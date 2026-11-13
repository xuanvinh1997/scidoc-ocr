import json

from scidoc.core.backends import OpenAIBackend, Task
from scidoc.core.models import BUILTIN_MODELS, Registry, custom_spec, to_canonical
from scidoc.core.models.outputs import from_doctags, from_layout_json, strip_math_delimiters
from PIL import Image


def spec(model_id):
    return next(s for s in BUILTIN_MODELS if s.id == model_id)


def test_catalog_ids_unique_and_have_prompts():
    ids = [s.id for s in BUILTIN_MODELS]
    assert len(ids) == len(set(ids))
    for s in BUILTIN_MODELS:
        assert s.prompts, s.id
        for t in ("document", "text", "formula", "table"):
            s.prompt_for(t)  # every task resolves (falls back if unsupported)


def test_registry_user_models_override_and_persist(tmp_path):
    reg = Registry(tmp_path / "models.json")
    custom = custom_spec("someorg/some-vlm")
    reg.add_or_update(custom)
    override = spec("xiaomi-ocr-0").from_dict({**spec("xiaomi-ocr-0").to_dict(), "name": "Xiaomi tweaked"})
    reg.add_or_update(override)
    reg2 = Registry(tmp_path / "models.json")
    assert reg2.get(custom.id).repo == "someorg/some-vlm"
    assert reg2.get("xiaomi-ocr-0").name == "Xiaomi tweaked"
    reg2.remove("xiaomi-ocr-0")
    assert Registry(tmp_path / "models.json").get("xiaomi-ocr-0").name == "Xiaomi-OCR-0"
    # unknown ids / raw repos become generic custom specs
    assert reg2.get("org/unknown").prompts["document"]
    assert spec("xiaomi-ocr-0").name == "Xiaomi-OCR-0"  # catalog untouched


def test_markdown_normalisation_nanonets_tags():
    raw = ("Intro $x^2$ text<page_number>3</page_number>\n\n<img>A plot of loss</img>\n\n"
           "<watermark>DRAFT</watermark>Para<table><tr><td>a</td></tr></table>")
    md = to_canonical(raw, spec("nanonets-ocr2-3b"), "document")
    assert "page_number" not in md and "DRAFT" not in md
    assert "*[Figure: A plot of loss]*" in md
    assert "| a |" in md or "<table>" in md


def test_deepseek_grounding_and_lighton_placeholders_removed():
    raw = "<|ref|>title<|/ref|><|det|>[[1,2,3,4]]<|/det|>\n# Title\n\n![text](80,175,920,225)\nBody."
    md = to_canonical(raw, spec("deepseek-ocr-2"), "document")
    assert md.startswith("# Title") and "Body." in md and "det" not in md and "](80" not in md


def test_formula_task_outputs_display_math():
    for raw in ("$$E=mc^2$$", "\\[E=mc^2\\]", "E=mc^2", "```latex\nE=mc^2\n```"):
        assert to_canonical(raw, spec("glm-ocr"), "formula") == "\\[\nE=mc^2\n\\]\n"
    assert strip_math_delimiters("\\begin{equation}a\\end{equation}") == "a"


def test_doctags_conversion():
    raw = ("<doctag><title><loc_1><loc_2><loc_3><loc_4>Paper</title>"
           "<section_header_level_1><loc_1><loc_2><loc_3><loc_4>Intro</section_header_level_1>"
           "<text><loc_1><loc_2><loc_3><loc_4>Hello world.</text>"
           "<formula><loc_1><loc_2><loc_3><loc_4>a^2+b^2</formula>"
           "<otsl><loc_1><loc_2><loc_3><loc_4><ched>x<ched>y<nl><fcel>1<fcel>2<nl></otsl>"
           "<page_footer><loc_1><loc_2><loc_3><loc_4>3</page_footer></doctag><end_of_utterance>")
    md = from_doctags(raw)
    assert "# Paper" in md and "## Intro" in md and "Hello world." in md
    assert "\\[\na^2+b^2\n\\]" in md and "| x | y |" in md
    assert "loc_" not in md


def test_layout_json_conversion():
    raw = json.dumps([
        {"bbox": [0, 0, 1, 1], "category": "Page-header", "text": "Journal"},
        {"bbox": [0, 0, 1, 1], "category": "Title", "text": "Deep Nets"},
        {"bbox": [0, 0, 1, 1], "category": "Text", "text": "We study $f(x)$."},
        {"bbox": [0, 0, 1, 1], "category": "Formula", "text": "$$y=f(x)$$"},
        {"bbox": [0, 0, 1, 1], "category": "Table", "text": "<table><tr><td>1</td></tr></table>"},
        {"bbox": [0, 0, 1, 1], "category": "Picture"},
    ])
    md = from_layout_json(raw)
    assert "Journal" not in md and "# Deep Nets" in md and "\\[\ny=f(x)\n\\]" in md and "<table>" in md


def test_openai_body_follows_spec():
    img = Image.new("RGB", (300, 300), "white")
    b = OpenAIBackend(spec("lighton-ocr-3-0.8b"), "http://x/v1").build_body(img, Task.DOCUMENT)
    content = b["messages"][-1]["content"]
    assert [p["type"] for p in content] == ["image_url"]  # image-only prompt
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert b["temperature"] == 0.2 and b["top_p"] == 0.9
    b = OpenAIBackend(spec("nanonets-ocr2-3b"), "http://x/v1").build_body(img, Task.FORMULA)
    assert b["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    b = OpenAIBackend(spec("granite-docling-258m"), "http://x/v1").build_body(img, Task.DOCUMENT)
    assert b["skip_special_tokens"] is False and b["max_tokens"] == 8192
