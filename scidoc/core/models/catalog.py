"""Built-in model catalog.

Prompts, decoding settings and output formats are copied from each model card
(researched Nov 2026). Models whose prompts live inside their own SDK pipeline
(MinerU2.5, chandra, surya, MiniCPM-OCR, olmOCR, FireRed-OCR) are not listed: they
cannot be driven correctly by a plain chat request.
"""

from __future__ import annotations

from .spec import RUNTIME_SERVER, ModelSpec

DEFAULT_MODEL_ID = "xiaomi-ocr-0"

# Used by general VLMs and user-added custom models.
GENERIC_PROMPTS = {
    "document": (
        "Convert this document page to Markdown. Keep the reading order and ignore page headers, "
        "footers and page numbers. Write every mathematical formula in LaTeX, using \\( \\) for inline "
        "math and \\[ \\] for display equations. Write tables as HTML <table>. Do not translate. "
        "Output only the converted content."
    ),
    "text": "Extract all text in the image exactly as written, preserving line breaks. Output only the text.",
    "formula": "Convert the formula in the image to LaTeX. Output only the LaTeX code, without $ or \\[ delimiters.",
    "table": "Convert the table in the image to an HTML <table>, using rowspan/colspan for merged cells. "
    "Output only the table.",
}

XIAOMI_PROMPTS = {
    "document": (
        "Extract all information from the main body of the document image and represent it "
        "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
        "format, formulas in the document should be represented using LATEX format, and the "
        "parsing should be organized according to the reading order."
    ),
    "text": "Extract the text in the image.",
    "table": "Parse the table in the image into OTSL.",
    "formula": "Identify the formula in the image and represent it using LATEX format.",
}

NANONETS_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. Return the tables in "
    "html format. Return the equations in LaTeX representation. If there is an image in the document and "
    "image caption is not present, add a small description of the image inside the <img></img> tag; "
    "otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: "
    "<watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: "
    "<page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."
)

DOTS_LAYOUT_PROMPT = (
    "Please output the layout information from the PDF image, including each layout element's bbox, its "
    "category, and the corresponding text content within the bbox.\n\n1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', "
    "'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n    - Picture: For the 'Picture' category, the text field "
    "should be omitted.\n    - Formula: Format its text as LaTeX.\n    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n4. Constraints:\n    - The "
    "output text must be the original text from the image, with no translation.\n    - All layout "
    "elements must be sorted according to human reading order.\n\n5. Final Output: The entire output "
    "must be a single JSON object.\n"
)

QWEN3VL_NOTE = "General-purpose VLM with SciDoc's generic OCR prompts. Strong multilingual support (incl. Vietnamese)."

BUILTIN_MODELS: list[ModelSpec] = [
    # ------------------------------------------------------------------ local, full page
    ModelSpec(
        id="xiaomi-ocr-0",
        name="Xiaomi-OCR-0",
        repo="xiaomi-research/Xiaomi-OCR-0",
        family="qwen3_5",
        params_b=0.87,
        license="Apache-2.0",
        languages=["en", "zh", "multilingual"],
        description="0.9B end-to-end document parser. OmniDocBench v1.6 96.8, UniMER 97.8. Default model.",
        strengths=["math", "tables", "handwriting", "fast"],
        prompts=XIAOMI_PROMPTS,
        chat_template_kwargs={"enable_thinking": False},
        dtype="fp16",
        serve_hint="vllm serve xiaomi-research/Xiaomi-OCR-0",
        verified=True,
    ),
    ModelSpec(
        id="lighton-ocr-3-0.8b",
        name="LightOnOCR-3 0.8B",
        repo="lightonai/LightOnOCR-3-0.8B",
        family="qwen3_5",
        params_b=0.85,
        license="Apache-2.0",
        languages=["en"],
        description="Image-only prompt, Markdown output with HTML tables. Very fast.",
        strengths=["fast", "tables"],
        prompts={"document": ""},
        chat_template_kwargs={"enable_thinking": False},
        generation={"do_sample": True, "temperature": 0.2, "top_p": 0.9},
        max_pixels=2048 * 1536,
        dtype="bf16",
        serve_hint="vllm serve lightonai/LightOnOCR-3-0.8B --trust-remote-code "
        "--default-chat-template-kwargs '{\"enable_thinking\": false}'",
    ),
    ModelSpec(
        id="lighton-ocr-3-4b",
        name="LightOnOCR-3 4B",
        repo="lightonai/LightOnOCR-3-4B",
        family="qwen3_5",
        params_b=4.54,
        license="Apache-2.0",
        languages=["en", "multilingual"],
        description="LightOn's best OCR model (\"recommended for most tasks\"). Markdown + HTML tables.",
        strengths=["tables", "accuracy"],
        prompts={"document": ""},
        chat_template_kwargs={"enable_thinking": False},
        generation={"do_sample": True, "temperature": 0.2, "top_p": 0.9},
        max_pixels=2048 * 1536,
        dtype="bf16",
        serve_hint="vllm serve lightonai/LightOnOCR-3-4B --trust-remote-code "
        "--default-chat-template-kwargs '{\"enable_thinking\": false}'",
    ),
    ModelSpec(
        id="nanonets-ocr2-3b",
        name="Nanonets-OCR2 3B",
        repo="nanonets/Nanonets-OCR2-3B",
        family="qwen2_5_vl",
        params_b=3.75,
        license="(not stated on card)",
        languages=["multilingual"],
        description="Markdown with $-LaTeX, HTML tables, checkboxes, signatures, watermarks, image captions.",
        strengths=["math", "tables", "forms", "handwriting"],
        prompts={"document": NANONETS_PROMPT},
        system_prompt="You are a helpful assistant.",
        dtype="bf16",
        serve_hint="vllm serve nanonets/Nanonets-OCR2-3B",
        notes="License is not stated on the model card — check before commercial use.",
    ),
    ModelSpec(
        id="qianfan-ocr",
        name="Qianfan-OCR (4B)",
        repo="baidu/Qianfan-OCR",
        family="qianfan_ocr",
        params_b=4.74,
        license="Apache-2.0",
        languages=["multilingual (192)"],
        description="End-to-end document → Markdown, HTML tables, LaTeX formulas, 192 languages.",
        strengths=["multilingual", "tables", "math"],
        prompts={"document": "Parse this document to Markdown."},
        dtype="bf16",
        serve_hint="vllm serve baidu/Qianfan-OCR --trust-remote-code "
        "--hf-overrides '{\"architectures\": [\"InternVLChatModel\"]}'",
    ),
    ModelSpec(
        id="hunyuan-ocr",
        name="HunyuanOCR",
        repo="tencent/HunyuanOCR",
        family="hunyuan_vl",
        params_b=1.12,
        license="Tencent Hunyuan Community License",
        languages=["multilingual", "en", "zh"],
        description="1B end-to-end parser: Markdown, HTML tables, LaTeX; headers/footers skipped.",
        strengths=["multilingual", "tables", "math"],
        prompts={
            "document": "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，"
            "文档中公式用latex格式表示，按照阅读顺序组织进行解析。",
            "text": "请提取图片中的文字内容。",
        },
        generation={"repetition_penalty": 1.08},
        dtype="bf16",
        serve_hint="see the model card (inference/vLLM/serve.sh)",
        notes="Custom community license — review terms before commercial use.",
    ),
    ModelSpec(
        id="granite-docling-258m",
        name="Granite-Docling 258M",
        repo="ibm-granite/granite-docling-258M",
        family="idefics3",
        params_b=0.26,
        license="Apache-2.0",
        languages=["en"],
        description="Tiny IBM model emitting DocTags (converted by SciDoc). Runs even on CPU.",
        strengths=["tiny", "tables"],
        prompts={
            "document": "Convert this page to docling.",
            "formula": "Convert formula to LaTeX.",
            "table": "Convert table to OTSL.",
        },
        output="doctags",
        output_by_task={"formula": "latex", "table": "markdown"},
        skip_special_tokens=False,
        generation={"max_new_tokens": 8192},
        dtype="bf16",
        serve_hint="vllm serve ibm-granite/granite-docling-258M --revision untied",
        notes="Needs bf16 or fp32: fp16 produces garbage ('!!!!').",
    ),
    ModelSpec(
        id="qwen3-vl-2b",
        name="Qwen3-VL 2B Instruct",
        repo="Qwen/Qwen3-VL-2B-Instruct",
        family="qwen3_vl",
        params_b=2.13,
        license="Apache-2.0",
        languages=["multilingual (32 OCR)", "vi"],
        description=QWEN3VL_NOTE,
        strengths=["multilingual", "vietnamese"],
        prompts=dict(GENERIC_PROMPTS),
        dtype="bf16",
        serve_hint="vllm serve Qwen/Qwen3-VL-2B-Instruct",
    ),
    ModelSpec(
        id="qwen3-vl-8b",
        name="Qwen3-VL 8B Instruct",
        repo="Qwen/Qwen3-VL-8B-Instruct",
        family="qwen3_vl",
        params_b=8.77,
        license="Apache-2.0",
        languages=["multilingual (32 OCR)", "vi"],
        description=QWEN3VL_NOTE + " Higher accuracy, needs ~24 GB.",
        strengths=["multilingual", "vietnamese", "accuracy"],
        prompts=dict(GENERIC_PROMPTS),
        dtype="bf16",
        serve_hint="vllm serve Qwen/Qwen3-VL-8B-Instruct",
    ),
    # ------------------------------------------------------------------ local, region level (snips)
    ModelSpec(
        id="glm-ocr",
        name="GLM-OCR",
        repo="zai-org/GLM-OCR",
        family="glm_ocr",
        params_b=1.33,
        license="MIT",
        languages=["zh", "en", "fr", "es", "ru", "de", "ja", "ko"],
        description="Region recognizer (text / formula / table). Best for snips; full pages need its layout SDK.",
        strengths=["math", "snips"],
        prompts={"text": "Text Recognition:", "formula": "Formula Recognition:", "table": "Table Recognition:"},
        output_by_task={"formula": "latex"},
        dtype="bf16",
        serve_hint="vllm serve zai-org/GLM-OCR",
        notes="Full-page parsing falls back to 'Text Recognition:' on the whole page (no layout step).",
    ),
    ModelSpec(
        id="paddleocr-vl-1.6",
        name="PaddleOCR-VL 1.6",
        repo="PaddlePaddle/PaddleOCR-VL-1.6",
        family="paddleocr_vl",
        params_b=0.96,
        license="Apache-2.0",
        languages=["en", "zh", "multilingual"],
        description="0.9B element recognizer (text / formula / table / chart). Best for snips.",
        strengths=["math", "tables", "snips"],
        prompts={"text": "OCR:", "formula": "Formula Recognition:", "table": "Table Recognition:"},
        output_by_task={"formula": "latex"},
        max_pixels=1280 * 28 * 28,
        dtype="bf16",
        serve_hint="see the model card (PaddleOCR genai_server, vLLM backend)",
        notes="Page-level parsing needs the PaddleOCR pipeline; here a full page uses the 'OCR:' prompt.",
    ),
    # ------------------------------------------------------------------ server only (custom code / CUDA)
    ModelSpec(
        id="deepseek-ocr-2",
        name="DeepSeek-OCR 2",
        repo="deepseek-ai/DeepSeek-OCR-2",
        family="deepseek_ocr",
        params_b=3.39,
        license="Apache-2.0",
        languages=["multilingual"],
        description="Visual-token-compression OCR, document → Markdown with layout grounding.",
        strengths=["tables", "math"],
        runtime=RUNTIME_SERVER,
        trust_remote_code=True,
        prompts={"document": "<|grounding|>Convert the document to markdown.", "text": "Free OCR."},
        serve_hint="CUDA + custom code; see the DeepSeek-OCR-2 GitHub for vLLM serving",
    ),
    ModelSpec(
        id="dots-ocr",
        name="dots.ocr",
        repo="rednote-hilab/dots.ocr",
        family="dots_ocr",
        params_b=3.04,
        license="MIT",
        languages=["en", "zh", "multilingual"],
        description="Layout + content in one JSON (bbox, category, Markdown/LaTeX/HTML), converted by SciDoc.",
        strengths=["tables", "math", "multilingual"],
        runtime=RUNTIME_SERVER,
        trust_remote_code=True,
        prompts={"document": DOTS_LAYOUT_PROMPT},
        output="layout-json",
        generation={"max_new_tokens": 24000},
        max_pixels=11_289_600,
        serve_hint="vllm serve <local DotsOCR dir> --trust-remote-code --chat-template-content-format string "
        "--served-model-name model  (register DotsOCR first, see model card)",
    ),
    ModelSpec(
        id="vintern-1b-v3.5",
        name="Vintern-1B v3.5 (Vietnamese)",
        repo="5CD-AI/Vintern-1B-v3_5",
        family="internvl_chat",
        params_b=0.94,
        license="MIT",
        languages=["vi", "en", "zh"],
        description="InternVL2.5-1B fine-tuned on Vietnamese documents. Markdown extraction.",
        strengths=["vietnamese"],
        runtime=RUNTIME_SERVER,
        trust_remote_code=True,
        prompts={"document": "Trích xuất thông tin chính trong ảnh và trả về dạng markdown."},
        generation={"repetition_penalty": 2.5, "max_new_tokens": 1024},
        serve_hint="vllm serve 5CD-AI/Vintern-1B-v3_5 --trust-remote-code  (InternVL architecture; untested)",
        notes="General extraction model, not a full-page LaTeX parser.",
    ),
]
