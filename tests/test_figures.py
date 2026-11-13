"""Figures: bbox markers from layout models, cropping, placeholders and export."""

from PIL import Image

from scidoc.core.export import export_file, export_text
from scidoc.core.figures import FigureStore, file_urls, insert_figure, materialize
from scidoc.core.models.outputs import from_doctags, from_layout_json


def _page():
    return Image.new("RGB", (1000, 1400), "white")


def test_doctags_picture_becomes_marker_and_caption():
    md = from_doctags("<doctag><picture><loc_50><loc_100><loc_450><loc_300>"
                      "<caption><loc_50><loc_305><loc_450><loc_320>Figure 1. Loss.</caption></picture></doctag>")
    assert "![Figure 1. Loss.](#bbox=0.1000,0.2000,0.9000,0.6000)" in md
    assert "*Figure 1. Loss.*" in md


def test_layout_json_pixels_are_normalised_by_model_input_size():
    md = from_layout_json('[{"bbox":[100,200,700,600],"category":"Picture"}]', (1000, 1400))
    assert "#bbox=0.1000,0.1429,0.7000,0.4286" in md
    # Without the input size a pixel box cannot be placed: dropped, as before.
    assert "#bbox" not in from_layout_json('[{"bbox":[100,200,700,600],"category":"Picture"}]')


def test_materialize_crops_markers(tmp_path):
    store = FigureStore("d", tmp_path)
    md = materialize("![Cap](#bbox=0.1,0.2,0.5,0.4)\n", _page(), 2, store)
    assert md.strip() == "![Cap](figures/p3-fig1.png)"
    w, h = Image.open(store.resolve("figures/p3-fig1.png")).size
    assert abs(w - 400) <= 10 and abs(h - 280) <= 14


def test_detected_boxes_fill_placeholders_in_reading_order(tmp_path):
    store = FigureStore("d", tmp_path)
    md = "A\n\n*[Figure: top]*\n\nB *[Figure: bottom]* C\n"
    out = materialize(md, _page(), 0, store, [(0.1, 0.6, 0.9, 0.8), (0.1, 0.1, 0.9, 0.3)])
    assert "![top](figures/p1-fig1.png)" in out and "![bottom](figures/p1-fig2.png)" in out
    assert out.index("top") < out.index("bottom")


def test_detected_boxes_ignored_when_page_already_has_figures(tmp_path):
    store = FigureStore("d", tmp_path)
    md = "![x](figures/p1-fig9.png)\n"
    assert materialize(md, _page(), 0, store, [(0.1, 0.1, 0.9, 0.3)]) == md


def test_insert_figure_replaces_placeholder_else_cursor():
    assert insert_figure("A *[Figure: p]* B\n", "![p](figures/a.png)").count("![p](figures/a.png)") == 1
    assert "*[Figure" not in insert_figure("A *[Figure: p]* B\n", "![p](figures/a.png)")
    assert insert_figure("A\n\nB\n", "![x](figures/a.png)", 2) == "A\n\n![x](figures/a.png)\n\nB\n"


def test_export_copies_or_embeds_figures(tmp_path):
    store = FigureStore("d", tmp_path / "cache")
    rel = store.save(Image.new("RGB", (10, 10), "red"), 0)
    pages = [f"Text\n\n![Plot]({rel})\n"]
    (tmp_path / "out").mkdir()
    out = export_file(pages, "md", tmp_path / "out" / "doc.md", figures=store)
    assert (out.parent / rel).exists()
    assert "data:image/png;base64," in export_text(pages, "html", figures=store)
    assert "\\includegraphics" in export_text(pages, "tex") and rel in export_text(pages, "tex")
    assert store.base.as_uri() in file_urls(f'<img src="{rel}">', store)


def test_split_blocks_keeps_display_math_and_fences_together():
    from scidoc.core.figures import split_blocks

    md = "A\n\n\\[\nx\n\ny\n\\]\n\n```\nc\n\nd\n```\n\n$$\nz\n\n$$\n\nB\n"
    assert split_blocks(md) == ["A", "\\[\nx\n\ny\n\\]", "```\nc\n\nd\n```", "$$\nz\n\n$$", "B"]


def test_figure_goes_before_its_caption_without_placeholder(tmp_path):
    # The screenshot case: the picture is at the top, the model wrote only the caption.
    store = FigureStore("d", tmp_path)
    md = "Figure 1.8 — Dispersion law for a free electron.\n\nenergy level for the electron…\n\nFinally, …\n"
    out = materialize(md, _page(), 0, store, [(0.05, 0.08, 0.95, 0.33)], [(0.1, 0.35, 0.9, 0.4)])
    assert out.startswith("![Figure](figures/p1-fig1.png)\n\nFigure 1.8")


def test_figure_placed_by_text_above_it_when_no_caption(tmp_path):
    store = FigureStore("d", tmp_path)
    md = "Para one is here.\n\nPara two is here.\n\nPara three is here.\n\nPara four is here.\n"
    texts = [(0.1, 0.05, 0.9, 0.15), (0.1, 0.16, 0.9, 0.26), (0.1, 0.70, 0.9, 0.80), (0.1, 0.81, 0.9, 0.91)]
    out = materialize(md, _page(), 0, store, [(0.1, 0.3, 0.9, 0.65)], texts)
    assert out.index("Para two") < out.index("![Figure]") < out.index("Para three")


def test_caption_regex_skips_body_text_mentioning_a_figure():
    from scidoc.core.figures import CAPTION_RE

    for cap in ("Figure 1.8 — Dispersion law", "Fig. 2: Setup", "**Figure 3.** Loss", "Hình 4. Sơ đồ", "Рис. 5 – x"):
        assert CAPTION_RE.match(cap), cap
    assert not CAPTION_RE.match("Figure 2 shows that the loss decreases.")
