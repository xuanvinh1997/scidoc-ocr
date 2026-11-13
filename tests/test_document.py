from scidoc.core.document import (
    DisplayMath,
    Heading,
    ListBlock,
    Paragraph,
    TableBlock,
    clean_model_output,
    join_pages,
    parse_blocks,
    render_html_body,
    render_latex_document,
    render_markdown,
    split_inline,
)
from scidoc.core.tables import Table


def test_otsl_simple():
    t = Table.from_otsl("<fcel>a<fcel>b<nl><fcel>1<ecel><nl>")
    assert (t.n_rows, t.n_cols) == (2, 2)
    assert t.to_markdown().splitlines()[0] == "| a | b |"
    assert "| 1 |  |" in t.to_markdown()


def test_otsl_spans():
    # header "x" spans two columns; "y" spans two rows
    t = Table.from_otsl("<fcel>x<lcel><fcel>z<nl><fcel>y<fcel>1<fcel>2<nl><ucel><fcel>3<fcel>4<nl>")
    cells = {(c.row, c.col): c for c in t.cells}
    assert cells[(0, 0)].colspan == 2
    assert cells[(1, 0)].rowspan == 2
    assert t.has_spans
    html = t.to_html()
    assert 'colspan="2"' in html and 'rowspan="2"' in html
    tex = t.to_latex(lambda s: s)
    assert "\\multicolumn{2}" in tex and "\\multirow{2}" in tex


def test_otsl_newline_rows():
    t = Table.from_otsl("<fcel>a<fcel>b\n<fcel>c<fcel>d")
    assert (t.n_rows, t.n_cols) == (2, 2)


def test_html_and_pipe_roundtrip():
    t = Table.from_otsl("<fcel>x<lcel><nl><fcel>1<fcel>2<nl>")
    t2 = Table.from_html(t.to_html())
    assert [(c.row, c.col, c.colspan) for c in t2.cells] == [(c.row, c.col, c.colspan) for c in t.cells]
    p = Table.from_pipe(["| a | b |", "|---|---|", "| 1 | 2 |"])
    assert p.n_rows == 2 and p.n_cols == 2


def test_clean_output_converts_otsl_and_tags():
    raw = "<think>\n\n</think>\n\n# Title\n\nText\n\n<fcel>a<fcel>b<nl><fcel>1<fcel>2<nl>\n\n\\[x=1\\]\n(3)\n"
    md = clean_model_output(raw)
    assert "<think>" not in md and "<fcel>" not in md
    assert "| a | b |" in md
    assert "\\tag{3}" in md


def test_parse_blocks():
    md = "# H\n\nPara with \\(x^2\\) and *em*.\n\n\\[\na+b\n\\]\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    kinds = [type(b) for b in parse_blocks(md)]
    assert kinds == [Heading, Paragraph, DisplayMath, ListBlock, TableBlock]


def test_single_line_display_math():
    blocks = parse_blocks("\\[\\mathbf {S} = 1 \\tag{5}\\]")
    assert isinstance(blocks[0], DisplayMath) and "\\tag{5}" in blocks[0].tex


def test_inline_math_split_and_currency():
    segs = split_inline("cost $5 and $10, but $x+y$ is math")
    maths = [c for k, c, _ in segs if k == "math"]
    assert maths == ["x+y"]


def test_html_escapes_and_marks_math():
    html = render_html_body(parse_blocks("a <script>alert(1)</script> \\(x<y\\)"))
    assert "<script>" not in html
    assert '<span class="tex">x&lt;y</span>' in html


def test_latex_document():
    tex = render_latex_document(parse_blocks("# Intro\n\n50% of \\(x_1\\) & more\n\n\\[a \\tag{1}\\]"))
    assert "\\section*{Intro}" in tex
    assert "50\\% of \\(x_1\\) \\& more" in tex
    assert "\\begin{equation*}" in tex
    assert tex.strip().endswith("\\end{document}")


def test_markdown_dollar():
    md = render_markdown(parse_blocks("see \\(a\\)\n\n\\[b\\]"), dollar=True)
    assert "$a$" in md and "$$\nb\n$$" in md


def test_join_pages_merges_split_paragraph():
    blocks = join_pages(["The result holds for every", "integer \\(n\\)."])
    assert len(blocks) == 1 and blocks[0].text == "The result holds for every integer \\(n\\)."
    blocks = join_pages(["Sentence ends.", "next page starts"])
    assert len(blocks) == 2
