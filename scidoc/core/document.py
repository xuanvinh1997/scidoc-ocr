"""Turn raw Xiaomi-OCR-0 output into a block model and render it to many formats.

Pipeline::

    raw model text --clean_model_output--> canonical Markdown (stored, user-editable)
    canonical Markdown --parse_blocks--> [Block]  --render_*--> md / html / tex / txt

Canonical Markdown uses ``\\( \\)`` / ``\\[ \\]`` math delimiters (like Mathpix MMD) and
GFM pipe tables, or HTML ``<table>`` when cells span rows/columns.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from markdown_it import MarkdownIt

from .tables import OTSL_CELL_RE, Table

# --------------------------------------------------------------------------- cleanup

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
_FENCE_WRAP_RE = re.compile(r"^\s*```(?:markdown|md|latex|tex)?\s*\n(.*)\n```\s*$", re.S)
_EQ_NUMBER_AFTER_DISPLAY = re.compile(r"\\\]\s*\n+\s*\((\d{1,3}[a-z]?)\)[ \t]*(?=\n|$)")


def clean_model_output(raw: str) -> str:
    """Normalise raw model output into canonical Markdown."""
    text = raw.replace("\r\n", "\n")
    text = _THINK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    m = _FENCE_WRAP_RE.match(text)
    if m:
        text = m.group(1)
    # "\] \n (5)" -> move equation number into the display as \tag{5}
    text = _EQ_NUMBER_AFTER_DISPLAY.sub(r"\\tag{\1}\\]", text)
    # Replace OTSL blocks by GFM / HTML tables.
    text = _replace_otsl_blocks(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _replace_otsl_blocks(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not _is_otsl_line(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        block = []
        while i < len(lines) and _is_otsl_line(lines[i]):
            block.append(lines[i])
            i += 1
        table = Table.from_otsl("\n".join(block))
        out.append(table.to_markdown() if table else "\n".join(block))
    return "\n".join(out)


def _is_otsl_line(line: str) -> bool:
    return bool(OTSL_CELL_RE.search(line) or re.search(r"<nl\s*/?>", line, re.I))


# --------------------------------------------------------------------------- blocks


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Paragraph:
    text: str


@dataclass
class DisplayMath:
    tex: str


@dataclass
class ListBlock:
    ordered: bool
    items: list[str] = field(default_factory=list)


@dataclass
class TableBlock:
    table: Table
    caption: str = ""


@dataclass
class Image:
    alt: str
    src: str


@dataclass
class CodeBlock:
    lang: str
    code: str


Block = Heading | Paragraph | DisplayMath | ListBlock | TableBlock | Image | CodeBlock

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*(?:([-*+•])|(\d+)[.)])\s+(.*)$")
_PIPE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]*)\)\s*$")
_ENV_BEGIN_RE = re.compile(
    r"^\s*\\begin\{(equation|align|gather|multline|eqnarray|displaymath)\*?\}"
)


def parse_blocks(md: str) -> list[Block]:
    lines = md.replace("\r\n", "\n").split("\n")
    blocks: list[Block] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            blocks.append(Paragraph(" ".join(s.strip() for s in para)))
            para.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            flush_para()
            i += 1
            continue

        # fenced code
        if s.startswith("```"):
            flush_para()
            lang = s[3:].strip()
            j = i + 1
            while j < n and not lines[j].strip().startswith("```"):
                j += 1
            blocks.append(CodeBlock(lang, "\n".join(lines[i + 1 : j])))
            i = j + 1
            continue

        # display math: \[ ... \]   or   $$ ... $$   or \begin{equation}...
        opener = None
        if s.startswith("\\["):
            opener, closer = "\\[", "\\]"
        elif s.startswith("$$"):
            opener, closer = "$$", "$$"
        if opener:
            j, body = _collect_until(lines, i, opener, closer)
            if body is not None:
                flush_para()
                blocks.append(DisplayMath(body.strip()))
                i = j + 1
                continue
        m_env = _ENV_BEGIN_RE.match(s)
        if m_env:
            env_end = re.compile(r"\\end\{" + re.escape(m_env.group(1)) + r"\*?\}")
            j = i
            while j < n and not env_end.search(lines[j]):
                j += 1
            flush_para()
            blocks.append(DisplayMath("\n".join(lines[i : j + 1]).strip()))
            i = j + 1
            continue

        # HTML table
        if s.lower().startswith("<table"):
            flush_para()
            j = i
            while j < n and "</table>" not in lines[j].lower():
                j += 1
            table = Table.from_html("\n".join(lines[i : j + 1]))
            if table:
                blocks.append(TableBlock(table))
            i = j + 1
            continue

        # OTSL (user pasted raw model output)
        if _is_otsl_line(s):
            flush_para()
            j = i
            while j < n and _is_otsl_line(lines[j]):
                j += 1
            table = Table.from_otsl("\n".join(lines[i:j]))
            if table:
                blocks.append(TableBlock(table))
            i = j
            continue

        # GFM pipe table
        if s.startswith("|") and i + 1 < n and _PIPE_SEP_RE.match(lines[i + 1]):
            flush_para()
            j = i
            while j < n and lines[j].strip().startswith("|"):
                j += 1
            table = Table.from_pipe(lines[i:j])
            if table:
                blocks.append(TableBlock(table))
            i = j
            continue

        m = _HEADING_RE.match(s)
        if m:
            flush_para()
            blocks.append(Heading(len(m.group(1)), m.group(2).strip()))
            i += 1
            continue

        m = _IMAGE_RE.match(s)
        if m:
            flush_para()
            blocks.append(Image(m.group(1), m.group(2)))
            i += 1
            continue

        m = _LIST_RE.match(line)
        if m and not para:
            ordered = m.group(2) is not None
            lb = ListBlock(ordered)
            while i < n:
                m = _LIST_RE.match(lines[i])
                if m and (m.group(2) is not None) == ordered:
                    lb.items.append(m.group(3).strip())
                elif lines[i].strip() and lines[i].startswith(("  ", "\t")) and lb.items:
                    lb.items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            blocks.append(lb)
            continue

        para.append(line)
        i += 1

    flush_para()
    return blocks


def _collect_until(lines: list[str], i: int, opener: str, closer: str) -> tuple[int, str | None]:
    """Collect a display-math block starting at line i. Returns (end_line, body)."""
    first = lines[i].strip()[len(opener) :]
    if closer in first:
        body, _, rest = first.partition(closer)
        if rest.strip():  # "\[x\] trailing text" -> not a standalone display, treat inline
            return i, None
        return i, body
    parts = [first]
    j = i + 1
    while j < len(lines):
        if closer in lines[j]:
            body, _, rest = lines[j].partition(closer)
            parts.append(body)
            return j, "\n".join(parts)
        parts.append(lines[j])
        j += 1
    return i, None  # unterminated; fall back to paragraph


def join_pages(pages: list[str]) -> list[Block]:
    """Parse pages and merge a paragraph that was split across a page break."""
    out: list[Block] = []
    for md in pages:
        blocks = parse_blocks(md)
        if (
            out
            and blocks
            and isinstance(out[-1], Paragraph)
            and isinstance(blocks[0], Paragraph)
            and not re.search(r"[.!?:;。！？]\s*$|\\\)\s*\.$", out[-1].text)
            and re.match(r"^[a-z(\\]", blocks[0].text)
        ):
            prev = out[-1].text
            joined = prev[:-1] + blocks[0].text if prev.endswith("-") else prev + " " + blocks[0].text
            out[-1] = Paragraph(joined)
            blocks = blocks[1:]
        out.extend(blocks)
    return out


# --------------------------------------------------------------------------- inline

_INLINE_MATH_RE = re.compile(
    r"\\\((?P<a>.+?)\\\)"  # \( ... \)
    r"|\\\[(?P<b>.+?)\\\]"  # \[ ... \]
    r"|\$\$(?P<c>.+?)\$\$"  # $$ ... $$
    r"|(?<![\\$\w])\$(?=\S)(?P<d>[^$\n]+?)(?<=\S)\$(?!\d)",  # $ ... $
    re.S,
)


def split_inline(text: str) -> list[tuple[str, str, bool]]:
    """Split text into ("text"|"math", content, display) segments."""
    segs: list[tuple[str, str, bool]] = []
    pos = 0
    for m in _INLINE_MATH_RE.finditer(text):
        if m.start() > pos:
            segs.append(("text", text[pos : m.start()], False))
        display = m.group("b") is not None or m.group("c") is not None
        tex = next(g for g in (m.group("a"), m.group("b"), m.group("c"), m.group("d")) if g is not None)
        segs.append(("math", tex.strip(), display))
        pos = m.end()
    if pos < len(text):
        segs.append(("text", text[pos:], False))
    return segs


# --------------------------------------------------------------------------- renderers

_md_inline = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
_md_inline.enable("strikethrough")


def _html_inline(text: str) -> str:
    out = []
    for kind, content, display in split_inline(text):
        if kind == "math":
            out.append(_tex_span(content, display))
        else:
            out.append(_md_inline.renderInline(content))
    return "".join(out)


def _tex_span(tex: str, display: bool) -> str:
    cls = "tex tex-display" if display else "tex"
    return f'<span class="{cls}">{html.escape(tex)}</span>'


def render_html_body(blocks: list[Block]) -> str:
    out: list[str] = []
    for b in blocks:
        if isinstance(b, Heading):
            out.append(f"<h{b.level}>{_html_inline(b.text)}</h{b.level}>")
        elif isinstance(b, Paragraph):
            out.append(f"<p>{_html_inline(b.text)}</p>")
        elif isinstance(b, DisplayMath):
            out.append(f'<div class="tex tex-display">{html.escape(b.tex)}</div>')
        elif isinstance(b, ListBlock):
            tag = "ol" if b.ordered else "ul"
            items = "".join(f"<li>{_html_inline(t)}</li>" for t in b.items)
            out.append(f"<{tag}>{items}</{tag}>")
        elif isinstance(b, TableBlock):
            out.append(f'<div class="table-wrap">{b.table.to_html(_html_inline)}</div>')
        elif isinstance(b, Image):
            out.append(f'<figure><img alt="{html.escape(b.alt)}" src="{html.escape(b.src)}"></figure>')
        elif isinstance(b, CodeBlock):
            out.append(f"<pre><code>{html.escape(b.code)}</code></pre>")
    return "\n".join(out)


def render_markdown(blocks: list[Block], dollar: bool = False) -> str:
    """Re-emit Markdown. ``dollar=True`` uses $..$/$$..$$ (GitHub, Obsidian, Typora)."""

    def inline(text: str) -> str:
        if not dollar:
            return text
        return "".join(
            (f"$${c}$$" if d else f"${c}$") if k == "math" else c for k, c, d in split_inline(text)
        )

    out: list[str] = []
    for b in blocks:
        if isinstance(b, Heading):
            out.append("#" * b.level + " " + inline(b.text))
        elif isinstance(b, Paragraph):
            out.append(inline(b.text))
        elif isinstance(b, DisplayMath):
            out.append(f"$$\n{b.tex}\n$$" if dollar else f"\\[\n{b.tex}\n\\]")
        elif isinstance(b, ListBlock):
            out.append(
                "\n".join(
                    (f"{k + 1}. " if b.ordered else "- ") + inline(t) for k, t in enumerate(b.items)
                )
            )
        elif isinstance(b, TableBlock):
            if dollar and not b.table.has_spans:
                for c in b.table.cells:
                    c.text = inline(c.text)
            out.append(b.table.to_markdown())
        elif isinstance(b, Image):
            out.append(f"![{b.alt}]({b.src})")
        elif isinstance(b, CodeBlock):
            out.append(f"```{b.lang}\n{b.code}\n```")
    return "\n\n".join(out) + "\n"


def render_text(blocks: list[Block]) -> str:
    """Plain text; math is kept as $LaTeX$ so nothing is lost."""

    def inline(text: str) -> str:
        parts = []
        for k, c, d in split_inline(text):
            if k == "math":
                parts.append(f"$${c}$$" if d else f"${c}$")
            else:
                parts.append(re.sub(r"(\*\*|__|\*|`)", "", c))
        return "".join(parts)

    out = []
    for b in blocks:
        if isinstance(b, (Heading, Paragraph)):
            out.append(inline(b.text))
        elif isinstance(b, DisplayMath):
            out.append(f"$$ {b.tex} $$")
        elif isinstance(b, ListBlock):
            out.append("\n".join(("%d. " % (k + 1) if b.ordered else "• ") + inline(t) for k, t in enumerate(b.items)))
        elif isinstance(b, TableBlock):
            out.append(inline(b.table.to_text()))
        elif isinstance(b, CodeBlock):
            out.append(b.code)
    return "\n\n".join(out) + "\n"


# ---- LaTeX

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(s: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in s)


def _latex_text(s: str) -> str:
    """Escape text and translate a few Markdown inline marks."""
    tokens = re.split(r"(\*\*[^*]+\*\*|__[^_]+__|\*[^*\s][^*]*\*|`[^`]+`)", s)
    out = []
    for t in tokens:
        if t.startswith("**") and t.endswith("**") and len(t) > 4:
            out.append(r"\textbf{" + _latex_escape(t[2:-2]) + "}")
        elif t.startswith("__") and t.endswith("__") and len(t) > 4:
            out.append(r"\textbf{" + _latex_escape(t[2:-2]) + "}")
        elif t.startswith("*") and t.endswith("*") and len(t) > 2:
            out.append(r"\emph{" + _latex_escape(t[1:-1]) + "}")
        elif t.startswith("`") and t.endswith("`") and len(t) > 2:
            out.append(r"\texttt{" + _latex_escape(t[1:-1]) + "}")
        else:
            out.append(_latex_escape(t))
    return "".join(out)


def _latex_inline(text: str) -> str:
    out = []
    for kind, content, display in split_inline(text):
        if kind == "math":
            out.append(f"\\[{content}\\]" if display else f"\\({content}\\)")
        else:
            out.append(_latex_text(content))
    return "".join(out)


_HAS_TAG = re.compile(r"\\tag\{")
_ENV_START = re.compile(r"^\s*\\begin\{")


def render_latex_body(blocks: list[Block]) -> str:
    out: list[str] = []
    sections = ["section", "subsection", "subsubsection", "paragraph", "subparagraph", "subparagraph"]
    for b in blocks:
        if isinstance(b, Heading):
            out.append(f"\\{sections[b.level - 1]}*{{{_latex_inline(b.text)}}}")
        elif isinstance(b, Paragraph):
            out.append(_latex_inline(b.text))
        elif isinstance(b, DisplayMath):
            if _ENV_START.match(b.tex):
                out.append(b.tex)
            elif _HAS_TAG.search(b.tex):
                out.append(f"\\begin{{equation*}}\n{b.tex}\n\\end{{equation*}}")
            else:
                out.append(f"\\[\n{b.tex}\n\\]")
        elif isinstance(b, ListBlock):
            env = "enumerate" if b.ordered else "itemize"
            items = "\n".join(f"  \\item {_latex_inline(t)}" for t in b.items)
            out.append(f"\\begin{{{env}}}\n{items}\n\\end{{{env}}}")
        elif isinstance(b, TableBlock):
            out.append(
                "\\begin{table}[htbp]\n\\centering\n\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n"
                + b.table.to_latex(_latex_inline)
                + "}\n\\end{table}"
            )
        elif isinstance(b, Image):
            if b.src.startswith(("http://", "https://", "#")):
                out.append(f"% image: {b.alt} ({b.src})")
            else:
                cap = f"\n\\caption{{{_latex_text(b.alt)}}}" if b.alt and b.alt != "Figure" else ""
                out.append("\\begin{figure}[htbp]\n\\centering\n"
                           f"\\includegraphics[width=\\linewidth,height=0.45\\textheight,keepaspectratio]{{{b.src}}}"
                           f"{cap}\n\\end{{figure}}")
        elif isinstance(b, CodeBlock):
            out.append(f"\\begin{{verbatim}}\n{b.code}\n\\end{{verbatim}}")
    return "\n\n".join(out) + "\n"


LATEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{iftex}
\ifPDFTeX
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
  \usepackage{lmodern}
\else
  \usepackage{fontspec}
\fi
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts,mathtools,bm}
\usepackage{graphicx,multirow,array,booktabs}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
"""


def render_latex_document(blocks: list[Block], title: str = "") -> str:
    head = LATEX_PREAMBLE
    if title:
        head += f"\\title{{{_latex_escape(title)}}}\n\\date{{}}\n"
    body = render_latex_body(blocks)
    return head + "\n\\begin{document}\n" + ("\\maketitle\n\n" if title else "\n") + body + "\n\\end{document}\n"
