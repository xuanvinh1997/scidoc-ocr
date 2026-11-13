"""Normalise each model's native output into SciDoc canonical Markdown.

Canonical Markdown = Markdown + LaTeX math (\\( \\) / \\[ \\] / $ / $$) + GFM or HTML tables.
"""

from __future__ import annotations

import json
import re

from ..document import clean_model_output
from ..figures import bbox_marker
from ..tables import Table
from .spec import ModelSpec

# --------------------------------------------------------------------------- shared cleanup

_SPECIAL_TOKENS = re.compile(r"<\|[a-z_]+\|>|<end_of_utterance>|<\|?endoftext\|?>|</?s>|<pad>", re.I)
_DS_GROUNDING = re.compile(r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>\s*", re.S)
_LIGHTON_GROUNDING = re.compile(r"^!\[[a-z_+]+\]\(\d+,\s*\d+,\s*\d+,\s*\d+\)\s*$", re.M | re.I)
_FRONT_MATTER = re.compile(r"\A\s*---\n.*?\n---\n", re.S)


def _normalize_markup(text: str) -> str:
    """Strip model-specific tags that are not part of the document content."""
    text = _DS_GROUNDING.sub("", text)
    text = _LIGHTON_GROUNDING.sub("", text)
    text = re.sub(r"<page_number>.*?</page_number>", "", text, flags=re.S)
    text = re.sub(r"<watermark>.*?</watermark>", "", text, flags=re.S)
    text = re.sub(r"<img>(.*?)</img>", lambda m: f"*[Figure: {m.group(1).strip()}]*", text, flags=re.S)
    text = re.sub(r"<signature>(.*?)</signature>", lambda m: f"*[Signature: {m.group(1).strip()}]*", text,
                  flags=re.S)
    # Inline HTML that some models use for emphasis / scripts.
    text = re.sub(r"<(?:b|strong)>(.*?)</(?:b|strong)>", r"**\1**", text, flags=re.S)
    text = re.sub(r"<(?:i|em)>(.*?)</(?:i|em)>", r"*\1*", text, flags=re.S)
    text = re.sub(r"<sup>(.*?)</sup>", r"\\(^{\1}\\)", text)
    text = re.sub(r"<sub>(.*?)</sub>", r"\\(_{\1}\\)", text)
    # Put HTML tables on their own lines so the block parser sees them.
    text = re.sub(r"\s*(<table\b)", r"\n\n\1", text, flags=re.I)
    text = re.sub(r"(</table>)\s*", r"\1\n\n", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    return text


def strip_math_delimiters(tex: str) -> str:
    tex = tex.strip()
    if tex.startswith("```"):
        tex = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", tex)
    for a, b in (("\\[", "\\]"), ("$$", "$$"), ("\\(", "\\)"), ("$", "$")):
        if tex.startswith(a) and tex.endswith(b) and len(tex) > len(a) + len(b):
            return tex[len(a) : -len(b)].strip()
    m = re.fullmatch(r"\\begin\{(equation|displaymath)\*?\}(.*)\\end\{\1\*?\}", tex, re.S)
    return m.group(2).strip() if m else tex


# --------------------------------------------------------------------------- converters


def from_markdown(raw: str) -> str:
    return clean_model_output(_normalize_markup(_FRONT_MATTER.sub("", raw)))


def from_latex(raw: str) -> str:
    tex = strip_math_delimiters(_SPECIAL_TOKENS.sub("", raw))
    return f"\\[\n{tex}\n\\]\n"


def from_plain(raw: str) -> str:
    return "\n\n".join(p.strip() for p in re.split(r"\n\s*\n", raw.strip()) if p.strip()) + "\n"


# ---- DocTags (IBM Docling)

_LOC = re.compile(r"<loc_\d+>")
_DOCTAG_LEAF = re.compile(
    r"<(title|section_header_level_(\d)|text|paragraph|caption|footnote|formula|code|list_item|"
    r"page_header|page_footer|otsl|picture|chart|checkbox_selected|checkbox_unselected)>(.*?)</\1>",
    re.S,
)


_DOCTAG_GRID = 500  # DocTags locations are on a 0..500 grid


def from_doctags(raw: str, image_size: tuple[int, int] | None = None) -> str:
    out: list[str] = []
    for m in _DOCTAG_LEAF.finditer(raw):
        tag, level, body = m.group(1), m.group(2), m.group(3)
        if tag in ("page_header", "page_footer"):
            continue
        if tag in ("picture", "chart"):
            cap = re.search(r"<caption>(.*?)</caption>", body, re.S)
            caption = _LOC.sub("", cap.group(1)).strip() if cap else ""
            locs = [int(v) for v in re.findall(r"<loc_(\d+)>", body[: cap.start() if cap else len(body)])[:4]]
            if len(locs) == 4:
                out.append(bbox_marker(caption or "Figure", tuple(v / _DOCTAG_GRID for v in locs)))
            if caption:
                out.append(f"*{caption}*")
            continue
        content = _LOC.sub("", body).strip()
        if tag == "otsl":
            otsl = re.sub(r"<(?:ched|rhed|srow)>", "<fcel>", content)
            table = Table.from_otsl(otsl)
            if table:
                out.append(table.to_markdown())
        elif tag == "formula":
            out.append(f"\\[\n{strip_math_delimiters(content)}\n\\]")
        elif tag == "title":
            out.append(f"# {content}")
        elif level:
            out.append("#" * min(6, int(level) + 1) + " " + content)
        elif tag == "list_item":
            out.append(content if re.match(r"^([-*•]|\d+[.)])\s", content) else f"- {content}")
        elif tag == "code":
            out.append(f"```\n{re.sub(r'^<_[^>]+_>', '', content)}\n```")
        elif tag.startswith("checkbox"):
            out.append(("☑ " if tag.endswith("selected") and "unselected" not in tag else "☐ ") + content)
        else:
            out.append(content)
    if not out:  # not DocTags after all
        return from_markdown(_SPECIAL_TOKENS.sub("", raw))
    return clean_model_output("\n\n".join(out))


# ---- layout JSON (dots.ocr)


def _norm_bbox(bbox, image_size: tuple[int, int] | None):
    """dots.ocr boxes are pixels of the image the model saw; also accept already-normalised boxes."""
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if max(x1, y1, x2, y2) <= 1.0:
        return x1, y1, x2, y2
    if not image_size:
        return None
    w, h = image_size
    return x1 / w, y1 / h, x2 / w, y2 / h


def from_layout_json(raw: str, image_size: tuple[int, int] | None = None) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        data = json.loads(s[min(i for i in (s.find("["), s.find("{")) if i >= 0):])
    except (ValueError, json.JSONDecodeError):
        return from_markdown(raw)  # truncated / not JSON: best effort
    if isinstance(data, dict):
        data = data.get("layout") or data.get("elements") or [data]
    out: list[str] = []
    for el in data if isinstance(data, list) else []:
        if not isinstance(el, dict):
            continue
        cat = str(el.get("category", "Text"))
        text = str(el.get("text") or "").strip()
        if cat == "Picture":
            box = _norm_bbox(el.get("bbox"), image_size)
            if box:
                out.append(bbox_marker("Figure", box))
            continue
        if cat in ("Page-header", "Page-footer") or not text:
            continue
        if cat == "Formula":
            out.append(f"\\[\n{strip_math_delimiters(text)}\n\\]")
        elif cat == "Title":
            out.append(text if text.startswith("#") else f"# {text}")
        elif cat == "Section-header":
            out.append(text if text.startswith("#") else f"## {text}")
        elif cat == "Table":
            out.append(_normalize_markup(text).strip())
        elif cat in ("Caption", "Footnote"):
            out.append(f"*{text}*" if not text.startswith("*") else text)
        else:
            out.append(text)
    return clean_model_output("\n\n".join(out))


CONVERTERS = {
    "markdown": from_markdown,
    "front-matter": from_markdown,
    "latex": from_latex,
    "plain": from_plain,
    "doctags": from_doctags,
    "layout-json": from_layout_json,
}


def to_canonical(raw: str, spec: ModelSpec, task: str, image_size: tuple[int, int] | None = None) -> str:
    """Convert raw model output for ``task`` to canonical Markdown.

    ``image_size`` is the (width, height) of the image the model actually saw; layout formats
    need it to turn pixel boxes into figure markers (see :mod:`scidoc.core.figures`).
    """
    fmt = spec.output_for(task)
    if task == "formula" and fmt == "markdown":
        fmt = "latex"
    # Grounding blocks are made of special tokens: remove them before the generic token strip.
    text = raw if fmt == "doctags" else _SPECIAL_TOKENS.sub("", _DS_GROUNDING.sub("", raw))
    if fmt in ("doctags", "layout-json"):
        return CONVERTERS[fmt](text, image_size)
    return CONVERTERS.get(fmt, from_markdown)(text)
