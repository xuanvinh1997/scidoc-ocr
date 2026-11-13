"""Figure images cropped from pages: storage, cropping and Markdown wiring (no Qt).

Figures are referenced from canonical Markdown with a path relative to the document's
figure folder base, e.g. ``![Figure 1](figures/p3-fig1.png)``.

OCR converters that know where a picture is (layout JSON, DocTags) emit a *bbox marker*
``![alt](#bbox=x1,y1,x2,y2)`` with coordinates normalised to 0..1. :func:`materialize`
crops those regions from the page image and rewrites the marker into a real image link.
"""

from __future__ import annotations

import base64
import re
import shutil
import threading
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from .cache import cache_root

FIGURES_DIR = "figures"
Box = tuple[float, float, float, float]  # normalised x1, y1, x2, y2

BBOX_MARK_RE = re.compile(r"!\[([^\]]*)\]\(#bbox=([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)\)")
PLACEHOLDER_RE = re.compile(r"\*\[Figure:\s*(.*?)\]\*")
FIGURE_LINK_RE = re.compile(r"!\[[^\]]*\]\((" + FIGURES_DIR + r"/[^)\s]+)\)")
_SRC_RE = re.compile(r'(src=")(' + FIGURES_DIR + r'/[^"]+)(")')


def bbox_marker(alt: str, box: Box) -> str:
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in box)
    alt = alt.replace("]", ")").replace("\n", " ").strip()
    return f"![{alt}](#bbox={x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f})"


class FigureStore:
    """Per-document folder of cropped figures (shared by every model and mode of that document)."""

    def __init__(self, digest: str, root: Path | None = None):
        self.base = (root or cache_root()) / (digest or "_unsaved")
        self.dir = self.base / FIGURES_DIR
        self._lock = threading.Lock()
        self.auto_straighten = False  # straighten photographed plots by their frame when cropping

    def save(self, img: Image.Image, page_index: int) -> str:
        """Store ``img`` as the next free ``p<page>-fig<k>.png``; return the Markdown-relative path."""
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            k = 1
            while (self.dir / f"p{page_index + 1}-fig{k}.png").exists():
                k += 1
            name = f"p{page_index + 1}-fig{k}.png"
            img.convert("RGB").save(self.dir / name)
        return f"{FIGURES_DIR}/{name}"

    def resolve(self, rel: str) -> Path:
        return self.base / rel


def crop(page: Image.Image, box: Box, pad: float = 0.004) -> Image.Image:
    w, h = page.size
    x1, y1, x2, y2 = box
    left, top = max(0, int((x1 - pad) * w)), max(0, int((y1 - pad) * h))
    right, bottom = min(w, int((x2 + pad) * w + 0.5)), min(h, int((y2 + pad) * h + 0.5))
    return page.crop((left, top, max(left + 1, right), max(top + 1, bottom)))


def crop_figure(page: Image.Image, box: Box, store: FigureStore) -> Image.Image:
    img = crop(page, box)
    if store.auto_straighten:
        from .straighten import auto_straighten

        img = auto_straighten(img)
    return img


def reading_order(boxes: Iterable[Box]) -> list[Box]:
    """Top-to-bottom, left column before right column (good enough for 1- and 2-column pages)."""

    def key(b: Box):
        col = 1 if b[0] > 0.45 and b[2] > 0.55 and (b[2] - b[0]) < 0.6 else 0
        return col, b[1], b[0]

    return sorted(boxes, key=key)


def _valid(box: Box) -> bool:
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= 1.0001 and 0 <= y1 < y2 <= 1.0001 and (x2 - x1) * (y2 - y1) > 1e-4


# "Figure 1.8 — …", "Fig. 2:", "Hình 3.", "Рис. 4 –" — but not "Figure 2 shows that …"
CAPTION_RE = re.compile(r"^\s*[*_]{0,2}\s*(?:figure|fig\.?|hình|рис\.?|图|abb\.?|figura)\s*"
                        r"[\dIVX][\w.\-]*[*_]{0,2}\s*[.:—–\-|]", re.I)


def split_blocks(md: str) -> list[str]:
    """Top-level Markdown blocks (blank-line separated, never splitting code fences or display math)."""
    blocks, cur = [], []
    fence = False
    math: str | None = None  # closing delimiter of an open display-math block
    for line in md.split("\n"):
        st = line.strip()
        if math is None and st.startswith("```"):
            fence = not fence
        elif not fence and math is None:
            if st.startswith("\\[") and not (len(st) > 2 and st.endswith("\\]")):
                math = "\\]"
            elif st == "$$":
                math = "$$"
        elif not fence and st.endswith(math):
            math = None
        if not st and not fence and math is None:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _order_key(b: Box):
    col = 1 if b[0] > 0.45 and b[2] > 0.55 and (b[2] - b[0]) < 0.6 else 0
    return col, b[1], b[0]


def _insertion_points(blocks: list[str], pictures: list[Box], text_boxes: list[Box] | None) -> list[int]:
    """For each picture (reading order), the block index to insert it before.

    1. A caption paragraph ("Figure 1.8 — …", "Fig. 2", "Hình 3") right below the picture wins.
    2. Otherwise the share of the page's text (by layout area) that precedes the picture is
       matched against the share of Markdown characters before each block boundary.
    """
    sizes = [len(b) for b in blocks]
    total = sum(sizes) or 1
    cum = [0]
    for n in sizes:
        cum.append(cum[-1] + n)
    captions = [i for i, b in enumerate(blocks) if CAPTION_RE.match(b)]
    texts = [t for t in (text_boxes or []) if _area(t) > 0]
    text_total = sum(_area(t) for t in texts)
    points, last = [], 0
    for pic in pictures:
        cap = next((i for i in captions if i >= last), None)
        if cap is not None:
            idx = cap
            captions.remove(cap)
        else:
            if text_total > 0:
                frac = sum(_area(t) for t in texts if _order_key(t) < _order_key(pic)) / text_total
            else:
                frac = pic[1]
            idx = min(range(len(cum)), key=lambda k: abs(cum[k] / total - frac))
        idx = max(idx, last)
        points.append(idx)
        last = idx
    return points


def materialize(md: str, page: Image.Image, page_index: int, store: FigureStore,
                detected: list[Box] | None = None, text_boxes: list[Box] | None = None) -> str:
    """Turn bbox markers (and, given ``detected`` picture boxes, placeholders) into image links.

    Detected boxes are only used when the page has no figure image yet. They replace the
    model's ``*[Figure: …]*`` placeholders in reading order; without placeholders each picture
    is inserted where it sits on the page (before its caption, or by the text that precedes it,
    using ``text_boxes`` from the layout model).
    """

    def from_marker(m: re.Match) -> str:
        box = tuple(float(m.group(i)) for i in range(2, 6))
        if not _valid(box):
            return ""
        return f"![{m.group(1) or 'Figure'}]({store.save(crop_figure(page, box, store), page_index)})"

    md = BBOX_MARK_RE.sub(from_marker, md)
    boxes = [b for b in (detected or []) if _valid(b)]
    if not boxes or FIGURE_LINK_RE.search(md):
        return md
    queue = reading_order(boxes)
    if PLACEHOLDER_RE.search(md):

        def from_placeholder(m: re.Match) -> str:
            if not queue:
                return m.group(0)
            rel = store.save(crop_figure(page, queue.pop(0), store), page_index)
            return f"\n\n![{m.group(1).strip() or 'Figure'}]({rel})\n\n"

        md = PLACEHOLDER_RE.sub(from_placeholder, md)
        return re.sub(r"\n{3,}", "\n\n", md)
    blocks = split_blocks(md)
    inserts: dict[int, list[str]] = {}
    for pic, at in zip(queue, _insertion_points(blocks, queue, text_boxes)):
        inserts.setdefault(at, []).append(f"![Figure]({store.save(crop_figure(page, pic, store), page_index)})")
    out: list[str] = []
    for i in range(len(blocks) + 1):
        out.extend(inserts.get(i, []))
        if i < len(blocks):
            out.append(blocks[i])
    return "\n\n".join(out) + "\n"


def insert_figure(md: str, link: str, position: int | None = None) -> str:
    """Put an image link into a page: replace the first placeholder, else insert at ``position`` (default end)."""
    m = PLACEHOLDER_RE.search(md)
    if m:
        out = md[: m.start()] + f"\n\n{link}\n\n" + md[m.end():]
    else:
        pos = len(md) if position is None else max(0, min(len(md), position))
        out = md[:pos].rstrip("\n") + ("\n\n" if md[:pos].strip() else "") + link + "\n\n" + md[pos:].lstrip("\n")
    return re.sub(r"\n{3,}", "\n\n", out)


# --------------------------------------------------------------------------- export helpers


def referenced(pages: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for md in pages:
        for m in FIGURE_LINK_RE.finditer(md):
            seen.setdefault(m.group(1))
    return list(seen)


def copy_figures(pages: Iterable[str], store: FigureStore | None, out_dir: Path) -> int:
    """Copy every figure the pages reference into ``out_dir/figures`` (links stay valid)."""
    if store is None:
        return 0
    n = 0
    for rel in referenced(pages):
        src = store.resolve(rel)
        if src.exists():
            dst = Path(out_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            n += 1
    return n


def inline_images(html: str, store: FigureStore | None) -> str:
    """Embed figure images as data URIs (self-contained HTML / PDF)."""
    if store is None:
        return html

    def repl(m: re.Match) -> str:
        p = store.resolve(m.group(2))
        if not p.exists():
            return m.group(0)
        return m.group(1) + "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() + m.group(3)

    return _SRC_RE.sub(repl, html)


def file_urls(html: str, store: FigureStore | None) -> str:
    """Point figure ``src`` attributes at the files on disk (live preview)."""
    if store is None:
        return html
    base = store.base.as_uri().rstrip("/")
    return _SRC_RE.sub(lambda m: f"{m.group(1)}{base}/{m.group(2)}{m.group(3)}", html)
