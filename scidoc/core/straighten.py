"""Straighten photographed figures (page curl + perspective) using the figure's own frame.

Plots almost always have a frame or axes that should be straight lines. We find the four
frame edges, model each one as a quadratic curve (robust RANSAC fit, so tick labels, digits
and hatching are ignored), and map the curved quadrilateral onto a rectangle with a Coons
patch (bilinearly blended transfinite interpolation). The margins outside the frame
(tick labels, axis title) are warped with the same, extrapolated mapping.

Pure numpy + Pillow: no OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

Curve = np.ndarray  # quadratic coefficients for np.polyval
Point = tuple[float, float]


@dataclass
class Frame:
    top: Curve  # y = top(x)
    bottom: Curve  # y = bottom(x)
    left: Curve  # x = left(y)
    right: Curve  # x = right(y)
    tl: Point
    tr: Point
    bl: Point
    br: Point
    support: float  # fraction of each edge covered by detected line pixels (worst edge)


# --------------------------------------------------------------------------- line detection


def dark_mask(img: Image.Image, level: int = 80) -> np.ndarray:
    """Ink pixels: dark in every channel (black frame lines, not coloured fills or hatching)."""
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    return a.max(axis=2) < level


def _line_samples(mask: np.ndarray, strip: int, fill: float) -> list[tuple[float, float]]:
    """(along, across) centres of horizontal ink runs that fill most of each vertical strip."""
    h, w = mask.shape
    pts = []
    for x0 in range(0, w - strip + 1, strip // 2):
        counts = mask[:, x0:x0 + strip].sum(axis=1)
        rows = np.flatnonzero(counts >= fill * strip)
        if rows.size == 0:
            continue
        # merge consecutive rows into runs, keep each run's centre
        splits = np.flatnonzero(np.diff(rows) > 1) + 1
        for run in np.split(rows, splits):
            pts.append((x0 + strip / 2, float(run.mean())))
    return pts


def _ransac_quadratic(pts: np.ndarray, extent: float, tol: float = 3.0, iters: int = 1500,
                      rng: np.random.Generator | None = None):
    """Best quadratic across ``pts`` (N×2: along, across) → (coeffs, inlier mask)."""
    rng = rng or np.random.default_rng(0)
    n = len(pts)
    if n < 6:
        return None, None
    best, best_score = None, 0.0
    x, y = pts[:, 0], pts[:, 1]
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        xs = np.sort(x[idx])
        # need three distinct, spread-out samples for a stable curve
        if np.ptp(xs) < extent * 0.25 or np.min(np.diff(xs)) < extent * 0.05:
            continue
        c = np.linalg.solve(np.vander(x[idx], 3), y[idx])
        if abs(c[0]) * extent ** 2 > extent * 0.15:  # sag > 15 % of the span: not a frame edge
            continue
        inl = np.abs(np.polyval(c, x) - y) < tol
        # score = how much of the span the inliers cover (not how many points: dense text loses)
        cover = np.unique((x[inl] / (extent / 60)).astype(int)).size
        if cover > best_score:
            best, best_score = inl, cover
    if best is None:
        return None, None
    c = np.polyfit(x[best], y[best], 2)
    inl = np.abs(np.polyval(c, x) - y) < tol
    c = np.polyfit(x[inl], y[inl], 2)
    # Page curl is often asymmetric (flat far from the spine, bending near it): when the edge is
    # well covered, a cubic through the same inliers follows it better than the quadratic.
    if inl.sum() >= 12 and np.ptp(x[inl]) > 0.6 * extent:
        c3 = np.polyfit(x[inl], y[inl], 3)
        if np.abs(np.polyval(c3, x[inl]) - y[inl]).mean() < 0.8 * np.abs(np.polyval(c, x[inl]) - y[inl]).mean():
            c = c3
    return c, inl


def _two_edges(pts: list[tuple[float, float]], extent: float, length: float):
    """The two longest roughly-parallel line curves → (first, second) sorted by position."""
    arr = np.array(pts, dtype=float)
    found = []
    for _ in range(4):
        c, inl = _ransac_quadratic(arr, extent)
        if c is None:
            break
        span = np.ptp(arr[inl, 0]) if inl.sum() > 1 else 0
        found.append((c, arr[inl, 0].min(), arr[inl, 0].max(), span / length))
        arr = arr[~inl]
        if len(arr) < 6:
            break
    long = [f for f in found if f[3] > 0.45]
    if len(long) < 2:
        return None
    long.sort(key=lambda f: f[3], reverse=True)
    a, b = long[0], long[1]
    mid = extent / 2
    if abs(np.polyval(a[0], mid) - np.polyval(b[0], mid)) < 0.25 * length:
        # two hits on the same side (e.g. axis + tick row): look further down the list
        others = [f for f in long[2:] if abs(np.polyval(a[0], mid) - np.polyval(f[0], mid)) >= 0.25 * length]
        if not others:
            return None
        b = others[0]
    return sorted((a, b), key=lambda f: np.polyval(f[0], mid))


def _intersect(h: Curve, v: Curve, guess: Point) -> Point:
    """Point where y = h(x) meets x = v(y) (fixed-point iteration, curves are nearly straight)."""
    x, y = guess
    for _ in range(20):
        y = float(np.polyval(h, x))
        x = float(np.polyval(v, y))
    return x, y


def detect_frame(img: Image.Image, strip: int | None = None) -> Frame | None:
    """Find the plot frame / axes box, or None when the figure has no clear four-sided frame."""
    mask = dark_mask(img)
    h, w = mask.shape
    s = strip or max(16, min(w, h) // 30)
    horiz = _two_edges(_line_samples(mask, s, 0.6), w, w)
    vert = _two_edges(_line_samples(mask.T, s, 0.6), h, h)
    if not horiz or not vert:
        return None
    (top, *_t), (bottom, *_b) = horiz
    (left, *_l), (right, *_r) = vert
    tl = _intersect(top, left, (w * 0.1, h * 0.1))
    tr = _intersect(top, right, (w * 0.9, h * 0.1))
    bl = _intersect(bottom, left, (w * 0.1, h * 0.9))
    br = _intersect(bottom, right, (w * 0.9, h * 0.9))
    if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]):
        return None
    frame = Frame(top, bottom, left, right, tl, tr, bl, br, 0.0)
    # A real frame is one unbroken line on every side. Rows of boxes in a flowchart, or a plot
    # with only two axes, line up into "edges" with gaps: reject them instead of warping.
    edges = [_ink_coverage(mask, line) for line in edge_polylines(frame, 200)]
    frame.support = min(c for c, _ in edges)
    if frame.support < MIN_COVERAGE or max(g for _, g in edges) > MAX_GAP:
        return None
    return frame


MIN_COVERAGE = 0.9  # fraction of each edge that must lie on ink
MAX_GAP = 0.06  # longest break allowed along an edge, as a fraction of its length


def _ink_coverage(mask: np.ndarray, line: list[Point], r: int = 3) -> tuple[float, float]:
    """(fraction of points with ink within ``r`` px, longest run without ink as a fraction)."""
    h, w = mask.shape
    hit = np.array([mask[max(0, int(y) - r):int(y) + r + 1, max(0, int(x) - r):int(x) + r + 1].any()
                    for x, y in line])
    gaps = np.diff(np.flatnonzero(np.r_[True, hit, True])) - 1
    return float(hit.mean()), float(gaps.max() / len(hit))


def distortion(frame: Frame) -> float:
    """Largest distance (px) between the frame and the axis-aligned rectangle it maps to."""
    f = frame
    worst = 0.0
    for line in edge_polylines(f, 40):
        pts = np.array(line)
        horizontal = np.ptp(pts[:, 0]) > np.ptp(pts[:, 1])
        k = 1 if horizontal else 0
        worst = max(worst, float(np.ptp(pts[:, k])))
    return worst


# --------------------------------------------------------------------------- warp


def _coons(frame: Frame, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Source (x, y) for normalised frame coordinates (u, v); u, v may extend beyond [0, 1]."""
    f = frame
    xt = f.tl[0] + u * (f.tr[0] - f.tl[0])
    xb = f.bl[0] + u * (f.br[0] - f.bl[0])
    yl = f.tl[1] + v * (f.bl[1] - f.tl[1])
    yr = f.tr[1] + v * (f.br[1] - f.tr[1])
    T = (xt, np.polyval(f.top, xt))
    B = (xb, np.polyval(f.bottom, xb))
    L = (np.polyval(f.left, yl), yl)
    R = (np.polyval(f.right, yr), yr)
    out = []
    for k in (0, 1):
        corners = ((1 - u) * (1 - v) * f.tl[k] + u * (1 - v) * f.tr[k]
                   + (1 - u) * v * f.bl[k] + u * v * f.br[k])
        out.append((1 - v) * T[k] + v * B[k] + (1 - u) * L[k] + u * R[k] - corners)
    return out[0], out[1]


def _bilinear(src: np.ndarray, x: np.ndarray, y: np.ndarray, fill: np.ndarray) -> np.ndarray:
    h, w = src.shape[:2]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    a = src[y0c, x0c]
    b = src[y0c, x0c + 1]
    c = src[y0c + 1, x0c]
    d = src[y0c + 1, x0c + 1]
    out = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    out[~inside] = fill
    return out


def straighten(img: Image.Image, frame: Frame | None = None) -> tuple[Image.Image, Frame | None]:
    """Return (straightened image, frame used). Unchanged image when no frame is found."""
    frame = frame or detect_frame(img)
    if frame is None:
        return img, None
    src = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = src.shape[:2]
    f = frame
    fw = (np.hypot(f.tr[0] - f.tl[0], f.tr[1] - f.tl[1]) + np.hypot(f.br[0] - f.bl[0], f.br[1] - f.bl[1])) / 2
    fh = (np.hypot(f.bl[0] - f.tl[0], f.bl[1] - f.tl[1]) + np.hypot(f.br[0] - f.tr[0], f.br[1] - f.tr[1])) / 2
    # keep the margins (tick labels, axis title) that surround the frame in the source
    m_left = max(0.0, min(f.tl[0], f.bl[0])) / fw
    m_right = max(0.0, w - max(f.tr[0], f.br[0])) / fw
    m_top = max(0.0, min(f.tl[1], f.tr[1])) / fh
    m_bottom = max(0.0, h - max(f.bl[1], f.br[1])) / fh
    out_w = int(round(fw * (1 + m_left + m_right)))
    out_h = int(round(fh * (1 + m_top + m_bottom)))
    u = (np.arange(out_w, dtype=np.float32) + 0.5) / fw - m_left
    v = (np.arange(out_h, dtype=np.float32) + 0.5) / fh - m_top
    uu, vv = np.meshgrid(u, v)
    sx, sy = _coons(frame, uu, vv)
    border = np.concatenate([src[0], src[-1], src[:, 0], src[:, -1]])
    fill = np.median(border, axis=0)
    out = _bilinear(src, sx, sy, fill)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)), frame


# --------------------------------------------------------------------------- manual handles

# Handle order used by the UI: corners and edge midpoints, clockwise from top-left.
HANDLE_NAMES = ("tl", "tm", "tr", "rm", "br", "bm", "bl", "lm")


def _through3(a: float, b: float, c: float, fa: float, fb: float, fc: float) -> Curve | None:
    """Quadratic f with f(a)=fa, f(b)=fb, f(c)=fc (None when the abscissas collapse)."""
    if not (a < b < c) or min(b - a, c - b) < 1e-3:
        return None
    return np.linalg.solve(np.vander(np.array([a, b, c], dtype=float), 3), np.array([fa, fb, fc], dtype=float))


def frame_from_handles(pts: list[Point]) -> Frame | None:
    """Frame from 8 handles (see HANDLE_NAMES); each edge is the parabola through its three handles."""
    tl, tm, tr, rm, br, bm, bl, lm = pts
    top = _through3(tl[0], tm[0], tr[0], tl[1], tm[1], tr[1])
    bottom = _through3(bl[0], bm[0], br[0], bl[1], bm[1], br[1])
    left = _through3(tl[1], lm[1], bl[1], tl[0], lm[0], bl[0])
    right = _through3(tr[1], rm[1], br[1], tr[0], rm[0], br[0])
    if any(c is None for c in (top, bottom, left, right)):
        return None
    return Frame(top, bottom, left, right, tuple(tl), tuple(tr), tuple(bl), tuple(br), 1.0)


def handles_for(frame: Frame | None, size: tuple[int, int], inset: float = 0.06) -> list[Point]:
    """8 handle positions for ``frame`` (or an inset rectangle when there is none)."""
    w, h = size
    if frame is None:
        x0, y0, x1, y1 = w * inset, h * inset, w * (1 - inset), h * (1 - inset)
        return [(x0, y0), ((x0 + x1) / 2, y0), (x1, y0), (x1, (y0 + y1) / 2),
                (x1, y1), ((x0 + x1) / 2, y1), (x0, y1), (x0, (y0 + y1) / 2)]
    f = frame
    xt = (f.tl[0] + f.tr[0]) / 2
    xb = (f.bl[0] + f.br[0]) / 2
    yl = (f.tl[1] + f.bl[1]) / 2
    yr = (f.tr[1] + f.br[1]) / 2
    return [f.tl, (xt, float(np.polyval(f.top, xt))), f.tr, (float(np.polyval(f.right, yr)), yr),
            f.br, (xb, float(np.polyval(f.bottom, xb))), f.bl, (float(np.polyval(f.left, yl)), yl)]


def edge_polylines(frame: Frame, n: int = 48) -> list[list[Point]]:
    """The four frame edges as polylines (for drawing)."""
    f = frame
    xt = np.linspace(f.tl[0], f.tr[0], n)
    xb = np.linspace(f.bl[0], f.br[0], n)
    yl = np.linspace(f.tl[1], f.bl[1], n)
    yr = np.linspace(f.tr[1], f.br[1], n)
    return [list(zip(xt, np.polyval(f.top, xt))), list(zip(np.polyval(f.right, yr), yr)),
            list(zip(xb, np.polyval(f.bottom, xb))), list(zip(np.polyval(f.left, yl), yl))]


def scaled(frame: Frame, s: float) -> Frame:
    """The same frame for an image scaled by ``s`` (used for fast previews)."""
    pts = [(x * s, y * s) for x, y in handles_for(frame, (0, 0))]
    return frame_from_handles(pts) or frame


def auto_straighten(img: Image.Image) -> Image.Image:
    """Straighten when a confident frame is found, else return ``img`` untouched."""
    frame = detect_frame(img)
    # Flat scans need no warp: resampling would only blur them.
    if frame is None or distortion(frame) < 2.0:
        return img
    return straighten(img, frame)[0]
