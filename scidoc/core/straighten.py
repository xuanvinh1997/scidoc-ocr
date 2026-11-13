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
    support = min(horiz[0][3], horiz[1][3], vert[0][3], vert[1][3])
    return Frame(top, bottom, left, right, tl, tr, bl, br, support)


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
