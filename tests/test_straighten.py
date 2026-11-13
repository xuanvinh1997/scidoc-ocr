"""Frame-based straightening of photographed plots."""

import numpy as np
from PIL import Image, ImageDraw

from scidoc.core.straighten import detect_frame, straighten


def _curved_plot(w=900, h=700, sag=30):
    """A plot frame whose top/bottom edges bow like a curled page, plus labels and hatching."""
    img = Image.new("RGB", (w, h), (205, 205, 200))
    d = ImageDraw.Draw(img)
    x0, x1, y0, y1 = 90, 840, 60, 600

    def bow(x, base, s):
        t = (x - x0) / (x1 - x0)
        return base - s * 4 * t * (1 - t)

    xs = np.arange(x0, x1 + 1)
    d.line([(x, bow(x, y0, sag)) for x in xs], fill="black", width=4)
    d.line([(x, bow(x, y1, -sag * 0.6)) for x in xs], fill="black", width=4)
    d.line([(x0, y0), (x0 + 6, y1)], fill="black", width=4)
    d.line([(x1, y0), (x1 + 10, y1)], fill="black", width=4)
    for k in range(0, 10):  # tick labels and short dashes that must not be taken for edges
        d.text((30, y0 + k * 55), f"-{k * 5}", fill="black")
        d.line((70, y0 + k * 55 + 5, 82, y0 + k * 55 + 5), fill="black", width=2)
    for x in range(x0 + 20, x1 - 20, 14):  # green hatching (not ink)
        d.line((x, 300, x + 40, 560), fill=(90, 120, 95), width=2)
    return img


def test_no_frame_returns_image_unchanged():
    img = Image.new("RGB", (400, 300), "white")
    out, frame = straighten(img)
    assert frame is None and out is img


def test_curved_frame_becomes_straight():
    img = _curved_plot()
    frame = detect_frame(img)
    assert frame is not None and frame.support > 0.6
    out, _ = straighten(img, frame)
    f2 = detect_frame(out)
    assert f2 is not None
    xs = np.linspace(f2.tl[0], f2.tr[0], 40)
    for curve in (f2.top, f2.bottom):
        ys = np.polyval(curve, xs)
        assert np.ptp(ys - np.polyval(np.polyfit(xs, ys, 1), xs)) < 3  # was ~30 px of sag
