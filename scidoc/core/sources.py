"""Load PDFs and images as page images."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = IMAGE_EXTS | PDF_EXTS


@dataclass
class Page:
    index: int
    image: Image.Image
    markdown: str = ""
    raw: str = ""
    status: str = "pending"  # pending | running | done | error | cancelled
    error: str = ""


@dataclass
class Source:
    name: str
    pages: list[Page] = field(default_factory=list)
    digest: str = ""  # content hash used for the result cache
    path: Path | None = None


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def image_digest(img: Image.Image) -> str:
    return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()[:24]


def load_pdf(path: Path, dpi: int = 150, page_range: tuple[int, int] | None = None) -> list[Image.Image]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        n = len(pdf)
        lo, hi = page_range or (1, n)
        out = []
        for i in range(max(1, lo) - 1, min(n, hi)):
            page = pdf[i]
            bitmap = page.render(scale=dpi / 72.0)
            out.append(bitmap.to_pil().convert("RGB"))
            page.close()
        return out
    finally:
        pdf.close()


def load_images(path: Path) -> list[Image.Image]:
    img = Image.open(path)
    frames = []
    try:  # multi-page TIFF / GIF
        i = 0
        while True:
            img.seek(i)
            frames.append(ImageOps.exif_transpose(img.copy()).convert("RGB"))
            i += 1
    except EOFError:
        pass
    return frames or [ImageOps.exif_transpose(img).convert("RGB")]


def open_source(path: str | Path, dpi: int = 150) -> Source:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        imgs = load_pdf(path, dpi)
    elif ext in IMAGE_EXTS:
        imgs = load_images(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return Source(
        name=path.stem,
        pages=[Page(i, im) for i, im in enumerate(imgs)],
        digest=f"{file_digest(path)}-{dpi}",
        path=path,
    )


def source_from_image(img: Image.Image, name: str = "Snip") -> Source:
    img = img.convert("RGB")
    return Source(name=name, pages=[Page(0, img)], digest=image_digest(img))
