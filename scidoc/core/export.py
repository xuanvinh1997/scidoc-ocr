"""Export recognised pages to files. PDF export lives in :mod:`scidoc.core.pdf_export` (needs Qt)."""

from __future__ import annotations

from pathlib import Path

from .document import join_pages, render_latex_document, render_markdown, render_text
from .figures import FigureStore, copy_figures, inline_images
from .render import pages_to_html

FORMATS = {
    "md": ("Markdown (\\( \\) math)", ".md"),
    "md-dollar": ("Markdown ($ math — GitHub/Obsidian)", ".md"),
    "tex": ("LaTeX document", ".tex"),
    "html": ("HTML (self-contained, KaTeX)", ".html"),
    "txt": ("Plain text", ".txt"),
    "pdf": ("Readable PDF (re-typeset)", ".pdf"),
}


def export_text(pages: list[str], fmt: str, title: str = "", figures: FigureStore | None = None) -> str:
    """Render pages; HTML embeds figure images when ``figures`` is given."""
    blocks = join_pages(pages)
    if fmt == "md":
        return render_markdown(blocks)
    if fmt == "md-dollar":
        return render_markdown(blocks, dollar=True)
    if fmt == "tex":
        return render_latex_document(blocks, title=title)
    if fmt == "txt":
        return render_text(blocks)
    if fmt == "html":
        return inline_images(pages_to_html(pages, title=title, standalone=True), figures)
    raise ValueError(f"not a text format: {fmt}")


def export_file(pages: list[str], fmt: str, out: Path, title: str = "",
                figures: FigureStore | None = None) -> Path:
    """Write ``out``. Markdown/LaTeX get a ``figures/`` folder next to them; HTML/PDF embed images."""
    out = Path(out)
    if fmt == "pdf":
        from .pdf_export import html_to_pdf

        html_to_pdf(inline_images(pages_to_html(pages, title=title, standalone=True), figures), out)
    else:
        out.write_text(export_text(pages, fmt, title, figures), encoding="utf-8")
        if fmt in FIGURE_FOLDER_FORMATS:
            copy_figures(pages, figures, out.parent)
    return out


FIGURE_FOLDER_FORMATS = ("md", "md-dollar", "tex")
