"""Readable-PDF export: render HTML+KaTeX in Qt WebEngine (Chromium) and print to PDF.

Works inside the GUI (non-blocking via callback) and from the CLI (blocking).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QEventLoop, QMarginsF, QTimer, QUrl
from PySide6.QtGui import QPageLayout, QPageSize


def _ensure_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
        app = QApplication(["scidoc"])
    return app


class PdfPrinter:
    """Keeps the page alive until printing finishes."""

    _alive: set["PdfPrinter"] = set()

    def __init__(self, html: str, out: Path, done: Callable[[bool, str], None], page_size: str = "A4"):
        from PySide6.QtWebEngineCore import QWebEnginePage

        self.out = Path(out)
        self.done = done
        self._tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8")
        self._tmp.write(html)
        self._tmp.close()
        self.page = QWebEnginePage()
        self.page.loadFinished.connect(self._loaded)
        self.page.pdfPrintingFinished.connect(self._printed)
        size = QPageSize(QPageSize.PageSizeId.Letter if page_size.lower() == "letter" else QPageSize.PageSizeId.A4)
        self.layout = QPageLayout(size, QPageLayout.Orientation.Portrait, QMarginsF(18, 18, 18, 18),
                                  QPageLayout.Unit.Millimeter)
        PdfPrinter._alive.add(self)
        self.page.load(QUrl.fromLocalFile(self._tmp.name))

    def _loaded(self, ok: bool) -> None:
        if not ok:
            self._finish(False, "failed to load HTML")
            return
        self._poll_fonts(0)

    def _poll_fonts(self, tries: int) -> None:
        def check(status):
            if status == "loaded" or tries > 50:
                self.page.printToPdf(str(self.out), self.layout)
            else:
                QTimer.singleShot(100, lambda: self._poll_fonts(tries + 1))

        self.page.runJavaScript("document.fonts ? document.fonts.status : 'loaded'", 0, check)

    def _printed(self, path: str, ok: bool) -> None:
        self._finish(ok, path if ok else "printToPdf failed")

    def _finish(self, ok: bool, msg: str) -> None:
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass
        PdfPrinter._alive.discard(self)
        self.page.deleteLater()
        self.done(ok, msg)


def html_to_pdf_async(html: str, out: Path, done: Callable[[bool, str], None], page_size: str = "A4") -> None:
    PdfPrinter(html, out, done, page_size)


def html_to_pdf(html: str, out: Path, page_size: str = "A4", timeout_ms: int = 120_000) -> Path:
    """Blocking variant (CLI)."""
    _ensure_app()
    loop = QEventLoop()
    result: dict = {}

    def done(ok, msg):
        result.update(ok=ok, msg=msg)
        loop.quit()

    html_to_pdf_async(html, out, done, page_size)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    if not result.get("ok"):
        raise RuntimeError(f"PDF export failed: {result.get('msg', 'timeout')}")
    return Path(out)
