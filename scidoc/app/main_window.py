"""SciDoc OCR main window."""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QBuffer, QIODevice, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QImage, QKeySequence, QPen, QPixmap, QShortcut
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QHBoxLayout,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.backends import Task
from ..core.cache import PageCache
from ..core.models import DEFAULT_MODEL_ID, Registry, local_status, remote_size
from ..core.document import clean_model_output, parse_blocks, render_html_body, render_latex_body
from ..core.export import FIGURE_FOLDER_FORMATS, FORMATS, export_text
from ..core.figures import PLACEHOLDER_RE, FigureStore, copy_figures, file_urls, inline_images, insert_figure
from ..core.layout import LayoutDetector, available as layout_available
from ..core.pdf_export import html_to_pdf_async
from ..core.render import ASSETS, html_document, pages_to_html
from ..core.sources import SUPPORTED_EXTS, Source, open_source, source_from_image
from .downloads import DownloadManager
from .model_manager import ModelManagerDialog, fmt_bytes
from .settings import AppSettings, SettingsDialog
from .snip import SnipOverlay
from .ui import (
    STATUS_ROLE,
    FloatingBar,
    Segmented,
    ThumbDelegate,
    Welcome,
    icon,
    overlay,
    panel_header,
    refresh_icons,
    set_icon,
    theme,
    tool_button,
)
from .worker import Job, OCRWorker, start_worker

TASK_LABELS = {"document": "Full page", "formula": "Formula", "text": "Text", "table": "Table"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- helpers


def pil_to_qimage(img: Image.Image) -> QImage:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    return QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888).copy()


def qimage_to_pil(qimg: QImage) -> Image.Image:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    qimg.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGB")


def is_dark() -> bool:
    return theme().dark


@dataclass
class DocState:
    id: int
    source: Source
    task: Task
    model_id: str
    cache: PageCache = None  # type: ignore[assignment]
    figures: FigureStore = None  # type: ignore[assignment]

    def __post_init__(self):
        self.figures = FigureStore(self.source.digest)

    def make_cache(self) -> PageCache:
        # Xiaomi-OCR-0 results keep the original <digest>/<task> layout for backward compatibility.
        sub = self.task.value if self.model_id == DEFAULT_MODEL_ID else f"{self.model_id}/{self.task.value}"
        self.cache = PageCache(self.source.digest, sub)
        return self.cache

    def load_cached(self) -> None:
        """(Re)load results of the current model+task from disk; reset pages without results."""
        self.make_cache()
        for p in self.source.pages:
            if p.status == "running":
                continue
            hit = self.cache.get(p.index)
            if hit:
                p.markdown, p.raw = hit
                p.status, p.error = "done", ""
            else:
                p.markdown, p.raw, p.status, p.error = "", "", "pending", ""


# --------------------------------------------------------------------------- widgets


class PageView(QGraphicsView):
    """Zoomable page image (Ctrl/⌘ + wheel to zoom) with an optional crop selection.

    In crop mode a drag draws a rectangle in image pixels; ``selectionMade`` reports it and a
    floating action bar (set with :meth:`set_crop_bar`) follows the selection.
    """

    selectionMade = Signal(QRectF)
    cropCancelled = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.item)
        self.sel = QGraphicsRectItem()
        self.sel.setZValue(1)
        self.sel.hide()
        self.scene().addItem(self.sel)
        self.setObjectName("PageView")
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._fit = True
        self._crop = False
        self._origin = None
        self._bar: QWidget | None = None
        self.restyle()

    def restyle(self):
        t = theme()
        pen = QPen(QColor(t.accent), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self.sel.setPen(pen)
        fill = QColor(t.accent)
        fill.setAlpha(28)
        self.sel.setBrush(fill)

    def set_image(self, img: Image.Image | None):
        self.item.setPixmap(QPixmap.fromImage(pil_to_qimage(img)) if img else QPixmap())
        self.scene().setSceneRect(self.item.boundingRect())
        self.clear_selection()
        self._fit = True
        self.fit_width()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit_width()
        self._place_bar()

    def zoom(self, factor: float):
        self._fit = False
        self.scale(factor, factor)
        self._place_bar()

    def fit_width(self):
        self._fit = True
        if self.item.pixmap().isNull():
            return
        self.resetTransform()
        w = self.item.pixmap().width()
        s = (self.viewport().width() - 24) / max(1, w)
        self.scale(s, s)
        self._place_bar()

    def wheelEvent(self, e):
        if e.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            self._fit = False
            f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
            self.scale(f, f)
            self._place_bar()
        else:
            super().wheelEvent(e)

    def scrollContentsBy(self, dx, dy):  # noqa: N802
        super().scrollContentsBy(dx, dy)
        self._place_bar()

    # ---- crop selection
    def set_crop_bar(self, bar: QWidget):
        self._bar = bar
        bar.setParent(self.viewport())
        bar.hide()

    def set_crop_mode(self, on: bool):
        self._crop = on
        self.setDragMode(QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag)
        if on:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.setFocus()
        else:
            self.viewport().unsetCursor()
            self.clear_selection()

    def selection(self) -> QRectF | None:
        return self.sel.rect() if self.sel.isVisible() else None

    def clear_selection(self):
        self.sel.hide()
        if self._bar:
            self._bar.hide()

    def _scene_point(self, pos):
        p = self.mapToScene(pos)
        r = self.item.boundingRect()
        p.setX(min(max(p.x(), r.left()), r.right()))
        p.setY(min(max(p.y(), r.top()), r.bottom()))
        return p

    def mousePressEvent(self, e):
        if self._crop and e.button() == Qt.MouseButton.LeftButton and not self.item.pixmap().isNull():
            self._origin = self._scene_point(e.position().toPoint())
            self.sel.setRect(QRectF(self._origin, self._origin))
            self.sel.show()
            if self._bar:
                self._bar.hide()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._origin is not None:
            self.sel.setRect(QRectF(self._origin, self._scene_point(e.position().toPoint())).normalized())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._origin is not None:
            self._origin = None
            r = self.sel.rect()
            if r.width() < 8 or r.height() < 8:
                self.clear_selection()
            else:
                self._place_bar()
                self.selectionMade.emit(r)
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if self._crop and e.key() == Qt.Key.Key_Escape:
            if self.sel.isVisible():
                self.clear_selection()
            else:
                self.cropCancelled.emit()
            e.accept()
            return
        super().keyPressEvent(e)

    def _place_bar(self):
        bar = self._bar
        if not bar or not self.sel.isVisible() or self._origin is not None:
            return
        bar.adjustSize()
        r = self.mapFromScene(self.sel.rect()).boundingRect()
        vp = self.viewport().rect()
        x = min(max(4, r.right() - bar.width()), vp.width() - bar.width() - 4)
        y = r.top() - bar.height() - 8
        if y < 4:
            y = min(r.bottom() + 8, vp.height() - bar.height() - 4)
        bar.move(x, max(4, y))
        bar.show()
        bar.raise_()


def _image_files(md) -> list[str]:
    if not md.hasUrls():
        return []
    return [u.toLocalFile() for u in md.urls() if u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in IMAGE_EXTS]


class MarkdownEditor(QPlainTextEdit):
    """Markdown editor that turns pasted / dropped images into figure links (via ``imageInserted``)."""

    imageInserted = Signal(QImage)

    def canInsertFromMimeData(self, md):  # noqa: N802
        return md.hasImage() or bool(_image_files(md)) or super().canInsertFromMimeData(md)

    def insertFromMimeData(self, md):  # noqa: N802
        if md.hasImage():
            self.imageInserted.emit(QImage(md.imageData()))
            return
        files = _image_files(md)
        if files:
            for f in files:
                self.imageInserted.emit(QImage(f))
            return
        super().insertFromMimeData(md)

    def dragEnterEvent(self, e):
        # Documents (PDFs…) dropped on the editor are opened by the window, not inserted as text.
        if e.mimeData().hasUrls() and not _image_files(e.mimeData()):
            e.ignore()
            return
        super().dragEnterEvent(e)


class PreviewPage(QWebEnginePage):
    def __init__(self, on_copy, parent=None):
        super().__init__(parent)
        self.on_copy = on_copy

    def javaScriptConsoleMessage(self, level, message, line, source):  # noqa: N802
        if message.startswith("SCIDOC_COPY:"):
            self.on_copy(message[len("SCIDOC_COPY:") :])

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):  # noqa: N802
        # Never navigate away from the preview (links in OCR output open externally).
        if nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(url)
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class Preview(QWebEngineView):
    """KaTeX preview. The shell page is loaded once; bodies are swapped in via JS."""

    def __init__(self, on_copy_tex):
        super().__init__()
        self.setPage(PreviewPage(on_copy_tex, self))
        self._ready = False
        self._pending: str | None = None
        self.loadFinished.connect(self._on_loaded)
        self.reload_shell()

    def reload_shell(self):
        self._ready = False
        shell = html_document("", dark=is_dark()).replace(
            "if (el && window.scidocBridge) window.scidocBridge(el.getAttribute('data-tex'));",
            "if (el) console.log('SCIDOC_COPY:' + el.getAttribute('data-tex'));",
        )
        self.setHtml(shell, QUrl.fromLocalFile(str(ASSETS) + "/"))

    def _on_loaded(self, ok):
        self._ready = True
        if self._pending is not None:
            body, self._pending = self._pending, None
            self.set_body(body)

    def set_body(self, body: str, keep_scroll: bool = True):
        if not self._ready:
            self._pending = body
            return
        js = (
            "(function(){var y=window.scrollY;document.body.innerHTML=%s;scidocRender(document);"
            "window.scrollTo(0,%s);})();" % (json.dumps(body), "y" if keep_scroll else "0")
        )
        self.page().runJavaScript(js)

    def to_mathml(self, tex: str, callback):
        self.page().runJavaScript(f"scidocToMathML({json.dumps(tex)}, true)", 0, callback)


# --------------------------------------------------------------------------- main window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SciDoc OCR")
        self.resize(1400, 880)
        self.setAcceptDrops(True)
        self.settings = AppSettings.load()
        self.registry = Registry()
        if not any(m.id == self.settings.doc_model for m in self.registry.all()):
            self.settings.doc_model = DEFAULT_MODEL_ID
        self.downloads = DownloadManager(self)
        self.downloads.progress.connect(self._on_download_progress)
        self.downloads.finished.connect(self._on_download_finished)
        self._backends: dict[str, tuple[tuple, object]] = {}  # model_id -> (settings key, backend)
        self._waiting_download: str | None = None
        self.docs: dict[int, DocState] = {}
        self._next_id = 1
        self.cur_doc: int | None = None
        self.cur_page = 0
        self.worker: OCRWorker | None = None
        self.thread = None
        self.queue: list[tuple[str, list[Job]]] = []  # (model_id, jobs)
        self._stream: dict[tuple[int, int], str] = {}
        self._loading_editor = False
        self._run_total = 0
        self._run_done = 0
        self._autocopy: set[int] = set()
        self._t0 = time.time()

        self._build_ui()
        self._build_actions()
        self._refresh_engine_label()
        self._show_empty()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # ---- pages rail
        self.page_list = QListWidget(objectName="Thumbs")
        self.page_list.setViewMode(QListWidget.ViewMode.ListMode)
        self.page_list.setItemDelegate(ThumbDelegate(self.page_list))
        self.page_list.setIconSize(QSize(ThumbDelegate.THUMB_W, ThumbDelegate.THUMB_H))
        self.page_list.setUniformItemSizes(True)
        self.page_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.page_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.page_list.currentRowChanged.connect(self._on_page_selected)
        self.pages_count = QLabel(objectName="PanelTitle")
        sidebar = QWidget()
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        sl.addWidget(panel_header(None, self.pages_count, title="Pages"))
        sl.addWidget(self.page_list)
        sidebar.setMinimumWidth(ThumbDelegate.THUMB_W + 32)

        # ---- original page: zoom + crop in the header, page navigation floating over the image
        self.page_view = PageView()
        self.page_view.selectionMade.connect(lambda _r: self.status_label.setText(
            "Insert the selection into the result, copy it or save it."))
        self.page_view.cropCancelled.connect(lambda: self.a_crop.setChecked(False))
        self.page_label = QLabel("–", objectName="PageNum")
        self.page_label.setMinimumWidth(64)
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav = FloatingBar()
        nav.add(tool_button("prev", "Previous page (PgUp)", lambda: self._step_page(-1)))
        nav.add(self.page_label)
        nav.add(tool_button("next", "Next page (PgDn)", lambda: self._step_page(1)))

        crop_bar = FloatingBar()
        insert = tool_button("image", "Insert the selected region into the result as an image",
                             self.insert_selection, text="Insert into result", role="white", object_name="Accent")
        crop_bar.add(insert)
        crop_bar.add(tool_button("copy", "Copy the selected region as an image", self.copy_selection, text="Copy"))
        crop_bar.add(tool_button("save", "Save the selected region as PNG", self.save_selection, text="Save…"))
        self.page_view.set_crop_bar(crop_bar)

        self.a_crop = QAction("Crop image", self, checkable=True)
        set_icon(self.a_crop, "crop")
        self.a_crop.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.a_crop.setToolTip("Select a figure on the page to insert it as an image  "
                               f"({self.a_crop.shortcut().toString(QKeySequence.SequenceFormat.NativeText)})")
        self.a_crop.toggled.connect(self._on_crop_toggled)
        crop_btn = QToolButton(objectName="Ghost")
        crop_btn.setDefaultAction(self.a_crop)
        crop_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        crop_btn.setIconSize(QSize(14, 14))

        fit = tool_button("fit", "Fit width", self.page_view.fit_width)
        page_hdr = panel_header(
            None, crop_btn, _vline(),
            tool_button("minus", "Zoom out", lambda: self.page_view.zoom(1 / 1.2)),
            fit,
            tool_button("plus", "Zoom in", lambda: self.page_view.zoom(1.2)),
            title="Original",
        )
        page_box = QWidget()
        pl = QVBoxLayout(page_box)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)
        pl.addWidget(page_hdr)
        pl.addWidget(overlay(self.page_view, nav, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter), 1)

        # ---- output: preview / markdown editor
        self.preview = Preview(self._copy_formula_text)
        self.editor = MarkdownEditor(objectName="Editor")
        self.editor.setPlaceholderText("Recognised Markdown + LaTeX appears here. Edit freely — the preview updates live.")
        f = self.editor.font()
        f.setFamilies(["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"])
        f.setPointSize(12)
        self.editor.setFont(f)
        self.editor.textChanged.connect(self._on_editor_changed)
        self.editor.imageInserted.connect(self._on_editor_image)
        self._edit_timer = QTimer(self, singleShot=True, interval=350, timeout=self._commit_edit)
        self._stream_timer = QTimer(self, singleShot=True, interval=400, timeout=self._render_stream)
        editor_box = QWidget()
        el = QVBoxLayout(editor_box)
        el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(0)
        el.addWidget(self.editor, 1)
        el.addWidget(QLabel("Paste or drop an image here to insert it at the cursor.", objectName="Hint"))
        self.editor_box = editor_box

        self.view_mode = Segmented([("Preview", "preview"), ("Split", "split"), ("Markdown", "markdown")])
        self.view_mode.changed.connect(self._apply_view_mode)
        self.scope = Segmented([("Page", "page"), ("Document", "doc")])
        self.scope.changed.connect(lambda _: self._refresh_preview(keep_scroll=False))
        copy_btn = tool_button("copy", "Copy results", text="Copy", object_name="Ghost")
        copy_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        cm = QMenu(copy_btn)
        cm.addAction("Page as Markdown", lambda: self.copy_page("md"))
        cm.addAction("Page as LaTeX", lambda: self.copy_page("tex"))
        cm.addSeparator()
        cm.addAction("Formula as LaTeX", lambda: self.copy_page("formula"))
        cm.addAction("Formula as MathML (Word)", lambda: self.copy_page("mathml"))
        cm.addSeparator()
        cm.addAction("Document as Markdown", lambda: self.copy_doc("md"))
        cm.addAction("Document as LaTeX", lambda: self.copy_doc("tex"))
        copy_btn.setMenu(cm)
        out_hdr = panel_header(self.view_mode, None, self.scope, copy_btn)
        out_hdr.setToolTip("Double-click an equation in the preview to copy its LaTeX")

        self.out_split = QSplitter(Qt.Orientation.Vertical)
        self.out_split.addWidget(self.preview)
        self.out_split.addWidget(editor_box)
        self.out_split.setSizes([520, 340])
        self.out_split.setChildrenCollapsible(False)
        out_box = QWidget()
        ol = QVBoxLayout(out_box)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(0)
        ol.addWidget(out_hdr)
        ol.addWidget(self.out_split, 1)

        work = QSplitter(Qt.Orientation.Horizontal)
        work.addWidget(sidebar)
        work.addWidget(page_box)
        work.addWidget(out_box)
        work.setSizes([ThumbDelegate.THUMB_W + 40, 620, 700])
        work.setStretchFactor(1, 1)
        work.setStretchFactor(2, 1)
        work.setCollapsible(1, False)
        work.setCollapsible(2, False)

        # ---- document tabs: they live in the top bar (see _build_actions)
        self.doc_tabs = QTabBar(objectName="DocTabs")
        self.doc_tabs.setMovable(True)
        self.doc_tabs.setDocumentMode(True)
        self.doc_tabs.setDrawBase(False)
        self.doc_tabs.setExpanding(False)
        self.doc_tabs.setUsesScrollButtons(True)
        self.doc_tabs.setElideMode(Qt.TextElideMode.ElideMiddle)
        self.doc_tabs.setIconSize(QSize(14, 14))
        self.doc_tabs.currentChanged.connect(self._on_doc_selected)

        self.welcome = Welcome(self.open_files, self.start_snip, self.paste_image, self.open_models)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.welcome)
        self.stack.addWidget(work)
        self.setCentralWidget(self.stack)
        self._apply_view_mode("preview")

        # ---- status bar: message · progress · engine
        self.status_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(120, 4)
        self.progress_label = QLabel()
        self.engine_label = QLabel()
        self.engine_label.setTextFormat(Qt.TextFormat.RichText)
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        sb.addWidget(self.status_label, 1)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.progress_label)
        sb.addPermanentWidget(self.engine_label)
        self._show_progress(False)

    def _show_progress(self, on: bool):
        self.progress.setVisible(on)
        self.progress_label.setVisible(on)

    def _apply_view_mode(self, mode):
        self.preview.setVisible(mode in ("preview", "split"))
        self.editor_box.setVisible(mode in ("markdown", "split"))
        self.scope.setEnabled(mode != "markdown")

    def _step_page(self, delta: int):
        if self.doc:
            row = max(0, min(len(self.doc.source.pages) - 1, self.cur_page + delta))
            self.page_list.setCurrentRow(row)

    def _build_actions(self):
        tb = QToolBar("Main", objectName="MainToolbar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, tip=None, icon_name=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if icon_name:
                set_icon(a, icon_name)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            sc = (f"  ({QKeySequence(shortcut).toString(QKeySequence.SequenceFormat.NativeText)})"
                  if shortcut else "")
            a.setToolTip((tip or text.rstrip("…")) + sc)
            return a

        self.a_open = act("Open…", self.open_files, QKeySequence.StandardKey.Open, "Open PDFs or images", "open")
        self.a_snip = act("Snip screen", self.start_snip, "Ctrl+Alt+M", "Capture a region of the screen", "snip")
        self.a_paste = act("Paste image", self.paste_image, "Ctrl+Shift+V", "OCR an image from the clipboard", "paste")
        self.a_run = act("Run", self.run_current, "Ctrl+R", "Recognise all unprocessed pages", "run")
        self.a_rerun = act("Re-run current page", self.rerun_page, "Ctrl+Shift+R", None, "rerun")
        self.a_rerun_all = act("Re-run all pages", self.rerun_all, None, None)
        self.a_stop = act("Stop", self.stop, "Esc", "Cancel recognition", "stop")
        self.a_settings = act("Settings", self.open_settings, QKeySequence.StandardKey.Preferences, None, "settings")
        self.a_models = act("Manage models…", self.open_models, "Ctrl+Shift+M",
                            "Download, choose and add OCR models", "models")

        # logo + app name (the name only shows on the empty state, where there are no tabs)
        logo = QLabel("∑", objectName="Logo")
        logo.setFixedSize(26, 26)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand = QWidget()
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(4, 0, 4, 0)
        bl.setSpacing(10)
        bl.addWidget(logo)
        self.app_name = QLabel("SciDoc OCR", objectName="AppName")
        bl.addWidget(self.app_name)
        tb.addWidget(brand)

        # New ▾ : open / snip / paste
        new_btn = QToolButton(objectName="Ghost")
        new_btn.setText("New")
        set_icon(new_btn, "plus")
        new_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        new_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        nm = QMenu(new_btn)
        for a in (self.a_open, self.a_snip, self.a_paste):
            nm.addAction(a)
        new_btn.setMenu(nm)
        self._new_action = tb.addWidget(new_btn)
        self._sep_action = tb.addSeparator()

        # document tabs take the free space
        tabs_host = QWidget()
        th = QHBoxLayout(tabs_host)
        th.setContentsMargins(0, 0, 0, 0)
        th.addWidget(self.doc_tabs)
        th.addStretch(1)
        tabs_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(tabs_host)

        # model · mode chip right next to Run
        self.mode_btn = QToolButton(objectName="Ghost")
        set_icon(self.mode_btn, "models", "muted")
        self.mode_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.mode_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.mode_btn.setToolTip("OCR model and what to recognise (the model is also the default for new documents)")
        self.mode_menu = QMenu(self.mode_btn)
        self.mode_menu.setToolTipsVisible(True)
        self.mode_btn.setMenu(self.mode_menu)
        self._model_group: QActionGroup | None = None
        self._task_group: QActionGroup | None = None
        self._fill_model_menu()
        self._mode_action = tb.addWidget(self.mode_btn)

        self.run_btn = QToolButton(objectName="Primary")
        self.run_btn.setDefaultAction(self.a_run)
        set_icon(self.run_btn, "run", "white")
        self.run_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.run_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        rm = QMenu(self.run_btn)
        rm.addAction(self.a_rerun)
        rm.addAction(self.a_rerun_all)
        self.run_btn.setMenu(rm)
        self.stop_btn = QToolButton(objectName="Danger")
        self.stop_btn.setDefaultAction(self.a_stop)
        set_icon(self.stop_btn, "stop", "white")
        self.stop_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._run_action = tb.addWidget(self.run_btn)
        self._stop_action = tb.addWidget(self.stop_btn)

        export_btn = QToolButton(objectName="Ghost")
        export_btn.setText("Export")
        set_icon(export_btn, "export")
        export_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        em = QMenu("Export", export_btn)
        for key, (label, _ext) in FORMATS.items():
            em.addAction(label, lambda k=key: self.export(k))
        export_btn.setMenu(em)
        self.export_btn = export_btn
        self._export_action = tb.addWidget(export_btn)

        settings_btn = QToolButton(objectName="Flat")
        settings_btn.setDefaultAction(self.a_settings)
        settings_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tb.addWidget(settings_btn)
        self._running = False
        self._set_running(False)

        # Ctrl/⌘+V pastes into the editor when it has focus (images become figures), else OCRs the clipboard.
        sc = QShortcut(QKeySequence.StandardKey.Paste, self)
        sc.activated.connect(lambda: self.editor.paste() if self.editor.hasFocus() else self.paste_image())
        QShortcut(QKeySequence(Qt.Key.Key_PageDown), self, lambda: self._step_page(1))
        QShortcut(QKeySequence(Qt.Key.Key_PageUp), self, lambda: self._step_page(-1))

        # menu bar (top of screen on macOS)
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        fm.addAction(self.a_open)
        fm.addMenu(em)
        fm.addSeparator()
        fm.addAction(self.a_settings)
        fm.addAction(act("Quit", self.close, QKeySequence.StandardKey.Quit))
        om = mb.addMenu("&OCR")
        for a in (self.a_snip, self.a_paste, None, self.a_run, self.a_rerun, self.a_rerun_all, self.a_stop, None,
                  self.a_crop, None, self.a_models):
            om.addSeparator() if a is None else om.addAction(a)
        vm = mb.addMenu("&View")
        for label, mode, key in (("Preview", "preview", "Ctrl+1"), ("Split", "split", "Ctrl+2"),
                                 ("Markdown", "markdown", "Ctrl+3")):
            vm.addAction(act(label, lambda _=False, m=mode: (self.view_mode.setCurrentData(m),
                                                             self._apply_view_mode(m)), key))
        hm = mb.addMenu("&Help")
        hm.addAction(act("About SciDoc OCR", self._about))

    def _set_running(self, running: bool):
        self._running = running
        self.a_stop.setEnabled(running)
        self.a_run.setEnabled(not running)
        self._sync_toolbar()

    def _sync_toolbar(self):
        """Document controls only make sense once a document is open."""
        has_docs = bool(self.docs)
        self.app_name.setVisible(not has_docs)
        self._mode_action.setVisible(has_docs)
        self._export_action.setVisible(has_docs)
        self._run_action.setVisible(has_docs and not self._running)
        self._stop_action.setVisible(self._running)

    # ------------------------------------------------------------------ documents
    def _add_doc(self, source: Source, task: Task, model_id: str | None = None) -> DocState:
        doc = DocState(self._next_id, source, task, model_id or self.settings.doc_model)
        self._next_id += 1
        self.docs[doc.id] = doc
        doc.load_cached()
        self.stack.setCurrentIndex(1)
        # addTab() emits currentChanged before the tab carries its doc id: wire it up silently,
        # then select explicitly.
        self.doc_tabs.blockSignals(True)
        i = self.doc_tabs.addTab(icon("doc", theme().muted, 14), source.name)
        self.doc_tabs.setTabData(i, doc.id)
        self.doc_tabs.setTabToolTip(i, f"{source.path or source.name} · {len(source.pages)} page(s)")
        close = tool_button("close", "Close document", role="muted", object_name="TabClose")
        close.setIconSize(QSize(12, 12))
        close.setFixedSize(18, 18)
        close.clicked.connect(lambda _=False, d=doc.id: self._close_doc(self._tab_of(d)))
        self.doc_tabs.setTabButton(i, QTabBar.ButtonPosition.RightSide, close)
        self.doc_tabs.setCurrentIndex(i)
        self.doc_tabs.blockSignals(False)
        self._on_doc_selected(i)
        self._sync_toolbar()
        return doc

    def _tab_of(self, doc_id: int) -> int:
        return next((i for i in range(self.doc_tabs.count()) if self.doc_tabs.tabData(i) == doc_id), -1)

    def open_files(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(self, "Open documents", "", f"Documents ({exts});;All files (*)")
        self.open_paths(paths)

    def open_paths(self, paths):
        for path in paths:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.status_label.setText(f"Rendering {Path(path).name}…")
                QApplication.processEvents()
                src = open_source(path, dpi=self.settings.dpi)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Cannot open", f"{path}\n\n{e}")
                continue
            finally:
                QApplication.restoreOverrideCursor()
            doc = self._add_doc(src, Task.DOCUMENT)
            self.status_label.setText(f"Opened {src.name} ({len(src.pages)} pages)")
            if self.settings.auto_run:
                self._run_doc(doc)

    def _close_doc(self, index: int):
        if index < 0:
            return
        doc_id = self.doc_tabs.tabData(index)
        self.docs.pop(doc_id, None)
        self._autocopy.discard(doc_id)
        self.doc_tabs.removeTab(index)
        if not self.docs:
            self.cur_doc = None
            self.page_list.clear()
            self.a_crop.setChecked(False)
            self._show_empty()
        self._sync_toolbar()

    @property
    def doc(self) -> DocState | None:
        return self.docs.get(self.cur_doc) if self.cur_doc else None

    def _on_doc_selected(self, index):
        if index < 0:
            return
        self.cur_doc = self.doc_tabs.tabData(index)
        doc = self.doc
        if doc is None:
            return
        self._sync_mode_btn()
        self._refresh_engine_label()
        self.page_list.blockSignals(True)
        self.page_list.clear()
        for p in doc.source.pages:
            thumb = p.image.copy()
            thumb.thumbnail((260, 340))
            it = QListWidgetItem(QIcon(QPixmap.fromImage(pil_to_qimage(thumb))), str(p.index + 1))
            self.page_list.addItem(it)
            self._update_page_item(doc.id, p.index)
        self.page_list.blockSignals(False)
        self._update_pages_count()
        self.page_list.setCurrentRow(0)
        self._on_page_selected(0)

    def _on_page_selected(self, row):
        doc = self.doc
        if not doc or row < 0 or row >= len(doc.source.pages):
            return
        self.cur_page = row
        page = doc.source.pages[row]
        self.page_label.setText(f"Page {row + 1} / {len(doc.source.pages)}")
        self.page_view.set_image(page.image)
        self._load_editor()
        self._refresh_preview(keep_scroll=False)

    def _on_task_changed(self, task_value: str):
        if self.doc and self.doc.task.value != task_value:
            self.doc.task = Task(task_value)
            self._reload_doc()
        self._sync_mode_btn()

    def _reload_doc(self):
        """Show the cached results of the current doc's model+task (switching model or mode)."""
        doc = self.doc
        doc.load_cached()
        for p in doc.source.pages:
            self._update_page_item(doc.id, p.index)
        self._load_editor()
        self._refresh_preview(keep_scroll=False)

    # ------------------------------------------------------------------ models
    def _fill_model_menu(self):
        """Model + mode chip menu: one exclusive group per section."""
        m = self.mode_menu
        m.clear()
        for g in (self._model_group, self._task_group):
            if g is not None:
                g.deleteLater()
        m.addSection("Model")
        self._model_group = QActionGroup(self)
        for spec in self.registry.all():
            if spec.runs_locally and self.settings.engine_for(spec) == "local":
                mark = "" if local_status(spec).downloaded else "  ⬇"
            else:
                mark = "  ☁"
            a = m.addAction(spec.name + mark)
            a.setCheckable(True)
            a.setData(spec.id)
            a.setToolTip(f"{spec.description}\n⬇ = not downloaded yet, ☁ = runs on the server")
            self._model_group.addAction(a)
        m.addAction(self.a_models)
        m.addSection("Recognise")
        self._task_group = QActionGroup(self)
        for value, label in TASK_LABELS.items():
            a = m.addAction(label)
            a.setCheckable(True)
            a.setData(value)
            self._task_group.addAction(a)
        self._model_group.triggered.connect(lambda a: self._on_model_changed(a.data()))
        self._task_group.triggered.connect(lambda a: self._on_task_changed(a.data()))
        self._sync_mode_btn()

    def _sync_mode_btn(self):
        model_id = self.doc.model_id if self.doc else self.settings.doc_model
        task = self.doc.task.value if self.doc else Task.DOCUMENT.value
        for g, value in ((self._model_group, model_id), (self._task_group, task)):
            for a in g.actions():
                a.setChecked(a.data() == value)
        self.mode_btn.setText(f"{self.registry.get(model_id).name}  ·  {TASK_LABELS.get(task, task)}")

    def _on_model_changed(self, model_id):
        if not model_id:
            return
        self.settings.doc_model = model_id
        self.settings.save()
        if self.doc and self.doc.model_id != model_id:
            self.doc.model_id = model_id
            self._reload_doc()
        self._sync_mode_btn()
        self._refresh_engine_label()

    def open_models(self):
        dlg = ModelManagerDialog(self.registry, self.settings, self.downloads, self)
        dlg.modelsChanged.connect(self._on_models_changed)
        dlg.exec()
        self._on_models_changed()

    def _on_models_changed(self):
        self.registry.reload()
        self._fill_model_menu()
        self._refresh_engine_label()
        if not self.docs:
            self._show_empty()

    def _update_page_item(self, doc_id, idx):
        if doc_id != self.cur_doc:
            return
        it = self.page_list.item(idx)
        if it:
            p = self.docs[doc_id].source.pages[idx]
            it.setData(STATUS_ROLE, p.status)
            it.setToolTip(f"Page {idx + 1}: " + (p.error or p.status))
            self.page_list.update(self.page_list.indexFromItem(it))
            self._update_pages_count()

    def _update_pages_count(self):
        doc = self.doc
        if doc:
            done = sum(1 for p in doc.source.pages if p.status == "done")
            self.pages_count.setText(f"{done}/{len(doc.source.pages)}")

    # ------------------------------------------------------------------ editor / preview
    def _load_editor(self):
        doc = self.doc
        page = doc.source.pages[self.cur_page] if doc else None
        self._loading_editor = True
        if page and page.status == "running":
            self.editor.setPlainText(self._stream.get((doc.id, page.index), ""))
            self.editor.setReadOnly(True)
        else:
            self.editor.setPlainText(page.markdown if page else "")
            self.editor.setReadOnly(page is None)
        self._loading_editor = False

    def _on_editor_changed(self):
        if not self._loading_editor and not self.editor.isReadOnly():
            self._edit_timer.start()

    def _commit_edit(self):
        doc = self.doc
        if not doc:
            return
        page = doc.source.pages[self.cur_page]
        md = self.editor.toPlainText()
        if md != page.markdown:
            page.markdown = md
            if page.status != "done" and md.strip():
                page.status = "done"
                self._update_page_item(doc.id, page.index)
            doc.cache.put(page.index, md)
        self._refresh_preview()

    def _refresh_preview(self, keep_scroll=True):
        doc = self.doc
        if not doc:
            self._show_empty()
            return
        if self.scope.currentData() == "doc":
            chunks = []
            for p in doc.source.pages:
                if p.markdown.strip():
                    chunks.append(f'<div class="page-label" style="margin-top:1.6em">page {p.index + 1}</div>')
                    chunks.append(render_html_body(parse_blocks(p.markdown)))
            body = "\n".join(chunks) or _empty_html("No recognised pages yet.")
        else:
            page = doc.source.pages[self.cur_page]
            if page.status == "running":
                md = clean_model_output(self._stream.get((doc.id, page.index), "")) or ""
                body = render_html_body(parse_blocks(md)) or _empty_html("Recognising…")
            elif page.markdown.strip():
                body = render_html_body(parse_blocks(page.markdown))
            elif page.status == "error":
                body = _empty_html(f"OCR failed: {page.error}")
            else:
                body = _empty_html("Not recognised yet — press Run (Ctrl+R).")
        self.preview.set_body(file_urls(body, doc.figures), keep_scroll)

    def _show_empty(self):
        self.stack.setCurrentIndex(0)
        spec = self.registry.get(self.settings.doc_model)
        if self.settings.engine_for(spec) == "openai":
            self.welcome.set_model(spec.name, f"runs on {self.settings.server_url}")
        else:
            ok = local_status(spec).downloaded
            self.welcome.set_model(spec.name, f"runs locally on {self.settings.device}"
                                   + ("" if ok else " · not downloaded yet"), ok)

    # ------------------------------------------------------------------ OCR
    def _backend_for(self, model_id: str):
        spec = self.registry.get(model_id)
        key = self.settings.backend_key(spec)
        cached = self._backends.get(model_id)
        if cached and cached[0] == key:
            return cached[1]
        backend = self.settings.make_backend(spec)
        if getattr(backend, "name", "") == "local":
            # Keep a single local model in (GPU) memory.
            for mid, (_k, b) in list(self._backends.items()):
                if getattr(b, "name", "") == "local":
                    b.unload()
                    del self._backends[mid]
        self._backends[model_id] = (key, backend)
        return backend

    def _refresh_engine_label(self):
        model_id = self.doc.model_id if self.doc else self.settings.doc_model
        cached = self._backends.get(model_id)
        spec = self.registry.get(model_id)
        if cached:
            text = cached[1].describe()
        elif self.settings.engine_for(spec) == "local":
            text = f"Local · {spec.name} · {self.settings.device}"
        else:
            text = f"Server · {spec.name} · {self.settings.server_url}"
        t = theme()
        self.engine_label.setText(f"<span style='color:{t.ok if cached else t.pending}'>●</span>&nbsp;{text}")
        self.engine_label.setToolTip("Model loaded" if cached else "Model loads on the first run")

    def _needs_download(self, model_id: str) -> bool:
        spec = self.registry.get(model_id)
        return self.settings.engine_for(spec) == "local" and not local_status(spec).downloaded

    def _on_download_progress(self, model_id, done, total):
        if model_id == self._waiting_download:
            pct = f" {100 * done / total:.0f}%" if total else ""
            self.status_label.setText(f"Downloading {self.registry.get(model_id).name}…{pct} "
                                      f"({fmt_bytes(done)} / {fmt_bytes(total)})")

    def _on_download_finished(self, model_id, ok, msg):
        self._fill_model_menu()
        if model_id != self._waiting_download:
            return
        self._waiting_download = None
        if ok:
            self.status_label.setText(f"Downloaded {self.registry.get(model_id).name}")
            self._start_next()
        else:
            self.status_label.setText(f"Download failed: {msg}")
            self._fail_queue(f"model download failed: {msg}")

    def _fail_queue(self, reason: str):
        for _mid, jobs in self.queue:
            for j in jobs:
                self._on_page_failed(j.doc_id, j.page_index, reason)
        self.queue.clear()
        self._finish_run()

    def _run_doc(self, doc: DocState, indices: list[int] | None = None, force=False):
        if indices is None:
            indices = [p.index for p in doc.source.pages if force or p.status != "done"]
        if not indices:
            self.status_label.setText("All pages already recognised. Use Re-run page to redo one.")
            return
        jobs = []
        for i in indices:
            p = doc.source.pages[i]
            p.status, p.error = "pending", ""
            self._update_page_item(doc.id, i)
            jobs.append(Job(doc.id, i, p.image, doc.task, doc.figures))
        self.queue.append((doc.model_id, jobs))
        self._run_total += len(jobs)
        self._update_progress()
        if self.worker is None:
            self._start_next()

    def _start_next(self):
        if not self.queue or self._waiting_download:
            return
        model_id, jobs = self.queue[0]
        if self._needs_download(model_id):
            spec = self.registry.get(model_id)
            size = fmt_bytes(remote_size(spec))
            if QMessageBox.question(
                self, "Download model?",
                f"{spec.name} is not downloaded yet ({size}). Download it now from Hugging Face?\n\n"
                f"Tip: manage models in Models… ({spec.repo})",
            ) != QMessageBox.StandardButton.Yes:
                self._fail_queue("model not downloaded")
                return
            self._waiting_download = model_id
            self._show_progress(True)
            self._set_running(True)
            self.downloads.start(spec)
            return
        self.queue.pop(0)
        backend = self._backend_for(model_id)
        self._refresh_engine_label()
        self.worker = OCRWorker(backend, jobs, self.settings.concurrency, detector=self._layout_detector())
        w = self.worker
        w.status.connect(self._on_status)
        w.pageStarted.connect(self._on_page_started)
        w.token.connect(self._on_token)
        w.pageDone.connect(self._on_page_done)
        w.pageFailed.connect(self._on_page_failed)
        self.thread = start_worker(w, self._on_worker_finished)
        self._set_running(True)
        self._show_progress(True)
        self._t0 = time.time()

    def run_current(self):
        if self.doc:
            self._run_doc(self.doc)

    def rerun_page(self):
        if self.doc:
            self.doc.cache.clear(self.cur_page)
            self._run_doc(self.doc, [self.cur_page], force=True)

    def rerun_all(self):
        if self.doc:
            for p in self.doc.source.pages:
                self.doc.cache.clear(p.index)
            self._run_doc(self.doc, force=True)

    def stop(self):
        if self._waiting_download:
            self.downloads.cancel(self._waiting_download)
        self.queue.clear()
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Stopping…")

    def _on_status(self, msg):
        self.status_label.setText(msg)
        self._refresh_engine_label()

    def _on_page_started(self, doc_id, idx):
        doc = self.docs.get(doc_id)
        if not doc:
            return
        doc.source.pages[idx].status = "running"
        self._stream[(doc_id, idx)] = ""
        self._update_page_item(doc_id, idx)
        self.status_label.setText(f"Recognising {doc.source.name} — page {idx + 1}/{len(doc.source.pages)}")
        if doc_id == self.cur_doc and idx == self.cur_page:
            self._load_editor()
            self._refresh_preview()

    def _on_token(self, doc_id, idx, text):
        key = (doc_id, idx)
        self._stream[key] = self._stream.get(key, "") + text
        if doc_id == self.cur_doc and idx == self.cur_page:
            self._loading_editor = True
            cur = self.editor.textCursor()
            cur.movePosition(cur.MoveOperation.End)
            cur.insertText(text)
            self._loading_editor = False
            if not self._stream_timer.isActive():
                self._stream_timer.start()

    def _render_stream(self):
        doc = self.doc
        if doc and doc.source.pages[self.cur_page].status == "running":
            self._refresh_preview()

    def _on_page_done(self, doc_id, idx, md, raw):
        self._stream.pop((doc_id, idx), None)
        self._run_done += 1
        self._update_progress()
        doc = self.docs.get(doc_id)
        if not doc:
            return
        p = doc.source.pages[idx]
        p.markdown, p.raw, p.status = md, raw, "done"
        doc.cache.put(idx, md, raw)
        self._update_page_item(doc_id, idx)
        if doc_id == self.cur_doc and idx == self.cur_page:
            self._load_editor()
            self._refresh_preview()
        elif doc_id == self.cur_doc and self.scope.currentData() == "doc":
            self._refresh_preview()
        if doc_id in self._autocopy:
            self._autocopy.discard(doc_id)
            self._copy_result(doc, p)

    def _on_page_failed(self, doc_id, idx, err):
        self._stream.pop((doc_id, idx), None)
        self._run_done += 1
        self._update_progress()
        doc = self.docs.get(doc_id)
        if not doc:
            return
        p = doc.source.pages[idx]
        p.status = "cancelled" if err == "cancelled" else "error"
        p.error = err
        self._update_page_item(doc_id, idx)
        if doc_id == self.cur_doc and idx == self.cur_page:
            self._load_editor()
            self._refresh_preview()

    def _on_worker_finished(self):
        # Called from QThread.finished: the thread has already stopped, nothing to wait for.
        self.worker = None
        self.thread = None
        if self.queue:
            self._start_next()
            return
        self._finish_run(f"Done — {self._run_done} page(s) in {time.time() - self._t0:.1f}s")

    def _finish_run(self, message: str | None = None):
        if self.worker is not None or self._waiting_download:
            return
        self._set_running(False)
        self._show_progress(False)
        if message:
            self.status_label.setText(message)
        self._run_total = self._run_done = 0

    def _update_progress(self):
        self.progress.setMaximum(max(1, self._run_total))
        self.progress.setValue(self._run_done)
        self.progress_label.setText(f"{self._run_done}/{self._run_total}")

    def _layout_detector(self) -> LayoutDetector | None:
        """Shared figure detector (loaded lazily on the worker thread), or None when disabled."""
        if not (self.settings.layout_detect and layout_available()):
            return None
        det = getattr(self, "_layout", None)
        if det is None or det.requested_device != self.settings.device:
            det = self._layout = LayoutDetector(self.settings.device)
        return det

    # ------------------------------------------------------------------ figures (crop / paste)
    def _on_crop_toggled(self, on: bool):
        if on and not self.doc:
            self.a_crop.setChecked(False)
            return
        self.page_view.set_crop_mode(on)
        if on:
            self.status_label.setText("Drag around a figure on the page. Esc to cancel.")

    def _selection_image(self) -> Image.Image | None:
        r, doc = self.page_view.selection(), self.doc
        if r is None or doc is None:
            return None
        img = doc.source.pages[self.cur_page].image
        box = (int(r.left()), int(r.top()), int(r.right() + 0.5), int(r.bottom() + 0.5))
        return img.crop(box)

    def insert_selection(self):
        img = self._selection_image()
        if img is None:
            return
        if self._insert_figure(img):
            self.page_view.clear_selection()
            self.a_crop.setChecked(False)

    def copy_selection(self):
        img = self._selection_image()
        if img is not None:
            QApplication.clipboard().setImage(pil_to_qimage(img))
            self.status_label.setText("Copied image")

    def save_selection(self):
        img = self._selection_image()
        if img is None:
            return
        doc = self.doc
        default = str((doc.source.path.parent if doc.source.path else Path.home())
                      / f"{doc.source.name}-p{self.cur_page + 1}-figure.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save image", default, "PNG image (*.png)")
        if path:
            img.save(path)
            self.status_label.setText(f"Saved {path}")

    def _insert_figure(self, img: Image.Image, at_cursor: bool = False) -> bool:
        """Store ``img`` with the document and link it from the current page."""
        doc = self.doc
        if not doc:
            return False
        page = doc.source.pages[self.cur_page]
        if page.status == "running":
            self.status_label.setText("Wait until this page is recognised, then insert the image.")
            return False
        rel = doc.figures.save(img, page.index)
        m = PLACEHOLDER_RE.search(page.markdown)
        link = f"![{(m.group(1).strip() if m else '') or 'Figure'}]({rel})"
        if at_cursor:
            cur = self.editor.textCursor()
            cur.insertText(("\n\n" if cur.positionInBlock() else "") + link + "\n\n")
            self.editor.setTextCursor(cur)
            self._edit_timer.stop()
            self._commit_edit()
        else:
            pos = self.editor.textCursor().position() if self.editor.isVisible() else None
            page.markdown = insert_figure(page.markdown, link, pos)
            page.status, page.error = "done", ""
            doc.cache.put(page.index, page.markdown)
            self._update_page_item(doc.id, page.index)
            self._load_editor()
            self._refresh_preview()
        self.status_label.setText(f"Inserted {rel}")
        return True

    def _on_editor_image(self, qimg: QImage):
        if qimg.isNull():
            self.status_label.setText("Could not read that image.")
            return
        self._insert_figure(qimage_to_pil(qimg), at_cursor=True)

    # ------------------------------------------------------------------ snip / paste
    def start_snip(self):
        self._was_visible = self.isVisible() and not self.isMinimized()
        self.hide()
        QTimer.singleShot(300, self._show_overlay)  # let the window disappear first

    def _show_overlay(self):
        self._overlay = SnipOverlay()
        self._overlay.snipped.connect(self._on_snipped)
        self._overlay.cancelled.connect(self._restore_window)
        self._overlay.start()

    def _restore_window(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_snipped(self, qimg: QImage):
        self._restore_window()
        self._ocr_image(qimage_to_pil(qimg), "Snip")

    def paste_image(self):
        cb = QApplication.clipboard()
        md = cb.mimeData()
        if md.hasImage():
            self._ocr_image(qimage_to_pil(cb.image()), "Pasted")
        elif md.hasUrls():
            self.open_paths([u.toLocalFile() for u in md.urls() if u.isLocalFile()])
        else:
            self.status_label.setText("Clipboard has no image.")

    def _ocr_image(self, img: Image.Image, label: str):
        src = source_from_image(img, f"{label} {time.strftime('%H:%M:%S')}")
        doc = self._add_doc(src, Task(self.settings.snip_task),
                            self.settings.snip_model or self.settings.doc_model)
        if doc.source.pages[0].status == "done":
            self._copy_result(doc, doc.source.pages[0])
        else:
            self._autocopy.add(doc.id)
            self._run_doc(doc)

    # ------------------------------------------------------------------ copy
    def _formula_wrap(self, tex: str) -> str:
        style = self.settings.formula_copy
        if style == "inline":
            return f"${tex}$"
        if style == "display":
            return f"\\[\n{tex}\n\\]"
        return tex

    @staticmethod
    def _page_formula(md: str) -> str | None:
        blocks = parse_blocks(md)
        from ..core.document import DisplayMath

        if len(blocks) == 1 and isinstance(blocks[0], DisplayMath):
            return blocks[0].tex
        return None

    def _copy_result(self, doc: DocState, page):
        """Auto-copy after a snip, Mathpix style."""
        tex = self._page_formula(page.markdown)
        text = self._formula_wrap(tex) if tex is not None else page.markdown.strip()
        QApplication.clipboard().setText(text)
        self.status_label.setText("Copied " + ("LaTeX" if tex is not None else "Markdown") + " to clipboard")

    def _copy_formula_text(self, tex: str):
        QApplication.clipboard().setText(self._formula_wrap(tex.strip()))
        self.status_label.setText("Copied equation LaTeX")

    def copy_page(self, kind):
        doc = self.doc
        if not doc:
            return
        md = doc.source.pages[self.cur_page].markdown
        cb = QApplication.clipboard()
        if kind == "md":
            cb.setText(md)
        elif kind == "tex":
            cb.setText(render_latex_body(parse_blocks(md)))
        elif kind in ("formula", "mathml"):
            tex = self._page_formula(md)
            if tex is None:
                self.status_label.setText("Current page is not a single formula (use Formula mode for snips).")
                return
            if kind == "formula":
                cb.setText(self._formula_wrap(tex))
            else:
                self.preview.to_mathml(tex, lambda s: (cb.setText(_strip_katex_mathml(s or "")),
                                                       self.status_label.setText("Copied MathML")))
                return
        self.status_label.setText("Copied")

    def copy_doc(self, kind):
        if self.doc:
            pages = [p.markdown for p in self.doc.source.pages if p.markdown.strip()]
            QApplication.clipboard().setText(export_text(pages, kind, self.doc.source.name))
            self.status_label.setText("Copied document")

    # ------------------------------------------------------------------ export
    def export(self, fmt):
        doc = self.doc
        if not doc:
            return
        pages = [p.markdown for p in doc.source.pages if p.markdown.strip()]
        if not pages:
            QMessageBox.information(self, "Nothing to export", "Run OCR first.")
            return
        missing = sum(1 for p in doc.source.pages if not p.markdown.strip())
        if missing and QMessageBox.question(
            self, "Incomplete", f"{missing} page(s) are not recognised yet. Export anyway?"
        ) != QMessageBox.StandardButton.Yes:
            return
        label, ext = FORMATS[fmt]
        default = str((doc.source.path.parent if doc.source.path else Path.home()) / (doc.source.name + ext))
        path, _ = QFileDialog.getSaveFileName(self, f"Export {label}", default, f"{label} (*{ext})")
        if not path:
            return
        if fmt == "pdf":
            self.status_label.setText("Typesetting PDF…")
            html = inline_images(pages_to_html(pages, title="", standalone=True), doc.figures)

            def done(ok, msg):
                self.status_label.setText(f"Saved {path}" if ok else f"PDF export failed: {msg}")

            html_to_pdf_async(html, Path(path), done, self.settings.page_size)
        else:
            Path(path).write_text(export_text(pages, fmt, title="", figures=doc.figures), encoding="utf-8")
            n = copy_figures(pages, doc.figures, Path(path).parent) if fmt in FIGURE_FOLDER_FORMATS else 0
            self.status_label.setText(f"Saved {path}" + (f" (+{n} image(s) in figures/)" if n else ""))

    # ------------------------------------------------------------------ misc
    def open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            self.settings = dlg.result_settings()
            self.settings.save()
            self._fill_model_menu()
            self._refresh_engine_label()

    def _about(self):
        QMessageBox.about(
            self,
            "SciDoc OCR",
            "<b>SciDoc OCR</b> — an open-source Mathpix alternative.<br>"
            "OCR engine: <a href='https://huggingface.co/xiaomi-research/Xiaomi-OCR-0'>Xiaomi-OCR-0</a> "
            "(Apache-2.0). Math rendering: KaTeX. UI: Qt for Python (PySide6).",
        )

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() or e.mimeData().hasImage():
            e.acceptProposedAction()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasUrls():
            self.open_paths([u.toLocalFile() for u in md.urls() if u.isLocalFile()])
        elif md.hasImage():
            self._ocr_image(qimage_to_pil(QImage(md.imageData())), "Dropped")

    def retheme(self):
        """Called when the OS switches light/dark: recolour icons and the preview."""
        refresh_icons(self)
        for i in range(self.doc_tabs.count()):
            self.doc_tabs.setTabIcon(i, icon("doc", theme().muted, 14))
        self.welcome.refresh()
        self.page_view.restyle()
        self._refresh_engine_label()
        if not self.docs:
            self._show_empty()
        self.preview.reload_shell()
        QTimer.singleShot(300, self._refresh_preview)

    def closeEvent(self, e):
        self.queue.clear()
        if self.worker:
            self.worker.cancel()
        if self.thread:
            self.thread.wait(5000)
        super().closeEvent(e)


def _vline() -> QWidget:
    w = QWidget()
    w.setFixedSize(9, 18)
    w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    w.setStyleSheet(f"background: transparent; border-left: 1px solid {theme().line}; margin-left: 4px;")
    return w


def _empty_html(msg: str) -> str:
    return f'<div class="empty">{msg}</div>'


def _strip_katex_mathml(s: str) -> str:
    """KaTeX wraps MathML in <span class="katex">; Word wants the bare <math> element."""
    start, end = s.find("<math"), s.rfind("</math>")
    return s[start : end + len("</math>")] if start >= 0 and end > start else s
