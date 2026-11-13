"""Look & feel: theme (light/dark), line icons, and small reusable widgets."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from PySide6.QtCore import QByteArray, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QKeySequence, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# --------------------------------------------------------------------------- theme


@dataclass(frozen=True)
class Theme:
    dark: bool
    window: str
    base: str
    panel: str
    text: str
    muted: str
    line: str
    hover: str
    accent: str
    accent_soft: str
    ok: str
    warn: str
    err: str
    pending: str
    dash: str


LIGHT = Theme(False, window="#f6f7f9", base="#ffffff", panel="#eceef2", text="#1d2329", muted="#5f6874",
              line="#e1e4e8", hover="#e8ebef", accent="#2f6fde", accent_soft="#dbe7fb", ok="#1e8e3e",
              warn="#b9770e", err="#c0392b", pending="#c3c9d1", dash="#b9c0ca")
DARK = Theme(True, window="#1b1e22", base="#15171a", panel="#24282d", text="#e6e9ec", muted="#9aa3ad",
             line="#2e3338", hover="#2a2f35", accent="#6ea0ff", accent_soft="#263a5c", ok="#4cc27a",
             warn="#e0a43a", err="#ef6b5e", pending="#5b636c", dash="#454c54")

_current: Theme = LIGHT


def theme() -> Theme:
    return _current


def system_is_dark() -> bool:
    try:
        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:  # Qt < 6.5
        return QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128


def apply_theme(app: QApplication, dark: bool | None = None) -> Theme:
    """Fusion + palette + a small stylesheet: identical look on macOS, Windows and Linux."""
    global _current
    t = DARK if (system_is_dark() if dark is None else dark) else LIGHT
    _current = t
    app.setStyle("Fusion")
    pal = QPalette()
    c = QColor
    pal.setColor(QPalette.ColorRole.Window, c(t.window))
    pal.setColor(QPalette.ColorRole.WindowText, c(t.text))
    pal.setColor(QPalette.ColorRole.Base, c(t.base))
    pal.setColor(QPalette.ColorRole.AlternateBase, c(t.panel))
    pal.setColor(QPalette.ColorRole.Text, c(t.text))
    pal.setColor(QPalette.ColorRole.Button, c(t.panel))
    pal.setColor(QPalette.ColorRole.ButtonText, c(t.text))
    pal.setColor(QPalette.ColorRole.Highlight, c(t.accent))
    pal.setColor(QPalette.ColorRole.HighlightedText, c("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, c(t.panel))
    pal.setColor(QPalette.ColorRole.ToolTipText, c(t.text))
    pal.setColor(QPalette.ColorRole.PlaceholderText, c(t.muted))
    pal.setColor(QPalette.ColorRole.Link, c(t.accent))
    pal.setColor(QPalette.ColorRole.Mid, c(t.line))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, c(t.muted))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET.format(t=t))
    return t


STYLESHEET = """
QMainWindow, QDialog {{ background: {t.window}; }}

/* ---- top bar */
QToolBar#MainToolbar {{ background: {t.window}; border: none; border-bottom: 1px solid {t.line};
                       padding: 9px 12px; spacing: 8px; }}
QToolBar::separator {{ background: {t.line}; width: 1px; margin: 5px 4px; }}
QLabel#Logo {{ background: {t.accent}; color: #ffffff; border-radius: 7px; font-weight: 700; font-size: 14px; }}
QLabel#AppName {{ font-weight: 600; font-size: 14px; color: {t.text}; }}

/* ---- buttons */
QToolButton {{ border: 1px solid transparent; border-radius: 7px; padding: 4px 6px; color: {t.text}; }}
QToolButton:hover {{ background: {t.hover}; }}
QToolButton:pressed {{ background: {t.panel}; }}
QToolButton:checked {{ background: {t.accent_soft}; color: {t.accent}; border-color: {t.accent}; }}
QToolButton:disabled {{ color: {t.muted}; }}
QToolButton#Ghost {{ background: {t.base}; border: 1px solid {t.line}; padding: 5px 10px; font-weight: 500; }}
QToolButton#Ghost:hover {{ border-color: {t.muted}; }}
QToolButton#Ghost:checked {{ background: {t.accent_soft}; border-color: {t.accent}; color: {t.accent}; }}
QToolButton#Ghost[popupMode="2"] {{ padding-right: 20px; }}
QToolButton#Ghost::menu-indicator {{ subcontrol-origin: padding; subcontrol-position: right center; right: 6px; }}
QToolButton#Primary {{ background: {t.accent}; color: #ffffff; font-weight: 600; padding: 5px 12px;
                      border: none; border-radius: 8px; }}
QToolButton#Primary:hover {{ background: {t.accent}; }}
QToolButton#Primary:disabled {{ background: {t.pending}; }}
QToolButton#Primary[popupMode="1"] {{ padding-right: 30px; }}
QToolButton#Primary::menu-button {{ border: none; border-left: 1px solid rgba(255,255,255,0.35); width: 24px;
                                   border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}
QToolButton#Danger {{ background: {t.err}; color: #ffffff; font-weight: 600; padding: 5px 12px; border: none;
                     border-radius: 8px; }}
QToolButton#TabClose {{ padding: 0; border-radius: 4px; }}
QToolButton#Flat {{ background: transparent; border: none; border-radius: 6px; padding: 4px; }}
QToolButton#Flat:hover {{ background: {t.hover}; }}
QToolButton#Accent {{ background: {t.accent}; color: #ffffff; font-weight: 600; border: none; border-radius: 6px;
                     padding: 4px 10px; }}

/* ---- segmented control */
QWidget#Segmented {{ background: {t.panel}; border-radius: 8px; }}
QWidget#Segmented QToolButton {{ border: none; border-radius: 6px; padding: 3px 10px; color: {t.muted};
                                font-weight: 500; background: transparent; }}
QWidget#Segmented QToolButton:hover {{ color: {t.text}; }}
QWidget#Segmented QToolButton:checked {{ background: {t.base}; color: {t.text}; font-weight: 600; }}

/* ---- document tabs (inside the top bar) */
QTabBar#DocTabs {{ background: transparent; }}
QTabBar#DocTabs::tab {{ background: transparent; color: {t.muted}; padding: 6px 8px 6px 10px; margin-right: 2px;
                       border: 1px solid transparent; border-radius: 8px; max-width: 220px; }}
QTabBar#DocTabs::tab:selected {{ background: {t.base}; color: {t.text}; border-color: {t.line}; }}
QTabBar#DocTabs::tab:hover:!selected {{ background: {t.hover}; color: {t.text}; }}
QTabBar#DocTabs QToolButton {{ background: transparent; }}

/* ---- panels */
QWidget#PanelHeader {{ background: {t.window}; border-bottom: 1px solid {t.line}; }}
QLabel#PanelTitle {{ color: {t.muted}; font-weight: 600; font-size: 11px; letter-spacing: 0.5px; }}
QLabel#PageNum {{ color: {t.text}; }}
QListWidget#Thumbs {{ background: {t.window}; border: none; outline: none; }}
QGraphicsView#PageView {{ background: {t.panel}; border: none; }}
QPlainTextEdit#Editor {{ background: {t.base}; border: none; padding: 8px 12px; selection-background-color: {t.accent}; }}
QFrame#Floating {{ background: {t.base}; border: 1px solid {t.line}; border-radius: 10px; }}
QLabel#Hint {{ background: {t.window}; color: {t.muted}; border-top: 1px solid {t.line}; padding: 5px 14px; }}
QSplitter::handle {{ background: {t.line}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

/* ---- status bar */
QStatusBar {{ background: {t.window}; border-top: 1px solid {t.line}; color: {t.muted}; min-height: 26px; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {t.muted}; padding: 0 4px; }}
QProgressBar {{ border: none; border-radius: 2px; background: {t.panel}; }}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 2px; }}

/* ---- welcome */
QLabel#Title {{ font-size: 24px; font-weight: 700; color: {t.text}; }}
QLabel#Muted {{ color: {t.muted}; }}
QLabel#Strong {{ font-weight: 600; font-size: 14px; color: {t.text}; }}
QLabel#Kbd {{ color: {t.muted}; border: 1px solid {t.line}; border-radius: 5px; padding: 1px 6px;
             background: {t.window}; font-size: 11px; }}
QLabel#IconTile {{ background: {t.accent_soft}; border-radius: 12px; }}
QPushButton#DropZone {{ background: {t.base}; border: 2px dashed {t.dash}; border-radius: 14px; }}
QPushButton#DropZone:hover, QPushButton#DropZone:focus {{ border-color: {t.accent}; }}
QPushButton#Card {{ background: {t.base}; border: 1px solid {t.line}; border-radius: 12px; text-align: left; }}
QPushButton#Card:hover, QPushButton#Card:focus {{ border-color: {t.accent}; }}
QFrame#ModelLine {{ background: {t.panel}; border-radius: 10px; }}

/* ---- misc */
QComboBox {{ min-height: 24px; padding: 1px 8px; }}
QMenu {{ background: {t.base}; border: 1px solid {t.line}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {t.accent_soft}; color: {t.text}; }}
QMenu::item:disabled {{ color: {t.muted}; }}
QMenu::separator {{ height: 1px; background: {t.line}; margin: 4px 6px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 4px; }}
QTableWidget {{ background: {t.base}; gridline-color: {t.line}; border: 1px solid {t.line}; border-radius: 6px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {t.dash}; border-radius: 3px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {t.muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QHeaderView::section {{ background: {t.window}; color: {t.muted}; border: none;
                       border-bottom: 1px solid {t.line}; padding: 5px; }}
"""

# --------------------------------------------------------------------------- icons (Lucide, ISC license)

_SVG = {
    "open": '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>',
    "snip": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
    "paste": '<rect width="8" height="4" x="8" y="2" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>',
    "run": '<polygon points="6 3 20 12 6 21 6 3" fill="{c}"/>',
    "stop": '<rect width="14" height="14" x="5" y="5" rx="2" fill="{c}"/>',
    "rerun": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    "copy": '<rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    "export": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    "settings": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>',
    "models": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "prev": '<path d="m15 18-6-6 6-6"/>',
    "next": '<path d="m9 18 6-6-6-6"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "minus": '<path d="M5 12h14"/>',
    "fit": '<polyline points="18 8 22 12 18 16"/><polyline points="6 8 2 12 6 16"/><line x1="2" x2="22" y1="12" y2="12"/>',
    "close": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "doc": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "crop": '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
    "image": '<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>',
    "upload": '<path d="M12 13v8"/><path d="m8 17 4-4 4 4"/><path d="M20 16.6A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 4 15.25"/>',
    "save": '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>',
}

# Icon colour roles, resolved against the current theme each time icons are (re)rendered.
_ROLES = {"text": lambda t: t.text, "muted": lambda t: t.muted, "accent": lambda t: t.accent,
          "white": lambda t: "#ffffff"}


def icon(name: str, color: str | None = None, size: int = 18) -> QIcon:
    """Render a Lucide stroke icon in the given colour (default: theme text), crisp on HiDPI."""
    color = color or theme().text
    body = _SVG[name].replace("{c}", color)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    ic = QIcon()
    for scale in (1, 2, 3):
        pm = QPixmap(size * scale, size * scale)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        renderer.render(p, QRectF(0, 0, size * scale, size * scale))
        p.end()
        pm.setDevicePixelRatio(scale)
        ic.addPixmap(pm)
    return ic


def set_icon(obj, name: str, role: str = "text"):
    """Give a QAction / QAbstractButton a themed icon that :func:`refresh_icons` can recolour."""
    obj.setProperty("iconName", name)
    obj.setProperty("iconRole", role)
    obj.setIcon(icon(name, _ROLES[role](theme())))
    return obj


def refresh_icons(root) -> None:
    """Re-render every themed icon below ``root`` (call after the light/dark theme changes)."""
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QAbstractButton

    for obj in [*root.findChildren(QAction), *root.findChildren(QAbstractButton)]:
        name = obj.property("iconName")
        if name:
            obj.setIcon(icon(name, _ROLES[obj.property("iconRole") or "text"](theme())))


def shortcut_text(seq) -> str:
    """Platform-native shortcut label (⌘⇧V on macOS, Ctrl+Shift+V elsewhere)."""
    return QKeySequence(seq).toString(QKeySequence.SequenceFormat.NativeText)


# --------------------------------------------------------------------------- widgets


class Segmented(QWidget):
    """Mutually exclusive pill buttons, API-compatible with QComboBox.currentData()."""

    changed = Signal(object)

    def __init__(self, options: list[tuple[str, object]], parent=None):
        super().__init__(parent)
        self.setObjectName("Segmented")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self._data = []
        for i, (label, data) in enumerate(options):
            b = QToolButton(text=label, checkable=True)
            b.setChecked(i == 0)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.group.addButton(b, i)
            lay.addWidget(b)
            self._data.append(data)
        self.group.idClicked.connect(lambda i: self.changed.emit(self._data[i]))

    def currentData(self):  # noqa: N802 - mirrors QComboBox
        return self._data[max(0, self.group.checkedId())]

    def setCurrentData(self, data):  # noqa: N802
        if data in self._data:
            self.group.button(self._data.index(data)).setChecked(True)


def panel_header(*widgets, title: str | None = None) -> QWidget:
    """A slim header row above a panel; ``None`` in widgets inserts a stretch."""
    w = QWidget(objectName="PanelHeader")
    w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    lay = QHBoxLayout(w)
    lay.setContentsMargins(12, 0, 8, 0)
    lay.setSpacing(4)
    if title:
        lay.addWidget(QLabel(title.upper(), objectName="PanelTitle"))
    for x in widgets:
        if x is None:
            lay.addStretch(1)
        else:
            lay.addWidget(x)
    w.setFixedHeight(42)
    return w


def tool_button(icon_name: str, tip: str, slot=None, text: str = "", role: str = "text",
                object_name: str = "Flat") -> QToolButton:
    b = QToolButton(objectName=object_name)
    set_icon(b, icon_name, role)
    b.setIconSize(QSize(16, 16))
    b.setToolTip(tip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if text:
        b.setText(text)
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    else:
        b.setAccessibleName(tip.split(" (")[0])
    if slot:
        b.clicked.connect(slot)
    return b


class FloatingBar(QFrame):
    """Rounded pill with a soft shadow, floating over a view (page navigation, crop actions)."""

    def __init__(self, parent=None):
        super().__init__(parent, objectName="Floating")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setOffset(QPointF(0, 3))
        shadow.setColor(QColor(0, 0, 0, 45))
        self.setGraphicsEffect(shadow)

    def add(self, w: QWidget) -> QWidget:
        self.layout().addWidget(w)
        return w


def overlay(base: QWidget, floating: QWidget, align: Qt.AlignmentFlag, margin: int = 14) -> QWidget:
    """Stack ``floating`` over ``base`` in the same grid cell (Qt's standard overlay pattern)."""
    host = QWidget()
    grid = QGridLayout(host)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.addWidget(base, 0, 0)
    holder = QWidget()
    hl = QVBoxLayout(holder)
    hl.setContentsMargins(margin, margin, margin, margin)
    hl.addWidget(floating)
    grid.addWidget(holder, 0, 0, align)
    return host


STATUS_ROLE = Qt.ItemDataRole.UserRole + 1


class ThumbDelegate(QStyledItemDelegate):
    """Compact page thumbnail with the page number and a status dot underneath."""

    THUMB_W, THUMB_H, LABEL_H = 64, 84, 18

    def paint(self, p: QPainter, opt, index):
        t = theme()
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = opt.rect
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        box = QRectF(r.x() + (r.width() - self.THUMB_W) / 2, r.y() + 8, self.THUMB_W, self.THUMB_H)
        image_bottom = box.bottom()
        icon_ = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon_, QIcon):
            dpr = p.device().devicePixelRatioF() if p.device() else 1.0
            pm = icon_.pixmap(box.size().toSize(), dpr)
            sz = pm.deviceIndependentSize().toSize().scaled(box.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF(box.x() + (box.width() - sz.width()) / 2, box.y(), sz.width(), sz.height())
            p.drawPixmap(target.toRect(), pm)
            pen = p.pen()
            pen.setColor(QColor(t.accent if selected else t.line))
            pen.setWidthF(2.0 if selected else 1.0)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            grow = 3.0 if selected else 0.5
            p.drawRoundedRect(target.adjusted(-grow, -grow, grow, grow), 4, 4)
            image_bottom = target.bottom()
        status = index.data(STATUS_ROLE) or "pending"
        color = {"done": t.ok, "running": t.accent, "error": t.err, "cancelled": t.warn}.get(status, t.pending)
        label = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        f = QFont(opt.font)
        f.setPointSizeF(max(8.0, f.pointSizeF() - 1))
        f.setBold(selected)
        p.setFont(f)
        fm = p.fontMetrics()
        dot, gap = 6, 5
        total = fm.horizontalAdvance(label) + dot + gap
        x = r.x() + (r.width() - total) / 2
        baseline = image_bottom + 8 + fm.ascent()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(color))
        p.drawEllipse(QRectF(x, baseline - fm.ascent() / 2 - dot / 2 - 1, dot, dot))
        p.setPen(QColor(t.text if selected else t.muted))
        p.drawText(int(x + dot + gap), int(baseline), label)
        p.restore()

    def sizeHint(self, opt, index):  # noqa: N802
        return QSize(self.THUMB_W + 24, self.THUMB_H + self.LABEL_H + 22)


# --------------------------------------------------------------------------- welcome


def _passive(w: QWidget) -> QWidget:
    """Labels inside a button must not swallow its clicks."""
    w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    return w


class _IconLabel(QLabel):
    def __init__(self, name: str, role: str, size: int, tile: int | None = None):
        super().__init__(objectName="IconTile" if tile else "")
        self.name, self.role, self.size_ = name, role, size
        if tile:
            self.setFixedSize(tile, tile)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.refresh()

    def refresh(self):
        self.setPixmap(icon(self.name, _ROLES[self.role](theme()), self.size_).pixmap(self.size_, self.size_))


class Welcome(QWidget):
    """Empty state: a big drop zone (also the Open button), Snip and Paste cards, and the model line."""

    def __init__(self, on_open, on_snip, on_paste, on_models, parent=None):
        super().__init__(parent)
        self._icons: list[_IconLabel] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 32, 16, 32)
        outer.addStretch(2)
        col_host = QWidget()
        col_host.setMaximumWidth(640)
        col_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        col = QVBoxLayout(col_host)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(20)

        head = QVBoxLayout()
        head.setSpacing(6)
        head.addWidget(QLabel("Start a document", objectName="Title"))
        sub = QLabel("Turn scientific PDFs, scans and screenshots into Markdown, LaTeX and clean PDFs.",
                     objectName="Muted")
        sub.setWordWrap(True)
        head.addWidget(sub)
        col.addLayout(head)

        # drop zone == Open button
        drop = QPushButton(objectName="DropZone")
        drop.setAccessibleName("Open documents")
        drop.setCursor(Qt.CursorShape.PointingHandCursor)
        drop.setMinimumHeight(200)
        drop.clicked.connect(on_open)
        dl = QVBoxLayout(drop)
        dl.setSpacing(10)
        dl.addStretch(1)
        tile = self._icon("upload", "accent", 22, tile=44)
        dl.addWidget(tile, 0, Qt.AlignmentFlag.AlignHCenter)
        t1 = _passive(QLabel("Drop PDFs or images here", objectName="Strong"))
        t2 = _passive(QLabel(objectName="Muted"))
        t2.setTextFormat(Qt.TextFormat.RichText)
        self._browse = t2
        for w in (t1, t2):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dl.addWidget(w)
        dl.addStretch(1)
        col.addWidget(drop)

        # secondary starts
        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self._card("snip", "Snip screen", "Result is copied automatically",
                                  shortcut_text("Ctrl+Alt+M"), on_snip), 0, 0)
        grid.addWidget(self._card("paste", "Paste image", "OCR what is on the clipboard",
                                  shortcut_text(QKeySequence.StandardKey.Paste), on_paste), 0, 1)
        col.addLayout(grid)

        # model line
        line = QFrame(objectName="ModelLine")
        ll = QHBoxLayout(line)
        ll.setContentsMargins(14, 10, 14, 10)
        ll.setSpacing(10)
        self.model_line = QLabel()
        self.model_line.setTextFormat(Qt.TextFormat.RichText)
        change = QLabel()
        change.setTextFormat(Qt.TextFormat.RichText)
        change.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse
                                       | Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
        change.linkActivated.connect(lambda _: on_models())
        self._change = change
        ll.addWidget(self.model_line, 1)
        ll.addWidget(change)
        col.addWidget(line)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(col_host, 100)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(3)
        self._model = ("", "")
        self.refresh()

    def _icon(self, name, role, size, tile=None) -> _IconLabel:
        w = _passive(_IconLabel(name, role, size, tile))
        self._icons.append(w)
        return w

    def _card(self, icon_name, title, sub, keys, slot) -> QPushButton:
        b = QPushButton(objectName="Card")
        b.setAccessibleName(title)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setMinimumHeight(68)
        b.clicked.connect(slot)
        lay = QHBoxLayout(b)
        lay.setContentsMargins(16, 12, 14, 12)
        lay.setSpacing(12)
        lay.addWidget(self._icon(icon_name, "accent", 20))
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(_passive(QLabel(title, objectName="Strong")))
        text.addWidget(_passive(QLabel(sub, objectName="Muted")))
        lay.addLayout(text, 1)
        lay.addWidget(_passive(QLabel(keys, objectName="Kbd")), 0, Qt.AlignmentFlag.AlignVCenter)
        return b

    def set_model(self, name: str, where: str, ok: bool = True):
        self._model = (name, where, ok)
        t = theme()
        dot = t.ok if ok else t.warn
        self.model_line.setText(f"<span style='color:{dot}'>●</span>&nbsp; Model <b>{name}</b> "
                                f"<span style='color:{t.muted}'>· {where}</span>")

    def refresh(self):
        """Re-apply theme colours that live outside the stylesheet."""
        t = theme()
        for w in self._icons:
            w.refresh()
        self._change.setText(f"<a href='models' style='color:{t.accent}; text-decoration:none'>Change model</a>")
        self._browse.setText(f"or <span style='color:{t.accent}; font-weight:500'>browse files</span>"
                             f" · PDF, PNG, JPG, TIFF &nbsp;({shortcut_text(QKeySequence.StandardKey.Open)})")
        if len(self._model) == 3:
            self.set_model(*self._model)


def is_mac() -> bool:
    return sys.platform == "darwin"
