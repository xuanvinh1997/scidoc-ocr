"""Screen snipping overlay: drag a rectangle anywhere on screen, get the image back."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


class SnipOverlay(QWidget):
    snipped = Signal(QImage)
    cancelled = Signal()

    def __init__(self):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        self.shot = screen.grabWindow(0)  # needs Screen Recording permission on macOS
        self.setGeometry(screen.geometry())
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.origin: QPoint | None = None
        self.current: QPoint | None = None

    def start(self):
        self.showFullScreen()
        self.activateWindow()
        self.raise_()

    def _rect(self) -> QRect:
        if not self.origin or not self.current:
            return QRect()
        return QRect(self.origin, self.current).normalized()

    def paintEvent(self, _):
        p = QPainter(self)
        p.drawPixmap(self.rect(), self.shot)
        p.fillRect(self.rect(), QColor(0, 0, 0, 110))
        r = self._rect()
        if r.isValid():
            dpr = self.shot.devicePixelRatio()
            src = QRect(int(r.x() * dpr), int(r.y() * dpr), int(r.width() * dpr), int(r.height() * dpr))
            p.drawPixmap(r, self.shot, src)
            p.setPen(QPen(QColor(47, 111, 222), 2))
            p.drawRect(r)
        p.setPen(QColor(255, 255, 255))
        p.drawText(self.rect().adjusted(0, 24, 0, 0), Qt.AlignmentFlag.AlignHCenter,
                   "Drag to select an equation, table or paragraph · Esc to cancel")

    def mousePressEvent(self, e):
        self.origin = self.current = e.position().toPoint()
        self.update()

    def mouseMoveEvent(self, e):
        self.current = e.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, e):
        self.current = e.position().toPoint()
        r = self._rect()
        self.close()
        if r.width() < 6 or r.height() < 6:
            self.cancelled.emit()
            return
        dpr = self.shot.devicePixelRatio()
        crop = self.shot.copy(QRect(int(r.x() * dpr), int(r.y() * dpr), int(r.width() * dpr), int(r.height() * dpr)))
        self.snipped.emit(crop.toImage())

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()
            self.cancelled.emit()
