"""Straighten dialog: drag 4 corners + 4 edge midpoints onto the figure's frame, see the result live."""

from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from ..core.straighten import (
    detect_frame,
    edge_polylines,
    frame_from_handles,
    handles_for,
    scaled,
    straighten,
)
from .ui import theme

PREVIEW_MAX = 640  # preview is computed on a downscaled copy so dragging stays smooth


class _Handle(QGraphicsEllipseItem):
    R = 7

    def __init__(self, on_move):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self._on_move = on_move
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
                      | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(2)
        t = theme()
        self.setBrush(QColor("#ffffff"))
        self.setPen(QPen(QColor(t.accent), 2))

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._on_move()
        return super().itemChange(change, value)


class _View(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(QColor(theme().panel))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fitInView(self.sceneRect().adjusted(-16, -16, 16, 16), Qt.AspectRatioMode.KeepAspectRatio)


class StraightenDialog(QDialog):
    """Returns the straightened image via :meth:`result_image` when accepted."""

    def __init__(self, img: Image.Image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Straighten figure")
        self.resize(1180, 680)
        self.img = img.convert("RGB")
        s = min(1.0, PREVIEW_MAX / max(self.img.size))
        self._scale = s
        self._small = self.img.resize((max(1, int(self.img.width * s)), max(1, int(self.img.height * s))),
                                      Image.Resampling.BILINEAR) if s < 1 else self.img
        self._auto = detect_frame(self.img)

        from .main_window import pil_to_qimage

        scene = QGraphicsScene(self)
        scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(pil_to_qimage(self.img))))
        scene.setSceneRect(QRectF(0, 0, self.img.width, self.img.height))
        pen = QPen(QColor(theme().accent), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._edges = QGraphicsPathItem()
        self._edges.setPen(pen)
        self._edges.setZValue(1)
        scene.addItem(self._edges)
        self._handles = [_Handle(self._changed) for _ in range(8)]
        for h in self._handles:
            scene.addItem(h)
        self.view = _View(scene)

        self.preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumWidth(360)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet(f"background: {theme().panel};")

        hint = QLabel("Drag the points onto the figure's frame or axes. Corners fix perspective; "
                      "the middle points follow the page curl.", objectName="Muted")
        hint.setWordWrap(True)
        self.status = QLabel(objectName="Muted")

        auto_btn = QPushButton("Auto-detect frame")
        auto_btn.setEnabled(self._auto is not None)
        auto_btn.clicked.connect(self._use_auto)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(lambda: self._set_handles(handles_for(None, self.img.size), auto=False))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        ok = bb.addButton("Use straightened image", QDialogButtonBox.ButtonRole.AcceptRole)
        ok.setDefault(True)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)

        panes = QHBoxLayout()
        panes.setSpacing(12)
        left = QVBoxLayout()
        left.addWidget(QLabel("Original", objectName="PanelTitle"))
        left.addWidget(self.view, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("Result", objectName="PanelTitle"))
        right.addWidget(self.preview, 1)
        panes.addLayout(left, 3)
        panes.addLayout(right, 2)
        buttons = QHBoxLayout()
        buttons.addWidget(auto_btn)
        buttons.addWidget(reset_btn)
        buttons.addWidget(self.status, 1)
        buttons.addWidget(bb)
        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addLayout(panes, 1)
        lay.addLayout(buttons)

        self._timer = QTimer(self, singleShot=True, interval=120, timeout=self._update_preview)
        self._use_auto()
        self.status.setText("Frame detected automatically." if self._auto else
                            "No frame found: place the points on the figure's edges.")

    # ---- handles
    def _use_auto(self):
        self._set_handles(handles_for(self._auto, self.img.size), auto=self._auto is not None)

    def _set_handles(self, pts, auto: bool = False):
        self._block = True
        for h, (x, y) in zip(self._handles, pts):
            h.setPos(QPointF(float(x), float(y)))
        self._block = False
        # The detected frame may use cubic edges, which 8 handles (parabolas) cannot express:
        # keep using it exactly until the user moves a point.
        self._auto_active = auto
        self._changed(user=False)

    def _points(self):
        return [(h.pos().x(), h.pos().y()) for h in self._handles]

    def _frame(self):
        if getattr(self, "_auto_active", False):
            return self._auto
        return frame_from_handles(self._points())

    def _changed(self, user: bool = True):
        if getattr(self, "_block", False) or not hasattr(self, "_timer"):
            return
        if user:
            self._auto_active = False
        frame = self._frame()
        path = QPainterPath()
        if frame is not None:
            for line in edge_polylines(frame):
                path.moveTo(*line[0])
                for p in line[1:]:
                    path.lineTo(*p)
        self._edges.setPath(path)
        self._timer.start()

    # ---- preview / result
    def _update_preview(self):
        frame = self._frame()
        if frame is None:
            self.status.setText("Points cross over: move them back into order.")
            return
        from .main_window import pil_to_qimage

        small, _ = straighten(self._small, scaled(frame, self._scale))
        pm = QPixmap.fromImage(pil_to_qimage(small))
        self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._timer.start()

    def result_image(self) -> Image.Image:
        frame = self._frame()
        return straighten(self.img, frame)[0] if frame is not None else self.img
