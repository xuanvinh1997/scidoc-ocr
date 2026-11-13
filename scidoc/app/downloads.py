"""Background model downloads with progress, shared by the main window and the Model Manager."""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal

from ..core.models import ModelSpec, download
from ..core.models.registry import DownloadCancelled


class DownloadManager(QObject):
    progress = Signal(str, object, object)  # model_id, done_bytes, total_bytes (python ints: may exceed 2^31)
    finished = Signal(str, bool, str)  # model_id, ok, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active: dict[str, threading.Event] = {}
        self._progress: dict[str, tuple[int, int]] = {}

    def is_active(self, model_id: str) -> bool:
        return model_id in self._active

    def last_progress(self, model_id: str) -> tuple[int, int]:
        return self._progress.get(model_id, (0, 0))

    def start(self, spec: ModelSpec) -> None:
        if spec.id in self._active:
            return
        cancel = threading.Event()
        self._active[spec.id] = cancel
        self._progress[spec.id] = (0, 0)

        def on_progress(done, total):
            self._progress[spec.id] = (done, total)
            self.progress.emit(spec.id, done, total)  # queued to the GUI thread

        def run():
            try:
                path = download(spec, on_progress, cancel)
                self._done(spec.id, True, path)
            except DownloadCancelled:
                self._done(spec.id, False, "cancelled")
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._done(spec.id, False, str(e))

        threading.Thread(target=run, daemon=True, name=f"download-{spec.id}").start()

    def _done(self, model_id: str, ok: bool, msg: str) -> None:
        self._active.pop(model_id, None)
        self._progress.pop(model_id, None)
        self.finished.emit(model_id, ok, msg)

    def cancel(self, model_id: str) -> None:
        ev = self._active.get(model_id)
        if ev:
            ev.set()
