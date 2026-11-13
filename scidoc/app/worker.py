"""Background OCR runner (QThread) that streams tokens back to the UI."""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, QThread, Signal

from ..core.backends import Cancelled, LocalBackend, Task, model_input_size
from ..core.figures import BBOX_MARK_RE, FIGURE_LINK_RE, FigureStore, materialize


@dataclass
class Job:
    doc_id: int
    page_index: int
    image: object  # PIL.Image
    task: Task
    figures: FigureStore | None = None  # where cropped figures go (None = keep markers as text)


def postprocess(raw: str, task: Task, spec=None, image_size: tuple[int, int] | None = None) -> str:
    """Canonical Markdown for a page, using the model's output converter."""
    from ..core.models import Registry, to_canonical

    spec = spec or Registry().get(None)
    return to_canonical(raw, spec, Task(task).value, image_size)


class OCRWorker(QObject):
    status = Signal(str)
    pageStarted = Signal(int, int)  # doc_id, page
    token = Signal(int, int, str)
    pageDone = Signal(int, int, str, str)  # doc_id, page, markdown, raw
    pageFailed = Signal(int, int, str)
    finished = Signal()

    def __init__(self, backend, jobs: list[Job], concurrency: int = 1, detector=None):
        super().__init__()
        self.backend = backend
        self.jobs = jobs
        # Layout detector (scidoc.core.layout.LayoutDetector) used to crop figures on full pages.
        self.detector = detector if any(j.task == Task.DOCUMENT and j.figures for j in jobs) else None
        self.concurrency = 1 if isinstance(backend, LocalBackend) else max(1, concurrency)
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            if isinstance(self.backend, LocalBackend):
                self.backend.load(self.status.emit)
                self.status.emit(self.backend.describe())
            if self.detector is not None and not self.detector.loaded:
                try:
                    self.detector.load(self.status.emit)
                except Exception as e:  # noqa: BLE001 - figures are optional, OCR goes on
                    traceback.print_exc()
                    self.status.emit(f"Layout detection unavailable: {e}")
                    self.detector = None
            if self.concurrency == 1:
                for job in self.jobs:
                    if self.cancel_event.is_set():
                        break
                    self._one(job)
            else:
                with ThreadPoolExecutor(self.concurrency) as pool:
                    list(pool.map(self._one, self.jobs))
        except Exception as e:  # model load failures etc.
            traceback.print_exc()
            self.status.emit(f"Error: {e}")
            for job in self.jobs:
                self.pageFailed.emit(job.doc_id, job.page_index, str(e))
        finally:
            self.finished.emit()

    def _one(self, job: Job):
        if self.cancel_event.is_set():
            self.pageFailed.emit(job.doc_id, job.page_index, "cancelled")
            return
        self.pageStarted.emit(job.doc_id, job.page_index)
        try:
            raw = self.backend.recognize(
                job.image,
                job.task,
                on_token=lambda t: self.token.emit(job.doc_id, job.page_index, t),
                cancel=self.cancel_event,
            )
            md = postprocess(raw, job.task, self.backend.spec, model_input_size(self.backend, job.image))
            if job.figures is not None:
                pictures, text_boxes = self._detect(job, md)
                md = materialize(md, job.image, job.page_index, job.figures, pictures, text_boxes)
            self.pageDone.emit(job.doc_id, job.page_index, md, raw)
        except Cancelled:
            self.pageFailed.emit(job.doc_id, job.page_index, "cancelled")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.pageFailed.emit(job.doc_id, job.page_index, str(e))


    def _detect(self, job: Job, md: str):
        """(picture, text) boxes from the layout model, only for full pages the OCR model left without figures."""
        if (self.detector is None or job.task != Task.DOCUMENT or BBOX_MARK_RE.search(md)
                or FIGURE_LINK_RE.search(md)):
            return None, None
        try:
            return self.detector.analyze(job.image)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.status.emit(f"Layout detection failed: {e}")
            return None, None


def start_worker(worker: OCRWorker, on_done) -> QThread:
    """Run ``worker`` on a new QThread; ``on_done`` is called on the GUI thread once the thread has stopped.

    Never call ``QThread.wait()`` from a slot of ``worker.finished``: that signal is delivered to
    the GUI thread *before* the queued ``quit()``, so waiting there deadlocks the UI.
    """
    th = QThread()
    worker.moveToThread(th)
    th.started.connect(worker.run)
    # quit() is thread-safe: call it directly from the worker thread instead of queueing it
    # behind GUI events.
    worker.finished.connect(th.quit, Qt.ConnectionType.DirectConnection)
    th.finished.connect(on_done)  # emitted after run() returned and the event loop exited
    th.finished.connect(worker.deleteLater)
    th.finished.connect(th.deleteLater)
    th.start()
    return th
