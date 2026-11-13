"""GUI regression: a finished OCR run must return the window to idle (no deadlock)."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWebEngineWidgets")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


class FakeBackend:
    name = "fake"

    def __init__(self):
        from scidoc.core.models import Registry

        self.spec = Registry().get(None)

    def describe(self):
        return "fake"

    def recognize(self, image, task, on_token=None, cancel=None):
        out = "# Title\n\nSome \\(x^2\\) text."
        for ch in out:
            if on_token:
                on_token(ch)
        time.sleep(0.02)
        return out


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    os.environ["SCIDOC_CACHE"] = str(tmp_path_factory.mktemp("cache"))
    QCoreApplication.setOrganizationName("SciDocTest")
    QCoreApplication.setApplicationName("SciDocTest")
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return QApplication.instance() or QApplication([])


def _pump_until(app, cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_run_completes_and_queue_drains(app):
    from scidoc.app.main_window import MainWindow
    from scidoc.core.backends import Task
    from scidoc.core.sources import Page, Source

    w = MainWindow()
    w.settings.auto_run = False
    fake = FakeBackend()
    w._backend_for = lambda model_id: fake
    w._needs_download = lambda model_id: False

    def make_doc(name, n):
        imgs = [Image.new("RGB", (64 + i, 64), "white") for i in range(n)]
        return w._add_doc(Source(name, [Page(i, im) for i, im in enumerate(imgs)], digest=""), Task.DOCUMENT)

    d1 = make_doc("a", 2)
    d2 = make_doc("b", 1)
    w._run_doc(d1)
    w._run_doc(d2)  # queued behind the first worker

    assert _pump_until(app, lambda: w.worker is None and not w.queue), "OCR run never returned to idle"
    assert all(p.status == "done" for d in (d1, d2) for p in d.source.pages)
    assert not w.a_stop.isEnabled()

    # A second run after completion must work too (thread objects were cleaned up correctly).
    w.rerun_page()
    assert _pump_until(app, lambda: w.worker is None), "second run deadlocked"
    w.close()
