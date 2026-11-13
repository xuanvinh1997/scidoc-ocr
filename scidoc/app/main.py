"""Desktop entry point: ``scidoc-app`` or ``python -m scidoc``."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication

    QCoreApplication.setOrganizationName("SciDoc")
    QCoreApplication.setApplicationName("SciDoc OCR")
    # QtWebEngine must be imported before the QApplication exists.
    import PySide6.QtWebEngineWidgets  # noqa: F401

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("SciDoc OCR")

    from .main_window import MainWindow
    from .ui import apply_theme

    apply_theme(app)
    win = MainWindow()

    def on_scheme_changed(*_):
        apply_theme(app)
        win.retheme()

    try:
        app.styleHints().colorSchemeChanged.connect(on_scheme_changed)
    except AttributeError:  # Qt < 6.5
        pass
    win.show()
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    if files:
        win.open_paths(files)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
