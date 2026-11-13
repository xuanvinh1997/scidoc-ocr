"""Build a standalone desktop bundle with PyInstaller (run on each target OS).

    uv pip install -e ".[local,dev]"
    python scripts/build_app.py            # -> dist/SciDoc OCR(.app|.exe)

The bundle includes torch when the `local` extra is installed (large, ~1-2 GB).
For a slim "server-only" build, install without `[local]`; the app then needs an
OpenAI-compatible endpoint (vLLM / SGLang / Ollama) configured in Settings.
"""

import os
import sys
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent.parent
sep = os.pathsep

args = [
    str(ROOT / "scidoc" / "app" / "main.py"),
    "--name", "SciDoc OCR",
    "--windowed",
    "--noconfirm",
    "--add-data", f"{ROOT / 'scidoc' / 'assets'}{sep}scidoc/assets",
    "--collect-submodules", "scidoc",
    "--hidden-import", "PySide6.QtWebEngineWidgets",
]
try:
    import torch  # noqa: F401

    args += ["--collect-data", "transformers", "--copy-metadata", "transformers", "--copy-metadata", "torch"]
except ImportError:
    print("torch not installed: building server-backend-only app", file=sys.stderr)

PyInstaller.__main__.run(args)
