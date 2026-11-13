"""On-disk cache of recognised pages so reopening a document never re-runs OCR."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def cache_root() -> Path:
    if env := os.environ.get("SCIDOC_CACHE"):
        return Path(env)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "scidoc"


class PageCache:
    def __init__(self, digest: str, task: str, root: Path | None = None):
        self.dir = (root or cache_root()) / digest / task
        self.enabled = bool(digest)

    def _path(self, index: int, kind: str) -> Path:
        return self.dir / f"page-{index + 1:04d}.{kind}"

    def get(self, index: int) -> tuple[str, str] | None:
        """Return (markdown, raw) or None. User edits are stored as markdown."""
        if not self.enabled:
            return None
        md = self._path(index, "md")
        if not md.exists():
            return None
        raw = self._path(index, "raw")
        return md.read_text(encoding="utf-8"), raw.read_text(encoding="utf-8") if raw.exists() else ""

    def put(self, index: int, markdown: str, raw: str | None = None) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path(index, "md").write_text(markdown, encoding="utf-8")
        if raw is not None:
            self._path(index, "raw").write_text(raw, encoding="utf-8")

    def clear(self, index: int) -> None:
        for kind in ("md", "raw"):
            self._path(index, kind).unlink(missing_ok=True)
