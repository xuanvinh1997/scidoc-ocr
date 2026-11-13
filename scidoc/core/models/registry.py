"""Model registry: built-in catalog + user-defined models, plus Hugging Face cache management."""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .catalog import BUILTIN_MODELS, DEFAULT_MODEL_ID
from .spec import ModelSpec


def config_dir() -> Path:
    if env := os.environ.get("SCIDOC_CONFIG"):
        return Path(env)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "scidoc"


class Registry:
    """All known models. User models live in ``<config>/models.json`` and may override built-ins."""

    def __init__(self, path: Path | None = None):
        self.path = path or (config_dir() / "models.json")
        self._user: dict[str, ModelSpec] = {}
        self.reload()

    def reload(self) -> None:
        self._user = {}
        if self.path.exists():
            try:
                for d in json.loads(self.path.read_text(encoding="utf-8")):
                    spec = ModelSpec.from_dict({**d, "builtin": False})
                    self._user[spec.id] = spec
            except (json.JSONDecodeError, TypeError) as e:
                print(f"scidoc: ignoring broken {self.path}: {e}", file=sys.stderr)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self._user.values()]
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def all(self) -> list[ModelSpec]:
        merged = {s.id: s for s in BUILTIN_MODELS}
        merged.update(self._user)
        return list(merged.values())

    def get(self, model_id: str | None) -> ModelSpec:
        model_id = model_id or DEFAULT_MODEL_ID
        for s in self.all():
            if s.id == model_id:
                return s
        # Allow a raw HF repo id / folder: treat it as a generic VLM with the default prompts.
        for s in self.all():
            if s.repo == model_id:
                return s
        return custom_spec(model_id)

    def add_or_update(self, spec: ModelSpec) -> None:
        spec = ModelSpec.from_dict({**spec.to_dict(), "builtin": False})  # never alias catalog objects
        self._user[spec.id] = spec
        self._save()

    def remove(self, model_id: str) -> None:
        if self._user.pop(model_id, None) is not None:
            self._save()

    def is_user(self, model_id: str) -> bool:
        return model_id in self._user


def custom_spec(repo: str, name: str | None = None) -> ModelSpec:
    """A user-supplied model: generic VLM chat with generic OCR prompts."""
    from .catalog import GENERIC_PROMPTS

    slug = repo.strip("/").replace("/", "--").lower()
    return ModelSpec(
        id=f"custom-{slug}",
        name=name or repo.split("/")[-1],
        repo=repo,
        family="generic",
        prompts=dict(GENERIC_PROMPTS),
        description="Custom model (generic vision-language chat prompts).",
        builtin=False,
    )


# --------------------------------------------------------------------------- HF cache


@dataclass
class LocalStatus:
    downloaded: bool
    size_bytes: int = 0
    path: str = ""


def local_status(spec: ModelSpec) -> LocalStatus:
    """Is the model fully available offline?"""
    if spec.is_local_path:
        p = Path(spec.repo).expanduser()
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0
        return LocalStatus(p.exists(), size, str(p))
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(spec.repo, local_files_only=True, allow_patterns=_ALLOW)
    except Exception:  # noqa: BLE001 - not (fully) cached
        return LocalStatus(False, cached_size(spec.repo))
    p = Path(path)
    weights = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return LocalStatus(bool(weights), size, path)


def cached_size(repo: str) -> int:
    try:
        from huggingface_hub import scan_cache_dir

        for r in scan_cache_dir().repos:
            if r.repo_id == repo and r.repo_type == "model":
                return r.size_on_disk
    except Exception:  # noqa: BLE001
        pass
    return 0


# Skip things a runtime never needs (GGUF variants, ONNX, images in the card, …).
_ALLOW = ["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja", "*.tiktoken", "tokenizer*", "*.bin"]
_IGNORE = ["*.gguf", "*.onnx", "onnx/*", "*.msgpack", "*.h5", "assets/*", "*.png", "*.jpg", "*.gif", "*.pdf"]


def remote_size(spec: ModelSpec) -> int:
    """Total bytes to download (0 if unknown)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(spec.repo, files_metadata=True)
        from fnmatch import fnmatch

        total = 0
        for s in info.siblings or []:
            name = s.rfilename
            if any(fnmatch(name, pat) for pat in _IGNORE):
                continue
            if any(fnmatch(name.split("/")[-1], pat) or fnmatch(name, pat) for pat in _ALLOW):
                total += s.size or 0
        return total
    except Exception:  # noqa: BLE001
        return 0


class DownloadCancelled(Exception):
    pass


def download(
    spec: ModelSpec,
    on_progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
) -> str:
    """Download a model snapshot into the HF cache. ``on_progress(done_bytes, total_bytes)``."""
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm as _tqdm

    total = remote_size(spec)
    lock = threading.Lock()
    state = {"done": 0}

    class Progress(_tqdm):  # huggingface_hub reports byte progress through tqdm instances
        def __init__(self, *a, **k):
            k["disable"] = True
            super().__init__(*a, **k)

        def update(self, n=1):
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled()
            if n and getattr(self, "unit", "") == "B":
                with lock:
                    state["done"] += n
                    done = state["done"]
                if on_progress:
                    on_progress(min(done, total) if total else done, total)
            return super().update(n)

    return snapshot_download(spec.repo, allow_patterns=_ALLOW, ignore_patterns=_IGNORE, tqdm_class=Progress)


def delete_local(spec: ModelSpec) -> int:
    """Remove all cached revisions of the model. Returns freed bytes."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    revs = [rev.commit_hash for r in info.repos if r.repo_id == spec.repo and r.repo_type == "model"
            for rev in r.revisions]
    if not revs:
        return 0
    strategy = info.delete_revisions(*revs)
    strategy.execute()
    return strategy.expected_freed_size
