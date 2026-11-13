"""Model management: catalog, registry, download/cache, output converters."""

from .catalog import BUILTIN_MODELS, DEFAULT_MODEL_ID, GENERIC_PROMPTS
from .outputs import to_canonical
from .registry import Registry, custom_spec, delete_local, download, local_status, remote_size
from .spec import RUNTIME_SERVER, RUNTIME_TRANSFORMERS, TASKS, ModelSpec

__all__ = [
    "BUILTIN_MODELS",
    "DEFAULT_MODEL_ID",
    "GENERIC_PROMPTS",
    "ModelSpec",
    "Registry",
    "RUNTIME_SERVER",
    "RUNTIME_TRANSFORMERS",
    "TASKS",
    "custom_spec",
    "delete_local",
    "download",
    "local_status",
    "remote_size",
    "to_canonical",
]
