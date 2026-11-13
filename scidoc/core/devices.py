"""Compute-device detection and policy for the local engine.

Support matrix
==============
=========  ==========  =====================================================
kind       status      notes
=========  ==========  =====================================================
cuda       supported   NVIDIA, Linux/Windows. fp16 (model-card verified) or bf16.
mps        supported   Apple Silicon, macOS. fp16. CPU fallback for missing ops.
cpu        supported   fp32. Slow but always available.
rocm       planned     AMD via PyTorch ROCm builds. Detected and listed, but not
                       selected unless SCIDOC_EXPERIMENTAL_ROCM=1.
=========  ==========  =====================================================

ROCm builds of PyTorch expose AMD GPUs through the ``torch.cuda`` API, so we tell
them apart via ``torch.version.hip``. Everything here takes the torch module as a
parameter so the policy is unit-testable without GPUs.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field

# Must be set before torch initialises MPS: run unsupported ops on CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROCM_FLAG = "SCIDOC_EXPERIMENTAL_ROCM"
PRECISIONS = ("auto", "fp16", "bf16", "fp32")


class DeviceError(RuntimeError):
    pass


@dataclass
class Device:
    id: str  # torch device string: "cuda:0", "mps", "cpu"
    kind: str  # cuda | rocm | mps | cpu
    name: str
    memory_gb: float | None = None
    supported: bool = True
    note: str = ""
    bf16: bool = False

    @property
    def label(self) -> str:
        mem = f", {self.memory_gb:.0f} GB" if self.memory_gb else ""
        tag = "" if self.supported else " — not yet supported"
        return f"{self.id} · {self.name}{mem}{tag}"


@dataclass
class TorchBuild:
    torch: str
    cuda: str | None = None
    hip: str | None = None
    mps_built: bool = False
    kernels: dict[str, bool] = field(default_factory=dict)

    @property
    def flavor(self) -> str:
        if self.hip:
            return f"ROCm {self.hip}"
        if self.cuda:
            return f"CUDA {self.cuda}"
        if self.mps_built:
            return "MPS"
        return "CPU"


def rocm_enabled() -> bool:
    return os.environ.get(ROCM_FLAG, "").strip().lower() in ("1", "true", "yes")


def _import_torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - depends on install
        raise DeviceError(
            "PyTorch is not installed. Install the local engine: see README → 'Install' "
            "(or switch Settings → OCR engine to an OpenAI-compatible server)."
        ) from e
    return torch


def torch_build(torch=None) -> TorchBuild:
    torch = torch or _import_torch()
    v = getattr(torch, "version", None)
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return TorchBuild(
        torch=getattr(torch, "__version__", "?"),
        cuda=getattr(v, "cuda", None),
        hip=getattr(v, "hip", None),
        mps_built=bool(mps and mps.is_built()),
        kernels={
            # Optional fused kernels for Qwen3.5's linear-attention layers (CUDA only).
            "flash-linear-attention": importlib.util.find_spec("fla") is not None,
            "causal-conv1d": importlib.util.find_spec("causal_conv1d") is not None,
        },
    )


def list_devices(torch=None) -> list[Device]:
    torch = torch or _import_torch()
    build = torch_build(torch)
    out: list[Device] = []
    if torch.cuda.is_available():
        is_rocm = bool(build.hip)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem = getattr(props, "total_memory", 0) / 2**30 or None
            if is_rocm:
                ok = rocm_enabled()
                note = (
                    "experimental (SCIDOC_EXPERIMENTAL_ROCM=1)" if ok
                    else f"ROCm support is planned; set {ROCM_FLAG}=1 to try it"
                )
                out.append(Device(f"cuda:{i}", "rocm", props.name, mem, ok, note))
            else:
                major = getattr(props, "major", 0)
                out.append(Device(f"cuda:{i}", "cuda", props.name, mem, True, bf16=major >= 8))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        out.append(Device("mps", "mps", _apple_chip_name(), None, True))
    out.append(Device("cpu", "cpu", "CPU", None, True))
    return out


def resolve_device(requested: str = "auto", torch=None) -> Device:
    """Pick the device. ``auto`` = first supported CUDA GPU → MPS → CPU."""
    devices = list_devices(torch)
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        for kind in ("cuda", "rocm", "mps"):
            for d in devices:
                if d.kind == kind and d.supported:
                    return d
        return devices[-1]  # cpu
    if requested == "cuda":
        requested = "cuda:0"
    for d in devices:
        if d.id == requested or d.kind == requested:
            if not d.supported:
                raise DeviceError(f"{d.label}: {d.note}")
            return d
    available = ", ".join(d.id for d in devices)
    raise DeviceError(f"Device '{requested}' is not available. Available: {available}")


def resolve_dtype(device: Device, precision: str = "auto", torch=None, preferred: str = "fp16"):
    """Pick the compute dtype.

    ``auto`` uses the model's ``preferred`` precision (from its model card) on GPUs and fp32 on CPU.
    bf16-preferring models fall back to fp32 — not fp16 — on GPUs without bf16, because some
    (e.g. Granite-Docling) overflow in fp16.
    """
    torch = torch or _import_torch()
    precision = (precision or "auto").lower()
    if precision == "auto":
        if device.kind == "cpu":
            precision = "fp32"
        else:
            precision = preferred if preferred in ("fp16", "bf16", "fp32") else "fp16"
            if precision == "bf16" and device.kind == "cuda" and not device.bf16:
                precision = "fp32"
    elif precision == "bf16" and device.kind == "cuda" and not device.bf16:
        raise DeviceError(f"{device.name} does not support bf16 (needs compute capability ≥ 8.0)")
    if precision == "fp16" and device.kind == "cpu":
        precision = "fp32"  # fp16 matmuls on CPU are slow or unsupported
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


def configure_backend(device: Device, torch=None) -> None:
    """Per-device global knobs, applied once before loading the model."""
    torch = torch or _import_torch()
    if device.kind == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(device.id)


def empty_cache(device: Device, torch=None) -> None:
    torch = torch or _import_torch()
    if device.kind in ("cuda", "rocm"):
        torch.cuda.empty_cache()
    elif device.kind == "mps":
        torch.mps.empty_cache()


def is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return name == "OutOfMemoryError" or "out of memory" in msg or "mps backend out of memory" in msg


def _apple_chip_name() -> str:
    import platform
    import subprocess

    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2
        ).stdout.strip() or "Apple GPU"
    except Exception:  # noqa: BLE001
        return f"Apple GPU ({platform.machine()})"


def report(torch=None) -> str:
    """Human-readable diagnostics for ``scidoc devices`` and bug reports."""
    torch = torch or _import_torch()
    b = torch_build(torch)
    lines = [f"PyTorch {b.torch} ({b.flavor} build)"]
    for d in list_devices(torch):
        mark = "✓" if d.supported else "–"
        lines.append(f"  {mark} {d.label}" + (f"  [{d.note}]" if d.note else ""))
    try:
        auto = resolve_device("auto", torch)
        lines.append(f"auto → {auto.id} ({auto.kind}), dtype {str(resolve_dtype(auto, 'auto', torch)).split('.')[-1]}")
    except DeviceError as e:
        lines.append(f"auto → error: {e}")
    if b.cuda and not b.hip:
        missing = [k for k, ok in b.kernels.items() if not ok]
        lines.append(
            "fast kernels: " + ("all installed" if not missing else
                                f"missing {', '.join(missing)} → uv pip install flash-linear-attention causal-conv1d "
                                "--no-build-isolation   (Linux; optional, faster decoding)")
        )
    if not torch.cuda.is_available() and not (b.mps_built and torch.backends.mps.is_available()):
        lines.append("hint: no GPU found. For NVIDIA install a CUDA build of torch (README → Install).")
    return "\n".join(lines)
