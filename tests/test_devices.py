"""Device policy tests with a fake torch module (no GPU needed)."""

from types import SimpleNamespace

import pytest

from scidoc.core import devices
from scidoc.core.devices import DeviceError, list_devices, resolve_device, resolve_dtype


def fake_torch(cuda=(), hip=None, cuda_version="13.0", mps=False):
    """cuda: list of (name, total_GB, compute_major)."""
    props = [SimpleNamespace(name=n, total_memory=int(g * 2**30), major=m) for n, g, m in cuda]
    return SimpleNamespace(
        __version__="2.14.0",
        version=SimpleNamespace(cuda=None if hip else (cuda_version if cuda else None), hip=hip),
        cuda=SimpleNamespace(
            is_available=lambda: bool(props),
            device_count=lambda: len(props),
            get_device_properties=lambda i: props[i],
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps, is_built=lambda: mps)),
        float16="fp16", bfloat16="bf16", float32="fp32",
    )


def test_auto_prefers_cuda_then_mps_then_cpu():
    t = fake_torch(cuda=[("RTX 4090", 24, 8)], mps=False)
    assert resolve_device("auto", t).id == "cuda:0"
    t = fake_torch(mps=True)
    assert resolve_device("auto", t).id == "mps"
    t = fake_torch()
    assert resolve_device("auto", t).id == "cpu"


def test_multi_gpu_selection():
    t = fake_torch(cuda=[("A100", 80, 8), ("T4", 16, 7)])
    assert [d.id for d in list_devices(t)] == ["cuda:0", "cuda:1", "cpu"]
    assert resolve_device("cuda:1", t).name == "T4"
    assert resolve_device("cuda", t).id == "cuda:0"
    with pytest.raises(DeviceError):
        resolve_device("cuda:5", t)
    with pytest.raises(DeviceError):
        resolve_device("mps", t)


def test_dtype_policy():
    t = fake_torch(cuda=[("A100", 80, 8), ("T4", 16, 7)])
    a100, t4, cpu = list_devices(t)
    assert resolve_dtype(a100, "auto", t) == "fp16"  # model-card verified
    assert resolve_dtype(a100, "bf16", t) == "bf16"
    with pytest.raises(DeviceError):
        resolve_dtype(t4, "bf16", t)  # Turing has no bf16
    assert resolve_dtype(cpu, "auto", t) == "fp32"
    assert resolve_dtype(cpu, "fp16", t) == "fp32"


def test_rocm_listed_but_not_selected(monkeypatch):
    monkeypatch.delenv(devices.ROCM_FLAG, raising=False)
    t = fake_torch(cuda=[("AMD Instinct MI300X", 192, 9)], hip="7.2")
    ds = list_devices(t)
    assert ds[0].kind == "rocm" and not ds[0].supported
    assert resolve_device("auto", t).id == "cpu"  # planned: falls back to CPU
    with pytest.raises(DeviceError, match="planned"):
        resolve_device("cuda:0", t)


def test_rocm_experimental_opt_in(monkeypatch):
    monkeypatch.setenv(devices.ROCM_FLAG, "1")
    t = fake_torch(cuda=[("AMD Radeon PRO W7900", 48, 11)], hip="7.2")
    d = resolve_device("auto", t)
    assert d.kind == "rocm" and d.supported
    assert resolve_dtype(d, "auto", t) == "fp16"


def test_oom_detection():
    class OutOfMemoryError(RuntimeError):
        pass

    assert devices.is_oom(OutOfMemoryError("x"))
    assert devices.is_oom(RuntimeError("MPS backend out of memory (MPS allocated: 30 GB)"))
    assert not devices.is_oom(ValueError("bad"))


def test_dtype_follows_model_preference():
    t = fake_torch(cuda=[("A100", 80, 8), ("T4", 16, 7)], mps=True)
    a100, t4, mps, cpu = list_devices(t)
    assert resolve_dtype(a100, "auto", t, preferred="bf16") == "bf16"
    assert resolve_dtype(t4, "auto", t, preferred="bf16") == "fp32"  # no bf16 on Turing → fp32, not fp16
    assert resolve_dtype(mps, "auto", t, preferred="bf16") == "bf16"
    assert resolve_dtype(cpu, "auto", t, preferred="bf16") == "fp32"
