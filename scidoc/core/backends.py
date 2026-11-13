"""OCR backends, driven by a :class:`~scidoc.core.models.spec.ModelSpec`.

* :class:`LocalBackend` runs a model in-process with 🤗 transformers (CUDA → MPS → CPU; ROCm planned).
* :class:`OpenAIBackend` talks to any OpenAI-compatible server: vLLM, SGLang, Ollama,
  LM Studio, llama.cpp ``llama-server``, Docker Model Runner, …

Both return the model's *raw* text; :func:`scidoc.core.models.to_canonical` normalises it.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from PIL import Image

from .models.catalog import DEFAULT_MODEL_ID
from .models.spec import ModelSpec

DEFAULT_MODEL = "xiaomi-research/Xiaomi-OCR-0"  # back-compat


class Task(str, Enum):
    DOCUMENT = "document"
    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"


TokenCallback = Callable[[str], None]


class Cancelled(Exception):
    pass


class OCRBackend(Protocol):
    name: str
    spec: ModelSpec

    def recognize(self, image: Image.Image, task: Task, on_token: TokenCallback | None = None,
                  cancel: threading.Event | None = None) -> str: ...

    def describe(self) -> str: ...


@dataclass
class GenerationConfig:
    max_new_tokens: int = 4096
    # Upper bound on pixels sent to the model (a model's own max_pixels may be lower).
    max_pixels: int = 4_000_000


def fit_pixels(img: Image.Image, max_pixels: int, min_pixels: int = 65_536) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    px = w * h
    if px > max_pixels:
        s = (max_pixels / px) ** 0.5
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)
    elif px < min_pixels:
        s = (min_pixels / px) ** 0.5
        img = img.resize((int(w * s) + 1, int(h * s) + 1), Image.Resampling.BICUBIC)
    return img


def model_input_size(backend, image: Image.Image) -> tuple[int, int]:
    """(width, height) of ``image`` after the same resizing ``backend.recognize`` applies."""
    spec, gen = getattr(backend, "spec", None), getattr(backend, "gen", None)
    if spec is None or gen is None:
        return image.size
    max_px, min_px = _pixel_budget(spec, gen)
    w, h = image.size
    px = w * h
    if px > max_px:
        s = (max_px / px) ** 0.5
        return max(1, int(w * s)), max(1, int(h * s))
    if px < min_px:
        s = (min_px / px) ** 0.5
        return int(w * s) + 1, int(h * s) + 1
    return w, h


def _resolve_spec(spec: ModelSpec | str | None) -> ModelSpec:
    if isinstance(spec, ModelSpec):
        return spec
    from .models.registry import Registry

    return Registry().get(spec or DEFAULT_MODEL_ID)


def _pixel_budget(spec: ModelSpec, gen: GenerationConfig) -> tuple[int, int]:
    return min(spec.max_pixels, gen.max_pixels), spec.min_pixels


def _max_tokens(spec: ModelSpec, gen: GenerationConfig) -> int:
    # The user's limit wins unless the model card asks for a larger budget (e.g. layout JSON).
    return max(gen.max_new_tokens, int(spec.generation.get("max_new_tokens", 0)))


# --------------------------------------------------------------------------- local


class LocalBackend:
    """In-process inference with transformers. The model is loaded lazily on first use."""

    name = "local"

    def __init__(self, spec: ModelSpec | str | None = None, device: str = "auto",
                 gen: GenerationConfig | None = None, precision: str = "auto"):
        self.spec = _resolve_spec(spec)
        self.requested_device = device
        self.precision = precision
        self.gen = gen or GenerationConfig()
        self._model = None
        self._processor = None
        self.device = None  # devices.Device once loaded
        self.dtype = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self.spec.repo

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, on_status: Callable[[str], None] | None = None) -> None:
        if self._model is not None:
            return
        if not self.spec.runs_locally:
            raise RuntimeError(
                f"{self.spec.name} needs custom code and runs through an OpenAI-compatible server. "
                f"Start one ({self.spec.serve_hint}) and select 'Server' in Settings."
            )
        from . import devices

        torch = devices._import_torch()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.utils import logging as hf_logging

        dev = devices.resolve_device(self.requested_device, torch)
        dtype = devices.resolve_dtype(dev, self.precision, torch, preferred=self.spec.dtype)
        devices.configure_backend(dev, torch)
        if dev.kind != "cuda":
            hf_logging.set_verbosity_error()  # CUDA-only kernel fallback notices are noise here
        if on_status:
            on_status(f"Loading {self.spec.name} on {dev.label} ({str(dtype).split('.')[-1]})…")
        kw = {"trust_remote_code": True} if self.spec.trust_remote_code else {}
        self._processor = AutoProcessor.from_pretrained(self.spec.repo, **kw)
        self._model = AutoModelForImageTextToText.from_pretrained(self.spec.repo, dtype=dtype, **kw)
        self._model = self._model.to(dev.id).eval()
        self.device, self.dtype = dev, dtype

    def unload(self) -> None:
        """Free GPU memory (when switching models)."""
        if self._model is None:
            return
        from . import devices

        dev = self.device
        self._model = self._processor = None
        import gc

        gc.collect()
        if dev is not None:
            devices.empty_cache(dev)

    def describe(self) -> str:
        if self.device is None:
            return f"Local · {self.spec.name} · {self.requested_device}"
        dt = str(self.dtype).split(".")[-1].replace("float", "fp").replace("bfp", "bf")
        return f"Local · {self.spec.name} · {self.device.kind.upper()} {self.device.name} · {dt}"

    def recognize(self, image, task, on_token=None, cancel=None) -> str:
        from . import devices

        self.load()
        max_px, min_px = _pixel_budget(self.spec, self.gen)
        for attempt in range(3):
            try:
                return self._generate(fit_pixels(image, max_px, min_px), Task(task), on_token, cancel)
            except Cancelled:
                raise
            except Exception as e:
                if not devices.is_oom(e) or attempt == 2:
                    raise
                devices.empty_cache(self.device)  # out of memory: retry at half the pixels
                max_px //= 2
        raise AssertionError("unreachable")

    def _messages(self, image, task: Task) -> list[dict]:
        _, prompt = self.spec.prompt_for(task.value)
        img_part = {"type": "image", "image": image}
        parts = [img_part]
        if prompt:
            text_part = {"type": "text", "text": prompt}
            parts = [text_part, img_part] if self.spec.text_first else [img_part, text_part]
        msgs = []
        if self.spec.system_prompt:
            msgs.append({"role": "system", "content": [{"type": "text", "text": self.spec.system_prompt}]})
        msgs.append({"role": "user", "content": parts})
        return msgs

    def _generate(self, image, task: Task, on_token, cancel) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        inputs = self._processor.apply_chat_template(
            self._messages(image, task), tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", **self.spec.chat_template_kwargs,
        ).to(self.device.id)
        if "pixel_values" in inputs and self.dtype is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        class _Stop(StoppingCriteria):
            def __call__(self, *_a, **_k):
                return bool(cancel and cancel.is_set())

        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                        skip_special_tokens=self.spec.skip_special_tokens)
        gen = {k: v for k, v in self.spec.generation.items() if k != "max_new_tokens"}
        gen.setdefault("do_sample", False)
        kwargs = dict(**inputs, max_new_tokens=_max_tokens(self.spec, self.gen), streamer=streamer,
                      stopping_criteria=StoppingCriteriaList([_Stop()]), **gen)
        error: list[BaseException] = []

        def run():
            try:
                with torch.inference_mode():
                    self._model.generate(**kwargs)
            except BaseException as e:  # surfaced below
                error.append(e)
                streamer.on_finalized_text("", stream_end=True)

        with self._lock:
            t = threading.Thread(target=run, daemon=True)
            t.start()
            parts = []
            for piece in streamer:
                parts.append(piece)
                if on_token and piece:
                    on_token(piece)
            t.join()
        if error:
            raise error[0]
        if cancel and cancel.is_set():
            raise Cancelled()
        return "".join(parts)


# --------------------------------------------------------------------------- remote


class OpenAIBackend:
    """Any server implementing ``POST /v1/chat/completions`` with image_url content."""

    name = "openai"

    def __init__(self, spec: ModelSpec | str | None = None, base_url: str = "http://localhost:8000/v1",
                 served_model: str = "", api_key: str = "", gen: GenerationConfig | None = None,
                 timeout: float = 600.0):
        self.spec = _resolve_spec(spec)
        self.base_url = base_url.rstrip("/")
        self.model = served_model or self.spec.repo
        self.api_key = api_key
        self.gen = gen or GenerationConfig()
        self.timeout = timeout

    def describe(self) -> str:
        return f"Server · {self.spec.name} · {self.base_url}"

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def list_models(self) -> list[str]:
        import httpx

        r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def build_body(self, image: Image.Image, task: Task) -> dict:
        max_px, min_px = _pixel_budget(self.spec, self.gen)
        image = fit_pixels(image, max_px, min_px)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        _, prompt = self.spec.prompt_for(Task(task).value)
        img_part = {"type": "image_url", "image_url": {"url": data_url}}
        parts = [img_part]
        if prompt:
            text_part = {"type": "text", "text": prompt}
            parts = [text_part, img_part] if self.spec.text_first else [img_part, text_part]
        messages = []
        if self.spec.system_prompt:
            messages.append({"role": "system", "content": self.spec.system_prompt})
        messages.append({"role": "user", "content": parts})
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": _max_tokens(self.spec, self.gen),
            "stream": True,
            "temperature": 0.0,
        }
        g = self.spec.generation
        if g.get("do_sample"):
            body["temperature"] = g.get("temperature", 0.7)
            if "top_p" in g:
                body["top_p"] = g["top_p"]
        if "repetition_penalty" in g:
            body["repetition_penalty"] = g["repetition_penalty"]  # vLLM / SGLang extension
        if self.spec.chat_template_kwargs:
            body["chat_template_kwargs"] = self.spec.chat_template_kwargs
        if not self.spec.skip_special_tokens:
            body["skip_special_tokens"] = False
        return body

    def recognize(self, image, task, on_token=None, cancel=None) -> str:
        import httpx

        body = self.build_body(image, Task(task))
        parts: list[str] = []
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", f"{self.base_url}/chat/completions", headers=self._headers(),
                               json=body) as r:
                if r.status_code >= 400:
                    r.read()
                    raise RuntimeError(f"Server error {r.status_code}: {r.text[:500]}")
                for line in r.iter_lines():
                    if cancel and cancel.is_set():
                        raise Cancelled()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            parts.append(piece)
                            if on_token:
                                on_token(piece)
        return "".join(parts)
