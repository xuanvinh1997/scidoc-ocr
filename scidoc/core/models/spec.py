"""Declarative description of an OCR model: how to load it, prompt it and read its output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TASKS = ("document", "text", "formula", "table")

# How the model is executed.
RUNTIME_TRANSFORMERS = "transformers"  # loads in-process with stock transformers classes
RUNTIME_SERVER = "server"  # custom code / special runtime: use vLLM, SGLang, Ollama… (OpenAI API)

# Output formats understood by scidoc.core.models.outputs
OUTPUT_FORMATS = (
    "markdown",  # Markdown + LaTeX; tables as Markdown/HTML/OTSL (auto-detected)
    "plain",  # plain text
    "latex",  # bare LaTeX (formula tasks)
    "doctags",  # IBM Docling DocTags
    "layout-json",  # list of {bbox, category, text} (dots.ocr style)
    "front-matter",  # YAML front matter + Markdown (olmOCR)
)


@dataclass
class ModelSpec:
    id: str  # stable slug used in settings, e.g. "xiaomi-ocr-0"
    name: str
    repo: str  # Hugging Face repo id or local folder
    family: str = "generic"
    params_b: float = 0.0
    license: str = ""
    languages: list[str] = field(default_factory=list)
    description: str = ""
    strengths: list[str] = field(default_factory=list)  # e.g. ["math", "tables", "handwriting"]
    runtime: str = RUNTIME_TRANSFORMERS
    trust_remote_code: bool = False
    # Prompting ---------------------------------------------------------------------------
    prompts: dict[str, str] = field(default_factory=dict)  # task -> prompt ("" = image only)
    system_prompt: str = ""
    text_first: bool = False  # put the prompt before the image in the user turn
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    # Decoding / preprocessing ------------------------------------------------------------
    generation: dict[str, Any] = field(default_factory=dict)  # merged into generate()/API body
    max_pixels: int = 4_000_000
    min_pixels: int = 65_536
    dtype: str = "bf16"  # preferred GPU precision from the model card: fp16 | bf16 | fp32
    skip_special_tokens: bool = True  # False for formats that live in special tokens (DocTags)
    output: str = "markdown"
    output_by_task: dict[str, str] = field(default_factory=dict)
    # Bookkeeping ---------------------------------------------------------------------------
    serve_hint: str = ""  # e.g. "vllm serve <repo> --trust-remote-code"
    notes: str = ""
    builtin: bool = True
    verified: bool = False  # end-to-end tested by SciDoc maintainers

    # ------------------------------------------------------------------ helpers
    def supports(self, task: str) -> bool:
        return task in self.prompts

    def prompt_for(self, task: str) -> tuple[str, str]:
        """(effective_task, prompt). Unsupported tasks fall back to the full-page prompt."""
        if task in self.prompts:
            return task, self.prompts[task]
        for fb in ("document", "text"):
            if fb in self.prompts:
                return fb, self.prompts[fb]
        raise ValueError(f"{self.name} has no usable prompt")

    def output_for(self, task: str) -> str:
        return self.output_by_task.get(task, self.output)

    @property
    def runs_locally(self) -> bool:
        return self.runtime == RUNTIME_TRANSFORMERS

    @property
    def is_local_path(self) -> bool:
        return self.repo.startswith(("/", "~", ".")) or (len(self.repo) > 1 and self.repo[1] == ":")

    def est_vram_gb(self) -> float:
        """Rough fp16 footprint: weights ×2 bytes plus ~35% for activations/KV cache."""
        return round(self.params_b * 2 * 1.35 + 0.5, 1) if self.params_b else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
