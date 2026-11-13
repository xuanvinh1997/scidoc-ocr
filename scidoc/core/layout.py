"""Page layout detection (Docling *Heron*, RT-DETRv2, Apache-2.0) used to find figures.

The detector is independent of the OCR model: it runs on the page image after recognition so
that pictures can be cropped even when the OCR model only writes ``<img>caption</img>``.
"""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

from .figures import Box

LAYOUT_REPO = "docling-project/docling-layout-heron"
LAYOUT_NAME = "Docling Heron layout"
PICTURE_LABELS = frozenset({"picture"})
# Regions whose content ends up in the OCR text (used to place figures between paragraphs).
TEXT_LABELS = frozenset({"text", "title", "section_header", "list_item", "formula", "caption", "footnote",
                         "table", "code", "document_index", "form", "key_value_region"})


@dataclass(frozen=True)
class Region:
    label: str  # lower-case Heron label: picture, table, formula, text, …
    score: float
    box: Box  # normalised to the page


def available() -> bool:
    """True when the local engine (torch + transformers) is installed."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


class LayoutDetector:
    def __init__(self, device: str = "auto", repo: str = LAYOUT_REPO, threshold: float = 0.6):
        self.requested_device = device
        self.repo = repo
        self.threshold = threshold
        self._model = None
        self._processor = None
        self._device = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, on_status: Callable[[str], None] | None = None) -> None:
        with self._lock:
            if self._model is not None:
                return
            from . import devices

            torch = devices._import_torch()
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

            dev = devices.resolve_device(self.requested_device, torch)
            if on_status:
                on_status(f"Loading {LAYOUT_NAME} on {dev.label}…")
            self._processor = RTDetrImageProcessor.from_pretrained(self.repo)
            self._model = RTDetrV2ForObjectDetection.from_pretrained(self.repo).to(dev.id).eval()
            self._device = dev.id

    def unload(self) -> None:
        with self._lock:
            self._model = self._processor = None

    def detect(self, image: Image.Image) -> list[Region]:
        self.load()
        import torch

        img = image.convert("RGB")
        w, h = img.size
        with self._lock, torch.inference_mode():
            inputs = self._processor(images=[img], return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)
            result = self._processor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([[h, w]], device=self._device), threshold=self.threshold
            )[0]
        labels = self._model.config.id2label
        regions = []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            regions.append(Region(str(labels[int(label)]).lower(), float(score),
                                  (max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h))))
        return regions

    def analyze(self, image: Image.Image) -> tuple[list[Box], list[Box]]:
        """(picture boxes, text boxes) for a page."""
        regions = self.detect(image)
        return self._pictures(regions), [r.box for r in regions if r.label in TEXT_LABELS]

    def pictures(self, image: Image.Image) -> list[Box]:
        return self._pictures(self.detect(image))

    @staticmethod
    def _pictures(regions: list[Region], min_area: float = 0.008) -> list[Box]:
        """Picture boxes, ignoring specks (logos, bullets) and boxes nested in a larger picture."""
        boxes = [r.box for r in regions if r.label in PICTURE_LABELS
                 and (r.box[2] - r.box[0]) * (r.box[3] - r.box[1]) >= min_area]
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        kept: list[Box] = []
        for b in boxes:
            if not any(b[0] >= k[0] - 0.01 and b[1] >= k[1] - 0.01 and b[2] <= k[2] + 0.01 and b[3] <= k[3] + 0.01
                       for k in kept):
                kept.append(b)
        return kept
