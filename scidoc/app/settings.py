"""Persistent settings (QSettings) and the Settings dialog."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

import json

from ..core.backends import GenerationConfig, LocalBackend, OpenAIBackend
from ..core.models import DEFAULT_MODEL_ID, ModelSpec


@dataclass
class AppSettings:
    backend: str = "local"  # local = run on this computer when the model allows | openai = always server
    doc_model: str = DEFAULT_MODEL_ID  # model used for opened documents
    snip_model: str = ""  # model used for snips/pastes ("" = same as doc_model)
    served_names: str = "{}"  # JSON {model_id: name on the server}, default = HF repo id
    device: str = "auto"
    precision: str = "auto"
    server_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    concurrency: int = 4
    dpi: int = 150
    max_tokens: int = 4096
    snip_task: str = "formula"
    formula_copy: str = "inline"  # inline $..$ | display \[..\] | raw
    auto_run: bool = True
    page_size: str = "A4"
    layout_detect: bool = True  # crop figures with the layout model when the OCR model gives no boxes
    straighten_figures: bool = True  # unbend photographed plots using their frame

    @classmethod
    def load(cls) -> "AppSettings":
        q = QSettings()
        s = cls()
        for f in fields(cls):
            if q.contains(f.name):
                v = q.value(f.name)
                if f.type in ("int", int):
                    v = int(v)
                elif f.type in ("bool", bool):
                    v = v in (True, "true", "1", 1)
                setattr(s, f.name, v)
        return s

    def save(self) -> None:
        q = QSettings()
        for k, v in asdict(self).items():
            q.setValue(k, v)

    # ---- models
    def served_name(self, spec: ModelSpec) -> str:
        try:
            return json.loads(self.served_names or "{}").get(spec.id, "") or spec.repo
        except json.JSONDecodeError:
            return spec.repo

    def set_served_name(self, spec: ModelSpec, name: str) -> None:
        try:
            d = json.loads(self.served_names or "{}")
        except json.JSONDecodeError:
            d = {}
        if name and name != spec.repo:
            d[spec.id] = name
        else:
            d.pop(spec.id, None)
        self.served_names = json.dumps(d)

    def engine_for(self, spec: ModelSpec) -> str:
        """'local' or 'openai'. Custom-code models always go through a server."""
        return "openai" if (self.backend == "openai" or not spec.runs_locally) else "local"

    def make_backend(self, spec: ModelSpec):
        gen = GenerationConfig(max_new_tokens=self.max_tokens)
        if self.engine_for(spec) == "openai":
            return OpenAIBackend(spec, self.server_url, self.served_name(spec), self.api_key, gen)
        return LocalBackend(spec, self.device, gen, precision=self.precision)

    def backend_key(self, spec: ModelSpec) -> tuple:
        """Settings that require a new backend instance when changed."""
        engine = self.engine_for(spec)
        if engine == "openai":
            return (spec.id, engine, self.server_url, self.served_name(spec), self.api_key, self.max_tokens)
        return (spec.id, engine, self.device, self.precision, self.max_tokens)


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self.s = settings
        lay = QVBoxLayout(self)

        # --- engine
        eng = QGroupBox("OCR engine")
        f = QFormLayout(eng)
        self.backend = QComboBox()
        self.backend.addItem("This computer when possible (server for custom-code models)", "local")
        self.backend.addItem("Always use the OpenAI-compatible server", "openai")
        self.backend.setCurrentIndex(0 if settings.backend == "local" else 1)
        f.addRow("Backend", self.backend)

        self.device = QComboBox()
        self.device.addItem("Auto (CUDA → MPS → CPU)", "auto")
        self.device_note = QLabel()
        self.device_note.setWordWrap(True)
        self._fill_devices(settings.device)
        self.precision = QComboBox()
        self.precision.addItem("Auto (fp16 on GPU, fp32 on CPU)", "auto")
        for p_ in ("fp16", "bf16", "fp32"):
            self.precision.addItem(p_, p_)
        self.precision.setCurrentIndex(max(0, self.precision.findData(settings.precision)))
        models_hint = QLabel("<small>Choose, download and add models in <b>Models…</b> (toolbar).</small>")
        f.addRow(models_hint)
        f.addRow("Device", self.device)
        f.addRow("", self.device_note)
        f.addRow("Precision", self.precision)

        self.server_url = QLineEdit(settings.server_url)
        self.api_key = QLineEdit(settings.api_key)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, 64)
        self.concurrency.setValue(settings.concurrency)
        test = QPushButton("Test connection")
        test.clicked.connect(self._test)
        row = QHBoxLayout()
        row.addWidget(self.server_url, 1)
        row.addWidget(test)
        f.addRow("Server URL", row)
        f.addRow("API key", self.api_key)
        f.addRow("Parallel pages", self.concurrency)
        self._server_rows = [self.api_key, self.concurrency]
        self._local_rows = [self.device, self.precision, self.device_note]
        hint = QLabel(
            "<small>Server example: <code>vllm serve xiaomi-research/Xiaomi-OCR-0</code> → "
            "<code>http://localhost:8000/v1</code>. Per-model serve commands and served names are in "
            "<b>Models…</b>.</small>"
        )
        hint.setWordWrap(True)
        f.addRow(hint)
        self.backend.currentIndexChanged.connect(self._sync)
        lay.addWidget(eng)

        # --- processing
        proc = QGroupBox("Processing")
        f2 = QFormLayout(proc)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 400)
        self.dpi.setSuffix(" dpi")
        self.dpi.setValue(settings.dpi)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(256, 32768)
        self.max_tokens.setSingleStep(512)
        self.max_tokens.setValue(settings.max_tokens)
        self.auto_run = QCheckBox("Start OCR automatically after opening a file")
        self.auto_run.setChecked(settings.auto_run)
        from ..core.layout import LAYOUT_NAME, available as layout_available

        self.layout_detect = QCheckBox(f"Find and crop figures with {LAYOUT_NAME} (~170 MB, downloaded once)")
        self.layout_detect.setChecked(settings.layout_detect and layout_available())
        self.layout_detect.setEnabled(layout_available())
        self.layout_detect.setToolTip(
            "Used on full pages when the OCR model reports no figure positions.\n"
            "Models with layout output (dots.ocr, Granite-Docling) are cropped from their own boxes."
            + ("" if layout_available() else "\nNeeds the local engine (torch + transformers).")
        )
        f2.addRow("PDF render resolution", self.dpi)
        f2.addRow("Max tokens per page", self.max_tokens)
        f2.addRow(self.auto_run)
        f2.addRow(self.layout_detect)
        self.straighten = QCheckBox("Straighten photographed figures (page curl, perspective) using the plot frame")
        self.straighten.setChecked(settings.straighten_figures)
        self.straighten.setToolTip("Applies to cropped figures that have a four-sided frame or axes box.\n"
                                   "Use Crop image → Straighten… to adjust any figure by hand.")
        f2.addRow(self.straighten)
        lay.addWidget(proc)

        # --- snip
        snip = QGroupBox("Snip && export")
        f3 = QFormLayout(snip)
        self.snip_task = QComboBox()
        for t in ("formula", "text", "table", "document"):
            self.snip_task.addItem(t.capitalize(), t)
        self.snip_task.setCurrentIndex(max(0, self.snip_task.findData(settings.snip_task)))
        self.formula_copy = QComboBox()
        self.formula_copy.addItem("Inline  $ … $", "inline")
        self.formula_copy.addItem("Display  \\[ … \\]", "display")
        self.formula_copy.addItem("Raw LaTeX", "raw")
        self.formula_copy.setCurrentIndex(max(0, self.formula_copy.findData(settings.formula_copy)))
        self.page_size = QComboBox()
        self.page_size.addItems(["A4", "Letter"])
        self.page_size.setCurrentText(settings.page_size)
        f3.addRow("Default snip mode", self.snip_task)
        f3.addRow("Copy formulas as", self.formula_copy)
        f3.addRow("PDF page size", self.page_size)
        lay.addWidget(snip)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._sync()

    def _fill_devices(self, current: str):
        """List detected devices; ROCm GPUs are shown but disabled until supported."""
        try:
            from ..core.devices import list_devices, torch_build

            build = torch_build()
            devs = list_devices()
            self.device_note.setText(f"<small>PyTorch {build.torch} · {build.flavor} build</small>")
        except Exception as e:  # noqa: BLE001 - torch missing or broken
            devs = []
            self.device_note.setText(f"<small>Local engine unavailable: {e}</small>")
        model = self.device.model()
        for d in devs:
            self.device.addItem(d.label, d.id)
            if not d.supported:
                item = model.item(self.device.count() - 1)
                item.setEnabled(False)
                item.setToolTip(d.note)
        idx = self.device.findData(current)
        if idx < 0 and current not in ("", "auto"):
            self.device.addItem(f"{current} (not detected)", current)
            idx = self.device.count() - 1
        self.device.setCurrentIndex(max(0, idx))

    def _sync(self):
        # Device settings matter whenever a model runs locally; server settings whenever a
        # server-only model is used — so both groups stay editable.
        always_server = self.backend.currentData() == "openai"
        for w in self._local_rows:
            w.setEnabled(not always_server)

    def _test(self):
        try:
            models = OpenAIBackend(None, self.server_url.text(), api_key=self.api_key.text()).list_models()
            QMessageBox.information(self, "Connected", "Models on server:\n" + "\n".join(models or ["(none)"]))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Connection failed", str(e))

    def result_settings(self) -> AppSettings:
        s = AppSettings(**asdict(self.s))
        s.backend = self.backend.currentData()
        s.device = self.device.currentData()
        s.precision = self.precision.currentData()
        s.server_url = self.server_url.text().strip()
        s.api_key = self.api_key.text()
        s.concurrency = self.concurrency.value()
        s.dpi = self.dpi.value()
        s.max_tokens = self.max_tokens.value()
        s.auto_run = self.auto_run.isChecked()
        s.layout_detect = self.layout_detect.isChecked()
        s.straighten_figures = self.straighten.isChecked()
        s.snip_task = self.snip_task.currentData()
        s.formula_copy = self.formula_copy.currentData()
        s.page_size = self.page_size.currentText()
        return s
