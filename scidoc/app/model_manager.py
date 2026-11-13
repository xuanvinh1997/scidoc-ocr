"""Model Manager: browse the catalog, download/delete weights, pick models, add custom models."""

from __future__ import annotations

import html
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core.models import (
    BUILTIN_MODELS,
    GENERIC_PROMPTS,
    RUNTIME_SERVER,
    RUNTIME_TRANSFORMERS,
    TASKS,
    ModelSpec,
    Registry,
    delete_local,
    local_status,
    remote_size,
)
from ..core.models.spec import OUTPUT_FORMATS
from .downloads import DownloadManager
from .settings import AppSettings

COLS = ["", "Model", "Params", "Runs on", "Status", "License"]


def fmt_bytes(n: int) -> str:
    if not n:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit in ("GB", "TB") else f"{n:.0f} {unit}"
        n /= 1024
    return str(n)


class _SizeFetcher(QObject):
    """Fetches download sizes from the Hub in the background."""

    fetched = Signal(str, object)

    def fetch(self, specs: list[ModelSpec]):
        def run():
            for s in specs:
                if not s.is_local_path:
                    self.fetched.emit(s.id, remote_size(s))

        threading.Thread(target=run, daemon=True).start()


class ModelManagerDialog(QDialog):
    modelsChanged = Signal()  # active models / registry changed

    def __init__(self, registry: Registry, settings: AppSettings, downloads: DownloadManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Models")
        self.resize(1180, 680)
        self.registry, self.settings, self.downloads = registry, settings, downloads
        self._status: dict[str, object] = {}
        self._remote: dict[str, int] = {}

        # ---- table
        self.search = QLineEdit(placeholderText="Search models, languages, strengths…")
        self.only_local = QCheckBox("Runs on this computer")
        self.only_downloaded = QCheckBox("Downloaded")
        for w in (self.search,):
            w.textChanged.connect(self._apply_filter)
        for w in (self.only_local, self.only_downloaded):
            w.toggled.connect(self._apply_filter)
        top = QHBoxLayout()
        top.addWidget(self.search, 1)
        top.addWidget(self.only_local)
        top.addWidget(self.only_downloaded)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setMinimumSectionSize(40)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # model name gets the spare width
        for c in (0, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        for c, width in ((4, 190), (5, 120)):  # long statuses / licenses elide instead of squeezing the name
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(c, width)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setWordWrap(False)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.doubleClicked.connect(lambda _: self._use("doc"))

        # ---- details
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(True)
        self.served = QLineEdit(placeholderText="name on the server (default: Hugging Face id)")
        self.served.editingFinished.connect(self._save_served)
        served_row = QHBoxLayout()
        served_row.addWidget(QLabel("Server model name"))
        served_row.addWidget(self.served, 1)

        self.b_download = QPushButton("Download")
        self.b_delete = QPushButton("Delete files")
        self.b_use_doc = QPushButton("Use for documents")
        self.b_use_snip = QPushButton("Use for snips")
        self.b_hf = QPushButton("Open on Hugging Face")
        self.b_edit = QPushButton("Edit…")
        self.b_reset = QPushButton("Remove / reset")
        self.b_download.clicked.connect(self._download)
        self.b_delete.clicked.connect(self._delete)
        self.b_use_doc.clicked.connect(lambda: self._use("doc"))
        self.b_use_snip.clicked.connect(lambda: self._use("snip"))
        self.b_hf.clicked.connect(self._open_hf)
        self.b_edit.clicked.connect(self._edit)
        self.b_reset.clicked.connect(self._remove)
        btns1 = QHBoxLayout()
        for b in (self.b_use_doc, self.b_use_snip):
            btns1.addWidget(b)
        btns2 = QHBoxLayout()
        for b in (self.b_download, self.b_delete, self.b_hf):
            btns2.addWidget(b)
        btns3 = QHBoxLayout()
        for b in (self.b_edit, self.b_reset):
            btns3.addWidget(b)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.details, 1)
        rl.addLayout(served_row)
        rl.addLayout(btns1)
        rl.addLayout(btns2)
        rl.addLayout(btns3)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addLayout(top)
        ll.addWidget(self.table, 1)
        add = QPushButton("Add custom model…")
        add.clicked.connect(self._add)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        bl = QHBoxLayout()
        bl.addWidget(add)
        bl.addStretch(1)
        bl.addWidget(refresh)
        ll.addLayout(bl)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([700, 480])
        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        lay.addWidget(close)

        downloads.progress.connect(self._on_progress)
        downloads.finished.connect(self._on_download_finished)
        self._sizes = _SizeFetcher()
        self._sizes.fetched.connect(self._on_size)
        self.refresh()
        self._sizes.fetch([s for s in self.registry.all() if s.runs_locally])

    # ------------------------------------------------------------------ data
    def refresh(self):
        self.registry.reload()
        cur = self._current_id()
        self.specs = sorted(
            self.registry.all(),
            key=lambda s: (s.id != self.settings.doc_model, not s.runs_locally, not s.builtin, s.params_b),
        )
        self._status = {s.id: local_status(s) if s.runs_locally else None for s in self.specs}
        self.table.setRowCount(len(self.specs))
        for r, s in enumerate(self.specs):
            self._fill_row(r, s)
        self._apply_filter()
        row = next((i for i, s in enumerate(self.specs) if s.id == cur), 0)
        self.table.selectRow(row)
        self._on_select()

    def _fill_row(self, r: int, s: ModelSpec):
        badges = []
        if s.id == self.settings.doc_model:
            badges.append("●doc")
        if s.id == (self.settings.snip_model or self.settings.doc_model):
            badges.append("●snip")
        name = s.name + ("" if s.builtin else "  (custom)") + ("  ✓ verified" if s.verified else "")
        cells = [
            " ".join(badges),
            name,
            f"{s.params_b:.2f}B" if s.params_b else "?",
            "this computer" if s.runs_locally else "server only",
            self._status_text(s),
            s.license or "?",
        ]
        for c, text in enumerate(cells):
            it = QTableWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, s.id)
            if c == 5:
                it.setToolTip(s.license)
            if c == 1:
                f = QFont()
                f.setBold(s.id == self.settings.doc_model)
                it.setFont(f)
                it.setToolTip(s.description)
            self.table.setItem(r, c, it)

    def _status_text(self, s: ModelSpec) -> str:
        if self.downloads.is_active(s.id):
            done, total = self.downloads.last_progress(s.id)
            pct = f"{100 * done / total:.0f}%" if total else fmt_bytes(done)
            return f"Downloading {pct}"
        st = self._status.get(s.id)
        if st is None:
            return "via server"
        if st.downloaded:
            return f"Downloaded ({fmt_bytes(st.size_bytes)})"
        size = self._remote.get(s.id)
        return f"Not downloaded ({fmt_bytes(size)})" if size else "Not downloaded"

    def _apply_filter(self):
        q = self.search.text().strip().lower()
        for r, s in enumerate(self.specs):
            hay = " ".join([s.name, s.repo, s.description, " ".join(s.languages), " ".join(s.strengths),
                            s.family]).lower()
            st = self._status.get(s.id)
            hide = (q and q not in hay) or (self.only_local.isChecked() and not s.runs_locally) or (
                self.only_downloaded.isChecked() and not (st and st.downloaded))
            self.table.setRowHidden(r, bool(hide))

    def _current_id(self) -> str | None:
        items = self.table.selectedItems() if hasattr(self, "table") else []
        return items[0].data(Qt.ItemDataRole.UserRole) if items else None

    def _spec(self) -> ModelSpec | None:
        mid = self._current_id()
        return next((s for s in self.specs if s.id == mid), None) if mid else None

    def _row_of(self, model_id: str) -> int:
        return next((i for i, s in enumerate(self.specs) if s.id == model_id), -1)

    # ------------------------------------------------------------------ details
    def _on_select(self):
        s = self._spec()
        if not s:
            self.details.clear()
            return
        st = self._status.get(s.id)
        downloading = self.downloads.is_active(s.id)
        self.b_download.setText("Cancel download" if downloading else "Download")
        self.b_download.setEnabled(s.runs_locally and not s.is_local_path and (downloading or not (st and st.downloaded)))
        self.b_delete.setEnabled(bool(st and st.downloaded and not s.is_local_path and not downloading))
        self.b_hf.setEnabled(not s.is_local_path)
        self.b_reset.setEnabled(self.registry.is_user(s.id))
        overrides_builtin = any(b.id == s.id for b in BUILTIN_MODELS)
        self.b_reset.setText("Reset to default" if overrides_builtin else "Remove custom")
        self.served.setText(self.settings.served_name(s) if self.settings.served_name(s) != s.repo else "")
        self.details.setHtml(self._details_html(s))

    def _details_html(self, s: ModelSpec) -> str:
        e = html.escape
        tasks = []
        for t in TASKS:
            if s.supports(t):
                tasks.append(f"<b>{t}</b>")
            else:
                fb, _ = s.prompt_for(t)
                tasks.append(f"<span style='color:gray'>{t} → uses {fb} prompt</span>")
        link = s.repo if s.is_local_path else f"<a href='https://huggingface.co/{e(s.repo)}'>{e(s.repo)}</a>"
        vram = s.est_vram_gb()
        rows = [
            ("Repository", link),
            ("Runs on", "this computer (transformers)" if s.runs_locally else
             "OpenAI-compatible server only (custom code)"),
            ("Parameters", f"{s.params_b:.2f} B" + (f" · ~{vram} GB GPU memory" if vram else "")),
            ("Precision", s.dtype),
            ("License", e(s.license or "?")),
            ("Languages", e(", ".join(s.languages) or "?")),
            ("Strengths", e(", ".join(s.strengths) or "—")),
            ("Tasks", " · ".join(tasks)),
            ("Output", e(s.output) + (f" ({', '.join(f'{k}: {v}' for k, v in s.output_by_task.items())})"
                                      if s.output_by_task else "")),
        ]
        body = "".join(f"<tr><td style='color:gray;padding-right:10px;vertical-align:top'>{k}</td>"
                       f"<td>{v}</td></tr>" for k, v in rows)
        serve = f"<p><b>Serve:</b><br><code>{e(s.serve_hint)}</code></p>" if s.serve_hint else ""
        notes = f"<p style='color:#b9770e'>⚠ {e(s.notes)}</p>" if s.notes else ""
        verified = ("<p style='color:#1e8449'>✓ Tested end-to-end in SciDoc.</p>" if s.verified else
                    "<p style='color:gray'>Settings from the model card; not yet tested end-to-end in SciDoc.</p>")
        return (f"<h3>{e(s.name)}</h3><p>{e(s.description)}</p><table>{body}</table>"
                f"{serve}{notes}{verified}")

    # ------------------------------------------------------------------ actions
    def _download(self):
        s = self._spec()
        if not s:
            return
        if self.downloads.is_active(s.id):
            self.downloads.cancel(s.id)
            return
        self.downloads.start(s)
        self._update_row(s.id)
        self._on_select()

    def _on_progress(self, model_id, done, total):
        r = self._row_of(model_id)
        if r >= 0:
            self.table.item(r, 4).setText(self._status_text(self.specs[r]))

    def _on_download_finished(self, model_id, ok, msg):
        s = next((x for x in self.specs if x.id == model_id), None)
        if s:
            self._status[s.id] = local_status(s)
            self._update_row(model_id)
            if model_id == self._current_id():
                self._on_select()
        if not ok and msg != "cancelled":
            QMessageBox.warning(self, "Download failed", f"{model_id}: {msg}")

    def _on_size(self, model_id, size):
        self._remote[model_id] = int(size or 0)
        self._update_row(model_id)

    def _update_row(self, model_id: str):
        r = self._row_of(model_id)
        if r >= 0:
            self._fill_row(r, self.specs[r])

    def _delete(self):
        s = self._spec()
        if not s:
            return
        st = self._status.get(s.id)
        if QMessageBox.question(self, "Delete model files",
                                f"Delete {s.name} ({fmt_bytes(st.size_bytes if st else 0)}) from the "
                                f"Hugging Face cache?") != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_local(s)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Delete failed", str(e))
        self._status[s.id] = local_status(s)
        self._update_row(s.id)
        self._on_select()
        self.modelsChanged.emit()

    def _use(self, which: str):
        s = self._spec()
        if not s:
            return
        if which == "doc":
            self.settings.doc_model = s.id
        else:
            self.settings.snip_model = "" if s.id == self.settings.doc_model else s.id
        self.settings.save()
        for r, spec in enumerate(self.specs):
            self._fill_row(r, spec)
        self.modelsChanged.emit()

    def _save_served(self):
        s = self._spec()
        if s:
            self.settings.set_served_name(s, self.served.text().strip())
            self.settings.save()
            self.modelsChanged.emit()

    def _open_hf(self):
        s = self._spec()
        if s and not s.is_local_path:
            QDesktopServices.openUrl(QUrl(f"https://huggingface.co/{s.repo}"))

    def _add(self):
        dlg = ModelEditDialog(None, self.registry, self)
        if dlg.exec():
            spec = dlg.result_spec()
            self.registry.add_or_update(spec)
            self.refresh()
            self.table.selectRow(self._row_of(spec.id))
            self.modelsChanged.emit()

    def _edit(self):
        s = self._spec()
        if not s:
            return
        dlg = ModelEditDialog(s, self.registry, self)
        if dlg.exec():
            self.registry.add_or_update(dlg.result_spec())
            self.refresh()
            self.modelsChanged.emit()

    def _remove(self):
        s = self._spec()
        if s and self.registry.is_user(s.id):
            self.registry.remove(s.id)
            if self.settings.doc_model == s.id and not any(b.id == s.id for b in self.registry.all()):
                self.settings.doc_model = ""
            self.settings.save()
            self.refresh()
            self.modelsChanged.emit()


# --------------------------------------------------------------------------- edit dialog


class ModelEditDialog(QDialog):
    """Add a custom model or override a built-in one's prompts/settings."""

    def __init__(self, spec: ModelSpec | None, registry: Registry, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit model" if spec else "Add custom model")
        self.resize(720, 640)
        self.orig = spec
        s = spec or ModelSpec(id="", name="", repo="", prompts=dict(GENERIC_PROMPTS), builtin=False)

        self.repo = QLineEdit(s.repo, placeholderText="org/model on Hugging Face, or a local folder")
        browse = QPushButton("Folder…")
        browse.clicked.connect(self._browse)
        check = QPushButton("Check")
        check.clicked.connect(self._check)
        repo_row = QHBoxLayout()
        repo_row.addWidget(self.repo, 1)
        repo_row.addWidget(browse)
        repo_row.addWidget(check)
        self.name = QLineEdit(s.name)
        self.runtime = QComboBox()
        self.runtime.addItem("This computer (transformers, stock architecture)", RUNTIME_TRANSFORMERS)
        self.runtime.addItem("OpenAI-compatible server only", RUNTIME_SERVER)
        self.runtime.setCurrentIndex(max(0, self.runtime.findData(s.runtime)))
        self.trust = QCheckBox("trust_remote_code (runs Python code from the repo)")
        self.trust.setChecked(s.trust_remote_code)
        self.params = QDoubleSpinBox(decimals=2, maximum=1000.0, suffix=" B")
        self.params.setValue(s.params_b)
        self.dtype = QComboBox()
        self.dtype.addItems(["bf16", "fp16", "fp32"])
        self.dtype.setCurrentText(s.dtype)
        self.output = QComboBox()
        self.output.addItems(list(OUTPUT_FORMATS))
        self.output.setCurrentText(s.output)
        self.max_px = QSpinBox(maximum=50_000_000, singleStep=100_000)
        self.max_px.setValue(s.max_pixels)
        self.system = QLineEdit(s.system_prompt)
        self.thinking_off = QCheckBox("Pass enable_thinking=False to the chat template")
        self.thinking_off.setChecked(s.chat_template_kwargs.get("enable_thinking") is False)
        self.license = QLineEdit(s.license)
        self.info = QLabel()
        self.info.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Model", repo_row)
        form.addRow("", self.info)
        form.addRow("Display name", self.name)
        form.addRow("Runs on", self.runtime)
        form.addRow("", self.trust)
        form.addRow("Parameters", self.params)
        form.addRow("Precision", self.dtype)
        form.addRow("Output format", self.output)
        form.addRow("Max image pixels", self.max_px)
        form.addRow("System prompt", self.system)
        form.addRow("", self.thinking_off)
        form.addRow("License", self.license)

        self.prompt_edits: dict[str, tuple[QCheckBox, QPlainTextEdit]] = {}
        tabs = QTabWidget()
        for t in TASKS:
            w = QWidget()
            v = QVBoxLayout(w)
            enabled = QCheckBox(f"Model supports the '{t}' task")
            enabled.setChecked(t in s.prompts)
            edit = QPlainTextEdit(s.prompts.get(t, GENERIC_PROMPTS.get(t, "")))
            edit.setPlaceholderText("Empty = send the image only, without a text prompt")
            v.addWidget(enabled)
            v.addWidget(edit)
            tabs.addTab(w, t.capitalize())
            self.prompt_edits[t] = (enabled, edit)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("Prompts"))
        lay.addWidget(tabs, 1)
        lay.addWidget(bb)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Model folder")
        if d:
            self.repo.setText(d)
            if not self.name.text():
                self.name.setText(Path(d).name)

    def _check(self):
        """Inspect the repo config: architecture, transformers support, parameter count."""
        repo = self.repo.text().strip()
        if not repo:
            return
        try:
            info = inspect_model(repo)
        except Exception as e:  # noqa: BLE001
            self.info.setText(f"<span style='color:#c0392b'>Could not read {html.escape(repo)}: "
                              f"{html.escape(str(e))}</span>")
            return
        if not self.name.text():
            self.name.setText(repo.rstrip("/").split("/")[-1])
        if info.get("params_b"):
            self.params.setValue(info["params_b"])
        if info.get("license") and not self.license.text():
            self.license.setText(info["license"])
        self.runtime.setCurrentIndex(self.runtime.findData(
            RUNTIME_TRANSFORMERS if info["native"] else RUNTIME_SERVER))
        self.trust.setChecked(info["custom_code"])
        color = "#1e8449" if info["native"] else "#b9770e"
        msg = ("supported natively — can run on this computer" if info["native"] else
               "not a stock transformers architecture — use a server (vLLM/SGLang)")
        self.info.setText(f"<span style='color:{color}'>model_type <b>{html.escape(str(info['model_type']))}</b>: "
                          f"{msg}</span>")

    def _accept(self):
        if not self.repo.text().strip():
            QMessageBox.warning(self, "Missing model", "Enter a Hugging Face id or folder.")
            return
        if not any(en.isChecked() for en, _ in self.prompt_edits.values()):
            QMessageBox.warning(self, "No tasks", "Enable at least one task prompt.")
            return
        self.accept()

    def result_spec(self) -> ModelSpec:
        from ..core.models import custom_spec

        repo = self.repo.text().strip()
        base = self.orig or custom_spec(repo)
        d = base.to_dict()
        d.update(
            repo=repo,
            name=self.name.text().strip() or repo.split("/")[-1],
            runtime=self.runtime.currentData(),
            trust_remote_code=self.trust.isChecked(),
            params_b=self.params.value(),
            dtype=self.dtype.currentText(),
            output=self.output.currentText(),
            max_pixels=self.max_px.value(),
            system_prompt=self.system.text(),
            license=self.license.text().strip(),
            prompts={t: e.toPlainText() for t, (en, e) in self.prompt_edits.items() if en.isChecked()},
            builtin=False,
            verified=False,
        )
        ctk = dict(base.chat_template_kwargs)
        if self.thinking_off.isChecked():
            ctk["enable_thinking"] = False
        else:
            ctk.pop("enable_thinking", None)
        d["chat_template_kwargs"] = ctk
        return ModelSpec.from_dict(d)


def inspect_model(repo: str) -> dict:
    """Read config.json (Hub or folder) and report whether stock transformers can load it."""
    import json

    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    p = Path(repo).expanduser()
    params_b, license_ = 0.0, ""
    if p.exists():
        cfg = json.loads((p / "config.json").read_text())
    else:
        from huggingface_hub import HfApi, hf_hub_download

        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        try:
            info = HfApi().model_info(repo, expand=["safetensors", "cardData"])
            if info.safetensors:
                params_b = info.safetensors.total / 1e9
            card = info.card_data.to_dict() if info.card_data else {}
            license_ = str(card.get("license") or "")
        except Exception:  # noqa: BLE001
            pass
    mt = cfg.get("model_type")
    return {
        "model_type": mt,
        "native": mt in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        "custom_code": bool(cfg.get("auto_map")),
        "params_b": round(params_b, 2),
        "license": license_,
    }
