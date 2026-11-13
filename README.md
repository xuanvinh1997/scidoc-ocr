# SciDoc OCR

SciDoc OCR is an open-source desktop alternative to Mathpix for Windows, macOS and Linux. It turns scanned or born-digital scientific documents (papers, textbooks, handwritten derivations, tables) into:

- **Markdown** with LaTeX math, using either `\( \)` or `$ $` delimiters
- a **LaTeX** document that you can compile
- **HTML**: one self-contained file with KaTeX and its fonts embedded, so it works offline
- a **readable PDF**, re-typeset with real math rendering
- **plain text**

Recognition uses [**Xiaomi-OCR-0**](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0), a 0.9B Apache-2.0 vision-language OCR model. It can run on your own machine or on any OpenAI-compatible server.

## Features

| | |
|---|---|
| **Documents** | Open PDFs or images (PNG/JPG/TIFF…) by dialog or drag-and-drop. Pages are recognised in reading order, with formulas as LaTeX and tables (including merged cells) parsed from the model's OTSL output. |
| **Snip** (`Ctrl+Alt+M`) | Drag a rectangle anywhere on screen. The result is OCR'd and **copied to the clipboard automatically**, like Mathpix Snip. |
| **Paste** (`Ctrl/⌘+V`) | OCR an image from the clipboard. |
| **Figures** | Pictures are cropped from the page and linked as `![…](figures/…)`. Positions come from the OCR model when it reports layout (dots.ocr, Granite-Docling), otherwise from the [Docling Heron](https://huggingface.co/docling-project/docling-layout-heron) layout model (Apache-2.0, ~170 MB, local engine only; toggle in Settings, `--no-figures` in the CLI). **Crop image** (`Ctrl/⌘+Shift+C`) lets you select any region by hand; pasting or dropping an image into the Markdown editor inserts it too. Markdown/LaTeX exports get a `figures/` folder; HTML/PDF embed the images. |
| **Modes** | Document, Formula, Text, Table: the four task prompts published for the model. |
| **Live editor** | Side-by-side view: original page, KaTeX preview, and editable Markdown. Edits re-render instantly and are cached. |
| **Copy** | A page or the whole document as Markdown or LaTeX. A formula as `$…$`, `\[…\]` or raw LaTeX, or as **MathML** (paste into Word for a native equation). Double-click any equation in the preview to copy its LaTeX. |
| **Streaming** | Tokens appear as the model generates them. Results are cached on disk, so reopening a file never runs OCR twice. |
| **CLI** | `scidoc convert paper.pdf -o paper.pdf -o paper.tex` for batch work. |

## Install

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). Choose **one** engine profile. Each profile installs the matching PyTorch build, and the versions are pinned in `uv.lock`:

| Platform | GPU | Command |
|---|---|---|
| macOS (Apple Silicon) | Metal / **MPS** | `uv sync --extra mps` |
| Linux / Windows | NVIDIA **CUDA** 13.0 | `uv sync --extra cuda` |
| any | none (CPU, very slow) | `uv sync --extra cpu` |
| any | remote server only | `uv sync` |

```bash
uv sync --extra mps          # or cuda / cpu
uv run scidoc devices        # check which GPU will be used
uv run scidoc-app            # launch the desktop app
```

The model (about 1.8 GB) downloads from Hugging Face on first use.

## Compute devices (local engine)

| Device | Status | Precision | Notes |
|---|---|---|---|
| **CUDA** (NVIDIA) | supported | fp16 (default), bf16 (Ampere or newer), fp32 | Multi-GPU: pick `cuda:N` in Settings or `--device cuda:1`. TF32 is enabled. |
| **MPS** (Apple Silicon) | supported | fp16 | `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically. |
| CPU | fallback | fp32 | Works, but takes many minutes per page. Use a GPU or a server. |
| **ROCm** (AMD) | planned | – | AMD GPUs are detected and listed, but not used yet (see below). |

- `auto` picks the first CUDA GPU, then MPS, then CPU.
- If the GPU runs out of memory, SciDoc frees the cache and retries the page at half the resolution, up to twice.
- `scidoc devices` prints the torch build, detected GPUs, the device and precision `auto` will choose, and missing optional kernels.

**Faster CUDA decoding (Linux, optional).** Xiaomi-OCR-0 is based on Qwen3.5, which uses linear-attention layers. On CUDA, fused kernels speed these up:

```bash
uv pip install flash-linear-attention causal-conv1d --no-build-isolation
```

**ROCm (experimental).** ROCm builds of PyTorch are recognised. SciDoc only uses them when you opt in:

```bash
uv sync                                    # app without torch
uv pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/rocm7.2
uv pip install transformers==5.17.0 accelerate
SCIDOC_EXPERIMENTAL_ROCM=1 uv run scidoc-app
```

## OCR engines (Settings → OCR engine)

**Local (default).** Runs on the device chosen above. On an M5 Max with MPS, a dense math page takes about 30 s, and a table or formula snip about 5 s.

**OpenAI-compatible server.** This is the best option for large batches, for example vLLM on an NVIDIA box. The app sends pages in parallel, and the number sent at once is set by "Parallel pages":

```bash
vllm serve xiaomi-research/Xiaomi-OCR-0               # → http://localhost:8000/v1
# or SGLang:  python -m sglang.launch_server --model-path xiaomi-research/Xiaomi-OCR-0   # → :30000/v1
# or Ollama (community build): ollama pull longwayxu/xiaomi-ocr-0:bf16   # → :11434/v1
```

## CLI

```bash
scidoc convert paper.pdf -o paper.pdf -o paper.md -o paper.tex     # format from extension
scidoc convert scan.png --task formula -o eq.tex
scidoc convert book.pdf --pages 10-25 --backend openai --url http://gpu-box:8000/v1 -o ch2.html
```

## How it works

```
PDF/image ─▶ page images (pypdfium2, 150 dpi)
          ─▶ Xiaomi-OCR-0 (task prompt from the model card, greedy, non-thinking)
          ─▶ clean_model_output: strip <think>, OTSL → GFM/HTML tables, "(5)" → \tag{5}
          ─▶ canonical Markdown  (cached, user-editable)
          ─▶ block model ─▶ Markdown │ LaTeX │ HTML+KaTeX │ text
                                          └▶ Qt WebEngine printToPdf ─▶ readable PDF
```

```
scidoc/
  core/       backends.py  (local transformers / OpenAI-compatible, streaming)
              tables.py    (OTSL / pipe / HTML tables → GFM / HTML / LaTeX multirow+multicolumn)
              document.py  (cleanup, block parser, Markdown / LaTeX / text renderers)
              render.py    (HTML + KaTeX)   pdf_export.py (Chromium print-to-PDF)
              devices.py   (CUDA / MPS / CPU detection, dtype policy, ROCm gate)
              sources.py   (PDF/image loading)   cache.py   export.py
  app/        PySide6 GUI: main_window.py, worker.py (QThread), snip.py, settings.py
  assets/     vendored KaTeX 0.19.1 (MIT)
  cli.py
```

## Packaging

```bash
uv sync --extra mps --extra dev      # or --extra cuda
uv run python scripts/build_app.py       # dist/SciDoc OCR.app / .exe / Linux folder
```

Run this on each target OS. If you install without `[local]`, the bundle is much smaller and uses a server backend only.

## Notes

- **macOS Snip:** grant *Screen Recording* permission to your terminal or the app. Without it, the capture shows only the desktop wallpaper.
- **LaTeX export:** the file compiles with pdfLaTeX, XeLaTeX or LuaLaTeX. Use XeLaTeX or LuaLaTeX for CJK text.
- **Tests:** `pytest`
