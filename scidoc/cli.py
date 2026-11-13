"""Command line.

    scidoc convert paper.pdf -o paper.pdf -o paper.tex [--model lighton-ocr-3-4b]
    scidoc models list | info <id> | download <id> | rm <id>
    scidoc devices
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .core.backends import GenerationConfig, LocalBackend, OpenAIBackend, Task, model_input_size
from .core.cache import PageCache
from .core.export import export_file
from .core.figures import FigureStore, materialize
from .core.layout import LayoutDetector, available as layout_available
from .core.models import DEFAULT_MODEL_ID, Registry, delete_local, download, local_status, to_canonical
from .core.sources import open_source

EXT_TO_FMT = {".md": "md", ".tex": "tex", ".html": "html", ".htm": "html", ".txt": "txt", ".pdf": "pdf"}


def _parse_range(s: str | None, n: int) -> range:
    if not s:
        return range(n)
    a, dash, b = s.partition("-")
    lo = int(a) if a else 1
    hi = int(b) if b else (n if dash else lo)
    return range(max(1, lo) - 1, min(n, hi))


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n else "—"


def cmd_convert(args) -> int:
    spec = Registry().get(args.model)
    gen = GenerationConfig(max_new_tokens=args.max_tokens)
    if args.backend == "openai" or not spec.runs_locally:
        backend = OpenAIBackend(spec, args.url, args.served_name or "", args.api_key, gen)
    else:
        backend = LocalBackend(spec, args.device, gen, precision=args.precision)
        backend.load(lambda m: print(m, file=sys.stderr))
    print(f"model: {spec.name} ({spec.repo}) via {backend.describe()}", file=sys.stderr)

    src = open_source(args.input, dpi=args.dpi)
    task = Task(args.task)
    sub = task.value if spec.id == DEFAULT_MODEL_ID else f"{spec.id}/{task.value}"
    cache = PageCache(src.digest if not args.no_cache else "", sub)
    figures = FigureStore(src.digest)
    detector = None
    if getattr(args, "figures", True) and layout_available() and task == Task.DOCUMENT:
        detector = LayoutDetector(getattr(args, "device", "auto") or "auto")
    pages = []
    for i in _parse_range(args.pages, len(src.pages)):
        hit = cache.get(i)
        if hit:
            md = hit[0]
            print(f"[{i + 1}/{len(src.pages)}] cached", file=sys.stderr)
        else:
            t = time.time()
            raw = backend.recognize(src.pages[i].image, task)
            image = src.pages[i].image
            pictures, text_boxes = detector.analyze(image) if detector else (None, None)
            md = materialize(to_canonical(raw, spec, task.value, model_input_size(backend, image)),
                             image, i, figures, pictures, text_boxes)
            cache.put(i, md, raw)
            print(f"[{i + 1}/{len(src.pages)}] {time.time() - t:.1f}s", file=sys.stderr)
        pages.append(md)

    outputs = args.output or [str(Path(args.input).with_suffix(".md"))]
    for out in outputs:
        out_p = Path(out)
        fmt = args.format or EXT_TO_FMT.get(out_p.suffix.lower())
        if not fmt:
            print(f"cannot infer format from {out}; use --format", file=sys.stderr)
            return 2
        export_file(pages, fmt, out_p, title=src.name if args.title else "", figures=figures)
        print(f"wrote {out_p}", file=sys.stderr)
    return 0


def cmd_models(args) -> int:
    reg = Registry()
    if args.action == "list":
        print(f"{'id':24} {'params':>7}  {'runs on':12} {'downloaded':>11}  name")
        for s in reg.all():
            st = local_status(s) if s.runs_locally else None
            dl = ("yes " + _gb(st.size_bytes)) if st and st.downloaded else ("no" if st else "server")
            tag = " *" if s.id == DEFAULT_MODEL_ID else ""
            print(f"{s.id:24} {s.params_b:6.2f}B  {'local' if s.runs_locally else 'server-only':12} "
                  f"{dl:>11}  {s.name}{tag}{'' if s.builtin else ' (custom)'}")
        return 0
    spec = reg.get(args.model)
    if args.action == "info":
        for k, v in spec.to_dict().items():
            if k == "prompts":
                for t, p in v.items():
                    print(f"prompt[{t}]: {p!r}")
            elif v not in ("", [], {}, None):
                print(f"{k}: {v}")
        return 0
    if args.action == "download":
        if not spec.runs_locally:
            print(f"{spec.name} runs on a server; nothing to download. Serve with:\n  {spec.serve_hint}")
            return 0
        last = [0.0]

        def progress(done, total):
            if time.time() - last[0] > 0.5 or done == total:
                last[0] = time.time()
                pct = f"{100 * done / total:5.1f}%" if total else ""
                print(f"\r{spec.name}: {pct} {_gb(done)} / {_gb(total)}", end="", file=sys.stderr)

        path = download(spec, progress)
        print(f"\n{path}", file=sys.stderr)
        return 0
    if args.action == "rm":
        print(f"freed {_gb(delete_local(spec))}")
        return 0
    return 2


def cmd_devices(_args) -> int:
    from .core.devices import DeviceError, report

    try:
        print(report())
    except DeviceError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scidoc", description="OCR scientific documents with open VLMs")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("convert", help="convert a PDF/image to md/tex/html/txt/pdf")
    c.add_argument("input")
    c.add_argument("-o", "--output", action="append", help="output file(s); format from extension")
    c.add_argument("--format", choices=["md", "md-dollar", "tex", "html", "txt", "pdf"])
    c.add_argument("--task", default="document", choices=[t.value for t in Task])
    c.add_argument("--pages", help="page range, e.g. 1-5 or 3")
    c.add_argument("--dpi", type=int, default=150)
    c.add_argument("-m", "--model", default=DEFAULT_MODEL_ID, help="model id (see `scidoc models list`) or HF repo")
    c.add_argument("--backend", default="local", choices=["local", "openai"])
    c.add_argument("--device", default="auto", help="auto | cuda | cuda:N | mps | cpu  (see `scidoc devices`)")
    c.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    c.add_argument("--url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL")
    c.add_argument("--served-name", help="model name on the server (default: HF repo id)")
    c.add_argument("--api-key", default="")
    c.add_argument("--max-tokens", type=int, default=4096)
    c.add_argument("--title", action="store_true", help="use file name as document title")
    c.add_argument("--no-cache", action="store_true")
    c.add_argument("--no-figures", dest="figures", action="store_false",
                   help="do not run the layout model to find and crop figures")
    c.set_defaults(func=cmd_convert)

    m = sub.add_parser("models", help="list / inspect / download / delete OCR models")
    m.add_argument("action", choices=["list", "info", "download", "rm"])
    m.add_argument("model", nargs="?", default=DEFAULT_MODEL_ID)
    m.set_defaults(func=cmd_models)

    d = sub.add_parser("devices", help="show detected GPUs / torch build (CUDA, MPS, ROCm)")
    d.set_defaults(func=cmd_devices)

    sub.add_parser("gui", help="launch the desktop app").set_defaults(func=lambda a: _gui())

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        return _gui()
    return args.func(args)


def _gui() -> int:
    from .app.main import main as gui_main

    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
