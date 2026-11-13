"""HTML rendering with KaTeX (used by the live preview, HTML export and PDF export)."""

from __future__ import annotations

import base64
import html
import re
from functools import lru_cache
from pathlib import Path

from .document import Block, join_pages, render_html_body

ASSETS = Path(__file__).resolve().parent.parent / "assets"
KATEX_DIR = ASSETS / "katex"

CSS = """
:root { --fg:#1d2329; --muted:#5b6672; --line:#d9dee3; --bg:#ffffff; --accent:#2f6fde; --code:#f4f6f8; }
:root.dark { --fg:#e6e9ec; --muted:#9aa5b1; --line:#3a424b; --bg:#1e2227; --accent:#7aa7ff; --code:#272c33; }
html { background: var(--bg); }
body { color: var(--fg); background: var(--bg); margin: 0 auto; max-width: 820px; padding: 28px 36px 60px;
       font: 16px/1.62 "Source Serif 4", "Charter", "Georgia", "Times New Roman", "Noto Serif", "Noto Serif CJK SC", serif; }
h1,h2,h3,h4,h5,h6 { font-family: -apple-system, "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif; line-height:1.25; margin: 1.4em 0 .5em; }
h1 { font-size: 1.7em; } h2 { font-size: 1.35em; } h3 { font-size: 1.15em; }
p { margin: .55em 0; text-align: justify; hyphens: auto; }
.tex-display { display:block; margin: .8em 0; overflow-x: auto; overflow-y: hidden; text-align:center; }
.table-wrap { overflow-x:auto; margin: 1em 0; }
table { border-collapse: collapse; margin: 0 auto; font-size: .92em; }
th, td { border: 1px solid var(--line); padding: 4px 9px; vertical-align: middle; }
th { background: var(--code); font-weight: 600; }
pre { background: var(--code); padding: 10px 12px; border-radius: 6px; overflow-x:auto; }
figure img { max-width: 100%; }
.page-sep { border: 0; border-top: 1px dashed var(--line); margin: 2em 0; }
.page-label { color: var(--muted); font: 12px -apple-system, "Segoe UI", sans-serif; text-align:center; margin-top:-1.6em; }
.katex-error { color: #c0392b; }
.empty { color: var(--muted); font-family: -apple-system, "Segoe UI", sans-serif; text-align:center; margin-top: 30vh; }
@media print {
  body { max-width: none; padding: 0; font-size: 11pt; }
  .page-sep, .page-label { display:none; }
  h1,h2,h3 { break-after: avoid; } table, .tex-display { break-inside: avoid; }
}
"""

RENDER_JS = r"""
(function(){
  function renderAll(root){
    if (typeof katex === 'undefined') return;
    (root||document).querySelectorAll('.tex').forEach(function(el){
      if (el.dataset.done) return;
      var tex = el.textContent, disp = el.classList.contains('tex-display');
      el.setAttribute('data-tex', tex);
      try { katex.render(tex, el, {displayMode: disp, throwOnError: false, strict: 'ignore', trust: false,
                                    macros: {"\\RR":"\\mathbb{R}"}}); }
      catch(e) { el.textContent = tex; el.classList.add('katex-error'); }
      el.dataset.done = '1';
    });
  }
  window.scidocRender = renderAll;
  window.scidocToMathML = function(tex, disp){
    return katex.renderToString(tex, {displayMode: !!disp, output: 'mathml', throwOnError: false});
  };
  renderAll(document);
  document.addEventListener('dblclick', function(ev){
    var el = ev.target.closest('[data-tex]');
    if (el && window.scidocBridge) window.scidocBridge(el.getAttribute('data-tex'));
  });
})();
"""


@lru_cache(maxsize=1)
def _katex_css_inline() -> str:
    """KaTeX CSS with woff2 fonts embedded as data URIs (for portable HTML)."""
    css = (KATEX_DIR / "katex.min.css").read_text(encoding="utf-8")

    def repl(m: re.Match) -> str:
        name = m.group(1)
        f = KATEX_DIR / "fonts" / f"{name}.woff2"
        if not f.exists():
            return ""
        b64 = base64.b64encode(f.read_bytes()).decode()
        return f"url(data:font/woff2;base64,{b64}) format(\"woff2\")"

    css = re.sub(r'url\(fonts/([^)]+?)\.woff2\) format\("woff2"\)', repl, css)
    # drop woff/ttf fallbacks we do not ship
    css = re.sub(r',\s*url\(fonts/[^)]+?\.(?:woff|ttf)\) format\("(?:woff|truetype)"\)', "", css)
    return css


@lru_cache(maxsize=1)
def _katex_js() -> str:
    return (KATEX_DIR / "katex.min.js").read_text(encoding="utf-8")


def html_document(body: str, title: str = "", standalone: bool = False, dark: bool = False) -> str:
    """Wrap an HTML body into a page with KaTeX.

    ``standalone=False`` references KaTeX via relative paths (load with base URL = ASSETS dir).
    ``standalone=True`` inlines KaTeX JS/CSS and fonts — a single portable file.
    """
    if standalone:
        katex_head = f"<style>{_katex_css_inline()}</style>\n<script>{_katex_js()}</script>"
    else:
        katex_head = '<link rel="stylesheet" href="katex/katex.min.css">\n<script src="katex/katex.min.js"></script>'
    cls = ' class="dark"' if dark else ""
    return f"""<!doctype html>
<html{cls}><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title or "SciDoc")}</title>
{katex_head}
<style>{CSS}</style>
</head><body>
{body}
<script>{RENDER_JS}</script>
</body></html>"""


def pages_to_html(pages: list[str], title: str = "", standalone: bool = False, page_markers: bool = False,
                  dark: bool = False) -> str:
    if page_markers:
        chunks = []
        for i, md in enumerate(pages):
            if i:
                chunks.append(f'<hr class="page-sep"><div class="page-label">page {i + 1}</div>')
            chunks.append(render_html_body(join_pages([md])))
        body = "\n".join(chunks)
    else:
        body = render_html_body(join_pages(pages))
    return html_document(body, title=title, standalone=standalone, dark=dark)


def blocks_to_html(blocks: list[Block], title: str = "", standalone: bool = False) -> str:
    return html_document(render_html_body(blocks), title=title, standalone=standalone)
