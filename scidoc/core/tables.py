"""Table model shared by every exporter.

Xiaomi-OCR-0 emits tables in OTSL (Optimized Table Structure Language):

    <fcel>text   filled cell (content follows until the next tag)
    <ecel>       empty cell
    <lcel>       slot covered by the cell to its left   (colspan += 1)
    <ucel>       slot covered by the cell above         (rowspan += 1)
    <xcel>       slot covered by both (2-D span)
    <nl>         end of row

We parse OTSL (and also GFM pipe tables / simple HTML tables, so that user-edited
Markdown round-trips) into a :class:`Table`, then render it as GFM, HTML or LaTeX.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

OTSL_TAG_RE = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl)\s*/?>", re.I)
OTSL_CELL_RE = re.compile(r"<(?:fcel|ecel|lcel|ucel|xcel)>", re.I)


@dataclass
class Cell:
    text: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1


@dataclass
class Table:
    cells: list[Cell] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_otsl(cls, src: str) -> "Table | None":
        """Parse an OTSL block. Tolerates newlines used in place of ``<nl>``."""
        src = src.replace("\r\n", "\n")
        # A real newline that is not already preceded by <nl> acts as a row break.
        src = re.sub(r"<nl\s*/?>\s*\n?", "\n", src, flags=re.I)
        grid: dict[tuple[int, int], Cell] = {}
        cells: list[Cell] = []
        row = col = 0
        pending: str | None = None  # text of the filled cell currently being read

        def place(text: str) -> None:
            nonlocal col
            cell = Cell(text=text.strip(), row=row, col=col)
            cells.append(cell)
            grid[(row, col)] = cell
            col += 1

        def flush() -> None:
            nonlocal pending
            if pending is not None:
                place(pending)
                pending = None

        def cover(dr: int, dc: int) -> None:
            """Mark the slot (row, col) as covered by the master cell up/left."""
            nonlocal col
            master = None
            if dc:  # look left
                master = grid.get((row, col - 1)) if col > 0 else None
            if master is None and dr:  # look up
                master = grid.get((row - 1, col)) if row > 0 else None
            if master is not None:
                if master.col + master.colspan - 1 < col:
                    master.colspan = col - master.col + 1
                if master.row + master.rowspan - 1 < row:
                    master.rowspan = row - master.row + 1
                grid[(row, col)] = master
            col += 1

        pos = 0
        for m in re.finditer(r"<(fcel|ecel|lcel|ucel|xcel)>|\n", src, flags=re.I):
            text = src[pos : m.start()]
            pos = m.end()
            if pending is not None:
                pending += text
            elif text.strip():
                # Text glued *before* a tag (seen in some outputs): treat as a cell.
                pending = text
            tok = m.group(1).lower() if m.group(1) else "nl"
            if tok == "fcel":
                flush()
                pending = ""
            elif tok == "ecel":
                flush()
                place("")
            elif tok == "lcel":
                flush()
                cover(0, 1)
            elif tok == "ucel":
                flush()
                cover(1, 0)
            elif tok == "xcel":
                flush()
                cover(1, 1)
            elif tok == "nl":
                flush()
                if col:
                    row += 1
                col = 0
        if pending is not None:
            pending += src[pos:]
        elif src[pos:].strip():
            pending = src[pos:]
        flush()
        return cls._finish(cells)

    @classmethod
    def from_pipe(cls, lines: list[str]) -> "Table | None":
        rows = []
        for i, line in enumerate(lines):
            s = line.strip()
            if i == 1 and re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", s):
                continue
            if s.startswith("|"):
                s = s[1:]
            if s.endswith("|") and not s.endswith("\\|"):
                s = s[:-1]
            rows.append([c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)])
        cells = [Cell(text=t, row=r, col=c) for r, row in enumerate(rows) for c, t in enumerate(row)]
        return cls._finish(cells)

    @classmethod
    def from_html(cls, src: str) -> "Table | None":
        p = _HTMLTableParser()
        p.feed(src)
        occupied: set[tuple[int, int]] = set()
        cells: list[Cell] = []
        for r, row in enumerate(p.rows):
            c = 0
            for text, rs, cs in row:
                while (r, c) in occupied:
                    c += 1
                cells.append(Cell(text=text.strip(), row=r, col=c, rowspan=rs, colspan=cs))
                for dr in range(rs):
                    for dc in range(cs):
                        occupied.add((r + dr, c + dc))
                c += cs
        return cls._finish(cells)

    @classmethod
    def _finish(cls, cells: list[Cell]) -> "Table | None":
        if not cells:
            return None
        n_rows = max(c.row + c.rowspan for c in cells)
        n_cols = max(c.col + c.colspan for c in cells)
        return cls(cells=cells, n_rows=n_rows, n_cols=n_cols)

    # ---------------------------------------------------------------- rendering
    @property
    def has_spans(self) -> bool:
        return any(c.rowspan > 1 or c.colspan > 1 for c in self.cells)

    def _rows(self) -> list[list[Cell]]:
        rows: list[list[Cell]] = [[] for _ in range(self.n_rows)]
        for c in self.cells:
            rows[c.row].append(c)
        for r in rows:
            r.sort(key=lambda c: c.col)
        return rows

    def _dense(self) -> list[list[str]]:
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            grid[c.row][c.col] = c.text
        return grid

    def to_markdown(self) -> str:
        """GFM pipe table when the grid is regular, otherwise an HTML table."""
        if self.has_spans:
            return self.to_html()
        grid = self._dense()
        esc = lambda s: s.replace("|", "\\|").replace("\n", " ")  # noqa: E731
        out = ["| " + " | ".join(esc(x) for x in grid[0]) + " |"]
        out.append("|" + "|".join(["---"] * self.n_cols) + "|")
        out += ["| " + " | ".join(esc(x) for x in row) + " |" for row in grid[1:]]
        return "\n".join(out)

    def to_html(self, render_cell=None) -> str:
        render_cell = render_cell or html.escape
        parts = ["<table>"]
        for r, row in enumerate(self._rows()):
            parts.append("<tr>")
            tag = "th" if r == 0 else "td"
            for c in row:
                attrs = ""
                if c.rowspan > 1:
                    attrs += f' rowspan="{c.rowspan}"'
                if c.colspan > 1:
                    attrs += f' colspan="{c.colspan}"'
                parts.append(f"<{tag}{attrs}>{render_cell(c.text)}</{tag}>")
            parts.append("</tr>")
        parts.append("</table>")
        return "".join(parts)

    def to_latex(self, render_cell) -> str:
        spec = "|" + "|".join(["c"] * self.n_cols) + "|"
        covered: dict[tuple[int, int], Cell] = {}
        for c in self.cells:
            for dr in range(c.rowspan):
                for dc in range(c.colspan):
                    covered[(c.row + dr, c.col + dc)] = c
        lines = [f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
        for r in range(self.n_rows):
            parts = []
            col = 0
            while col < self.n_cols:
                c = covered.get((r, col))
                if c is None:
                    parts.append("")
                    col += 1
                    continue
                body = render_cell(c.text) if (c.row == r) else ""
                if c.rowspan > 1 and c.row == r:
                    body = f"\\multirow{{{c.rowspan}}}{{*}}{{{body}}}"
                if c.colspan > 1:
                    body = f"\\multicolumn{{{c.colspan}}}{{|c|}}{{{body}}}"
                parts.append(body)
                col += c.colspan
            lines.append(" & ".join(parts) + r" \\")
            # \cline under columns whose cell does not continue to the next row
            if r == self.n_rows - 1:
                lines.append("\\hline")
            else:
                segs = _cline_segments(
                    [not (covered.get((r, k)) is not None and covered.get((r, k)) is covered.get((r + 1, k)))
                     for k in range(self.n_cols)]
                )
                if segs == [(1, self.n_cols)]:
                    lines.append("\\hline")
                else:
                    lines.append(" ".join(f"\\cline{{{a}-{b}}}" for a, b in segs))
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    def to_text(self) -> str:
        return "\n".join("\t".join(row) for row in self._dense())


def _cline_segments(flags: list[bool]) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            segs.append((start + 1, i))
            start = None
    return segs


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self._cell: list[str] | None = None
        self._span = (1, 1)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            self._cell = []
            self._span = (_int(a.get("rowspan")), _int(a.get("colspan")))
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append(("".join(self._cell), *self._span))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _int(v) -> int:
    try:
        return max(1, min(999, int(v)))
    except (TypeError, ValueError):
        return 1
