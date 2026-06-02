"""Export saved q-numbers to the formats a working mathematician uses.

One saved entry or a whole list renders to four texts:

- ``to_json``  the entries as JSON, the round-trip format;
- ``to_csv``   one row per constant, for a spreadsheet;
- ``to_latex`` a booktabs table ready to paste into a paper;
- ``to_magma`` Magma code that rebuilds each value as a Laurent series in q.

Every function returns a string and writes nothing. ``write_export`` is the one
function that touches the disk, and only when handed a path. The whole module is
core: sympy for rendering the input nicely, the standard library for the rest.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import sympy as sp

from .store import SavedEntry

FORMATS = ("json", "csv", "latex", "magma")
_EXTENSION = {"json": ".json", "csv": ".csv", "latex": ".tex", "magma": ".m"}


def extension_for(fmt: str) -> str:
    """The conventional file extension for a format name."""
    try:
        return _EXTENSION[fmt]
    except KeyError:
        raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")


def _as_list(entries: SavedEntry | Iterable[SavedEntry]) -> list[SavedEntry]:
    if isinstance(entries, SavedEntry):
        return [entries]
    return list(entries)


def render(entries: SavedEntry | Iterable[SavedEntry], fmt: str) -> str:
    """Render one entry or a list in the named format, returning the text."""
    items = _as_list(entries)
    if fmt == "json":
        return to_json(items)
    if fmt == "csv":
        return to_csv(items)
    if fmt == "latex":
        return to_latex(items)
    if fmt == "magma":
        return to_magma(items)
    raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")


def write_export(text: str, path: str | Path) -> Path:
    """Write rendered export text to a path. The only function here that writes."""
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(text)
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# JSON.
# --------------------------------------------------------------------------


def to_json(entries: SavedEntry | Iterable[SavedEntry]) -> str:
    """The entries as a JSON object with a small header, round-trippable."""
    items = _as_list(entries)
    payload = {
        "tool": "qreals",
        "format": "saved-results",
        "exported": _now(),
        "entries": [e.to_dict() for e in items],
    }
    return json.dumps(payload, indent=2) + "\n"


# --------------------------------------------------------------------------
# CSV (one row per constant).
# --------------------------------------------------------------------------

_CSV_COLUMNS = [
    "label",
    "input",
    "n",
    "valuation",
    "coefficients",
    "timestamp",
    "qprov_id",
]


def to_csv(entries: SavedEntry | Iterable[SavedEntry]) -> str:
    """One row per constant. Coefficients are a space-separated list in one cell."""
    items = _as_list(entries)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for e in items:
        writer.writerow(
            [
                e.label,
                e.input,
                e.n,
                e.valuation,
                " ".join(str(c) for c in e.coefficients),
                e.timestamp,
                e.qprov_id or "",
            ]
        )
    return buffer.getvalue()


# --------------------------------------------------------------------------
# LaTeX table.
# --------------------------------------------------------------------------


def _input_latex(text: str) -> str:
    """LaTeX for the input expression, falling back to upright text if unparsed."""
    try:
        return str(sp.latex(sp.sympify(text)))
    except Exception:  # noqa: BLE001 - a label such as an arithmetic combination
        escaped = text.replace("\\", " ").replace("_", r"\_").replace("^", r"\^{}")
        return r"\mathrm{" + escaped + "}"


def _laurent_latex(valuation: int, coeffs: list[int]) -> str:
    """A Laurent polynomial in q as LaTeX, low power to high, with an O() tail."""
    terms: list[str] = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        power = valuation + i
        if power == 0:
            mono = ""
        elif power == 1:
            mono = "q"
        else:
            mono = f"q^{{{power}}}"
        mag = abs(c)
        if mono == "":
            body = str(mag)
        elif mag == 1:
            body = mono
        else:
            body = f"{mag} {mono}"
        sign = "-" if c < 0 else "+"
        terms.append(f"{sign} {body}")
    tail = f"O(q^{{{valuation + len(coeffs)}}})"
    if not terms:
        return "0 + " + tail
    first = terms[0]
    head = first[2:] if first.startswith("+ ") else first.replace("- ", "-")
    return " ".join([head] + terms[1:]) + " + " + tail


def to_latex(entries: SavedEntry | Iterable[SavedEntry]) -> str:
    """A booktabs table of the saved values, ready to paste into a paper.

    A qprov-id column appears only when at least one entry carries an id, so the
    common table stays three columns wide.
    """
    items = _as_list(entries)
    with_qprov = any(e.qprov_id for e in items)
    colspec = "llll" if with_qprov else "lll"
    header = r"$x$ & order $N$ & $[x]_q$"
    if with_qprov:
        header += r" & qprov id"
    rows = []
    for e in items:
        cells = [
            f"${_input_latex(e.input)}$",
            f"${e.n}$",
            f"${_laurent_latex(e.valuation, e.coefficients)}$",
        ]
        if with_qprov:
            cells.append(r"\texttt{" + (e.qprov_id or "") + "}" if e.qprov_id else "")
        rows.append("    " + " & ".join(cells) + r" \\")
    rows_tex = "\n".join(rows)
    return (
        "% q-deformed reals exported by qreals. Needs \\usepackage{booktabs}.\n"
        "\\begin{table}[ht]\n"
        "  \\centering\n"
        f"  \\begin{{tabular}}{{{colspec}}}\n"
        "    \\toprule\n"
        f"    {header} \\\\\n"
        "    \\midrule\n"
        f"{rows_tex}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "  \\caption{q-deformed reals computed with \\texttt{qreals} via the "
        "Morier-Genoud-Ovsienko continued-fraction construction.}\n"
        "  \\label{tab:qreals}\n"
        "\\end{table}\n"
    )


def latex_document(table: str) -> str:
    """Wrap a table snippet in a standalone document, for a quick compile."""
    return (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{amssymb}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage[margin=2cm]{geometry}\n"
        "\\begin{document}\n"
        f"{table}"
        "\\end{document}\n"
    )


# --------------------------------------------------------------------------
# Magma.
# --------------------------------------------------------------------------


def _magma_ident(text: str, used: set[str]) -> str:
    """A valid, unique Magma identifier derived from the input expression."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    base = "q_" + slug if slug else "q_value"
    name = base
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def _magma_exponent(valuation: int) -> str:
    if valuation == 0:
        return "q^i"
    if valuation > 0:
        return f"q^(i + {valuation})"
    return f"q^(i - {abs(valuation)})"


def to_magma(entries: SavedEntry | Iterable[SavedEntry]) -> str:
    """Magma code rebuilding each value as a Laurent series in q over Q.

    For each entry it assigns the coefficient sequence, then the series built
    from it with the right valuation and an ``O(q^prec)`` precision marker. The
    output is valid Magma: paste it into a session whose ring is the one named
    on the first line.
    """
    items = _as_list(entries)
    used: set[str] = set()
    lines = [
        "// q-deformed reals exported by qreals.",
        "// Each value is a Laurent series in q over the rationals.",
        "R<q> := LaurentSeriesRing(RationalField());",
        "",
    ]
    for e in items:
        name = _magma_ident(e.input, used)
        coeff_name = f"coeffs_{name}"
        seq = "[" + ", ".join(str(c) for c in e.coefficients) + "]"
        prec = e.valuation + len(e.coefficients)
        lines.append(
            f"// {e.label}  computed by qreals to order {e.n} at {e.timestamp}"
        )
        if e.qprov_id:
            lines.append(f"// qprov id: {e.qprov_id}")
        lines.append(f"{coeff_name} := {seq};")
        lines.append(
            f"{name} := (&+[{coeff_name}[i+1]*{_magma_exponent(e.valuation)} "
            f": i in [0..#{coeff_name}-1]]) + O(q^{prec});"
        )
        lines.append("")
    return "\n".join(lines)
