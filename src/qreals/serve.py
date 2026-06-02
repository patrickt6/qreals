r"""A localhost web app over the qreals engine, rendered with MathJax.

`qreals serve` starts a small web app bound to 127.0.0.1 that exposes the same
computations as the command line through a browser. The page is a single-page
app: a home screen of operation cards, a per-operation view with a live MathJax
preview of the parsed input, and a result panel whose typeset output is the
dominant element. It POSTs to /compute, which calls the exact same engine
functions the CLI calls (the compute_* functions in qreals.app), and to
/preview, which renders the parsed input. Nothing here recomputes the math.

Saved results live in the browser (localStorage), mirroring the CLI saved list:
each result can be kept, re-opened, and removed without touching the disk.

The web framework is an optional extra. FastAPI is preferred; if it is not
importable the module falls back to Flask. If neither is installed the module
still imports cleanly and raises a clear ImportError only when build_app or
serve is actually called, so the core install never needs a web framework.

Install the extra with:  pip install qreals[serve]
Start it with:          qreals serve
"""

from __future__ import annotations

import json
import socket
from typing import Any

import sympy as sp

from . import app as _app
from . import exports
from . import features as _features
from . import format_laurent
from ._parsing import parse_real
from .store import SavedEntry

# --------------------------------------------------------------------------
# The operation registry. Every CLI capability appears here, grouped for the
# card home screen, with the fields each one needs (name, label, example, and
# type). The example doubles as the default value shown in the form. compute()
# below dispatches on the key and calls the matching qreals.app compute_*
# function; this registry adds no math of its own.
# --------------------------------------------------------------------------

_REAL = "real"
_RATIONAL = "rational"
_INTEGER = "integer"
_SEQUENCE = "sequence"


def _f(
    name: str, label: str, example: str, type_: str = "text", **extra: Any
) -> dict[str, Any]:
    field: dict[str, Any] = {
        "name": name,
        "label": label,
        "example": example,
        "type": type_,
    }
    field.update(extra)
    return field


OPERATIONS: dict[str, dict[str, Any]] = {
    # ---- q-rationals -----------------------------------------------------
    "rational": {
        "name": "Exact q-rational",
        "symbol": "[p/s]_q",
        "group": "q-rationals",
        "blurb": "The exact rational function in q for a fraction p/s.",
        "input_kind": _RATIONAL,
        "fields": [_f("input", "rational p/s", "3/2")],
    },
    "qint": {
        "name": "q-integer",
        "symbol": "[n]_q",
        "group": "q-rationals",
        "blurb": "The q-analog of a whole number, [n]_q and [n]_(1/q).",
        "input_kind": _INTEGER,
        "fields": [_f("input", "whole number n", "5", "int")],
    },
    "factor": {
        "name": "Factor R(q), S(q)",
        "symbol": "R(q) / S(q)",
        "group": "q-rationals",
        "blurb": "Factor R(q) and S(q) of [a/b]_q over Z[q], labelling the "
        "cyclotomic factors.",
        "input_kind": _RATIONAL,
        "fields": [_f("input", "rational a/b", "7/5")],
    },
    "roots": {
        "name": "Roots of R(q)",
        "symbol": "roots of R(q)",
        "group": "q-rationals",
        "blurb": "Plot the complex roots of the numerator R(q) of [a/b]_q on the "
        "unit circle, splitting cyclotomic (n-gon vertices) from the core.",
        "input_kind": _RATIONAL,
        "fields": [_f("input", "rational a/b", "7/5")],
    },
    "jump-gap": {
        "name": "Jump gap",
        "symbol": "[p/s]_q^+ - [p/s]_q^-",
        "group": "q-rationals",
        "blurb": "The right and left q-versions of p/s and the factored gap "
        "between them.",
        "input_kind": _RATIONAL,
        "fields": [_f("input", "rational p/s", "3/5")],
    },
    "exact-rational": {
        "name": "Exact rational function",
        "symbol": "P(q) / Q(q)",
        "group": "q-rationals",
        "blurb": "The exact [x]_q = P/Q for a q-rational x. Give a second "
        "rational y to get the exact difference [x]_q - [y]_q = "
        "(P_x Q_y - P_y Q_x)/(Q_x Q_y), factored, with the Q_x | Q_y "
        "divisibilities. Leave y blank for the single rational function.",
        "input_kind": _RATIONAL,
        "fields": [
            _f("input", "rational x", "7/5"),
            _f("y", "rational y (optional)", "3/2"),
        ],
    },
    # ---- friezes ---------------------------------------------------------
    "frieze": {
        "name": "Conway-Coxeter frieze",
        "symbol": "frieze of a/b",
        "group": "Friezes",
        "blurb": "The Conway-Coxeter frieze of a/b > 1 with the q-coefficient "
        "overlay on every cell, drawn as an interactive diagonal lattice.",
        "input_kind": _RATIONAL,
        "fields": [_f("input", "rational a/b (> 1)", "19/7")],
    },
    # ---- visuals ---------------------------------------------------------
    "coeff-surface": {
        "name": "Coefficient landscape",
        "symbol": "c_n([x]_q)",
        "group": "Visuals",
        "blurb": "A 3D surface of the Taylor coefficients c_n of [x]_q as both n "
        "and x vary: zero-runs are valleys, sign flips cross the floor.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "start x_min", "1"),
            _f("x_max", "end x_max", "2"),
            _f("n", "coefficients per x (N)", "14", "int"),
        ],
    },
    "root-sweep": {
        "name": "Root migration",
        "symbol": "roots of R(q) vs b",
        "group": "Visuals",
        "blurb": "Stack the complex roots of R(q) for [a/b]_q as the denominator "
        "b sweeps: cyclotomic roots stay pinned on the unit circle, the core drifts.",
        "input_kind": _INTEGER,
        "fields": [
            _f("input", "numerator a", "7", "int"),
            _f("b_max", "max denominator b", "12", "int"),
        ],
    },
    "radius-grid": {
        "name": "Radius landscape",
        "symbol": "radius of [a/b]_q",
        "group": "Visuals",
        "blurb": "The radius of convergence (nearest pole |q|) of [a/b]_q over a "
        "Farey grid, read on a number line, as Ford circles, or as an (a, b) grid.",
        "input_kind": _INTEGER,
        "fields": [
            _f("input", "max denominator b", "8", "int"),
            _f("a_max", "max numerator a", "12", "int"),
        ],
    },
    # ---- q-reals ---------------------------------------------------------
    "coefficients": {
        "name": "Coefficients",
        "symbol": "[x]_q",
        "group": "q-reals",
        "blurb": "The first N Taylor coefficients of [x]_q for any real x.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "pi"),
            _f("n", "how many coefficients N", "12", "int"),
        ],
    },
    "laurent": {
        "name": "Laurent expansion",
        "symbol": "[x]_q + O(q^k)",
        "group": "q-reals",
        "blurb": "[x]_q written out to a chosen power, with its integer-part "
        "prefix.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "pi"),
            _f("order", "highest power q^order", "12", "int"),
        ],
    },
    "prefix": {
        "name": "Integer-part prefix",
        "symbol": "[floor(x)]_q",
        "group": "q-reals",
        "blurb": "The forced opening block of [x]_q fixed by floor(x).",
        "input_kind": _REAL,
        "fields": [_f("input", "real number x (>= 0)", "pi")],
    },
    "locked": {
        "name": "Convergent locking",
        "symbol": "S_n",
        "group": "q-reals",
        "blurb": "How many Laurent coefficients the n-th convergent of x pins "
        "down.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "pi"),
            _f("n", "convergent index n", "2", "int"),
        ],
    },
    "shift": {
        "name": "Shift by one",
        "symbol": "[x +/- 1]_q",
        "group": "q-reals",
        "blurb": "Move the argument by one: [x+1]_q = q[x]_q + 1, "
        "[x-1]_q = ([x]_q - 1)/q.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "pi"),
            _f(
                "direction",
                "which shift",
                "up",
                "select",
                choices=[
                    {"value": "up", "label": "up: [x+1]_q"},
                    {"value": "down", "label": "down: [x-1]_q"},
                ],
            ),
            _f("order", "highest power q^order", "12", "int"),
        ],
    },
    "readouts": {
        "name": "Coefficient read-outs",
        "symbol": "c_k pattern",
        "group": "q-reals",
        "blurb": "First nonzero, first negative, largest size, and zero count.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "sqrt(2)"),
            _f("n", "how many coefficients N", "30", "int"),
        ],
    },
    "radius": {
        "name": "Radius of convergence",
        "symbol": "rho([x]_q)",
        "group": "q-reals",
        "blurb": "A running-max estimate of the radius of convergence of [x]_q.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "pi"),
            _f("n", "how many coefficients N", "60", "int"),
        ],
    },
    "fingerprint": {
        "name": "Fingerprint",
        "symbol": "phi([x]_q)",
        "group": "q-reals",
        "blurb": "A named, fixed-length feature vector of [x]_q for "
        "nearest-neighbour.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "pi"),
            _f(
                "n_coeffs",
                "coefficients in the fingerprint",
                str(_features.DEFAULT_N_COEFFS),
                "int",
            ),
        ],
    },
    "certificate": {
        "name": "Certificate",
        "symbol": "[x]_q certificate",
        "group": "q-reals",
        "blurb": "The coefficients of [x]_q with a ready-to-paste LaTeX table.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "pi"),
            _f("n", "how many coefficients N", "12", "int"),
        ],
    },
    # ---- arithmetic ------------------------------------------------------
    "q-sum": {
        "name": "q-sum",
        "symbol": "[x]_q + [y]_q",
        "group": "Arithmetic",
        "blurb": "The series sum [x]_q + [y]_q via the q-Gosper engine.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "3/2"),
            _f("y", "real number y", "13/5"),
            _f("n", "how many coefficients N", "12", "int"),
        ],
    },
    "q-product": {
        "name": "q-product",
        "symbol": "[x]_q * [y]_q",
        "group": "Arithmetic",
        "blurb": "The series product [x]_q * [y]_q.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "3/2"),
            _f("y", "real number y", "13/5"),
            _f("n", "how many coefficients N", "12", "int"),
        ],
    },
    "deficit": {
        "name": "Deficit",
        "symbol": "[x op y]_q - ([x]_q op [y]_q)",
        "group": "Arithmetic",
        "blurb": "How far [x]_q +/* [y]_q sits from [x +/* y]_q, with the q=1 "
        "and q=0 checks.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x", "3/2"),
            _f("y", "real number y", "5/2"),
            _f(
                "op",
                "which operation",
                "add",
                "select",
                choices=[
                    {"value": "add", "label": "add: [x+y]_q vs [x]_q + [y]_q"},
                    {"value": "mul", "label": "mul: [x*y]_q vs [x]_q * [y]_q"},
                ],
            ),
            _f("n", "how many coefficients N", "12", "int"),
        ],
    },
    # ---- transfer matrices -----------------------------------------------
    "quad-arith": {
        "name": "Quadratic-irrational arithmetic",
        "symbol": "x o y  via  M_x (x) M_y",
        "group": "Transfer matrices",
        "blurb": "Exact x op y for two quadratic irrationals, read off from the "
        "dominant eigenvector of the continued-fraction transfer matrix "
        "K = M_x (x) M_y. Reproduces the golden + silver worked example; the "
        "four eigenvalues of K are plotted on the complex plane.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "quadratic irrational x", "(1 + sqrt(5))/2"),
            _f("y", "quadratic irrational y", "1 + sqrt(2)"),
            _f(
                "op",
                "operation",
                "add",
                "select",
                choices=[
                    {"value": "add", "label": "add:  x + y"},
                    {"value": "sub", "label": "sub:  x - y"},
                    {"value": "mul", "label": "mul:  x * y"},
                    {"value": "div", "label": "div:  x / y"},
                ],
            ),
        ],
    },
    # ---- symmetry --------------------------------------------------------
    "negation": {
        "name": "q-negation",
        "symbol": "[-x]_q",
        "group": "Symmetry",
        "blurb": "The Jouteur negation [-x]_q and the x to -x symmetry.",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x (>= 0)", "sqrt(2)"),
            _f("n", "how many coefficients N", "12", "int"),
        ],
    },
    "finiteness": {
        "name": "Negation-sum finiteness",
        "symbol": "[x]_q + [-x]_q",
        "group": "Symmetry",
        "blurb": "Whether [x]_q + [-x]_q is a finite Laurent polynomial "
        "(Ovsienko Example 6.4).",
        "input_kind": _REAL,
        "fields": [
            _f("input", "real number x (>= 0)", "sqrt(2)"),
            _f("n", "how many coefficients N", "12", "int"),
        ],
    },
    # ---- lookup ----------------------------------------------------------
    "oeis": {
        "name": "OEIS lookup",
        "symbol": "A-number",
        "group": "Lookup",
        "blurb": "Look a coefficient sequence up in the OEIS, re-verified "
        "against b-files.",
        "input_kind": _SEQUENCE,
        "fields": [_f("input", "coefficient sequence", "1,1,2,5,14,42,132,429")],
    },
}

# The order the groups appear on the home screen.
GROUP_ORDER = ["q-rationals", "Friezes", "Visuals", "q-reals", "Arithmetic", "Transfer matrices", "Symmetry", "Lookup"]

# The LaTeX form of each operation's headline symbol, rendered with MathJax on
# the home cards and the per-operation header. (The plain "symbol" string above
# stays as an accessible label and a fallback.)
_TEX_SYMBOL = {
    "rational": r"\left[\tfrac{p}{s}\right]_q",
    "qint": r"[n]_q",
    "factor": r"\dfrac{R(q)}{S(q)}",
    "roots": r"\{\,q : R(q)=0\,\}",
    "jump-gap": r"\left[\tfrac{p}{s}\right]_q^{+}-\left[\tfrac{p}{s}\right]_q^{-}",
    "exact-rational": r"\dfrac{P(q)}{Q(q)}",
    "frieze": r"\left[\tfrac{a}{b}\right]_q\ \text{frieze}",
    "coeff-surface": r"c_n\!\left([x]_q\right)",
    "root-sweep": r"\{\,q:R(q)=0\,\}\ \text{vs}\ b",
    "radius-grid": r"\rho\!\left(\left[\tfrac{a}{b}\right]_q\right)",
    "coefficients": r"[x]_q",
    "laurent": r"[x]_q+O(q^{k})",
    "prefix": r"\big[\lfloor x\rfloor\big]_q",
    "locked": r"S_n",
    "shift": r"[x\pm 1]_q",
    "readouts": r"(c_0,c_1,c_2,\dots)",
    "radius": r"\rho\!\left([x]_q\right)",
    "fingerprint": r"\varphi\!\left([x]_q\right)",
    "certificate": r"[x]_q",
    "q-sum": r"[x]_q+[y]_q",
    "q-product": r"[x]_q\cdot[y]_q",
    "deficit": r"[x+y]_q-\left([x]_q+[y]_q\right)",
    "negation": r"[-x]_q",
    "finiteness": r"[x]_q+[-x]_q",
    "quad-arith": r"x\circ y\ \text{via}\ M_x\otimes M_y",
    "oeis": r"(a_n)\to\text{A-number}",
}


def _annotate_registry() -> None:
    """Add the MathJax symbol and per-field LaTeX defaults to OPERATIONS.

    The card and panel symbols render as math; a math-input field shows its
    example as a typeset default (for instance 3/2 as a fraction). Computing the
    LaTeX once here keeps the page template free of any math of its own.
    """
    for op, meta in OPERATIONS.items():
        meta["tex"] = _TEX_SYMBOL.get(op, "")
        if meta["input_kind"] in (_REAL, _RATIONAL):
            for field in meta["fields"]:
                if field["name"] in ("input", "y"):
                    try:
                        field["tex"] = str(sp.latex(sp.sympify(field["example"])))
                    except Exception:  # noqa: BLE001 - keep the raw example
                        field["tex"] = field["example"]


_annotate_registry()


# --------------------------------------------------------------------------
# The compute bridge. Each branch calls the same engine entry point the CLI
# uses and returns {latex, text, rows, meta}. Nothing here recomputes the math.
# --------------------------------------------------------------------------


def _laurent_entry(
    input_label: str, coeffs: list[int], valuation: int = 0
) -> SavedEntry:
    """Wrap a coefficient list as a SavedEntry so exports.to_latex can render it."""
    return SavedEntry(
        input=input_label,
        n=len(coeffs),
        coefficients=[int(c) for c in coeffs],
        label=f"[{input_label}]_q",
        kind="coeffs",
        valuation=int(valuation),
    )


def _latex_for_coeffs(coeffs: list[int], valuation: int = 0) -> str:
    """LaTeX for a Laurent series, reusing the exports renderer."""
    return exports._laurent_latex(int(valuation), [int(c) for c in coeffs])


def _sym(text: str) -> str:
    """LaTeX for a parsed real, falling back to the raw text if it will not parse."""
    try:
        return str(sp.latex(parse_real(text)))
    except Exception:  # noqa: BLE001 - a label is shown verbatim if unparsed
        return str(text)


def _idx(i: int) -> str:
    return "none" if i == -1 else str(i)


def _int_arg(args: dict[str, Any], name: str, default: int) -> int:
    value = args.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _frieze_backend() -> Any:
    """Return a frieze module: the installed qfrieze if its q-overlay works, else
    the vendored copy in qreals._frieze.

    The serve frieze card reuses qfrieze's mathematics rather than recomputing a
    frieze. An overlay-capable qfrieze (v0.1B and later) is preferred when it is
    importable; older builds that ship only the integer frieze (their
    frieze_coefficients raises) fall back to the vendored module, which carries
    the same conventions and qfrieze's MIT attribution.
    """
    try:
        import qfrieze as _qf  # type: ignore

        if getattr(_qf, "OVERLAY_AVAILABLE", False):
            return _qf
    except Exception:  # noqa: BLE001 - any import/attribute issue falls back
        pass
    from . import _frieze as _vendored

    return _vendored


def _frieze_data(input_text: str, cols_periods: int = 1) -> dict[str, Any]:
    """Build the JSON-ready frieze of a/b: the integer triangle plus the
    q-coefficient overlay, ready for the front end to draw as a diagonal lattice.

    The math comes entirely from the frieze backend (qfrieze or the vendored
    copy); nothing here recomputes a frieze. ``cols_periods`` is how many
    horizontal periods to lay out (the frieze is periodic with period = width).
    """
    p, s = _app._parse_rational(input_text)
    backend = _frieze_backend()
    f = backend.frieze(p, s)
    cm = backend.frieze_coefficients(p, s)

    width = int(f.width)
    cols = max(1, int(cols_periods)) * width
    last = len(f.rows) - 1
    drawn = list(range(1, last))  # skip the two all-0 borders

    rows_out: list[dict[str, Any]] = []
    for i in drawn:
        is_border = i == 1 or i == last - 1
        cells = []
        for j in range(cols):
            value = int(f.cell(i, j))
            poly = cm.cell(i, j)
            cells.append(
                {
                    "value": value,
                    "coeffs": [int(c) for c in poly.coeffs()],
                    "tex": poly.to_latex(),
                    "txt": poly.to_superscript(),
                    "border": is_border,
                }
            )
        rows_out.append({"border": is_border, "cells": cells})

    qnum = cm.q_numerator()
    return {
        "r": int(f.r),
        "s": int(f.s),
        "cf": [int(a) for a in f.cf],
        "quiddity": [int(c) for c in f.quiddity],
        "n_polygon": int(f.n_polygon),
        "width": width,
        "cols": cols,
        "rows": rows_out,
        "q_numerator": {"tex": qnum.to_latex(), "txt": qnum.to_superscript()},
    }


def compute(
    op: str, input_text: str, args: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run one operation and return {latex, text, rows, meta}.

    op is a key of OPERATIONS; input_text is the user's main input; args carries
    the other fields ("y", "n", "order", "direction", "op", "n_coeffs"). The
    branch calls the matching qreals.app compute_* function and renders the
    result with the existing LaTeX path. Errors are returned as a payload with
    an "error" field rather than raised, so the page can show them.
    """
    args = args or {}
    n = _int_arg(args, "n", OPERATIONS.get(op, {}).get("default_n", 12))
    y = str(args.get("y") or "")

    try:
        if op == "rational":
            p, s = _app._parse_rational(input_text)
            res = _app.compute_rational(p, s)
            data = res["data"]
            expr = sp.sympify(data["expr"])
            latex = r"\left[\tfrac{%d}{%d}\right]_q = %s" % (p, s, sp.latex(expr))
            return {
                "latex": latex,
                "text": data["expr"],
                "rows": [["value at q = 1", str(data["at_q_eq_1"])]],
                "meta": {"op": op, "p": p, "s": s},
            }

        if op == "exact-rational":
            from .qrat_exact import q_rational_difference, q_rational_exact
            from .rational import q as _q

            def _qlab(a: int, b: int) -> str:
                return (r"[%d]_q" % a) if b == 1 else (
                    r"\left[\tfrac{%d}{%d}\right]_q" % (a, b)
                )

            yv = str(args.get("y") or "").strip()
            if not yv:
                ex = q_rational_exact(input_text)
                P, Q = sp.factor(ex.P), sp.factor(ex.Q)
                latex = r"%s = \dfrac{%s}{%s}" % (
                    _qlab(ex.a, ex.b),
                    sp.latex(P),
                    sp.latex(Q),
                )
                at1 = sp.cancel(ex.P / ex.Q).subs(_q, 1)
                return {
                    "latex": latex,
                    "text": "[%s]_q = (%s)/(%s)"
                    % (sp.Rational(ex.a, ex.b), sp.sstr(P), sp.sstr(Q)),
                    "rows": [
                        ["numerator P(q)", sp.sstr(P)],
                        ["denominator Q(q)", sp.sstr(Q)],
                        ["continued fraction", str(ex.cf)],
                        ["at q = 1", "%s  (ordinary %s)" % (at1, sp.Rational(ex.a, ex.b))],
                    ],
                    "meta": {"op": op, "a": ex.a, "b": ex.b},
                }
            d = q_rational_difference(input_text, yv)
            num_f, den_f = sp.factor(d.num), sp.factor(d.den)
            xl, yl = _qlab(d.x.a, d.x.b), _qlab(d.y.a, d.y.b)
            latex = r"%s - %s = \dfrac{%s}{%s}" % (
                xl,
                yl,
                sp.latex(num_f),
                sp.latex(den_f),
            )
            return {
                "latex": latex,
                "text": "[%s]_q - [%s]_q = (%s)/(%s)"
                % (
                    sp.Rational(d.x.a, d.x.b),
                    sp.Rational(d.y.a, d.y.b),
                    sp.sstr(num_f),
                    sp.sstr(den_f),
                ),
                "rows": [
                    ["[x]_q", "(%s)/(%s)" % (sp.sstr(sp.factor(d.x.P)), sp.sstr(sp.factor(d.x.Q)))],
                    ["[y]_q", "(%s)/(%s)" % (sp.sstr(sp.factor(d.y.P)), sp.sstr(sp.factor(d.y.Q)))],
                    ["numerator P_x Q_y - P_y Q_x", sp.sstr(sp.factor(d.num_unreduced))],
                    ["denominator Q_x Q_y", sp.sstr(sp.factor(d.den_unreduced))],
                    ["Q_x | Q_y", "yes" if d.qx_divides_qy else "no"],
                    ["Q_y | Q_x", "yes" if d.qy_divides_qx else "no"],
                    ["Q_x = Q_y up to a unit", "yes" if d.q_equal_up_to_unit else "no"],
                    ["gcd(Q_x, Q_y)", sp.sstr(d.gcd_Q)],
                    [
                        "difference at q = 1",
                        "%s  (ordinary %s - %s)"
                        % (
                            d.value_at_1,
                            sp.Rational(d.x.a, d.x.b),
                            sp.Rational(d.y.a, d.y.b),
                        ),
                    ],
                ],
                "meta": {
                    "op": op,
                    "q_equal_up_to_unit": bool(d.q_equal_up_to_unit),
                },
            }

        if op == "qint":
            nn = int(str(input_text).strip())
            res = _app.compute_qint(nn)
            data = res["data"]
            latex = r"[%d]_q = %s" % (nn, sp.latex(sp.sympify(data["q_int"])))
            return {
                "latex": latex,
                "text": data["q_int"],
                "rows": [
                    [f"[{nn}]_(1/q)", str(data["q_int_qinv"])],
                    ["value at q = 1", str(nn)],
                ],
                "meta": {"op": op, "n": nn},
            }

        if op == "coefficients":
            res = _app.compute_coeffs(input_text, n)
            coeffs = res["data"]["coefficients"]
            latex = r"\left[%s\right]_q = %s" % (_sym(input_text), _latex_for_coeffs(coeffs))
            return {
                "latex": latex,
                "text": format_laurent(coeffs),
                "rows": [["coefficients c_0..", ", ".join(str(c) for c in coeffs)]],
                "meta": {"op": op, "n": n},
            }

        if op == "laurent":
            order = _int_arg(args, "order", 12)
            res = _app.compute_laurent(input_text, order)
            data = res["data"]
            latex = r"\left[%s\right]_q = %s" % (
                _sym(input_text),
                _latex_for_coeffs(data["coefficients"]),
            )
            return {
                "latex": latex,
                "text": format_laurent(data["coefficients"]),
                "rows": [
                    [
                        "integer-part prefix",
                        format_laurent(data["integer_part_prefix"]),
                    ],
                    ["floor(x)", str(data["floor"])],
                ],
                "meta": {"op": op, "order": order},
            }

        if op == "prefix":
            res = _app.compute_prefix(input_text)
            data = res["data"]
            latex = r"\text{prefix of } \left[%s\right]_q = %s" % (
                _sym(input_text),
                _latex_for_coeffs(data["prefix"]),
            )
            return {
                "latex": latex,
                "text": format_laurent(data["prefix"]),
                "rows": [["floor(x)", str(data["floor"])]],
                "meta": {"op": op},
            }

        if op == "locked":
            res = _app.compute_locked(input_text, n)
            data = res["data"]
            s_n = int(data["S_n"])
            latex = (
                r"S_{%d} = %d \quad\Longrightarrow\quad \text{%d coefficients locked}"
                % (n, s_n, int(data["locked"]))
            )
            return {
                "latex": latex,
                "text": f"the {n}-th convergent fixes q^0 through q^{max(s_n - 2, 0)}",
                "rows": [
                    [f"continued fraction (first {n})", str(data["cf_terms"])],
                    ["partial sum S_n", str(s_n)],
                    ["coefficients locked in", str(data["locked"])],
                    ["first power that may differ", f"q^{s_n - 1}"],
                ],
                "meta": {"op": op, "n": n},
            }

        if op == "shift":
            order = _int_arg(args, "order", 12)
            direction = "down" if str(args.get("direction", "up")) == "down" else "up"
            res = _app.compute_shift(input_text, order, direction)
            data = res["data"]
            sign = "+" if direction == "up" else "-"
            formula = "q[x]_q + 1" if direction == "up" else "([x]_q - 1)/q"
            latex = r"\left[%s %s 1\right]_q = %s" % (
                _sym(input_text),
                sign,
                _latex_for_coeffs(data["shifted_coefficients"]),
            )
            return {
                "latex": latex,
                "text": format_laurent(data["shifted_coefficients"]),
                "rows": [
                    ["[x]_q", format_laurent(data["input_coefficients"])],
                    ["formula", formula],
                ],
                "meta": {"op": op, "order": order, "direction": direction},
            }

        if op == "readouts":
            res = _app.compute_readouts(input_text, n)
            data = res["data"]
            latex = r"\left[%s\right]_q = %s" % (
                _sym(input_text),
                _latex_for_coeffs(data["coefficients"]),
            )
            return {
                "latex": latex,
                "text": format_laurent(data["coefficients"]),
                "rows": [
                    ["first nonzero coefficient index", _idx(data["first_nonzero_index"])],
                    ["first negative coefficient index", _idx(data["first_negative_index"])],
                    ["largest absolute coefficient", str(data["max_abs"])],
                    ["number of zero coefficients", str(data["zeros"])],
                ],
                "meta": {"op": op, "n": n},
            }

        if op in ("q-sum", "q-product"):
            kind = "mul" if op == "q-product" else "add"
            res = _app.compute_arith(input_text, y, n, kind)
            coeffs = res["data"]["coefficients"]
            sym = r"\cdot" if kind == "mul" else "+"
            latex = r"\left[%s\right]_q %s \left[%s\right]_q = %s" % (
                _sym(input_text),
                sym,
                _sym(y),
                _latex_for_coeffs(coeffs),
            )
            name = "product" if kind == "mul" else "sum"
            opc = "*" if kind == "mul" else "+"
            return {
                "latex": latex,
                "text": format_laurent(coeffs),
                "rows": [
                    [
                        "note",
                        f"series {name}, not [x {opc} y]_q; the MGO map x -> [x]_q "
                        "is not a ring homomorphism",
                    ]
                ],
                "meta": {"op": op, "y": y, "n": n},
            }

        if op == "quad-arith":
            from .transfer import quad_arith as _quad_arith

            qop = str(args.get("op", "add"))
            if qop not in ("add", "sub", "mul", "div"):
                qop = "add"
            r = _quad_arith(input_text, y, qop)
            tex_sym = {"add": "+", "sub": "-", "mul": r"\cdot", "div": "/"}[qop]
            latex = (
                r"\begin{aligned}"
                r"\left(%s\right) %s \left(%s\right) &= %s \\[5pt]"
                r"K = M_x \otimes M_y &= %s"
                r"\end{aligned}"
            ) % (
                sp.latex(r.x),
                tex_sym,
                sp.latex(r.y),
                sp.latex(r.value),
                sp.latex(r.matrix),
            )
            dominant = complex(sp.N(r.dominant_eigenvalue))
            points = []
            for eigenvalue in r.eigenvalues:
                c = complex(sp.N(eigenvalue))
                points.append(
                    {
                        "re": c.real,
                        "im": c.imag,
                        "mod": abs(c),
                        "dominant": abs(c - dominant) < 1e-9,
                        "label": sp.sstr(eigenvalue),
                    }
                )
            txt_sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[qop]
            rows = [
                ["x %s y (exact)" % txt_sym, sp.sstr(r.value)],
                ["decimal", "%.12g" % r.decimal],
                ["dominant eigenvalue of K", sp.sstr(r.dominant_eigenvalue)],
                [
                    "verified against direct arithmetic",
                    "yes" if r.verified else "no",
                ],
            ]
            return {
                "latex": latex,
                "text": "%s %s %s = %s"
                % (sp.sstr(r.x), txt_sym, sp.sstr(r.y), sp.sstr(r.value)),
                "rows": rows,
                "meta": {
                    "op": op,
                    "matrix": [[int(v) for v in row] for row in r.matrix.tolist()],
                    "eigen": {"points": points},
                    "verified": r.verified,
                },
            }

        if op == "deficit":
            dop = "mul" if str(args.get("op", "add")) == "mul" else "add"
            res = _app.compute_deficit(input_text, y, n, dop)
            data = res["data"]
            sym = r"\cdot" if dop == "mul" else "+"
            target = r"\left[%s %s %s\right]_q" % (_sym(input_text), sym, _sym(y))
            engine = r"\left(\left[%s\right]_q %s \left[%s\right]_q\right)" % (
                _sym(input_text),
                sym,
                _sym(y),
            )
            latex = r"%s - %s = %s" % (
                target,
                engine,
                _latex_for_coeffs(data["deficit"]),
            )
            rows: list[list[str]] = []
            if data.get("exact"):
                rows.append(["exact closed form in q", str(data["exact"])])
            rows.append(["deficit at q = 1", str(data["deficit_at_q1"])])
            rows.append(["deficit at q = 0", str(data["deficit_at_q0"])])
            return {
                "latex": latex,
                "text": format_laurent(data["deficit"]),
                "rows": rows,
                "meta": {"op": op, "y": y, "n": n},
            }

        if op == "negation":
            res = _app.compute_negation(input_text, n)
            data = res["data"]
            latex = r"\left[-%s\right]_q = %s" % (
                _sym(input_text),
                exports._laurent_latex(
                    int(data["neg_valuation"]),
                    [int(c) for c in data["neg_coefficients"]],
                ),
            )
            return {
                "latex": latex,
                "text": _app._format_laurent_v(
                    int(data["neg_valuation"]), list(data["neg_coefficients"])
                ),
                "rows": [
                    [
                        "[x]_q + [-x]_q",
                        _app._format_laurent_v(
                            int(data["sum_valuation"]), list(data["sum_coefficients"])
                        ),
                    ],
                    ["[x]_q + [-x]_q finite?", "yes" if data["finite"] else "no"],
                ],
                "meta": {"op": op, "n": n, "finite": data["finite"]},
            }

        if op == "finiteness":
            res = _app.compute_negsum(input_text, n)
            data = res["data"]
            verdict = (
                "finite (Laurent polynomial)" if data["finite"] else "infinite series"
            )
            latex = r"\left[%s\right]_q + \left[-%s\right]_q = %s" % (
                _sym(input_text),
                _sym(input_text),
                exports._laurent_latex(
                    int(data["valuation"]),
                    [int(c) for c in data["sum_coefficients"]],
                ),
            )
            return {
                "latex": latex,
                "text": f"[{input_text}]_q + [-{input_text}]_q is {verdict}",
                "rows": [
                    ["finite?", "yes" if data["finite"] else "no"],
                    [
                        "reason",
                        "finite exactly for pure square roots (trace-zero "
                        "quadratics), Ovsienko Example 6.4",
                    ],
                ],
                "meta": {"op": op, "n": n, "finite": data["finite"]},
            }

        if op == "jump-gap":
            p, s = _app._parse_rational(input_text)
            res = _app.compute_jumpgap(p, s)
            data = res["data"]
            latex = r"\left[\tfrac{%d}{%d}\right]_q^{+} - \left[\tfrac{%d}{%d}\right]_q^{-} = %s" % (
                p,
                s,
                p,
                s,
                sp.latex(sp.sympify(data["gap"])),
            )
            return {
                "latex": latex,
                "text": f"gap = {data['gap']}",
                "rows": [
                    ["exponent E (det M_q = q^E)", str(data["exponent"])],
                    ["continued fraction", str(data["cf"])],
                    ["S^+ (right denominator)", str(data["s_plus"])],
                    ["S^- (left denominator)", str(data["s_minus"])],
                    ["[p/s]_q^+ (limit from above)", str(data["right"])],
                    ["[p/s]_q^- (limit from below)", str(data["left"])],
                ],
                "meta": {"op": op, "p": p, "s": s, "exponent": data["exponent"]},
            }

        if op == "frieze":
            data = _frieze_data(input_text)
            latex = r"\left[\tfrac{%d}{%d}\right]_q:\quad R(q) = %s" % (
                data["r"],
                data["s"],
                data["q_numerator"]["tex"],
            )
            # Plain-text form: one drawn row per line, cells separated by " | ",
            # with the staggered half-cell offset shown as leading spaces.
            text_lines = []
            for ri, row in enumerate(data["rows"]):
                indent = "   " if (ri % 2) else ""
                text_lines.append(
                    indent + "  |  ".join(c["txt"] for c in row["cells"])
                )
            cf = ", ".join(str(a) for a in data["cf"])
            quid = ", ".join(str(c) for c in data["quiddity"])
            header = (
                f"q-frieze of {data['r']}/{data['s']}   CF = [{cf}]   "
                f"quiddity = ({quid})   ({data['n_polygon']}-gon)"
            )
            text = header + "\n" + "\n".join(text_lines)
            return {
                "latex": latex,
                "text": text,
                "rows": [
                    ["regular continued fraction", f"[{cf}]"],
                    ["quiddity (row 2 = [c_i]_q)", f"({quid})"],
                    ["polygon size n", str(data["n_polygon"])],
                    ["numerator R(q) of [a/b]_q at q = 1", str(data["r"])],
                ],
                "meta": {"op": op, "frieze": data},
            }

        if op == "radius":
            res = _app.compute_radius(input_text, n)
            data = res["data"]
            if data["infinite"]:
                value_tex, value_txt = r"\infty", "infinite"
            else:
                value_tex = "%.6f" % data["radius"]
                value_txt = value_tex
            latex = r"\rho\!\left(\left[%s\right]_q\right) \approx %s" % (
                _sym(input_text),
                value_tex,
            )
            return {
                "latex": latex,
                "text": f"radius estimate (N={n}): {value_txt}",
                "rows": [
                    ["N", str(n)],
                    ["method", "running-max root test, biased high at finite N"],
                ],
                "meta": {"op": op, "n": n},
            }

        if op == "oeis":
            res = _app.compute_oeis(input_text)
            data = res["data"]
            hits = data.get("hits", [])
            if hits:
                anums = ", ".join(h.get("anum", "?") for h in hits[:5])
                rows = [
                    [
                        h.get("anum", "?"),
                        str(h.get("name", ""))[:70],
                    ]
                    for h in hits[:8]
                ]
                return {
                    "latex": r"\text{OEIS: } " + anums,
                    "text": f"OEIS hits: {anums}",
                    "rows": rows,
                    "meta": {"op": op, "hits": len(hits)},
                }
            return {
                "latex": r"\text{no OEIS match}",
                "text": "no OEIS match (or OEIS unreachable)",
                "rows": [],
                "meta": {"op": op, "hits": 0},
            }

        if op == "fingerprint":
            n_coeffs = _int_arg(args, "n_coeffs", _features.DEFAULT_N_COEFFS)
            res = _app.compute_fingerprint(input_text, n_coeffs=n_coeffs)
            data = res["data"]
            feats = data["features"]
            latex = r"\varphi\!\left(\left[%s\right]_q\right) \in \mathbb{R}^{%d}" % (
                _sym(input_text),
                len(data["values"]),
            )
            rows = [[name, str(feats[name])] for name in data["names"][:12]]
            return {
                "latex": latex,
                "text": ", ".join(str(v) for v in data["values"]),
                "rows": rows,
                "meta": {"op": op, "n_features": len(data["values"])},
            }

        if op == "factor":
            from .factor import denominator_expr, factor_qreal, numerator_expr

            result = factor_qreal(input_text)
            R = numerator_expr(result)
            S = denominator_expr(result)
            latex = r"\left[\tfrac{%d}{%d}\right]_q = \dfrac{%s}{%s}" % (
                result.a,
                result.b,
                sp.latex(R),
                sp.latex(S),
            )
            cyclo = (
                ", ".join(
                    f"Phi_{d}^{e}" for d, e in sorted(result.cyclotomic_R.items())
                )
                or "none"
            )
            return {
                "latex": latex,
                "text": f"R(q) = {sp.sstr(R)}    S(q) = {sp.sstr(S)}",
                "rows": [
                    ["R irreducible over Z[q]?", "yes" if result.is_irreducible_R else "no"],
                    [
                        "R a pure product of cyclotomics?",
                        "yes" if result.is_pure_cyclotomic_R else "no",
                    ],
                    ["cyclotomic support of R", cyclo],
                ],
                "meta": {
                    "op": op,
                    "a": result.a,
                    "b": result.b,
                    "is_irreducible_R": result.is_irreducible_R,
                    "is_pure_cyclotomic_R": result.is_pure_cyclotomic_R,
                },
            }

        if op == "roots":
            from .factor import (
                classify_roots,
                denominator_expr,
                factor_qreal,
                numerator_expr,
            )
            from .rational import q as _q

            result = factor_qreal(input_text)
            # The plotted roots are those of the normalised R (R(0) = 1); the
            # q^k valuation prefix is a shift, not a cyclotomic/core root, so it
            # is shown as a note rather than a point at the origin.
            R = sp.expand(numerator_expr(result) / _q**result.k)
            S = denominator_expr(result)
            plot = classify_roots(result)
            # The coefficients of the normalised R and of S (ascending powers of
            # q) let the 3D view evaluate |R(q)/S(q)| on a grid in the browser;
            # both have integer coefficients, so they round-trip exactly as ints.
            R_poly = sp.Poly(R, _q)
            R_coeffs = [int(c) for c in reversed(R_poly.all_coeffs())]
            S_poly = sp.Poly(S, _q)
            S_coeffs = [int(c) for c in reversed(S_poly.all_coeffs())]
            # Poles of [a/b]_q are the roots of S(q); the nearest one to the
            # origin is the radius of convergence of the power series [a/b]_q.
            poles = []
            for root in (S_poly.nroots() if S_poly.degree() >= 1 else []):
                c = complex(root)
                poles.append(
                    {"re": float(c.real), "im": float(c.imag), "mod": abs(c)}
                )
            radius = min((p["mod"] for p in poles), default=None)
            latex = r"R(q) = %s" % sp.latex(R)
            cyclo = (
                ", ".join(
                    f"Phi_{d}^{e}" for d, e in sorted(result.cyclotomic_R.items())
                )
                or "none"
            )
            n_cyc = sum(1 for r in plot["roots"] if r["kind"] == "cyclotomic")
            n_core = sum(1 for r in plot["roots"] if r["kind"] == "core")
            rows = [
                ["degree of R(q)", str(plot["degree"])],
                ["cyclotomic support {d: e_d}", cyclo],
                ["roots on the unit circle (cyclotomic)", str(n_cyc)],
                ["non-cyclotomic core degree", str(plot["core_degree"])],
                ["roots off the unit circle (core)", str(n_core)],
            ]
            if result.k:
                rows.append(["q^k prefix split off (k)", str(result.k)])
            if radius is not None:
                rows.append(
                    [
                        "radius of convergence (nearest pole |q|)",
                        "%.6f" % radius,
                    ]
                )
            return {
                "latex": latex,
                "text": f"R(q) = {sp.sstr(R)}",
                "rows": rows,
                "meta": {
                    "op": op,
                    "a": result.a,
                    "b": result.b,
                    "roots": plot["roots"],
                    "coeffs": R_coeffs,
                    "s_coeffs": S_coeffs,
                    "poles": poles,
                    "radius": radius,
                    "cyclotomic_support": plot["cyclotomic_support"],
                    "core_degree": plot["core_degree"],
                    "degree": plot["degree"],
                },
            }

        if op == "coeff-surface":
            from . import q_real_truncated

            N = max(2, min(_int_arg(args, "n", 14), 40))
            M = 25  # number of x samples across the band

            def _real_field(text: str, fallback: str) -> sp.Expr:
                try:
                    return sp.nsimplify(parse_real(text))
                except Exception:  # noqa: BLE001 - fall back to the default band
                    return sp.sympify(fallback)

            x_lo = _real_field(input_text, "1")
            x_hi = _real_field(args.get("x_max"), "2")
            if x_hi == x_lo:
                x_hi = x_lo + 1
            if x_hi < x_lo:
                x_lo, x_hi = x_hi, x_lo
            xs: list[float] = []
            Z: list[list[int]] = []
            for j in range(M):
                xj = x_lo + (x_hi - x_lo) * sp.Rational(j, M - 1)
                coeffs = q_real_truncated(str(xj), N)
                xs.append(float(xj))
                Z.append([int(c) for c in coeffs])
            latex = (
                r"c_n\!\left(\left[x\right]_q\right)\ \text{for }"
                r"x\in\left[%s,\,%s\right],\ n=0,\dots,%d"
                % (sp.latex(x_lo), sp.latex(x_hi), N - 1)
            )
            return {
                "latex": latex,
                "text": (
                    f"coefficient surface c_n of [x]_q for x in "
                    f"[{float(x_lo):g}, {float(x_hi):g}] ({M} samples), n = 0..{N - 1}"
                ),
                "rows": [
                    ["x band", f"[{float(x_lo):g}, {float(x_hi):g}]"],
                    ["x samples", str(M)],
                    ["coefficients per x (N)", str(N)],
                ],
                "meta": {
                    "op": op,
                    "plot3d": {"kind": "coeff-surface", "n": list(range(N)),
                               "x": xs, "z": Z},
                },
            }

        if op == "root-sweep":
            import math

            from .factor import classify_roots, factor_qreal

            a = int(str(input_text).strip())
            if a == 0:
                return {"error": "the numerator a must be non-zero"}
            b_max = max(1, min(_int_arg(args, "b_max", 12), 40))
            points: list[dict[str, Any]] = []
            used_b: list[int] = []
            for b in range(1, b_max + 1):
                if math.gcd(abs(a), b) != 1:
                    continue  # a/b would not reduce to numerator a
                used_b.append(b)
                plot = classify_roots(factor_qreal((a, b)))
                for r in plot["roots"]:
                    points.append({
                        "re": float(r["re"]), "im": float(r["im"]),
                        "b": b, "kind": r["kind"], "d": r["d"],
                    })
            latex = (
                r"\{\,q:R(q)=0\,\}\ \text{of}\ \left[\tfrac{%d}{b}\right]_q,"
                r"\ b=1,\dots,%d" % (a, b_max)
            )
            return {
                "latex": latex,
                "text": (
                    f"roots of R(q) for [{a}/b]_q over b in "
                    f"{{{', '.join(str(x) for x in used_b)}}} "
                    f"({len(points)} roots total)"
                ),
                "rows": [
                    ["numerator a", str(a)],
                    ["denominators swept", str(len(used_b))],
                    ["roots plotted", str(len(points))],
                ],
                "meta": {
                    "op": op, "a": a,
                    "plot3d": {"kind": "root-sweep", "points": points,
                               "b_max": b_max},
                },
            }

        if op == "radius-grid":
            import math

            from .rational import q as _q
            from .factor import denominator_expr, factor_qreal

            b_max = max(1, min(int(str(input_text).strip()), 16))
            a_max = max(1, min(_int_arg(args, "a_max", 12), 40))
            points = []
            omitted = 0
            for b in range(1, b_max + 1):
                for a in range(1, a_max + 1):
                    if math.gcd(a, b) != 1:
                        continue
                    # the radius is the modulus of the nearest pole, i.e. the
                    # smallest-modulus root of S(q); compute it and its location
                    # once here, the same pole math the roots view uses
                    s_poly = sp.Poly(denominator_expr(factor_qreal((a, b))), _q)
                    if s_poly.degree() < 1:
                        # integers (b = 1) and other pole-free values have an
                        # infinite radius; they carry no height, so leave them out
                        omitted += 1
                        continue
                    nearest = min(
                        (complex(root) for root in s_poly.nroots()), key=abs,
                    )
                    points.append({
                        "a": a, "b": b, "val": a / b, "r": abs(nearest),
                        "pole_re": float(nearest.real),
                        "pole_im": float(nearest.imag),
                    })
            below = sum(1 for pt in points if pt["r"] < 1.0)
            latex = (
                r"\rho\!\left(\left[\tfrac{a}{b}\right]_q\right)"
                r"\ \text{for }a\le %d,\ b\le %d" % (a_max, b_max)
            )
            return {
                "latex": latex,
                "text": (
                    f"radius of convergence of [a/b]_q over the Farey grid "
                    f"a<= {a_max}, b<= {b_max} ({len(points)} fractions, "
                    f"{omitted} pole-free omitted, {below} with radius < 1)"
                ),
                "rows": [
                    ["max denominator b", str(b_max)],
                    ["max numerator a", str(a_max)],
                    ["fractions plotted", str(len(points))],
                    ["radius < 1 (pole left the unit circle)", str(below)],
                    ["pole-free (infinite radius) omitted", str(omitted)],
                ],
                "meta": {
                    "op": op,
                    "plot3d": {"kind": "radius-grid", "points": points,
                               "a_max": a_max, "b_max": b_max},
                },
            }

        if op == "certificate":
            res = _app.compute_coeffs(input_text, n)
            coeffs = res["data"]["coefficients"]
            entry = _laurent_entry(input_text, coeffs)
            latex = r"\left[%s\right]_q = %s" % (
                _sym(input_text),
                _latex_for_coeffs(coeffs),
            )
            return {
                "latex": latex,
                "text": exports.to_latex(entry),
                "rows": [
                    [
                        "LaTeX table",
                        "the plain-text box is a booktabs table; paste it into a "
                        "paper with \\usepackage{booktabs}",
                    ]
                ],
                "meta": {"op": op, "n": n},
            }

        return {"error": f"unknown operation {op!r}"}
    except Exception as exc:  # noqa: BLE001 - report the error to the page
        return {"error": str(exc)}


def export_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Turn a {title, items:[{op,input,args,note}]} bundle into a standalone .tex."""
    from . import exports
    sections = []
    for it in bundle.get("items", []):
        res = compute(str(it.get("op", "")), str(it.get("input", "")), it.get("args") or {})
        if "error" in res:
            continue
        body = r"\[" + res["latex"] + r"\]"
        rows = "".join(r"\item %s: %s" % (k, v) for k, v in (res.get("rows") or []))
        note = (r"\par\textit{%s}" % it["note"]) if it.get("note") else ""
        sections.append(r"\section*{%s}%s\begin{itemize}%s\end{itemize}%s" %
                        (it.get("input", ""), body, rows or r"\item ~", note))
    table = "\n".join(sections) or "(empty)"
    return {"tex": exports.latex_document(table), "title": bundle.get("title", "qreals")}


def _app_result_for(op: str, input_text: str, args: dict[str, Any]) -> dict[str, Any]:
    """The raw app.py Result dict for an op, for certificate building. Raises for
    ops without a clean MGO derivation (plots, exact-rational, etc.)."""
    n = _int_arg(args, "n", OPERATIONS.get(op, {}).get("default_n", 12))
    if op == "rational":
        p, s = _app._parse_rational(input_text)
        return _app.compute_rational(p, s)
    if op == "qint":
        return _app.compute_qint(int(str(input_text).strip()))
    if op == "coefficients":
        return _app.compute_coeffs(input_text, n)
    if op == "laurent":
        return _app.compute_laurent(input_text, _int_arg(args, "order", 12))
    if op == "prefix":
        return _app.compute_prefix(input_text)
    if op == "locked":
        return _app.compute_locked(input_text, n)
    if op == "shift":
        direction = "down" if str(args.get("direction", "up")) == "down" else "up"
        return _app.compute_shift(input_text, _int_arg(args, "order", 12), direction)
    raise ValueError(f"no derivation for op '{op}'")


def compute_certificate(
    op: str, input_text: str, args: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A referee-checkable derivation for a result, via certificate.build_certificate."""
    from . import certificate as _cert
    from . import provenance as _prov

    import re as _re

    def _clean(s: str) -> str:
        # turn the [[cite:key|text]] placeholders into their human text
        return _re.sub(r"\[\[cite:[^|\]]*\|([^\]]*)\]\]", r"\1", str(s or "")).strip()

    args = args or {}
    try:
        result = _app_result_for(op, input_text, args)
        cert = _cert.build_certificate(result)
        v = cert.referee
        # The fold recursion as proper display LaTeX (MathJax renders it like the
        # main result), mirroring certificate._referee_tex — not the terminal ascii.
        recursion_tex = ""
        if v.folds:
            m = len(cert.even_cf)
            recursion_tex = (
                r"\begin{aligned}"
                + (r"R_{%d} &= [a_{%d}]_{q^{-1}}, \\ " % (m, m))
                + r"R_i &= [a_i]_q + q^{a_i}/R_{i+1} && (i\ \text{odd}), \\ "
                + r"R_i &= [a_i]_{q^{-1}} + q^{-a_i}/R_{i+1} && (i\ \text{even})"
                + r"\end{aligned}"
            )
        return {
            "title": _clean(cert.title),
            "headline": _clean(v.headline),
            "recursionTex": recursion_tex,
            "gaussNote": r"[a]_q = \dfrac{1-q^a}{1-q}" if v.folds else "",
            "structure": [_clean(s) for s in v.structure],
            "folds": [
                {"pos": fr.pos, "a": fr.a, "ratio": fr.ratio, "degree": fr.degree}
                for fr in v.folds
            ],
            "witness": _clean(v.witness) if v.witness else None,
            "citations": cert.citation_keys(),
            "provenance_available": _prov.qprov_available(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def preview(op: str, input_text: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """LaTeX for the parsed input of an operation, for the live preview.

    This parses with the same helpers the engine uses, but does no computation:
    it shows what the input will be read as (for example pi as the symbol pi, or
    3/5 as a fraction). Returns {latex} ("" when the input does not yet parse).
    """
    args = args or {}
    meta = OPERATIONS.get(op, {})
    kind = meta.get("input_kind", _REAL)
    try:
        if kind == _RATIONAL:
            p, s = _app._parse_rational(input_text)
            base = r"\left[\tfrac{%d}{%d}\right]_q" % (p, s)
            if op == "jump-gap":
                base = r"\left[\tfrac{%d}{%d}\right]_q^{+} - \left[\tfrac{%d}{%d}\right]_q^{-}" % (
                    p,
                    s,
                    p,
                    s,
                )
            elif op == "factor":
                base = r"R(q),\ S(q)\ \text{of}\ \left[\tfrac{%d}{%d}\right]_q" % (p, s)
            elif op == "roots":
                base = r"\{\,q : R(q)=0\,\}\ \text{of}\ \left[\tfrac{%d}{%d}\right]_q" % (p, s)
            elif op == "frieze":
                base = r"\text{frieze of}\ \left[\tfrac{%d}{%d}\right]_q" % (p, s)
            return {"latex": base}

        if kind == _INTEGER:
            nn = int(str(input_text).strip())
            return {"latex": r"[%d]_q" % nn}

        if kind == _SEQUENCE:
            from . import oeis as _oeis

            seq = _oeis.parse_sequence(input_text)
            shown = ", ".join(str(t) for t in seq[:12])
            if len(seq) > 12:
                shown += r", \dots"
            return {"latex": r"(%s)" % shown}

        # real-valued input (possibly with a second value y)
        xl = sp.latex(_app._parse_real(input_text))
        if op in ("q-sum", "q-product", "deficit"):
            yl = sp.latex(_app._parse_real(str(args.get("y") or "")))
            if op == "deficit":
                sym = r"\cdot" if str(args.get("op", "add")) == "mul" else "+"
                base = r"\left[%s %s %s\right]_q - \left(\left[%s\right]_q %s \left[%s\right]_q\right)" % (
                    xl,
                    sym,
                    yl,
                    xl,
                    sym,
                    yl,
                )
            else:
                sym = r"\cdot" if op == "q-product" else "+"
                base = r"\left[%s\right]_q %s \left[%s\right]_q" % (xl, sym, yl)
            return {"latex": base}
        if op == "quad-arith":
            yl = sp.latex(_app._parse_real(str(args.get("y") or "")))
            sym = {"add": "+", "sub": "-", "mul": r"\cdot", "div": "/"}.get(
                str(args.get("op", "add")), "+"
            )
            return {"latex": r"\left(%s\right) %s \left(%s\right)" % (xl, sym, yl)}
        if op == "negation":
            return {"latex": r"\left[-%s\right]_q" % xl}
        if op == "finiteness":
            return {"latex": r"\left[%s\right]_q + \left[-%s\right]_q" % (xl, xl)}
        if op == "shift":
            sign = "-" if str(args.get("direction", "up")) == "down" else "+"
            return {"latex": r"\left[%s %s 1\right]_q" % (xl, sign)}
        return {"latex": r"\left[%s\right]_q" % xl}
    except Exception:  # noqa: BLE001 - an unfinished input simply has no preview
        return {"latex": ""}


# --------------------------------------------------------------------------
# The page.
# --------------------------------------------------------------------------


import importlib.resources as _res


def _asset(name: str) -> str:
    """Read a frontend asset shipped under qreals/web/."""
    return (_res.files("qreals.web") / name).read_text(encoding="utf-8")


def _index_html() -> str:
    """Compose the single-page app from the carved-out web assets.

    The JS carries the OPERATIONS registry and group order via the __OPS__ and
    __GROUPS__ tokens; CSS and JS are injected into the template. Nothing here
    recomputes math.
    """
    css = _asset("app.css")
    js = (
        _asset("app.js")
        .replace("__OPS__", json.dumps(OPERATIONS))
        .replace("__GROUPS__", json.dumps(GROUP_ORDER))
    )
    return _asset("template.html").replace("__CSS__", css).replace("__JS__", js)


# --------------------------------------------------------------------------
# App construction. FastAPI preferred, Flask fallback.
# --------------------------------------------------------------------------


def _have(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def build_app() -> Any:
    """Build and return the web application (FastAPI app, else Flask app).

    Raises ImportError with an install hint when neither framework is present.
    """
    if _have("fastapi"):
        return _build_fastapi_app()
    if _have("flask"):
        return _build_flask_app()
    raise ImportError(
        "qreals serve needs a web framework. Install the extra with:  "
        "pip install qreals[serve]"
    )


def _build_fastapi_app() -> Any:
    from fastapi import FastAPI
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse

    application = FastAPI(title="qreals", docs_url=None, redoc_url=None)

    # Plain Starlette handlers registered with add_route, so FastAPI never
    # inspects their signatures for query/body parameters. This sidesteps the
    # interaction between `from __future__ import annotations` and FastAPI's
    # parameter detection, which otherwise mistakes the Request argument for a
    # query field. The JSON body shape is free-form {op, input, args}.
    async def index(_request: Request) -> Any:
        return HTMLResponse(_index_html())

    async def _payload(request: Request) -> dict[str, Any]:
        raw = await request.body()
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            return {}

    async def compute_endpoint(request: Request) -> Any:
        payload = await _payload(request)
        result = compute(
            str(payload.get("op", "")),
            str(payload.get("input", "")),
            payload.get("args") or {},
        )
        return JSONResponse(result)

    async def preview_endpoint(request: Request) -> Any:
        payload = await _payload(request)
        result = preview(
            str(payload.get("op", "")),
            str(payload.get("input", "")),
            payload.get("args") or {},
        )
        return JSONResponse(result)

    async def certificate_endpoint(request: Request) -> Any:
        payload = await _payload(request)
        result = compute_certificate(
            str(payload.get("op", "")),
            str(payload.get("input", "")),
            payload.get("args") or {},
        )
        return JSONResponse(result)

    async def export_endpoint(request: Request) -> Any:
        payload = await _payload(request)
        return JSONResponse(export_bundle(payload))

    application.add_route("/", index, methods=["GET"])
    application.add_route("/compute", compute_endpoint, methods=["POST"])
    application.add_route("/preview", preview_endpoint, methods=["POST"])
    application.add_route("/certificate", certificate_endpoint, methods=["POST"])
    application.add_route("/export", export_endpoint, methods=["POST"])
    return application


def _build_flask_app() -> Any:
    from flask import Flask, jsonify, request

    application = Flask(__name__)

    @application.get("/")
    def index() -> Any:
        return _index_html()

    @application.post("/compute")
    def compute_endpoint() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        result = compute(
            str(payload.get("op", "")),
            str(payload.get("input", "")),
            payload.get("args") or {},
        )
        return jsonify(result)

    @application.post("/preview")
    def preview_endpoint() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        result = preview(
            str(payload.get("op", "")),
            str(payload.get("input", "")),
            payload.get("args") or {},
        )
        return jsonify(result)

    @application.post("/certificate")
    def certificate_endpoint() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        result = compute_certificate(
            str(payload.get("op", "")),
            str(payload.get("input", "")),
            payload.get("args") or {},
        )
        return jsonify(result)

    @application.post("/export")
    def export_endpoint() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        return jsonify(export_bundle(payload))

    return application


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, preferred: int, tries: int = 20) -> int:
    """Return the preferred port if free, else the next free port after it."""
    for offset in range(tries):
        candidate = preferred + offset
        if _port_is_free(host, candidate):
            return candidate
    raise OSError(
        f"no free port found in {preferred}..{preferred + tries - 1} on {host}"
    )


def serve(port: int = 8000, open_browser: bool = True) -> int:
    """Start the local server on 127.0.0.1, picking a free port from `port`.

    Opens the browser to the served URL unless open_browser is False. Returns a
    process exit code (0 on a clean shutdown). Needs the serve extra.
    """
    host = "127.0.0.1"
    chosen = _pick_port(host, port)
    url = f"http://{host}:{chosen}"
    application = build_app()

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    print(f"qreals serve running at {url}  (Ctrl+C to stop)")

    if _have("fastapi"):
        import uvicorn

        uvicorn.run(application, host=host, port=chosen, log_level="warning")
    else:  # Flask fallback
        application.run(host=host, port=chosen)
    return 0
