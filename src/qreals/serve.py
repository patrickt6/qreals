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

from . import __version__
from . import app as _app
from . import exports
from . import features as _features
from . import format_laurent
from . import formatter as fmt
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
    "s-properties": {
        "name": "Denominator S(q) properties",
        "symbol": "S(q) of [a/d]_q",
        "group": "q-rationals",
        "blurb": "The cyclotomic structure of the denominator S(q): its factors "
        "Phi_k, the index set T and saturation index e* = lcm(T) (the minimal n "
        "with S | [n]_q), deg S against the bound d-1, the S(1)=d and S(0)=1 "
        "invariants, and whether S is the full [d]_q, a proper collapse, or the "
        "impossibility branch (divides no [n]_q).",
        "input_kind": _RATIONAL,
        "fields": [_f("input", "rational a/d", "5/12")],
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
    "s-atlas": {
        "name": "S(q) cyclotomic atlas",
        "symbol": "regime of S(a/d)",
        "group": "Visuals",
        "blurb": "A coloured (a, d) grid of the denominator S(q)'s regime: full "
        "[d]_q, a proper cyclotomic collapse, non-squarefree, or non-cyclotomic, "
        "with the index set T on hover and a Phi_k appearance tally.",
        "input_kind": _INTEGER,
        "fields": [
            _f("input", "max denominator d", "12", "int"),
            _f("a_max", "max numerator a (0 = no cap)", "0", "int"),
        ],
    },
    "saturation-explorer": {
        "name": "Saturation index e*",
        "symbol": "e*(a/d) vs a",
        "group": "Visuals",
        "blurb": "For a fixed d, the saturation index e* = lcm(T) and the minimal "
        "saturating n as a ranges over the residues coprime to d, with the "
        "impossibility residues (no finite n) marked.",
        "input_kind": _INTEGER,
        "fields": [_f("input", "denominator d", "12", "int")],
    },
    "degree-collapse": {
        "name": "deg S vs d-1 collapse map",
        "symbol": "deg S vs d-1",
        "group": "Visuals",
        "blurb": "deg S plotted against the bound d-1 over the coprime grid. The "
        "diagonal is the a == +/-1 saturating locus (S = [d]_q); the drop below "
        "it is the collapse depth, the totient weight of the dropped Phi_k.",
        "input_kind": _INTEGER,
        "fields": [
            _f("input", "max denominator d", "12", "int"),
            _f("a_max", "max numerator a (0 = no cap)", "0", "int"),
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
    "s-properties": r"S(q)\ \text{of}\ \left[\tfrac{a}{d}\right]_q",
    "s-atlas": r"S\!\left(\tfrac{a}{d}\right)\ \text{regime}",
    "saturation-explorer": r"e^{\star}\!\left(\tfrac{a}{d}\right)",
    "degree-collapse": r"\deg S\ \text{vs}\ d-1",
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


# A longer "What is this?" description for each operation, shown under the blurb
# on the per-operation page. These carry inline LaTeX (\( ... \)) typeset by
# MathJax, so a newcomer can read what the tool computes and why it matters, not
# just its one-line card summary.
_ABOUT: dict[str, str] = {
    "rational": (
        r"The \(q\)-rational \(\left[\tfrac{p}{s}\right]_q\) is the "
        r"Morier-Genoud and Ovsienko \(q\)-deformation of the fraction "
        r"\(\tfrac{p}{s}\): a rational function \(N(q)/S(q)\) in the formal "
        r"variable \(q\) that returns the ordinary value \(\tfrac{p}{s}\) when "
        r"\(q=1\). It is built from the (negative) continued fraction of "
        r"\(\tfrac{p}{s}\) by replacing each integer \(c\) with the "
        r"\(q\)-integer \([c]_q\)."
    ),
    "qint": (
        r"The \(q\)-integer is "
        r"\([n]_q = 1+q+q^2+\cdots+q^{\,n-1} = \dfrac{1-q^{\,n}}{1-q}\), the "
        r"q-analog of a whole number \(n\). At \(q=1\) it collapses to \(n\). It "
        r"factors over \(\mathbb{Z}[q]\) as the product of the cyclotomic "
        r"polynomials \(\Phi_k(q)\) over the divisors \(k\mid n\) with \(k\ge2\)."
    ),
    "factor": (
        r"Writes \(\left[\tfrac{a}{b}\right]_q = q^{k}\,R(q)/S(q)\) with "
        r"\(R(0)=1\) and factors both the numerator \(R(q)\) and denominator "
        r"\(S(q)\) into irreducibles over \(\mathbb{Z}[q]\), labelling each "
        r"factor as a cyclotomic polynomial \(\Phi_d(q)\) (a root of unity) or a "
        r"non-cyclotomic ``core'' factor. \(R\) is often irreducible when "
        r"\(a\) is prime."
    ),
    "s-properties": (
        r"Analyzes the denominator \(S(q)\) of "
        r"\(\left[\tfrac{a}{d}\right]_q\). \(S\) is a monic polynomial with "
        r"\(S(0)=1\) and \(S(1)=d\); when it is a squarefree product of "
        r"cyclotomics \(S=\prod_{k\in T}\Phi_k\) it divides \([n]_q\) exactly "
        r"when the saturation index \(e^\star=\operatorname{lcm}(T)\) divides "
        r"\(n\). Its degree obeys \(\deg S\le d-1\), with equality iff "
        r"\(S=[d]_q\) iff \(a\equiv\pm1\pmod d\). Click any property below for a "
        r"full explanation."
    ),
    "roots": (
        r"Plots the complex roots of the numerator \(R(q)\) (the zeros of "
        r"\(\left[\tfrac{a}{b}\right]_q\)) and overlays the roots of the "
        r"denominator \(S(q)\) (its poles). Cyclotomic factors give roots on the "
        r"unit circle \(|q|=1\); a non-cyclotomic factor can sit off it. The "
        r"pole nearest the origin sets the radius of convergence \(\rho\) of the "
        r"power series \(\left[\tfrac{a}{b}\right]_q\)."
    ),
    "jump-gap": (
        r"A rational \(p/s\) has two \(q\)-versions: a right value "
        r"\(\left[\tfrac{p}{s}\right]_q^{+}\) (the limit from above) and a left "
        r"value \(\left[\tfrac{p}{s}\right]_q^{-}\) (from below). Their gap is "
        r"the single rational function "
        r"\(\left[\tfrac{p}{s}\right]_q^{+}-\left[\tfrac{p}{s}\right]_q^{-} "
        r"= (1-q)\,q^{E}/(S^{+}S^{-})\) (Jouteur, arXiv:2503.02122)."
    ),
    "exact-rational": (
        r"The exact \(\left[x\right]_q = P(q)/Q(q)\) for a \(q\)-rational "
        r"\(x\), factored. Give a second rational \(y\) for the exact difference "
        r"\(\left[x\right]_q-\left[y\right]_q = "
        r"\dfrac{P_xQ_y-P_yQ_x}{Q_xQ_y}\), with the divisibilities "
        r"\(Q_x\mid Q_y\) that force the denominators to agree up to a unit."
    ),
    "s-atlas": (
        r"A coloured \((a,d)\) grid of the regime of the denominator "
        r"\(S(a/d)\): the full \([d]_q\), a proper cyclotomic collapse, "
        r"non-squarefree, or non-cyclotomic. The full \([d]_q\) cells trace the "
        r"\(a\equiv\pm1\pmod d\) diagonals; the two impossibility regimes "
        r"(which divide no \([n]_q\)) stand out. A companion view tallies how "
        r"often each \(\Phi_k\) appears as \(d\) grows."
    ),
    "saturation-explorer": (
        r"For a fixed denominator \(d\), the saturation index "
        r"\(e^\star(a/d)=\operatorname{lcm}(T)\) and the minimal \(n\) with "
        r"\(S\mid[n]_q\), as \(a\) ranges over the residues coprime to \(d\). "
        r"\(e^\star\) is the least \(n\) that makes the difference of two "
        r"equal-tail \(q\)-rationals a finite Laurent polynomial; the "
        r"impossibility residues (no finite \(n\)) are marked on the floor."
    ),
    "degree-collapse": (
        r"Plots \(\deg S\) against the bound \(d-1\) over the coprime grid. "
        r"The dashed diagonal \(\deg S=d-1\) is the saturating "
        r"\(a\equiv\pm1\pmod d\) locus (\(S=[d]_q\)); the drop below it is the "
        r"collapse depth, which for a squarefree \(S\) equals the totient weight "
        r"\(\sum_{k\,\text{dropped}}\varphi(k)\) of the cancelled cyclotomic "
        r"factors."
    ),
    "coefficients": (
        r"The first \(N\) Taylor coefficients \(c_0,c_1,\dots\) of "
        r"\(\left[x\right]_q=\sum_k c_k q^{k}\) for any real \(x\). These "
        r"stabilise as the continued fraction of \(x\) is refined, so the "
        r"leading coefficients are exact."
    ),
    "laurent": (
        r"The MGO Laurent expansion of \(\left[x\right]_q\) through "
        r"\(q^{\text{order}}\), together with its integer-part prefix: the first "
        r"\(\lfloor x\rfloor\) coefficients are all \(1\) and the coefficient at "
        r"\(q^{\lfloor x\rfloor}\) is forced to \(0\)."
    ),
    "prefix": (
        r"The forced opening block of \(\left[x\right]_q\): the first "
        r"\(\lfloor x\rfloor\) coefficients equal \(1\) and the next is \(0\), "
        r"so \(\left[x\right]_q\) begins like \([\lfloor x\rfloor]_q\) "
        r"regardless of the fractional part."
    ),
    "locked": (
        r"How many Laurent coefficients of \(\left[x\right]_q\) are pinned down "
        r"by the \(n\)-th convergent of \(x\): the partial sum \(S_n\) of the "
        r"continued-fraction terms fixes the coefficients \(q^0\) through "
        r"\(q^{\,S_n-2}\)."
    ),
    "shift": (
        r"Moves the argument by one using the exact relations "
        r"\(\left[x+1\right]_q = q\left[x\right]_q + 1\) and "
        r"\(\left[x-1\right]_q = (\left[x\right]_q-1)/q\)."
    ),
    "readouts": (
        r"Pattern read-outs over the first \(N\) coefficients of "
        r"\(\left[x\right]_q\): the first nonzero index, the first negative "
        r"index, the largest absolute coefficient, and the number of zeros. "
        r"Useful for spotting zero-runs and sign flips."
    ),
    "radius": (
        r"A running-max root-test estimate of the radius of convergence "
        r"\(\rho\!\left(\left[x\right]_q\right)=\exp\!\big(-\max_k "
        r"\tfrac{\ln|c_k|}{k}\big)\). It is biased high at finite \(N\) and "
        r"decreases toward the true radius as \(N\) grows. For a rational "
        r"\(x\), \(\rho\) is the modulus of the nearest pole of \(S(q)\)."
    ),
    "fingerprint": (
        r"A named, fixed-length, deterministic feature vector "
        r"\(\varphi\!\left(\left[x\right]_q\right)\in\mathbb{R}^{m}\) of a "
        r"constant, combining continued-fraction terms, coefficients, and a "
        r"radius estimate, for nearest-neighbour comparison between constants."
    ),
    "certificate": (
        r"The coefficients of \(\left[x\right]_q\) with a ready-to-paste LaTeX "
        r"<code>booktabs</code> table and a referee-checkable derivation, so a "
        r"result can be dropped straight into a paper."
    ),
    "q-sum": (
        r"The series sum \(\left[x\right]_q+\left[y\right]_q\) of the two "
        r"\(q\)-series, coefficient by coefficient. Note this is not "
        r"\(\left[x+y\right]_q\): the map \(x\mapsto\left[x\right]_q\) is not a "
        r"ring homomorphism."
    ),
    "q-product": (
        r"The series product \(\left[x\right]_q\cdot\left[y\right]_q\), "
        r"coefficient by coefficient. As with the sum, this is not "
        r"\(\left[xy\right]_q\); the deficit measures the gap."
    ),
    "deficit": (
        r"The deficit \(\left[x\circ y\right]_q-\left(\left[x\right]_q\circ"
        r"\left[y\right]_q\right)\) for \(\circ\in\{+,\cdot\}\): how far the "
        r"series sum or product sits from the \(q\)-analog of the real "
        r"\(x\circ y\). At \(q=1\) both sides agree so the deficit is \(0\); the "
        r"\(q=0\) value is a structural check."
    ),
    "quad-arith": (
        r"Exact \(x\circ y\) for two quadratic irrationals, read off from the "
        r"dominant (Perron) eigenvector of the continued-fraction transfer "
        r"matrix \(K=M_x\otimes M_y\). Reproduces the golden-plus-silver worked "
        r"example and cross-checks against direct arithmetic."
    ),
    "negation": (
        r"The Jouteur \(q\)-negation \(\left[-x\right]_q\), an involution from "
        r"the \(\mathrm{PGL}_2(\mathbb{Z})\) action (arXiv:2503.02122), and the "
        r"sum \(\left[x\right]_q+\left[-x\right]_q\) with its finiteness verdict."
    ),
    "finiteness": (
        r"Tests whether \(\left[x\right]_q+\left[-x\right]_q\) is a finite "
        r"Laurent polynomial. It is finite exactly for pure square roots "
        r"(trace-zero quadratics), where \(-x\) is the Galois conjugate and the "
        r"sum is the \(q\)-trace (Ovsienko, Example 6.4)."
    ),
    "frieze": (
        r"The Conway-Coxeter frieze of \(\tfrac{a}{b}>1\) with the "
        r"\(q\)-coefficient polynomial overlaid on every cell. Each diamond "
        r"satisfies the unimodular rule \(ad-bc=1\); at \(q=1\) it becomes the "
        r"classical integer frieze."
    ),
    "coeff-surface": (
        r"A 3D surface of the Taylor coefficients \(c_n\!\left("
        r"\left[x\right]_q\right)\) as both \(n\) and \(x\) vary: zero-runs are "
        r"valleys and sign flips cross the floor."
    ),
    "root-sweep": (
        r"Stacks the complex roots of \(R(q)\) for "
        r"\(\left[\tfrac{a}{b}\right]_q\) as the denominator \(b\) sweeps: "
        r"cyclotomic roots stay pinned on the unit circle while the core drifts."
    ),
    "radius-grid": (
        r"The radius of convergence "
        r"\(\rho\!\left(\left[\tfrac{a}{b}\right]_q\right)\) (the nearest pole "
        r"\(|q|\)) over a Farey grid, drawn as a number line, Ford circles, or "
        r"an \((a,b)\) grid."
    ),
    "oeis": (
        r"Looks a coefficient sequence up in the Online Encyclopedia of Integer "
        r"Sequences, re-verifying the top hits against the full b-file and "
        r"checking small mod-\(p\) reductions."
    ),
}

# Click-to-open explanations for each property row of the S(q) properties panel.
# Keyed by the exact row label the s-properties branch emits; the values carry
# inline LaTeX and a list of the values the property can take and what they mean.
_SPROPS_GLOSSARY: dict[str, dict[str, Any]] = {
    "kind of S": {
        "title": "Kind of S (the regime)",
        "tex": (
            r"Which of four shapes the denominator \(S(q)\) takes. The first two "
            r"are the saturating regimes (a finite saturation index exists); the "
            r"last two are the impossibility branch, where \(S\) divides no "
            r"\([n]_q\), so the difference of two equal-tail \(q\)-rationals over "
            r"\(S\) is never a finite Laurent polynomial."
        ),
        "values": [
            r"<b>full \([d]_q\)</b>: \(S=[d]_q\), the whole \(q\)-integer "
            r"(no collapse), occurring exactly when \(a\equiv\pm1\pmod d\);",
            r"<b>proper collapse</b>: a strict squarefree-cyclotomic "
            r"subproduct of \([d]_q\) (e.g. \(5/12\) gives "
            r"\(\Phi_2\Phi_3\Phi_4\)); still divides some \([n]_q\);",
            r"<b>non-squarefree</b>: a repeated cyclotomic factor (e.g. "
            r"\(3/8\) gives \(\Phi_2^{2}\Phi_4\)); divides no \([n]_q\);",
            r"<b>non-cyclotomic</b>: carries a non-cyclotomic core factor "
            r"(e.g. \(2/15\)); divides no \([n]_q\).",
        ],
    },
    "cyclotomic index set T": {
        "title": "Cyclotomic index set T",
        "tex": (
            r"The set \(T\) of indices \(k\) for which the cyclotomic polynomial "
            r"\(\Phi_k(q)\) divides \(S\), so (in the squarefree case) "
            r"\(S=\prod_{k\in T}\Phi_k\). Since "
            r"\([n]_q=\prod_{k\mid n,\,k\ge2}\Phi_k\) is squarefree, every "
            r"admissible \(S\) is a subset product of these factors (Remark 2)."
        ),
        "values": [
            r"a finite set of integers \(k\ge2\), e.g. \(T=\{2,3,4\}\) for "
            r"\(5/12\);",
            r"<b>empty</b>: \(S\) has no cyclotomic factor at all (a purely "
            r"non-cyclotomic denominator).",
        ],
    },
    "saturation index e* = lcm(T)": {
        "title": "Saturation index e*",
        "tex": (
            r"The saturation index "
            r"\(e^\star(S)=\operatorname{lcm}(T)\) is the smallest \(n\ge1\) for "
            r"which \(S(q)\) divides \([n]_q\). Because a cyclotomic \(\Phi_k\) "
            r"divides \([n]_q\) iff \(k\mid n\), the product "
            r"\(\prod_{k\in T}\Phi_k\) divides \([n]_q\) iff every \(k\in T\) "
            r"divides \(n\), i.e. iff \(\operatorname{lcm}(T)\mid n\). It is the "
            r"least shift making the equal-tail difference finite."
        ),
        "values": [
            r"a positive integer (the minimal \(n\)), e.g. \(e^\star=12\) for "
            r"\(5/12\) and \(e^\star=15\) for \(4/15\);",
            r"<b>undefined</b>: \(S\) is not a squarefree product of "
            r"cyclotomics, so no \(n\) works.",
        ],
    },
    "minimal n with S | [n]_q": {
        "title": "Minimal saturating n",
        "tex": (
            r"The smallest \(n\) with \(S(q)\mid[n]_q\). It equals the "
            r"saturation index \(e^\star\), and every multiple of \(e^\star\) "
            r"also works; nothing smaller does."
        ),
        "values": [
            r"\(n=e^\star\) when \(S\) is squarefree-cyclotomic;",
            r"<b>none</b>: no finite \(n\) exists (the impossibility branch).",
        ],
    },
    "deg S  (bound d-1)": {
        "title": "Degree of S and the bound",
        "tex": (
            r"The polynomial degree \(\deg S\), shown against the upper bound "
            r"\(d-1\). The degree theorem says \(\deg S\le d-1\) for every "
            r"\(a/d\), with equality iff \(S=[d]_q\). For a squarefree \(S\), "
            r"\(\deg S=\sum_{k\in T}\varphi(k)\) (a sum of Euler totients)."
        ),
        "values": [
            r"any \(0\le\deg S\le d-1\); equality \(\deg S=d-1\) is the "
            r"no-collapse case \(S=[d]_q\);",
            r"a smaller value signals a collapse, the gap being the totient "
            r"weight of the dropped \(\Phi_k\).",
        ],
    },
    "deg S = d-1  (so S = [d]_q)": {
        "title": "Saturates the degree bound?",
        "tex": (
            r"Whether the degree bound is tight, \(\deg S=d-1\). This happens "
            r"if and only if \(S=[d]_q\), the full \(q\)-integer, which happens "
            r"if and only if \(a\equiv\pm1\pmod d\)."
        ),
        "values": [
            r"<b>yes</b>: \(S=[d]_q\), the saturating \(a\equiv\pm1\) locus;",
            r"<b>no</b>: a proper collapse or the impossibility branch, "
            r"\(\deg S<d-1\).",
        ],
    },
    "equality locus a == +/-1 (mod d)": {
        "title": "Equality locus",
        "tex": (
            r"Whether the numerator sits on the locus "
            r"\(a\equiv1\) or \(a\equiv d-1\pmod d\). The degree theorem "
            r"identifies this locus exactly with the case \(S=[d]_q\), so it is "
            r"the predicted \(\deg S=d-1\) set."
        ),
        "values": [
            r"<b>on locus</b>: \(a\equiv\pm1\pmod d\), so \(S=[d]_q\);",
            r"<b>off locus</b>: every other residue, where \(S\) collapses or "
            r"leaves the cyclotomic world.",
        ],
    },
    "S(1) = d": {
        "title": "The invariant S(1) = d",
        "tex": (
            r"Evaluating the denominator at \(q=1\) always returns the ordinary "
            r"denominator \(d\): \(S(1)=d\). This is a built-in consistency "
            r"check, since \(\left[\tfrac{a}{d}\right]_q\to\tfrac{a}{d}\) as "
            r"\(q\to1\)."
        ),
        "values": [
            r"<b>ok</b>: \(S(1)=d\) as it must;",
            r"<b>FAIL</b>: would indicate a bug; never expected.",
        ],
    },
    "S(0) = 1": {
        "title": "The invariant S(0) = 1",
        "tex": (
            r"The constant term of the reduced denominator is always \(1\): "
            r"\(S(0)=1\), so \(q\nmid S\). This makes \(S\) the same element of "
            r"\(\mathbb{Z}[q]\) and the Laurent ring \(\mathbb{Z}[q,q^{-1}]\), "
            r"with no \(q\)-power unit to clear."
        ),
        "values": [
            r"<b>ok</b>: \(S(0)=1\) as it must;",
            r"<b>FAIL</b>: would indicate a bug; never expected.",
        ],
    },
    "S squarefree": {
        "title": "Is S squarefree?",
        "tex": (
            r"Whether every irreducible factor of \(S\) occurs to the first "
            r"power. Saturation needs a squarefree \(S\): a repeated factor "
            r"(e.g. \(\Phi_2^{2}\) for \(3/8\)) cannot divide the squarefree "
            r"\([n]_q\), so such an \(S\) divides no \([n]_q\)."
        ),
        "values": [
            r"<b>yes</b>: a clean subset product of distinct \(\Phi_k\);",
            r"<b>no</b>: a repeated cyclotomic factor, the non-squarefree "
            r"impossibility branch.",
        ],
    },
    "S a product of cyclotomics": {
        "title": "Is S a product of cyclotomics?",
        "tex": (
            r"Whether \(S\) factors entirely into cyclotomic polynomials "
            r"\(\Phi_k\), with no non-cyclotomic ``core'' factor. Only a "
            r"cyclotomic \(S\) can divide some \([n]_q\)."
        ),
        "values": [
            r"<b>yes</b>: \(S=\prod_{k\in T}\Phi_k\) (possibly with "
            r"repeats);",
            r"<b>no</b>: \(S\) carries a non-cyclotomic core (e.g. \(2/15\)), "
            r"so it divides no \([n]_q\).",
        ],
    },
}


def _annotate_registry() -> None:
    """Add the MathJax symbol and per-field LaTeX defaults to OPERATIONS.

    The card and panel symbols render as math; a math-input field shows its
    example as a typeset default (for instance 3/2 as a fraction). The longer
    "What is this?" description is attached too. Computing the LaTeX once here
    keeps the page template free of any math of its own.
    """
    for op, meta in OPERATIONS.items():
        meta["tex"] = _TEX_SYMBOL.get(op, "")
        meta["about"] = _ABOUT.get(op, "")
        if meta["input_kind"] in (_REAL, _RATIONAL):
            for field in meta["fields"]:
                if field["name"] in ("input", "y"):
                    field["tex"] = fmt.to_tex(field["example"])


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
        return fmt.to_tex(parse_real(text))
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
            latex = "%s = %s" % (fmt.qrat_tex(p, s), fmt.to_tex(expr))
            return {
                "latex": latex,
                "text": data["expr"],
                "rows": [["value at q = 1", str(data["at_q_eq_1"])]],
                "meta": {"op": op, "p": p, "s": s},
            }

        if op == "exact-rational":
            from .qrat_exact import q_rational_difference, q_rational_exact
            from .rational import q as _q

            yv = str(args.get("y") or "").strip()
            if not yv:
                ex = q_rational_exact(input_text)
                P, Q = sp.factor(ex.P), sp.factor(ex.Q)
                latex = "%s = %s" % (
                    fmt.qrat_tex(ex.a, ex.b),
                    fmt.display_fraction_tex(fmt.to_tex(P), fmt.to_tex(Q)),
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
            xl, yl = fmt.qrat_tex(d.x.a, d.x.b), fmt.qrat_tex(d.y.a, d.y.b)
            latex = "%s - %s = %s" % (
                xl,
                yl,
                fmt.display_fraction_tex(fmt.to_tex(num_f), fmt.to_tex(den_f)),
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
            latex = "%s = %s" % (fmt.qint_tex(nn), fmt.to_tex(data["q_int"]))
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
            latex = "%s = %s" % (fmt.qreal_tex(_sym(input_text)), _latex_for_coeffs(coeffs))
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
            latex = "%s = %s" % (
                fmt.qreal_tex(_sym(input_text)),
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
            latex = r"\text{prefix of } %s = %s" % (
                fmt.qreal_tex(_sym(input_text)),
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
            latex = "%s = %s" % (
                fmt.qreal_tex("%s %s 1" % (_sym(input_text), sign)),
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
            latex = "%s = %s" % (
                fmt.qreal_tex(_sym(input_text)),
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
            latex = "%s %s %s = %s" % (
                fmt.qreal_tex(_sym(input_text)),
                sym,
                fmt.qreal_tex(_sym(y)),
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
                fmt.to_tex(r.x),
                tex_sym,
                fmt.to_tex(r.y),
                fmt.to_tex(r.value),
                fmt.to_tex(r.matrix),
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
            target = fmt.qreal_tex("%s %s %s" % (_sym(input_text), sym, _sym(y)))
            engine = r"\left(%s %s %s\right)" % (
                fmt.qreal_tex(_sym(input_text)),
                sym,
                fmt.qreal_tex(_sym(y)),
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
            latex = "%s = %s" % (
                fmt.qreal_tex("-%s" % _sym(input_text)),
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
            latex = "%s + %s = %s" % (
                fmt.qreal_tex(_sym(input_text)),
                fmt.qreal_tex("-%s" % _sym(input_text)),
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
            latex = "%s^{+} - %s^{-} = %s" % (
                fmt.qrat_tex(p, s),
                fmt.qrat_tex(p, s),
                fmt.to_tex(data["gap"]),
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
            latex = r"%s:\quad R(q) = %s" % (
                fmt.qrat_tex(data["r"], data["s"]),
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
            latex = r"\rho\!\left(%s\right) \approx %s" % (
                fmt.qreal_tex(_sym(input_text)),
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
            latex = r"\varphi\!\left(%s\right) \in \mathbb{R}^{%d}" % (
                fmt.qreal_tex(_sym(input_text)),
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
            latex = "%s = %s" % (
                fmt.qrat_tex(result.a, result.b),
                fmt.display_fraction_tex(fmt.to_tex(R), fmt.to_tex(S)),
            )
            cyclo = (
                ", ".join(
                    fmt.phi_label(d, e) for d, e in sorted(result.cyclotomic_R.items())
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

        if op == "s-properties":
            from .factor import denominator_expr, factor_qreal, s_properties

            result = factor_qreal(input_text)
            S = denominator_expr(result)
            p = s_properties(input_text)
            latex = r"S(q)\ \text{of}\ %s = %s" % (
                fmt.qrat_tex(p.a, p.d),
                fmt.to_tex(S),
            )
            t_str = (
                "{" + ", ".join(str(k) for k in p.index_set_T) + "}"
                if p.index_set_T
                else "empty"
            )
            if p.saturation_index is None:
                kind_str = (
                    "non-cyclotomic (divides no [n]_q)"
                    if not p.is_cyclotomic
                    else "non-squarefree (divides no [n]_q)"
                )
                sat_str = "none"
            elif p.is_full_qint:
                kind_str = "full q-integer [d]_q"
                sat_str = f"{p.saturation_index} (= d)"
            else:
                kind_str = "proper cyclotomic collapse"
                sat_str = str(p.saturation_index)
            rows = [
                ["kind of S", kind_str],
                ["cyclotomic index set T", t_str],
                ["saturation index e* = lcm(T)", sat_str],
                ["minimal n with S | [n]_q", sat_str],
                ["deg S  (bound d-1)", f"{p.deg_S}  ({p.deg_bound})"],
                ["deg S = d-1  (so S = [d]_q)", "yes" if p.saturates_bound else "no"],
                [
                    "equality locus a == +/-1 (mod d)",
                    "yes" if p.equality_locus else "no",
                ],
                ["S(1) = d", f"{p.S_at_1}  ({'ok' if p.S_at_1_ok else 'FAIL'})"],
                ["S(0) = 1", f"{p.S_at_0}  ({'ok' if p.S_at_0_ok else 'FAIL'})"],
                ["S squarefree", "yes" if p.is_squarefree else "no"],
                ["S a product of cyclotomics", "yes" if p.is_cyclotomic else "no"],
            ]
            # Attach the click-to-open glossary for the rows that have one, so
            # the front end can make each property label expand to an
            # explanation with LaTeX and a list of what its values mean.
            row_info = {
                label: _SPROPS_GLOSSARY[label]
                for label, _ in rows
                if label in _SPROPS_GLOSSARY
            }
            return {
                "latex": latex,
                "text": f"S(q) = {sp.sstr(S)}",
                "rows": rows,
                "meta": {
                    "op": op,
                    "a": p.a,
                    "d": p.d,
                    "index_set_T": p.index_set_T,
                    "saturation_index": p.saturation_index,
                    "deg_S": p.deg_S,
                    "deg_bound": p.deg_bound,
                    "saturates_bound": p.saturates_bound,
                    "is_full_qint": p.is_full_qint,
                    "is_collapse": p.is_collapse,
                    "is_squarefree": p.is_squarefree,
                    "is_cyclotomic": p.is_cyclotomic,
                    "S_at_1": p.S_at_1,
                    "S_at_0": p.S_at_0,
                    "equality_locus": p.equality_locus,
                    "rowInfo": row_info,
                },
            }

        if op == "roots":
            from .factor import (
                classify_poles,
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
            # classify_poles labels each pole by its exact cyclotomic index, so
            # the overlay can mark a cyclotomic pole (on |q| = 1) apart from a
            # core pole (which can leave the unit circle and drop the radius
            # below 1), and tag the nearest pole that sets the radius.
            pole_data = classify_poles(result)
            poles = pole_data["poles"]
            radius = pole_data["radius"]
            radius_index = pole_data["radius_index"]
            latex = "R(q) = %s" % fmt.to_tex(R)
            cyclo = (
                ", ".join(
                    fmt.phi_label(d, e) for d, e in sorted(result.cyclotomic_R.items())
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
            n_cyc_pole = sum(1 for p in poles if p["kind"] == "cyclotomic")
            n_core_pole = sum(1 for p in poles if p["kind"] == "core")
            if poles:
                rows.append(
                    [
                        "poles of [a/b]_q (zeros of S): cyclotomic / core",
                        f"{n_cyc_pole} / {n_core_pole}",
                    ]
                )
            if radius is not None:
                where = (
                    f"on {fmt.phi_label(radius_index)} (|q| = 1)"
                    if radius_index is not None
                    else "a non-cyclotomic core pole (|q| may be < 1)"
                )
                rows.append(
                    [
                        "radius of convergence (nearest pole |q|)",
                        "%.6f  (%s)" % (radius, where),
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
                    "radius_index": radius_index,
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
                % (fmt.to_tex(x_lo), fmt.to_tex(x_hi), N - 1)
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

        if op == "s-atlas":
            from .factor import REGIME_LABELS

            d_max = max(2, min(int(str(input_text).strip()), 24))
            a_max_raw = _int_arg(args, "a_max", 0)
            a_max = a_max_raw if a_max_raw > 0 else None
            res = _app.compute_satlas(d_max, a_max)
            atlas = res["data"]
            counts = atlas["regime_counts"]
            latex = (
                r"S\!\left(\tfrac{a}{d}\right)\ \text{regime over}\ d\le %d" % d_max
            )
            rows = [
                [REGIME_LABELS[r], str(counts[r])]
                for r in ("full", "collapse", "nonsquarefree", "noncyclotomic")
            ]
            rows.append(["fractions", str(len(atlas["cells"]))])
            appearances = atlas["index_appearances"]
            rows.append(
                [
                    "cyclotomic factor appearances",
                    ", ".join(f"{fmt.phi_label(k)}:{n}" for k, n in appearances.items()),
                ]
            )
            # the same tally as typeset math, so the row reads
            # Phi_2 : 17,  Phi_3 : 14, ... with real subscripts (rendered by the
            # front end when meta.rowTex names the row).
            appearances_tex = ",\\ ".join(
                r"%s\!:\!%d" % (fmt.phi_tex(k), n) for k, n in appearances.items()
            )
            return {
                "latex": latex,
                "text": (
                    f"S(q) regime atlas over the coprime grid d <= {d_max}: "
                    f"{counts['full']} full [d]_q, {counts['collapse']} collapse, "
                    f"{counts['nonsquarefree']} non-squarefree, "
                    f"{counts['noncyclotomic']} non-cyclotomic"
                ),
                "rows": rows,
                "meta": {
                    "op": op,
                    "plot3d": {"kind": "s-atlas", **atlas},
                    "rowTex": {"cyclotomic factor appearances": appearances_tex},
                },
            }

        if op == "saturation-explorer":
            from .factor import REGIME_LABELS

            d = max(2, int(str(input_text).strip()))
            res = _app.compute_saturation(d)
            ex = res["data"]
            points = ex["points"]
            finite = [pt for pt in points if pt["e_star"] is not None]
            latex = r"e^{\star}\!\left(\tfrac{a}{%d}\right)\ \text{vs}\ a" % d
            rows = [
                [
                    f"{pt['a']}/{d}",
                    (str(pt["e_star"]) if pt["e_star"] is not None else "none")
                    + f"  ({REGIME_LABELS[pt['regime']]})",
                ]
                for pt in points
            ]
            return {
                "latex": latex,
                "text": (
                    f"saturation index e* across a/{d}: {len(finite)} of "
                    f"{len(points)} residues have a finite e*"
                ),
                "rows": rows,
                "meta": {"op": op, "plot3d": {"kind": "saturation", **ex}},
            }

        if op == "degree-collapse":
            d_max = max(2, min(int(str(input_text).strip()), 24))
            a_max_raw = _int_arg(args, "a_max", 0)
            a_max = a_max_raw if a_max_raw > 0 else None
            res = _app.compute_degcollapse(d_max, a_max)
            dc = res["data"]
            cells = dc["cells"]
            saturating = sum(1 for c in cells if c["saturates_bound"])
            latex = r"\deg S\ \text{vs}\ d-1\ \text{over}\ d\le %d" % d_max
            return {
                "latex": latex,
                "text": (
                    f"deg S vs d-1 over the coprime grid d <= {d_max}: "
                    f"{saturating} of {len(cells)} fractions saturate the bound "
                    f"(S = [d]_q); collapse-depth totient law "
                    f"{'holds' if dc['depth_law_holds'] else 'FAILS'}"
                ),
                "rows": [
                    ["fractions", str(len(cells))],
                    ["on the diagonal deg S = d-1 (S = [d]_q)", str(saturating)],
                    [
                        "collapse-depth law (squarefree S)",
                        "holds" if dc["depth_law_holds"] else "FAILS",
                    ],
                ],
                "meta": {"op": op, "plot3d": {"kind": "degree-collapse", **dc}},
            }

        if op == "certificate":
            res = _app.compute_coeffs(input_text, n)
            coeffs = res["data"]["coefficients"]
            entry = _laurent_entry(input_text, coeffs)
            latex = "%s = %s" % (
                fmt.qreal_tex(_sym(input_text)),
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
            base = fmt.qrat_tex(p, s)
            if op == "jump-gap":
                base = "%s^{+} - %s^{-}" % (fmt.qrat_tex(p, s), fmt.qrat_tex(p, s))
            elif op == "factor":
                base = r"R(q),\ S(q)\ \text{of}\ %s" % fmt.qrat_tex(p, s)
            elif op == "roots":
                base = r"\{\,q : R(q)=0\,\}\ \text{of}\ %s" % fmt.qrat_tex(p, s)
            elif op == "frieze":
                base = r"\text{frieze of}\ %s" % fmt.qrat_tex(p, s)
            return {"latex": base}

        if kind == _INTEGER:
            nn = int(str(input_text).strip())
            return {"latex": fmt.qint_tex(nn)}

        if kind == _SEQUENCE:
            from . import oeis as _oeis

            seq = _oeis.parse_sequence(input_text)
            shown = ", ".join(str(t) for t in seq[:12])
            if len(seq) > 12:
                shown += r", \dots"
            return {"latex": r"(%s)" % shown}

        # real-valued input (possibly with a second value y)
        xl = fmt.to_tex(_app._parse_real(input_text))
        if op in ("q-sum", "q-product", "deficit"):
            yl = fmt.to_tex(_app._parse_real(str(args.get("y") or "")))
            if op == "deficit":
                sym = r"\cdot" if str(args.get("op", "add")) == "mul" else "+"
                base = r"%s - \left(%s %s %s\right)" % (
                    fmt.qreal_tex("%s %s %s" % (xl, sym, yl)),
                    fmt.qreal_tex(xl),
                    sym,
                    fmt.qreal_tex(yl),
                )
            else:
                sym = r"\cdot" if op == "q-product" else "+"
                base = "%s %s %s" % (fmt.qreal_tex(xl), sym, fmt.qreal_tex(yl))
            return {"latex": base}
        if op == "quad-arith":
            yl = fmt.to_tex(_app._parse_real(str(args.get("y") or "")))
            sym = {"add": "+", "sub": "-", "mul": r"\cdot", "div": "/"}.get(
                str(args.get("op", "add")), "+"
            )
            return {"latex": r"\left(%s\right) %s \left(%s\right)" % (xl, sym, yl)}
        if op == "negation":
            return {"latex": fmt.qreal_tex("-%s" % xl)}
        if op == "finiteness":
            return {"latex": "%s + %s" % (fmt.qreal_tex(xl), fmt.qreal_tex("-%s" % xl))}
        if op == "shift":
            sign = "-" if str(args.get("direction", "up")) == "down" else "+"
            return {"latex": fmt.qreal_tex("%s %s 1" % (xl, sign))}
        return {"latex": fmt.qreal_tex(xl)}
    except Exception:  # noqa: BLE001 - an unfinished input simply has no preview
        return {"latex": ""}


# --------------------------------------------------------------------------
# The page.
# --------------------------------------------------------------------------


import importlib.resources as _res


def _asset(name: str) -> str:
    """Read a frontend asset shipped under qreals/web/."""
    return (_res.files("qreals.web") / name).read_text(encoding="utf-8")


# Content types for the vendored static files (the local MathJax bundle).
_VENDOR_TYPES = {
    ".js": "application/javascript",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".css": "text/css",
}


def _vendor_asset(rel: str) -> tuple[bytes, str] | None:
    """Read a vendored static file shipped under qreals/web/vendor/.

    MathJax and its fonts are vendored there so every page renders all math
    with no network access. Returns (bytes, content type), or None for a
    missing or unsafe path.
    """
    parts = [p for p in str(rel).replace("\\", "/").split("/") if p]
    if not parts or any(p == ".." for p in parts):
        return None
    node = _res.files("qreals.web") / "vendor"
    for part in parts:
        node = node / part
    try:
        data = node.read_bytes()
    except (FileNotFoundError, IsADirectoryError, OSError):
        return None
    name = parts[-1]
    ext = name[name.rfind(".") :] if "." in name else ""
    return data, _VENDOR_TYPES.get(ext, "application/octet-stream")


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


# --------------------------------------------------------------------------
# Optional update check. On the first request the server asks the PyPI JSON API
# for the latest released qreals version and compares it with the running one,
# so the page can show an unobtrusive "update available" banner. It is
# best-effort: a short timeout, cached once per process, and silent on any
# failure (offline, PyPI down, or not yet published). Set the environment
# variable QREALS_NO_UPDATE_CHECK to any value to turn it off (CI, air-gapped).
# --------------------------------------------------------------------------

_PYPI_JSON_URL = "https://pypi.org/pypi/qreals/json"
_update_cache: dict[str, Any] | None = None

# A short changelog shipped with the package, keyed by version. Each note is a
# {change, helps} pair carrying inline LaTeX (\( ... \)); the web page renders
# them as the "What's new" patch notes, both in the update banner and from a
# footer link. Keep newest first.
CHANGELOG: dict[str, dict[str, Any]] = {
    "0.1.3": {
        "summary": "Denominator S(q) tools and an update checker.",
        "notes": [
            {
                "change": r"Added the \(S(q)\) cyclotomic-factor atlas: a "
                r"coloured \((a,d)\) grid of each denominator's regime.",
                "helps": r"See at a glance where \(S=[d]_q\), where it "
                r"collapses, and where it divides no \([n]_q\).",
            },
            {
                "change": r"Added the saturation-index explorer for a fixed "
                r"\(d\): \(e^\star(a/d)=\operatorname{lcm}(T)\) versus \(a\).",
                "helps": r"Reads off the least \(n\) making an equal-tail "
                r"difference finite, and the residues where none exists.",
            },
            {
                "change": r"Added the degree-collapse map: \(\deg S\) against "
                r"the bound \(d-1\).",
                "helps": r"Shows how tight the bound \(\deg S\le d-1\) is and "
                r"how deep each collapse runs.",
            },
            {
                "change": r"The roots view now overlays the poles of \(S(q)\), "
                r"labelled by cyclotomic index, marking the nearest as the "
                r"radius of convergence \(\rho\).",
                "helps": r"Connects the denominator's factorisation to the "
                r"pole structure and the radius question.",
            },
            {
                "change": r"Each S(q) property is now click-to-explain, and "
                r"every tool has a LaTeX description.",
                "helps": r"Makes the panel self-documenting for a newcomer.",
            },
        ],
    },
    "0.1.2": {
        "summary": "Earlier release.",
        "notes": [
            {
                "change": r"The denominator \(S(q)\) properties panel: index "
                r"set \(T\), saturation index, degree bound, and invariants.",
                "helps": r"Surfaces the structure of \(S(q)\), not just its "
                r"factored form.",
            },
        ],
    },
}


def changelog_for(version: str | None) -> dict[str, Any] | None:
    """The changelog entry for a version (summary + notes), or None."""
    if not version:
        return None
    return CHANGELOG.get(version)


def _parse_version(text: str) -> tuple[int, ...]:
    """A lenient dotted-numeric version key, e.g. '0.1.10' -> (0, 1, 10).

    Only the leading numeric components are read; a trailing pre-release or
    local tag (rc1, +local) is dropped, which is enough to compare releases.
    """
    parts: list[int] = []
    for chunk in str(text).split("."):
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        if num == "":
            break
        parts.append(int(num))
    return tuple(parts)


def _fetch_latest_version(timeout: float = 2.5) -> str | None:
    """The latest qreals version on PyPI, or None on any failure."""
    import urllib.request

    req = urllib.request.Request(
        _PYPI_JSON_URL, headers={"User-Agent": f"qreals/{__version__}"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https
        data = json.loads(resp.read().decode("utf-8"))
    version = data.get("info", {}).get("version")
    return str(version) if version else None


def check_for_update(force: bool = False) -> dict[str, Any]:
    """Compare the running version with the latest on PyPI (cached, fail-silent).

    Returns {current, latest, update_available, checked}. `latest` is None and
    `checked` is False when the check was skipped (opt-out) or could not reach
    PyPI; `update_available` is True only when PyPI is strictly newer. The
    result is cached for the life of the process, so it is one request per
    `qreals serve`, not one per page load.
    """
    global _update_cache
    if _update_cache is not None and not force:
        return _update_cache
    import os

    result: dict[str, Any] = {
        "current": __version__,
        "latest": None,
        "update_available": False,
        "checked": False,
        # the running version's patch notes, always available so the page can
        # show "What's new" even when there is no update to announce.
        "current_notes": changelog_for(__version__),
        # the new version's notes when the shipped changelog happens to know
        # them; usually None for a strictly-newer remote version.
        "latest_notes": None,
    }
    if os.environ.get("QREALS_NO_UPDATE_CHECK"):
        _update_cache = result
        return result
    try:
        latest = _fetch_latest_version()
    except Exception:  # noqa: BLE001 - the check must never break serve
        latest = None
    if latest:
        result["latest"] = latest
        result["checked"] = True
        result["update_available"] = (
            _parse_version(latest) > _parse_version(__version__)
        )
        result["latest_notes"] = changelog_for(latest)
    _update_cache = result
    return result


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

    async def version_endpoint(_request: Request) -> Any:
        return JSONResponse(check_for_update())

    async def vendor_endpoint(request: Request) -> Any:
        from starlette.responses import Response

        found = _vendor_asset(str(request.path_params.get("path", "")))
        if found is None:
            return Response(status_code=404)
        data, ctype = found
        return Response(content=data, media_type=ctype)

    application.add_route("/vendor/{path:path}", vendor_endpoint, methods=["GET"])
    application.add_route("/", index, methods=["GET"])
    application.add_route("/compute", compute_endpoint, methods=["POST"])
    application.add_route("/preview", preview_endpoint, methods=["POST"])
    application.add_route("/certificate", certificate_endpoint, methods=["POST"])
    application.add_route("/export", export_endpoint, methods=["POST"])
    application.add_route("/version", version_endpoint, methods=["GET"])
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

    @application.get("/version")
    def version_endpoint() -> Any:
        return jsonify(check_for_update())

    @application.get("/vendor/<path:rel>")
    def vendor_endpoint(rel: str) -> Any:
        from flask import Response

        found = _vendor_asset(rel)
        if found is None:
            return Response("not found", status=404)
        data, ctype = found
        return Response(data, mimetype=ctype)

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
