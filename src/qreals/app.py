"""Guided interface to qreals plus a scripting CLI.

Running ``qreals`` with no arguments opens an arrow-key menu (questionary for
navigation, rich for output) that can perform every computation in the public
API, with an example and validation on each input and formatted results. The
same capabilities are available headless for agents and scripts as
subcommands, e.g. ``qreals rational 3 2`` or ``qreals coeffs pi 12 --json``.

The design keeps three concerns apart so each capability can be exercised
without a terminal:

- ``compute_*`` functions take parsed inputs and return a plain result dict.
- ``render_result`` turns a result dict into text (rich if available, else
  builtins, or JSON for ``--json``).
- ``prompt_*`` functions gather and validate input interactively.

The smoke test in ``tests/test_app.py`` builds the menu and calls a
``compute_*`` function directly, so it never needs a TTY.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

import sympy as sp

from . import (
    __version__,
    coefficient_max_abs,
    coeffs_locked_by_convergent,
    deficit,
    factor_qreal,
    finite_xnegx,
    first_negative_coefficient_index,
    first_nonzero_coefficient_index,
    format_laurent,
    integer_part_prefix,
    jumpgap,
    mgo_laurent,
    negation_panel,
    negation_sum,
    number_of_zeros,
    q,
    q_add,
    q_int,
    q_int_qinv,
    q_mul,
    q_neg,
    q_rational,
    q_real_truncated,
    quad_arith,
    radius,
    shift_down,
    shift_up,
)
from . import exports, features
from ._parsing import parse_real
from .store import SavedEntry, SavedStore

# Result dicts carry a "title", an ordered list of "blocks" to render, and a
# flat "data" dict for --json. A block is one of:
#   {"kind": "poly",  "label": str, "text": str}
#   {"kind": "kv",    "pairs": [(label, value), ...]}
#   {"kind": "table", "columns": [str, ...], "rows": [[cell, ...], ...]}
#   {"kind": "note",  "text": str}
Result = dict[str, Any]


# --------------------------------------------------------------------------
# Input parsing and validation. Validators return True on success or a short
# error string, matching what questionary's `validate=` expects.
# --------------------------------------------------------------------------


def _parse_real(text: str) -> sp.Expr:
    """Parse a sympy-readable real, raising ValueError on anything else."""
    text = text.strip()
    if not text:
        raise ValueError("enter a value, for example pi")
    try:
        value = parse_real(text)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        raise ValueError("could not read that; try pi, sqrt(2), (1+sqrt(5))/2, 22/7")
    if value.is_real is not True:
        raise ValueError("must be a real number")
    if value.is_finite is False:
        raise ValueError("must be finite")
    return value


def _validate_real(text: str) -> bool | str:
    try:
        _parse_real(text)
    except ValueError as exc:
        return str(exc)
    return True


def _parse_rational(text: str) -> tuple[int, int]:
    """Parse one rational p/s, raising ValueError on anything not rational."""
    value = _parse_real(text)
    if value.is_rational is not True:
        raise ValueError("must be a rational p/s, for example 3/5")
    r = sp.Rational(value)
    return int(r.p), int(r.q)


def _validate_rational(text: str) -> bool | str:
    try:
        _parse_rational(text)
    except ValueError as exc:
        return str(exc)
    return True


def _make_int_validator(
    *, low: int | None = None, nonzero: bool = False
) -> Callable[[str], bool | str]:
    def _validate(text: str) -> bool | str:
        text = text.strip()
        try:
            value = int(text)
        except ValueError:
            return "enter a whole number"
        if nonzero and value == 0:
            return "must not be zero"
        if low is not None and value < low:
            return f"must be at least {low}"
        return True

    return _validate


# --------------------------------------------------------------------------
# Pure computations. Each returns a Result dict and never prompts or prints.
# --------------------------------------------------------------------------


def compute_rational(p: int, s: int) -> Result:
    """Exact [p/s]_q as a rational function in q."""
    expr = sp.sympify(q_rational(p, s))
    at_one = sp.simplify(expr.subs(q, 1))
    return {
        "kind": "rational",
        "title": f"[{p}/{s}]_q  (exact)",
        "blocks": [
            {"kind": "poly", "label": f"[{p}/{s}]_q", "text": sp.sstr(expr)},
            {
                "kind": "kv",
                "pairs": [
                    ("at q = 1", f"{at_one}   (the ordinary value {sp.Rational(p, s)})")
                ],
            },
        ],
        "data": {
            "p": p,
            "s": s,
            "expr": sp.sstr(expr),
            "at_q_eq_1": str(at_one),
        },
    }


def _qrat_label(a: int, b: int) -> str:
    """[a]_q for an integer denominator, else [a/b]_q."""
    return f"[{a}]_q" if b == 1 else f"[{a}/{b}]_q"


def compute_exact_rational(x: str, y: str = "") -> Result:
    """Exact [x]_q = P/Q for a q-rational x, or the exact difference [x]_q - [y]_q.

    With y empty this is the single exact rational function P(q)/Q(q), factored,
    with its continued fraction and q = 1 value. With y given it is the June 1
    board: [x]_q - [y]_q = (P_x Q_y - P_y Q_x)/(Q_x Q_y), reduced and factored,
    plus the Q_x | Q_y / Q_y | Q_x divisibilities that force Q_x = Q_y up to a
    unit. Everything is exact over Q[q].
    """
    from .qrat_exact import q_rational_difference, q_rational_exact

    y = (y or "").strip()
    if not y:
        ex = q_rational_exact(x)
        P, Q = sp.factor(ex.P), sp.factor(ex.Q)
        xl = _qrat_label(ex.a, ex.b)
        return {
            "kind": "exact-rational",
            "title": f"{xl}  (exact P/Q)",
            "blocks": [
                {
                    "kind": "poly",
                    "label": xl,
                    "text": f"({sp.sstr(P)}) / ({sp.sstr(Q)})",
                },
                {
                    "kind": "kv",
                    "pairs": [
                        ("numerator P(q)", sp.sstr(P)),
                        ("denominator Q(q)", sp.sstr(Q)),
                        ("continued fraction", str(ex.cf)),
                        (
                            "at q = 1",
                            f"{sp.cancel(ex.P / ex.Q).subs(q, 1)}   "
                            f"(the ordinary value {sp.Rational(ex.a, ex.b)})",
                        ),
                    ],
                },
            ],
            "data": {
                "a": ex.a,
                "b": ex.b,
                "P": sp.sstr(sp.expand(ex.P)),
                "Q": sp.sstr(sp.expand(ex.Q)),
                "cf": list(ex.cf),
            },
        }
    d = q_rational_difference(x, y)
    xl, yl = _qrat_label(d.x.a, d.x.b), _qrat_label(d.y.a, d.y.b)
    num_f, den_f = sp.factor(d.num), sp.factor(d.den)
    return {
        "kind": "exact-rational",
        "title": f"{xl} - {yl}  (exact)",
        "blocks": [
            {
                "kind": "poly",
                "label": f"{xl} - {yl}",
                "text": f"({sp.sstr(num_f)}) / ({sp.sstr(den_f)})",
            },
            {
                "kind": "kv",
                "pairs": [
                    (xl, f"({sp.sstr(sp.factor(d.x.P))}) / ({sp.sstr(sp.factor(d.x.Q))})"),
                    (yl, f"({sp.sstr(sp.factor(d.y.P))}) / ({sp.sstr(sp.factor(d.y.Q))})"),
                    ("numerator P_x Q_y - P_y Q_x", sp.sstr(sp.factor(d.num_unreduced))),
                    ("denominator Q_x Q_y", sp.sstr(sp.factor(d.den_unreduced))),
                    ("Q_x | Q_y", "yes" if d.qx_divides_qy else "no"),
                    ("Q_y | Q_x", "yes" if d.qy_divides_qx else "no"),
                    ("Q_x = Q_y up to a unit", "yes" if d.q_equal_up_to_unit else "no"),
                    ("gcd(Q_x, Q_y)", sp.sstr(d.gcd_Q)),
                    (
                        "at q = 1",
                        f"{d.value_at_1}   (the ordinary "
                        f"{sp.Rational(d.x.a, d.x.b)} - {sp.Rational(d.y.a, d.y.b)})",
                    ),
                ],
            },
        ],
        "data": {
            "x": [d.x.a, d.x.b],
            "y": [d.y.a, d.y.b],
            "num": sp.sstr(d.num),
            "den": sp.sstr(d.den),
            "qx_divides_qy": d.qx_divides_qy,
            "qy_divides_qx": d.qy_divides_qx,
            "q_equal_up_to_unit": d.q_equal_up_to_unit,
        },
    }


def compute_jumpgap(p: int, s: int) -> Result:
    """The two q-versions of p/s and the factored gap between them.

    A rational p/s has a right version [p/s]_q^+ (the limit from above, the
    q_rational value) and a left version [p/s]_q^- (the limit from below). The
    gap between them is the single rational function
    (1 - q) q^E / (S^+ S^-) (Jouteur arXiv:2503.02122).
    """
    gap = jumpgap(p, s)
    return {
        "kind": "jumpgap",
        "title": f"jump gap of {p}/{s}:  [{p}/{s}]_q^+ - [{p}/{s}]_q^-",
        "blocks": [
            {
                "kind": "kv",
                "pairs": [
                    ("even-length continued fraction", str(list(gap.cf))),
                    ("exponent E  (det M_q = q^E)", str(gap.exponent)),
                ],
            },
            {
                "kind": "poly",
                "label": "S^+  (right q-denominator)",
                "text": sp.sstr(gap.s_plus),
            },
            {
                "kind": "poly",
                "label": "S^-  (left q-denominator)",
                "text": sp.sstr(gap.s_minus),
            },
            {
                "kind": "poly",
                "label": f"[{p}/{s}]_q^+  (right, limit from above)",
                "text": sp.sstr(gap.right),
            },
            {
                "kind": "poly",
                "label": f"[{p}/{s}]_q^-  (left, limit from below)",
                "text": sp.sstr(gap.left),
            },
            {
                "kind": "poly",
                "label": f"gap = [{p}/{s}]_q^+ - [{p}/{s}]_q^-",
                "text": sp.sstr(gap.gap),
            },
            {
                "kind": "note",
                "text": (
                    "gap = (1 - q) q^E / (S^+ S^-), with S^+(1) = S^-(1) = s "
                    "(Jouteur arXiv:2503.02122, Def 1.2, Prop 1.1, Prop 4.8)"
                ),
            },
        ],
        "data": {
            "p": p,
            "s": s,
            "cf": list(gap.cf),
            "exponent": gap.exponent,
            "s_plus": sp.sstr(gap.s_plus),
            "s_minus": sp.sstr(gap.s_minus),
            "right": sp.sstr(gap.right),
            "left": sp.sstr(gap.left),
            "gap": sp.sstr(gap.gap),
            "checks": gap.checks(),
        },
    }


def _factor_product_str(
    content: sp.Expr,
    factors: list[tuple[sp.Expr, int]],
    k: int = 0,
) -> str:
    """Pretty-print a factorisation as q^k times content times labelled factors.

    Cyclotomic factors print as Phi_d(q); the rest print as their sympy form.
    """
    from .factor import _cyclotomic_index

    parts: list[str] = []
    if k:
        parts.append(f"q^{k}")
    if content != sp.Integer(1):
        parts.append(sp.sstr(content))
    for fac, mult in factors:
        d = _cyclotomic_index(fac)
        label = f"Phi_{d}(q)" if d is not None else f"({sp.sstr(fac)})"
        parts.append(label if mult == 1 else f"{label}^{mult}")
    return " * ".join(parts) if parts else "1"


def compute_factor(a: int, b: int) -> Result:
    """Factor the numerator R(q) and denominator S(q) of [a/b]_q over Z[q].

    Each irreducible factor is classified as a cyclotomic polynomial Phi_d(q)
    or a non-cyclotomic core factor, answering the open questions
    on when R(q) is irreducible (often when a is prime) and how the cyclotomic
    factorisation of [m/1]_q generalises to a fraction.
    """
    result = factor_qreal((a, b))
    r_str = _factor_product_str(result.content_R, result.factors_R, result.k)
    s_str = _factor_product_str(result.content_S, result.factors_S)
    cyclo_r = ", ".join(
        f"Phi_{d}^{e}" for d, e in sorted(result.cyclotomic_R.items())
    )
    core_r = ", ".join(f"({sp.sstr(f)})^{m}" for f, m in result.core_R)
    return {
        "kind": "factor",
        "title": f"factorisation of [{a}/{b}]_q over Z[q]",
        "blocks": [
            {"kind": "poly", "label": f"R(q) numerator of [{a}/{b}]_q", "text": r_str},
            {
                "kind": "poly",
                "label": f"S(q) denominator of [{a}/{b}]_q",
                "text": s_str,
            },
            {
                "kind": "kv",
                "pairs": [
                    ("cyclotomic support of R", cyclo_r or "none"),
                    ("non-cyclotomic core of R", core_r or "none"),
                    (
                        "R is irreducible over Z[q]",
                        "yes" if result.is_irreducible_R else "no",
                    ),
                    (
                        "R is a pure product of cyclotomic factors",
                        "yes" if result.is_pure_cyclotomic_R else "no",
                    ),
                ],
            },
            {
                "kind": "note",
                "text": (
                    "[m/1]_q = [m]_q factors as the product of Phi_d(q) over d | m, "
                    "d > 1; for a general fraction the numerator can carry a "
                    "non-cyclotomic core, which is what makes R irreducible when a "
                    "is prime"
                ),
            },
        ],
        "data": {
            "a": a,
            "b": b,
            "k": result.k,
            "content_R": sp.sstr(result.content_R),
            "factors_R": [(sp.sstr(f), m) for f, m in result.factors_R],
            "cyclotomic_R": {str(d): e for d, e in result.cyclotomic_R.items()},
            "core_R": [(sp.sstr(f), m) for f, m in result.core_R],
            "content_S": sp.sstr(result.content_S),
            "factors_S": [(sp.sstr(f), m) for f, m in result.factors_S],
            "cyclotomic_S": {str(d): e for d, e in result.cyclotomic_S.items()},
            "core_S": [(sp.sstr(f), m) for f, m in result.core_S],
            "is_irreducible_R": result.is_irreducible_R,
            "is_pure_cyclotomic_R": result.is_pure_cyclotomic_R,
        },
    }


def compute_sprops(a: int, b: int) -> Result:
    """The mathematical properties of the denominator S(q) of [a/b]_q.

    The factor tool surfaces the numerator R(q) but says little about the
    denominator S(q); this analyzes S directly. S is the q-denominator;
    when it is a squarefree product of cyclotomics S = prod_{k in T} Phi_k it
    divides [n]_q iff the saturation index e* = lcm(T) divides n, so e* is the
    minimal n that makes the difference of equal-tail q-rationals finite. The
    degree obeys deg S <= d-1 with equality iff S = [d]_q iff a == +/-1 (mod d);
    S(1) = d and S(0) = 1 are built-in invariants; S can be the full [d]_q, a
    proper cyclotomic collapse, or the impossibility branch (non-squarefree or
    non-cyclotomic, dividing no [n]_q).
    """
    from .factor import s_properties

    p = s_properties((a, b))

    if p.saturation_index is None:
        if not p.is_cyclotomic:
            kind_str = "non-cyclotomic (no finite difference)"
        else:
            kind_str = "non-squarefree (impossibility branch)"
        sat_str = "none: S divides no [n]_q"
    elif p.is_full_qint:
        kind_str = "full q-integer [d]_q (no collapse)"
        sat_str = f"n = {p.saturation_index} = d (minimal)"
    else:
        kind_str = "proper cyclotomic collapse"
        sat_str = f"n = {p.saturation_index} (minimal)"

    t_str = (
        "{" + ", ".join(str(k) for k in p.index_set_T) + "}"
        if p.index_set_T
        else "empty"
    )
    return {
        "kind": "sprops",
        "title": f"denominator S(q) of [{a}/{b}]_q: cyclotomic structure",
        "blocks": [
            {"kind": "poly", "label": f"S(q) denominator of [{a}/{b}]_q", "text": p.S_str},
            {
                "kind": "kv",
                "pairs": [
                    ("kind of S", kind_str),
                    ("cyclotomic index set T", t_str),
                    (
                        "saturation index e* = lcm(T)",
                        str(p.saturation_index) if p.saturation_index else "undefined",
                    ),
                    ("minimal n with S | [n]_q", sat_str),
                    ("deg S", f"{p.deg_S}  (bound d-1 = {p.deg_bound})"),
                    (
                        "deg S = d-1 (so S = [d]_q)",
                        "yes" if p.saturates_bound else "no",
                    ),
                    (
                        "equality locus a == +/-1 (mod d)",
                        f"a mod d = {p.a_mod_d}, "
                        + ("on locus" if p.equality_locus else "off locus"),
                    ),
                    ("S(1) = d invariant", f"S(1) = {p.S_at_1}  ({'ok' if p.S_at_1_ok else 'FAIL'})"),
                    ("S(0) = 1 invariant", f"S(0) = {p.S_at_0}  ({'ok' if p.S_at_0_ok else 'FAIL'})"),
                    ("S is squarefree", "yes" if p.is_squarefree else "no"),
                    ("S is a product of cyclotomics", "yes" if p.is_cyclotomic else "no"),
                ],
            },
            {
                "kind": "note",
                "text": (
                    "S(q) divides [n]_q exactly when S is a squarefree product of "
                    "cyclotomics and lcm of its indices divides n; then the difference "
                    "of two equal-tail q-rationals over S is a finite Laurent "
                    "polynomial at every multiple of e*. A non-squarefree S (e.g. 3/8 "
                    "gives Phi_2^2 Phi_4) or a non-cyclotomic S (e.g. 2/15) divides no "
                    "[n]_q, so that difference is never finite. deg S = d-1 picks out "
                    "exactly S = [d]_q, the a == +/-1 (mod d) tails."
                ),
            },
        ],
        "data": {
            "a": p.a,
            "d": p.d,
            "S": p.S_str,
            "index_set_T": p.index_set_T,
            "multiplicities": {str(k): e for k, e in p.multiplicities.items()},
            "is_cyclotomic": p.is_cyclotomic,
            "is_squarefree": p.is_squarefree,
            "saturation_index": p.saturation_index,
            "minimal_saturating_n": p.minimal_saturating_n,
            "deg_S": p.deg_S,
            "deg_bound": p.deg_bound,
            "saturates_bound": p.saturates_bound,
            "is_full_qint": p.is_full_qint,
            "is_collapse": p.is_collapse,
            "S_at_1": p.S_at_1,
            "S_at_1_ok": p.S_at_1_ok,
            "S_at_0": p.S_at_0,
            "S_at_0_ok": p.S_at_0_ok,
            "a_mod_d": p.a_mod_d,
            "equality_locus": p.equality_locus,
        },
    }


def _t_set_str(indices: list[int]) -> str:
    """Render a cyclotomic index set T as {k, ...} or 'empty'."""
    return "{" + ", ".join(str(k) for k in indices) + "}" if indices else "empty"


_ATLAS_MAX_ROWS = 80  # cap the per-fraction CLI table; the summary stays full


def compute_satlas(d_max: int, a_max: int | None = None) -> Result:
    """The S(q) cyclotomic-factor atlas over the coprime (a, d) grid.

    For every proper fraction a/d (0 < a/d < 1) up to the bounds this reports
    the regime of the q-denominator S(q): the full [d]_q, a proper cyclotomic
    collapse, or the impossibility branch (non-squarefree like 3/8, or
    non-cyclotomic like 2/15). A regime tally and a count of how often each
    Phi_k appears make Remark 2 (S is a subset product of the cyclotomic factors
    of [d]_q) visible as d grows.
    """
    from .factor import REGIME_LABELS, REGIME_ORDER, s_atlas

    atlas = s_atlas(d_max, a_max)
    cells = atlas["cells"]

    count_rows = [
        [REGIME_LABELS[r], str(atlas["regime_counts"][r])] for r in REGIME_ORDER
    ]
    index_rows = [
        [f"Phi_{k}", str(n)] for k, n in atlas["index_appearances"].items()
    ]
    cell_rows = [
        [
            f"{c['a']}/{c['d']}",
            REGIME_LABELS[c["regime"]],
            _t_set_str(c["T"]),
            f"{c['deg_S']}/{c['deg_bound']}",
            str(c["saturation_index"]) if c["saturation_index"] else "-",
        ]
        for c in cells[:_ATLAS_MAX_ROWS]
    ]
    truncated = len(cells) > _ATLAS_MAX_ROWS
    bound_str = f"d <= {d_max}" + (f", a <= {a_max}" if a_max else "")
    blocks: list[dict[str, Any]] = [
        {
            "kind": "kv",
            "pairs": [
                ("grid", f"{bound_str}, proper coprime fractions 0 < a/d < 1"),
                ("fractions", str(len(cells))),
            ],
        },
        {
            "kind": "table",
            "columns": ["regime", "count"],
            "rows": count_rows,
        },
        {
            "kind": "table",
            "columns": ["cyclotomic factor", "appearances"],
            "rows": index_rows,
        },
        {
            "kind": "table",
            "columns": ["a/d", "regime", "T", "deg S/(d-1)", "e*"],
            "rows": cell_rows,
        },
    ]
    if truncated:
        blocks.append(
            {
                "kind": "note",
                "text": (
                    f"per-fraction table truncated to the first {_ATLAS_MAX_ROWS} of "
                    f"{len(cells)} fractions; the regime tally above covers all of "
                    "them. Use --json for the full grid, or the web atlas for the "
                    "coloured heat map."
                ),
            }
        )
    blocks.append(
        {
            "kind": "note",
            "text": (
                "the two saturating regimes (full [d]_q and proper collapse) are the "
                "squarefree-cyclotomic S that divide [n]_q at n = e* = lcm(T); the "
                "two impossibility regimes (non-squarefree, non-cyclotomic) divide no "
                "[n]_q. The a == +/-1 (mod d) locus is exactly the full [d]_q cells."
            ),
        }
    )
    return {
        "kind": "satlas",
        "title": f"S(q) cyclotomic-factor atlas  ({bound_str})",
        "blocks": blocks,
        "data": atlas,
    }


def compute_saturation(d: int) -> Result:
    """The saturation index e* and minimal saturating n as a ranges over a/d.

    For a fixed denominator d this sweeps every numerator a coprime to d and
    reports e*(a/d) = lcm(T), the minimal n with S | [n]_q, and the regime,
    flagging the impossibility residues that have no finite n. This is the
    Saturation box read across a whole denominator: e* is the least n making the
    difference of two equal-tail q-rationals over S a finite Laurent polynomial.
    """
    from .factor import REGIME_LABELS, saturation_explorer

    ex = saturation_explorer(d)
    points = ex["points"]
    rows = [
        [
            f"{pt['a']}/{d}",
            str(pt["a_mod_d"]),
            REGIME_LABELS[pt["regime"]],
            str(pt["e_star"]) if pt["e_star"] is not None else "none (no finite n)",
            _t_set_str(pt["T"]),
            str(pt["deg_S"]),
        ]
        for pt in points
    ]
    finite = sum(1 for pt in points if pt["e_star"] is not None)
    return {
        "kind": "saturation",
        "title": f"saturation index e* across a/{d}",
        "blocks": [
            {
                "kind": "kv",
                "pairs": [
                    ("denominator d", str(d)),
                    ("numerators a coprime to d", str(len(points))),
                    ("with a finite saturation index e*", str(finite)),
                ],
            },
            {
                "kind": "table",
                "columns": ["a/d", "a mod d", "regime", "e* (minimal n)", "T", "deg S"],
                "rows": rows,
            },
            {
                "kind": "note",
                "text": (
                    "S | [n]_q iff e* = lcm(T) divides n, so the minimal saturating n "
                    "is e* and every multiple of it also works. The impossibility "
                    "residues (non-squarefree or non-cyclotomic S) divide no [n]_q, so "
                    "the equal-tail difference is never finite there."
                ),
            },
        ],
        "data": ex,
    }


def compute_degcollapse(d_max: int, a_max: int | None = None) -> Result:
    """deg S against the bound d-1 over the coprime grid, with the collapse depth.

    The diagonal deg S = d-1 is the saturating a == +/-1 locus (S = [d]_q); the
    drop below it is the collapse depth d-1-deg S, which for a squarefree S is
    the totient weight of the dropped Phi_k (the degree note's law). The depth_ok
    column cross-checks that equality on each squarefree cell.
    """
    from .factor import REGIME_LABELS, degree_collapse

    dc = degree_collapse(d_max, a_max)
    cells = dc["cells"]
    rows = [
        [
            f"{c['a']}/{c['d']}",
            f"{c['deg_S']}/{c['deg_bound']}",
            str(c["drop"]),
            str(c["totient_sum"]),
            _t_set_str(c["dropped"]) if c["dropped"] else "-",
            REGIME_LABELS[c["regime"]],
        ]
        for c in cells[:_ATLAS_MAX_ROWS]
    ]
    truncated = len(cells) > _ATLAS_MAX_ROWS
    saturating = sum(1 for c in cells if c["saturates_bound"])
    bound_str = f"d <= {d_max}" + (f", a <= {a_max}" if a_max else "")
    blocks: list[dict[str, Any]] = [
        {
            "kind": "kv",
            "pairs": [
                ("grid", f"{bound_str}, proper coprime fractions 0 < a/d < 1"),
                ("fractions", str(len(cells))),
                ("on the diagonal deg S = d-1 (S = [d]_q)", str(saturating)),
                (
                    "collapse-depth law drop = sum phi(dropped) on squarefree S",
                    "holds" if dc["depth_law_holds"] else "FAILS",
                ),
            ],
        },
        {
            "kind": "table",
            "columns": ["a/d", "deg S/(d-1)", "drop", "sum phi(dropped)", "dropped", "regime"],
            "rows": rows,
        },
    ]
    if truncated:
        blocks.append(
            {
                "kind": "note",
                "text": (
                    f"per-fraction table truncated to the first {_ATLAS_MAX_ROWS} of "
                    f"{len(cells)} fractions. Use --json for the full grid, or the web "
                    "collapse map for the scatter against the diagonal."
                ),
            }
        )
    blocks.append(
        {
            "kind": "note",
            "text": (
                "deg S <= d-1 always, with equality iff S = [d]_q iff a == +/-1 "
                "(mod d). Below the diagonal the drop d-1-deg S equals the totient "
                "weight of the dropped cyclotomic factors for a squarefree S (the "
                "5/12 drop of Phi_6 Phi_12 is 2+4 = 6, the 4/15 drop of Phi_15 is 8)."
            ),
        }
    )
    return {
        "kind": "degcollapse",
        "title": f"deg S vs d-1 collapse map  ({bound_str})",
        "blocks": blocks,
        "data": dc,
    }


def compute_qint(n: int) -> Result:
    """The q-integers [n]_q and [n]_{q^-1}."""
    a = sp.sympify(q_int(n))
    b = sp.sympify(q_int_qinv(n))
    return {
        "kind": "qint",
        "title": f"q-integer  [{n}]_q",
        "blocks": [
            {"kind": "poly", "label": f"[{n}]_q", "text": sp.sstr(a)},
            {"kind": "poly", "label": f"[{n}]_(q^-1)", "text": sp.sstr(b)},
            {
                "kind": "kv",
                "pairs": [("[n]_q at q = 1", f"{sp.simplify(a.subs(q, 1))}   (= {n})")],
            },
        ],
        "data": {"n": n, "q_int": sp.sstr(a), "q_int_qinv": sp.sstr(b)},
    }


def compute_coeffs(x: str, n: int) -> Result:
    """First n stable Taylor coefficients of [x]_q."""
    coeffs = q_real_truncated(x, n)
    return {
        "kind": "coeffs",
        "title": f"[{x}]_q  (first {n} coefficients)",
        "blocks": [
            {"kind": "poly", "label": f"[{x}]_q", "text": format_laurent(coeffs)},
            {
                "kind": "table",
                "columns": ["power q^k", "coefficient c_k"],
                "rows": [[f"q^{k}", str(c)] for k, c in enumerate(coeffs)],
            },
        ],
        "data": {"x": x, "n": n, "coefficients": coeffs},
    }


def compute_laurent(x: str, order: int) -> Result:
    """Laurent expansion of [x]_q through q^order, with its integer-part prefix."""
    coeffs = mgo_laurent(x, order)
    prefix = integer_part_prefix(x)
    floor_t = len(prefix) - 1
    return {
        "kind": "laurent",
        "title": f"[{x}]_q  (MGO Laurent expansion to q^{order})",
        "blocks": [
            {"kind": "poly", "label": f"[{x}]_q", "text": format_laurent(coeffs)},
            {
                "kind": "poly",
                "label": f"integer-part prefix (floor = {floor_t})",
                "text": format_laurent(prefix),
            },
        ],
        "data": {
            "x": x,
            "order": order,
            "coefficients": coeffs,
            "integer_part_prefix": prefix,
            "floor": floor_t,
        },
    }


def compute_prefix(x: str) -> Result:
    """The forced opening block [floor(x)]_q + 0*q^floor(x) of [x]_q."""
    prefix = integer_part_prefix(x)
    floor_t = len(prefix) - 1
    return {
        "kind": "prefix",
        "title": f"integer-part prefix of [{x}]_q",
        "blocks": [
            {"kind": "kv", "pairs": [("floor(x)", str(floor_t))]},
            {"kind": "poly", "label": "forced prefix", "text": format_laurent(prefix)},
            {
                "kind": "note",
                "text": (
                    "the first floor(x) coefficients are all 1 and the coefficient "
                    "at q^floor(x) is forced to 0; the fractional part can change only "
                    "higher powers"
                ),
            },
        ],
        "data": {"x": x, "floor": floor_t, "prefix": prefix},
    }


def compute_locked(x: str, n: int) -> Result:
    """How many Laurent coefficients the n-th convergent of x locks in."""
    terms = _cf_terms(x, n)
    if len(terms) < n:
        raise ValueError(
            f"x = {x} has only {len(terms)} continued-fraction terms; "
            f"pick a convergent index n <= {len(terms)}"
        )
    s_n, count = coeffs_locked_by_convergent(terms, n)
    return {
        "kind": "locked",
        "title": f"convergent locking for [{x}]_q",
        "blocks": [
            {
                "kind": "kv",
                "pairs": [(f"continued fraction (first {n} terms)", str(terms[:n]))],
            },
            {
                "kind": "kv",
                "pairs": [
                    ("partial sum S_n", str(s_n)),
                    ("coefficients locked in", str(count)),
                    ("first power that may differ", f"q^{s_n - 1}"),
                ],
            },
            {
                "kind": "note",
                "text": (
                    f"the {n}-th convergent agrees with [x]_q on q^0 through q^{s_n - 2}"
                ),
            },
        ],
        "data": {"x": x, "n": n, "cf_terms": terms[:n], "S_n": s_n, "locked": count},
    }


def compute_shift(x: str, order: int, direction: str) -> Result:
    """[x+1]_q (up) or [x-1]_q (down) from the coefficients of [x]_q."""
    coeffs = mgo_laurent(x, order)
    if direction == "up":
        shifted = shift_up(coeffs)
        label, formula = f"[{x} + 1]_q", "q*[x]_q + 1"
    else:
        shifted = shift_down(coeffs)
        label, formula = f"[{x} - 1]_q", "([x]_q - 1)/q"
    return {
        "kind": "shift",
        "title": f"shift {direction}:  {label} = {formula}",
        "blocks": [
            {"kind": "poly", "label": f"[{x}]_q", "text": format_laurent(coeffs)},
            {"kind": "poly", "label": label, "text": format_laurent(shifted)},
        ],
        "data": {
            "x": x,
            "order": order,
            "direction": direction,
            "input_coefficients": coeffs,
            "shifted_coefficients": shifted,
        },
    }


def compute_readouts(x: str, n: int) -> Result:
    """The pattern read-outs over the first n coefficients of [x]_q."""
    coeffs = q_real_truncated(x, n)
    first_nonzero = first_nonzero_coefficient_index(coeffs)
    first_negative = first_negative_coefficient_index(coeffs)
    max_abs = coefficient_max_abs(coeffs)
    zeros = number_of_zeros(coeffs)
    return {
        "kind": "readouts",
        "title": f"coefficient read-outs for [{x}]_q  (first {n})",
        "blocks": [
            {"kind": "poly", "label": f"[{x}]_q", "text": format_laurent(coeffs)},
            {
                "kind": "table",
                "columns": ["read-out", "value"],
                "rows": [
                    ["first nonzero coefficient index", _index_str(first_nonzero)],
                    ["first negative coefficient index", _index_str(first_negative)],
                    ["largest absolute coefficient", str(max_abs)],
                    ["number of zero coefficients", str(zeros)],
                ],
            },
        ],
        "data": {
            "x": x,
            "n": n,
            "coefficients": coeffs,
            "first_nonzero_index": first_nonzero,
            "first_negative_index": first_negative,
            "max_abs": max_abs,
            "zeros": zeros,
        },
    }


def compute_arith(x: str, y: str, n: int, op: str) -> Result:
    """The series sum [x]_q + [y]_q or product [x]_q * [y]_q (first n coeffs)."""
    if op == "mul":
        coeffs = q_mul(x, y, n)
        sym, name = "*", "product"
    else:
        coeffs = q_add(x, y, n)
        sym, name = "+", "sum"
    label = f"[{x}]_q {sym} [{y}]_q"
    return {
        "kind": "arith",
        "title": f"{label}  (series {name}, first {n} coefficients)",
        "blocks": [
            {"kind": "poly", "label": label, "text": format_laurent(coeffs)},
            {
                "kind": "note",
                "text": (
                    f"this is the {name} of the two q-series, not [{x} {sym} {y}]_q; "
                    "the MGO map x -> [x]_q is not a ring homomorphism"
                ),
            },
        ],
        "data": {"x": x, "y": y, "n": n, "op": op, "coefficients": coeffs},
    }


_QUAD_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def compute_quad_arith(x: str, y: str, op: str) -> Result:
    """Exact x op y for quadratic irrationals via the CF transfer matrix.

    Builds K = M_x (x) M_y, takes its dominant eigenvalue's eigenvector, and
    reads off the closed form (the golden+silver worked example). The result is
    cross-checked against direct arithmetic on x and y.
    """
    r = quad_arith(x, y, op)
    sym = _QUAD_SYMBOL[op]
    label = f"{x} {sym} {y}"
    return {
        "kind": "quad-arith",
        "title": f"{label}  (quadratic-irrational arithmetic)",
        "blocks": [
            {"kind": "poly", "label": label, "text": sp.sstr(r.value)},
            {
                "kind": "kv",
                "pairs": [
                    ("decimal", f"{r.decimal:.12g}"),
                    ("dominant eigenvalue of K", sp.sstr(r.dominant_eigenvalue)),
                    (
                        "matrix method verified against direct arithmetic",
                        "yes" if r.verified else "no",
                    ),
                ],
            },
            {
                "kind": "note",
                "text": (
                    f"transfer matrix K = M_x (x) M_y = {r.matrix.tolist()}; "
                    "the closed form is the read-out of its dominant (Perron) "
                    "eigenvector in the monomial basis (xy, x, y, 1)"
                ),
            },
        ],
        "data": {
            "x": x,
            "y": y,
            "op": op,
            "value": sp.sstr(r.value),
            "decimal": r.decimal,
            "verified": r.verified,
            "dominant_eigenvalue": sp.sstr(r.dominant_eigenvalue),
            "matrix": r.matrix.tolist(),
            "eigenvalues": [sp.sstr(e) for e in r.eigenvalues],
        },
    }


def compute_negation(x: str, n: int) -> Result:
    """The Jouteur negation [-x]_q, the sum [x]_q + [-x]_q, and its finiteness."""
    neg_v, neg_c = q_neg(x, n)
    sum_v, sum_c = negation_sum(x, n)
    finite = finite_xnegx(x)
    return {
        "kind": "negation",
        "title": f"q-negation  [-{x}]_q  and the x -> -x symmetry",
        "blocks": [
            {
                "kind": "poly",
                "label": f"[-{x}]_q",
                "text": _format_laurent_v(neg_v, neg_c),
            },
            {
                "kind": "poly",
                "label": f"[{x}]_q + [-{x}]_q",
                "text": _format_laurent_v(sum_v, sum_c),
            },
            {
                "kind": "kv",
                "pairs": [
                    (
                        "[x]_q + [-x]_q finite (Laurent polynomial)?",
                        "yes" if finite else "no",
                    ),
                ],
            },
            {
                "kind": "note",
                "text": (
                    "negation is the Jouteur PGL_2(Z) action (arXiv:2503.02122), an "
                    "involution; the sum is finite exactly for pure square roots "
                    "(trace-zero quadratics), Ovsienko Example 6.4"
                ),
            },
        ],
        "data": {
            "x": x,
            "n": n,
            "neg_valuation": neg_v,
            "neg_coefficients": neg_c,
            "sum_valuation": sum_v,
            "sum_coefficients": sum_c,
            "finite": finite,
        },
    }


def compute_deficit(x: str, y: str, n: int, op: str) -> Result:
    """The deficit [x op y]_q - (series sum or product) for op '+' or '*'."""
    sym = "*" if op in ("*", "mul") else "+"
    d = deficit(x, y, sym, n)
    target_label = f"[{x} {sym} {y}]_q"
    engine_label = f"[{x}]_q {sym} [{y}]_q"
    engine_name = "product" if sym == "*" else "sum"
    blocks: list[dict[str, Any]] = [
        {"kind": "poly", "label": f"[{x}]_q", "text": format_laurent(d.x_series)},
        {"kind": "poly", "label": f"[{y}]_q", "text": format_laurent(d.y_series)},
        {
            "kind": "poly",
            "label": f"engine value  {engine_label}  (series {engine_name})",
            "text": format_laurent(d.engine),
        },
        {
            "kind": "poly",
            "label": f"target  {target_label}  (q-series of the real {x} {sym} {y})",
            "text": format_laurent(d.target),
        },
        {
            "kind": "poly",
            "label": f"deficit  {target_label} - ({engine_label})",
            "text": format_laurent(d.deficit),
        },
    ]
    if d.exact is not None:
        blocks.append(
            {
                "kind": "poly",
                "label": "deficit (exact closed form in q)",
                "text": sp.sstr(d.exact),
            }
        )
    q1_value = (
        str(d.deficit_at_q1)
        if d.deficit_at_q1 is not None
        else "0 in closed form (a truncated series is not summable at q = 1)"
    )
    blocks.append(
        {
            "kind": "kv",
            "pairs": [
                ("deficit at q = 1", q1_value),
                ("deficit at q = 0", str(d.deficit_at_q0)),
            ],
        }
    )
    blocks.append(
        {
            "kind": "note",
            "text": (
                f"the engine value is the series {engine_name} {engine_label}, not "
                f"{target_label}; the MGO map x -> [x]_q is not a ring homomorphism. "
                "at q = 1 both sides collapse to the ordinary value so the deficit is "
                "0; at q = 0 the gap theorem forces it to -1 for a sum of x, y >= 1"
            ),
        }
    )
    return {
        "kind": "deficit",
        "title": f"deficit  {target_label} - {engine_label}  (first {n} coefficients)",
        "blocks": blocks,
        "data": {
            "x": x,
            "y": y,
            "n": n,
            "op": sym,
            "x_series": d.x_series,
            "y_series": d.y_series,
            "engine": d.engine,
            "target": d.target,
            "deficit": d.deficit,
            "exact": None if d.exact is None else sp.sstr(d.exact),
            "deficit_at_q1": d.deficit_at_q1,
            "deficit_at_q0": d.deficit_at_q0,
        },
    }


def compute_negsum(x: str, n: int) -> Result:
    """The negation sum [x]_q + [-x]_q for one x and whether it is finite."""
    panel = negation_panel(x, n)
    return {
        "kind": "negsum",
        "title": f"negation sum  [{x}]_q + [-{x}]_q  (Ovsienko Example 6.4)",
        "blocks": [
            {
                "kind": "poly",
                "label": f"[{x}]_q + [-{x}]_q",
                "text": _format_laurent_v(panel.valuation, panel.sum_coeffs),
            },
            {
                "kind": "kv",
                "pairs": [
                    (
                        "[x]_q + [-x]_q finite (Laurent polynomial)?",
                        "yes" if panel.finite else "no",
                    ),
                ],
            },
            {
                "kind": "note",
                "text": (
                    "the sum is finite exactly for pure square roots (trace-zero "
                    "quadratics), where -x is the Galois conjugate and the sum is the "
                    "q-trace; otherwise it is a non-terminating Laurent series. "
                    "Ovsienko Example 6.4, with the Jouteur negation arXiv:2503.02122"
                ),
            },
        ],
        "data": {
            "x": x,
            "n": n,
            "valuation": panel.valuation,
            "sum_coefficients": panel.sum_coeffs,
            "finite": panel.finite,
        },
    }


def compute_radius(x: str, n: int) -> Result:
    """The running-max estimate of the radius of convergence of [x]_q."""
    r = radius(x, n)
    return {
        "kind": "radius",
        "title": f"radius of convergence estimate for [{x}]_q  (N = {n})",
        "blocks": [
            {
                "kind": "kv",
                "pairs": [
                    ("estimate", "infinite" if r == float("inf") else f"{r:.6f}")
                ],
            },
            {
                "kind": "note",
                "text": (
                    "running-max root-test slope: exp(-max_k (ln|c_k|)/k). It is "
                    "biased high and decreases toward the true radius as N grows"
                ),
            },
        ],
        "data": {
            "x": x,
            "n": n,
            "radius": (None if r == float("inf") else r),
            "infinite": r == float("inf"),
        },
    }


def compute_oeis(sequence: str, do_modp: bool = True, do_bfile: bool = True) -> Result:
    """Look a coefficient sequence up in the OEIS, re-verified against b-files."""
    from . import oeis as _oeis

    seq = _oeis.parse_sequence(sequence)
    res = _oeis.lookup(seq, do_modp=do_modp, do_bfile=do_bfile)
    n = len(seq)
    preview = ", ".join(str(t) for t in seq[:12]) + (", ..." if n > 12 else "")
    blocks: list[dict[str, Any]] = [
        {"kind": "kv", "pairs": [(f"input ({n} terms)", preview)]}
    ]
    if not res.hits:
        blocks.append(
            {
                "kind": "note",
                "text": (
                    "no OEIS match for this sequence (or OEIS was unreachable). "
                    "responses are cached on disk, so a repeat runs offline"
                ),
            }
        )
    else:
        rows = []
        for i, h in enumerate(res.hits, 1):
            sign = "-" if h.transform == "identity" else h.transform
            if h.fully_verified:
                bfile = f"all {n} ok"
            elif h.diverged:
                bfile = f"diverges at term {h.diverge_term}"
            elif h.bfile_checked:
                bfile = "partial"
            else:
                bfile = "-"
            rows.append(
                [str(i), h.anum, f"{h.prefix_len}/{n}", sign, bfile, h.name[:48]]
            )
        blocks.append(
            {
                "kind": "table",
                "columns": ["#", "A-number", "prefix", "sign", "b-file", "name"],
                "rows": rows,
            }
        )
    if res.modp_hits:
        modp_rows = []
        for p in sorted(res.modp_hits):
            for h in res.modp_hits[p]:
                modp_rows.append([f"mod {p}", h.anum, str(h.prefix_len), h.name[:48]])
        blocks.append(
            {
                "kind": "table",
                "columns": ["reduction", "A-number", "prefix", "name"],
                "rows": modp_rows,
            }
        )
    blocks.append(
        {
            "kind": "note",
            "text": (
                "ranked by matching-prefix length with signs reconciled; the top "
                "hits are re-checked against the full b-file. needs the requests "
                "extra (pip install qreals[oeis])"
            ),
        }
    )
    return {
        "kind": "oeis",
        "title": f"OEIS lookup  ({n} terms)",
        "blocks": blocks,
        "data": _oeis.result_as_dict(res),
    }


def _fmt_feature(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.4f}"


def compute_fingerprint(
    x: str,
    n_cf: int = features.DEFAULT_N_CF,
    n_coeffs: int = features.DEFAULT_N_COEFFS,
    n_radius: int = features.DEFAULT_N_RADIUS,
) -> Result:
    """A named, fixed-length, deterministic fingerprint of [x]_q."""
    fp = features.featurize(x, n_cf=n_cf, n_coeffs=n_coeffs, n_radius=n_radius)
    mapping = fp.as_dict()
    scalar_names = features.feature_names(n_cf, n_coeffs, n_radius)[:15]
    scalar_rows = [[name, _fmt_feature(mapping[name])] for name in scalar_names]
    cf = [int(mapping[f"cf_{i}"]) for i in range(n_cf)]
    cf_sums = [int(mapping[f"cf_partial_sum_{i}"]) for i in range(1, n_cf + 1)]
    coeffs = [int(mapping[f"c_{i}"]) for i in range(n_coeffs)]
    return {
        "kind": "fingerprint",
        "title": f"fingerprint of [{x}]_q  ({len(fp.values)} features)",
        "blocks": [
            {"kind": "table", "columns": ["feature", "value"], "rows": scalar_rows},
            {
                "kind": "kv",
                "pairs": [
                    ("partial quotients (cf)", str(cf)),
                    ("partial sums", str(cf_sums)),
                    ("coefficients c_0..", str(coeffs)),
                ],
            },
            {
                "kind": "note",
                "text": (
                    "a deterministic feature vector of the same length for every x "
                    "at these settings, for nearest-neighbour over constants; raw "
                    "coefficients can dominate Euclidean distance, so normalise for "
                    "a shape-based comparison"
                ),
            },
        ],
        "data": {
            "x": fp.x,
            "params": list(fp.params),
            "names": fp.names,
            "values": fp.values,
            "features": mapping,
        },
    }


def _format_laurent_v(valuation: int, coeffs: list[int]) -> str:
    """Render a Laurent result (valuation, coeffs) as a readable q-polynomial."""
    terms: list[str] = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        power = valuation + i
        mag = abs(c)
        if power == 0:
            mono = f"{mag}"
        elif power == 1:
            mono = "q" if mag == 1 else f"{mag}*q"
        else:
            mono = f"q^{power}" if mag == 1 else f"{mag}*q^{power}"
        sign = "-" if c < 0 else "+"
        terms.append(f"{sign} {mono}")
    if not terms:
        body = "0"
    else:
        body = terms[0].replace("+ ", "") if terms[0].startswith("+") else terms[0]
        body = " ".join([body] + terms[1:])
    return f"{body} + O(q^{valuation + len(coeffs)})"


def _index_str(i: int) -> str:
    return "none" if i == -1 else str(i)


def _cf_terms(x_str: str, count: int) -> list[int]:
    """First `count` regular continued-fraction terms of x (fewer if it ends)."""
    x = parse_real(x_str)
    terms: list[int] = []
    for term in sp.continued_fraction_iterator(x):
        terms.append(int(term))
        if len(terms) >= count:
            break
    return terms


# --------------------------------------------------------------------------
# Rendering. rich is used when importable; otherwise builtins.
# --------------------------------------------------------------------------


def _make_console() -> Any | None:
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return None


def render_result(
    result: Result,
    console: Any | None = None,
    as_json: bool = False,
    verify_result: bool = True,
) -> None:
    """Render a result and, by default, print the inline verification stamp.

    The stamp runs the cheap cross-checks for this input and prints one line. It
    writes nothing and needs no extras. In JSON mode the stamp is folded into
    the payload under "verification" instead of printed, so the stream stays
    valid JSON.
    """
    stamp = _stamp_for(result) if verify_result else None
    if as_json:
        payload = dict(result["data"])
        if stamp is not None and stamp.checks:
            payload["verification"] = stamp.as_dict()
        print(json.dumps(payload, indent=2))
        return
    if console is not None:
        _render_rich(result, console)
    else:
        _render_plain(result)
    if stamp is not None and stamp.checks:
        line = stamp.line()
        if console is not None:
            from rich.text import Text

            style = "green" if stamp.ok else "yellow"
            console.print(Text(line, style=style))
        else:
            print(line)


def _stamp_for(result: Result) -> Any | None:
    """Verify a computation result, returning None for non-computation screens."""
    if not isinstance(result, dict) or "kind" not in result:
        return None
    from .verify import verify

    try:
        return verify(result)
    except Exception:  # noqa: BLE001 - a stamp must never break the result
        return None


def _render_rich(result: Result, console: Any) -> None:
    # Titles and results carry q-polynomials with [..] in them; rich reads [..]
    # as style markup, so every dynamic string goes through Text to render
    # literally rather than vanishing.
    from rich.table import Table
    from rich.text import Text

    console.print()
    console.rule(Text(result["title"], style="bold cyan"))
    for block in result["blocks"]:
        kind = block["kind"]
        if kind == "poly":
            line = Text()
            line.append(f"{block['label']} = ", style="bold")
            line.append(block["text"], style="green")
            console.print(line)
        elif kind == "kv":
            for label, value in block["pairs"]:
                line = Text()
                line.append(f"{label}: ", style="bold")
                line.append(str(value))
                console.print(line)
        elif kind == "table":
            table = Table(show_header=True, header_style="bold")
            for column in block["columns"]:
                table.add_column(column)
            for row in block["rows"]:
                table.add_row(*[Text(str(cell)) for cell in row])
            console.print(table)
        elif kind == "note":
            console.print(Text(block["text"], style="dim italic"))
    console.print()


def _render_plain(result: Result) -> None:
    print()
    print(f"== {result['title']} ==")
    for block in result["blocks"]:
        kind = block["kind"]
        if kind == "poly":
            print(f"{block['label']} = {block['text']}")
        elif kind == "kv":
            for label, value in block["pairs"]:
                print(f"{label}: {value}")
        elif kind == "table":
            widths = [len(c) for c in block["columns"]]
            for row in block["rows"]:
                for i, cell in enumerate(row):
                    widths[i] = max(widths[i], len(str(cell)))
            header = "  ".join(
                c.ljust(widths[i]) for i, c in enumerate(block["columns"])
            )
            print(header)
            print("  ".join("-" * widths[i] for i in range(len(widths))))
            for row in block["rows"]:
                print(
                    "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
                )
        elif kind == "note":
            print(f"note: {block['text']}")
    print()


# --------------------------------------------------------------------------
# Capability registry. Each capability ties together a menu label, the
# headless subcommand name, the interactive prompt, and the computation.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Capability:
    key: str  # subcommand name, e.g. "rational"
    title: str  # menu label
    summary: str  # one-line description
    prompt: Callable[[Any], dict[str, Any] | None]  # interactive input gathering
    compute: Callable[..., Result]  # pure computation, takes the prompt's dict


def _ask_real(qst: Any, message: str, example: str, default: str = "") -> str | None:
    answer = qst.text(
        f"{message}  (for example {example})",
        default=default,
        validate=_validate_real,
    ).ask()
    return None if answer is None else answer.strip()


def _ask_int(
    qst: Any,
    message: str,
    default: str,
    *,
    low: int | None = None,
    nonzero: bool = False,
) -> int | None:
    answer = qst.text(
        message, default=default, validate=_make_int_validator(low=low, nonzero=nonzero)
    ).ask()
    return None if answer is None else int(answer)


def _prompt_rational(qst: Any) -> dict[str, Any] | None:
    p = _ask_int(qst, "numerator p", "3")
    if p is None:
        return None
    s = _ask_int(qst, "denominator s (nonzero)", "2", nonzero=True)
    if s is None:
        return None
    return {"p": p, "s": s}


def _prompt_jumpgap(qst: Any) -> dict[str, Any] | None:
    answer = qst.text(
        "rational p/s  (for example 3/5)",
        default="3/5",
        validate=_validate_rational,
    ).ask()
    if answer is None:
        return None
    p, s = _parse_rational(answer.strip())
    return {"p": p, "s": s}


def _prompt_factor(qst: Any) -> dict[str, Any] | None:
    answer = qst.text(
        "rational a/b  (for example 7/5)",
        default="7/5",
        validate=_validate_rational,
    ).ask()
    if answer is None:
        return None
    a, b = _parse_rational(answer.strip())
    return {"a": a, "b": b}


def _prompt_sprops(qst: Any) -> dict[str, Any] | None:
    answer = qst.text(
        "rational a/d  (for example 5/12)",
        default="5/12",
        validate=_validate_rational,
    ).ask()
    if answer is None:
        return None
    a, b = _parse_rational(answer.strip())
    return {"a": a, "b": b}


def _prompt_satlas(qst: Any) -> dict[str, Any] | None:
    d_max = _ask_int(qst, "max denominator d", "12", low=2)
    if d_max is None:
        return None
    a_max = _ask_int(qst, "max numerator a (0 for no cap)", "0", low=0)
    if a_max is None:
        return None
    return {"d_max": d_max, "a_max": a_max or None}


def _prompt_saturation(qst: Any) -> dict[str, Any] | None:
    d = _ask_int(qst, "denominator d", "12", low=2)
    return None if d is None else {"d": d}


def _prompt_degcollapse(qst: Any) -> dict[str, Any] | None:
    d_max = _ask_int(qst, "max denominator d", "12", low=2)
    if d_max is None:
        return None
    a_max = _ask_int(qst, "max numerator a (0 for no cap)", "0", low=0)
    if a_max is None:
        return None
    return {"d_max": d_max, "a_max": a_max or None}


def _prompt_qint(qst: Any) -> dict[str, Any] | None:
    n = _ask_int(qst, "integer n", "5")
    return None if n is None else {"n": n}


def _prompt_coeffs(qst: Any) -> dict[str, Any] | None:
    x = _ask_real(qst, "real number x", "pi", default="pi")
    if x is None:
        return None
    n = _ask_int(qst, "how many coefficients N", "12", low=1)
    return None if n is None else {"x": x, "n": n}


def _prompt_laurent(qst: Any) -> dict[str, Any] | None:
    x = _ask_real(qst, "real number x", "pi", default="pi")
    if x is None:
        return None
    order = _ask_int(qst, "highest power q^order", "12", low=0)
    return None if order is None else {"x": x, "order": order}


def _prompt_prefix(qst: Any) -> dict[str, Any] | None:
    x = _ask_real(qst, "real number x (>= 0)", "pi", default="pi")
    return None if x is None else {"x": x}


def _prompt_locked(qst: Any) -> dict[str, Any] | None:
    x = _ask_real(qst, "real number x", "pi", default="pi")
    if x is None:
        return None
    n = _ask_int(qst, "convergent index n", "2", low=1)
    return None if n is None else {"x": x, "n": n}


def _prompt_shift(qst: Any) -> dict[str, Any] | None:
    direction = qst.select(
        "which shift?",
        choices=["up: [x+1]_q = q*[x]_q + 1", "down: [x-1]_q = ([x]_q - 1)/q"],
    ).ask()
    if direction is None:
        return None
    direction = "up" if direction.startswith("up") else "down"
    default_x = "pi" if direction == "down" else "pi-2"
    note = (
        "x (>= 1 so the constant term is 1)" if direction == "down" else "real number x"
    )
    x = _ask_real(qst, note, "pi", default=default_x)
    if x is None:
        return None
    order = _ask_int(qst, "highest power q^order", "12", low=1)
    return None if order is None else {"x": x, "order": order, "direction": direction}


def _prompt_readouts(qst: Any) -> dict[str, Any] | None:
    x = _ask_real(qst, "real number x", "sqrt(2)", default="sqrt(2)")
    if x is None:
        return None
    n = _ask_int(qst, "how many coefficients N", "30", low=1)
    return None if n is None else {"x": x, "n": n}


def _prompt_arith(qst: Any) -> dict[str, Any] | None:
    op_choice = qst.select(
        "which operation?",
        choices=["add: [x]_q + [y]_q", "mul: [x]_q * [y]_q"],
    ).ask()
    if op_choice is None:
        return None
    op = "mul" if op_choice.startswith("mul") else "add"
    x = _ask_real(qst, "real number x (>= 0)", "3/2", default="3/2")
    if x is None:
        return None
    y = _ask_real(qst, "real number y (>= 0)", "13/5", default="13/5")
    if y is None:
        return None
    n = _ask_int(qst, "how many coefficients N", "12", low=1)
    return None if n is None else {"x": x, "y": y, "n": n, "op": op}


def _prompt_quad_arith(qst: Any) -> dict[str, Any] | None:
    op_choice = qst.select(
        "which operation?",
        choices=["add: x + y", "sub: x - y", "mul: x * y", "div: x / y"],
    ).ask()
    if op_choice is None:
        return None
    op = op_choice.split(":", 1)[0]
    x = _ask_real(qst, "quadratic irrational x", "(1 + sqrt(5))/2", default="(1 + sqrt(5))/2")
    if x is None:
        return None
    y = _ask_real(qst, "quadratic irrational y", "1 + sqrt(2)", default="1 + sqrt(2)")
    return None if y is None else {"x": x, "y": y, "op": op}


def _prompt_negation(qst: Any) -> dict[str, Any] | None:
    x = _ask_real(qst, "real number x (>= 0)", "sqrt(2)", default="sqrt(2)")
    if x is None:
        return None
    n = _ask_int(qst, "how many coefficients N", "12", low=1)
    return None if n is None else {"x": x, "n": n}


def _prompt_deficit(qst: Any) -> dict[str, Any] | None:
    op_choice = qst.select(
        "which operation?",
        choices=["add: [x + y]_q vs [x]_q + [y]_q", "mul: [x*y]_q vs [x]_q * [y]_q"],
    ).ask()
    if op_choice is None:
        return None
    op = "mul" if op_choice.startswith("mul") else "add"
    x = _ask_real(qst, "real number x (>= 0)", "3/2", default="3/2")
    if x is None:
        return None
    y = _ask_real(qst, "real number y (>= 0)", "5/2", default="5/2")
    if y is None:
        return None
    n = _ask_int(qst, "how many coefficients N", "12", low=1)
    return None if n is None else {"x": x, "y": y, "n": n, "op": op}


def _prompt_negsum(qst: Any) -> dict[str, Any] | None:
    x = _ask_real(qst, "real number x (>= 0)", "sqrt(2)", default="sqrt(2)")
    if x is None:
        return None
    n = _ask_int(qst, "how many coefficients N", "12", low=1)
    return None if n is None else {"x": x, "n": n}


def _prompt_radius(qst: Any) -> dict[str, Any] | None:
    x = _ask_real(qst, "real number x (>= 0)", "pi", default="pi")
    if x is None:
        return None
    n = _ask_int(qst, "how many coefficients N", "60", low=2)
    return None if n is None else {"x": x, "n": n}


def _validate_sequence(text: str) -> bool | str:
    from . import oeis as _oeis

    try:
        seq = _oeis.parse_sequence(text)
    except ValueError as exc:
        return str(exc)
    return True if seq else "enter at least one integer"


def _prompt_oeis(qst: Any) -> dict[str, Any] | None:
    answer = qst.text(
        "coefficient sequence to look up  (for example 1,1,2,5,14,42,132,429)",
        default="1,1,2,5,14,42,132,429",
        validate=_validate_sequence,
    ).ask()
    return None if answer is None else {"sequence": answer.strip()}


def _prompt_fingerprint(qst: Any) -> dict[str, Any] | None:
    x = _ask_real(qst, "real number x", "pi", default="pi")
    if x is None:
        return None
    n = _ask_int(
        qst,
        "how many coefficients in the fingerprint",
        str(features.DEFAULT_N_COEFFS),
        low=1,
    )
    return None if n is None else {"x": x, "n_coeffs": n}


CAPABILITIES: list[Capability] = [
    Capability(
        "rational",
        "Exact q-rational  [p/s]_q",
        "the exact rational function in q for a fraction p/s",
        _prompt_rational,
        compute_rational,
    ),
    Capability(
        "jumpgap",
        "Gap between a rational's two q-versions",
        "the right and left versions of p/s and the factored gap between them",
        _prompt_jumpgap,
        compute_jumpgap,
    ),
    Capability(
        "factor",
        "Factor R(q), S(q) of [a/b]_q",
        "factor the numerator and denominator over Z[q], labelling Phi_d factors",
        _prompt_factor,
        compute_factor,
    ),
    Capability(
        "sprops",
        "Denominator S(q) properties of [a/d]_q",
        "cyclotomic factors of S, saturation index e*, deg S vs d-1, S(1)=d, "
        "collapse vs full [d]_q",
        _prompt_sprops,
        compute_sprops,
    ),
    Capability(
        "satlas",
        "S(q) cyclotomic-factor atlas (a/d grid)",
        "regime of S(q) (full [d]_q / collapse / non-squarefree / non-cyclotomic) "
        "over a coprime (a, d) grid, with the Phi_k appearance tally",
        _prompt_satlas,
        compute_satlas,
    ),
    Capability(
        "saturation",
        "Saturation index e* across a/d (fixed d)",
        "e* = lcm(T) and the minimal saturating n as a ranges over the residues "
        "coprime to a fixed d, flagging the impossibility residues",
        _prompt_saturation,
        compute_saturation,
    ),
    Capability(
        "degcollapse",
        "deg S vs d-1 collapse map",
        "deg S against the bound d-1 over the grid, with the collapse depth "
        "d-1-deg S = totient weight of the dropped Phi_k",
        _prompt_degcollapse,
        compute_degcollapse,
    ),
    Capability(
        "qint",
        "q-integer  [n]_q",
        "the q-analog of a whole number, [n]_q and [n]_(q^-1)",
        _prompt_qint,
        compute_qint,
    ),
    Capability(
        "coeffs",
        "q-real coefficients  [x]_q",
        "the first N stable Taylor coefficients of [x]_q for any real x",
        _prompt_coeffs,
        compute_coeffs,
    ),
    Capability(
        "laurent",
        "Laurent expansion of [x]_q (MGO)",
        "[x]_q written out to a chosen power, with its integer-part prefix",
        _prompt_laurent,
        compute_laurent,
    ),
    Capability(
        "prefix",
        "Integer-part prefix of [x]_q",
        "the forced opening block [floor(x)]_q then a 0 at q^floor(x)",
        _prompt_prefix,
        compute_prefix,
    ),
    Capability(
        "locked",
        "Convergent locking",
        "how many coefficients the n-th convergent of x pins down",
        _prompt_locked,
        compute_locked,
    ),
    Capability(
        "shift",
        "Shift relations  [x +/- 1]_q",
        "move the argument by one: [x+1]_q = q*[x]_q + 1, [x-1]_q = ([x]_q - 1)/q",
        _prompt_shift,
        compute_shift,
    ),
    Capability(
        "readouts",
        "Coefficient read-outs",
        "first nonzero, first negative, largest size, and zero count",
        _prompt_readouts,
        compute_readouts,
    ),
    Capability(
        "arith",
        "Arithmetic  [x]_q +/* [y]_q",
        "the series sum or product of two q-reals (not [x +/* y]_q)",
        _prompt_arith,
        compute_arith,
    ),
    Capability(
        "quad",
        "Quadratic-irrational arithmetic  x +/-/*// y",
        "exact x op y for quadratic irrationals via the CF transfer matrix "
        "(the golden+silver worked example), cross-checked against direct arithmetic",
        _prompt_quad_arith,
        compute_quad_arith,
    ),
    Capability(
        "deficit",
        "Deficit of two q-reals",
        "how far [x]_q +/* [y]_q sits from [x +/* y]_q, with the q=1 and q=0 checks",
        _prompt_deficit,
        compute_deficit,
    ),
    Capability(
        "negate",
        "q-negation  [-x]_q  and x -> -x",
        "the Jouteur negation and whether [x]_q + [-x]_q is finite (Ex. 6.4)",
        _prompt_negation,
        compute_negation,
    ),
    Capability(
        "negsum",
        "Negation sum  [x]_q + [-x]_q",
        "the x -> -x sum and whether it is finite (Ovsienko Ex. 6.4)",
        _prompt_negsum,
        compute_negsum,
    ),
    Capability(
        "radius",
        "Radius of convergence of [x]_q",
        "the running-max slope estimate, biased high at finite N",
        _prompt_radius,
        compute_radius,
    ),
    Capability(
        "oeis",
        "Look this sequence up in OEIS",
        "search the OEIS for a coefficient sequence, re-verified against b-files",
        _prompt_oeis,
        compute_oeis,
    ),
    Capability(
        "fingerprint",
        "Fingerprint a constant",
        "a named, fixed-length feature vector of [x]_q for nearest-neighbour",
        _prompt_fingerprint,
        compute_fingerprint,
    ),
]

CAPABILITY_BY_KEY: dict[str, Capability] = {c.key: c for c in CAPABILITIES}


_DOCTOR_LABEL = "Doctor / environment check"
_SAVED_LABEL = "My saved list"


def build_menu_choices() -> list[str]:
    """The main-menu labels: capabilities, the saved list, Doctor, Help, Quit."""
    return [c.title for c in CAPABILITIES] + [
        _SAVED_LABEL,
        _DOCTOR_LABEL,
        "Help / About",
        "Quit",
    ]


# --------------------------------------------------------------------------
# Interactive loop.
# --------------------------------------------------------------------------

_BANNER = r"""
   __ _ _ __ ___  __ _ | |___
  / _` | '__/ _ \/ _` || / __|     q-deformed rationals and reals
 | (_| | | |  __/ (_| || \__ \     via MGO continued fractions
  \__, |_|  \___|\__,_||_|___/
     |_|
"""


def _help_result() -> Result:
    rows = [[c.title, c.summary] for c in CAPABILITIES]
    return {
        "title": f"qreals {__version__}  -  help and about",
        "blocks": [
            {
                "kind": "note",
                "text": (
                    "qreals computes the q-analog [x]_q of a number, after "
                    "Morier-Genoud and Ovsienko, Forum Math. Sigma 8 (2020). "
                    "Pick a capability, answer one prompt at a time, read the result."
                ),
            },
            {"kind": "table", "columns": ["capability", "what it does"], "rows": rows},
            {
                "kind": "note",
                "text": (
                    "for scripts: qreals rational 3 2, qreals coeffs pi 12, "
                    "qreals laurent pi --order 12 (add --json for machine output)"
                ),
            },
            {
                "kind": "note",
                "text": (
                    "every result prints a one-line verification stamp by default. "
                    "qreals certify rational 3 2 prints the full derivation (--save "
                    "writes a .tex, --pdf opens one). qreals doctor checks this "
                    "environment."
                ),
            },
            {
                "kind": "note",
                "text": (
                    "keep results in a personal list that survives across sessions "
                    "('My saved list'), and export one result or the whole list to "
                    "JSON, CSV, a LaTeX table, or Magma. Headless: qreals batch "
                    '"pi,sqrt(2),3/2" --order 12 --format magma -o out.m, and '
                    "qreals export --format csv -o saved.csv. Files are written only "
                    "when you export."
                ),
            },
        ],
        "data": {
            "version": __version__,
            "capabilities": {c.key: c.summary for c in CAPABILITIES},
        },
    }


def _import_questionary() -> Any | None:
    try:
        import questionary

        return questionary
    except ImportError:
        return None


def _is_interactive() -> bool:
    """True when both ends are a real terminal, as arrow keys need."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (ValueError, AttributeError):
        return False


def run_interactive() -> int:
    """The arrow-key menu. Returns a process exit code."""
    qst = _import_questionary()
    console = _make_console()
    if qst is None:
        _say(
            console,
            "The interactive menu needs the optional interface extras.\n"
            "Install them with:  pip install qreals[app]\n"
            "Or use a subcommand, for example:  qreals coeffs pi 12",
        )
        return 1
    if not _is_interactive():
        _say(
            console,
            "The interactive menu needs a terminal for arrow-key navigation.\n"
            "Run qreals directly in your terminal, or use a subcommand, "
            "for example:  qreals coeffs pi 12   (try qreals --help).",
        )
        return 1

    if console is not None:
        console.print(_BANNER, style="bold cyan")
    else:
        print(_BANNER)

    try:
        while True:
            choice = qst.select(
                "What would you like to compute?",
                choices=build_menu_choices(),
            ).ask()
            if choice is None or choice == "Quit":
                _say(console, "Bye.")
                return 0
            if choice == _SAVED_LABEL:
                _saved_menu(qst, console)
                continue
            if choice == _DOCTOR_LABEL:
                run_doctor(console)
                continue
            if choice == "Help / About":
                render_result(_help_result(), console)
                continue
            capability = next(c for c in CAPABILITIES if c.title == choice)
            _run_capability(capability, qst, console)
    except (KeyboardInterrupt, EOFError):
        _say(console, "\nBye.")
        return 0


_SHOW = "Show the derivation (terminal, keeps no file)"
_PDF = "View it as a PDF (opens, keeps no file)"
_SAVE = "Save a certificate here (writes a file)"
_ADD_SAVED = "Add this result to my saved list"
_EXPORT_ONE = "Export this result (writes a file)"
_AGAIN = "Do another"
_BACK = "Back to menu"


def _run_capability(capability: Capability, qst: Any, console: Any | None) -> None:
    while True:
        _say(console, f"\n{capability.title}  -  {capability.summary}")
        try:
            params = capability.prompt(qst)
        except KeyboardInterrupt:
            return
        if params is None:
            return
        try:
            result = capability.compute(**params)
        except Exception as exc:  # surface the math error, stay in the menu
            _say(console, f"could not compute: {exc}")
            nxt = qst.select("Next?", choices=[_AGAIN, _BACK]).ask()
            if nxt != _AGAIN:
                return
            continue
        render_result(result, console)
        if _post_compute_menu(result, qst, console) == "back":
            return


def _post_compute_menu(result: Result, qst: Any, console: Any | None) -> str:
    """After a computation: keep it, export it, derive it, or move on."""
    while True:
        nxt = qst.select(
            "Next?",
            choices=[
                _ADD_SAVED,
                _EXPORT_ONE,
                _SHOW,
                _PDF,
                _SAVE,
                _AGAIN,
                _BACK,
            ],
        ).ask()
        if nxt is None or nxt == _BACK:
            return "back"
        if nxt == _AGAIN:
            return "again"
        if nxt == _ADD_SAVED:
            _add_result_to_saved(result, console)
        elif nxt == _EXPORT_ONE:
            _export_one(result, qst, console)
        elif nxt == _SHOW:
            _certificate_action(result, console, "show")
        elif nxt == _PDF:
            _certificate_action(result, console, "pdf")
        elif nxt == _SAVE:
            _certificate_action(result, console, "save")


def _certificate_action(result: Result, console: Any | None, what: str) -> None:
    certificate = _load_certificate()
    if certificate is None:
        _say(
            console,
            "Certificates need the proof extras. Install them with:  "
            "pip install qreals[proof]",
        )
        return
    try:
        cert = certificate.build_certificate(result)
    except Exception as exc:  # noqa: BLE001 - report, stay in the menu
        _say(console, f"could not build a certificate for this result: {exc}")
        return
    if what == "show":
        cert.render_terminal(console)
    elif what == "pdf":
        cert.view_pdf(console)
    elif what == "save":
        # An explicit save also records in .qprov when qprov is importable, so
        # the certificate can be cited later; the id is printed so it is visible.
        written = cert.save(".", qprov=True)
        _say(console, f"wrote {written['tex']}")
        if written["pdf"]:
            _say(console, f"wrote {written['pdf']}")
        else:
            _say(
                console,
                "no TeX engine found; compile the .tex with pdflatex to get a PDF",
            )
        if written["qprov_id"]:
            _say(console, f"recorded in your .qprov store as {written['qprov_id']}")


def _load_certificate() -> Any | None:
    try:
        from . import certificate

        return certificate
    except Exception:  # noqa: BLE001 - proof extras absent
        return None


def _say(console: Any | None, message: str) -> None:
    if console is not None:
        console.print(message)
    else:
        print(message)


# --------------------------------------------------------------------------
# Saved list and exports. A computed result can be kept in a personal list that
# persists across sessions in the per-user data directory, and any single result
# or the whole list can be written out as JSON, CSV, a LaTeX table, or Magma.
# Nothing here writes until the user asks to add an item or export.
# --------------------------------------------------------------------------


def entry_from_result(result: Result) -> SavedEntry:
    """Build a saved entry from a computation result that carries a q-series.

    Supports the kinds with a q-coefficient list: coeffs, laurent, readouts and
    arith (the q^0.. series), shift (the shifted series), and negate (the q-
    negation, with its valuation). Other kinds raise, so the caller can report
    that there is nothing to keep.
    """
    data = result.get("data", {})
    kind = str(result.get("kind", "coeffs"))
    x = str(data.get("x", ""))
    valuation = 0
    if kind == "shift":
        coeffs = data.get("shifted_coefficients")
        delta = "+ 1" if data.get("direction") == "up" else "- 1"
        input_label = f"{x} {delta}"
        label = f"[{x} {delta}]_q"
    elif kind == "negation":
        coeffs = data.get("neg_coefficients")
        valuation = int(data.get("neg_valuation", 0))
        input_label = f"-{x}"
        label = f"[-{x}]_q"
    elif kind == "arith":
        coeffs = data.get("coefficients")
        sym = "*" if data.get("op") == "mul" else "+"
        input_label = f"[{x}]_q {sym} [{data.get('y')}]_q"
        label = input_label
    else:
        coeffs = data.get("coefficients")
        input_label = x
        label = f"[{x}]_q"
    if not isinstance(coeffs, list):
        raise ValueError(
            "this result has no q-coefficient series to keep; compute coeffs, "
            "laurent, readouts, arith, shift, or negate first"
        )
    n = data.get("n")
    if n is None:
        n = int(data["order"]) + 1 if "order" in data else len(coeffs)
    return SavedEntry(
        input=input_label,
        n=int(n),
        coefficients=[int(c) for c in coeffs],
        label=label,
        kind=kind,
        valuation=valuation,
    )


def _saved_list_result(entries: list[SavedEntry], store_path: Any) -> Result:
    """A renderable view of the saved list (no "kind", so it skips the stamp)."""
    rows = []
    for i, e in enumerate(entries):
        preview = exports._laurent_latex(e.valuation, e.coefficients)
        if len(preview) > 48:
            preview = preview[:45] + "..."
        rows.append([str(i), e.label, str(e.n), e.timestamp, e.qprov_id or "-"])
    blocks: list[dict[str, Any]] = []
    if entries:
        blocks.append(
            {
                "kind": "table",
                "columns": ["#", "value", "N", "saved at (UTC)", "qprov id"],
                "rows": rows,
            }
        )
    else:
        blocks.append(
            {
                "kind": "note",
                "text": "your saved list is empty. Compute a value, then "
                "choose 'Add this result to my saved list'.",
            }
        )
    blocks.append({"kind": "note", "text": f"stored at {store_path}"})
    return {
        "title": f"My saved list  ({len(entries)} item{'s' if len(entries) != 1 else ''})",
        "blocks": blocks,
        "data": {"entries": [e.to_dict() for e in entries]},
    }


def _add_result_to_saved(
    result: Result, console: Any | None, store: SavedStore | None = None
) -> None:
    store = store or SavedStore()
    try:
        entry = entry_from_result(result)
    except ValueError as exc:
        _say(console, f"cannot save this result: {exc}")
        return
    store.add(entry)
    _say(console, f"saved {entry.label} to your list ({store.path})")


def _ask_export_path(
    qst: Any, console: Any | None, fmt: str, default_stem: str
) -> str | None:
    """Ask where to write. A blank answer defaults to the current directory only
    after an explicit confirm, so nothing lands in the working folder by surprise.
    """
    ext = exports.extension_for(fmt)
    raw = qst.text(
        f"file path to write  (blank uses ./{default_stem}{ext})", default=""
    ).ask()
    if raw is None:
        return None
    answer = str(raw).strip()
    if answer:
        return answer
    confirm = qst.select(
        f"Write {default_stem}{ext} to the current directory?",
        choices=["Yes", "No, cancel"],
    ).ask()
    return f"{default_stem}{ext}" if confirm == "Yes" else None


def _ask_qprov(qst: Any) -> bool:
    """Offer the optional qprov link, only when qprov is importable."""
    from . import provenance

    if not provenance.qprov_available():
        return False
    choice = qst.select(
        "Link each item to a qprov record? (records a run in your .qprov store)",
        choices=["No", "Yes, record and attach ids"],
    ).ask()
    return bool(choice == "Yes, record and attach ids")


def _write_entries(
    entries: list[SavedEntry],
    fmt: str,
    path: str,
    console: Any | None,
    use_qprov: bool,
) -> None:
    if use_qprov:
        from . import provenance

        provenance.annotate(entries)
    text = exports.render(entries, fmt)
    written = exports.write_export(text, path)
    _say(console, f"wrote {written}")


def _export_one(result: Result, qst: Any, console: Any | None) -> None:
    try:
        entry = entry_from_result(result)
    except ValueError as exc:
        _say(console, f"cannot export this result: {exc}")
        return
    fmt = qst.select("Which format?", choices=list(exports.FORMATS)).ask()
    if fmt is None:
        return
    path = _ask_export_path(qst, console, fmt, "qreals-result")
    if path is None:
        return
    _write_entries([entry], fmt, path, console, _ask_qprov(qst))


_SAVED_VIEW = "View my saved list"
_SAVED_REMOVE = "Remove an item"
_SAVED_EXPORT = "Export my saved list (writes a file)"
_SAVED_CLEAR = "Clear the whole list"
_SAVED_BACK = "Back to menu"


def _saved_menu(qst: Any, console: Any | None, store: SavedStore | None = None) -> None:
    store = store or SavedStore()
    while True:
        count = len(store.all())
        choice = qst.select(
            f"My saved list  ({count} item{'s' if count != 1 else ''})",
            choices=[
                _SAVED_VIEW,
                _SAVED_REMOVE,
                _SAVED_EXPORT,
                _SAVED_CLEAR,
                _SAVED_BACK,
            ],
        ).ask()
        if choice is None or choice == _SAVED_BACK:
            return
        if choice == _SAVED_VIEW:
            render_result(_saved_list_result(store.all(), store.path), console)
        elif choice == _SAVED_REMOVE:
            _remove_saved(qst, console, store)
        elif choice == _SAVED_EXPORT:
            _export_saved(qst, console, store)
        elif choice == _SAVED_CLEAR:
            confirm = qst.select(
                "Remove every saved item?",
                choices=["No, keep them", "Yes, clear the list"],
            ).ask()
            if confirm == "Yes, clear the list":
                _say(console, f"cleared {store.clear()} item(s)")


def _remove_saved(qst: Any, console: Any | None, store: SavedStore) -> None:
    entries = store.all()
    if not entries:
        _say(console, "your saved list is empty; nothing to remove")
        return
    labels = [
        f"{i}: {e.label}  (N={e.n}, {e.timestamp})" for i, e in enumerate(entries)
    ]
    cancel = "Cancel"
    choice = qst.select("Remove which item?", choices=labels + [cancel]).ask()
    if choice is None or choice == cancel:
        return
    index = int(choice.split(":", 1)[0])
    removed = store.remove(index)
    _say(console, f"removed {removed.label}")


def _export_saved(qst: Any, console: Any | None, store: SavedStore) -> None:
    entries = store.all()
    if not entries:
        _say(console, "your saved list is empty; nothing to export")
        return
    fmt = qst.select("Which format?", choices=list(exports.FORMATS)).ask()
    if fmt is None:
        return
    path = _ask_export_path(qst, console, fmt, "qreals-saved")
    if path is None:
        return
    _write_entries(entries, fmt, path, console, _ask_qprov(qst))


# --------------------------------------------------------------------------
# Environment check.
# --------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _tex_engine_name() -> str | None:
    """The TeX engine the certificate layer would use, or None."""
    try:
        from .certificate import find_tex_engine

        return find_tex_engine()
    except Exception:  # noqa: BLE001 - fall back to a direct probe
        import shutil

        for engine in ("pdflatex", "tectonic", "latexmk"):
            if shutil.which(engine):
                return engine
        return None


def doctor_report() -> dict[str, Any]:
    """Capability flags for the current environment, used by `qreals doctor`."""
    import platform

    has_questionary = _module_available("questionary")
    stdin_tty = _is_a_tty(sys.stdin)
    stdout_tty = _is_a_tty(sys.stdout)
    menu_will_run = has_questionary and stdin_tty and stdout_tty
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "stdin_tty": stdin_tty,
        "stdout_tty": stdout_tty,
        "questionary": has_questionary,
        "rich": _module_available("rich"),
        "tex_engine": _tex_engine_name(),
        "menu_will_run": menu_will_run,
    }


def _is_a_tty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (ValueError, AttributeError):
        return False


def run_doctor(console: Any | None = None) -> int:
    """Print the environment report and a one-line verdict on the menu."""
    report = doctor_report()
    tex = report["tex_engine"] or "none on PATH"
    lines = [
        "qreals doctor",
        f"  operating system : {report['os']}",
        f"  python           : {report['python']}",
        f"  stdin is a tty   : {report['stdin_tty']}",
        f"  stdout is a tty  : {report['stdout_tty']}",
        f"  questionary      : {'available' if report['questionary'] else 'missing'}",
        f"  rich             : {'available' if report['rich'] else 'missing'}",
        f"  tex engine       : {tex}",
    ]
    if report["menu_will_run"]:
        verdict = "verdict: the interactive menu will run here."
    elif not report["questionary"]:
        verdict = "verdict: install qreals[app] for the menu; subcommands work now."
    else:
        verdict = (
            "verdict: no terminal; use subcommands, for example qreals coeffs pi 12."
        )
    for line in lines:
        _say(console, line)
    _say(console, verdict)
    return 0


# --------------------------------------------------------------------------
# Headless CLI for agents and scripts.
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qreals",
        description=(
            "q-deformed rationals and reals. Run with no arguments for a guided "
            "arrow-key menu, or use a subcommand below for scripting."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("menu", help="open the interactive arrow-key menu (the default)")
    sub.add_parser(
        "doctor", help="report OS, Python, TTY, and which optional extras import"
    )

    p_certify = sub.add_parser(
        "certify",
        help="print a human-auditable certificate (a derivation, not a machine proof)",
    )
    p_certify.add_argument(
        "kind",
        choices=["rational", "qint", "coeffs", "laurent"],
        help="which computation to certify",
    )
    p_certify.add_argument(
        "args", nargs="*", help="the inputs, e.g. rational 3 2, coeffs pi 12"
    )
    p_certify.add_argument(
        "--order", type=int, default=12, help="for laurent: highest power"
    )
    p_certify.add_argument(
        "--max-terms",
        type=int,
        default=20,
        help="cap monomials shown per polynomial in the PDF (default 20)",
    )
    p_certify.add_argument(
        "--save", action="store_true", help="write a .tex (and .pdf) here"
    )
    p_certify.add_argument(
        "--pdf", action="store_true", help="open a temporary PDF, keep no file"
    )
    p_certify.add_argument(
        "--qprov",
        action="store_true",
        help="with --save, also record the run in your .qprov store (needs qprov)",
    )

    def add_json(p: argparse.ArgumentParser) -> None:
        p.add_argument("--json", action="store_true", help="print the result as JSON")

    p_rational = sub.add_parser("rational", help="exact [p/s]_q")
    p_rational.add_argument("p", type=int)
    p_rational.add_argument("s", type=int)
    add_json(p_rational)

    p_jumpgap = sub.add_parser(
        "jumpgap", help="the two q-versions of p/s and the factored gap between them"
    )
    p_jumpgap.add_argument("p", type=int)
    p_jumpgap.add_argument("s", type=int)
    add_json(p_jumpgap)

    p_factor = sub.add_parser(
        "factor",
        help="factor R(q), S(q) of [a/b]_q over Z[q], labelling Phi_d factors",
    )
    p_factor.add_argument("fraction", help="the rational a/b, e.g. 7/5")
    add_json(p_factor)

    p_sprops = sub.add_parser(
        "sprops",
        help="properties of the denominator S(q): cyclotomic factors, "
        "saturation index e*, deg S vs d-1, S(1)=d, collapse vs full [d]_q",
    )
    p_sprops.add_argument("fraction", help="the rational a/d, e.g. 5/12")
    add_json(p_sprops)

    p_satlas = sub.add_parser(
        "satlas",
        help="S(q) cyclotomic-factor atlas over a coprime (a, d) grid, "
        "classified by regime with a Phi_k appearance tally",
    )
    p_satlas.add_argument(
        "d_max", type=int, help="max denominator d in the grid, e.g. 12"
    )
    p_satlas.add_argument(
        "--a-max", type=int, default=None, help="cap the numerator a (default a < d)"
    )
    add_json(p_satlas)

    p_saturation = sub.add_parser(
        "saturation",
        help="saturation index e* = lcm(T) and the minimal saturating n as a "
        "ranges over the residues coprime to a fixed d",
    )
    p_saturation.add_argument("d", type=int, help="the denominator d, e.g. 12")
    add_json(p_saturation)

    p_degcollapse = sub.add_parser(
        "degcollapse",
        help="deg S vs the bound d-1 over the grid, with the collapse depth "
        "(d-1-deg S = totient weight of the dropped Phi_k)",
    )
    p_degcollapse.add_argument(
        "d_max", type=int, help="max denominator d in the grid, e.g. 12"
    )
    p_degcollapse.add_argument(
        "--a-max", type=int, default=None, help="cap the numerator a (default a < d)"
    )
    add_json(p_degcollapse)

    p_exact = sub.add_parser(
        "exact",
        help="exact [x]_q = P/Q, or the exact difference [x]_q - [y]_q over Q(q)",
    )
    p_exact.add_argument("x", help="a rational x, e.g. 7/5")
    p_exact.add_argument(
        "y", nargs="?", default="", help="optional rational y for the difference"
    )
    add_json(p_exact)

    p_serve = sub.add_parser(
        "serve",
        help="start a localhost MathJax web UI over the engine (needs [serve])",
    )
    p_serve.add_argument(
        "--port", type=int, default=8000, help="preferred port (falls back if taken)"
    )
    p_serve.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )

    p_qint = sub.add_parser("qint", help="the q-integers [n]_q and [n]_(q^-1)")
    p_qint.add_argument("n", type=int)
    add_json(p_qint)

    p_coeffs = sub.add_parser("coeffs", help="first N coefficients of [x]_q")
    p_coeffs.add_argument("x")
    p_coeffs.add_argument("n", type=int)
    add_json(p_coeffs)

    p_laurent = sub.add_parser("laurent", help="[x]_q to a chosen power")
    p_laurent.add_argument("x")
    p_laurent.add_argument("--order", type=int, default=12)
    add_json(p_laurent)

    p_prefix = sub.add_parser("prefix", help="integer-part prefix of [x]_q")
    p_prefix.add_argument("x")
    add_json(p_prefix)

    p_locked = sub.add_parser(
        "locked", help="coefficients the n-th convergent locks in"
    )
    p_locked.add_argument("x")
    p_locked.add_argument("n", type=int)
    add_json(p_locked)

    p_shift = sub.add_parser("shift", help="[x+1]_q (--up) or [x-1]_q (--down)")
    p_shift.add_argument("x")
    p_shift.add_argument("--order", type=int, default=12)
    group = p_shift.add_mutually_exclusive_group()
    group.add_argument(
        "--up", action="store_true", help="raise the argument by one (default)"
    )
    group.add_argument("--down", action="store_true", help="lower the argument by one")
    add_json(p_shift)

    p_readouts = sub.add_parser(
        "readouts", help="pattern read-outs over the first N coefficients"
    )
    p_readouts.add_argument("x")
    p_readouts.add_argument("n", type=int)
    add_json(p_readouts)

    p_arith = sub.add_parser(
        "arith", help="series sum [x]_q+[y]_q or product [x]_q*[y]_q"
    )
    p_arith.add_argument("x")
    p_arith.add_argument("y")
    p_arith.add_argument("n", type=int)
    group = p_arith.add_mutually_exclusive_group()
    group.add_argument("--add", action="store_true", help="series sum (default)")
    group.add_argument("--mul", action="store_true", help="series product")
    add_json(p_arith)

    p_quad = sub.add_parser(
        "quad",
        help="exact x op y for quadratic irrationals via the CF transfer matrix",
    )
    p_quad.add_argument("x", help="quadratic irrational, e.g. (1+sqrt(5))/2")
    p_quad.add_argument("y", help="quadratic irrational, e.g. 1+sqrt(2)")
    p_quad.add_argument(
        "--op", choices=["add", "sub", "mul", "div"], default="add", help="operation"
    )
    add_json(p_quad)

    p_deficit = sub.add_parser(
        "deficit", help="deficit [x op y]_q - (series sum or product), op + or *"
    )
    p_deficit.add_argument("x")
    p_deficit.add_argument("y")
    p_deficit.add_argument("n", type=int)
    group = p_deficit.add_mutually_exclusive_group()
    group.add_argument("--add", action="store_true", help="op + (default)")
    group.add_argument("--mul", action="store_true", help="op *")
    add_json(p_deficit)

    p_negate = sub.add_parser(
        "negate", help="Jouteur [-x]_q, [x]_q+[-x]_q, and finiteness"
    )
    p_negate.add_argument("x")
    p_negate.add_argument("n", type=int)
    add_json(p_negate)

    p_negsum = sub.add_parser(
        "negsum", help="negation sum [x]_q+[-x]_q and its finiteness (Ex. 6.4)"
    )
    p_negsum.add_argument("x")
    p_negsum.add_argument("n", type=int)
    add_json(p_negsum)

    p_radius = sub.add_parser("radius", help="radius-of-convergence estimate of [x]_q")
    p_radius.add_argument("x")
    p_radius.add_argument("n", type=int)
    add_json(p_radius)

    p_oeis = sub.add_parser("oeis", help="look a coefficient sequence up in the OEIS")
    p_oeis.add_argument(
        "sequence", help='comma-separated integers, e.g. "1,1,2,5,14,42"'
    )
    p_oeis.add_argument("--no-modp", action="store_true", help="skip mod-p reductions")
    p_oeis.add_argument(
        "--no-bfile", action="store_true", help="skip b-file re-verification"
    )
    add_json(p_oeis)

    p_fp = sub.add_parser(
        "fingerprint", help="named, fixed-length fingerprint of [x]_q"
    )
    p_fp.add_argument("x")
    p_fp.add_argument(
        "--n-cf",
        type=int,
        default=features.DEFAULT_N_CF,
        help="continued-fraction terms",
    )
    p_fp.add_argument(
        "--n-coeffs",
        type=int,
        default=features.DEFAULT_N_COEFFS,
        help="q-series coefficients",
    )
    p_fp.add_argument(
        "--n-radius",
        type=int,
        default=features.DEFAULT_N_RADIUS,
        help="inv_radius window",
    )
    add_json(p_fp)

    p_batch = sub.add_parser(
        "batch",
        help="compute [x]_q for a list of constants and write one export file",
    )
    p_batch.add_argument(
        "constants", help='comma-separated reals, e.g. "pi,sqrt(2),3/2"'
    )
    p_batch.add_argument(
        "--order",
        type=int,
        default=12,
        help="number of coefficients N per constant (default 12)",
    )
    p_batch.add_argument(
        "--format", choices=exports.FORMATS, required=True, help="output format"
    )
    p_batch.add_argument(
        "-o",
        "--output",
        help="file to write; without it the export is printed to stdout",
    )
    p_batch.add_argument(
        "--qprov",
        action="store_true",
        help="record each value in qprov and attach its id (needs qprov)",
    )

    p_export = sub.add_parser("export", help="write the saved list in a format")
    p_export.add_argument(
        "--format", choices=exports.FORMATS, required=True, help="output format"
    )
    p_export.add_argument(
        "-o",
        "--output",
        help="file to write; without it the export is printed to stdout",
    )
    p_export.add_argument(
        "--qprov",
        action="store_true",
        help="record each saved value in qprov and attach its id (needs qprov)",
    )

    p_saved = sub.add_parser("saved", help="list, remove from, or clear the saved list")
    saved_group = p_saved.add_mutually_exclusive_group()
    saved_group.add_argument(
        "--remove", type=int, metavar="INDEX", help="remove the entry at INDEX"
    )
    saved_group.add_argument(
        "--clear", action="store_true", help="remove every saved entry"
    )
    add_json(p_saved)

    return parser


def _run_headless(args: argparse.Namespace) -> int:
    console = None if getattr(args, "json", False) else _make_console()
    as_json = getattr(args, "json", False)
    try:
        if args.command == "rational":
            result = compute_rational(args.p, args.s)
        elif args.command == "jumpgap":
            result = compute_jumpgap(args.p, args.s)
        elif args.command == "factor":
            a, b = _parse_rational(args.fraction)
            result = compute_factor(a, b)
        elif args.command == "sprops":
            a, b = _parse_rational(args.fraction)
            result = compute_sprops(a, b)
        elif args.command == "satlas":
            result = compute_satlas(args.d_max, args.a_max)
        elif args.command == "saturation":
            result = compute_saturation(args.d)
        elif args.command == "degcollapse":
            result = compute_degcollapse(args.d_max, args.a_max)
        elif args.command == "exact":
            result = compute_exact_rational(args.x, args.y)
        elif args.command == "qint":
            result = compute_qint(args.n)
        elif args.command == "coeffs":
            result = compute_coeffs(args.x, args.n)
        elif args.command == "laurent":
            result = compute_laurent(args.x, args.order)
        elif args.command == "prefix":
            result = compute_prefix(args.x)
        elif args.command == "locked":
            result = compute_locked(args.x, args.n)
        elif args.command == "shift":
            direction = "down" if args.down else "up"
            result = compute_shift(args.x, args.order, direction)
        elif args.command == "readouts":
            result = compute_readouts(args.x, args.n)
        elif args.command == "arith":
            result = compute_arith(args.x, args.y, args.n, "mul" if args.mul else "add")
        elif args.command == "quad":
            result = compute_quad_arith(args.x, args.y, args.op)
        elif args.command == "deficit":
            result = compute_deficit(
                args.x, args.y, args.n, "mul" if args.mul else "add"
            )
        elif args.command == "negate":
            result = compute_negation(args.x, args.n)
        elif args.command == "negsum":
            result = compute_negsum(args.x, args.n)
        elif args.command == "radius":
            result = compute_radius(args.x, args.n)
        elif args.command == "oeis":
            result = compute_oeis(
                args.sequence, do_modp=not args.no_modp, do_bfile=not args.no_bfile
            )
        elif args.command == "fingerprint":
            result = compute_fingerprint(
                args.x, n_cf=args.n_cf, n_coeffs=args.n_coeffs, n_radius=args.n_radius
            )
        else:  # pragma: no cover - argparse guards this
            raise ValueError(f"unknown command {args.command!r}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    render_result(result, console, as_json=as_json)
    return 0


def _emit_export(text: str, output: str | None) -> int:
    """Write the export to a file when a path is given, else print it."""
    if output:
        print(f"wrote {exports.write_export(text, output)}")
    else:
        print(text)
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    constants = [c.strip() for c in args.constants.split(",") if c.strip()]
    if not constants:
        print("error: no constants given", file=sys.stderr)
        return 1
    entries: list[SavedEntry] = []
    for c in constants:
        try:
            coeffs = q_real_truncated(c, args.order)
        except Exception as exc:  # noqa: BLE001 - report the math error and stop
            print(f"error: could not compute [{c}]_q: {exc}", file=sys.stderr)
            return 1
        entries.append(
            SavedEntry(
                input=c,
                n=args.order,
                coefficients=[int(v) for v in coeffs],
                kind="coeffs",
            )
        )
    if args.qprov:
        from . import provenance

        provenance.annotate(entries)
    return _emit_export(exports.render(entries, args.format), args.output)


def _run_export(args: argparse.Namespace) -> int:
    entries = SavedStore().all()
    if not entries:
        print(
            "your saved list is empty; add results in the menu or run qreals batch",
            file=sys.stderr,
        )
        return 1
    if args.qprov:
        from . import provenance

        provenance.annotate(entries)
    return _emit_export(exports.render(entries, args.format), args.output)


def _run_saved(args: argparse.Namespace) -> int:
    store = SavedStore()
    if args.clear:
        print(f"cleared {store.clear()} item(s)")
        return 0
    if args.remove is not None:
        try:
            removed = store.remove(args.remove)
        except IndexError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"removed {removed.label}")
        return 0
    result = _saved_list_result(store.all(), store.path)
    as_json = getattr(args, "json", False)
    console = None if as_json else _make_console()
    render_result(result, console, as_json=as_json, verify_result=False)
    return 0


def _result_for_certify(kind: str, args: list[str], order: int) -> Result:
    """Build the computation result a certify subcommand should certify."""
    if kind == "rational":
        if len(args) != 2:
            raise ValueError(
                "certify rational needs p and s, e.g. certify rational 3 2"
            )
        return compute_rational(int(args[0]), int(args[1]))
    if kind == "qint":
        if len(args) != 1:
            raise ValueError("certify qint needs n, e.g. certify qint 5")
        return compute_qint(int(args[0]))
    if kind == "coeffs":
        if len(args) != 2:
            raise ValueError("certify coeffs needs x and N, e.g. certify coeffs pi 12")
        return compute_coeffs(args[0], int(args[1]))
    if kind == "laurent":
        if len(args) != 1:
            raise ValueError(
                "certify laurent needs x, e.g. certify laurent pi --order 12"
            )
        return compute_laurent(args[0], order)
    raise ValueError(f"unknown certify kind {kind!r}")


def _run_certify(args: argparse.Namespace) -> int:
    console = _make_console()
    certificate = _load_certificate()
    if certificate is None:
        print(
            "error: certificates need the proof extras; install with pip install qreals[proof]",
            file=sys.stderr,
        )
        return 1
    try:
        result = _result_for_certify(args.kind, args.args, args.order)
        cert = certificate.build_certificate(result)
        cert.max_terms = args.max_terms
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.save:
        written = cert.save(".", qprov=args.qprov)
        print(f"wrote {written['tex']}")
        if written["pdf"]:
            print(f"wrote {written['pdf']}")
        else:
            print("no TeX engine found; compile the .tex with pdflatex to get a PDF")
        if args.qprov:
            if written["qprov_id"]:
                print(f"recorded in your .qprov store as {written['qprov_id']}")
            else:
                print("qprov not available; saved the certificate without recording")
        return 0
    if args.pdf:
        cert.view_pdf(console)
        return 0
    cert.render_terminal(console)  # the default: print the derivation, keep no file
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. No arguments opens the menu; a subcommand runs headless."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv == ["menu"]:
        return run_interactive()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        return run_interactive()
    if args.command == "doctor":
        return run_doctor(_make_console())
    if args.command == "certify":
        return _run_certify(args)
    if args.command == "batch":
        return _run_batch(args)
    if args.command == "export":
        return _run_export(args)
    if args.command == "saved":
        return _run_saved(args)
    if args.command == "serve":
        return _run_serve(args)
    return _run_headless(args)


def _run_serve(args: argparse.Namespace) -> int:
    """Start the local web UI, reporting a clear message if the extra is missing."""
    try:
        from . import serve as _serve_module
    except Exception as exc:  # noqa: BLE001 - import error surfaces to the user
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        return _serve_module.serve(port=args.port, open_browser=not args.no_browser)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nqreals serve stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
