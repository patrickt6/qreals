"""Generate qnumbers-site/data/operations-deficit.json from qreals.

Every value in the JSON is produced by the same functions the CLI uses:
qreals.deficit, qreals.q_rational, qreals.q_gosper, qreals.negation_sum,
qreals.finite_xnegx. The site page reads this file, so the page and the CLI
share one source of truth. Run from the qreals repo root:

    python gen_operations_json.py
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, "src")

import sympy as sp  # noqa: E402

import qreals  # noqa: E402
from qreals import (  # noqa: E402
    deficit,
    finite_xnegx,
    negation_sum,
    q,
    q_gosper,
    q_rational,
)

N = 12

OUT = Path(__file__).resolve().parent.parent / "qnumbers-site" / "data" / "operations-deficit.json"

# Curated rational inputs for the operation battery, as canonical reduced
# strings. All pairs (with repetition) over both + and * are precomputed.
OP_VALUES = [
    "1/2", "1/3", "2/3", "1/4", "3/4", "2/5", "3/5",
    "1", "3/2", "5/2", "4/3", "5/3", "5/4", "7/4", "7/5", "2", "3",
]

# Inputs for the negation battery. Pure square roots are the finite case
# (trace-zero quadratics, Ovsienko Example 6.4); the rest are infinite.
NEG_VALUES = [
    "sqrt(2)", "sqrt(3)", "sqrt(5)", "sqrt(6)", "sqrt(7)",
    "(1+sqrt(5))/2", "1/2", "3/2", "5/2", "2", "3",
]


def canon(s: str) -> str:
    """Reduced canonical string for a rational input: '6/4' -> '3/2', '2/1' -> '2'."""
    fr = Fraction(s)
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def q_rational_latex(s: str) -> str:
    fr = Fraction(s)
    return sp.latex(q_rational(fr.numerator, fr.denominator))


def laurent_latex(valuation: int, coeffs: list[int], finite: bool) -> str:
    """LaTeX for a Laurent series given its valuation and coefficient list.

    Finite: the exact polynomial with trailing zeros trimmed. Infinite: the
    leading terms then a continuation mark.
    """
    if finite:
        # Trim trailing zeros so the exact finite polynomial is shown.
        last = -1
        for i, c in enumerate(coeffs):
            if c != 0:
                last = i
        terms = coeffs[: last + 1]
        expr = sum(c * q ** (valuation + i) for i, c in enumerate(terms))
        return sp.latex(sp.nsimplify(expr)) if expr != 0 else "0"
    # Infinite: show terms through a modest cutoff, then the continuation mark.
    cutoff = min(len(coeffs), 8)
    expr = sum(coeffs[i] * q ** (valuation + i) for i in range(cutoff))
    return sp.latex(expr) + r" + \cdots"


def op_entry(xs: str, ys: str, op: str) -> dict:
    d = deficit(xs, ys, op, N)
    fr_x, fr_y = Fraction(xs), Fraction(ys)
    engine = q_gosper(
        Fraction(fr_x.numerator, fr_x.denominator),
        Fraction(fr_y.numerator, fr_y.denominator),
        "add" if op == "+" else "mul",
    )
    target_value = (fr_x + fr_y) if op == "+" else (fr_x * fr_y)
    return {
        "x": xs,
        "y": ys,
        "op": op,
        "result": canon(str(target_value)),
        "x_latex": sp.latex(sp.sympify(xs)),
        "y_latex": sp.latex(sp.sympify(ys)),
        "result_latex": sp.latex(sp.sympify(str(target_value))),
        "x_q": q_rational_latex(xs),
        "y_q": q_rational_latex(ys),
        "engine": sp.latex(sp.cancel(engine)),
        "target": sp.latex(q_rational(target_value.numerator, target_value.denominator)),
        "deficit": sp.latex(d.exact),
        "deficit_factored": sp.latex(sp.factor(d.exact)),
        "q1": d.deficit_at_q1,
        "q0": d.deficit_at_q0,
    }


def classify(xs: str) -> str:
    """integer, sqrt (trace-zero quadratic), rational, quadratic, or higher."""
    val = sp.sympify(xs)
    if val.is_integer:
        return "integer"
    if val.is_rational:
        return "rational"
    t = sp.Symbol("t")
    poly = sp.Poly(sp.minimal_polynomial(val, t), t)
    if poly.degree() == 2:
        return "sqrt" if poly.coeff_monomial(t) == 0 else "quadratic"
    return "higher"


def neg_verdict(kind: str, finite: bool) -> str:
    """A precise one-sentence verdict for the negation sum, by input kind."""
    if kind == "integer":
        return ("x is an integer, so [x]_q is a polynomial and the sum is "
                "trivially a finite Laurent polynomial.")
    if kind == "sqrt":
        return ("x is a trace-zero quadratic (a pure square root), so the sum "
                "collapses to a finite Laurent polynomial (Ovsienko Example 6.4).")
    if kind == "rational":
        return ("x is a non-integer rational, so [x]_q is an infinite series and "
                "the sum does not terminate; the leading terms are shown.")
    if kind == "quadratic":
        return ("x is a quadratic with nonzero trace, not a pure square root, so "
                "the sum does not terminate (Ovsienko Example 6.4); leading terms shown.")
    return ("x is not a trace-zero quadratic, so the sum does not terminate "
            "(Ovsienko Example 6.4); the leading terms are shown.")


def neg_entry(xs: str) -> dict:
    valuation, coeffs = negation_sum(xs, N)
    finite = bool(finite_xnegx(xs))
    kind = classify(xs)
    return {
        "x": xs,
        "x_latex": sp.latex(sp.sympify(xs)),
        "finite": finite,
        "kind": kind,
        "verdict": neg_verdict(kind, finite),
        "sum": laurent_latex(valuation, coeffs, finite),
        "valuation": valuation,
    }


def main() -> None:
    operations: dict[str, dict] = {}
    vals = [canon(v) for v in OP_VALUES]
    for op in ("+", "*"):
        for i, a in enumerate(vals):
            for b in vals[i:]:
                lo, hi = sorted([a, b])
                key = f"{lo}|{hi}|{op}"
                operations[key] = op_entry(lo, hi, op)
                print("op", key)

    negation: dict[str, dict] = {}
    for v in NEG_VALUES:
        negation[v] = neg_entry(v)
        print("neg", v)

    payload = {
        "generated_by": f"qreals {qreals.__version__}: deficit, q_rational, q_gosper, negation_sum, finite_xnegx",
        "N": N,
        "key_format": "x|y|op with x,y canonical reduced rationals sorted lexicographically, op in {+,*}",
        "operations": operations,
        "negation": negation,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1)
    OUT.write_text(text, encoding="utf-8")
    # A script-tag loader with identical data, so the page also works when
    # opened directly from disk (file://), where fetch of a local JSON is
    # blocked by some browsers. The page prefers the JSON and falls back to this.
    js = OUT.with_suffix(".js")
    js.write_text("window.OPERATIONS_DEFICIT = " + text + ";\n", encoding="utf-8")
    print(f"\nwrote {OUT} : {len(operations)} op entries, {len(negation)} negation entries")
    print(f"wrote {js} (script-tag fallback)")


if __name__ == "__main__":
    main()
