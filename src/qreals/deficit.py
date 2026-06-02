r"""The q-arithmetic deficit: how far the engine value sits from [x op y]_q.

The MGO map x |-> [x]_q is not a ring homomorphism, so the series sum [x]_q +
[y]_q is not [x+y]_q and the series product [x]_q * [y]_q is not [x*y]_q. This
module names the gap directly. For an operation op in {"+", "*"} the deficit is

    D = [x op y]_q  -  (engine value),

where the engine value is the series sum [x]_q + [y]_q for "+" (arithmetic.q_add,
itself cross-checked by the bihomographic gosper engine) and the series product
[x]_q * [y]_q for "*" (arithmetic.q_mul). The target [x op y]_q is the genuine
q-series of the real number x op y, read off the verified q_real_truncated path
(and the exact q_rational when both inputs are rational).

Two invariants pin the sum deficit. At q = 1 every q-real collapses to its
ordinary value (RAT Corollary 1.7), so the engine value and the target agree
there and D(1) = 0. At q = 0 the gap theorem (REAL Theorem 2) forces constant
term 1 on each [.]_q, so for x, y >= 1 the engine has constant term 2 against the
target's 1 and D(0) = -1. Both are reported.

For rational x and y the gosper engine returns the engine value as an exact
rational function in q, so the deficit is exact: deficit("3/2", "5/2", "+") reads
q^3 - 1. For an irrational input only the truncated series is available, so the
closed form is omitted and D at q = 1 is left out (a truncated series cannot be
summed at q = 1, though D(1) = 0 still holds in closed form).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from ._parsing import parse_real
from .arithmetic import finite_xnegx, negation_sum, q_add, q_mul
from .gosper import q_gosper
from .rational import q, q_rational
from .truncated import q_real_truncated

# Public op spellings on the left, the internal gosper/arithmetic name on the
# right. "+" and "*" are the documented inputs; the word forms are accepted too.
_OPS: dict[str, str] = {"+": "add", "add": "add", "*": "mul", "mul": "mul"}


@dataclass(frozen=True)
class Deficit:
    """The deficit between the engine value and the target [x op y]_q.

    Fields:
        x, y: the two inputs, as given.
        op: the operation, normalised to "+" or "*".
        N: the number of Taylor coefficients carried.
        x_series, y_series: the first N coefficients of [x]_q and [y]_q.
        engine: the first N coefficients of the engine value, the series sum
            [x]_q + [y]_q for "+" or the series product [x]_q * [y]_q for "*".
        target: the first N coefficients of [x op y]_q, the q-series of the real
            number x op y.
        deficit: the first N coefficients of target - engine.
        deficit_at_q1: D at q = 1; exact when both inputs are rational, else None
            (a truncated series cannot be summed at q = 1, though it is 0 in
            closed form).
        deficit_at_q0: D at q = 0, the constant coefficient of the deficit.
        exact: the deficit as a reduced rational function in q when both inputs
            are rational (q^3 - 1 for x = 3/2, y = 5/2), else None.
    """

    x: str
    y: str
    op: str
    N: int
    x_series: list[int]
    y_series: list[int]
    engine: list[int]
    target: list[int]
    deficit: list[int]
    deficit_at_q1: int | None
    deficit_at_q0: int
    exact: sp.Expr | None


def deficit(x: str, y: str, op: str = "+", N: int = 12) -> Deficit:
    """The deficit [x op y]_q - (engine value) for op "+" or "*" (x, y >= 0).

    Reuses the verified pieces: q_real_truncated for [x]_q, [y]_q, and the target
    [x op y]_q; q_add / q_mul for the engine value; and, for rational inputs, the
    bihomographic q_gosper engine and q_rational for the exact closed form. See
    docs/CORRECTNESS.md.
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    key = _OPS.get(op)
    if key is None:
        raise ValueError(f'op must be "+" or "*", got {op!r}')
    op_sym = "+" if key == "add" else "*"

    x_series = q_real_truncated(x, N)
    y_series = q_real_truncated(y, N)

    xv = parse_real(x)
    yv = parse_real(y)
    target_value = xv + yv if key == "add" else xv * yv
    target = q_real_truncated(str(target_value), N)

    engine = q_add(x, y, N) if key == "add" else q_mul(x, y, N)
    deficit_coeffs = [t - e for t, e in zip(target, engine)]

    exact: sp.Expr | None = None
    at_q1: int | None = None
    at_q0 = int(deficit_coeffs[0])
    if xv.is_rational and yv.is_rational:
        rx, ry = sp.Rational(xv), sp.Rational(yv)
        engine_exact = q_gosper(
            Fraction(int(rx.p), int(rx.q)), Fraction(int(ry.p), int(ry.q)), key
        )
        rt = sp.Rational(target_value)
        target_exact = q_rational(int(rt.p), int(rt.q))
        exact = sp.cancel(target_exact - engine_exact)
        at_q1 = int(exact.subs(q, 1))
        at_q0 = int(exact.subs(q, 0))

    return Deficit(
        x=x,
        y=y,
        op=op_sym,
        N=N,
        x_series=x_series,
        y_series=y_series,
        engine=engine,
        target=target,
        deficit=deficit_coeffs,
        deficit_at_q1=at_q1,
        deficit_at_q0=at_q0,
        exact=exact,
    )


@dataclass(frozen=True)
class NegationPanel:
    """The x -> -x symmetry for one input (Ovsienko Example 6.4).

    Fields:
        x: the input, as given.
        N: the number of Laurent coefficients carried.
        valuation: the lowest power of q in the sum [x]_q + [-x]_q.
        sum_coeffs: the coefficients of [x]_q + [-x]_q from q^valuation up.
        finite: whether the sum terminates as a finite Laurent polynomial, the
            operational finite-order observation of finite_xnegx (finite iff x is
            a trace-zero quadratic, i.e. a pure square root).
    """

    x: str
    N: int
    valuation: int
    sum_coeffs: list[int]
    finite: bool


def negation_panel(x: str, N: int = 12) -> NegationPanel:
    """[x]_q + [-x]_q and whether it is finite, for one real x >= 0 (Ex. 6.4).

    Bundles the verified negation_sum and finite_xnegx so a single call gives the
    sum (as a valuation and coefficient list, since [-x]_q carries negative powers
    of q) and the finite-or-infinite verdict. See docs/CORRECTNESS.md.
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    valuation, sum_coeffs = negation_sum(x, N)
    return NegationPanel(
        x=x,
        N=N,
        valuation=valuation,
        sum_coeffs=sum_coeffs,
        finite=finite_xnegx(x),
    )
