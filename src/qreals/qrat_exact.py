r"""Exact q-rational functions and their differences over Q(q).

For a rational x = a/b the continued fraction terminates, so [x]_q is a genuine
element of Q(q): a reduced ratio P_x(q)/Q_x(q), not a truncated series. This
module exposes that exact form and, more importantly, the exact *difference* of
two q-rationals,

    [x]_q - [y]_q = P_x/Q_x - P_y/Q_y = (P_x Q_y - P_y Q_x) / (Q_x Q_y),

which is the worked example from the June 1 board. Writing N(q) = P_x Q_y -
P_y Q_x and D(q) = Q_x Q_y, the difference reduces to a single P(q)/Q(q), and
the board's observation is that the denominators are tightly linked: Q_x | Q_y
and Q_y | Q_x can both hold, forcing Q_x = Q_y up to a unit. This module
reports those divisibilities exactly (polynomial remainders over Q[q]), so the
conjecture can be checked on any pair instead of by hand.

Everything is exact: sympy over Q[q], no floating point. Setting q = 1 sends
[x]_q to the ordinary x, so the difference at q = 1 is x - y, used as the
defining sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._parsing import parse_real
from .rational import q, q_rational


@dataclass(frozen=True)
class QRatExact:
    """[a/b]_q as a reduced rational function P/Q in q.

    Fields:
        a, b: the input fraction a/b in lowest terms (b > 0).
        P, Q: the reduced numerator and denominator, P/Q = [a/b]_q with
            gcd(P, Q) = 1 over Q[q].
        cf: the regular continued fraction [a_0; a_1, ...] of a/b, the [a],
            [b], [c] structure the q-rational is built from.
    """

    a: int
    b: int
    P: sp.Expr
    Q: sp.Expr
    cf: list[int]


@dataclass(frozen=True)
class QRatDifference:
    """The exact difference [x]_q - [y]_q of two q-rationals.

    Fields:
        x, y: the two operands as QRatExact (each carrying its P/Q and cf).
        num_unreduced: P_x Q_y - P_y Q_x, the numerator before cancellation.
        den_unreduced: Q_x Q_y, the denominator before cancellation.
        num, den: the reduced difference num/den with gcd(num, den) = 1.
        qx_divides_qy: True when Q_x divides Q_y over Q[q].
        qy_divides_qx: True when Q_y divides Q_x over Q[q].
        q_equal_up_to_unit: True when Q_x and Q_y are associates (each divides
            the other), the board's Q_x = Q_y "up to a unit".
        gcd_Q: gcd(Q_x, Q_y) over Q[q].
        value_at_1: the difference at q = 1, equal to the ordinary x - y.
    """

    x: QRatExact
    y: QRatExact
    num_unreduced: sp.Expr
    den_unreduced: sp.Expr
    num: sp.Expr
    den: sp.Expr
    qx_divides_qy: bool
    qy_divides_qx: bool
    q_equal_up_to_unit: bool
    gcd_Q: sp.Expr
    value_at_1: sp.Expr


def _parse_q_rational(text: str) -> tuple[int, int]:
    """Read a rational a/b from text, rejecting anything irrational.

    Accepts "7/5", "7", "3/2", or any expression sympify reduces to a rational
    (for example "1 + 1/2"). Raises ValueError for irrational input such as
    sqrt(2), since [sqrt(2)]_q is a power series, not a rational function.
    """
    expr = sp.nsimplify(parse_real(text))
    if not expr.is_rational:
        raise ValueError(
            f"exact rational function needs a rational input; {text!r} is not rational"
        )
    r = sp.Rational(expr)
    return int(r.p), int(r.q)


def q_rational_exact(text: str) -> QRatExact:
    """[a/b]_q as a reduced rational function, for a rational a/b given as text."""
    a, b = _parse_q_rational(text)
    expr = sp.cancel(q_rational(a, b))
    P, Q = sp.fraction(expr)
    cf = [int(t) for t in sp.continued_fraction(sp.Rational(a, b))]
    return QRatExact(a=a, b=b, P=sp.expand(P), Q=sp.expand(Q), cf=cf)


def _divides(divisor: sp.Expr, dividend: sp.Expr) -> bool:
    """Polynomial divisibility divisor | dividend over Q[q]."""
    d = sp.Poly(divisor, q)
    n = sp.Poly(dividend, q)
    if d.is_zero:
        return n.is_zero
    return n.rem(d).is_zero


def q_rational_difference(x_text: str, y_text: str) -> QRatDifference:
    """The exact difference [x]_q - [y]_q for two rationals x, y given as text."""
    x = q_rational_exact(x_text)
    y = q_rational_exact(y_text)

    num_unreduced = sp.expand(x.P * y.Q - y.P * x.Q)
    den_unreduced = sp.expand(x.Q * y.Q)

    reduced = sp.cancel(sp.Rational(1) * num_unreduced / den_unreduced)
    num, den = sp.fraction(reduced)
    num, den = sp.expand(num), sp.expand(den)

    qx_divides_qy = _divides(x.Q, y.Q)
    qy_divides_qx = _divides(y.Q, x.Q)
    gcd_Q = sp.expand(sp.gcd(x.Q, y.Q))
    value_at_1 = sp.nsimplify(reduced.subs(q, 1))

    return QRatDifference(
        x=x,
        y=y,
        num_unreduced=num_unreduced,
        den_unreduced=den_unreduced,
        num=num,
        den=den,
        qx_divides_qy=qx_divides_qy,
        qy_divides_qx=qy_divides_qx,
        q_equal_up_to_unit=qx_divides_qy and qy_divides_qx,
        gcd_Q=gcd_Q,
        value_at_1=value_at_1,
    )
