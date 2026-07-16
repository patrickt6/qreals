"""The q-rational [p/s]_q as an exact rational function in q.

For a rational the continued fraction terminates, so the MGO formula produces
a genuine element of Q(q) rather than a truncated series. This path keeps the
result symbolic: q_rational(3, 2) returns (q**2 + q + 1)/(q + 1), not its
Taylor coefficients.

Setting q = 1 collapses [n]_q = 1 + q + ... + q^{n-1} back to n, so every
q-rational specialises to the ordinary rational at q = 1. The test suite uses
that as the defining sanity check.
"""

from __future__ import annotations

import sympy as sp

from .continuant import continuant_pair, continuant_symbolic
from .continuant import q as q
from .continuant import q_int as q_int
from .continuant import q_int_qinv as q_int_qinv
from .continued_fraction import make_even_length


def mgo_build(a: list[int]) -> sp.Expr:
    """Evaluate the even-length MGO continued fraction symbolically.

    Delegates to the shared fold primitive: the cancelled first-column ratio
    of the block product in `continuant`.
    """
    return continuant_symbolic(a)


def q_rational_pair(p: int, s: int) -> tuple[sp.Poly, sp.Poly]:
    """[p/s]_q as a reduced pair (N, S) of polynomials over ZZ, for p/s > 0.

    Same value as q_rational, computed through `continuant.continuant_pair`
    (the q^a-scaled Poly block product; see its docstring for the scaling)
    with polynomial arithmetic instead of symbolic cancellation, so it stays
    fast at large s. The raw pair is gcd-reduced here and normalised with S
    monic, and the two routes are asserted equal in the test suite.
    """
    p, s = int(p), int(s)
    if s == 0:
        raise ZeroDivisionError("denominator zero")
    if s < 0:
        p, s = -p, -s
    if p <= 0:
        raise ValueError("q_rational_pair needs a positive rational p/s")
    cf = make_even_length([int(t) for t in sp.continued_fraction(sp.Rational(p, s))])
    num, den = continuant_pair(cf)
    g = num.gcd(den)
    num, den = num.exquo(g), den.exquo(g)
    # Normalise so S is monic; the MGO denominator is monic with S(0) = 1,
    # so after exact division the leading unit is +/-1.
    lc = den.LC()
    if lc != 1:
        num = num.quo_ground(lc)
        den = den.quo_ground(lc)
    return num, den


def q_rational(p: int, s: int) -> sp.Expr:
    """[p/s]_q as a reduced rational function in q, for integers p, s != 0."""
    p = int(p)
    s = int(s)
    if s == 0:
        raise ZeroDivisionError("denominator zero")
    if p == s:
        return sp.Integer(1)
    quotients = sp.continued_fraction(sp.Rational(p, s))
    return mgo_build(make_even_length(list(quotients)))
