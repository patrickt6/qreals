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

from .continued_fraction import make_even_length

q = sp.Symbol("q")


def q_int(n: int) -> sp.Expr:
    """[n]_q for n in Z, as an exact expression in q."""
    n = int(n)
    if n == 0:
        return sp.Integer(0)
    if n > 0:
        return sum((q**i for i in range(n)), sp.Integer(0))
    return -q_int(-n) / q ** (-n)


def q_int_qinv(n: int) -> sp.Expr:
    """[n]_{q^{-1}}, the q -> q^{-1} substitution of [n]_q."""
    n = int(n)
    if n == 0:
        return sp.Integer(0)
    if n > 0:
        return q_int(n) / q ** (n - 1)
    return -q_int_qinv(-n) * q ** (-n)


def mgo_build(a: list[int]) -> sp.Expr:
    """Evaluate the even-length MGO continued fraction symbolically."""
    n = len(a)
    if n == 0:
        return sp.Integer(0)

    def term(i: int, ai: int) -> sp.Expr:
        return q_int(ai) if (i + 1) % 2 == 1 else q_int_qinv(ai)

    def num_above(i: int, ai: int) -> sp.Expr:
        return q**ai if (i + 1) % 2 == 1 else q ** (-ai)

    result = term(n - 1, a[n - 1])
    for i in range(n - 2, -1, -1):
        result = term(i, a[i]) + num_above(i, a[i]) / result
    return sp.cancel(result)


def _qint_poly(n: int) -> sp.Poly:
    """[n]_q for n >= 0 as a Poly over ZZ."""
    return sp.Poly([1] * int(n) if n > 0 else [0], q, domain="ZZ")


def q_rational_pair(p: int, s: int) -> tuple[sp.Poly, sp.Poly]:
    """[p/s]_q as a reduced pair (N, S) of polynomials over ZZ, for p/s > 0.

    Same value as q_rational, computed through the continuant matrix product
    with polynomial arithmetic instead of symbolic cancellation, so it stays
    fast at large s. Each even-position MGO block is scaled by q^{a_i} to
    clear the q^{-1} entries (the scaling cancels in the final ratio):
    q^a [a]_{q^{-1}} = q [a]_q, so the scaled block is [[q [a]_q, 1],
    [q^a, 0]]. The result is normalised with S monic, and the two routes are
    asserted equal in the test suite.
    """
    p, s = int(p), int(s)
    if s == 0:
        raise ZeroDivisionError("denominator zero")
    if s < 0:
        p, s = -p, -s
    if p <= 0:
        raise ValueError("q_rational_pair needs a positive rational p/s")
    cf = make_even_length([int(t) for t in sp.continued_fraction(sp.Rational(p, s))])
    one = sp.Poly(1, q, domain="ZZ")
    zero = sp.Poly(0, q, domain="ZZ")
    m = [[one, zero], [zero, one]]
    for i, a in enumerate(cf):
        if (i + 1) % 2 == 1:
            blk = [[_qint_poly(a), sp.Poly(q**a, q, domain="ZZ")], [one, zero]]
        else:
            blk = [
                [sp.Poly(q, q, domain="ZZ") * _qint_poly(a), one],
                [sp.Poly(q**a, q, domain="ZZ"), zero],
            ]
        m = [
            [
                m[0][0] * blk[0][0] + m[0][1] * blk[1][0],
                m[0][0] * blk[0][1] + m[0][1] * blk[1][1],
            ],
            [
                m[1][0] * blk[0][0] + m[1][1] * blk[1][0],
                m[1][0] * blk[0][1] + m[1][1] * blk[1][1],
            ],
        ]
    num, den = m[0][0], m[1][0]
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
