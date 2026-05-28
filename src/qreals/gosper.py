r"""The q-deformed bihomographic engine, an independent route to q-real arithmetic.

This is the q-analogue of the classical bihomographic continued-fraction state
machine (a "Gosper" engine). The classical engine carries a 2x4 integer state
and ingests partial quotients of x and y by right-multiplying by the
Kronecker-factored blocks A(t) (x) I and I (x) A(t), with the continued-fraction
block A(t) = [[t, 1], [1, 0]]. Here every entry is promoted to its MGO
q-deformed counterpart, so the state carries Laurent polynomials in q and the
engine computes a bilinear function of the q-reals [x]_q and [y]_q directly from
the continued fractions of x and y.

The MGO q-block, in the exact regular-CF convention `rational` and `truncated`
use. For an even-length regular continued fraction a = [a_1, ..., a_{2m}],

    A_q^{(i)}(a) = [[ [a]_q    , q^{ a} ], [1, 0]]   at odd  positions (1-indexed),
                   [[ [a]_{1/q}, q^{-a} ], [1, 0]]   at even positions,

and [x]_q = R_x / S_x where (R_x, S_x)^T is the first column of the product of
the blocks. With monomial columns (XY, X, Y, 1) the 2x4 state
S = [[a,b,c,d],[e,f,g,h]] represents z(X, Y) = (aXY+bX+cY+d)/(eXY+fX+gY+h).
Ingesting an x-quotient right-multiplies by A_q (x) I; a y-quotient by I (x) A_q.
After full ingestion the first-column ratio of S . (prod P_q)(prod Q_q) is
z([x]_q, [y]_q): for op="add" that is [x]_q + [y]_q, for op="mul" it is
[x]_q * [y]_q.

Why this is here. `arithmetic.q_add` and `arithmetic.q_mul` compute the same
quantities the cheap way, by expanding each q-real series and adding or
convolving the coefficient lists. This engine reaches the result by a different
algorithm (a state machine over rational functions in q, never forming the two
series separately), so agreement between the two is a real cross-check rather
than a re-run of one code path. The test suite uses it exactly that way; see
docs/CORRECTNESS.md.

Caveat (the load-bearing one). For op="add" this computes [x]_q + [y]_q, the sum
of the two q-series, NOT [x+y]_q, the q-deformation of the real sum. The MGO map
x |-> [x]_q is not additive, so these differ already at q^0 (constant term 2 vs
1); the deficit D = [x+y]_q - ([x]_q + [y]_q) is a separate object and is not
what this engine returns.
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp

from .continued_fraction import make_even_length
from .rational import q, q_int, q_int_qinv

# Bilinear operation coefficients in monomial order (XY, X, Y, 1), two rows.
# These are the only operations the engine is verified for in this package.
_OPS: dict[str, tuple[int, int, int, int, int, int, int, int]] = {
    "add": (0, 1, 1, 0, 0, 0, 0, 1),  # z = X + Y  -> [x]_q + [y]_q
    "mul": (1, 0, 0, 0, 0, 0, 0, 1),  # z = X * Y  -> [x]_q * [y]_q
}


def q_cf(fr: Fraction) -> list[int]:
    """The even-length regular continued fraction the engine ingests for fr."""
    quotients = sp.continued_fraction(sp.Rational(fr.numerator, fr.denominator))
    return make_even_length([int(t) for t in quotients])


def q_block(i: int, a: int) -> sp.Matrix:
    """The MGO 2x2 q-continuant block at 0-indexed position i with digit a.

    Even i (1-indexed odd) carries [a]_q with q^{a} above; odd i (1-indexed even)
    carries [a]_{1/q} with q^{-a} above, matching `rational.mgo_build`.
    """
    if i % 2 == 0:
        return sp.Matrix([[q_int(a), q**a], [1, 0]])
    return sp.Matrix([[q_int_qinv(a), q ** (-a)], [1, 0]])


def kron(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    """Kronecker product in the (XY, X, Y, 1) = (X, 1) (x) (Y, 1) block layout."""
    return sp.Matrix(
        sp.BlockMatrix(
            [[left[i, j] * right for j in range(left.cols)] for i in range(left.rows)]
        )
    )


def q_convergent_matrix(cf: list[int]) -> sp.Matrix:
    """Product of MGO q-blocks for an even-length CF; first column is (R, S)."""
    m = sp.eye(2)
    for i, a in enumerate(cf):
        m = m * q_block(i, a)
    return m


def q_real_rational(fr: Fraction) -> sp.Expr:
    """[x]_q as a reduced rational function in q, via the q-block product.

    This reproduces `rational.q_rational` for the same fraction; it is kept here
    so the engine is self-contained as an independent route.
    """
    m = q_convergent_matrix(q_cf(fr))
    return sp.cancel(m[0, 0] / m[1, 0])


def _state_matrix(op: str) -> sp.Matrix:
    if op not in _OPS:
        raise ValueError(f"op must be one of {sorted(_OPS)}, got {op!r}")
    a, b, c, d, e, f, g, h = _OPS[op]
    return sp.Matrix([[a, b, c, d], [e, f, g, h]])


def q_gosper(x: Fraction, y: Fraction, op: str = "add") -> sp.Expr:
    """z([x]_q, [y]_q) as a reduced rational function in q, via the 2x4 state.

    op="add" gives [x]_q + [y]_q; op="mul" gives [x]_q * [y]_q. Inputs are
    rationals (the continued fraction must terminate for the state to close).
    """
    m = _state_matrix(op)
    for i, a in enumerate(q_cf(x)):
        m = m * kron(q_block(i, a), sp.eye(2))
    for j, b in enumerate(q_cf(y)):
        m = m * kron(sp.eye(2), q_block(j, b))
    return sp.cancel(m[0, 0] / m[1, 0])


def _valuation(expr: sp.Expr) -> int:
    """Lowest power of q in a Laurent rational function: val(num) - val(den)."""
    expr = sp.cancel(expr)
    if expr == 0:
        return 0
    num, den = sp.fraction(expr)
    pn = sp.Poly(sp.expand(num), q)
    pd = sp.Poly(sp.expand(den), q)
    vn = min(m[0] for m in pn.monoms()) if pn.terms() else 0
    vd = min(m[0] for m in pd.monoms()) if pd.terms() else 0
    return int(vn - vd)


def laurent_coeffs(
    expr: sp.Expr, hi: int, lo: int | None = None
) -> tuple[int, list[int]]:
    """Integer Laurent coefficients of expr for exponents lo..hi (inclusive).

    Returns (lo, [c_lo, ..., c_hi]). When lo is None it defaults to the
    valuation, so the list starts at the first possibly-nonzero coefficient.
    """
    expr = sp.cancel(expr)
    if lo is None:
        lo = _valuation(expr) if expr != 0 else 0
    n = hi - lo + 1
    if n <= 0:
        return lo, []
    shifted = sp.cancel(expr * q ** (-lo))  # regular at q = 0
    ser = shifted.series(q, 0, n).removeO()
    poly = sp.Poly(sp.expand(ser), q)
    out = [int(poly.coeff_monomial(q**k)) for k in range(n)]
    return lo, out


def gosper_coeffs(x: Fraction, y: Fraction, op: str, N: int) -> list[int]:
    """First N Taylor coefficients [c_0..c_{N-1}] of z([x]_q, [y]_q).

    For x, y > 0 both q-reals are regular at q = 0 and the result is a Taylor
    series, so the valuation is taken to be 0.
    """
    _, coeffs = laurent_coeffs(q_gosper(x, y, op), N - 1, lo=0)
    return coeffs
