r"""The MGO continued-fraction fold: one primitive, five backends.

Every q-real in this package is built by the same recurrence, the
Morier-Genoud and Ovsienko (MGO) q-deformation of the regular continued
fraction. For an even-length CF word a = [a_1, ..., a_2m] the fold is the
2x2 block product

    M(a) = B_1(a_1) B_2(a_2) ... B_2m(a_2m),

    B_i(a) = [[ [a]_q,     q^a  ], [1, 0]]   at odd  1-indexed positions,
    B_i(a) = [[ [a]_{1/q}, q^-a ], [1, 0]]   at even 1-indexed positions,

with [a]_q = 1 + q + ... + q^{a-1} and [a]_{1/q} = q^{-(a-1)} [a]_q, and the
q-real is the first-column ratio

    [x]_q = M(a)[0, 0] / M(a)[1, 0].

This module is the one place that owns that recurrence. Five backends
evaluate it in different representations; each states its block form in
terms of the definition above, and the callers in `rational`, `truncated`,
`fastseries`, `gosper` and `q_sum` are thin delegations:

  * continuant_matrix: the literal product, a sympy 2x2 Matrix with entries
    in Z[q, q^{-1}]. Works for any integer quotients, including negatives.
  * continuant_pair: the same product with every even-position block scaled
    by q^a, which clears the q^{-1} entries so the product lives in sympy
    Polys over ZZ (the scaling cancels in the first-column ratio):
    q^a [a]_{1/q} = q [a]_q, so the scaled block is [[q [a]_q, 1], [q^a, 0]].
    Needs non-negative quotients (a Poly cannot hold q^{-a}).
  * continuant_symbolic: the first-column ratio of continuant_matrix as one
    cancelled sympy expression; sp.cancel of the descending fold
    term_i + above_i / (...) reaches the identical canonical form.
  * continuant_series: the descending fold over the truncated-series kernel
    in `series`, with one series inversion per CF term. Each block enters as
    its (term, above) pair: term_i is the series of [a]_q or [a]_{1/q},
    above_i is the monomial q^a or q^-a. Handles any quotients the kernel
    can invert, including the Laurent range of negative inputs.
  * continuant_fast: the fold as a numerator/denominator pair of truncated
    Laurent polynomials, term + above / (N/D) = (term N + above D) / N,
    which is multiplication-only with a single long division at the end.
    It covers words with a_1 >= 1 and a_i >= 1 elsewhere, and raises
    ArithmeticError on anything else so callers fall back to
    continuant_series (the contract `truncated.q_real_truncated` relies on).

The q-integer building blocks [n]_q and [n]_{1/q} are defined here in both
the symbolic and the series representation; `rational` and `truncated`
re-export them under their historical names.
"""

from __future__ import annotations

from typing import TypeVar

import sympy as sp

from . import series
from .series import Series

q = sp.Symbol("q")


# ----------------------------------------------------------------------------
# block entries, symbolic representation
# ----------------------------------------------------------------------------
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


def mgo_block(i: int, a: int) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """The entries (b00, b01, b10, b11) of B_{i+1}(a), 0-indexed position i.

    Even i (1-indexed odd) is [[ [a]_q, q^a ], [1, 0]]; odd i is
    [[ [a]_{1/q}, q^-a ], [1, 0]], exactly the module-docstring recurrence.
    """
    if i % 2 == 0:
        return q_int(a), q**a, sp.Integer(1), sp.Integer(0)
    return q_int_qinv(a), q ** (-a), sp.Integer(1), sp.Integer(0)


# ----------------------------------------------------------------------------
# block entries, series representation
# ----------------------------------------------------------------------------
def q_int_series(n: int, prec: int) -> Series:
    """[n]_q as a series at q = 0, truncated to q^prec."""
    n = int(n)
    if n == 0:
        return 0, []
    if n > 0:
        return series.normalise((0, [1] * min(n, prec)))
    # [-m]_q = -[m]_q / q^m, so valuation -m with all coefficients -1.
    m = -n
    return series.trim(series.normalise((-m, [-1] * m)), prec)


def q_int_qinv_series(n: int, prec: int) -> Series:
    """[n]_{q^{-1}} = q^{-(n-1)} [n]_q for n > 0, truncated to q^prec."""
    n = int(n)
    if n == 0:
        return 0, []
    if n > 0:
        coeffs = [1] * min(n, prec - (-(n - 1)))
        return series.trim(series.normalise((-(n - 1), coeffs)), prec)
    m = -n
    return series.scalar_mul(
        series.mul(q_int_qinv_series(m, prec), series.q_pow(m, prec), prec), -1, prec
    )


# ----------------------------------------------------------------------------
# the shared 2x2 fold
# ----------------------------------------------------------------------------
# The fold is the same for the symbolic and the Poly backend; only the block
# entries differ. Entries are anything with ring + and * (sp.Expr, sp.Poly).
_T = TypeVar("_T", sp.Expr, sp.Poly)

Mat2 = tuple[_T, _T, _T, _T]


def _mat2_mul(m: Mat2[_T], b: Mat2[_T]) -> Mat2[_T]:
    """Row-major 2x2 product (m00, m01, m10, m11) * (b00, b01, b10, b11)."""
    m00, m01, m10, m11 = m
    b00, b01, b10, b11 = b
    return (
        m00 * b00 + m01 * b10,
        m00 * b01 + m01 * b11,
        m10 * b00 + m11 * b10,
        m10 * b01 + m11 * b11,
    )


def continuant_matrix(word: list[int]) -> sp.Matrix:
    """M(word) as a sympy 2x2 Matrix; the first column is (R, S).

    The literal block product of the module-docstring recurrence, with
    entries in Z[q, q^{-1}]. An empty word gives the identity.
    """
    m: Mat2[sp.Expr] = (sp.Integer(1), sp.Integer(0), sp.Integer(0), sp.Integer(1))
    for i, a in enumerate(word):
        m = _mat2_mul(m, mgo_block(i, a))
    return sp.Matrix([[m[0], m[1]], [m[2], m[3]]])


def _qint_poly(n: int) -> sp.Poly:
    """[n]_q for n >= 0 as a Poly over ZZ."""
    return sp.Poly([1] * int(n) if n > 0 else [0], q, domain="ZZ")


def _poly_block(i: int, a: int) -> Mat2[sp.Poly]:
    """The q^a-scaled MGO block at 0-indexed position i, as Polys over ZZ.

    Odd 1-indexed positions keep the recurrence block [[ [a]_q, q^a ],
    [1, 0]] unchanged; even positions are scaled by q^a, turning
    [[ [a]_{1/q}, q^-a ], [1, 0]] into [[ q [a]_q, 1 ], [q^a, 0]]
    (q^a [a]_{1/q} = q [a]_q). The scaling cancels in the first-column ratio.
    """
    one = sp.Poly(1, q, domain="ZZ")
    zero = sp.Poly(0, q, domain="ZZ")
    if i % 2 == 0:
        return _qint_poly(a), sp.Poly(q**a, q, domain="ZZ"), one, zero
    return (
        sp.Poly(q, q, domain="ZZ") * _qint_poly(a),
        one,
        sp.Poly(q**a, q, domain="ZZ"),
        zero,
    )


def continuant_pair(word: list[int]) -> tuple[sp.Poly, sp.Poly]:
    """The first column (R, S) of the scaled block product, as Polys over ZZ.

    Same recurrence as continuant_matrix, with every even-position block
    scaled by q^a so the product stays polynomial; the common scale factor
    q^{a_2 + a_4 + ...} multiplies both R and S and cancels in R / S. The
    pair is returned raw (not gcd-reduced). Quotients must be non-negative,
    the shape sp.continued_fraction produces for a positive rational.
    """
    m: Mat2[sp.Poly] = (
        sp.Poly(1, q, domain="ZZ"),
        sp.Poly(0, q, domain="ZZ"),
        sp.Poly(0, q, domain="ZZ"),
        sp.Poly(1, q, domain="ZZ"),
    )
    for i, a in enumerate(word):
        m = _mat2_mul(m, _poly_block(i, a))
    return m[0], m[2]


def continuant_symbolic(word: list[int]) -> sp.Expr:
    """The even-length MGO continued fraction as one cancelled expression.

    The first-column ratio M[0, 0] / M[1, 0] of the block product, through
    sp.cancel; identical to cancelling the descending fold
    term_i + above_i / (...). The empty word is 0 by convention.
    """
    if not word:
        return sp.Integer(0)
    m = continuant_matrix(word)
    return sp.cancel(m[0, 0] / m[1, 0])


# ----------------------------------------------------------------------------
# series backend: the descending fold with one inversion per term
# ----------------------------------------------------------------------------
def continuant_series(word: list[int], prec: int) -> Series:
    """The fold over the truncated-series kernel, one inversion per term.

    Folds the recurrence bottom-up as result_i = term_i + above_i / result_
    {i+1}, where (term_i, above_i) are the top-row block entries of B_{i+1}
    in series form: ([a]_q, q^a) at odd 1-indexed positions, ([a]_{1/q},
    q^-a) at even ones. This is the general-case backend: it accepts any
    quotients the series kernel can invert, at O(len(word) * prec^2) cost.
    The empty word is the zero series.
    """
    n = len(word)
    if n == 0:
        return 0, []

    def term(i: int, ai: int) -> Series:
        return (
            q_int_series(ai, prec) if (i + 1) % 2 == 1 else q_int_qinv_series(ai, prec)
        )

    def num_above(i: int, ai: int) -> Series:
        return series.q_pow(ai if (i + 1) % 2 == 1 else -ai, prec)

    result = term(n - 1, word[n - 1])
    for i in range(n - 2, -1, -1):
        inv = series.invert(result, prec)
        result = series.add(
            term(i, word[i]), series.mul(num_above(i, word[i]), inv, prec), prec
        )
    return result


# ----------------------------------------------------------------------------
# fast backend: multiplication-only fold, one long division at the end
# ----------------------------------------------------------------------------
# A truncated Laurent polynomial: (valuation, coefficients low-to-high).
# The zero polynomial is (0, []).
LPoly = tuple[int, list[int]]

# Extra coefficients kept beyond the requested precision at every fold step,
# absorbing the small valuation drift between numerator and denominator.
_PAD = 64


def _lmul(a: LPoly, b: LPoly, keep_below: int) -> LPoly:
    """a * b, dropping exponents >= keep_below."""
    va, ca = a
    vb, cb = b
    if not ca or not cb:
        return (0, [])
    v = va + vb
    width = min(len(ca) + len(cb) - 1, keep_below - v)
    if width <= 0:
        return (0, [])
    out = [0] * width
    short, long_ = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    for i, si in enumerate(short):
        if si == 0 or i >= width:
            continue
        lim = min(len(long_), width - i)
        if si == 1:
            for j in range(lim):
                out[i + j] += long_[j]
        else:
            for j in range(lim):
                out[i + j] += si * long_[j]
    return (v, out)


def _ladd(a: LPoly, b: LPoly) -> LPoly:
    va, ca = a
    vb, cb = b
    if not ca:
        return b
    if not cb:
        return a
    v = min(va, vb)
    end = max(va + len(ca), vb + len(cb))
    out = [0] * (end - v)
    for i, x in enumerate(ca):
        out[va - v + i] += x
    for i, x in enumerate(cb):
        out[vb - v + i] += x
    return (v, out)


def _lstrip(a: LPoly) -> LPoly:
    """Drop leading and trailing zero coefficients, adjusting the valuation."""
    v, c = a
    i = 0
    while i < len(c) and c[i] == 0:
        i += 1
    if i == len(c):
        return (0, [])
    j = len(c)
    while c[j - 1] == 0:
        j -= 1
    return (v + i, c[i:j])


def _term_pair(pos: int, ai: int) -> tuple[LPoly, LPoly]:
    """(term, above) at 0-indexed position pos: the top-row entries of the
    recurrence block B_{pos+1} as truncated Laurent polynomials. Odd
    1-indexed positions carry [a]_q with q^a above; even positions carry
    [a]_{1/q} = q^{-(a-1)} [a]_q with q^-a above."""
    if (pos + 1) % 2 == 1:
        return (0, [1] * ai), (ai, [1])
    return (-(ai - 1), [1] * ai), (-ai, [1])


def continuant_fast(word: list[int], prec: int) -> list[int]:
    """First prec Taylor coefficients of [x]_q for the even-length CF word.

    Folds the recurrence as a numerator/denominator pair, term + above /
    (N/D) = (term N + above D) / N, which is multiplication-only, then
    performs one long division at the end. Truncation soundness: each fold
    step is linear in (N, D) and the final quotient mod q^prec depends only
    on N and D mod q^(prec + small drift), so both are trimmed to a padded
    window after every step, with the common valuation normalised away.

    Raises ArithmeticError on any convention or drift violation (odd or
    empty word, non-positive partial quotient, quotient valuation or leading
    coefficient off the expected 1 + O(q) shape of [x]_q for x >= 1);
    callers fall back to continuant_series, which handles the general case.
    """
    n = len(word)
    if n == 0 or n % 2 != 0:
        raise ArithmeticError("need a non-empty even-length CF word")
    if any(a <= 0 for a in word[1:]) or word[0] < 1:
        raise ArithmeticError(f"non-positive partial quotient in {word[:8]}...")

    N, _ = _term_pair(n - 1, word[n - 1])
    D: LPoly = (0, [1])
    for pos in range(n - 2, -1, -1):
        t, ab = _term_pair(pos, word[pos])
        v = min(N[0], D[0])
        keep_below = v + prec + _PAD
        N, D = _ladd(_lmul(t, N, keep_below), _lmul(ab, D, keep_below)), N
        # Normalise the common valuation away and re-trim.
        N, D = _lstrip(N), _lstrip(D)
        v = min(N[0], D[0])
        N = (N[0] - v, N[1])
        D = (D[0] - v, D[1])

    N, D = _lstrip(N), _lstrip(D)
    off = N[0] - D[0]  # valuation of the quotient
    if off != 0:
        raise ArithmeticError(f"quotient valuation {off}, expected 0 for x >= 1")
    d = D[1]
    d0 = d[0]
    if d0 not in (1, -1):
        raise ArithmeticError(f"denominator lowest coefficient {d0}, expected +-1")
    num = list(N[1]) + [0] * max(0, prec - len(N[1]))
    out = [0] * prec
    for k in range(prec):
        if k < len(num):
            c = num[k] * d0  # divide by +-1
        else:
            c = 0
        out[k] = c
        if c:
            lim = min(len(d), (len(num) - k))
            for j in range(1, lim):
                num[k + j] -= c * d[j]
    if out[0] != 1:
        raise ArithmeticError(f"c_0 = {out[0]}, expected 1 for x >= 1")
    return out
