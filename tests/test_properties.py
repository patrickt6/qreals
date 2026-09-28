"""Property-based checks over random rationals (Hypothesis).

The same identities as test_identities.py, on random reduced fractions
p/s with |p| <= 40 and s <= 12. The "ci" profile in conftest.py is
derandomized, so every run and every machine sees the same examples.

Sources: [MGO20] arXiv:1812.00170v3, [MGO22] arXiv:1908.04365v3,
[J25] arXiv:2503.02122v1 (see test_identities.py for the exact locations).
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from conftest import is_zero, rationals, taylor
from qreals import gosper_coeffs, q, q_add, q_mul, q_rational, q_real_truncated
from qreals.continuant import (
    continuant_fast,
    continuant_pair,
    continuant_series,
    continuant_symbolic,
)
from qreals.continued_fraction import make_even_length
from qreals.rational import q_rational_pair
from test_identities import hj_digits, left_version, negative_cf_q


def qr(x: Fraction) -> sp.Expr:
    return q_rational(x.numerator, x.denominator)


def even_cf(x: Fraction) -> list[int]:
    cf = sp.continued_fraction(sp.Rational(x.numerator, x.denominator))
    return make_even_length([int(t) for t in cf])


@given(rationals())
def test_translation(x: Fraction) -> None:
    """[x + 1]_q = q [x]_q + 1. Source: [MGO22] (3), [J25] (1)."""
    assert is_zero(qr(x + 1) - (q * qr(x) + 1))


@given(rationals())
def test_s_action(x: Fraction) -> None:
    """[-1/x]_q = -1/(q [x]_q). Source: [J25] (1)."""
    assert is_zero(qr(-1 / x) - (-1 / (q * qr(x))))


@given(rationals())
def test_negation_and_reciprocal_by_inversion(x: Fraction) -> None:
    """[-x]_q = -q^-1 [x]_(q^-1) and [1/x]_q = 1/[x]_(q^-1). Source: [J25] (4)."""
    X_inv = qr(x).subs(q, 1 / q)
    assert is_zero(qr(-x) + X_inv / q)
    assert is_zero(qr(1 / x) - 1 / X_inv)


@given(rationals())
def test_q_equals_one(x: Fraction) -> None:
    """[x]_q at q = 1 is x. Source: [MGO20] Section 1.1."""
    assert sp.cancel(qr(x)).subs(q, 1) == sp.Rational(x.numerator, x.denominator)


@given(rationals(min_value=1))
def test_positivity_above_one(x: Fraction) -> None:
    """For x > 1, R and S have positive coefficients with unit ends, and
    R(1) = r, S(1) = s. Source: [MGO20] Proposition 1.3, Corollary 1.7."""
    R, S = q_rational_pair(x.numerator, x.denominator)
    assert R.eval(1) == x.numerator and S.eval(1) == x.denominator
    for P in (R, S):
        c = P.all_coeffs()
        assert all(t > 0 for t in c) and c[0] == 1 and c[-1] == 1


@given(rationals(min_value=1), rationals(min_value=1))
def test_theorem_2(x: Fraction, y: Fraction) -> None:
    """R S' - S R' has non-negative coefficients when x > y > 1.
    Source: [MGO20] Theorem 2."""
    if x == y:
        return
    big, small = (x, y) if x > y else (y, x)
    R, S = q_rational_pair(big.numerator, big.denominator)
    R2, S2 = q_rational_pair(small.numerator, small.denominator)
    coeffs = (R * S2 - S * R2).all_coeffs()
    assert all(c >= 0 for c in coeffs) and coeffs[0] > 0


@given(rationals(min_value=1))
def test_negative_continued_fraction_reference(x: Fraction) -> None:
    """Engine = [MGO20] formula (1.2), a reference that shares no code with
    the package. Source: [MGO20] Theorem 1."""
    assert is_zero(qr(x) - negative_cf_q(hj_digits(x)))


@given(rationals(min_value=0))
def test_jouteur_left_version(x: Fraction) -> None:
    """N_q . [x]_q = [-x]^flat_q on rationals. Source: [J25] Theorem 1.5."""
    X = qr(x)
    J = (-X + 1 - 1 / q) / ((q - 1) * X + 1)
    assert is_zero(J - left_version(-x))


# --------------------------------------------------------------------------
# cross-path agreement
# --------------------------------------------------------------------------


@given(rationals(min_value=0))
def test_symbolic_pair_and_gosper_routes_agree(x: Fraction) -> None:
    """q_rational (sp.cancel of the block product), q_rational_pair (Poly
    product, gcd-reduced) and gosper.q_real_rational agree exactly.

    These share the block definition continuant.mgo_block; the independent
    check of that definition is the negative CF reference above.
    """
    from qreals.gosper import q_real_rational

    R, S = q_rational_pair(x.numerator, x.denominator)
    X = qr(x)
    assert is_zero(X - R.as_expr() / S.as_expr())
    assert is_zero(X - q_real_rational(x))
    word = even_cf(x)
    assert is_zero(X - continuant_symbolic(word))
    Rp, Sp = continuant_pair(word)
    assert is_zero(X - Rp.as_expr() / Sp.as_expr())


@given(rationals(min_value=0), st.integers(min_value=1, max_value=30))
def test_truncated_series_equals_exact_taylor(x: Fraction, n: int) -> None:
    """q_real_truncated on a rational = Taylor expansion of q_rational.

    The Taylor coefficients come from test-local long division
    (conftest.taylor), not from the package's series kernel.
    """
    got = q_real_truncated(f"{x.numerator}/{x.denominator}", n)
    assert got == taylor(qr(x), n)


@given(rationals(min_value=1), st.integers(min_value=1, max_value=40))
def test_fast_and_series_backends_agree(x: Fraction, n: int) -> None:
    """continuant_fast (multiplication-only) = continuant_series (one
    inversion per term) on words with a_1 >= 1."""
    word = even_cf(x)
    fast = continuant_fast(word, n)
    v, c = continuant_series(word, n)
    series_coeffs = [0] * n
    for k in range(n):
        if 0 <= k - v < len(c):
            series_coeffs[k] = c[k - v]
    assert fast == series_coeffs


@given(
    st.lists(st.integers(min_value=1, max_value=6), min_size=2, max_size=8).filter(
        lambda w: len(w) % 2 == 0
    ),
    st.integers(min_value=1, max_value=40),
)
def test_backends_agree_on_raw_words(word: list[int], n: int) -> None:
    """The fast and series folds agree on arbitrary positive even words, and
    both agree with the Taylor series of the symbolic fold."""
    fast = continuant_fast(word, n)
    v, c = continuant_series(word, n)
    assert v == 0
    assert fast == (c + [0] * n)[:n]
    assert fast == taylor(continuant_symbolic(word), n)


@given(rationals(min_value=0, max_num=12, max_den=5), rationals(min_value=0, max_num=12, max_den=5))
def test_series_arithmetic_matches_gosper(x: Fraction, y: Fraction) -> None:
    """q_add / q_mul (coefficient lists) = the bihomographic gosper engine
    (a 2x4 state machine over rational functions), on rationals."""
    N = 10
    sx, sy = f"{x.numerator}/{x.denominator}", f"{y.numerator}/{y.denominator}"
    assert q_add(sx, sy, N) == gosper_coeffs(x, y, "add", N)
    assert q_mul(sx, sy, N) == gosper_coeffs(x, y, "mul", N)


# --------------------------------------------------------------------------
# exactness: no floats anywhere in the outputs
# --------------------------------------------------------------------------


def _no_float(expr: sp.Expr) -> bool:
    return not any(isinstance(a, sp.Float) for a in sp.preorder_traversal(expr))


@given(rationals())
def test_outputs_are_exact(x: Fraction) -> None:
    """Every output is int, sympy Integer/Rational, or a Poly over ZZ."""
    X = qr(x)
    assert _no_float(X)
    if x > 0:
        R, S = q_rational_pair(x.numerator, x.denominator)
        assert R.domain == sp.ZZ and S.domain == sp.ZZ
        coeffs = q_real_truncated(f"{x.numerator}/{x.denominator}", 12)
        assert all(type(c) is int for c in coeffs)


def test_irrational_outputs_are_ints() -> None:
    """Truncated coefficients of irrationals are Python ints, never floats."""
    for x in ["pi", "E", "sqrt(2)", "(1+sqrt(5))/2", "2**(1/3)"]:
        assert all(type(c) is int for c in q_real_truncated(x, 25))
