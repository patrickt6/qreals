"""Unit tests for the numeric kernel and the continued-fraction utilities.

These pin the contracts the higher layers rely on (valuations, truncation,
the ArithmeticError fallback, the even-length normalisation), checked
against values computed by hand or by test-local arithmetic.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from conftest import is_zero, series_expr, taylor
from qreals import q, q_rational, q_real_truncated, series
from qreals.continuant import (
    _ladd,
    _lmul,
    _lstrip,
    continuant_fast,
    continuant_matrix,
    continuant_pair,
    continuant_series,
    continuant_symbolic,
    mgo_block,
    q_int,
    q_int_qinv,
    q_int_qinv_series,
    q_int_series,
)
from qreals.continued_fraction import cf_partials, make_even_length
from qreals.rational import q_rational_pair

# --------------------------------------------------------------------------
# series kernel
# --------------------------------------------------------------------------


def dense(s: series.Series, lo: int, hi: int) -> list[int]:
    """Coefficients of q^lo .. q^(hi-1) of a kernel series."""
    v, c = s
    return [c[k - v] if 0 <= k - v < len(c) else 0 for k in range(lo, hi)]


def test_trim_and_normalise() -> None:
    assert series.trim((0, [1, 2, 3, 4]), 2) == (0, [1, 2])
    assert series.trim((3, [1, 2]), 2) == (3, [])
    assert series.trim((-2, [1, 2, 3]), 5) == (-2, [1, 2, 3])
    assert series.trim((0, [1, 2, 3]), 3) == (0, [1, 2, 3])
    assert series.normalise((0, [0, 0, 5, 0])) == (2, [5])
    assert series.normalise((-1, [0, 0])) == (0, [])
    assert series.normalise((4, [7])) == (4, [7])
    assert series.normalise((1, [3, 0, 2])) == (1, [3, 0, 2])


def test_add_mul_scalar() -> None:
    a = (0, [1, 1, 1])  # 1 + q + q^2
    b = (-1, [2, 0, 1])  # 2/q + q
    assert dense(series.add(a, b, 10), -1, 3) == [2, 1, 2, 1]
    assert dense(series.add(a, b, 2), -1, 3) == [2, 1, 2, 0]
    assert series.add((0, [1]), (0, [-1]), 5) == (0, [])
    assert dense(series.mul(a, b, 10), -1, 4) == [2, 2, 3, 1, 1]
    assert dense(series.mul(a, b, 1), -1, 4) == [2, 2, 0, 0, 0]
    assert series.mul((0, []), a, 5) == (0, [])
    assert series.mul(a, (0, []), 5) == (0, [])
    assert series.scalar_mul(a, 0, 5) == (0, [])
    assert series.scalar_mul(a, -3, 5) == (0, [-3, -3, -3])
    assert series.add_int(a, 4, 5) == (0, [5, 1, 1])
    assert series.add_int(a, 0, 5) == (0, [1, 1, 1])
    assert series.q_pow(3, 5) == (3, [1])
    assert series.q_pow(5, 5) == (0, [])
    assert series.q_pow(-2, 5) == (-2, [1])


def test_mul_skips_zero_coefficients() -> None:
    a = (0, [1, 0, 2])
    b = (0, [3, 4])
    assert dense(series.mul(a, b, 10), 0, 5) == [3, 4, 6, 8, 0]


@given(
    st.integers(min_value=-4, max_value=4),
    st.lists(st.integers(min_value=-5, max_value=5), min_size=1, max_size=8),
    st.sampled_from([1, -1]),
    st.integers(min_value=1, max_value=25),
)
def test_invert_unit_leading(v: int, tail: list[int], sign: int, prec: int) -> None:
    """a * invert(a) = 1 through q^(prec - 1), for a leading unit.

    invert(a, P) keeps exponents below P only, so for a of valuation v < 0
    the inverse is requested to P = prec - v to make the product exact
    through q^(prec - 1).
    """
    a = (v, [sign] + tail)
    short = series.invert(a, prec)
    assert all(short[0] + i < prec for i in range(len(short[1])))
    P = prec + max(0, -v)
    inv = series.invert(a, P)
    assert inv[0] == -v and inv[1][0] == sign
    prod = series.mul(a, inv, prec)
    lo = min(0, prod[0])
    assert dense(prod, lo, prec) == [1 if k == 0 else 0 for k in range(lo, prec)]


def test_invert_general_leading() -> None:
    """A non-unit leading coefficient routes to the Fraction fallback."""
    a = (0, [2, 4])  # 2 + 4q = 2 (1 + 2q); inverse has 1/2: not integral
    with pytest.raises(ValueError):
        series.invert(a, 5)
    b = (1, [2])  # 2q: inverse q^-1 / 2 is not integral either
    with pytest.raises(ValueError):
        series.invert(b, 5)
    c = (2, [3, 3])  # 3q^2 (1 + q): the inverse carries 1/3
    with pytest.raises(ValueError):
        series.invert_general(c, 4)
    with pytest.raises(ZeroDivisionError):
        series.invert((0, []), 5)
    with pytest.raises(ZeroDivisionError):
        series.invert_general((0, []), 5)


def test_invert_general_integral_case() -> None:
    """invert_general is exact when the inverse happens to be integral:
    a = -1 * (1 - q) has leading -1 but is forced through the general path
    here directly; the result must be -(1 + q + q^2 + ...)."""
    inv = series.invert_general((0, [-1, 1]), 6)
    assert dense(inv, 0, 6) == [-1, -1, -1, -1, -1, -1]
    inv2 = series.invert_general((-2, [1, 1]), 4)
    assert inv2[0] == 2
    assert dense(inv2, 2, 4) == [1, -1]
    assert dense(series.invert_general((0, [1, 0, 1]), 7), 0, 7) == [1, 0, -1, 0, 1, 0, -1]


# --------------------------------------------------------------------------
# q-integer blocks in both representations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [-4, -3, -1, 0, 1, 2, 5])
def test_q_int_series_matches_symbolic(n: int) -> None:
    prec = 8
    for sym, ser in ((q_int(n), q_int_series(n, prec)), (q_int_qinv(n), q_int_qinv_series(n, prec))):
        lo = -10
        expected = series_expr(dense(ser, lo, prec), lo)
        diff = sp.expand(sp.cancel(sym) - expected)
        # agreement on every exponent below prec
        assert all(diff.coeff(q, k) == 0 for k in range(lo, prec)), (n, diff)


def test_q_int_series_truncates() -> None:
    assert q_int_series(10, 3) == (0, [1, 1, 1])
    assert q_int_qinv_series(4, 1) == (-3, [1, 1, 1, 1])
    assert q_int_qinv_series(4, -1) == (-3, [1, 1])
    assert q_int_series(-3, -1) == (-3, [-1, -1])


def test_mgo_block_entries() -> None:
    b00, b01, b10, b11 = mgo_block(0, 3)
    assert sp.expand(b00) == 1 + q + q**2 and b01 == q**3 and b10 == 1 and b11 == 0
    c00, c01, c10, c11 = mgo_block(1, 3)
    assert is_zero(c00 - (1 + 1 / q + 1 / q**2)) and c01 == q**-3
    assert c10 == 1 and c11 == 0


def test_empty_words() -> None:
    assert continuant_matrix([]) == sp.eye(2)
    assert continuant_symbolic([]) == 0
    assert continuant_series([], 5) == (0, [])
    R, S = continuant_pair([])
    assert R.as_expr() == 1 and S.as_expr() == 0


def test_pair_is_raw_and_scaled() -> None:
    """continuant_pair is not gcd-reduced: [2, 2] gives q (R, S) of 5/2."""
    R, S = continuant_pair([2, 2])
    assert sp.expand(R.as_expr() - q * (1 + 2 * q + q**2 + q**3)) == 0
    assert sp.expand(S.as_expr() - q * (1 + q)) == 0


# --------------------------------------------------------------------------
# the fast fold and its ArithmeticError contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word",
    [[], [2], [1, 2, 3], [0, 3], [2, 0], [2, -1], [-1, 1], [0, 1, 2, 2]],
)
def test_fast_rejects_outside_its_domain(word: list[int]) -> None:
    with pytest.raises(ArithmeticError):
        continuant_fast(word, 10)


@pytest.mark.parametrize("word", [[1, 1], [2, 2], [1, 2, 2, 1], [3, 7, 15, 1], [1, 1, 1, 1, 1, 1]])
def test_fast_matches_symbolic(word: list[int]) -> None:
    for prec in (1, 2, 7, 30):
        assert continuant_fast(word, prec) == taylor(continuant_symbolic(word), prec)


def test_fast_long_word_uses_padding() -> None:
    """A deep word (many fold steps) still matches the exact Taylor series;
    this exercises the padded trim window after each step."""
    word = [1, 2] * 40
    assert continuant_fast(word, 60) == taylor(continuant_pair(word), 60)
    word = [3, 1, 1, 4] * 15
    assert continuant_fast(word, 90) == taylor(continuant_pair(word), 90)


def test_series_fold_handles_leading_zero() -> None:
    """Words with a_1 = 0 (x < 1) go through continuant_series only."""
    for x in [Fraction(1, 2), Fraction(3, 7), Fraction(2, 9)]:
        word = make_even_length([int(t) for t in sp.continued_fraction(sp.Rational(x.numerator, x.denominator))])
        v, c = continuant_series(word, 12)
        assert dense((v, c), 0, 12) == taylor(q_rational(x.numerator, x.denominator), 12)


def test_lpoly_helpers() -> None:
    assert _lstrip((2, [0, 0, 3, 4, 0])) == (4, [3, 4])
    assert _lstrip((1, [0, 0])) == (0, [])
    assert _lstrip((0, [5])) == (0, [5])
    assert _ladd((0, []), (3, [1])) == (3, [1])
    assert _ladd((3, [1]), (0, [])) == (3, [1])
    assert _ladd((0, [1, 1]), (1, [2, 2])) == (0, [1, 3, 2])
    assert _ladd((1, [2]), (-1, [5])) == (-1, [5, 0, 2])
    assert _lmul((0, [1, 1]), (0, [1, 1]), 10) == (0, [1, 2, 1])
    assert _lmul((0, [1, 1]), (0, [1, 1]), 2) == (0, [1, 2])
    assert _lmul((0, [2, 3]), (1, [1, -1]), 10) == (1, [2, 1, -3])
    assert _lmul((5, [1]), (0, [1]), 3) == (0, [])
    assert _lmul((0, []), (0, [1]), 3) == (0, [])
    assert _lmul((0, [0, 1]), (0, [1, 1]), 10) == (0, [0, 1, 1])


# --------------------------------------------------------------------------
# continued fractions
# --------------------------------------------------------------------------


def cf_value(a: list[int]) -> Fraction:
    val = Fraction(a[-1])
    for t in reversed(a[:-1]):
        val = t + 1 / val
    return val


@pytest.mark.parametrize(
    "a,expected",
    [
        ([2, 2], [2, 2]),
        ([2], [1, 1]),
        ([1], [0, 1]),
        ([0], [-1, 1]),
        ([-3], [-4, 1]),
        ([1, 2, 2], [1, 2, 1, 1]),
        ([0, 3, 6, 3], [0, 3, 6, 3]),
        ([3, 7, 15], [3, 7, 14, 1]),
        ([2, 3, 1], [2, 4]),
        ([1, 1, 1], [1, 2]),
    ],
)
def test_make_even_length(a: list[int], expected: list[int]) -> None:
    out = make_even_length(a)
    assert out == expected
    assert len(out) % 2 == 0
    assert cf_value(out) == cf_value(a)


def test_make_even_length_does_not_mutate_and_rejects() -> None:
    a = [1, 2, 2]
    make_even_length(a)
    assert a == [1, 2, 2]
    with pytest.raises(ValueError):
        make_even_length([1, 2, 0])


@given(st.integers(min_value=1, max_value=300), st.integers(min_value=1, max_value=300))
def test_even_length_value_preserved(p: int, s: int) -> None:
    cf = [int(t) for t in sp.continued_fraction(sp.Rational(p, s))]
    out = make_even_length(cf)
    assert len(out) % 2 == 0 and cf_value(out) == Fraction(p, s)


def test_cf_partials_stopping_rule() -> None:
    """Stops at the first depth where a1 + ... + an >= max_sum + 1."""
    assert cf_partials("sqrt(2)", 0) == [1]
    assert cf_partials("sqrt(2)", 1) == [1, 2]
    assert cf_partials("sqrt(2)", 2) == [1, 2]
    assert cf_partials("sqrt(2)", 3) == [1, 2, 2]
    assert cf_partials("pi", 10) == [3, 7, 15]
    assert cf_partials("pi", 9) == [3, 7]
    assert cf_partials("3/2", 100) == [1, 2]  # finite CF ends early
    assert cf_partials("(1+sqrt(5))/2", 4) == [1, 1, 1, 1, 1]
    assert cf_partials("sqrt(2)", 10**6, max_depth=3) == [1, 2, 2, 2]
    assert cf_partials("E", 5) == [2, 1, 2, 1]


def test_cf_partials_non_periodic_irrational() -> None:
    """Cube roots have no periodic form; the iterator fallback is used."""
    assert cf_partials("2**(1/3)", 10) == [1, 3, 1, 5, 1]


# --------------------------------------------------------------------------
# truncated path: edge cases and errors
# --------------------------------------------------------------------------


def test_truncated_constants_and_errors() -> None:
    assert q_real_truncated("0", 4) == [0, 0, 0, 0]
    assert q_real_truncated("1", 4) == [1, 0, 0, 0]
    assert q_real_truncated("1", 0) == []
    assert q_real_truncated("5", 7) == [1, 1, 1, 1, 1, 0, 0]
    with pytest.raises(ValueError):
        q_real_truncated("-3/2", 5)
    with pytest.raises(ValueError):
        q_real_truncated("-sqrt(2)", 5)


def test_truncated_below_one_uses_series_fallback() -> None:
    """x in (0, 1): a_1 = 0, fast path raises, series fold takes over."""
    assert q_real_truncated("1/2", 8) == [0, 1, -1, 1, -1, 1, -1, 1]
    assert q_real_truncated("sqrt(2)-1", 10) == taylor_sqrt2_minus_1(10)


def taylor_sqrt2_minus_1(n: int) -> list[int]:
    """[sqrt(2) - 1]_q = ([sqrt(2)]_q - 1)/q by [MGO22] (3)."""
    c = q_real_truncated("sqrt(2)", n + 1)
    return c[1 : n + 1]


def test_pair_rejects_bad_input() -> None:
    with pytest.raises(ZeroDivisionError):
        q_rational_pair(3, 0)
    with pytest.raises(ValueError):
        q_rational_pair(-3, 2)
    with pytest.raises(ValueError):
        q_rational_pair(0, 2)
    with pytest.raises(ValueError):
        q_rational_pair(3, -2)  # the sign moves to p, and -3/2 is not positive
    R, S = q_rational_pair(-5, -2)
    assert sp.expand(R.as_expr() - (1 + 2 * q + q**2 + q**3)) == 0


def test_rational_rejects_zero_denominator() -> None:
    with pytest.raises(ZeroDivisionError):
        q_rational(1, 0)
    assert q_rational(4, 4) == 1
    assert is_zero(q_rational(6, 4) - q_rational(3, 2))
    assert is_zero(q_rational(-6, -4) - q_rational(3, 2))


# --------------------------------------------------------------------------
# contracts pinned after mutation testing (see docs/VERIFICATION.md)
# --------------------------------------------------------------------------


def test_zero_series_is_canonical() -> None:
    """[0]_q in series form is the canonical zero (0, [])."""
    assert q_int_series(0, 5) == (0, [])
    assert q_int_qinv_series(0, 5) == (0, [])


def test_mat2_mul_is_a_general_product() -> None:
    from qreals.continuant import _mat2_mul

    a = (sp.Integer(1), sp.Integer(2), sp.Integer(3), sp.Integer(4))
    b = (sp.Integer(5), sp.Integer(6), sp.Integer(7), sp.Integer(8))
    assert _mat2_mul(a, b) == (19, 22, 43, 50)


def test_lpoly_multiplication_is_exact_integer_arithmetic() -> None:
    """_lmul must multiply (never divide) and return Python ints."""
    v, c = _lmul((0, [1, 3]), (0, [2, 5]), 10)
    assert (v, c) == (0, [2, 11, 15])
    assert all(type(t) is int for t in c)
    v, c = _lmul((0, [2, 3]), (1, [1, -1]), 10)
    assert all(type(t) is int for t in c)
    assert _lmul((0, []), (0, [1, 1, 1]), 5) == (0, [])
    assert _lmul((5, [1]), (0, [1]), 5) == (0, [])


def test_lstrip_odd_leading_zeros() -> None:
    assert _lstrip((0, [0, 5])) == (1, [5])
    assert _lstrip((-2, [0, 0, 0, 7, 1])) == (1, [7, 1])


@pytest.mark.parametrize("word", [[0, 3], [2, 0], [2, 3, 0, 1], [0, 1, 2, 2]])
def test_fast_rejects_non_positive_quotients_at_the_guard(word: list[int]) -> None:
    """The domain guard itself fires (not a later drift check)."""
    with pytest.raises(ArithmeticError, match="non-positive partial quotient"):
        continuant_fast(word, 10)


@pytest.mark.parametrize("word", [[], [2], [1, 2, 3]])
def test_fast_rejects_odd_or_empty_words_at_the_guard(word: list[int]) -> None:
    with pytest.raises(ArithmeticError, match="even-length"):
        continuant_fast(word, 10)


def test_fast_returns_python_ints() -> None:
    for word in ([1, 2] * 20, [3, 7, 15, 1], [2, 1, 2, 1, 1, 4, 1, 1]):
        out = continuant_fast(word, 40)
        assert all(type(t) is int for t in out)


def test_series_add_truncates_both_operands() -> None:
    assert series.add((0, [1]), (0, [1, 1, 1, 1]), 2) == (0, [2, 1])
    assert series.add((0, [1, 1, 1, 1]), (0, [1]), 2) == (0, [2, 1])
    assert series.add((1, [1]), (-1, [1, 0, 0, 5]), 2) == (-1, [1, 0, 1])


def test_series_mul_drops_terms_at_or_beyond_prec() -> None:
    assert series.mul((3, [1]), (3, [1]), 5) == (0, [])
    assert series.mul((3, [1, 1]), (1, [1]), 5) == (4, [1])


def test_scalar_mul_by_one_is_identity() -> None:
    a = (2, [1, -1, 3])
    assert series.scalar_mul(a, 1, 10) == a


def test_invert_general_positive_valuation_and_long_operand() -> None:
    """The Fraction path keeps prec + v coefficients for valuation v > 0 and
    handles an operand longer than the working window."""
    inv = series.invert_general((2, [1, 1]), 4)  # 1/(q^2 (1 + q))
    assert inv[0] == -2
    assert dense(inv, -2, 4) == [1, -1, 1, -1, 1, -1]
    a = (0, [-1] + [1] * 9)
    inv = series.invert_general(a, 10)
    prod = series.mul(a, inv, 10)
    assert dense(prod, 0, 10) == [1] + [0] * 9
    assert series.invert(a, 10) == inv


def test_cf_partials_deep_periodic_expansion() -> None:
    """A quadratic surd unrolls through its periodic form past depth 250,
    where the iterator fallback would exceed the recursion limit; the
    default depth cap is 500 (501 quotients)."""
    a = cf_partials("sqrt(2)", 700)
    assert a[0] == 1 and all(t == 2 for t in a[1:]) and len(a) == 351
    assert len(cf_partials("sqrt(2)", 10**6)) == 501


def test_mgo_build_series_delegates() -> None:
    from qreals.truncated import mgo_build_series

    for word in ([2, 2], [0, 2], [1, 2, 2, 1]):
        assert mgo_build_series(word, 12) == continuant_series(word, 12)
