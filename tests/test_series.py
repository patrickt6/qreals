import pytest

from qreals import series


def test_invert_roundtrip_with_unit_leading_coefficient():
    prec = 30
    a = series.normalise((0, [1, 1, 1]))
    prod = series.mul(a, series.invert(a, prec), prec)
    assert prod == (0, [1])


def test_invert_roundtrip_with_negative_leading_coefficient():
    prec = 20
    a = series.normalise((0, [-1, 2, 3]))
    prod = series.mul(a, series.invert(a, prec), prec)
    assert prod == (0, [1])


def test_addition_is_commutative():
    prec = 20
    a = (0, [1, 2, 3])
    b = (1, [4, 5])
    assert series.add(a, b, prec) == series.add(b, a, prec)


def test_q_pow_at_or_beyond_precision_is_zero():
    assert series.q_pow(50, 10) == (0, [])


def test_invert_zero_series_raises():
    with pytest.raises(ZeroDivisionError):
        series.invert((0, []), 10)


def test_invert_non_unit_leading_over_integers_raises():
    # A leading coefficient other than +-1 is not a unit in Z[[q]], so the
    # integer inverse does not exist and the Fraction fallback rejects it.
    with pytest.raises(ValueError):
        series.invert((0, [2, 1]), 10)
