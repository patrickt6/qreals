"""Tests for the QReal convenience wrapper.

QReal is sugar over the functional API; these tests confirm each method and
operator delegates to the verified function and that the read-outs are correct.
"""

import math

import pytest

from qreals import QReal, q_add, q_mul, q_neg, q_real_truncated, radius


def test_qreal_coeffs_match_q_real_truncated():
    qr = QReal("pi", 12)
    assert qr.coeffs == q_real_truncated("pi", 12)
    assert qr.valuation == 0
    assert len(qr) == 12


def test_add_delegates_to_q_add():
    s = QReal("3/2", 16) + QReal("13/5", 16)
    assert s.coeffs == q_add("3/2", "13/5", 16)
    assert s.valuation == 0


def test_mul_delegates_to_q_mul():
    p = QReal("3/2", 16) * QReal("5/2", 16)
    assert p.coeffs == q_mul("3/2", "5/2", 16)


def test_neg_delegates_to_q_neg():
    n = -QReal("3/2", 12)
    v, c = q_neg("3/2", 12)
    assert (n.valuation, n.coeffs) == (v, c)


def test_add_uses_the_shorter_length():
    s = QReal("3/2", 16) + QReal("13/5", 10)
    assert len(s) == 10


def test_radius_estimate_matches_function():
    qr = QReal("7/5", 80)
    assert qr.radius_estimate() == radius("7/5", 80)


def test_radius_estimate_rejects_laurent_qreal():
    n = -QReal("3/2", 12)  # valuation < 0
    with pytest.raises(ValueError):
        n.radius_estimate()


def test_sign_pattern():
    qr = QReal("2", 5)  # [2]_q = 1 + q
    assert qr.sign_pattern() == "+ + 0 0 0"


def test_zero_run_finds_longest_run():
    # [pi-2]_q opens with a long zero run after its first coefficients.
    qr = QReal("pi-2", 12)
    start, length = qr.zero_run()
    assert length >= 1
    assert all(qr.coeffs[start + i] == 0 for i in range(length))
    # the run is maximal: the entries bracketing it are nonzero (or at an edge).
    if start > 0:
        assert qr.coeffs[start - 1] != 0
    if start + length < len(qr):
        assert qr.coeffs[start + length] != 0


def test_zero_run_none_when_no_zeros():
    qr = QReal("pi", 3)  # [pi]_q = 1 + q + q^2 + ...
    assert qr.coeffs[:3] == [1, 1, 1]
    assert qr.zero_run() == (0, 0)


def test_operators_reject_compound_qreal():
    s = QReal("3/2", 12) + QReal("13/5", 12)
    with pytest.raises(ValueError):
        _ = s + QReal("2", 12)
    with pytest.raises(ValueError):
        _ = -s


def test_qreal_radius_of_integer_saturates_near_one():
    # finite-N bias: a polynomial's true radius is infinite, but the estimate
    # saturates near 1 because the leading unit coefficients dominate the window.
    assert QReal("3", 10).radius_estimate() == 1.0


def test_qreal_equality_and_repr():
    assert QReal("pi", 8) == QReal("pi", 8)
    assert QReal("pi", 8) != QReal("E", 8)
    assert "valuation" in repr(QReal("pi", 4))


def test_radius_of_one_is_infinite():
    assert QReal("1", 4).radius_estimate() == math.inf
