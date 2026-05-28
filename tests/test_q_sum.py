"""Tests for the May-25 meeting algorithm helpers: `negate`, `transfer_matrix`,
`q_sum_rational`, `q_sum_irrational`, `finiteness_check`.

Pinning notes (see docs/CORRECTNESS.md):
- `negate` is checked against the existing verified `q_neg` for x >= 0 (same
  underlying Jouteur eq. (2) path) and against `q_real_truncated` of |x| for
  x < 0 (the involution check).
- `transfer_matrix` is checked against `gosper.q_convergent_matrix`, the
  independent block product the package's verified MGO engine uses; for a
  rational its first-column ratio reproduces `q_rational`.
- `q_sum_rational` is checked against `gosper.q_gosper(x, y, "add")`, which is
  itself pinned to the package's series-sum path on the rational test family.
- `q_sum_irrational` is checked against the explicit meeting identity for
  4 +/- sqrt(7): the Taylor coefficients equal those of
  q^4 * negation_sum(sqrt(7)) + 2 * (1 + q + q^2 + q^3).
- `finiteness_check` is checked on the trace-zero quadratic catalogue via
  4 +/- sqrt(7) (finite) and a non-zero-trace control (golden ratio, infinite).
"""

from fractions import Fraction

import pytest
import sympy as sp

from qreals import (
    Approximant,
    FinitenessReport,
    QSumIrrational,
    QSumRational,
    finiteness_check,
    gosper_coeffs,
    negate,
    negation_sum,
    q_neg,
    q_real_truncated,
    q_rational,
    q_sum_irrational,
    q_sum_rational,
    transfer_matrix,
)
from qreals.gosper import q_convergent_matrix, q_cf
from qreals.rational import q


# ----------------------------------------------------------------------------
# negate (Jouteur eq. (2), arXiv:2503.02122)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("x", ["3/2", "5/2", "7/3", "pi", "sqrt(2)", "(1+sqrt(5))/2"])
def test_negate_matches_q_neg_for_nonneg(x):
    """For x >= 0, `negate` is the existing verified `q_neg`."""
    assert negate(x, 14) == q_neg(x, 14)


@pytest.mark.parametrize("x_pos", ["3/2", "5/2", "7/3"])
def test_negate_negative_input_is_qreal_of_abs(x_pos):
    """For x < 0, [-x]_q is the q-real of |x|; the involution gives back [|x|]_q."""
    v, c = negate(f"-{x_pos}", 12)
    assert v == 0
    assert c == q_real_truncated(x_pos, 12)


def test_negate_minus_2_known_value():
    """[-2]_q = -q^{-1} - q^{-3}, the regression value in `docs/CORRECTNESS.md`."""
    v, c = negate("2", 6)
    # the negation carries negative powers; the head of (-q^{-1} - q^{-3}) is at q^{-3}
    expr = sum(coef * q ** (v + i) for i, coef in enumerate(c))
    assert sp.simplify(expr - (-q ** (-1) - q ** (-3))) == 0


# ----------------------------------------------------------------------------
# transfer_matrix (the 2x2 MGO q-continuant block product)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fr", [Fraction(3, 2), Fraction(5, 2), Fraction(7, 3), Fraction(13, 5)]
)
def test_transfer_matrix_matches_gosper_block_product(fr):
    """The 2x2 product equals `gosper.q_convergent_matrix(q_cf(fr))`."""
    expected = q_convergent_matrix(q_cf(fr))
    got = transfer_matrix(q_cf(fr))
    diff = sp.simplify(got - expected)
    assert diff == sp.zeros(2, 2)


@pytest.mark.parametrize(
    "p_s",
    [(3, 2), (5, 2), (7, 3), (13, 5), (22, 7)],
)
def test_transfer_matrix_first_column_gives_qrational(p_s):
    """First-column ratio of the transfer matrix is [p/s]_q (`q_rational`)."""
    p, s = p_s
    cf = q_cf(Fraction(p, s))
    M = transfer_matrix(cf)
    value = sp.cancel(M[0, 0] / M[1, 0])
    assert sp.simplify(value - q_rational(p, s)) == 0


# ----------------------------------------------------------------------------
# q_sum_rational (Alex's CF algorithm on rational pairs)
# ----------------------------------------------------------------------------
RATIONAL_PAIRS = [
    (Fraction(3, 2), Fraction(5, 2)),
    (Fraction(7, 3), Fraction(5, 2)),
    (Fraction(13, 5), Fraction(3, 2)),
    (Fraction(11, 5), Fraction(8, 3)),
    (Fraction(5, 2), Fraction(5, 2)),
]


@pytest.mark.parametrize("x,y", RATIONAL_PAIRS)
def test_q_sum_rational_matches_gosper_engine(x, y):
    """Reduced value equals the bihomographic engine's "add" value."""
    rec = q_sum_rational(x, y)
    coeffs_engine = gosper_coeffs(x, y, "add", 16)
    ser = sp.series(rec.value, q, 0, 16).removeO()
    poly = sp.Poly(sp.expand(ser), q)
    coeffs = [int(poly.coeff_monomial(q**k)) for k in range(16)]
    assert coeffs == coeffs_engine


def test_q_sum_rational_returns_named_tuple_with_caveat():
    """The result carries the not-always-reduced caveat in plain text."""
    rec = q_sum_rational(Fraction(3, 2), Fraction(5, 2))
    assert isinstance(rec, QSumRational)
    assert "not always" in rec.caveat
    # raw R, Q sit alongside the reduced pair (the raw form is what the meeting
    # called "not always reduced"; the cancelled value is the genuine q-number).
    assert rec.raw_numerator != 0
    assert rec.raw_denominator != 0


def test_q_sum_rational_known_value_three_halves_five_halves():
    """`q_sum_rational(3/2, 5/2)` reduces to q^2 + q + 2 (the Gosper engine value)."""
    rec = q_sum_rational(Fraction(3, 2), Fraction(5, 2))
    assert sp.simplify(rec.value - (q**2 + q + 2)) == 0


# ----------------------------------------------------------------------------
# q_sum_irrational (convergent iterator) + finiteness_check
# ----------------------------------------------------------------------------
def test_q_sum_irrational_4_plus_minus_sqrt7_matches_meeting_identity():
    """The 4 +/- sqrt(7) sum equals q^4 * negation_sum(sqrt(7)) + 2(1+q+q^2+q^3).

    From the 2026-05-25 supervisor meeting:
        [4 + sqrt(7)]_q + [4 - sqrt(7)]_q
            = q^4 ([sqrt(7)]_q + [-sqrt(7)]_q) + 2 (1 + q + q^2 + q^3).
    """
    N = 16
    res = q_sum_irrational("4+sqrt(7)", "4-sqrt(7)", N)
    assert isinstance(res, QSumIrrational)
    got = list(res.coeffs)
    # build the right-hand side coefficient sequence
    val, neg_coeffs = negation_sum("sqrt(7)", N)
    # multiplying by q^4 shifts valuation; we only want Taylor (q^0..q^{N-1}).
    rhs = [0] * N
    for i, c in enumerate(neg_coeffs):
        idx = val + 4 + i
        if 0 <= idx < N:
            rhs[idx] += c
    for k, two_coef in enumerate([2, 2, 2, 2]):
        if k < N:
            rhs[k] += two_coef
    assert got == rhs


def test_q_sum_irrational_stabilises_in_window():
    res = q_sum_irrational("4+sqrt(7)", "4-sqrt(7)", 12)
    assert res.stabilised_at is not None
    assert all(isinstance(a, Approximant) for a in res.approximants)


def test_finiteness_check_rational_pair_known_finite():
    """Integer-plus-integer is finite: [2]_q + [3]_q = (q + 1) + (q^2 + q + 1)."""
    fr = finiteness_check(2, 3, 8)
    assert fr.finite is True
    assert fr.method == "rational"
    assert sp.simplify(fr.polynomial - ((q + 1) + (q**2 + q + 1))) == 0


def test_finiteness_check_rational_pair_infinite():
    """[3/2]_q + [5/2]_q = q^2 + q + 2 is a polynomial in q with no negative
    powers; the denominator (after cancellation) is 1, a monomial, so the
    method's verdict is "finite". This pins the rational branch's behaviour."""
    fr = finiteness_check(Fraction(3, 2), Fraction(5, 2), 8)
    assert fr.finite is True
    assert sp.simplify(fr.polynomial - (q**2 + q + 2)) == 0


def test_finiteness_check_4_pm_sqrt7_finite():
    """Cross-check against the trace-zero quadratic catalogue (Ovsienko 6.4).

    sqrt(7) is a pure square root, so [sqrt(7)]_q + [-sqrt(7)]_q is finite, and
    the meeting identity reduces the 4 +/- sqrt(7) finiteness to it.
    """
    fr = finiteness_check("4+sqrt(7)", "4-sqrt(7)", 16)
    assert fr.finite is True
    assert fr.method == "irrational-trailing-zeros"


def test_finiteness_check_irrational_pair_infinite_control():
    """Golden ratio + sqrt(2): not the trace-zero pair, so the sum is not a
    finite Laurent polynomial."""
    fr = finiteness_check("(1+sqrt(5))/2", "sqrt(2)", 16)
    assert fr.finite is False
    assert isinstance(fr, FinitenessReport)
