"""Regression tests reproducing the worked examples of the two MGO papers.

Sources cited in the comments below:

  RAT  = S. Morier-Genoud and V. Ovsienko, "q-deformed rationals and
         q-continued fractions", Forum Math. Sigma 8 (2020), e13
         (arXiv:1812.00170). This is the paper named in the package docs;
         it carries the worked examples for q_rational.

  REAL = S. Morier-Genoud and V. Ovsienko, "On q-deformed real numbers"
         (arXiv:1908.04365), the companion that builds [x]_q for real x as a
         power series. It carries the worked Taylor series for the quadratic
         irrationals and the functional equations that q_real_truncated must
         satisfy. The functional-equation checks are independent of the
         continued-fraction construction the code runs, so they cross-check
         the series by a second route.

Every value below is transcribed from the cited paper, not produced by the
code; the test asserts the code reproduces it.
"""

import sympy as sp

from qreals import q, q_rational, q_real_truncated


def _rateq(got: sp.Expr, want_str: str) -> bool:
    """True when got equals the rational function want_str as elements of Q(q)."""
    return sp.simplify(sp.cancel(got - sp.sympify(want_str))) == 0


# === RAT: q_rational worked examples =====================================


def test_rat_intro_5_2_and_5_3():
    # RAT, Introduction (p. 2): the two examples that show the "quantized" 5
    # differs with the denominator.
    assert _rateq(q_rational(5, 2), "(1 + 2*q + q**2 + q**3)/(1 + q)")
    assert _rateq(q_rational(5, 3), "(1 + q + 2*q**2 + q**3)/(1 + q + q**2)")


def test_rat_example_1_2b_first_nontrivial():
    # RAT, Example 1.2(b): the first non-trivial q-rationals.
    assert _rateq(q_rational(5, 2), "(1 + 2*q + q**2 + q**3)/(1 + q)")
    assert _rateq(q_rational(5, 3), "(1 + q + 2*q**2 + q**3)/(1 + q + q**2)")
    assert _rateq(q_rational(7, 3), "(1 + 2*q + 2*q**2 + q**3 + q**4)/(1 + q + q**2)")
    assert _rateq(
        q_rational(7, 4), "(1 + q + 2*q**2 + 2*q**3 + q**4)/(1 + q + q**2 + q**3)"
    )
    assert _rateq(
        q_rational(7, 5), "(1 + q + 2*q**2 + 2*q**3 + q**4)/(1 + q + 2*q**2 + q**3)"
    )


def test_rat_example_1_2a_r_over_r_minus_one():
    # RAT, Example 1.2(a): [r/(r-1)]_q = [r]_q / [r-1]_q, the only family where
    # the q-rational is the quotient of the two q-integers.
    for r in (2, 3, 4, 5, 8, 12):
        num = sum(q**i for i in range(r))
        den = sum(q**i for i in range(r - 1))
        assert _rateq(q_rational(r, r - 1), str(num / den)), r


def test_rat_example_1_2c_denominator_two_family():
    # RAT, Example 1.2(c): the q-rationals with denominator [2]_q.
    # [(2m+1)/2]_q = (1 + 2q + ... + 2q^{m-1} + q^m + q^{m+1}) / (1 + q).
    for m in (1, 2, 3, 4, 5):
        num = 1 + sum(2 * q**k for k in range(1, m)) + q**m + q ** (m + 1)
        assert _rateq(q_rational(2 * m + 1, 2), str(num / (1 + q))), m


def test_rat_example_1_2d_denominator_three_families():
    # RAT, Example 1.2(d): the q-rationals with denominator [3]_q. The ramp of
    # coefficients 1,2,3,...,3 only appears for m >= 2.
    for m in (2, 3, 4, 5):
        base = 1 + 2 * q + sum(3 * q**k for k in range(2, m))
        num1 = base + 2 * q**m + q ** (m + 1) + q ** (m + 2)
        num2 = base + 2 * q**m + 2 * q ** (m + 1) + q ** (m + 2)
        assert _rateq(q_rational(3 * m + 1, 3), str(num1 / (1 + q + q**2))), m
        assert _rateq(q_rational(3 * m + 2, 3), str(num2 / (1 + q + q**2))), m


# === REAL: q_real_truncated worked Taylor series =========================
#
# Coefficient lists transcribed from REAL, Section 4 (eqns 13, 15 and the
# square-root displays on p. 11). Index k is the coefficient of q^k.

GOLDEN = [
    1,
    0,
    1,
    -1,
    2,
    -4,
    8,
    -17,
    37,
    -82,
    185,
    -423,
    978,
    -2283,
    5373,
    -12735,
    30372,
    -72832,
    175502,
    -424748,
    1032004,
]
SILVER = [
    1,
    1,
    0,
    0,
    1,
    0,
    -2,
    1,
    4,
    -5,
    -7,
    18,
    7,
    -55,
    18,
    146,
    -155,
    -322,
    692,
    476,
    -2446,
    307,
    7322,
    -6276,
    -18277,
    33061,
    33376,
    -129238,
    -10899,
]
SQRT2 = [
    1,
    0,
    0,
    1,
    0,
    -2,
    1,
    4,
    -5,
    -7,
    18,
    7,
    -55,
    18,
    146,
    -155,
    -322,
    692,
    476,
    -2446,
    307,
    7322,
    -6276,
    -18277,
    33061,
    33376,
]
SQRT3 = [
    1,
    0,
    1,
    0,
    -1,
    2,
    -2,
    -1,
    7,
    -12,
    7,
    18,
    -59,
    78,
    -1,
    -228,
    514,
    -469,
    -506,
    2591,
    -4338,
    1837,
    9405,
    -27430,
    33390,
    10329,
]
SQRT5 = [
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    -1,
    -1,
    -1,
    3,
    4,
    -1,
    -6,
    -11,
    2,
    25,
    22,
    -10,
    -70,
    -71,
    67,
    208,
    168,
    -222,
]
SQRT7 = [
    1,
    1,
    0,
    1,
    -1,
    2,
    -3,
    4,
    -6,
    8,
    -9,
    9,
    -5,
    -9,
    40,
    -101,
    215,
    -411,
    724,
    -1195,
    1845,
    -2623,
    3324,
    -3412,
    1696,
    4157,
]


def test_real_golden_ratio_series():
    # REAL, eqn (13) and the full series on p. 9; the alternating Generalized
    # Catalan numbers A004148 of REAL, Proposition 4.2.
    assert q_real_truncated("(1+sqrt(5))/2", len(GOLDEN)) == GOLDEN


def test_real_silver_ratio_series():
    # REAL, eqn (15) and the series on p. 10; the silver ratio 1 + sqrt(2).
    assert q_real_truncated("1+sqrt(2)", len(SILVER)) == SILVER


def test_real_square_root_series():
    # REAL, p. 11 displays for sqrt(2), sqrt(3), sqrt(5), sqrt(7).
    assert q_real_truncated("sqrt(2)", len(SQRT2)) == SQRT2
    assert q_real_truncated("sqrt(3)", len(SQRT3)) == SQRT3
    assert q_real_truncated("sqrt(5)", len(SQRT5)) == SQRT5
    assert q_real_truncated("sqrt(7)", len(SQRT7)) == SQRT7


def test_real_sqrt2_is_silver_shifted_by_one_power():
    # REAL p. 11 remark: the coefficients of [sqrt(2)]_q are those of the
    # silver ratio with the power of q shifted by one, because
    # sqrt(2) = (1 + sqrt(2)) - 1. The series kernel drops the constant term.
    silver = q_real_truncated("1+sqrt(2)", 26)
    sqrt2 = q_real_truncated("sqrt(2)", 25)
    assert sqrt2 == silver[1:]


# === REAL: functional equations (algorithm-independent cross-check) =======
#
# REAL, Propositions 4.2, 4.4 and 4.5 give a quadratic A*[x]^2 + B*[x] + C = 0
# that [x]_q satisfies. We plug the computed series in and confirm the
# residual vanishes to high order. Squaring an N-term series is exact to
# order N - 1, so with N = 26 a check through q^18 has margin to spare.


def _poly(coeffs):
    return sum(c * q**k for k, c in enumerate(coeffs))


def _residual_vanishes(coeffs, A, B, C, order):
    series = _poly(coeffs)
    residual = sp.expand(A * series**2 + B * series + C)
    tail = sp.series(residual, q, 0, order).removeO()
    return sp.expand(tail) == 0


def test_real_functional_equations():
    N, order = 26, 18
    g = q_real_truncated("(1+sqrt(5))/2", N)
    si = q_real_truncated("1+sqrt(2)", N)
    r2 = q_real_truncated("sqrt(2)", N)
    r3 = q_real_truncated("sqrt(3)", N)
    r5 = q_real_truncated("sqrt(5)", N)
    r7 = q_real_truncated("sqrt(7)", N)
    # REAL eqn (14): q [phi]^2 - (q^2 + q - 1)[phi] - 1 = 0.
    assert _residual_vanishes(g, q, -(q**2 + q - 1), -1, order)
    # REAL eqn (16): q [delta]^2 - (q^3 + 2q - 1)[delta] - 1 = 0.
    assert _residual_vanishes(si, q, -(q**3 + 2 * q - 1), -1, order)
    # REAL eqn (17): q^2 [.]^2 - (q^3 - 1)[.] - (q^2 + 1) = 0.
    assert _residual_vanishes(r2, q**2, -(q**3 - 1), -(q**2 + 1), order)
    # REAL eqn (18): q^2 [.]^2 - (q^3 + q^2 - q - 1)[.] - (q^2 + q + 1) = 0.
    assert _residual_vanishes(r3, q**2, -(q**3 + q**2 - q - 1), -(q**2 + q + 1), order)
    # REAL eqn (19): q^3 [.]^2 - (q^5 + q^3 - q^2 - 1)[.] - (q^4+q^3+q^2+q+1) = 0.
    assert _residual_vanishes(
        r5, q**3, -(q**5 + q**3 - q**2 - 1), -(q**4 + q**3 + q**2 + q + 1), order
    )
    # REAL eqn (20): q^3 [.]^2 - (q^5 + q^4 - q - 1)[.] - (q^4+2q^3+q^2+2q+1) = 0.
    assert _residual_vanishes(
        r7, q**3, -(q**5 + q**4 - q - 1), -(q**4 + 2 * q**3 + q**2 + 2 * q + 1), order
    )
