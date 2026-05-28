"""Tests for arithmetic between q-reals.

Every operation is checked against an independent path: q_add and q_mul against
the bihomographic `gosper` engine (a different algorithm) and against the
coefficient combination of two `q_real_truncated` calls; q_neg against its
own algebraic involution back to `q_real_truncated`; radius against the exact
nearest-pole modulus of a rational [x]_q. See docs/CORRECTNESS.md.
"""

from fractions import Fraction

import pytest
import sympy as sp

from qreals import (
    finite_xnegx,
    gosper_coeffs,
    negation_sum,
    q_add,
    q_mul,
    q_neg,
    q_real_truncated,
    radius,
)
from qreals.arithmetic import _jouteur_neg, _qreal_series
from qreals.rational import q, q_rational

# Rational test family (x, y), all > 0, drawn from the q-Gosper battery.
RATIONAL_PAIRS = [
    (Fraction(3, 2), Fraction(5, 2)),
    (Fraction(3, 2), Fraction(13, 5)),
    (Fraction(7, 3), Fraction(5, 2)),
    (Fraction(27, 19), Fraction(7, 3)),
    (Fraction(4, 3), Fraction(5, 3)),
    (Fraction(22, 7), Fraction(11, 5)),
    (Fraction(8, 3), Fraction(9, 4)),
    (Fraction(41, 29), Fraction(13, 7)),
    (Fraction(5, 2), Fraction(5, 2)),
    (Fraction(11, 3), Fraction(16, 7)),
]

N = 20


def _frac(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


# --------------------------------------------------------------------------
# q_add: series sum [x]_q + [y]_q
# --------------------------------------------------------------------------
@pytest.mark.parametrize("x,y", RATIONAL_PAIRS)
def test_q_add_matches_engine_and_brute_force(x, y):
    lib = q_add(_frac(x), _frac(y), N)
    engine = gosper_coeffs(x, y, "add", N)
    brute = [
        a + b
        for a, b in zip(q_real_truncated(_frac(x), N), q_real_truncated(_frac(y), N))
    ]
    assert lib == engine == brute


@pytest.mark.parametrize(
    "x,y", [("pi", "sqrt(2)"), ("E", "(1+sqrt(5))/2"), ("sqrt(3)", "22/7")]
)
def test_q_add_on_irrationals_is_the_series_sum(x, y):
    lib = q_add(x, y, N)
    brute = [a + b for a, b in zip(q_real_truncated(x, N), q_real_truncated(y, N))]
    assert lib == brute


def test_q_add_is_not_q_of_the_real_sum():
    # The headline caveat: [x]_q + [y]_q is not [x+y]_q. With x+y = 4 the real
    # sum's q-real has constant term 1, the series sum has constant term 2.
    assert q_add("3/2", "5/2", 6)[0] == 2
    assert q_real_truncated("4", 6)[0] == 1


# --------------------------------------------------------------------------
# q_mul: series product [x]_q * [y]_q
# --------------------------------------------------------------------------
@pytest.mark.parametrize("x,y", RATIONAL_PAIRS)
def test_q_mul_matches_engine_and_convolution(x, y):
    lib = q_mul(_frac(x), _frac(y), N)
    engine = gosper_coeffs(x, y, "mul", N)
    assert lib == engine


def test_q_mul_convolution_matches_q_real_truncated():
    cx, cy = q_real_truncated("3/2", N), q_real_truncated("13/5", N)
    expect = [0] * N
    for i, a in enumerate(cx):
        for j, b in enumerate(cy):
            if i + j < N:
                expect[i + j] += a * b
    assert q_mul("3/2", "13/5", N) == expect


# --------------------------------------------------------------------------
# q_neg: the Jouteur negation (Ovsienko Example 6.4)
# --------------------------------------------------------------------------
def test_q_neg_jouteur_is_an_involution_back_to_q_real_truncated():
    # Applying the Jouteur negation twice must return [x]_q exactly.
    for x in ["3/2", "13/5", "22/7", "8/3"]:
        prec = N + 8
        twice = _jouteur_neg(_jouteur_neg(_qreal_series(x, prec), prec), prec)
        from qreals import series as ser

        v, c = ser.normalise(twice)
        assert v == 0
        assert c[:N] == q_real_truncated(x, N)


def test_q_neg_of_integer_two():
    # [-2]_q = (-(1+q) + 1 - 1/q)/q^2 = -q^{-1} - q^{-3}.
    assert q_neg("2", 6) == (-3, [-1, 0, -1, 0, 0, 0])


def test_q_neg_symbolic_identity_and_involution():
    # The closed identity (*) and the involution, proved over a free symbol A.
    A = sp.Symbol("A")
    jouteur = sp.cancel((-A + 1 - 1 / q) / ((q - 1) * A + 1))
    rhs = sp.cancel(((q - 1) * A**2 + (1 - 1 / q)) / ((q - 1) * A + 1))
    assert sp.simplify(sp.cancel(A + jouteur) - rhs) == 0
    twice = sp.cancel(jouteur.subs(A, jouteur))
    assert sp.simplify(twice - A) == 0


# --------------------------------------------------------------------------
# finite_xnegx: the x -> -x finiteness criterion
# --------------------------------------------------------------------------
@pytest.mark.parametrize("x", ["sqrt(2)", "sqrt(3)", "sqrt(5)", "sqrt(6)", "sqrt(7)"])
def test_negation_sum_finite_for_pure_square_roots(x):
    assert finite_xnegx(x) is True


@pytest.mark.parametrize(
    "x", ["(1+sqrt(5))/2", "1+sqrt(2)", "(3+sqrt(13))/2", "5/7", "pi"]
)
def test_negation_sum_infinite_otherwise(x):
    assert finite_xnegx(x) is False


def test_negation_sum_sqrt2_closed_form():
    # [sqrt2]_q + [-sqrt2]_q = -q^{-2} + q^{1}, a short Laurent polynomial.
    v, c = negation_sum("sqrt(2)", 16)
    assert v == -2
    assert c[0] == -1 and c[3] == 1
    assert all(c[i] == 0 for i in range(len(c)) if i not in (0, 3))


# --------------------------------------------------------------------------
# radius: running-max slope estimate
# --------------------------------------------------------------------------
def _nearest_pole(p: int, s: int) -> float:
    den = sp.fraction(sp.cancel(q_rational(p, s)))[1]
    return min(abs(complex(r)) for r in sp.Poly(den, q).all_roots())


@pytest.mark.parametrize("p,s", [(7, 5), (8, 5), (11, 9)])
def test_radius_converges_to_nearest_pole_from_above(p, s):
    pole = _nearest_pole(p, s)
    estimates = [radius(f"{p}/{s}", n) for n in (20, 40, 80, 160)]
    # monotone decreasing and staying at or above the true radius
    for a, b in zip(estimates, estimates[1:]):
        assert a >= b - 1e-12
    assert estimates[-1] >= pole - 1e-6
    assert estimates[-1] == pytest.approx(pole, abs=2e-2)


def test_radius_of_constant_is_infinite():
    import math

    assert radius("1", 5) == math.inf


def test_radius_irrational_is_between_zero_and_one():
    r = radius("pi", 80)
    assert 0.0 < r <= 1.0


def test_arithmetic_rejects_negative_x():
    with pytest.raises(ValueError):
        q_add("-1", "2", 5)
    with pytest.raises(ValueError):
        q_neg("-1", 5)
