import pytest
import sympy as sp

from qreals import q, q_rational, q_real_truncated


def _taylor(expr, N):
    truncated = sp.series(sp.expand(expr), q, 0, N).removeO()
    poly = sp.Poly(truncated, q)
    return [int(poly.coeff_monomial(q**k)) for k in range(N)]


def test_q_one_is_a_constant():
    assert q_real_truncated("1", 8) == [1, 0, 0, 0, 0, 0, 0, 0]


def test_integer_two_is_one_plus_q():
    assert q_real_truncated("2", 6) == [1, 1, 0, 0, 0, 0]


def test_integer_three():
    assert q_real_truncated("3", 6) == [1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("x", ["pi", "sqrt(2)", "(1+sqrt(5))/2", "E", "3/2", "5/7"])
def test_proposition_1_1_coefficient_stability(x):
    short = q_real_truncated(x, 40)
    long = q_real_truncated(x, 90)
    assert short == long[:40]


@pytest.mark.parametrize(
    "p,s", [(3, 2), (1, 2), (19, 7), (5, 7), (7, 5), (22, 7), (12, 5), (2, 1)]
)
def test_truncated_path_matches_exact_path_on_rationals(p, s):
    N = 30
    assert q_real_truncated(f"({p})/({s})", N) == _taylor(q_rational(p, s), N)


def test_golden_ratio_constant_term():
    assert q_real_truncated("(1+sqrt(5))/2", 5)[0] == 1
