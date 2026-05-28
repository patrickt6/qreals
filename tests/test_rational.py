import pytest
import sympy as sp

from qreals import q, q_rational


def test_q_rational_3_2_matches_paper():
    assert sp.simplify(q_rational(3, 2) - (1 + q + q**2) / (1 + q)) == 0


def test_q_rational_1_2_matches_paper():
    assert sp.simplify(q_rational(1, 2) - q / (1 + q)) == 0


def test_equal_numerator_denominator_is_one():
    assert q_rational(5, 5) == 1


@pytest.mark.parametrize(
    "p,s",
    [
        (3, 2),
        (1, 2),
        (19, 7),
        (5, 7),
        (22, 7),
        (7, 5),
        (12, 5),
        (100, 3),
        (2, 9),
        (2, 1),
        (3, 1),
    ],
)
def test_specialises_to_ordinary_rational_at_q_equals_one(p, s):
    assert sp.nsimplify(q_rational(p, s).subs(q, 1)) == sp.Rational(p, s)


def test_zero_denominator_raises():
    with pytest.raises(ZeroDivisionError):
        q_rational(1, 0)
