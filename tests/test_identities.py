"""Structural identities that hold for every rational, checked exactly.

Sources (each opened and read for this suite):

* [MGO20] Morier-Genoud, Ovsienko, arXiv:1812.00170v3.
* [MGO22] Morier-Genoud, Ovsienko, arXiv:1908.04365v3.
* [J25] P. Jouteur, "Symmetries of the q-deformed real projective line",
  arXiv:2503.02122v1.

Each identity is checked on a fixed grid of rationals here and on random
rationals in test_properties.py. All comparisons are exact: sp.cancel of a
difference of rational functions, or equality of integer lists.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
import sympy as sp

from conftest import is_zero, series_expr, taylor
from qreals import negate, q, q_int, q_neg, q_rational, q_real_truncated
from qreals.continuant import continuant_matrix, continuant_pair
from qreals.continued_fraction import cf_partials, make_even_length
from qreals.rational import q_rational_pair


def grid(max_num: int = 12, max_den: int = 7, positive: bool = False):
    out = []
    lo = 1 if positive else -max_num
    for p in range(lo, max_num + 1):
        for s in range(1, max_den + 1):
            if p != 0 and math.gcd(p, s) == 1:
                out.append(Fraction(p, s))
    return out


GRID = grid()
POS_GRID = grid(positive=True)
GT1_GRID = [x for x in POS_GRID if x > 1]


def qr(x: Fraction) -> sp.Expr:
    return q_rational(x.numerator, x.denominator)


def inv_q(expr: sp.Expr) -> sp.Expr:
    return expr.subs(q, 1 / q)


# --------------------------------------------------------------------------
# the modular group action
# --------------------------------------------------------------------------


@pytest.mark.parametrize("x", GRID, ids=str)
def test_translation(x: Fraction) -> None:
    """[x + 1]_q = q [x]_q + 1 for every rational x, negatives included.

    Source: [MGO22] equation (3); [J25] equation (1).
    """
    assert is_zero(qr(x + 1) - (q * qr(x) + 1))


@pytest.mark.parametrize("x", GRID, ids=str)
def test_s_action(x: Fraction) -> None:
    """[-1/x]_q = -1/(q [x]_q), the q-deformed S : x -> -1/x.

    This is the convention the code follows: S_q = [[0, -q^-1], [1, 0]]
    acting by fractional-linear maps sends [x]_q to -q^-1/[x]_q.

    Source: [J25] equation (1) and (6); [MGO20] Definition 4.4 for S_q.
    """
    assert is_zero(qr(-1 / x) - (-1 / (q * qr(x))))


@pytest.mark.parametrize("x", GRID, ids=str)
def test_negation_by_inversion(x: Fraction) -> None:
    """[-x]_q = -q^-1 [x]_(q^-1) for rational x.

    Source: [J25] equation (4).
    """
    assert is_zero(qr(-x) - (-inv_q(qr(x)) / q))


@pytest.mark.parametrize("x", GRID, ids=str)
def test_reciprocal_by_inversion(x: Fraction) -> None:
    """[1/x]_q = 1/[x]_(q^-1) for rational x.

    Source: [J25] equation (4).
    """
    assert is_zero(qr(1 / x) - 1 / inv_q(qr(x)))


def test_zero_and_one() -> None:
    """[0]_q = 0 and [1]_q = 1, the base of the translation orbit.

    Source: [MGO20], Introduction ([a]_q for a = 0, 1).
    """
    assert q_rational(0, 1) == 0
    assert q_rational(1, 1) == 1
    assert q_rational(3, 3) == 1


# --------------------------------------------------------------------------
# specialisation and positivity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("x", GRID, ids=str)
def test_q_equals_one_gives_classical_value(x: Fraction) -> None:
    """[r/s]_q at q = 1 is r/s.

    Source: [MGO20], Section 1.1, "R(1) = r, S(1) = s".
    """
    value = sp.cancel(qr(x)).subs(q, 1)
    assert value == sp.Rational(x.numerator, x.denominator)


@pytest.mark.parametrize("x", POS_GRID, ids=str)
def test_numerator_denominator_at_one(x: Fraction) -> None:
    """The reduced pair satisfies R(1) = r and S(1) = s exactly.

    Source: [MGO20], Section 1.1 and Corollary 1.7.
    """
    R, S = q_rational_pair(x.numerator, x.denominator)
    assert R.eval(1) == x.numerator
    assert S.eval(1) == x.denominator


@pytest.mark.parametrize("x", GT1_GRID, ids=str)
def test_total_positivity_and_unit_ends(x: Fraction) -> None:
    """For r/s > 1, R and S have positive coefficients, and their leading
    and constant coefficients equal 1.

    The paper works only with r/s > 1 ("Throughout the paper, we work only
    with the case r/s > 1"); for r/s < 1 the constant term of R is 0.

    Source: [MGO20], Proposition 1.3 and the statement before Example 1.2.
    """
    R, S = q_rational_pair(x.numerator, x.denominator)
    for P in (R, S):
        coeffs = P.all_coeffs()
        assert all(c > 0 for c in coeffs)
        assert coeffs[0] == 1 and coeffs[-1] == 1


def test_theorem_2_cross_difference_is_positive() -> None:
    """For r/s > r'/s' > 1, R S' - S R' has positive integer coefficients.

    Source: [MGO20], Theorem 2, formula (1.4).
    """
    pts = sorted(set(GT1_GRID))
    for i, small in enumerate(pts):
        for big in pts[i + 1 : i + 8]:
            R, S = q_rational_pair(big.numerator, big.denominator)
            R2, S2 = q_rational_pair(small.numerator, small.denominator)
            X = R * S2 - S * R2
            coeffs = [c for c in X.all_coeffs()]
            assert X.eval(1) == big.numerator * small.denominator - big.denominator * small.numerator
            assert all(c >= 0 for c in coeffs) and coeffs[0] > 0, (big, small)


# --------------------------------------------------------------------------
# an independent reference: the negative continued fraction, [MGO20] (1.2)
# --------------------------------------------------------------------------


def hj_digits(x: Fraction) -> list[int]:
    """Negative (Hirzebruch-Jung) continued fraction digits of x > 1.

    x = c1 - 1/(c2 - 1/(...)), with c_i = ceil of the running remainder;
    every digit after the first is >= 2.
    """
    out = []
    while True:
        c = math.ceil(x)
        out.append(c)
        if c == x:
            return out
        x = 1 / (c - x)


def negative_cf_q(digits: list[int]) -> sp.Expr:
    """[MGO20] formula (1.2): [c1]_q - q^(c1-1) / ([c2]_q - q^(c2-1)/(...))."""

    def qn(n: int) -> sp.Expr:
        return sum((q**i for i in range(n)), sp.Integer(0))

    value = qn(digits[-1])
    for c in reversed(digits[:-1]):
        value = qn(c) - q ** (c - 1) / value
    return value


@pytest.mark.parametrize("x", GT1_GRID, ids=str)
def test_regular_equals_negative_continued_fraction(x: Fraction) -> None:
    """The regular-CF engine agrees with the negative CF formula (1.2).

    This reference shares no code with the package: the digits and the fold
    are both written in this test from the paper's formula (1.2). Theorem 1
    states that the two q-deformations coincide.

    Source: [MGO20], Definition 1.1 (b) and Theorem 1.
    """
    assert is_zero(qr(x) - negative_cf_q(hj_digits(x)))


def test_negative_cf_digits_of_paper_examples() -> None:
    """5/2 = [[3, 2]], 5/3 = [[2, 3]], 7/5 = [[2, 2, 3]].

    Source: [MGO20], Example 2.2.
    """
    assert hj_digits(Fraction(5, 2)) == [3, 2]
    assert hj_digits(Fraction(5, 3)) == [2, 3]
    assert hj_digits(Fraction(7, 5)) == [2, 2, 3]


# --------------------------------------------------------------------------
# matrices
# --------------------------------------------------------------------------


def even_cf(x: Fraction) -> list[int]:
    return make_even_length(
        [int(t) for t in sp.continued_fraction(sp.Rational(x.numerator, x.denominator))]
    )


@pytest.mark.parametrize("x", GRID, ids=str)
def test_block_product_determinant(x: Fraction) -> None:
    """det M(a) = q^(a1 - a2 + a3 - ... - a2m) for the unscaled block product.

    Each odd block [[ [a]_q, q^a ], [1, 0]] has determinant -q^a and each
    even block [[ [a]_(1/q), q^-a ], [1, 0]] has determinant -q^-a, so an
    even-length product has determinant q^(alternating sum): unimodular up
    to a power of q.

    Source: [MGO20] formula (1.1) written as a block product; the block form
    is the one in the qreals.continuant module docstring.
    """
    word = even_cf(x)
    M = continuant_matrix(word)
    alt = sum(a if i % 2 == 0 else -a for i, a in enumerate(word))
    assert is_zero(M.det() - q**alt)


@pytest.mark.parametrize("x", GT1_GRID, ids=str)
def test_scaled_matrix_first_column_and_previous_convergent(x: Fraction) -> None:
    """The scaled matrix M~+ of [MGO22] (9) has first column (qR, qS) up to a
    common factor, and its second column is the previous convergent
    [a1, ..., a_(2m-1)]_q.

    Source: [MGO22], equations (9) and (10).
    """
    word = even_cf(x)
    R, S = continuant_pair(word)
    assert is_zero(R.as_expr() / S.as_expr() - qr(x))
    # the full scaled product, built here from (9) directly
    M = sp.eye(2)
    for i, a in enumerate(word):
        if i % 2 == 0:
            M = M * sp.Matrix([[q_int(a), q**a], [1, 0]])
        else:
            M = M * sp.Matrix([[q * q_int(a), 1], [q**a, 0]])
    assert sp.expand(M[0, 0] - R.as_expr()) == 0
    assert sp.expand(M[1, 0] - S.as_expr()) == 0
    prev = word[:-1]
    # [a1, ..., a_(2m-1)] is an odd-length word; its value as a rational
    val = Fraction(prev[-1])
    for a in reversed(prev[:-1]):
        val = a + 1 / val
    assert is_zero(M[0, 1] / M[1, 1] - qr(val))
    # det M~+ = q^(a1 + ... + a2m)
    assert is_zero(M.det() - q ** sum(word))


# --------------------------------------------------------------------------
# q-reals: stabilisation and the gap theorem
# --------------------------------------------------------------------------


def convergent(a: list[int], k: int) -> Fraction:
    """The k-th convergent [a1, ..., ak] of a regular continued fraction."""
    val = Fraction(a[k - 1])
    for t in reversed(a[: k - 1]):
        val = t + 1 / val
    return val


CONVERGENT_CASES = ["sqrt(2)", "sqrt(3)", "(1+sqrt(5))/2", "E", "pi", "sqrt(7)", "sqrt(13)"]


@pytest.mark.parametrize("x", CONVERGENT_CASES)
def test_convergent_cross_determinant(x: str) -> None:
    """R_n S_(n-1) - S_n R_(n-1) = (-1)^n q^(a1 + ... + a_(2 floor(n/2)) - 1).

    For even n this is [MGO22] equation (11), q^(a1 + ... + an - 1), whose
    proof goes through det M~+ of an even-length word. For odd n the
    printed (11) does not hold: the exponent stops at a_(n-1) and the sign
    is negative. test_eq11_odd_case_from_printed_values shows this from the
    paper's own printed q-rationals, with no engine code involved.

    Source: [MGO22], Proposition 1.1 and equation (11), Section 3.2.
    """
    a = cf_partials(x, 60)[:7]
    for n in range(2, len(a) + 1):
        c0, c1 = convergent(a, n - 1), convergent(a, n)
        R0, S0 = q_rational_pair(c0.numerator, c0.denominator)
        R1, S1 = q_rational_pair(c1.numerator, c1.denominator)
        D = (R1 * S0 - S1 * R0).as_expr()
        exponent = sum(a[: 2 * (n // 2)]) - 1
        assert sp.expand(D - (-1) ** n * q**exponent) == 0, (n, D)


def test_eq11_odd_case_from_printed_values() -> None:
    """[MGO22] (11) at n = 3 for sqrt(2) = [1, 2, 2, ...], from printed data.

    x_2 = 3/2 and x_3 = 7/5. Using [3/2]_q = (1 + q + q^2)/(1 + q) from
    [MGO20] Example 1.2 (c) and [7/5]_q = (1 + q + 2q^2 + 2q^3 + q^4) /
    (1 + q + 2q^2 + q^3) from [MGO20] Example 1.2 (b), the cross
    determinant is -q^2, not the q^(1 + 2 + 2 - 1) = q^4 of (11). So the
    literal statement of Proposition 1.1 is too strong for odd n; the
    engine agrees with the printed q-rationals.
    """
    R2, S2 = 1 + q + q**2, 1 + q
    R3, S3 = 1 + q + 2 * q**2 + 2 * q**3 + q**4, 1 + q + 2 * q**2 + q**3
    assert sp.expand(R3 * S2 - S3 * R2) == -(q**2)


@pytest.mark.parametrize("x", CONVERGENT_CASES)
def test_consecutive_convergents_stabilize(x: str) -> None:
    """Consecutive convergents agree on the first a1 + ... + a_(2 floor(n/2))
    - 1 Taylor coefficients and the next one differs by exactly 1.

    This is the series form of test_convergent_cross_determinant: both
    denominators start 1 + O(q), so the difference of the two series is
    +/- q^e + O(q^(e+1)). For even n it is [MGO22] Proposition 1.1.
    """
    a = cf_partials(x, 40)
    for n in range(2, min(len(a), 7) + 1):
        c0, c1 = convergent(a, n - 1), convergent(a, n)
        stable = sum(a[: 2 * (n // 2)]) - 1
        if stable > 60:  # pi's 292 would ask sympy for 318 series terms
            break
        t0 = taylor(qr(c0), stable + 1)
        t1 = taylor(qr(c1), stable + 1)
        assert t0[:stable] == t1[:stable]
        assert abs(t0[stable] - t1[stable]) == 1


@pytest.mark.parametrize("x", ["sqrt(2)", "E", "pi", "(1+sqrt(5))/2", "sqrt(7)", "3/7", "22/7"])
def test_truncation_is_stable(x: str) -> None:
    """The first N coefficients do not change when more are requested.

    This is the guarantee q_real_truncated relies on (the cf_partials
    stopping rule, at least S_n - 1 >= N stable terms).
    """
    long = q_real_truncated(x, 60)
    for N in (1, 2, 5, 13, 30):
        assert q_real_truncated(x, N) == long[:N]


@pytest.mark.parametrize(
    "x", ["1", "7/5", "sqrt(2)", "2", "5/2", "E", "3", "pi", "29/10", "7/2", "4", "9/2"]
)
def test_gap_theorem_half_open(x: str) -> None:
    """For k <= x < k + 1 (k >= 1): c_0 = ... = c_(k-1) = 1 and c_k = 0.

    [MGO22] Theorem 2 states it for k <= x <= k + 1. At the right endpoint
    x = k + 1 the q-integer [k+1]_q = 1 + ... + q^k has c_k = 1, so the
    closed statement fails there; this test uses the half-open interval,
    and test_gap_theorem_right_endpoint pins the endpoint behaviour.

    Source: [MGO22], Theorem 2 and Proposition 6.2.
    """
    val = sp.sympify(x)
    k = int(sp.floor(val))
    c = q_real_truncated(x, k + 4)
    assert c[:k] == [1] * k
    assert c[k] == 0


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_gap_theorem_right_endpoint(k: int) -> None:
    """At x = k + 1 the q^k coefficient is 1, not 0.

    This documents the endpoint case of [MGO22] Theorem 2 / Proposition 6.2
    ("iff 1 <= x <= 2"): [2]_q = 1 + q has a nonzero q^1 coefficient.
    """
    c = q_real_truncated(str(k + 1), k + 3)
    assert c[k] == 1
    assert c[: k + 1] == [1] * (k + 1)


def test_negative_rational_laurent_shape() -> None:
    """For -k <= x < 1 - k, [x]_q = -q^(-k) + (higher powers).

    Source: [MGO22], equation (4).
    """
    for x in [Fraction(-1), Fraction(-1, 2), Fraction(-3, 2), Fraction(-7, 3), Fraction(-5)]:
        k = -math.floor(x)
        expr = sp.cancel(qr(x) * q**k)
        assert expr.subs(q, 0) == -1


# --------------------------------------------------------------------------
# the Jouteur negation N_q
# --------------------------------------------------------------------------


def jouteur(A: sp.Expr) -> sp.Expr:
    """[J25] equation (2): (-A + 1 - q^-1) / ((q - 1) A + 1)."""
    return (-A + 1 - 1 / q) / ((q - 1) * A + 1)


def left_version(x: Fraction) -> sp.Expr:
    """[x]^flat_q = M_q . 1/(1 - q), the left q-rational of [J25] Def. 1.2.

    M_q is the MGO block product for x, which sends infinity = 1/0 to the
    right version [x]_q (first-column ratio); applying it to 1/(1 - q)
    instead gives the left version.
    """
    M = continuant_matrix(even_cf(x))
    z = 1 / (1 - q)
    return (M[0, 0] * z + M[0, 1]) / (M[1, 0] * z + M[1, 1])


@pytest.mark.parametrize("x", POS_GRID, ids=str)
def test_jouteur_negation_of_rational_is_left_version(x: Fraction) -> None:
    """N_q . [x]_q = [-x]^flat_q for rational x (det N = -1 swaps sides).

    So the Jouteur formula is NOT the (right, MGO) q-rational [-x]_q on
    rationals; the two differ already at x = 1 (-q^-2 versus -q^-1).

    Source: [J25], Theorem 1.5 (first part), Definition 1.2.
    """
    assert is_zero(jouteur(qr(x)) - left_version(-x))
    assert not is_zero(jouteur(qr(x)) - qr(-x))


def test_left_integers() -> None:
    """[n]^flat_q = 1 + q + ... + q^(n-2) + q^n.

    Source: [J25], Section 1.1, "The left q-integers are".
    """
    for n in range(1, 8):
        expected = sum((q**i for i in range(n - 1)), sp.Integer(0)) + q**n
        assert is_zero(left_version(Fraction(n)) - expected)


@pytest.mark.parametrize("x,shift", [("sqrt(2)", 2), ("sqrt(3)", 2), ("(1+sqrt(5))/2", 2), ("pi", 4), ("E", 3)])
def test_jouteur_negation_of_irrational_is_mgo(x: str, shift: int) -> None:
    """For irrational x, N_q . [x]_q = [-x]_q, the MGO series of -x.

    [-x]_q is built independently from the translation rule: with
    y = shift - x in (0, 1), [-x]_q = [y - shift]_q and
    [y - 1]_q = ([y]_q - 1)/q applied shift times.

    Source: [J25], Theorem 1.5 (second part); [MGO22] equation (3).
    """
    N = 20
    y = f"{shift}-({x})"
    Y = series_expr(q_real_truncated(y, N + shift + 2))
    for _ in range(shift):
        Y = sp.expand((Y - 1) / q)
    v, c = q_neg(x, N)
    got = series_expr(c, v)
    # compare coefficients from q^v through q^(v + N - 1)
    diff = sp.expand((got - Y) * q ** (-v))
    low = [diff.coeff(q, k) for k in range(N)]
    assert all(t == 0 for t in low), low


@pytest.mark.parametrize("x", ["3/2", "7/5", "sqrt(2)", "pi", "5"])
def test_negate_is_an_involution(x: str) -> None:
    """negate(-x) recovers [x]_q: N_q is an involution in PGL_2.

    Source: [J25], Section 1.2, relation N^2 = 1 preserved by N_q.
    """
    N = 16
    v, c = negate(f"-({x})", N)
    assert v == 0
    assert c == q_real_truncated(x, N)
    assert negate(x, N) == q_neg(x, N)
