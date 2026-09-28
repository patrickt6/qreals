"""Known values from the literature, each checked against the engine.

Sources (each opened and read for this suite):

* [MGO20] S. Morier-Genoud, V. Ovsienko, "q-deformed rationals and
  q-continued fractions", Forum Math. Sigma 8 (2020), arXiv:1812.00170v3.
* [MGO22] S. Morier-Genoud, V. Ovsienko, "On q-deformed real numbers",
  Experimental Mathematics 31 (2022), arXiv:1908.04365v3.
* [A004148] OEIS A004148, Generalized Catalan numbers; the b-file excerpt
  used here is stored in tests/data/A004148.txt with its URL and fetch date.

Every expected value below is copied from the source, not from the engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sympy as sp

from conftest import is_zero, poly_coeffs, series_expr
from qreals import q, q_int, q_int_qinv, q_rational, q_real_truncated
from qreals.rational import q_rational_pair

# --------------------------------------------------------------------------
# q-integers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, 13))
def test_q_integer_is_geometric_sum(n: int) -> None:
    """[n]_q = (1 - q^n)/(1 - q) = 1 + q + ... + q^(n-1).

    Source: [MGO20], Introduction, first display.
    """
    assert sp.expand(q_int(n)) == sum(q**i for i in range(n))
    assert is_zero(q_int(n) - (1 - q**n) / (1 - q))


@pytest.mark.parametrize("n", range(1, 10))
def test_q_integer_inverse_substitution(n: int) -> None:
    """[n]_(q^-1) is [n]_q with q replaced by 1/q.

    Source: [MGO20], Definition 1.1, where [a]_(q^-1) enters formula (1.1).
    """
    assert is_zero(q_int_qinv(n) - q_int(n).subs(q, 1 / q))


def test_q_integer_zero_and_negative() -> None:
    """[0]_q = 0 and [-n]_q = -q^(-n) [n]_q, the value the translation rule
    [x - 1]_q = ([x]_q - 1)/q forces on negative integers.

    Source: [MGO22], equation (3), used as the definition for x < 1.
    """
    assert q_int(0) == 0
    for n in range(1, 8):
        assert is_zero(q_int(-n) - (q_int(-n + 1) - 1) / q)


# --------------------------------------------------------------------------
# q-rationals: [MGO20] Example 1.2
# --------------------------------------------------------------------------

# (r, s, numerator R, denominator S), coefficients lowest degree first.
MGO20_EXAMPLE_1_2B = [
    (5, 2, [1, 2, 1, 1], [1, 1]),
    (5, 3, [1, 1, 2, 1], [1, 1, 1]),
    (7, 3, [1, 2, 2, 1, 1], [1, 1, 1]),
    (7, 4, [1, 1, 2, 2, 1], [1, 1, 1, 1]),
    (7, 5, [1, 1, 2, 2, 1], [1, 1, 2, 1]),
]


@pytest.mark.parametrize("r,s,num,den", MGO20_EXAMPLE_1_2B)
def test_mgo20_example_1_2b(r: int, s: int, num: list[int], den: list[int]) -> None:
    """The first non-trivial q-rationals 5/2, 5/3, 7/3, 7/4, 7/5.

    Source: [MGO20], Example 1.2 (b), p. 5; 5/2 and 5/3 also in the
    Introduction, p. 2.
    """
    R, S = q_rational_pair(r, s)
    assert [int(c) for c in reversed(R.all_coeffs())] == num
    assert [int(c) for c in reversed(S.all_coeffs())] == den
    expected = series_expr(num) / series_expr(den)
    assert is_zero(q_rational(r, s) - expected)


@pytest.mark.parametrize("r", range(2, 14))
def test_mgo20_example_1_2a(r: int) -> None:
    """[r/(r-1)]_q = [r]_q / [r-1]_q (and s = 1 gives [r]_q itself).

    Source: [MGO20], Example 1.2 (a), p. 5.
    """
    assert is_zero(q_rational(r, r - 1) - q_int(r) / q_int(r - 1))
    assert is_zero(q_rational(r, 1) - q_int(r))


@pytest.mark.parametrize("m", range(1, 10))
def test_mgo20_example_1_2c_denominator_two(m: int) -> None:
    """[(2m+1)/2]_q = (1 + 2q + ... + 2q^(m-1) + q^m + q^(m+1)) / (1 + q).

    Source: [MGO20], Example 1.2 (c), p. 5.
    """
    num = [1] + [2] * (m - 1) + [1, 1]
    assert is_zero(q_rational(2 * m + 1, 2) - series_expr(num) / (1 + q))


@pytest.mark.parametrize("m", range(2, 9))
def test_mgo20_example_1_2d_denominator_three(m: int) -> None:
    """[(3m+1)/3]_q and [(3m+2)/3]_q over the denominator 1 + q + q^2.

    Numerators 1 + 2q + 3q^2 + ... + 3q^(m-1) + 2q^m + q^(m+1) + q^(m+2) and
    1 + 2q + 3q^2 + ... + 3q^(m-1) + 2q^m + 2q^(m+1) + q^(m+2), where the run
    of 3s starts at q^2. The printed pattern opens with 1 + 2q, so it
    describes m >= 2; at m = 1 the values are 4/3 and 5/3, given by
    Example 1.2 (a) and (b) instead (checked in those tests).

    Source: [MGO20], Example 1.2 (d), p. 5.
    """
    head = [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3][:m]
    num1 = head + [2, 1, 1]
    num2 = head + [2, 2, 1]
    den = 1 + q + q**2
    assert is_zero(q_rational(3 * m + 1, 3) - series_expr(num1) / den)
    assert is_zero(q_rational(3 * m + 2, 3) - series_expr(num2) / den)


def test_mgo22_seven_fifths_series() -> None:
    """[7/5]_q = 1 + q^3 - 2q^5 + q^6 + 3q^7 - 3q^8 - 4q^9 + 7q^10 + 4q^11
    - 14q^12 + ...

    Source: [MGO22], Section 2.2, the example after Figure 1.
    """
    expected = [1, 0, 0, 1, 0, -2, 1, 3, -3, -4, 7, 4, -14]
    assert q_real_truncated("7/5", len(expected)) == expected


# Fibonacci convergents phi_n = F_(n+1)/F_n of the golden ratio.
MGO22_PHI_CONVERGENTS = [
    (13, 8, [1, 2, 3, 3, 3, 1], [1, 2, 2, 2, 1]),
    (34, 21, [1, 3, 5, 7, 7, 6, 4, 1], [1, 3, 4, 5, 4, 3, 1]),
    (55, 34, [1, 4, 7, 10, 11, 10, 7, 4, 1], [1, 4, 6, 7, 7, 5, 3, 1]),
]


@pytest.mark.parametrize("r,s,num,den", MGO22_PHI_CONVERGENTS)
def test_mgo22_golden_convergents(r: int, s: int, num: list[int], den: list[int]) -> None:
    """[phi_6]_q, [phi_8]_q, [phi_9]_q as printed.

    Source: [MGO22], Section 4.1, the display after Remark 4.1.
    """
    R, S = q_rational_pair(r, s)
    assert [int(c) for c in reversed(R.all_coeffs())] == num
    assert [int(c) for c in reversed(S.all_coeffs())] == den


def test_mgo22_golden_convergent_series_stabilize() -> None:
    """Taylor series of [phi_6]_q, [phi_8]_q, [phi_9]_q through q^12.

    Source: [MGO22], Section 4.1, "the corresponding Taylor series".
    """
    table = {
        (13, 8): [1, 0, 1, -1, 2, -3, 3, -3, 4, -5, 5, -5, 6],
        (34, 21): [1, 0, 1, -1, 2, -4, 8, -16, 30, -55, 103, -195, 368],
        (55, 34): [1, 0, 1, -1, 2, -4, 8, -17, 37, -82, 184, -414, 932],
    }
    for (r, s), expected in table.items():
        assert q_real_truncated(f"{r}/{s}", 13) == expected


# --------------------------------------------------------------------------
# q-reals: [MGO22] Sections 4 and 5
# --------------------------------------------------------------------------

MGO22_SERIES = {
    # Section 4.1, "The full series (13) starts as follows".
    "(1+sqrt(5))/2": [1, 0, 1, -1, 2, -4, 8, -17, 37, -82, 185, -423, 978,
                      -2283, 5373, -12735, 30372, -72832, 175502, -424748,
                      1032004],
    # Section 4.2, the silver ratio 1 + sqrt(2).
    "1+sqrt(2)": [1, 1, 0, 0, 1, 0, -2, 1, 4, -5, -7, 18, 7, -55, 18, 146,
                  -155, -322, 692, 476, -2446, 307, 7322, -6276, -18277,
                  33061, 33376, -129238, -10899],
    # Section 4.3.
    "sqrt(2)": [1, 0, 0, 1, 0, -2, 1, 4, -5, -7, 18, 7, -55, 18, 146, -155,
                -322, 692, 476, -2446, 307, 7322, -6276, -18277, 33061, 33376],
    "sqrt(3)": [1, 0, 1, 0, -1, 2, -2, -1, 7, -12, 7, 18, -59, 78, -1, -228,
                514, -469, -506, 2591, -4338, 1837, 9405, -27430, 33390, 10329],
    "sqrt(5)": [1, 1, 0, 0, 0, 0, 1, 0, -1, -1, -1, 3, 4, -1, -6, -11, 2, 25,
                22, -10, -70, -71, 67, 208, 168, -222],
    "sqrt(7)": [1, 1, 0, 1, -1, 2, -3, 4, -6, 8, -9, 9, -5, -9, 40, -101, 215,
                -411, 724, -1195, 1845, -2623, 3324, -3412, 1696, 4157],
    # Section 5.1, first 40 terms of [e]_q.
    "E": [1, 1, 0, 1, 0, -1, 2, -3, 3, -1, -3, 9, -17, 25, -29, 23, 2, -54,
          134, -232, 320, -347, 243, 71, -660, 1531, -2575, 3504, -3804, 2747,
          488, -6537, 15395, -25819, 34716, -36780, 24771, 9096, -70197,
          156811],
}  # fmt: skip

# Section 5.2, the first 80 terms of [pi]_q (q^0 .. q^79), nonzero entries.
MGO22_PI_NONZERO = {
    0: 1, 1: 1, 2: 1, 10: 1, 12: -1, 13: -1, 15: 1, 16: 1, 20: -1, 21: -2,
    22: -1, 23: 2, 24: 4, 25: 1, 27: -4, 28: -4, 29: -2, 30: 1, 31: 5, 32: 8,
    33: 3, 34: -3, 35: -10, 36: -12, 37: -5, 38: 8, 39: 19, 40: 20, 41: 2,
    42: -18, 43: -32, 44: -25, 46: 31, 47: 51, 48: 45, 49: -7, 50: -65,
    51: -94, 52: -57, 53: 35, 54: 122, 55: 140, 56: 72, 57: -76, 58: -209,
    59: -234, 60: -90, 61: 171, 62: 383, 63: 363, 64: 76, 65: -364, 66: -650,
    67: -545, 68: -6, 69: 702, 70: 1101, 71: 790, 72: -180, 73: -1329,
    74: -1824, 75: -1113, 76: 642, 77: 2454, 78: 2982, 79: 1415,
}  # fmt: skip


@pytest.mark.parametrize("x", sorted(MGO22_SERIES))
def test_mgo22_published_series(x: str) -> None:
    """Published Taylor coefficients of [x]_q for quadratic irrationals and e.

    Source: [MGO22], Sections 4.1 (golden ratio), 4.2 (silver ratio),
    4.3 (square roots of 2, 3, 5, 7) and 5.1 (e).
    """
    expected = MGO22_SERIES[x]
    assert q_real_truncated(x, len(expected)) == expected


def test_mgo22_pi_first_80_terms() -> None:
    """[pi]_q through q^79, including the vanishing coefficient of q^45.

    Source: [MGO22], Section 5.2.
    """
    expected = [MGO22_PI_NONZERO.get(k, 0) for k in range(80)]
    assert q_real_truncated("pi", 80) == expected


def _a004148() -> dict[int, int]:
    path = Path(__file__).parent / "data" / "A004148.txt"
    out: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        n, a = line.split()
        out[int(n)] = int(a)
    return out


def test_golden_ratio_is_signed_generalized_catalan() -> None:
    """phi_k = (-1)^k a(k-1) for k >= 2, a = OEIS A004148, checked to k = 60.

    Source: [MGO22], Proposition 4.2; values from the A004148 b-file.
    """
    a = _a004148()
    coeffs = q_real_truncated("(1+sqrt(5))/2", 61)
    assert coeffs[0] == 1 and coeffs[1] == 0
    for k in range(2, 61):
        assert coeffs[k] == (-1) ** k * a[k - 1], k


# --------------------------------------------------------------------------
# functional equations of quadratic irrationals, [MGO22] (14), (16)-(20)
# --------------------------------------------------------------------------

# x: (A2, A1, A0) with A2 [x]^2 + A1 [x] + A0 = 0 as a power series identity.
FUNCTIONAL_EQUATIONS = {
    "(1+sqrt(5))/2": (q, -(q**2 + q - 1), -1),                        # (14)
    "1+sqrt(2)": (q, -(q**3 + 2 * q - 1), -1),                        # (16)
    "sqrt(2)": (q**2, -(q**3 - 1), -(q**2 + 1)),                      # (17)
    "sqrt(3)": (q**2, -(q**3 + q**2 - q - 1), -(q**2 + q + 1)),       # (18)
    "sqrt(5)": (q**3, -(q**5 + q**3 - q**2 - 1),
                -(q**4 + q**3 + q**2 + q + 1)),                       # (19)
    "sqrt(7)": (q**3, -(q**5 + q**4 - q - 1),
                -(q**4 + 2 * q**3 + q**2 + 2 * q + 1)),               # (20)
}  # fmt: skip


@pytest.mark.parametrize("x", sorted(FUNCTIONAL_EQUATIONS))
def test_mgo22_functional_equations(x: str) -> None:
    """The quadratic functional equation of [x]_q holds through q^(N-1).

    [x]_q is known to N terms, so every coefficient of the residual below
    q^N is exact and must vanish. N = 80 goes far past the printed terms.

    Source: [MGO22], equations (14), (16), (17), (18), (19), (20).
    """
    N = 80
    A = series_expr(q_real_truncated(x, N))
    a2, a1, a0 = FUNCTIONAL_EQUATIONS[x]
    residual = sp.expand(a2 * A**2 + a1 * A + a0)
    low = [c for c in poly_coeffs(residual)[:N]]
    assert all(c == 0 for c in low), low[:10]
