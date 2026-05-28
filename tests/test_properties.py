"""Property-based cross-checks for every public function of qreals.

Each public name re-exported from ``qreals.__init__`` is checked against an
independent method: the q = 1 specialisation, the Gauss closed form, a
second computation path, a brute-force reference, or a theorem proved in the
MGO papers. Citations:

  RAT  = Morier-Genoud, Ovsienko, "q-deformed rationals and q-continued
         fractions", Forum Math. Sigma 8 (2020), e13 (arXiv:1812.00170).
  REAL = Morier-Genoud, Ovsienko, "On q-deformed real numbers"
         (arXiv:1908.04365).

Symbolic cases run with the Hypothesis deadline disabled because sympy
cancellation is not uniformly fast; correctness, not timing, is the subject.
"""

from math import gcd

import pytest
import sympy as sp
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from qreals import (
    coefficient_max_abs,
    first_negative_coefficient_index,
    first_nonzero_coefficient_index,
    format_laurent,
    integer_part_prefix,
    mgo_laurent,
    number_of_zeros,
    q,
    q_int,
    q_int_qinv,
    q_rational,
    q_real_truncated,
    shift_down,
    shift_up,
)


def _taylor(expr: sp.Expr, n: int) -> list[int]:
    """Independent Taylor read-out: the q^0..q^{n-1} coefficients of expr."""
    truncated = sp.series(expr, q, 0, n).removeO()
    poly = sp.Poly(truncated, q)
    return [int(poly.coeff_monomial(q**k)) for k in range(n)]


def _numer_denom(p: int, s: int) -> tuple[sp.Poly, sp.Poly]:
    """Canonical (R, S) of [p/s]_q for p > s coprime: constant term 1 each."""
    num, den = sp.fraction(sp.cancel(q_rational(p, s)))
    c = sp.expand(num).subs(q, 0)
    return sp.Poly(sp.expand(num / c), q), sp.Poly(sp.expand(den / c), q)


# === q_rational =========================================================


@given(p=st.integers(1, 60), s=st.integers(1, 60))
@settings(deadline=None, max_examples=60)
def test_q_rational_specialises_to_ordinary_rational_at_q_one(p, s):
    # RAT, Corollary 1.7(iii): R(1) = r, S(1) = s, hence [p/s]_q|_{q=1} = p/s.
    assume(p != s)
    assert sp.nsimplify(q_rational(p, s).subs(q, 1)) == sp.Rational(p, s)


@given(p=st.integers(2, 40), s=st.integers(1, 40))
@settings(deadline=None, max_examples=40)
def test_q_rational_corollary_1_7_invariants(p, s):
    # RAT, Corollary 1.7: for r/s > 1 coprime, R and S have constant term 1,
    # leading coefficient 1, positive integer coefficients, R(1) = r, S(1) = s.
    g = gcd(p, s)
    r, ss = p // g, s // g
    assume(r > ss)  # the corollary is stated for r/s > 1
    R, S = _numer_denom(p, s)
    assert R.subs(q, 0) == 1 and S.subs(q, 0) == 1
    assert R.LC() == 1 and S.LC() == 1
    assert all(c > 0 for c in R.all_coeffs())
    assert all(c > 0 for c in S.all_coeffs())
    assert R.subs(q, 1) == r and S.subs(q, 1) == ss


@given(p=st.integers(2, 40), s=st.integers(1, 40))
@settings(deadline=None, max_examples=40)
def test_q_rational_proposition_1_8_q_equals_minus_one(p, s):
    # RAT, Proposition 1.8: R(-1), S(-1) lie in {-1, 0, 1}; R(-1) = 0 iff r is
    # even; S(-1) = 0 iff s is even.
    g = gcd(p, s)
    r, ss = p // g, s // g
    assume(r > ss)
    R, S = _numer_denom(p, s)
    assert R.subs(q, -1) in (-1, 0, 1)
    assert S.subs(q, -1) in (-1, 0, 1)
    assert (R.subs(q, -1) == 0) == (r % 2 == 0)
    assert (S.subs(q, -1) == 0) == (ss % 2 == 0)


@pytest.mark.parametrize(
    "a,b,c,d",
    [
        (3, 2, 2, 1),
        (5, 3, 3, 2),
        (7, 5, 4, 3),
        (7, 2, 3, 1),
        (22, 7, 19, 6),
        (5, 2, 7, 3),
    ],
)
def test_q_rational_corollary_1_4_farey_determinant_is_a_monomial(a, b, c, d):
    # RAT, Corollary 1.4: for Farey neighbours (a*d - c*b = +-1), the
    # determinant R1 S2 - S1 R2 is a single power of q (up to sign).
    assert abs(a * d - c * b) == 1
    R1, S1 = _numer_denom(a, b)
    R2, S2 = _numer_denom(c, d)
    det = sp.expand(R1.as_expr() * S2.as_expr() - S1.as_expr() * R2.as_expr())
    assert det == 0 or len(sp.Add.make_args(det)) == 1


# === q_real_truncated ===================================================


@given(p=st.integers(1, 50), s=st.integers(1, 50), n=st.integers(2, 24))
@settings(deadline=None, max_examples=50)
def test_truncated_matches_taylor_of_exact_rational(p, s, n):
    # REAL p. 1: for a rational the CF terminates, so the stable series is the
    # Taylor expansion of the exact q_rational. The two paths must agree.
    assume(p != s)
    got = q_real_truncated(f"({p})/({s})", n)
    want = _taylor(q_rational(p, s), n)
    assert got == want


@given(
    x=st.sampled_from(
        ["pi", "sqrt(2)", "(1+sqrt(5))/2", "E", "sqrt(3)", "5/7", "22/7"]
    ),
    short=st.integers(5, 30),
    extra=st.integers(1, 25),
)
@settings(deadline=None, max_examples=40)
def test_truncation_stability(x, short, extra):
    # REAL, Theorem 1 and Proposition 1.1: the first N coefficients do not move
    # as N grows. Asking for more must not change the ones already returned.
    a = q_real_truncated(x, short)
    b = q_real_truncated(x, short + extra)
    assert a == b[:short]


def test_mgo_laurent_is_truncated_shifted_by_one():
    # mgo_laurent(x, order) returns q^0..q^order, i.e. q_real_truncated(x, order+1).
    for x in ("pi", "sqrt(2)", "(1+sqrt(5))/2", "12/5"):
        for order in (0, 5, 11):
            assert mgo_laurent(x, order) == q_real_truncated(x, order + 1)


# === integer_part_prefix (REAL, Theorem 2: the gap theorem) =============


@given(p=st.integers(1, 60), s=st.integers(1, 30))
@settings(deadline=None, max_examples=40)
def test_integer_part_prefix_is_the_real_series_opening(p, s):
    # REAL, Theorem 2: for floor(x) = t the series opens 1,...,1 (t ones) then
    # a 0 at q^t. The forced prefix must equal the actual leading coefficients.
    t = int(sp.floor(sp.Rational(p, s)))
    prefix = integer_part_prefix(sp.Rational(p, s))
    assert prefix == [1] * t + [0]
    assert q_real_truncated(f"({p})/({s})", t + 1) == prefix


# === shift_down / shift_up (REAL, eqn 3: the translation group) =========


@given(c=st.lists(st.integers(-1000, 1000), min_size=0, max_size=30))
def test_shift_down_inverts_shift_up(c):
    # shift_up is c -> [1] + c (i.e. q[x]_q + 1); shift_down is its inverse.
    assert shift_up(c) == [1] + c
    assert shift_down(shift_up(c)) == c


@pytest.mark.parametrize("x", ["pi", "sqrt(2)+2", "E+1", "(7+sqrt(5))/2"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_shift_chain_matches_direct_translation(x, k):
    # REAL, eqn (3): repeatedly applying shift_down to [x]_q yields [x-k]_q.
    base = mgo_laurent(x, 14)
    shifted = base
    for _ in range(k):
        shifted = shift_down(shifted)
    direct = q_real_truncated(f"({x})-{k}", len(shifted))
    assert shifted == direct


# === q_int / q_int_qinv =================================================


@given(n=st.integers(-12, 12))
def test_q_int_matches_gauss_closed_form(n):
    # The Gauss q-integer [n]_q. For n > 0 it is (q^n - 1)/(q - 1); for all n
    # it specialises to n at q = 1.
    assert sp.nsimplify(q_int(n).subs(q, 1)) == n
    if n > 0:
        assert sp.cancel(q_int(n) - (q**n - 1) / (q - 1)) == 0


@given(n=st.integers(-12, 12))
def test_q_int_qinv_is_q_int_under_q_to_qinv(n):
    # [n]_{q^{-1}} is [n]_q with q replaced by 1/q; it also specialises to n.
    assert sp.cancel(q_int_qinv(n) - q_int(n).subs(q, sp.Integer(1) / q)) == 0
    assert sp.nsimplify(q_int_qinv(n).subs(q, 1)) == n


# === coefficient read-outs ==============================================


@given(c=st.lists(st.integers(-50, 50), min_size=0, max_size=40))
def test_coefficient_readouts_match_brute_force(c):
    nz = next((i for i, v in enumerate(c) if v != 0), -1)
    neg = next((i for i, v in enumerate(c) if v < 0), -1)
    assert first_nonzero_coefficient_index(c) == nz
    assert first_negative_coefficient_index(c) == neg
    assert coefficient_max_abs(c) == max((abs(v) for v in c), default=0)
    assert number_of_zeros(c) == sum(1 for v in c if v == 0)


# === format_laurent =====================================================


@given(c=st.lists(st.integers(-30, 30), min_size=1, max_size=15))
def test_format_laurent_round_trips_through_sympy(c):
    # The rendered polynomial, reparsed, must recover the input coefficients.
    rendered = format_laurent(c)
    body = rendered.split(" + O(q^")[0]
    poly = (
        sp.Poly(sp.sympify(body, locals={"q": q}), q) if sp.sympify(body) != 0 else None
    )
    recovered = [0] * len(c)
    if poly is not None:
        for k in range(len(c)):
            recovered[k] = int(poly.coeff_monomial(q**k))
    assert recovered == c


def test_shift_down_rejects_empty_input():
    with pytest.raises(ValueError):
        shift_down([])


# === remaining public exports: the q symbol and the version string ======


def test_q_is_the_sympy_symbol_used_by_q_rational():
    assert q == sp.Symbol("q")
    assert q_rational(3, 2).free_symbols <= {q}


def test_version_matches_packaging_metadata():
    import tomllib
    from pathlib import Path

    import qreals

    root = Path(__file__).resolve().parent.parent
    meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert qreals.__version__ == meta["project"]["version"]
