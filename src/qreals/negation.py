r"""The negation sum G(x) = [x]_q + [-x]_q with certified coefficient locking.

This module studies Ovsienko's Example 6.4: for which real x is
G(x) = [x]_q + [-x]_q a finite Laurent polynomial. It gives an exact route for
two classes of x:

  * x rational: G(x) is computed as an exact rational function of q by the
    REVERSAL RULE, [-x]_q := -q^-1 [x]_{1/q} (substitute q -> 1/q in the exact
    rational function [x]_q, then multiply by -q^-1), so finiteness is a
    certain, closed-form fact, not a finite-order guess.

    LEFT/RIGHT CONVENTION, READ BEFORE TOUCHING THIS PATH. At a rational x the
    q-deformation has genuinely different left and right limits, so there is
    no series-limit way to define [-x]_q at a rational point: a convergent
    computation is convention-dependent and, at rationals, gives the wrong
    answer for this project. The convention used throughout MGO,
    Leclere-Morier-Genoud, this library, and every dataset shipped from this
    project is the reversal rule above, NOT the Jouteur PGL_2(Z) formula
    negate.py uses for genuinely irrational x (that formula is the analytic
    continuation appropriate off the rationals; at a rational input it
    disagrees with the reversal rule and must not be used for this gate).
    Concretely, under the reversal rule, G(1/s) = 1 - q^-1 exactly for every
    integer s >= 2 (checked below for s = 2, 3, 5), and G(17/12) = q - q^-2
    exactly, both proven classification results, not observations. A rational
    x is NOT restricted to "infinite" here: "trace-zero quadratics are the
    only finite case" is an irrational-x statement (Ovsienko Example 6.4) and
    does not extend to rationals, where the reversal rule can and does also
    produce finite Laurent polynomials (1/s and 17/12 above are both finite).

  * x = a + b*sqrt(D), a, b rational, D squarefree: [x]_q and [-x]_q are each
    built as a product of q-deformed Hirzebruch-Jung step matrices

        M(c) = [[ [c]_q, -q^(c-1) ], [ 1, 0 ]]

    over the HJ convergents of x (see `quadratic.hj_terms`), then read off as
    a Laurent series by exact integer-coefficient long division of the
    matrix's first column. A coefficient is LOCKED once two successive
    convergents agree on it; `negation_sum_exact` reports how many leading
    coefficients are locked and, once enough of them are locked and all zero
    past a low-degree window, calls the series finite-looking.

GATE-2 (resolved). An earlier draft of this experiment flagged a possible
discrepancy: the raw [sqrt(2)]_q Taylor coefficients computed via the
Hirzebruch-Jung route (1, 0, 0, 1, 0, -2, 1, 4, -5, ...) were compared against
a cited value "1, 1, 0, -1, 2, -4, 9, -21" attributed to arXiv:1908.04365 and
did not match. Reading the actual paper (Section 4.3, "The q-square roots of
2, 3, 5 and 7") shows the cited comparison value was wrong: the paper prints

    [sqrt(2)]_q = 1 + q^3 - 2q^5 + q^6 + 4q^7 - 5q^8 - 7q^9 + 18q^10 + ...

i.e. coefficients 1, 0, 0, 1, 0, -2, 1, 4, -5, -7, 18, ..., which is exactly
what both the Hirzebruch-Jung route here and the pre-existing regular-CF route
in `truncated.q_real_truncated` compute. The same cross-check was repeated for
sqrt(3), sqrt(5), and the golden ratio (1+sqrt(5))/2, all matching the paper
exactly. There is no bug and the two continued-fraction conventions agree, as
the Jouteur left/right symmetry requires; the "1, 1, 0, -1, 2, -4, 9, -21"
figure is retracted. The G(sqrt(2)) = q - q^-2 identity holds under this
(now singly confirmed) convention, checked directly below and in the tests.

Coefficient locking, in more detail. Unlike the regular-CF route, whose
locking bound is `expansions.coeffs_locked_by_convergent` (stated for regular
continued fractions with positive partial quotients and the partial-sum bound
S_n - 1), the Hirzebruch-Jung route here uses the direct, empirical form of
the same idea: a coefficient is locked once the Laurent series computed from
convergent n and from convergent n - 1 agree on it. This is the same
principle (successive convergents pin down a growing low-degree window) but
is verified by direct comparison rather than by reusing the regular-CF sum
bound, because the HJ partial quotients do not obey the same positivity
(c_1 may be any integer) that the S_n bound is stated for.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from ._parsing import parse_real
from .quadratic import QuadraticIrrational, hj_terms
from .rational import q_rational

_q = sp.Symbol("q")

# Laurent polynomials represented as {degree: Fraction}, truncated to a fixed
# window during matrix products so degrees stay bounded through the fold.
LaurentDict = dict[int, Fraction]

Mat = tuple[LaurentDict, LaurentDict, LaurentDict, LaurentDict]


# ----------------------------------------------------------------------------
# Laurent-polynomial arithmetic (truncated window), the hot path
# ----------------------------------------------------------------------------
def _trunc(d: LaurentDict, lo: int, hi: int) -> LaurentDict:
    return {k: v for k, v in d.items() if lo <= k <= hi and v != 0}


def _lmul(a: LaurentDict, b: LaurentDict, lo: int, hi: int) -> LaurentDict:
    out: dict[int, Fraction] = {}
    for ka, va in a.items():
        if va == 0:
            continue
        for kb, vb in b.items():
            if vb == 0:
                continue
            k = ka + kb
            if k < lo or k > hi:
                continue
            out[k] = out.get(k, 0) + va * vb
    return {k: v for k, v in out.items() if v != 0}


def _ladd(a: LaurentDict, b: LaurentDict) -> LaurentDict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


def q_bracket(c: int) -> LaurentDict:
    """[c]_q as a Laurent dict, for any integer c (c <= 0 gives negative powers)."""
    d: dict[int, Fraction] = {}
    if c > 0:
        for i in range(c):
            d[i] = Fraction(1)
    elif c < 0:
        for i in range(c, 0):
            d[i] = Fraction(-1)
    return d


def _M(c: int) -> Mat:
    """M(c) = [[ [c]_q, -q^(c-1) ], [ 1, 0 ]] as a block of four Laurent dicts."""
    return (q_bracket(c), {c - 1: Fraction(-1)}, {0: Fraction(1)}, {})


_IDENTITY: Mat = ({0: Fraction(1)}, {}, {}, {0: Fraction(1)})


def _mat_mul(A: Mat, B: Mat, lo: int, hi: int) -> Mat:
    a00, a01, a10, a11 = A
    b00, b01, b10, b11 = B
    c00 = _ladd(_lmul(a00, b00, lo, hi), _lmul(a01, b10, lo, hi))
    c01 = _ladd(_lmul(a00, b01, lo, hi), _lmul(a01, b11, lo, hi))
    c10 = _ladd(_lmul(a10, b00, lo, hi), _lmul(a11, b10, lo, hi))
    c11 = _ladd(_lmul(a10, b01, lo, hi), _lmul(a11, b11, lo, hi))
    return (_trunc(c00, lo, hi), _trunc(c01, lo, hi), _trunc(c10, lo, hi), _trunc(c11, lo, hi))


def laurent_divide(num: LaurentDict, den: LaurentDict, max_deg: int) -> LaurentDict:
    """num / den as a Laurent series, exact integer-coefficient long division.

    Both num and den are truncated Laurent dicts; the quotient is returned up
    to absolute degree max_deg.
    """
    if not num:
        return {}
    if not den:
        raise ZeroDivisionError("zero denominator series")
    vn = min(num)
    vd = min(den)
    numc = {k - vn: v for k, v in num.items()}
    denc = {k - vd: v for k, v in den.items()}
    d0 = denc.get(0)
    if not d0:
        raise ZeroDivisionError("denominator series has no leading term")
    order = max_deg - (vn - vd) + 2
    if order < 0:
        return {}
    c: dict[int, Fraction] = {}
    for n in range(order + 1):
        s = numc.get(n, Fraction(0))
        for i in range(1, n + 1):
            di = denc.get(i)
            if di:
                s -= di * c[n - i]
        c[n] = s / d0
    return {(vn - vd + n): v for n, v in c.items() if v != 0}


# ----------------------------------------------------------------------------
# locked-coefficient series for a QuadraticIrrational, via HJ convergents
# ----------------------------------------------------------------------------
WINDOW_LO = -12
DEFAULT_LOCK_TARGET = 55


def locked_series(
    x: QuadraticIrrational,
    depth: int,
    max_hj_terms: int = 200,
    lock_target: int = DEFAULT_LOCK_TARGET,
) -> tuple[LaurentDict, int]:
    """Laurent coefficients of [x]_q locked by agreement of successive HJ convergents.

    Returns (series, locked_depth): series holds the coefficients over
    [WINDOW_LO, WINDOW_LO + depth) that have stabilized between two successive
    convergents, and locked_depth counts how many of those (from the bottom)
    agree. Stops early once locked_depth reaches lock_target or the HJ term
    budget is exhausted.
    """
    hi = WINDOW_LO + depth
    terms, _, _ = hj_terms(x, max_hj_terms)
    M: Mat = _IDENTITY
    prev_conv: LaurentDict | None = None
    locked_depth = 0
    final_conv: LaurentDict | None = None
    window = list(range(WINDOW_LO, hi))
    for c in terms:
        M = _mat_mul(M, _M(c), WINDOW_LO, hi)
        try:
            conv = laurent_divide(M[0], M[2], max_deg=hi)
        except ZeroDivisionError:
            continue
        if prev_conv is not None:
            d = 0
            for deg in window:
                if conv.get(deg, 0) == prev_conv.get(deg, 0):
                    d += 1
                else:
                    break
            locked_depth = d
            final_conv = conv
            if locked_depth >= lock_target:
                break
        prev_conv = conv
        final_conv = conv
    series = {deg: final_conv.get(deg, 0) for deg in window[:locked_depth]} if final_conv else {}
    return series, locked_depth


# ----------------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Verdict:
    kind: str  # "finite_looking", "infinite", or "insufficient_depth"
    first_nonzero_tail_index: int | None
    polynomial: LaurentDict


def default_lock_target(depth: int, min_zero_run: int = 30, tail_start: int = 6, buffer: int = 10) -> int:
    """A lock target big enough to judge finiteness, but not the whole window.

    Locking the entire requested depth is far more expensive than needed: the
    verdict needs locked_depth large enough that the window
    [WINDOW_LO, WINDOW_LO + locked_depth) contains min_zero_run degrees at or
    past tail_start (see `classify`), so this asks for a modest margin past
    that bound rather than forcing agreement all the way out to `depth`,
    which matters a lot for sweep throughput at large depth.
    """
    required = -WINDOW_LO + tail_start + min_zero_run
    return min(depth, required + buffer)


def classify(series: LaurentDict, locked_depth: int, min_zero_run: int = 30, tail_start: int = 6) -> Verdict:
    """finite_looking / infinite / insufficient_depth from a locked series."""
    if locked_depth < tail_start + min_zero_run:
        return Verdict("insufficient_depth", None, {})
    tail_indices = [WINDOW_LO + i for i in range(locked_depth) if WINDOW_LO + i >= tail_start]
    first_nonzero = None
    for d in tail_indices:
        if series.get(d, 0) != 0:
            first_nonzero = d
            break
    if first_nonzero is not None:
        return Verdict("infinite", first_nonzero, {})
    if len(tail_indices) < min_zero_run:
        return Verdict("insufficient_depth", None, {})
    return Verdict("finite_looking", None, dict(series))


@dataclass(frozen=True)
class NegationSumResult:
    valuation: int
    locked_coefficients: dict[int, Fraction]
    verdict: str
    first_nonzero_tail_index: int | None
    polynomial: dict[int, Fraction]
    locked_depth: int


# ----------------------------------------------------------------------------
# exact rational path
# ----------------------------------------------------------------------------
def _q_rational_pos(a: Fraction) -> sp.Expr:
    """[a]_q for a rational a >= 0, as an exact expr in q (`rational.q_rational`)."""
    if a == 0:
        return sp.Integer(0)
    return q_rational(a.numerator, a.denominator)


def _reversal_negation(A: sp.Expr) -> sp.Expr:
    """[-x]_q at a rational x, by the reversal rule [-x]_q := -q^-1 [x]_{1/q}.

    A is the exact rational function [x]_q for x >= 0 (from `q_rational`).
    Substituting q -> 1/q and multiplying by -q^-1 gives the reversal-rule
    value of [-x]_q. This is the convention this project uses at rational
    points (see the module docstring); it is NOT the Jouteur PGL_2(Z) formula
    `negate.py` applies for irrational x, which disagrees with it here.
    """
    reflected = A.subs(_q, 1 / _q)
    return sp.cancel(-reflected / _q)


def negation_sum_rational_exact(a: Fraction) -> tuple[str, dict[int, Fraction] | None, sp.Expr]:
    """G(a) = [a]_q + [-a]_q for rational a, exact via the reversal rule.

    G is even in a (G(-a) = G(a)): [-a]_q and [-(-a)]_q = [a]_q are each other's
    reversal-rule image, so this always works from A = [|a|]_q regardless of
    the sign of a.

    Returns (verdict, polynomial, expr): verdict is "finite" or "infinite",
    polynomial is a {degree: Fraction} dict when finite (else None), and expr
    is the exact rational function of q in every case (so an "infinite"
    verdict still comes with an exact object to expand as a series to any
    order, rather than only a yes/no answer).
    """
    A = _q_rational_pos(abs(a))
    if a == 0:
        g = sp.Integer(0)
    else:
        g = sp.cancel(A + _reversal_negation(A))
    numer, denom = sp.fraction(g)
    denom = sp.expand(denom)
    denom_poly = sp.Poly(denom, _q) if denom.free_symbols else None
    is_monomial = denom_poly is None or denom_poly.is_monomial
    if not is_monomial:
        return "infinite", None, g
    # denom is c * q^k (or a constant); shift the numerator down by k and
    # divide by c to read off the finite Laurent polynomial exactly.
    if denom_poly is None:
        shift = 0
        c0 = denom
    else:
        monoms = denom_poly.monoms()
        shift = monoms[0][0] if monoms else 0
        c0 = denom_poly.coeffs()[0] if denom_poly.coeffs() else sp.Integer(1)
    numer_poly = sp.Poly(sp.expand(numer), _q)
    out: dict[int, Fraction] = {}
    for monom, coeff in zip(numer_poly.monoms(), numer_poly.coeffs()):
        deg = monom[0] - shift
        val = sp.nsimplify(coeff / c0)
        out[deg] = Fraction(int(sp.numer(val)), int(sp.denom(val)))
    return "finite", {k: v for k, v in out.items() if v != 0}, g


def _laurent_valuation(expr: sp.Expr) -> int:
    """The valuation (lowest power of q with a nonzero coefficient) of expr."""
    if expr == 0:
        return 0
    numer, denom = sp.fraction(sp.together(expr))
    numer_poly = sp.Poly(sp.expand(numer), _q)
    denom_poly = sp.Poly(sp.expand(denom), _q)
    numer_low = min(m[0] for m in numer_poly.monoms())
    denom_low = min(m[0] for m in denom_poly.monoms())
    return numer_low - denom_low


def exact_laurent_expansion(expr: sp.Expr, n_terms: int) -> dict[int, Fraction]:
    """n_terms EXACT Laurent coefficients of a rational function expr in q.

    Unlike a locked-coefficient computation, every entry here is proven
    correct: expr is an exact closed form, so any prefix of its Laurent
    series is exact, not an empirical observation. Used for the rational
    "infinite" verdict, where the answer is certain but has no finite
    polynomial to report in full.
    """
    if expr == 0:
        return {}
    lowest = _laurent_valuation(expr)
    ser = sp.series(expr, _q, 0, lowest + n_terms).removeO()
    poly_expr = sp.Poly(sp.expand(sp.together(ser * _q ** (-lowest))), _q)
    out: dict[int, Fraction] = {}
    for monom, coeff in zip(poly_expr.monoms(), poly_expr.coeffs()):
        k = monom[0] + lowest
        if lowest <= k < lowest + n_terms:
            val = sp.nsimplify(coeff)
            out[k] = Fraction(int(sp.numer(val)), int(sp.denom(val)))
    return {k: v for k, v in out.items() if v != 0}


# ----------------------------------------------------------------------------
# parsing: rational vs a + b*sqrt(D)
# ----------------------------------------------------------------------------
def _parse_quadratic_or_rational(x_repr: str) -> tuple[str, Fraction] | tuple[str, QuadraticIrrational]:
    expr = sp.sympify(x_repr)
    expr = sp.radsimp(sp.nsimplify(expr))
    rad_atoms = [a for a in expr.atoms(sp.Pow) if a.exp == sp.Rational(1, 2)]
    if not rad_atoms:
        return "rational", Fraction(sp.nsimplify(expr))
    if len(rad_atoms) > 1:
        raise ValueError(f"expected a single square-root term, got {x_repr}")
    rad = rad_atoms[0]
    d_val = rad.base
    if not d_val.is_Integer or int(d_val) <= 1:
        raise ValueError(f"expected sqrt(D) for an integer D > 1, got {rad}")
    D = int(d_val)
    b_expr = sp.simplify(expr.coeff(rad))
    a_expr = sp.simplify(expr - b_expr * rad)
    a = Fraction(sp.nsimplify(a_expr))
    b = Fraction(sp.nsimplify(b_expr))
    return "quadratic", QuadraticIrrational.from_ab(a, b, D)


def negation_sum_exact(
    x: str,
    depth: int = 120,
    min_zero_run: int = 30,
    lock_target: int | None = None,
) -> NegationSumResult:
    """G(x) = [x]_q + [-x]_q, exact for rational x and locked for quadratic x.

    Args:
        x: a sympy-parseable string, either a plain rational ("3/2") or an
            element of Q(sqrt D) ("1 + sqrt(2)/3", "-2*sqrt(3)").
        depth: how many Laurent coefficients (per side) to attempt to lock,
            for the quadratic path.
        min_zero_run: consecutive zero coefficients past degree 6 required to
            call the tail finite-looking.
        lock_target: convergent-agreement target; defaults to a value derived
            from depth.

    Returns:
        A NegationSumResult. For rational x, verdict is "finite" or
        "infinite", both exact (proven, not observed) via the reversal rule.
        "finite" carries the complete Laurent polynomial and sets locked_depth
        to -1 (a sentinel meaning "exact and complete, no depth needed", never
        paired with empty data unless G is identically zero). "infinite"
        carries `depth` genuinely exact Laurent coefficients (a proven prefix
        of an infinite series, not a guess) and the true first_nonzero_tail_index
        found directly from that exact expansion; locked_depth is set to
        `depth` in this case, matching how many exact coefficients were
        computed. For quadratic x, verdict is one of "finite_looking",
        "infinite", "insufficient_depth".
    """
    kind, value = _parse_quadratic_or_rational(x)
    if kind == "rational":
        verdict, poly, expr = negation_sum_rational_exact(value)
        if verdict == "finite":
            return NegationSumResult(
                valuation=min(poly) if poly else 0,
                locked_coefficients=dict(poly) if poly else {},
                verdict="finite",
                first_nonzero_tail_index=None,
                polynomial=dict(poly) if poly else {},
                locked_depth=-1,
            )
        coeffs = exact_laurent_expansion(expr, depth)
        tail_start = 6
        first_nonzero = next((d for d in sorted(coeffs) if d >= tail_start and coeffs[d] != 0), None)
        return NegationSumResult(
            valuation=min(coeffs) if coeffs else 0,
            locked_coefficients=coeffs,
            verdict="infinite",
            first_nonzero_tail_index=first_nonzero,
            polynomial={},
            locked_depth=depth,
        )

    assert isinstance(value, QuadraticIrrational)
    target = lock_target if lock_target is not None else default_lock_target(depth, min_zero_run)
    window_depth = target if lock_target is None else depth
    pos_series, pos_depth = locked_series(value, window_depth, lock_target=target)
    neg_series, neg_depth = locked_series(-value, window_depth, lock_target=target)
    locked_depth = min(pos_depth, neg_depth)
    window = list(range(WINDOW_LO, WINDOW_LO + locked_depth))
    g_series = {deg: pos_series.get(deg, 0) + neg_series.get(deg, 0) for deg in window}
    g_series = {k: v for k, v in g_series.items() if v != 0}
    verdict_obj = classify({**g_series}, locked_depth, min_zero_run=min_zero_run)
    coeffs = {deg: pos_series.get(deg, 0) + neg_series.get(deg, 0) for deg in window}
    return NegationSumResult(
        valuation=min(coeffs) if coeffs else 0,
        locked_coefficients=coeffs,
        verdict=verdict_obj.kind,
        first_nonzero_tail_index=verdict_obj.first_nonzero_tail_index,
        polynomial=verdict_obj.polynomial,
        locked_depth=locked_depth,
    )


def polynomial_string(poly: dict[int, Fraction]) -> str:
    """Render a {degree: coeff} Laurent dict as a compact q-polynomial string."""
    if not poly:
        return "0"
    parts = []
    for d in sorted(poly, reverse=True):
        v = poly[d]
        if v == 0:
            continue
        parts.append(f"({v})*q^{d}")
    return " + ".join(parts) if parts else "0"


# ----------------------------------------------------------------------------
# route arithmetic.negation_sum through the exact path (fixes the truncation bug)
# ----------------------------------------------------------------------------
def negation_sum_fixed(x: str, N: int) -> tuple[int, list[int]]:
    """A drop-in, exact-routed replacement for `arithmetic.negation_sum`.

    Same signature and same return shape (valuation, N coefficients starting
    at q^valuation), but for rational x the coefficients come from the exact
    closed form (never a finite-order guess), and for quadratic irrationals
    they come from the locked HJ-convergent series here rather than a single
    fixed-precision truncation.
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    try:
        kind, value = _parse_quadratic_or_rational(x)
    except Exception:
        kind, value = None, None

    if kind == "rational":
        assert isinstance(value, Fraction)
        verdict, poly, expr = negation_sum_rational_exact(value)
        if verdict == "finite":
            v = min(poly) if poly else 0
            hi = v + N
            coeffs = [int(poly.get(d, 0)) for d in range(v, hi)]
            return v, coeffs
        # Exact rational function, genuinely infinite: expand it as a Laurent
        # series to exactly the requested precision, no truncation heuristic.
        lowest = _laurent_valuation(expr)
        exact = exact_laurent_expansion(expr, N)
        coeffs = [int(exact.get(lowest + k, 0)) for k in range(N)]
        return lowest, coeffs

    if kind == "quadratic":
        result = negation_sum_exact(x, depth=max(N + 70, 100))
        if result.verdict == "finite_looking":
            poly = result.polynomial
            v = min(poly) if poly else 0
            hi = v + N
            return v, [int(poly.get(d, 0)) for d in range(v, hi)]
        if result.verdict == "infinite" and result.locked_depth >= N:
            v = min(result.locked_coefficients)
            hi = v + N
            return v, [int(result.locked_coefficients.get(d, 0)) for d in range(v, hi)]

    # Fall back to the original truncated-series path (e.g. a real x that is
    # not rational and not a simple a + b*sqrt(D) quadratic, such as pi).
    from .arithmetic import _negation_sum_truncated

    return _negation_sum_truncated(x, N)
