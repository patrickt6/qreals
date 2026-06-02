r"""Continued-fraction algorithm for the q-real sum [x]_q + [y]_q.

The algorithm runs the
bihomographic state machine on the addition coefficients (0, 1, 1, 0; 0, 0, 0,
1) over the MGO q-blocks, ingesting the continued-fraction digits of x and y.
For rational x, y > 1 the continued fractions terminate, the state closes, and
the first-column ratio of the 2x4 state is the value R(q) / Q(q) = [x]_q +
[y]_q (the series sum, not [x+y]_q; the latter is the genuine q-real of x + y
and is reachable only through a separate deficit term).

The not-always-reduced caveat. The algorithm's
output R / Q is not always a reduced rational function: only sufficient
conditions for the finiteness of [x]_q + [y]_q come out of it directly, and the
genuine q-number is the cancelled form. `q_sum_rational` returns both the raw
state ratio and its sympy-cancelled reduction, with the caveat written into the
result.

For irrational x, y > 1 the continued fractions are infinite, so the algorithm
is run on a sequence of below-convergents x_n -> x, y_n -> y (the rational
algorithm applies to each). The convergent sums R_n(q) / Q_n(q) tend, as formal
Laurent series, to the limit [x]_q + [y]_q (the limit-of-the-sum-equals-sum-of-
the-limits step is independent of the algorithm).
`q_sum_irrational` returns the approximant sequence together with the Taylor
coefficients to which the run stabilises.

`transfer_matrix(cf)` is the 2x2 MGO q-continuant block product the algorithm
relies on. Its building block is T_q(a) = [[ [a]_q, q^a ],
[1, 0]] at 1-indexed-odd positions, [[ [a]_{1/q}, q^{-a} ], [1, 0]] at even
positions; the first column is (R_x, S_x) with [x]_q = R_x / S_x. The 4x4
transfer matrix is the Kronecker product T_q(a) (x) T_q(b),
the same blocks acting on V (x) V; this module reaches it implicitly through
`gosper.kron`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from ._parsing import parse_real
from .continued_fraction import make_even_length
from .gosper import _state_matrix, kron, q_block
from .rational import q


# ----------------------------------------------------------------------------
# the transfer matrix used by both rational and irrational variants
# ----------------------------------------------------------------------------
def transfer_matrix(cf: list[int]) -> sp.Matrix:
    """The 2x2 MGO q-continuant block product for a regular CF.

    Args:
        cf: a regular continued fraction as a list of integers (the algorithm
            ingests it in even-length form, padding via the qreals convention
            if needed).

    Returns:
        A sympy 2x2 matrix M with entries in Z[q, q^{-1}]. The first column is
        (R_x, S_x), giving [x]_q = R_x / S_x. For [x]_q + [y]_q the algorithm
        builds the Kronecker product of the two such matrices (one for x, one
        for y) inside the bihomographic state; that 4x4 view is the V (x) V
        operator picture.
    """
    cf = make_even_length([int(t) for t in cf])
    matrix = sp.eye(2)
    for i, a in enumerate(cf):
        matrix = matrix * q_block(i, a)
    return matrix


# ----------------------------------------------------------------------------
# the CF algorithm on rational inputs
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class QSumRational:
    """Result of the CF algorithm for [x]_q + [y]_q on rational x, y.

    Fields:
        x, y: the inputs as `Fraction`.
        cf_x, cf_y: the even-length regular CFs the algorithm ingested.
        raw_numerator, raw_denominator: the algorithm's R(q), Q(q), as sympy
            expressions in q. These come straight from the bihomographic state
            and are not always reduced (see module docstring).
        value: the reduced rational function R(q) / Q(q) = [x]_q + [y]_q. This
            is the same q-Gosper value `gosper.q_gosper(x, y, "add")` returns.
        reduced_numerator, reduced_denominator: the cancelled (R, Q).
        caveat: a one-line note recording the not-always-reduced property.
    """

    x: Fraction
    y: Fraction
    cf_x: tuple[int, ...]
    cf_y: tuple[int, ...]
    raw_numerator: sp.Expr
    raw_denominator: sp.Expr
    value: sp.Expr
    reduced_numerator: sp.Expr
    reduced_denominator: sp.Expr
    caveat: str


def _frac_cf(f: Fraction) -> list[int]:
    quotients = sp.continued_fraction(sp.Rational(f.numerator, f.denominator))
    return make_even_length([int(t) for t in quotients])


def q_sum_rational(x, y) -> QSumRational:
    """The CF algorithm for [x]_q + [y]_q on rational x, y > 0.

    Args:
        x, y: anything Fraction will accept (int, str like "3/2", a Fraction).

    Returns:
        A `QSumRational` carrying the raw R, Q from the bihomographic state,
        the cancelled value, and the not-always-reduced caveat.
    """
    fx = Fraction(x)
    fy = Fraction(y)
    cf_x = _frac_cf(fx)
    cf_y = _frac_cf(fy)
    state = _state_matrix("add")
    for i, a in enumerate(cf_x):
        state = state * kron(q_block(i, a), sp.eye(2))
    for j, b in enumerate(cf_y):
        state = state * kron(sp.eye(2), q_block(j, b))
    raw_num = sp.expand(state[0, 0])
    raw_den = sp.expand(state[1, 0])
    value = sp.cancel(raw_num / raw_den)
    red_num, red_den = sp.fraction(value)
    return QSumRational(
        x=fx,
        y=fy,
        cf_x=tuple(cf_x),
        cf_y=tuple(cf_y),
        raw_numerator=raw_num,
        raw_denominator=raw_den,
        value=value,
        reduced_numerator=sp.expand(red_num),
        reduced_denominator=sp.expand(red_den),
        caveat=(
            "The CF algorithm does not always return a reduced fraction; "
            "the genuine q-number sum is the cancelled value."
        ),
    )


# ----------------------------------------------------------------------------
# the convergent iterator for irrational inputs
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Approximant:
    """One step of the convergent iterator: a below-convergent and its value."""

    n: int
    x_n: Fraction
    y_n: Fraction
    value: sp.Expr


@dataclass(frozen=True)
class QSumIrrational:
    """Result of the convergent iterator for [x]_q + [y]_q on irrational x, y > 1.

    Fields:
        x, y: the original sympy expressions for the inputs.
        N: the number of stable Taylor coefficients requested.
        approximants: the sequence of `Approximant`, indexed by an even CF
            depth n = 2, 4, ..., so each x_n, y_n lies below x, y.
        coeffs: the Taylor coefficients [c_0, ..., c_{N-1}] to which the
            sequence stabilised on the last step.
        stabilised_at: the first even depth n at which the next step left
            `coeffs` unchanged; None if no stabilisation was observed in the
            window.
    """

    x: sp.Expr
    y: sp.Expr
    N: int
    approximants: tuple[Approximant, ...]
    coeffs: tuple[int, ...]
    stabilised_at: int | None


def _below_convergent(x_sym: sp.Expr, even_depth: int) -> Fraction:
    """The below-convergent of x at CF index k = even_depth (0-indexed even).

    Even-indexed convergents lie below x; we read the first even_depth + 1
    partial quotients off `sp.continued_fraction_iterator` and evaluate the
    truncated CF as a `Fraction`.
    """
    if even_depth < 0:
        raise ValueError("even_depth must be >= 0")
    if even_depth % 2 != 0:
        raise ValueError("even_depth must be even (below-convergent)")
    it = sp.continued_fraction_iterator(x_sym)
    cf: list[int] = []
    for _ in range(even_depth + 1):
        cf.append(int(next(it)))
    val = Fraction(cf[-1])
    for t in reversed(cf[:-1]):
        val = t + Fraction(1) / val
    return val


def _taylor_coeffs(expr: sp.Expr, N: int) -> tuple[int, ...]:
    if expr == 0:
        return tuple([0] * N)
    ser = sp.series(expr, q, 0, N).removeO()
    poly = sp.Poly(sp.expand(ser), q)
    return tuple(int(poly.coeff_monomial(q**k)) for k in range(N))


def q_sum_irrational(x, y, N: int) -> QSumIrrational:
    """Convergent iterator R_n / Q_n -> [x]_q + [y]_q for irrational x, y > 1.

    Runs the CF algorithm on the below-convergents x_n, y_n of x and y at
    even CF depths n = 2, 4, ..., up to a depth that fixes the first N Taylor
    coefficients of the sum. Returns the approximant sequence and the limit
    Taylor coefficients.

    Args:
        x, y: anything sympy.sympify accepts and reads as a real number > 1.
        N: the number of stable Taylor coefficients to stabilise on.

    Returns:
        A `QSumIrrational` with the approximant sequence and the limit coefficients.
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    x_sym = parse_real(x)
    y_sym = parse_real(y)
    approximants: list[Approximant] = []
    last: tuple[int, ...] | None = None
    stabilised_at: int | None = None
    max_depth = max(N + 2, 6)
    for n in range(2, max_depth + 1, 2):
        x_n = _below_convergent(x_sym, n)
        y_n = _below_convergent(y_sym, n)
        rec = q_sum_rational(x_n, y_n)
        approximants.append(Approximant(n=n, x_n=x_n, y_n=y_n, value=rec.value))
        coeffs = _taylor_coeffs(rec.value, N)
        if last is not None and coeffs == last and stabilised_at is None:
            stabilised_at = n
            last = coeffs
            break
        last = coeffs
    return QSumIrrational(
        x=x_sym,
        y=y_sym,
        N=N,
        approximants=tuple(approximants),
        coeffs=last if last is not None else tuple([0] * N),
        stabilised_at=stabilised_at,
    )


# ----------------------------------------------------------------------------
# the empirical finiteness check
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class FinitenessReport:
    """Verdict on whether [x]_q + [y]_q is a finite Laurent polynomial.

    Fields:
        finite: the verdict.
        polynomial: the cancelled Laurent polynomial when finite and the input
            is rational; the truncated Taylor coefficients otherwise.
        stabilised_at: the convergent depth at which the irrational case
            settled, or None.
        method: "rational" if the verdict is exact, "irrational-trailing-zeros"
            if it is the operational below-convergent run.
    """

    finite: bool
    polynomial: sp.Expr | tuple[int, ...] | None
    stabilised_at: int | None
    method: str


def finiteness_check(x, y, N: int) -> FinitenessReport:
    """Does [x]_q + [y]_q stabilise to a finite Laurent polynomial up to order N?

    For rational x, y the value is exact, and the verdict is "finite Laurent
    polynomial" iff the cancelled denominator is a monomial in q (a single
    power of q in the denominator is exactly the algorithm's output shape).
    For irrational x, y the function runs the convergent
    iterator to even depth ~N and reports "finite" when the truncated Taylor
    coefficients show a long trailing run of zeros, a finite-order observation
    consistent with the negation-finiteness theorem on the trace-zero quadratic
    catalogue (Ovsienko Example 6.4; the worked 4 +/- sqrt(7) example).

    Args:
        x, y: anything sympy.sympify accepts.
        N: the order to which the empirical run watches the Taylor tail; for
            rational inputs the verdict is exact and N is unused except to set
            the reported polynomial's series order if needed.

    Returns:
        A `FinitenessReport`.
    """
    if N < 4:
        raise ValueError("N must be at least 4")
    x_sym = parse_real(x)
    y_sym = parse_real(y)
    if x_sym.is_Rational and y_sym.is_Rational:
        rec = q_sum_rational(x_sym, y_sym)
        den_poly = sp.Poly(sp.expand(rec.reduced_denominator), q)
        finite = len(den_poly.terms()) == 1
        return FinitenessReport(
            finite=bool(finite),
            polynomial=rec.value if finite else None,
            stabilised_at=None,
            method="rational",
        )
    res = q_sum_irrational(x, y, N)
    coeffs = res.coeffs
    trailing = 0
    for c in reversed(coeffs):
        if c == 0:
            trailing += 1
        else:
            break
    finite = trailing >= N // 2
    return FinitenessReport(
        finite=bool(finite),
        polynomial=coeffs if finite else None,
        stabilised_at=res.stabilised_at,
        method="irrational-trailing-zeros",
    )
