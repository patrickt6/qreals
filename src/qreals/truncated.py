"""The q-real [x]_q as a truncated power series, for any real x.

This is the path for irrationals (pi, sqrt(2), the golden ratio) and for any
input where the exact rational function is not wanted. The MGO formula is
evaluated bottom-up over the truncated-series kernel in `series`, so the
result is the first N stable integer Taylor coefficients of [x]_q.

For a rational p/s the CF terminates and this returns the Taylor expansion of
the exact function computed in `rational`; the two agree coefficient for
coefficient, which the test suite checks.
"""

from __future__ import annotations

import sympy as sp

from . import series
from .continued_fraction import cf_partials, make_even_length
from .series import Series


def q_int_series(n: int, prec: int) -> Series:
    """[n]_q as a series at q = 0, truncated to q^prec."""
    n = int(n)
    if n == 0:
        return 0, []
    if n > 0:
        return series.normalise((0, [1] * min(n, prec)))
    # [-m]_q = -[m]_q / q^m, so valuation -m with all coefficients -1.
    m = -n
    return series.trim(series.normalise((-m, [-1] * m)), prec)


def q_int_qinv_series(n: int, prec: int) -> Series:
    """[n]_{q^{-1}} = q^{-(n-1)} [n]_q for n > 0, truncated to q^prec."""
    n = int(n)
    if n == 0:
        return 0, []
    if n > 0:
        coeffs = [1] * min(n, prec - (-(n - 1)))
        return series.trim(series.normalise((-(n - 1), coeffs)), prec)
    m = -n
    return series.scalar_mul(
        series.mul(q_int_qinv_series(m, prec), series.q_pow(m, prec), prec), -1, prec
    )


def mgo_build_series(a: list[int], prec: int) -> Series:
    """Evaluate the even-length MGO continued fraction over the series kernel.

    Odd positions (1-indexed) carry [a_i]_q with q^{a_i} above; even positions
    carry [a_i]_{q^{-1}} with q^{-a_i} above. The recursion folds from the
    innermost term outward.
    """
    n = len(a)
    if n == 0:
        return 0, []

    def term(i: int, ai: int) -> Series:
        return (
            q_int_series(ai, prec) if (i + 1) % 2 == 1 else q_int_qinv_series(ai, prec)
        )

    def num_above(i: int, ai: int) -> Series:
        return series.q_pow(ai if (i + 1) % 2 == 1 else -ai, prec)

    result = term(n - 1, a[n - 1])
    for i in range(n - 2, -1, -1):
        inv = series.invert(result, prec)
        result = series.add(
            term(i, a[i]), series.mul(num_above(i, a[i]), inv, prec), prec
        )
    return result


def q_real_truncated(x_repr: str, N: int) -> list[int]:
    """First N stable Taylor coefficients of [x]_q.

    Args:
        x_repr: a sympy-parseable string, e.g. "pi", "sqrt(2)",
            "(1+sqrt(5))/2", "E", "3/2".
        N: number of stable coefficients required, per MGO Proposition 1.1.

    Returns:
        A list of N integers [c_0, c_1, ..., c_{N-1}], where c_k is the
        coefficient of q^k.

    Raises:
        ValueError: if x < 0. The truncated path returns the coefficients of
            q^0, q^1, ..., but [x]_q for x < 0 is a Laurent series carrying
            negative powers of q (for an integer, [-m]_q = -[m]_q / q^m), so
            its q^0.. coefficients are all zero and carry no information here.
            Use the q-integer path for [n]_q or the q-negation path for [-x]_q.
    """
    if sp.sympify(x_repr).is_negative:
        raise ValueError(
            f"[x]_q is built here for x >= 0; for x = {x_repr} < 0 the series "
            "lives in negative powers of q. Use [n]_q (q-integer) or [-x]_q "
            "(q-negation) instead"
        )
    a = cf_partials(x_repr, N)
    if len(a) == 1 and a[0] in (0, 1):
        # [0]_q = 0 and [1]_q = 1 are constants whose single-quotient CF sits
        # outside the even-length normalisation. Integers >= 2 split normally.
        out = [0] * N
        if a[0] == 1 and N > 0:
            out[0] = 1
        return out
    a = make_even_length(a)
    prec = N + 5
    v, coeffs = mgo_build_series(a, prec)
    out = [0] * N
    for k in range(N):
        idx = k - v
        if 0 <= idx < len(coeffs):
            out[k] = int(coeffs[idx])
    return out
