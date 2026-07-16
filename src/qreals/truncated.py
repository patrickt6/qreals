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

from ._parsing import parse_real
from .continuant import continuant_fast, continuant_series
from .continuant import q_int_qinv_series as q_int_qinv_series
from .continuant import q_int_series as q_int_series
from .continued_fraction import cf_partials, make_even_length
from .series import Series


def mgo_build_series(a: list[int], prec: int) -> Series:
    """Evaluate the even-length MGO continued fraction over the series kernel.

    Odd positions (1-indexed) carry [a_i]_q with q^{a_i} above; even positions
    carry [a_i]_{q^{-1}} with q^{-a_i} above. The recursion folds from the
    innermost term outward; the fold lives in `continuant.continuant_series`.
    """
    return continuant_series(a, prec)


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
    if parse_real(x_repr).is_negative:
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
    try:
        # Fast path: multiplication-only fold with one final long division,
        # `continuant.continuant_fast`. It covers CF words with a positive
        # leading partial quotient (x >= 1) and raises ArithmeticError
        # otherwise, in which case the exact inversion-per-term fold below
        # takes over.
        return continuant_fast(a, prec)[:N]
    except ArithmeticError:
        pass
    v, coeffs = continuant_series(a, prec)
    out = [0] * N
    for k in range(N):
        idx = k - v
        if 0 <= idx < len(coeffs):
            out[k] = int(coeffs[idx])
    return out
