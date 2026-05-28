r"""Arithmetic between q-reals: series sum, series product, and q-negation.

The stable core here works on coefficient lists over the truncated-series kernel
in `series`, so it stays pure Python (sympy is used only to read the continued
fraction of x, through `q_real_truncated`). Three things live here, with one
caveat each, all spelled out in docs/CORRECTNESS.md:

- `q_add(x, y, N)` and `q_mul(x, y, N)` return the first N Taylor coefficients of
  the series sum [x]_q + [y]_q and the series product [x]_q * [y]_q. These are
  the sum and product of the two q-series, NOT [x+y]_q or [x*y]_q: the MGO map
  x |-> [x]_q is not a ring homomorphism. `gosper` reaches the same two
  quantities by a different algorithm, which the tests use as a cross-check.

- `q_neg(x, N)` returns the q-deformed negation [-x]_q of Jouteur
  (arXiv:2503.02122, eq. 2), as a Laurent result (valuation, coeffs). This is
  the PGL_2(Z)-action negation, which is NOT the coefficient-wise negation of
  [x]_q and NOT the MGO series of the real number -x. It is an involution, and
  `negation_sum` / `finite_xnegx` use it to study Ovsienko's Example 6.4 (for
  which real x is [x]_q + [-x]_q a finite Laurent polynomial).

- `radius(x, N)` estimates the radius of convergence of the power series [x]_q
  from its first N coefficients, by the running-max root-test slope. The
  finite-N value is biased high; see its docstring and the docs.
"""

from __future__ import annotations

import math

from . import series
from .series import Series
from .truncated import q_real_truncated

# A Laurent result: the valuation v and the dense coefficient list starting at
# q^v. q_neg and negation_sum return this, since [-x]_q has negative powers.
LaurentCoeffs = tuple[int, list[int]]

# How many extra coefficients to carry internally so that division (which mixes
# in the highest input terms) leaves N reliable coefficients at the end.
_BUFFER = 12


def _require_nonnegative(x_repr: str) -> None:
    import sympy as sp

    value = sp.sympify(x_repr)
    if value.is_negative:
        raise ValueError(
            f"x must be >= 0 (the truncated [x]_q path is built for x >= 0); got {x_repr}"
        )


def _qreal_series(x_repr: str, prec: int) -> Series:
    """[x]_q for x >= 0 as a kernel Series (valuation, coeffs), exact to prec."""
    _require_nonnegative(x_repr)
    return series.normalise((0, q_real_truncated(x_repr, prec)))


def _pad(coeffs: list[int], n: int) -> list[int]:
    """Right-pad with zeros (or truncate) to exactly n entries."""
    if len(coeffs) >= n:
        return coeffs[:n]
    return coeffs + [0] * (n - len(coeffs))


# ----------------------------------------------------------------------------
# series sum and product:  [x]_q + [y]_q  and  [x]_q * [y]_q
# ----------------------------------------------------------------------------
def q_add(x: str, y: str, N: int) -> list[int]:
    """First N Taylor coefficients of the series sum [x]_q + [y]_q (x, y >= 0).

    This is the coefficient-wise sum of the two q-series. It is verified against
    the bihomographic `gosper` engine, an independent algorithm, on rationals.
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    cx = q_real_truncated(x, N)
    cy = q_real_truncated(y, N)
    return [a + b for a, b in zip(cx, cy)]


def q_mul(x: str, y: str, N: int) -> list[int]:
    """First N Taylor coefficients of the series product [x]_q * [y]_q (x, y >= 0).

    This is the Cauchy product (convolution) of the two q-series, cross-checked
    against the `gosper` engine's "mul" value on rationals.
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    cx = q_real_truncated(x, N)
    cy = q_real_truncated(y, N)
    out = [0] * N
    for i, a in enumerate(cx):
        if a == 0:
            continue
        for j, b in enumerate(cy):
            if i + j >= N:
                break
            out[i + j] += a * b
    return out


# ----------------------------------------------------------------------------
# q-negation (Jouteur) and the x -> -x symmetry (Ovsienko Example 6.4)
# ----------------------------------------------------------------------------
def _jouteur_neg(A: Series, prec: int) -> Series:
    """[-x]_q from A = [x]_q via Jouteur eq. (2), over the series kernel.

    [-x]_q = (-A + 1 - q^{-1}) / ((q - 1) A + 1).
    """
    qA = series.mul((1, [1]), A, prec)  # q * A
    qm1A = series.add(qA, series.scalar_mul(A, -1, prec), prec)  # (q - 1) A
    den = series.add_int(qm1A, 1, prec)  # (q - 1) A + 1
    num = series.add(series.scalar_mul(A, -1, prec), (0, [1]), prec)  # -A + 1
    num = series.add(num, (-1, [-1]), prec)  # -A + 1 - q^{-1}
    return series.mul(num, series.invert(den, prec), prec)


def q_neg(x: str, N: int) -> LaurentCoeffs:
    """The Jouteur q-negation [-x]_q (x >= 0) as (valuation, N coefficients).

    [-x]_q is a Laurent series (it carries negative powers of q), so the result
    is returned as a valuation together with the coefficient list from q^v up.
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    prec = N + _BUFFER
    neg = _jouteur_neg(_qreal_series(x, prec), prec)
    v, c = neg
    return v, _pad(c, N)


def negation_sum(x: str, N: int) -> LaurentCoeffs:
    """[x]_q + [-x]_q (x >= 0) as (valuation, N coefficients), Ovsienko Ex. 6.4."""
    if N < 1:
        raise ValueError("N must be at least 1")
    prec = N + _BUFFER
    A = _qreal_series(x, prec)
    total = series.add(A, _jouteur_neg(A, prec), prec)
    v, c = total
    return v, _pad(c, N)


def finite_xnegx(x: str, order: int = 48) -> bool:
    """Does [x]_q + [-x]_q terminate as a finite Laurent polynomial? (Ex. 6.4)

    Computed numerically to the given order: the sum is reported finite when its
    coefficients past the leading block are a long run of zeros. The proven
    criterion (finite iff x is a trace-zero quadratic, i.e. a pure square root)
    and the closed identity behind it are in docs/CORRECTNESS.md; this is the
    operational check, honest about being a finite-order observation.
    """
    if order < 8:
        raise ValueError("order must be at least 8 to judge termination")
    _, coeffs = negation_sum(x, order)
    last_nonzero = -1
    for i, c in enumerate(coeffs):
        if c != 0:
            last_nonzero = i
    trailing_zeros = (len(coeffs) - 1) - last_nonzero
    # A terminating Laurent polynomial leaves a long zero tail in this window; a
    # genuine infinite series keeps producing nonzero coefficients near the top.
    return trailing_zeros >= order // 2


# ----------------------------------------------------------------------------
# radius of convergence estimate
# ----------------------------------------------------------------------------
def radius(x: str, N: int) -> float:
    """Running-max root-test estimate of the radius of convergence of [x]_q.

    The Cauchy-Hadamard radius is R = 1 / limsup_k |c_k|^{1/k}. This returns the
    finite-N reciprocal of the running maximum of |c_k|^{1/k} over 1 <= k < N,
    i.e. exp(-max_k (ln|c_k|) / k). Returns +inf only when no coefficient past
    the constant term is nonzero in the window (x = 0 or x = 1).

    Finite-N bias, stated honestly: the running maximum over a finite window is
    at most the true limsup, so this estimate is at least the true radius. It is
    biased high and decreases toward R from above as N grows. For an integer x,
    where [x]_q is a polynomial (true radius infinite), the low-order unit
    coefficients pin the estimate near 1 rather than revealing the infinite
    radius; that saturation is itself a face of the finite-N bias.
    """
    if N < 2:
        raise ValueError("N must be at least 2 to estimate a slope")
    coeffs = q_real_truncated(x, N)
    max_slope: float | None = None
    for k in range(1, len(coeffs)):
        c = coeffs[k]
        if c == 0:
            continue
        slope = math.log(abs(c)) / k
        if max_slope is None or slope > max_slope:
            max_slope = slope
    if max_slope is None:
        return math.inf
    return math.exp(-max_slope)
