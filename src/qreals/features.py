"""A named, fixed-length fingerprint of a real number's q-analog.

`featurize(x)` turns a real number x into a vector of named, deterministic
features drawn from two sources: the regular continued fraction of x (which
drives the MGO construction) and the integer Taylor coefficients of its q-analog
[x]_q. The vector is fixed length for a given set of parameters, so two
constants are directly comparable: the point is exploration and nearest-neighbour
search over constants, not training a model.

The output is a :class:`Fingerprint`: a parallel pair of `names` and `values`
(plain floats), with `as_dict()` for a name -> value mapping and `as_numpy()`
for a numpy array when numpy is installed. No numpy is needed to build a
fingerprint or to compare two of them; numpy is the only optional dependency,
behind the ``qreals[features]`` extra, and is used only by `as_numpy`. There is
no torch and no model here.

The features, in the order they appear in the vector:

Scalar features (16):

- ``valuation``: index of the first nonzero coefficient of [x]_q. 0 for x >= 1,
  1 for 0 < x < 1.
- ``n_zeros``: how many of the first ``n_coeffs`` coefficients are zero.
- ``longest_zero_run``: length of the longest run of consecutive zero
  coefficients in that window.
- ``first_negative_index``: index of the first negative coefficient, or
  ``n_coeffs`` if none is negative in the window.
- ``n_sign_changes``: sign changes between consecutive nonzero coefficients.
- ``n_sign_runs``: number of maximal same-sign runs over the nonzero
  coefficients (``n_sign_changes + 1``, or 0 when all coefficients are zero).
- ``longest_sign_run``: the longest such run.
- ``max_abs_coeff``: the largest |c_k| in the window.
- ``mean_abs_coeff``: the mean of |c_k| over the window.
- ``log10_max_abs_coeff``: ``log10(1 + max_abs_coeff)``, a scale-insensitive
  measure of how big the coefficients get.
- ``log10_mean_abs_coeff``: ``log10(1 + mean_abs_coeff)``.
- ``inv_radius``: the running-max root-test slope ``max_k (ln|c_k|)/k`` over the
  first ``n_radius`` coefficients. This is 1 / (radius-of-convergence estimate);
  it is 0 exactly when [x]_q is a polynomial (x a nonnegative integer), and
  larger when the coefficients grow faster. Always finite, unlike the radius
  itself, which is what makes it usable as a coordinate.
- ``cf_len``: number of regular continued-fraction terms actually available, up
  to ``n_cf`` (a terminating, i.e. rational, expansion shows up as cf_len <
  n_cf).
- ``cf_max``: the largest partial quotient among the first ``n_cf`` terms.
- ``cf_sum``: the sum of the first ``n_cf`` partial quotients (its growth sets
  how fast coefficients lock in, MGO Proposition 1.1).

Block features:

- ``cf_0 .. cf_{n_cf-1}``: the partial quotients (continued-fraction terms) of
  x, padded with 0 past a terminating expansion. ``cf_0`` is floor(x).
- ``cf_partial_sum_1 .. cf_partial_sum_{n_cf}``: cumulative sums of those
  partial quotients. A plateau marks where a rational expansion terminated.
- ``c_0 .. c_{n_coeffs-1}``: the signed integer Taylor coefficients of [x]_q,
  the most direct fingerprint of the series. These can grow large for some
  constants, so for scale-insensitive nearest-neighbour either normalise the
  vector or lean on the log-magnitude and shape scalars above.

All features are derived from exact integer coefficients and the exact symbolic
continued fraction, so `featurize` is deterministic: the same x and parameters
give the same vector every time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import sympy as sp

from ._parsing import parse_real
from .truncated import q_real_truncated

DEFAULT_N_CF = 8
DEFAULT_N_COEFFS = 24
DEFAULT_N_RADIUS = 64


@dataclass(frozen=True)
class Fingerprint:
    """A named, fixed-length feature vector for one real number.

    ``names`` and ``values`` are parallel lists of the same length. ``x`` is the
    input as given, and ``params`` records ``(n_cf, n_coeffs, n_radius)`` so two
    fingerprints are only comparable when their params agree.
    """

    x: str
    names: list[str]
    values: list[float]
    params: tuple[int, int, int]

    @property
    def vector(self) -> list[float]:
        """The feature values, in the fixed order given by ``names``."""
        return self.values

    def as_dict(self) -> dict[str, float]:
        """A name -> value mapping of the fingerprint."""
        return dict(zip(self.names, self.values))

    def as_numpy(self) -> Any:
        """The values as a numpy float array (needs ``qreals[features]``)."""
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised when numpy absent
            raise ImportError(
                "as_numpy needs numpy; install it with pip install qreals[features]"
            ) from exc
        return np.asarray(self.values, dtype=float)


def _parse_real(x_repr: str) -> sp.Expr:
    value = parse_real(x_repr)
    if value.is_real is not True:
        raise ValueError(f"x must be a real number, got {x_repr!r}")
    if value.is_finite is False:
        raise ValueError(f"x must be finite, got {x_repr!r}")
    return value


def _cf_terms(value: sp.Expr, count: int) -> list[int]:
    """First ``count`` regular continued-fraction terms of x (fewer if it ends)."""
    terms: list[int] = []
    for term in sp.continued_fraction_iterator(value):
        terms.append(int(term))
        if len(terms) >= count:
            break
    return terms


def _sign_run_stats(coeffs: Sequence[int]) -> tuple[int, int, int]:
    """(sign changes, number of sign runs, longest sign run) over nonzero terms."""
    signs = [1 if c > 0 else -1 for c in coeffs if c != 0]
    if not signs:
        return 0, 0, 0
    changes = 0
    longest = 1
    current = 1
    for prev, cur in zip(signs, signs[1:]):
        if cur != prev:
            changes += 1
            current = 1
        else:
            current += 1
        longest = max(longest, current)
    return changes, changes + 1, longest


def _longest_zero_run(coeffs: Sequence[int]) -> int:
    longest = 0
    current = 0
    for c in coeffs:
        if c == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _inv_radius(coeffs: Sequence[int]) -> float:
    """Running-max root-test slope max_k (ln|c_k|)/k; 0 when [x]_q is a polynomial.

    This is the reciprocal of the radius-of-convergence estimate in `arithmetic`,
    kept in this form so it is always a finite number (the radius itself is
    infinite for integer x). Same finite-N bias caveat applies, the other way up.
    """
    best: float | None = None
    for k in range(1, len(coeffs)):
        c = coeffs[k]
        if c == 0:
            continue
        slope = math.log(abs(c)) / k
        if best is None or slope > best:
            best = slope
    return 0.0 if best is None else best


def feature_names(
    n_cf: int = DEFAULT_N_CF,
    n_coeffs: int = DEFAULT_N_COEFFS,
    n_radius: int = DEFAULT_N_RADIUS,
) -> list[str]:
    """The fixed feature names, in vector order, for the given parameters."""
    scalars = [
        "valuation",
        "n_zeros",
        "longest_zero_run",
        "first_negative_index",
        "n_sign_changes",
        "n_sign_runs",
        "longest_sign_run",
        "max_abs_coeff",
        "mean_abs_coeff",
        "log10_max_abs_coeff",
        "log10_mean_abs_coeff",
        "inv_radius",
        "cf_len",
        "cf_max",
        "cf_sum",
    ]
    cf = [f"cf_{i}" for i in range(n_cf)]
    cf_sums = [f"cf_partial_sum_{i}" for i in range(1, n_cf + 1)]
    coeffs = [f"c_{i}" for i in range(n_coeffs)]
    return scalars + cf + cf_sums + coeffs


def featurize(
    x: str | int | float | sp.Expr,
    *,
    n_cf: int = DEFAULT_N_CF,
    n_coeffs: int = DEFAULT_N_COEFFS,
    n_radius: int = DEFAULT_N_RADIUS,
) -> Fingerprint:
    """A named, fixed-length, deterministic fingerprint of [x]_q.

    Args:
        x: a sympy-parseable real, e.g. "pi", "sqrt(2)", "(1+sqrt(5))/2", "3/2".
        n_cf: how many continued-fraction terms (and their partial sums) to keep.
        n_coeffs: how many signed Taylor coefficients of [x]_q to keep.
        n_radius: window for the ``inv_radius`` slope feature; must be >= 2.

    Returns:
        A :class:`Fingerprint`. The names and length depend only on the three
        parameters, so fingerprints built with the same parameters are directly
        comparable (see :func:`feature_distance`, :func:`nearest`).

    The math each feature reads is documented at the top of this module.
    """
    if n_cf < 1:
        raise ValueError("n_cf must be at least 1")
    if n_coeffs < 1:
        raise ValueError("n_coeffs must be at least 1")
    if n_radius < 2:
        raise ValueError("n_radius must be at least 2")

    x_repr = str(x)
    value = _parse_real(x_repr)

    depth = max(n_coeffs, n_radius)
    coeffs = q_real_truncated(x_repr, depth)
    window = coeffs[:n_coeffs]

    valuation = next((i for i, c in enumerate(window) if c != 0), -1)
    n_zeros = sum(1 for c in window if c == 0)
    longest_zero_run = _longest_zero_run(window)
    first_negative = next((i for i, c in enumerate(window) if c < 0), None)
    first_negative_index = float(n_coeffs if first_negative is None else first_negative)
    n_sign_changes, n_sign_runs, longest_sign_run = _sign_run_stats(window)
    abs_coeffs = [abs(c) for c in window]
    max_abs = max(abs_coeffs) if abs_coeffs else 0
    mean_abs = sum(abs_coeffs) / len(abs_coeffs) if abs_coeffs else 0.0
    inv_radius = _inv_radius(coeffs[:n_radius])

    cf = _cf_terms(value, n_cf)
    cf_len = len(cf)
    cf_padded = cf + [0] * (n_cf - cf_len)
    cf_max = max(cf_padded) if cf_padded else 0
    cf_sum = sum(cf_padded)
    cf_partial_sums: list[int] = []
    running = 0
    for term in cf_padded:
        running += term
        cf_partial_sums.append(running)

    scalar_values = [
        float(valuation),
        float(n_zeros),
        float(longest_zero_run),
        first_negative_index,
        float(n_sign_changes),
        float(n_sign_runs),
        float(longest_sign_run),
        float(max_abs),
        float(mean_abs),
        math.log10(1.0 + max_abs),
        math.log10(1.0 + mean_abs),
        float(inv_radius),
        float(cf_len),
        float(cf_max),
        float(cf_sum),
    ]
    values = (
        scalar_values
        + [float(t) for t in cf_padded]
        + [float(s) for s in cf_partial_sums]
        + [float(c) for c in window]
    )
    names = feature_names(n_cf, n_coeffs, n_radius)
    assert len(names) == len(values)  # the fixed-length contract
    return Fingerprint(
        x=x_repr, names=names, values=values, params=(n_cf, n_coeffs, n_radius)
    )


def feature_distance(a: Fingerprint, b: Fingerprint) -> float:
    """Euclidean distance between two fingerprints (same parameters required).

    The raw coefficient features can dominate this distance for fast-growing
    constants; normalise the vectors first if you want a shape-based comparison.
    """
    if a.params != b.params:
        raise ValueError(
            f"fingerprints use different parameters {a.params} vs {b.params}; "
            "compare only fingerprints built the same way"
        )
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a.values, b.values)))


def nearest(
    target: Fingerprint, pool: Sequence[Fingerprint], k: int = 1
) -> list[tuple[Fingerprint, float]]:
    """The ``k`` fingerprints in ``pool`` closest to ``target`` by Euclidean distance.

    Returns ``(fingerprint, distance)`` pairs sorted nearest first. A small helper
    for nearest-neighbour exploration over a set of constants.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    scored = [(fp, feature_distance(target, fp)) for fp in pool]
    scored.sort(key=lambda pair: pair[1])
    return scored[:k]
