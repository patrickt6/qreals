"""Small read-outs over a coefficient list produced by q_real_truncated.

These answer the questions that come up when looking for patterns in the
Taylor coefficients of a q-real: where does the first nonzero term sit, where
does the first sign change happen, how large do the coefficients get.
"""

from __future__ import annotations

from collections.abc import Iterable


def first_nonzero_coefficient_index(coeffs: Iterable[int]) -> int:
    for i, c in enumerate(coeffs):
        if c != 0:
            return i
    return -1


def first_negative_coefficient_index(coeffs: Iterable[int]) -> int:
    for i, c in enumerate(coeffs):
        if c < 0:
            return i
    return -1


def coefficient_max_abs(coeffs: Iterable[int]) -> int:
    return max((abs(c) for c in coeffs), default=0)


def number_of_zeros(coeffs: Iterable[int]) -> int:
    return sum(1 for c in coeffs if c == 0)
