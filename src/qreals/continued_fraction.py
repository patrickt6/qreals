"""Continued-fraction utilities for the MGO construction.

Two jobs. First, produce enough partial quotients of a real x that their sum
clears the stopping bound from MGO Proposition 1.1. Second, rewrite any
regular continued fraction into the even-length form the MGO formula expects.

The MGO formula is stated for an even-length regular CF [a_1, ..., a_{2m}].
A regular CF coming out of the Euclidean algorithm can have odd length, so it
is normalised by at most one of two moves: split a final entry >= 2 as
[..., a_k - 1, 1], or absorb a trailing 1 as [..., a_{k-1} + 1]. For the
Euclidean output of r/s > 1 in lowest terms the last entry is always >= 2, so
in practice only the split move fires.
"""

from __future__ import annotations

import sympy as sp

from ._parsing import parse_real


def cf_partials(x_repr: str, max_sum: int, max_depth: int = 500) -> list[int]:
    """Partial quotients of x = sympify(x_repr) until their sum exceeds max_sum.

    MGO Proposition 1.1: stopping the CF at the first depth n where the
    partial-quotient sum S_n = a_1 + ... + a_n satisfies S_n >= N + 1 fixes
    exactly S_n - 1 stable power-series coefficients of [x]_q, which is at
    least N. Calling with max_sum = N realises that bound.
    """
    x = parse_real(x_repr)
    out: list[int] = []
    total = 0
    for k, ai in enumerate(sp.continued_fraction_iterator(x)):
        out.append(int(ai))
        total += int(ai)
        if total >= max_sum + 1:
            return out
        if k >= max_depth:
            break
    return out


def make_even_length(a: list[int]) -> list[int]:
    """Return the even-length regular CF representative of a."""
    a = list(a)
    if len(a) % 2 == 0:
        return a
    if len(a) == 1:
        # A single term [a0] is the integer a0; [a0 - 1, 1] has the same value
        # (a0 - 1 + 1/1 = a0) and is even-length, for any a0 including 0, 1 and
        # negatives. Without this the integers and unit fractions whose CF is a
        # single quotient (1 = [1], 0 = [0], -3 = [-3]) had no even-length form.
        return [a[0] - 1, 1]
    if a[-1] >= 2:
        a[-1] -= 1
        a.append(1)
        return a
    if a[-1] == 1:
        a.pop()
        a[-1] += 1
        return a
    raise ValueError(f"cannot make even-length CF from {a!r}")
