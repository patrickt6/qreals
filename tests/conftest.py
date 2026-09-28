"""Shared test configuration and helpers.

Hypothesis runs under a fixed profile: derandomized (the same examples on
every run and every machine), no example database, and no per-example
deadline (sympy's first call in a process can be slow). Set
HYPOTHESIS_PROFILE=thorough locally for a larger, randomized search, or
HYPOTHESIS_PROFILE=mutation for the smaller run used under mutmut.
"""

from __future__ import annotations

import os
from fractions import Fraction

import sympy as sp
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from qreals import q

settings.register_profile(
    "ci",
    derandomize=True,
    database=None,
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "mutation",
    derandomize=True,
    database=None,
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "thorough",
    derandomize=False,
    database=None,
    deadline=None,
    max_examples=1000,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


def rationals(
    min_value: Fraction | int | None = None,
    max_num: int = 40,
    max_den: int = 12,
    exclude_zero: bool = True,
) -> st.SearchStrategy[Fraction]:
    """Reduced fractions p/s with |p| <= max_num and 1 <= s <= max_den."""

    def build(pair: tuple[int, int]) -> Fraction:
        return Fraction(pair[0], pair[1])

    strat = st.tuples(
        st.integers(min_value=-max_num, max_value=max_num),
        st.integers(min_value=1, max_value=max_den),
    ).map(build)
    if exclude_zero:
        strat = strat.filter(lambda x: x != 0)
    if min_value is not None:
        lo = Fraction(min_value)
        strat = strat.filter(lambda x: x > lo)
    return strat


def is_zero(expr: sp.Expr) -> bool:
    """Exact test that a rational function in q is identically zero."""
    return sp.cancel(sp.together(expr)) == 0


def taylor(expr: "sp.Expr | tuple[sp.Poly, sp.Poly]", n: int) -> list[int]:
    """The first n Taylor coefficients of a rational function at q = 0.

    Plain power-series long division with Fraction arithmetic, written here
    so it shares no code with the package's series kernel. Raises if expr
    has a pole at q = 0 or a non-integer coefficient appears.
    """
    if isinstance(expr, tuple):  # a (numerator, denominator) pair of Polys
        num, den = expr
    else:
        num, den = sp.fraction(sp.cancel(sp.together(expr)))
    nc = [Fraction(int(c)) for c in reversed(sp.Poly(num, q).all_coeffs())]
    dc = [Fraction(int(c)) for c in reversed(sp.Poly(den, q).all_coeffs())]
    shift = next(i for i, c in enumerate(dc) if c != 0)
    dc = dc[shift:]
    lead = next((i for i, c in enumerate(nc) if c != 0), len(nc))
    if lead < shift:
        raise ValueError("pole at q = 0; not a Taylor series")
    nc = nc[shift:] + [Fraction(0)] * (n + len(dc))
    out: list[int] = []
    for k in range(n):
        c = nc[k] / dc[0]
        if c.denominator != 1:
            raise ValueError("non-integer Taylor coefficient")
        out.append(int(c))
        if c:
            for j in range(1, len(dc)):
                nc[k + j] -= c * dc[j]
    return out


def poly_coeffs(expr: sp.Expr) -> list[int]:
    """Coefficients of a polynomial in q, lowest degree first."""
    return [int(c) for c in reversed(sp.Poly(sp.expand(expr), q).all_coeffs())]


def series_expr(coeffs: list[int], valuation: int = 0) -> sp.Expr:
    """The truncated series sum c_k q^(valuation + k) as a sympy expression."""
    return sum(
        (sp.Integer(c) * q ** (valuation + k) for k, c in enumerate(coeffs)),
        sp.Integer(0),
    )
