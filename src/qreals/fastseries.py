"""Fast truncated Taylor coefficients of [x]_q from an even-length CF word.

Historical home of the multiplication-only MGO fold. The implementation now
lives in `continuant.continuant_fast`, the fast backend of the shared fold
primitive: the recurrence is folded as a numerator/denominator pair of
truncated Laurent polynomials,

    term + above / (N/D) = (term * N + above * D) / N,

with ONE long division at the end, instead of the one-inversion-per-term
fold of `truncated.mgo_build_series` (O(n_terms * prec^2)).

The CF-word convention is the module-docstring recurrence of `continuant`:
a non-empty even-length list of partial quotients where odd 1-indexed
positions carry [a]_q with q^a above and even positions carry [a]_{q^{-1}}
with q^{-a} above. Words the fast fold does not cover raise ArithmeticError
and the caller falls back to the exact inversion path in `truncated`; exact
agreement between the two paths is pinned by the test suite.
"""

from __future__ import annotations

from .continuant import LPoly as LPoly
from .continuant import continuant_fast


def mgo_fast_coefficients(terms: list[int], prec: int) -> list[int]:
    """First prec Taylor coefficients of [x]_q for the even-length CF `terms`.

    Multiplication-only fold with a single long division at the end. Raises
    ArithmeticError on any convention or drift violation (odd or empty word,
    non-positive partial quotient, quotient valuation or leading coefficient
    off the expected 1 + O(q) shape); callers fall back to
    `truncated.mgo_build_series`, which handles the general case. Delegates
    to `continuant.continuant_fast`.
    """
    return continuant_fast(terms, prec)
