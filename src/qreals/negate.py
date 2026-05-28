r"""The Jouteur q-negation, exposed as `qreals.negate`.

Given A = [x]_q, the q-deformed negation of Jouteur, arXiv:2503.02122 eq. (2),
is

    [-x]_q = (-A + 1 - q^{-1}) / ((q - 1) A + 1).

`arithmetic.q_neg(x, N)` runs this for x >= 0 and returns the result as a
Laurent (valuation, coefficients) pair, because [-x]_q carries negative powers
of q for positive x. `negate` is the same construction extended to accept any
real x: for x < 0 the input series A is itself a Jouteur image of [|x|]_q, and
applying the formula a second time returns the q-real of -x = |x| (the
involution [-(-y)]_q = [y]_q).

The Jouteur PGL_2(Z) action on q-reals is the route Ovsienko's Example 6.4
uses to ask when [x]_q + [-x]_q is a finite Laurent polynomial; the criterion
(pure square root) is checked in `arithmetic.finite_xnegx` and through the
catalogue scan in `q_gosper.negation_finiteness` in the wider research code.
"""

from __future__ import annotations

import sympy as sp

from . import series
from .arithmetic import _BUFFER, _jouteur_neg, _pad
from .series import Series
from .truncated import q_real_truncated

LaurentCoeffs = tuple[int, list[int]]


def _qreal_laurent_series(x_repr: str, prec: int) -> Series:
    """[x]_q as a kernel Laurent series, valid for any real x.

    For x >= 0 this is the verified `q_real_truncated` path with valuation 0.
    For x < 0 the q-real is the Jouteur image of [|x|]_q (negation eq. 2 of
    arXiv:2503.02122 maps positive q-reals to their negative counterparts),
    so the same single application of the formula builds it.
    """
    val = sp.sympify(x_repr)
    if val.is_negative:
        pos = _qreal_laurent_series(str(-val), prec)
        return _jouteur_neg(pos, prec)
    return series.normalise((0, q_real_truncated(x_repr, prec)))


def negate(x: str, N: int) -> LaurentCoeffs:
    """The Jouteur q-negation [-x]_q for any real x, as (valuation, N coeffs).

    Args:
        x: a sympy-parseable string, e.g. "pi", "sqrt(2)", "3/2", "-3/2".
        N: the number of Laurent coefficients to return, starting at q^v.

    Returns:
        A pair (v, [c_v, ..., c_{v+N-1}]), where v is the valuation of the
        Laurent series [-x]_q and the list is the next N coefficients.

    Reference:
        A. Jouteur, "Modular group action on q-deformed real numbers",
        arXiv:2503.02122, eq. (2).
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    prec = N + _BUFFER
    a_series = _qreal_laurent_series(x, prec)
    v, c = _jouteur_neg(a_series, prec)
    return v, _pad(c, N)
