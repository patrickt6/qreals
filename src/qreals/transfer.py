r"""Arithmetic on quadratic irrationals via continued-fraction transfer matrices.

This is the classical (un-deformed, q = 1) route described in the project's
worked example "sum of golden and silver ratio". For two quadratic irrationals
x and y it builds the 4x4 transfer matrix

    K = M_x (x) M_y,

where M_x is the 2x2 continuant matrix of one period of the continued fraction
of x (the q = 1 specialisation of the MGO ``q_block``), and ``(x)`` is the
Kronecker product in the monomial basis (XY, X, Y, 1). K is the matrix whose
powers the algorithm iterates; its dominant (Perron) eigenvalue has an
eigenvector proportional to (t_x t_y, t_x, t_y, 1), where t_x, t_y are the
purely-periodic continued-fraction tails. Folding back any pre-period of x and
y by the finite homographies A_x, A_y turns that into the vector

    V = (A_x (x) A_y) . v_dom   propto  (xy, x, y, 1),

and the bilinear operation z(x, y) = (a xy + b x + c y + d)/(e xy + f x + g y + h)
is read off directly as

    z = (a V_1 + b V_2 + c V_3 + d V_4) / (e V_1 + f V_2 + g V_3 + h V_4).

For x = phi = [1; 1, 1, ...] and y = delta = [2; 2, 2, ...] and op = "add" this
returns the exact (3 + 2 sqrt(2) + sqrt(5)) / 2.

Every closed form is cross-checked against direct sympy arithmetic on x and y
(the ``verified`` flag), so the matrix route is never trusted on its own.

This module is the q = 1 sibling of ``gosper.py`` and reuses its ``kron``. It is
the scaffolding the q-deformed extension (the open "which eigenvalue dominates"
problem) will build on.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.ntheory.continued_fraction import continued_fraction_periodic

from ._parsing import parse_real
from .gosper import kron

# Bilinear operation read-out covectors in the monomial basis (XY, X, Y, 1).
# value = (num . V) / (den . V) for the folded dominant eigenvector V.
_OPS: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "add": ((0, 1, 1, 0), (0, 0, 0, 1)),   # x + y
    "sub": ((0, 1, -1, 0), (0, 0, 0, 1)),  # x - y
    "mul": ((1, 0, 0, 0), (0, 0, 0, 1)),   # x * y
    "div": ((0, 1, 0, 0), (0, 0, 1, 0)),   # x / y
}

_OP_SYMBOL = {"add": "+", "sub": "-", "mul": r"\cdot", "div": "/"}


@dataclass
class QuadArith:
    """The result of one quadratic-irrational operation x op y.

    Attributes:
        x, y: the inputs as sympy expressions.
        op: one of "add", "sub", "mul", "div".
        matrix: the 4x4 transfer matrix K = M_x (x) M_y (the period block).
        eigenvalues: the four eigenvalues of K (products of the 2x2 spectra).
        dominant_eigenvalue: the eigenvalue of K of largest magnitude.
        dominant_vector: its eigenvector (the eigenvector of K).
        value: the exact closed form of x op y, read off from the eigenvector.
        decimal: a float evaluation of value.
        verified: True iff value equals x op y computed directly in sympy.
    """

    x: sp.Expr
    y: sp.Expr
    op: str
    matrix: sp.Matrix
    eigenvalues: list[sp.Expr]
    dominant_eigenvalue: sp.Expr
    dominant_vector: sp.Matrix
    value: sp.Expr
    decimal: float
    verified: bool


def _as_expr(x: str | sp.Expr) -> sp.Expr:
    return parse_real(x) if isinstance(x, str) else x


def periodic_cf(x: str | sp.Expr) -> tuple[list[int], list[int]]:
    """Pre-period and period of the regular continued fraction of x.

    Returns (pre, period). For a purely periodic value (e.g. the golden ratio)
    pre is empty. Raises ValueError if x is not a real quadratic irrational
    (i.e. its minimal polynomial does not have degree 2, which rules out
    rationals and higher-degree algebraic numbers).
    """
    xe = _as_expr(x)
    t = sp.Symbol("_t")
    poly = sp.Poly(sp.minimal_polynomial(xe, t), t)
    if poly.degree() != 2:
        raise ValueError(
            f"{x!r} is not a quadratic irrational "
            f"(minimal polynomial has degree {poly.degree()})"
        )
    a, b, c = (int(k) for k in poly.all_coeffs())
    disc = b * b - 4 * a * c
    # x is a root of a t^2 + b t + c, so x = (-b + sqrt(disc)) / (2a); the CF of
    # (P + sqrt(D))/Q comes from sympy's periodic continued-fraction routine.
    cf = continued_fraction_periodic(-b, 2 * a, disc)
    pre: list[int] = []
    period: list[int] = []
    for element in cf:
        if isinstance(element, list):
            period = [int(k) for k in element]
        else:
            pre.append(int(element))
    if not period:
        raise ValueError(f"{x!r} has no periodic continued fraction")
    return pre, period


def continuant_matrix(cf: list[int]) -> sp.Matrix:
    """Product of the continuant blocks [[a, 1], [1, 0]] over the digits of cf.

    This is the q = 1 specialisation of the MGO q-continuant ``gosper.q_block``.
    The empty list gives the identity (an absent pre-period).
    """
    m = sp.eye(2)
    for a in cf:
        m = m * sp.Matrix([[a, 1], [1, 0]])
    return m


def combined_matrix(x: str | sp.Expr, y: str | sp.Expr) -> sp.Matrix:
    """The 4x4 transfer matrix K = M_x (x) M_y built from the CF periods of x, y."""
    _, period_x = periodic_cf(x)
    _, period_y = periodic_cf(y)
    return kron(continuant_matrix(period_x), continuant_matrix(period_y))


def _dominant_eigenpair(m: sp.Matrix) -> tuple[sp.Expr, sp.Matrix]:
    """The eigenvalue of largest magnitude of a 2x2 matrix and its eigenvector."""
    best_value: sp.Expr | None = None
    best_vector: sp.Matrix | None = None
    best_magnitude = -1.0
    for value, _multiplicity, vectors in m.eigenvects():
        magnitude = abs(complex(sp.N(value)))
        if magnitude > best_magnitude:
            best_magnitude = magnitude
            best_value = value
            best_vector = vectors[0]
    assert best_value is not None and best_vector is not None
    return best_value, best_vector


def _simplify_value(expr: sp.Expr) -> sp.Expr:
    """Rationalise denominators and denest radicals for a readable closed form."""
    return sp.sqrtdenest(sp.expand(sp.radsimp(sp.cancel(expr))))


def quad_arith(x: str | sp.Expr, y: str | sp.Expr, op: str) -> QuadArith:
    """Compute x op y for quadratic irrationals via the transfer-matrix route.

    op is one of "add", "sub", "mul", "div". Raises ValueError on an unknown op
    or a non-quadratic-irrational input.
    """
    if op not in _OPS:
        raise ValueError(f"op must be one of {sorted(_OPS)}, got {op!r}")
    xe, ye = _as_expr(x), _as_expr(y)

    pre_x, period_x = periodic_cf(xe)
    pre_y, period_y = periodic_cf(ye)
    rx = continuant_matrix(period_x)
    ry = continuant_matrix(period_y)
    ax = continuant_matrix(pre_x)
    ay = continuant_matrix(pre_y)

    matrix = kron(rx, ry)

    # Spectrum of K is every product of an Rx eigenvalue with an Ry eigenvalue.
    lam_x = [value for value, _m, _v in rx.eigenvects()]
    lam_y = [value for value, _m, _v in ry.eigenvects()]
    eigenvalues = [sp.simplify(lx * ly) for lx in lam_x for ly in lam_y]

    val_x, vec_x = _dominant_eigenpair(rx)
    val_y, vec_y = _dominant_eigenpair(ry)
    dominant_eigenvalue = sp.simplify(val_x * val_y)
    dominant_vector = kron(vec_x, vec_y)  # propto (t_x t_y, t_x, t_y, 1)

    # Fold any pre-period back in: V propto (xy, x, y, 1).
    folded = kron(ax, ay) * dominant_vector
    num_c, den_c = _OPS[op]
    numerator = sum(num_c[i] * folded[i] for i in range(4))
    denominator = sum(den_c[i] * folded[i] for i in range(4))
    value = _simplify_value(numerator / denominator)

    direct = {"add": xe + ye, "sub": xe - ye, "mul": xe * ye, "div": xe / ye}[op]
    verified = sp.simplify(value - direct) == 0
    decimal = float(sp.N(value))

    return QuadArith(
        x=xe,
        y=ye,
        op=op,
        matrix=matrix,
        eigenvalues=eigenvalues,
        dominant_eigenvalue=dominant_eigenvalue,
        dominant_vector=dominant_vector,
        value=value,
        decimal=decimal,
        verified=verified,
    )
