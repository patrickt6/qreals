r"""Factor the numerator R(q) and denominator S(q) of a q-rational [a/b]_q.

An open question concerns the shape of the numerator R(q)
of [a/b]_q as an element of Z[q]: when is R(q) irreducible (the worked
observation is that R is often irreducible when a is prime), and how does the
cyclotomic-style factorisation seen in the q-integer case [m/1]_q = [m]_q
extend to a general fraction? This module answers both by factoring R and S
exactly over Z[q] with sympy.factor_list and classifying each irreducible
factor as a cyclotomic polynomial Phi_d(q) or a non-cyclotomic "core" factor.

For an integer m, [m/1]_q = [m]_q = 1 + q + ... + q^{m-1} factors as the
product of Phi_d(q) over the divisors d | m with d > 1 (the classical
identity used by qint_factor). For a general fraction the numerator can carry
a non-cyclotomic core, and that core is exactly what makes R irreducible when
a is prime in the cases observed.

Here [a/b]_q = q^k R(q)/S(q) with R(0) = 1. The result
records k, the power of q split off the raw numerator to normalise R(0) = 1,
so the convention is explicit. For many fractions k is 0, but it can be
positive (for example [3/5]_q has k = 1), so the field carries real
information. Everything here is exact: sympy over Z[q], no floating point.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from ._parsing import parse_real
from .rational import q, q_rational


@dataclass(frozen=True)
class QRealFactor:
    """The structured factorisation of [a/b]_q over Z[q].

    Fields:
        a, b: the input fraction a/b in lowest terms (b > 0).
        k: the power of q split off the raw numerator so the normalised R has
            R(0) = 1, matching [a/b]_q = q^k R(q)/S(q). Often 0, but can be
            positive (for example [3/5]_q has k = 1).
        content_R: the integer content of the numerator (the leading rational
            coefficient pulled out by factor_list, kept as a sympy Integer or
            Rational so the round-trip is exact).
        factors_R: the irreducible factors of R as (factor, multiplicity)
            pairs, each factor a primitive sympy Expr in q.
        cyclotomic_R: the cyclotomic support of R, a dict d -> e_d giving the
            exponent of Phi_d(q) (d >= 1) among the factors of R.
        core_R: the non-cyclotomic core factors of R as (factor, multiplicity)
            pairs; empty when R is a pure product of cyclotomic polynomials.
        content_S, factors_S, cyclotomic_S, core_S: the same data for the
            denominator S(q).
        is_irreducible_R: True when R has exactly one irreducible factor of
            multiplicity one (ignoring integer content), i.e. R is irreducible
            over Z[q] up to a unit.
        is_pure_cyclotomic_R: True when R is a product of cyclotomic factors
            only, with no non-cyclotomic core.
    """

    a: int
    b: int
    k: int
    content_R: sp.Expr
    factors_R: list[tuple[sp.Expr, int]]
    cyclotomic_R: dict[int, int]
    core_R: list[tuple[sp.Expr, int]]
    content_S: sp.Expr
    factors_S: list[tuple[sp.Expr, int]]
    cyclotomic_S: dict[int, int]
    core_S: list[tuple[sp.Expr, int]]
    is_irreducible_R: bool
    is_pure_cyclotomic_R: bool


def _coerce_fraction(x: Fraction | str | tuple[int, int] | list[int]) -> Fraction:
    """Read a/b from a Fraction, a string 'a/b', or an (a, b) pair."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str):
        text = x.strip()
        try:
            return Fraction(text)
        except ValueError:
            # The web UI's math editor sends engine-style rationals such as
            # "(7)/(5)" that Fraction cannot read; parse those with sympy, the
            # same path the rest of the engine uses, and require a rational.
            value = sp.nsimplify(parse_real(text))
            if value.is_rational is not True:
                raise ValueError(
                    f"input {x!r} is not a rational a/b"
                ) from None
            r = sp.Rational(value)
            return Fraction(int(r.p), int(r.q))
    if isinstance(x, (tuple, list)):
        if len(x) != 2:
            raise ValueError("a pair input must be (a, b)")
        a, b = x
        return Fraction(int(a), int(b))
    raise TypeError(
        f"cannot read a rational from {x!r}; pass Fraction, 'a/b', or (a, b)"
    )


def _cyclotomic_index(factor: sp.Expr) -> int | None:
    """Return d if factor equals Phi_d(q) for some d >= 1, else None.

    A monic irreducible factor of a polynomial over Z[q] is cyclotomic exactly
    when it equals cyclotomic_poly(d, q); its degree is the Euler totient
    phi(d), so only d with phi(d) == deg(factor) need to be tested.
    """
    poly = sp.Poly(sp.expand(factor), q)
    deg = poly.degree()
    if deg < 1:
        return None
    target = poly.as_expr()
    # phi(d) = deg bounds d: only the d whose totient matches the degree can
    # match, and d <= 2 * deg^2 + 1 is a safe span for those.
    for d in range(1, 2 * deg * deg + 2):
        if sp.totient(d) != deg:
            continue
        if sp.expand(sp.cyclotomic_poly(d, q) - target) == 0:
            return d
    return None


def _factor_poly(
    expr: sp.Expr,
) -> tuple[
    sp.Expr,
    list[tuple[sp.Expr, int]],
    dict[int, int],
    list[tuple[sp.Expr, int]],
]:
    """Factor a polynomial in q over Z and classify each irreducible factor.

    Returns (content, factors, cyclotomic, core) where content is the integer
    content, factors is the list of (irreducible factor, multiplicity) pairs,
    cyclotomic maps d -> e_d over the factors that equal Phi_d(q), and core is
    the list of (factor, multiplicity) pairs that are not cyclotomic.
    """
    content, factor_pairs = sp.factor_list(sp.expand(expr), q)
    factors: list[tuple[sp.Expr, int]] = []
    cyclotomic: dict[int, int] = {}
    core: list[tuple[sp.Expr, int]] = []
    for fac, mult in factor_pairs:
        fac_expr = fac.as_expr() if isinstance(fac, sp.Poly) else sp.sympify(fac)
        factors.append((fac_expr, int(mult)))
        d = _cyclotomic_index(fac_expr)
        if d is not None:
            cyclotomic[d] = cyclotomic.get(d, 0) + int(mult)
        else:
            core.append((fac_expr, int(mult)))
    return sp.sympify(content), factors, cyclotomic, core


def _split_q_power(expr: sp.Expr) -> tuple[int, sp.Expr]:
    """Split q^k out of a polynomial so the remainder has a non-zero constant.

    Returns (k, remainder) with expr == q**k * remainder and remainder(0) != 0.
    For many q-rationals k is 0, but it can be positive (for example the
    numerator of [3/5]_q carries one factor of q, so k = 1).
    """
    poly = sp.Poly(sp.expand(expr), q)
    if poly.is_zero:
        return 0, sp.Integer(0)
    k = int(min(monom[0] for monom in poly.monoms()))
    if k == 0:
        return 0, poly.as_expr()
    return k, sp.expand(poly.as_expr() / q**k)


def factor_qreal(x: Fraction | str | tuple[int, int] | list[int]) -> QRealFactor:
    r"""Factor the numerator and denominator of [a/b]_q over Z[q].

    The input x is a rational given as a Fraction, a string "a/b", or an
    (a, b) pair. The value [a/b]_q is taken from q_rational (the exact MGO
    continued-fraction value); its numerator R(q) and denominator S(q) are
    each factored with sympy.factor_list and every irreducible factor is
    classified as a cyclotomic polynomial Phi_d(q) or a non-cyclotomic core
    factor. The numerator is normalised so R has R(0) = 1, matching
    [a/b]_q = q^k R(q)/S(q), with the split-off power k recorded.

    Returns a QRealFactor record carrying R's content, irreducible factors,
    cyclotomic support, and core, the same for S, and the two booleans
    is_irreducible_R and is_pure_cyclotomic_R.
    """
    frac = _coerce_fraction(x)
    a = int(frac.numerator)
    b = int(frac.denominator)

    value = sp.cancel(q_rational(a, b))
    num, den = sp.fraction(sp.together(value))
    num = sp.expand(num)
    den = sp.expand(den)

    k, num_norm = _split_q_power(num)

    content_R, factors_R, cyclotomic_R, core_R = _factor_poly(num_norm)
    content_S, factors_S, cyclotomic_S, core_S = _factor_poly(den)

    is_irreducible_R = len(factors_R) == 1 and factors_R[0][1] == 1
    is_pure_cyclotomic_R = len(core_R) == 0 and len(factors_R) > 0

    return QRealFactor(
        a=a,
        b=b,
        k=k,
        content_R=content_R,
        factors_R=factors_R,
        cyclotomic_R=cyclotomic_R,
        core_R=core_R,
        content_S=content_S,
        factors_S=factors_S,
        cyclotomic_S=cyclotomic_S,
        core_S=core_S,
        is_irreducible_R=is_irreducible_R,
        is_pure_cyclotomic_R=is_pure_cyclotomic_R,
    )


def numerator_expr(result: QRealFactor) -> sp.Expr:
    """Rebuild R(q) exactly from a QRealFactor: q^k * content * prod factors^mult.

    The round-trip identity numerator_expr(factor_qreal(x)) == R(q) holds as a
    sympy expression, so callers can verify the factorisation is exact.
    """
    out = sp.Integer(1) * result.content_R
    for fac, mult in result.factors_R:
        out = out * fac**mult
    return sp.expand(q**result.k * out)


def denominator_expr(result: QRealFactor) -> sp.Expr:
    """Rebuild S(q) exactly from a QRealFactor: content * prod factors^mult."""
    out = sp.Integer(1) * result.content_S
    for fac, mult in result.factors_S:
        out = out * fac**mult
    return sp.expand(out)


def classify_roots(result: QRealFactor) -> dict:
    """The complex roots of R(q), each tagged by its exact factor class.

    A display helper for the serve visualizer: the cyclotomic-vs-core split of
    each root comes from the exact Z[q] factorisation in `result`, not from
    numerical proximity to the unit circle. A root is "cyclotomic" exactly when
    the irreducible factor it solves is a Phi_d(q) (so it is a primitive d-th
    root of unity and sits on the unit circle, an n-gon vertex); it is "core"
    when its factor is a non-cyclotomic core factor. The complex coordinates
    themselves are numerical (sympy nroots) and are for plotting only; the kind
    label is exact.

    Returns a dict with:
        roots: list of {re, im, kind, d, mult}, one entry per root of each
            irreducible factor (kind is "cyclotomic" or "core"; d is the
            cyclotomic index for cyclotomic factors, else None; mult is the
            multiplicity of that factor in R).
        cyclotomic_support: {str(d): e_d}, the cyclotomic support of R.
        core_degree: the total degree of the non-cyclotomic core of R.
        degree: the degree of R(q).
    """
    roots: list[dict] = []
    degree = 0
    for fac, mult in result.factors_R:
        poly = sp.Poly(sp.expand(fac), q)
        if poly.degree() < 1:
            continue
        degree += poly.degree() * mult
        d = _cyclotomic_index(fac)
        kind = "cyclotomic" if d is not None else "core"
        # one plotted point per algebraic root: nroots gives the deg(factor)
        # roots of the squarefree irreducible factor, repeated by its
        # multiplicity in R so the point count matches the degree of R.
        for r in poly.nroots():
            c = complex(r)
            for _ in range(int(mult)):
                roots.append(
                    {
                        "re": float(c.real),
                        "im": float(c.imag),
                        "kind": kind,
                        "d": d,
                        "mult": int(mult),
                    }
                )
    core_degree = sum(
        sp.Poly(sp.expand(fac), q).degree() * mult for fac, mult in result.core_R
    )
    return {
        "roots": roots,
        "cyclotomic_support": {
            str(d): e for d, e in sorted(result.cyclotomic_R.items())
        },
        "core_degree": int(core_degree),
        "degree": degree,
    }
