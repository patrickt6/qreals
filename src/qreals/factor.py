r"""Factor the numerator R(q) and denominator S(q) of a q-rational [a/b]_q.

An open question concerns the shape of the numerator R(q)
of [a/b]_q as an element of Z[q]: when is R(q) irreducible (the worked
observation is that R is often irreducible when a is prime), and how does the
cyclotomic-style factorisation seen in the q-integer case [m/1]_q = [m]_q
extend to a general fraction? This module answers both by factoring R and S
exactly over Z[q] with sympy.factor_list and classifying each irreducible
factor as a cyclotomic polynomial Phi(d) or a non-cyclotomic "core" factor.

For an integer m, [m/1]_q = [m]_q = 1 + q + ... + q^{m-1} factors as the
product of Phi(d) over the divisors d | m with d > 1 (the classical
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
from math import lcm

import sympy as sp

from . import formatter
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
            exponent of Phi(d) (d >= 1) among the factors of R.
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
    """Return d if factor equals Phi(d) for some d >= 1, else None.

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
    cyclotomic maps d -> e_d over the factors that equal Phi(d), and core is
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
    classified as a cyclotomic polynomial Phi(d) or a non-cyclotomic core
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


@dataclass(frozen=True)
class SProperties:
    """The mathematical properties of the denominator S(q) of [a/d]_q.

    S(q) is the central object of the denominator question: for a rational
    x = a/d in lowest terms, [a/d]_q = q^k R(q)/S(q) with S monic, S(0) = 1.
    The theory (prime_power_saturation, degree_bound_proof) says: when S is a
    squarefree product of cyclotomics S = prod_{k in T} Phi(k), it divides [n]_q
    iff e_star = lcm(T) divides n; deg S <= d-1 with equality iff S = [d]_q iff
    a == +/-1 (mod d); S(1) = d always; S(0) = 1 always. Some S are a proper
    "collapse" (a strict cyclotomic subproduct of [d]_q, finite difference,
    minimal saturating n = e_star), and some are the impossibility branch:
    non-squarefree (e.g. 3/8 gives Phi(2)^2 Phi(4)) or non-cyclotomic (e.g. 2/15),
    which divide no [n]_q at all, so the difference of equal-tail q-rationals is
    never finite.

    Fields:
        a, d: the input fraction a/d in lowest terms (d > 0).
        S_str: the factored S(q), cyclotomic factors printed as Phi(k).
        index_set_T: the sorted list of cyclotomic indices k with Phi(k) | S
            (only meaningful in the squarefree-cyclotomic regime; the indices of
            the cyclotomic part otherwise).
        multiplicities: k -> e_k, the exponent of Phi(k) in S (all 1 when
            squarefree).
        is_cyclotomic: True when S is a pure product of cyclotomics (no core).
        is_squarefree: True when S is squarefree (every factor multiplicity 1).
        saturation_index: e_star = lcm(T) when S is squarefree-cyclotomic, the
            minimal n with S | [n]_q; None in the impossibility branch (S divides
            no [n]_q).
        minimal_saturating_n: the smallest n >= 1 with S | [n]_q, equal to
            saturation_index, or None when no such n exists.
        deg_S: the polynomial degree of S(q).
        deg_bound: d - 1, the upper bound from the degree theorem.
        saturates_bound: True iff deg_S == d-1, iff S == [d]_q, iff a == +/-1
            (mod d).
        is_full_qint: True iff S(q) == [d]_q (the no-collapse case).
        is_collapse: True iff S is a proper squarefree-cyclotomic subproduct of
            [d]_q (finite difference, but S != [d]_q).
        S_at_1: S(1), which always equals d (a built-in invariant check).
        S_at_1_ok: True iff S(1) == d (the invariant holds).
        S_at_0: S(0), which always equals 1 (q does not divide S).
        S_at_0_ok: True iff S(0) == 1.
        a_mod_d: a reduced mod d, for the equality-locus reading.
        equality_locus: True iff a == 1 or a == d-1 (mod d), the predicted
            deg S = d-1 locus.
    """

    a: int
    d: int
    S_str: str
    index_set_T: list[int]
    multiplicities: dict[int, int]
    is_cyclotomic: bool
    is_squarefree: bool
    saturation_index: int | None
    minimal_saturating_n: int | None
    deg_S: int
    deg_bound: int
    saturates_bound: bool
    is_full_qint: bool
    is_collapse: bool
    S_at_1: int
    S_at_1_ok: bool
    S_at_0: int
    S_at_0_ok: bool
    a_mod_d: int
    equality_locus: bool


def s_properties(x: Fraction | str | tuple[int, int] | list[int]) -> SProperties:
    r"""Analyze the q-denominator S(q) of [a/d]_q: its cyclotomic structure.

    This is the S(q) companion to factor_qreal. Where factor_qreal reports the
    factorisation of both R and S, this function surfaces the *mathematical
    properties* of S: the cyclotomic index
    set T, the saturation index e_star = lcm(T) and the minimal n with
    S | [n]_q, the degree deg S against the bound d-1 (with the a == +/-1 mod d
    equality flag), the S(1) = d and S(0) = 1 invariants, and whether S is the
    full [d]_q, a proper cyclotomic collapse, or the impossibility branch
    (non-squarefree / non-cyclotomic, dividing no [n]_q).

    Everything is exact over Z[q]: the factorisation is sympy.factor_list, the
    saturation reasoning is the squarefree-cyclotomic divisor theory of the two
    proof notes, never a numerical test.
    """
    result = factor_qreal(x)
    d = result.b
    a = result.a
    S = denominator_expr(result)

    is_cyclotomic = len(result.core_S) == 0
    multiplicities = dict(sorted(result.cyclotomic_S.items()))
    is_squarefree = is_cyclotomic and all(e == 1 for e in multiplicities.values())

    index_set_T = sorted(multiplicities.keys())
    # Saturation only makes sense when S is a squarefree product of cyclotomics;
    # then S | [n]_q iff every k in T divides n iff lcm(T) | n, so the minimal
    # saturating n is lcm(T). A non-squarefree S (e.g. Phi(2)^2 Phi(4) for 3/8) or
    # a non-cyclotomic S (e.g. 2/15) divides no [n]_q, so no n saturates.
    if is_squarefree and index_set_T:
        saturation_index: int | None = lcm(*index_set_T)
    elif is_squarefree and not index_set_T:
        # S == 1 (only for the empty tail; degenerate), vacuously saturated at 1.
        saturation_index = 1
    else:
        saturation_index = None
    minimal_saturating_n = saturation_index

    poly_S = sp.Poly(sp.expand(S), q)
    deg_S = poly_S.degree()
    deg_bound = d - 1
    saturates_bound = deg_S == deg_bound

    qint_d = sp.expand(sum(q**i for i in range(d)))
    is_full_qint = sp.expand(S - qint_d) == 0
    is_collapse = is_squarefree and is_cyclotomic and not is_full_qint

    S_at_1 = int(sp.expand(S).subs(q, 1))
    S_at_0 = int(sp.expand(S).subs(q, 0))
    a_mod_d = a % d
    equality_locus = a_mod_d == 1 or a_mod_d == d - 1

    return SProperties(
        a=a,
        d=d,
        S_str=_factor_label(result.content_S, result.factors_S),
        index_set_T=index_set_T,
        multiplicities=multiplicities,
        is_cyclotomic=is_cyclotomic,
        is_squarefree=is_squarefree,
        saturation_index=saturation_index,
        minimal_saturating_n=minimal_saturating_n,
        deg_S=deg_S,
        deg_bound=deg_bound,
        saturates_bound=saturates_bound,
        is_full_qint=is_full_qint,
        is_collapse=is_collapse,
        S_at_1=S_at_1,
        S_at_1_ok=S_at_1 == d,
        S_at_0=S_at_0,
        S_at_0_ok=S_at_0 == 1,
        a_mod_d=a_mod_d,
        equality_locus=equality_locus,
    )


def _factor_label(content: sp.Expr, factors: list[tuple[sp.Expr, int]]) -> str:
    """Pretty-print a factorisation with cyclotomic factors as Phi(k)."""
    parts: list[str] = []
    if content != sp.Integer(1):
        parts.append(str(content))
    for fac, mult in factors:
        d = _cyclotomic_index(fac)
        label = (
            formatter.phi_applied_label(d) if d is not None else f"({sp.sstr(fac)})"
        )
        parts.append(label if mult == 1 else f"{label}^{mult}")
    return " * ".join(parts) if parts else "1"


def s_regime(p: SProperties) -> str:
    """The one-word regime of a denominator S(q), from its SProperties.

    Returns one of:
        "full"          S == [d]_q (no collapse), a == +/-1 (mod d);
        "collapse"      a proper squarefree-cyclotomic subproduct of [d]_q;
        "nonsquarefree" cyclotomic but with a repeated factor (e.g. 3/8 gives
                        Phi(2)^2 Phi(4)); divides no [n]_q;
        "noncyclotomic" carries a non-cyclotomic core (e.g. 2/15); divides no
                        [n]_q.
    The first two are the saturating regimes (a finite saturation index e*);
    the last two are the impossibility branch (no n with S | [n]_q).
    """
    if p.is_full_qint:
        return "full"
    if p.is_collapse:
        return "collapse"
    if not p.is_cyclotomic:
        return "noncyclotomic"
    return "nonsquarefree"


# Human labels and a stable display order for the four regimes, shared by the
# CLI tables and the web atlas legend so both read the same.
REGIME_LABELS: dict[str, str] = {
    "full": "full [d]_q",
    "collapse": "proper collapse",
    "nonsquarefree": "non-squarefree (no n)",
    "noncyclotomic": "non-cyclotomic (no n)",
}
REGIME_ORDER: list[str] = ["full", "collapse", "nonsquarefree", "noncyclotomic"]


def _divisor_indices(d: int) -> list[int]:
    """The cyclotomic indices of [d]_q: the divisors k of d with k >= 2."""
    return [k for k in range(2, d + 1) if d % k == 0]


def _dropped_indices(p: SProperties) -> list[int]:
    """For a squarefree S, the cyclotomic factors of [d]_q absent from S.

    The dropped set is the subset-product bookkeeping of Remark 2, so it is only
    meaningful when S is a squarefree product of cyclotomics: the divisor indices
    of d not in the index set T of S. For the full [d]_q this is empty; for a
    proper collapse it is the factors that cancelled. A non-squarefree or
    non-cyclotomic S is outside the subset-product picture, so this returns the
    empty list there (no totient-weight reading applies).
    """
    if not p.is_squarefree:
        return []
    return [k for k in _divisor_indices(p.d) if k not in set(p.index_set_T)]


def _collapse_depth_check(p: SProperties) -> tuple[int, int, bool]:
    """The collapse depth d-1-deg S and the totient weight of the dropped Phi(k).

    The degree note says the drop below the bound equals the sum of the
    totients of the dropped cyclotomic factors. Returns (drop, totient_sum,
    matches) where drop = d-1-deg S and totient_sum = sum phi(k) over the
    dropped indices; matches is their equality (always True for a cyclotomic S,
    a built-in cross-check of the degree note).
    """
    drop = p.deg_bound - p.deg_S
    totient_sum = int(sum(int(sp.totient(k)) for k in _dropped_indices(p)))
    return drop, totient_sum, drop == totient_sum


def s_atlas_cell(a: int, d: int) -> dict:
    """The atlas datum for one fraction a/d: its regime and S(q) summary.

    A single cell of the S(q) cyclotomic-factor atlas. Everything is exact via
    s_properties; the regime is the key field the heat map colours by.
    """
    p = s_properties((a, d))
    drop, totient_sum, depth_ok = _collapse_depth_check(p)
    return {
        "a": a,
        "d": d,
        "regime": s_regime(p),
        "T": list(p.index_set_T),
        "dropped": _dropped_indices(p),
        "deg_S": p.deg_S,
        "deg_bound": p.deg_bound,
        "saturation_index": p.saturation_index,
        "collapse_depth": drop,
        "totient_sum": totient_sum,
        "depth_ok": depth_ok,
        "S_str": p.S_str,
    }


def _coprime_numerators(d: int, a_max: int | None) -> list[int]:
    """The numerators a with 1 <= a < d, gcd(a, d) = 1, capped by a_max.

    The atlas and the explorers range over proper fractions 0 < a/d < 1 in
    lowest terms, the setting of the two proof notes (0 < x < 1).
    """
    from math import gcd

    top = d - 1 if a_max is None else min(a_max, d - 1)
    return [a for a in range(1, top + 1) if gcd(a, d) == 1]


def s_atlas(d_max: int, a_max: int | None = None) -> dict:
    """The S(q) cyclotomic-factor atlas over the coprime grid 2 <= d <= d_max.

    For every proper fraction a/d in lowest terms (0 < a/d < 1) up to the given
    bounds this records the regime of its q-denominator S(q) (full [d]_q,
    proper collapse, non-squarefree, or non-cyclotomic), its index set T, degree
    and saturation index. A companion tally counts, for each cyclotomic index k,
    how often Phi(k) appears across the grid, making Remark 2 (S is a subset
    product of the cyclotomic factors of [n]_q) visible as d grows.

    Returns {cells, d_max, a_max, regime_counts, index_appearances}.
    """
    cells: list[dict] = []
    regime_counts: dict[str, int] = {r: 0 for r in REGIME_ORDER}
    index_appearances: dict[int, int] = {}
    for d in range(2, d_max + 1):
        for a in _coprime_numerators(d, a_max):
            cell = s_atlas_cell(a, d)
            cells.append(cell)
            regime_counts[cell["regime"]] += 1
            for k in cell["T"]:
                index_appearances[k] = index_appearances.get(k, 0) + 1
    return {
        "cells": cells,
        "d_max": d_max,
        "a_max": a_max,
        "regime_counts": regime_counts,
        "index_appearances": dict(sorted(index_appearances.items())),
    }


def saturation_explorer(d: int) -> dict:
    """The saturation index e* and minimal saturating n as a ranges over a/d.

    For a fixed denominator d this sweeps every numerator a coprime to d
    (0 < a < d) and records the saturation index e*(a/d) = lcm(T) and the
    minimal n with S | [n]_q, flagging the impossibility residues (no finite n).
    This is the Saturation box read across a whole denominator: e* is the least
    n making the difference of two equal-tail q-rationals over S finite.

    Returns {d, points} with points = [{a, e_star, regime, T, deg_S, ...}].
    """
    points: list[dict] = []
    for a in _coprime_numerators(d, None):
        p = s_properties((a, d))
        points.append(
            {
                "a": a,
                "d": d,
                "e_star": p.saturation_index,
                "minimal_n": p.minimal_saturating_n,
                "regime": s_regime(p),
                "T": list(p.index_set_T),
                "deg_S": p.deg_S,
                "a_mod_d": p.a_mod_d,
                "equality_locus": p.equality_locus,
            }
        )
    return {"d": d, "points": points}


def degree_collapse(d_max: int, a_max: int | None = None) -> dict:
    """deg S against the bound d-1 over the coprime grid, with the collapse depth.

    For each proper fraction a/d this gives deg S and the bound d-1; the
    diagonal deg S = d-1 is the saturating a == +/-1 locus (S = [d]_q), and the
    drop below it is the collapse depth d-1-deg S, which for a cyclotomic S
    equals the totient weight of the dropped Phi(k) (the degree note's law). The
    depth_ok flag cross-checks that equality on every cyclotomic cell.

    Returns {cells, d_max, a_max, depth_law_holds} where depth_law_holds is
    True iff the totient-sum law held on every cyclotomic cell of the grid.
    """
    cells: list[dict] = []
    depth_law_holds = True
    for d in range(2, d_max + 1):
        for a in _coprime_numerators(d, a_max):
            p = s_properties((a, d))
            drop, totient_sum, depth_ok = _collapse_depth_check(p)
            regime = s_regime(p)
            # The collapse-depth law (drop = sum of totients of the dropped
            # Phi(k)) is a squarefree-regime statement: it covers the full [d]_q
            # and the proper collapses, where S is a subset product of the
            # cyclotomic factors of [d]_q. A non-squarefree S (e.g. Phi(2)^2 Phi(4)
            # for 3/8) or a non-cyclotomic S is outside the subset-product
            # bookkeeping, so it is excluded from the law check.
            if p.is_squarefree and not depth_ok:
                depth_law_holds = False
            cells.append(
                {
                    "a": a,
                    "d": d,
                    "deg_S": p.deg_S,
                    "deg_bound": p.deg_bound,
                    "drop": drop,
                    "totient_sum": totient_sum,
                    "depth_ok": depth_ok,
                    "dropped": _dropped_indices(p),
                    "regime": regime,
                    "saturates_bound": p.saturates_bound,
                }
            )
    return {
        "cells": cells,
        "d_max": d_max,
        "a_max": a_max,
        "depth_law_holds": depth_law_holds,
    }


def classify_poles(result: QRealFactor) -> dict:
    """The complex roots of S(q) (the poles of [a/d]_q), tagged by factor class.

    The pole companion to classify_roots: each root of the denominator S(q) is a
    pole of the rational function [a/d]_q, and is labelled by the exact factor
    it solves. A pole is "cyclotomic" when its irreducible factor is Phi(k)
    (so it is a primitive k-th root of unity on the unit circle) and "core" when
    the factor is non-cyclotomic (where the pole can leave |q| = 1 and drop the
    radius of convergence below 1). The complex coordinates are numerical
    (sympy nroots), for plotting only; the kind label and index k are exact.

    Returns {poles, radius, radius_index} where poles is a list of
    {re, im, mod, kind, d, mult}, radius is the least modulus (the radius of
    convergence of [a/d]_q), and radius_index is the cyclotomic index of the
    nearest pole when it is cyclotomic, else None.
    """
    poles: list[dict] = []
    for fac, mult in result.factors_S:
        poly = sp.Poly(sp.expand(fac), q)
        if poly.degree() < 1:
            continue
        d = _cyclotomic_index(fac)
        kind = "cyclotomic" if d is not None else "core"
        for r in poly.nroots():
            c = complex(r)
            for _ in range(int(mult)):
                poles.append(
                    {
                        "re": float(c.real),
                        "im": float(c.imag),
                        "mod": abs(c),
                        "kind": kind,
                        "d": d,
                        "mult": int(mult),
                    }
                )
    radius = min((p["mod"] for p in poles), default=None)
    radius_index = None
    if radius is not None:
        for p in poles:
            if abs(p["mod"] - radius) < 1e-9:
                radius_index = p["d"]
                break
    return {"poles": poles, "radius": radius, "radius_index": radius_index}


def classify_roots(result: QRealFactor) -> dict:
    """The complex roots of R(q), each tagged by its exact factor class.

    A display helper for the serve visualizer: the cyclotomic-vs-core split of
    each root comes from the exact Z[q] factorisation in `result`, not from
    numerical proximity to the unit circle. A root is "cyclotomic" exactly when
    the irreducible factor it solves is a Phi(d) (so it is a primitive d-th
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
