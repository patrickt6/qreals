r"""The headline cyclotomic factorisation of a q-rational [a/b]_q.

This is the one view that puts the cyclotomic anatomy of a q-rational front
and centre: the fully factored form

    [a/b]_q = q^k R(q)/S(q),

with R and S written as explicit products of cyclotomic polynomials Phi(e)
(each labelled by its index e) and any non-cyclotomic "core" factor flagged,
plus the brick strip of the denominator (one Phi(e) per divisor e >= 2 of d,
marked kept, dropped, or repeated relative to the full [d]_q) and the
one-line verdict (the class of S, the index set T, the saturation index
e* = lcm(T), deg S against the d-1 bound, and the S(1) = d invariant).

It is pure composition over the exact engine: factor.factor_qreal does the
factorisation, factor.s_properties reads off the denominator theory, and
bricks.prime_power_rule gives each brick's value at q = 1. No floating point,
no new arithmetic of its own.

JSON schema (the --json output of `qreals cyclotomic`; keys are stable):
    a, d            int    the fraction in lowest terms (d > 0)
    k               int    the power of q split off the numerator, R(0) = 1
    R, S            str    R(q), S(q) factored, ascii (Phi(e) for cyclotomics)
    R_tex, S_tex    str    the same, TeX (cyclotomic factors as \Phi indices)
    headline_tex    str    [a/d]_q = q^k R/S, fully factored, ready to render
    cyclotomic_R    {str(e): int}   exponent of Phi(e) in R
    cyclotomic_S    {str(e): int}   exponent of Phi(e) in S
    core_R, core_S  [str]  the non-cyclotomic core factors (ascii), if any
    bricks          [ {e, in_qint_d, mult, deg, value_at_1, state} ]
                    one per cyclotomic index of [d]_q or of S; state is
                    "kept" | "dropped" | "repeated" | "extra"
    index_set_T     [int]  the cyclotomic index set of S (sorted)
    saturation_index  int | null   e* = lcm(T), the minimal n with S | [n]_q
    deg_S, deg_bound  int  deg S and d - 1
    klass           str    "full" | "collapse" | "repeated" | "noncyclotomic"
    klass_label     str    a human sentence for the class
    is_cyclotomic, is_squarefree, is_full_qint  bool
    S_at_1          int    S(1), always d
    equality_locus  bool   a == +/-1 (mod d), the deg S = d-1 locus
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from . import formatter as fmt
from .bricks import prime_power_rule
from .factor import (
    _cyclotomic_index,
    denominator_expr,
    factor_qreal,
    numerator_expr,
    s_properties,
)


@dataclass(frozen=True)
class Brick:
    """One cyclotomic brick of the denominator picture.

    e is the cyclotomic index; in_qint_d says whether Phi(e) is a factor of
    the full [d]_q (i.e. e | d, e >= 2); mult is its multiplicity in S (0 when
    dropped); deg is phi(e) (the degree of Phi(e)); value_at_1 is p when e is a
    prime power p^j and 1 otherwise; state is one of kept, dropped, repeated,
    extra.
    """

    e: int
    in_qint_d: bool
    mult: int
    deg: int
    value_at_1: int
    state: str


@dataclass(frozen=True)
class CyclotomicView:
    """The full headline view of [a/d]_q (schema in the module docstring)."""

    a: int
    d: int
    k: int
    R: sp.Expr
    S: sp.Expr
    R_tex: str
    S_tex: str
    headline_tex: str
    cyclotomic_R: dict[int, int]
    cyclotomic_S: dict[int, int]
    core_R: list[tuple[sp.Expr, int]]
    core_S: list[tuple[sp.Expr, int]]
    bricks: list[Brick]
    index_set_T: list[int]
    saturation_index: int | None
    deg_S: int
    deg_bound: int
    klass: str
    klass_label: str
    is_cyclotomic: bool
    is_squarefree: bool
    is_full_qint: bool
    S_at_1: int
    equality_locus: bool
    # The ascii factored forms (Phi(e) spelling), captured at build time from
    # the same QRealFactor as everything else so view_data never has to
    # re-factor the fraction.
    R_ascii: str = ""
    S_ascii: str = ""


_CLASS_LABEL = {
    "full": "S is the full q-integer [d]_q (no collapse); this is the "
    "a == +/-1 (mod d) locus",
    "collapse": "S is a proper cyclotomic collapse: a strict squarefree "
    "subproduct of [d]_q that still divides some [n]_q",
    "repeated": "S has a repeated cyclotomic factor (non-squarefree), so it "
    "divides no [n]_q",
    "noncyclotomic": "S carries a non-cyclotomic core factor, so it divides "
    "no [n]_q",
}


def _classify(p) -> str:
    """Map an SProperties record to the headline class key."""
    if not p.is_cyclotomic:
        return "noncyclotomic"
    if not p.is_squarefree:
        return "repeated"
    return "full" if p.is_full_qint else "collapse"


def _bricks(d: int, cyclotomic_S: dict[int, int]) -> list[Brick]:
    """The brick strip: every Phi(e) of [d]_q or of S, with its state."""
    divisors_d = {int(e) for e in sp.divisors(d) if int(e) >= 2}
    indices = sorted(divisors_d | set(cyclotomic_S))
    out: list[Brick] = []
    for e in indices:
        in_d = e in divisors_d
        mult = int(cyclotomic_S.get(e, 0))
        if mult == 0:
            state = "dropped"
        elif mult >= 2:
            state = "repeated"
        elif not in_d:
            state = "extra"
        else:
            state = "kept"
        out.append(
            Brick(
                e=e,
                in_qint_d=in_d,
                mult=mult,
                deg=int(sp.totient(e)),
                value_at_1=prime_power_rule(e),
                state=state,
            )
        )
    return out


def _factored_tex(content: sp.Expr, factors: list[tuple[sp.Expr, int]]) -> str:
    return fmt.factored_tex(content, factors, _cyclotomic_index)


def _factored_ascii(content: sp.Expr, factors: list[tuple[sp.Expr, int]]) -> str:
    return fmt.factored_ascii(content, factors, _cyclotomic_index)


def cyclotomic_view(x: Fraction | str | tuple[int, int] | list[int]) -> CyclotomicView:
    """Build the headline cyclotomic view of the q-rational x = a/d.

    x is read by factor_qreal (a Fraction, an 'a/b' string, an (a, b) pair, or
    the engine's '(a)/(b)' spelling). Everything returned is exact over Z[q].

    The fraction is factored exactly once: the QRealFactor from factor_qreal
    is passed straight into s_properties as its precomputed record, and the
    ascii factored forms view_data needs are captured here too.
    """
    f = factor_qreal(x)
    p = s_properties(x, precomputed=f)
    R = numerator_expr(f)
    S = denominator_expr(f)

    R_tex = _factored_tex(f.content_R, f.factors_R)
    S_tex = _factored_tex(f.content_S, f.factors_S)
    qpow = "" if f.k == 0 else fmt.q_power_tex(f.k) + r" \cdot "
    headline_tex = "%s = %s%s" % (
        fmt.qrat_tex(f.a, f.b),
        qpow,
        fmt.display_fraction_tex(R_tex, S_tex),
    )

    klass = _classify(p)
    return CyclotomicView(
        a=f.a,
        d=f.b,
        k=f.k,
        R=R,
        S=S,
        R_tex=R_tex,
        S_tex=S_tex,
        headline_tex=headline_tex,
        cyclotomic_R=dict(sorted(f.cyclotomic_R.items())),
        cyclotomic_S=dict(sorted(f.cyclotomic_S.items())),
        core_R=list(f.core_R),
        core_S=list(f.core_S),
        bricks=_bricks(f.b, f.cyclotomic_S),
        index_set_T=list(p.index_set_T),
        saturation_index=p.saturation_index,
        deg_S=p.deg_S,
        deg_bound=p.deg_bound,
        klass=klass,
        klass_label=_CLASS_LABEL[klass],
        is_cyclotomic=p.is_cyclotomic,
        is_squarefree=p.is_squarefree,
        is_full_qint=p.is_full_qint,
        S_at_1=p.S_at_1,
        equality_locus=p.equality_locus,
        R_ascii=_factored_ascii(f.content_R, f.factors_R),
        S_ascii=_factored_ascii(f.content_S, f.factors_S),
    )


def view_data(view: CyclotomicView) -> dict:
    """The stable JSON object of the view (schema in the module docstring)."""
    v = view
    return {
        "a": v.a,
        "d": v.d,
        "k": v.k,
        "R": v.R_ascii,
        "S": v.S_ascii,
        "R_tex": v.R_tex,
        "S_tex": v.S_tex,
        "headline_tex": v.headline_tex,
        "cyclotomic_R": {str(e): m for e, m in v.cyclotomic_R.items()},
        "cyclotomic_S": {str(e): m for e, m in v.cyclotomic_S.items()},
        "core_R": [
            fmt.poly_ascii(fac, wrap=10**9) + ("" if m == 1 else f"^{m}")
            for fac, m in v.core_R
        ],
        "core_S": [
            fmt.poly_ascii(fac, wrap=10**9) + ("" if m == 1 else f"^{m}")
            for fac, m in v.core_S
        ],
        "bricks": [
            {
                "e": b.e,
                "in_qint_d": b.in_qint_d,
                "mult": b.mult,
                "deg": b.deg,
                "value_at_1": b.value_at_1,
                "state": b.state,
            }
            for b in v.bricks
        ],
        "index_set_T": list(v.index_set_T),
        "saturation_index": v.saturation_index,
        "minimal_saturating_n": v.saturation_index,
        "deg_S": v.deg_S,
        "deg_bound": v.deg_bound,
        "klass": v.klass,
        "klass_label": v.klass_label,
        "is_cyclotomic": v.is_cyclotomic,
        "is_squarefree": v.is_squarefree,
        "is_full_qint": v.is_full_qint,
        "S_at_1": v.S_at_1,
        "equality_locus": v.equality_locus,
    }


def view_tex(view: CyclotomicView) -> str:
    """A TeX block of the view, ready to paste into notes (compiles standalone)."""
    v = view
    bricks_rows = []
    for b in v.bricks:
        state = b.state
        bricks_rows.append(
            rf"{fmt.phi_tex(b.e)} & {b.deg} & {b.value_at_1} & \text{{{state}}} \\"
        )
    sat = (
        str(v.saturation_index)
        if v.saturation_index is not None
        else r"\text{none}"
    )
    t_set = (
        r"\{" + ", ".join(str(e) for e in v.index_set_T) + r"\}"
        if v.index_set_T
        else r"\varnothing"
    )
    lines = [
        r"\begin{align*}",
        rf"{v.headline_tex} \\",
        rf"\deg S &= {v.deg_S} \quad (\text{{bound }} d - 1 = {v.deg_bound}),"
        rf" \quad S(1) = {v.S_at_1} = d \\",
        rf"T &= {t_set}, \quad e^\ast = \operatorname{{lcm}}(T) = {sat} \\",
        r"\text{class} &= \text{" + v.klass + r"}",
        r"\end{align*}",
        r"\[",
        r"\begin{array}{c|c|c|c}",
        r"\Phi(e) & \deg=\varphi(e) & \text{value at }q=1 & \text{state} \\ \hline",
        *bricks_rows,
        r"\end{array}",
        r"\]",
    ]
    return "\n".join(lines)
