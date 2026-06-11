r"""The one-shot dossier of the q-denominator S(q) of [a/d]_q.

For a rational a/d in lowest terms, [a/d]_q = q^k N(q)/S(q) with S monic and
S(0) = 1. This module computes everything the denominator question asks of a
single fraction: S expanded and factored, its cyclotomic index set T (or the
non-cyclotomic verdict), deg S against the bound d-1, the S(1) = d invariant,
the class of S, the residue a^2 mod d, and, in the collapse case, every
coprime split d = d_+ d_- with the discrepancy [d_+]_q [d_-]_q / S classified
EXACT, POLYNOMIAL, or RATIO.

The classes:
    FULL      S equals the full q-integer of d (every cyclotomic factor of
              index dividing d, each once); exactly the a == +/-1 (mod d)
              tails.
    COLLAPSE  S is a proper squarefree product of cyclotomic factors with
              indices dividing d.
    REPEATED  S is a product of cyclotomic factors but one appears with
              multiplicity at least 2 (e.g. 3/8).
    NONCYC    S carries a non-cyclotomic irreducible factor (e.g. 2/15).

Everything is exact over Z[q]. The cyclotomic structure of S is read off by
trial division by the cyclotomic polynomials of the divisors of d (and, for
any remaining cofactor, by full factorisation with the factor module's
classifier), reusing the engine's q_rational_pair value; the result is
cross-checked against the factor-module route in the test suite.

JSON schema (the --json output of `qreals denom`, all values built from this
module's dossier; keys are stable):
    a, d            int   the fraction in lowest terms (d > 0)
    cf              [int] the regular continued fraction of a/d
    k               int   the power of q split off the numerator
    N               str   the numerator polynomial, expanded ascii
    S               str   the denominator polynomial, expanded ascii
    S_factored      str   the factored S, cyclotomic factors by index
    T               [int] the cyclotomic index set of S (sorted)
    multiplicities  {str(e): int}  exponent of each cyclotomic index in S
    noncyclotomic_cofactor  str | null   the factored non-cyclotomic part
    is_cyclotomic_product   bool
    deg_S, deg_bound        int   deg S and d - 1
    S_at_1          int   S(1), always d
    S_at_1_ok       bool
    klass           str   FULL | COLLAPSE | REPEATED | NONCYC
    a_sq_mod_d      int   a^2 mod d
    splits          [ {d_plus, d_minus: int, realized: bool,
                       discrepancy_class: "EXACT"|"POLYNOMIAL"|"RATIO",
                       discrepancy: str (factored)} ]
                    every coprime split of d (collapse class only, else [])
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd

import sympy as sp

from . import formatter
from .factor import _cyclotomic_index
from .rational import q, q_rational_pair


@dataclass(frozen=True)
class Split:
    """One coprime split d = d_plus * d_minus with its discrepancy.

    The discrepancy is the exact ratio of the q-integer product of the two
    parts to S, recorded as cyclotomic exponent vectors: num_exponents and
    den_exponents map a cyclotomic index e to its (positive) exponent in the
    numerator and denominator of the reduced ratio. The class is EXACT when
    the ratio is 1, POLYNOMIAL when the reduced denominator is trivial, and
    RATIO otherwise. realized flags the splits whose parts both see a on the
    a == +/-1 equality locus (a == +/-1 mod d_plus and mod d_minus).
    """

    d_plus: int
    d_minus: int
    realized: bool
    num_exponents: dict[int, int]
    den_exponents: dict[int, int]
    klass: str


@dataclass(frozen=True)
class DenomDossier:
    """The full denominator dossier of one fraction a/d."""

    a: int
    d: int
    cf: list[int]
    k: int
    N: sp.Poly
    S: sp.Poly
    multiplicities: dict[int, int]
    cofactor_factors: list[tuple[sp.Expr, int]]
    klass: str
    splits: list[Split]

    @property
    def index_set(self) -> list[int]:
        return sorted(self.multiplicities)

    @property
    def is_cyclotomic_product(self) -> bool:
        return not self.cofactor_factors

    @property
    def deg_S(self) -> int:
        return int(self.S.degree())

    @property
    def S_at_1(self) -> int:
        return int(self.S.eval(1))


def cf_str(cf: list[int]) -> str:
    """A regular continued fraction as [a0; a1, a2, ...]."""
    if len(cf) == 1:
        return f"[{cf[0]}]"
    return f"[{cf[0]}; " + ", ".join(str(t) for t in cf[1:]) + "]"


def parse_fraction(text: str) -> tuple[int, int]:
    """Read a positive rational a/d from 'a/d' (or 'a' for an integer)."""
    frac = Fraction(text.strip())
    if frac <= 0:
        raise ValueError("the fraction a/d must be positive")
    return frac.numerator, frac.denominator


def _cyclotomic_structure(
    S: sp.Poly, d: int
) -> tuple[dict[int, int], list[tuple[sp.Expr, int]]]:
    """The cyclotomic multiplicities of S and its non-cyclotomic cofactor.

    Trial-divides S by the cyclotomic polynomial of every divisor e >= 2 of d
    (the indices the theory allows for a cyclotomic S); whatever remains is
    fully factored and classified with the factor module's cyclotomic test,
    so a cyclotomic factor at an unexpected index is still found.
    """
    mult: dict[int, int] = {}
    rem = S
    for e in sorted(int(t) for t in sp.divisors(d)):
        if e < 2:
            continue
        phi = sp.Poly(sp.cyclotomic_poly(e, q), q, domain="ZZ")
        while rem.degree() >= phi.degree():
            quo, r = sp.div(rem, phi)
            if not r.is_zero:
                break
            rem = quo
            mult[e] = mult.get(e, 0) + 1
    cofactor: list[tuple[sp.Expr, int]] = []
    if rem.degree() > 0:
        _, pairs = sp.factor_list(rem.as_expr(), q)
        for fac, m in pairs:
            fac_expr = fac.as_expr() if isinstance(fac, sp.Poly) else sp.sympify(fac)
            e = _cyclotomic_index(fac_expr)
            if e is not None:
                mult[e] = mult.get(e, 0) + int(m)
            else:
                cofactor.append((fac_expr, int(m)))
    return mult, cofactor


def _classify(d: int, mult: dict[int, int], cofactor) -> str:
    if cofactor:
        return "NONCYC"
    if any(m > 1 for m in mult.values()):
        return "REPEATED"
    full = {e for e in (int(t) for t in sp.divisors(d)) if e >= 2}
    if set(mult) == full:
        return "FULL"
    return "COLLAPSE"


def _qint_indices(n: int) -> dict[int, int]:
    """The cyclotomic exponent vector of [n]_q: 1 at every divisor e >= 2."""
    return {e: 1 for e in (int(t) for t in sp.divisors(n)) if e >= 2}


def _coprime_splits(d: int) -> list[tuple[int, int]]:
    """Every unordered coprime split d = d_plus * d_minus with d_plus <= d_minus."""
    out = []
    for d_plus in sorted(int(t) for t in sp.divisors(d)):
        d_minus = d // d_plus
        if d_plus > d_minus:
            break
        if gcd(d_plus, d_minus) == 1:
            out.append((d_plus, d_minus))
    return out


def _on_locus(a: int, m: int) -> bool:
    """True when a == +/-1 (mod m); vacuously true at m = 1."""
    if m == 1:
        return True
    r = a % m
    return r == 1 % m or r == (m - 1) % m


def _splits(a: int, d: int, mult: dict[int, int]) -> list[Split]:
    """Every coprime split of d with its discrepancy [d+]_q [d-]_q / S.

    Both the q-integer product and S are products of cyclotomic polynomials
    here (collapse class), so the reduced discrepancy is read off the
    exponent vectors exactly, with no polynomial division.
    """
    out: list[Split] = []
    for d_plus, d_minus in _coprime_splits(d):
        prod: dict[int, int] = {}
        for part in (d_plus, d_minus):
            for e, m in _qint_indices(part).items():
                prod[e] = prod.get(e, 0) + m
        num: dict[int, int] = {}
        den: dict[int, int] = {}
        for e in sorted(set(prod) | set(mult)):
            diff = prod.get(e, 0) - mult.get(e, 0)
            if diff > 0:
                num[e] = diff
            elif diff < 0:
                den[e] = -diff
        if not num and not den:
            klass = "EXACT"
        elif not den:
            klass = "POLYNOMIAL"
        else:
            klass = "RATIO"
        out.append(
            Split(
                d_plus=d_plus,
                d_minus=d_minus,
                realized=_on_locus(a, d_plus) and _on_locus(a, d_minus),
                num_exponents=num,
                den_exponents=den,
                klass=klass,
            )
        )
    return out


def denom_dossier(a: int, d: int) -> DenomDossier:
    """The denominator dossier of [a/d]_q, exact over Z[q]."""
    frac = Fraction(int(a), int(d))
    if frac <= 0:
        raise ValueError("the fraction a/d must be positive")
    a, d = frac.numerator, frac.denominator
    N, S = q_rational_pair(a, d)
    k = 0
    if not N.is_zero:
        k = int(min(m[0] for m in N.monoms()))
        if k:
            N = N.exquo(sp.Poly(q**k, q, domain="ZZ"))
    cf = [int(t) for t in sp.continued_fraction(sp.Rational(a, d))]
    mult, cofactor = _cyclotomic_structure(S, d)
    klass = _classify(d, mult, cofactor)
    splits = _splits(a, d, mult) if klass == "COLLAPSE" else []
    return DenomDossier(
        a=a,
        d=d,
        cf=cf,
        k=k,
        N=N,
        S=S,
        multiplicities=dict(sorted(mult.items())),
        cofactor_factors=cofactor,
        klass=klass,
        splits=splits,
    )


# --------------------------------------------------------------------------
# Rendering, ASCII and TeX, both through the shared formatter.
# --------------------------------------------------------------------------


def _exponents_ascii(exps: dict[int, int]) -> str:
    if not exps:
        return "1"
    return " ".join(formatter.phi_label(e, m) for e, m in sorted(exps.items()))


def _exponents_tex(exps: dict[int, int]) -> str:
    if not exps:
        return "1"
    return r" \, ".join(formatter.phi_tex(e, m) for e, m in sorted(exps.items()))


def split_discrepancy_ascii(s: Split) -> str:
    """The factored discrepancy of one split."""
    if s.klass == "EXACT":
        return "1"
    if s.klass == "POLYNOMIAL":
        return _exponents_ascii(s.num_exponents)
    return formatter.fraction_layout(
        _exponents_ascii(s.num_exponents), _exponents_ascii(s.den_exponents)
    )


def split_discrepancy_tex(s: Split) -> str:
    if s.klass == "EXACT":
        return "1"
    if s.klass == "POLYNOMIAL":
        return _exponents_tex(s.num_exponents)
    return formatter.fraction_tex(
        _exponents_tex(s.num_exponents), _exponents_tex(s.den_exponents)
    )


def s_factored_ascii(dossier: DenomDossier) -> str:
    """S(q) factored, cyclotomic factors by index, cofactor marked."""
    parts = [
        formatter.phi_label(e, m) for e, m in sorted(dossier.multiplicities.items())
    ]
    for fac, m in dossier.cofactor_factors:
        inner = f"({formatter.poly_ascii(fac, wrap=10**9)})"
        label = inner if m == 1 else f"{inner}^{m}"
        parts.append(f"{label} [non-cyclotomic]")
    return " * ".join(parts) if parts else "1"


def s_factored_tex(dossier: DenomDossier) -> str:
    parts = [
        formatter.phi_tex(e, m) for e, m in sorted(dossier.multiplicities.items())
    ]
    for fac, m in dossier.cofactor_factors:
        inner = rf"\left({formatter.poly_tex(fac)}\right)"
        label = inner if m == 1 else inner + rf"^{{{m}}}"
        parts.append(label + r"\ \text{(non-cyclotomic)}")
    return r" \, ".join(parts) if parts else "1"


def dossier_tex(dossier: DenomDossier) -> str:
    """The TeX block of the dossier, ready to paste into notes.

    Compiles standalone inside a minimal article preamble (amsmath); the CI
    wraps it and runs pdflatex.
    """
    p = dossier
    lines = [
        r"\begin{align*}",
        rf"x &= {p.a}/{p.d}, \quad [a_0; a_1, \ldots] = {cf_str(p.cf)} \\",
        rf"S(q) &= {formatter.poly_tex(p.S.as_expr())} \\",
        rf"S(q) &= {s_factored_tex(p)} \\",
        rf"\deg S &= {p.deg_S}, \quad d - 1 = {p.d - 1}, \quad S(1) = {p.S_at_1} \\",
        r"\text{class} &= \text{" + p.klass + r"}, \quad "
        + formatter.congruence_tex("a^2", str((p.a * p.a) % p.d), p.d),
    ]
    for s in p.splits:
        lines.append(
            r"\\ "
            + rf"{p.d} &= {s.d_plus} \cdot {s.d_minus}: \quad "
            + rf"{formatter.fraction_tex(formatter.qint_tex(s.d_plus) + r' \, ' + formatter.qint_tex(s.d_minus), 'S')}"
            + rf" = {split_discrepancy_tex(s)} \quad \text{{({s.klass})}}"
        )
    lines.append(r"\end{align*}")
    return "\n".join(lines)


def dossier_data(dossier: DenomDossier) -> dict:
    """The stable JSON object of the dossier (schema in the module docstring)."""
    p = dossier
    return {
        "a": p.a,
        "d": p.d,
        "cf": list(p.cf),
        "k": p.k,
        "N": formatter.poly_ascii(p.N.as_expr(), wrap=10**9),
        "S": formatter.poly_ascii(p.S.as_expr(), wrap=10**9),
        "S_factored": s_factored_ascii(p),
        "T": p.index_set,
        "multiplicities": {str(e): m for e, m in p.multiplicities.items()},
        "noncyclotomic_cofactor": (
            None
            if p.is_cyclotomic_product
            else " * ".join(
                f"({formatter.poly_ascii(f, wrap=10**9)})" + ("" if m == 1 else f"^{m}")
                for f, m in p.cofactor_factors
            )
        ),
        "is_cyclotomic_product": p.is_cyclotomic_product,
        "deg_S": p.deg_S,
        "deg_bound": p.d - 1,
        "S_at_1": p.S_at_1,
        "S_at_1_ok": p.S_at_1 == p.d,
        "klass": p.klass,
        "a_sq_mod_d": (p.a * p.a) % p.d,
        "splits": [
            {
                "d_plus": s.d_plus,
                "d_minus": s.d_minus,
                "realized": s.realized,
                "discrepancy_class": s.klass,
                "discrepancy": split_discrepancy_ascii(s),
            }
            for s in p.splits
        ],
    }
