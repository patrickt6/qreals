r"""Fractions that share the same q-denominator polynomial S.

For a rational a/d in lowest terms the denominator S(q) of [a/d]_q is a monic
integer polynomial with S(0) = 1 and S(1) = d. This module answers: which other
numerators b coprime to d give the same S, and, optionally, which fractions at
other moduli give an S identical up to a unit power of q.

The sharing class at d is the set of numerators b coprime to d (1 <= b < d) with
S(b/d) = S(a/d), found by grouping every coprime numerator by its S, exactly as
the reverse table does. The class carries the involution b -> -b^{-1} mod d: two
numerators paired by it share S, and at a prime modulus the class is exactly
{b, -b^{-1} mod d}. At a modulus divisible by p^2 with p == 1 (mod 4) a square
root of minus one mod p can enlarge the class beyond that pair; those roots are
reported so the extra pairing is explained, not just observed.

The across search compares S(a/d) against S(b/d') for every modulus d' up to a
declared bound and every numerator b coprime to d', reporting a match when the
two polynomials agree up to a unit power of q, with that unit q^k shown. The
comparison is exact over Z[q]: a unit multiple means one coefficient sequence is
the other shifted by a fixed power, detected from the integer exponent vectors.
Since S(1) = d, any such match necessarily shares the modulus d, so the across
search recovers the same-modulus partners while declaring the full range it
covered (silent truncation would be a gate failure).

JSON schema, same-d mode (the --json output of `qreals twin`, keys are stable):
    a, d            int    the fraction in lowest terms (d > 0)
    mode            str    "same-d"
    search_range    str    the scanned set of numerators
    S               str    the shared denominator, expanded ascii
    S_factored      str    the factored denominator
    klass           str    FULL | COLLAPSE | REPEATED | NONCYC
    T               [int] | null   the cyclotomic index set (null if non-cyclotomic)
    members         [int]  the numerators sharing S, ascending
    class_size      int
    pairing         [ {a: int, neg_inv: int, paired: bool} ]
                    each member, its image under b -> -b^{-1} mod d, and
                    whether that image is itself in the class
    sqrt_minus_one  [ {p: int, roots: [int]} ]
                    primes p with p^2 | d and i^2 = -1 (mod p), with the roots i

JSON schema, across mode:
    a, d            int
    mode            str    "across"
    dmax            int    the search bound on the modulus d'
    search_range    str    the scanned range of moduli
    S, S_factored, klass   as above, for a/d
    matches         [ {b: int, d: int, k: int, unit: str} ]
                    each fraction b/d' whose S matches up to the unit q^k
    match_count     int
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd

import sympy as sp

from . import formatter
from .denom import DenomDossier, denom_dossier, s_factored_ascii
from .rational import q


@dataclass(frozen=True)
class TwinClass:
    """The sharing class of one fraction a/d at its own modulus."""

    a: int
    d: int
    rep: DenomDossier  # the dossier of a/d, the shared denominator and its class
    members: list[int]
    pairing: list[tuple[int, int, bool]]  # (b, -b^{-1} mod d, image in class)
    sqrt_minus_one: list[tuple[int, list[int]]]  # (p, roots i with i^2 = -1 mod p)


@dataclass(frozen=True)
class TwinAcross:
    """The fractions at moduli up to a bound whose S matches a/d up to a unit."""

    a: int
    d: int
    dmax: int
    rep: DenomDossier
    matches: list[tuple[int, int, int]]  # (b, d', k) with S(b/d') = q^k S(a/d)


def _normalize(a: int, d: int) -> tuple[int, int]:
    frac = Fraction(int(a), int(d))
    if frac <= 0:
        raise ValueError("the fraction a/d must be positive")
    return frac.numerator, frac.denominator


def _involution(b: int, d: int) -> int:
    """The image of b under b -> -b^{-1} (mod d); b must be coprime to d."""
    return (-pow(b, -1, d)) % d


def _sqrt_minus_one(d: int) -> list[tuple[int, list[int]]]:
    """Primes p with p^2 | d and a square root of minus one mod p, with roots."""
    out: list[tuple[int, list[int]]] = []
    for p, e in sorted(sp.factorint(d).items()):
        p = int(p)
        if int(e) >= 2 and p % 4 == 1:
            roots = sorted(i for i in range(1, p) if (i * i) % p == p - 1)
            out.append((p, roots))
    return out


def twin_class(a: int, d: int) -> TwinClass:
    """The sharing class of a/d at modulus d, exact over Z[q].

    The representative dossier (a/d itself) is computed with full factoring for
    display; the scan over the other numerators groups by the coefficient tuple
    of S only, so it skips the costly factoring of each candidate.
    """
    a, d = _normalize(a, d)
    if d < 2:
        raise ValueError("the modulus d must be at least 2")
    rep = denom_dossier(a, d)
    key = tuple(rep.S.all_coeffs())
    members = [
        b
        for b in range(1, d)
        if gcd(b, d) == 1
        and tuple(denom_dossier(b, d, factor_cofactor=False).S.all_coeffs()) == key
    ]
    member_set = set(members)
    pairing = [(b, _involution(b, d), _involution(b, d) in member_set) for b in members]
    return TwinClass(
        a=a,
        d=d,
        rep=rep,
        members=members,
        pairing=pairing,
        sqrt_minus_one=_sqrt_minus_one(d),
    )


def _as_poly(p) -> sp.Poly:
    return p if isinstance(p, sp.Poly) else sp.Poly(sp.expand(p), q, domain="ZZ")


def unit_shift(p1, p2) -> int | None:
    """The integer k with q^k * p1 == p2, or None if no unit power relates them.

    Two polynomials differ by a unit power of q exactly when their ascending
    coefficient sequences are identical and every exponent is offset by the same
    k. Exact over Z[q]: only integer exponents and coefficients are compared.
    """
    t1 = sorted(_as_poly(p1).terms(), key=lambda t: t[0][0])
    t2 = sorted(_as_poly(p2).terms(), key=lambda t: t[0][0])
    if len(t1) != len(t2) or not t1:
        return None
    if [c for _, c in t1] != [c for _, c in t2]:
        return None
    shifts = {e2[0] - e1[0] for (e1, _), (e2, _) in zip(t1, t2)}
    if len(shifts) != 1:
        return None
    return int(shifts.pop())


def twin_across(a: int, d: int, dmax: int) -> TwinAcross:
    """Fractions at moduli 2..dmax whose S matches a/d up to a unit power of q."""
    a, d = _normalize(a, d)
    dmax = int(dmax)
    if dmax < 2:
        raise ValueError("the search bound dmax must be at least 2")
    rep = denom_dossier(a, d)
    matches: list[tuple[int, int, int]] = []
    for dp in range(2, dmax + 1):
        for b in range(1, dp):
            if gcd(b, dp) != 1:
                continue
            if dp == d and b == a:
                continue
            Sb = denom_dossier(b, dp, factor_cofactor=False).S
            k = unit_shift(rep.S, Sb)
            if k is not None:
                matches.append((b, dp, k))
    return TwinAcross(a=a, d=d, dmax=dmax, rep=rep, matches=matches)


# --------------------------------------------------------------------------
# Rendering, ASCII, through the shared formatter.
# --------------------------------------------------------------------------


def s_factored(rep: DenomDossier) -> str:
    """The shared denominator S factored, reusing the dossier renderer."""
    return s_factored_ascii(rep)


def index_set_ascii(rep: DenomDossier) -> str:
    if rep.is_cyclotomic_product:
        return "{" + ", ".join(str(e) for e in rep.index_set) + "}"
    return "not a cyclotomic product"


def members_ascii(members: list[int]) -> str:
    return "{" + ", ".join(str(m) for m in members) + "}"


def sqrt_minus_one_ascii(tc: TwinClass) -> str:
    """The square-root-of-minus-one note for the class, or an empty string."""
    if not tc.sqrt_minus_one:
        return ""
    parts = []
    for p, roots in tc.sqrt_minus_one:
        roots_str = "{" + ", ".join(str(r) for r in roots) + "}"
        parts.append(
            f"{p}^2 divides {tc.d} and "
            + formatter.congruence_ascii("i^2", "-1", p)
            + f" for i in {roots_str}"
        )
    return (
        "; ".join(parts)
        + ". A square root of minus one yields the pairing beyond a -> -a^{-1}."
    )


def twin_class_data(tc: TwinClass) -> dict:
    """The stable JSON object of the sharing class (schema in the docstring)."""
    rep = tc.rep
    return {
        "a": tc.a,
        "d": tc.d,
        "mode": "same-d",
        "search_range": "numerators b coprime to d with 1 <= b < d",
        "S": formatter.poly_ascii(rep.S.as_expr(), wrap=10**9),
        "S_factored": s_factored(rep),
        "klass": rep.klass,
        "T": rep.index_set if rep.is_cyclotomic_product else None,
        "members": list(tc.members),
        "class_size": len(tc.members),
        "pairing": [
            {"a": b, "neg_inv": n, "paired": paired} for b, n, paired in tc.pairing
        ],
        "sqrt_minus_one": [
            {"p": p, "roots": list(roots)} for p, roots in tc.sqrt_minus_one
        ],
    }


def twin_across_data(ta: TwinAcross) -> dict:
    """The stable JSON object of the across search (schema in the docstring)."""
    rep = ta.rep
    return {
        "a": ta.a,
        "d": ta.d,
        "mode": "across",
        "dmax": ta.dmax,
        "search_range": f"moduli d' with 2 <= d' <= {ta.dmax}, numerators coprime to d'",
        "S": formatter.poly_ascii(rep.S.as_expr(), wrap=10**9),
        "S_factored": s_factored(rep),
        "klass": rep.klass,
        "matches": [
            {"b": b, "d": dp, "k": k, "unit": formatter.q_power(k)}
            for b, dp, k in ta.matches
        ],
        "match_count": len(ta.matches),
    }
