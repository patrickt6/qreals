r"""The exact divisibility microscope of S(q) at one cyclotomic index e.

For a rational a/d in lowest terms the denominator S(q) of [a/d]_q is a monic
integer polynomial with S(0) = 1. This module answers one question exactly:
does the cyclotomic polynomial of index e divide S over Z[q]? Equivalently,
is the primitive e-th root of unity a root of S? The answer is a yes/no
verdict plus the recurrence trace that produces it, every value carried in the
power basis of Z[zeta_e] as an integer coefficient vector, so the whole trace
is reproducible by hand.

The trace replays the q-continuant recurrence of the continued fraction of
a/d: the partial denominators S_0, S_1, ..., S_n are the denominators of the
successive convergents, built from the same transfer-matrix blocks the engine
uses for S, reduced to lowest terms so each is monic with S_k(0) = 1. The last
partial denominator is S itself (cross-checked against the denominator dossier
in the test suite). Each S_k is evaluated at q = zeta_e by reduction modulo the
cyclotomic polynomial of index e, which is the minimal polynomial of zeta_e, so
the value is an integer vector against the basis 1, zeta_e, ..., zeta_e^{m-1}
with m = phi(e). The verdict is positive exactly when the final vector is zero.

A determinant identity is carried as a sanity row: the transfer matrix has
determinant (-1)^n q^E with E the sum of the continued-fraction entries, and
the same value is recomputed from the explicit matrix entries; the two agree
exactly in Z[zeta_e], confirming the arithmetic.

Everything is exact over Z and Z[zeta_e]: integer coefficient vectors, no
floating point in any value the user sees.

JSON schema (the --json output of `qreals why`, keys are stable):
    a, d            int    the fraction in lowest terms (d > 0)
    e               int    the cyclotomic index
    phi_e           int    phi(e), the degree of the power basis
    basis           str    the declared power basis 1, zeta_e, ...
    cf              [int]  the even-length continued fraction of a/d
    verdict         bool   the cyclotomic polynomial of e divides S over Z[q]
    final_vector    [int]  S(zeta_e) in the power basis (zero iff verdict)
    trace           [ {k: int, cf_entry: int, vector: [int]} ]
                    one row per partial denominator S_k, k = 1..n
    determinant     { vector: [int], closed_form_vector: [int], ok: bool }
                    det M(zeta_e) the long way and the closed form (-1)^n
                    zeta_e^E, and whether they match
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from . import formatter
from .continued_fraction import make_even_length
from .denom import denom_dossier
from .rational import q


@dataclass(frozen=True)
class TraceStep:
    """One partial denominator S_k and its value at zeta_e."""

    k: int
    cf_entry: int
    vector: list[int]


@dataclass(frozen=True)
class WhyDossier:
    """The divisibility microscope of S(q) at one index e."""

    a: int
    d: int
    e: int
    phi_e: int
    cf: list[int]
    S: sp.Poly
    verdict: bool
    trace: list[TraceStep]
    det_vector: list[int]
    det_closed_vector: list[int]
    det_ok: bool

    @property
    def final_vector(self) -> list[int]:
        return self.trace[-1].vector if self.trace else [0] * self.phi_e


def parse_index(text: str) -> int:
    """Read the cyclotomic index e (a positive integer) from text."""
    e = int(str(text).strip())
    if e < 1:
        raise ValueError("the cyclotomic index e must be at least 1")
    return e


def _qint_poly(n: int) -> sp.Poly:
    """[n]_q for n >= 0 as a Poly over ZZ."""
    return sp.Poly([1] * int(n) if n > 0 else [0], q, domain="ZZ")


def _blocks(cf: list[int]) -> list[list[list[sp.Poly]]]:
    """The transfer-matrix blocks of the even-length continued fraction.

    These are the same blocks the engine multiplies to build S (each
    even-position block is scaled by q^{a_i} to clear the q^{-1} entries),
    so the partial products reproduce the engine's denominator exactly.
    """
    one = sp.Poly(1, q, domain="ZZ")
    zero = sp.Poly(0, q, domain="ZZ")
    out: list[list[list[sp.Poly]]] = []
    for i, a in enumerate(cf):
        if (i + 1) % 2 == 1:
            out.append([[_qint_poly(a), sp.Poly(q**a, q, domain="ZZ")], [one, zero]])
        else:
            out.append(
                [
                    [sp.Poly(q, q, domain="ZZ") * _qint_poly(a), one],
                    [sp.Poly(q**a, q, domain="ZZ"), zero],
                ]
            )
    return out


def _matmul(m, blk):
    return [
        [
            m[0][0] * blk[0][0] + m[0][1] * blk[1][0],
            m[0][0] * blk[0][1] + m[0][1] * blk[1][1],
        ],
        [
            m[1][0] * blk[0][0] + m[1][1] * blk[1][0],
            m[1][0] * blk[0][1] + m[1][1] * blk[1][1],
        ],
    ]


def _reduced_denominator(num: sp.Poly, den: sp.Poly) -> sp.Poly:
    """The convergent denominator reduced to lowest terms and made monic."""
    g = num.gcd(den)
    num, den = num.exquo(g), den.exquo(g)
    lc = den.LC()
    if lc != 1:
        den = den.quo_ground(lc)
    return den


def _power_basis_vector(poly: sp.Poly, phi: sp.Poly, m: int) -> list[int]:
    """The value of poly at zeta_e in the power basis 1, zeta_e, ..., zeta_e^{m-1}.

    Reduction modulo the cyclotomic polynomial phi of index e (the minimal
    polynomial of zeta_e) gives an integer remainder of degree below m; its
    ascending coefficients are the basis vector. Exact over Z, no floats.
    """
    rem = poly.rem(phi)
    vec = [0] * m
    for monom, coeff in rem.terms():
        vec[monom[0]] = int(coeff)
    return vec


def why_dossier(a: int, d: int, e: int) -> WhyDossier:
    """The divisibility microscope of S(q) at index e, exact over Z[zeta_e]."""
    e = parse_index(e)
    frac = Fraction(int(a), int(d))
    if frac <= 0:
        raise ValueError("the fraction a/d must be positive")
    a, d = frac.numerator, frac.denominator

    dossier = denom_dossier(a, d, factor_cofactor=False)
    S = dossier.S
    cf = make_even_length([int(t) for t in sp.continued_fraction(sp.Rational(a, d))])

    phi = sp.Poly(sp.cyclotomic_poly(e, q), q, domain="ZZ")
    m = int(phi.degree())

    one = sp.Poly(1, q, domain="ZZ")
    zero = sp.Poly(0, q, domain="ZZ")
    matrix = [[one, zero], [zero, one]]
    trace: list[TraceStep] = []
    for i, blk in enumerate(_blocks(cf)):
        matrix = _matmul(matrix, blk)
        den = _reduced_denominator(matrix[0][0], matrix[1][0])
        trace.append(
            TraceStep(
                k=i + 1,
                cf_entry=int(cf[i]),
                vector=_power_basis_vector(den, phi, m),
            )
        )

    verdict = all(c == 0 for c in trace[-1].vector) if trace else False

    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    sign = -1 if len(cf) % 2 else 1
    closed = sp.Poly(sign * q ** sum(cf), q, domain="ZZ")
    det_vector = _power_basis_vector(det, phi, m)
    det_closed_vector = _power_basis_vector(closed, phi, m)

    return WhyDossier(
        a=a,
        d=d,
        e=e,
        phi_e=m,
        cf=cf,
        S=S,
        verdict=verdict,
        trace=trace,
        det_vector=det_vector,
        det_closed_vector=det_closed_vector,
        det_ok=det_vector == det_closed_vector,
    )


# --------------------------------------------------------------------------
# Rendering, ASCII, through the shared formatter.
# --------------------------------------------------------------------------


def verdict_line(dossier: WhyDossier) -> str:
    """The one-line verdict: the cyclotomic polynomial of e divides S or not."""
    relation = "divides S" if dossier.verdict else "does not divide S"
    return f"{formatter.phi_label(dossier.e)} {relation}"


def basis_line(dossier: WhyDossier) -> str:
    """The declared power basis of Z[zeta_e], with its dimension."""
    return (
        f"{formatter.zeta_basis_ascii(dossier.e, dossier.phi_e)}"
        f"  (m = phi({dossier.e}) = {dossier.phi_e})"
    )


def why_data(dossier: WhyDossier) -> dict:
    """The stable JSON object of the dossier (schema in the module docstring)."""
    p = dossier
    return {
        "a": p.a,
        "d": p.d,
        "e": p.e,
        "phi_e": p.phi_e,
        "basis": formatter.zeta_basis_ascii(p.e, p.phi_e),
        "cf": list(p.cf),
        "verdict": p.verdict,
        "final_vector": list(p.final_vector),
        "trace": [
            {"k": s.k, "cf_entry": s.cf_entry, "vector": list(s.vector)}
            for s in p.trace
        ],
        "determinant": {
            "vector": list(p.det_vector),
            "closed_form_vector": list(p.det_closed_vector),
            "ok": p.det_ok,
        },
    }
