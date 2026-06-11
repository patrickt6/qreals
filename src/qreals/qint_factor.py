r"""Recover the multiset {a_1, ..., a_r} from a q-integer product P(q).

The q-integer uniqueness property: given

    P(q) = [a_1]_q [a_2]_q ... [a_r]_q,    a_i >= 2,

the multiset {a_1, ..., a_r} is determined by P(q). The proof is short and
constructive. Each q-integer factors over Z[q] as

    [n]_q = prod_{d | n, d > 1} Phi(d),

where Phi(d) is the d-th cyclotomic polynomial; the Phi(d) are irreducible and
distinct, and Z[q] is a UFD, so the exponent vector

    e_d = the exponent of Phi(d) in the factorisation of P(q)        (d >= 2)

is uniquely determined by P(q). Phi(d) divides [n]_q iff d | n, so

    e_d = sum_i [d | a_i] = sum_{n: d | n} m_n,

where m_n is the multiplicity of n in {a_i}. The classical "downward"
Mobius pair g(n) = sum_{d | n} f(d) <=> f(n) = sum_{d | n} mu(n/d) g(d)
inverts a sum over divisors; here e_d sums over multiples, so the matrix
v with rows v_n (entries [d | n]) is lower-triangular with ones on the
diagonal and its row-inverse is the "upward" Mobius transform:

    m_n = sum_{d in supp, n | d} mu(d / n) e_d,

with the sum ranging over d in the cyclotomic support (every d with e_d > 0).

If every m_n is a non-negative integer the multiset is the one with those
multiplicities; otherwise P(q) is not a product of q-integers. The
non-negativity check is what verifies the conjecture on the input: a fitted
multiset and a recomputed product must agree with P bit for bit.

The factorial identity

    [n]_q! = [n]_q [n-1]_q ... [2]_q

has, in the recovered multiset, exactly one copy of each integer 2, 3, ..., n.
Equivalently, the cyclotomic exponent of Phi(d) in [n]_q! is floor(n / d) for
d >= 2 (since Phi(d) divides [m]_q for exactly the multiples of d in 2..n),
which the test suite cross-checks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import sympy as sp
from sympy import cyclotomic_poly, mobius

from .rational import q, q_int


@dataclass(frozen=True)
class QIntFactor:
    """The structured record returned by qint_factor.

    Fields:
        status: "product" if P(q) factors as a product of q-integers >= 2,
            "not_product" otherwise.
        multiset: list of (n, multiplicity) pairs sorted by n when status is
            "product"; empty list when status is "not_product".
        cyclotomic_exponents: dict d -> e_d (d >= 2, e_d > 0) reading the
            cyclotomic factorisation of P over Z[q].
        mobius_transform: dict n -> m_n for every n appearing in the divisor
            span of the support; values are integers, possibly negative when
            status is "not_product".
        obstruction_indices: dict n -> m_n for the indices that block the
            decomposition; empty when status is "product". The keys are
            exactly the n with m_n < 0 or m_n a non-integer rational.
        leading_unit: the leading content of P that is not a cyclotomic factor
            (1 for a clean q-integer product, otherwise an explanation).
    """

    status: str
    multiset: list[tuple[int, int]]
    cyclotomic_exponents: dict[int, int]
    mobius_transform: dict[int, int]
    obstruction_indices: dict[int, int]
    leading_unit: sp.Expr


def _coerce_poly(P) -> sp.Poly:
    """Accept a sympy expression, Poly, or coefficient list and return Poly(P, q)."""
    if isinstance(P, sp.Poly):
        if P.gen != q:
            return sp.Poly(P.as_expr(), q)
        return P
    if isinstance(P, (list, tuple)):
        # Coefficient list: P[0] + P[1] q + P[2] q^2 + ...
        expr = sum(sp.Integer(c) * q**i for i, c in enumerate(P))
        return sp.Poly(expr, q)
    return sp.Poly(sp.sympify(P), q)


def _cyclotomic_exponents(P: sp.Poly) -> tuple[dict[int, int], sp.Expr]:
    """Strip cyclotomic factors Phi(d) (d >= 2) from P by trial division.

    Returns (exponents, leftover). exponents maps d -> e_d for d such that
    Phi(d) divides P^k for some k >= 1. leftover is the unit content that did
    not match any cyclotomic factor; for a clean q-integer product the
    leftover is +-1.
    """
    expr = sp.expand(P.as_expr())
    poly = sp.Poly(expr, q)
    if poly.is_zero:
        raise ValueError("P is the zero polynomial; not a product of q-integers")

    deg = poly.degree()
    exponents: dict[int, int] = {}

    # Largest d we ever need: Phi(d) has degree phi(d) >= 1, so d <= deg(P) + 1
    # is more than enough (degree of [n]_q is n - 1, so n - 1 <= deg <= n - 1
    # for the largest factor; using deg + 2 gives a safety margin).
    d_max = deg + 2

    for d in range(2, d_max + 1):
        phi_d = sp.Poly(cyclotomic_poly(d, q), q)
        # Trial-divide by Phi(d) while the division is exact.
        while True:
            quotient, remainder = sp.div(poly, phi_d, q)
            if remainder.is_zero:
                poly = quotient
                exponents[d] = exponents.get(d, 0) + 1
                if poly.degree() == 0:
                    break
            else:
                break
        if poly.degree() == 0:
            break

    leftover = sp.simplify(poly.as_expr())
    return exponents, leftover


def _mobius_inverse(exponents: dict[int, int]) -> dict[int, int]:
    """Apply upward Mobius inversion on the cyclotomic support.

    Each a_i in the multiset is a multiple of every d in its divisor set, and
    Phi(a_i) appears with multiplicity one in [a_i]_q, so the support
    {d: e_d > 0} contains every a_i itself. Candidate n therefore range over
    the support. The inversion of e_d = sum_{n: d | n} m_n on the
    divisibility poset is the dual ("upward") pair:

        m_n = sum_{d in supp, n | d} mu(d / n) e_d.
    """
    if not exponents:
        return {}
    support = sorted(exponents)
    transform: dict[int, int] = {}
    for n in support:
        total = 0
        for d in support:
            if d % n != 0:
                continue
            mu = int(mobius(d // n))
            if mu == 0:
                continue
            total += mu * exponents[d]
        transform[n] = total
    return transform


def qint_factor(P) -> QIntFactor:
    r"""Recover the multiset of integer factors from a q-integer product P(q).

    P may be a sympy expression in q, a sympy Poly in q, or a coefficient list
    [c_0, c_1, ...] with P = c_0 + c_1 q + c_2 q^2 + ... Returns a QIntFactor
    record:

      * status="product" with the recovered multiset when P = prod_i [a_i]_q,
        a_i >= 2; the round-trip product prod_i [a_i]_q is recomputed and
        compared with P to refuse a fitted but wrong answer.
      * status="not_product" with the negative Mobius indices marked in
        obstruction_indices when P does not factor as a q-integer product;
        no multiset is returned in that case.

    The implementation is the explicit algorithm from the proof: cyclotomic
    factorisation over Z[q] then Mobius inversion on the divisibility poset.
    """
    poly = _coerce_poly(P)

    if poly.is_zero:
        raise ValueError("P is the zero polynomial; not a product of q-integers")

    exponents, leftover = _cyclotomic_exponents(poly)

    # The leftover must be a unit (a non-zero constant) for P to lie in the
    # cyclotomic submonoid; if not, it is at most a not-product.
    if sp.simplify(leftover) == 0:
        return QIntFactor(
            status="not_product",
            multiset=[],
            cyclotomic_exponents=dict(exponents),
            mobius_transform={},
            obstruction_indices={},
            leading_unit=sp.Integer(0),
        )

    transform = _mobius_inverse(exponents)

    obstruction = {n: m for n, m in transform.items() if m < 0}
    if obstruction or sp.nsimplify(leftover) not in (sp.Integer(1), sp.Integer(-1)):
        return QIntFactor(
            status="not_product",
            multiset=[],
            cyclotomic_exponents=dict(exponents),
            mobius_transform=dict(transform),
            obstruction_indices=dict(obstruction),
            leading_unit=sp.simplify(leftover),
        )

    multiset = [(n, m) for n, m in sorted(transform.items()) if m > 0]

    # Round-trip cross-check: recompute the product and demand symbolic equality.
    recovered = sp.Integer(1)
    for n, mult in multiset:
        recovered = recovered * (q_int(n) ** mult)
    if leftover != sp.Integer(1):
        recovered = -recovered
    if sp.simplify(sp.expand(recovered) - poly.as_expr()) != 0:
        # The Mobius result was non-negative integer, but the recomputed product
        # does not match P; refuse to claim a multiset and report the failure
        # as an obstruction with the surviving discrepancy.
        return QIntFactor(
            status="not_product",
            multiset=[],
            cyclotomic_exponents=dict(exponents),
            mobius_transform=dict(transform),
            obstruction_indices={n: m for n, m in transform.items() if m != 0},
            leading_unit=sp.simplify(leftover),
        )

    return QIntFactor(
        status="product",
        multiset=multiset,
        cyclotomic_exponents=dict(exponents),
        mobius_transform=dict(transform),
        obstruction_indices={},
        leading_unit=sp.simplify(leftover),
    )


def qint_factor_peeling(P) -> QIntFactor:
    r"""Recover the multiset of integer factors via a root-order peeling argument.

    Independent algorithm to qint_factor, by a root-order argument:
    the roots of [n]_q are precisely the n-th roots of unity other than 1, so
    among the roots of P the highest multiplicative order coincides with the
    largest a_i in the multiset. Equivalently, max{d : e_d > 0} is the largest
    factor and [max_d]_q must divide P; the multiset is built by repeatedly
    peeling off [d*]_q for d* the current largest support index.

    Operating on the cyclotomic exponent vector (e_d) shared with qint_factor:

      while any e_d > 0:
        d* := max{d : e_d > 0}
        if there exists d' | d*, d' >= 2, with e_{d'} == 0:
          return not_product, obstruction = d*       # [d*]_q does not divide
        else:
          for every d' | d*, d' >= 2:
            e_{d'} -= 1
          append d* to multiset

    Returns the same QIntFactor record shape as qint_factor. mobius_transform is
    left empty (peeling does not compute it). obstruction_indices, on failure,
    maps the candidate d* to its negative slack -1 to mark which step failed.
    """
    poly = _coerce_poly(P)

    if poly.is_zero:
        raise ValueError("P is the zero polynomial; not a product of q-integers")

    exponents, leftover = _cyclotomic_exponents(poly)

    if sp.simplify(leftover) == 0:
        return QIntFactor(
            status="not_product",
            multiset=[],
            cyclotomic_exponents=dict(exponents),
            mobius_transform={},
            obstruction_indices={},
            leading_unit=sp.Integer(0),
        )

    if sp.nsimplify(leftover) not in (sp.Integer(1), sp.Integer(-1)):
        return QIntFactor(
            status="not_product",
            multiset=[],
            cyclotomic_exponents=dict(exponents),
            mobius_transform={},
            obstruction_indices={},
            leading_unit=sp.simplify(leftover),
        )

    working = dict(exponents)
    recovered: list[int] = []

    while any(v > 0 for v in working.values()):
        d_star = max(d for d, v in working.items() if v > 0)
        divisors = [d for d in sp.divisors(d_star) if d >= 2]
        missing = [d for d in divisors if working.get(d, 0) == 0]
        if missing:
            return QIntFactor(
                status="not_product",
                multiset=[],
                cyclotomic_exponents=dict(exponents),
                mobius_transform={},
                obstruction_indices={d_star: -1},
                leading_unit=sp.simplify(leftover),
            )
        for d in divisors:
            working[d] -= 1
        recovered.append(d_star)

    counts = Counter(recovered)
    multiset = sorted(counts.items())

    round_trip = sp.Integer(1)
    for n, mult in multiset:
        round_trip = round_trip * (q_int(n) ** mult)
    if leftover != sp.Integer(1):
        round_trip = -round_trip
    if sp.simplify(sp.expand(round_trip) - poly.as_expr()) != 0:
        return QIntFactor(
            status="not_product",
            multiset=[],
            cyclotomic_exponents=dict(exponents),
            mobius_transform={},
            obstruction_indices={n: m for n, m in counts.items()},
            leading_unit=sp.simplify(leftover),
        )

    return QIntFactor(
        status="product",
        multiset=multiset,
        cyclotomic_exponents=dict(exponents),
        mobius_transform={},
        obstruction_indices={},
        leading_unit=sp.simplify(leftover),
    )


def qint_product(multiset: Iterable[int]) -> sp.Expr:
    """Multiply [a_i]_q over a multiset of integers >= 2 and return the expanded sum.

    Helper: lets callers spot-check the algorithm on a known multiset, and lets
    the scan script enumerate products without depending on the q_int internal.
    """
    result = sp.Integer(1)
    for a in multiset:
        a = int(a)
        if a < 2:
            raise ValueError(f"multiset entries must be >= 2, got {a}")
        result = result * q_int(a)
    return sp.expand(result)


def canonical_multiset(multiset: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Sorted (value, multiplicity) tuple form of a multiset, for hashing."""
    counts = Counter(int(a) for a in multiset)
    return tuple(sorted(counts.items()))
