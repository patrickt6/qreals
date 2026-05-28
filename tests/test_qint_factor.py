"""Tests for qint_factor: the Mobius reconstructor for q-integer products.

Charles's uniqueness conjecture (2026-05-26): given P(q) = [a_1]_q ... [a_r]_q
with a_i >= 2, the multiset {a_i} is determined by P. The reconstructor
factors P over the cyclotomic monoid and Mobius-inverts on the divisibility
poset to recover the multiset. The tests below cover the worked cases from
the PDF, the [n]_q! factorial identity for n = 2..8, fifty random multisets,
and the obstruction case where P is not a q-integer product.
"""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import combinations_with_replacement

import pytest
import sympy as sp

from qreals import canonical_multiset, q_int, qint_factor, qint_product


def _product(multiset):
    """The polynomial prod [a_i]_q, fully expanded."""
    return sp.expand(qint_product(multiset))


def _multiset_to_pairs(multiset):
    """Convert a flat multiset list into the (value, multiplicity) form."""
    return [(a, m) for a, m in canonical_multiset(multiset)]


# the worked cases from QNumbersUniqueness.pdf -----------------------------------

WORKED_CASES = [
    [6],
    [2, 3],
    [2, 6],
    [3, 4],
    [2, 2, 3],
    [12],
]


@pytest.mark.parametrize("multiset", WORKED_CASES)
def test_worked_cases_round_trip(multiset):
    """Each worked multiset reconstructs from its q-integer product."""
    P = _product(multiset)
    result = qint_factor(P)
    assert result.status == "product"
    assert result.multiset == _multiset_to_pairs(multiset)
    # Round-trip cross-check is implicit in qint_factor, but spell it out:
    recovered = _product([a for a, m in result.multiset for _ in range(m)])
    assert sp.expand(recovered - P) == 0


def test_six_vs_two_three_are_distinguished():
    """[6]_q != [2]_q [3]_q, and the reconstructor refuses to confuse them."""
    P6 = _product([6])
    P23 = _product([2, 3])
    assert sp.expand(P6 - P23) != 0
    r6 = qint_factor(P6)
    r23 = qint_factor(P23)
    assert r6.multiset == [(6, 1)]
    assert r23.multiset == [(2, 1), (3, 1)]


def test_two_six_vs_three_four_are_distinguished():
    """{2, 6} and {3, 4} give different q-products and recover correctly."""
    P26 = _product([2, 6])
    P34 = _product([3, 4])
    assert sp.expand(P26 - P34) != 0
    assert qint_factor(P26).multiset == [(2, 1), (6, 1)]
    assert qint_factor(P34).multiset == [(3, 1), (4, 1)]


def test_three_four_vs_two_two_three_are_distinguished():
    """{3, 4} and {2, 2, 3}: [4]_q = [2]_q (1 + q^2), not [2]_q^2."""
    P34 = _product([3, 4])
    P223 = _product([2, 2, 3])
    assert sp.expand(P34 - P223) != 0
    assert qint_factor(P34).multiset == [(3, 1), (4, 1)]
    assert qint_factor(P223).multiset == [(2, 2), (3, 1)]


# the [n]_q! factorial identity --------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_q_factorial_recovery(n):
    """[n]_q! = [2]_q [3]_q ... [n]_q recovers the multiset {2, 3, ..., n}."""
    multiset = list(range(2, n + 1))
    P = _product(multiset)
    result = qint_factor(P)
    assert result.status == "product"
    assert result.multiset == _multiset_to_pairs(multiset)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_q_factorial_cyclotomic_floor_law(n):
    """Phi_d in [n]_q! has exponent floor(n / d) for every d >= 2.

    Since Phi_d divides [m]_q iff d | m, the exponent of Phi_d in [n]_q! is
    the count of multiples of d in {2, 3, ..., n}, which equals floor(n / d)
    for d >= 2 (the multiple m = d itself sits in 2..n iff d <= n).
    """
    multiset = list(range(2, n + 1))
    P = _product(multiset)
    result = qint_factor(P)
    for d in range(2, n + 1):
        expected = n // d
        assert result.cyclotomic_exponents.get(d, 0) == expected, (
            f"Phi_{d} in [{n}]_q!: expected floor({n}/{d}) = {expected}, "
            f"got {result.cyclotomic_exponents.get(d, 0)}"
        )


# fifty random multisets ---------------------------------------------------------


def _random_multisets(rng, count, a_max, r_max):
    """Sample `count` distinct multisets with entries in 2..a_max and r <= r_max."""
    seen = set()
    out = []
    while len(out) < count:
        r = rng.randint(1, r_max)
        ms = sorted(rng.randint(2, a_max) for _ in range(r))
        key = tuple(ms)
        if key in seen:
            continue
        seen.add(key)
        out.append(ms)
    return out


def test_fifty_random_multisets_round_trip():
    """Fifty pseudo-random multisets recover to themselves."""
    rng = random.Random(202605260000)
    samples = _random_multisets(rng, count=50, a_max=15, r_max=5)
    for ms in samples:
        P = _product(ms)
        result = qint_factor(P)
        assert result.status == "product", f"failed on {ms}: {result}"
        assert result.multiset == _multiset_to_pairs(ms), (
            f"round-trip mismatch on {ms}: got {result.multiset}"
        )


# obstruction cases (not a q-integer product) ------------------------------------


def test_phi_4_alone_is_not_a_product():
    """Phi_4(q) on its own is not a q-integer: [4]_q = Phi_2 Phi_4 carries Phi_2."""
    phi4 = sp.cyclotomic_poly(4, sp.Symbol("q"))
    result = qint_factor(phi4)
    assert result.status == "not_product"
    assert result.multiset == []


def test_phi_3_phi_4_alone_is_not_a_product():
    """Phi_3 Phi_4 misses the Phi_2 that [4]_q = Phi_2 Phi_4 carries."""
    qsym = sp.Symbol("q")
    P = sp.cyclotomic_poly(3, qsym) * sp.cyclotomic_poly(4, qsym)
    result = qint_factor(P)
    assert result.status == "not_product"
    assert result.multiset == []


def test_phi_6_alone_is_not_a_product():
    """Phi_6 alone is not a q-integer: [6]_q = Phi_2 Phi_3 Phi_6 also carries Phi_2 Phi_3."""
    phi6 = sp.cyclotomic_poly(6, sp.Symbol("q"))
    result = qint_factor(phi6)
    assert result.status == "not_product"
    assert result.multiset == []


# the conjecture on a small exhaustive grid --------------------------------------


def test_no_collision_on_small_grid():
    """Every multiset with a_i in 2..7, r <= 3 maps to a distinct product."""
    a_max = 7
    r_max = 3
    products: dict[bytes, tuple] = {}
    count = 0
    for r in range(1, r_max + 1):
        for ms in combinations_with_replacement(range(2, a_max + 1), r):
            P = sp.expand(_product(list(ms)))
            key = sp.Poly(P, sp.Symbol("q")).all_coeffs()
            key = tuple(int(c) for c in key)
            assert key not in products or products[key] == ms, (
                f"collision: {ms} and {products[key]} share the same product"
            )
            products[key] = ms
            count += 1
    assert count >= 50  # the grid is large enough to be interesting
