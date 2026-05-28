"""Tests for the bihomographic q-Gosper engine.

The engine is the independent algorithm that verifies q_add / q_mul. These tests
pin it to the package ground truth (q_rational, q_real_truncated) and check the
structural identities the port relies on (the q-block substitution and the
commutation of the two ingestion sides). See docs/CORRECTNESS.md.
"""

from fractions import Fraction

import sympy as sp

from qreals import gosper_coeffs, q_gosper, q_rational, q_real_truncated
from qreals.gosper import kron, q_block, q_cf, q_convergent_matrix, q_real_rational
from qreals.rational import q, q_int, q_int_qinv

RATIONALS = [(3, 2), (27, 19), (13, 5), (7, 3), (22, 7), (5, 8), (41, 29)]


def test_engine_reproduces_q_rational():
    # The one-variable block product equals q_rational coefficient for coefficient.
    for p, s in RATIONALS:
        engine = q_real_rational(Fraction(p, s))
        assert sp.simplify(engine - q_rational(p, s)) == 0


def test_engine_add_equals_series_sum_brute_force():
    N = 16
    for p, s, a, b in [(3, 2, 5, 2), (7, 3, 22, 7), (8, 3, 9, 4)]:
        engine = gosper_coeffs(Fraction(p, s), Fraction(a, b), "add", N)
        brute = [
            u + v
            for u, v in zip(
                q_real_truncated(f"{p}/{s}", N), q_real_truncated(f"{a}/{b}", N)
            )
        ]
        assert engine == brute


def test_engine_mul_equals_series_product_brute_force():
    N = 16
    p, s, a, b = 3, 2, 5, 2
    engine = gosper_coeffs(Fraction(p, s), Fraction(a, b), "mul", N)
    cx, cy = q_real_truncated(f"{p}/{s}", N), q_real_truncated(f"{a}/{b}", N)
    expect = [0] * N
    for i, u in enumerate(cx):
        for j, v in enumerate(cy):
            if i + j < N:
                expect[i + j] += u * v
    assert engine == expect


def test_q_block_implements_the_mgo_substitution():
    # An odd-position block sends u -> [t]_q + q^t/u; even -> [t]_{1/q} + q^{-t}/u.
    u = sp.Symbol("u")
    for t in range(1, 6):
        odd = q_block(0, t)
        sub_odd = (odd[0, 0] * u + odd[0, 1]) / (odd[1, 0] * u + odd[1, 1])
        assert sp.simplify(sub_odd - (q_int(t) + q**t / u)) == 0
        even = q_block(1, t)
        sub_even = (even[0, 0] * u + even[0, 1]) / (even[1, 0] * u + even[1, 1])
        assert sp.simplify(sub_even - (q_int_qinv(t) + q ** (-t) / u)) == 0


def test_ingestion_sides_commute():
    # P_q = A_q (x) I and Q_q = I (x) A_q act on different tensor factors, so they
    # commute; this is the structural fact the Kronecker factorisation rests on.
    px = kron(q_block(0, 2), sp.eye(2))
    qy = kron(sp.eye(2), q_block(0, 3))
    assert sp.simplify(px * qy - qy * px) == sp.zeros(4, 4)


def test_first_column_is_tensor_of_convergent_vectors():
    # (prod P)(prod Q) = M_x (x) M_y, so the engine's leading ratio is z([x]_q,[y]_q).
    cfx, cfy = q_cf(Fraction(3, 2)), q_cf(Fraction(7, 3))
    mx, my = q_convergent_matrix(cfx), q_convergent_matrix(cfy)
    pprod = sp.eye(4)
    for i, a in enumerate(cfx):
        pprod = pprod * kron(q_block(i, a), sp.eye(2))
    qprod = sp.eye(4)
    for j, b in enumerate(cfy):
        qprod = qprod * kron(sp.eye(2), q_block(j, b))
    assert sp.simplify(pprod * qprod - kron(mx, my)) == sp.zeros(4, 4)


def test_engine_add_is_x_plus_y_series_not_q_of_real_sum():
    # constant term 2 (sum of two q-series), not the 1 of [x+y]_q.
    val = q_gosper(Fraction(3, 2), Fraction(5, 2), "add")
    assert val.subs(q, 0) == 2
