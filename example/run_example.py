"""End-to-end demo for qreals.

Computes a few exact q-rationals and the leading Taylor coefficients of a few
q-deformed reals, then prints a short read-out of each. Run with:

    python run_example.py

The package must be importable first (pip install -e . from the repo root).
"""
from __future__ import annotations

import qreals
from qreals import coefficients as co


def main() -> None:
    print("Exact q-rationals (elements of Q(q)):")
    for p, s in [(3, 2), (1, 2), (19, 7)]:
        print(f"  [{p}/{s}]_q = {qreals.q_rational(p, s)}")

    N = 30
    print(f"\nq-deformed reals, first Taylor coefficients of [x]_q (N = {N}):")
    constants = [
        ("pi", "pi"),
        ("sqrt(2)", "sqrt(2)"),
        ("phi", "(1+sqrt(5))/2"),
        ("e", "E"),
    ]
    for label, x in constants:
        c = qreals.q_real_truncated(x, N)
        first_neg = co.first_negative_coefficient_index(c)
        print(f"  [{label}]_q")
        print(f"    first 12 coefficients: {c[:12]}")
        print(
            f"    first nonzero at q^{co.first_nonzero_coefficient_index(c)}, "
            f"first negative at {'q^' + str(first_neg) if first_neg >= 0 else 'none'}, "
            f"max |coefficient| = {co.coefficient_max_abs(c)}"
        )


if __name__ == "__main__":
    main()
