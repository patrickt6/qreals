"""Lemmas about q-real Laurent expansions, as first-class functions.

Four lemmas about the Laurent expansion of [x]_q, each turned into a tested
function:

  1. Integer-part prefix. For real x with t = floor(x), the Laurent expansion
     of [x]_q opens with [t]_q = 1 + q + ... + q^{t-1} and then a forced 0
     coefficient at q^t, the clean seam between integer and fractional parts.
     `integer_part_prefix`. (note 1)

  2. Convergents pin down coefficients. For a positive irrational x with
     continued fraction [a_1, a_2, ...] and partial sum S_n = a_1 + ... + a_n,
     the n-th convergent computes [x]_q correctly on a known number of low
     powers. `coeffs_locked_by_convergent`. (note 2)

  3. The MGO positive form. [x]_q is the Laurent expansion produced by the MGO
     continued-fraction formula; `mgo_laurent` reads its coefficients off to a
     requested order. (note 3)

  4. The shift relations. [x-1]_q = ([x]_q - 1)/q lowers the argument by one
     and [x+1]_q = q[x]_q + 1 raises it. `shift_down` and `shift_up` are the
     tool that moves a coefficient question between unit intervals. (note 5)

CLI:
    python -m qreals.expansions pi --order 12
prints the Laurent expansion of [x]_q to the requested order plus its forced
integer-part prefix.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from ._parsing import parse_real
from .truncated import q_real_truncated


def integer_part_prefix(x: str | int | float | sp.Expr) -> list[int]:
    """The forced opening block [floor(x)]_q + 0*q^floor(x) of [x]_q (note 1).

    For real x with t = floor(x), the Laurent expansion of [x]_q begins with t
    coefficients all equal to 1 (spelling out [t]_q) followed by a 0 coefficient
    at q^t. This returns that prefix as a list of t + 1 integers,
    [1, 1, ..., 1, 0]; the fractional part of x can perturb only powers q^{t+1}
    and higher.

    Args:
        x: a sympy-parseable real, e.g. "22/7", "pi", 3, "pi-2".

    Returns:
        [1] * t + [0], where t = floor(x).
    """
    t = int(sp.floor(parse_real(x)))
    if t < 0:
        raise ValueError(f"integer-part lemma is stated for x >= 0, got floor(x) = {t}")
    return [1] * t + [0]


def coeffs_locked_by_convergent(cf_terms: Sequence[int], n: int) -> tuple[int, int]:
    """How many Laurent coefficients the n-th convergent of x locks in (note 2).

    With continued fraction x = [a_1, a_2, ...] and partial sum
    S_n = a_1 + ... + a_n, the n-th convergent x_n agrees with [x]_q on every
    power strictly below q^{S_n - 1}. Returns (S_n, count), where
    count = S_n - 1 is the number of locked coefficients, i.e. the powers
    q^0, ..., q^{S_n - 2}. The first power that may differ is q^{S_n - 1}.

    Note on the off-by-one. Note 2 phrased the cutoff as "below q^{S_n}", which
    would make count = S_n. Direct computation refutes that: for x = pi,
    a_1 = 3, a_2 = 7, S_2 = 10, the convergent [3,7] = 22/7 gives
    [22/7]_q = 1 + q + q^2 + q^9 + ... whereas the true [pi]_q has a 0 at q^9
    (its next term is at q^10). So [22/7]_q and [pi]_q already differ at
    q^9 = q^{S_2 - 1}: agreement holds only below q^{S_n - 1}, locking S_n - 1
    coefficients, not S_n. This matches the existing `continued_fraction`
    docstring and is the resolution of the numerical tension flagged in note 5.

    Args:
        cf_terms: the continued-fraction partial quotients [a_1, a_2, ...].
        n: convergent index, 1 <= n <= len(cf_terms).

    Returns:
        (S_n, count) with S_n = a_1 + ... + a_n and count = S_n - 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    terms = [int(a) for a in cf_terms]
    if n > len(terms):
        raise ValueError(
            f"need at least n = {n} continued-fraction terms, got {len(terms)}"
        )
    s_n = sum(terms[:n])
    if s_n < 1:
        raise ValueError(
            f"convergent locking is stated for positive x with partial sum "
            f"S_n >= 1; got S_n = {s_n} (the lemma does not apply to x <= 0, "
            f"and for 0 < x < 1 the leading quotient is 0, so take n >= 2)"
        )
    return s_n, s_n - 1


def mgo_laurent(x: str | int | float | sp.Expr, order: int) -> list[int]:
    """Laurent coefficients of [x]_q through q^order, via the MGO formula (note 3).

    Evaluates the MGO positive-form continued fraction for x over the
    truncated-series kernel and reads off the coefficients of q^0 up to and
    including q^order. Returns order + 1 integers [c_0, ..., c_order] with c_k
    the coefficient of q^k. For x >= 1 the series has valuation 0 (c_0 = 1);
    for 0 < x < 1 it has valuation 1 (c_0 = 0).

    Args:
        x: a sympy-parseable real, e.g. "22/7", "pi", "sqrt(2)".
        order: highest power q^order to return; must be >= 0.

    Returns:
        [c_0, c_1, ..., c_order].
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    return q_real_truncated(str(x), order + 1)


def shift_down(coeffs: Sequence[int]) -> list[int]:
    """Coefficients of [x-1]_q = ([x]_q - 1)/q from those of [x]_q (note 5).

    Subtracts 1 from the constant term and divides by q, i.e. drops the
    constant coefficient and shifts every higher power down by one. This is
    exact only when the constant term is 1, which holds for any [x]_q with
    x >= 1 (the integer-part lemma forces it); a different constant term means
    the input is not such a series, so the division by q would produce a
    negative power, and this raises.

    Returns one fewer coefficient than the input (the top power is lost to the
    division, matching the precision the truncated series can support).
    """
    c = [int(v) for v in coeffs]
    if not c:
        raise ValueError("empty coefficient list")
    if c[0] != 1:
        raise ValueError(
            f"shift_down needs constant term 1 (an [x]_q with x >= 1), got c_0 = {c[0]}"
        )
    return c[1:]


def shift_up(coeffs: Sequence[int]) -> list[int]:
    """Coefficients of [x+1]_q = q[x]_q + 1 from those of [x]_q (note 5).

    Multiplies by q and adds 1: the new constant term is 1 and every old
    coefficient moves up one power. Inverse of `shift_down`. Returns one more
    coefficient than the input.
    """
    return [1] + [int(v) for v in coeffs]


def _monomial(k: int) -> str:
    if k == 0:
        return ""
    if k == 1:
        return "q"
    return f"q^{k}"


def format_laurent(coeffs: Sequence[int]) -> str:
    """Render a coefficient list [c_0, c_1, ...] as a readable q-polynomial.

    Appends an O(q^N) tail where N = len(coeffs), the first power not held.
    """
    coeffs = [int(c) for c in coeffs]
    parts: list[str] = []
    for k, c in enumerate(coeffs):
        if c == 0:
            continue
        mono = _monomial(k)
        if c == 1:
            term = mono if mono else "1"
        elif c == -1:
            term = f"-{mono}" if mono else "-1"
        else:
            term = f"{c}*{mono}" if mono else f"{c}"
        parts.append(term)
    if not parts:
        body = "0"
    else:
        body = parts[0]
        for term in parts[1:]:
            if term.startswith("-"):
                body += f" - {term[1:]}"
            else:
                body += f" + {term}"
    return f"{body} + O(q^{len(coeffs)})"


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m qreals.expansions",
        description="Laurent expansion of the q-real [x]_q via the MGO formula.",
    )
    parser.add_argument(
        "x", help="a sympy-parseable real, e.g. pi, sqrt(2), 22/7, pi-2"
    )
    parser.add_argument(
        "--order",
        type=int,
        default=12,
        help="highest power q^order to print (default 12)",
    )
    args = parser.parse_args(argv)

    coeffs = mgo_laurent(args.x, args.order)
    prefix = integer_part_prefix(args.x)
    t = len(prefix) - 1

    print(f"[{args.x}]_q = {format_laurent(coeffs)}")
    print(f"integer-part prefix (floor = {t}): {format_laurent(prefix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
