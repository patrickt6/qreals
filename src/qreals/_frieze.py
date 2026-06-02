"""Conway-Coxeter frieze of a rational r/s > 1, with the q-coefficient overlay.

This is a minimal, vendored copy of the frieze math from the sibling project
qfrieze (https://github.com/patrickt6/qfrieze), reused here so `qreals serve`
can draw a frieze without recomputing the mathematics itself. It is kept tiny
on purpose: only the pieces the serve frieze card needs (quiddity, the integer
frieze rows, and the q-deformed overlay) are carried over, with the same
conventions and self-checks as qfrieze.

The serve back end imports the real qfrieze when an overlay-capable build is
installed, and falls back to this module otherwise (see serve._frieze_backend).

Vendored from qfrieze, MIT License, Copyright (c) 2026 Patrick Taylor. The
mathematics is pinned to:

  S. Morier-Genoud and V. Ovsienko, "The Farey Boat" (quiddity and the integer
  frieze, Section 1.1-1.2, Fact 1, Cor. 1.1 and 1.4); and "Quantum Numbers and
  q-Deformed Conway-Coxeter Friezes", Math. Intelligencer 43 (2021), 61-70,
  arXiv:2011.10809, eq. 3 (the q-unimodular rule that fixes the overlay).

The MIT permission notice from qfrieze's LICENSE travels with this copy:

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software ... subject to the condition that the above copyright notice
  and this permission notice be included in all copies or substantial portions
  of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
  KIND.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd

__all__ = [
    "QPoly",
    "Frieze",
    "CoefficientMapping",
    "frieze",
    "frieze_coefficients",
    "quiddity_of",
    "regular_cf",
    "negative_cf",
]

CITATION = (
    "Morier-Genoud and Ovsienko, arXiv:2011.10809 (2020); "
    "Math. Intelligencer 43 (2021), 61-70, eq. 3 (q-unimodular rule)"
)

_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


# --------------------------------------------------------------------------- #
# Continued fractions
# --------------------------------------------------------------------------- #
def regular_cf(r: int, s: int) -> list[int]:
    """Canonical regular continued fraction [a_1, ..., a_l] of r/s."""
    if s == 0:
        raise ValueError("denominator must be non-zero")
    a: list[int] = []
    while s:
        q, rem = divmod(r, s)
        a.append(q)
        r, s = s, rem
    if len(a) > 1 and a[-1] == 1:
        a.pop()
        a[-1] += 1
    return a


def _make_even(cf: list[int]) -> list[int]:
    """An even-length regular CF for the same value ([..., a] = [..., a-1, 1])."""
    if len(cf) % 2 == 0:
        return list(cf)
    even = list(cf)
    even[-1] -= 1
    even.append(1)
    return even


def _cf_value(cf: list[int]) -> Fraction:
    """Evaluate a regular continued fraction to an exact Fraction."""
    val = Fraction(cf[-1])
    for a in reversed(cf[:-1]):
        val = a + 1 / val
    return val


def negative_cf(r: int, s: int) -> list[int]:
    """Negative (Hirzebruch-Jung) continued fraction with every c_i >= 2."""
    if s <= 0 or r <= s:
        raise ValueError(f"negative_cf needs r/s > 1, got {r}/{s}")
    c: list[int] = []
    while True:
        q = -(-r // s)  # ceil(r / s)
        c.append(q)
        num, den = s, q * s - r
        if den == 0:
            break
        r, s = num, den
    return c


# --------------------------------------------------------------------------- #
# Quiddity
# --------------------------------------------------------------------------- #
def quiddity_of(r: int, s: int) -> list[int]:
    """Cyclic Conway-Coxeter quiddity of T_{r/s} (Farey Boat construction)."""
    g = gcd(r, s)
    r, s = r // g, s // g
    if s <= 0:
        raise ValueError("denominator must be positive")
    if s == 1:
        raise ValueError(
            f"{r} is an integer; friezes need a non-integer rational r/s > 1"
        )
    if r <= s:
        raise ValueError(
            f"r/s = {r}/{s} <= 1; friezes are defined for r/s > 1"
        )

    forward = negative_cf(r, s)
    even = _make_even(regular_cf(r, s))
    rev_val = _cf_value(list(reversed(even)))
    backward = negative_cf(rev_val.numerator, rev_val.denominator)

    quiddity = [1, *forward, 1, *backward]

    n = len(quiddity)
    total = sum(quiddity)
    if total != 3 * n - 6:
        raise AssertionError(
            f"quiddity {quiddity} for {r}/{s} violates sum = 3n-6"
        )
    return quiddity


# --------------------------------------------------------------------------- #
# Integer frieze
# --------------------------------------------------------------------------- #
@dataclass
class Frieze:
    """A classical Conway-Coxeter integer frieze for r/s.

    ``rows`` runs top to bottom and includes the all-0 borders (rows[0] and
    rows[-1], not drawn) and the all-1 borders (rows[1] and rows[-2]).
    """

    r: int
    s: int
    cf: list[int]
    quiddity: list[int]
    rows: list[list[int]]
    n_polygon: int
    width: int
    height: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.height:
            self.height = len(self.rows)

    def cell(self, i: int, j: int) -> int:
        return self.rows[i][j % self.width]

    def diamond_check(self) -> bool:
        """Every staggered 2x2 diamond satisfies a*d - b*c == 1."""
        n = self.width
        for i in range(1, len(self.rows) - 1):
            above, mid, below = self.rows[i - 1], self.rows[i], self.rows[i + 1]
            for k in range(n):
                a = mid[k]
                d = mid[(k + 1) % n]
                b = above[(k + 1) % n]
                c = below[k]
                if a * d - b * c != 1:
                    return False
        return True


def _build_rows(quiddity: list[int]) -> list[list[int]]:
    """Generate the full frieze (with 0/1 borders) from the quiddity row."""
    n = len(quiddity)
    rows: list[list[int]] = [[0] * n, [1] * n, list(quiddity)]
    ones = [1] * n
    while rows[-1] != ones:
        prev2, prev1 = rows[-2], rows[-1]
        new = []
        for k in range(n):
            num = prev1[k] * prev1[(k + 1) % n] - 1
            den = prev2[(k + 1) % n]
            q, rem = divmod(num, den)
            if rem != 0:
                raise AssertionError(f"non-integer frieze entry at col {k}: {num}/{den}")
            new.append(q)
        rows.append(new)
        if len(rows) > n + 3:
            raise AssertionError(
                f"frieze for quiddity {quiddity} did not close; convention bug"
            )
    rows.append([0] * n)
    return rows


def frieze(r: int, s: int) -> Frieze:
    """Build the classical Conway-Coxeter integer frieze for r/s > 1."""
    g = gcd(r, s)
    r0, s0 = r // g, s // g
    quiddity = quiddity_of(r0, s0)
    cf = regular_cf(r0, s0)
    rows = _build_rows(quiddity)
    n = len(quiddity)
    f = Frieze(
        r=r0,
        s=s0,
        cf=cf,
        quiddity=quiddity,
        rows=rows,
        n_polygon=n,
        width=n,
        height=len(rows),
    )
    if not f.diamond_check():
        raise AssertionError(f"diamond_check failed for {r0}/{s0}; construction is wrong")
    return f


# --------------------------------------------------------------------------- #
# q-coefficient overlay
# --------------------------------------------------------------------------- #
class QPoly:
    """An exact polynomial in q with non-negative integer coefficients.

    Stored as a tuple ``c`` with ``c[e]`` the coefficient of q**e, no trailing
    zeros.
    """

    __slots__ = ("c",)

    def __init__(self, coeffs=()) -> None:
        c = list(coeffs)
        while c and c[-1] == 0:
            c.pop()
        self.c = tuple(c)

    @classmethod
    def zero(cls) -> "QPoly":
        return cls(())

    @classmethod
    def one(cls) -> "QPoly":
        return cls((1,))

    @classmethod
    def monomial(cls, exp: int, coeff: int = 1) -> "QPoly":
        if exp < 0:
            raise ValueError("overlay polynomials have no negative powers of q")
        return cls((0,) * exp + (coeff,))

    @classmethod
    def q_int(cls, n: int) -> "QPoly":
        """The Euler q-integer [n]_q = 1 + q + ... + q**(n-1) for n >= 0."""
        if n < 0:
            raise ValueError("quiddity entries are positive; [n]_q needs n >= 0")
        return cls((1,) * n)

    def __add__(self, other: "QPoly") -> "QPoly":
        a, b = self.c, other.c
        if len(a) < len(b):
            a, b = b, a
        out = list(a)
        for i, v in enumerate(b):
            out[i] += v
        return QPoly(out)

    def __sub__(self, other: "QPoly") -> "QPoly":
        a, b = self.c, other.c
        out = list(a) + [0] * (len(b) - len(a))
        for i, v in enumerate(b):
            out[i] -= v
        return QPoly(out)

    def __mul__(self, other: "QPoly") -> "QPoly":
        a, b = self.c, other.c
        if not a or not b:
            return QPoly(())
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai:
                for jx, bj in enumerate(b):
                    out[i + jx] += ai * bj
        return QPoly(out)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QPoly) and self.c == other.c

    def __hash__(self) -> int:
        return hash(self.c)

    @property
    def degree(self) -> int:
        return len(self.c) - 1

    def is_zero(self) -> bool:
        return not self.c

    def eval(self, q: int) -> int:
        acc = 0
        for coeff in reversed(self.c):
            acc = acc * q + coeff
        return acc

    def coeffs(self) -> list[int]:
        return list(self.c)

    def _terms(self):
        for e in range(self.degree, -1, -1):
            a = self.c[e]
            if a:
                yield e, a

    def __str__(self) -> str:
        if self.is_zero():
            return "0"
        out = []
        for e, a in self._terms():
            if e == 0:
                out.append(str(a))
            elif e == 1:
                out.append("q" if a == 1 else f"{a}q")
            else:
                out.append(f"q^{e}" if a == 1 else f"{a}q^{e}")
        return " + ".join(out)

    def to_superscript(self) -> str:
        if self.is_zero():
            return "0"
        out = []
        for e, a in self._terms():
            if e == 0:
                out.append(str(a))
            elif e == 1:
                out.append("q" if a == 1 else f"{a}q")
            else:
                sup = str(e).translate(_SUPERSCRIPT)
                out.append(f"q{sup}" if a == 1 else f"{a}q{sup}")
        return " + ".join(out)

    def to_latex(self) -> str:
        if self.is_zero():
            return "0"
        out = []
        for e, a in self._terms():
            if e == 0:
                out.append(str(a))
            elif e == 1:
                out.append("q" if a == 1 else f"{a}q")
            else:
                out.append(f"q^{{{e}}}" if a == 1 else f"{a}q^{{{e}}}")
        return " + ".join(out)

    def __repr__(self) -> str:
        return f"QPoly({self.c!r})"


@dataclass
class CoefficientMapping:
    """The q-deformed Conway-Coxeter frieze of r/s (the overlay)."""

    r: int
    s: int
    quiddity: list[int]
    n_polygon: int
    width: int
    rows: list[list[QPoly] | None]
    convention: str = CITATION
    _frieze: Frieze | None = field(default=None, repr=False)

    def cell(self, i: int, j: int) -> QPoly:
        row = self.rows[i]
        if row is None:
            raise IndexError(f"row {i} is a 0-border with no q-polynomial")
        return row[j % self.width]

    def matches_integer_frieze(self) -> bool:
        f = self._frieze
        if f is None:
            return False
        for i, row in enumerate(self.rows):
            if row is None:
                continue
            for j, poly in enumerate(row):
                if poly.eval(1) != f.rows[i][j]:
                    return False
        return True

    def verify_q_unimodular(self) -> bool:
        n = self.width
        c = self.quiddity
        rows = self.rows
        for m in range(3, len(rows) - 1):
            if rows[m - 2] is None or rows[m - 1] is None or rows[m] is None:
                continue
            for col in range(n):
                lhs = rows[m - 1][col] * rows[m - 1][(col + 1) % n] - (
                    rows[m - 2][(col + 1) % n] * rows[m][col]
                )
                exp = sum((c[(col + t) % n] - 1) for t in range(m - 2))
                if lhs != QPoly.monomial(exp):
                    return False
        return True

    def verify(self) -> bool:
        return self.matches_integer_frieze() and self.verify_q_unimodular()

    def q_numerator(self) -> QPoly:
        """R(q): the numerator of [r/s]_q; equals the largest cell."""
        f = self._frieze
        best = QPoly.zero()
        best_int = -1
        for i, row in enumerate(self.rows):
            if row is None:
                continue
            for j, poly in enumerate(row):
                v = f.rows[i][j] if f is not None else poly.eval(1)
                if v > best_int:
                    best_int, best = v, poly
        return best


def frieze_coefficients(r: int, s: int) -> CoefficientMapping:
    """Build the q-deformed Conway-Coxeter frieze (overlay) of r/s > 1.

    Self-verified before returning: it must specialise to the integer frieze at
    q = 1 and satisfy the q-unimodular rule (eq. 3) on every diamond.
    """
    f = frieze(r, s)
    n = f.width
    c = f.quiddity
    height = len(f.rows)

    rows: list[list[QPoly] | None] = [None] * height
    rows[1] = [QPoly.one() for _ in range(n)]
    rows[2] = [QPoly.q_int(c[col]) for col in range(n)]
    for m in range(3, height - 1):
        prev1, prev2 = rows[m - 1], rows[m - 2]
        assert prev1 is not None and prev2 is not None
        row = []
        for col in range(n):
            head = QPoly.q_int(c[col])
            term1 = head * prev1[(col + 1) % n]
            term2 = QPoly.monomial(c[col] - 1) * prev2[(col + 2) % n]
            row.append(term1 - term2)
        rows[m] = row

    cm = CoefficientMapping(
        r=f.r,
        s=f.s,
        quiddity=list(c),
        n_polygon=n,
        width=n,
        rows=rows,
        _frieze=f,
    )
    if not cm.verify():
        raise AssertionError(
            f"q-overlay for {f.r}/{f.s} failed self-check; convention bug"
        )
    return cm
