r"""Exact elements of Q(sqrt D), for D a squarefree positive integer.

A `QuadraticIrrational` holds x = (p + r*sqrt(D)) / s exactly, with integer p,
r, s, s > 0, and gcd(gcd(|p|, |r|), s) = 1. No floating point is used for any
comparison, floor, ceil, or continued-fraction step: every exact decision
(sign, ceiling) reduces to comparing squares of integers.

Two continued-fraction conventions are relevant to this module and to
`negation`, and it matters which one is used where:

  * The regular (floor-based) continued fraction, x = [a_1, a_2, ...] with
    a_1 possibly <= 0 and a_i >= 1 for i >= 2, folded by the MGO formula
    (`continued_fraction.py`, `continuant.py`). This is the convention MGO's
    own paper (arXiv:1908.04365) uses for [x]_q.

  * The Hirzebruch-Jung (ceiling) continued fraction, x = c_1 - 1/(c_2 - 1/(...))
    with c_i >= 2 for i >= 2 (any integer for c_1), which this module
    implements as `hj_terms`/`HJTerm` below. It is a different but standard
    encoding of the same real number, useful because every step is a single
    ceiling and the recursion never needs a sign check on a partial quotient.

GATE-2 finding (see `negation.py` for the full writeup): both conventions,
correctly implemented, reproduce the same [x]_q for every quadratic irrational
tested (sqrt(2), sqrt(3), sqrt(5), the golden ratio); their agreement is
required by the Jouteur left/right symmetry and was checked directly against
the worked examples printed in arXiv:1908.04365, Section 4.3. There is no bug
and no live discrepancy; a stray transcription of the wrong coefficients for
sqrt(2) had been floated as a possible mismatch and is now retracted.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def is_squarefree(d: int) -> bool:
    """True if the positive integer d has no repeated prime factor."""
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    n = d
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        while n % i == 0:
            n //= i
        i += 1
    return True


def sign_p_plus_r_sqrtD(p: int, r: int, D: int) -> int:
    """Exact sign of p + r*sqrt(D) for integers p, r and squarefree D > 1.

    Never evaluates sqrt(D) as a float; reduces to a comparison of squares.
    """
    if r == 0:
        return _sign(p)
    if p == 0:
        return _sign(r)
    if r > 0:
        if p > 0:
            return 1
        # p < 0 < r: p + r*sqrt(D) > 0  <=>  r^2 D > p^2
        return _sign(r * r * D - p * p)
    # r < 0
    if p < 0:
        return -1
    # p > 0 > r: p + r*sqrt(D) > 0  <=>  p^2 > r^2 D
    return _sign(p * p - r * r * D)


def _reduce(p: int, r: int, s: int) -> tuple[int, int, int]:
    if s == 0:
        raise ZeroDivisionError("s must be nonzero")
    if s < 0:
        p, r, s = -p, -r, -s
    g = gcd(gcd(abs(p), abs(r)), s)
    if g == 0:
        g = 1
    return p // g, r // g, s // g


def _sqrt_bracket(D: int, digits: int = 40) -> Fraction:
    """A Fraction within 10^-digits of sqrt(D), for seeding integer searches."""
    scale = 10 ** digits
    return Fraction(isqrt(D * scale * scale), scale)


@dataclass(frozen=True)
class QuadraticIrrational:
    """x = (p + r*sqrt(D)) / s, exact, with D a fixed squarefree integer > 1."""

    p: int
    r: int
    s: int
    D: int

    def __post_init__(self) -> None:
        if self.D <= 1:
            raise ValueError(f"D must be a squarefree integer > 1, got {self.D}")
        if not is_squarefree(self.D):
            raise ValueError(f"D = {self.D} is not squarefree")
        p, r, s = _reduce(self.p, self.r, self.s)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "s", s)

    @classmethod
    def from_ab(cls, a: Fraction, b: Fraction, D: int) -> "QuadraticIrrational":
        """Build (a + b*sqrt(D)) from rationals a, b, over a common denominator."""
        s = a.denominator * b.denominator // gcd(a.denominator, b.denominator)
        p = a.numerator * (s // a.denominator)
        r = b.numerator * (s // b.denominator)
        return cls(p, r, s, D)

    def as_fraction_pair(self) -> tuple[Fraction, Fraction]:
        """(a, b) rationals with self == a + b*sqrt(D)."""
        return Fraction(self.p, self.s), Fraction(self.r, self.s)

    def decimal(self) -> float:
        return (self.p + self.r * (self.D ** 0.5)) / self.s

    def sign(self) -> int:
        return sign_p_plus_r_sqrtD(self.p, self.r, self.D)

    def is_zero(self) -> bool:
        return self.p == 0 and self.r == 0

    def __neg__(self) -> "QuadraticIrrational":
        return QuadraticIrrational(-self.p, -self.r, self.s, self.D)

    def __add__(self, other: "QuadraticIrrational | Fraction | int") -> "QuadraticIrrational":
        if isinstance(other, QuadraticIrrational):
            if other.D != self.D:
                raise ValueError("cannot add QuadraticIrrational values over different D")
            s = self.s * other.s // gcd(self.s, other.s)
            p = self.p * (s // self.s) + other.p * (s // other.s)
            r = self.r * (s // self.s) + other.r * (s // other.s)
            return QuadraticIrrational(p, r, s, self.D)
        f = Fraction(other)
        s = self.s * f.denominator // gcd(self.s, f.denominator)
        p = self.p * (s // self.s) + f.numerator * (s // f.denominator)
        r = self.r * (s // self.s)
        return QuadraticIrrational(p, r, s, self.D)

    def __sub__(self, other: "QuadraticIrrational | Fraction | int") -> "QuadraticIrrational":
        if isinstance(other, QuadraticIrrational):
            return self + (-other)
        return self + (-Fraction(other))

    def scalar_mul(self, f: Fraction | int) -> "QuadraticIrrational":
        """Multiply by a rational scalar (keeps the value in Q(sqrt D))."""
        f = Fraction(f)
        p = self.p * f.numerator
        r = self.r * f.numerator
        s = self.s * f.denominator
        return QuadraticIrrational(p, r, s, self.D)

    def reciprocal(self) -> "QuadraticIrrational":
        """1/self, exact. Raises ZeroDivisionError if self is zero."""
        if self.is_zero():
            raise ZeroDivisionError("reciprocal of zero QuadraticIrrational")
        # 1 / ((p + r sqrt D)/s) = s(p - r sqrt D) / (p^2 - D r^2)
        norm = self.p * self.p - self.D * self.r * self.r
        if norm == 0:
            raise ZeroDivisionError("norm is zero; self is not a valid quadratic unit here")
        p = self.s * self.p
        r = -self.s * self.r
        s = norm
        return QuadraticIrrational(p, r, s, self.D)

    def floor(self) -> int:
        """Exact floor(self), via a high-precision rational sqrt bracket."""
        approx = _sqrt_bracket(self.D)
        n0 = int((self.p + self.r * approx) / self.s)
        # Correct n0 in either direction by exact integer sign checks.
        n = n0 - 2
        while sign_p_plus_r_sqrtD(self.p - n * self.s, self.r, self.D) < 0:
            n += 1
        while sign_p_plus_r_sqrtD(self.p - (n + 1) * self.s, self.r, self.D) >= 0:
            n += 1
        return n

    def ceil(self) -> int:
        f = self.floor()
        if self.p == f * self.s and self.r == 0:
            return f
        return f + 1

    def __eq__(self, other: object) -> bool:
        if isinstance(other, QuadraticIrrational):
            return (self.p, self.r, self.s, self.D) == (other.p, other.r, other.s, other.D)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.p, self.r, self.s, self.D))

    def __repr__(self) -> str:
        return f"QuadraticIrrational({self.p}, {self.r}, {self.s}, D={self.D})"


def hj_terms(x: QuadraticIrrational, num_terms: int) -> tuple[list[int], int | None, int | None]:
    """Hirzebruch-Jung (ceiling) continued fraction of x, up to num_terms terms.

    x = c_1 - 1/(c_2 - 1/(c_3 - ...)), generated by c = ceil(x); x_next =
    1/(c - x). c_1 can be any integer; every later term is >= 2 for a genuine
    quadratic irrational, because c - x lies in (0, 1) and its reciprocal
    exceeds 1, forcing the next ceiling to be at least 2.

    Returns (terms, period_start, period_length). period_start and
    period_length are both None if no period was found within num_terms (this
    can only happen if num_terms is too small: every quadratic irrational has
    an eventually periodic HJ continued fraction, detected here by hashing
    the (p, r, s) state and finding a repeat).
    """
    terms: list[int] = []
    seen: dict[tuple[int, int, int], int] = {}
    cur = x
    period_start = None
    period_length = None
    while len(terms) < num_terms:
        state = (cur.p, cur.r, cur.s)
        if state in seen:
            period_start = seen[state]
            period_length = len(terms) - period_start
            break
        seen[state] = len(terms)
        c = cur.ceil()
        terms.append(c)
        # x_next = 1 / (c - x)
        diff = QuadraticIrrational(c * cur.s - cur.p, -cur.r, cur.s, cur.D)
        if diff.is_zero():
            # x was exactly the integer c; terminates (only possible for a
            # rational input passed in by mistake, never for a genuine
            # irrational element with r != 0 after reduction).
            return terms, None, None
        cur = diff.reciprocal()
    if period_start is not None:
        cyc = terms[period_start:]
        while len(terms) < num_terms:
            terms.append(cyc[(len(terms) - period_start) % len(cyc)])
    return terms[:num_terms], period_start, period_length
