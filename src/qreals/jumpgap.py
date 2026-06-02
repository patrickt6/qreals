r"""The one-sided jump gap of a q-deformed rational, [p/s]_q^+ - [p/s]_q^-.

The Morier-Genoud-Ovsienko (MGO) map x |-> [x]_q is discontinuous at every
rational: a rational p/s carries two distinct q-deformations, and which one the
deformation approaches depends on the side of approach. Jouteur
(arXiv:2503.02122) names the two the *right* and *left* versions and pins down
which is which one-sided limit (Definition 1.2, Proposition 1.1, Proposition
4.8):

    [p/s]_q^+  (limit from above, x -> p/s with x > p/s) = the RIGHT version,
               the original MGO deformation, equal to qreals.q_rational(p, s);
    [p/s]_q^-  (limit from below, x -> p/s with x < p/s) = the LEFT version.

Both are read off the SAME q-deformed continuant matrix M_q of the even-length
regular continued fraction of p/s, applied to two q-deformations of the point at
infinity:

    [p/s]_q^+ = M_q . (1, 0)^T          (first column: R / S^+),
    [p/s]_q^- = M_q . (1, 1 - q)^T      (1/(1-q) is the left version of infinity).

Writing M_q = [[R, R'], [S, S']], the gap telescopes through det(M_q) = q^E:

    gap(p/s) = [p/s]_q^+ - [p/s]_q^- = (1 - q) q^E / (S^+ S^-),

with S^+ = S the right q-denominator, S^- = S + (1 - q) S' the left one, and
E = (sum of even-position digits) - (sum of odd-position digits) of the
even-length continued fraction. Both denominators are q-analogues of s:
S^+(1) = S^-(1) = s.

The right version is cross-checked against the independent qreals.q_rational
path (the oracle), and both q-denominators are checked to collapse to s at
q = 1, so trust is verified rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .continued_fraction import make_even_length
from .rational import q, q_int, q_int_qinv, q_rational


def cf_terms(p: int, s: int) -> list[int]:
    """Regular continued-fraction partial quotients of p/s."""
    return [int(t) for t in sp.continued_fraction(sp.Rational(int(p), int(s)))]


def alt_cf_sum(cf: list[int]) -> int:
    """E = (sum of even-position digits) - (sum of odd-position digits)."""
    return sum(a for i, a in enumerate(cf) if i % 2 == 0) - sum(
        a for i, a in enumerate(cf) if i % 2 == 1
    )


def _q_block(i: int, a: int) -> sp.Matrix:
    """MGO 2x2 q-continuant block at 0-indexed position i with digit a.

    Even i (1-indexed odd) carries [a]_q with q^a above; odd i carries
    [a]_{1/q} with q^{-a} above, the exact convention of qreals.q_rational.
    """
    if i % 2 == 0:
        return sp.Matrix([[q_int(a), q**a], [1, 0]])
    return sp.Matrix([[q_int_qinv(a), q ** (-a)], [1, 0]])


def continuant_matrix(p: int, s: int) -> tuple[sp.Matrix, list[int]]:
    """The q-deformed continuant matrix M_q of p/s, and its even-length CF."""
    cf = make_even_length(cf_terms(p, s))
    matrix = sp.eye(2)
    for i, a in enumerate(cf):
        matrix = matrix * _q_block(i, a)
    return matrix, cf


def right_version(p: int, s: int) -> sp.Expr:
    """[p/s]_q^+ = limit from above = MGO right version = M_q . (1, 0)^T."""
    matrix, _ = continuant_matrix(p, s)
    return sp.cancel(matrix[0, 0] / matrix[1, 0])


def left_version(p: int, s: int) -> sp.Expr:
    """[p/s]_q^- = limit from below = left version = M_q . (1, 1 - q)^T."""
    matrix, _ = continuant_matrix(p, s)
    num = matrix[0, 0] + (1 - q) * matrix[0, 1]
    den = matrix[1, 0] + (1 - q) * matrix[1, 1]
    return sp.cancel(num / den)


def jump_gap(p: int, s: int) -> sp.Expr:
    """gap(p/s) = [p/s]_q^+ - [p/s]_q^-, an exact rational function of q."""
    matrix, _ = continuant_matrix(p, s)
    right = matrix[0, 0] / matrix[1, 0]
    left = (matrix[0, 0] + (1 - q) * matrix[0, 1]) / (
        matrix[1, 0] + (1 - q) * matrix[1, 1]
    )
    return sp.cancel(right - left)


@dataclass(frozen=True)
class JumpGap:
    """The two q-versions of p/s and the factored gap between them.

    Every expression field is an element of Q(q). The gap satisfies
    gap = (1 - q) q^E / (S^+ S^-) with E the exponent and S^+, S^- the right and
    left q-denominators; both denominators equal s at q = 1.
    """

    p: int
    s: int
    cf: tuple[int, ...]  # the even-length regular continued fraction of p/s
    right: sp.Expr  # [p/s]_q^+, the right version (limit from above)
    left: sp.Expr  # [p/s]_q^-, the left version (limit from below)
    gap: sp.Expr  # right - left, factored
    exponent: int  # E, the q-power in (1 - q) q^E (det M_q = q^E)
    s_plus: sp.Expr  # S^+, the right q-denominator, factored
    s_minus: sp.Expr  # S^-, the left q-denominator, factored

    def denominators_at_one(self) -> tuple[int, int]:
        """S^+ and S^- specialised at q = 1; both must equal s."""
        return (
            int(sp.nsimplify(self.s_plus.subs(q, 1))),
            int(sp.nsimplify(self.s_minus.subs(q, 1))),
        )

    def right_matches_oracle(self) -> bool:
        """The right version equals qreals.q_rational(p, s), the oracle."""
        return bool(sp.simplify(self.right - q_rational(self.p, self.s)) == 0)

    def closed_form_holds(self) -> bool:
        """gap == (1 - q) q^E / (S^+ S^-)."""
        target = (1 - q) * q**self.exponent / (self.s_plus * self.s_minus)
        return bool(sp.simplify(self.gap - target) == 0)

    def checks(self) -> dict[str, bool]:
        """The oracle checks, computed (not asserted): the right version matches
        q_rational, and S^+ and S^- both collapse to s at q = 1."""
        plus_one, minus_one = self.denominators_at_one()
        return {
            "right_matches_oracle": self.right_matches_oracle(),
            "s_plus_at_one_is_s": plus_one == self.s,
            "s_minus_at_one_is_s": minus_one == self.s,
        }


def jumpgap(p: int, s: int) -> JumpGap:
    """Both q-versions of p/s and the factored gap between them.

    Builds the q-continuant matrix once and reads off the right version
    [p/s]_q^+ = R / S^+, the left version [p/s]_q^- = (R + (1-q)R') / S^-, the
    gap, the exponent E, and the two q-denominators S^+ and S^-. Nothing is
    recomputed; the returns are the existing catalogue values, factored.
    """
    p = int(p)
    s = int(s)
    if s == 0:
        raise ZeroDivisionError("denominator zero")
    matrix, cf = continuant_matrix(p, s)
    s_plus = matrix[1, 0]
    s_minus = matrix[1, 0] + (1 - q) * matrix[1, 1]
    right = sp.factor(matrix[0, 0] / s_plus)
    left = sp.factor((matrix[0, 0] + (1 - q) * matrix[0, 1]) / s_minus)
    gap = sp.factor(right - left)
    return JumpGap(
        p=p,
        s=s,
        cf=tuple(cf),
        right=right,
        left=left,
        gap=gap,
        exponent=alt_cf_sum(cf),
        s_plus=sp.factor(s_plus),
        s_minus=sp.factor(s_minus),
    )
