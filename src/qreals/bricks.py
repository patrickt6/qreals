r"""The cyclotomic reference card of a q-integer [n]_q.

The classical identity [n]_q = prod over the divisors e of n with e >= 2 of
the cyclotomic polynomial of index e is the brick decomposition of the
q-integer: each brick is irreducible over Z[q], has degree phi(e) (Euler's
totient), and evaluates at q = 1 to p when e is a prime power p^k and to 1
otherwise. This module computes the card for one n: the expanded [n]_q, one
table row per brick (expanded polynomial, degree, value at 1), the divisors
of n partitioned into prime powers and composite non-prime-powers, the lcm of
any user-supplied subset of divisors, and the exact evaluation of every
factor at a rational point or at a root of unity.

Evaluation is exact everywhere. At a rational q0 each factor is evaluated
over Q. At a root of unity of order e0 (the `zeta_e0` syntax) each factor is
reduced modulo the cyclotomic polynomial of index e0, so its value is the
integer coefficient vector in the power basis of Z[zeta_e0]; the value is
zero exactly when the vector is zero. No floating point appears anywhere.

JSON schema (the --json output of `qreals bricks`; keys are stable):
    n               int    the input
    qint            str    [n]_q expanded, ascii ascending powers
    rows            [ {e: int, phi: str (expanded), deg: int, phi_at_1: int} ]
                    one row per divisor e >= 2 of n, sorted by e
    prime_powers    [int]  the divisors e >= 2 of n that are prime powers
    composites      [int]  the divisors e >= 2 of n that are composite
                           non-prime-powers
    lcm             null | {subset: [int], value: int}
    at              null | {point: str, kind: "rational" | "zeta",
                            values: [ {factor: str, value: str,
                                       is_zero: bool} ]}
                    the values list covers [n]_q then every table row
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm

import sympy as sp

from . import formatter
from .rational import q, q_int


@dataclass(frozen=True)
class BrickRow:
    """One brick: the cyclotomic factor of index e in [n]_q.

    deg is phi(e) (the polynomial degree, asserted equal to Euler's totient)
    and at_1 is the value at q = 1 computed by evaluation; rule_at_1 is the
    closed rule (p for a prime power e = p^k, 1 otherwise), kept separately
    so the two routes can be cross-checked.
    """

    e: int
    phi: sp.Poly
    deg: int
    at_1: int
    rule_at_1: int


@dataclass(frozen=True)
class AtValue:
    """The exact value of one factor at the evaluation point.

    For a rational point the value is a sympy Rational and vector is None.
    For a root of unity of order e0 the value is the integer coefficient
    vector in the power basis 1, zeta, ..., zeta^(phi(e0) - 1) of Z[zeta_e0]
    and rational is None.
    """

    label: str
    rational: sp.Rational | None
    vector: tuple[int, ...] | None

    @property
    def is_zero(self) -> bool:
        if self.vector is not None:
            return all(c == 0 for c in self.vector)
        return self.rational == 0


@dataclass(frozen=True)
class BricksCard:
    """The full cyclotomic reference card of [n]_q."""

    n: int
    qint: sp.Poly
    rows: list[BrickRow]
    prime_powers: list[int]
    composites: list[int]
    lcm_subset: list[int] | None
    lcm_value: int | None
    at_point: str | None
    at_kind: str | None
    at_values: list[AtValue]


def prime_power_rule(e: int) -> int:
    """The closed rule for the value of the brick of index e at q = 1.

    p when e = p^k is a prime power (k >= 1), 1 otherwise. The card computes
    the value by evaluation as well and asserts the two agree.
    """
    factors = sp.factorint(int(e))
    if len(factors) == 1:
        return int(next(iter(factors)))
    return 1


def _brick_divisors(n: int) -> list[int]:
    """The brick indices of [n]_q: the divisors e of n with e >= 2, sorted."""
    return sorted(int(e) for e in sp.divisors(int(n)) if int(e) >= 2)


def parse_point(text: str) -> tuple[str, object]:
    """Read an evaluation point: a rational 'a/b' or a root of unity 'zeta_e'.

    Returns ("rational", Fraction) or ("zeta", e). The root-of-unity syntax
    is the shared formatter's own label, so the accepted spelling can never
    drift from the printed one.
    """
    text = text.strip()
    zeta_prefix = formatter.zeta_label(1)[:-1]
    if text.startswith(zeta_prefix):
        tail = text[len(zeta_prefix):]
        try:
            e = int(tail)
        except ValueError:
            raise ValueError(
                f"a root-of-unity point is {formatter.zeta_label(12)} with a "
                f"whole-number order, got {text!r}"
            ) from None
        if e < 1:
            raise ValueError("the root-of-unity order must be at least 1")
        return "zeta", e
    try:
        frac = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise ValueError(
            f"an evaluation point is a rational like 1/2 or a root of unity "
            f"like {formatter.zeta_label(12)}, got {text!r}"
        ) from None
    return "rational", frac


def eval_at_rational(poly: sp.Poly, q0: Fraction) -> sp.Rational:
    """The exact value of an integer polynomial at a rational point."""
    return sp.Rational(poly.eval(sp.Rational(q0.numerator, q0.denominator)))


def eval_at_zeta(poly: sp.Poly, e0: int) -> tuple[int, ...]:
    """The exact value of an integer polynomial at a root of unity of order e0.

    Z[zeta_e0] is Z[x] modulo the cyclotomic polynomial of index e0, so the
    value is the remainder of the polynomial under division by that monic
    integer polynomial: an integer coefficient vector in the power basis
    1, zeta, ..., zeta^(phi(e0) - 1). The value is zero in Z[zeta_e0] exactly
    when every coordinate is zero. No floating point is involved.
    """
    modulus = sp.Poly(sp.cyclotomic_poly(int(e0), q), q, domain="ZZ")
    rem = poly.rem(modulus)
    width = modulus.degree()
    coeffs = [0] * width
    for (k,), c in rem.terms():
        coeffs[int(k)] = int(c)
    return tuple(coeffs)


def zeta_vector_ascii(vector: tuple[int, ...], e0: int) -> str:
    """An element of Z[zeta_e0] in the power basis, ascending, explicit signs."""
    zeta = formatter.zeta_label(e0)
    pieces: list[str] = []
    for k, c in enumerate(vector):
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        size = abs(c)
        if k == 0:
            body = str(size)
        else:
            mono = zeta if k == 1 else f"{zeta}^{k}"
            body = mono if size == 1 else f"{size}*{mono}"
        if not pieces:
            pieces.append(body if sign == "+" else f"-{body}")
        else:
            pieces.append(f"{sign} {body}")
    return " ".join(pieces) if pieces else "0"


def zeta_vector_tex(vector: tuple[int, ...], e0: int) -> str:
    """The TeX twin of zeta_vector_ascii."""
    zeta = formatter.zeta_tex(e0)
    pieces: list[str] = []
    for k, c in enumerate(vector):
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        size = abs(c)
        if k == 0:
            body = str(size)
        else:
            mono = zeta if k == 1 else zeta + rf"^{{{k}}}"
            body = mono if size == 1 else f"{size} {mono}"
        if not pieces:
            pieces.append(body if sign == "+" else f"-{body}")
        else:
            pieces.append(f"{sign} {body}")
    return " ".join(pieces) if pieces else "0"


def at_value_ascii(v: AtValue, e0: int | None) -> str:
    """One evaluation value rendered in ascii."""
    if v.vector is not None:
        return zeta_vector_ascii(v.vector, int(e0))
    return str(v.rational)


def _evaluate(card_rows: list[BrickRow], qint: sp.Poly, n: int, point: str):
    """The exact values of [n]_q and every brick at the parsed point."""
    kind, parsed = parse_point(point)
    values: list[AtValue] = []
    if kind == "rational":
        label = str(sp.Rational(parsed.numerator, parsed.denominator))
        values.append(
            AtValue(
                label=formatter.qint_label(n),
                rational=eval_at_rational(qint, parsed),
                vector=None,
            )
        )
        for row in card_rows:
            values.append(
                AtValue(
                    label=formatter.phi_label(row.e),
                    rational=eval_at_rational(row.phi, parsed),
                    vector=None,
                )
            )
        return label, kind, values
    e0 = int(parsed)
    label = formatter.zeta_label(e0)
    values.append(
        AtValue(
            label=formatter.qint_label(n),
            rational=None,
            vector=eval_at_zeta(qint, e0),
        )
    )
    for row in card_rows:
        values.append(
            AtValue(
                label=formatter.phi_label(row.e),
                rational=None,
                vector=eval_at_zeta(row.phi, e0),
            )
        )
    return label, kind, values


def parse_lcm_subset(text: str) -> list[int]:
    """Read a comma-separated divisor subset, e.g. '2,3,4'."""
    out: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            raise ValueError(
                f"the lcm subset is comma-separated whole numbers, got {token!r}"
            ) from None
    if not out:
        raise ValueError("the lcm subset is empty")
    return out


def bricks_card(
    n: int, lcm_subset: list[int] | None = None, at: str | None = None
) -> BricksCard:
    """The cyclotomic reference card of [n]_q, exact over Z[q].

    n is a positive integer. lcm_subset, when given, is a list of divisors of
    n whose least common multiple the card reports. at, when given, is an
    evaluation point: a rational like '1/2' or a root of unity in the
    formatter's zeta spelling; every factor is then evaluated exactly.

    Each row's value at 1 is computed by evaluation and asserted against the
    closed rule (p for a prime power e = p^k, 1 otherwise) in both
    directions; a disagreement raises rather than printing a wrong card.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")
    qint = sp.Poly(sp.expand(q_int(n)), q, domain="ZZ")
    rows: list[BrickRow] = []
    for e in _brick_divisors(n):
        phi = sp.Poly(sp.cyclotomic_poly(e, q), q, domain="ZZ")
        deg = int(phi.degree())
        if deg != int(sp.totient(e)):
            raise ArithmeticError(f"degree of the brick of index {e} is not phi(e)")
        at_1 = int(phi.eval(1))
        rule = prime_power_rule(e)
        if at_1 != rule:
            raise ArithmeticError(
                f"value at 1 of the brick of index {e} disagrees with the "
                f"prime-power rule: {at_1} vs {rule}"
            )
        rows.append(BrickRow(e=e, phi=phi, deg=deg, at_1=at_1, rule_at_1=rule))

    prime_powers = [r.e for r in rows if r.rule_at_1 != 1]
    composites = [r.e for r in rows if r.rule_at_1 == 1]

    lcm_value: int | None = None
    subset: list[int] | None = None
    if lcm_subset is not None:
        subset = sorted(int(e) for e in lcm_subset)
        bad = [e for e in subset if e < 1 or n % e != 0]
        if bad:
            raise ValueError(
                f"the lcm subset must consist of divisors of {n}, got {bad}"
            )
        lcm_value = lcm(*subset)

    at_point: str | None = None
    at_kind: str | None = None
    at_values: list[AtValue] = []
    if at is not None:
        at_point, at_kind, at_values = _evaluate(rows, qint, n, at)

    return BricksCard(
        n=n,
        qint=qint,
        rows=rows,
        prime_powers=prime_powers,
        composites=composites,
        lcm_subset=subset,
        lcm_value=lcm_value,
        at_point=at_point,
        at_kind=at_kind,
        at_values=at_values,
    )


def _at_order(card: BricksCard) -> int | None:
    """The root-of-unity order of the evaluation point, when it is one."""
    if card.at_kind != "zeta":
        return None
    zeta_prefix = formatter.zeta_label(1)[:-1]
    return int(card.at_point[len(zeta_prefix):])


def card_data(card: BricksCard) -> dict:
    """The stable JSON object of the card (schema in the module docstring)."""
    e0 = _at_order(card)
    at = None
    if card.at_point is not None:
        at = {
            "point": card.at_point,
            "kind": card.at_kind,
            "values": [
                {
                    "factor": v.label,
                    "value": at_value_ascii(v, e0),
                    "is_zero": v.is_zero,
                }
                for v in card.at_values
            ],
        }
    return {
        "n": card.n,
        "qint": formatter.poly_ascii(card.qint.as_expr(), wrap=10**9),
        "rows": [
            {
                "e": r.e,
                "phi": formatter.poly_ascii(r.phi.as_expr(), wrap=10**9),
                "deg": r.deg,
                "phi_at_1": r.at_1,
            }
            for r in card.rows
        ],
        "prime_powers": list(card.prime_powers),
        "composites": list(card.composites),
        "lcm": (
            None
            if card.lcm_subset is None
            else {"subset": list(card.lcm_subset), "value": card.lcm_value}
        ),
        "at": at,
    }


def card_tex(card: BricksCard) -> str:
    """The TeX block of the card, ready to paste into notes.

    Compiles standalone inside a minimal article preamble (amsmath); the CI
    wraps it and runs pdflatex.
    """
    e0 = _at_order(card)
    lines = [
        r"\begin{align*}",
        rf"{formatter.qint_tex(card.n)} &= "
        + formatter.poly_tex(card.qint.as_expr())
        + r" \\",
    ]
    for r in card.rows:
        lines.append(
            rf"{formatter.phi_tex(r.e)} &= {formatter.poly_tex(r.phi.as_expr())},"
            rf" \quad \deg = {r.deg}, \quad"
            rf" {formatter.phi_tex(r.e)}(1) = {r.at_1} \\"
        )
    pp = ", ".join(str(e) for e in card.prime_powers) or r"\varnothing"
    comp = ", ".join(str(e) for e in card.composites) or r"\varnothing"
    lines.append(
        r"\text{prime powers} &= \{" + pp + r"\}, \quad"
        r" \text{composite non-prime-powers} = \{" + comp + r"\}"
    )
    if card.lcm_subset is not None:
        subset = ", ".join(str(e) for e in card.lcm_subset)
        lines.append(
            rf"\\ \operatorname{{lcm}}\{{{subset}\}} &= {card.lcm_value}"
        )
    if card.at_point is not None:
        point_tex = (
            formatter.zeta_tex(e0) if e0 is not None else sp.latex(sp.Rational(card.at_point))
        )
        for v in card.at_values:
            value_tex = (
                zeta_vector_tex(v.vector, e0)
                if v.vector is not None
                else sp.latex(v.rational)
            )
            label_tex = v.label
            if label_tex.startswith(formatter.phi_label(1)[:-1]):
                e = int(label_tex.split(formatter.phi_label(1)[:-1], 1)[1])
                label_tex = formatter.phi_tex(e)
            else:
                label_tex = formatter.qint_tex(card.n)
            lines.append(
                rf"\\ {label_tex}\bigl({point_tex}\bigr) &= {value_tex}"
            )
    lines.append(r"\end{align*}")
    return "\n".join(lines)
