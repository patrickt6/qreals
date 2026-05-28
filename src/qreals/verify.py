"""Inline cross-checks for a qreals result, the everyday trust signal.

After every computation qreals runs the cheap checks that apply to that input
and prints one line, for example::

    verified: q=1 matches 3/2, exact = truncated to 12

This module is core. It imports only sympy and the other core modules, never
the interface or the certificate layer, so the stamp works with `pip install
qreals` alone and never needs a TeX engine. It writes nothing.

Each check recomputes a value a second, independent way and compares:

- q=1 specialisation. For a rational input, setting q=1 collapses [n]_q back
  to n, so [p/s]_q at q=1 must equal the ordinary p/s (RAT Corollary 1.7).
- exact = truncated. For a rational input the continued fraction terminates,
  so the truncated series must equal the Taylor expansion of the exact
  rational function computed by `q_rational`.
- truncation stable. The first N coefficients of [x]_q must not change when
  more are requested (REAL Theorem 1).
- shift law. [x+1]_q = q [x]_q + 1 (REAL eqn 3). Recomputing [x+1]_q from its
  own continued fraction and comparing to q*[x]_q + 1 is an algorithm-level
  cross-check that holds for every real x.

For an irrational input the first two checks cannot run (there is no
terminating rational function and the series diverges at q=1); the stamp says
so plainly rather than claiming a pass.

The mathematics follows Morier-Genoud and Ovsienko, "q-deformed rationals and
q-continued fractions", Forum Math. Sigma 8 (2020); see docs/CORRECTNESS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import sympy as sp

from . import series as _series
from .arithmetic import (
    _jouteur_neg,
    _qreal_series,
    q_add,
    q_mul,
    radius,
)
from .gosper import gosper_coeffs
from .rational import q, q_int, q_rational
from .truncated import q_real_truncated

# A check is "pass", "fail", "na" (cannot run for this input), or "error"
# (the check itself raised). The stamp claims success only when every check
# that ran passed.
_GOOD = "pass"
_BAD = "fail"
_NA = "na"
_ERR = "error"


@dataclass(frozen=True)
class Check:
    """One independent cross-check and how it came out."""

    label: str  # short phrasing for the one-line stamp
    status: str  # _GOOD | _BAD | _NA | _ERR
    detail: str  # full sentence for the certificate cross-check section


@dataclass
class Stamp:
    """The set of checks run for one computation, plus the one-line summary."""

    checks: list[Check] = field(default_factory=list)

    @property
    def ran(self) -> list[Check]:
        return [c for c in self.checks if c.status in (_GOOD, _BAD, _ERR)]

    @property
    def ok(self) -> bool:
        """True when at least one check ran and none failed or errored."""
        ran = self.ran
        return bool(ran) and all(c.status == _GOOD for c in ran)

    def line(self) -> str:
        """The one-line stamp, honest about what did and did not run."""
        if not self.checks:
            return ""
        passed = [c.label for c in self.checks if c.status == _GOOD]
        failed = [c.label for c in self.checks if c.status == _BAD]
        errored = [c.label for c in self.checks if c.status == _ERR]
        na = [c.label for c in self.checks if c.status == _NA]
        if failed:
            return "verification FAILED: " + "; ".join(failed)
        parts: list[str] = []
        if passed:
            parts.append("verified: " + ", ".join(passed))
        else:
            parts.append("verified: no check applied to this input")
        if errored:
            parts.append("could not check: " + "; ".join(errored))
        if na:
            parts.append("n/a: " + "; ".join(na))
        return "; ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "line": self.line(),
            "ok": self.ok,
            "checks": [
                {"label": c.label, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
        }


def _taylor_coeffs(expr: sp.Expr, n: int) -> list[int]:
    """First n integer Taylor coefficients of a q-rational at q = 0."""
    series = sp.series(expr, q, 0, n).removeO()
    return [int(series.coeff(q, k)) for k in range(n)]


def _check_q_at_one(expr: sp.Expr, ordinary: sp.Rational, name: str) -> Check:
    value = sp.simplify(expr.subs(q, 1))
    passed = sp.simplify(value - ordinary) == 0
    return Check(
        label=f"q=1 matches {ordinary}",
        status=_GOOD if passed else _BAD,
        detail=f"{name} at q=1 is {value}, the ordinary value {ordinary}",
    )


def _check_exact_equals_truncated(expr: sp.Expr, x_repr: str, n: int) -> Check:
    try:
        exact = _taylor_coeffs(expr, n)
        trunc = q_real_truncated(x_repr, n)
    except Exception as exc:  # noqa: BLE001 - report rather than crash the stamp
        return Check(
            label="exact = truncated",
            status=_ERR,
            detail=f"could not compare exact and truncated paths: {exc}",
        )
    passed = exact == trunc
    return Check(
        label=f"exact = truncated to {n}",
        status=_GOOD if passed else _BAD,
        detail=(
            f"the Taylor expansion of the exact rational function and the "
            f"truncated series agree on q^0..q^{n - 1}: {exact}"
            if passed
            else f"exact {exact} differs from truncated {trunc}"
        ),
    )


def _check_truncation_stable(x_repr: str, n: int) -> Check:
    try:
        base = q_real_truncated(x_repr, n)
        deeper = q_real_truncated(x_repr, n + 4)
    except Exception as exc:  # noqa: BLE001
        return Check(
            label="truncation stable",
            status=_ERR,
            detail=f"could not test truncation stability: {exc}",
        )
    passed = deeper[:n] == base
    return Check(
        label=f"truncation stable to {n}",
        status=_GOOD if passed else _BAD,
        detail=(
            f"the first {n} coefficients are unchanged when {n + 4} are asked for"
            if passed
            else f"coefficients moved: {base} then {deeper[:n]}"
        ),
    )


def _check_shift_law(x_repr: str, n: int) -> Check:
    try:
        base = q_real_truncated(x_repr, n)
        raised = q_real_truncated(f"({x_repr})+1", n)
    except Exception as exc:  # noqa: BLE001
        return Check(
            label="shift law [x+1]=q[x]+1",
            status=_ERR,
            detail=f"could not test the shift law: {exc}",
        )
    expected = ([1] + base)[:n]
    passed = raised == expected
    return Check(
        label="shift law [x+1]=q[x]+1",
        status=_GOOD if passed else _BAD,
        detail=(
            "[x+1]_q computed from its own continued fraction equals q*[x]_q + 1"
            if passed
            else f"[x+1]_q is {raised}, but q*[x]_q + 1 is {expected}"
        ),
    )


def _rational_of(x_repr: str) -> sp.Rational | None:
    value = sp.sympify(x_repr)
    if value.is_rational:
        return sp.Rational(value)
    return None


def verify_rational(p: int, s: int, n: int = 12) -> Stamp:
    """Checks for an exact [p/s]_q."""
    expr = sp.sympify(q_rational(p, s))
    ordinary = sp.Rational(p, s)
    return Stamp(
        [
            _check_q_at_one(expr, ordinary, f"[{p}/{s}]_q"),
            _check_exact_equals_truncated(expr, f"{p}/{s}", n),
        ]
    )


def verify_qint(n: int, prec: int = 12) -> Stamp:
    """Checks for a q-integer [n]_q."""
    expr = q_int(n)
    checks = [_check_q_at_one(expr, sp.Integer(n), f"[{n}]_q")]
    if n >= 0:
        checks.append(_check_exact_equals_truncated(expr, str(n), max(n + 2, prec)))
    else:
        checks.append(
            Check(
                label="exact = truncated",
                status=_NA,
                detail=f"the truncated series path is built for n >= 0, here n = {n}",
            )
        )
    return Stamp(checks)


def verify_series(x_repr: str, n: int) -> Stamp:
    """Checks for a [x]_q series read off to N coefficients."""
    n = max(int(n), 1)
    rational = _rational_of(x_repr)
    checks: list[Check] = [
        _check_truncation_stable(x_repr, n),
        _check_shift_law(x_repr, n),
    ]
    if rational is not None:
        expr = sp.sympify(q_rational(rational.p, rational.q))
        checks.append(_check_q_at_one(expr, rational, f"[{rational}]_q"))
        checks.append(_check_exact_equals_truncated(expr, str(rational), n))
    else:
        checks.append(
            Check(
                label="q=1 specialisation",
                status=_NA,
                detail=f"the series [{x_repr}]_q diverges at q=1; q=1 is for rationals",
            )
        )
        checks.append(
            Check(
                label="exact rational function",
                status=_NA,
                detail=f"{x_repr} is irrational, so the continued fraction does not terminate",
            )
        )
    return Stamp(checks)


def _as_fraction(x_repr: str) -> Fraction | None:
    value = sp.sympify(x_repr)
    if value.is_rational:
        r = sp.Rational(value)
        return Fraction(int(r.p), int(r.q))
    return None


def verify_arith(x: str, y: str, n: int, op: str) -> Stamp:
    """Checks for a series sum or product [x]_q (+/*) [y]_q."""
    cx, cy = q_real_truncated(x, n), q_real_truncated(y, n)
    if op == "mul":
        lib = q_mul(x, y, n)
        comb = [0] * n
        for i, a in enumerate(cx):
            for j, b in enumerate(cy):
                if i + j < n:
                    comb[i + j] += a * b
        comb_label = "matches [x]_q * [y]_q series"
    else:
        lib = q_add(x, y, n)
        comb = [a + b for a, b in zip(cx, cy)]
        comb_label = "matches [x]_q + [y]_q series"
    checks = [
        Check(
            label=comb_label,
            status=_GOOD if lib == comb else _BAD,
            detail="the result equals the term-by-term combination of the two q-series",
        )
    ]
    fx, fy = _as_fraction(x), _as_fraction(y)
    if fx is not None and fy is not None:
        eng = gosper_coeffs(fx, fy, op, n)
        checks.append(
            Check(
                label="bihomographic engine agrees",
                status=_GOOD if lib == eng else _BAD,
                detail="the independent q-Gosper state machine returns the same coefficients",
            )
        )
    else:
        checks.append(
            Check(
                label="bihomographic engine agrees",
                status=_NA,
                detail="the engine path needs terminating continued fractions (rationals)",
            )
        )
    return Stamp(checks)


def verify_negation(x: str, n: int) -> Stamp:
    """Checks for the Jouteur negation [-x]_q: it must be an involution."""
    prec = n + 8
    twice = _series.normalise(
        _jouteur_neg(_jouteur_neg(_qreal_series(x, prec), prec), prec)
    )
    v, c = twice
    ok = v == 0 and c[:n] == q_real_truncated(x, n)
    return Stamp(
        [
            Check(
                label="negation involutive",
                status=_GOOD if ok else _BAD,
                detail="applying the Jouteur negation twice returns [x]_q (REAL/Jouteur PGL_2 action)",
            )
        ]
    )


def verify_radius(x: str, n: int) -> Stamp:
    """Checks for the radius estimate: it is biased high and decreases with N."""
    coarse = radius(x, max(2, n // 2))
    fine = radius(x, n)
    ok = fine <= coarse + 1e-9
    return Stamp(
        [
            Check(
                label="estimate decreases with N",
                status=_GOOD if ok else _BAD,
                detail="the running-max slope estimate is biased high and falls toward the true radius",
            )
        ]
    )


def verify_jumpgap(p: int, s: int) -> Stamp:
    """Checks for a one-sided jump gap of p/s.

    The right version is recomputed by the independent q_rational path (the
    oracle) and compared, and the two q-denominators are specialised at q = 1,
    where each must collapse to the ordinary denominator s.
    """
    from .jumpgap import jumpgap as _jumpgap

    gap = _jumpgap(p, s)
    oracle_ok = sp.simplify(gap.right - q_rational(p, s)) == 0
    plus_one, minus_one = gap.denominators_at_one()
    return Stamp(
        [
            Check(
                label=f"right version matches q_rational({p}, {s})",
                status=_GOOD if oracle_ok else _BAD,
                detail=(
                    "the right version [p/s]_q^+ equals the q_rational oracle "
                    "computed by the independent continued-fraction path"
                ),
            ),
            Check(
                label=f"S^+ and S^- equal {s} at q=1",
                status=_GOOD if (plus_one == s and minus_one == s) else _BAD,
                detail=(
                    f"both q-denominators collapse to the ordinary s = {s} at "
                    f"q = 1 (S^+(1) = {plus_one}, S^-(1) = {minus_one})"
                ),
            ),
        ]
    )


def _series_n(result: dict[str, Any]) -> int:
    data = result.get("data", {})
    coeffs = data.get("coefficients")
    if isinstance(coeffs, list) and coeffs:
        return len(coeffs)
    if "order" in data:
        return int(data["order"]) + 1
    if result.get("kind") == "locked" and "S_n" in data:
        return int(data["S_n"])
    if result.get("kind") == "prefix" and "floor" in data:
        return int(data["floor"]) + 2
    if "n" in data:
        return int(data["n"])
    return 12


def verify(result: dict[str, Any]) -> Stamp:
    """Run the cross-checks that apply to a computation result dict.

    Dispatches on result["kind"]. An unknown or absent kind (the help screen,
    say) yields an empty stamp, which renders as no line at all.
    """
    kind = result.get("kind")
    data = result.get("data", {})
    if kind == "rational":
        return verify_rational(int(data["p"]), int(data["s"]))
    if kind == "qint":
        return verify_qint(int(data["n"]))
    if kind in ("coeffs", "laurent", "prefix", "locked", "shift", "readouts"):
        return verify_series(str(data["x"]), _series_n(result))
    if kind == "arith":
        return verify_arith(
            str(data["x"]), str(data["y"]), int(data["n"]), str(data.get("op", "add"))
        )
    if kind in ("negation", "finite"):
        return verify_negation(str(data["x"]), int(data.get("n", 12)))
    if kind == "radius":
        return verify_radius(str(data["x"]), int(data["n"]))
    if kind == "jumpgap":
        return verify_jumpgap(int(data["p"]), int(data["s"]))
    return Stamp([])
