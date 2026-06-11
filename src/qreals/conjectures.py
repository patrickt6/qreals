"""A conjecture registry with a uniform falsification harness.

Each registered conjecture is one entry: a name, the exact statement in
neutral mathematical prose, a deterministic iterator over instances, a
predicate, and a nearest-miss metric. The runner scans instances in order,
stops at the FIRST counterexample (printing the full dossier of the offending
instance and exiting 1), and otherwise reports the range covered, the
instance count, the wall time, and the three nearest misses by the registered
metric (exiting 0).

Long scans checkpoint to a state file at a fixed interval and resume with
``--resume``; the final report of a resumed run is byte-identical to the
report of an uninterrupted run except for the wall-time field.

Registered names:
    divisor        every cyclotomic index of S divides d
    sqrt-law       all roots of S on the unit circle forces a^2 = 1 (mod d)
    indices-2ju    product-law discrepancy indices have the form 2^j u, u odd
    mult-two       no cyclotomic factor of any S has multiplicity above 2
    floor3         the q-continuant is injective on tails with entries >= 3
    negsum-period  the continued fraction period decides negation-sum
                   finiteness

One deliberately false entry, ``planted-degbound``, is registered but hidden
from the listing; the test suite uses it to prove the harness finds a minimal
counterexample (gate G5.1).

JSON schema (the --json output of ``qreals conj NAME``; keys are stable):
    name               str    the conjecture name
    statement          str    the registered statement
    space              str    the instance space and its ordering
    miss_metric        str    the registered nearest-miss metric
    until              int    the scan bound N
    range              str    the range covered, written out
    instances_checked  int    instances consumed by the scan
    wall_time_seconds  float  total wall time (the only nondeterministic key)
    survived           bool   true when no counterexample was found
    counterexample     null | {label: str, violates: str, dossier: [str]}
    nearest_misses     [ {miss: int, label: str} ]  at most three, nearest
                       first (smaller miss = nearer to a counterexample)

``registry_data()`` is the stable JSON object of ``qreals conj list``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Callable, Iterator

import sympy as sp

from . import formatter
from .rational import q

_STATE_FORMAT = 1


# ---------------------------------------------------------------------------
# outcome of one predicate check
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """The verdict of one instance check.

    ok is the predicate. miss, when not None, is the registered nearest-miss
    value of the instance (an int; smaller = nearer to a counterexample) and
    label describes the instance for the miss table. On a violation,
    violates restates what failed in one line and dossier carries the full
    printout of the offending instance.
    """

    ok: bool
    miss: int | None = None
    label: str = ""
    violates: str = ""
    dossier: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Conjecture:
    """One registry entry: statement, instance space, predicate, metric."""

    name: str
    statement: str
    space: str
    miss_metric: str
    default_until: int
    instances: Callable[[int], Iterator[Any]]
    check: Callable[[Any, dict], Outcome]
    range_label: Callable[[int], str]
    hidden: bool = False


# ---------------------------------------------------------------------------
# shared instance spaces and dossiers
# ---------------------------------------------------------------------------


def _fractions(until: int) -> Iterator[list[int]]:
    """Reduced fractions a/d with 2 <= d <= until, ascending d then a."""
    for d in range(2, until + 1):
        for a in range(1, d):
            if gcd(a, d) == 1:
                yield [a, d]


def _fraction_range(until: int) -> str:
    return f"reduced fractions a/d with 2 <= d <= {until}, ascending d then a"


def _fraction_dossier_lines(a: int, d: int) -> list[str]:
    """The full denom dossier of a/d, one line per fact, ASCII."""
    from . import denom as denom_mod

    p = denom_mod.denom_dossier(a, d)
    data = denom_mod.dossier_data(p)
    lines = [
        f"fraction: {p.a}/{p.d}",
        f"continued fraction: {denom_mod.cf_str(p.cf)}",
        f"S(q) = {data['S']}",
        f"S(q) = {data['S_factored']}",
        f"index set T = {data['T']}",
        f"deg S = {data['deg_S']}, d - 1 = {data['deg_bound']}",
        f"S(1) = {data['S_at_1']}",
        f"class = {data['klass']}",
        formatter.congruence_ascii("a^2", str(data["a_sq_mod_d"]), p.d),
    ]
    for s in data["splits"]:
        realized = "  (realized)" if s["realized"] else ""
        lines.append(
            f"split {p.d} = {s['d_plus']} * {s['d_minus']}: "
            f"{s['discrepancy_class']}, discrepancy {s['discrepancy']}{realized}"
        )
    return lines


def _cofactor_degree(p) -> int:
    return sum(int(sp.degree(f, q)) * m for f, m in p.cofactor_factors)


# ---------------------------------------------------------------------------
# the six conjectures and the planted entry
# ---------------------------------------------------------------------------


def _check_divisor(inst: list[int], state: dict) -> Outcome:
    from .denom import denom_dossier

    a, d = inst
    p = denom_dossier(a, d)
    bad = sorted(e for e in p.multiplicities if d % e)
    if bad:
        return Outcome(
            ok=False,
            violates=(
                f"{formatter.phi_label(bad[0])} divides S but {bad[0]} "
                f"does not divide d = {d}"
            ),
            dossier=_fraction_dossier_lines(a, d),
        )
    deg = _cofactor_degree(p)
    if deg:
        return Outcome(
            ok=True,
            miss=-deg,
            label=f"{a}/{d}: non-cyclotomic cofactor of degree {deg}",
        )
    return Outcome(ok=True)


def _check_sqrt_law(inst: list[int], state: dict) -> Outcome:
    from .denom import denom_dossier

    a, d = inst
    p = denom_dossier(a, d)
    a_sq = (a * a) % d
    if p.is_cyclotomic_product:
        if a_sq != 1 % d:
            return Outcome(
                ok=False,
                violates=(
                    "every root of S is a root of unity yet "
                    + formatter.congruence_ascii("a^2", str(a_sq), d)
                ),
                dossier=_fraction_dossier_lines(a, d),
            )
        return Outcome(ok=True)
    if a_sq != 1 % d:
        deg = _cofactor_degree(p)
        return Outcome(
            ok=True,
            miss=deg,
            label=(
                f"{a}/{d}: a^2 = {a_sq} (mod {d}) and the non-cyclotomic "
                f"part has degree {deg}"
            ),
        )
    return Outcome(ok=True)


def _v2(n: int) -> int:
    v = 0
    while n % 2 == 0:
        n //= 2
        v += 1
    return v


def _check_indices_2ju(inst: list[int], state: dict) -> Outcome:
    from .denom import denom_dossier

    a, d = inst
    p = denom_dossier(a, d)
    best: tuple[int, str] | None = None
    for s in p.splits:
        if not s.realized or s.d_plus == 1:
            continue
        indices = sorted(set(s.num_exponents) | set(s.den_exponents))
        odd = [e for e in indices if e % 2 == 1]
        if odd:
            return Outcome(
                ok=False,
                violates=(
                    f"discrepancy index {odd[0]} of the split "
                    f"{d} = {s.d_plus} * {s.d_minus} is odd"
                ),
                dossier=_fraction_dossier_lines(a, d),
            )
        if indices:
            v = min(_v2(e) for e in indices)
            label = (
                f"{a}/{d}: split {d} = {s.d_plus} * {s.d_minus}, "
                f"discrepancy indices {indices}, least 2-adic valuation {v}"
            )
            if best is None or v < best[0]:
                best = (v, label)
    if best is not None:
        return Outcome(ok=True, miss=best[0], label=best[1])
    return Outcome(ok=True)


def _check_mult_two(inst: list[int], state: dict) -> Outcome:
    from .denom import denom_dossier

    a, d = inst
    p = denom_dossier(a, d)
    if not p.multiplicities:
        return Outcome(ok=True)
    e_max, m_max = max(p.multiplicities.items(), key=lambda em: (em[1], em[0]))
    if m_max > 2:
        return Outcome(
            ok=False,
            violates=(
                f"{formatter.phi_label(e_max)} divides S with "
                f"multiplicity {m_max}"
            ),
            dossier=_fraction_dossier_lines(a, d),
        )
    if m_max == 2:
        return Outcome(
            ok=True,
            miss=3 - m_max,
            label=(
                f"{a}/{d}: {formatter.phi_label(e_max)} appears with "
                "multiplicity 2"
            ),
        )
    return Outcome(ok=True)


def _check_planted_degbound(inst: list[int], state: dict) -> Outcome:
    from .denom import denom_dossier

    a, d = inst
    p = denom_dossier(a, d)
    if p.deg_S >= d - 2:
        return Outcome(
            ok=False,
            violates=f"deg S = {p.deg_S} is not below d - 2 = {d - 2}",
            dossier=_fraction_dossier_lines(a, d),
        )
    return Outcome(ok=True, miss=(d - 2) - p.deg_S - 1, label=f"{a}/{d}")


# -- floor3: the q-continuant on tails ---------------------------------------


def _factored_ascii(poly: sp.Poly) -> str:
    """A polynomial factored over Z[q], cyclotomic factors labelled by index."""
    from .factor import _cyclotomic_index

    content, pairs = sp.factor_list(poly.as_expr(), q)
    return formatter.factored_ascii(content, pairs, _cyclotomic_index)


def _qint_poly(n: int) -> sp.Poly:
    return sp.Poly(sum(q**i for i in range(n)), q, domain="ZZ")


def continuant(tail: list[int]) -> sp.Poly:
    """The q-continuant K of a tail, exact over Z[q].

    K() = 1, K(t1) = [t1]_q, and
    K(t1, ..., tm) = [t1]_q K(t2, ..., tm) - q^(t1 - 1) K(t3, ..., tm).
    """
    if not tail:
        return sp.Poly(1, q, domain="ZZ")
    k_after = sp.Poly(1, q, domain="ZZ")
    k_here = _qint_poly(tail[-1])
    for t in reversed(tail[:-1]):
        k_here, k_after = (
            _qint_poly(t) * k_here - sp.Poly(q ** (t - 1), q, domain="ZZ") * k_after,
            k_here,
        )
    return k_here


def _tails(until: int, floor: int = 3) -> Iterator[list[int]]:
    """Tails with entries >= floor, ordered by entry sum then lexicographic."""

    def comps(s: int) -> Iterator[list[int]]:
        if s == 0:
            yield []
            return
        for head in range(floor, s + 1):
            rest = s - head
            if rest == 0:
                yield [head]
            elif rest >= floor:
                for tail in comps(rest):
                    yield [head] + tail

    for s in range(floor, until + 1):
        yield from comps(s)


def _tail_str(tail: list[int]) -> str:
    return "(" + ", ".join(str(t) for t in tail) + ")"


def _check_floor3(inst: list[int], state: dict) -> Outcome:
    tail = list(inst)
    poly = continuant(tail)
    coeffs = [int(c) for c in reversed(poly.all_coeffs())]
    key = ",".join(str(c) for c in coeffs)
    seen: dict[str, list[int]] = state.setdefault("seen", {})
    if key in seen and seen[key] != tail:
        other = seen[key]
        return Outcome(
            ok=False,
            violates=(
                f"tails {_tail_str(other)} and {_tail_str(tail)} share the "
                "same q-continuant"
            ),
            dossier=[
                f"tail: {_tail_str(other)}",
                f"tail: {_tail_str(tail)}",
                f"K(q) = {formatter.poly_ascii(poly.as_expr())}",
                f"K(q) = {_factored_ascii(poly)}",
                f"deg K = {int(poly.degree())}, K(1) = {int(poly.eval(1))}",
            ],
        )
    coarse_key = f"{int(poly.degree())}:{int(poly.eval(1))}"
    bucket: list[list[Any]] = state.setdefault("coarse", {}).setdefault(coarse_key, [])
    best: tuple[int, str] | None = None
    for other_coeffs, other_tail in bucket:
        if other_coeffs == coeffs:
            continue
        diffs = sum(1 for x, y in zip(other_coeffs, coeffs) if x != y)
        diffs += abs(len(other_coeffs) - len(coeffs))
        label = (
            f"tails {_tail_str(other_tail)} and {_tail_str(tail)}: equal "
            f"degree and value at 1, {diffs} differing coefficients"
        )
        if best is None or diffs < best[0]:
            best = (diffs, label)
    seen[key] = tail
    bucket.append([coeffs, tail])
    if best is not None:
        return Outcome(ok=True, miss=best[0], label=best[1])
    return Outcome(ok=True)


# -- negsum-period: the period decides negation-sum finiteness ---------------


def _nonsquares(until: int) -> Iterator[int]:
    for D in range(2, until + 1):
        if sp.integer_nthroot(D, 2)[1]:
            continue
        yield D


def _surd_period(D: int) -> list[int]:
    _, period = sp.continued_fraction_periodic(0, 1, D)
    return [int(t) for t in period]


def _check_negsum_period(inst: int, state: dict) -> Outcome:
    from .arithmetic import finite_xnegx

    D = int(inst)
    period = _surd_period(D)
    finite = bool(finite_xnegx(f"sqrt({D})", order=48))
    key = ",".join(str(t) for t in period)
    periods: dict[str, list[Any]] = state.setdefault("periods", {})
    if key in periods:
        first_d, first_finite = periods[key]
        if bool(first_finite) != finite:
            return Outcome(
                ok=False,
                violates=(
                    f"sqrt({first_d}) and sqrt({D}) share the period "
                    f"({key}) with opposite finiteness verdicts"
                ),
                dossier=[
                    f"D = {first_d}: period ({key}), "
                    f"finite = {bool(first_finite)}",
                    f"D = {D}: period ({key}), finite = {finite}",
                ],
            )
    else:
        periods[key] = [D, finite]
    by_len: dict[str, list[Any]] = state.setdefault("by_length", {})
    len_key = str(len(period))
    slot = by_len.setdefault(len_key, [None, None])
    idx = 0 if finite else 1
    if slot[idx] is None:
        slot[idx] = D
    other = slot[1 - idx]
    if other is not None:
        return Outcome(
            ok=True,
            miss=len(period),
            label=(
                f"sqrt({other}) and sqrt({D}) share period length "
                f"{len(period)} with opposite verdicts"
            ),
        )
    return Outcome(ok=True)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Conjecture] = {
    c.name: c
    for c in [
        Conjecture(
            name="divisor",
            statement=(
                "Every cyclotomic index of the denominator S of a reduced "
                "fraction a/d divides d."
            ),
            space="reduced fractions a/d up to denominator N",
            miss_metric=(
                "minus the degree of the non-cyclotomic cofactor of S "
                "(more non-cyclotomic mass = nearer)"
            ),
            default_until=60,
            instances=_fractions,
            check=_check_divisor,
            range_label=_fraction_range,
        ),
        Conjecture(
            name="sqrt-law",
            statement=(
                "If every root of the denominator S of a reduced fraction "
                "a/d is a root of unity, then a^2 = 1 (mod d)."
            ),
            space="reduced fractions a/d up to denominator N",
            miss_metric=(
                "degree of the non-cyclotomic part of S among instances "
                "with a^2 != 1 (mod d) (smaller = nearer)"
            ),
            default_until=60,
            instances=_fractions,
            check=_check_sqrt_law,
            range_label=_fraction_range,
        ),
        Conjecture(
            name="indices-2ju",
            statement=(
                "Every cyclotomic index in the product-law discrepancy "
                "[d+]_q [d-]_q / S of a realized coprime split d = d+ d- "
                "has the form 2^j u with u odd and j >= 1."
            ),
            space="reduced fractions a/d up to denominator N",
            miss_metric=(
                "least 2-adic valuation among the discrepancy indices "
                "(valuation 1 = nearest to an odd index)"
            ),
            default_until=60,
            instances=_fractions,
            check=_check_indices_2ju,
            range_label=_fraction_range,
        ),
        Conjecture(
            name="mult-two",
            statement=(
                "No cyclotomic factor of the denominator S of any reduced "
                "fraction appears with multiplicity above 2."
            ),
            space="reduced fractions a/d up to denominator N",
            miss_metric="3 minus the largest multiplicity (0 would violate)",
            default_until=60,
            instances=_fractions,
            check=_check_mult_two,
            range_label=_fraction_range,
        ),
        Conjecture(
            name="floor3",
            statement=(
                "The q-continuant is injective on tails whose entries are "
                "all at least 3."
            ),
            space="tails with entries >= 3, ordered by entry sum up to N",
            miss_metric=(
                "differing-coefficient count between two continuants of "
                "equal degree and equal value at 1"
            ),
            default_until=24,
            instances=_tails,
            check=_check_floor3,
            range_label=lambda until: (
                f"tails with entries >= 3 and entry sum <= {until}, "
                "ascending sum then lexicographic"
            ),
        ),
        Conjecture(
            name="negsum-period",
            statement=(
                "Whether [sqrt(D)]_q + [-sqrt(D)]_q is a finite Laurent "
                "polynomial is decided by the continued fraction period of "
                "sqrt(D)."
            ),
            space="non-square integers D up to N",
            miss_metric=(
                "shared period length between two surds with opposite "
                "finiteness verdicts (shorter = nearer)"
            ),
            default_until=60,
            instances=_nonsquares,
            check=_check_negsum_period,
            range_label=lambda until: f"non-square integers 2 <= D <= {until}",
        ),
        Conjecture(
            name="planted-degbound",
            statement="deg S < d - 2 for every reduced fraction a/d.",
            space="reduced fractions a/d up to denominator N",
            miss_metric="d - 3 - deg S (0 = nearest)",
            default_until=60,
            instances=_fractions,
            check=_check_planted_degbound,
            range_label=_fraction_range,
            hidden=True,
        ),
    ]
}


def registry_data() -> dict:
    """The stable JSON object of `qreals conj list` (visible entries only)."""
    return {
        "conjectures": [
            {
                "name": c.name,
                "statement": c.statement,
                "space": c.space,
                "miss_metric": c.miss_metric,
                "default_until": c.default_until,
            }
            for c in REGISTRY.values()
            if not c.hidden
        ]
    }


def registry_lines() -> list[str]:
    """The human listing of the registry, one block per visible entry."""
    lines: list[str] = []
    for c in REGISTRY.values():
        if c.hidden:
            continue
        lines.append(c.name)
        lines.append(f"  statement: {c.statement}")
        lines.append(f"  instances: {c.space}")
        lines.append(f"  nearest-miss metric: {c.miss_metric}")
        lines.append(f"  default --until: {c.default_until}")
    return lines


# ---------------------------------------------------------------------------
# the runner: scan, checkpoint, resume, report
# ---------------------------------------------------------------------------


def default_state_path(name: str) -> Path:
    from .store import user_data_dir

    return user_data_dir() / "conj" / f"{name}.state.json"


def _fresh_state(name: str, until: int) -> dict:
    return {
        "format": _STATE_FORMAT,
        "name": name,
        "until": until,
        "cursor": 0,
        "checked": 0,
        "elapsed": 0.0,
        "done": False,
        "state": {},
        "misses": [],
        "counterexample": None,
    }


def _load_state(path: Path, name: str, until: int) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != _STATE_FORMAT or raw.get("name") != name:
        raise ValueError(f"the state file {path} does not belong to '{name}'")
    if raw.get("until") != until:
        raise ValueError(
            f"the state file {path} was written with --until {raw.get('until')}; "
            "resume with the same bound or start fresh without --resume"
        )
    return raw


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


def _record_miss(state: dict, miss: int, label: str) -> None:
    entry = [miss, state["cursor"], label]
    misses = state["misses"]
    misses.append(entry)
    misses.sort(key=lambda m: (m[0], m[1]))
    del misses[3:]


def run_conjecture(
    name: str,
    until: int | None = None,
    resume: bool = False,
    state_file: str | Path | None = None,
    checkpoint_seconds: float = 60.0,
) -> dict:
    """Scan one conjecture and return the report object (see module doc).

    The report carries everything both renderings (human and JSON) print;
    wall_time_seconds is the only nondeterministic field.
    """
    if name not in REGISTRY:
        known = ", ".join(c for c in REGISTRY if not REGISTRY[c].hidden)
        raise ValueError(f"unknown conjecture '{name}'; registered: {known}")
    entry = REGISTRY[name]
    until = entry.default_until if until is None else int(until)
    path = Path(state_file) if state_file else default_state_path(name)

    if resume and path.exists():
        state = _load_state(path, name, until)
    else:
        state = _fresh_state(name, until)

    start = time.monotonic()
    last_checkpoint = start

    if not state["done"]:
        it = entry.instances(until)
        for _ in range(state["cursor"]):
            next(it)
        for inst in it:
            out = entry.check(inst, state["state"])
            state["cursor"] += 1
            state["checked"] += 1
            if not out.ok:
                state["counterexample"] = {
                    "label": out.label or _instance_label(inst),
                    "violates": out.violates,
                    "dossier": out.dossier,
                }
                break
            if out.miss is not None:
                _record_miss(state, int(out.miss), out.label)
            now = time.monotonic()
            if now - last_checkpoint >= checkpoint_seconds:
                state["elapsed"] += now - start
                start = now
                last_checkpoint = now
                _write_state(path, state)
        state["done"] = True
        state["elapsed"] += time.monotonic() - start
        _write_state(path, state)

    return {
        "name": entry.name,
        "statement": entry.statement,
        "space": entry.space,
        "miss_metric": entry.miss_metric,
        "until": until,
        "range": entry.range_label(until),
        "instances_checked": state["checked"],
        "wall_time_seconds": round(float(state["elapsed"]), 2),
        "survived": state["counterexample"] is None,
        "counterexample": state["counterexample"],
        "nearest_misses": [
            {"miss": m[0], "label": m[2]} for m in state["misses"]
        ],
    }


def _instance_label(inst: Any) -> str:
    if isinstance(inst, list) and len(inst) == 2 and all(
        isinstance(t, int) for t in inst
    ):
        return f"{inst[0]}/{inst[1]}"
    if isinstance(inst, list):
        return _tail_str(inst)
    return str(inst)


def report_lines(report: dict) -> list[str]:
    """The human rendering of a report, built from the same object as --json."""
    lines = [
        f"conjecture: {report['name']}",
        f"statement: {report['statement']}",
    ]
    if report["counterexample"] is not None:
        c = report["counterexample"]
        lines.append(f"COUNTEREXAMPLE: {c['label']}")
        lines.append(f"violates: {c['violates']}")
        lines.extend("  " + line for line in c["dossier"])
        return lines
    lines.append(f"range covered: {report['range']}")
    lines.append(f"instances checked: {report['instances_checked']}")
    lines.append(f"wall time: {report['wall_time_seconds']:.2f} s")
    lines.append(f"nearest misses (metric: {report['miss_metric']}):")
    if report["nearest_misses"]:
        for m in report["nearest_misses"]:
            lines.append(f"  miss = {m['miss']}  {m['label']}")
    else:
        lines.append("  none recorded in this range")
    lines.append("verdict: survives the scanned range")
    return lines
