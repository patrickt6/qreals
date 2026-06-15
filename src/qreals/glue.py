r"""Prime-power sharing-class anomaly hunter.

At a prime-power modulus m = p^e the numerators a coprime to m group by their
q-denominator polynomial S(q) into sharing classes. The generic class is the
two-element pair {a, -a^{-1} mod m}; a class with MORE THAN TWO members is an
anomaly. This module collects those exceptional classes for one modulus, and
scans a range of primes recording, per prime, whether p^e is exceptional and
the residue p mod 4.

A class is found by grouping every numerator a coprime to m (1 <= a < m) by the
coefficient sequence of S(q), exactly as the reverse table groups numerators,
but here every numerator is grouped (including the ones whose S is not a
cyclotomic product, which is where the larger classes actually appear). For a
modulus divisible by p^2 with p == 1 (mod 4) a square root of minus one mod p
explains the enlargement; those roots are reported so the extra structure is
shown, not just observed. Each exceptional class shares one residue mod p, and
that residue is one of the square roots of minus one.

Whether the exceptional primes are exactly those with p == 1 (mod 4) is an open
observation, not a settled theorem: the output reports the residue p mod 4 next
to each prime and states the association only as something checked over the
listed range. The summary never claims it as fact.

Long prime scans checkpoint to a state file at a fixed interval and resume with
``--resume``; the deterministic part of a resumed run's report (the prime rows,
their p mod 4 column, and the exceptional verdicts) is byte-identical to an
uninterrupted run. Only the per-prime timing differs.

JSON schema, single mode (the --json output of ``qreals glue P``, keys stable):
    p              int    the prime
    exp            int    the exponent e
    modulus        int    m = p^e
    mode           str    "single"
    p_mod_4        int    p mod 4, computed
    checked_range  str    the scanned set of numerators, written out
    sqrt_minus_one [int]  square roots of minus one mod p (empty if none)
    classes        [ {numerators: [int], residues_mod_p: [int],
                      sqrt_minus_one_residues: [int], size: int} ]
                   sharing classes of size > 2, ordered by smallest numerator
    exceptional    bool   true when at least one class has size > 2

JSON schema, scan mode (the --json output of ``qreals glue --scan A..B``):
    mode                str    "scan"
    lo, hi              int    the inclusive prime range bounds
    exp                 int    the exponent e
    checked_range       str    the scanned range, written out
    rows                [ {p: int, p_mod_4: int, exceptional: bool} ]
                        one row per prime in [lo, hi], ascending
    exceptional_primes  [int]  the primes whose p^e is exceptional
    timings             [ {p: int, seconds: float} ]  per-prime wall time, the
                        only nondeterministic field
    wall_time_seconds   float  total wall time (nondeterministic)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import sympy as sp

from .denom import denom_dossier

_STATE_FORMAT = 1


@dataclass(frozen=True)
class GlueClass:
    """One sharing class of size greater than two at a prime-power modulus."""

    numerators: list[int]
    residues_mod_p: list[int]
    sqrt_minus_one_residues: list[int]

    @property
    def size(self) -> int:
        return len(self.numerators)


@dataclass(frozen=True)
class GlueReport:
    """The exceptional sharing classes of one prime-power modulus p^e."""

    p: int
    exp: int
    classes: list[GlueClass]
    sqrt_minus_one: list[int]

    @property
    def modulus(self) -> int:
        return self.p**self.exp

    @property
    def exceptional(self) -> bool:
        return any(c.size > 2 for c in self.classes)


@dataclass(frozen=True)
class GlueScanRow:
    """One prime of a scan: its p mod 4 and whether p^e is exceptional."""

    p: int
    p_mod_4: int
    exceptional: bool


def _require_prime(p: int) -> int:
    p = int(p)
    if p < 2 or not sp.isprime(p):
        raise ValueError(f"p must be a prime; {p} is not")
    return p


def sqrt_minus_one(p: int) -> list[int]:
    """The square roots of minus one mod p (a residue i with i^2 == -1 mod p)."""
    p = int(p)
    if p < 2:
        return []
    return sorted(i for i in range(1, p) if (i * i) % p == (p - 1) % p)


def _classes_by_s(m: int) -> dict[tuple, list[int]]:
    """Numerators coprime to m grouped by the coefficient sequence of S(q).

    Every coprime numerator is grouped, the cyclotomic and the non-cyclotomic
    alike, so the larger sharing classes are not skipped. The denominator is
    taken from the engine's dossier with the cofactor factoring switched off,
    so the scan only pays for S itself.
    """
    by_s: dict[tuple, list[int]] = {}
    for a in range(1, m):
        if gcd(a, m) != 1:
            continue
        key = tuple(denom_dossier(a, m, factor_cofactor=False).S.all_coeffs())
        by_s.setdefault(key, []).append(a)
    return by_s


def glue_report(p: int, exp: int = 2) -> GlueReport:
    """The exceptional sharing classes of p^e, exact over Z[q].

    The classes are the groups of more than two numerators that share one
    denominator S; each is reported with its numerators, their residues mod p,
    and the square roots of minus one mod p that the class sits on.
    """
    p = _require_prime(p)
    exp = int(exp)
    if exp < 1:
        raise ValueError("the exponent e must be at least 1")
    m = p**exp
    roots = sqrt_minus_one(p)
    root_set = set(roots)
    classes: list[GlueClass] = []
    for numerators in _classes_by_s(m).values():
        if len(numerators) <= 2:
            continue
        numerators = sorted(numerators)
        residues = [a % p for a in numerators]
        involved = sorted(set(residues) & root_set)
        classes.append(
            GlueClass(
                numerators=numerators,
                residues_mod_p=residues,
                sqrt_minus_one_residues=involved,
            )
        )
    classes.sort(key=lambda c: c.numerators[0])
    return GlueReport(p=p, exp=exp, classes=classes, sqrt_minus_one=roots)


def is_exceptional(p: int, exp: int = 2) -> bool:
    """True when p^e has a sharing class of more than two numerators."""
    return glue_report(p, exp).exceptional


def checked_range_single(m: int) -> str:
    """The documented set of numerators checked at a single modulus."""
    return f"checked numerators a coprime to {m} with 1 <= a < {m}"


def checked_range_scan(lo: int, hi: int, exp: int) -> str:
    """The documented prime range checked by a scan."""
    return f"checked primes p with {lo} <= p <= {hi}, modulus p^{exp}"


def parse_scan_range(text: str) -> tuple[int, int]:
    """Read a scan range a..b with 2 <= a <= b."""
    lo, sep, hi = text.partition("..")
    if not sep:
        raise ValueError("the scan range must be written a..b, e.g. 3..30")
    a, b = int(lo), int(hi)
    if a < 2 or b < a:
        raise ValueError("the scan range needs 2 <= a <= b")
    return a, b


def primes_in_range(lo: int, hi: int) -> list[int]:
    """The primes p with lo <= p <= hi, ascending."""
    return [int(p) for p in sp.primerange(int(lo), int(hi) + 1)]


# --------------------------------------------------------------------------
# JSON objects, built once and shared by the human and --json renderings.
# --------------------------------------------------------------------------


def report_data(report: GlueReport) -> dict:
    """The stable JSON object of one prime-power report (schema in the doc)."""
    return {
        "p": report.p,
        "exp": report.exp,
        "modulus": report.modulus,
        "mode": "single",
        "p_mod_4": report.p % 4,
        "checked_range": checked_range_single(report.modulus),
        "sqrt_minus_one": list(report.sqrt_minus_one),
        "classes": [
            {
                "numerators": list(c.numerators),
                "residues_mod_p": list(c.residues_mod_p),
                "sqrt_minus_one_residues": list(c.sqrt_minus_one_residues),
                "size": c.size,
            }
            for c in report.classes
        ],
        "exceptional": report.exceptional,
    }


def numerators_ascii(c: GlueClass) -> str:
    """A class's numerators as {a, ...}."""
    return "{" + ", ".join(str(a) for a in c.numerators) + "}"


def residues_ascii(c: GlueClass) -> str:
    """A class's residues mod p, aligned with its numerators."""
    return ", ".join(str(r) for r in c.residues_mod_p)


def sqrt_minus_one_ascii(c: GlueClass) -> str:
    """The square roots of minus one the class sits on, or a dash."""
    if not c.sqrt_minus_one_residues:
        return "-"
    return "{" + ", ".join(str(r) for r in c.sqrt_minus_one_residues) + "}"


# --------------------------------------------------------------------------
# the prime scan: checkpoint, resume, per-prime timing.
# --------------------------------------------------------------------------


def default_state_path(lo: int, hi: int, exp: int) -> Path:
    from .store import user_data_dir

    return user_data_dir() / "glue" / f"{lo}-{hi}-e{exp}.state.json"


def _fresh_state(lo: int, hi: int, exp: int) -> dict:
    return {
        "format": _STATE_FORMAT,
        "lo": lo,
        "hi": hi,
        "exp": exp,
        "cursor": 0,
        "rows": [],
        "timings": [],
        "elapsed": 0.0,
        "done": False,
    }


def _load_state(path: Path, lo: int, hi: int, exp: int) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        raw.get("format") != _STATE_FORMAT
        or raw.get("lo") != lo
        or raw.get("hi") != hi
        or raw.get("exp") != exp
    ):
        raise ValueError(
            f"the state file {path} was written for a different scan; "
            "resume with the same range and exponent or start fresh"
        )
    return raw


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


def run_scan(
    lo: int,
    hi: int,
    exp: int = 2,
    resume: bool = False,
    state_file: str | Path | None = None,
    checkpoint_seconds: float = 60.0,
) -> dict:
    """Scan the primes in [lo, hi] and return the report object (see the doc).

    The deterministic part of the report (rows and exceptional primes) is
    independent of timing; timings and wall_time_seconds are the only
    nondeterministic fields, so a resumed run matches an uninterrupted one
    byte for byte once timing is stripped.
    """
    lo, hi = int(lo), int(hi)
    exp = int(exp)
    if exp < 1:
        raise ValueError("the exponent e must be at least 1")
    primes = primes_in_range(lo, hi)
    path = Path(state_file) if state_file else default_state_path(lo, hi, exp)

    if resume and path.exists():
        state = _load_state(path, lo, hi, exp)
    else:
        state = _fresh_state(lo, hi, exp)

    start = time.monotonic()
    last_checkpoint = start

    if not state["done"]:
        for p in primes[state["cursor"] :]:
            t0 = time.monotonic()
            exc = is_exceptional(p, exp)
            dt = time.monotonic() - t0
            state["rows"].append([p, p % 4, exc])
            state["timings"].append([p, round(dt, 4)])
            state["cursor"] += 1
            now = time.monotonic()
            if now - last_checkpoint >= checkpoint_seconds:
                state["elapsed"] += now - start
                start = now
                last_checkpoint = now
                _write_state(path, state)
        state["done"] = True
        state["elapsed"] += time.monotonic() - start
        _write_state(path, state)

    rows = [
        {"p": r[0], "p_mod_4": r[1], "exceptional": bool(r[2])}
        for r in state["rows"]
    ]
    return {
        "mode": "scan",
        "lo": lo,
        "hi": hi,
        "exp": exp,
        "checked_range": checked_range_scan(lo, hi, exp),
        "rows": rows,
        "exceptional_primes": [r["p"] for r in rows if r["exceptional"]],
        "timings": [{"p": t[0], "seconds": t[1]} for t in state["timings"]],
        "wall_time_seconds": round(float(state["elapsed"]), 2),
    }


def scan_rows(report: dict) -> list[GlueScanRow]:
    """The scan rows as typed records, for callers that want the objects."""
    return [
        GlueScanRow(p=r["p"], p_mod_4=r["p_mod_4"], exceptional=r["exceptional"])
        for r in report["rows"]
    ]
