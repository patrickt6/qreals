r"""The reverse table at one modulus: numerators of d grouped by identical S.

For a fixed denominator d this maps the denominator dossier over every
numerator a coprime to d (1 <= a < d) and groups the numerators by identical
denominator polynomial S. Per group it records the cyclotomic index set T,
the factored S, the residues of the group's numerators modulo each
prime-power part of d (in a fixed order, ascending prime, so diffs are
stable), and the coprime splits d = d_+ d_- with their discrepancy class and
the group numerators that realize each split. Numerators that realize a
RATIO split are flagged.

The footer counts c(d), the number of numerators whose S is a product of
DISTINCT cyclotomic polynomials (class FULL or COLLAPSE), alongside the
REPEATED and non-cyclotomic counts. Numerators with a non-cyclotomic S are
not grouped (each has its own S); they are listed once and counted.

Everything reuses the denom dossier; this module computes no denominator of
its own.

JSON schema (the --json output of `qreals collapse`, keys are stable):
    d                int    the modulus
    prime_powers     [int]  the prime-power parts of d, ascending prime
    numerators_total int    the count of a coprime to d, 1 <= a < d
    groups           [ {numerators: [int], klass: str, T: [int],
                        multiplicities: {str(e): int}, S_factored: str,
                        residues: {str(pp): [int]}  (aligned to numerators),
                        ratio_numerators: [int],
                        splits: [ {d_plus, d_minus: int,
                                   discrepancy_class: str,
                                   discrepancy: str (factored),
                                   realized_by: [int]} ]} ]
                     one entry per distinct cyclotomic-product S, ordered by
                     smallest numerator
    c                int    count of numerators with squarefree cyclotomic S
    repeated_count   int    count with a repeated cyclotomic factor
    noncyclotomic_count       int
    noncyclotomic_numerators  [int]

The --range d1..d2 mode emits one line per d, "d c(d)", pipeable into
`qreals oeis` (take the second column); its JSON is {d1, d2, rows: [[d, c]]}.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import sympy as sp

from . import formatter
from .denom import (
    DenomDossier,
    Split,
    denom_dossier,
    s_factored_ascii,
    s_factored_tex,
    split_discrepancy_ascii,
)


@dataclass(frozen=True)
class GroupSplit:
    """One coprime split of d, shared by a whole group, with its realizers."""

    d_plus: int
    d_minus: int
    klass: str
    discrepancy: str
    realized_by: list[int]


@dataclass(frozen=True)
class Group:
    """The numerators of d sharing one cyclotomic-product denominator S."""

    numerators: list[int]
    dossier: DenomDossier  # of the smallest numerator in the group
    splits: list[GroupSplit]

    @property
    def klass(self) -> str:
        return self.dossier.klass

    @property
    def index_set(self) -> list[int]:
        return self.dossier.index_set

    @property
    def ratio_numerators(self) -> list[int]:
        flagged: set[int] = set()
        for s in self.splits:
            if s.klass == "RATIO":
                flagged.update(s.realized_by)
        return sorted(flagged)


@dataclass(frozen=True)
class CollapseTable:
    """The reverse table of one modulus d."""

    d: int
    prime_powers: list[int]
    numerators_total: int
    groups: list[Group]
    repeated_count: int
    noncyclotomic_numerators: list[int]

    @property
    def c(self) -> int:
        return sum(
            len(g.numerators)
            for g in self.groups
            if g.klass in ("FULL", "COLLAPSE")
        )

    def residues(self, group: Group) -> dict[int, list[int]]:
        """The group's numerator residues per prime-power part, aligned."""
        return {pp: [a % pp for a in group.numerators] for pp in self.prime_powers}


def prime_power_parts(d: int) -> list[int]:
    """The prime-power parts of d, in ascending prime order (fixed for diffs)."""
    return [int(p) ** int(e) for p, e in sorted(sp.factorint(d).items())]


def _on_locus(a: int, m: int) -> bool:
    if m == 1:
        return True
    r = a % m
    return r == 1 % m or r == (m - 1) % m


def _group_splits(numerators: list[int], splits: list[Split]) -> list[GroupSplit]:
    return [
        GroupSplit(
            d_plus=s.d_plus,
            d_minus=s.d_minus,
            klass=s.klass,
            discrepancy=split_discrepancy_ascii(s),
            realized_by=[
                a
                for a in numerators
                if _on_locus(a, s.d_plus) and _on_locus(a, s.d_minus)
            ],
        )
        for s in splits
    ]


def collapse_table(d: int) -> CollapseTable:
    """The reverse table of d, exact over Z[q], built on the denom dossier."""
    d = int(d)
    if d < 2:
        raise ValueError("the modulus d must be at least 2")
    by_s: dict[tuple, tuple[DenomDossier, list[int]]] = {}
    noncyc: list[int] = []
    repeated = 0
    total = 0
    for a in range(1, d):
        if gcd(a, d) != 1:
            continue
        total += 1
        p = denom_dossier(a, d)
        if not p.is_cyclotomic_product:
            noncyc.append(a)
            continue
        if p.klass == "REPEATED":
            repeated += 1
        key = tuple(p.S.all_coeffs())
        if key in by_s:
            by_s[key][1].append(a)
        else:
            by_s[key] = (p, [a])
    groups = []
    for dossier, numerators in sorted(by_s.values(), key=lambda t: t[1][0]):
        groups.append(
            Group(
                numerators=numerators,
                dossier=dossier,
                splits=_group_splits(numerators, dossier.splits),
            )
        )
    return CollapseTable(
        d=d,
        prime_powers=prime_power_parts(d),
        numerators_total=total,
        groups=groups,
        repeated_count=repeated,
        noncyclotomic_numerators=noncyc,
    )


def parse_range(text: str) -> tuple[int, int]:
    """Read a range d1..d2 with 2 <= d1 <= d2."""
    lo, sep, hi = text.partition("..")
    if not sep:
        raise ValueError("the range must be written d1..d2, e.g. 2..120")
    d1, d2 = int(lo), int(hi)
    if d1 < 2 or d2 < d1:
        raise ValueError("the range needs 2 <= d1 <= d2")
    return d1, d2


def range_row(d: int) -> tuple[int, int]:
    """One row of the c(d) sequence: (d, c(d))."""
    return d, collapse_table(d).c


# --------------------------------------------------------------------------
# Rendering, ASCII and TeX, both through the shared formatter.
# --------------------------------------------------------------------------


def numerators_ascii(group: Group) -> str:
    """The group's numerators, RATIO realizers flagged."""
    flagged = set(group.ratio_numerators)
    return ", ".join(
        f"{a} [RATIO]" if a in flagged else str(a) for a in group.numerators
    )


def index_set_ascii(group: Group) -> str:
    if group.dossier.is_cyclotomic_product:
        return "{" + ", ".join(str(e) for e in group.index_set) + "}"
    return "not a cyclotomic product"


def table_data(table: CollapseTable) -> dict:
    """The stable JSON object of the table (schema in the module docstring)."""
    t = table
    return {
        "d": t.d,
        "prime_powers": list(t.prime_powers),
        "numerators_total": t.numerators_total,
        "groups": [
            {
                "numerators": list(g.numerators),
                "klass": g.klass,
                "T": g.index_set,
                "multiplicities": {
                    str(e): m for e, m in g.dossier.multiplicities.items()
                },
                "S_factored": s_factored_ascii(g.dossier),
                "residues": {
                    str(pp): res for pp, res in t.residues(g).items()
                },
                "ratio_numerators": g.ratio_numerators,
                "splits": [
                    {
                        "d_plus": s.d_plus,
                        "d_minus": s.d_minus,
                        "discrepancy_class": s.klass,
                        "discrepancy": s.discrepancy,
                        "realized_by": list(s.realized_by),
                    }
                    for s in g.splits
                ],
            }
            for g in t.groups
        ],
        "c": t.c,
        "repeated_count": t.repeated_count,
        "noncyclotomic_count": len(t.noncyclotomic_numerators),
        "noncyclotomic_numerators": list(t.noncyclotomic_numerators),
    }


def table_tex(table: CollapseTable) -> str:
    """The TeX block of the table, ready to paste into notes.

    Compiles standalone inside a minimal article preamble (amsmath); the CI
    wraps it and runs pdflatex.
    """
    t = table
    lines = [r"\begin{align*}", rf"d &= {t.d} \\"]
    for g in t.groups:
        nums = ", ".join(
            (rf"{a}^{{*}}" if a in set(g.ratio_numerators) else str(a))
            for a in g.numerators
        )
        lines.append(
            rf"a \in \{{{nums}\}} &: \quad S = {s_factored_tex(g.dossier)}"
            + rf" \quad \text{{({g.klass})}} \\"
        )
    lines.append(
        rf"c({t.d}) &= {t.c}, \quad \text{{repeated}} = {t.repeated_count},"
        rf" \quad \text{{non-cyclotomic}} = {len(t.noncyclotomic_numerators)}"
    )
    lines.append(r"\end{align*}")
    if any(g.ratio_numerators for g in t.groups):
        lines.append(r"% * marks a numerator realizing a RATIO split")
    return "\n".join(lines)
