r"""The brick lattice explorer data layer for the /lattice page.

For a denominator d, the divisor lattice has one node per divisor e >= 2 of
d, with an edge e1 -> e2 whenever e1 divides e2 and e2 / e1 is prime (the
covering relations of the divisibility order). For a numerator a coprime to
d, each node carries the state of its cyclotomic brick (of index e) in the
denominator S(q) of [a/d]_q:

    kept      the factor of index e divides S exactly once (solid node)
    dropped   it does not divide S (faded node)
    repeated  it divides S with multiplicity >= 2 (multiplicity badge)

When S carries a non-cyclotomic factor the per-node states are not the right
picture, so the payload carries a banner instead and the nodes render in a
neutral state.

Everything here is read off the denom module's dossier; this module computes
no polynomial arithmetic of its own. All TeX strings come from the shared
formatter emitters (the G0.13 rule). Output is deterministic: nodes, edges,
and panel rows are emitted in sorted order with no timestamps.

Scaling: the lattice view is built for d up to LATTICE_MAX_D. Above that the
payload degrades to a labelled divisor table (mode "table") with an explicit
notice, and no dossier is computed, so the page never hangs. Input d is
rejected above HARD_MAX_D so the divisor factorisation itself stays fast.
"""

from __future__ import annotations

from math import gcd
from typing import Any

import sympy as sp

from . import denom as denom_mod
from . import formatter
from .rational import q as _q

# The lattice view (and the per-numerator states) cover d up to here.
LATTICE_MAX_D = 10000
# Inputs above this are rejected outright (factorising d must stay instant).
HARD_MAX_D = 10**7
# Expanded S(q) is included in the panel only up to this degree; above it the
# panel says so explicitly and points at the CLI (no silent truncation).
PANEL_EXPANDED_MAX_DEG = 200
# Below this d the non-cyclotomic cofactor is factored into irreducibles for
# the panel; above it the cofactor stays unfactored (factoring a remainder of
# degree in the thousands takes minutes, far past the page's 2-second budget;
# the kept/dropped/repeated states and the class are identical either way).
FACTOR_COFACTOR_MAX_D = 1000

DEFAULT_D = 60
DEFAULT_A = 19


def coprime_residues(d: int) -> list[int]:
    """The residues 1 <= a < d coprime to d (a = 1 alone when d = 1)."""
    if d == 1:
        return [1]
    return [r for r in range(1, d) if gcd(r, d) == 1]


def _divisor_nodes(d: int) -> list[int]:
    """The lattice node set: divisors e >= 2 of d, ascending."""
    return [int(e) for e in sp.divisors(d) if int(e) >= 2]


def _omega(e: int) -> int:
    """The number of prime factors of e counted with multiplicity."""
    return int(sum(sp.factorint(e).values()))


def _edges(nodes: list[int]) -> list[list[int]]:
    """The covering edges e1 -> e2 (e1 | e2, e2/e1 prime), sorted."""
    node_set = set(nodes)
    out: list[list[int]] = []
    for e2 in nodes:
        for p in sp.primefactors(e2):
            e1 = e2 // int(p)
            if e1 in node_set:
                out.append([e1, e2])
    return sorted(out)


def _layout(nodes: list[int]) -> dict[int, dict[str, float]]:
    """Deterministic positions: rank = Omega(e) bottom-up, spread per rank."""
    ranks: dict[int, list[int]] = {}
    for e in nodes:
        ranks.setdefault(_omega(e), []).append(e)
    max_rank = max(ranks) if ranks else 1
    min_rank = min(ranks) if ranks else 1
    span = max(max_rank - min_rank, 1)
    pos: dict[int, dict[str, float]] = {}
    for rank, row in sorted(ranks.items()):
        row.sort()
        for i, e in enumerate(row):
            pos[e] = {
                "x": (i + 1) / (len(row) + 1),
                "y": 1.0 - (rank - min_rank) / span,
            }
    return pos


def _node_state(e: int, mult: dict[int, int]) -> tuple[str, int]:
    m = int(mult.get(e, 0))
    if m == 0:
        return "dropped", 0
    if m == 1:
        return "kept", 1
    return "repeated", m


def _cofactor_deg(fac: sp.Expr) -> int:
    return int(sp.degree(fac, gen=_q))


def _s_factored_tex(p: denom_mod.DenomDossier) -> str:
    """S factored for the panel, any oversized cofactor named not printed."""
    parts = [formatter.phi_tex(e, m) for e, m in sorted(p.multiplicities.items())]
    for fac, m in p.cofactor_factors:
        deg = _cofactor_deg(fac)
        if deg > PANEL_EXPANDED_MAX_DEG:
            parts.append(
                r"\text{(non-cyclotomic factor of degree %d, omitted here)}" % deg
            )
            continue
        inner = rf"\left({formatter.poly_tex(fac)}\right)"
        label = inner if m == 1 else inner + rf"^{{{m}}}"
        parts.append(label + r"\ \text{(non-cyclotomic)}")
    return r" \, ".join(parts) if parts else "1"


def _s_factored_ascii(p: denom_mod.DenomDossier) -> str:
    parts = [
        formatter.phi_label(e, m) for e, m in sorted(p.multiplicities.items())
    ]
    for fac, m in p.cofactor_factors:
        deg = _cofactor_deg(fac)
        if deg > PANEL_EXPANDED_MAX_DEG:
            parts.append(f"(non-cyclotomic factor of degree {deg}, omitted here)")
            continue
        inner = f"({formatter.poly_ascii(fac, wrap=10**9)})"
        label = inner if m == 1 else f"{inner}^{m}"
        parts.append(f"{label} [non-cyclotomic]")
    return " * ".join(parts) if parts else "1"


def _panel(dossier: denom_mod.DenomDossier) -> dict[str, Any]:
    """The side-panel dossier: TeX rows from the shared emitters plus the
    CLI ASCII rendering as the plain-text fallback."""
    p = dossier
    t_str = (
        "\\{" + ", ".join(str(k) for k in p.index_set) + "\\}"
        if p.is_cyclotomic_product
        else r"\text{not a cyclotomic product}"
    )
    rows: list[dict[str, str]] = [
        {"label": "fraction", "tex": formatter.qrat_tex(p.a, p.d)},
        {"label": "continued fraction", "tex": r"\text{%s}" % denom_mod.cf_str(p.cf)},
        {"label": "S(q) factored", "tex": _s_factored_tex(p)},
    ]
    if p.deg_S <= PANEL_EXPANDED_MAX_DEG:
        rows.append(
            {"label": "S(q) expanded", "tex": formatter.poly_tex(p.S.as_expr())}
        )
    else:
        rows.append(
            {
                "label": "S(q) expanded",
                "tex": r"\text{degree %d; expanded form omitted here, "
                r"available from the command line}" % p.deg_S,
            }
        )
    rows += [
        {"label": "index set T", "tex": t_str},
        {
            "label": "deg S (bound d - 1)",
            "tex": r"\deg S = %d, \quad d - 1 = %d" % (p.deg_S, p.d - 1),
        },
        {"label": "invariants", "tex": r"S(1) = %d, \quad S(0) = 1" % p.S_at_1},
        {"label": "class", "tex": r"\text{%s}" % p.klass},
        {
            "label": "residue",
            "tex": formatter.congruence_tex("a^2", str((p.a * p.a) % p.d), p.d),
        },
    ]
    if p.is_cyclotomic_product and not any(m > 1 for m in p.multiplicities.values()):
        if p.index_set:
            e_star = sp.ilcm(*p.index_set) if len(p.index_set) > 1 else p.index_set[0]
            rows.append(
                {
                    "label": "saturation index",
                    "tex": r"e^{\star} = \operatorname{lcm}(T) = %d" % int(e_star),
                }
            )
    splits = [
        {
            "label": "%d = %d * %d" % (p.d, s.d_plus, s.d_minus),
            "tex": formatter.fraction_tex(
                formatter.qint_tex(s.d_plus) + r" \, " + formatter.qint_tex(s.d_minus),
                "S",
            )
            + " = "
            + denom_mod.split_discrepancy_tex(s)
            + r" \quad \text{(%s%s)}"
            % (s.klass, ", realized" if s.realized else ""),
        }
        for s in p.splits
    ]
    return {"rows": rows, "splits": splits, "text": _panel_ascii(p)}


def _panel_ascii(p: denom_mod.DenomDossier) -> str:
    """The CLI ASCII rendering of the dossier (the G0.17 fallback text)."""
    t_str = (
        "{" + ", ".join(str(k) for k in p.index_set) + "}"
        if p.is_cyclotomic_product
        else "not a cyclotomic product"
    )
    lines = [
        f"denominator dossier of [{p.a}/{p.d}]_q",
        f"continued fraction: {denom_mod.cf_str(p.cf)}",
        f"S(q) factored: {_s_factored_ascii(p)}",
    ]
    if p.deg_S <= PANEL_EXPANDED_MAX_DEG:
        lines.append("S(q) = " + formatter.poly_ascii(p.S.as_expr()))
    else:
        lines.append(
            f"S(q) expanded omitted (degree {p.deg_S}); "
            "available from the command line"
        )
    lines += [
        f"index set T: {t_str}",
        f"deg S = {p.deg_S}  (bound d-1 = {p.d - 1})",
        f"S(1) = {p.S_at_1}  ({'ok' if p.S_at_1 == p.d else 'FAIL'})",
        f"class: {p.klass}",
        formatter.congruence_ascii("a^2", str((p.a * p.a) % p.d), p.d),
    ]
    for s in p.splits:
        lines.append(
            f"{p.d} = {s.d_plus} * {s.d_minus}: "
            f"[{s.d_plus}]_q [{s.d_minus}]_q / S = "
            f"{denom_mod.split_discrepancy_ascii(s)}  ({s.klass}"
            + (", realized)" if s.realized else ")")
        )
    return "\n".join(lines)


def lattice_data(d_text: str, a_text: str) -> dict[str, Any]:
    """The JSON payload of the /lattice page for inputs d and a.

    Returns {"error": message} on a bad input instead of raising, so the page
    can show it. The payload schema (stable):
        mode      "lattice" | "table"
        d, a      ints (a reduced into the coprime range; table mode omits a)
        residues  the coprime residues for the slider (lattice mode)
        nodes     [{e, tex, state, mult, x, y}]  (states neutral under banner)
        edges     [[e1, e2]] covering relations
        banner    str | null  the non-cyclotomic banner
        notice    str | null  the degrade notice (table mode)
        panel     {rows, splits, text}  the dossier side panel (lattice mode)
    """
    try:
        d = int(str(d_text).strip())
    except (TypeError, ValueError):
        return {"error": "d must be a whole number"}
    if d < 2:
        return {"error": "d must be at least 2"}
    if d > HARD_MAX_D:
        return {"error": f"d must be at most {HARD_MAX_D}"}

    nodes = _divisor_nodes(d)
    edges = _edges(nodes)
    pos = _layout(nodes)

    if d > LATTICE_MAX_D:
        return {
            "mode": "table",
            "d": d,
            "residues": [],
            "nodes": [
                {
                    "e": e,
                    "tex": formatter.phi_tex(e),
                    "state": "none",
                    "mult": 0,
                    "x": pos[e]["x"],
                    "y": pos[e]["y"],
                }
                for e in nodes
            ],
            "edges": edges,
            "banner": None,
            "notice": (
                f"d = {d} exceeds {LATTICE_MAX_D}, so the page shows the "
                "labelled divisor table instead of the drawn lattice; "
                f"per-numerator factor states need d <= {LATTICE_MAX_D}."
            ),
            "panel": None,
        }

    raw_a = "" if a_text is None else str(a_text).strip()
    if raw_a == "":
        a = DEFAULT_A if d == DEFAULT_D else 1
    else:
        try:
            a = int(raw_a)
        except (TypeError, ValueError):
            return {"error": "a must be a whole number"}
    residues = coprime_residues(d)
    if a < 1 or a >= d or gcd(a, d) != 1:
        return {
            "error": f"a must be a residue in 1..{d - 1} coprime to d = {d}"
        }

    dossier = denom_mod.denom_dossier(
        a, d, factor_cofactor=d <= FACTOR_COFACTOR_MAX_D
    )
    mult = dossier.multiplicities
    # Defensive: a cyclotomic index outside the divisors of d (the theory
    # says none exists) still gets a node rather than vanishing silently.
    extra = sorted(e for e in mult if e not in set(nodes))
    if extra:
        nodes = sorted(set(nodes) | set(extra))
        edges = _edges(nodes)
        pos = _layout(nodes)
    banner = None
    if not dossier.is_cyclotomic_product:
        banner = (
            "S(q) carries a non-cyclotomic factor for this a/d, so the "
            "kept/dropped picture over the divisor lattice does not apply; "
            "see the dossier panel for the factorisation."
        )

    out_nodes: list[dict[str, Any]] = []
    for e in nodes:
        if banner is None:
            state, m = _node_state(e, mult)
        else:
            state, m = "none", 0
        out_nodes.append(
            {
                "e": e,
                "tex": formatter.phi_tex(e),
                "state": state,
                "mult": m,
                "x": pos[e]["x"],
                "y": pos[e]["y"],
            }
        )

    return {
        "mode": "lattice",
        "d": d,
        "a": a,
        "residues": residues,
        "nodes": out_nodes,
        "edges": edges,
        "banner": banner,
        "notice": None,
        "panel": _panel(dossier),
    }
