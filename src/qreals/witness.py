r"""The smallest-example generator over reduced fractions a/d.

Given a named property, this walks the reduced fractions a/d in a fixed order
(ascending d then a) and returns the smallest one with that property, rendered
as the one-shot denominator dossier of the tool that owns it, plus one line
stating the ordering under which it is minimal and the range certified below
it. Minimality is certified, not assumed: the scan visits every reduced
fraction at or before the witness in the order, and the visited count is part
of the result so a test can confirm the enumeration was exhaustive.

Registered properties (each a pure predicate on a/d, unit-tested on its own):
    ratio-cofactor    the denominator dossier has a coprime split whose
                      discrepancy is a proper ratio (class RATIO)
    no-collapse       S is the full q-integer of d (class FULL, no collapse)
    odd-multiplicity  a cyclotomic factor of S appears with an odd
                      multiplicity of at least 3
    sqrt-fail         a^2 = 1 (mod d) yet S is not a product of cyclotomic
                      polynomials (the sqrt-law boundary)
    glue              a/d sits in a sharing class of more than two numerators
                      at its modulus

The optional --where predicate is a small safe mini-language: simple
comparisons (==, !=, <, <=, >, >=) on the allowed field names (d, a, klass,
deg_S, a_sq_mod_d) and integer or string literals, joined by and / or. It is
compiled by a restricted walk over ast.parse that whitelists node types and
evaluated by walking that tree; Python eval and exec are never used on the
input, so an out-of-grammar or malicious predicate raises a clean error and
runs no code.

JSON schema (the --json output of ``qreals witness NAME``; keys are stable):
    property        str    the property name
    nth             int    which smallest was requested (1 = the smallest)
    where           str | null   the --where predicate, if any
    ordering        str    the order the search runs in
    search_cap      int    the largest denominator the search would cover
    found           bool   true when a witness was found in range
    scanned         int    reduced fractions visited at or before the witness
    matches_before  int    matches strictly before the witness (nth - 1)
    witness         null | {a: int, d: int}
    minimal_line    str    the one-line minimality statement
    uncovered       str | null   what was not covered on an early exit
    dossier         null | the tool-1 denominator dossier data of the witness
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import gcd
from typing import Callable

from . import denom as denom_mod

# The largest denominator the search covers before giving up (G0.18). The
# scan stops at the first match, so this only bounds the no-match case.
DEFAULT_D_CAP = 120

ORDERING = "ascending d then a"


# ---------------------------------------------------------------------------
# the property registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Property:
    """One registry entry: a name, a neutral description, and a predicate."""

    name: str
    description: str
    predicate: Callable[[int, int], bool]


def _fast_dossier(a: int, d: int):
    """The denominator dossier of a/d, skipping the costly cofactor factoring.

    Every field the predicates and the --where fields read (klass,
    multiplicities, splits, is_cyclotomic_product, deg_S) is identical with or
    without that step.
    """
    return denom_mod.denom_dossier(a, d, factor_cofactor=False)


def _pred_ratio_cofactor(a: int, d: int) -> bool:
    p = _fast_dossier(a, d)
    return any(s.klass == "RATIO" for s in p.splits)


def _pred_no_collapse(a: int, d: int) -> bool:
    return _fast_dossier(a, d).klass == "FULL"


def _pred_odd_multiplicity(a: int, d: int) -> bool:
    p = _fast_dossier(a, d)
    return any(m >= 3 and m % 2 == 1 for m in p.multiplicities.values())


def _pred_sqrt_fail(a: int, d: int) -> bool:
    p = _fast_dossier(a, d)
    return (a * a) % d == 1 % d and not p.is_cyclotomic_product


def _pred_glue(a: int, d: int) -> bool:
    from .twin import twin_class

    return len(twin_class(a, d).members) > 2


PROPERTIES: dict[str, Property] = {
    p.name: p
    for p in [
        Property(
            "ratio-cofactor",
            "the denominator dossier has a coprime split whose discrepancy "
            "[d+]_q [d-]_q / S is a proper ratio (discrepancy class RATIO)",
            _pred_ratio_cofactor,
        ),
        Property(
            "no-collapse",
            "S is the full q-integer of d with no collapse (class FULL, the "
            "a == +/-1 (mod d) tails)",
            _pred_no_collapse,
        ),
        Property(
            "odd-multiplicity",
            "a cyclotomic factor of S appears with an odd multiplicity of at "
            "least 3",
            _pred_odd_multiplicity,
        ),
        Property(
            "sqrt-fail",
            "a^2 = 1 (mod d) yet S is not a product of cyclotomic polynomials, "
            "the boundary of the sqrt-law",
            _pred_sqrt_fail,
        ),
        Property(
            "glue",
            "a/d lies in a sharing class of more than two numerators at its "
            "modulus",
            _pred_glue,
        ),
    ]
}


def registry_data() -> dict:
    """The stable JSON object of ``qreals witness list``."""
    return {
        "properties": [
            {"name": p.name, "description": p.description}
            for p in PROPERTIES.values()
        ]
    }


def registry_lines() -> list[str]:
    """The human listing of the registry, one block per property."""
    lines: list[str] = []
    for p in PROPERTIES.values():
        lines.append(p.name)
        lines.append(f"  {p.description}")
    return lines


# ---------------------------------------------------------------------------
# the --where mini-language: a restricted, whitelisted AST, never eval/exec
# ---------------------------------------------------------------------------


class WhereError(ValueError):
    """A --where predicate outside the allowed grammar."""


_ALLOWED_FIELDS = {"d", "a", "klass", "deg_S", "a_sq_mod_d"}
_ALLOWED_CMP = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def compile_where(text: str) -> ast.Expression:
    """Parse and validate a --where predicate, raising WhereError if invalid.

    The grammar is only simple comparisons joined by and / or on the allowed
    field names and integer or string literals. Anything else (calls,
    attribute access, arithmetic, chained comparisons, multiple statements,
    unknown names) is rejected before any evaluation, and the input is never
    handed to Python eval or exec.
    """
    text = (text or "").strip()
    if not text:
        raise WhereError("the --where predicate is empty")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise WhereError(f"could not parse the --where predicate: {text!r}")
    _validate_node(tree.body)
    return tree


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise WhereError("only 'and' and 'or' may join comparisons")
        for value in node.values:
            _validate_node(value)
        return
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise WhereError("chained comparisons are not allowed")
        if not isinstance(node.ops[0], _ALLOWED_CMP):
            raise WhereError("only ==, !=, <, <=, >, >= comparisons are allowed")
        _validate_operand(node.left)
        _validate_operand(node.comparators[0])
        return
    raise WhereError(
        "only comparisons on d, a, klass, deg_S, a_sq_mod_d joined by and / or "
        "are allowed"
    )


def _validate_operand(node: ast.AST) -> None:
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_FIELDS:
            raise WhereError(
                f"unknown field {node.id!r}; allowed: "
                + ", ".join(sorted(_ALLOWED_FIELDS))
            )
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, str)):
        if isinstance(node.value, bool):
            raise WhereError("only integer and string literals are allowed")
        return
    raise WhereError("operands must be an allowed field or an int / string literal")


def eval_where(tree: ast.AST, env: dict) -> bool:
    """Evaluate a compiled --where predicate against a field environment.

    Walks the whitelisted tree directly; Python eval and exec are never used.
    """
    if isinstance(tree, ast.Expression):
        return eval_where(tree.body, env)
    if isinstance(tree, ast.BoolOp):
        results = [eval_where(v, env) for v in tree.values]
        return all(results) if isinstance(tree.op, ast.And) else any(results)
    if isinstance(tree, ast.Compare):
        left = _operand_value(tree.left, env)
        right = _operand_value(tree.comparators[0], env)
        return _apply_compare(tree.ops[0], left, right)
    raise WhereError("malformed --where predicate")


def _operand_value(node: ast.AST, env: dict):
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise WhereError("malformed operand in --where predicate")


def _apply_compare(op: ast.AST, left, right) -> bool:
    try:
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
    except TypeError:
        raise WhereError("cannot order a string against an integer")
    raise WhereError("unsupported comparison in --where predicate")


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WitnessResult:
    """The outcome of one smallest-example search."""

    property: str
    nth: int
    where: str | None
    search_cap: int
    found: bool
    a: int | None
    d: int | None
    dossier: object | None
    scanned: int
    matches_before: int


def find_witness(
    property: str,
    nth: int = 1,
    where: str | None = None,
    d_cap: int = DEFAULT_D_CAP,
) -> WitnessResult:
    """Find the nth smallest reduced fraction a/d with the named property.

    The search runs in the ascending d then a order, filtered by the optional
    --where predicate, and stops at the nth match. The visited count
    (``scanned``) covers every reduced fraction at or before the witness, so
    minimality is certified rather than assumed.
    """
    if property not in PROPERTIES:
        known = ", ".join(PROPERTIES)
        raise ValueError(f"unknown property '{property}'; registered: {known}")
    nth = int(nth)
    if nth < 1:
        raise ValueError("--nth must be at least 1")
    prop = PROPERTIES[property]
    tree = compile_where(where) if where else None

    scanned = 0
    matches = 0
    for d in range(2, int(d_cap) + 1):
        for a in range(1, d):
            if gcd(a, d) != 1:
                continue
            scanned += 1
            if tree is not None:
                p = _fast_dossier(a, d)
                env = {
                    "d": d,
                    "a": a,
                    "klass": p.klass,
                    "deg_S": p.deg_S,
                    "a_sq_mod_d": (a * a) % d,
                }
                if not eval_where(tree, env):
                    continue
            if prop.predicate(a, d):
                matches += 1
                if matches == nth:
                    return WitnessResult(
                        property=property,
                        nth=nth,
                        where=where,
                        search_cap=int(d_cap),
                        found=True,
                        a=a,
                        d=d,
                        dossier=denom_mod.denom_dossier(a, d),
                        scanned=scanned,
                        matches_before=matches - 1,
                    )
    return WitnessResult(
        property=property,
        nth=nth,
        where=where,
        search_cap=int(d_cap),
        found=False,
        a=None,
        d=None,
        dossier=None,
        scanned=scanned,
        matches_before=matches,
    )


# ---------------------------------------------------------------------------
# rendering, shared by the human and JSON outputs
# ---------------------------------------------------------------------------


def minimal_line(res: WitnessResult) -> str:
    """The one line stating the ordering and the range certified below."""
    if not res.found:
        return uncovered_line(res)
    where = f", among fractions with {res.where}," if res.where else ""
    if res.nth == 1:
        return (
            f"minimal under {ORDERING}{where} no smaller example exists for "
            f"d <= {res.d}; every reduced fraction before {res.a}/{res.d} in "
            f"this order was checked ({res.scanned - 1} of them)"
        )
    return (
        f"the {res.nth}-th smallest under {ORDERING}{where} {res.matches_before} "
        f"smaller examples exist, all with d <= {res.d}; every reduced fraction "
        f"before {res.a}/{res.d} in this order was checked ({res.scanned - 1} of them)"
    )


def uncovered_line(res: WitnessResult) -> str:
    """What was not covered when no witness was found in range (G0.18)."""
    where = f" with {res.where}" if res.where else ""
    return (
        f"no reduced fraction{where} with d <= {res.search_cap} has the "
        f"'{res.property}' property; fractions with d > {res.search_cap} were "
        "not covered"
    )


def witness_data(res: WitnessResult) -> dict:
    """The stable JSON object of a witness search (schema in the module doc)."""
    return {
        "property": res.property,
        "nth": res.nth,
        "where": res.where,
        "ordering": ORDERING,
        "search_cap": res.search_cap,
        "found": res.found,
        "scanned": res.scanned,
        "matches_before": res.matches_before,
        "witness": ({"a": res.a, "d": res.d} if res.found else None),
        "minimal_line": minimal_line(res),
        "uncovered": (None if res.found else uncovered_line(res)),
        "dossier": (denom_mod.dossier_data(res.dossier) if res.found else None),
    }
