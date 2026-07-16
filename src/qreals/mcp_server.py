"""MCP server: the exact qreals engine as Model Context Protocol tools.

``qreals mcp`` runs a stdio MCP server named "qreals" that exposes the
computations an LLM most often needs as typed tools: exact q-rationals,
Taylor coefficients, factorisations, the denominator dossier, the jump gap,
the negation panel, series arithmetic, and a single-input OEIS lookup. Every
tool is a thin wrapper over an existing engine function, so the numbers a
model sees through MCP are exactly the numbers the CLI and the test suite
see; the model narrates, the engine computes.

The ``mcp`` package (the official python SDK, ``pip install "mcp[cli]"`` or
the ``qreals[mcp]`` extra) is imported lazily inside :func:`build_server`,
so importing this module, and everything else in the package, works without
it; only actually running the server needs it. The tool wrapper functions
below have no MCP dependency at all, which is also what the tests exercise.

Every tool returns JSON-native data (ints, strings, bools, lists, dicts);
sympy expressions are serialized through ``str``. Size arguments are clamped
to :data:`MAX_N` so a confused caller cannot request a million coefficients.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict
from typing import Any

SERVER_NAME = "qreals"

# The hard ceiling for every size argument (coefficient counts, Laurent
# orders). Large enough for real use, small enough to stay interactive.
MAX_N = 512

_INSTRUCTIONS = (
    "Exact q-deformed rationals and reals (Morier-Genoud and Ovsienko "
    "continued fractions). Every tool computes exactly over Z[q] or as an "
    "exact truncated series; nothing is estimated. Real-number inputs are "
    "sympy-readable strings such as '3/2', 'sqrt(2)', '(1+sqrt(5))/2', 'pi'."
)


def _clamp(n: int, low: int) -> int:
    """Clamp a size argument to the [low, MAX_N] window."""
    return max(low, min(int(n), MAX_N))


def _jsonable(value: Any) -> Any:
    """Force a value to JSON-native types via a dumps round trip.

    Anything json does not know (sympy Integers and expressions, tuples
    inside lists) is serialized with ``str`` or coerced to lists, so every
    tool result is guaranteed transportable over the MCP wire.
    """
    return json.loads(json.dumps(value, default=str))


def _fraction(x: str) -> tuple[int, int]:
    """Parse a rational input string like '7/5' into (numerator, denominator)."""
    from .app import _parse_rational

    return _parse_rational(x)


# --------------------------------------------------------------------------- #
# Tool functions. Plain typed python functions; FastMCP builds the schemas
# from the signatures and these docstrings, so the docstrings are written
# for the calling model: they say exactly what the inputs mean.
# --------------------------------------------------------------------------- #
def q_rational(a: int, b: int) -> dict[str, Any]:
    """Exact q-deformed rational [a/b]_q as a rational function in q.

    Args:
        a: integer numerator, e.g. 3.
        b: nonzero integer denominator, e.g. 2.

    Returns p and s (the fraction), expr (the exact rational function in q,
    as a sympy string), and at_q_eq_1 (its value at q = 1, which is always
    the ordinary fraction a/b).
    """
    from .app import compute_rational

    return dict(_jsonable(compute_rational(int(a), int(b))["data"]))


def q_coefficients(x: str, n: int = 16) -> dict[str, Any]:
    """First n Taylor coefficients of the q-real [x]_q.

    Args:
        x: a real number as a sympy-readable string, e.g. "3/2", "sqrt(2)",
            "(1+sqrt(5))/2", "pi".
        n: how many coefficients to return (clamped to 1..512).

    Returns x, n, and coefficients (exact integers c_0, c_1, ... with
    [x]_q = sum c_k q^k). The coefficients are stable: they do not change
    when n grows.
    """
    from . import q_real_truncated

    n = _clamp(n, 1)
    coeffs = q_real_truncated(x, n)
    return {"x": x, "n": n, "coefficients": [int(c) for c in coeffs]}


def q_factor(x: str) -> dict[str, Any]:
    """Factor the numerator R(q) and denominator S(q) of [x]_q over Z[q].

    Args:
        x: a rational number as a string, e.g. "7/5" (irrationals have no
            finite R/S and are rejected).

    Returns the factor lists of R and S with multiplicities, the cyclotomic
    support (which cyclotomic polynomials Phi(d) appear), the non-cyclotomic
    core, and whether R is irreducible or a pure cyclotomic product. This is
    the same record the CLI command ``qreals factor x --json`` prints.
    """
    from .app import compute_factor

    a, b = _fraction(x)
    return dict(_jsonable(compute_factor(a, b)["data"]))


def s_properties(x: str) -> dict[str, Any]:
    """Properties of the q-denominator S(q) of a rational [a/d]_q.

    Args:
        x: a rational number as a string, e.g. "5/12".

    Returns the exact structure record of S(q): its cyclotomic index set T,
    the saturation index e* = lcm(T) (None when S divides no [n]_q), deg S
    against the bound d - 1, the S(1) = d and S(0) = 1 invariants, and the
    squarefree / cyclotomic / full-q-integer flags. All fields are theorems
    of the Z[q] factorisation, not heuristics.
    """
    from .app import compute_sprops

    a, b = _fraction(x)
    return dict(_jsonable(compute_sprops(a, b)["data"]))


def denom_dossier(a: int, d: int) -> dict[str, Any]:
    """The one-shot denominator dossier of [a/d]_q.

    Args:
        a: integer numerator of the fraction.
        d: nonzero integer denominator of the fraction.

    Returns S(q) expanded and factored, the cyclotomic index set, the class
    (FULL, COLLAPSE, REPEATED, or NONCYC), deg S against d - 1, the residue
    a^2 mod d, and for a collapse every coprime split d = d_+ d_- with its
    exact discrepancy. This is the deepest single-fraction view the engine
    has.
    """
    from .app import compute_denom

    return dict(_jsonable(compute_denom(int(a), int(d))["data"]))


def jump_gap(p: int, s: int) -> dict[str, Any]:
    """The two q-versions of the rational p/s and the factored gap between them.

    Args:
        p: integer numerator.
        s: nonzero integer denominator.

    Returns the right and left q-versions [p/s]_q^+ and [p/s]_q^- (limits
    from above and below), their denominators S^+ and S^-, the exponent E,
    and the exact gap (1 - q) q^E / (S^+ S^-), all as sympy strings, plus
    the built-in consistency checks.
    """
    from .app import compute_jumpgap

    return dict(_jsonable(compute_jumpgap(int(p), int(s))["data"]))


def negation_panel(x: str, n: int = 48) -> dict[str, Any]:
    """The negation sum [x]_q + [-x]_q and a finiteness verdict (HEURISTIC).

    Args:
        x: a real number as a string, e.g. "sqrt(19)" or "1+sqrt(2)".
        n: the Laurent order of the computed window (clamped to 8..512).

    Returns the valuation and coefficients of [x]_q + [-x]_q through order n
    and the boolean ``finite``. IMPORTANT: the finite verdict is a
    truncation-order heuristic, not a proof. It only says the sum looks
    terminated within the order-n window (``verdict_kind`` names the order);
    a deeper window can overturn it, and sqrt(19) is a known example where
    shallow windows mislead. Treat ``finite`` as evidence, never as a
    theorem.
    """
    from .app import compute_negsum

    n = _clamp(n, 8)
    data = dict(_jsonable(compute_negsum(x, n)["data"]))
    data["verdict_kind"] = f"heuristic-order-{n}"
    data["verdict_note"] = (
        "finite is a truncation-order heuristic at the order above, not a proof"
    )
    return data


def q_add(x: str, y: str, n: int = 16) -> dict[str, Any]:
    """The series sum [x]_q + [y]_q (NOT [x+y]_q), first n coefficients.

    Args:
        x: a real number as a string, e.g. "sqrt(2)".
        y: a real number as a string, e.g. "3/2".
        n: how many coefficients to return (clamped to 1..512).

    Returns the exact coefficients of the coefficientwise sum of the two
    q-series. Note the MGO map is not a ring homomorphism: this is not the
    q-series of the real x + y.
    """
    from . import q_add as _q_add

    n = _clamp(n, 1)
    return {"x": x, "y": y, "n": n, "coefficients": [int(c) for c in _q_add(x, y, n)]}


def q_mul(x: str, y: str, n: int = 16) -> dict[str, Any]:
    """The series product [x]_q * [y]_q (NOT [x*y]_q), first n coefficients.

    Args:
        x: a real number as a string, e.g. "sqrt(2)".
        y: a real number as a string, e.g. "3/2".
        n: how many coefficients to return (clamped to 1..512).

    Returns the exact coefficients of the Cauchy product of the two
    q-series. Note the MGO map is not a ring homomorphism: this is not the
    q-series of the real x * y.
    """
    from . import q_mul as _q_mul

    n = _clamp(n, 1)
    return {"x": x, "y": y, "n": n, "coefficients": [int(c) for c in _q_mul(x, y, n)]}


def oeis_lookup(x: str, n: int = 24) -> dict[str, Any]:
    """Look the q-series coefficients of [x]_q up in the OEIS.

    Args:
        x: a real number as a string, e.g. "4/15" or "sqrt(2)".
        n: how many Taylor coefficients to compute and query (clamped to
            1..512).

    Returns the queried coefficients and the OEIS hits, each with its
    A-number, name, match length, alignment offset, sign transform, and a
    b-file re-verification verdict (fully_verified, or diverged with the
    first disagreeing term). Needs network access and the requests package;
    when that is missing the result is {"error": ...} instead.
    """
    from . import oeis_bulk
    from .oeis import OeisUnavailable

    n = _clamp(n, 1)
    try:
        results = oeis_bulk.sweep_oeis([x], n=n)
    except OeisUnavailable as exc:
        return {
            "error": f"{exc}; install the extra with pip install qreals[oeis]"
        }
    res = results[0]
    return dict(
        _jsonable(
            {
                "input": res.input,
                "coeffs": res.coeffs,
                "error": res.error,
                "hits": [asdict(h) for h in res.hits],
            }
        )
    )


# The registry FastMCP tools are built from; order is the menu order an
# MCP client shows.
TOOLS: tuple[Any, ...] = (
    q_rational,
    q_coefficients,
    q_factor,
    s_properties,
    denom_dossier,
    jump_gap,
    negation_panel,
    q_add,
    q_mul,
    oeis_lookup,
)


def build_server() -> Any:
    """Build the FastMCP server with every tool registered.

    The mcp package is imported here, lazily, so the rest of the package
    never depends on it; a missing install raises ImportError with the
    exact pip command to run.
    """
    try:
        fastmcp = importlib.import_module("mcp.server.fastmcp")
    except ImportError as exc:  # pragma: no cover - exercised via main()
        raise ImportError(
            "the MCP server needs the mcp package; install it with "
            'pip install "mcp[cli]" (or pip install "qreals[mcp]")'
        ) from exc
    server = fastmcp.FastMCP(SERVER_NAME, instructions=_INSTRUCTIONS)
    for fn in TOOLS:
        server.add_tool(fn)
    return server


def main() -> int:
    """Entry point for ``qreals mcp``: run the stdio server until EOF."""
    try:
        server = build_server()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    server.run("stdio")
    return 0
