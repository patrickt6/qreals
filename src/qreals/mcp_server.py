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

Failures leave as structured payloads rather than bare tracebacks: see
:class:`ToolFailure`. Alongside the tools the server publishes read-only
catalog resources (:data:`RESOURCES`) describing tool routing, the input
grammar, and the verdict vocabulary, so a client can see what the engine
covers without spending tool calls to find out.
"""

from __future__ import annotations

import functools
import importlib
import json
import sys
from collections.abc import Callable
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


# --------------------------------------------------------------------------- #
# The error contract.
#
# A failing tool returns metadata the calling model can act on, not a bare
# traceback and not a uniform "operation failed": those strip the information
# needed to choose between retrying, fixing the arguments, and giving up. The
# four categories carry different recovery decisions:
#
#   transient   the attempt failed but the request was sound (network, service
#               down); retrying can succeed.
#   validation  the arguments were malformed or outside the engine's domain;
#               retrying unchanged cannot succeed.
#   business    a rule of the mathematics forbids the request (for example an
#               irrational passed where only a rational has a finite R/S).
#   permission  the server lacks a capability or credential it needs, such as
#               an optional dependency that was never installed.
#
# build_server also raises these through the MCP layer so the protocol-level
# isError flag is set for the client, while the payload keeps the structure the
# model reads.
# --------------------------------------------------------------------------- #
ERROR_CATEGORIES = ("transient", "validation", "business", "permission")


class ToolFailure(Exception):
    """A tool failure carrying the metadata an agent needs to recover.

    Args:
        category: one of :data:`ERROR_CATEGORIES`.
        description: a human-readable account of what went wrong, specific
            enough that the model can tell the caller why.
        retryable: whether trying the identical call again could succeed.
            Defaults to True only for transient failures.
        remedy: what would have to change for the call to work.
    """

    def __init__(
        self,
        category: str,
        description: str,
        *,
        retryable: bool | None = None,
        remedy: str | None = None,
    ) -> None:
        super().__init__(description)
        if category not in ERROR_CATEGORIES:
            raise ValueError(f"unknown error category {category!r}")
        self.category = category
        self.description = description
        self.retryable = category == "transient" if retryable is None else retryable
        self.remedy = remedy

    def payload(self) -> dict[str, Any]:
        """The structured error result the tool returns."""
        out: dict[str, Any] = {
            "isError": True,
            "errorCategory": self.category,
            "isRetryable": self.retryable,
            "description": self.description,
        }
        if self.remedy:
            out["remedy"] = self.remedy
        return out


def _tool(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Wrap a tool so every failure leaves as a structured payload.

    Expected failures raise :class:`ToolFailure` and keep their category.
    Anything else is reported as a non-retryable validation failure carrying
    the real exception text: in this engine an unplanned exception is almost
    always an input the domain does not accept (an unparseable expression, a
    cubic irrational, a zero denominator), and the concrete message is what
    lets the model correct the call.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except ToolFailure as exc:
            return exc.payload()
        except Exception as exc:  # noqa: BLE001 - this is the error boundary
            return ToolFailure(
                "validation",
                f"{fn.__name__} could not run on these arguments: {exc}",
                remedy="check the input formats in this tool's description",
            ).payload()

    return wrapper


def _clamp(n: int, low: int) -> int:
    """Clamp a size argument to the [low, MAX_N] window."""
    return max(low, min(int(n), MAX_N))


def _nonzero(value: int, field: str) -> int:
    """Return an integer denominator, rejecting zero as a validation failure."""
    value = int(value)
    if value == 0:
        raise ToolFailure(
            "validation",
            f"{field} is zero, so the fraction is undefined",
            remedy=f"pass a nonzero {field}",
        )
    return value


def _jsonable(value: Any) -> Any:
    """Force a value to JSON-native types via a dumps round trip.

    Anything json does not know (sympy Integers and expressions, tuples
    inside lists) is serialized with ``str`` or coerced to lists, so every
    tool result is guaranteed transportable over the MCP wire.
    """
    return json.loads(json.dumps(value, default=str))


def _fraction(x: str) -> tuple[int, int]:
    """Parse a rational input string like '7/5' into (numerator, denominator).

    Rejects irrationals as a business failure rather than a malformed one:
    "sqrt(2)" is a perfectly well formed real, it simply has no finite R/S,
    so the model should be told to move to a different tool rather than to
    fix its syntax.
    """
    from .app import _parse_rational

    try:
        a, b = _parse_rational(x)
    except Exception as exc:  # noqa: BLE001 - classified below
        raise ToolFailure(
            "business",
            f"{x!r} is not a rational, and only rationals have a finite "
            f"numerator and denominator over Z[q] ({exc})",
            remedy="pass a rational such as '7/5'; for an irrational such as "
            "'sqrt(2)' use q_coefficients or negation_panel instead",
        ) from exc
    return a, _nonzero(b, "denominator")


# --------------------------------------------------------------------------- #
# Tool functions. Plain typed python functions; FastMCP builds the schemas
# from the signatures and these docstrings, so the docstrings are written
# for the calling model: they say exactly what the inputs mean.
# --------------------------------------------------------------------------- #
@_tool
def q_rational(a: int, b: int) -> dict[str, Any]:
    """Exact q-deformed rational [a/b]_q as a closed-form rational function in q.

    Use this when the question is "what is [a/b]_q", and you want the closed
    form R(q)/S(q) rather than a series. For the Taylor coefficients of the
    same object use q_coefficients; for the structure of the denominator S
    use s_properties or denom_dossier.

    Args:
        a: integer numerator, e.g. 3.
        b: nonzero integer denominator, e.g. 2.

    Returns p and s (the fraction), expr (the exact rational function in q,
    as a sympy string), and at_q_eq_1 (its value at q = 1, which is always
    the ordinary fraction a/b).
    """
    from .app import compute_rational

    return dict(_jsonable(compute_rational(int(a), _nonzero(b, "denominator"))["data"]))


@_tool
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


@_tool
def q_factor(x: str) -> dict[str, Any]:
    """Factor BOTH R(q) and S(q) of a rational [x]_q over Z[q].

    This is the numerator-and-denominator tool. Reach for it when the question
    is about R, about irreducibility, or about the factorisation of the whole
    fraction. If the question is only about the denominator S, s_properties
    answers it with the invariants already derived, and denom_dossier answers
    it in full; neither of those reports R at all.

    Args:
        x: a rational number as a string, e.g. "7/5" or "4/15". Irrationals
            are rejected: they have no finite R/S.

    Returns the factor lists of R and S with multiplicities, the cyclotomic
    support (which cyclotomic polynomials Phi(d) appear), the non-cyclotomic
    core, and whether R is irreducible or a pure cyclotomic product. This is
    the same record the CLI command ``qreals factor x --json`` prints.
    """
    from .app import compute_factor

    a, b = _fraction(x)
    return dict(_jsonable(compute_factor(a, b)["data"]))


@_tool
def s_properties(x: str) -> dict[str, Any]:
    """The compact invariant summary of the q-denominator S(q) of [a/d]_q.

    The quick denominator read: one fraction in, the derived invariants out.
    Use it to answer a specific question about S (its degree, its index set,
    whether it is squarefree or cyclotomic). Use denom_dossier instead when
    you want everything about the denominator including the collapse splits,
    and q_factor when you need the numerator R as well.

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


@_tool
def denom_dossier(a: int, d: int) -> dict[str, Any]:
    """The exhaustive denominator report for [a/d]_q, including collapse splits.

    The widest single-fraction denominator view, and the only tool that
    classifies the fraction (FULL, COLLAPSE, REPEATED, NONCYC) and enumerates
    the coprime splits behind a collapse. Prefer s_properties when a compact
    invariant summary is enough, and q_factor when the numerator R matters.
    Note this tool takes a and d as separate integers, not one "a/d" string.

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

    return dict(_jsonable(compute_denom(int(a), _nonzero(d, "denominator"))["data"]))


@_tool
def jump_gap(p: int, s: int) -> dict[str, Any]:
    """The two one-sided q-versions of p/s and the exact gap between them.

    The only tool concerned with the discontinuity of the q-deformation at a
    rational: it computes the limits from above and below and factors their
    difference. It is not a general factorisation tool; for the factorisation
    of the single object [p/s]_q use q_factor.

    Args:
        p: integer numerator.
        s: nonzero integer denominator.

    Returns the right and left q-versions [p/s]_q^+ and [p/s]_q^- (limits
    from above and below), their denominators S^+ and S^-, the exponent E,
    and the exact gap (1 - q) q^E / (S^+ S^-), all as sympy strings, plus
    the built-in consistency checks.
    """
    from .app import compute_jumpgap

    return dict(_jsonable(compute_jumpgap(int(p), _nonzero(s, "denominator"))["data"]))


def _fraction_jsonable(value: Any) -> Any:
    """A Fraction as an int (denominator 1) or a "p/q" string, else pass through."""
    from fractions import Fraction

    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return value


def _polynomial_jsonable(poly: dict[int, Any]) -> dict[str, Any]:
    """A {exponent: Fraction} Laurent dict as JSON-safe {"exponent": coeff}."""
    return {str(deg): _fraction_jsonable(coeff) for deg, coeff in sorted(poly.items())}


@_tool
def negation_panel(x: str, n: int = 48) -> dict[str, Any]:
    """The negation sum G(x) = [x]_q + [-x]_q with a certified finiteness verdict.

    Backed by the exact engine (`negation.negation_sum_exact`), not a
    truncation-order guess. For rational x the verdict is proven by the exact
    reversal rule [-x]_q := -q^-1 [x]_{1/q}: "finite" means the reduced
    denominator of the resulting exact rational function in q is a single
    monomial, so G(x) is a genuine finite Laurent polynomial, and "infinite"
    means it provably is not. For irrational quadratic x (a + b*sqrt(D)),
    successive Hirzebruch-Jung convergents are used to lock a growing window
    of leading coefficients; "finite" here means enough of the locked window
    (min_zero_run consecutive zero coefficients past a low-degree start) is an
    exact zero tail, "infinite" means an exact nonzero locked coefficient was
    found in that tail, and "insufficient_depth" means neither has happened
    yet within the requested depth (ask for more depth, not more trust).

    Args:
        x: a real number as a string, e.g. "sqrt(19)", "1+sqrt(2)", "1/3".
            Supported forms are a plain rational or a + b*sqrt(D) with a, b
            rational and D a squarefree integer > 1.
        n: requested Laurent depth; mapped to the exact engine's depth
            argument as max(n, 120) so shallow requests still lock enough
            coefficients to reach a real verdict (clamped to 8..512 first).

    Returns:
        x, depth (the depth actually used), verdict (one of "finite",
        "infinite", "finite_looking", "insufficient_depth"), certified (True
        only for "finite"/"infinite", both proofs, never observations),
        valuation, locked_depth (-1 for an exact finite rational verdict,
        meaning complete and depth-independent), first_nonzero_tail_index
        (set when infinite), polynomial (the {exponent: coefficient} Laurent
        map, JSON-safe, present when finite or finite_looking), and
        polynomial_string (a human-readable q-polynomial rendering of the
        same data). A "finite_looking" verdict is strong evidence over the
        locked window, not a proof; treat it accordingly.
    """
    from .negation import negation_sum_exact, polynomial_string

    n = _clamp(n, 8)
    depth = max(n, 120)
    result = negation_sum_exact(x, depth=depth)
    certified = result.verdict in ("finite", "infinite")
    out: dict[str, Any] = {
        "x": x,
        "depth": depth,
        "verdict": result.verdict,
        "certified": certified,
        "valuation": result.valuation,
        "locked_depth": result.locked_depth,
        "first_nonzero_tail_index": result.first_nonzero_tail_index,
    }
    if result.verdict in ("finite", "finite_looking"):
        out["polynomial"] = _polynomial_jsonable(result.polynomial)
        out["polynomial_string"] = polynomial_string(result.polynomial)
    else:
        out["polynomial"] = {}
        out["polynomial_string"] = None
    if result.verdict == "finite":
        out["note"] = (
            "finite is proven exactly by the reversal rule: the reduced "
            "denominator of the exact rational function in q is a monomial, "
            "so the polynomial above is complete and exact"
        )
    elif result.verdict == "infinite":
        out["note"] = (
            "infinite is proven exactly: a genuinely nonzero locked or exact "
            "coefficient was found at first_nonzero_tail_index, so no finite "
            "Laurent polynomial exists"
        )
    elif result.verdict == "finite_looking":
        out["note"] = (
            "finite_looking is strong evidence over the locked window, not a "
            "proof; a deeper request could still overturn it"
        )
    else:
        out["note"] = (
            "insufficient_depth: neither a finite nor an infinite verdict "
            "was reached within the requested depth; ask for more depth"
        )
    return dict(_jsonable(out))


@_tool
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


@_tool
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


@_tool
def oeis_lookup(x: str, n: int = 24) -> dict[str, Any]:
    """Look the q-series coefficients of [x]_q up in the OEIS. Needs network access.

    The only tool here that leaves the machine, so it is the only one that can
    fail for reasons unrelated to the mathematics. A completed search that
    matched nothing is a success with matched = false, and is reported as
    distinct from a search that could not be run at all: do not read an empty
    hit list as evidence that the sequence is absent from the OEIS unless
    matched is present and false.

    Args:
        x: a real number as a string, e.g. "4/15" or "sqrt(2)".
        n: how many Taylor coefficients to compute and query (clamped to
            1..512). Longer queries match more strictly.

    Returns the queried coefficients, matched (whether the completed search
    found anything), and hits, each with its A-number, name, match length,
    alignment offset, sign transform, and a b-file re-verification verdict
    (fully_verified, or diverged with the first disagreeing term).
    """
    from . import oeis_bulk
    from .oeis import OeisUnavailable

    n = _clamp(n, 1)
    try:
        results = oeis_bulk.sweep_oeis([x], n=n)
    except OeisUnavailable as exc:
        # A capability the server does not have, not a failed lookup: no
        # number of retries installs a missing package.
        raise ToolFailure(
            "permission",
            f"the OEIS lookup capability is not available: {exc}",
            retryable=False,
            remedy="install the extra with pip install qreals[oeis]",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - network and service faults
        raise ToolFailure(
            "transient",
            f"the OEIS lookup could not complete: {exc}",
            remedy="retry; if it keeps failing the OEIS service may be down",
        ) from exc

    res = results[0]
    if res.error:
        # The sweep returned, but this input's lookup did not succeed. That is
        # an access failure, and reporting it as zero hits would tell the model
        # the sequence is unknown to the OEIS when nothing was ever checked.
        raise ToolFailure(
            "transient",
            f"the OEIS lookup for {x!r} did not complete: {res.error}",
            remedy="retry the lookup; the coefficients themselves are unaffected",
        )
    return dict(
        _jsonable(
            {
                "input": res.input,
                "coeffs": res.coeffs,
                "matched": bool(res.hits),
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


# --------------------------------------------------------------------------- #
# Resources: read-only catalogs the client can load as context. They exist so a
# model does not have to spend tool calls discovering what the engine covers,
# which inputs parse, or what a verdict is allowed to license. They are static
# descriptions of the surface, never computed mathematics.
# --------------------------------------------------------------------------- #
def catalog_tools() -> str:
    """Which tool answers which question, and where the boundaries fall."""
    return json.dumps(
        {
            "closed form of [x]_q": "q_rational (rationals, as R/S)",
            "Taylor coefficients of [x]_q": "q_coefficients (any real)",
            "factorisation of numerator and denominator": "q_factor",
            "denominator invariants only, compact": "s_properties",
            "denominator in full, with collapse splits": "denom_dossier",
            "one-sided limits and the gap at a rational": "jump_gap",
            "is [x]_q + [-x]_q a finite Laurent polynomial": "negation_panel",
            "coefficientwise sum or product of two q-series": "q_add, q_mul",
            "does this coefficient sequence appear in the OEIS": "oeis_lookup",
            "overlaps_to_know": [
                "q_factor, s_properties and denom_dossier all take a rational "
                "and all describe S; only q_factor reports the numerator R, "
                "and only denom_dossier reports the class and collapse splits",
                "q_add and q_mul are coefficientwise operations on two "
                "q-series; the MGO map is not a ring homomorphism, so they "
                "are not the q-series of x + y or x * y",
            ],
        },
        indent=2,
    )


def catalog_inputs() -> str:
    """The input grammar, with the distinction that decides tool choice."""
    return json.dumps(
        {
            "real_inputs": {
                "accepted_by": ["q_coefficients", "q_add", "q_mul", "oeis_lookup"],
                "format": "a sympy-readable string",
                "examples": ["3/2", "sqrt(2)", "(1+sqrt(5))/2", "pi"],
            },
            "rational_inputs": {
                "accepted_by": ["q_factor", "s_properties"],
                "format": "a rational as a string",
                "examples": ["7/5", "4/15", "5/12"],
                "note": "irrationals are refused with errorCategory business, "
                "because they have no finite R/S at all",
            },
            "integer_pair_inputs": {
                "accepted_by": ["q_rational", "denom_dossier", "jump_gap"],
                "format": "two separate integer arguments, not one string",
            },
            "negation_inputs": {
                "accepted_by": ["negation_panel"],
                "format": "a rational, or a + b*sqrt(D) with a, b rational and "
                "D a squarefree integer > 1",
                "examples": ["1/3", "17/12", "sqrt(19)", "3+sqrt(2)"],
            },
            "size_arguments": {
                "clamped_to": MAX_N,
                "note": "an oversized n is silently clamped, never honoured",
            },
        },
        indent=2,
    )


def catalog_verdicts() -> str:
    """What each negation verdict licenses, and what it does not."""
    return json.dumps(
        {
            "finite": {
                "certified": True,
                "means": "proven exactly by the reversal rule; the returned "
                "polynomial is complete and depth-independent",
            },
            "infinite": {
                "certified": True,
                "means": "proven exactly; a nonzero locked or exact "
                "coefficient was found in the tail",
            },
            "finite_looking": {
                "certified": False,
                "means": "strong evidence over the locked window only; a "
                "deeper request could overturn it",
                "do_not": "report this as a proof or as 'finite'",
            },
            "insufficient_depth": {
                "certified": False,
                "means": "no verdict was reached at this depth",
                "do_not": "read this as evidence either way; ask for more depth",
            },
            "error_categories": {
                "transient": "retrying the identical call can succeed",
                "validation": "the arguments were malformed; fix them",
                "business": "the mathematics forbids the request; change tool",
                "permission": "a capability or credential is missing here",
            },
        },
        indent=2,
    )


# uri -> (name, description, loader)
RESOURCES: dict[str, tuple[str, str, Callable[[], str]]] = {
    "qreals://catalog/tools": (
        "tool routing catalog",
        "Which qreals tool answers which question, and how the overlapping "
        "rational-structure tools differ. Read before choosing among them.",
        catalog_tools,
    ),
    "qreals://catalog/inputs": (
        "input grammar catalog",
        "The accepted input forms per tool, with examples, and which tools "
        "take strings versus separate integers.",
        catalog_inputs,
    ),
    "qreals://catalog/verdicts": (
        "verdict and error vocabulary",
        "What each negation finiteness verdict licenses, which are certified "
        "proofs, and what each error category means for recovery.",
        catalog_verdicts,
    ),
}


def build_server() -> Any:
    """Build the FastMCP server with every tool and catalog resource registered.

    The mcp package is imported here, lazily, so the rest of the package
    never depends on it; a missing install raises ImportError with the
    exact pip command to run.

    Tools are registered behind a shim that re-raises a structured failure as
    an MCP tool error, so a failing call sets the protocol isError flag for
    the client while the payload keeps the errorCategory and isRetryable
    metadata the model reads.
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
        server.add_tool(_raising(fn))
    for uri, (name, description, loader) in RESOURCES.items():
        server.resource(
            uri, name=name, description=description, mime_type="application/json"
        )(loader)
    return server


def _raising(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Re-raise a structured failure payload as an MCP tool error.

    The payload is carried through as its JSON text, so the client sees
    isError set and the model still receives the full metadata rather than a
    bare sentence.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = fn(*args, **kwargs)
        if isinstance(result, dict) and result.get("isError"):
            raise RuntimeError(json.dumps(result))
        return result

    return wrapper


def main() -> int:
    """Entry point for ``qreals mcp``: run the stdio server until EOF."""
    try:
        server = build_server()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    server.run("stdio")
    return 0
