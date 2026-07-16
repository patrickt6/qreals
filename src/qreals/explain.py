"""LLM narration of engine results: ``qreals explain`` and explain_result.

The division of labour is strict: the engine computes, the model narrates.
:func:`explain_result` sends one CLI result dict (the object a ``--json``
run prints) to the Claude API and returns a short referee-style prose
summary of what the computation shows. The system prompt forbids the model
from introducing any number not present in the input, so the narration can
never smuggle a value into the record; every number in the output is a
quote of the engine's own output.

The anthropic SDK is imported lazily through the :func:`_make_client` seam
(the ``qreals[ai]`` extra), so the package imports without it, and the API
key comes from the ``ANTHROPIC_API_KEY`` environment variable. Both missing
pieces surface as :class:`ExplainUnavailable` with a one-line actionable
message; the CLI turns that into a clean exit, never a traceback. The seam
is also what the tests monkeypatch, so the suite never touches the network.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "claude-opus-4-8"
MAX_TOKENS = 1024

# The narration contract. The no-new-numbers rule is the load-bearing one:
# it keeps the prose auditable against the input record alone.
SYSTEM_PROMPT = (
    "You summarize one computation record from qreals, an exact engine for "
    "q-deformed rationals and reals (Morier-Genoud and Ovsienko continued "
    "fractions). The user message contains the complete record as JSON. "
    "Write a short referee-style summary of what the computation shows. "
    "Hard rules: do not introduce any number that is not present in the "
    "input; quote values exactly as they appear. Write plain prose in an "
    "objective voice, no first or second person, no bullet points, no "
    "headings, no code blocks. Never use em dashes or en dashes; use "
    "commas, hyphens, or separate sentences instead. State what was "
    "computed, what the record establishes, and what it does not establish; "
    "label any heuristic field as heuristic. Keep it under 150 words."
)


class ExplainUnavailable(RuntimeError):
    """Raised when the anthropic SDK or the API key is missing."""


def _make_client() -> Any:
    """Build an Anthropic client; the tests monkeypatch this seam.

    The import lives here so the package never depends on anthropic at
    import time, matching the lazy-extra pattern used everywhere else.
    """
    try:
        anthropic = importlib.import_module("anthropic")
    except ImportError as exc:
        raise ExplainUnavailable(
            "qreals explain needs the anthropic package; install it with "
            'pip install anthropic (or pip install "qreals[ai]")'
        ) from exc
    return anthropic.Anthropic()


def explain_result(result: dict[str, Any], *, model: str | None = None) -> str:
    """A short prose summary of one CLI result dict, via the Claude API.

    Args:
        result: the object a ``--json`` run prints (any of the CLI's result
            payloads; the whole dict is sent verbatim as the record).
        model: optional Claude model id override; default DEFAULT_MODEL.

    Returns:
        The summary text. Raises :class:`ExplainUnavailable` when the
        anthropic package is not installed or ANTHROPIC_API_KEY is unset.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ExplainUnavailable(
            "ANTHROPIC_API_KEY is not set; export it to use qreals explain"
        )
    client = _make_client()
    payload = json.dumps(result, indent=2, sort_keys=True, default=str)
    response = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this qreals computation record:\n\n" + payload
                ),
            }
        ],
    )
    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", "") == "text"
    ]
    return "\n".join(parts).strip()


def run_cli(path: str | None, *, model: str | None = None) -> int:
    """The ``qreals explain`` command body: read JSON, print the summary.

    Args:
        path: a file holding one result JSON object; None or "-" reads
            stdin, so ``qreals rational 3 2 --json | qreals explain`` works.
        model: optional Claude model id override.

    Returns the process exit code: 0 on success, 1 when the LLM pieces are
    missing (one-line message, no traceback), 2 on unreadable input.
    """
    try:
        if path is None or path == "-":
            text = sys.stdin.read()
        else:
            text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"error: not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(result, dict):
        print("error: expected one JSON object (a --json result)", file=sys.stderr)
        return 2
    try:
        print(explain_result(result, model=model))
    except ExplainUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
