"""Tolerant parsing of user-supplied real expressions.

``sympy.sympify`` rejects implicit multiplication, so ``3sqrt(2)``, ``2pi`` or
``3 sqrt(2)`` raise a SyntaxError even though their meaning is unambiguous. The
web math editor and ordinary users routinely write that form. ``parse_real``
accepts it by running sympy's implicit-multiplication (and caret) parser
transformations, while staying a strict superset of ``sympify`` for every input
that already worked: ``pi``, ``sqrt(2)``, ``(1+sqrt(5))/2``, ``22/7``, ``-3/2``
all parse to exactly what they did before.

Every place in the engine that turns a user string into a sympy expression goes
through this, so the whole tool (CLI and web) reads ``3sqrt(2)`` identically.
"""

from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# Standard sympy parsing plus: implicit multiplication (3sqrt(2) -> 3*sqrt(2),
# 2pi -> 2*pi) and ^ as exponentiation (sqrt(2)^2) rather than bitwise xor.
_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def parse_real(text: object) -> sp.Expr:
    """Parse a user real expression, accepting implicit multiplication.

    Args:
        text: a string such as ``"3sqrt(2)"``, ``"pi"``, ``"(1+sqrt(5))/2"``,
            ``"22/7"``; an already-built sympy expression is returned unchanged.

    Returns:
        The sympy expression.

    Raises:
        sympy.SympifyError: if the text cannot be parsed (so callers that catch
            SympifyError, as the plain ``sympify`` path did, keep working).
    """
    if isinstance(text, sp.Basic):
        return text
    try:
        return parse_expr(str(text).strip(), transformations=_TRANSFORMS, evaluate=True)
    except Exception as exc:  # noqa: BLE001 - normalise to the sympify error type
        raise sp.SympifyError(str(text)) from exc
