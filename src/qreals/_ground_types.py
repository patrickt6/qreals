"""Engage python-flint ground types in sympy when available.

sympy only uses FLINT for its polynomial arithmetic when the environment
variable SYMPY_GROUND_TYPES is set to "flint" before sympy is first imported;
it does not pick flint up automatically. Installing qreals[fast] is meant to
be enough on its own, so this module opts in when flint is importable and the
user has not already chosen ground types. qreals/__init__.py imports this
module before anything that imports sympy.
"""

from __future__ import annotations

import os
import sys

if "SYMPY_GROUND_TYPES" not in os.environ and "sympy" not in sys.modules:
    try:
        import flint  # noqa: F401
    except ImportError:
        pass
    else:
        os.environ["SYMPY_GROUND_TYPES"] = "flint"
