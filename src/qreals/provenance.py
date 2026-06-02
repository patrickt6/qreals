"""Optional one-way bridge from a saved q-number to a qprov record.

If the separate ``qprov`` project is importable, a saved entry can be recorded
as an external run and carry the returned id, so the kept number links back to a
reproducible record. This is off by default and the core never imports qprov:
the import here is lazy, inside the function that records, and any failure
degrades to no id rather than raising. The dependency points one way only, from
this interface helper to qprov, never the reverse.
"""

from __future__ import annotations

import importlib.util

from .store import SavedEntry


def qprov_available() -> bool:
    """True when qprov can be imported in this environment."""
    return importlib.util.find_spec("qprov") is not None


def record_entry(entry: SavedEntry) -> str | None:
    """Record one saved entry in qprov and return its id, or None.

    Returns None, without touching qprov, when qprov is not installed or the
    record call fails, so callers can treat the id as best-effort.
    """
    try:
        import qprov
    except Exception:  # noqa: BLE001 - qprov is fully optional
        return None
    try:
        qprov_id: str | None = qprov.register_external(
            function_name="qreals.saved",
            inputs={"input": entry.input, "n": entry.n, "valuation": entry.valuation},
            outputs={"coefficients": entry.coefficients},
            tags={"tool": "qreals", "artifact": "saved-result"},
            notes=f"qreals saved value {entry.label}",
        )
        return qprov_id
    except Exception:  # noqa: BLE001 - degrade to no id
        return None


def annotate(entries: list[SavedEntry]) -> list[SavedEntry]:
    """Fill in a qprov id for any entry missing one, in place, and return them.

    Entries that already carry an id are left as they are. With qprov absent this
    is a no-op, so an export with provenance requested still succeeds, just
    without ids.
    """
    if not qprov_available():
        return entries
    for entry in entries:
        if not entry.qprov_id:
            entry.qprov_id = record_entry(entry)
    return entries
