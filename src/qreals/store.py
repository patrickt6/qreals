"""A personal saved list of computed q-numbers that persists across sessions.

A professor computes [x]_q for the constants they care about and keeps them in
one place to revisit and export later. This module is the storage layer: it
appends, lists, and removes entries, and reads and writes one JSON file under
the operating system's per-user data directory, never the working folder.

Each entry records the input, the order N, the coefficients (the q^0.. Taylor
list of [x]_q), and a timestamp; an optional qprov id links it to a recorded
run (see ``provenance``). The location is resolved in this order:

1. an explicit ``path`` passed to ``SavedStore``;
2. the ``QREALS_DATA_DIR`` environment variable, if set;
3. ``platformdirs.user_data_dir("qreals")`` when platformdirs is importable;
4. a standard-library fallback to the same per-OS location.

The store stays in the core: sympy plus the standard library, with platformdirs
preferred but optional. Writing happens only when the user adds or removes an
entry; reading a missing file yields an empty list, so the first session starts
clean without creating anything.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STORE_FILENAME = "saved.json"
_ENV_DIR = "QREALS_DATA_DIR"


def _stdlib_user_data_dir() -> Path:
    """The per-user data directory by OS, using only the standard library.

    Matches what ``platformdirs.user_data_dir("qreals")`` returns on each
    platform, so the location is the same whether or not platformdirs is
    installed.
    """
    import sys

    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
        return Path(base) / "qreals"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "qreals"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "qreals"


def user_data_dir() -> Path:
    """The directory the saved list lives in, preferring platformdirs.

    The ``QREALS_DATA_DIR`` environment variable overrides everything, which
    keeps tests and one-off runs off the real per-user store. Nothing is created
    here; the directory is made only when an entry is first written.
    """
    override = os.environ.get(_ENV_DIR)
    if override:
        return Path(override)
    try:
        import platformdirs

        return Path(platformdirs.user_data_dir("qreals", appauthor=False))
    except ImportError:
        return _stdlib_user_data_dir()


@dataclass
class SavedEntry:
    """One kept q-number: the input, the order N, the coefficients, a timestamp.

    ``valuation`` is the power of the first coefficient (0 for the q^0.. series
    of [x]_q, used so a Laurent result can be kept faithfully). ``label`` is the
    human-readable name of the value (for example ``[pi]_q``). ``qprov_id`` is
    set only when the user asks for provenance and qprov is importable.
    """

    input: str
    n: int
    coefficients: list[int]
    timestamp: str = ""
    label: str = ""
    kind: str = "coeffs"
    valuation: int = 0
    qprov_id: str | None = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.label:
            self.label = f"[{self.input}]_q"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SavedEntry:
        fields = {
            "input",
            "n",
            "coefficients",
            "timestamp",
            "label",
            "kind",
            "valuation",
            "qprov_id",
        }
        return cls(**{k: v for k, v in raw.items() if k in fields})


@dataclass
class SavedStore:
    """Append-only-with-removal access to the saved list on disk.

    Construct with no arguments to use the per-user location, or pass an explicit
    ``path`` to a JSON file (tests pass a temporary one). Every method reads the
    file fresh, so two processes never work from a stale in-memory copy.
    """

    path: Path = field(default_factory=lambda: user_data_dir() / _STORE_FILENAME)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def all(self) -> list[SavedEntry]:
        """Every saved entry, oldest first. Empty when the file does not exist."""
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as handle:
            raw = json.load(handle)
        return [SavedEntry.from_dict(item) for item in raw.get("entries", [])]

    def add(self, entry: SavedEntry) -> SavedEntry:
        """Append an entry, creating the data directory and file if needed."""
        entries = self.all()
        entries.append(entry)
        self._write(entries)
        return entry

    def remove(self, index: int) -> SavedEntry:
        """Remove the entry at a zero-based index and return it."""
        entries = self.all()
        if not 0 <= index < len(entries):
            raise IndexError(
                f"no saved entry at position {index}; the list has {len(entries)}"
            )
        removed = entries.pop(index)
        self._write(entries)
        return removed

    def clear(self) -> int:
        """Remove every entry, returning how many were dropped."""
        count = len(self.all())
        self._write([])
        return count

    def _write(self, entries: list[SavedEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "entries": [e.to_dict() for e in entries]}
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
