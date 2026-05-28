"""Shared fixtures for the test suite.

The saved-list store reads ``QREALS_DATA_DIR`` before any per-user location, so
pointing it at a fresh temporary directory for every test keeps the suite off
the real per-user store and gives each test a clean slate.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_saved_store(tmp_path, monkeypatch):
    monkeypatch.setenv("QREALS_DATA_DIR", str(tmp_path / "qreals-data"))
