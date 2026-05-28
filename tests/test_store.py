"""The persistent saved list: location, round-trip, and removal.

The store keeps a personal list of computed q-numbers under the per-user data
directory. These tests pin the location away from the working folder, confirm an
add survives a fresh reload, and check removal and clearing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qreals.store import SavedEntry, SavedStore, user_data_dir


def _entry(x: str = "pi", n: int = 12) -> SavedEntry:
    return SavedEntry(input=x, n=n, coefficients=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0])


def test_user_data_dir_honours_the_env_override(tmp_path, monkeypatch):
    target = tmp_path / "explicit-data"
    monkeypatch.setenv("QREALS_DATA_DIR", str(target))
    assert user_data_dir() == target


def test_user_data_dir_is_outside_the_working_folder(monkeypatch):
    # With no override the location is a per-user path, never the current dir.
    monkeypatch.delenv("QREALS_DATA_DIR", raising=False)
    location = user_data_dir()
    assert location.is_absolute()
    assert Path.cwd() not in location.parents
    assert location != Path.cwd()


def test_entry_fills_in_timestamp_and_label():
    entry = SavedEntry(input="sqrt(2)", n=8, coefficients=[1, 0, 1])
    assert entry.label == "[sqrt(2)]_q"
    assert entry.timestamp  # ISO 8601, set on construction
    assert entry.timestamp.endswith("+00:00")


def test_reading_a_missing_store_creates_nothing(tmp_path):
    store = SavedStore(path=tmp_path / "nope" / "saved.json")
    assert store.all() == []
    assert not store.path.exists()
    assert not store.path.parent.exists()


def test_add_persist_reload_round_trip(tmp_path):
    path = tmp_path / "saved.json"
    SavedStore(path=path).add(_entry("pi", 12))
    SavedStore(path=path).add(_entry("sqrt(2)", 8))

    # A brand-new store object reads both back, with every field intact.
    reloaded = SavedStore(path=path).all()
    assert [e.input for e in reloaded] == ["pi", "sqrt(2)"]
    assert reloaded[0].n == 12
    assert reloaded[0].coefficients == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert reloaded[0].label == "[pi]_q"
    assert reloaded[0].timestamp


def test_remove_drops_one_entry(tmp_path):
    path = tmp_path / "saved.json"
    store = SavedStore(path=path)
    store.add(_entry("pi"))
    store.add(_entry("sqrt(2)", 8))
    removed = store.remove(0)
    assert removed.input == "pi"
    assert [e.input for e in SavedStore(path=path).all()] == ["sqrt(2)"]


def test_remove_out_of_range_raises(tmp_path):
    store = SavedStore(path=tmp_path / "saved.json")
    store.add(_entry())
    with pytest.raises(IndexError):
        store.remove(5)


def test_clear_empties_the_list(tmp_path):
    path = tmp_path / "saved.json"
    store = SavedStore(path=path)
    store.add(_entry("pi"))
    store.add(_entry("e"))
    assert store.clear() == 2
    assert SavedStore(path=path).all() == []


def test_store_writes_under_the_data_dir_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    monkeypatch.setenv("QREALS_DATA_DIR", str(data_dir))
    SavedStore().add(_entry())
    # the file lands in the data dir, and nothing is written to the cwd
    assert (data_dir / "saved.json").exists()
    assert os.listdir(tmp_path) == ["data"]
