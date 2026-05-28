"""The inline verification stamp: on by default, writing nothing.

These tests confirm the everyday trust signal. The stamp must run the cheap
cross-checks, print one line, and never create a file or need a TeX engine.
"""

from __future__ import annotations

import os

from qreals import Stamp, verify
from qreals import app


def test_verify_rational_passes_its_checks():
    stamp = verify({"kind": "rational", "data": {"p": 3, "s": 2}})
    assert isinstance(stamp, Stamp)
    assert stamp.ok
    line = stamp.line()
    assert line.startswith("verified:")
    assert "q=1 matches 3/2" in line
    assert "exact = truncated" in line


def test_verify_irrational_marks_q_at_one_na_not_pass():
    stamp = verify({"kind": "coeffs", "data": {"x": "pi", "n": 12}})
    assert stamp.ok  # the checks that ran passed
    line = stamp.line()
    assert "truncation stable" in line
    assert "shift law" in line
    # q=1 cannot run for an irrational; it must be reported n/a, not claimed.
    assert "n/a" in line
    statuses = {c.label: c.status for c in stamp.checks}
    assert any(s == "na" for s in statuses.values())


def test_unknown_kind_yields_an_empty_stamp():
    stamp = verify({"kind": "help", "data": {}})
    assert stamp.checks == []
    assert stamp.line() == ""


def test_render_prints_the_stamp_by_default_and_writes_no_file(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    result = app.compute_rational(3, 2)
    app.render_result(result, console=None)
    out = capsys.readouterr().out
    assert "verified:" in out
    assert os.listdir(tmp_path) == []  # the stamp wrote nothing


def test_render_can_turn_the_stamp_off():
    result = app.compute_coeffs("pi", 6)
    # rendering without verification still works and prints no stamp line.
    app.render_result(result, console=None, verify_result=False)


def test_headless_subcommand_prints_the_stamp_and_writes_no_file(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    code = app.main(["coeffs", "pi", "12"])
    assert code == 0
    assert "verified:" in capsys.readouterr().out
    assert os.listdir(tmp_path) == []


def test_json_output_folds_the_stamp_in_rather_than_printing_it(capsys):
    import json

    app.render_result(app.compute_rational(3, 2), console=None, as_json=True)
    payload = json.loads(capsys.readouterr().out)  # still valid JSON
    assert payload["verification"]["ok"] is True
    assert payload["verification"]["line"].startswith("verified:")
