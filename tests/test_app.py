"""Smoke tests for the guided interface.

These never open a terminal. They build the menu, drive each capability's
`compute_*` headlessly, and confirm the bare command launches the menu and
exits cleanly when there is no input (questionary returns None without a TTY).
"""

from __future__ import annotations

import json

import pytest

from qreals import app


def test_menu_lists_every_capability_plus_help_and_quit():
    choices = app.build_menu_choices()
    # one entry per capability, then the saved list, Doctor, Help/About and Quit.
    assert choices[-2:] == ["Help / About", "Quit"]
    assert app._DOCTOR_LABEL in choices
    assert app._SAVED_LABEL in choices
    assert len(choices) == len(app.CAPABILITIES) + 4
    titles = {c.title for c in app.CAPABILITIES}
    assert titles.issubset(set(choices))


def test_registry_covers_the_public_api():
    keys = {c.key for c in app.CAPABILITIES}
    assert keys == {
        "rational",
        "jumpgap",
        "qint",
        "coeffs",
        "laurent",
        "prefix",
        "locked",
        "shift",
        "readouts",
        "arith",
        "deficit",
        "negate",
        "negsum",
        "radius",
        "oeis",
        "fingerprint",
    }


def test_compute_rational_matches_the_paper_example():
    result = app.compute_rational(3, 2)
    assert result["data"]["expr"] == "(q**2 + q + 1)/(q + 1)"
    assert result["data"]["at_q_eq_1"] == "3/2"


def test_compute_coeffs_pi():
    result = app.compute_coeffs("pi", 12)
    assert result["data"]["coefficients"] == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]


def test_every_capability_computes_headlessly():
    inputs = {
        "rational": {"p": 3, "s": 2},
        "jumpgap": {"p": 3, "s": 5},
        "qint": {"n": 5},
        "coeffs": {"x": "pi", "n": 8},
        "laurent": {"x": "pi", "order": 8},
        "prefix": {"x": "pi"},
        "locked": {"x": "pi", "n": 2},
        "shift": {"x": "pi", "order": 8, "direction": "up"},
        "readouts": {"x": "sqrt(2)", "n": 12},
        "arith": {"x": "3/2", "y": "13/5", "n": 8, "op": "add"},
        "deficit": {"x": "3/2", "y": "5/2", "n": 8, "op": "add"},
        "negate": {"x": "sqrt(2)", "n": 12},
        "negsum": {"x": "sqrt(2)", "n": 12},
        "radius": {"x": "pi", "n": 30},
        "fingerprint": {"x": "pi", "n_coeffs": 16},
    }
    for capability in app.CAPABILITIES:
        if capability.key == "oeis":
            continue  # needs the network; covered separately in test_oeis.py
        result = capability.compute(**inputs[capability.key])
        assert "title" in result and result["blocks"] and "data" in result


def test_render_result_plain_and_json(capsys):
    result = app.compute_coeffs("pi", 6)
    app.render_result(result, console=None)
    plain = capsys.readouterr().out
    assert "[pi]_q" in plain

    app.render_result(result, console=None, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["coefficients"][0] == 1


def test_headless_main_runs_a_subcommand(capsys):
    code = app.main(["rational", "3", "2"])
    assert code == 0
    assert "q**2 + q + 1" in capsys.readouterr().out


def test_headless_json_flag(capsys):
    code = app.main(["coeffs", "pi", "6", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["coefficients"] == [1, 1, 1, 0, 0, 0]


def test_headless_arith_add_json(capsys):
    code = app.main(["arith", "3/2", "13/5", "6", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["coefficients"][0] == 2  # constant term 1 + 1
    assert payload["verification"]["ok"] is True


def test_headless_negate_and_radius(capsys):
    assert app.main(["negate", "sqrt(2)", "12"]) == 0
    out = capsys.readouterr().out
    assert "[-sqrt(2)]_q" in out
    assert app.main(["radius", "pi", "40"]) == 0
    assert "radius" in capsys.readouterr().out.lower()


def test_bare_command_launches_menu_and_exits(monkeypatch, capsys):
    # Simulate a terminal, then make the menu's first question return None
    # (what a Ctrl-C / closed prompt yields). Launching the menu once and
    # exiting cleanly is the success signal.
    calls = {"select": 0}

    class FakeQuestion:
        def ask(self):
            calls["select"] += 1
            return None

    def fake_select(*args, **kwargs):
        return FakeQuestion()

    import questionary

    monkeypatch.setattr(app, "_is_interactive", lambda: True)
    monkeypatch.setattr(questionary, "select", fake_select)
    code = app.main([])
    assert code == 0
    assert calls["select"] == 1  # the main menu was shown once, then exited
    assert "q-deformed" in capsys.readouterr().out  # the banner printed


def test_bare_command_without_a_terminal_explains_itself(monkeypatch, capsys):
    monkeypatch.setattr(app, "_is_interactive", lambda: False)
    code = app.main([])
    assert code == 1
    assert "terminal" in capsys.readouterr().out.lower()


class _Answer:
    def __init__(self, value):
        self._value = value

    def ask(self):
        return self._value


class _ScriptedQuestionary:
    """Stand-in for questionary that hands back scripted answers in order."""

    def __init__(self, text_answers, select_answers):
        self._text = list(text_answers)
        self._select = list(select_answers)

    def text(self, *args, **kwargs):
        return _Answer(self._text.pop(0))

    def select(self, *args, **kwargs):
        return _Answer(self._select.pop(0))


def test_interactive_run_drives_a_text_only_capability(capsys):
    # coeffs prompts for x then N, then asks Next? -> Back to menu.
    qst = _ScriptedQuestionary(
        text_answers=["pi", "8"], select_answers=["Back to menu"]
    )
    app._run_capability(app.CAPABILITY_BY_KEY["coeffs"], qst, console=None)
    out = capsys.readouterr().out
    assert "[pi]_q" in out
    assert "1 + q + q^2" in out


def test_interactive_run_drives_a_select_then_text_capability(capsys):
    # shift asks a select (direction) first, then x and order.
    qst = _ScriptedQuestionary(
        text_answers=["pi", "8"],
        select_answers=["up: [x+1]_q = q*[x]_q + 1", "Back to menu"],
    )
    app._run_capability(app.CAPABILITY_BY_KEY["shift"], qst, console=None)
    out = capsys.readouterr().out
    assert "[pi + 1]_q" in out


def test_interactive_run_reports_a_math_error_and_stays(capsys):
    # downshift of a value in (0,1) raises; the loop should report and not crash.
    qst = _ScriptedQuestionary(
        text_answers=["pi-3", "8"],
        select_answers=["down: [x-1]_q = ([x]_q - 1)/q", "Back to menu"],
    )
    app._run_capability(app.CAPABILITY_BY_KEY["shift"], qst, console=None)
    assert "could not compute" in capsys.readouterr().out


def test_headless_fingerprint_json_is_fixed_length_and_deterministic(capsys):
    code = app.main(["fingerprint", "pi", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["names"] and len(payload["names"]) == len(payload["values"])
    assert payload["features"]["c_0"] == 1.0  # [pi]_q opens 1, 1, 1, ...
    # a second run must produce the identical vector
    app.main(["fingerprint", "pi", "--json"])
    again = json.loads(capsys.readouterr().out)
    assert again["values"] == payload["values"]


def test_compute_oeis_renders_a_hit_without_network(monkeypatch, capsys):
    from qreals import oeis

    fake = oeis.LookupResult(
        input_seq=[1, 1, 2, 5, 14, 42],
        hits=[
            oeis.Hit(
                anum="A000108",
                name="Catalan numbers",
                transform="identity",
                prefix_len=6,
                fully_verified=True,
                bfile_checked=True,
                bfile_len=200,
            )
        ],
        modp_hits={},
    )
    monkeypatch.setattr(oeis, "lookup", lambda *a, **k: fake)
    result = app.compute_oeis("1,1,2,5,14,42")
    assert result["kind"] == "oeis"
    app.render_result(result, console=None)
    out = capsys.readouterr().out
    assert "A000108" in out and "all 6 ok" in out


def test_compute_locked_rejects_too_large_index():
    with pytest.raises(ValueError):
        app.compute_locked("3/2", 50)


def test_compute_shift_down_requires_constant_term_one():
    # 0 < pi - 3 < 1 means [pi-3]_q has constant term 0, so a downshift is
    # undefined and should raise rather than return garbage.
    with pytest.raises(ValueError):
        app.compute_shift("pi-3", 8, "down")


# --------------------------------------------------------------------------
# Saved list and exports.
# --------------------------------------------------------------------------


def test_entry_from_result_keeps_input_order_and_coefficients():
    entry = app.entry_from_result(app.compute_coeffs("pi", 12))
    assert entry.input == "pi"
    assert entry.n == 12
    assert entry.coefficients == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert entry.label == "[pi]_q"


def test_entry_from_result_handles_laurent_shift_and_negate():
    laurent = app.entry_from_result(app.compute_laurent("pi", 8))
    assert laurent.n == 9 and len(laurent.coefficients) == 9  # order 8 -> q^0..q^8

    shifted = app.entry_from_result(app.compute_shift("pi", 8, "up"))
    assert shifted.label == "[pi + 1]_q"

    negated = app.entry_from_result(app.compute_negation("sqrt(2)", 10))
    assert negated.label == "[-sqrt(2)]_q"
    assert negated.coefficients  # the q-negation coefficients are kept


def test_entry_from_result_rejects_a_result_without_a_series():
    with pytest.raises(ValueError):
        app.entry_from_result(app.compute_rational(3, 2))


def test_batch_writes_one_file_per_run(tmp_path, capsys):
    out = tmp_path / "atlas.json"
    code = app.main(
        ["batch", "pi,sqrt(2),3/2", "--order", "8", "--format", "json", "-o", str(out)]
    )
    assert code == 0
    assert "wrote" in capsys.readouterr().out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [e["input"] for e in payload["entries"]] == ["pi", "sqrt(2)", "3/2"]
    assert all(len(e["coefficients"]) == 8 for e in payload["entries"])


def test_batch_to_stdout_writes_no_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = app.main(["batch", "pi", "--order", "6", "--format", "csv"])
    assert code == 0
    out = capsys.readouterr().out
    assert "pi" in out
    assert list(tmp_path.iterdir()) == []  # no file written without -o


def test_export_dumps_the_saved_list(tmp_path, capsys):
    from qreals.store import SavedEntry, SavedStore

    SavedStore().add(SavedEntry(input="pi", n=6, coefficients=[1, 1, 1, 0, 0, 0]))
    out = tmp_path / "saved.csv"
    code = app.main(["export", "--format", "csv", "-o", str(out)])
    assert code == 0
    body = out.read_text(encoding="utf-8").splitlines()
    assert body[0].startswith("label,input,n")
    assert any(line.startswith("[pi]_q,pi,6") for line in body[1:])


def test_export_of_an_empty_list_reports_and_fails(capsys):
    code = app.main(["export", "--format", "json"])
    assert code == 1
    assert "empty" in capsys.readouterr().err


def test_saved_remove_and_clear_headless(capsys):
    from qreals.store import SavedEntry, SavedStore

    store = SavedStore()
    store.add(SavedEntry(input="pi", n=6, coefficients=[1, 1, 1, 0, 0, 0]))
    store.add(SavedEntry(input="e", n=6, coefficients=[1, 1, 0, 0, 0, 0]))

    assert app.main(["saved", "--remove", "0"]) == 0
    assert "removed [pi]_q" in capsys.readouterr().out
    assert [e.input for e in SavedStore().all()] == ["e"]

    assert app.main(["saved", "--clear"]) == 0
    assert "cleared 1 item" in capsys.readouterr().out
    assert SavedStore().all() == []


def test_computing_a_result_writes_nothing_until_an_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from qreals.store import SavedStore

    result = app.compute_coeffs("pi", 12)
    app.render_result(result, console=None)
    # rendering a computation creates no file, and the saved store stays absent
    assert list(tmp_path.iterdir()) == []
    assert not SavedStore().path.exists()


def test_interactive_add_to_saved_then_view(capsys):
    from qreals.store import SavedStore

    result = app.compute_coeffs("pi", 12)
    app._add_result_to_saved(result, console=None)
    assert "saved [pi]_q" in capsys.readouterr().out
    assert len(SavedStore().all()) == 1

    # the saved-list view renders the kept value
    app.render_result(
        app._saved_list_result(SavedStore().all(), SavedStore().path), None
    )
    assert "[pi]_q" in capsys.readouterr().out
