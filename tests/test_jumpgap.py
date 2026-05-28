"""Tests for the one-sided jump gap [p/s]_q^+ - [p/s]_q^-.

The worked values are pinned to jump_findings.md (computations/q_jump in the
research codebase, qprov record a2e94ea7). Comparisons use sympy equality, not
string form, so an equal-but-differently-written factorisation still passes.
"""

from __future__ import annotations

import json

import sympy as sp

from qreals import JumpGap, jumpgap, q, q_int, q_rational
from qreals import app


def _unit_fraction_form(s: int) -> sp.Expr:
    """The closed form (1 - q) q^{s-1} / ([s]_q (1 + q^2 [s-1]_q)) (jump_findings)."""
    return sp.cancel((1 - q) * q ** (s - 1) / (q_int(s) * (1 + q**2 * q_int(s - 1))))


def test_jumpgap_1_5_matches_jump_findings_and_unit_form():
    gap = jumpgap(1, 5)
    assert isinstance(gap, JumpGap)
    worked = sp.sympify(
        "-q**4*(q - 1)/((q**4 + q**3 + q**2 + q + 1)*(q**5 + q**4 + q**3 + q**2 + 1))"
    )
    assert sp.simplify(gap.gap - worked) == 0
    assert gap.exponent == -5
    assert gap.cf == (0, 5)
    # the right version is the qreals q_rational oracle value
    assert sp.simplify(gap.right - q_rational(1, 5)) == 0
    # the unit-fraction closed form, recomputed from q and s only
    assert sp.simplify(gap.gap - _unit_fraction_form(5)) == 0


def test_jumpgap_3_5_matches_jump_findings():
    gap = jumpgap(3, 5)
    worked = sp.sympify(
        "-q**3*(q - 1)/((q**3 + q**2 + 2*q + 1)*(q**4 + q**3 + q**2 + q + 1))"
    )
    assert sp.simplify(gap.gap - worked) == 0
    assert gap.exponent == -2
    assert gap.cf == (0, 1, 1, 2)
    assert sp.simplify(gap.right - q_rational(3, 5)) == 0


def test_gap_equals_its_closed_form_factors():
    # gap = (1 - q) q^E / (S^+ S^-) for the right and left q-denominators.
    for p, s in [(1, 5), (3, 5), (2, 5), (5, 7), (1, 2), (1, 3), (2, 1), (3, 1)]:
        gap = jumpgap(p, s)
        target = (1 - q) * q**gap.exponent / (gap.s_plus * gap.s_minus)
        assert sp.simplify(gap.gap - target) == 0
        assert gap.closed_form_holds()


def test_oracle_checks_hold_on_a_sample():
    # the right version matches q_rational and both denominators equal s at q=1.
    for p, s in [(1, 5), (3, 5), (2, 5), (5, 7), (2, 1), (3, 1)]:
        gap = jumpgap(p, s)
        checks = gap.checks()
        assert checks["right_matches_oracle"]
        assert checks["s_plus_at_one_is_s"]
        assert checks["s_minus_at_one_is_s"]
        assert gap.denominators_at_one() == (s, s)


def test_left_version_differs_from_right_and_makes_the_gap():
    gap = jumpgap(3, 5)
    assert sp.simplify(gap.right - gap.left) != 0
    assert sp.simplify(gap.gap - (gap.right - gap.left)) == 0


# --------------------------------------------------------------------------
# The capability in the guided interface.
# --------------------------------------------------------------------------


def test_compute_jumpgap_result_shape():
    result = app.compute_jumpgap(3, 5)
    assert result["kind"] == "jumpgap"
    data = result["data"]
    assert data["p"] == 3 and data["s"] == 5
    assert data["exponent"] == -2
    assert data["cf"] == [0, 1, 1, 2]
    assert data["checks"]["right_matches_oracle"]
    assert data["checks"]["s_plus_at_one_is_s"]
    assert data["checks"]["s_minus_at_one_is_s"]


def test_headless_jumpgap_prints_versions_and_stamp(capsys):
    code = app.main(["jumpgap", "3", "5"])
    assert code == 0
    out = capsys.readouterr().out
    assert "[3/5]_q^+" in out
    assert "[3/5]_q^-" in out
    # the verification stamp prints under the result
    assert "matches q_rational(3, 5)" in out


def test_headless_jumpgap_json_carries_a_passing_stamp(capsys):
    code = app.main(["jumpgap", "1", "5", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exponent"] == -5
    assert payload["checks"]["right_matches_oracle"]
    assert payload["verification"]["ok"] is True


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


def test_interactive_jumpgap_menu_run_headlessly(capsys):
    # the menu asks for one rational p/s, then Next? -> Back to menu.
    qst = _ScriptedQuestionary(text_answers=["3/5"], select_answers=["Back to menu"])
    app._run_capability(app.CAPABILITY_BY_KEY["jumpgap"], qst, console=None)
    out = capsys.readouterr().out
    assert "[3/5]_q^+" in out
    assert "[3/5]_q^-" in out
    assert "gap" in out


def test_jumpgap_prompt_rejects_an_irrational():
    assert app._validate_rational("pi") != True  # noqa: E712 - validator returns a str
    assert app._validate_rational("3/5") is True
