"""Tests for the q-arithmetic deficit and the negation panel.

The deficit reuses the verified pieces: q_add and q_mul for the engine value,
q_real_truncated for the target series, and, for rational inputs, the
bihomographic q_gosper engine and q_rational for the exact closed form. Each
check here recomputes a quantity a second way: the truncated-series deficit
against the Taylor expansion of the exact closed form, and the q=1 and q=0
invariants of the sum deficit (D(1)=0, D(0)=-1). See docs/CORRECTNESS.md.
"""

from __future__ import annotations

import json

import pytest
import sympy as sp

from qreals import app, deficit, negation_panel, q, q_add, q_mul, q_real_truncated

RATIONAL_SUM_PAIRS = [("3/2", "5/2"), ("3/2", "13/5"), ("7/3", "5/2"), ("4/3", "5/3")]
RATIONAL_MUL_PAIRS = [("3/2", "4/3"), ("3/2", "5/2"), ("7/3", "5/2")]


def _taylor(expr: sp.Expr, n: int) -> list[int]:
    """First n integer Taylor coefficients of a q-rational at q = 0."""
    ser = sp.series(expr, q, 0, n).removeO()
    return [int(ser.coeff(q, k)) for k in range(n)]


# --------------------------------------------------------------------------
# The headline case: the (3/2, 5/2) sum deficit is q^3 - 1.
# --------------------------------------------------------------------------
def test_sum_deficit_three_halves_five_halves_is_q_cubed_minus_one():
    d = deficit("3/2", "5/2", "+", 12)
    # head -1, 0, 0, 1, 0, 0  (the polynomial q^3 - 1)
    assert d.deficit[:6] == [-1, 0, 0, 1, 0, 0]
    assert sp.simplify(d.exact - (q**3 - 1)) == 0
    assert sp.sstr(d.exact) == "q**3 - 1"
    # the q=1 invariant: engine and target agree at q=1, so the deficit is 0
    assert d.deficit_at_q1 == 0
    # the q=0 invariant: the gap theorem forces constant term 2 vs 1, deficit -1
    assert d.deficit_at_q0 == -1


# --------------------------------------------------------------------------
# Series deficit equals the Taylor expansion of the exact closed form, both ops.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("x,y", RATIONAL_SUM_PAIRS)
def test_sum_deficit_series_matches_exact_and_invariants(x, y):
    n = 18
    d = deficit(x, y, "+", n)
    assert d.op == "+"
    assert d.deficit == _taylor(d.exact, n)
    # engine is the series sum, target is the q-series of the real sum
    assert d.engine == q_add(x, y, n)
    assert d.x_series == q_real_truncated(x, n)
    assert d.y_series == q_real_truncated(y, n)
    assert d.deficit == [t - e for t, e in zip(d.target, d.engine)]
    # invariants: D(1) = 0 for every pair, D(0) = -1 once x, y >= 1
    assert d.deficit_at_q1 == 0
    assert d.deficit_at_q0 == -1


@pytest.mark.parametrize("x,y", RATIONAL_MUL_PAIRS)
def test_product_deficit_series_matches_exact_and_q1_invariant(x, y):
    n = 18
    d = deficit(x, y, "*", n)
    assert d.op == "*"
    assert d.deficit == _taylor(d.exact, n)
    assert d.engine == q_mul(x, y, n)
    assert d.deficit == [t - e for t, e in zip(d.target, d.engine)]
    # the product engine and target also agree at q=1, so D(1) = 0
    assert d.deficit_at_q1 == 0
    # for x, y >= 1 the product of unit constant terms is 1, matching the target
    assert d.deficit_at_q0 == 0


def test_op_word_forms_match_the_symbols():
    assert deficit("3/2", "5/2", "add", 8).deficit == deficit("3/2", "5/2", "+", 8).deficit
    assert deficit("3/2", "4/3", "mul", 8).deficit == deficit("3/2", "4/3", "*", 8).deficit


# --------------------------------------------------------------------------
# Irrational inputs: a series deficit but no exact closed form.
# --------------------------------------------------------------------------
def test_irrational_inputs_give_a_series_deficit_without_a_closed_form():
    d = deficit("sqrt(2)", "sqrt(3)", "+", 10)
    assert d.exact is None
    assert d.deficit_at_q1 is None  # a truncated series is not summable at q=1
    assert d.deficit_at_q0 == d.deficit[0]
    assert len(d.deficit) == 10
    assert d.deficit == [t - e for t, e in zip(d.target, d.engine)]


def test_deficit_rejects_bad_op_and_nonpositive_n():
    with pytest.raises(ValueError):
        deficit("3/2", "5/2", "-", 8)
    with pytest.raises(ValueError):
        deficit("3/2", "5/2", "+", 0)


# --------------------------------------------------------------------------
# The negation panel: [x]_q + [-x]_q and its finiteness (Ovsienko Example 6.4).
# --------------------------------------------------------------------------
def test_negation_panel_finite_for_a_pure_square_root():
    panel = negation_panel("sqrt(2)", 16)
    assert panel.finite is True
    # [sqrt2]_q + [-sqrt2]_q = -q^{-2} + q, a short Laurent polynomial
    assert panel.valuation == -2
    assert panel.sum_coeffs[0] == -1 and panel.sum_coeffs[3] == 1


def test_negation_panel_infinite_for_the_golden_ratio():
    assert negation_panel("(1+sqrt(5))/2", 24).finite is False


def test_negation_panel_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        negation_panel("sqrt(2)", 0)


# --------------------------------------------------------------------------
# Headless app smoke: the deficit and negation-sum entries build and run.
# --------------------------------------------------------------------------
def test_app_deficit_entry_builds_and_runs_one_case():
    result = app.compute_deficit("3/2", "5/2", 8, "add")
    assert result["kind"] == "deficit"
    assert result["title"] and result["blocks"] and "data" in result
    assert result["data"]["deficit"][:4] == [-1, 0, 0, 1]
    assert result["data"]["exact"] == "q**3 - 1"
    assert result["data"]["deficit_at_q1"] == 0
    assert result["data"]["deficit_at_q0"] == -1
    # the q=1 and q=0 checks are shown on screen, in a kv block
    kv_labels = [
        label
        for block in result["blocks"]
        if block["kind"] == "kv"
        for (label, _value) in block["pairs"]
    ]
    assert "deficit at q = 1" in kv_labels
    assert "deficit at q = 0" in kv_labels


def test_app_deficit_headless_json(capsys):
    code = app.main(["deficit", "3/2", "5/2", "8", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deficit"][:4] == [-1, 0, 0, 1]
    assert payload["exact"] == "q**3 - 1"
    assert payload["deficit_at_q1"] == 0


def test_app_negsum_entry_builds_and_runs():
    result = app.compute_negsum("sqrt(2)", 12)
    assert result["kind"] == "negsum"
    assert result["data"]["finite"] is True
    assert result["title"] and result["blocks"] and "data" in result
