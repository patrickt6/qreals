"""CLI smoke tests: every documented command runs and prints the documented result.

Commands run as `python -m qreals ...` in a subprocess, the same entry point
the `qreals` script calls. Output is compared as parsed JSON where the
command offers --json, and on stable lines otherwise (rich tables wrap with
the terminal width, and `conj` prints a wall time).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from qreals import app as app_mod

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, COLUMNS="100", NO_COLOR="1", TERM="dumb")
    proc = subprocess.run(
        [sys.executable, "-m", "qreals", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if check:
        assert proc.returncode == 0, proc.stderr
    return proc


def run_json(*args: str) -> object:
    return json.loads(run(*args, "--json").stdout)


# --------------------------------------------------------------------------
# the --help worked examples: output must equal the epilog, byte for byte
# --------------------------------------------------------------------------

EPILOG_TOOLS = ["denom", "why", "twin", "bricks", "conj", "glue", "witness", "collapse", "check"]


def epilog_command_and_payload(tool: str) -> tuple[list[str], str]:
    text = getattr(app_mod, f"_{tool}_help_epilog")()
    lines = text.splitlines()
    assert lines[0] == "worked example:" and lines[1] == ""
    cmd = lines[2].strip()
    assert cmd.startswith("$ qreals ")
    payload = textwrap.dedent("\n".join(lines[3:])) + "\n"
    return cmd.split()[2:], payload


@pytest.mark.parametrize("tool", EPILOG_TOOLS)
def test_help_worked_example_matches_output(tool: str) -> None:
    """The worked example printed by `qreals <tool> --help` is the literal
    stdout of the command it shows."""
    args, payload = epilog_command_and_payload(tool)
    out = run(*args).stdout
    assert out == payload
    json.loads(out)


def test_help_shows_the_epilog() -> None:
    out = run("denom", "--help").stdout
    assert "worked example:" in out and "$ qreals denom 4/15 --json" in out


def test_top_level_help_lists_every_subcommand() -> None:
    parser = app_mod._build_parser()
    sub = next(a for a in parser._actions if getattr(a, "choices", None))
    names = sorted(sub.choices)
    out = run("--help").stdout
    for name in names:
        assert name in out


# --------------------------------------------------------------------------
# README and docs/HOW-IT-WORKS.md worked examples
# --------------------------------------------------------------------------


def test_readme_denom_19_60() -> None:
    """README "Worked example": qreals denom 19/60."""
    out = run("denom", "19/60").stdout
    for line in [
        "fraction in lowest terms: 19/60",
        "continued fraction: [0; 3, 6, 3]",
        "S(q) factored = Phi_2 * Phi_3 * Phi_4 * Phi_5 * Phi_6",
        "class: COLLAPSE",
    ]:
        assert line in out
    assert "60 = 3 * 20" in out and "RATIO" in out and "(Phi_10 Phi_20) / Phi_6" in out
    data = run_json("denom", "19/60")
    assert data["klass"] == "COLLAPSE"
    assert data["cf"] == [0, 3, 6, 3]
    assert data["S_at_1"] == 60 and data["S_at_1_ok"] is True
    split = next(s for s in data["splits"] if (s["d_plus"], s["d_minus"]) == (3, 20))
    assert split["discrepancy_class"] == "RATIO" and split["realized"] is True


def test_readme_conj_floor3() -> None:
    """README "Worked example": qreals conj floor3 --until 24."""
    out = run("conj", "floor3", "--until", "24").stdout
    for line in [
        "conjecture: floor3",
        "statement: The q-continuant is injective on tails whose entries are all at least 3.",
        "range covered: tails with entries >= 3 and entry sum <= 24, ascending sum then lexicographic",
        "instances checked: 5895",
        "verdict: survives the scanned range",
    ]:
        assert line in out


def test_how_it_works_qint_4() -> None:
    """docs/HOW-IT-WORKS.md section 1.1: qreals qint 4."""
    out = run("qint", "4").stdout
    assert "[4]_q = q**3 + q**2 + q + 1" in out
    assert "[4]_(q^-1) = (q**3 + q**2 + q + 1)/q**3" in out
    assert "[n]_q at q = 1: 4" in out


def test_how_it_works_coeffs_sqrt2() -> None:
    """docs/HOW-IT-WORKS.md section 1.3: qreals coeffs "sqrt(2)" 12."""
    data = run_json("coeffs", "sqrt(2)", "12")
    assert data["coefficients"] == [1, 0, 0, 1, 0, -2, 1, 4, -5, -7, 18, 7]
    assert data["verification"]["ok"] is True
    out = run("coeffs", "sqrt(2)", "12").stdout
    assert "1 + q^3 - 2*q^5 + q^6 + 4*q^7 - 5*q^8 - 7*q^9 + 18*q^10 + 7*q^11" in out.replace("\n", " ")


def test_how_it_works_denom_4_15() -> None:
    """docs/HOW-IT-WORKS.md section 1.2: qreals denom 4/15 --json."""
    data = run_json("denom", "4/15")
    assert data["cf"] == [0, 3, 1, 3]
    assert data["N"] == "1 + q + q^2 + q^3"
    assert data["S"] == "1 + 2*q + 3*q^2 + 3*q^3 + 3*q^4 + 2*q^5 + q^6"
    assert data["S_factored"] == "Phi_3 * Phi_5"


def test_check_show_schema() -> None:
    data = json.loads(run("check", "--show-schema").stdout)
    assert data["schema"] == "the literal string 'qreals-claim/1'"


# --------------------------------------------------------------------------
# core subcommands, values cross-checked against the literature
# --------------------------------------------------------------------------


def test_rational_5_2() -> None:
    """[5/2]_q = (1 + 2q + q^2 + q^3)/(1 + q), [MGO20] Example 1.2 (b)."""
    out = run("rational", "5", "2").stdout
    assert "(q**3 + q**2 + 2*q + 1)/(q + 1)" in out


def test_coeffs_golden_ratio() -> None:
    """[MGO22] Section 4.1."""
    data = run_json("coeffs", "(1+sqrt(5))/2", "12")
    assert data["coefficients"] == [1, 0, 1, -1, 2, -4, 8, -17, 37, -82, 185, -423]


def test_shift_up_and_down() -> None:
    """qreals shift: [x+1]_q = q[x]_q + 1 and [x-1]_q = ([x]_q - 1)/q,
    [MGO22] (3). The shifted list must equal the engine run on x +/- 1."""
    from qreals import q_real_truncated

    up = run_json("shift", "3/2", "--up")
    assert up["input_coefficients"] == q_real_truncated("3/2", 13)
    assert up["shifted_coefficients"][:13] == q_real_truncated("5/2", 13)
    down = run_json("shift", "7/2", "--down")
    assert down["shifted_coefficients"][:10] == q_real_truncated("5/2", 10)


def test_negate_matches_engine() -> None:
    """qreals negate: the Jouteur [-x]_q and the negation sum."""
    from qreals import q_neg

    data = run_json("negate", "sqrt(2)", "6")
    v, c = q_neg("sqrt(2)", 6)
    assert data["neg_valuation"] == v and data["neg_coefficients"] == c
    assert data["verification"]["ok"] is True


def test_exact_rational_function() -> None:
    """qreals exact 5/2: P/Q = (1 + 2q + q^2 + q^3)/(1 + q), [MGO20] Ex. 1.2."""
    data = run_json("exact", "5/2")
    assert data["P"] == "q**3 + q**2 + 2*q + 1" and data["Q"] == "q + 1"
    assert data["cf"] == [2, 2]


def test_bad_input_exits_nonzero() -> None:
    proc = run("rational", "5/3", check=False)
    assert proc.returncode != 0
    proc = run("no-such-command", check=False)
    assert proc.returncode != 0


def test_negation_sweep_small(tmp_path: Path) -> None:
    """README "Negation-sum sweeps": a tiny grid writes shards and SUMMARY.md."""
    out = tmp_path / "sweep"
    run(
        "negation-sweep", "--D", "2", "--qa-max", "1", "--qb-max", "1",
        "--pa-max", "1", "--pb-max", "1", "--depth", "24", "--min-zero-run", "8",
        "--out", str(out), "--workers", "1",
    )  # fmt: skip
    assert (out / "SUMMARY.md").exists()
    shards = list(out.glob("shard_*.csv"))
    assert shards and shards[0].read_text().strip()


# --------------------------------------------------------------------------
# the other surfaces: MCP tool registry and the local REST API
# --------------------------------------------------------------------------


def test_mcp_registry_counts() -> None:
    """README: 10 MCP tools and 3 catalog resources."""
    from qreals import mcp_server

    assert len(mcp_server.TOOLS) == 10
    assert len(mcp_server.RESOURCES) == 3
    out = mcp_server.q_coefficients("sqrt(2)", 8)
    assert out["coefficients"][:8] == [1, 0, 0, 1, 0, -2, 1, 4]
    bad = mcp_server.q_rational(1, 0)
    assert bad["isError"] is True


def test_rest_api_compute() -> None:
    """README: POST /api/v1/compute (checked on the Flask mirror app)."""
    pytest.importorskip("flask")
    from qreals import serve

    client = serve._build_flask_app().test_client()
    r = client.post("/api/v1/compute", json={"op": "rational", "input": "5/2"})
    assert r.status_code == 200
    assert r.get_json()["text"] == "(q**3 + q**2 + 2*q + 1)/(q + 1)"
    r = client.post("/api/v1/compute", json={"op": "no-such-op", "input": "1"})
    assert r.status_code == 400
