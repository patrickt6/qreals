"""The export formats: JSON, CSV, a LaTeX table, and Magma code.

Each format is checked for well-formedness without a terminal: the JSON and CSV
are parsed back, the LaTeX is compiled headlessly when a TeX engine is present,
and the Magma is parsed to confirm it assigns the coefficient sequences it
should. A final test confirms that rendering writes nothing on its own.
"""

from __future__ import annotations

import csv
import io
import json
import re

import pytest

from qreals import exports
from qreals.store import SavedEntry

_PI = SavedEntry(input="pi", n=12, coefficients=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
_SQRT2 = SavedEntry(input="sqrt(2)", n=8, coefficients=[1, 0, 0, 1, 0, -2, 1, 4])
_ENTRIES = [_PI, _SQRT2]


# -- JSON ------------------------------------------------------------------


def test_json_parses_and_carries_every_entry():
    payload = json.loads(exports.to_json(_ENTRIES))
    assert payload["tool"] == "qreals"
    assert [e["input"] for e in payload["entries"]] == ["pi", "sqrt(2)"]
    assert payload["entries"][0]["coefficients"] == _PI.coefficients


def test_json_round_trips_through_savedentry():
    payload = json.loads(exports.to_json(_PI))
    rebuilt = SavedEntry.from_dict(payload["entries"][0])
    assert rebuilt.input == _PI.input
    assert rebuilt.coefficients == _PI.coefficients
    assert rebuilt.n == _PI.n


# -- CSV -------------------------------------------------------------------


def test_csv_is_one_row_per_constant_and_parses():
    rows = list(csv.reader(io.StringIO(exports.to_csv(_ENTRIES))))
    header, body = rows[0], rows[1:]
    assert header[:3] == ["label", "input", "n"]
    assert len(body) == len(_ENTRIES)
    by_name = {r[1]: r for r in body}
    coeffs_cell = by_name["pi"][header.index("coefficients")]
    assert [int(t) for t in coeffs_cell.split()] == _PI.coefficients


# -- LaTeX -----------------------------------------------------------------


def test_latex_is_a_booktabs_table_with_a_row_per_entry():
    tex = exports.to_latex(_ENTRIES)
    assert r"\begin{table}" in tex and r"\end{table}" in tex
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
    assert tex.count(r"\\") == len(_ENTRIES) + 1  # one header row plus the data rows
    assert r"\pi" in tex and r"\sqrt{2}" in tex  # inputs rendered as math


def _tex_engine():
    from qreals import certificate

    return certificate.find_tex_engine()


@pytest.mark.skipif(_tex_engine() is None, reason="no TeX engine on PATH")
def test_latex_table_compiles_to_a_pdf(tmp_path):
    from qreals import certificate

    document = exports.latex_document(exports.to_latex(_ENTRIES))
    engine = certificate.find_tex_engine()
    pdf = certificate._compile_tex(
        document, "qreals-export-table", str(tmp_path), engine
    )
    assert pdf is not None, "the exported LaTeX table did not compile to a PDF"


# -- Magma -----------------------------------------------------------------

_COEFF_ASSIGN = re.compile(r"^(coeffs_\w+) := \[([^\]]*)\];$", re.M)
_SERIES_ASSIGN = re.compile(
    r"^(q_\w+) := \(&\+\[(coeffs_\w+)\[i\+1\]\*.*\) \+ O\(q\^(\d+)\);$", re.M
)


def test_magma_assigns_the_expected_coefficient_sequences():
    code = exports.to_magma(_ENTRIES)
    assert "LaurentSeriesRing(RationalField())" in code

    coeff_assigns = _COEFF_ASSIGN.findall(code)
    assert len(coeff_assigns) == len(_ENTRIES)
    for entry, (_name, body) in zip(_ENTRIES, coeff_assigns):
        parsed = [int(t) for t in body.split(",")] if body.strip() else []
        assert parsed == entry.coefficients

    series = _SERIES_ASSIGN.findall(code)
    assert len(series) == len(_ENTRIES)
    for entry, (name, coeff_name, prec) in zip(_ENTRIES, series):
        assert coeff_name == "coeffs_" + name  # the series uses its own sequence
        assert int(prec) == entry.valuation + len(entry.coefficients)


def test_magma_handles_a_negative_valuation():
    entry = SavedEntry(
        input="neg", n=4, coefficients=[1, -1, 2, -3], valuation=-2, label="[neg]_q"
    )
    code = exports.to_magma(entry)
    assert "q^(i - 2)" in code
    assert "O(q^2)" in code  # valuation -2 plus four terms


# -- provenance column -----------------------------------------------------


def test_qprov_id_surfaces_in_every_format_when_present():
    tagged = SavedEntry(
        input="pi", n=6, coefficients=[1, 1, 1, 0, 0, 0], qprov_id="abc123"
    )
    assert "abc123" in exports.to_json(tagged)
    assert "abc123" in exports.to_csv(tagged)
    assert "abc123" in exports.to_magma(tagged)
    latex = exports.to_latex([tagged])
    assert "qprov id" in latex and r"\texttt{abc123}" in latex


def test_latex_omits_the_qprov_column_when_no_entry_has_an_id():
    assert "qprov id" not in exports.to_latex(_ENTRIES)


# -- writing only on request ----------------------------------------------


def test_rendering_writes_nothing_and_write_export_writes_one_file(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    for fmt in exports.FORMATS:
        exports.render(_ENTRIES, fmt)
    assert list(tmp_path.iterdir()) == []  # rendering alone left the folder empty

    out = tmp_path / "saved.json"
    written = exports.write_export(exports.to_json(_ENTRIES), out)
    assert written == out and out.exists()


def test_render_accepts_a_single_entry():
    rows = list(csv.reader(io.StringIO(exports.render(_PI, "csv"))))
    assert len(rows) == 2  # header plus one data row
