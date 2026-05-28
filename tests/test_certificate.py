"""The certificate: a human-auditable derivation, rendered without writing.

Terminal and .tex generation need no TeX engine and are always tested. The
PDF-compile path is tested only when a TeX engine is on PATH.
"""

from __future__ import annotations

import os
import re
import tempfile

import pytest

from qreals import app, certificate

# A certificate line stays inside the text block when its hbox is not overfull.
# A few points of overhang are routine for fractions and table rules; anything
# past this budget is the wall-of-terms defect this guardrail watches for.
_OVERFULL = re.compile(r"Overfull \\hbox \(([0-9.]+)pt")
_OVERFULL_BUDGET_PT = 5.0
# The two worst cases: pi to 60 coefficients reaches q^292 with hundreds of
# terms per fold, and a rational whose q-form is a long ratio.
_WORST_CASES = [
    ("pi-60-coeffs", {"kind": "coeffs", "data": {"x": "pi", "n": 60}}),
    ("long-rational", {"kind": "rational", "data": {"p": 355, "s": 113}}),
]


def _worst_overfull_pt(cert: certificate.Certificate) -> float:
    """Compile the certificate and return the widest overfull hbox in points."""
    engine = certificate.find_tex_engine()
    assert engine is not None  # guarded by the skipif on the caller
    with tempfile.TemporaryDirectory() as work:
        pdf = certificate._compile_tex(cert.to_tex(), cert.slug, work, engine)
        assert pdf is not None, "the certificate did not compile to a PDF"
        log_path = os.path.join(work, f"{cert.slug}.log")
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            log = handle.read()
    boxes = [float(pt) for pt in _OVERFULL.findall(log)]
    return max(boxes) if boxes else 0.0


def test_certificate_text_has_the_three_sections_for_a_rational():
    cert = certificate.build_certificate({"kind": "rational", "data": {"p": 3, "s": 2}})
    text = cert.text()
    # (a) the continued fraction and its even-length MGO form
    assert "continued fraction" in text
    assert "even-length MGO form" in text
    # (b) the fold evaluated step by step down to the result
    assert "folded step by step" in text
    assert "fold in a_1" in text
    assert "(q**2 + q + 1)/(q + 1)" in text
    # (c) the independent cross-checks
    assert "independent cross-checks" in text
    assert "q=1 matches 3/2" in text
    # citation and the honest framing
    assert "Forum Math. Sigma 8 (2020)" in text
    assert "docs/CORRECTNESS.md" in text
    assert "not a formal machine proof" in text


def test_certificate_text_for_an_irrational_shows_the_convergent_and_na_checks():
    cert = certificate.build_certificate(
        {"kind": "coeffs", "data": {"x": "pi", "n": 12}}
    )
    text = cert.text()
    assert "convergent" in text
    assert "1 + q + q^2 + q^10" in text  # the known [pi]_q coefficients
    assert "[n/a]" in text  # q=1 cannot run for pi and says so


# --- the referee view: a short prose proof, with the dump demoted ----------
# The redesign (docs/cert-critique-and-spec.md) makes the default view a referee
# view: the recursion stated once, named subscripted N_i/D_i, a per-fold table,
# and a closing sanity check, with the full polynomial trace kept in an appendix.

_REFEREE_KINDS = [
    ("rational", {"kind": "rational", "data": {"p": 333, "s": 106}}),
    ("qint", {"kind": "qint", "data": {"n": 5}}),
    ("coeffs", {"kind": "coeffs", "data": {"x": "pi", "n": 12}}),
    ("coeffs-rational", {"kind": "coeffs", "data": {"x": "22/7", "n": 12}}),
]


@pytest.mark.parametrize("name,result", _REFEREE_KINDS, ids=[n for n, _ in _REFEREE_KINDS])
def test_terminal_opens_with_a_referee_view_and_keeps_the_full_dump(name, result):
    text = certificate.build_certificate(result).text()
    # the referee view leads
    assert text.index("Referee view") < text.index("independent cross-checks")
    # a closing sanity check that ties back to a known value
    assert "Sanity check" in text
    assert "match" in text
    # the full intermediate-polynomial trace is demoted, not dropped
    appendix_at = text.index("Appendix: full intermediate-polynomial trace")
    assert appendix_at > text.index("Sanity check")
    assert "(b) MGO formula folded step by step" in text  # the dump is still there


@pytest.mark.parametrize(
    "name,result",
    [r for r in _REFEREE_KINDS if r[0] != "qint"],
    ids=[n for n, _ in _REFEREE_KINDS if n != "qint"],
)
def test_referee_view_states_the_recursion_once_with_named_quantities(name, result):
    text = certificate.build_certificate(result).text()
    # the MGO recursion is written out, not just named
    assert "folded inside out" in text
    assert "R_i = [a_i]_q" in text
    assert "MGO eqn 1.1" in text  # its citation, resolved inline
    # named, subscripted quantities and the telescoping fact, in words
    assert "R_i = N_i / D_i" in text
    assert "D_i = N_(i+1)" in text
    # the per-fold table: step, partial quotient, the partial as a ratio, degree
    assert "step i" in text and "partial R_i" in text and "deg N_i" in text
    assert "N_1 / D_1" in text


def test_pi_referee_view_compares_against_the_known_reference_coefficients():
    # The worked example from the spec: [pi]_q has c0=c1=c2=1, c3..c9=0, c10=1.
    cert = certificate.build_certificate({"kind": "coeffs", "data": {"x": "pi", "n": 12}})
    known = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert cert.coeffs == known  # the computed coefficients
    text = cert.text()
    # the full coefficient row appears with no interior ellipsis
    assert ", ".join(str(c) for c in known) in text
    # the sanity check states the match against the deeper convergent
    assert "all 12 computed coefficients agree with the deeper convergent" in text
    # a numeric witness the reader can reproduce on a calculator
    assert "Numeric witness at q = 1/2" in text


def test_qint_referee_view_has_no_fold_and_ties_back_at_q_equals_one():
    text = certificate.build_certificate({"kind": "qint", "data": {"n": 5}}).text()
    assert "no continued-fraction fold is required" in text
    assert "all-ones polynomial" in text
    assert "at q = 1 is the ordinary integer 5" in text


@pytest.mark.parametrize("name,result", _REFEREE_KINDS, ids=[n for n, _ in _REFEREE_KINDS])
def test_tex_carries_both_the_referee_view_and_the_appendix(name, result):
    tex = certificate.build_certificate(result).to_tex()
    assert r"\section*{Referee view}" in tex
    assert r"\section*{Appendix: full intermediate-polynomial trace}" in tex
    # the referee view comes before the appendix
    assert tex.index(r"\section*{Referee view}") < tex.index(
        r"\section*{Appendix: full intermediate-polynomial trace}"
    )
    # the closing comparison table is present
    assert "Sanity check" in tex


def test_tex_states_the_recursion_as_a_typeset_display_for_fold_kinds():
    tex = certificate.build_certificate(
        {"kind": "coeffs", "data": {"x": "pi", "n": 12}}
    ).to_tex()
    assert r"\begin{aligned}" in tex  # the recursion is a typeset display
    assert "R_{i+1}" in tex
    assert r"D_i = N_{i+1}" in tex  # the telescoping fact in words


def test_html_opens_with_a_referee_view_and_keeps_the_appendix():
    page = certificate.build_certificate(
        {"kind": "coeffs", "data": {"x": "pi", "n": 12}}
    ).to_html()
    assert "<h2>Referee view</h2>" in page
    assert "<h2>Appendix: full intermediate-polynomial trace</h2>" in page
    assert page.index("Referee view") < page.index("Appendix")


def test_to_tex_is_a_standalone_document_with_the_citation():
    cert = certificate.build_certificate({"kind": "rational", "data": {"p": 3, "s": 2}})
    tex = cert.to_tex()
    assert r"\documentclass" in tex
    assert r"\begin{document}" in tex and r"\end{document}" in tex
    assert "Forum Math. Sigma" in tex
    assert "CORRECTNESS.md" in tex


def test_render_terminal_prints_the_derivation(capsys):
    cert = certificate.build_certificate({"kind": "qint", "data": {"n": 5}})
    cert.render_terminal(console=None)
    out = capsys.readouterr().out
    assert "Certificate for the q-integer [5]_q" in out
    assert "independent cross-checks" in out


def test_save_writes_a_tex_into_the_chosen_directory(tmp_path):
    cert = certificate.build_certificate({"kind": "rational", "data": {"p": 3, "s": 2}})
    written = cert.save(str(tmp_path), qprov=False)
    assert os.path.exists(written["tex"])
    with open(written["tex"], encoding="utf-8") as handle:
        assert r"\documentclass" in handle.read()


@pytest.mark.skipif(
    certificate.find_tex_engine() is None, reason="no TeX engine on PATH"
)
def test_save_compiles_a_pdf_when_a_tex_engine_is_present(tmp_path):
    cert = certificate.build_certificate({"kind": "rational", "data": {"p": 3, "s": 2}})
    written = cert.save(str(tmp_path), qprov=False)
    assert written["pdf"] is not None
    assert os.path.exists(written["pdf"])


def test_headless_certify_prints_the_derivation_by_default_and_keeps_no_file(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    code = app.main(["certify", "rational", "3", "2"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Certificate for [3/2]_q" in out
    assert "folded step by step" in out
    assert os.listdir(tmp_path) == []  # the default certify wrote nothing


def test_headless_certify_save_writes_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = app.main(["certify", "coeffs", "pi", "12", "--save"])
    assert code == 0
    written = os.listdir(tmp_path)
    assert any(name.endswith(".tex") for name in written)


def test_certify_rejects_wrong_argument_count(capsys):
    code = app.main(["certify", "rational", "3"])
    assert code == 1
    assert "error" in capsys.readouterr().err.lower()


def test_to_tex_elides_long_polynomials_without_a_tex_engine():
    # The elision is in the .tex itself, so it is testable with no TeX engine.
    cert = certificate.build_certificate(
        {"kind": "coeffs", "data": {"x": "pi", "n": 60}}
    )
    tex = cert.to_tex()
    assert "terms omitted" in tex  # the appendix dump keeps the head/dots/tail note
    assert r"\dots" in tex
    # the referee view never elides a polynomial: the terminal text carries the
    # whole coefficient row and no "(K terms omitted)" note. The "..." that does
    # appear is continued-fraction notation ([a_1, ..., a_m]), not an elision.
    text = cert.text()
    assert "terms omitted" not in text
    assert len(cert.coeffs) == 60
    assert ", ".join(str(c) for c in cert.coeffs) in text


# Irrational and mixed inputs the certificate must build from. The defect this
# regression watches: a sharp even-length convergent (sqrt(2) and friends) once
# round-tripped through nsimplify back to the irrational and broke the build.
# Euler's number is spelled "E" in sympy parsing.
_CERTIFIABLE_INPUTS = [
    "sqrt(2)",
    "(1+sqrt(5))/2",
    "sqrt(3)",
    "sqrt(5)",
    "E",
    "1/sqrt(2)",
    "3/2",
    "22/7",
    "7",
]


@pytest.mark.parametrize("x", _CERTIFIABLE_INPUTS)
def test_certificate_builds_for_irrational_and_mixed_inputs(x):
    # The menu path: read-outs for [x]_q, then a certificate over that result.
    # No TeX engine is needed to build the object or its .tex source.
    cert = certificate.build_certificate(app.compute_readouts(x, 30))
    tex = cert.to_tex()
    assert "folded step by step" in tex  # the (b) derivation is present
    assert cert.convergent.is_Rational  # the convergent is the exact rational


@pytest.mark.skipif(
    certificate.find_tex_engine() is None, reason="no TeX engine on PATH"
)
@pytest.mark.parametrize("x", _CERTIFIABLE_INPUTS)
def test_irrational_certificate_compiles_within_the_line_budget(x):
    # Reuse the line-overflow guardrail on each input that now builds, so a deep
    # irrational convergent cannot quietly start running lines off the page.
    cert = certificate.build_certificate(app.compute_readouts(x, 30))
    worst = _worst_overfull_pt(cert)
    assert worst <= _OVERFULL_BUDGET_PT, (
        f"{x}: an overfull hbox of {worst}pt exceeds the "
        f"{_OVERFULL_BUDGET_PT}pt budget; a certificate line ran off the page"
    )


@pytest.mark.skipif(
    certificate.find_tex_engine() is None, reason="no TeX engine on PATH"
)
@pytest.mark.parametrize(
    "name,result", _WORST_CASES, ids=[name for name, _ in _WORST_CASES]
)
def test_worst_case_certificate_keeps_every_line_in_the_text_block(name, result):
    # The guardrail: build the worst-case certificate, compile it, and read the
    # LaTeX log. No line may overflow the text block past the small budget.
    cert = certificate.build_certificate(result)
    worst = _worst_overfull_pt(cert)
    assert worst <= _OVERFULL_BUDGET_PT, (
        f"{name}: an overfull hbox of {worst}pt exceeds the "
        f"{_OVERFULL_BUDGET_PT}pt budget; a certificate line ran off the page"
    )
