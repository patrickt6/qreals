"""The capability-by-input-by-action matrix, as a regression test.

This pins the audit behind the sqrt(2)-certificate fix: every interactive
capability is exercised across a spread of input types (positive and negative
integers, proper/improper/unit rationals, recognizable irrationals, sub-one
values, zero, and the smallest sizes), and every follow-up action is run on the
result (terminal render in plain and rich, the JSON dump, the certificate
terminal view and saved .tex, the PDF compile when a TeX engine is on PATH, and
the headless certify command).

Two contracts are locked in:

- a supported (capability, input) cell computes and renders through every
  action without raising;
- a genuinely unsupported (capability, input) cell raises a clean ``ValueError``
  with a recognizable message, never a cryptic crash or a quiet wrong answer.

The PDF-compile cases skip when no TeX engine is found, so the matrix runs in a
plain checkout. See ``notes/input-output-matrix-audit.md`` for the bugs this
caught and how they were fixed.
"""

from __future__ import annotations

import contextlib
import io
import os

import pytest

from qreals import app, certificate

# Inputs that the [x]_q series capabilities accept (x >= 0): the full spread of
# types the audit covers.
X_SPREAD = [
    "7",  # positive integer
    "3/5",  # proper rational
    "7/3",  # improper rational
    "5/5",  # unit rational (= 1), CF is the single term [1]
    "sqrt(2)",
    "sqrt(3)",
    "sqrt(5)",
    "(1+sqrt(5))/2",  # golden ratio
    "E",  # Euler's number (sympy spelling)
    "pi",
    "1/sqrt(2)",  # sub-one irrational
    "1/3",  # sub-one rational
    "0",  # zero, CF is the single term [0]
]
# x >= 1, where the down-shift [x-1]_q stays a power series (constant term 1).
X_GE_ONE = ["7", "7/3", "5/5", "sqrt(2)", "sqrt(3)", "sqrt(5)", "(1+sqrt(5))/2", "E", "pi"]


def _render_all(result: dict) -> None:
    """Run the three render actions: plain terminal, rich terminal, JSON."""
    from rich.console import Console

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        app.render_result(result, console=None, as_json=False)
        app.render_result(result, console=None, as_json=True)
    con = Console(file=io.StringIO(), force_terminal=False, width=100)
    app.render_result(result, console=con, as_json=False)


def _cert_no_compile(result: dict, tmp_path) -> certificate.Certificate:
    """Build the certificate and run the no-engine actions: terminal and .tex."""
    cert = certificate.build_certificate(result)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        cert.render_terminal(console=None)
    cert.to_tex()
    written = cert.save(str(tmp_path), compile_pdf=False, qprov=False)
    assert os.path.exists(written["tex"])
    return cert


# --------------------------------------------------------------------------
# Supported cells: compute, render (plain/rich/JSON), build a certificate, and
# write its .tex, all without raising.
# --------------------------------------------------------------------------

# Each entry: (id, builder, certifiable). builder() returns the result dict.
_SUPPORTED: list[tuple[str, object, bool]] = []


def _add(idtag: str, builder, certifiable: bool = True) -> None:
    _SUPPORTED.append((idtag, builder, certifiable))


# rational: every rational input including negative, zero and the unit p=s.
for _p, _s in [(7, 1), (3, 5), (7, 3), (5, 5), (1, 3), (0, 1), (1, 1), (-3, 1)]:
    _add(f"rational-{_p}-{_s}", (lambda p=_p, s=_s: app.compute_rational(p, s)))

# qint: positive, negative, zero, and the smallest size n = 1.
for _n in [7, -3, 0, 1]:
    _add(f"qint-{_n}", (lambda n=_n: app.compute_qint(n)))

# coeffs and readouts over the full nonnegative spread, plus the smallest size.
for _x in X_SPREAD:
    _add(f"coeffs-{_x}-N12", (lambda x=_x: app.compute_coeffs(x, 12)))
    _add(f"readouts-{_x}-N30", (lambda x=_x: app.compute_readouts(x, 30)))
_add("coeffs-pi-N1", (lambda: app.compute_coeffs("pi", 1)))
_add("readouts-sqrt2-N1", (lambda: app.compute_readouts("sqrt(2)", 1)))

# laurent over the spread at order 12, plus the smallest orders 1 and 0.
for _x in X_SPREAD:
    _add(f"laurent-{_x}-O12", (lambda x=_x: app.compute_laurent(x, 12)))
_add("laurent-pi-O1", (lambda: app.compute_laurent("pi", 1)))
_add("laurent-1/3-O0", (lambda: app.compute_laurent("1/3", 0)))

# prefix over the nonnegative spread (the integer-part lemma needs x >= 0).
for _x in X_SPREAD:
    _add(f"prefix-{_x}", (lambda x=_x: app.compute_prefix(x)))

# locked: positive x with enough terms and a meaningful partial sum.
for _x, _n in [("pi", 2), ("sqrt(2)", 2), ("7", 1), ("1/3", 2), ("(1+sqrt(5))/2", 2)]:
    _add(f"locked-{_x}-n{_n}", (lambda x=_x, n=_n: app.compute_locked(x, n)))

# shift up over the spread; shift down only for x >= 1; smallest order too.
for _x in X_SPREAD:
    _add(f"shift-up-{_x}-O12", (lambda x=_x: app.compute_shift(x, 12, "up")))
for _x in X_GE_ONE:
    _add(f"shift-down-{_x}-O12", (lambda x=_x: app.compute_shift(x, 12, "down")))
_add("shift-up-pi-O1", (lambda: app.compute_shift("pi", 1, "up")))
_add("shift-down-pi-O1", (lambda: app.compute_shift("pi", 1, "down")))

# arithmetic: add and mul over nonnegative pairs, including zero and irrationals;
# the certificate is a series derivation over x only, so it is not asserted here.
for _idt, _x, _y in [
    ("rat-rat", "3/2", "13/5"),
    ("int-int", "2", "3"),
    ("irr-irr", "sqrt(2)", "sqrt(3)"),
    ("subone", "1/3", "1/sqrt(2)"),
    ("zero", "0", "5"),
]:
    for _op in ("add", "mul"):
        _add(
            f"arith-{_idt}-{_op}-N12",
            (lambda x=_x, y=_y, op=_op: app.compute_arith(x, y, 12, op)),
            certifiable=False,
        )
_add("arith-rat-add-N1", (lambda: app.compute_arith("3/2", "13/5", 1, "add")), False)

# negation over the nonnegative spread, plus the smallest size.
for _x in X_SPREAD:
    _add(f"negate-{_x}-N12", (lambda x=_x: app.compute_negation(x, 12)), certifiable=False)
_add("negate-sqrt2-N1", (lambda: app.compute_negation("sqrt(2)", 1)), False)


@pytest.mark.parametrize(
    "builder,certifiable",
    [(b, c) for (_id, b, c) in _SUPPORTED],
    ids=[i for (i, _b, _c) in _SUPPORTED],
)
def test_supported_cell_runs_every_action(builder, certifiable, tmp_path):
    result = builder()
    _render_all(result)
    if certifiable:
        _cert_no_compile(result, tmp_path)


# --------------------------------------------------------------------------
# Unsupported cells: a clean ValueError, never a cryptic crash or a quiet wrong
# answer. The message substring is asserted so the degrade stays explanatory.
# --------------------------------------------------------------------------

_UNSUPPORTED: list[tuple[str, object, str]] = [
    # negative x on the series path: [x]_q lives in negative powers of q.
    ("coeffs-neg", (lambda: app.compute_coeffs("-3", 12)), "x >= 0"),
    ("laurent-neg", (lambda: app.compute_laurent("-3", 12)), "x >= 0"),
    ("readouts-neg", (lambda: app.compute_readouts("-3", 30)), "x >= 0"),
    ("shift-up-neg", (lambda: app.compute_shift("-3", 12, "up")), "x >= 0"),
    ("shift-down-neg", (lambda: app.compute_shift("-3", 12, "down")), "x >= 0"),
    ("negate-neg", (lambda: app.compute_negation("-3", 12)), ">= 0"),
    ("prefix-neg", (lambda: app.compute_prefix("-3")), "x >= 0"),
    # down-shift below 1: [x-1]_q would need a negative power.
    ("shift-down-proper", (lambda: app.compute_shift("3/5", 12, "down")), "constant term 1"),
    ("shift-down-subone-irr", (lambda: app.compute_shift("1/sqrt(2)", 12, "down")), "constant term 1"),
    ("shift-down-zero", (lambda: app.compute_shift("0", 12, "down")), "constant term 1"),
    # convergent locking with a non-positive partial sum (x <= 0, or 0<x<1 at n=1).
    ("locked-neg", (lambda: app.compute_locked("-3", 1)), "S_n"),
    ("locked-subone-n1", (lambda: app.compute_locked("1/3", 1)), "S_n"),
    ("locked-zero-n1", (lambda: app.compute_locked("0", 1)), "S_n"),
    # convergent index past the available continued-fraction terms.
    ("locked-int-n2", (lambda: app.compute_locked("7", 2)), "continued-fraction terms"),
    ("locked-zero-n2", (lambda: app.compute_locked("0", 2)), "continued-fraction terms"),
]


@pytest.mark.parametrize(
    "builder,message",
    [(b, m) for (_id, b, m) in _UNSUPPORTED],
    ids=[i for (i, _b, _m) in _UNSUPPORTED],
)
def test_unsupported_cell_degrades_cleanly(builder, message):
    with pytest.raises(ValueError) as exc:
        builder()
    assert message in str(exc.value)


# --------------------------------------------------------------------------
# The PDF-compile action, over a representative subset, skipped with no engine.
# It covers each certificate path and the inputs the build once broke on (the
# unit rational, zero, the negative rational, and a deep irrational convergent).
# --------------------------------------------------------------------------

_PDF_CASES = [
    ("rational-3-2", lambda: app.compute_rational(3, 2)),
    ("rational-unit", lambda: app.compute_rational(5, 5)),
    ("rational-zero", lambda: app.compute_rational(0, 1)),
    ("rational-neg", lambda: app.compute_rational(-3, 1)),
    ("rational-long", lambda: app.compute_rational(355, 113)),
    ("qint-5", lambda: app.compute_qint(5)),
    ("qint-neg", lambda: app.compute_qint(-3)),
    ("coeffs-unit", lambda: app.compute_coeffs("5/5", 12)),
    ("coeffs-zero", lambda: app.compute_coeffs("0", 12)),
    ("coeffs-sqrt2", lambda: app.compute_coeffs("sqrt(2)", 30)),
    ("laurent-subone-O0", lambda: app.compute_laurent("1/3", 0)),
]


@pytest.mark.skipif(
    certificate.find_tex_engine() is None, reason="no TeX engine on PATH"
)
@pytest.mark.parametrize(
    "builder", [b for (_id, b) in _PDF_CASES], ids=[i for (i, _b) in _PDF_CASES]
)
def test_certificate_pdf_compiles(builder, tmp_path):
    cert = certificate.build_certificate(builder())
    written = cert.save(str(tmp_path), compile_pdf=True, qprov=False)
    assert written["pdf"] is not None
    assert os.path.exists(written["pdf"])


# --------------------------------------------------------------------------
# The headless certify command, the fourth render path, over its four kinds and
# the inputs that once broke it (unit rational, zero). Exit 0 and no traceback.
# --------------------------------------------------------------------------

_CERTIFY_CASES = [
    ("rational-3-2", ["rational", "3", "2"]),
    ("rational-unit", ["rational", "5", "5"]),
    ("rational-zero", ["rational", "0", "1"]),
    ("qint-5", ["qint", "5"]),
    ("qint-neg", ["qint", "-3"]),
    ("coeffs-sqrt2", ["coeffs", "sqrt(2)", "12"]),
    ("coeffs-unit", ["coeffs", "5/5", "12"]),
    ("coeffs-zero", ["coeffs", "0", "12"]),
    ("laurent-pi", ["laurent", "pi", "--order", "12"]),
    ("laurent-zero", ["laurent", "0", "--order", "1"]),
]


@pytest.mark.parametrize(
    "argv", [a for (_id, a) in _CERTIFY_CASES], ids=[i for (i, _a) in _CERTIFY_CASES]
)
def test_headless_certify_default_view(argv, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = app.main(["certify", *argv])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Traceback" not in captured.err
    assert os.listdir(tmp_path) == []  # the default certify writes nothing
