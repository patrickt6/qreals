"""`qreals doctor`: report the environment and which extras import, headlessly."""

from __future__ import annotations

from qreals import app


def test_doctor_report_has_every_capability_flag():
    report = app.doctor_report()
    for key in (
        "os",
        "python",
        "stdin_tty",
        "stdout_tty",
        "questionary",
        "rich",
        "tex_engine",
        "menu_will_run",
    ):
        assert key in report
    assert isinstance(report["stdin_tty"], bool)
    assert isinstance(report["stdout_tty"], bool)
    assert isinstance(report["questionary"], bool)
    assert isinstance(report["rich"], bool)
    # tex_engine is a name or None; never a bare True/False.
    assert report["tex_engine"] is None or isinstance(report["tex_engine"], str)


def test_doctor_runs_headlessly_and_reports(capsys):
    code = app.main(["doctor"])
    assert code == 0
    out = capsys.readouterr().out
    assert "qreals doctor" in out
    assert "operating system" in out
    assert "python" in out
    assert "verdict:" in out


def test_doctor_menu_verdict_reflects_no_tty(capsys):
    # In the test harness stdin/stdout are not a tty, so the menu will not run;
    # the verdict must say so rather than promise the menu.
    report = app.doctor_report()
    assert report["menu_will_run"] is False
    app.run_doctor(console=None)
    assert "verdict:" in capsys.readouterr().out
