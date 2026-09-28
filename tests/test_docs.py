"""The README's factual claims, pinned so a stale number fails the suite.

Each row of the README "At a glance" table is recomputed here from the
source tree, and the generated tool-reference block must equal a fresh
render of `python -m qreals.readme`.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

from qreals import app as app_mod
from qreals import mcp_server, readme

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def glance_value(label: str) -> str:
    m = re.search(rf"^\| {re.escape(label)} \| ([^|(]+?) \(", README, re.MULTILINE)
    assert m, f"README At a glance row {label!r} is missing"
    return m.group(1).strip()


def test_tool_reference_block_is_current() -> None:
    """The TOOL-SECTIONS block equals a fresh render from the --help epilogs."""
    start = README.index(readme.BEGIN)
    end = README.index(readme.END) + len(readme.END)
    assert README[start:end] == readme.render_block()


def test_module_count() -> None:
    files = [p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts]
    assert int(glance_value("Python modules")) == len(files)


def test_subcommand_count() -> None:
    parser = app_mod._build_parser()
    sub = next(a for a in parser._actions if getattr(a, "choices", None))
    assert int(glance_value("CLI subcommands")) == len(sub.choices)


def test_mcp_tool_count() -> None:
    assert int(glance_value("MCP tools")) == len(mcp_server.TOOLS)
    assert f"{len(mcp_server.TOOLS)} typed tools" in README
    assert f"{len(mcp_server.RESOURCES)} read-only catalog resources" in README


def test_python_versions_match_pyproject_and_ci() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    versions = [
        c.rsplit(":: ", 1)[1]
        for c in meta["project"]["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    ]
    assert glance_value("Python versions") == ", ".join(versions)
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    for v in versions:
        assert f'"{v}"' in workflow


def test_test_count() -> None:
    """The README test count equals `pytest --collect-only -q` right now."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    m = re.search(r"(\d+) tests? collected", proc.stdout)
    assert m, proc.stdout[-2000:]
    assert int(glance_value("Tests")) == int(m.group(1))


def test_no_dangling_references() -> None:
    """Every repository path the README mentions exists."""
    for path in re.findall(r"\((docs/[A-Za-z_-]+\.md)\)", README):
        assert (ROOT / path).exists(), path
    assert "tools/" not in README
