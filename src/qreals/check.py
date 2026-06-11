"""Claim files and the `qreals check` re-verification tool.

A claim file is a small JSON document that pins the `--json` output of one
public CLI invocation by its SHA-256 hash:

    {
      "schema": "qreals-claim/1",
      "tool": "denom",
      "args": ["19/60"],
      "expected_sha256": "<hex digest of the --json stdout>",
      "label": "free-text description of the cited fact",
      "runtime_seconds": 0.41
    }

`qreals check DIR` replays every `*.json` claim in DIR through the public
CLI entry point (`python -m qreals TOOL ARGS --json`), hashes the stdout,
and prints PASS or DRIFT per claim.  Any drift, error, or claim left
unrun because the total time budget was exhausted makes the exit code 1.

`qreals check DIR --new TOOL ARGS... --label "..."` creates a claim by
running the tool once and recording its hash and runtime.

Replays deliberately go through a subprocess running the installed public
entry point, never through private imports, so every claim is exactly
reproducible by a reader of the public repository.

Claim labels ship with the public repository, so they are linted with the
same rules as the tree: no em or en dashes, and no hit against the private
blocklist file named by the QREALS_BLOCKLIST environment variable (one
case-insensitive pattern per line; the file lives outside the repository
and is simply skipped when the variable is unset or the file is absent).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CLAIM_SCHEMA_ID = "qreals-claim/1"
BLOCKLIST_ENV = "QREALS_BLOCKLIST"
DEFAULT_BUDGET_SECONDS = 60.0

# The documented claim schema (G0.7).  --show-schema prints exactly this.
CLAIM_SCHEMA: dict[str, str] = {
    "schema": "the literal string 'qreals-claim/1'",
    "tool": "a public qreals subcommand, e.g. 'denom'",
    "args": "list of CLI argument strings (without --json)",
    "expected_sha256": "hex SHA-256 of the tool's --json stdout",
    "label": "free-text description; linted, ships publicly",
    "runtime_seconds": "wall time of the recording run, float",
}

# Written with escapes so this module passes its own lint.
_DASHES = {"\u2013": "en dash (U+2013)", "\u2014": "em dash (U+2014)"}

# Files exempt from the dash rule, mirroring the existing CI grep: the page
# server and the vendored web assets legitimately contain typographic dashes
# (third-party MathJax sources and CSS). The blocklist rule still applies.
_DASH_EXEMPT_PARTS = ("serve.py", "web")


def _blocklist_patterns() -> tuple[list["re.Pattern[str]"], bool]:
    """Compiled patterns from the private blocklist, and whether it loaded.

    Each non-comment line is a case-insensitive regular expression (a line
    that fails to compile is taken as a literal substring).

    The blocklist lives OUTSIDE the repository and is named by the
    QREALS_BLOCKLIST environment variable.  When the variable is unset or
    the file is missing (the public CI case) the blocklist part of the
    lint is skipped gracefully.
    """
    path = os.environ.get(BLOCKLIST_ENV, "")
    if not path:
        return [], False
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return [], False
    patterns = [
        line.strip() for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            compiled.append(re.compile(re.escape(pattern), re.IGNORECASE))
    return compiled, True


def lint_text(text: str) -> tuple[list[str], bool]:
    """Violations in one string, plus whether the blocklist was consulted.

    Always checks the dash rule; checks the private blocklist when it is
    available.  Blocklist hits never echo the matched pattern itself
    (the pattern is private); they report only its line number.
    """
    violations: list[str] = []
    for char, name in _DASHES.items():
        if char in text:
            violations.append(f"contains an {name}")
    patterns, loaded = _blocklist_patterns()
    for i, pattern in enumerate(patterns, start=1):
        if pattern.search(text):
            violations.append(f"matches private blocklist entry {i}")
    return violations, loaded


def _is_text_file(path: Path) -> bool:
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    return b"\x00" not in raw[:4096]


def _tracked_files(root: Path) -> list[Path] | None:
    """git-tracked files under root, or None when root is not a repository.

    The R1 lint is about what ships publicly, so inside a repository only
    tracked files are linted (local-only tests and notes are not pushed).
    """
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    return sorted(
        root / name for name in proc.stdout.split("\0") if name
    )


def lint_tree(root: Path) -> tuple[list[str], bool]:
    """Lint the public text files under root; returns (hits, loaded).

    Used by the local pre-push hook and the CI grep step (R1, G0.1-G0.3).
    Inside a git repository only tracked files are linted; otherwise the
    tree is walked with hidden directories, caches, and build output
    skipped.
    """
    skip_dirs = {".git", "__pycache__", ".pytest_cache", "dist", "build",
                 "node_modules", ".venv"}
    hits: list[str] = []
    loaded = False
    files = _tracked_files(root)
    if files is None:
        files = sorted(p for p in root.rglob("*") if p.is_file())
    for path in files:
        if any(part in skip_dirs or part.startswith(".") for part in path.parts):
            continue
        if not _is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        violations, loaded = lint_text(text)
        if any(part in _DASH_EXEMPT_PARTS for part in path.parts):
            violations = [v for v in violations if "dash" not in v]
        for v in violations:
            hits.append(f"{path}: {v}")
    return hits, loaded


def replay(tool: str, args: list[str]) -> tuple[str, str, int, float]:
    """Run one public CLI invocation; return (sha256, stdout, rc, seconds).

    The replay is a subprocess running `python -m qreals TOOL ARGS --json`,
    the same entry point a reader of the public repository has.  Nothing
    is imported from outside the installed package.
    """
    argv = [sys.executable, "-m", "qreals", tool, *args]
    if "--json" not in argv:
        argv.append("--json")
    start = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, text=True)
    seconds = time.monotonic() - start
    digest = hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()
    return digest, proc.stdout, proc.returncode, seconds


def load_claim(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("tool", "args", "expected_sha256", "label"):
        if key not in data:
            raise ValueError(f"{path.name}: claim is missing the '{key}' field")
    if not isinstance(data["args"], list):
        raise ValueError(f"{path.name}: 'args' must be a list of strings")
    return data


def new_claim(
    claims_dir: Path, tool: str, args: list[str], label: str
) -> tuple[Path, dict[str, Any]]:
    """Create one claim by running the tool and recording its hash.

    The label is linted first (G14.4); a violating label is refused
    because claim files ship with the public repository.
    """
    violations, _ = lint_text(label)
    if violations:
        raise ValueError("label fails the lint: " + "; ".join(violations))
    digest, _, rc, seconds = replay(tool, args)
    if rc != 0:
        raise ValueError(f"'{tool}' exited {rc}; not recording a claim")
    claim = {
        "schema": CLAIM_SCHEMA_ID,
        "tool": tool,
        "args": args,
        "expected_sha256": digest,
        "label": label,
        "runtime_seconds": round(seconds, 3),
    }
    claims_dir.mkdir(parents=True, exist_ok=True)
    name = f"{tool}-{digest[:12]}.json"
    path = claims_dir / name
    path.write_text(json.dumps(claim, indent=2) + "\n", encoding="utf-8")
    return path, claim


def check_dir(
    claims_dir: Path, budget_seconds: float = DEFAULT_BUDGET_SECONDS
) -> dict[str, Any]:
    """Replay every claim in claims_dir under a total time budget.

    Returns one report object that both renderings (human and JSON) print
    from (G0.7).  Claims that did not run because the budget was exhausted
    are listed explicitly; silent truncation is a gate failure (G0.18).
    """
    files = sorted(claims_dir.glob("*.json"))
    report: dict[str, Any] = {
        "claims_dir": str(claims_dir),
        "budget_seconds": budget_seconds,
        "claims": [],
        "not_run": [],
        "ok": True,
    }
    start = time.monotonic()
    for i, path in enumerate(files):
        elapsed = time.monotonic() - start
        if elapsed >= budget_seconds:
            report["not_run"] = [p.name for p in files[i:]]
            report["ok"] = False
            break
        entry: dict[str, Any] = {"file": path.name}
        try:
            claim = load_claim(path)
        except (ValueError, json.JSONDecodeError) as exc:
            entry.update(status="ERROR", detail=str(exc))
            report["claims"].append(entry)
            report["ok"] = False
            continue
        entry["label"] = claim["label"]
        label_violations, _ = lint_text(claim["label"])
        digest, _, rc, seconds = replay(claim["tool"], list(claim["args"]))
        entry["runtime_seconds"] = round(seconds, 3)
        if label_violations:
            entry.update(
                status="DRIFT",
                detail="label fails the lint: " + "; ".join(label_violations),
            )
            report["ok"] = False
        elif rc != 0:
            entry.update(status="DRIFT", detail=f"tool exited {rc}")
            report["ok"] = False
        elif digest != claim["expected_sha256"]:
            entry.update(status="DRIFT", detail="output hash changed")
            report["ok"] = False
        else:
            entry["status"] = "PASS"
        report["claims"].append(entry)
    report["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return report


def report_lines(report: dict[str, Any]) -> list[str]:
    """Human rendering of a check_dir report (same object as the JSON)."""
    lines: list[str] = []
    for entry in report["claims"]:
        label = entry.get("label", "")
        runtime = entry.get("runtime_seconds")
        suffix = f" ({runtime:.3f}s)" if runtime is not None else ""
        lines.append(f"{entry['status']:5s} {entry['file']}  {label}{suffix}")
        if entry.get("detail"):
            lines.append(f"      {entry['detail']}")
    for name in report["not_run"]:
        lines.append(f"SKIP  {name}  not run: time budget exhausted")
    if report["not_run"]:
        lines.append(
            "budget of %.1fs exhausted; %d claim(s) NOT covered"
            % (report["budget_seconds"], len(report["not_run"]))
        )
    n = len(report["claims"])
    passed = sum(1 for e in report["claims"] if e["status"] == "PASS")
    lines.append(
        f"{passed}/{n} claims pass in {report['elapsed_seconds']:.3f}s "
        f"(budget {report['budget_seconds']:.1f}s)"
    )
    return lines
