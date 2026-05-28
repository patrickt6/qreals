"""Build one certificate of each computation kind, for the critique-and-spec.

Writes, per kind, a terminal-text dump (<slug>.txt), and a .tex/.pdf when a
TeX engine is present, into this directory. Run from anywhere:

    python docs/cert-samples/_build_samples.py
"""

from __future__ import annotations

import io
import os
import sys
import traceback
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)

from qreals import app  # noqa: E402
from qreals import certificate as cert_mod  # noqa: E402

# One representative result of every kind that the menu/CLI can produce.
RESULTS = {
    "rational": lambda: app.compute_rational(333, 106),
    "qint": lambda: app.compute_qint(5),
    "coeffs_rational": lambda: app.compute_coeffs("333/106", 12),
    "coeffs_irrational": lambda: app.compute_coeffs("pi", 12),
    "laurent": lambda: app.compute_laurent("sqrt(2)", 12),
    "prefix": lambda: app.compute_prefix("pi"),
    "locked": lambda: app.compute_locked("pi", 4),
    "shift": lambda: app.compute_shift("sqrt(2)", 12, "up"),
    "readouts": lambda: app.compute_readouts("pi", 12),
    "arith_add": lambda: app.compute_arith("1/2", "1/3", 12, "add"),
    "arith_mul": lambda: app.compute_arith("1/2", "1/3", 12, "mul"),
    "negation": lambda: app.compute_negation("sqrt(2)", 12),
    "radius": lambda: app.compute_radius("pi", 24),
    "oeis": lambda: {"kind": "oeis", "title": "oeis", "blocks": [], "data": {"sequence": "1,1,1"}},
    "fingerprint": lambda: {"kind": "fingerprint", "title": "fp", "blocks": [], "data": {"x": "pi"}},
}

engine = cert_mod.find_tex_engine()
print(f"TeX engine: {engine}")

summary = []
for name, make in RESULTS.items():
    line = {"name": name}
    try:
        result = make()
        line["result_kind"] = result.get("kind")
    except Exception as exc:  # noqa: BLE001
        line["compute_error"] = repr(exc)
        summary.append(line)
        print(f"[{name}] compute failed: {exc!r}")
        continue
    try:
        cert = cert_mod.build_certificate(result)
    except Exception as exc:  # noqa: BLE001
        line["build_error"] = repr(exc)
        summary.append(line)
        print(f"[{name}] build_certificate failed: {exc!r}")
        continue
    line["cert_title"] = cert.title
    line["cert_slug"] = cert.slug
    # terminal text
    buf = io.StringIO()
    with redirect_stdout(buf):
        cert.render_terminal(None)
    txt_path = os.path.join(HERE, f"{name}.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    line["txt"] = os.path.basename(txt_path)
    # tex + pdf
    try:
        cert.slug = f"{name}-{cert.slug}"
        written = cert.save(HERE, compile_pdf=True, qprov=False)
        line["tex"] = os.path.basename(written["tex"]) if written["tex"] else None
        line["pdf"] = os.path.basename(written["pdf"]) if written["pdf"] else None
    except Exception as exc:  # noqa: BLE001
        line["save_error"] = repr(exc)
        traceback.print_exc()
    summary.append(line)
    print(f"[{name}] -> title={cert.title!r}")

print("\n=== SUMMARY ===")
for line in summary:
    print(line)
