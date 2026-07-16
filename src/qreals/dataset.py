"""ML-ready dataset exports from the exact engine, stdlib writers only.

Two labelled tables the research program keeps rebuilding by hand, produced
here as row generators plus CSV and JSONL writers with no dependency beyond
the standard library (no pandas, no pyarrow). Everything a model would treat
as a value is an exact int, a string, or a bool; floats never appear where
exactness matters, so a round trip through CSV or JSONL loses nothing.

The negation table (`negation_dataset`) serves the open finiteness question
(Ovsienko Example 6.4, the x -> -x symmetry): one row per sqrt(d), with the
continued-fraction period length, the valuation and leading Laurent
coefficients of [x]_q + [-x]_q from `deficit.negation_panel`, the trailing
zero run in the computed window, and the finiteness verdict. The verdict is
NOT ground truth: it is the finite-order heuristic of `arithmetic.finite_xnegx`
(the sum is called finite when the top half of the order-N window is all
zeros), applied at this table's order N to the panel's own coefficients.
Every row therefore carries verdict_kind = "heuristic-order-N" spelling out
the order, so a downstream model cannot silently mistake a truncation
artifact for a label. The true criterion is open; sqrt(19) is a known example
where shallow windows mislead.

The atlas table (`atlas_dataset`) is the S(q) denominator atlas as rows: one
row per coprime proper fraction a/d, with the degree of S against its bound
d - 1, the regime class label from `factor.s_regime` (full, collapse,
nonsquarefree, noncyclotomic), and the saturation index e* (empty in the
impossibility branch, where S divides no [n]_q). These labels are exact
theorems of the factorisation, not heuristics.
"""

from __future__ import annotations

import csv
import json
from math import gcd
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import sympy as sp

from .deficit import negation_panel
from .factor import s_properties, s_regime

# A cell of an exported row: exact int, string, bool, or missing. No floats.
Cell = int | str | bool | None
Row = dict[str, Cell]

DEFAULT_NEGATION_ORDER = 400
N_COEFF_COLUMNS = 16


def _sqrt_period_length(d: int) -> int:
    """The period length of the regular continued fraction of sqrt(d).

    A quadratic surd has an eventually periodic CF; sympy returns it as
    [a0, ..., [period]] with the periodic block as the trailing list. A
    perfect square gives a finite CF (sqrt(d) is an integer), reported as
    period length 0.
    """
    root = sp.sqrt(sp.Integer(d))
    if root.is_integer:
        return 0
    cf = sp.continued_fraction(root)
    last = cf[-1]
    return len(last) if isinstance(last, list) else 0


def _trailing_zero_run(coeffs: Sequence[int]) -> int:
    """The length of the run of zeros at the top of a coefficient window."""
    run = 0
    for c in reversed(coeffs):
        if c != 0:
            break
        run += 1
    return run


def negation_dataset(
    d_values: Iterable[int], N: int = DEFAULT_NEGATION_ORDER
) -> Iterator[Row]:
    """Rows for the negation-finiteness question, one per sqrt(d).

    Args:
        d_values: the positive integers d to compute sqrt(d) rows for.
        N: the Laurent order carried by `deficit.negation_panel`; both the
            coefficient window and the finiteness heuristic are order N.
            Must be at least 16 (the coefficient columns) and at least 8
            (the minimum order at which termination can be judged at all).

    Yields one row (a plain dict) per d, in the given order, with columns:

    - ``d``: the input integer.
    - ``period_length``: CF period of sqrt(d) (0 for a perfect square).
    - ``valuation``: the lowest power of q in [x]_q + [-x]_q.
    - ``c_0`` .. ``c_15``: the first 16 Laurent coefficients of the sum,
      starting at q^valuation, exact ints.
    - ``trailing_zero_run``: zeros at the top of the order-N window.
    - ``finite_verdict``: True when the sum looks terminated at order N,
      by the same rule as `arithmetic.finite_xnegx` (the trailing zero run
      covers at least half the window), applied to the panel's coefficients.
    - ``verdict_kind``: the string "heuristic-order-N" with the actual N,
      e.g. "heuristic-order-400". The verdict is a finite-order observation,
      never ground truth; models consuming this column must treat it as a
      noisy label for the open criterion.
    """
    if N < max(N_COEFF_COLUMNS, 8):
        raise ValueError(
            f"N must be at least {max(N_COEFF_COLUMNS, 8)} to fill the "
            "coefficient columns and judge termination"
        )
    for d_raw in d_values:
        d = int(d_raw)
        if d < 1:
            raise ValueError(f"d must be a positive integer, got {d}")
        panel = negation_panel(f"sqrt({d})", N)
        run = _trailing_zero_run(panel.sum_coeffs)
        row: Row = {
            "d": d,
            "period_length": _sqrt_period_length(d),
            "valuation": int(panel.valuation),
        }
        for i in range(N_COEFF_COLUMNS):
            row[f"c_{i}"] = (
                int(panel.sum_coeffs[i]) if i < len(panel.sum_coeffs) else 0
            )
        row["trailing_zero_run"] = run
        # The finite_xnegx rule at order N, on the panel's own window: a
        # terminating Laurent polynomial leaves a long zero tail; a genuine
        # infinite series keeps producing nonzero coefficients near the top.
        row["finite_verdict"] = run >= N // 2
        row["verdict_kind"] = f"heuristic-order-{N}"
        yield row


def atlas_dataset(d_max: int, a_max: int | None = None) -> Iterator[Row]:
    """Rows of the S(q) denominator atlas over the coprime grid up to d_max.

    Args:
        d_max: sweep denominators 2 <= d <= d_max.
        a_max: optional cap on the numerator a for each d.

    Yields one row per proper fraction a/d in lowest terms (0 < a/d < 1), in
    (d, a) order, with columns:

    - ``a``, ``d``: the fraction in lowest terms.
    - ``deg_S``: the degree of the q-denominator S(q) of [a/d]_q.
    - ``deg_bound``: the theorem bound d - 1 (equality iff a == +/-1 mod d).
    - ``regime``: the class label from `factor.s_regime`, one of "full",
      "collapse", "nonsquarefree", "noncyclotomic".
    - ``saturation_index``: e* = lcm(T), the minimal n with S | [n]_q, or
      None (empty in CSV, null in JSONL) in the impossibility branch where
      S divides no [n]_q.

    Unlike the negation verdict, every column here is exact: the regime and
    the saturation index are consequences of the Z[q] factorisation.
    """
    if d_max < 2:
        raise ValueError("d_max must be at least 2")
    for d in range(2, d_max + 1):
        cap = d - 1 if a_max is None else min(a_max, d - 1)
        for a in range(1, cap + 1):
            if gcd(a, d) != 1:
                continue
            p = s_properties((a, d))
            yield {
                "a": a,
                "d": d,
                "deg_S": int(p.deg_S),
                "deg_bound": int(p.deg_bound),
                "regime": s_regime(p),
                "saturation_index": p.saturation_index,
            }


# --------------------------------------------------------------------------- #
# Writers. Standard library only; every row must share the first row's
# columns so the files are rectangular and header-stable.
# --------------------------------------------------------------------------- #
def to_csv(rows: Iterable[Mapping[str, Cell]], path: str | Path) -> int:
    """Write rows to a CSV file with a header; returns the row count.

    The first row fixes the column order; every later row must carry exactly
    the same keys or a ValueError names the offender. Values are written by
    str(): ints as full decimal strings (no float rounding, arbitrarily
    large), bools as "True"/"False", None as the empty string.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    header: list[str] | None = None
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for row in rows:
            if header is None:
                header = list(row.keys())
                writer.writerow(header)
            elif set(row.keys()) != set(header):
                raise ValueError(
                    f"row {count} has columns {sorted(row.keys())}, "
                    f"expected {sorted(header)}"
                )
            writer.writerow(
                ["" if row[k] is None else str(row[k]) for k in header]
            )
            count += 1
        if header is None:
            raise ValueError("no rows to write")
    return count


def to_jsonl(rows: Iterable[Mapping[str, Cell]], path: str | Path) -> int:
    """Write rows to a JSON Lines file, one object per line; returns the count.

    Ints stay exact (JSON integers are unbounded), bools become true/false,
    None becomes null. Every row must carry the same keys as the first, in
    the same declared order, so the file is column-stable for loaders that
    stream it.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    header: list[str] | None = None
    with open(out, "w", encoding="utf-8") as fh:
        for row in rows:
            if header is None:
                header = list(row.keys())
            elif set(row.keys()) != set(header):
                raise ValueError(
                    f"row {count} has columns {sorted(row.keys())}, "
                    f"expected {sorted(header)}"
                )
            fh.write(json.dumps({k: row[k] for k in header}) + "\n")
            count += 1
        if header is None:
            raise ValueError("no rows to write")
    return count
