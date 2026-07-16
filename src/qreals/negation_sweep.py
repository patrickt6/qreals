r"""Sweep driver for G(x) = [x]_q + [-x]_q over a rational grid a + b*sqrt(D).

Builds an evidence grid at a fixed squarefree D, with a = pa/qa and
b = pb/qb ranging over rationals with growing denominators and numerators,
and computes the certified, locked-coefficient verdict for each point via
`negation`. One worker process per shard: each worker owns a CSV file it
appends to and flushes after every row, so a crash or Ctrl-C loses at most the
in-flight row. `--resume` reads every existing shard file's (pa, qa, pb, qb)
column and skips those combinations. SUMMARY.md is rewritten by whichever
worker crosses a multiple of 500 of its own rows, from all shard files on
disk at that moment, so it stays roughly current across the whole sweep.

CLI:
    python -m qreals.negation_sweep --D 2 --qa-max 4 --qb-max 4 \
        --pa-max 8 --pb-max 8 --depth 120 --out OUTDIR
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from multiprocessing import Process
from pathlib import Path

from .negation import WINDOW_LO, classify, default_lock_target, locked_series, polynomial_string
from .quadratic import QuadraticIrrational

CSV_FIELDS = [
    "pa",
    "qa",
    "pb",
    "qb",
    "x_decimal",
    "on_axis",
    "locked_depth",
    "verdict",
    "first_nonzero_tail_index",
    "G_polynomial",
    "elapsed_sec",
]


@dataclass(frozen=True)
class SweepGridPoint:
    pa: int
    qa: int
    pb: int
    qb: int


def build_grid(qa_max: int, qb_max: int, pa_max: int, pb_max: int) -> list[SweepGridPoint]:
    """The deduplicated grid of (pa, qa, pb, qb), lowest terms, pb >= 1.

    a = pa/qa ranges over every rational with 1 <= qa <= qa_max and
    |pa| <= pa_max in lowest terms; b = pb/qb ranges the same way but with
    pb >= 1 (b = 0 collapses to a plain rational, already covered elsewhere,
    and the sign of b carries no new information since [x]_q + [-x]_q is
    even in b through the D-fixed reflection sqrt(D) -> -sqrt(D) composed
    with a -> a).
    """
    seen: set[tuple[Fraction, Fraction]] = set()
    grid: list[SweepGridPoint] = []
    for qa in range(1, qa_max + 1):
        for pa in range(-pa_max, pa_max + 1):
            if pa != 0 and gcd(abs(pa), qa) != 1:
                continue
            a = Fraction(pa, qa)
            for qb in range(1, qb_max + 1):
                for pb in range(1, pb_max + 1):
                    if gcd(pb, qb) != 1:
                        continue
                    b = Fraction(pb, qb)
                    key = (a, b)
                    if key in seen:
                        continue
                    seen.add(key)
                    grid.append(SweepGridPoint(pa, qa, pb, qb))
    grid.sort(key=lambda g: (g.qa, abs(g.pa), g.qb, g.pb))
    return grid


def compute_row(point: SweepGridPoint, D: int, depth: int, min_zero_run: int) -> dict:
    """One sweep row for a + b*sqrt(D), certified via locked HJ convergents."""
    t0 = time.perf_counter()
    a = Fraction(point.pa, point.qa)
    b = Fraction(point.pb, point.qb)
    x = QuadraticIrrational.from_ab(a, b, D)
    lock_target = default_lock_target(depth, min_zero_run)
    # The Laurent window only needs to reach lock_target, not the caller's
    # full requested depth: classify() never inspects past locked_depth, and
    # keeping the truncated-matrix window small is what makes the sweep fast.
    window_depth = lock_target
    pos_series, pos_depth = locked_series(x, window_depth, lock_target=lock_target)
    neg_series, neg_depth = locked_series(-x, window_depth, lock_target=lock_target)
    locked_depth = min(pos_depth, neg_depth)
    window = range(WINDOW_LO, WINDOW_LO + locked_depth)
    g_series = {d: pos_series.get(d, 0) + neg_series.get(d, 0) for d in window}
    verdict = classify(g_series, locked_depth, min_zero_run=min_zero_run)
    elapsed = time.perf_counter() - t0
    on_axis = a == 0 and b.denominator == 1
    return {
        "pa": point.pa,
        "qa": point.qa,
        "pb": point.pb,
        "qb": point.qb,
        "x_decimal": f"{float(a) + float(b) * (D ** 0.5):.6f}",
        "on_axis": on_axis,
        "locked_depth": locked_depth,
        "verdict": verdict.kind,
        "first_nonzero_tail_index": (
            verdict.first_nonzero_tail_index
            if verdict.first_nonzero_tail_index is not None
            else ""
        ),
        "G_polynomial": polynomial_string(verdict.polynomial) if verdict.kind == "finite_looking" else "",
        "elapsed_sec": f"{elapsed:.6f}",
    }


def _completed_combos(out_dir: Path) -> set[tuple[int, int, int, int]]:
    done: set[tuple[int, int, int, int]] = set()
    for shard in sorted(out_dir.glob("shard_*.csv")):
        try:
            with open(shard, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done.add((int(row["pa"]), int(row["qa"]), int(row["pb"]), int(row["qb"])))
        except (OSError, csv.Error, KeyError, ValueError):
            continue
    return done


def _write_summary(out_dir: Path, D: int, t0: float) -> None:
    counts: dict[str, int] = {}
    counts_axis: dict[tuple[str, bool], int] = {}
    off_axis_hits: list[str] = []
    total = 0
    for shard in sorted(out_dir.glob("shard_*.csv")):
        try:
            with open(shard, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    verdict = row["verdict"]
                    on_axis = row["on_axis"] in ("True", "true", "1")
                    counts[verdict] = counts.get(verdict, 0) + 1
                    counts_axis[(verdict, on_axis)] = counts_axis.get((verdict, on_axis), 0) + 1
                    if verdict == "finite_looking" and not on_axis:
                        off_axis_hits.append(
                            f"pa={row['pa']} qa={row['qa']} pb={row['pb']} qb={row['qb']} "
                            f"x={row['x_decimal']} G={row['G_polynomial']}"
                        )
        except (OSError, csv.Error, KeyError):
            continue
    elapsed = time.time() - t0
    throughput = total / elapsed if elapsed > 0 else 0.0
    lines = [
        f"# Negation sweep summary (D = {D})",
        "",
        f"Rows computed so far: {total}",
        f"Elapsed: {elapsed:.1f} s, throughput: {throughput:.2f} rows/sec",
        "",
        "## Counts by verdict",
        "",
    ]
    for verdict, n in sorted(counts.items()):
        lines.append(f"- {verdict}: {n}")
    lines.append("")
    lines.append("## Counts by verdict and axis")
    lines.append("")
    for (verdict, on_axis), n in sorted(counts_axis.items()):
        axis_label = "on-axis" if on_axis else "off-axis"
        lines.append(f"- {verdict}, {axis_label}: {n}")
    lines.append("")
    if off_axis_hits:
        lines.append("## OFF-AXIS FINITE_LOOKING HITS (flagged)")
        lines.append("")
        for hit in off_axis_hits:
            lines.append(f"- OFF-AXIS FINITE_LOOKING: {hit}")
    else:
        lines.append("## Off-axis finite_looking hits: none so far")
    lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def _worker_loop(
    worker_id: int,
    points: list[SweepGridPoint],
    D: int,
    depth: int,
    min_zero_run: int,
    out_dir: Path,
    t0: float,
    summary_every: int = 500,
) -> None:
    shard_path = out_dir / f"shard_{worker_id}.csv"
    write_header = not shard_path.exists() or shard_path.stat().st_size == 0
    count = 0
    with open(shard_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()
        for point in points:
            row = compute_row(point, D, depth, min_zero_run)
            writer.writerow(row)
            f.flush()
            count += 1
            if count % summary_every == 0:
                _write_summary(out_dir, D, t0)
    _write_summary(out_dir, D, t0)


def _chunk(items: list, n: int) -> list[list]:
    n = max(1, n)
    return [items[i::n] for i in range(n)]


def run_sweep(
    D: int,
    qa_max: int,
    qb_max: int,
    pa_max: int,
    pb_max: int,
    depth: int,
    min_zero_run: int,
    out_dir: str | Path,
    workers: int = 1,
    resume: bool = False,
) -> Path:
    """Run the full sweep, writing CSV shards and SUMMARY.md under out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    grid = build_grid(qa_max, qb_max, pa_max, pb_max)
    if resume:
        done = _completed_combos(out_path)
        grid = [p for p in grid if (p.pa, p.qa, p.pb, p.qb) not in done]
    t0 = time.time()
    if workers <= 1:
        _worker_loop(0, grid, D, depth, min_zero_run, out_path, t0)
    else:
        chunks = _chunk(grid, workers)
        procs = [
            Process(
                target=_worker_loop,
                args=(i, chunk, D, depth, min_zero_run, out_path, t0),
            )
            for i, chunk in enumerate(chunks)
            if chunk
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    _write_summary(out_path, D, t0)
    return out_path


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m qreals.negation_sweep",
        description="Sweep G(x) = [x]_q + [-x]_q over a + b*sqrt(D) grids.",
    )
    parser.add_argument("--D", type=int, required=True, help="squarefree D > 1")
    parser.add_argument("--qa-max", type=int, required=True)
    parser.add_argument("--qb-max", type=int, required=True)
    parser.add_argument("--pa-max", type=int, required=True)
    parser.add_argument("--pb-max", type=int, required=True)
    parser.add_argument("--depth", type=int, default=120)
    parser.add_argument("--min-zero-run", type=int, default=30)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    out = run_sweep(
        args.D,
        args.qa_max,
        args.qb_max,
        args.pa_max,
        args.pb_max,
        args.depth,
        args.min_zero_run,
        args.out,
        workers=args.workers,
        resume=args.resume,
    )
    print(f"wrote sweep output to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
