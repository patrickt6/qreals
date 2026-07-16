"""Bulk OEIS matching over many q-series coefficient sequences.

`qreals.oeis` answers the single-sequence question: is this one coefficient
sequence already catalogued? This module answers the sweep question: given a
whole grid of inputs (every fraction of an atlas run, a list of surds), which
of their q-series are catalogued, how deep does each match go, and at what
offset? Every hit is a potential theorem pointer, so the sweep is built to be
run over large input lists and re-run cheaply.

For each input string ("4/15", "sqrt(2)") the first ``n`` Taylor coefficients
of [x]_q are computed on the verified `q_real_truncated` path, the sequence is
submitted to the OEIS search endpoint through the same request machinery as
`qreals.oeis` (same sign-reconciliation transforms, same alignment-offset
matching), and every hit whose matching prefix reaches ``min_match`` terms is
re-verified against the entry's full b-file exactly the way `oeis.lookup`
does, so a deep divergence ("matched 20 terms, diverged at term 25") is
surfaced rather than silently trusted.

Two properties make the sweep re-runnable:

- caching: every raw OEIS response (search payload and b-file) is stored as a
  JSON file keyed by a hash of the query, in ``cache_dir``. The default cache
  is ``~/.cache/qreals/oeis``; the ``QREALS_CACHE_DIR`` environment variable
  overrides the root. A re-run over the same inputs never hits the network.
- rate limiting: at least ``rate_limit_s`` seconds pass between consecutive
  network calls (cache hits do not wait), so a large sweep is polite to the
  OEIS servers by construction.

The network dependency is ``requests`` behind the ``qreals[oeis]`` extra, the
same as `qreals.oeis`; :class:`qreals.oeis.OeisUnavailable` is raised if it is
missing when a query actually needs the network. The test suite never touches
the network: the two fetch seams ``_search`` and ``_bfile`` are module-level
functions precisely so they can be monkeypatched with fixture payloads.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import oeis as _oeis
from .truncated import q_real_truncated

DEFAULT_N = 24
DEFAULT_MIN_MATCH = 12
DEFAULT_RATE_LIMIT_S = 1.0
DEFAULT_TIMEOUT = 20
MAX_HITS_PER_INPUT = 5

# The sleep seam for the rate limiter, monkeypatchable in tests so timing
# behaviour is asserted without real waiting.
_sleep: Callable[[float], None] = time.sleep


def _default_cache_dir() -> Path:
    """The bulk-sweep cache root: ``~/.cache/qreals/oeis`` by default.

    The ``QREALS_CACHE_DIR`` environment variable overrides the root, matching
    the convention in `qreals.oeis`; the sweep keeps its files under an
    ``oeis`` subdirectory of the override so one variable steers every cache.
    """
    override = os.environ.get("QREALS_CACHE_DIR")
    if override:
        return Path(override) / "oeis"
    return Path.home() / ".cache" / "qreals" / "oeis"


# --------------------------------------------------------------------------- #
# Fetch seams. Both delegate to the request + cache machinery in qreals.oeis
# (same URL, same User-Agent, same on-disk JSON cache format) rather than
# reimplementing it. Tests monkeypatch these two names with fixture data.
# --------------------------------------------------------------------------- #
def _search(query: str, cache_dir: Path, timeout: int) -> list[dict[str, Any]]:
    """Raw OEIS search results for a comma-joined sequence query (cached)."""
    return _oeis._search_raw(query, cache_dir, None, timeout)


def _bfile(anum: str, cache_dir: Path, timeout: int) -> list[int] | None:
    """All b-file terms for an A-number (cached); None when unavailable."""
    return _oeis._fetch_bfile(anum, cache_dir, None, timeout)


def _is_cached(kind: str, ident: str, cache_dir: Path) -> bool:
    """True when the response for this query is already on disk (no wait needed)."""
    return _oeis._read_cache(_oeis._cache_path(cache_dir, kind, ident)) is not None


class _RateLimiter:
    """Enforce a minimum interval between consecutive network calls."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = float(interval_s)
        self._last: float | None = None

    def wait(self) -> None:
        """Sleep just long enough that interval_s has passed since the last call."""
        if self.interval_s <= 0:
            self._last = time.monotonic()
            return
        now = time.monotonic()
        if self._last is not None:
            remaining = self.interval_s - (now - self._last)
            if remaining > 0:
                _sleep(remaining)
        self._last = time.monotonic()


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BulkHit:
    """One OEIS entry matched by one swept input.

    Fields:
        anum: the OEIS A-number, e.g. "A000045".
        name: the entry's name line.
        match_length: leading terms of the (sign-reconciled) input matched
            against the entry's data field.
        offset: alignment offset into the candidate (the entry may carry a few
            extra leading terms before the input begins).
        transform: which sign reconciliation matched ("identity" if none).
        bfile_checked: True when the entry's b-file was fetched and compared.
        bfile_match_length: leading terms matched against the full b-file.
        fully_verified: True when every computed input coefficient agrees with
            the b-file (the whole window is confirmed, not just the prefix).
        diverged: True when the b-file disagrees somewhere inside the window.
        diverge_term: 1-based index of the first disagreement, when diverged.
    """

    anum: str
    name: str
    match_length: int
    offset: int
    transform: str
    bfile_checked: bool = False
    bfile_match_length: int | None = None
    fully_verified: bool = False
    diverged: bool = False
    diverge_term: int | None = None


@dataclass(frozen=True)
class SweepResult:
    """The sweep outcome for one input: its coefficients and ranked hits.

    Fields:
        input: the input string as given ("4/15", "sqrt(2)").
        coeffs: the first n Taylor coefficients of [x]_q that were queried
            (empty when the input failed to parse or compute).
        hits: the OEIS hits with match_length >= min_match, ranked by
            match_length (longest first), each b-file re-verified.
        error: None on success, else a one-line description of why this input
            produced no query (bad input) or why its search failed (network).
    """

    input: str
    coeffs: list[int]
    hits: list[BulkHit]
    error: str | None = None


def _verify_hit(
    hit: BulkHit, coeffs: Sequence[int], terms: list[int] | None
) -> BulkHit:
    """Re-verify one hit against its b-file terms, the way oeis.lookup does.

    The input is reconciled with the hit's sign transform and re-aligned
    against the full b-file; full agreement marks the hit fully_verified,
    anything less marks it diverged with the 1-based first-disagreement index.
    A missing b-file (terms is None) leaves the hit unchecked.
    """
    if terms is None:
        return hit
    reconciled = _oeis._TRANSFORM_BY_NAME[hit.transform](list(coeffs))
    match_len, _off = _oeis._best_alignment(reconciled, terms)
    if match_len >= len(reconciled):
        return BulkHit(
            anum=hit.anum,
            name=hit.name,
            match_length=hit.match_length,
            offset=hit.offset,
            transform=hit.transform,
            bfile_checked=True,
            bfile_match_length=match_len,
            fully_verified=True,
        )
    return BulkHit(
        anum=hit.anum,
        name=hit.name,
        match_length=hit.match_length,
        offset=hit.offset,
        transform=hit.transform,
        bfile_checked=True,
        bfile_match_length=match_len,
        fully_verified=False,
        diverged=True,
        diverge_term=match_len + 1,
    )


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def sweep_oeis(
    inputs: Iterable[str],
    n: int = DEFAULT_N,
    min_match: int = DEFAULT_MIN_MATCH,
    cache_dir: str | Path | None = None,
    rate_limit_s: float = DEFAULT_RATE_LIMIT_S,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_hits: int = MAX_HITS_PER_INPUT,
) -> list[SweepResult]:
    """Match the q-series of every input against the OEIS, with re-verification.

    Args:
        inputs: input strings for `q_real_truncated`, e.g. "4/15", "sqrt(2)".
        n: how many Taylor coefficients of [x]_q to compute and query.
        min_match: a hit must match at least this many leading terms of the
            input (sign-reconciled) to be kept; shorter matches are noise.
        cache_dir: where raw OEIS responses are cached as JSON files keyed by
            a hash of the query. Defaults to ``~/.cache/qreals/oeis`` (created
            if missing); the ``QREALS_CACHE_DIR`` environment variable
            overrides the root.
        rate_limit_s: minimum seconds between consecutive network calls.
            Cache hits never wait, so re-runs are instant.
        timeout: per-request timeout in seconds.
        max_hits: cap on the number of hits kept per input.

    Returns:
        One :class:`SweepResult` per input, in input order. Every kept hit is
        re-verified against the entry's full b-file (fully_verified, or
        diverged with the first disagreeing term), matching `oeis.lookup`.

    An unparseable input becomes a result with an ``error`` and no hits; a
    network failure on one query degrades that input to no hits rather than
    aborting the sweep. Only a missing ``requests`` install raises, since that
    is a setup error, not a data condition.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if min_match < 1:
        raise ValueError("min_match must be at least 1")
    cache = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    limiter = _RateLimiter(rate_limit_s)

    # Identical coefficient sequences (and repeated A-numbers) are fetched
    # once per sweep; the memo sits above the disk cache.
    search_memo: dict[str, list[dict[str, Any]]] = {}
    bfile_memo: dict[str, list[int] | None] = {}

    results: list[SweepResult] = []
    for x in inputs:
        try:
            coeffs = q_real_truncated(x, n)
        except (ValueError, TypeError) as exc:
            results.append(SweepResult(input=x, coeffs=[], hits=[], error=str(exc)))
            continue

        query = ",".join(str(c) for c in coeffs)
        error: str | None = None
        if query in search_memo:
            raw = search_memo[query]
        else:
            if not _is_cached("search", query, cache):
                limiter.wait()
            try:
                raw = _search(query, cache, timeout)
            except _oeis.OeisUnavailable:
                raise
            except Exception as exc:  # network down or bad payload: no hits
                raw = []
                error = f"search failed: {exc}"
            search_memo[query] = raw

        # Rank candidates by sign-reconciled matching-prefix length, keep the
        # ones clearing min_match, best transform per A-number.
        by_anum: dict[str, BulkHit] = {}
        for r in raw:
            candidate = _oeis._parse_data_field(r["data"])
            m, tname, off = _oeis.best_transform_match(coeffs, candidate)
            if m < min_match:
                continue
            anum = _oeis._anum(r["number"])
            current = by_anum.get(anum)
            if current is None or m > current.match_length:
                by_anum[anum] = BulkHit(
                    anum=anum,
                    name=str(r.get("name", "")),
                    match_length=m,
                    offset=off,
                    transform=tname,
                )
        hits = sorted(by_anum.values(), key=lambda h: (-h.match_length, h.anum))
        hits = hits[:max_hits]

        # Every kept hit is re-verified against the full b-file.
        verified: list[BulkHit] = []
        for hit in hits:
            if hit.anum in bfile_memo:
                terms = bfile_memo[hit.anum]
            else:
                if not _is_cached("bfile", hit.anum, cache):
                    limiter.wait()
                try:
                    terms = _bfile(hit.anum, cache, timeout)
                except _oeis.OeisUnavailable:
                    raise
                except Exception:
                    terms = None
                bfile_memo[hit.anum] = terms
            verified.append(_verify_hit(hit, coeffs, terms))

        results.append(
            SweepResult(input=x, coeffs=list(coeffs), hits=verified, error=error)
        )
    return results


def summarize(results: Sequence[SweepResult]) -> str:
    """A plain-text report of a sweep, hits ranked by match length.

    Every (input, hit) pair across the sweep is flattened into one ranked
    table, longest match first (ties by A-number, then input), each line
    carrying the verification verdict: "verified" when the whole coefficient
    window agrees with the b-file, "DIVERGED at term k" when it does not, and
    "unverified" when no b-file was available. Inputs with no hits and inputs
    that errored are listed at the end so a sweep is auditable at a glance.
    """
    flat: list[tuple[str, BulkHit]] = []
    no_hits: list[str] = []
    errored: list[tuple[str, str]] = []
    for res in results:
        if res.error is not None:
            errored.append((res.input, res.error))
        if res.hits:
            for hit in res.hits:
                flat.append((res.input, hit))
        elif res.error is None:
            no_hits.append(res.input)
    flat.sort(key=lambda pair: (-pair[1].match_length, pair[1].anum, pair[0]))

    lines: list[str] = []
    n_with_hits = len({inp for inp, _ in flat})
    lines.append(
        f"OEIS sweep: {len(results)} input(s), {n_with_hits} with hits, "
        f"{len(flat)} hit(s) total"
    )
    for rank, (inp, hit) in enumerate(flat, start=1):
        if hit.fully_verified:
            verdict = f"verified against b-file ({hit.bfile_match_length} terms)"
        elif hit.diverged:
            verdict = f"DIVERGED at term {hit.diverge_term}"
        else:
            verdict = "unverified (no b-file)"
        note = "" if hit.transform == "identity" else f" [{hit.transform}]"
        lines.append(
            f"{rank:3d}. {inp}: {hit.anum} match {hit.match_length} "
            f"offset {hit.offset}{note}, {verdict}"
        )
        if hit.name:
            lines.append(f"     {hit.name}")
    if no_hits:
        lines.append("no hits: " + ", ".join(no_hits))
    for inp, msg in errored:
        lines.append(f"error for {inp}: {msg}")
    return "\n".join(lines)
