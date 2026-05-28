"""Look a coefficient sequence up in the OEIS, with re-verification.

A q-series produces an integer coefficient sequence; the obvious next question
is whether that sequence is already catalogued. This module submits a sequence
to the OEIS public API (``oeis.org/search``), ranks the hits by matching-prefix
length (reconciling a handful of sign conventions, since a q-series can arrive
negated or alternating relative to the canonical unsigned entry), re-verifies
each top hit against its full b-file so a deep divergence such as "matched the
first 20 terms but diverged at term 25" is surfaced, and also tries mod-p
reductions of the input. Every HTTP response is cached on disk so re-running the
same sequence does not hit the network again.

This is an optional helper. The only network dependency is ``requests``, behind
the ``qreals[oeis]`` extra; the rest of qreals does not import it. Two failure
modes are handled without crashing:

- ``requests`` not installed: ``lookup`` raises :class:`OeisUnavailable`, a clear
  error pointing at ``pip install qreals[oeis]``. Test with :func:`available`.
- the network is down or OEIS is unreachable: each query fails quietly and the
  lookup returns an empty :class:`LookupResult` (no hits) rather than raising.

Adapted from the standalone ``qoeis`` tool. The math context (q-Taylor and
q-continued-fraction coefficient sequences) is the same one the rest of qreals
computes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

OEIS_SEARCH_URL = "https://oeis.org/search"
OEIS_BASE = "https://oeis.org"
USER_AGENT = "qreals/0.1 (OEIS lookup helper for q-number research; +https://oeis.org)"
DEFAULT_PRIMES = (2, 3, 5, 7, 11)
DEFAULT_TIMEOUT = 20
OFFSET_LIMIT = 8  # how many leading terms a candidate may have before our input begins
MIN_MODP_PREFIX = (
    4  # mod-p "hits" matching fewer leading terms than this are dropped as noise
)


class OeisUnavailable(RuntimeError):
    """Raised when an OEIS lookup is requested but ``requests`` is not installed."""


def available() -> bool:
    """True when the ``requests`` dependency for OEIS lookups is importable."""
    return importlib.util.find_spec("requests") is not None


def _get_requests() -> Any:
    """Import ``requests`` lazily, with a clear error if the extra is missing."""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise OeisUnavailable(
            "OEIS lookup needs the requests dependency; install it with "
            "pip install qreals[oeis]"
        ) from exc
    return requests


def _default_cache_dir() -> Path:
    """A per-user cache directory, kept out of the installed package tree.

    Honours ``QREALS_CACHE_DIR`` if set, then the platform convention
    (``%LOCALAPPDATA%`` on Windows, ``$XDG_CACHE_HOME`` or ``~/.cache`` elsewhere).
    """
    override = os.environ.get("QREALS_CACHE_DIR")
    if override:
        return Path(override) / "oeis"
    if os.name == "nt":
        nt_root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(nt_root) / "qreals" / "oeis-cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "qreals" / "oeis-cache"


DEFAULT_CACHE_DIR = _default_cache_dir()


# --------------------------------------------------------------------------- #
# Sign-reconciliation transforms.
#
# A q-series coefficient list may arrive negated, alternating, or in absolute
# value relative to the canonical OEIS entry. Each transform is applied to *our*
# input before comparing against a candidate. They are ordered by preference, so
# a plain identity match always wins ties.
# --------------------------------------------------------------------------- #
def _t_identity(a: Sequence[int]) -> list[int]:
    return list(a)


def _t_abs(a: Sequence[int]) -> list[int]:
    return [abs(x) for x in a]


def _t_alt(a: Sequence[int]) -> list[int]:
    """Multiply by (-1)**i: undoes a sign that alternates starting positive."""
    return [x if i % 2 == 0 else -x for i, x in enumerate(a)]


def _t_alt_shift(a: Sequence[int]) -> list[int]:
    """Multiply by (-1)**(i+1): undoes a sign that alternates starting negative."""
    return [-x if i % 2 == 0 else x for i, x in enumerate(a)]


def _t_neg(a: Sequence[int]) -> list[int]:
    return [-x for x in a]


TRANSFORMS: list[tuple[str, Callable[[Sequence[int]], list[int]]]] = [
    ("identity", _t_identity),
    ("abs", _t_abs),
    ("alt", _t_alt),
    ("alt-shift", _t_alt_shift),
    ("neg", _t_neg),
]
_TRANSFORM_BY_NAME = dict(TRANSFORMS)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def _prefix_match_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Number of leading positions where ``a`` and ``b`` agree."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _best_alignment(
    transformed: Sequence[int], candidate: Sequence[int]
) -> tuple[int, int]:
    """Longest leading agreement of ``transformed`` against ``candidate``.

    Returns ``(match_len, offset)`` where ``candidate[offset:]`` is the block that
    lines up with ``transformed[0:]``. A small offset is allowed so a candidate
    with extra leading terms (e.g. ``0, 0, 1, 1, ...``) still matches.
    """
    best_len, best_off = 0, 0
    limit = min(len(candidate), OFFSET_LIMIT)
    for j in range(max(1, limit)):
        m = _prefix_match_len(transformed, candidate[j:])
        if m > best_len:
            best_len, best_off = m, j
            if best_len == len(transformed):
                break
    return best_len, best_off


def best_transform_match(
    input_seq: Sequence[int], candidate: Sequence[int]
) -> tuple[int, str, int]:
    """Find the sign transform giving the longest leading match against ``candidate``.

    Returns ``(match_len, transform_name, offset)``. Identity is preferred on ties.
    """
    best = (0, "identity", 0)
    for name, fn in TRANSFORMS:
        m, off = _best_alignment(fn(input_seq), candidate)
        if m > best[0]:
            best = (m, name, off)
            if m == len(input_seq) and name == "identity":
                break  # a full identity match cannot be beaten
    return best


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_sequence(text: str | Sequence[int] | None) -> list[int]:
    """Parse a human-typed string of integers into a list of ints.

    Tolerates commas, surrounding brackets, whitespace, and trailing ellipses,
    e.g. ``"1, 1, 2, 4, 9, 21, ..."`` -> ``[1, 1, 2, 4, 9, 21]``.
    """
    if text is None:
        raise ValueError("no sequence given")
    if not isinstance(text, str):
        return [int(x) for x in text]
    tokens = re.findall(r"-?\d+", text)
    if not tokens:
        raise ValueError(f"no integers found in {text!r}")
    return [int(t) for t in tokens]


def _coerce_seq(sequence: str | Sequence[int]) -> list[int]:
    return (
        parse_sequence(sequence)
        if isinstance(sequence, str)
        else [int(x) for x in sequence]
    )


def _join(seq: Sequence[int]) -> str:
    return ",".join(str(x) for x in seq)


def _anum(number: int) -> str:
    return f"A{int(number):06d}"


def _parse_data_field(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_bfile(text: str) -> list[int]:
    """Parse a b-file's text into its list of values (in index order)."""
    terms: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            terms.append(int(parts[-1]))  # lines are "index value"; the value is last
        except ValueError:
            continue
    return terms


# --------------------------------------------------------------------------- #
# Disk cache + HTTP
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: Path, kind: str, ident: str) -> Path:
    digest = hashlib.sha256(ident.encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"{kind}_{digest}.json"


def _read_cache(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_cache(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
        os.replace(tmp, path)  # atomic; safe under concurrent writers
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _extract_results(payload: Any) -> list[dict[str, Any]]:
    """The OEIS json endpoint returns a bare list on hits and ``null`` on a miss;
    older deployments wrap the list in ``{"results": [...]}``. Handle all three."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("results") or []
    return []


def _search_raw(
    query: str, cache_dir: Path, session: Any, timeout: int
) -> list[dict[str, Any]]:
    """Return the raw list of OEIS result dicts for a query string (cached)."""
    cpath = _cache_path(cache_dir, "search", query)
    cached = _read_cache(cpath)
    if cached is not None:
        results_cached: list[dict[str, Any]] = cached.get("results", [])
        return results_cached
    getter = session.get if session is not None else _get_requests().get
    resp = getter(
        OEIS_SEARCH_URL,
        params={"q": query, "fmt": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    results = _extract_results(resp.json())
    _write_cache(cpath, {"query": query, "results": results})
    return results


def _fetch_bfile(
    anum: str, cache_dir: Path, session: Any, timeout: int
) -> Optional[list[int]]:
    """Return all b-file terms for ``anum`` (e.g. ``"A000108"``), cached. None on failure."""
    cpath = _cache_path(cache_dir, "bfile", anum)
    cached = _read_cache(cpath)
    if cached is not None:
        cached_terms: list[int] | None = cached.get("terms")
        return cached_terms
    requests = _get_requests()
    url = f"{OEIS_BASE}/{anum}/b{anum[1:]}.txt"
    getter = session.get if session is not None else requests.get
    try:
        resp = getter(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
        terms = _parse_bfile(resp.text)
    except (requests.RequestException, ValueError):
        return None
    _write_cache(cpath, {"anum": anum, "terms": terms})
    return terms


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    anum: str
    name: str
    transform: str  # which sign reconciliation matched ("identity" if none needed)
    prefix_len: int  # leading terms matched against the OEIS 'data' field
    offset: int = 0  # alignment offset into the candidate
    source: str = "raw"  # "raw" or "mod-2", "mod-3", ...
    data_terms: list[int] = field(default_factory=list)
    # b-file re-verification (filled in for the top hits only)
    bfile_checked: bool = False
    bfile_len: Optional[int] = None
    bfile_match_len: Optional[int] = None
    fully_verified: bool = False
    diverged: bool = False
    diverge_term: Optional[int] = None  # 1-based term index of the first disagreement
    input_value: Optional[int] = None  # reconciled input value at the divergence
    bfile_value: Optional[int] = None  # b-file value at the divergence


@dataclass
class LookupResult:
    input_seq: list[int]
    hits: list[Hit]
    modp_hits: dict[int, list[Hit]]
    primes: tuple[int, ...] = DEFAULT_PRIMES

    @property
    def top(self) -> Optional[Hit]:
        return self.hits[0] if self.hits else None


def _verify_hit_bfile(
    hit: Hit, input_seq: Sequence[int], cache_dir: Path, session: Any, timeout: int
) -> Hit:
    terms = _fetch_bfile(hit.anum, cache_dir, session, timeout)
    if terms is None:
        return hit
    reconciled = _TRANSFORM_BY_NAME[hit.transform](input_seq)
    match_len, off = _best_alignment(reconciled, terms)
    hit.bfile_checked = True
    hit.bfile_len = len(terms)
    hit.bfile_match_len = match_len
    if match_len >= len(reconciled):
        hit.fully_verified = True
    else:
        hit.diverged = True
        hit.diverge_term = match_len + 1
        hit.input_value = reconciled[match_len]
        idx = off + match_len
        hit.bfile_value = terms[idx] if idx < len(terms) else None
    return hit


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def lookup(
    sequence: str | Sequence[int],
    *,
    primes: Sequence[int] = DEFAULT_PRIMES,
    max_hits: int = 10,
    bfile_top: int = 3,
    do_modp: bool = True,
    do_bfile: bool = True,
    cache_dir: str | Path | None = None,
    session: Any = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_workers: int = 6,
) -> LookupResult:
    """Look up ``sequence`` in the OEIS.

    ``sequence`` may be a string (``"1,1,2,5,..."``) or any iterable of ints.

    Returns a :class:`LookupResult` whose ``hits`` are ranked by matching-prefix
    length (sign-reconciled), with the top ``bfile_top`` hits re-verified against
    their full b-files, plus ``modp_hits`` for the mod-p reductions of the input.

    Needs the ``requests`` dependency (``pip install qreals[oeis]``); raises
    :class:`OeisUnavailable` if it is missing. If OEIS is unreachable, every
    query fails quietly and the result carries no hits.
    """
    seq = _coerce_seq(sequence)
    if not seq:
        raise ValueError("empty sequence")
    _get_requests()  # fail fast and clearly if the extra is not installed
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    # Assemble every query up front so they can run concurrently.
    queries: dict[str, str] = {_join(seq): "raw"}
    if any(x < 0 for x in seq):
        queries.setdefault(_join([abs(x) for x in seq]), "abs-search")
    modp_inputs: dict[int, list[int]] = {}
    if do_modp:
        for p in primes:
            reduced = [x % p for x in seq]
            modp_inputs[p] = reduced
            queries.setdefault(_join(reduced), f"mod-{p}")

    results_by_query: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_search_raw, q, cache_dir, session, timeout): q for q in queries
        }
        for fut, q in futures.items():
            try:
                results_by_query[q] = fut.result()
            except Exception:  # network down, timeout, bad payload: degrade to no hits
                results_by_query[q] = []

    # Primary hits come from the raw (and absolute-value) searches.
    hits_by_anum: dict[str, Hit] = {}
    for q, src in queries.items():
        if src not in ("raw", "abs-search"):
            continue
        for r in results_by_query.get(q, []):
            candidate = _parse_data_field(r["data"])
            m, tname, off = best_transform_match(seq, candidate)
            if m == 0:
                continue
            anum = _anum(r["number"])
            current = hits_by_anum.get(anum)
            if current is None or m > current.prefix_len:
                hits_by_anum[anum] = Hit(
                    anum=anum,
                    name=r.get("name", ""),
                    transform=tname,
                    prefix_len=m,
                    offset=off,
                    source="raw",
                    data_terms=candidate,
                )
    hits = sorted(hits_by_anum.values(), key=lambda h: (-h.prefix_len, h.anum))[
        :max_hits
    ]

    if do_bfile and hits:
        top = hits[:bfile_top]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(
                pool.map(
                    lambda h: _verify_hit_bfile(h, seq, cache_dir, session, timeout),
                    top,
                )
            )

    # Mod-p reduced hits (no sign reconciliation; the reduction is already canonical).
    modp_hits: dict[int, list[Hit]] = {}
    if do_modp:
        for p, reduced in modp_inputs.items():
            collected: list[Hit] = []
            for r in results_by_query.get(_join(reduced), []):
                candidate = _parse_data_field(r["data"])
                m, off = _best_alignment(reduced, candidate)
                if m >= MIN_MODP_PREFIX:
                    collected.append(
                        Hit(
                            anum=_anum(r["number"]),
                            name=r.get("name", ""),
                            transform="identity",
                            prefix_len=m,
                            offset=off,
                            source=f"mod-{p}",
                            data_terms=candidate,
                        )
                    )
            collected.sort(key=lambda h: (-h.prefix_len, h.anum))
            if collected:
                modp_hits[p] = collected[:3]

    return LookupResult(
        input_seq=seq, hits=hits, modp_hits=modp_hits, primes=tuple(primes)
    )


def hit_as_dict(hit: Hit) -> dict[str, Any]:
    """A flat, JSON-serialisable view of a :class:`Hit`."""
    from dataclasses import asdict

    return asdict(hit)


def result_as_dict(res: LookupResult) -> dict[str, Any]:
    """A flat, JSON-serialisable view of a :class:`LookupResult`."""
    return {
        "input": res.input_seq,
        "hits": [hit_as_dict(h) for h in res.hits],
        "modp_hits": {
            p: [hit_as_dict(h) for h in hs] for p, hs in res.modp_hits.items()
        },
    }
