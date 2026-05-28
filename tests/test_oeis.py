"""Tests for qreals.oeis.

The live tests hit the OEIS API and are skipped (not failed) when OEIS is
unreachable or the requests extra is absent, so the suite never errors offline.
The offline tests cover the pure parsing/matching logic, b-file re-verification,
and the two graceful-degradation paths: requests missing, and the network down.
"""

from __future__ import annotations

import socket

import pytest

from qreals import oeis
from qreals.oeis import (
    Hit,
    OeisUnavailable,
    _parse_bfile,
    _verify_hit_bfile,
    available,
    best_transform_match,
    lookup,
    parse_sequence,
)


def _online(host: str = "oeis.org", port: int = 443, timeout: float = 4.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


live = pytest.mark.skipif(
    not (available() and _online()),
    reason="OEIS not reachable or requests extra absent",
)


# --------------------------- live tests --------------------------- #
@live
def test_catalan_hits_a000108():
    res = lookup("1,1,2,5,14,42,132,429", do_modp=False)
    assert res.top is not None
    assert res.top.anum == "A000108"
    assert res.top.prefix_len == 8
    assert res.top.transform == "identity"
    assert res.top.fully_verified is True


@live
def test_random_sequence_returns_no_hit():
    res = lookup("7,0,91,0,0,238,1,99999989", do_modp=False)
    assert res.hits == []
    assert res.top is None


# ----------------------------- offline unit tests -------------------------- #
def test_parse_sequence_tolerates_ellipsis_and_signs():
    assert parse_sequence("1,1,2,4,9,21,...") == [1, 1, 2, 4, 9, 21]
    assert parse_sequence("[1, -1, 1, -2]") == [1, -1, 1, -2]


def test_best_transform_match_reconciles_signs():
    seq = [1, -1, 1, -2, 4]
    candidate = [1, 1, 1, 2, 4, 8, 17]
    match_len, name, offset = best_transform_match(seq, candidate)
    assert match_len == 5
    assert name in {"abs", "alt"}  # both recover the unsigned prefix here
    assert offset == 0


def test_bfile_parse_skips_comments_and_blanks():
    text = "\n# header comment\n0 1\n1 1\n2 2\n3 5\n\n# end\n"
    assert _parse_bfile(text) == [1, 1, 2, 5]


def test_bfile_reverification_surfaces_divergence(monkeypatch):
    monkeypatch.setattr(
        oeis, "_fetch_bfile", lambda *a, **k: [1, 1, 2, 5, 99, 100, 101]
    )
    hit = Hit(anum="A999999", name="synthetic", transform="identity", prefix_len=5)
    _verify_hit_bfile(hit, [1, 1, 2, 5, 14], cache_dir=".", session=None, timeout=1)
    assert hit.bfile_checked is True
    assert hit.diverged is True
    assert hit.fully_verified is False
    assert hit.bfile_match_len == 4
    assert hit.diverge_term == 5
    assert hit.input_value == 14
    assert hit.bfile_value == 99


def test_bfile_reverification_confirms_full_match(monkeypatch):
    monkeypatch.setattr(
        oeis, "_fetch_bfile", lambda *a, **k: [1, 1, 2, 5, 14, 42, 132, 429, 1430]
    )
    hit = Hit(anum="A000108", name="Catalan", transform="identity", prefix_len=8)
    _verify_hit_bfile(
        hit, [1, 1, 2, 5, 14, 42, 132, 429], cache_dir=".", session=None, timeout=1
    )
    assert hit.fully_verified is True
    assert hit.diverged is False
    assert hit.bfile_match_len == 8


def test_lookup_without_requests_raises_a_clear_error(monkeypatch):
    def _no_requests():
        raise OeisUnavailable("OEIS lookup needs the requests dependency")

    monkeypatch.setattr(oeis, "_get_requests", _no_requests)
    with pytest.raises(OeisUnavailable):
        lookup("1,1,2,5,14,42")


def test_lookup_degrades_to_no_hits_when_network_is_down(monkeypatch, tmp_path):
    # requests is importable, but every HTTP call raises (the offline case).
    class _FakeRequests:
        class RequestException(Exception):
            pass

        @staticmethod
        def get(*a, **k):
            raise ConnectionError("network down")

    monkeypatch.setattr(oeis, "_get_requests", lambda: _FakeRequests)
    res = lookup("1,1,2,5,14,42", do_modp=True, cache_dir=tmp_path)
    assert res.hits == []
    assert res.top is None


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError):
        lookup([])
