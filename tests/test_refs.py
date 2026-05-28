"""The certificate reference registry and the hyperlinks it drives.

These tests pin three things the goal needs to stay true: every citation a
certificate emits is a key in the hard-coded registry, each citation renders as
a working hyperlink in every view (LaTeX, HTML, terminal), and the checked-in
URL note `docs/REFERENCES.md` names the same URLs as the registry, so a future
broken link cannot drift in unnoticed.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from qreals import certificate, refs

# Build one certificate of every kind `build_certificate` handles. Citations are
# kind-dependent (a series cites MGO Proposition 1.1, a rational does not), so
# the set of inputs has to cover every branch.
_RESULTS = [
    {"kind": "rational", "data": {"p": 3, "s": 2}},
    {"kind": "qint", "data": {"n": 5}},
    {"kind": "coeffs", "data": {"x": "pi", "n": 12}},
    {"kind": "coeffs", "data": {"x": "sqrt(2)", "n": 12}},
    {"kind": "rational", "data": {"p": 355, "s": 113}},
]


def _certs():
    return [certificate.build_certificate(r) for r in _RESULTS]


# --- the registry itself --------------------------------------------------


def test_registry_entries_are_well_formed():
    for key, ref in refs.REFERENCES.items():
        assert ref.key == key
        assert ref.url.startswith("https://"), key
        assert ref.citation and "http" not in ref.citation, key
        if ref.doi:
            assert ref.doi_url == f"https://doi.org/{ref.doi}"


def test_unknown_citation_key_raises_with_a_helpful_message():
    with pytest.raises(KeyError) as exc:
        refs.reference("not-a-real-key")
    assert "qreals.refs.REFERENCES" in str(exc.value)
    with pytest.raises(KeyError):
        refs.render("[[cite:not-a-real-key|x]]", "tex")


def test_at_least_the_three_named_sources_are_registered():
    # The goal names these explicitly; confirm they are present with their ids.
    assert refs.REFERENCES["mgo-rat"].arxiv == "1812.00170"
    assert refs.REFERENCES["mgo-rat"].doi == "10.1017/fms.2020.9"
    assert refs.REFERENCES["mgo-survey"].arxiv == "2503.23834"


# --- every citation used in certificate generation is registered ----------


def test_every_cite_placeholder_in_the_module_is_a_registry_key():
    # Static sweep: every [[cite:KEY|...]] literal in certificate.py must name a
    # registered key. This catches a typo even on a branch no test input hits.
    source = inspect.getsource(certificate)
    keys = re.findall(r"\[\[cite:([a-z0-9-]+)\|", source)
    assert keys, "expected the certificate module to carry cite placeholders"
    for key in keys:
        assert key in refs.REFERENCES, key


@pytest.mark.parametrize("cert", _certs(), ids=lambda c: c.slug)
def test_rendered_views_resolve_every_citation_to_a_registered_source(cert):
    # Dynamic sweep: rendering raises on an unknown key, so a clean render of all
    # three views over every kind proves the keys are all registered. Also assert
    # no placeholder leaked through unrendered.
    for view in (cert.text(), cert.to_tex(), cert.to_html()):
        assert "[[cite:" not in view
    for view in (cert.text(), cert.to_tex(), cert.to_html()):
        for key in cert.citation_keys():
            assert key in refs.REFERENCES


# --- the hyperlinks render in each format ---------------------------------


def test_mgo_proposition_1_1_renders_as_a_hyperlink_in_tex_and_html():
    # The goal's worked example: MGO Proposition 1.1 must be a link to the paper.
    cert = certificate.build_certificate({"kind": "coeffs", "data": {"x": "pi", "n": 12}})
    real = refs.REFERENCES["mgo-real"]
    assert r"\href{" + real.url + "}{MGO Proposition 1.1}" in cert.to_tex()
    assert f'<a href="{real.url}">MGO Proposition 1.1</a>' in cert.to_html()


@pytest.mark.parametrize("cert", _certs(), ids=lambda c: c.slug)
def test_tex_has_an_href_for_every_used_source(cert):
    tex = cert.to_tex()
    assert r"\usepackage{hyperref}" in tex
    assert r"\section*{Sources}" in tex
    for key in cert.citation_keys():
        assert r"\href{" + refs.REFERENCES[key].url + "}" in tex


@pytest.mark.parametrize("cert", _certs(), ids=lambda c: c.slug)
def test_html_has_an_anchor_for_every_used_source(cert):
    page = cert.to_html()
    assert "<h2>Sources</h2>" in page
    for key in cert.citation_keys():
        assert f'href="{refs.REFERENCES[key].url}"' in page


@pytest.mark.parametrize("cert", _certs(), ids=lambda c: c.slug)
def test_terminal_view_ends_with_a_numbered_sources_list_carrying_urls(cert):
    text = cert.text()
    assert "\nSources\n" in text
    used = cert.citation_keys()
    assert used, "every certificate cites at least one source"
    numbers = refs.numbering(used)
    for key in used:
        ref = refs.REFERENCES[key]
        # the inline marker and the matching numbered Sources entry and its URL
        assert f"[{numbers[key]}]" in text
        assert ref.url in text


# --- the checked-in note stays in step with the registry ------------------


def _references_note() -> str:
    note = Path(refs.__file__).resolve().parents[2] / "docs" / "REFERENCES.md"
    return note.read_text(encoding="utf-8")


def test_references_note_lists_exactly_the_registry_urls():
    note = _references_note()
    for ref in refs.REFERENCES.values():
        assert ref.url in note, f"{ref.key}: arXiv URL missing from REFERENCES.md"
        if ref.doi_url:
            assert ref.doi_url in note, f"{ref.key}: DOI URL missing from REFERENCES.md"
    # and the note carries no stray https URL that the registry does not know
    known = set()
    for ref in refs.REFERENCES.values():
        known.add(ref.url)
        if ref.doi_url:
            known.add(ref.doi_url)
    for url in re.findall(r"https://\S+", note):
        url = url.rstrip(".,)")
        # only the full reference URLs (a DOI with a path, an arXiv abstract),
        # not a bare host the prose mentions when describing the resolve check
        if "doi.org/" in url or "arxiv.org/abs/" in url:
            assert url in known, f"REFERENCES.md names an unregistered URL: {url}"
