"""Hard-coded bibliographic registry for qreals certificates.

A certificate cites a small, fixed set of published sources. Every in-text
citation routes through this registry so it renders as a working hyperlink: an
``\\href`` in LaTeX, a numbered, URL-carrying entry in the terminal, and an
``<a>`` in the HTML view. The set is hard-coded on purpose; nothing here fetches
at runtime.

Each URL was checked to resolve at the time of writing. The companion note
``docs/REFERENCES.md`` records the same URLs so a future broken link is caught;
``tests/test_refs.py`` keeps the note and this registry in step.

This module has no dependencies beyond the standard library, so importing it
costs the core nothing; only the certificate layer (the optional extra
``qreals[proof]``) uses it.

Citations are written in prose as a placeholder ``[[cite:KEY|label]]`` where KEY
is a registry key (lowercase, hyphenated, so it survives LaTeX escaping) and
label is the text shown to the reader. ``render`` rewrites the placeholder for a
given output format; ``used_keys`` reports which keys a body cites, in registry
order.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

# A citation placeholder: [[cite:key|label]]. Keys are lowercase and hyphenated
# (no underscore, no brace) so they pass through the LaTeX escaper untouched and
# can be rewritten to an \href after escaping.
_CITE = re.compile(r"\[\[cite:([a-z0-9-]+)\|([^\]]+)\]\]")


@dataclass(frozen=True)
class Reference:
    """One bibliographic entry and the URLs that locate it.

    ``url`` is the canonical landing page (the arXiv abstract page, which carries
    both the paper and links to the published version). ``arxiv`` and ``doi`` are
    the identifiers; ``doi_url`` builds the resolver link.
    """

    key: str
    citation: str  # full human-readable reference, no URL
    url: str  # canonical landing URL (arXiv abstract page)
    arxiv: str | None = None  # arXiv identifier, e.g. "1812.00170"
    doi: str | None = None  # DOI, e.g. "10.1017/fms.2020.9"

    @property
    def doi_url(self) -> str | None:
        return f"https://doi.org/{self.doi}" if self.doi else None

    def full(self) -> str:
        """The reference as one plain-text line, identifiers appended."""
        tail: list[str] = []
        if self.arxiv:
            tail.append(f"arXiv:{self.arxiv}")
        if self.doi:
            tail.append(f"doi:{self.doi}")
        suffix = (" " + ", ".join(tail) + ".") if tail else ""
        return self.citation + "." + suffix


# The whole set a certificate can cite. Insertion order is the order Sources are
# numbered and listed. URLs checked to resolve on 2026-05-24 (see
# docs/REFERENCES.md).
REFERENCES: dict[str, Reference] = {
    "mgo-rat": Reference(
        key="mgo-rat",
        citation=(
            "S. Morier-Genoud and V. Ovsienko, q-deformed rationals and "
            "q-continued fractions, Forum Math. Sigma 8 (2020), e13"
        ),
        url="https://arxiv.org/abs/1812.00170",
        arxiv="1812.00170",
        doi="10.1017/fms.2020.9",
    ),
    "mgo-real": Reference(
        key="mgo-real",
        citation=(
            "S. Morier-Genoud and V. Ovsienko, On q-deformed real numbers, "
            "Exp. Math. 31 (2022), no. 2, 652-660"
        ),
        url="https://arxiv.org/abs/1908.04365",
        arxiv="1908.04365",
        doi="10.1080/10586458.2019.1671922",
    ),
    "mgo-survey": Reference(
        key="mgo-survey",
        citation=(
            "S. Morier-Genoud and V. Ovsienko, q-deformed rationals and "
            "irrationals (a survey for the Mathematical Omnibus, 2nd ed.)"
        ),
        url="https://arxiv.org/abs/2503.23834",
        arxiv="2503.23834",
    ),
}


def reference(key: str) -> Reference:
    """The registry entry for `key`, or a clear error naming the valid keys."""
    try:
        return REFERENCES[key]
    except KeyError:
        valid = ", ".join(REFERENCES)
        raise KeyError(
            f"unknown citation key {key!r}; a certificate citation must use a key "
            f"in qreals.refs.REFERENCES ({valid})"
        ) from None


def used_keys(body: str) -> list[str]:
    """Citation keys cited in `body`, unique, in registry (Sources) order.

    Each key is validated against the registry, so a typo in a placeholder is
    caught here rather than rendered as a dead link.
    """
    found = set()
    for match in _CITE.finditer(body):
        key = match.group(1)
        reference(key)  # validate; raises on an unknown key
        found.add(key)
    return [key for key in REFERENCES if key in found]


def numbering(used: list[str]) -> dict[str, int]:
    """Map each used key to its 1-based Sources number, in registry order."""
    ordered = [key for key in REFERENCES if key in set(used)]
    return {key: i + 1 for i, key in enumerate(ordered)}


def render(body: str, fmt: str, numbers: dict[str, int] | None = None) -> str:
    """Rewrite every ``[[cite:key|label]]`` in `body` for output format `fmt`.

    ``fmt`` is "tex" (an ``\\href`` to the URL), "html" (an ``<a>``), or "text"
    (the label with a ``[n]`` marker that points to the Sources list, where
    `numbers` gives n). An unknown key raises through ``reference``.
    """

    def repl(match: re.Match[str]) -> str:
        key, label = match.group(1), match.group(2)
        ref = reference(key)
        if fmt == "tex":
            return r"\href{" + ref.url + "}{" + label + "}"
        if fmt == "html":
            return f'<a href="{html.escape(ref.url, quote=True)}">{html.escape(label)}</a>'
        if fmt == "text":
            if numbers is None:
                return label
            return f"{label} [{numbers[key]}]"
        raise ValueError(f"unknown render format {fmt!r}")

    return _CITE.sub(repl, body)
