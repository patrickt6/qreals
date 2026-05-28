"""Human-auditable certificates for a qreals computation.

A certificate is a derivation a reader can check by hand, not a formal machine
proof. For one input it shows:

(a) the continued fraction of x and its even-length MGO form;
(b) the MGO continued-fraction formula folded step by step down to the result;
(c) the independent cross-checks that hold for that case (the same checks the
    inline stamp runs, shown in full here).

Every source it cites comes from the hard-coded registry in ``qreals.refs``: an
in-text citation (for example MGO Proposition 1.1) renders as a hyperlink to the
paper, and each certificate ends with a hyperlinked Sources list. It also points
the reader at docs/CORRECTNESS.md, where every public function is mapped to its
theorem.

This module is the interface layer behind the optional extra ``qreals[proof]``.
It imports the core (never the other way round) and degrades cleanly: the
terminal view and saving a .tex need no TeX engine; only the PDF paths require
one, found on PATH as pdflatex, then tectonic, then latexmk.

Three rendering paths, with one rule: only SAVE writes a file.

- ``render_terminal`` prints the derivation and checks as formatted text.
- ``view_pdf`` compiles a temporary PDF, opens it in the system viewer, and
  deletes the temp file, leaving nothing behind.
- ``save`` writes a .tex (plus a .pdf when a TeX engine is found) into a chosen
  directory, for keeping or attaching to a paper, and can optionally record the
  run in the user's .qprov store if qprov is importable.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from . import refs
from .continued_fraction import cf_partials, make_even_length
from .expansions import format_laurent
from .rational import q, q_int, q_int_qinv, q_rational
from .truncated import q_real_truncated
from .verify import Stamp, verify

_TEX_ENGINES = ("pdflatex", "tectonic", "latexmk")


def find_tex_engine() -> str | None:
    """The first of pdflatex, tectonic, latexmk found on PATH, or None.

    The order is the same on every operating system; no path is hard-coded.
    """
    for engine in _TEX_ENGINES:
        if shutil.which(engine):
            return engine
    return None


# --------------------------------------------------------------------------
# Building the derivation.
# --------------------------------------------------------------------------


@dataclass
class FoldRow:
    """One row of the referee per-fold table: a partial as a named ratio.

    ``pos`` is the 1-based MGO position i, ``a`` the partial quotient a_i,
    ``ratio`` the partial written as the subscripted ratio N_i / D_i (or the
    closed form for the innermost term), and ``degree`` the degree of N_i after
    reducing R_i to lowest terms. The table carries names and degrees, never the
    full polynomials, so no monomial run is elided.
    """

    pos: int
    a: int
    ratio: str
    degree: int


@dataclass
class RefereeView:
    """The half-page proof a mathematician reads: recursion, structure, fold
    table, then the deliverable tied back to a value the reader can check.

    The full intermediate-polynomial trace is not here; it stays in the
    certificate appendix (the (a) and (b) sections) for byte-by-byte re-checking.
    """

    recursion: list[str]  # the MGO fold stated once, with its citation
    structure: list[str]  # the structural facts in words (D_i = N_{i+1}, etc.)
    folds: list[FoldRow]  # the per-fold table (empty for a q-integer)
    deliverable_lines: list[str]  # the result, the closing line, with full vector
    compare_title: str  # heading for the tie-back table
    compare_columns: list[str]
    compare_rows: list[list[str]]
    compare_summary: str  # the match verdict
    headline: str = ""  # one escape-safe result sentence for the typeset views
    coeff_row: list[int] | None = None  # the full coefficient row, never elided
    witness: str | None = None  # the numeric witness at q_0 = 1/2


@dataclass
class Certificate:
    """A built derivation, renderable to the terminal, a temp PDF, or a file."""

    title: str
    input_line: str
    cf: list[int]
    even_cf: list[int]
    steps: list[tuple[str, sp.Expr]]
    result_label: str
    result_expr: sp.Expr | None
    coeffs: list[int] | None
    convergent: sp.Rational | None
    stamp: Stamp
    slug: str
    referee: RefereeView
    qprov_inputs: dict[str, Any] = field(default_factory=dict)
    qprov_outputs: dict[str, Any] = field(default_factory=dict)
    # PDF rendering knobs. They touch only the LaTeX paths; the terminal text,
    # JSON, and coefficient read-outs always carry the full exact object.
    max_terms: int = 20
    coeff_table_threshold: int = 30
    wrap: str = "breqn"

    def citation_keys(self) -> list[str]:
        """The registry keys this certificate cites, in Sources (registry) order.

        Every certificate cites mgo-rat (the section (b) construction note) and
        mgo-survey (the Sources overview); the [x]_q series certificates also
        cite mgo-real, whose Proposition 1.1 is the stabilisation bound named in
        the input line. The scan reads those exact placeholders, so it matches
        what every rendered view emits.
        """
        seed = self.input_line + " [[cite:mgo-rat|x]] [[cite:mgo-survey|x]]"
        return refs.used_keys(seed)

    # -- terminal ----------------------------------------------------------

    def text(self) -> str:
        """The derivation as plain text: the referee view, then a full appendix.

        The referee view comes first (recursion, structure, per-fold table, the
        deliverable with its tie-back table). The full intermediate-polynomial
        trace is demoted to the appendix at the foot, so nothing is lost for
        byte-by-byte re-checking.
        """
        r = self.referee
        body: list[str] = []
        body.append(f"= {self.title} =")
        body.append("")
        body.append(self.input_line)
        body.append("")
        body.append("Referee view")
        body.append("")
        if r.recursion:
            body.append("The recursion.")
            body.extend(f"    {line}" for line in r.recursion)
            body.append("")
        if r.structure:
            body.append("Structure.")
            body.extend(f"    {line}" for line in r.structure)
            body.append("")
        if r.folds:
            body.append("The fold, inside out (partials as ratios; degrees after reducing):")
            body.append(_text_table(
                ["step i", "quotient a_i", "partial R_i", "deg N_i"],
                [[str(f.pos), str(f.a), f.ratio, str(f.degree)] for f in r.folds],
            ))
            body.append("")
        body.extend(r.deliverable_lines)
        body.append("")
        body.append(r.compare_title)
        body.append(_text_table(r.compare_columns, r.compare_rows))
        body.append(f"    {r.compare_summary}")
        if r.witness:
            body.append(f"    {r.witness}")
        body.append("")
        body.append("(c) independent cross-checks")
        for check in self.stamp.checks:
            mark = {
                "pass": "[pass]",
                "fail": "[FAIL]",
                "na": "[n/a] ",
                "error": "[err] ",
            }.get(check.status, "[?]")
            body.append(f"    {mark} {check.detail}")
        body.append(f"    summary: {self.stamp.line()}")
        body.append("")
        body.append("Sources")
        body.append(
            "    Overview: [[cite:mgo-survey|q-deformed rationals and irrationals]]."
        )
        body.append("    Each numbered source below opens at the cited paper.")

        # The appendix carries the full machine trace, rendered after the Sources
        # list so the referee view and its references read first.
        appendix: list[str] = []
        appendix.append("Appendix: full intermediate-polynomial trace")
        appendix.append("")
        appendix.append("(a) continued fraction and even-length MGO form")
        appendix.append(f"    regular continued fraction: {self.cf}")
        appendix.append(f"    even-length MGO form:       {self.even_cf}")
        if self.convergent is not None:
            appendix.append(
                f"    even-length form evaluates to the convergent {self.convergent}"
            )
        appendix.append("")
        appendix.append("(b) MGO formula folded step by step")
        appendix.append("    odd positions carry [a]_q with q^a above; even positions")
        appendix.append("    carry [a]_(q^-1) with q^-a above ([[cite:mgo-rat|MGO eqn 1.1]]).")
        for label, expr in self.steps:
            appendix.append(f"    {label}:  {sp.sstr(sp.cancel(expr))}")
        appendix.append(f"    {self.result_label} = {sp.sstr(self.result_expr)}")
        if self.coeffs is not None:
            appendix.append(f"    Taylor coefficients: {format_laurent(self.coeffs)}")

        # Resolve every citation once over the whole document, number the sources,
        # then assemble: referee view, Sources list, appendix, closing note.
        prose = "\n".join(body)
        appendix_prose = "\n".join(appendix)
        used = refs.used_keys(prose + "\n" + appendix_prose)
        numbers = refs.numbering(used)
        out = [refs.render(prose, "text", numbers)]
        for key in used:
            ref = refs.reference(key)
            out.append(f"    [{numbers[key]}] {ref.full()}")
            out.append(f"        {ref.url}")
            if ref.doi_url:
                out.append(f"        {ref.doi_url}")
        out.append("    Per-function theorem and check mapping: docs/CORRECTNESS.md.")
        out.append("")
        out.append(refs.render(appendix_prose, "text", numbers))
        out.append("")
        out.append(
            "This is a human-auditable derivation, not a formal machine proof;"
        )
        out.append("every line above is checkable by hand.")
        return "\n".join(out)

    def render_terminal(self, console: Any = None) -> None:
        """Print the derivation. Uses rich for a rule if available, else builtins."""
        body = self.text()
        if console is not None:
            from rich.panel import Panel
            from rich.text import Text

            console.print(Panel(Text(body), title="qreals certificate", expand=False))
        else:
            print(body)

    # -- HTML --------------------------------------------------------------

    def to_html(self) -> str:
        """A standalone HTML view of the derivation, every citation an ``<a>``.

        The same three sections as the terminal and PDF, with each in-text
        citation a clickable link and a hyperlinked Sources list at the end.
        Math is shown as plain text in ``<code>`` rather than typeset; this view
        is for reading and following links, the PDF is the typeset copy.
        """
        esc = html.escape
        r = self.referee
        parts: list[str] = [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>{esc(self.title)}</title></head><body>",
            f"<h1>{esc(self.title)}</h1>",
            f"<p>{esc(self.input_line)}</p>",
            "<h2>Referee view</h2>",
        ]
        if r.recursion:
            parts.append("<p><strong>The recursion.</strong><br>")
            parts.append("<br>".join(esc(line) for line in r.recursion))
            parts.append("</p>")
        if r.structure:
            parts.append(
                "<p><strong>Structure.</strong> "
                + " ".join(esc(s) for s in r.structure)
                + "</p>"
            )
        if r.folds:
            parts.append(
                '<table border="1"><thead><tr><th>step i</th>'
                "<th>quotient a_i</th><th>partial R_i</th><th>deg N_i</th>"
                "</tr></thead><tbody>"
            )
            for f in r.folds:
                parts.append(
                    f"<tr><td>{f.pos}</td><td>{f.a}</td>"
                    f"<td>{esc(f.ratio)}</td><td>{f.degree}</td></tr>"
                )
            parts.append("</tbody></table>")
        parts.append(f"<p><strong>Result.</strong> {esc(r.headline)}</p>")
        if r.coeff_row is not None:
            parts.append(
                "<p>Coefficient row (full, no elision): <code>"
                + esc(", ".join(str(c) for c in r.coeff_row))
                + "</code></p>"
            )
        parts.append(f"<p>{esc(r.compare_title)}</p>")
        parts.append(
            '<table border="1"><thead><tr>'
            + "".join(f"<th>{esc(c)}</th>" for c in r.compare_columns)
            + "</tr></thead><tbody>"
        )
        for row in r.compare_rows:
            parts.append(
                "<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in row) + "</tr>"
            )
        parts.append("</tbody></table>")
        parts.append(f"<p>{esc(r.compare_summary)}</p>")
        if r.witness:
            parts.append(f"<p>{esc(r.witness)}</p>")
        parts.append("<h2>Independent cross-checks</h2><ul>")
        for check in self.stamp.checks:
            parts.append(f"<li>[{esc(check.status)}] {esc(check.detail)}</li>")
        parts.append("</ul>")
        parts.append(f"<p>Summary: {esc(self.stamp.line())}.</p>")
        parts.append("<h2>Sources</h2>")
        parts.append(
            "<p>Overview: [[cite:mgo-survey|q-deformed rationals and irrationals]]. "
            "Each source below opens at the cited paper.</p>"
        )

        # The appendix carries the full machine trace, after the Sources list.
        ap: list[str] = ["<h2>Appendix: full intermediate-polynomial trace</h2>"]
        ap.append("<h3>(a) Continued fraction and even-length MGO form</h3>")
        ap.append(f"<p>Regular continued fraction: <code>{esc(_cf_text(self.cf))}</code>.<br>")
        ap.append(f"Even-length MGO form: <code>{esc(_cf_text(self.even_cf))}</code>.")
        if self.convergent is not None:
            ap.append(
                "<br>The even-length form evaluates to the convergent "
                f"<code>{esc(str(self.convergent))}</code>."
            )
        ap.append("</p>")
        ap.append("<h3>(b) MGO formula folded step by step</h3>")
        ap.append(
            "<p>Odd positions carry [a]_q with q^a above; even positions carry "
            "[a]_(q^-1) with q^-a above ([[cite:mgo-rat|MGO eqn 1.1]]).</p>"
        )
        ap.append("<ul>")
        for label, expr in self.steps:
            ap.append(
                f"<li><strong>{esc(label)}:</strong> "
                f"<code>{esc(sp.sstr(sp.cancel(expr)))}</code></li>"
            )
        ap.append("</ul>")
        ap.append(
            f"<p><strong>{esc(self.result_label)}</strong> = "
            f"<code>{esc(sp.sstr(self.result_expr))}</code></p>"
        )
        if self.coeffs is not None:
            ap.append(
                f"<p>Taylor coefficients: <code>{esc(format_laurent(self.coeffs))}</code></p>"
            )

        body = "\n".join(parts)
        appendix = "\n".join(ap)
        used = refs.used_keys(body + "\n" + appendix)
        body = refs.render(body, "html")
        appendix = refs.render(appendix, "html")
        items: list[str] = []
        for key in used:
            ref = refs.reference(key)
            links: list[str] = []
            if ref.arxiv:
                links.append(
                    f'<a href="{esc(ref.url, quote=True)}">arXiv:{esc(ref.arxiv)}</a>'
                )
            if ref.doi and ref.doi_url:
                links.append(
                    f'<a href="{esc(ref.doi_url, quote=True)}">doi:{esc(ref.doi)}</a>'
                )
            tail = (" " + ", ".join(links) + ".") if links else ""
            items.append(f"<li>{esc(ref.citation)}.{tail}</li>")
        return (
            body
            + "\n<ul>\n"
            + "\n".join(items)
            + "\n</ul>\n"
            + "<p>Per-function theorem and check mapping: "
            + "<code>docs/CORRECTNESS.md</code>.</p>\n"
            + appendix
            + "\n<p><em>This is a human-auditable derivation, not a formal machine "
            + "proof; every line above is checkable by hand.</em></p>\n"
            + "</body></html>\n"
        )

    # -- LaTeX -------------------------------------------------------------

    def to_tex(
        self,
        *,
        max_terms: int | None = None,
        coeff_table_threshold: int | None = None,
        wrap: str | None = None,
    ) -> str:
        """A standalone LaTeX document for the derivation, compilable by pdflatex.

        Long polynomials are elided to ``max_terms`` monomials (head, ``\\dots``,
        tail, and a "(K terms omitted)" note) and every display is line-broken to
        the text block, so no line runs off the page. The full exact object stays
        available through the terminal, JSON, and coefficient read-outs; only the
        printed PDF is shortened. ``wrap`` is "breqn" (automatic breaking at + and
        -, the default) or "allowbreak" (a fallback for engines without breqn).
        """
        max_terms = self.max_terms if max_terms is None else max_terms
        threshold = (
            self.coeff_table_threshold
            if coeff_table_threshold is None
            else coeff_table_threshold
        )
        wrap = self.wrap if wrap is None else wrap

        referee_tex = self._referee_tex()

        # The appendix keeps the full line-by-line trace, elided to fit the page.
        step_blocks = []
        for label, expr in self.steps:
            block = _value_block(sp.cancel(expr), wrap, r"\footnotesize", max_terms)
            step_blocks.append(
                r"\noindent\textbf{" + _tex_escape(label) + ":}\n" + block
            )
        steps_tex = "\n\n".join(step_blocks)

        result_tex = ""
        if self.result_expr is not None:
            block = _value_block(
                sp.cancel(self.result_expr), wrap, r"\footnotesize", max_terms
            )
            result_tex = (
                r"\noindent\textbf{Result.} "
                + _tex_escape(self.result_label)
                + ":\n"
                + block
            )

        coeffs_tex = ""
        if self.coeffs is not None:
            nonzero = sum(1 for c in self.coeffs if c != 0)
            if nonzero > threshold:
                coeffs_tex = (
                    r"\noindent Taylor coefficients (nonzero terms; the series "
                    rf"continues as $O(q^{{{len(self.coeffs)}}})$):"
                    + "\n"
                    + _coeff_longtable(self.coeffs)
                )
            else:
                body, omitted = _coeffs_poly_latex(self.coeffs, max_terms)
                note = f"{omitted} terms omitted" if omitted else None
                coeffs_tex = (
                    r"\noindent Taylor coefficients:"
                    + "\n"
                    + _display(body, wrap, note, r"\small")
                )

        checks_tex = "\n".join(
            rf"\item \textbf{{[{check.status}]}} {_tex_escape(check.detail)}"
            for check in self.stamp.checks
        )
        convergent_tex = ""
        if self.convergent is not None:
            convergent_tex = (
                r"The even-length form evaluates to the convergent "
                rf"$ {sp.latex(self.convergent)} $.\par"
            )

        body = (
            _preamble(wrap)
            + rf"""\title{{{_tex_escape(self.title)}}}
\author{{qreals certificate}}
\date{{\today}}
\begin{{document}}
\maketitle
\sloppy

\noindent {_tex_escape(self.input_line)}

{referee_tex}

\section*{{Independent cross-checks}}
\begin{{itemize}}
{checks_tex}
\end{{itemize}}
Summary: {_tex_escape(self.stamp.line())}.

\section*{{Sources}}
\noindent Overview: [[cite:mgo-survey|q-deformed rationals and irrationals]].
Each source below opens at the cited paper.\par
"""
        )
        # Resolve every in-text citation to an \href, then list the sources used.
        # The appendix is rendered after the Sources list so the referee view reads
        # first; it carries the full intermediate-polynomial trace.
        appendix = rf"""\section*{{Appendix: full intermediate-polynomial trace}}
\subsection*{{(a) Continued fraction and even-length MGO form}}
Regular continued fraction: {_cf_text(self.cf)}. \par
Even-length MGO form: {_cf_text(self.even_cf)}. \par
{convergent_tex}

\subsection*{{(b) MGO formula folded step by step}}
Odd positions carry $[a]_q$ with $q^a$ above; even positions carry
$[a]_{{q^{{-1}}}}$ with $q^{{-a}}$ above ([[cite:mgo-rat|MGO eqn.~1.1]]).

{steps_tex}

{result_tex}

{coeffs_tex}
"""
        used = refs.used_keys(body + appendix)
        body = refs.render(body, "tex")
        appendix = refs.render(appendix, "tex")
        items = "\n".join(_tex_source_item(refs.reference(key)) for key in used)
        return (
            body
            + "\\begin{itemize}\n"
            + items
            + "\n\\end{itemize}\n"
            + "\\noindent Per-function theorem and check mapping: "
            + "\\texttt{docs/CORRECTNESS.md}.\n\n"
            + appendix
            + "\n\\medskip\n"
            + "\\noindent This is a human-auditable derivation, not a formal "
            + "machine proof;\nevery line above is checkable by hand.\n"
            + "\\end{document}\n"
        )

    def _referee_tex(self) -> str:
        """The referee view as LaTeX: recursion, structure, fold table, result,
        and the tie-back comparison table. Citations stay as placeholders for the
        single ``refs.render`` pass in ``to_tex``.
        """
        r = self.referee
        parts: list[str] = [r"\section*{Referee view}"]
        if r.recursion:
            m = len(self.even_cf)
            parts.append(
                r"\noindent The even-length form $[a_1, \dots, a_{%d}]_q$ is folded "
                r"inside out, with $R_i$ the partial from position $i$ "
                r"([[cite:mgo-rat|MGO eqn.~1.1]]):" % m
            )
            parts.append(
                r"\[\begin{aligned}"
                + (r"R_{%d} &= [a_{%d}]_{q^{-1}}, \\" % (m, m))
                + r"R_i &= [a_i]_q + q^{a_i}/R_{i+1} && (i\ \text{odd}), \\"
                + r"R_i &= [a_i]_{q^{-1}} + q^{-a_i}/R_{i+1} && (i\ \text{even}),"
                + r"\end{aligned}\]"
            )
            parts.append(
                r"\noindent with $[a]_q = (1-q^a)/(1-q)$ the Gauss $q$-integer.\par"
            )
        if r.structure:
            parts.append(
                r"\noindent\textbf{Structure.} Write each partial as "
                r"$R_i = N_i/D_i$. The recursion sets the next denominator to the "
                r"current numerator, $D_i = N_{i+1}$, so the fold telescopes; the "
                r"degrees in the table are after reducing each $R_i$ to lowest "
                r"terms. Odd folds carry $[a]_q$ and even folds $[a]_{q^{-1}}$, the "
                r"$q \leftrightarrow q^{-1}$ duality ([[cite:mgo-rat|MGO eqn.~1.1]]); "
                r"the numerator coefficients are non-negative, and the coefficient "
                r"row below is given in full, not elided.\par"
            )
        if r.folds:
            rows = " ".join(
                r"$%d$ & $%d$ & $N_{%d}/D_{%d}$ & $%d$ \\"
                % (f.pos, f.a, f.pos, f.pos, f.degree)
                for f in r.folds
            )
            parts.append(
                r"\begin{center}\begin{tabular}{rrcr}\toprule "
                r"step $i$ & quotient $a_i$ & partial $R_i$ & $\deg N_i$ \\\midrule "
                + rows
                + r"\bottomrule\end{tabular}\end{center}"
            )
        parts.append(r"\noindent\textbf{Result.} " + _tex_escape(r.headline) + r"\par")
        if r.coeff_row is not None:
            row = ", ".join(str(c) for c in r.coeff_row)
            parts.append(
                r"\begin{flushleft}\small\noindent Coefficient row (full, no "
                r"elision): " + row + r"\end{flushleft}"
            )
        parts.append(r"\noindent " + _tex_escape(r.compare_title) + r"\par")
        colspec = "l" * len(r.compare_columns)
        header = " & ".join(_tex_escape(c) for c in r.compare_columns) + r" \\"
        body_rows = " ".join(
            " & ".join(_tex_escape(str(c)) for c in row) + r" \\"
            for row in r.compare_rows
        )
        parts.append(
            r"\begin{center}\begin{tabular}{" + colspec + r"}\toprule "
            + header
            + r"\midrule "
            + body_rows
            + r"\bottomrule\end{tabular}\end{center}"
        )
        parts.append(r"\noindent " + _tex_escape(r.compare_summary) + r"\par")
        if r.witness:
            parts.append(r"\noindent " + _tex_escape(r.witness) + r"\par")
        return "\n".join(parts)

    # -- PDF view (no file kept) ------------------------------------------

    def view_pdf(self, console: Any = None) -> bool:
        """Compile a temp PDF, open it in the system viewer, then delete it.

        Returns True if a viewer was launched. With no TeX engine, prints a
        one-line note on how to compile and returns False, leaving the terminal
        view as the way to read the derivation.
        """
        engine = find_tex_engine()
        if engine is None:
            _say(
                console,
                "no TeX engine on PATH (pdflatex, tectonic, latexmk); cannot make a "
                "PDF. Read the derivation with the terminal view, or save the .tex "
                "and compile it elsewhere.",
            )
            return False
        with tempfile.TemporaryDirectory() as work:
            pdf_in_work = _compile_tex(self.to_tex(), self.slug, work, engine)
            if pdf_in_work is None:
                _say(
                    console,
                    f"{engine} did not produce a PDF; the .tex may need attention.",
                )
                return False
            # Copy out so the temp directory can close; open, then remove.
            handle = tempfile.NamedTemporaryFile(
                prefix=f"{self.slug}-", suffix=".pdf", delete=False
            )
            handle.close()
            shutil.copyfile(pdf_in_work, handle.name)
        _open_in_viewer(handle.name)
        _remove_after_open(handle.name)
        _say(
            console,
            "opened the certificate PDF in your viewer; the temporary file is removed.",
        )
        return True

    # -- SAVE (the only path that writes a file) --------------------------

    def save(
        self, directory: str = ".", *, compile_pdf: bool = True, qprov: bool = False
    ) -> dict[str, Any]:
        """Write a .tex (and a .pdf if a TeX engine is found) into `directory`.

        This is the only certificate path that leaves a file. Returns a dict
        with the paths written and the qprov id if a run was recorded. The
        qprov bridge is off by default so scripts never touch the store without
        being asked; pass qprov=True to record the run for later citation.
        """
        os.makedirs(directory, exist_ok=True)
        tex_path = os.path.join(directory, f"{self.slug}.tex")
        with open(tex_path, "w", encoding="utf-8") as handle:
            handle.write(self.to_tex())
        result: dict[str, Any] = {"tex": tex_path, "pdf": None, "qprov_id": None}
        engine = find_tex_engine() if compile_pdf else None
        if engine is not None:
            with tempfile.TemporaryDirectory() as work:
                pdf_in_work = _compile_tex(self.to_tex(), self.slug, work, engine)
                if pdf_in_work is not None:
                    pdf_path = os.path.join(directory, f"{self.slug}.pdf")
                    shutil.copyfile(pdf_in_work, pdf_path)
                    result["pdf"] = pdf_path
        if qprov:
            result["qprov_id"] = _record_in_qprov(
                self.slug, self.qprov_inputs, self.qprov_outputs, tex_path
            )
        return result


# --------------------------------------------------------------------------
# Construction from a result dict.
# --------------------------------------------------------------------------


def _mgo_fold_steps(a: list[int]) -> list[tuple[str, sp.Expr]]:
    """The symbolic MGO fold of an even-length CF, one entry per folded term."""
    n = len(a)
    steps: list[tuple[str, sp.Expr]] = []

    def term(i: int, ai: int) -> sp.Expr:
        return q_int(ai) if (i + 1) % 2 == 1 else q_int_qinv(ai)

    def num_above(i: int, ai: int) -> sp.Expr:
        return q**ai if (i + 1) % 2 == 1 else q ** (-ai)

    result = term(n - 1, a[n - 1])
    steps.append((f"innermost term a_{n} = {a[n - 1]}", sp.cancel(result)))
    for i in range(n - 2, -1, -1):
        result = sp.cancel(term(i, a[i]) + num_above(i, a[i]) / result)
        steps.append((f"fold in a_{i + 1} = {a[i]}", result))
    return steps


def _slug(parts: list[object]) -> str:
    raw = "qreals-certificate-" + "-".join(str(p) for p in parts)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")


# --------------------------------------------------------------------------
# Referee view: the half-page proof built from the same fold the appendix dumps.
# --------------------------------------------------------------------------

_Q0 = sp.Rational(1, 2)  # the fixed point for the numeric witness, in (0, 1)


def _q_max_degree(expr: sp.Expr) -> int:
    """The highest power of q in a polynomial expression (0 for a constant)."""
    poly = sp.expand(expr)
    if poly == 0 or not poly.has(q):
        return 0
    return int(sp.Poly(poly, q).degree())


def _fold_section(even_cf: list[int], steps: list[tuple[str, sp.Expr]]) -> tuple[
    list[str], list[str], list[FoldRow]
]:
    """The recursion text, the structural facts, and the per-fold table.

    Shared by the rational and series referee views; the q-integer has no fold
    and does not call this.
    """
    m = len(even_cf)
    recursion = [
        f"The even-length form [a_1, ..., a_{m}]_q is folded inside out, R_i the "
        "partial from position i ([[cite:mgo-rat|MGO eqn 1.1]]):",
        f"    R_{m} = [a_{m}]_(q^-1)",
        "    R_i = [a_i]_q      + q^(a_i)  / R_(i+1)      (i odd)",
        "    R_i = [a_i]_(q^-1) + q^(-a_i) / R_(i+1)      (i even),",
        "    with [a]_q = (1 - q^a)/(1 - q) the Gauss q-integer.",
    ]
    structure = [
        "Write each partial as R_i = N_i / D_i. The recursion sets the next "
        "denominator to the current numerator, D_i = N_(i+1), so the fold "
        "telescopes; the degrees below are after reducing each R_i to lowest terms.",
        "Odd folds carry [a]_q, even folds [a]_(q^-1): the q <-> q^-1 duality "
        "([[cite:mgo-rat|MGO eqn 1.1]]). The numerator coefficients are "
        "non-negative, and the full coefficient row is given below rather than "
        "elided, so the shape is visible.",
    ]
    folds: list[FoldRow] = []
    for j, (_label, expr) in enumerate(steps):
        pos = m - j
        num, _den = sp.fraction(sp.together(sp.cancel(expr)))
        folds.append(FoldRow(pos, int(even_cf[pos - 1]), f"N_{pos} / D_{pos}",
                             _q_max_degree(num)))
    return recursion, structure, folds


def _numeric_witness(expr: sp.Expr | None, coeffs: list[int] | None) -> str | None:
    """A calculator-checkable line: the exact function and the series at q = 1/2.

    Both are evaluated at q_0 = 1/2; the difference is at the truncation order
    q_0^N, the independent confirmation the spec asks for.
    """
    if expr is None or not coeffs:
        return None
    try:
        exact = sp.Rational(sp.cancel(expr.subs(q, _Q0)))
    except (TypeError, ValueError):
        return None
    partial = sp.Rational(
        sum((sp.Integer(int(c)) * _Q0**k for k, c in enumerate(coeffs)), sp.Integer(0))
    )
    diff = abs(exact - partial)
    return (
        f"Numeric witness at q = 1/2: the exact rational function gives "
        f"{float(exact):.8f}, the truncated series {float(partial):.8f}; the "
        f"difference {float(diff):.2e} sits at the q^{len(coeffs)} truncation order."
    )


def _text_table(columns: list[str], rows: list[list[str]]) -> str:
    """A monospace table, four-space indented, columns left-aligned to width."""
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    out = ["    " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))]
    out.append("    " + "  ".join("-" * widths[i] for i in range(len(widths))))
    for row in rows:
        out.append(
            "    " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
        )
    return "\n".join(out)


def build_certificate(result: dict[str, Any], *, depth: int = 12) -> Certificate:
    """Turn a computation result dict into a Certificate.

    Handles the kinds with a clean MGO derivation: rational, qint, coeffs and
    laurent. Other [x]_q kinds (prefix, locked, shift, readouts) are certified
    through their underlying x as a coefficient derivation.
    """
    kind = result.get("kind")
    data = result.get("data", {})

    if kind == "rational":
        p, s = int(data["p"]), int(data["s"])
        return _certificate_rational(p, s)
    if kind == "qint":
        return _certificate_qint(int(data["n"]))
    # everything else is a [x]_q series derivation
    x_repr = str(data["x"])
    n = data.get("n") or (int(data["order"]) + 1 if "order" in data else None)
    if n is None and isinstance(data.get("coefficients"), list):
        n = len(data["coefficients"])
    return _certificate_series(x_repr, int(n or depth))


def _certificate_rational(p: int, s: int) -> Certificate:
    value = sp.Rational(p, s)
    if p == s:
        cf = [1]
        even = [1]
        steps = [("trivial: equal numerator and denominator", sp.Integer(1))]
        expr = sp.Integer(1)
    else:
        cf = [int(t) for t in sp.continued_fraction(value)]
        even = make_even_length(list(cf))
        steps = _mgo_fold_steps(even)
        expr = sp.cancel(sp.sympify(q_rational(p, s)))
    # The q^0.. Taylor table applies only for value >= 0; a negative q-rational
    # is a Laurent object in negative powers, so its table is dropped (the
    # derivation, exact function and cross-checks still certify it).
    coeffs = q_real_truncated(f"{p}/{s}", 12) if value >= 0 else None
    referee = _referee_rational(p, s, value, expr, even, steps, coeffs)
    return Certificate(
        title=f"Certificate for [{p}/{s}]_q (exact q-rational)",
        input_line=f"Input: the rational p/s = {p}/{s} = {value}.",
        cf=cf,
        even_cf=even,
        steps=steps,
        result_label=f"[{p}/{s}]_q",
        result_expr=expr,
        coeffs=coeffs,
        convergent=value,
        stamp=verify({"kind": "rational", "data": {"p": p, "s": s}}),
        slug=_slug(["rational", p, s]),
        referee=referee,
        qprov_inputs={"p": p, "s": s},
        qprov_outputs={"expr": sp.sstr(expr), "at_q_eq_1": str(value)},
    )


def _referee_rational(
    p: int,
    s: int,
    value: sp.Rational,
    expr: sp.Expr,
    even: list[int],
    steps: list[tuple[str, sp.Expr]],
    coeffs: list[int] | None,
) -> RefereeView:
    """Referee view for an exact q-rational: tie-back to p/s at q = 1 and the prefix."""
    if p == s:
        recursion: list[str] = []
        structure = [
            "With equal numerator and denominator [p/s]_q = [1]_q = 1; no fold is needed."
        ]
        folds: list[FoldRow] = []
    else:
        recursion, structure, folds = _fold_section(even, steps)
    at_one = sp.simplify(expr.subs(q, 1))
    deliverable = [f"[{p}/{s}]_q = {sp.sstr(expr)}  (exact rational function in q)."]
    columns = ["quantity", "computed", "known", "match"]
    rows = [
        [
            "value at q = 1",
            str(at_one),
            str(value),
            "yes" if sp.simplify(at_one - value) == 0 else "no",
        ]
    ]
    if coeffs is not None:
        floor_t = int(value)
        want = [1] * floor_t + [0]
        got = coeffs[: floor_t + 1]
        rows.append(
            [
                "integer-part prefix",
                format_laurent(got),
                f"first {floor_t} ones then 0",
                "yes" if got == want else "no",
            ]
        )
    headline = (
        f"[{p}/{s}]_q = 1."
        if p == s
        else (
            f"[{p}/{s}]_q is the exact rational function shown in the appendix; "
            "its coefficient row opens as below."
        )
    )
    return RefereeView(
        recursion=recursion,
        structure=structure,
        folds=folds,
        deliverable_lines=deliverable,
        compare_title=(
            "Sanity check: the q-rational specialises to p/s at q = 1 and opens "
            "with the integer-part prefix."
        ),
        compare_columns=columns,
        compare_rows=rows,
        compare_summary=(
            f"[{p}/{s}]_q at q = 1 is the ordinary value {value}."
        ),
        headline=headline,
        coeff_row=coeffs,
        witness=_numeric_witness(expr, coeffs),
    )


def _certificate_qint(n: int) -> Certificate:
    expr = q_int(n)
    coeffs = q_real_truncated(str(n), max(n + 2, 12)) if n >= 0 else None
    return Certificate(
        title=f"Certificate for the q-integer [{n}]_q",
        input_line=f"Input: the integer n = {n}.",
        cf=[n],
        even_cf=[n],
        steps=[(f"[{n}]_q is the Gauss q-integer 1 + q + ... + q^(n-1)", expr)],
        result_label=f"[{n}]_q",
        result_expr=expr,
        coeffs=coeffs,
        convergent=sp.Integer(n),
        stamp=verify({"kind": "qint", "data": {"n": n}}),
        slug=_slug(["qint", n]),
        referee=_referee_qint(n, expr),
        qprov_inputs={"n": n},
        qprov_outputs={"expr": sp.sstr(expr)},
    )


def _referee_qint(n: int, expr: sp.Expr) -> RefereeView:
    """Referee view for a q-integer: an all-ones polynomial, value n at q = 1.

    There is no continued-fraction fold, so the recursion and the per-fold table
    are empty; the deliverable is the closed form, with no O(q^N) tail since
    [n]_q is an exact finite polynomial.
    """
    if n >= 0:
        deliverable = [
            f"[{n}]_q = 1 + q + ... + q^{n - 1}, the Gauss q-integer: an all-ones "
            f"polynomial of degree {n - 1} (exact and finite, no O(q^N) tail)."
            if n >= 1
            else "[0]_q = 0, the empty sum."
        ]
        rows = [["value at q = 1", str(sp.simplify(expr.subs(q, 1))), str(n), "yes"]]
        if n >= 1:
            coeff_row = [1] * n
            rows.append(
                [
                    "coefficient row",
                    str(coeff_row),
                    f"{n} ones",
                    "yes",
                ]
            )
    else:
        deliverable = [
            f"[{n}]_q = {sp.sstr(expr)}, a Laurent polynomial in negative powers of q."
        ]
        rows = [["value at q = 1", str(sp.simplify(expr.subs(q, 1))), str(n), "yes"]]
    if n >= 1:
        headline = (
            f"[{n}]_q = 1 + q + ... + q^(n-1), an all-ones polynomial of degree "
            f"{n - 1} (exact and finite)."
        )
    elif n == 0:
        headline = "[0]_q = 0, the empty sum."
    else:
        headline = f"[{n}]_q is a Laurent polynomial in negative powers of q."
    return RefereeView(
        recursion=[],
        structure=[
            "[n]_q is the Gauss q-integer; no continued-fraction fold is required."
        ],
        folds=[],
        deliverable_lines=deliverable,
        compare_title="Sanity check: tie-back to the ordinary integer at q = 1.",
        compare_columns=["quantity", "computed", "known", "match"],
        compare_rows=rows,
        compare_summary=f"[{n}]_q at q = 1 is the ordinary integer {n}.",
        headline=headline,
        coeff_row=([1] * n if n >= 1 else None),
        witness=_numeric_witness(expr, [1] * n if n >= 1 else None),
    )


def _certificate_series(x_repr: str, n: int) -> Certificate:
    n = max(int(n), 1)
    cf = cf_partials(x_repr, n)
    even = make_even_length(list(cf))
    # The even-length CF is a finite integer list, so reducing it yields the
    # exact rational convergent directly. An earlier nsimplify round-trip could
    # re-name a sharp convergent back to the irrational it approximates (for
    # sqrt(2) it returned sqrt(2)), which sp.Rational then rejected. Take the
    # reduced value as is and guard that it is rational before sp.Rational.
    convergent = sp.continued_fraction_reduce(even)
    if not getattr(convergent, "is_Rational", False):
        raise ValueError(
            f"even-length continued fraction for {x_repr} did not reduce to a "
            f"rational convergent (got {convergent}); cannot build a certificate"
        )
    convergent = sp.Rational(convergent)
    steps = _mgo_fold_steps(even)
    expr = sp.cancel(sp.sympify(q_rational(convergent.p, convergent.q)))
    coeffs = q_real_truncated(x_repr, n)
    referee = _referee_series(x_repr, n, convergent, expr, even, steps, coeffs)
    return Certificate(
        title=f"Certificate for [{x_repr}]_q (first {n} coefficients)",
        input_line=(
            f"Input: the real number x = {x_repr}. The continued fraction is "
            f"truncated to the depth that locks the first {n} coefficients "
            f"([[cite:mgo-real|MGO Proposition 1.1]])."
        ),
        cf=cf,
        even_cf=even,
        steps=steps,
        result_label=f"[{convergent}]_q (the convergent, whose Taylor expansion gives [x]_q)",
        result_expr=expr,
        coeffs=coeffs,
        convergent=convergent,
        stamp=verify({"kind": "coeffs", "data": {"x": x_repr, "n": n}}),
        slug=_slug(["real", x_repr, n]),
        referee=referee,
        qprov_inputs={"x": x_repr, "n": n},
        qprov_outputs={"coefficients": coeffs},
    )


def _referee_series(
    x_repr: str,
    n: int,
    convergent: sp.Rational,
    expr: sp.Expr,
    even: list[int],
    steps: list[tuple[str, sp.Expr]],
    coeffs: list[int],
) -> RefereeView:
    """Referee view for a [x]_q series: tie the first coefficients to the
    next-deeper convergent's overlap, the stabilisation of MGO Proposition 1.1.
    """
    recursion, structure, folds = _fold_section(even, steps)
    deliverable = [
        f"The first {n} coefficients of [{x_repr}]_q, read from the Taylor "
        f"expansion of R_1 = N_1 / D_1, in full (no interior elision):",
        "    " + ", ".join(str(c) for c in coeffs),
        f"    as a series: {format_laurent(coeffs)}",
    ]
    deeper = q_real_truncated(x_repr, n + 4)
    match_all = deeper[:n] == coeffs
    visible = min(n, 12)
    columns = ["power k", "computed c_k", "deeper convergent", "match"]
    rows = [
        [str(k), str(coeffs[k]), str(deeper[k]),
         "yes" if coeffs[k] == deeper[k] else "no"]
        for k in range(visible)
    ]
    summary = (
        f"all {n} computed coefficients agree with the deeper convergent that "
        "locks them."
        if match_all
        else "MISMATCH: a coefficient moved when the convergent deepened."
    )
    return RefereeView(
        recursion=recursion,
        structure=structure,
        folds=folds,
        deliverable_lines=deliverable,
        compare_title=(
            "Sanity check: the leading coefficients against the next-deeper "
            "convergent, the stabilisation of [[cite:mgo-real|MGO Proposition 1.1]]."
        ),
        compare_columns=columns,
        compare_rows=rows,
        compare_summary=summary,
        headline=(
            f"The first {n} coefficients of [{x_repr}]_q, from the Taylor "
            "expansion of the locking convergent; the exact rational function "
            "R_1 = N_1/D_1 is in the appendix."
        ),
        coeff_row=coeffs,
        witness=_numeric_witness(expr, coeffs),
    )


# --------------------------------------------------------------------------
# LaTeX, PDF and qprov helpers.
# --------------------------------------------------------------------------


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _tex_source_item(ref: refs.Reference) -> str:
    """One ``\\item`` for the Sources list: the citation plus clickable ids."""
    links = []
    if ref.arxiv:
        links.append(r"\href{" + ref.url + "}{arXiv:" + ref.arxiv + "}")
    if ref.doi:
        links.append(r"\href{" + ref.doi_url + "}{doi:" + ref.doi + "}")
    tail = (" " + ", ".join(links) + ".") if links else ""
    return r"\item " + _tex_escape(ref.citation) + "." + tail


def _preamble(wrap: str) -> str:
    """The document preamble. A 2cm margin widens the text block; longtable and
    booktabs set the coefficient table; pdflscape is the landscape last resort;
    hyperref makes the citations clickable; breqn (when chosen) breaks long
    displays at + and -. hyperref is loaded before breqn, the order that compiles
    cleanly with both."""
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{longtable}",
        r"\usepackage{booktabs}",
        r"\usepackage{pdflscape}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue,citecolor=blue}",
    ]
    if wrap == "breqn":
        lines.append(r"\usepackage{breqn}")
    return "\n".join(lines) + "\n"


def _cf_text(seq: list[int]) -> str:
    """A continued fraction as a plain-text comma list, which wraps at the commas
    instead of running off the page the way an inline math list does."""
    return "[" + ", ".join(str(int(t)) for t in seq) + "]"


def _join(terms: list[sp.Expr], lead_op: bool) -> str:
    """LaTeX for a run of monomials, signs carried as + and - between them.

    With ``lead_op`` true the first term also gets its leading operator, so the
    run can follow a ``\\dots`` without losing the sign between them.
    """
    parts: list[str] = []
    for i, term in enumerate(terms):
        ltx = sp.latex(term).lstrip()
        neg = ltx.startswith("-")
        body = ltx[1:].lstrip() if neg else ltx
        if i == 0 and not lead_op:
            parts.append(f"-{body}" if neg else body)
        else:
            parts.append(f"{'-' if neg else '+'} {body}")
    return " ".join(parts)


def _elide_sum(expr: sp.Expr, max_terms: int) -> tuple[str, int]:
    """LaTeX for a polynomial, capped at ``max_terms`` monomials.

    Returns the rendered body and the count of omitted terms. Above the cap it
    keeps the leading half and trailing half (by power) joined with ``\\dots``.
    """
    terms = sp.expand(expr).as_ordered_terms()
    n = len(terms)
    if n <= max_terms:
        return _join(terms, lead_op=False), 0
    head = (max_terms + 1) // 2
    tail = max_terms - head
    omitted = n - head - tail
    body = (
        _join(terms[:head], lead_op=False)
        + r" + \dots "
        + _join(terms[-tail:], lead_op=True)
    )
    return body, omitted


# A \frac numerator or denominator is one unbreakable box, so it is kept inline
# only while short enough to fit the text width; longer ratios are split into
# separate N(q) and D(q) displays that breqn can break at + and -.
_FRAC_INLINE = 12


def _value_block(expr: sp.Expr, wrap: str, size: str, max_terms: int) -> str:
    """A display block for a (possibly rational) q-expression that fits the page.

    A polynomial, or a short ratio, is one display. A long ratio becomes a line
    naming it as $N(q)/D(q)$ followed by a broken display of each part, since a
    ``\\frac`` cannot break across lines.
    """
    expr = sp.cancel(expr)
    num, den = sp.fraction(sp.together(expr))
    if den == 1:
        body, omitted = _elide_sum(num, max_terms)
        note = f"{omitted} terms omitted" if omitted else None
        return _display(body, wrap, note, size)

    num_count = len(sp.expand(num).as_ordered_terms())
    den_count = len(sp.expand(den).as_ordered_terms())
    num_l, num_o = _elide_sum(num, max_terms)
    den_l, den_o = _elide_sum(den, max_terms)

    if num_count <= _FRAC_INLINE and den_count <= _FRAC_INLINE:
        cut = []
        if num_o:
            cut.append(f"{num_o} numerator")
        if den_o:
            cut.append(f"{den_o} denominator")
        note = (" and ".join(cut) + " terms omitted") if cut else None
        return _display(r"\frac{" + num_l + "}{" + den_l + "}", wrap, note, size)

    num_note = f"{num_o} terms omitted" if num_o else None
    den_note = f"{den_o} terms omitted" if den_o else None
    return (
        r"\noindent The value is the ratio $N(q)/D(q)$ with:\par"
        + "\n"
        + _display("N(q) = " + num_l, wrap, num_note, size)
        + "\n\n"
        + _display("D(q) = " + den_l, wrap, den_note, size)
    )


def _coeffs_poly_latex(coeffs: list[int], max_terms: int) -> tuple[str, int]:
    """LaTeX for a Taylor-coefficient list, low power to high, capped and tailed.

    Returns the body (ending in the ``O(q^N)`` tail) and the omitted-term count.
    """
    terms: list[tuple[str, str]] = []  # (operator, body) low power to high
    for k, c in enumerate(coeffs):
        if c == 0:
            continue
        mono = "" if k == 0 else ("q" if k == 1 else f"q^{{{k}}}")
        op = "-" if c < 0 else "+"
        if mono == "":
            body = str(abs(c))
        elif abs(c) == 1:
            body = mono
        else:
            body = f"{abs(c)} {mono}"
        terms.append((op, body))
    tail_o = rf"O(q^{{{len(coeffs)}}})"

    def render(seq: list[tuple[str, str]], lead_op: bool) -> str:
        out: list[str] = []
        for i, (op, body) in enumerate(seq):
            if i == 0 and not lead_op:
                out.append(f"-{body}" if op == "-" else body)
            else:
                out.append(f"{op} {body}")
        return " ".join(out)

    if not terms:
        return "0 + " + tail_o, 0
    if len(terms) <= max_terms:
        return render(terms, lead_op=False) + " + " + tail_o, 0
    head = (max_terms + 1) // 2
    tail = max_terms - head
    omitted = len(terms) - head - tail
    body = (
        render(terms[:head], lead_op=False)
        + r" + \dots "
        + render(terms[-tail:], lead_op=True)
        + " + "
        + tail_o
    )
    return body, omitted


def _coeff_longtable(coeffs: list[int]) -> str:
    """A two-column (power, coefficient) longtable of the nonzero coefficients.

    It paginates down the page, so a series of any length fits the text width.
    """
    rows = "\n".join(rf"${k}$ & ${c}$ \\" for k, c in enumerate(coeffs) if c != 0)
    return (
        "\\begin{center}\n"
        "\\begin{longtable}{r r}\n"
        "\\toprule\n"
        "power $k$ & coefficient $c_k$ \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{longtable}\n"
        "\\end{center}\n"
    )


def _display(body: str, wrap: str, note: str | None, size: str) -> str:
    """A display block that fits the text width, with an optional omission note.

    ``wrap`` "breqn" sets the body in ``dmath*`` so it breaks at + and -; the
    "allowbreak" fallback inserts breakpoints by hand in a left-aligned, ``size``
    block for engines that cannot load breqn.
    """
    if wrap == "breqn":
        block = "{" + size + "%\n\\begin{dmath*}\n" + body + "\n\\end{dmath*}\n}"
    else:
        broken = body.replace(" + ", " + \\allowbreak ").replace(
            " - ", " - \\allowbreak "
        )
        block = (
            "\\begin{flushleft}\n" + size + "$\\displaystyle " + broken + "$\n"
            "\\end{flushleft}"
        )
    if note:
        block += "\n\n{\\footnotesize\\itshape(" + note + ")}\\par"
    return block


def _compile_tex(tex: str, slug: str, work: str, engine: str) -> str | None:
    """Compile `tex` in `work` with `engine`; return the PDF path or None."""
    tex_path = os.path.join(work, f"{slug}.tex")
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(tex)
    if engine == "tectonic":
        cmd = [engine, "--outdir", work, tex_path]
        runs = 1
    elif engine == "latexmk":
        cmd = [
            engine,
            "-pdf",
            "-interaction=nonstopmode",
            f"-output-directory={work}",
            tex_path,
        ]
        runs = 1
    else:  # pdflatex
        cmd = [
            engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            work,
            tex_path,
        ]
        runs = 2  # a second pass settles \maketitle and references
    for _ in range(runs):
        try:
            subprocess.run(cmd, cwd=work, capture_output=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
    pdf_path = os.path.join(work, f"{slug}.pdf")
    return pdf_path if os.path.exists(pdf_path) else None


def _open_in_viewer(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606 - opening our own temp PDF
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def _remove_after_open(path: str, attempts: int = 8) -> None:
    """Delete a temp file once the viewer has had a moment to open it.

    On Windows the viewer may briefly hold the file open, so removal is retried
    a few times before giving up.
    """
    for _ in range(attempts):
        time.sleep(0.4)
        try:
            os.remove(path)
            return
        except OSError:
            continue


def _record_in_qprov(
    slug: str, inputs: dict[str, Any], outputs: dict[str, Any], tex_path: str
) -> str | None:
    """One-way bridge: if qprov imports, record the saved certificate.

    The core never imports qprov; this lives in the interface layer. Any
    failure degrades to just the certificate, returning None.
    """
    try:
        import qprov
    except Exception:  # noqa: BLE001 - qprov is fully optional
        return None
    try:
        qprov_id: str | None = qprov.register_external(
            function_name=f"qreals.certificate.{slug}",
            inputs=inputs,
            outputs=outputs,
            tags={"tool": "qreals", "artifact": "certificate"},
            notes=f"qreals certificate saved to {tex_path}",
        )
        return qprov_id
    except Exception:  # noqa: BLE001
        return None


def _say(console: Any, message: str) -> None:
    if console is not None:
        console.print(message)
    else:
        print(message)
