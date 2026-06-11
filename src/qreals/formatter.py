r"""The single owner of every ASCII math convention in qreals output.

Every CLI tool renders math through this module: q-integers as [n]_q,
cyclotomic polynomials by index, powers of q as q^k, roots of unity as
zeta_e, fractions as N(q) / S(q), and polynomials expanded in ascending
powers with explicit signs, wrapped at term boundaries (never mid-term).

Each ASCII emitter has a to_tex() twin so pages and --tex output are built
from the same data. No other module may hand-roll these strings; the test
suite greps the source tree for the cyclotomic prefix outside this file and
fails if it appears.
"""

from __future__ import annotations

import sympy as sp

from .rational import q

# The one place the cyclotomic ASCII prefix is spelled.
_PHI_PREFIX = "Phi_"
_ZETA_PREFIX = "zeta_"

DEFAULT_WRAP = 78


# --------------------------------------------------------------------------
# ASCII emitters.
# --------------------------------------------------------------------------


def qint_label(n: int) -> str:
    """The q-integer [n]_q."""
    return f"[{int(n)}]_q"


def phi_label(e: int, mult: int = 1) -> str:
    """The cyclotomic polynomial of index e, with an optional exponent."""
    base = f"{_PHI_PREFIX}{int(e)}"
    return base if mult == 1 else f"{base}^{int(mult)}"


def phi_applied_label(e: int, mult: int = 1) -> str:
    """The cyclotomic polynomial of index e applied to q, with an exponent."""
    base = f"{_PHI_PREFIX}{int(e)}(q)"
    return base if mult == 1 else f"{base}^{int(mult)}"


def q_power(k: int) -> str:
    """The monomial q^k (q for k = 1, 1 for k = 0)."""
    k = int(k)
    if k == 0:
        return "1"
    if k == 1:
        return "q"
    return f"q^{k}"


def zeta_label(e: int) -> str:
    """The primitive root of unity of order e."""
    return f"{_ZETA_PREFIX}{int(e)}"


def fraction_layout(num: str, den: str) -> str:
    """A fraction rendered on one line as num / den, parenthesised as needed."""

    def wrap(s: str) -> str:
        return s if _is_atomic(s) else f"({s})"

    return f"{wrap(num)} / {wrap(den)}"


def _is_atomic(s: str) -> bool:
    """True when a rendered expression needs no parentheses in a fraction."""
    if " " in s or "+" in s.lstrip("+-") or "*" in s:
        return False
    body = s.lstrip("+-")
    return bool(body) and ("-" not in body)


def _terms_ascending(expr: sp.Expr) -> list[tuple[int, sp.Integer]]:
    """The (exponent, integer coefficient) terms of a polynomial, ascending."""
    poly = sp.Poly(sp.expand(expr), q)
    terms = [(int(m[0]), c) for m, c in poly.terms()]
    return sorted(terms, key=lambda t: t[0])


def _term_ascii(k: int, c: sp.Integer) -> str:
    """One monomial c*q^k without a leading sign (the sign is handled apart)."""
    c = abs(int(c))
    if k == 0:
        return str(c)
    mono = q_power(k)
    return mono if c == 1 else f"{c}*{mono}"


def poly_ascii(expr: sp.Expr, wrap: int = DEFAULT_WRAP) -> str:
    """A polynomial in q, expanded in ascending powers with explicit signs.

    Long polynomials wrap at term boundaries: every line is at most `wrap`
    characters and no term is ever split across lines. Continuation lines
    start with the sign of their first term.
    """
    terms = _terms_ascending(expr)
    if not terms:
        return "0"
    pieces: list[str] = []
    for i, (k, c) in enumerate(terms):
        sign = "-" if int(c) < 0 else "+"
        body = _term_ascii(k, c)
        if i == 0:
            pieces.append(body if sign == "+" else f"-{body}")
        else:
            pieces.append(f"{sign} {body}")
    lines: list[str] = []
    current = pieces[0]
    for piece in pieces[1:]:
        candidate = f"{current} {piece}"
        if len(candidate) > wrap and current:
            lines.append(current)
            current = piece
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def factored_ascii(
    content: sp.Expr,
    factors: list[tuple[sp.Expr, int]],
    cyclotomic_index,
) -> str:
    """A factorisation with cyclotomic factors shown by index.

    `factors` is a list of (irreducible factor, multiplicity) pairs and
    `cyclotomic_index` maps a factor to its cyclotomic index or None (the
    classifier from the factor module). Non-cyclotomic factors print expanded
    in parentheses.
    """
    parts: list[str] = []
    if sp.sympify(content) != sp.Integer(1):
        parts.append(str(content))
    for fac, mult in factors:
        e = cyclotomic_index(fac)
        if e is not None:
            parts.append(phi_label(e, mult))
        else:
            label = f"({poly_ascii(fac, wrap=10**9)})"
            parts.append(label if mult == 1 else f"{label}^{int(mult)}")
    return " * ".join(parts) if parts else "1"


def congruence_ascii(lhs: str, rhs: str, mod: int) -> str:
    """A congruence lhs == rhs (mod m)."""
    return f"{lhs} == {rhs} (mod {int(mod)})"


# --------------------------------------------------------------------------
# TeX twins. Same data, TeX output; used by --tex and any served page.
# --------------------------------------------------------------------------


def qint_tex(n: int) -> str:
    """The q-integer in TeX."""
    return f"[{int(n)}]_q"


def phi_tex(e: int, mult: int = 1) -> str:
    """The cyclotomic polynomial of index e in TeX."""
    base = rf"\Phi_{{{int(e)}}}"
    return base if mult == 1 else base + rf"^{{{int(mult)}}}"


def q_power_tex(k: int) -> str:
    """The monomial q^k in TeX."""
    k = int(k)
    if k == 0:
        return "1"
    if k == 1:
        return "q"
    return rf"q^{{{k}}}"


def zeta_tex(e: int) -> str:
    """The primitive root of unity of order e in TeX."""
    return rf"\zeta_{{{int(e)}}}"


def fraction_tex(num: str, den: str) -> str:
    """A TeX fraction from already-rendered numerator and denominator."""
    return rf"\frac{{{num}}}{{{den}}}"


def poly_tex(expr: sp.Expr) -> str:
    """A polynomial in q in TeX, ascending powers, explicit signs."""
    terms = _terms_ascending(expr)
    if not terms:
        return "0"
    out: list[str] = []
    for i, (k, c) in enumerate(terms):
        sign = "-" if int(c) < 0 else "+"
        cc = abs(int(c))
        mono = q_power_tex(k)
        body = str(cc) if k == 0 else (mono if cc == 1 else f"{cc} {mono}")
        if i == 0:
            out.append(body if sign == "+" else f"-{body}")
        else:
            out.append(f"{sign} {body}")
    return " ".join(out)


def factored_tex(
    content: sp.Expr,
    factors: list[tuple[sp.Expr, int]],
    cyclotomic_index,
) -> str:
    """The TeX twin of factored_ascii."""
    parts: list[str] = []
    if sp.sympify(content) != sp.Integer(1):
        parts.append(str(content))
    for fac, mult in factors:
        e = cyclotomic_index(fac)
        if e is not None:
            parts.append(phi_tex(e, mult))
        else:
            inner = rf"\left({poly_tex(fac)}\right)"
            parts.append(inner if mult == 1 else inner + rf"^{{{int(mult)}}}")
    return r" \, ".join(parts) if parts else "1"


def congruence_tex(lhs: str, rhs: str, mod: int) -> str:
    """A congruence in TeX."""
    return rf"{lhs} \equiv {rhs} \pmod{{{int(mod)}}}"
