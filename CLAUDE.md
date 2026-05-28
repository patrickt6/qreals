# CLAUDE.md - qreals

## What this repo is

`qreals` computes q-deformed rational and real numbers via the
Morier-Genoud-Ovsienko (MGO) continued-fraction construction. It exposes two
entry points: `q_rational(p, s)` returns the exact `[p/s]_q` as a rational
function in q, and `q_real_truncated(x, N)` returns the first N stable Taylor
coefficients of `[x]_q` for any real x. The math follows MGO, "q-deformed
rationals and q-continued fractions", Forum Math. Sigma 8 (2020), Definition
1.1 and Proposition 1.1.

It is the standalone pure-Python form of the MGO construction, separated from
a larger research codebase (where the same construction has a Sage
implementation) so that it runs without Sage.

## Scope

- In scope: `q_rational` (exact), `q_real_truncated` (stable series), the
  truncated-series kernel, the coefficient read-outs, arithmetic between q-reals
  (`q_add`, `q_mul`, the Jouteur `q_neg`, `finite_xnegx`, `radius`), the
  bihomographic `gosper` engine that cross-checks them, and the `QReal` wrapper.
  Every such computation is pinned to an independent path and documented in
  `docs/CORRECTNESS.md`.
- Also in scope: two optional exploration helpers, each behind a light extra and
  importing nothing heavy into core. `featurize` (`features.py`) builds a named,
  fixed-length fingerprint of `[x]_q` for nearest-neighbour over constants (pure
  Python; numpy only for `as_numpy`, extra `features`). `oeis.lookup`
  (`oeis.py`) searches the OEIS for a coefficient sequence and re-verifies hits
  against the b-file (extra `oeis`, needs `requests`; lazy import, no network in
  core). No torch, no transformers, no model training.
- Out of scope: provenance and reproducibility tracking (that is the separate
  `qprov` project), a Sage backend, plotting, anything that is research output
  rather than the engine itself.

## Working in this repo

- Conventional commits, lowercase, present tense.
- One logical change per commit. No `git add -A`; stage specific files.
- Python 3.11+. The only runtime dependency is `sympy`; dev adds `pytest`
  and `ruff`.
- Tests run via `python -m pytest tests/ -v`.
- No banned words in commits, README, or docs: leverages, seamlessly,
  robust, cutting-edge, state-of-the-art, comprehensive, streamlined,
  harnesses, powerful, intuitive, sophisticated, novel, innovative,
  revolutionize, unlock.
- No em-dashes (U+2014) or en-dashes (U+2013) in any tracked file. Use
  hyphens, commas, or rephrase.

## Public API surface

The source of truth for what exists is `src/qreals/__init__.py`. Anything not
re-exported there is internal and may change without notice.

## Layout

```
qreals/
  src/qreals/      package source
  tests/           pytest suite
  example/         runnable end-to-end demo
  README.md        user-facing docs
  CLAUDE.md        this file (context for AI coding agents)
```
