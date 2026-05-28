# Changelog

All notable changes to this project are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-05-24

The first public release: the engine, separated from a larger research codebase
so it runs without Sage, with docs, types, CI, and PyPI trusted publishing.

### Added

- Documentation site built with mkdocs-material and mkdocstrings: a five-minute
  quickstart, the MGO construction in plain language, the correctness notes, and
  a full API reference generated from the docstrings.
- `py.typed` marker and complete type hints; the package passes `mypy --strict`.
- GitHub Actions CI: pytest on Python 3.11, 3.12, and 3.13 across Linux, macOS,
  and Windows, plus `ruff check`, `ruff format --check`, and `mypy --strict`.
- Release workflow that publishes to PyPI via Trusted Publishing (OIDC) on a tag.
- `CITATION.cff` and this changelog.

- `q_rational(p, s)`: the exact `[p/s]_q` as a reduced rational function in q,
  via the MGO continued-fraction formula.
- `q_real_truncated(x, N)`: the first N stable Taylor coefficients of `[x]_q` for
  any real x, with the stability guarantee of MGO Proposition 1.1.
- The truncated-series kernel (exact integer Laurent series with Newton
  inversion) and the continued-fraction utilities behind both paths.
- Arithmetic between q-reals: `q_add`, `q_mul`, the Jouteur `q_neg`,
  `negation_sum`, `finite_xnegx` (Ovsienko Example 6.4), and `radius`.
- The bihomographic q-Gosper engine (`q_gosper`, `gosper_coeffs`), an
  independent algorithm used to cross-check the arithmetic.
- The `QReal` convenience wrapper with read-outs and operators.
- The May-14 board lemmas as functions: `integer_part_prefix`,
  `coeffs_locked_by_convergent`, `mgo_laurent`, `shift_down`, `shift_up`,
  `format_laurent`, and the coefficient read-outs.
- The inline verification stamp (`verify`, `Stamp`): cheap independent
  cross-checks printed with every result.
- Human-auditable certificates (`qreals certify`), terminal, PDF, or saved
  `.tex`, behind the `proof` extra, with an optional qprov bridge.
- Two optional exploration helpers behind light extras: `oeis.lookup` (OEIS
  search with b-file re-verification) and `featurize` (a fixed-length
  fingerprint of `[x]_q`).
- A guided arrow-key interface and a headless scripting CLI (`qreals`), with a
  `doctor` environment check.
- `docs/CORRECTNESS.md`: for every public function, the theorem it computes and
  the independent check that confirms it.
- A saved-results workflow. The interactive app can add a computed result to a
  personal list ("My saved list"), view it, and remove items; the list persists
  across sessions under the per-user data directory (via platformdirs, with a
  standard-library fallback to the same per-OS location), never the working
  folder. Each entry keeps the input, the order N, the coefficients, and a
  timestamp. New module `qreals.store` (`SavedStore`, `SavedEntry`,
  `user_data_dir`); the `QREALS_DATA_DIR` environment variable overrides the
  location.
- Exports for one result or the whole list to JSON, CSV (one row per constant),
  a booktabs LaTeX table ready to paste into a paper, and Magma code that
  rebuilds each value as a Laurent series in q over the rationals. New module
  `qreals.exports`. Files are written only on an explicit export; an interactive
  export defaults to the current directory only after a confirm.
- Headless `qreals batch "pi,sqrt(2),3/2" --order N --format json|csv|latex|magma`
  to compute a list of constants into one file, `qreals export` to write the
  saved list, and `qreals saved` to list, remove from, or clear it. Without
  `-o`, batch and export print to stdout and write nothing.
- An optional, off-by-default qprov link (`qreals.provenance`): when qprov is
  importable, an exported item or list can carry a qprov id tying the saved
  number to a recorded run. The core never imports qprov; the import is lazy.
- `jumpgap(p, s)`: the two q-versions of a rational, the right version
  `[p/s]_q^+` (the limit from above, equal to `q_rational(p, s)`) and the left
  version `[p/s]_q^-` (the limit from below), with the factored gap between them
  and its closed-form factors: the exponent E and the right and left
  q-denominators S^+ and S^- (`gap = (1 - q) q^E / (S^+ S^-)`, Jouteur
  arXiv:2503.02122). New module `qreals.jumpgap` (`jumpgap`, `JumpGap`). A
  "Gap between a rational's two q-versions" menu entry and the headless
  `qreals jumpgap p s`; the right version is cross-checked against the
  `q_rational` oracle and both denominators against s at q = 1, shown in the
  verification stamp.
- `deficit(x, y, op, N)`: the gap `[x op y]_q - (engine value)` for op `+` or
  `*`, where the engine value is the series sum `[x]_q + [y]_q` or product
  `[x]_q * [y]_q`. Holds `[x]_q`, `[y]_q`, the engine value, the target
  `[x op y]_q`, the deficit, and its values at q = 1 and q = 0; for rational
  inputs it adds the exact closed form, so `deficit("3/2", "5/2", "+", N)` reads
  `q^3 - 1` (D(1) = 0, D(0) = -1). Reuses the verified `q_add`, `q_mul`,
  `q_real_truncated`, and the bihomographic `gosper` engine with `q_rational`.
  New module `qreals.deficit` (`deficit`, `Deficit`, plus `negation_panel` /
  `NegationPanel` bundling `negation_sum` and `finite_xnegx` for Ovsienko
  Example 6.4). A "Deficit of two q-reals" and a "Negation sum" menu entry, and
  the headless `qreals deficit x y N` and `qreals negsum x N`.

### Fixed

- Certificate PDFs no longer run long Laurent polynomials off the page. Each
  polynomial is capped at `max_terms` monomials (head, `\dots`, tail, and a
  "(K terms omitted)" note), displays break to the text width with breqn (or an
  allowbreak fallback), long ratios split into `N(q)` and `D(q)`, and very long
  coefficient series print as a paginating two-column longtable. The full exact
  object is unchanged in the terminal, JSON, and coefficient read-outs. A
  guardrail test compiles the worst cases and fails on any overfull box past
  5pt, skipping when no TeX engine is on PATH.

[Unreleased]: https://github.com/patrickt6/qreals/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/patrickt6/qreals/releases/tag/v0.1.0
