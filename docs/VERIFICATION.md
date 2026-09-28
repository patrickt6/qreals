# Verification

This page lists every check in the qreals test suite: what it proves, what
it does not prove, how to rerun it, and the current results.

Results below were measured on 2026-09-28 against the commit that added
this file.

| Measure | Value |
|---|---|
| Collected tests (`python -m pytest --collect-only -q`) | 1160 |
| Test functions (`def test_` in `tests/*.py`) | 116 |
| Python versions in CI | 3.11, 3.12, 3.13 |
| Mutation score, core math modules, first run | 87.3% (974 of 1116 mutants killed or timed out) |
| Mutation score, after adding tests for survivors | 90.0% (1004 of 1116) |

Most of the 1160 collected tests are one identity checked on each rational
of a fixed grid (for example the 114 fractions of the translation grid);
`tests/test_identities.py` alone contributes 968 of them. The test-function
count is the better measure of how many distinct checks exist.

## How to rerun

```bash
git clone https://github.com/patrickt6/qreals
cd qreals
pip install -e ".[dev]"
python -m pytest                               # the whole suite
HYPOTHESIS_PROFILE=thorough python -m pytest tests/test_properties.py   # 1000 random examples per property
```

CI runs `python -m pytest` on every push and pull request
(`.github/workflows/tests.yml`). The suite is deterministic: the default
Hypothesis profile (`ci`, in `tests/conftest.py`) is derandomized and keeps
no example database, so every run sees the same examples.

Mutation testing (about 6 to 10 minutes with 10 worker processes):

```bash
pip install mutmut
HYPOTHESIS_PROFILE=mutation mutmut run
mutmut results
```

The `[tool.mutmut]` table in `pyproject.toml` selects the modules and tests.
Run it from a clean clone or a separate git worktree, and from an
environment where qreals is *not* installed in editable mode, so the tests
import the mutated copy under `mutants/` and not the working tree.

## Sources

Every expected value in the known-value tests was copied from one of these,
opened and read for this suite:

- [MGO20] S. Morier-Genoud, V. Ovsienko, "q-deformed rationals and
  q-continued fractions", Forum Math. Sigma 8 (2020), arXiv:1812.00170v3.
- [MGO22] S. Morier-Genoud, V. Ovsienko, "On q-deformed real numbers",
  Experimental Mathematics 31 (2022), arXiv:1908.04365v3.
- [J25] P. Jouteur, "Symmetries of the q-deformed real projective line",
  arXiv:2503.02122v1.
- OEIS A004148 (Generalized Catalan numbers), b-file fetched 2026-09-28 from
  https://oeis.org/A004148/b004148.txt; the terms used are stored in
  `tests/data/A004148.txt`.

Section, example and equation numbers refer to the arXiv versions listed
above. The journal versions were not checked.

## The checks

### 1. Known values from the literature (`tests/test_known_values.py`)

| Check | Source |
|---|---|
| `[n]_q = 1 + q + ... + q^(n-1)` and `[n]_(q^-1)` | [MGO20] Introduction, Definition 1.1 |
| `[5/2]_q`, `[5/3]_q`, `[7/3]_q`, `[7/4]_q`, `[7/5]_q`, numerator and denominator coefficient by coefficient | [MGO20] Example 1.2 (b) |
| `[r/(r-1)]_q = [r]_q/[r-1]_q`, and the closed forms for denominators 2 and 3 | [MGO20] Example 1.2 (a), (c), (d) |
| Taylor series of `[7/5]_q` through `q^12` | [MGO22] Section 2.2 |
| Fibonacci convergents `[13/8]_q`, `[34/21]_q`, `[55/34]_q` and their series | [MGO22] Section 4.1 |
| First 21 to 40 coefficients of the golden ratio, silver ratio, sqrt(2), sqrt(3), sqrt(5), sqrt(7) and e | [MGO22] Sections 4.1 to 5.1 |
| First 80 coefficients of `[pi]_q`, including the zero at `q^45` | [MGO22] Section 5.2 |
| Golden-ratio coefficients equal `(-1)^k a(k-1)` with `a` = A004148, for k up to 60 | [MGO22] Proposition 4.2, OEIS A004148 |
| The quadratic functional equations of `[phi]_q`, `[1+sqrt(2)]_q`, `[sqrt(2)]_q`, `[sqrt(3)]_q`, `[sqrt(5)]_q`, `[sqrt(7)]_q`, as power-series identities through `q^79` | [MGO22] equations (14), (16) to (20) |

**Proves:** on these inputs the engine reproduces the published numbers
exactly. The functional-equation checks go well past the printed terms.

**Does not prove:** correctness on inputs nobody has published, or
correctness of the published values themselves (see the findings below).

### 2. Identities for every rational (`tests/test_identities.py`, `tests/test_properties.py`)

Each identity is checked exactly (`sp.cancel` of a difference of rational
functions) on a fixed grid of 114 reduced fractions `p/s` with
`0 < |p| <= 12`, `s <= 7` (57 of them positive, 39 above 1), and on 60 derandomized random fractions with
`|p| <= 40`, `s <= 12` (Hypothesis).

| Identity | Domain | Source |
|---|---|---|
| `[x+1]_q = q[x]_q + 1` | all rationals, negatives included | [MGO22] (3), [J25] (1) |
| `[-1/x]_q = -1/(q[x]_q)` | all nonzero rationals | [J25] (1), `S_q` of [MGO20] Definition 4.4 |
| `[-x]_q = -q^-1 [x]_(q^-1)` and `[1/x]_q = 1/[x]_(q^-1)` | all nonzero rationals | [J25] (4) |
| `[r/s]_q` at `q = 1` is `r/s`; `R(1) = r`, `S(1) = s` | all rationals | [MGO20] Section 1.1 |
| R and S have positive coefficients, with leading and constant coefficients 1 | `r/s > 1` | [MGO20] Proposition 1.3 |
| `R S' - S R'` has non-negative coefficients | `r/s > r'/s' > 1` | [MGO20] Theorem 2 |
| `det M(a) = q^(a1 - a2 + ... - a2m)` for the block product; `det M~+ = q^(a1 + ... + a2m)` | all even-length words from rationals | [MGO20] (1.1), [MGO22] (9) |
| The scaled matrix `M~+` has first column `(qR, qS)` up to a common factor and second column the previous convergent | `r/s > 1` | [MGO22] (9), (10) |
| Consecutive convergents: `R_n S_(n-1) - S_n R_(n-1) = (-1)^n q^(a1 + ... + a_(2 floor(n/2)) - 1)`, and the matching Taylor-series agreement | 7 irrationals, n up to 7 | [MGO22] Proposition 1.1, (11); see finding 2 |
| Gap theorem: `c_0 = ... = c_(k-1) = 1`, `c_k = 0` | `k <= x < k+1` | [MGO22] Theorem 2; see finding 3 |
| Negative-x Laurent shape `[x]_q = -q^(-k) + ...` | `-k <= x < 1-k` | [MGO22] (4) |
| Jouteur negation `N_q . [x]_q` equals the *left* q-rational `[-x]^flat_q` | positive rationals | [J25] Theorem 1.5, Definition 1.2 |
| `N_q . [x]_q` equals the MGO series `[-x]_q` | sqrt(2), sqrt(3), phi, pi, e | [J25] Theorem 1.5; `[-x]_q` built from [MGO22] (3) |
| `negate` is an involution | rationals and irrationals | [J25] Section 1.2 |

**Proves:** on every grid point and every generated example, the identity
holds exactly, with no tolerance.

**Does not prove:** the identity for all rationals. The grids and the
random draws are bounded (numerators up to 40, denominators up to 12). The
identities for irrational inputs are checked on the listed numbers only,
to 20 to 80 coefficients.

### 3. An independent reference implementation (`test_regular_equals_negative_continued_fraction`)

The package folds the *regular* continued fraction. The test suite contains
its own implementation of the *negative* (Hirzebruch-Jung) continued
fraction formula (1.2) of [MGO20], digits and fold both written from the
paper, sharing no code with the package. [MGO20] Theorem 1 says the two
q-deformations coincide, and the test checks it on every rational `> 1` in
the grid and on random ones.

This matters because the package's own alternative paths (below) all take
their 2x2 blocks from one function, `continuant.mgo_block`, and its Poly
twin `_poly_block`. A wrong block would pass every internal agreement check;
it would not pass this one, or the known values of section 1.

### 4. Agreement between the package's own code paths (`tests/test_properties.py`)

| Paths compared | Shared code |
|---|---|
| `q_rational` (sympy cancel of the block product), `q_rational_pair` (Poly product, gcd reduced), `continuant_symbolic`, `continuant_pair`, `gosper.q_real_rational` | all use the MGO block definition |
| `continuant_fast` (multiplication-only fold, one long division) against `continuant_series` (one series inversion per term) | block definition; different arithmetic |
| `q_real_truncated` on a rational against the Taylor expansion of the exact `q_rational` | the Taylor expansion is computed by long division written in `tests/conftest.py`, not by the package's series kernel |
| `q_add`, `q_mul` (coefficient lists) against `gosper_coeffs` (a 2x4 bihomographic state machine over rational functions) | block definition only |

**Proves:** the fast and general algorithms agree exactly; a bug in one
arithmetic layer (series inversion, Laurent padding, gcd reduction) shows up
as a disagreement.

**Does not prove:** the shared block definition; that is what sections 1
and 3 check.

### 5. Exact arithmetic

`test_outputs_are_exact`, `test_irrational_outputs_are_ints`,
`test_fast_returns_python_ints` and
`test_lpoly_multiplication_is_exact_integer_arithmetic` check that results
contain no `sp.Float`, that Poly results are over `ZZ`, and that every
coefficient returned by the truncated paths is a Python `int` (not a float
that happens to compare equal).

**Does not cover:** `radius` and `radius_estimate`, which return a float by
design (a root-test estimate, documented in `arithmetic.py`).

### 6. Kernel and continued-fraction contracts (`tests/test_kernel.py`)

Unit tests of the truncated-series kernel (`series.py`: trim, add,
multiply, Newton inversion with the Fraction fallback), the truncated
Laurent helpers of the fast fold, the `ArithmeticError` contract of
`continuant_fast` that `q_real_truncated` relies on for its fallback, the
even-length normalisation, the `cf_partials` stopping rule, and the error
cases (`x < 0`, zero denominators).

### 7. CLI, docs and other surfaces (`tests/test_cli.py`, `tests/test_docs.py`)

- Each of the nine `--help` worked examples (`denom`, `why`, `twin`,
  `bricks`, `conj`, `glue`, `witness`, `collapse`, `check`) is run with
  `python -m qreals`, and its stdout must equal the example byte for byte.
- The README worked examples (`denom 19/60`, `conj floor3 --until 24`) and
  the `docs/HOW-IT-WORKS.md` examples (`qint 4`, `coeffs "sqrt(2)" 12`,
  `denom 4/15 --json`) are run and their stated lines checked.
- `rational`, `coeffs`, `shift`, `negate`, `exact` and a small
  `negation-sweep` are run and their JSON checked against the library.
- The MCP tool registry (10 tools, 3 resources) and the REST endpoint
  `POST /api/v1/compute` (on the Flask mirror app) are exercised.
- `tests/test_docs.py` recomputes every row of the README "At a glance"
  table (module count, test count, Python versions against `pyproject.toml`
  and the CI matrix, MCP tool count, subcommand count) and requires the
  README tool-reference block to equal a fresh `python -m qreals.readme`
  render.

**Does not cover:** the interactive menu, `qreals serve` as a running
server (the app object is tested, not a live socket), the stdio MCP
transport (the tool functions are tested, not the protocol), the network
helpers (`oeis`, `oeis-sweep`, `explain`), certificate PDF output, and most
of the research subcommands (`hunt`, `dataset`, `satlas`, `sweep`, and
others) beyond the worked examples above.

## Mutation testing

mutmut 3.8.0 mutated `continuant.py`, `continued_fraction.py`,
`series.py`, `truncated.py` and `rational.py` (1116 mutants), running the
math test files (`test_known_values`, `test_identities`, `test_properties`,
`test_kernel`) under the `mutation` Hypothesis profile (20 examples per
property).

| | First run | After new tests |
|---|---|---|
| Killed | 969 | 999 |
| Timed out (counted as killed) | 5 | 5 |
| Survived | 138 | 112 |
| No covering test | 4 | 0 |
| Score, (killed + timed out) / total | 87.3% | 90.0% |

The first run exposed real gaps, and each got a test in the last section of
`tests/test_kernel.py`:

- The Laurent multiply `_lmul` could divide instead of multiply and still
  pass, because `2.0 == 2` in Python: the exactness tests now check types.
- `_lstrip` with an odd number of leading zeros.
- `series.mul` keeping a term at or beyond the precision.
- `series.add` truncating its second operand.
- `scalar_mul` by 1.
- `invert_general` with a positive valuation or with an operand longer than
  the working window.
- The `continuant_fast` domain guard: a later drift check also raised
  `ArithmeticError`, so a broken guard went unnoticed.
- `_mat2_mul` as a general 2x2 product.
- The canonical zero series.
- The periodic continued-fraction path beyond depth 250.
- `truncated.mgo_build_series`, which had no covering test.

The 112 survivors that remain were each read and classified (this is a
judgement, not a proof):

- 32 change only an error-message string.
- 80 are equivalent, or unreachable for valid input. They include:
  - `domain="ZZ"` removed, since sympy infers `ZZ` anyway;
  - index shifts that keep the parity, such as `(i - 1) % 2` for
    `(i + 1) % 2`;
  - padding and trim changes that a later trim erases;
  - swapping the operands of a commutative product;
  - the `else: c = 0` branch of the final long division in
    `continuant_fast`, which cannot run because the numerator is padded to
    `prec`;
  - the `lc != 1` normalisation in `q_rational_pair`, which never fires
    because sympy's gcd leaves a monic denominator;
  - routing a leading coefficient of -1 to the Fraction inverse, which
    gives the same value by a slower path.

  For a few of these the equivalence rests on an argument, not a proof. For
  example, `d0 = -1` in `continuant_fast` never occurs because of total
  positivity.

Excluding those 112 by that classification, every remaining mutant is
killed.

## Findings

The findings below concern the arXiv versions listed under Sources; the
journal versions may differ.

1. **No bug was found in the engine.** Every published value and every
   identity above holds exactly.
2. **Parity in [MGO22] Proposition 1.1.** In arXiv:1908.04365v3 the
   proposition gives that consecutive convergents `x_(n-1)`, `x_n` agree on `a1 + ... + an - 1`
   terms, with the cross determinant `q^(a1 + ... + an - 1)` (equation 11).
   The suite confirms this for even `n`, the case the proof's determinant
   argument covers. For odd `n` the printed q-rationals give a different sign
   and exponent: `[3/2]_q` ([MGO20] Example 1.2 (c)) and `[7/5]_q` ([MGO20] Example
   1.2 (b)), consecutive convergents of sqrt(2), have cross determinant
   `-q^2`, not `q^4` (`test_eq11_odd_case_from_printed_values`, which uses
   only the printed polynomials). The engine gives the same `-q^2`. The pattern the suite pins is
   `(-1)^n q^(a1 + ... + a_(2 floor(n/2)) - 1)`. It also matches the paper's
   remark that `[3, 7, 15, 1, 292]` fixes `[pi]_q` "up to degree 317". The
   package's truncation rule (`cf_partials`) still keeps at least `N` stable
   coefficients in both parities (`test_truncation_is_stable`).
3. **Endpoint of the gap theorem.** [MGO22] Theorem 2 and Proposition 6.2
   are written for `k <= x <= k + 1`. At `x = k + 1` the q-integer
   `[k+1]_q` has `c_k = 1`, so the suite tests the interval `k <= x < k + 1`. The suite checks `k <= x < k + 1` and
   pins the endpoint behaviour separately (`test_gap_theorem_right_endpoint`).
4. **Range of [MGO20] Example 1.2 (d).** The printed pattern
   opens with `1 + 2q + 3q^2`; at `m = 1` the values `4/3` and `5/3` are
   those of Example 1.2 (a) and (b), so the suite applies (d) for `m >= 2`.
5. **The Jouteur negation is not the MGO q-rational of `-x` on
   rationals.** Its value there is the *left* q-rational ([J25] Theorem
   1.5). The two differ already at `x = 1`: `-q^-2` against `-q^-1`. On
   irrationals the two agree. `arithmetic.q_neg` and `negate` implement the
   Jouteur formula and say so. `negate` on a negative rational input
   therefore starts from the left version, not from `q_rational(-p, s)`.

## Performance note

The README says the fast fold is about 300x faster than the
inversion-per-term fold at depth 512. No test pins this number. A single
measurement on 2026-09-28 (Python 3.11, macOS, 512 stable coefficients)
gave 310x for sqrt(2) (0.030 s against 9.28 s) and 537x for the golden
ratio (0.042 s against 22.3 s).

## What remains unverified

- Inputs outside the tested ranges: rationals with numerators above 40 or
  denominators above 12 in the random checks, and continued fractions with
  very large partial quotients other than pi's 292.
- The cyclotomic and denominator-structure tools (`denom`, `collapse`,
  `bricks`, `conj`, `why`, `twin`, `glue`, `witness`) are checked only
  through their worked examples and the README examples, not against an
  independent factorisation.
- The research heuristics (`finite_xnegx`, the negation-sweep
  `finite_looking` verdicts, `radius`) report finite-order observations. The
  suite runs them but does not check the mathematical claims they suggest.
- Modules outside the five mutated ones were not mutation tested.
- The speed ratio in the performance note is one measurement on one
  machine, not a test.
