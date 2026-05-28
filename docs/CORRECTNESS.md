# Correctness of qreals

This document states, for every public function of `qreals`, the theorem it
computes, the algorithm it runs, and the independent check that confirms it.
The aim is that no public result rests on the implementation alone: each is
pinned to a published theorem and to a test that recomputes the same value a
second way.

## Source papers and a citation note

Two papers define the mathematics. The package README and `CLAUDE.md` name
only the first; the stability and series results actually live in the second,
and this document cites them precisely.

- **RAT**: S. Morier-Genoud and V. Ovsienko, "q-deformed rationals and
  q-continued fractions", Forum Math. Sigma 8 (2020), e13 (arXiv:1812.00170).
  Defines `[p/s]_q` and proves its algebraic properties.
- **REAL**: S. Morier-Genoud and V. Ovsienko, "On q-deformed real numbers"
  (arXiv:1908.04365). Builds `[x]_q` as a power series for real x and proves
  the stabilisation that `q_real_truncated` depends on.

Correction to the package docs: the "Proposition 1.1" that the README and the
`continued_fraction` docstring cite for the stable-coefficient bound is
Proposition 1.1 of **REAL**, not of the Forum Math. Sigma paper. The
existence of the limit series is **REAL** Theorem 1. Definition 1.1 of the
construction is from **RAT**. The code is correct; only the paper label was
imprecise.

## How verification is organised

- `tests/test_rational.py`, `tests/test_truncated.py`, `tests/test_series.py`,
  `tests/test_expansions.py`: anchored unit tests (pre-existing).
- `tests/test_mgo_paper.py`: every worked example transcribed from RAT and
  REAL, asserted against the code. The reproduced numbers come from the
  papers, not the engine.
- `tests/test_properties.py`: Hypothesis property tests and the
  paper-theorem invariants (RAT Corollaries 1.4, 1.7, Proposition 1.8; REAL
  Theorems 1, 2 and the functional equations).
- `tests/test_gosper.py`, `tests/test_arithmetic.py`, `tests/test_qreal.py`:
  the q-real arithmetic (series sum and product, the Jouteur negation and its
  x -> -x finiteness, and the radius estimate), each pinned to an independent
  path - the bihomographic engine, the negation's own involution, or the exact
  nearest-pole modulus of a rational. Documented per function below.

The strongest independent check is the set of **functional equations** in REAL
Section 4. A series `[x]_q` for a quadratic irrational satisfies a quadratic
`A(q) [x]_q^2 + B(q) [x]_q + C(q) = 0`. Substituting the computed coefficients
and confirming the residual vanishes tests the output without re-running the
continued-fraction construction, so a construction bug cannot hide behind it.

---

## `q_rational(p, s)`

**Theorem.** For coprime integers `r/s > 1`, `[r/s]_q = R(q)/S(q)` is the
reduced rational function of RAT Definition 1.1, with `R, S` the unique coprime
polynomials with positive leading terms. (RAT Definition 1.1, eqns 1.1-1.3.)

**Algorithm.** Take the regular continued fraction of `p/s`, pad it to even
length, and fold RAT eqn (1.1) from the innermost term outward, alternating
`[a]_q` with `[a]_{q^{-1}}` and numerators `q^{a}` with `q^{-a}`; cancel.

**Independent checks.**
- `q=1` specialisation equals `p/s` exactly (RAT Corollary 1.7(iii):
  `R(1)=r`, `S(1)=s`). Property test over random `p, s`.
- RAT Corollary 1.7: `R(0)=S(0)=1`, leading coefficients 1, and (RAT
  Proposition 1.3) all coefficients positive. Property test.
- RAT Proposition 1.8: `R(-1), S(-1)` lie in `{-1, 0, 1}`, with `R(-1)=0` iff
  `r` is even and `S(-1)=0` iff `s` is even. Property test.
- RAT Corollary 1.4: for Farey neighbours (`r s' - r' s = +-1`) the determinant
  `R S' - S R'` is a single power of q. Parametrised test.
- The closed forms of RAT Introduction and Example 1.2(a)-(d) reproduced
  verbatim. Regression test.

---

## `q_real_truncated(x, N)`

**Theorem.** For real `x`, the Taylor coefficients at `q=0` of the q-deformed
convergents `[x_n]_q` stabilise; the limit coefficients are integers
independent of the approximating sequence (REAL Theorem 1). The n-th
convergent fixes the first `S_n - 1` coefficients, where `S_n = a_1 + ... + a_n`
is the partial-quotient sum (REAL Proposition 1.1).

**Algorithm.** Accumulate partial quotients of `x` until `S_n >= N + 1`, pad to
even length, and fold RAT eqn (1.1) over a truncated integer-coefficient
Laurent-series kernel (Newton inversion); read off `c_0 .. c_{N-1}`. Since
`S_n >= N + 1`, at least `S_n - 1 >= N` coefficients are stable.

**Independent checks.**
- On rationals the continued fraction terminates, so the output must equal the
  Taylor expansion of `q_rational(p, s)`; checked coefficient for coefficient
  by a second sympy series computation. Property test.
- Truncation stability: the first `N` coefficients do not change when more are
  requested (REAL Theorem 1). Property test over pi, e, sqrt(2), sqrt(3), the
  golden ratio, and rationals.
- The explicit series of REAL Section 4 for the golden ratio, the silver
  ratio, sqrt(2), sqrt(3), sqrt(5), sqrt(7) reproduced verbatim. Regression
  test.
- The functional equations REAL (14), (16), (17)-(20) hold for the computed
  series to high order. Algorithm-independent regression test.

**Unproven within the tests.** The stabilisation guarantee is for the infinite
limit; the tests confirm it only up to finite `N` (consistency between two
truncation lengths), and rely on REAL Theorem 1 and Proposition 1.1 for the
unbounded claim. For transcendental `x` the partial quotients come from
sympy's `continued_fraction_iterator`, so correctness at the depth used
inherits sympy's numerical continued-fraction accuracy for that constant.

---

## `mgo_laurent(x, order)`

**Theorem.** The Laurent coefficients of `[x]_q` from `q^0` through `q^order`
(REAL construction, same object as `q_real_truncated`).

**Algorithm.** Returns `q_real_truncated(str(x), order + 1)` after validating
`order >= 0`.

**Independent checks.** Equal to `q_real_truncated(x, order + 1)` by
construction (property test across several x and orders); the RAT/REAL worked
values flow through `q_real_truncated` above; the board value
`[22/7]_q = 1 + q + q^2 + q^9 + O(q^10)` is a regression test.

---

## `integer_part_prefix(x)`

**Theorem.** For real `x >= 0` with `t = floor(x)`, the series `[x]_q` opens
with `t` coefficients equal to 1 (spelling `[t]_q`) and a forced 0 at `q^t`
(REAL Theorem 2, the gap theorem). The fractional part can perturb only
`q^{t+1}` and higher.

**Algorithm.** Return `[1] * t + [0]` for `t = floor(x)`; raise for `t < 0`,
where the gap theorem is not stated.

**Independent check.** The returned prefix equals the actual leading
coefficients computed by `q_real_truncated(x, t + 1)`. Property test over
random rationals.

---

## `coeffs_locked_by_convergent(cf_terms, n)`

**Theorem.** For `x = [a_1, a_2, ...]` with partial sum `S_n = a_1 + ... + a_n`,
two consecutive q-deformed convergents share their first `S_n - 1` coefficients
and differ by 1 at `q^{S_n - 1}` (REAL Proposition 1.1). Returns
`(S_n, S_n - 1)`.

**Algorithm.** Validate `1 <= n <= len(cf_terms)`, sum the first `n` terms,
return `(S_n, S_n - 1)`.

**Independent check.** The lemma's content is verified against real series in
`tests/test_expansions.py`: for x = pi the convergent `[3,7] = 22/7` agrees
with `[pi]_q` on exactly `S_2 - 1 = 9` coefficients and first differs at
`q^9`. The off-by-one ("below `q^{S_n}`" on the source board) was wrong; the
proven and tested bound is `S_n - 1`, which this function returns.

---

## `shift_down(coeffs)` and `shift_up(coeffs)`

**Theorem.** The translation action `[x+1]_q = q [x]_q + 1` and
`[x-1]_q = ([x]_q - 1)/q` (REAL eqn 3). `shift_up` raises the argument by one,
`shift_down` lowers it.

**Algorithm.** `shift_up(c) = [1] + c`. `shift_down(c) = c[1:]`, valid only when
`c[0] == 1` (the constant term of any `[x]_q` with `x >= 1`, forced by the gap
theorem); otherwise it raises.

**Independent checks.**
- `shift_down(shift_up(c)) == c` for any coefficient list. Property test.
- A shift chain matches direct computation: applying `shift_down` k times to
  `[x]_q` equals `q_real_truncated(x - k, ...)` for x with `x - k >= 1`.
  Parametrised test over pi, sqrt(2)+2, e+1, (7+sqrt(5))/2.
- `shift_down` raises on an empty list and on a non-unit constant term.

---

## `format_laurent(coeffs)`

**Theorem.** Rendering only: a coefficient list as a readable q-polynomial with
an `O(q^N)` tail, `N = len(coeffs)`.

**Algorithm.** Join nonzero terms with sign handling, append `+ O(q^N)`.

**Independent check.** Round trip: reparsing the rendered body with sympy and
reading the coefficients back recovers the input list. Property test over
random integer lists; exact strings checked in `tests/test_expansions.py`.

---

## `q_int(n)` and `q_int_qinv(n)`

**Theorem.** The Gauss q-integer `[n]_q = (q^n - 1)/(q - 1) = 1 + q + ... +
q^{n-1}` for `n > 0`, extended to `n < 0` by `[-m]_q = -[m]_q / q^m`; and its
`q -> q^{-1}` form `[n]_{q^{-1}}`. (RAT and REAL, Gauss convention.) These are
the building blocks of the continued-fraction folds.

**Algorithm.** `q_int(n)`: geometric sum for `n > 0`, sign-and-rescale recursion
for `n < 0`. `q_int_qinv(n) = [n]_q / q^{n-1}` for `n > 0`, recursion for `n < 0`.

**Independent checks.**
- `q_int(n)` equals the closed form `(q^n - 1)/(q - 1)` for `n > 0` and
  specialises to `n` at `q = 1`. Property test over `-12 <= n <= 12`.
- `q_int_qinv(n)` equals `q_int(n)` with `q` replaced by `1/q`, and specialises
  to `n` at `q = 1`. Property test.

---

## Coefficient read-outs

`first_nonzero_coefficient_index`, `first_negative_coefficient_index`,
`coefficient_max_abs`, `number_of_zeros`.

**Theorem.** None; these are list summaries used for pattern hunting over a
computed coefficient sequence.

**Algorithm.** A single pass each: first index matching a predicate (or `-1`),
maximum absolute value (or `0`), count of zeros.

**Independent check.** Each is compared against a separate brute-force
comprehension over random integer lists. Property test.

---

## Arithmetic between q-reals: the bihomographic engine (`gosper`)

A third paper underlies the arithmetic functions:

- **NEG**: A. Jouteur, "Modular group action on q-deformed real numbers"
  (arXiv:2503.02122). Gives the PGL_2(Z) action on q-reals, in particular the
  negation used below (its eq. 2). The finiteness question this enables is
  Ovsienko, "Modular invariant q-deformed numbers: first steps", Example 6.4.

The engine in `gosper` is the q-analogue of the classical bihomographic
continued-fraction state machine. It is an independent route to `[x]_q + [y]_q`
and `[x]_q * [y]_q`: it never forms the two series separately, so its agreement
with the series functions below is a genuine cross-check, not a re-run.

**Theorem (what the engine computes).** With the 2x4 state for the bilinear map
`z(X,Y)`, ingesting the continued-fraction quotients of x and y by
right-multiplication by `A_q (x) I` and `I (x) A_q`, the first-column ratio of
the final state is `z([x]_q, [y]_q)`. For the addition coefficients this is
`[x]_q + [y]_q`; for the multiplication coefficients it is `[x]_q * [y]_q`.

**Justification (Gosper / Kronecker factorisation).** The classical proof's only
ring-level inputs are the Kronecker mixed-product property
`(A (x) B)(C (x) D) = AC (x) BD` and the continuant identity, both valid over
`Z[q, q^{-1}]`. Replacing the classical block `A(t) = [[t,1],[1,0]]` by the MGO
q-block changes only one thing the proof flags: the single-step ingestion
identity (q-Lemma 1), because the off-diagonal is now `q^{+-a}` rather than 1.
That identity holds for a fully general block `[[alpha, beta],[1,0]]`: it sends
`u |-> alpha + beta/u`, and specialising `(alpha, beta) = ([a]_q, q^{a})`
recovers the MGO recursion. With it,
`(prod P_q)(prod Q_q) = M_x^{(q)} (x) M_y^{(q)}` and the first column is the
tensor of the two convergent vectors, so the leading ratio is `z([x]_q, [y]_q)`.

**What is proven (symbolically, in the tests).**
- `tests/test_gosper.py`: the q-block performs the substitution
  `u |-> [a]_q + q^{a}/u` (odd) and `[a]_{1/q} + q^{-a}/u` (even); the two
  ingestion sides commute (`P_q Q_q = Q_q P_q`); and the Kronecker factorisation
  `(prod P_q)(prod Q_q) = M_x (x) M_y` holds with the first column equal to the
  tensor of convergent vectors. These are exact sympy identities over `Z[q]`.
- The one-variable block product reproduces `q_rational` (so the engine is
  attached to the right object): `q_real_rational(p/s) == q_rational(p, s)`.

**The load-bearing caveat.** For "add" the engine returns `[x]_q + [y]_q`, the
sum of the two q-series, NOT `[x+y]_q`, the q-deformation of the real sum. The
MGO map `x |-> [x]_q` is not additive: each `[x]_q` has constant term 1 for
`x > 1`, so the sum has constant term 2 while `[x+y]_q` has constant term 1, and
they already differ at `q^0`. The deficit `D = [x+y]_q - ([x]_q + [y]_q)` is a
separate object; `gosper` does not compute it and does not claim to.

---

## `q_add(x, y, N)` and `q_mul(x, y, N)`

**Theorem.** The first N Taylor coefficients of the series sum `[x]_q + [y]_q`
and series product `[x]_q * [y]_q`, for real `x, y >= 0`. By definition of
power-series addition and multiplication these are the term-by-term sum and the
Cauchy product (convolution) of the coefficient sequences of `[x]_q` and
`[y]_q`, each of which is the verified output of `q_real_truncated`.

**Algorithm.** Compute `q_real_truncated(x, N)` and `q_real_truncated(y, N)` and
combine: a coordinate-wise sum for `q_add`, a truncated convolution for `q_mul`.

**Independent checks.**
- The bihomographic `gosper` engine, a different algorithm, returns the same
  coefficients on the rational test family (10 pairs, to `q^19`): `q_add` equals
  `gosper_coeffs(..., "add", N)` and `q_mul` equals `gosper_coeffs(..., "mul",
  N)`. Parametrised test in `tests/test_arithmetic.py`; the engine is itself
  pinned to the package ground truth above.
- For irrational inputs (pi, e, sqrt(3), the golden ratio) the result equals the
  series combination of two independent `q_real_truncated` calls. Parametrised
  test.

**What is not claimed.** These are not `[x+y]_q` or `[x*y]_q`; a regression test
asserts the gap explicitly (constant term 2 for `q_add("3/2","5/2")` against
constant term 1 for `[4]_q`). The size of that gap is `deficit`, below.

---

## `deficit(x, y, op, N)` and `negation_panel(x, N)`

**What it is.** `deficit` names the gap that `q_add` and `q_mul` are careful not
to claim away. For `op` in `{"+", "*"}` and real `x, y >= 0` it returns

`D = [x op y]_q  -  (engine value)`,

where the engine value is the series sum `[x]_q + [y]_q` for `"+"` and the series
product `[x]_q * [y]_q` for `"*"`, and the target `[x op y]_q` is the genuine
q-series of the real number `x op y`. Because the MGO map `x |-> [x]_q` is not a
ring homomorphism, `D` is not zero; it is the exact measure of how far the engine
sits from the q-real of the combined argument.

**Algorithm and reuse.** Nothing new is computed. The result reuses the verified
paths above: `q_real_truncated` for `[x]_q`, `[y]_q`, and the target
`[x op y]_q`; `q_add` / `q_mul` for the engine value; and the deficit series is
their coefficient-wise difference. When both inputs are rational the bihomographic
`gosper` engine returns the engine value as an exact rational function in q and
`q_rational` returns the exact target, so the deficit is given in closed form:
`deficit("3/2", "5/2", "+", N)` reads `q^3 - 1`.

**Invariants of the sum deficit (the q=1 and q=0 checks).**
- `D(1) = 0`. At `q = 1` every q-real collapses to its ordinary value (RAT
  Corollary 1.7), so the engine value and the target are the same real number and
  their difference is 0. This holds for both operations.
- `D(0) = -1` for a sum of `x, y >= 1`. The gap theorem (REAL Theorem 2) forces
  constant term 1 on each `[.]_q`, so the series sum opens with constant term 2
  while `[x+y]_q` opens with 1, and `D(0) = 1 - 2 = -1`. (For a product of such
  `x, y` the constant terms multiply to 1, matching the target, so the product
  deficit has `D(0) = 0`.)

For the `(3/2, 5/2)` sum these give `q^3 - 1`, whose head is `-1, 0, 0, 1, 0, 0`,
with `D(1) = 0` and `D(0) = -1`.

**Independent checks.**
- The truncated-series deficit equals the Taylor expansion of the exact closed
  form, computed a second way from `gosper` and `q_rational`, on a family of
  rational pairs for both operations. Parametrised test in `tests/test_deficit.py`.
- Both invariants are asserted: `D(1) = 0` for every pair and operation; `D(0)`
  is `-1` for sums and `0` for products of inputs `>= 1`. The `(3/2, 5/2)` closed
  form `q^3 - 1` is a regression test.
- For irrational inputs only the truncated series is available: `exact` is `None`
  and `D` at `q = 1` is reported as unavailable (a truncated series is not
  summable at `q = 1`, though `D(1) = 0` still holds in closed form), while the
  `q = 0` value is the deficit's constant coefficient. Asserted directly.

**`negation_panel(x, N)`.** A presentation helper, not a new claim: for one real
`x >= 0` it bundles `negation_sum(x, N)` (the sum `[x]_q + [-x]_q`, with its
valuation since `[-x]_q` carries negative powers of q) and the `finite_xnegx(x)`
verdict (finite iff `x` is a pure square root, Ovsienko Example 6.4), both
verified in their own section below. Tested in `tests/test_deficit.py`
(`sqrt(2)` finite with sum `-q^{-2} + q`, the golden ratio infinite).

---

## `q_neg(x, N)`, `negation_sum(x, N)`, `finite_xnegx(x)`

**Theorem (negation).** For `A = [x]_q`, the q-deformed negation of NEG eq. 2 is

`[-x]_q = (-A + 1 - q^{-1}) / ((q - 1) A + 1)`,

a Laurent series in q. As a Mobius map in A it has matrix
`[[-1, 1 - q^{-1}], [q - 1, 1]]`, whose square is `(q - 1 + q^{-1}) I`; the map
is therefore an involution, `[-(-x)]_q = [x]_q`.

**Algorithm.** Build `A` as a kernel series from `q_real_truncated(x, N + 12)`,
apply the formula with the truncated-series operations (add, multiply, invert),
and return the result as `(valuation, coeffs)` since it carries negative powers.

**Independent checks.**
- Involution back to the brute-force path: applying the negation twice returns
  `q_real_truncated(x, N)` exactly, on a family of rationals (this is the
  cross-check against the independent series path; the negation has its own
  algorithm so the round-trip is not built in). Test in `tests/test_arithmetic.py`.
- The closed identity `[x]_q + [-x]_q = ((q-1)A^2 + (1 - q^{-1}))/((q-1)A + 1)`
  and the involution are proven symbolically over a free symbol `A` with sympy.
- A worked value: `[-2]_q = -q^{-1} - q^{-3}` (regression test).

**Theorem (finiteness, Ovsienko Example 6.4).** `[x]_q + [-x]_q` is a finite
Laurent polynomial iff x is a trace-zero quadratic irrational, i.e. a pure
square root `sqrt(D)`. For such x, `-x` is the Galois conjugate and the sum is
the q-trace (the sum of the two q-real branches), which is finite; for any other
x the sum is a non-terminating Laurent series.

**What is proven vs only checked.** The closed identity above is proven
symbolically. The criterion (finite iff pure square root) is the cited result;
its proof for the quadratic cases is the q-trace identity in the research module
`computations/q_gosper/negation_finiteness.py`. `finite_xnegx` itself is an
**operational, finite-order observation**: it computes the sum to order N and
reports "finite" when the coefficients past the leading block are a long run of
zeros. It is honest about being a numerical termination test, not a proof of
termination; the proof is the criterion it is checked against. Tests confirm it
returns True on `sqrt(2), sqrt(3), sqrt(5), sqrt(6), sqrt(7)` and False on the
golden ratio, the silver ratio `1+sqrt(2)`, the bronze ratio, `5/7`, and pi -
exactly the trace-zero split.

---

## `negate(x, N)`

**Theorem.** The Jouteur q-negation of NEG eq. 2 takes A = [x]_q to

`[-x]_q = (-A + 1 - q^{-1}) / ((q - 1) A + 1)`,

a Laurent series in q, the PGL_2(Z)-action negation on q-reals. It is an
involution: `[-(-x)]_q = [x]_q`. `negate` is the same construction as `q_neg`
extended to accept any real x; for x < 0 the q-real [x]_q is itself the Jouteur
image of [|x|]_q, so a single application of the formula returns [-x]_q = [|x|]_q.

**Source.** A. Jouteur, "Modular group action on q-deformed real numbers",
arXiv:2503.02122, eq. (2).

**Algorithm.** Build `A = [x]_q` as a kernel Laurent series via
`q_real_truncated(|x|, N+12)` and, when x < 0, one application of `_jouteur_neg`;
apply the formula again over the truncated-series operations and pad to N
coefficients.

**Independent checks (test_q_sum.py).**
- For x >= 0, `negate(x, N)` equals the existing verified `q_neg(x, N)` on
  rationals (3/2, 5/2, 7/3), pi, sqrt(2), and the golden ratio.
- For x < 0, the involution check: `negate(-p/s, N)` equals
  `q_real_truncated(p/s, N)` exactly, on 3/2, 5/2, 7/3 (parametrised).
- The regression value `[-2]_q = -q^{-1} - q^{-3}` is asserted from the
  reconstructed Laurent expression.

---

## `transfer_matrix(cf)`

**Theorem.** The 2x2 MGO q-continuant block product for a regular continued
fraction. Each digit a contributes the q-block

`T_q^{(i)}(a) = [[ [a]_q, q^{a} ], [1, 0]]` (1-indexed odd) or
`[[ [a]_{1/q}, q^{-a} ], [1, 0]]` (1-indexed even),

following the MGO recursion of RAT eqn (1.1). The first column of the product
is `(R_x, S_x)` with `[x]_q = R_x / S_x`. (Same block as the bihomographic
`gosper` engine; same form as the meeting summary's `T_q(a)`.)

**Algorithm.** Even-length-normalise the CF (`continued_fraction.make_even_length`)
and right-multiply the q-blocks in order.

**Independent checks (test_q_sum.py).**
- `transfer_matrix(q_cf(fr))` equals `gosper.q_convergent_matrix(q_cf(fr))` as
  a 2x2 sympy matrix, for fr in {3/2, 5/2, 7/3, 13/5}.
- First-column ratio reproduces `q_rational(p, s)` for (p, s) in
  {(3,2), (5,2), (7,3), (13,5), (22,7)}; this is the same agreement the engine
  already records, here exposed as the explicit transfer matrix.

---

## `q_sum_rational(x, y)`

**Theorem.** For rational x, y > 0 with the bihomographic state ingested by
right-multiplication by `q_block (x) I` for x-quotients and `I (x) q_block`
for y-quotients (the addition coefficients (0, 1, 1, 0; 0, 0, 0, 1)), the
first-column ratio of the 2x4 final state is `[x]_q + [y]_q`. (Same Kronecker
factorisation argument as the bihomographic engine for "add"; the algorithm
the 2026-05-25 meeting attributed to Alex.)

**Caveat.** The raw R, Q the algorithm produces are not always reduced; only
the cancelled value is the genuine q-number sum. The result carries both
`raw_numerator, raw_denominator` and `reduced_numerator, reduced_denominator`,
plus a plain-text `caveat` field stating it.

**Independent checks (test_q_sum.py).**
- The reduced value's Taylor coefficients to order 16 equal
  `gosper_coeffs(x, y, "add", 16)` on the rational test family (3/2, 5/2),
  (7/3, 5/2), (13/5, 3/2), (11/5, 8/3), (5/2, 5/2).
- The regression value `q_sum_rational(3/2, 5/2).value = q^2 + q + 2`,
  matching the q-Gosper engine's exact rational function.
- The caveat is asserted to mention "not always".

---

## `q_sum_irrational(x, y, N)`

**Theorem.** For x, y > 1, the regular below-convergents x_n -> x, y_n -> y
satisfy `[x_n]_q + [y_n]_q -> [x]_q + [y]_q` as formal Laurent series in q.
(The limit-of-the-sum-equals-sum-of-the-limits step the 2026-05-25 meeting
agreed is independent of the algorithm itself; the rational algorithm
`q_sum_rational` gives each `[x_n]_q + [y_n]_q` exactly.)

**Algorithm.** For n = 2, 4, ..., compute the below-convergent of x and y at
CF index n, run `q_sum_rational` on the rational pair, and watch the first N
Taylor coefficients. Stop when the coefficients are unchanged between two
consecutive even depths; return the approximant sequence and the stable
coefficient list.

**Independent checks (test_q_sum.py).**
- 4 +/- sqrt(7), the worked example of the meeting: the first 16 Taylor
  coefficients equal the right-hand side of the meeting identity
  `[4 + sqrt(7)]_q + [4 - sqrt(7)]_q = q^4 ([sqrt(7)]_q + [-sqrt(7)]_q) + 2 (1
  + q + q^2 + q^3)`, with `negation_sum("sqrt(7)", 16)` supplying the bracket
  on the right.
- Stabilisation: `stabilised_at` is set in the test window for the
  4 +/- sqrt(7) input.

**Unproven within the tests.** Convergence in the ring of formal Laurent series
is the meeting's "still has to be verified" point; the tests confirm only that
the truncated Taylor coefficients have settled by the chosen even depth.

---

## `finiteness_check(x, y, N)`

**Theorem-anchored test.** Returns a verdict on whether `[x]_q + [y]_q` is a
finite Laurent polynomial in `C[q, q^{-1}]`. For rational x, y the value is
exact (`q_sum_rational`); the verdict is "finite" iff the cancelled
denominator is a monomial in q. For irrational x, y the function runs the
convergent iterator to even depth ~N and reports "finite" when the truncated
Taylor coefficients show a long trailing run of zeros, an operational,
finite-order observation in the same spirit as `finite_xnegx`. The
cross-check is the meeting reduction `[4 + sqrt(7)]_q + [4 - sqrt(7)]_q =
q^4 ([sqrt(7)]_q + [-sqrt(7)]_q) + 2 (1 + q + q^2 + q^3)`, which makes the
4 +/- sqrt(7) sum finite iff sqrt(7) is in the trace-zero quadratic catalogue
of Ovsienko's Example 6.4, which `negation_finiteness` (in the wider research
code) already pins.

**Independent checks (test_q_sum.py).**
- Integer pair (2, 3): finite, value `(q + 1) + (q^2 + q + 1)`.
- Rational pair (3/2, 5/2): finite, value `q^2 + q + 2` (regression).
- 4 +/- sqrt(7): finite (trace-zero catalogue via the meeting reduction).
- Golden ratio + sqrt(2) (a non-trace-zero pair): not finite (control).

**What is not claimed.** The irrational verdict is the empirical
trailing-zero observation; the proof for the symmetric case is the
trace-zero criterion in `negation_finiteness` (research code).

---

## `radius(x, N)`

**Theorem.** None exact. `[x]_q` is a power series with integer coefficients;
its radius of convergence is `R = 1 / limsup_k |c_k|^{1/k}` (Cauchy-Hadamard).
`radius` returns the finite-N estimate `exp(-max_{1<=k<N, c_k != 0} (ln|c_k|)/k)`,
the reciprocal of the running maximum of `|c_k|^{1/k}`, or `+inf` when no
coefficient past `q^0` is nonzero in the window.

**Algorithm.** A single pass over `q_real_truncated(x, N)` tracking the maximum
slope.

**Independent check.** For a rational x, `[x]_q = R(q)/S(q)` is a rational
function, and its radius of convergence is the modulus of the nearest pole, the
smallest-modulus root of `S(q)`. `tests/test_arithmetic.py` computes that root
modulus independently with sympy for x in `7/5, 8/5, 11/9` (poles at
`0.755, 0.570, 0.826`, off the unit circle) and confirms the estimate is
monotone decreasing in N and converges to the pole from above, matching within
`2e-2` by N = 160.

**Finite-N bias, stated honestly.** The running maximum over a finite window is
at most the true limsup, so the estimate is **at least** the true radius: it is
biased high and decreases toward R as N grows. For a polynomial (an integer x,
true radius infinite) the leading unit coefficients pin the estimate near 1
rather than revealing the infinite radius - the same finite-N bias, at its
extreme. The estimate is reported as an estimate, never as the exact radius.

---

## Verification stamps for the arithmetic capabilities

The inline stamp (above) extends to the new computations, with the same rule
that it only reports checks that actually ran:

- **arith.** "matches `[x]_q (+/*) [y]_q` series" (the result equals the
  term-by-term combination of two `q_real_truncated` series) and "bihomographic
  engine agrees" (the independent state machine returns the same coefficients,
  for rational inputs; n/a otherwise).
- **negate.** "negation involutive" (applying the Jouteur negation twice returns
  `[x]_q`).
- **radius.** "estimate decreases with N" (the running-max estimate is biased
  high and falls toward the true radius).

Tested in `tests/test_arithmetic.py`, `tests/test_gosper.py`, and the headless
stamp assertions in `tests/test_app.py`.

---

## `q` and `__version__`

`q` is the sympy `Symbol("q")` that `q_rational` returns its results in; a test
confirms `q == Symbol("q")` and that `q_rational` introduces no other free
symbol. `__version__` is the package version string; a test confirms it matches
the `project.version` field of `pyproject.toml`.

---

## The inline verification stamp (`qreals.verify`)

Every computation prints one line of cross-checks by default. The logic is in
`qreals.verify`, which is core: it imports only sympy and the other core
modules, never the interface or certificate layer, so the stamp works with the
core install alone and never needs a TeX engine. It writes nothing.

The stamp does not introduce a new claim; it reruns, for one input, the same
independent checks this document already lists, and reports honestly what ran:

- **q=1 specialisation.** For a rational input, `[p/s]_q` at q=1 equals the
  ordinary p/s (RAT Corollary 1.7). For an irrational input the series diverges
  at q=1, so the check is reported n/a, not passed.
- **exact = truncated.** For a rational input, the Taylor expansion of the exact
  `q_rational` equals the truncated series, the same check as `q_real_truncated`
  above. For an irrational input there is no terminating rational function, so
  this is reported n/a.
- **truncation stable.** The first N coefficients do not change when N + 4 are
  requested (REAL Theorem 1).
- **shift law.** `[x+1]_q` recomputed from its own continued fraction equals
  `q [x]_q + 1` (REAL eqn 3). This is an algorithm-level cross-check: `[x+1]_q`
  has different partial quotients, so agreement is not built in.

A check that cannot run for an input is marked n/a with the reason; a check that
raises is marked as a check error; the stamp claims success only when every
check that ran passed. Tested in `tests/test_verify.py`: the stamp prints by
default, writes no file, marks q=1 n/a for an irrational, and folds into the
`--json` payload rather than corrupting it.

## Certificates (`qreals.certificate`)

A certificate is a human-auditable derivation, not a formal machine proof. For
one input it shows (a) the continued fraction of x and its even-length MGO form,
(b) the MGO formula folded step by step to the result, and (c) the cross-checks
above in full. Every line is checkable by hand. The module is the interface
layer behind the optional extra `qreals[proof]`; it imports the core, never the
other way round, and the qprov bridge is one-way and off by default.

Every source a certificate cites comes from the hard-coded registry in
`qreals.refs` (the dict `REFERENCES`); nothing is fetched at runtime. An in-text
citation (RAT eqn 1.1 in section (b), REAL Proposition 1.1 in a series
certificate's input line, the survey as the overview) routes through the registry
so it renders as a working hyperlink: an `\href` in LaTeX, an `<a>` in the HTML
view, and a numbered, URL-carrying entry in the terminal. Each certificate ends
with a hyperlinked Sources list of exactly the references it used. The registry
URLs are recorded in `docs/REFERENCES.md`, the checked-in note that
`tests/test_refs.py` keeps in step with the registry so a future broken link is
caught.

Rendering: terminal, HTML, LaTeX, and PDF, with one rule, only SAVE writes a
file. The terminal view, the HTML view, and saving a `.tex` need no TeX engine;
the PDF paths use the first of pdflatex, tectonic, latexmk found on PATH, in that
order on every operating system, with no hard-coded path. Tested in
`tests/test_certificate.py` (the text carries the three sections and the
citation, the `.tex` is a standalone document, save writes a `.tex` into the
chosen directory, the default `certify` writes nothing, and the PDF-compile test
is skipped when no TeX engine is present) and `tests/test_refs.py` (every
citation key a certificate emits is in the registry, each renders as a hyperlink
in all three views, and the URL note matches the registry).

## Running the suite

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=qreals --cov-report=term-missing
```

At the time of writing, 206 tests pass. Line coverage of the modules carrying a
mathematical claim stays high; the uncovered lines are command-line entry
points and defensive branches (empty-input guards, the max-depth fallback, the
negative-integer series helpers, and the certificate's PDF-view and rich
rendering paths reached only with a viewer and a TeX engine), none of which
carry a mathematical claim.
