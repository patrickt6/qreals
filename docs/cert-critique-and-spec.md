# Certificate critique and redesign spec

This document surveys how `qreals` builds a certificate for every computation
type, builds one sample of each, records what makes each hard to read, and gives
one cross-type redesign spec. It generalizes the hand-written critique of the
`[pi]_q` coefficient certificate (kept at `../../qrealspdfcritique.md`) from that
one case to all of them. Steps 13 and 14 implement the spec; this step only
diagnoses and specifies, and does not change `certificate.py`.

The built samples this document refers to live in `cert-samples/`, one terminal
dump (`<name>.txt`), one `.tex`, and one `.pdf` per kind, produced by
`cert-samples/_build_samples.py` with `pdflatex` on PATH.

## 1. How a certificate is produced today

The path is the same for every kind:

1. A `compute_*` function in `app.py` returns a result dict carrying a `kind`
   string, display `blocks`, and a `data` payload.
2. `certificate.build_certificate(result)` turns that dict into a `Certificate`
   dataclass. It branches on `kind`:
   - `kind == "rational"` calls `_certificate_rational(p, s)`;
   - `kind == "qint"` calls `_certificate_qint(n)`;
   - every other kind falls through to one line,
     `x_repr = str(data["x"])`, and then `_certificate_series(x_repr, n)`,
     where `n` is read from `data["n"]`, else `data["order"] + 1`, else
     `len(data["coefficients"])`, else the default depth 12.
3. The `Certificate` renders three ways: `render_terminal` (plain text, no
   eliding), `to_tex` / `view_pdf` (LaTeX, polynomials elided to `max_terms`),
   and `save` (writes the `.tex` and, when a TeX engine is found, the `.pdf`).

Two entry points reach step 2. The `certify` CLI subcommand restricts its
`kind` argument to `{rational, qint, coeffs, laurent}`, the four kinds the
branch above handles on target. The interactive menu's post-compute actions
(Show / PDF / Save certificate, in `_certificate_action`) are offered for the
result of every capability, so they reach `build_certificate` with kinds the
fall-through branch cannot certify on target. The next section is the
consequence.

## 2. Inventory: what each type actually certifies

Every certificate shares one body: section (a) the continued fraction and its
even-length MGO form, (b) the MGO fold step by step, an optional Taylor line,
(c) the cross-check stamp, and a references block. The table records, per kind,
the deliverable the user asked for, what `build_certificate` puts in the
certificate, and whether they match.

| kind | user's deliverable | what the certificate certifies | on target? | sample |
|---|---|---|---|---|
| `rational` | exact `[p/s]_q` | `[p/s]_q` fold and value | yes | `rational.*` |
| `qint` | `[n]_q` | `[n]_q` Gauss integer | yes | `qint.*` |
| `coeffs` (rational x) | first N coeffs of `[x]_q` | `[x]_q` fold and coeffs | yes | `coeffs_rational.*` |
| `coeffs` (irrational x) | first N coeffs of `[x]_q` | `[x]_q` fold and coeffs | yes | `coeffs_irrational.*` |
| `laurent` | `[x]_q` to a chosen power | `[x]_q` fold and coeffs | yes | `laurent.*` |
| `prefix` | forced integer-part prefix | `[x]_q` to depth 12, prefix absent | no | `prefix.*` |
| `locked` | `S_n` and locked-coefficient count | `[x]_q` to n coeffs, count absent | no | `locked.*` |
| `shift` | `[x+1]_q` or `[x-1]_q` | `[x]_q`, shifted series absent | no | `shift.*` |
| `readouts` | read-out table over the coeffs | `[x]_q` to n coeffs, table absent | no | `readouts.*` |
| `arith` | `[x]_q + [y]_q` or product | `[x]_q` only; y and the sum absent | no | `arith_add.*`, `arith_mul.*` |
| `negation` / `finite` | `[-x]_q` and `[x]_q + [-x]_q` | `[x]_q`, negation and sum absent | no | `negation.*` |
| `radius` | radius-of-convergence estimate | `[x]_q` to n coeffs, estimate absent | no | `radius.*` |
| `oeis` | OEIS match for a sequence | nothing: `build_certificate` raises `KeyError('x')` | no (errors) | not built |
| `fingerprint` | feature vector of `[x]_q` | `[x]_q` to depth 12, vector absent | no | `fingerprint.*` |

So four kinds out of fourteen produce a certificate about what the user
computed. The other ten either certify the wrong object or fail to build:

- `arith` keeps only `data["x"]`, the first operand, and drops the second
  operand and the operation. `arith_add.txt` is titled
  `Certificate for [1/2]_q (first 12 coefficients)`; the sum
  `[1/2]_q + [1/3]_q` it was asked to certify never appears.
- `negation` certifies `[sqrt(2)]_q`; the involution `[-sqrt(2)]_q` and the
  sum whose finiteness is the whole point (Ovsienko Example 6.4) are absent.
- `radius` certifies `[pi]_q` to 24 coefficients; the radius number, the only
  output of the computation, is absent.
- `shift`, `prefix`, `locked`, `readouts`, `fingerprint` likewise reduce to a
  plain `[x]_q` coefficient certificate, with the shifted series, the prefix,
  the locking count, the read-out table, and the feature vector all dropped.
- A second consequence of the single fall-through: `_certificate_series` always
  builds its stamp with `verify({"kind": "coeffs", ...})`. The dedicated
  checks `verify_arith`, `verify_negation`, and `verify_radius` in `verify.py`
  are never reached through any certificate. The cross-checks printed under an
  `arith` or `negation` certificate are the series checks for `[x]_q` (for
  example `arith_add.txt` lists "[x+1]_q computed from its own continued
  fraction equals q*[x]_q + 1", a check on `[1/2]_q`, not on the sum).

This mismatch is the first finding the redesign must fix, and it sits above the
readability findings in section 3: a certificate that is easy to read but
certifies the wrong object is worse than one that is hard to read.

## 3. Shared readability findings

The hand critique gave a five-point checklist for the `[pi]_q` case. Each point
generalizes, and each is reproduced by a built sample below. Quotations are from
the samples in `cert-samples/`.

### 3.1 Ellipses delete the palindrome's center

The MGO numerators are palindromic, and the elision keeps the first ten and last
ten monomials by power and cuts the middle, which is exactly the symmetric
center. For `[333/106]_q` the numerator has 25 terms; the center plateau is the
run `22 q^{14} + 22 q^{13} + 22 q^{12} + 22 q^{11} + 22 q^{10}`. The PDF
(`coeffs_irrational-*.tex`, line 60) renders it as

```
N(q) = q^{24} + 2 q^{23} + ... + 22 q^{15} + \dots + 21 q^9 + 20 q^8 + ... + 1
```

with "(5 terms omitted)" under it. The five omitted terms are precisely the flat
symmetric center, so the one feature a reader would scan for to confirm the
palindrome is the one feature removed. The terminal view does not elide at all;
it prints every monomial of every fold (see any `*.txt`), which is the opposite
failure, a wall of coefficients with the symmetry present but unannounced.

### 3.2 Reused N(q) and D(q) with no subscripts

Once a numerator or denominator passes `_FRAC_INLINE` (12 terms) the value is
split into a line naming it `N(q)/D(q)` and two displays. The names carry no
fold index, so the same `N(q)` and `D(q)` are written for fold a_2, fold a_1,
and the Result block. In `coeffs_irrational-*.tex` the `D(q)` printed for
fold a_1 (line 68) is character-for-character the `N(q)` printed for fold a_2
(line 44). That equality, `the new denominator is the previous numerator`, is
the single structural fact of the fold, and the document never states it; a
reader has to notice it by comparing two unlabelled displays a page apart.

### 3.3 The recursion is never written

Section (b) of every sample says only "odd positions carry `[a]_q` with `q^a`
above; even positions carry `[a]_(q^-1)` with `q^-a` above (MGO eqn 1.1)". The
formula being folded is named, never written. A referee has to open MGO to
follow line 3 of a derivation that claims to be checkable by hand.

### 3.4 The deliverable is buried, and restated before it appears

Every sample ends section (b) with a bold `Result.` line, then the Taylor
coefficients as the last line. The `Result.` value is identical to the last
fold (in `coeffs_irrational`, line 74 repeats line 60), so the prominent line is
a restatement, and the actual deliverable, the coefficients
`1 + q + q^2 + q^10 + O(q^12)`, is the final unemphasized line. For `radius`,
`arith`, `negation`, and the may14 kinds the true deliverable is not on the page
at all (section 2).

### 3.5 No closing tie-back to a known value

No sample ends by comparing its result to something the reader already believes.
The `rational` and `qint` samples come closest, with a stamp line
"the Taylor expansion of the exact rational function and the truncated series
agree on q^0..q^11", but that compares two internal computations, not the result
against an external anchor (the value at q = 1, the prefix law, the convergent
overlap, a published coefficient list). The irrational samples have no q = 1
anchor available and offer no substitute.

### 3.6 One body for every kind, so nothing is kind-specific

Because eleven kinds share `_certificate_series`, the certificate cannot say
anything that is true of one kind and not another. The shift law, the negation
involution, the locking bound, the radius method note, the non-homomorphism
warning, all of which the `compute_*` functions already produce as `blocks`,
are discarded when the result becomes a certificate.

## 4. Cross-type redesign spec

The fix the hand critique states in one sentence is: a human-facing certificate
should print structure plus a small numerical witness, not the full intermediate
polynomials. The spec below makes that the default for every kind and keeps the
full dump as a separate artifact. Steps 13 and 14 follow it.

### 4.1 Two artifacts, referee view as the default

Each certificate run yields two things:

- a **referee view**, the default for terminal, PDF, and the saved `.tex`: a few
  lines per fold, no symmetry-hiding ellipsis, the recursion stated once, named
  subscripted quantities, the deliverable up top, and a closing tie-back table;
- a **full dump**, the current line-by-line transcript, written only to an
  appendix section or a sibling file (for example `<slug>-dump.tex` or a clearly
  marked appendix), for anyone who wants to re-run the arithmetic byte for byte.

The referee reads the referee view; the dump exists for machine re-verification.
Neither one drops the exact objects, which remain available through the JSON and
coefficient read-outs as today.

### 4.2 Certify the deliverable, not the input

`build_certificate` must branch so that each kind's certificate is about what the
kind computed, using the `data` payload the `compute_*` function already fills:

- `arith` certifies the sum or product series from `data["coefficients"]`,
  shows both `[x]_q` and `[y]_q`, and states it is not `[x op y]_q`;
- `negation` certifies `[-x]_q` and the sum `[x]_q + [-x]_q` and reports the
  finiteness verdict;
- `radius` certifies the estimate and the slope data behind it;
- `shift`, `prefix`, `locked`, `readouts` certify their own deliverable
  (the shifted series, the prefix, the locking bound, the read-out table);
- `oeis` and `fingerprint` are not MGO fold derivations and should either be
  declined with a one-line message ("no fold certificate for this kind; use the
  result export") or given a minimal provenance card, not routed through
  `_certificate_series`. Decide this in Step 13; do not let them reach the fold
  path and certify an unrelated `[x]_q`.

Each kind's stamp uses its own verifier (`verify_arith`, `verify_negation`,
`verify_radius`, `verify_series`), not always the coeffs stamp.

### 4.3 State the recursion once

At the top of section (b), write the MGO fold explicitly, once, so the body is
self-contained:

```
[a_k, ..., a_n]_q is folded from the inside out by
    R_n      = [a_n]_(q^-1)
    R_i      = [a_i]_q     + q^{a_i}  / R_{i+1}   (i odd, 1-indexed)
    R_i      = [a_i]_(q^-1) + q^{-a_i} / R_{i+1}   (i even)
with [a]_q = (1 - q^a)/(1 - q) the Gauss q-integer.
```

Then the per-fold lines reference `R_i` rather than restating the rule.

### 4.4 Named, subscripted quantities, with the fold law stated

Write each fold result as `R_i = N_i(q) / D_i(q)` with the index carried. State
once, near the top, the structural identity `D_i(q) = N_{i+1}(q)` (the new
denominator is the previous numerator), then a reader can confirm it by reading
two subscripts instead of comparing two unlabelled displays.

### 4.5 No symmetry-hiding ellipsis

Never cut the middle of a palindromic polynomial. Two acceptable presentations:

- state "N_i is palindromic of degree d (coefficient of q^k equals coefficient
  of q^{d-k})" and list only the coefficient vector for k = 0..floor(d/2); or
- print the coefficient vector as a compact row, full length, since a row of
  integers fits the page where a row of monomials does not.

If a length cap is still wanted for very high degree, cap by showing the
coefficient vector, not by deleting interior monomials.

### 4.6 Final coefficient-comparison table

End every certificate with a table that ties the derived coefficients to a value
the reader can check independently, plus one small numerical witness: evaluate
both the exact rational function from the fold and the truncated series at a
fixed rational q_0 in (0, 1) (for example q_0 = 1/2) and show the two numbers
agree. That witness is independent of the symbolic Taylor path and reproducible
on a calculator. The table's "known value" column is filled per kind in
section 5.

## 5. Per-type referee view

What each kind's referee view should show, beyond the shared frame of 4.3 to 4.6.

- **rational `[p/s]_q`**: the even-length CF and convergent; the fold as
  `R_i = N_i/D_i` with degrees; the exact value; tie-back table with the
  value at q = 1 equal to p/s and the prefix pattern (first floor(p/s)
  coefficients equal 1, coefficient at q^{floor} equal 0) as the known column.
- **qint `[n]_q`**: state `[n]_q = 1 + q + ... + q^{n-1}`, an all-ones
  palindrome; no fold needed; tie-back is the value at q = 1 equal to n. Drop
  the misleading `O(q^N)` tail, since `[n]_q` is an exact finite polynomial.
- **coeffs (rational x)**: as `rational`, presented as a series to N, with the
  tie-back table comparing c_0..c_{N-1} against the value at q = 1 and the
  prefix law.
- **coeffs (irrational x)**: the convergent that locks N coefficients, named as
  such; the fold in subscripted form; tie-back table comparing the first
  coefficients against the next-deeper convergent's overlap (the stability that
  is MGO Proposition 1.1) and, where available, a published list such as the
  MGO Prop 1.1 values for `[pi]_q`. No q = 1 anchor; use the numeric q_0
  witness of 4.6 instead.
- **laurent**: as coeffs, with the chosen highest power stated and the tie-back
  table over the requested power range.
- **prefix**: state the prefix law once, show floor(x), the forced prefix
  `[1, ..., 1, 0]`, and a tie-back row showing it equals the first floor(x) + 1
  coefficients of `[x]_q`.
- **locked**: state the locking bound (the n-th convergent agrees with `[x]_q`
  on q^0 through q^{S_n - 2}); show the CF terms, the partial sum S_n, and the
  count; tie-back by exhibiting that the n-th and (n+1)-th convergents agree on
  those powers and first differ at q^{S_n - 1}.
- **shift**: state the shift law once (`[x+1]_q = q[x]_q + 1`, or
  `[x-1]_q = ([x]_q - 1)/q`); show `[x]_q` and the shifted series as two
  coefficient rows; tie-back termwise on the first few coefficients.
- **readouts**: show the coefficient series once, then the read-out table
  (first nonzero index, first negative index, largest absolute coefficient,
  zero count); each read-out is its own witness against the displayed row.
- **arith**: show `[x]_q`, `[y]_q`, and the sum or product series as three
  rows; state it is not `[x op y]_q` (the map is not a ring homomorphism);
  tie-back termwise, c_k(sum) = c_k(x) + c_k(y) for the sum.
- **negation**: show `[x]_q`, `[-x]_q` (the Jouteur PGL_2(Z) involution), and
  the sum `[x]_q + [-x]_q`; tie-back is the finiteness verdict, the sum is a
  finite Laurent polynomial exactly for trace-zero quadratics (pure square
  roots), Ovsienko Example 6.4.
- **radius**: show the running-max root-test value, the few largest
  `|c_k|^{1/k}` contributors as a short table, and the method note; tie-back is
  the trend, the estimate decreases toward the true radius as N grows.
- **oeis / fingerprint**: not fold derivations; per 4.2, decline with a message
  or emit a provenance card (input sequence or feature settings, the lookup or
  feature result, and the source), kept distinct from the MGO certificate.

## 6. What Steps 13 and 14 take from this

- Step 13: change `build_certificate` dispatch so each kind certifies its own
  deliverable (4.2), wire each kind to its own verifier stamp, and decide the
  `oeis` / `fingerprint` outcome. Add the `Certificate` fields the referee view
  needs (named subscripted folds, the recursion text, the tie-back table data,
  the numeric witness at q_0).
- Step 14: render the referee view as the default in terminal, PDF, and `.tex`,
  move the full dump to an appendix or sibling artifact (4.1), and implement the
  no-ellipsis palindrome presentation (4.5) and the closing comparison table
  (4.6).

All work stays behind `qreals[proof]`; the core remains sympy-only. The samples
in `cert-samples/` are the before state to compare the redesign against.
