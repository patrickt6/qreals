# Input-output matrix audit

A sweep of every interactive capability against a spread of input types
(positive and negative integers; proper, improper and unit rationals;
recognizable irrationals sqrt(2), sqrt(3), sqrt(5), the golden ratio, e, pi;
sub-one values; zero; and the smallest sizes N or order = 1) and every
follow-up action (terminal render, JSON dump, certificate terminal view, saved
.tex, compiled PDF, and the headless certify command).

The trigger was the sqrt(2) certificate, which once broke on a sharp
even-length convergent. The worry: other (capability, input) pairs break the
same quiet way. The audit ran each cell headless and collected the traceback.

The matrix is now a regression test, `tests/test_matrix.py`. The PDF-compile
cells skip when no TeX engine is on PATH.

## What was broken

One root cause produced 56 of the 71 failing cells across three surfaces;
the other 15 were already clean messages. After the fix, every remaining cell
is a clean, explanatory `ValueError` for a genuinely unsupported pair.

### Bug 1: a single-term continued fraction had no even-length form

`make_even_length` covered the odd-length cases that arise from a multi-term
continued fraction (split a last term >= 2, or absorb a trailing 1) but raised
`cannot make even-length CF from [a]` on a one-term list. A one-term list is
exactly what the inputs whose continued fraction is a single quotient produce:
the unit rational `5/5` and `1` give `[1]`, zero gives `[0]`, a negative
integer gives `[-3]`.

The computation paths sidestepped this (`q_real_truncated` special-cases `[0]`
and `[1]`; `q_rational` special-cases `p == s`), so terminal and JSON output
were fine. But three paths called `make_even_length` directly and crashed:

- the certificate builder (`_certificate_series`, `_certificate_rational`) for
  the unit rational and zero, the same family as the sqrt(2) defect;
- `q_rational(0, 1)` and any `p/s` reducing to zero;
- `q_rational` for a negative integer.

Cause: `make_even_length` was not total over valid integer continued
fractions. Fix: a single term `[a]` is the integer `a`, and `[a - 1, 1]` has
the same value (`a - 1 + 1/1 = a`) for any integer `a`, including 0, 1 and
negatives. This is the move already used for `a >= 2`, now also covering the
remaining single-term cases. `src/qreals/continued_fraction.py`.

With this fix the unit-rational and zero certificates build and compile, and
`q_rational` returns the correct `[0]_q = 0` and `[-3]_q = -(1 + q + q^2)/q^3`
(equal to `q_int(-3)`).

### Bug 2: the negative-rational certificate pulled a q^0 Taylor table

After Bug 1, `compute_rational(-3, 1)` computes the exact rational function,
but its certificate also asked `q_real_truncated` for a Taylor table. For a
negative value `[x]_q` is a Laurent object in negative powers, so that table
does not apply. Fix: `_certificate_rational` drops the table for value < 0,
matching how `_certificate_qint` already drops it for negative n. The
derivation, exact function, and cross-checks still certify the result.
`src/qreals/certificate.py`.

## What degrades cleanly (not bugs, made explicit)

These pairs are out of a capability's stated domain. They were made to fail
with a one-line explanation rather than a cryptic crash or a quiet wrong
answer.

- Negative x on the series path (coeffs, laurent, readouts, shift). After
  Bug 1 made `make_even_length` total, `q_real_truncated` would have returned
  all-zero coefficients for a negative x (its series lives in negative powers),
  a misleading result. It now raises a `ValueError` pointing to the q-integer
  and q-negation capabilities. `src/qreals/truncated.py`.
- Convergent locking with a non-positive partial sum. For x <= 0, and for
  0 < x < 1 at n = 1 (leading quotient 0), `S_n` was <= 0 and the locked count
  `S_n - 1` came out negative, a quiet nonsense result. It now raises with the
  positive-x precondition stated. `src/qreals/expansions.py`.
- Down-shift below 1 (`shift_down`), integer-part prefix of a negative x, and
  the q-negation / arithmetic path on a negative x already raised clear
  messages; the audit confirms they stay clean.

## Counts

- Distinct bugs found and fixed: 2 (the single-term even-length crash across
  three surfaces, and the negative-rational certificate table).
- 71 cells failed in the first sweep. The two fixes turned the 56 cryptic or
  blocking failures (certificate cannot build for a valid input;
  `cannot make even-length CF`) into working cells. The 31 that remain are
  clean, expected `ValueError`s for unsupported pairs, each asserted in the
  regression test; 5 of them are newly explicit (convergent locking with
  `S_n <= 0` used to return a negative count without complaint).
