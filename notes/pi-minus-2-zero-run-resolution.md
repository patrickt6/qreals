# Resolving the may14 note-5 tension: the zero-run of [pi-2]_q ends at a_7

## The question

For a real x in (1, 2), the q-real is forced to open
`[x]_q = 1 + 0*q + a_2 q^2 + a_3 q^3 + ...` (integer-part lemma, note 1). The
open question (note 5) asks how far the leading run of zeros in the tail
extends. The worked case is x = pi - 2 ~ 1.1416, reached from [pi]_q by two
applications of the shift-down identity `[x-1]_q = ([x]_q - 1)/q`.

note 5 flagged a tension: the board concluded `a_2 = a_3 = ... = a_7 = 0` for
pi - 2, but reasoned that if the q^9 coefficient of [pi]_q were 1 (as note 3's
`[22/7]_q = 1 + q + q^2 + q^9 + ...` suggested), the run would instead stop one
term earlier, at a_6. The two readings differ by one. The deciding fact is the
q^9 coefficient of [pi]_q.

## The computation

Computed with `qreals.mgo_laurent`. The expansion of [pi]_q is stable through
q^12 because the convergent [3, 7, 15] = 333/106 has partial-quotient sum
S_3 = 25, which locks every power below q^24. The values below also match the
deeper convergent [3, 7, 15, 1] = 355/113 coefficient for coefficient.

```
[pi]_q   = 1 + q + q^2 + q^10 - q^12 + O(q^13)
           c = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1]   (c_0 .. c_12)

[22/7]_q = 1 + q + q^2 + q^9 - q^10 + O(q^13)
           c = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0]   (c_0 .. c_12)
```

The q^9 coefficient of [pi]_q is **0**, not 1. Its first tail term sits at
q^10. The coefficient that is 1 at q^9 belongs to [22/7]_q, not to [pi]_q: the
two series first disagree exactly at q^9.

Applying shift-down twice (`qreals.shift_down`), which matches the direct
computation of [pi-2]_q coefficient for coefficient:

```
[pi-1]_q = 1 + q + q^9 - q^11 - q^12 + O(q^13)
[pi-2]_q = 1 + q^8 - q^10 - q^11 + O(q^12)
           c = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1]   (c_0 .. c_11)
```

## The answer

The leading zero-run of [pi-2]_q is `a_2 = a_3 = a_4 = a_5 = a_6 = a_7 = 0`,
and the first nonzero tail coefficient is `a_8 = 1` (the q^8 term). **The run
ends at a_7.** The board's `a_2 = ... = a_7 = 0` is correct; note 5's
alternative (run stopping at a_6) does not occur.

## Where the off-by-one actually was

The tension came from the convergent-agreement bound, not from the shift
arithmetic. note 2 stated that the n-th convergent agrees with [x]_q on every
power **below q^{S_n}** (S_n = a_1 + ... + a_n). The correct bound is one lower:
every power **below q^{S_n - 1}**, locking S_n - 1 coefficients (powers q^0
through q^{S_n - 2}).

For pi this is exactly the off-by-one above: S_2 = 3 + 7 = 10, so [22/7]_q
locks [pi]_q through q^8 only (nine coefficients), and the q^9 = q^{S_2 - 1}
term of [22/7]_q does **not** transfer. note 3's "every coefficient through q^9
transfers" overshoots by one term; that single phantom q^9 term is the entire
source of the tension.

The qreals engine already encodes the correct count: `continued_fraction.py`
stops the continued fraction at the first depth where S_n - 1 reaches the
requested precision, and `qreals.coeffs_locked_by_convergent([3,7,15], 2)`
returns `(S_n, count) = (10, 9)`. The empirical divergence of [22/7]_q and
[pi]_q at q^9 confirms the count of 9 is tight.

## Reproduce

```
python -m qreals.expansions pi   --order 12
python -m qreals.expansions 22/7 --order 12
python -m qreals.expansions pi-2 --order 12
```
