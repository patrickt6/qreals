# qreals

[![CI](https://github.com/patrickt6/qreals/actions/workflows/ci.yml/badge.svg)](https://github.com/patrickt6/qreals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patrickt6/qreals/blob/main/notebooks/quickstart.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/patrickt6/qreals/main?labpath=notebooks%2Fquickstart.ipynb)

Compute q-deformed rational and real numbers. Pure Python, exact integer
coefficients, one dependency.

**The question this answers:** given a real number x, what is its q-analog
`[x]_q`, and what are the coefficients of its power series in q?

## For researchers: install and open the app in two steps

```bash
pip install -e .[app]   # 1. install with the interactive interface
qreals                  # 2. open the arrow-key menu
```

The menu walks you through every computation one prompt at a time, with an
example and a check on each input. No Python and no flags to memorise. If you
prefer code or scripts, see the [Quickstart](#quickstart) below; the full
documentation (quickstart, the MGO math in plain language, the correctness
proofs, and the API) builds with `mkdocs serve` after `pip install -e .[docs]`.

## What `[x]_q` is, and why anyone computes it

An ordinary integer n has a standard q-analog,

```
[n]_q = 1 + q + q^2 + ... + q^{n-1}
```

which collapses back to n when q = 1. Morier-Genoud and Ovsienko extended
this from integers to rationals and then to all real numbers, using continued
fractions. For a rational p/s the result `[p/s]_q` is an honest rational
function of q. For an irrational x it is a power series in q whose
coefficients are integers, and the series is the object people study: its
coefficients carry arithmetic information about x, and which patterns appear
in them is an open research question.

This package implements that construction directly. It does two things:

- `q_rational(p, s)` returns the exact `[p/s]_q` as a reduced rational
  function in q.
- `q_real_truncated(x, N)` returns the first N power-series coefficients of
  `[x]_q` for any real x you can write down, with a guarantee that those N
  coefficients are stable (they will not change if you ask for more).

The math is from one paper: Morier-Genoud and Ovsienko, "q-deformed rationals
and q-continued fractions", Forum Math. Sigma 8 (2020), e13. The
implementation follows its Definition 1.1 and Proposition 1.1.

## Install

Requires Python 3.11 or newer. The only runtime dependency is sympy. Once
v0.1.0 is published, the install is one line:

```bash
pip install qreals
```

That release is staged and waiting on a single web-only step (see
[RELEASING.md](./RELEASING.md)); until it is live, install straight from git,
which works today:

```bash
pip install "git+https://github.com/patrickt6/qreals.git"
```

To work on the package itself, install it from a clone in editable mode:

```bash
pip install -e .
```

For the test tools as well:

```bash
pip install -e .[dev]
```

For the guided menu described below, install the interface extra:

```bash
pip install -e .[app]
```

For the full step-by-step certificates (terminal or PDF), install the proof
extra. PDF output also needs a TeX engine on PATH (pdflatex, tectonic, or
latexmk); the inline verification stamp below needs neither and is always on.

```bash
pip install -e .[proof]
```

Two optional exploration helpers each sit behind a light extra, so the core
stays at sympy only. `qreals[oeis]` adds OEIS lookup (its one dependency is
`requests`); `qreals[features]` adds numpy for the vector side of the
fingerprint, which otherwise builds in pure Python.

```bash
pip install -e .[oeis]       # OEIS lookup
pip install -e .[features]   # numpy for Fingerprint.as_numpy
```

## The `qreals` command

If you would rather not write Python, install `qreals[app]` and run the bare
command:

```bash
qreals
```

That opens a menu you move through with the arrow keys and Enter. There is one
entry per computation: exact `[p/s]_q`, the q-integers `[n]_q`, the
coefficients of `[x]_q`, the MGO Laurent expansion, the integer-part prefix,
convergent locking, the shift relations `[x +/- 1]_q`, the coefficient
read-outs, the arithmetic `[x]_q +/* [y]_q`, the q-negation `[-x]_q` with its
`x -> -x` finiteness, the radius-of-convergence estimate, an OEIS lookup for a
coefficient sequence, and a fixed-length fingerprint of a constant. Each one
asks for its inputs one at a time, shows an example, checks what you type, and
prints a formatted result. No flags to memorise.

The same computations are available as subcommands for scripts and agents, with
`--json` for machine-readable output:

```bash
qreals rational 3 2
qreals coeffs pi 12
qreals laurent pi --order 12
qreals shift pi --down --order 12
qreals arith 3/2 13/5 12          # [3/2]_q + [13/5]_q
qreals arith 3/2 5/2 12 --mul     # [3/2]_q * [5/2]_q
qreals negate "sqrt(2)" 12        # [-sqrt(2)]_q and is [x]_q + [-x]_q finite?
qreals radius pi 60               # radius-of-convergence estimate
qreals oeis "1,1,2,5,14,42,132"   # look the sequence up in the OEIS (needs [oeis])
qreals fingerprint pi             # named, fixed-length fingerprint of [pi]_q
qreals coeffs "(1+sqrt(5))/2" 30 --json
```

`python -m qreals` is equivalent to the bare command. Run `qreals --help` to
list every subcommand.

## Verification, every time

Every computation prints a one-line stamp by default. It reruns the cheap
cross-checks that apply to that input and reports them in one line:

```
qreals coeffs pi 12
...
verified: truncation stable to 12, shift law [x+1]=q[x]+1; n/a: q=1 specialisation; exact rational function
```

The checks are independent recomputations: the q = 1 specialisation equals the
ordinary value for a rational, the exact and truncated paths agree, the first N
coefficients hold when more are asked for, and `[x+1]_q = q[x]_q + 1` recomputed
from its own continued fraction. When a check cannot run for an input (q = 1 on
an irrational, say), the stamp says n/a rather than claiming a pass. The stamp
is core, needs no extra and no TeX engine, and writes nothing.

For the full reasoning, `qreals certify` prints a human-auditable derivation:
the continued fraction and its even-length MGO form, the formula folded step by
step to the result, and the cross-checks in full. It cites the source paper and
points at `docs/CORRECTNESS.md`. It is a derivation to read, not a formal
machine proof.

```bash
qreals certify rational 3 2          # print the derivation, keep no file
qreals certify coeffs pi 12 --pdf    # open it as a PDF, keep no file
qreals certify coeffs pi 12 --save   # write a .tex (and .pdf) here
```

Only `--save` writes a file. With `--save --qprov`, and qprov installed, the run
is also recorded in your `.qprov` store for later citation.

`qreals doctor` reports the operating system, Python version, whether stdin and
stdout are a terminal, whether questionary, rich, and a TeX engine are present,
and whether the interactive menu will run here.

## Keeping and exporting results

A computed value can go into a personal saved list that survives across
sessions. In the app, after a computation, choose "Add this result to my saved
list"; "My saved list" then lets you view it, remove an item, or export it. The
list lives under your per-user data directory (via platformdirs, with a
standard-library fallback to the same location), never the working folder, so it
is there the next time you open the app. Each entry keeps the input, the order
N, the coefficients, and a timestamp.

Any single result or the whole list exports to four formats: JSON (the
round-trip format), CSV (one row per constant), a booktabs LaTeX table ready to
paste into a paper, and Magma code that rebuilds each value as a Laurent series
in q over the rationals. A file is written only when you export; an interactive
export uses the current directory only after you confirm.

The same exports are available headless:

```bash
# compute a list of constants straight into one file
qreals batch "pi,sqrt(2),3/2" --order 12 --format magma -o atlas.m
qreals batch "pi,(1+sqrt(5))/2" --order 20 --format latex -o table.tex

# write, list, or tidy the saved list
qreals export --format csv -o saved.csv
qreals saved                 # show the saved list
qreals saved --remove 0      # drop one entry
qreals saved --clear         # empty the list
```

Without `-o`, `batch` and `export` print to stdout and write nothing. With
`--qprov`, and qprov installed, each exported value also carries a qprov id
linking it to a recorded run; this is off by default and the core never imports
qprov.

## Quickstart

To try qreals with zero setup, open the
[quickstart notebook in Colab](https://colab.research.google.com/github/patrickt6/qreals/blob/main/notebooks/quickstart.ipynb)
or [on Binder](https://mybinder.org/v2/gh/patrickt6/qreals/main?labpath=notebooks%2Fquickstart.ipynb):
both install the package and run the examples below for you. To run it locally,
`pip install qreals` (or the git install above until the release is live), then:

```python
import qreals

# Exact q-rationals, as elements of Q(q).
qreals.q_rational(3, 2)      # (q**2 + q + 1)/(q + 1)
qreals.q_rational(1, 2)      # q/(q + 1)

# First 30 stable Taylor coefficients of [x]_q, for any real x.
qreals.q_real_truncated("pi", 30)
qreals.q_real_truncated("sqrt(2)", 30)
qreals.q_real_truncated("(1+sqrt(5))/2", 30)   # the golden ratio
qreals.q_real_truncated("E", 30)
```

The string passed to `q_real_truncated` is parsed by sympy, so anything sympy
understands works: `"pi"`, `"sqrt(2)"`, `"(1+sqrt(5))/2"`, `"E"`, `"3/2"`.

The coefficients come back as a plain list of Python ints, c_0 first:

```python
qreals.q_real_truncated("pi", 12)
# [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
```

A few read-outs over a coefficient list are included for pattern hunting:

```python
from qreals import (
    first_nonzero_coefficient_index,
    first_negative_coefficient_index,
    coefficient_max_abs,
    number_of_zeros,
)
```

## Arithmetic between q-reals

The series operations on q-reals, plus a convenience wrapper, all verified
against an independent computation (see `docs/CORRECTNESS.md`):

```python
import qreals

# The series sum and product: [x]_q + [y]_q and [x]_q * [y]_q (first N coeffs).
qreals.q_add("3/2", "13/5", 8)     # [2, 1, 1, 0, 0, 1, -3, 6]
qreals.q_mul("3/2", "5/2", 8)

# The deficit: how far the series sum/product sits from [x op y]_q (op + or *).
d = qreals.deficit("3/2", "5/2", "+", 8)
d.deficit                          # [-1, 0, 0, 1, 0, 0, 0, 0]
d.exact                            # q**3 - 1  (exact for rational inputs)
d.deficit_at_q1, d.deficit_at_q0   # (0, -1)   the q=1 and q=0 checks

# The Jouteur q-negation [-x]_q (a Laurent series): (valuation, coefficients).
qreals.q_neg("2", 6)               # (-3, [-1, 0, -1, 0, 0, 0])  = -q^-1 - q^-3

# Ovsienko Example 6.4: is [x]_q + [-x]_q a finite Laurent polynomial?
qreals.finite_xnegx("sqrt(2)")     # True  (pure square root)
qreals.finite_xnegx("(1+sqrt(5))/2")  # False (golden ratio, trace 1)

# Radius-of-convergence estimate of the power series [x]_q.
qreals.radius("pi", 60)            # ~0.91, biased high at finite N
```

Two cautions, both spelled out in the docstrings and `docs/CORRECTNESS.md`:
`q_add` and `q_mul` are the sum and product of the two **series**, not
`[x + y]_q` or `[x * y]_q` (the MGO map `x -> [x]_q` is not a ring
homomorphism); and `q_neg` is the PGL_2(Z)-action negation of Jouteur, an
involution, not the coefficient-wise negation of `[x]_q`. `deficit` measures the
first gap directly, exact in q for rational inputs, with the invariants
`D(1) = 0` (both sides agree at `q = 1`) and `D(0) = -1` for a sum of `x, y >= 1`.

A `QReal` wrapper offers the read-outs and operators over a held series:

```python
from qreals import QReal

x = QReal("3/2", 16)
y = QReal("13/5", 16)
(x + y).coeffs           # delegates to q_add
(x * y).coeffs           # delegates to q_mul
(-QReal("sqrt(2)", 16))  # delegates to q_neg (a Laurent QReal)
QReal("pi", 60).radius_estimate()
QReal("pi-2", 12).sign_pattern()   # "+  ..." signs of the coefficients
QReal("pi-2", 12).zero_run()       # (start, length) of the longest zero run
```

The functional API (`q_add`, `q_mul`, `q_neg`, `radius`, ...) is the stable
core; `QReal` is sugar over it.

For the underlying algorithm, the bihomographic q-Gosper engine is exposed as
`qreals.q_gosper(x, y, op)` and `qreals.gosper_coeffs(...)`; the arithmetic
functions are cross-checked against it.

## Using the coefficients as data

Because `q_real_truncated` returns exact integer coefficients for any real x, on
demand and to any length, its output is convenient as data for experiments on the
coefficient sequences, including machine-learning experiments. Two properties
help. The labels are exact, so there is no measurement noise to model around. And
the same engine that generates an example can check a prediction about it, so it
also serves as a verifier. A typical loop generates coefficient sequences for
many constants, attaches whatever label a question needs, and studies how the
sequence relates to the label.

## Two exploration helpers

Both are optional, each behind a light extra, and neither adds a heavy
dependency to the core.

### OEIS lookup

`qreals.oeis.lookup` takes a coefficient sequence and searches the OEIS for it.
It ranks hits by matching-prefix length, reconciles the sign conventions a
q-series can carry (negated, alternating, absolute value), re-verifies the top
hits against the full b-file so a deep divergence is caught, and also tries
mod-p reductions of the input. Every response is cached on disk.

```python
import qreals

res = qreals.oeis.lookup("1,1,2,5,14,42,132,429")
res.top.anum            # "A000108"  (Catalan numbers)
res.top.fully_verified  # True, checked against the b-file
```

It needs the `requests` dependency (`pip install qreals[oeis]`) and raises a
clear `OeisUnavailable` if it is missing. When OEIS is unreachable the lookup
returns no hits rather than raising, so an offline call degrades quietly.

### Fingerprint of a constant

`qreals.featurize(x)` returns a named, fixed-length, deterministic feature
vector of `[x]_q`: continued-fraction partial quotients and their partial sums,
the signed Taylor coefficients, sign-run and zero-run counts, coefficient
magnitudes, and an inverse-radius growth estimate. The length depends only on
the parameters, so two constants are directly comparable, which is the point:
exploration and nearest-neighbour over constants, not model training.

```python
import qreals

fp = qreals.featurize("pi")
fp.as_dict()["cf_0"]     # 3.0   (floor of pi)
fp.as_dict()["c_0"]      # 1.0   (constant term of [pi]_q)
len(fp.values)           # same length for every constant at these settings

a, b = qreals.featurize("pi"), qreals.featurize("sqrt(2)")
qreals.feature_distance(a, b)          # Euclidean distance
qreals.nearest(a, [b, qreals.featurize("E")])   # nearest-neighbour search
```

Building a fingerprint is pure Python; numpy is needed only for
`Fingerprint.as_numpy` (`pip install qreals[features]`). Each feature is
documented in `src/qreals/features.py`.

## How the stable-coefficient guarantee works

The construction runs the continued fraction of x through the MGO formula.
The depth needed is not a tuning knob: MGO Proposition 1.1 says that if you
stop the continued fraction at the first point where the partial quotients
sum to at least N + 1, then exactly that-sum-minus-one of the resulting power
series coefficients agree with the true `[x]_q`. `q_real_truncated(x, N)`
accumulates partial quotients until the sum clears N, so the N coefficients it
returns are stable by that proposition.

Two consequences worth noting:

- For a rational p/s the continued fraction terminates, so the truncated
  series is just the Taylor expansion of the exact `q_rational(p, s)`. The
  test suite checks that the two paths agree coefficient for coefficient.
- The worst case for depth is the golden ratio, whose continued fraction is
  all ones, so reaching N stable coefficients takes depth N + 1. Everything
  else is faster.

## Two computation paths

| Input | Function | Returns |
|---|---|---|
| rational p/s | `q_rational(p, s)` | exact rational function in q |
| any real x | `q_real_truncated(x, N)` | first N integer coefficients of the q-series |

Coefficients are exact throughout. The series path holds each coefficient as
a Python int and inverts series with Newton's iteration over the integers, so
nothing is lost to floating point.

## What is checked against the paper

The test suite anchors the implementation to known values and to the
defining properties:

- `q_rational(3, 2)` equals `(1 + q + q^2)/(1 + q)` and `q_rational(1, 2)`
  equals `q/(1 + q)`, the worked examples in the source material.
- Every q-rational specialises to its ordinary value at q = 1.
- The first N coefficients from `q_real_truncated` do not change when N grows,
  which is the Proposition 1.1 stability claim.
- The exact and truncated paths agree on rationals.

```bash
python -m pytest tests/ -v
```

## Relationship to qprov

`qreals` is the engine that produces numbers; [qprov](https://github.com/patrickt6/qprov)
is a separate local-first tool that records where a number came from. They fit
together: a paper states a coefficient, qprov links that statement to the
exact recorded computation, and `qreals` is the code that computation runs.
Neither depends on the other, and you can use either alone.

## Project layout

```
qreals/
  pyproject.toml
  README.md
  CLAUDE.md                    context for AI coding agents
  src/qreals/
    __init__.py                public API
    continued_fraction.py      partial quotients + even-length normalisation
    series.py                  truncated Laurent series kernel in q
    truncated.py               [x]_q as a stable power series
    rational.py                [p/s]_q as an exact rational function
    arithmetic.py              q_add, q_mul, q_neg, finite_xnegx, radius
    gosper.py                  bihomographic q-Gosper engine (the cross-check)
    qreal.py                   QReal convenience wrapper
    coefficients.py            read-outs over a coefficient list
    features.py                fixed-length fingerprint of [x]_q (numpy optional)
    oeis.py                    OEIS lookup for a coefficient sequence (qreals[oeis])
    verify.py                  inline cross-check stamp (core, sympy only)
    certificate.py             full derivation, terminal/PDF/save (qreals[proof])
  tests/                       pytest suite
  example/
    run_example.py             runnable end-to-end demo
```

## Limitations

- `q_real_truncated` takes a sympy-parseable string and computes the
  continued fraction symbolically, so the cost of reaching N coefficients is
  set by how slowly x's continued fraction sum grows. Constants with small
  partial quotients (the golden ratio, sqrt(2)) cost the most.
- Coefficients grow fast for some constants. The size shown for the golden
  ratio at N = 30 is already in the billions; at large N the integers become
  the dominant cost. This is inherent to the series, not to the method.
- A Sage version of the same construction, using Sage's Laurent series ring,
  exists in a larger research codebase. This package is the standalone
  pure-Python form so it runs without Sage.

## References

- S. Morier-Genoud and V. Ovsienko, "q-deformed rationals and q-continued
  fractions", Forum Math. Sigma 8 (2020), e13.
- S. Morier-Genoud and V. Ovsienko, "On q-deformed real numbers"
  (arXiv:1908.04365).
- A. Jouteur, "Modular group action on q-deformed real numbers"
  (arXiv:2503.02122) - the negation used by `q_neg`.
- V. Ovsienko, "Modular invariant q-deformed numbers: first steps" - Example 6.4,
  the `x -> -x` finiteness studied by `finite_xnegx`.

## Citing qreals

Citation metadata is in [CITATION.cff](./CITATION.cff); GitHub renders a "Cite
this repository" button from it. The release history is in
[CHANGELOG.md](./CHANGELOG.md).

## License

MIT. See [LICENSE](./LICENSE).
