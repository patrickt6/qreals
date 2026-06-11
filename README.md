# qreals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)

![The qreals web app, run with `qreals serve`](https://github.com/patrickt6/qreals/releases/download/v0.1.1/image.png)

A q-number replaces an ordinary number x with a series in a variable q that
collapses back to x at q = 1. `qreals` computes these exactly, and ships a
browser app to explore them.

## Getting started

You need **Python 3.11 or newer**. Check with `python --version` (Windows) or
`python3 --version` (macOS). If you do not have it, install it first from
[python.org](https://www.python.org/downloads/) (on Windows, tick "Add Python
to PATH" in the installer).

### 1. Install

**Windows** (PowerShell):

```powershell
py -m pip install qreals
```

**macOS** (Terminal):

```bash
python3 -m pip install qreals
```

### 2. Run the web app

```bash
qreals serve
```

This starts a local server and opens the app in your browser at
<http://127.0.0.1:8000>. Use `qreals serve --port 9000` to pick a different
port, or `qreals serve --no-browser` to skip opening a window. Press
`Ctrl+C` in the terminal to stop it.

### 3. Update to the latest version

Re-run the install command with `--upgrade` to get the newest release:

**Windows:**

```powershell
py -m pip install --upgrade qreals
```

**macOS:**

```bash
python3 -m pip install --upgrade qreals
```

> Prefer the source? Download the repo as a ZIP from the green **Code** button
> on GitHub (or clone it), then run `pip install .` from inside the folder.

## What it does

For a fraction p/s you get an exact rational function `[p/s]_q`, and for any
real x you get the exact integer coefficients of its power series `[x]_q`, to
any length. The math is from Morier-Genoud and Ovsienko, "q-deformed rationals
and q-continued fractions" (Forum Math. Sigma, 2020).

Every feature below is reachable three ways: a Python function, a `qreals`
subcommand (add `--json` for machine output), and a card in `qreals serve`.

## Features

**q-rationals**

| Feature | What it does |
|---|---|
| Exact q-rational `[p/s]_q` | The exact rational function in q for a fraction p/s. |
| q-integer `[n]_q` | The q-analog of a whole number, `[n]_q` and `[n]_{1/q}`. |
| Factor R(q), S(q) | Factor numerator and denominator of `[a/b]_q` over Z[q], labelling each cyclotomic factor. |
| Roots of R(q) | Plot the complex roots of R(q) on the unit circle, splitting cyclotomic roots from the core. |
| Jump gap | The right and left q-versions of p/s and the factored gap between them. |
| Denominator dossier `qreals denom a/d` | One screen per fraction: S(q) expanded and factored, the cyclotomic index set T, deg S vs d-1, the S(1) = d check, the class (FULL / COLLAPSE / REPEATED / NONCYC), a^2 mod d, and every coprime split d = d+ d- with its discrepancy classified EXACT, POLYNOMIAL, or RATIO. Flags: `--json`, `--tex`. |
| Reverse table `qreals collapse d` | The dossier mapped over every numerator coprime to d, grouped by identical S: index set T, factored S, residues per prime-power part of d, realized splits with discrepancy class, RATIO numerators flagged, and the c(d) / non-cyclotomic counts. `--range d1..d2` emits the c(d) sequence one line per d, ready to pipe into `qreals oeis`. Flags: `--json`, `--tex`. |
| Conjecture falsifier `qreals conj NAME` | A registry of falsifiable conjectures (`divisor`, `sqrt-law`, `indices-2ju`, `mult-two`, `floor3`, `negsum-period`) scanned counterexample-first: the run stops at the FIRST counterexample, prints its full dossier, and exits 1; a survivor reports the range covered, the instance count, the wall time, and the three nearest misses by the entry's registered metric, exiting 0. Long scans checkpoint every 60 seconds and continue with `--resume`. `qreals conj list` prints the registry with each statement. Flags: `--until N`, `--resume`, `--state FILE`, `--json`. |

**q-reals**

| Feature | What it does |
|---|---|
| Coefficients `[x]_q` | The first N Taylor coefficients of `[x]_q` for any real x. |
| Laurent expansion | `[x]_q` written out to a chosen power, with its integer-part prefix. |
| Integer-part prefix | The forced opening block of `[x]_q` fixed by `floor(x)`. |
| Convergent locking | How many coefficients the n-th convergent of x pins down. |
| Shift by one `[x ± 1]_q` | `[x+1]_q = q[x]_q + 1`, `[x-1]_q = ([x]_q - 1)/q`. |
| Coefficient read-outs | First nonzero, first negative, largest size, zero count. |
| Radius of convergence | A running-max estimate of the radius of convergence of `[x]_q`. |
| Fingerprint | A named, fixed-length feature vector of `[x]_q` for nearest-neighbour. |
| Certificate | The coefficients of `[x]_q` with a ready-to-paste LaTeX table. |

**Arithmetic** (via the q-Gosper engine)

| Feature | What it does |
|---|---|
| q-sum `[x]_q + [y]_q` | The series sum of the two q-reals. |
| q-product `[x]_q · [y]_q` | The series product of the two q-reals. |
| Deficit | How far `[x]_q +/* [y]_q` sits from `[x +/* y]_q`, with the q=1 and q=0 checks. |

**Symmetry**

| Feature | What it does |
|---|---|
| q-negation `[-x]_q` | The Jouteur negation `[-x]_q` and the x → −x symmetry. |
| Negation-sum finiteness | Whether `[x]_q + [-x]_q` is a finite Laurent polynomial. |

**Visuals** (in `qreals serve`)

| Feature | What it does |
|---|---|
| Coefficient landscape | A 3D surface of the Taylor coefficients of `[x]_q` as n and x vary. |
| Root migration | The complex roots of R(q) as the denominator sweeps. |
| Radius landscape | The radius of convergence of `[a/b]_q` over a Farey grid. |
| Conway-Coxeter frieze | The frieze of a/b > 1 with the q-coefficient overlay on every cell. |

**Lookup**

| Feature | What it does |
|---|---|
| OEIS lookup | Look a coefficient sequence up in the OEIS, re-verified against the b-file. |

### Worked example

```
$ qreals denom 19/60
fraction in lowest terms: 19/60
continued fraction: [0; 3, 6, 3]
S(q) factored = Phi_2 * Phi_3 * Phi_4 * Phi_5 * Phi_6
class: COLLAPSE
60 = 3 * 20   RATIO   (Phi_10 Phi_20) / Phi_6
```

```
$ qreals conj floor3 --until 24
conjecture: floor3
statement: The q-continuant is injective on tails whose entries are all at least 3.
range covered: tails with entries >= 3 and entry sum <= 24, ascending sum then lexicographic
instances checked: 5895
...
verdict: survives the scanned range
```

`qreals denom --help` and `qreals conj --help` carry full worked examples with
their exact output; the test suite replays them, so the documentation cannot
drift from the tools.

## Optional extras

The web app works out of the box. These extras add the terminal interfaces:

```bash
pip install "qreals[app,proof,oeis]"
```

| Extra | What it adds |
|---|---|
| `[app]` | Guided arrow-key menu, run `qreals` with no arguments |
| `[proof]` | Step-by-step certificates, run `qreals certify` |
| `[oeis]` | OEIS lookup |
| `[features]` | numpy for `Fingerprint.as_numpy` |
| `[fast]` | python-flint for fast exact polynomial arithmetic (large denominators) |

## License

MIT. See [LICENSE](./LICENSE).

Created by Patrick Taylor.
