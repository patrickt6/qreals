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

## License

MIT. See [LICENSE](./LICENSE).

Created by Patrick Taylor.
