# Quickstart (five minutes)

This page takes you from a clone to a computed `[x]_q` and the interactive app.

## 1. Install

Requires Python 3.11 or newer. The only runtime dependency is sympy.

```bash
pip install -e .
```

To also get the arrow-key app described below, add the interface extra:

```bash
pip install -e .[app]
```

## 2. Open the app

With the `app` extra installed, run the bare command:

```bash
qreals
```

That opens a menu you move through with the arrow keys and Enter. There is one
entry per computation: exact `[p/s]_q`, the q-integers `[n]_q`, the coefficients
of `[x]_q`, the MGO Laurent expansion, the integer-part prefix, convergent
locking, the shift relations `[x +/- 1]_q`, the coefficient read-outs, the
arithmetic `[x]_q +/* [y]_q`, the q-negation `[-x]_q` with its `x -> -x`
finiteness, the radius-of-convergence estimate, an OEIS lookup, and a
fixed-length fingerprint of a constant. Each entry asks for its inputs one at a
time, shows an example, checks what you type, and prints a formatted result.

If you would rather not install the extra, every entry is also a subcommand:

```bash
qreals rational 3 2
qreals coeffs pi 12
qreals coeffs "(1+sqrt(5))/2" 30 --json
```

`python -m qreals` is equivalent to the bare command. Run `qreals --help` to
list every subcommand, and `qreals doctor` to check that the menu will run in
your terminal.

## 3. Compute from Python

```python
import qreals

# Exact q-rationals, as elements of Q(q).
qreals.q_rational(3, 2)      # (q**2 + q + 1)/(q + 1)
qreals.q_rational(1, 2)      # q/(q + 1)

# First N stable Taylor coefficients of [x]_q, for any real x.
qreals.q_real_truncated("pi", 12)
# [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
qreals.q_real_truncated("sqrt(2)", 30)
qreals.q_real_truncated("(1+sqrt(5))/2", 30)   # the golden ratio
```

The string passed to `q_real_truncated` is parsed by sympy, so anything sympy
understands works: `"pi"`, `"sqrt(2)"`, `"(1+sqrt(5))/2"`, `"E"`, `"3/2"`. The
coefficients come back as a plain list of Python ints, c_0 first.

## 4. Verification, every time

Every computation prints a one-line stamp by default. It reruns the cheap
cross-checks that apply to that input:

```
qreals coeffs pi 12
...
verified: truncation stable to 12, shift law [x+1]=q[x]+1; n/a: q=1 specialisation; exact rational function
```

The checks are independent recomputations. When a check cannot run for an input
(q = 1 on an irrational, say), the stamp says n/a rather than claiming a pass.
For the full reasoning, `qreals certify coeffs pi 12` prints a human-auditable
derivation. See [Correctness and proofs](CORRECTNESS.md) for what each check
means.

## 5. Run the test suite

```bash
pip install -e .[dev]
python -m pytest tests/ -v
```

The suite anchors the implementation to the worked examples in the source paper
and to the defining properties (the q = 1 specialisation, truncation stability,
and exact-equals-truncated agreement on rationals).
