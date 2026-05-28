# The MGO construction

This page explains what `[x]_q` is in plain language, and why the coefficients
`qreals` returns are trustworthy. The mathematics is from one paper:
Morier-Genoud and Ovsienko, "q-deformed rationals and q-continued fractions",
Forum Math. Sigma 8 (2020), e13, Definition 1.1 and Proposition 1.1.

## From integers to rationals to reals

Start with the q-integer everyone knows:

```
[n]_q = 1 + q + q^2 + ... + q^{n-1}
```

Set q = 1 and it collapses back to n. The MGO idea is to extend the map
`x -> [x]_q` so that it makes sense first for rationals and then for all reals,
while keeping the q = 1 specialisation honest.

For a rational the route is the continued fraction. Write x = p/s as a regular
continued fraction `[a_1, a_2, ..., a_m]`, normalise it to even length, and fold
the MGO formula through it: odd positions (counting from one) carry `[a]_q` with
`q^a` above, even positions carry `[a]_{q^{-1}}` with `q^{-a}` above. The result
`[p/s]_q` is an honest rational function of q. For example,

```
[3/2]_q = (1 + q + q^2) / (1 + q)
[1/2]_q = q / (1 + q)
```

Both specialise to the ordinary value at q = 1.

## The series for an irrational

For an irrational x the continued fraction does not terminate, so `[x]_q` is not
a rational function but a power series in q with integer coefficients. That
series is the object people study: its coefficients carry arithmetic information
about x, and which patterns appear in them is an open research question. The
golden ratio, sqrt(2), pi, and e all give different and structured coefficient
sequences.

`q_real_truncated(x, N)` returns the first N coefficients of that series.

## The stable-coefficient guarantee

The depth of continued fraction needed is not a tuning knob. MGO Proposition 1.1
says: stop the continued fraction at the first point where the partial quotients
sum to at least N + 1, and exactly that-sum-minus-one of the resulting power
series coefficients agree with the true `[x]_q`. They will not change if you ask
for more.

`q_real_truncated(x, N)` accumulates partial quotients until the sum clears N, so
the N coefficients it returns are stable by that proposition. Two consequences:

- For a rational p/s the continued fraction terminates, so the truncated series
  is just the Taylor expansion of the exact `q_rational(p, s)`. The test suite
  checks that the two paths agree coefficient for coefficient.
- The worst case for depth is the golden ratio, whose continued fraction is all
  ones, so reaching N stable coefficients takes depth N + 1. Everything else is
  faster.

## Exactness

Coefficients are exact throughout. The series kernel holds each coefficient as a
Python int and inverts series with Newton's iteration over the integers, so
nothing is lost to floating point. This matters when the coefficients are used as
data: there is no measurement noise to model around, and the same engine that
generates an example can check a prediction about it.

## Arithmetic, and two cautions

`qreals` also computes the series sum and product of two q-reals, the Jouteur
q-negation, and a radius-of-convergence estimate. Two cautions, both spelled out
in the [correctness notes](CORRECTNESS.md) and the docstrings:

- `q_add` and `q_mul` are the sum and product of the two **series** `[x]_q` and
  `[y]_q`, not `[x + y]_q` or `[x * y]_q`. The MGO map `x -> [x]_q` is not a ring
  homomorphism, so these differ already at the constant term.
- `q_neg` is the PGL_2(Z)-action negation of Jouteur, an involution, not the
  coefficient-wise negation of `[x]_q` and not the MGO series of the real number
  -x. It is the tool behind Ovsienko's Example 6.4: for which real x is
  `[x]_q + [-x]_q` a finite Laurent polynomial?

## Why a second engine

Every arithmetic result is cross-checked against an independent algorithm, the
bihomographic q-Gosper engine (`q_gosper`, `gosper_coeffs`). It reaches the same
quantity by a state machine over rational functions in q, never forming the two
series separately, so agreement between the two paths is a real check rather than
a re-run of one code path. The [correctness notes](CORRECTNESS.md) map every
public function to the theorem it computes and the check that confirms it.
