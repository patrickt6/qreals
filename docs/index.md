# qreals

Compute q-deformed rational and real numbers. Pure Python, exact integer
coefficients, one runtime dependency (sympy).

**The question this answers:** given a real number x, what is its q-analog
`[x]_q`, and what are the coefficients of its power series in q?

An ordinary integer n has a standard q-analog,

```
[n]_q = 1 + q + q^2 + ... + q^{n-1}
```

which collapses back to n when q = 1. Morier-Genoud and Ovsienko (MGO) extended
this from integers to rationals and then to all real numbers, using continued
fractions. `qreals` implements that construction directly:

- `q_rational(p, s)` returns the exact `[p/s]_q` as a reduced rational function
  in q.
- `q_real_truncated(x, N)` returns the first N power-series coefficients of
  `[x]_q` for any real x, with a guarantee that those N coefficients are stable.

## Where to go next

- [Quickstart](quickstart.md): install, run the interactive app, and read the
  first coefficients in five minutes.
- [The MGO construction](math.md): what `[x]_q` is, in plain language, and how
  the stable-coefficient guarantee works.
- [Correctness and proofs](CORRECTNESS.md): for every public function, the
  theorem it computes and the independent check that confirms it.
- [API reference](api.md): every public function and class, generated from the
  source.

## Two computation paths

| Input | Function | Returns |
|---|---|---|
| rational p/s | `q_rational(p, s)` | exact rational function in q |
| any real x | `q_real_truncated(x, N)` | first N integer coefficients of the q-series |

Coefficients are exact throughout. The series path holds each coefficient as a
Python int and inverts series with Newton's iteration over the integers, so
nothing is lost to floating point.

## References

- S. Morier-Genoud and V. Ovsienko, "q-deformed rationals and q-continued
  fractions", Forum Math. Sigma 8 (2020), e13.
- S. Morier-Genoud and V. Ovsienko, "On q-deformed real numbers"
  (arXiv:1908.04365).
- A. Jouteur, "Modular group action on q-deformed real numbers"
  (arXiv:2503.02122), the negation used by `q_neg`.
- V. Ovsienko, "Modular invariant q-deformed numbers: first steps", Example 6.4,
  the `x -> -x` finiteness studied by `finite_xnegx`.
