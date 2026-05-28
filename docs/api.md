# API reference

The source of truth for what is public is `src/qreals/__init__.py`. Everything
below is re-exported from the top-level `qreals` namespace; anything not listed
here is internal and may change without notice. Each entry is generated from the
docstring in the source.

## The two computation paths

::: qreals.rational
    options:
      members:
        - q_rational
        - q_int
        - q_int_qinv

::: qreals.truncated
    options:
      members:
        - q_real_truncated

The module symbol `qreals.q` is the sympy symbol used in the rational-function
results of `q_rational`.

## Arithmetic between q-reals

::: qreals.arithmetic
    options:
      members:
        - q_add
        - q_mul
        - q_neg
        - negation_sum
        - finite_xnegx
        - radius

::: qreals.qreal
    options:
      members:
        - QReal

## The bihomographic cross-check engine

::: qreals.gosper
    options:
      members:
        - q_gosper
        - gosper_coeffs

## Verification stamp

::: qreals.verify
    options:
      members:
        - verify
        - Stamp

## Laurent expansions and the board lemmas

::: qreals.expansions
    options:
      members:
        - integer_part_prefix
        - coeffs_locked_by_convergent
        - mgo_laurent
        - shift_down
        - shift_up
        - format_laurent

## Coefficient read-outs

::: qreals.coefficients
    options:
      members:
        - first_nonzero_coefficient_index
        - first_negative_coefficient_index
        - coefficient_max_abs
        - number_of_zeros

## Exploration helpers (optional extras)

::: qreals.features
    options:
      members:
        - featurize
        - feature_names
        - feature_distance
        - nearest
        - Fingerprint

::: qreals.oeis
    options:
      members:
        - lookup
        - available
        - LookupResult
        - Hit
        - OeisUnavailable
