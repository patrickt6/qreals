"""A generalized anomaly hunter over q-real feature fingerprints.

The productized form of a workflow the project has already run by hand twice:
the S1 phenomenon hunt (sweep a family, look for the inputs whose statistics
sit far from the crowd) and the t2 artifact catch (an "anomaly" that turned
out to be a computation artifact, found the same way). `glue.py` hunts one
specific anomaly (sharing-class sizes at prime powers); this module hunts any
of them: compute a dict of named numeric properties per input, fit the
population statistics per property, and rank the inputs by how far outside
the population they sit.

The default property set is the fingerprint from `features.featurize`, whose
vector mixes continued-fraction shape, coefficient shape, and growth scalars,
so an input can stand out for many different reasons; the witness report says
which feature did it. Any custom property function mapping an input string to
a dict of named numbers can be swapped in, so a hunt can run over s_properties
degrees, deficit magnitudes, or anything else the engine computes.

The statistics are deliberately cheap and dependency-free (numpy stays an
optional extra elsewhere in qreals and is not used here): per feature, the
population mean and population standard deviation; per input and feature, the
z-score (value - mean) / std, taken as 0 where the population is constant;
per input, the outlier score max |z| over the features, with ties broken by
the sum of |z|. Everything is deterministic: the same inputs in the same
order give byte-identical rankings, witnesses included.

A score is a screening statistic, not a verdict. A large max |z| marks an
input worth a human look (or worth `qreals why` and a dossier); it proves
nothing by itself, and the two hand-run hunts show both outcomes: a real
phenomenon and an artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import gcd
from typing import Callable, Iterable, Mapping, Sequence

from .factor import s_properties, s_regime
from .features import featurize

# An input string in, a dict of named numeric properties out.
PropertyFn = Callable[[str], Mapping[str, float]]

DEFAULT_TOP = 10
DEFAULT_WITNESSES = 3

FRACTION_KINDS = ("fingerprint", "sprops")


@dataclass(frozen=True)
class Witness:
    """One feature's evidence that an input is an outlier.

    Fields:
        feature: the property name, e.g. "inv_radius" or "deg_S".
        value: this input's value of the property.
        mean: the population mean of the property over all hunted inputs.
        std: the population standard deviation over all hunted inputs.
        z: the z-score (value - mean) / std, 0 when std is 0.
    """

    feature: str
    value: float
    mean: float
    std: float
    z: float


@dataclass(frozen=True)
class Outlier:
    """One ranked input with the features that make it stand out.

    Fields:
        input: the input string, e.g. "4/15".
        score: the outlier score, max |z| over all features.
        z_sum: the sum of |z| over all features, the tie-breaker.
        witnesses: the strongest features for this input, largest |z| first;
            the first witness is the feature realising the score.
    """

    input: str
    score: float
    z_sum: float
    witnesses: list[Witness]


def _fingerprint_properties(x: str) -> Mapping[str, float]:
    """The default property set: the full featurize fingerprint as a dict."""
    return featurize(x).as_dict()


def _sprops_properties(x: str) -> Mapping[str, float]:
    """Numeric properties of the q-denominator S(q) of a fraction a/d.

    Encodes the s_properties record as plain numbers: the degree of S against
    its bound d - 1, the collapse drop, the saturation index e* (encoded -1 in
    the impossibility branch, where no n has S | [n]_q), the size of the
    cyclotomic index set, and the 0/1 regime flags. The encoding is fixed so
    hunts over different ranges are comparable.
    """
    p = s_properties(x)
    e_star = -1.0 if p.saturation_index is None else float(p.saturation_index)
    return {
        "deg_S": float(p.deg_S),
        "deg_bound": float(p.deg_bound),
        "collapse_drop": float(p.deg_bound - p.deg_S),
        "saturation_index": e_star,
        "n_cyclotomic_indices": float(len(p.index_set_T)),
        "is_cyclotomic": 1.0 if p.is_cyclotomic else 0.0,
        "is_squarefree": 1.0 if p.is_squarefree else 0.0,
        "is_full_qint": 1.0 if s_regime(p) == "full" else 0.0,
        "equality_locus": 1.0 if p.equality_locus else 0.0,
    }


def _population_stats(values: Sequence[float]) -> tuple[float, float]:
    """Population mean and population standard deviation, in pure python."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(variance)


def hunt(
    property_fn: PropertyFn | None,
    inputs: Iterable[str],
    top: int = DEFAULT_TOP,
    *,
    witnesses: int = DEFAULT_WITNESSES,
) -> list[Outlier]:
    """Rank the inputs whose properties sit farthest from the population.

    Args:
        property_fn: maps an input string to a dict of named numeric
            properties. None uses the fingerprint from `features.featurize`,
            so any real-number input works out of the box. Every input must
            produce the same property names (featurize guarantees this).
        inputs: the input strings to hunt over, e.g. fractions or surds. At
            least two are required; one input has no population to deviate
            from.
        top: how many ranked outliers to return.
        witnesses: how many per-input witness features to attach, largest
            |z| first.

    Returns:
        The ``top`` inputs ranked by outlier score (max |z| over features),
        ties broken by the sum of |z| and then by the input string, so the
        ranking is deterministic. Each carries its witness features with the
        population mean and standard deviation they were judged against.
    """
    if top < 1:
        raise ValueError("top must be at least 1")
    if witnesses < 1:
        raise ValueError("witnesses must be at least 1")
    fn: PropertyFn = property_fn if property_fn is not None else (
        _fingerprint_properties
    )

    names: list[str] | None = None
    table: list[tuple[str, dict[str, float]]] = []
    for x in inputs:
        props = {str(k): float(v) for k, v in fn(x).items()}
        if names is None:
            names = list(props.keys())
        elif set(props.keys()) != set(names):
            raise ValueError(
                f"property names for {x!r} differ from the first input's; "
                "every input must produce the same named properties"
            )
        table.append((x, props))
    if names is None or len(table) < 2:
        raise ValueError("hunt needs at least two inputs to form a population")

    stats = {
        name: _population_stats([props[name] for _, props in table])
        for name in names
    }

    outliers: list[Outlier] = []
    for x, props in table:
        scored: list[Witness] = []
        for name in names:
            mean, std = stats[name]
            z = 0.0 if std == 0.0 else (props[name] - mean) / std
            scored.append(
                Witness(feature=name, value=props[name], mean=mean, std=std, z=z)
            )
        score = max(abs(w.z) for w in scored)
        z_sum = sum(abs(w.z) for w in scored)
        scored.sort(key=lambda w: (-abs(w.z), w.feature))
        outliers.append(
            Outlier(input=x, score=score, z_sum=z_sum, witnesses=scored[:witnesses])
        )
    outliers.sort(key=lambda o: (-o.score, -o.z_sum, o.input))
    return outliers[:top]


def _coprime_fractions(
    d_range: Iterable[int], a_max: int | None
) -> list[str]:
    """The proper fractions a/d in lowest terms over the given denominators."""
    inputs: list[str] = []
    for d in d_range:
        d = int(d)
        if d < 2:
            continue
        cap = d - 1 if a_max is None else min(a_max, d - 1)
        for a in range(1, cap + 1):
            if gcd(a, d) == 1:
                inputs.append(f"{a}/{d}")
    return inputs


def hunt_fractions(
    kind: str,
    d_range: Iterable[int],
    *,
    a_max: int | None = None,
    top: int = DEFAULT_TOP,
    witnesses: int = DEFAULT_WITNESSES,
) -> list[Outlier]:
    """Hunt for anomalous coprime fractions a/d over a denominator range.

    The convenience front end for the grid the project sweeps most: every
    proper fraction a/d in lowest terms (0 < a/d < 1), d over ``d_range``,
    numerators capped by ``a_max`` when given.

    Args:
        kind: which property set to hunt over. "fingerprint" uses the
            featurize vector of [a/d]_q (coefficient and CF shape);
            "sprops" uses the numeric S(q) denominator properties (degree,
            collapse drop, saturation index, regime flags).
        d_range: the denominators to sweep, e.g. range(2, 50).
        a_max: optional cap on the numerator a for each d.
        top: how many ranked outliers to return.
        witnesses: how many witness features to attach per outlier.

    Returns:
        The ranked outliers, as :func:`hunt` returns them; deterministic for
        a given range since the fractions are generated in (d, a) order.
    """
    if kind not in FRACTION_KINDS:
        raise ValueError(
            f"kind must be one of {FRACTION_KINDS}, got {kind!r}"
        )
    fn = _fingerprint_properties if kind == "fingerprint" else _sprops_properties
    inputs = _coprime_fractions(d_range, a_max)
    if len(inputs) < 2:
        raise ValueError(
            "the fraction grid has fewer than two inputs; widen d_range"
        )
    return hunt(fn, inputs, top, witnesses=witnesses)
