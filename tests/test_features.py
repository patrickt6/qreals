"""Tests for qreals.featurize.

These confirm the fingerprint is named, fixed length, deterministic, and that the
features read the q-series the way the docstring claims. The numpy path is tested
only when numpy is installed.
"""

from __future__ import annotations

import importlib.util

import pytest

from qreals import feature_distance, feature_names, featurize, nearest
from qreals.features import DEFAULT_N_CF, DEFAULT_N_COEFFS, Fingerprint


def test_fingerprint_is_named_and_fixed_length():
    fp = featurize("pi")
    assert isinstance(fp, Fingerprint)
    assert len(fp.names) == len(fp.values)
    assert fp.names == feature_names()
    # 15 scalars + n_cf partial quotients + n_cf partial sums + n_coeffs coeffs.
    assert len(fp.values) == 15 + 2 * DEFAULT_N_CF + DEFAULT_N_COEFFS


def test_fingerprint_length_is_the_same_for_every_constant():
    a = featurize("pi")
    b = featurize("sqrt(2)")
    c = featurize("(1+sqrt(5))/2")
    assert a.names == b.names == c.names
    assert len({len(a.values), len(b.values), len(c.values)}) == 1


def test_featurize_is_deterministic():
    first = featurize("sqrt(2)")
    second = featurize("sqrt(2)")
    assert first.names == second.names
    assert first.values == second.values  # exact, bit-for-bit


def test_features_read_the_continued_fraction():
    fp = featurize("pi").as_dict()
    # pi = [3; 7, 15, 1, 292, ...]
    assert fp["cf_0"] == 3.0
    assert fp["cf_1"] == 7.0
    assert fp["cf_2"] == 15.0
    assert fp["cf_partial_sum_1"] == 3.0
    assert fp["cf_partial_sum_2"] == 10.0  # 3 + 7


def test_features_read_the_q_series():
    fp = featurize("pi").as_dict()
    # [pi]_q = 1 + q + q^2 + ... , so it opens 1, 1, 1 with valuation 0.
    assert fp["valuation"] == 0.0
    assert fp["c_0"] == 1.0
    assert fp["c_1"] == 1.0
    assert fp["c_2"] == 1.0


def test_fractional_constant_has_valuation_one():
    fp = featurize("1/2").as_dict()
    # 0 < 1/2 < 1, so [1/2]_q has constant term 0 and valuation 1.
    assert fp["valuation"] == 1.0
    assert fp["c_0"] == 0.0


def test_integer_has_zero_inverse_radius():
    # [5]_q is a polynomial 1 + q + q^2 + q^3 + q^4, so its coefficients do not
    # grow: the running-max slope (1 / radius) is 0.
    fp = featurize("5").as_dict()
    assert fp["inv_radius"] == 0.0


def test_partial_sums_plateau_for_a_rational():
    # 3/2 = [1; 2], a terminating continued fraction, so beyond cf_len the padded
    # partial quotients are 0 and the cumulative sums stop growing.
    fp = featurize("3/2").as_dict()
    assert fp["cf_len"] == 2.0
    last = fp[f"cf_partial_sum_{DEFAULT_N_CF}"]
    assert fp[f"cf_partial_sum_{DEFAULT_N_CF - 1}"] == last  # already plateaued


def test_feature_distance_and_nearest():
    pi = featurize("pi")
    sqrt2 = featurize("sqrt(2)")
    golden = featurize("(1+sqrt(5))/2")
    assert feature_distance(pi, pi) == 0.0
    assert feature_distance(pi, sqrt2) == feature_distance(sqrt2, pi)  # symmetric
    hits = nearest(pi, [sqrt2, golden, pi])
    assert hits[0][0] is pi  # a constant is its own nearest neighbour
    assert hits[0][1] == 0.0


def test_feature_distance_rejects_mismatched_parameters():
    a = featurize("pi", n_coeffs=12)
    b = featurize("pi", n_coeffs=16)
    with pytest.raises(ValueError):
        feature_distance(a, b)


def test_parameters_change_the_length_predictably():
    fp = featurize("pi", n_cf=4, n_coeffs=10, n_radius=20)
    assert len(fp.values) == 15 + 2 * 4 + 10
    assert fp.params == (4, 10, 20)


def test_bad_parameters_are_rejected():
    with pytest.raises(ValueError):
        featurize("pi", n_radius=1)
    with pytest.raises(ValueError):
        featurize("pi", n_cf=0)


@pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None, reason="numpy not installed"
)
def test_as_numpy_returns_the_vector():
    import numpy as np

    fp = featurize("pi")
    arr = fp.as_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(fp.values),)
    assert list(arr) == fp.values
