import math

import numpy as np
import pytest

from data_loader import load_parameters
from montecarlo import sample_parameters, sample_triangular


def test_triangular_within_bounds_and_mean():
    rng = np.random.default_rng(0)
    values = sample_triangular(16000, 17000, 17000, 10000, rng)

    assert values.min() >= 16000 and values.max() <= 17000, "values must stay inside [min, max]"
    assert values.mean() == pytest.approx((16000 + 17000 + 17000) / 3, rel=0.01), "mean of triangular is (min+likely+max)/3"


def test_triangular_degenerate_range():
    rng = np.random.default_rng(0)
    values = sample_triangular(4, 4, 4, 100, rng)

    assert len(values) == 100 and (values == 4).all(), "min == max must give a constant"


def test_triangular_missing_likely_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="likely"):
        sample_triangular(0, math.nan, 23000, 10, rng)


def test_triangular_unordered_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_triangular(10, 5, 20, 10, rng)


def test_same_seed_same_numbers_different_seed_different_numbers():
    params = {"S0": (16000, 17000, 17000)}

    a = sample_parameters(params, 100, seed=1)["S0"]
    b = sample_parameters(params, 100, seed=1)["S0"]
    c = sample_parameters(params, 100, seed=2)["S0"]

    assert (a == b).all(), "same seed must reproduce the sample"
    assert not (a == c).all(), "different seed must change the sample"


def test_sample_parameters_from_csv():
    params = load_parameters("input")
    names = ["S0_phase1", "R0_engaged", "G_S"]

    sample = sample_parameters(params, 1000, seed=42, names=names)

    assert set(sample) == set(names), "only the requested parameters are sampled"
    for name in names:
        low, _, high = params[name]
        assert len(sample[name]) == 1000, f"{name}: wrong number of draws"
        assert sample[name].min() >= low and sample[name].max() <= high, f"{name}: outside [min, max]"


def test_sample_parameters_all_inputs_fails_on_missing_likely():
    params = load_parameters("input")

    with pytest.raises(ValueError, match="likely"):
        sample_parameters(params, 10, seed=0)   # A_R / A_S have no 'likely'