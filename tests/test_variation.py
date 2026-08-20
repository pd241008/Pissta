import numpy as np
import pytest

from variation.sampler import sample_correlated_process
from foundations.config_loader import VariationParams


def test_sample_process_variation_shape():
    vp = VariationParams()
    coords = {"g0": (0.0, 0.0), "g1": (1.0, 0.0)}
    rng = np.random.default_rng(42)
    samples = sample_correlated_process(10, vp, rng, gate_coords=coords)
    assert samples["L_nm"].shape == (10, 2)


def test_sample_process_variation_spread():
    vp = VariationParams()
    coords = {"g0": (0.0, 0.0)}
    rng = np.random.default_rng(42)
    samples = sample_correlated_process(10000, vp, rng, gate_coords=coords)
    assert samples["L_nm"].std() > 0.1


def test_sample_process_variation_correlation():
    vp = VariationParams()
    coords = {"g0": (0.0, 0.0), "g1": (1.0, 0.0)}
    rng = np.random.default_rng(42)
    samples = sample_correlated_process(10000, vp, rng, gate_coords=coords)
    corr = np.corrcoef(samples["L_nm"][:, 0], samples["L_nm"][:, 1])[0, 1]
    assert 0.0 < corr < 1.0
