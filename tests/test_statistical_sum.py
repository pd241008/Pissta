import math

import numpy as np
import pytest

from foundations.config_loader import load_config
from ssta.statistical_sum import gaussian_sum, propagate_path
from timing.delay import compute_delay_moments
from variation.analytical import compute_process_moments


def _synthetic_moments(n: int, seed: int = 0):
    """Random PSD-ish covariance structure with per-gate independent components.

    The per-gate random term appears in var_d[i] but NOT in cov_d's
    diagonal -- mirroring the real pipeline, where the Pelgrom / random L,W
    components enter var_d but not the shared covariance matrix. A naive
    "sum the full covariance submatrix" fix would drop exactly these terms.
    """
    rng = np.random.default_rng(seed)
    b = rng.uniform(0.05, 1.0, size=(n, n + 1))   # positive weights -> all covariances > 0
    cov_d = b @ b.T + np.eye(n) * 0.5             # SPD shared-part covariance
    indep = rng.uniform(0.01, 0.1, size=n)           # per-gate independent variance
    var_d = np.diag(cov_d).copy() + indep
    mean_d = rng.uniform(0.5, 2.0, size=n)
    idx = {f"G{i+1}": i for i in range(n)}
    return mean_d, var_d, cov_d, idx


def test_two_gate_path_bit_compatible_with_gaussian_sum():
    mean_d, var_d, cov_d, idx = _synthetic_moments(5)
    mu, var = propagate_path(["G1", "G3"], mean_d, var_d, cov_d, idx)
    mu_ref, var_ref = gaussian_sum(
        float(mean_d[idx["G1"]]), float(var_d[idx["G1"]]),
        float(mean_d[idx["G3"]]), float(var_d[idx["G3"]]),
        float(cov_d[idx["G1"], idx["G3"]]),
    )
    assert mu == mu_ref and var == var_ref


def test_three_gate_path_matches_closed_form():
    mean_d, var_d, cov_d, idx = _synthetic_moments(5)
    path = ["G1", "G3", "G5"]
    mu, var = propagate_path(path, mean_d, var_d, cov_d, idx)
    assert mu == pytest.approx(sum(float(mean_d[idx[g]]) for g in path), rel=1e-12)
    closed = sum(float(var_d[idx[g]]) for g in path)
    for a in range(len(path)):
        for b in range(a + 1, len(path)):
            closed += 2.0 * float(cov_d[idx[path[a]], idx[path[b]]])
    assert var == pytest.approx(closed, rel=1e-12)


def test_pairs_each_gate_against_full_accumulated_set():
    """The fix's core property: the middle-gate covariance term is present."""
    mean_d, var_d, cov_d, idx = _synthetic_moments(5)
    path = ["G1", "G2", "G3", "G4"]
    _, var = propagate_path(path, mean_d, var_d, cov_d, idx)
    # naive first-gate-only pairing (the old bug) for comparison
    naive = float(var_d[idx[path[0]]])
    for g in path[1:]:
        naive += float(var_d[idx[g]]) + 2.0 * float(cov_d[idx[path[0]], idx[g]])
    assert var > naive  # strictly larger when Cov(mid, last) terms are positive
    # and exactly the naive value plus the dropped pairwise terms
    dropped = 2.0 * sum(
        float(cov_d[idx[path[a]], idx[path[b]]]) for a in range(1, len(path)) for b in range(a + 1, len(path))
    )
    assert var == pytest.approx(naive + dropped, rel=1e-12)


def test_frozen_config_at_g4_regression_values():
    """Known values on the frozen six-gate config (post-fix).

    AT_G4 mean is unchanged by the fix; std gains the 2*Cov(G2,G4) term
    (0.007990 = 10.27% of the old variance). MC empirical std: 0.296434.
    """
    cfg = load_config("foundations/stage3_config.json")
    pm = compute_process_moments(cfg.variation_params)
    dm = compute_delay_moments(cfg.timing_params, cfg.variation_params, pm)
    mu, var = propagate_path(["G1", "G2", "G4"], dm["mean_d"], dm["var_d"], dm["cov_d"], dm["idx"])
    assert mu == pytest.approx(6.410875078798936, rel=1e-12)
    assert math.sqrt(var) == pytest.approx(0.292914, abs=1e-5)
