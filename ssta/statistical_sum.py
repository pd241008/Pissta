"""
Statistical sum propagation for Gaussian random variables.

If X ~ N(μ_X, σ_X²) and Y ~ N(μ_Y, σ_Y²) with Cov(X,Y) = σ_XY,
then X+Y ~ N(μ_X+μ_Y, σ_X² + σ_Y² + 2σ_XY).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def gaussian_sum(
    mu_x: float,
    var_x: float,
    mu_y: float,
    var_y: float,
    cov_xy: float,
) -> Tuple[float, float]:
    mu_z = mu_x + mu_y
    var_z = var_x + var_y + 2.0 * cov_xy
    return float(mu_z), float(var_z)


def propagate_path(
    path: List[str],
    mean_d: np.ndarray,
    var_d: np.ndarray,
    cov_d: np.ndarray,
    idx: Dict[str, int],
) -> Tuple[float, float]:
    """Exact Gaussian path propagation: pair each gate against the full
    accumulated set, not just the first gate.

    Var(sum) = sum_i var_i + 2 * sum_{i<j} cov_ij over ALL path pairs.
    """
    mu = float(mean_d[idx[path[0]]])
    var = float(var_d[idx[path[0]]])
    for n, name in enumerate(path[1:], start=1):
        i = idx[name]
        # Cov(running sum, d_k) = sum of covariances against EVERY gate
        # already accumulated. Pairing only against path[0] drops the
        # middle-gate covariance terms on 3+-gate paths.
        cov_running = sum(float(cov_d[idx[prev], i]) for prev in path[:n])
        mu, var = gaussian_sum(
            mu_x=mu,
            var_x=var,
            mu_y=float(mean_d[i]),
            var_y=float(var_d[i]),
            cov_xy=cov_running,
        )
    return float(mu), float(var)


def covariance_between_paths(
    path_a: List[str],
    path_b: List[str],
    cov_d: np.ndarray,
    idx: Dict[str, int],
) -> float:
    total = 0.0
    for ga in path_a:
        for gb in path_b:
            total += float(cov_d[idx[ga], idx[gb]])
    return float(total)
