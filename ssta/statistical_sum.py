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
    mu = float(mean_d[idx[path[0]]])
    var = float(var_d[idx[path[0]]])
    for name in path[1:]:
        i = idx[name]
        mu, var = gaussian_sum(
            mu_x=mu,
            var_x=var,
            mu_y=float(mean_d[i]),
            var_y=float(var_d[i]),
            cov_xy=float(cov_d[idx[path[0]], i]),
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
