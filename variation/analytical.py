"""
Analytical closed-form mean, variance, and covariance of L, W, Vth.

Reuses stage3_config.json parameters exactly. No Monte Carlo sampling.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from foundations.config_loader import VariationParams


def _build_spatial_covariance(
    coords: Dict[str, Tuple[float, float]],
    sigma: float,
    lam: float,
) -> np.ndarray:
    names = list(coords.keys())
    n = len(names)
    cov = np.zeros((n, n))
    for i, gi in enumerate(names):
        xi, yi = coords[gi]
        for j, gj in enumerate(names):
            xj, yj = coords[gj]
            d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            cov[i, j] = sigma ** 2 * np.exp(-d / lam)
    return cov


def compute_process_moments(
    params: VariationParams,
) -> Dict[str, np.ndarray]:
    names = list(params.gate_coords.keys())
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}

    cov_l = _build_spatial_covariance(params.gate_coords, params.spatial_sigma_l, params.spatial_lambda)
    cov_w = _build_spatial_covariance(params.gate_coords, params.spatial_sigma_w, params.spatial_lambda)
    cov_vth = _build_spatial_covariance(params.gate_coords, params.spatial_sigma_vth, params.spatial_lambda)

    mean_l = np.full(n, params.l_nom_nm)
    mean_w = np.full(n, params.w_nom_nm)
    mean_vth = np.full(n, params.vth_nom_v)

    var_l = np.zeros(n)
    var_w = np.zeros(n)
    var_vth = np.zeros(n)

    cov_l_full = np.zeros((n, n))
    cov_w_full = np.zeros((n, n))
    cov_vth_full = np.zeros((n, n))

    for i in range(n):
        var_l[i] = params.inter_die_sigma_l ** 2 + cov_l[i, i] + params.l_random_sigma_nm ** 2
        var_w[i] = params.inter_die_sigma_w ** 2 + cov_w[i, i] + params.w_random_sigma_nm ** 2

        d_um = float(np.sqrt(
            params.gate_coords[names[i]][0] ** 2 + params.gate_coords[names[i]][1] ** 2
        ))
        w_um = params.w_nom_nm * 1e-3
        l_um = params.l_nom_nm * 1e-3
        vth_random_sigma = float(np.sqrt(
            (params.vth_pelgrom_A_v_um ** 2) / (w_um * l_um) + (params.vth_pelgrom_S_v_um * d_um) ** 2
        ))
        var_vth[i] = params.inter_die_sigma_vth ** 2 + cov_vth[i, i] + vth_random_sigma ** 2

        for j in range(n):
            cov_l_full[i, j] = params.inter_die_sigma_l ** 2 + cov_l[i, j]
            cov_w_full[i, j] = params.inter_die_sigma_w ** 2 + cov_w[i, j]
            cov_vth_full[i, j] = params.inter_die_sigma_vth ** 2 + cov_vth[i, j]

    return {
        "names": names,
        "idx": idx,
        "mean_l": mean_l,
        "mean_w": mean_w,
        "mean_vth": mean_vth,
        "var_l": var_l,
        "var_w": var_w,
        "var_vth": var_vth,
        "cov_l": cov_l_full,
        "cov_w": cov_w_full,
        "cov_vth": cov_vth_full,
    }
