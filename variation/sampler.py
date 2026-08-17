"""
Correlated process variation sampler.

Generates correlated L, W, Vth samples with three components:
1. Inter-die: one global scalar per Monte Carlo sample per parameter.
2. Spatial intra-die: correlated via distance-based exponential covariance.
3. Random mismatch: independent per-gate Pelgrom-style noise.

Output arrays have shape (N_samples, 6) for L, W, Vth.
"""

from __future__ import annotations

from foundations.config_loader import VariationParams
from typing import Dict

import numpy as np


def _build_spatial_covariance(
    coords: Dict[str, tuple[float, float]],
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


def _sample_spatial_intra_die(
    n_samples: int,
    cov: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(cov)
    mask = eigvals > 1e-12
    eigvals_r = eigvals[mask]
    eigvecs_r = eigvecs[:, mask]
    z = rng.standard_normal((len(eigvals_r), n_samples))
    samples = (eigvecs_r * np.sqrt(eigvals_r)[None, :]) @ z
    return samples.T


def _pelgrom_vth_sigma(
    w_nm: float,
    l_nm: float,
    a_vth: float,
    s_vth: float,
    d_um: float,
) -> float:
    w_um = w_nm * 1e-3
    l_um = l_nm * 1e-3
    return float(np.sqrt((a_vth ** 2) / (w_um * l_um) + (s_vth * d_um) ** 2))


def sample_correlated_process(
    n_samples: int,
    params: VariationParams,
    rng: np.random.Generator,
    gate_coords: Dict[str, tuple[float, float]] | None = None,
) -> Dict[str, np.ndarray]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if gate_coords is not None:
        names = list(gate_coords.keys())
        coords = gate_coords
    else:
        names = list(params.gate_coords.keys())
        coords = params.gate_coords
    n_gates = len(names)

    inter_l = rng.normal(0.0, params.inter_die_sigma_l, n_samples)
    inter_w = rng.normal(0.0, params.inter_die_sigma_w, n_samples)
    inter_vth = rng.normal(0.0, params.inter_die_sigma_vth, n_samples)

    cov_l = _build_spatial_covariance(coords, params.spatial_sigma_l, params.spatial_lambda)
    cov_w = _build_spatial_covariance(coords, params.spatial_sigma_w, params.spatial_lambda)
    cov_vth = _build_spatial_covariance(coords, params.spatial_sigma_vth, params.spatial_lambda)

    spatial_l = _sample_spatial_intra_die(n_samples, cov_l, rng)
    spatial_w = _sample_spatial_intra_die(n_samples, cov_w, rng)
    spatial_vth = _sample_spatial_intra_die(n_samples, cov_vth, rng)

    l_random = np.zeros((n_samples, n_gates))
    w_random = np.zeros((n_samples, n_gates))
    vth_random = np.zeros((n_samples, n_gates))

    for idx, name in enumerate(names):
        l_random[:, idx] = rng.normal(0.0, params.l_random_sigma_nm, n_samples)
        w_random[:, idx] = rng.normal(0.0, params.w_random_sigma_nm, n_samples)

        d_um = float(np.sqrt(
            coords[name][0] ** 2 + coords[name][1] ** 2
        ))
        sigma_vth = _pelgrom_vth_sigma(
            params.w_nom_nm, params.l_nom_nm,
            params.vth_pelgrom_A_v_um,
            params.vth_pelgrom_S_v_um,
            d_um,
        )
        vth_random[:, idx] = rng.normal(0.0, sigma_vth, n_samples)

    L = np.zeros((n_samples, n_gates))
    W = np.zeros((n_samples, n_gates))
    Vth = np.zeros((n_samples, n_gates))

    for idx in range(n_gates):
        L[:, idx] = params.l_nom_nm + inter_l + spatial_l[:, idx] + l_random[:, idx]
        W[:, idx] = params.w_nom_nm + inter_w + spatial_w[:, idx] + w_random[:, idx]
        Vth[:, idx] = params.vth_nom_v + inter_vth + spatial_vth[:, idx] + vth_random[:, idx]

    if np.any(L <= 0) or np.any(W <= 0):
        raise RuntimeError("Generated non-positive geometry. Adjust parameters.")

    return {"L_nm": L, "W_nm": W, "Vth_v": Vth}
