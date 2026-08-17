"""
Analytical linearized gate delay model.

Linearizes the alpha-power delay around nominal process parameters and
returns per-gate delay means, variances, and covariances.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from foundations.config_loader import TimingParams, VariationParams


def nominal_delay(
    load_ff: float,
    gate_params: TimingParams,
    vth_nom: float,
    l_nom: float,
    w_nom: float,
) -> float:
    vdd = gate_params.vdd_v
    base = load_ff * vdd / (gate_params.k * (vdd - vth_nom) ** gate_params.alpha)
    geom = (l_nom / l_nom) / np.sqrt(w_nom / w_nom)
    return float(base * geom)


def delay_partials(
    load_ff: float,
    gate_params: TimingParams,
    vth_nom: float,
    l_nom: float,
    w_nom: float,
) -> Dict[str, float]:
    vdd = gate_params.vdd_v
    d_nom = nominal_delay(load_ff, gate_params, vth_nom, l_nom, w_nom)
    return {
        "vth": d_nom * gate_params.alpha / (vdd - vth_nom),
        "l": d_nom / l_nom,
        "w": -d_nom / (2.0 * w_nom),
    }


def compute_delay_moments(
    timing_params: TimingParams,
    variation_params: VariationParams,
    process_moments: Dict[str, np.ndarray],
    gate_loads: Dict[str, float] | None = None,
) -> Dict[str, np.ndarray]:
    names = process_moments["names"]
    n = len(names)
    idx = process_moments["idx"]

    if gate_loads is None:
        gate_loads = timing_params.gate_loads

    mean_d = np.zeros(n)
    var_d = np.zeros(n)
    cov_d = np.zeros((n, n))

    partials = {}
    for name in names:
        if name not in gate_loads:
            raise KeyError(f"Gate '{name}' not found in gate_loads. Available: {list(gate_loads.keys())}")
        load = float(gate_loads[name])
        partials[name] = delay_partials(
            load_ff=load,
            gate_params=timing_params,
            vth_nom=variation_params.vth_nom_v,
            l_nom=variation_params.l_nom_nm,
            w_nom=variation_params.w_nom_nm,
        )
        mean_d[idx[name]] = nominal_delay(
            load_ff=load,
            gate_params=timing_params,
            vth_nom=variation_params.vth_nom_v,
            l_nom=variation_params.l_nom_nm,
            w_nom=variation_params.w_nom_nm,
        )

    for i, gi in enumerate(names):
        p_i = partials[gi]
        var_d[i] = (
            p_i["vth"] ** 2 * process_moments["var_vth"][i] +
            p_i["l"] ** 2 * process_moments["var_l"][i] +
            p_i["w"] ** 2 * process_moments["var_w"][i]
        )
        for j, gj in enumerate(names):
            p_j = partials[gj]
            cov_d[i, j] = (
                p_i["vth"] * p_j["vth"] * process_moments["cov_vth"][i, j] +
                p_i["l"] * p_j["l"] * process_moments["cov_l"][i, j] +
                p_i["w"] * p_j["w"] * process_moments["cov_w"][i, j]
            )

    return {
        "names": names,
        "idx": idx,
        "mean_d": mean_d,
        "var_d": var_d,
        "cov_d": cov_d,
    }
