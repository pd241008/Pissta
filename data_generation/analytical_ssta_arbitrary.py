"""
Generalized analytical SSTA for arbitrary timing DAGs.

Extends Stage 4's analytical pipeline to work with any valid timing graph,
not just the hardcoded branching structure. Uses iterative Clark MAX for
nodes with multiple predecessors.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from foundations.config_loader import TimingParams, VariationParams
from timing.graph import TimingGraph, Gate
from timing.clark_max import clark_max
from variation.analytical import compute_process_moments
from timing.delay import compute_delay_moments


def _propagate_node_at(
    node: str,
    AT_mean: Dict[str, float],
    AT_var: Dict[str, float],
    delay_mean: Dict[str, float],
    delay_var: Dict[str, float],
    cov_delay: np.ndarray,
    idx: Dict[str, int],
    predecessors: Dict[str, List[str]],
) -> Tuple[float, float]:
    """Compute AT distribution for a single node given its predecessors."""
    preds = predecessors.get(node, [])
    if not preds:
        return delay_mean[node], delay_var[node]

    # Start with first predecessor
    mu = AT_mean[preds[0]] + delay_mean[node]
    var = AT_var[preds[0]] + delay_var[node] + 2.0 * cov_delay[idx[preds[0]], idx[node]]

    # Combine remaining predecessors using Clark MAX iteratively
    for pred in preds[1:]:
        mu_pred = AT_mean[pred]
        var_pred = AT_var[pred]
        cov_pred_node = cov_delay[idx[pred], idx[node]]

        # AT_pred + delay_node
        mu_pred_total = mu_pred + delay_mean[node]
        var_pred_total = var_pred + delay_var[node] + 2.0 * cov_pred_node

        # Covariance between current combined AT and this predecessor's AT
        # Approximate using shared gate covariances
        cov_combined_pred = 0.0
        for other_pred in preds[: preds.index(pred) + 1]:
            cov_combined_pred += cov_delay[idx[other_pred], idx[node]]

        rho = cov_combined_pred / (math.sqrt(max(var, 1e-18)) * math.sqrt(max(var_pred_total, 1e-18)))
        rho = max(-1.0, min(1.0, rho))

        mu_max, var_max = clark_max(mu, var, mu_pred_total, var_pred_total, rho)
        mu, var = mu_max, var_max

    return mu, var


def compute_analytical_ssta_arbitrary(
    graph: TimingGraph,
    timing_params: TimingParams,
    variation_params: VariationParams,
) -> Dict:
    """Compute analytical SSTA for an arbitrary timing graph."""
    order = graph.topological_order()
    n_gates = len(graph.gates)
    idx = {name: i for i, name in enumerate(order)}

    # Compute per-gate delay moments
    process_moments = compute_process_moments(variation_params)
    graph_gate_loads = {name: graph.gates[name].load_ff for name in graph.gates}
    delay_moments = compute_delay_moments(timing_params, variation_params, process_moments, gate_loads=graph_gate_loads)

    delay_mean = {name: float(delay_moments["mean_d"][idx[name]]) for name in order}
    delay_var = {name: float(delay_moments["var_d"][idx[name]]) for name in order}

    # Build predecessor map
    predecessors: Dict[str, List[str]] = {name: [] for name in order}
    for g, succs in graph.successors.items():
        for s in succs:
            predecessors[s].append(g)

    # Propagate AT moments
    AT_mean: Dict[str, float] = {}
    AT_var: Dict[str, float] = {}

    for name in order:
        preds = predecessors[name]
        if not preds:
            AT_mean[name] = delay_mean[name]
            AT_var[name] = delay_var[name]
        else:
            # Combine all predecessors
            mu_combined = AT_mean[preds[0]] + delay_mean[name]
            var_combined = AT_var[preds[0]] + delay_var[name] + 2.0 * delay_moments["cov_d"][idx[preds[0]], idx[name]]

            for pred in preds[1:]:
                mu_pred = AT_mean[pred] + delay_mean[name]
                var_pred = AT_var[pred] + delay_var[name] + 2.0 * delay_moments["cov_d"][idx[pred], idx[name]]

                # Approximate covariance between combined and this predecessor
                cov_combined_pred = 0.0
                for p in preds[: preds.index(pred) + 1]:
                    cov_combined_pred += delay_moments["cov_d"][idx[p], idx[name]]

                rho = cov_combined_pred / (math.sqrt(max(var_combined, 1e-18)) * math.sqrt(max(var_pred, 1e-18)))
                rho = max(-1.0, min(1.0, rho))

                mu_max, var_max = clark_max(mu_combined, var_combined, mu_pred, var_pred, rho)
                mu_combined, var_combined = mu_max, var_max

            AT_mean[name] = mu_combined
            AT_var[name] = var_combined

    sink = graph.sinks()[0]
    return {
        "AT_mean": AT_mean,
        "AT_var": AT_var,
        "delay_mean": delay_mean,
        "delay_var": delay_var,
        "sink_mean": AT_mean[sink],
        "sink_var": AT_var[sink],
        "sink_std": math.sqrt(max(AT_var[sink], 0.0)),
    }
