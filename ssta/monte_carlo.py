"""
Vectorized Monte Carlo timing analysis for a branching timing graph.

Computes per-gate delays, propagates arrival times, tracks the critical path,
and returns the critical-path delay distribution plus per-sample labels.

Sample columns are ordered by variation_params.gate_coords key order and are
mapped to gates by NAME, not by topological position (the two orders differ
for arbitrary Stage 6A DAGs).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from foundations.config_loader import VariationParams
from timing.graph import TimingGraph, GateParams, build_branching_graph, build_branching_graph_from_config, alpha_power_delay
from variation.sampler import sample_correlated_process


def run_branching_monte_carlo(
    n_samples: int = 10_000,
    seed: int = 42,
    variation_params: VariationParams | None = None,
    gate_params: GateParams | None = None,
    graph: TimingGraph | None = None,
) -> Dict[str, np.ndarray]:
    if graph is None:
        graph = build_branching_graph()
    if variation_params is None:
        variation_params = VariationParams()
    if gate_params is None:
        gate_params = GateParams()

    rng = np.random.default_rng(seed)
    samples = sample_correlated_process(n_samples, variation_params, rng)

    order = graph.topological_order()
    n_gates = len(graph.gates)

    sample_names = list(variation_params.gate_coords.keys())
    missing = set(graph.gates) - set(sample_names)
    if missing:
        raise KeyError(f"Gates missing from variation_params.gate_coords: {sorted(missing)}")
    col_idx = {name: i for i, name in enumerate(sample_names)}

    l_idx = {name: idx for idx, name in enumerate(order)}
    L = samples["L_nm"]
    W = samples["W_nm"]
    Vth = samples["Vth_v"]

    delays = np.zeros((n_samples, n_gates))
    for name in order:
        gate = graph.gates[name]
        delays[:, l_idx[name]] = alpha_power_delay(
            Vth[:, col_idx[name]],
            np.full(n_samples, gate.load_ff),
            gate_params,
            L[:, col_idx[name]],
            W[:, col_idx[name]],
            variation_params.l_nom_nm,
            variation_params.w_nom_nm,
        )

    AT = np.zeros((n_samples, n_gates))
    for name in order:
        idx = l_idx[name]
        preds = [p for p, succs in graph.successors.items() if name in succs]
        if not preds:
            AT[:, idx] = delays[:, idx]
        else:
            pred_indices = [l_idx[p] for p in preds]
            AT[:, idx] = np.max(AT[:, pred_indices], axis=1) + delays[:, idx]

    sinks = graph.sinks()
    if len(sinks) != 1:
        raise ValueError(f"Expected exactly one sink, got {sinks}")
    sink = sinks[0]
    sink_preds = sorted(
        [p for p, succs in graph.successors.items() if sink in succs]
    )
    if len(sink_preds) < 2:
        raise ValueError(f"Sink {sink} has fewer than 2 predecessors: {sink_preds}")

    sink_pred_indices = [l_idx[p] for p in sink_preds]
    pred_arrival = AT[:, sink_pred_indices]

    winner_idx = np.argmax(pred_arrival, axis=1)
    path_labels = {}
    for j, pred_name in enumerate(sink_preds):
        path_labels[f"path_{pred_name}_critical"] = winner_idx == j

    return {
        "L_nm": L,
        "W_nm": W,
        "Vth_v": Vth,
        "delays": delays,
        "arrival_times": AT,
        "critical_path_delay": AT[:, l_idx[sink]],
        "sink": sink,
        "sink_predecessors": sink_preds,
        "path_labels": path_labels,
        "gate_names": order,
    }
