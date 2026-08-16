"""
Sanity checks for Stage 3.

Compares branching DAG against:
1. A simple chain through the same variation samples.
2. Gaussian mean+3σ approximation.
"""

from __future__ import annotations

import json

import numpy as np

from foundations.config_loader import load_config
from ssta.monte_carlo import run_branching_monte_carlo
from timing.graph import Gate, TimingGraph, build_branching_graph_from_config
from variation.sampler import sample_correlated_process
from timing.graph import alpha_power_delay


def build_chain_graph() -> TimingGraph:
    gates = {
        "G1": Gate("G1", 1.0, 0.0, 0.0),
        "G2": Gate("G2", 1.1, 0.0, 2.0),
        "G4": Gate("G4", 1.2, 2.0, 2.0),
        "G6": Gate("G6", 1.4, 4.0, 0.0),
    }
    successors = {
        "G1": ["G2"],
        "G2": ["G4"],
        "G4": ["G6"],
        "G6": [],
    }
    return TimingGraph(gates=gates, successors=successors)


def run_chain_with_same_variation(
    n_samples: int,
    seed: int,
    variation_params,
    gate_params,
    graph: TimingGraph,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = sample_correlated_process(n_samples, variation_params, rng)

    order = graph.topological_order()
    AT = np.zeros(n_samples)
    for name in order:
        gate = graph.gates[name]
        idx = order.index(name)
        delay = alpha_power_delay(
            samples["Vth_v"][:, idx],
            np.full(n_samples, gate.load_ff),
            gate_params,
            samples["L_nm"][:, idx],
            samples["W_nm"][:, idx],
            variation_params.l_nom_nm,
            variation_params.w_nom_nm,
        )
        preds = [p for p, succs in graph.successors.items() if name in succs]
        if not preds:
            AT = delay
        else:
            AT = AT + delay
    return AT


def main() -> None:
    config = load_config("foundations/stage3_config.json")
    n_samples = 10_000
    seed = 42

    graph = build_branching_graph_from_config(
        timing_params=config.timing_params,
        dag=config.dag.successors,
        variation_params=config.variation_params,
    )
    branch_results = run_branching_monte_carlo(
        n_samples=n_samples,
        seed=seed,
        variation_params=config.variation_params,
        gate_params=config.timing_params,
        graph=graph,
    )
    cpd = branch_results["critical_path_delay"]

    chain_graph = build_chain_graph()
    chain_delay = run_chain_with_same_variation(
        n_samples, seed, config.variation_params, config.timing_params, chain_graph
    )

    branch_mean = float(np.mean(cpd))
    branch_std = float(np.std(cpd, ddof=1))
    chain_mean = float(np.mean(chain_delay))
    chain_std = float(np.std(chain_delay, ddof=1))

    gaussian_approx_p99_87 = branch_mean + 3.0 * branch_std

    report = {
        "branching": {
            "mean": branch_mean,
            "std": branch_std,
            "p95": float(np.quantile(cpd, 0.95)),
            "p99": float(np.quantile(cpd, 0.99)),
            "p99_87": float(np.quantile(cpd, 0.9987)),
        },
        "chain_same_variation": {
            "mean": chain_mean,
            "std": chain_std,
            "p95": float(np.quantile(chain_delay, 0.95)),
            "p99": float(np.quantile(chain_delay, 0.99)),
            "p99_87": float(np.quantile(chain_delay, 0.9987)),
        },
        "gaussian_approx_p99_87": gaussian_approx_p99_87,
        "branching_vs_chain_delta_mean": branch_mean - chain_mean,
        "branching_vs_chain_delta_p99_87": float(np.quantile(cpd, 0.9987)) - float(np.quantile(chain_delay, 0.9987)),
        "path_split": {
            name: float(np.count_nonzero(arr) / n_samples)
            for name, arr in branch_results["path_labels"].items()
        },
    }

    with open("results/stage3_sanity_checks.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=== Sanity Checks ===")
    print(f"Branching mean : {branch_mean:.6f}")
    print(f"Chain mean     : {chain_mean:.6f}")
    print(f"Delta mean     : {report['branching_vs_chain_delta_mean']:.6f}")
    print(f"Branching P99.87: {report['branching']['p99_87']:.6f}")
    print(f"Chain P99.87    : {report['chain_same_variation']['p99_87']:.6f}")
    print(f"Delta P99.87    : {report['branching_vs_chain_delta_p99_87']:.6f}")
    print(f"Gaussian approx (mean+3*std): {gaussian_approx_p99_87:.6f}")
    print(f"Actual P99.87            : {report['branching']['p99_87']:.6f}")
    for name, frac in report["path_split"].items():
        print(f"{name}: {frac:.4f}")
    print("Saved results/stage3_sanity_checks.json")


if __name__ == "__main__":
    main()
