"""
Reproducibility check for Stage 3 (N=100k).

Uses stage3_config.json for parameters. Runs the branching DAG and a simple
chain through the SAME variation draws to quantify the MAX-induced gap.
Saves raw .npy for the reference seed (42) so Stage 4 can reuse it.
"""

from __future__ import annotations

import json
import os
import time

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


def evaluate(n_samples: int, seed: int, config) -> dict:
    graph = build_branching_graph_from_config(
        timing_params=config.timing_params,
        dag=config.dag.successors,
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

    branch_p99_87 = float(np.quantile(cpd, 0.9987))
    chain_p99_87 = float(np.quantile(chain_delay, 0.9987))
    branch_mean = float(np.mean(cpd))
    branch_std = float(np.std(cpd, ddof=1))
    gaussian_approx = branch_mean + 3.0 * branch_std

    return {
        "branch_mean": branch_mean,
        "branch_std": branch_std,
        "branch_p95": float(np.quantile(cpd, 0.95)),
        "branch_p99": float(np.quantile(cpd, 0.99)),
        "branch_p99_87": branch_p99_87,
        "chain_mean": float(np.mean(chain_delay)),
        "chain_std": float(np.std(chain_delay, ddof=1)),
        "chain_p95": float(np.quantile(chain_delay, 0.95)),
        "chain_p99": float(np.quantile(chain_delay, 0.99)),
        "chain_p99_87": chain_p99_87,
        "delta_p99_87": branch_p99_87 - chain_p99_87,
        "gaussian_approx_p99_87": gaussian_approx,
        "path_split": {
            name: float(np.count_nonzero(arr) / n_samples)
            for name, arr in branch_results["path_labels"].items()
        },
    }


def main() -> None:
    config = load_config("foundations/stage3_config.json")
    exp = config.experiment
    n_samples = exp.n_samples
    seeds = exp.seeds
    output_dir = exp.output_dir
    os.makedirs(output_dir, exist_ok=True)

    report = {"n_samples": n_samples, "config_file": "foundations/stage3_config.json", "runs": []}

    for seed in seeds:
        print(f"\n=== Seed {seed} (N={n_samples:,}) ===")
        t0 = time.time()
        result = evaluate(n_samples, seed, config)
        elapsed = time.time() - t0
        report["runs"].append({"seed": seed, **result})
        print(f"  Branching mean    : {result['branch_mean']:.6f}")
        print(f"  Chain mean        : {result['chain_mean']:.6f}")
        print(f"  Delta mean        : {result['branch_mean'] - result['chain_mean']:.6f}")
        print(f"  Branching P99.87  : {result['branch_p99_87']:.6f}")
        print(f"  Chain P99.87      : {result['chain_p99_87']:.6f}")
        print(f"  Delta P99.87      : {result['delta_p99_87']:.6f}")
        print(f"  Gaussian approx   : {result['gaussian_approx_p99_87']:.6f}")
        for name, frac in result["path_split"].items():
            print(f"  {name}: {frac:.4f}")
        print(f"  Elapsed           : {elapsed:.1f}s")

    deltas = [r["delta_p99_87"] for r in report["runs"]]
    report["delta_p99_87_mean"] = float(np.mean(deltas))
    report["delta_p99_87_std"] = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

    print("\n=== Summary across seeds ===")
    print(f"Delta P99.87 values : {[round(d, 6) for d in deltas]}")
    print(f"Mean delta          : {report['delta_p99_87_mean']:.6f}")
    print(f"Std of deltas       : {report['delta_p99_87_std']:.6f}")

    with open(exp.reproducibility_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {exp.reproducibility_file}")


if __name__ == "__main__":
    main()
