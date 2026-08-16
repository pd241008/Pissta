"""
Asymmetric sanity check for Stage 3.

Uses stage3_config_asymmetric.json where Path 2 has an extra gate (G7).
Expects the critical-path fraction to move strongly away from 50/50.
If it does not, the AT/argmax logic has a bug.
"""

from __future__ import annotations

import json
import os

import numpy as np

from foundations.config_loader import load_config
from ssta.monte_carlo import run_branching_monte_carlo
from ssta.statistics import summarize_delay, critical_path_split
from timing.graph import build_branching_graph_from_config


def main() -> None:
    config = load_config("foundations/stage3_config_asymmetric.json")
    exp = config.experiment
    n_samples = exp.n_samples
    seed = exp.reference_seed
    output_dir = exp.output_dir
    os.makedirs(output_dir, exist_ok=True)

    graph = build_branching_graph_from_config(
        timing_params=config.timing_params,
        dag=config.dag.successors,
        variation_params=config.variation_params,
    )
    results = run_branching_monte_carlo(
        n_samples=n_samples,
        seed=seed,
        variation_params=config.variation_params,
        gate_params=config.timing_params,
        graph=graph,
    )

    cpd = results["critical_path_delay"]
    stats = summarize_delay(cpd)
    split = critical_path_split(path_labels=results["path_labels"])

    summary = {
        "config_file": "stage3_config_asymmetric.json",
        "n_samples": n_samples,
        "seed": seed,
        "stats": stats,
        "critical_path_split": split,
        "sink": results["sink"],
        "sink_predecessors": results["sink_predecessors"],
        "gates": results["gate_names"],
        "dag": config.dag.successors,
        "interpretation": (
            "Path 2 has an extra gate (G7) and should dominate. "
            "A fraction near 50/50 indicates a bug in AT propagation or argmax."
        ),
    }

    with open(exp.asymmetric_check_file, "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        exp.raw_data_file,
        L_nm=results["L_nm"],
        W_nm=results["W_nm"],
        Vth_v=results["Vth_v"],
        delays=results["delays"],
        arrival_times=results["arrival_times"],
        critical_path_delay=cpd,
        **{k: v for k, v in results["path_labels"].items()},
    )

    print("=== Asymmetric Sanity Check ===")
    print(f"DAG     : {config.dag.successors}")
    print(f"Samples : {n_samples:,}")
    for k in ["mean", "std", "p95", "p99", "p99_87"]:
        print(f"{k.upper():<7}: {stats[k]:.6f}")
    for name, frac in split.items():
        print(f"{name}: {frac:.4f}")
    path2_key = f"path_{results['sink_predecessors'][-1]}_critical"
    if path2_key in split and split[path2_key] > 0.80:
        print("PASS: Longer path dominates as expected.")
    else:
        print("FAIL: Longer path does not dominate. Check AT/argmax logic.")
    print(f"Saved {exp.asymmetric_check_file}")
    print(f"Saved raw {exp.raw_data_file}")


if __name__ == "__main__":
    main()
