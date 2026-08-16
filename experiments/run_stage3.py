"""
Stage 3 orchestration script — reference run.

Loads parameters from stage3_config.json, runs N=100k Monte Carlo on the
branching timing graph, computes statistics, and persists:
  - histogram PNG
  - summary JSON
  - raw .npy arrays for Stage 4 reuse
  - compressed NPZ archive
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from foundations.config_loader import load_config
from ssta.monte_carlo import run_branching_monte_carlo
from ssta.statistics import summarize_delay, critical_path_split
from timing.graph import build_branching_graph_from_config


def main() -> None:
    config = load_config("foundations/stage3_config.json")
    exp = config.experiment
    n_samples = exp.n_samples
    seed = exp.reference_seed

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
        "n_samples": n_samples,
        "seed": seed,
        "config_file": "foundations/stage3_config.json",
        "stats": stats,
        "critical_path_split": split,
        "sink": results["sink"],
        "sink_predecessors": results["sink_predecessors"],
        "gates": results["gate_names"],
        "note": "Normalized/toy units — not calibrated to real technology.",
    }

    os.makedirs(exp.output_dir, exist_ok=True)
    with open(exp.summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.hist(cpd, bins=100, density=True, alpha=0.7, edgecolor="black")
    plt.title("Stage 3: Critical-Path Delay Distribution (Branching DAG, N=100k)")
    plt.xlabel("Critical-path delay")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp.output_dir, "stage3_histogram.png"), dpi=150)
    plt.close()

    np.save(os.path.join(exp.output_dir, "stage3_cpd.npy"), cpd)
    for label, arr in results["path_labels"].items():
        np.save(os.path.join(exp.output_dir, f"stage3_{label}.npy"), arr)
    np.save(os.path.join(exp.output_dir, "stage3_L_nm.npy"), results["L_nm"])
    np.save(os.path.join(exp.output_dir, "stage3_W_nm.npy"), results["W_nm"])
    np.save(os.path.join(exp.output_dir, "stage3_Vth_v.npy"), results["Vth_v"])
    np.save(os.path.join(exp.output_dir, "stage3_arrival_times.npy"), results["arrival_times"])

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

    print("=== Stage 3 Reference Run (Branching DAG) ===")
    print(f"Config : foundations/stage3_config.json")
    print(f"Samples: {n_samples:,}")
    print(f"Seed   : {seed}")
    for k in ["mean", "std", "p95", "p99", "p99_87"]:
        print(f"{k.upper():<7}: {stats[k]:.6f}")
    for name, frac in split.items():
        print(f"{name}: {frac:.4f}")
    print(f"Saved summary -> {exp.summary_file}")
    print(f"Saved histogram -> {exp.output_dir}/stage3_histogram.png")
    print(f"Saved raw .npy + .npz -> {exp.output_dir}/")


if __name__ == "__main__":
    main()
