"""
Stage 6A — Training Data Generation

Generates a validated dataset of (DAG structure → MC-labeled delay statistics)
pairs with a clean train/val/test split by graph.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from foundations.config_loader import load_config, VariationParams
from ssta.monte_carlo import run_branching_monte_carlo
from data_generation.graph_generator import generate_dataset, print_graph_summary, GeneratedGraph
from data_generation.analytical_ssta_arbitrary import compute_analytical_ssta_arbitrary


def graph_to_dict(graph: GeneratedGraph) -> dict:
    """Convert GeneratedGraph to a JSON-serializable dict."""
    return {
        "graph_id": graph.graph_id,
        "source": graph.source,
        "sink": graph.sink,
        "successors": graph.successors,
        "gates": {
            name: {"load_ff": gate.load_ff, "x": gate.x, "y": gate.y}
            for name, gate in graph.gates.items()
        },
        "coordinates": {name: list(coord) for name, coord in graph.coordinates.items()},
        "gate_loads": graph.gate_loads,
        "reconvergence_points": graph.reconvergence_points,
    }


def compute_physics_features(graph: GeneratedGraph, config, variation_params=None) -> dict:
    """Compute physics-derived features for a graph."""
    from timing.graph import TimingGraph, Gate
    from timing.delay import compute_delay_moments, nominal_delay, delay_partials
    from variation.analytical import compute_process_moments

    if variation_params is None:
        variation_params = config.variation_params

    timing_params = config.timing_params

    # Build TimingGraph
    gates = {
        name: Gate(name=name, load_ff=gate.load_ff, x=gate.x, y=gate.y)
        for name, gate in graph.gates.items()
    }
    timing_graph = TimingGraph(gates=gates, successors=graph.successors)

    # Analytical SSTA
    analytical = compute_analytical_ssta_arbitrary(
        timing_graph, timing_params, variation_params
    )

    # Per-gate linearized sensitivities
    process_moments = compute_process_moments(variation_params)
    graph_gate_loads = {name: gate.load_ff for name, gate in graph.gates.items()}
    delay_moments = compute_delay_moments(timing_params, variation_params, process_moments, gate_loads=graph_gate_loads)

    sensitivities = {}
    for name in timing_graph.topological_order():
        gate = timing_graph.gates[name]
        partials = delay_partials(
            load_ff=gate.load_ff,
            gate_params=timing_params,
            vth_nom=variation_params.vth_nom_v,
            l_nom=variation_params.l_nom_nm,
            w_nom=variation_params.w_nom_nm,
        )
        sensitivities[name] = partials

    return {
        "analytical_ssta": analytical,
        "sensitivities": sensitivities,
    }


def validate_label_quality(
    graphs: List[GeneratedGraph],
    config,
    n_samples: int = 10_000,
    n_seeds: int = 2,
) -> dict:
    """Validate label quality by checking seed-to-seed noise."""
    validation_graphs = graphs[:20]

    rng = np.random.default_rng(42)

    noise_results = []
    for graph in validation_graphs:
        labels = []
        for seed in range(n_seeds):
            graph_coords = {name: (gate.x, gate.y) for name, gate in graph.gates.items()}
            graph_variation_params = replace(config.variation_params, gate_coords=graph_coords)
            graph_gate_loads = {name: gate.load_ff for name, gate in graph.gates.items()}
            graph_timing_params = replace(config.timing_params, gate_loads=graph_gate_loads)

            results = run_branching_monte_carlo(
                n_samples=n_samples,
                seed=seed,
                variation_params=graph_variation_params,
                gate_params=graph_timing_params,
                graph=_to_timing_graph(graph),
            )
            cpd = results["critical_path_delay"]
            labels.append({
                "mean": float(np.mean(cpd)),
                "std": float(np.std(cpd, ddof=1)),
            })

        mean_noise = abs(labels[0]["mean"] - labels[1]["mean"]) / max(labels[0]["mean"], 1e-9)
        std_noise = abs(labels[0]["std"] - labels[1]["std"]) / max(labels[0]["std"], 1e-9)
        noise_results.append({
            "graph_id": graph.graph_id,
            "mean_noise": mean_noise,
            "std_noise": std_noise,
        })

    mean_noises = [r["mean_noise"] for r in noise_results]
    std_noises = [r["std_noise"] for r in noise_results]

    return {
        "mean_noise_mean": float(np.mean(mean_noises)),
        "mean_noise_std": float(np.std(mean_noises, ddof=1)),
        "std_noise_mean": float(np.mean(std_noises)),
        "std_noise_std": float(np.std(std_noises, ddof=1)),
        "per_graph": noise_results,
    }


def _to_timing_graph(graph: GeneratedGraph) -> TimingGraph:
    """Convert GeneratedGraph to TimingGraph for MC analysis."""
    from timing.graph import TimingGraph, Gate

    gates = {
        name: Gate(name=name, load_ff=gate.load_ff, x=gate.x, y=gate.y)
        for name, gate in graph.gates.items()
    }
    return TimingGraph(gates=gates, successors=graph.successors)


def main() -> None:
    t0 = time.time()
    config = load_config("foundations/stage3_config.json")

    # Step 1: Generate graphs
    print("=== Generating random DAGs ===")
    target_n_graphs = 2000
    graphs = generate_dataset(
        n_graphs=target_n_graphs,
        n_gates_range=(6, 14),
        min_reconvergence=1,
        seed=42,
    )
    print(f"Generated {len(graphs)} graphs")

    # Step 2: Validate generator (20 graphs)
    print("\n=== Validating generator ===")
    validation_graphs = graphs[:20]
    val_sizes = [len(g.gates) for g in validation_graphs]
    val_reconv_counts = [len(g.reconvergence_points) for g in validation_graphs]
    print(f"Gate count range: {min(val_sizes)}-{max(val_sizes)}")
    print(f"Reconvergence points range: {min(val_reconv_counts)}-{max(val_reconv_counts)}")
    print(f"Mean reconvergence points: {np.mean(val_reconv_counts):.2f}")

    # Step 3-4: Generate MC labels and physics features
    print("\n=== Generating MC labels and physics features ===")
    dataset = {}
    generation_times = []
    skip_reasons = {"mc_error": 0}

    for i, graph in enumerate(graphs):
        graph_t0 = time.time()
        timing_graph = _to_timing_graph(graph)

        # Create variation params with this graph's coordinates
        graph_coords = {name: (gate.x, gate.y) for name, gate in graph.gates.items()}
        graph_variation_params = replace(config.variation_params, gate_coords=graph_coords)

        try:
            # MC labels (N=10,000)
            mc_results = run_branching_monte_carlo(
                n_samples=10_000,
                seed=42,
                variation_params=graph_variation_params,
                gate_params=config.timing_params,
                graph=timing_graph,
            )
            cpd = mc_results["critical_path_delay"]
            mc_labels = {
                "mean": float(np.mean(cpd)),
                "std": float(np.std(cpd, ddof=1)),
            }

            # Physics features
            physics = compute_physics_features(graph, config, variation_params=graph_variation_params)

            dataset[graph.graph_id] = {
                "graph": graph_to_dict(graph),
                "mc_labels": mc_labels,
                "physics_features": physics,
                "wall_time_s": time.time() - graph_t0,
            }

            generation_times.append(time.time() - graph_t0)
        except Exception as e:
            skip_reasons["mc_error"] += 1
            print(f"  MC failed for {graph.graph_id}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(graphs)} graphs (dataset: {len(dataset)})")

    n_processed = len(graphs) - skip_reasons["mc_error"]
    print(f"Skipped {skip_reasons['mc_error']} graphs (MC error)")
    print(f"Successfully processed: {len(dataset)} graphs")

    total_time = time.time() - t0
    print(f"\nTotal generation time: {total_time:.1f}s")
    if generation_times:
        print(f"Mean time per graph: {np.mean(generation_times):.3f}s")

    # Step 6: Validate label quality
    print("\n=== Validating label quality ===")
    noise = validate_label_quality(graphs, config, n_samples=10_000, n_seeds=2)
    if "error" not in noise:
        print(f"Mean noise (mean): {noise['mean_noise_mean']:.4f} ± {noise['mean_noise_std']:.4f}")
        print(f"Mean noise (std):  {noise['std_noise_mean']:.4f} ± {noise['std_noise_std']:.4f}")
    else:
        print(f"  Validation skipped: {noise['error']}")

    # Step 7: Split dataset (stratified by reconvergence count)
    print("\n=== Splitting dataset ===")
    graph_ids = list(dataset.keys())
    rng = np.random.default_rng(42)

    # Group by reconvergence count
    by_recon: Dict[int, List[str]] = {}
    for gid in graph_ids:
        nrecon = len(dataset[gid]["graph"]["reconvergence_points"])
        by_recon.setdefault(nrecon, []).append(gid)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for nrecon, group in by_recon.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        train_ids.extend(group[:n_train])
        val_ids.extend(group[n_train:n_train + n_val])
        test_ids.extend(group[n_train + n_val:])

    # Final shuffle within each split
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    for split_name, split_ids in splits.items():
        nrecon_dist = {}
        for gid in split_ids:
            nrecon = len(dataset[gid]["graph"]["reconvergence_points"])
            nrecon_dist[nrecon] = nrecon_dist.get(nrecon, 0) + 1
        print(f"  {split_name}: {len(split_ids)} graphs")
        for k in sorted(nrecon_dist.keys()):
            print(f"    nrecon={k}: {nrecon_dist[k]}")

    # Step 8: Save to disk
    print("\n=== Saving dataset ===")
    output_dir = Path("data_generation/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    with open(output_dir / "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    # Save splits
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # Save manifest
    actual_sizes = [len(d["graph"]["gates"]) for d in dataset.values()]
    actual_reconv = [len(d["graph"]["reconvergence_points"]) for d in dataset.values()]

    manifest = {
        "target_n_graphs": target_n_graphs,
        "n_generated": len(graphs),
        "n_dataset": len(dataset),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "skip_reasons": skip_reasons,
        "total_generation_time_s": total_time,
        "mean_time_per_graph_s": float(np.mean(generation_times)) if generation_times else 0.0,
        "label_noise": noise,
        "config_file": "foundations/stage3_config.json",
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save summary stats (computed from actual dataset, not pre-filter set)
    means = [d["mc_labels"]["mean"] for d in dataset.values()]
    stds = [d["mc_labels"]["std"] for d in dataset.values()]

    summary = {
        "mean_delay": {"mean": float(np.mean(means)), "std": float(np.std(means, ddof=1))},
        "std_delay": {"mean": float(np.mean(stds)), "std": float(np.std(stds, ddof=1))},
        "n_graphs": len(dataset),
        "gates_per_graph": {
            "min": int(min(actual_sizes)),
            "max": int(max(actual_sizes)),
            "mean": float(np.mean(actual_sizes)),
        },
        "reconvergence_points_per_graph": {
            "min": int(min(actual_reconv)),
            "max": int(max(actual_reconv)),
            "mean": float(np.mean(actual_reconv)),
        },
    }
    with open(output_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved dataset to {output_dir}")
    print(f"  dataset.pkl: {os.path.getsize(output_dir / 'dataset.pkl') / 1024 / 1024:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
