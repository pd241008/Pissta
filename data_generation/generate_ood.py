"""
OOD topology generation for Stage 7 Step 6.

Generates out-of-distribution timing DAGs (larger, deeper, higher reconvergence
than the training distribution) and labels them with Monte Carlo ground truth.

Training distribution: n_gates 6-14, nrecon 1-6 (run_stage6a.py).
OOD definition:        n_gates 15-25, min_reconvergence=2.

Uses graph_generator.py::generate_random_dag() — the SAME post-fix code path
as run_stage6a.py (ADR-006, bug #5: sink-reconvergence forced by construction
via first_split_done on source→sink edge). This is NOT a reimplementation;
the OOD generator calls the validated generator with different range params.

MC labels: N=10,000, seed=42 (Stage 6A convention, NOT the N=100k Stage 3
locked reference — flagged explicitly per project convention).
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

from foundations.config_loader import load_config
from ssta.monte_carlo import run_branching_monte_carlo
from data_generation.graph_generator import generate_random_dag, GeneratedGraph

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data_generation" / "data"

# OOD parameters — strictly outside training distribution (6-14 gates).
OOD_N_GRAPHS = 100
OOD_N_GATES_MIN = 15
OOD_N_GATES_MAX = 25
OOD_MIN_RECONVERGENCE = 2
OOD_MC_N_SAMPLES = 10_000
OOD_MC_SEED = 42
OOD_GENERATION_SEED = 12345


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


def _to_timing_graph(graph: GeneratedGraph):
    """Convert GeneratedGraph to TimingGraph for MC analysis."""
    from timing.graph import TimingGraph, Gate
    gates = {
        name: Gate(name=name, load_ff=gate.load_ff, x=gate.x, y=gate.y)
        for name, gate in graph.gates.items()
    }
    return TimingGraph(gates=gates, successors=graph.successors)


def _validate_sink_reconvergence(graph: GeneratedGraph) -> bool:
    """Explicit check that sink has ≥2 predecessors (the ADR-006 fix).

    The generator's _build_dag() forces first_split_done on source→sink,
    which guarantees this by construction. This is a belt-and-suspenders
    validation — the same class of check that caught bug #5 in Stage 6A.
    """
    pred_count = {}
    for g, succs in graph.successors.items():
        for s in succs:
            pred_count[s] = pred_count.get(s, 0) + 1
    sink_preds = pred_count.get(graph.sink, 0)
    return sink_preds >= 2


def main() -> None:
    config = load_config(str(REPO_ROOT / "foundations" / "stage3_config.json"))

    print("=== OOD Topology Generation ===")
    print(f"n_gates: {OOD_N_GATES_MIN}-{OOD_N_GATES_MAX} (training: 6-14)")
    print(f"min_reconvergence: {OOD_MIN_RECONVERGENCE} (training: ≥1)")
    print(f"n_graphs: {OOD_N_GRAPHS}")
    print(f"MC: N={OOD_MC_N_SAMPLES}, seed={OOD_MC_SEED} (Stage 6A convention)")
    print()

    # Step 1: Generate OOD graphs.
    # Calls graph_generator.py::generate_random_dag() — the same post-fix code
    # path used by run_stage6a.py. The sink-reconvergence fix (ADR-006, bug #5)
    # is in _build_dag(): first_split_done forces source→sink split on the
    # first operation, guaranteeing sink has ≥2 predecessors by construction.
    print("=== Generating OOD DAGs ===")
    rng = np.random.default_rng(OOD_GENERATION_SEED)
    graphs = []
    for i in range(OOD_N_GRAPHS):
        n_gates = int(rng.integers(OOD_N_GATES_MIN, OOD_N_GATES_MAX + 1))
        graph = generate_random_dag(
            n_gates=n_gates,
            graph_id=f"ood_{i:06d}",
            rng=rng,
            min_reconvergence=OOD_MIN_RECONVERGENCE,
        )
        graphs.append(graph)
    print(f"Generated {len(graphs)} OOD graphs")

    # Step 2: Validate sink-reconvergence (explicit check).
    all_valid = True
    for g in graphs:
        if not _validate_sink_reconvergence(g):
            print(f"  WARNING: {g.graph_id} sink has <2 predecessors")
            all_valid = False
    if all_valid:
        print(f"Sink-reconvergence validation: ALL {len(graphs)} graphs have ≥2 sink predecessors")
    else:
        print("Sink-reconvergence validation: FAILED — see warnings above")
        sys.exit(1)

    # Step 3: Report graph statistics.
    sizes = [len(g.gates) for g in graphs]
    reconv_counts = [len(g.reconvergence_points) for g in graphs]
    print(f"Gate count range: {min(sizes)}-{max(sizes)} (mean {np.mean(sizes):.1f})")
    print(f"Reconvergence range: {min(reconv_counts)}-{max(reconv_counts)} (mean {np.mean(reconv_counts):.1f})")
    nrecon_dist = {}
    for rc in reconv_counts:
        nrecon_dist[rc] = nrecon_dist.get(rc, 0) + 1
    for k in sorted(nrecon_dist):
        print(f"  nrecon={k}: {nrecon_dist[k]}")
    print()

    # Step 4: MC label each graph (N=10k, seed 42 — Stage 6A convention).
    print("=== MC labeling ===")
    dataset = {}
    generation_times = []
    skip_reasons = {"mc_error": 0}

    t0 = time.time()
    for i, graph in enumerate(graphs):
        graph_t0 = time.time()
        timing_graph = _to_timing_graph(graph)

        graph_coords = {name: (gate.x, gate.y) for name, gate in graph.gates.items()}
        graph_variation_params = replace(config.variation_params, gate_coords=graph_coords)
        graph_gate_loads = {name: gate.load_ff for name, gate in graph.gates.items()}
        graph_timing_params = replace(config.timing_params, gate_loads=graph_gate_loads)

        try:
            mc_results = run_branching_monte_carlo(
                n_samples=OOD_MC_N_SAMPLES,
                seed=OOD_MC_SEED,
                variation_params=graph_variation_params,
                gate_params=graph_timing_params,
                graph=timing_graph,
            )
            cpd = mc_results["critical_path_delay"]
            mc_labels = {
                "mean": float(np.mean(cpd)),
                "std": float(np.std(cpd, ddof=1)),
            }

            dataset[graph.graph_id] = {
                "graph": graph_to_dict(graph),
                "mc_labels": mc_labels,
                "wall_time_s": time.time() - graph_t0,
            }
            generation_times.append(time.time() - graph_t0)
        except Exception as e:
            skip_reasons["mc_error"] += 1
            print(f"  MC failed for {graph.graph_id}: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(graphs)} graphs")

    total_time = time.time() - t0
    print(f"Successfully labeled: {len(dataset)}/{len(graphs)} graphs")
    print(f"Total MC time: {total_time:.1f}s ({np.mean(generation_times)*1000:.0f}ms/graph)")
    print()

    # Step 5: Compute normalization reference stats from training data.
    # These are NOT used for normalization (that's locked to train stats);
    # they're reported so the evaluator can check whether OOD features are
    # within the training normalization range.
    print("=== Normalization reference (from OOD data) ===")
    ood_means = [d["mc_labels"]["mean"] for d in dataset.values()]
    ood_stds = [d["mc_labels"]["std"] for d in dataset.values()]
    ood_gates = [len(d["graph"]["gates"]) for d in dataset.values()]

    # Load training stats for comparison.
    with open(DATA_DIR / "summary_stats.json") as f:
        train_stats = json.load(f)

    print(f"  OOD mean delay:  {np.mean(ood_means):.4f} ± {np.std(ood_means):.4f}")
    print(f"  Train mean delay: {train_stats['mean_delay']['mean']:.4f} ± {train_stats['mean_delay']['std']:.4f}")
    print(f"  OOD std delay:   {np.mean(ood_stds):.4f} ± {np.std(ood_stds):.4f}")
    print(f"  Train std delay:  {train_stats['std_delay']['mean']:.4f} ± {train_stats['std_delay']['std']:.4f}")
    print(f"  OOD gate count:  {np.mean(ood_gates):.1f} ± {np.std(ood_gates):.1f}")
    print(f"  Train gate count: {train_stats['gates_per_graph']['mean']:.1f}")
    print()

    # Step 6: Save OOD dataset.
    output_path = DATA_DIR / "ood_dataset.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    manifest = {
        "description": "OOD topology dataset for Stage 7 Step 6",
        "n_graphs": len(dataset),
        "n_generated": len(graphs),
        "n_gates_range": [OOD_N_GATES_MIN, OOD_N_GATES_MAX],
        "min_reconvergence": OOD_MIN_RECONVERGENCE,
        "mc_n_samples": OOD_MC_N_SAMPLES,
        "mc_seed": OOD_MC_SEED,
        "generation_seed": OOD_GENERATION_SEED,
        "total_generation_time_s": total_time,
        "skip_reasons": skip_reasons,
        "nrecon_distribution": nrecon_dist,
        "normalization_note": "MC labels use N=10k seed=42 (Stage 6A convention, NOT N=100k Stage 3 reference)",
    }
    with open(DATA_DIR / "ood_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved OOD dataset: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.0f} KB")
    print(f"  Manifest: {DATA_DIR / 'ood_manifest.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
