#!/usr/bin/env python3
"""
generate_pissta.py — regenerate a Pissta dataset release from a manifest contract.

Versions
--------
  pissta-2k    2,000 graphs — the PAPER-EXACT Stage 6A dataset (frozen artifact;
               this mode verifies it, it does not rebuild it).
  pissta-5k    5,000 graphs — graphs 000000–001999 are verbatim copies of
               pissta-2k; graphs 002000+ are extension graphs.
  pissta-10k  10,000 graphs — same construction, extended to 10k.
  pissta-ood100  100 graphs — frozen OOD-topology companion (Stage 7 Step 6).

Determinism contract (all claims verified 2026-09-17, see manifest)
-------------------------------------------------------------------
  * Graph topology, coordinates, gate-count/nrecon distributions and the
    stratified train/val/test split: bit-exact from generation seed 42 with
    the current repo code (data_generation/graph_generator.py @ d697021).
  * MC labels (the supervised targets): bit-exact. The sampler maps variation
    samples to gates by NAME (ssta/monte_carlo.py @ 2873f73), N=10,000, seed 42
    per graph — identical convention for every dataset version.
  * Gate-load VALUES per graph: bit-exact multiset (hash-order independent).
  * Gate-load PAIRING (which gate receives which value): the original 2k build
    consumed draws in Python set-iteration order, i.e. it depended on the build
    process's PYTHONHASHSEED. That pairing is therefore part of the frozen 2k
    artifact and is NOT re-derived. Extension graphs (002000+) use a canonical,
    process-independent pairing: load values sorted ascending, assigned to gate
    names in lexicographic order.

Nesting guarantee: pissta-2k ⊂ pissta-5k ⊂ pissta-10k as graph sets (IDs
graph_000000–graph_001999 are byte-identical entries, MC labels included).
Splits are regenerated per version by the frozen stratified procedure (seed 42,
70/15/15 by reconvergence count), so split membership differs across versions
by design; graph sets do not.

Usage
-----
  # Verify + publish the frozen paper dataset (full MC label re-verification):
  python zenodo/scripts/generate_pissta.py --version pissta-2k

  # Build companion versions (requires the frozen 2k pickle as prefix source):
  python zenodo/scripts/generate_pissta.py --version pissta-5k
  python zenodo/scripts/generate_pissta.py --version pissta-10k

  # Publish the frozen OOD companion:
  python zenodo/scripts/generate_pissta.py --version pissta-ood100

  # Pipeline smoke test (50 extension graphs, non-citable output):
  python zenodo/scripts/generate_pissta.py --version pissta-5k --smoke

Every mode writes <out-dir>/<version>/manifest.json recording process constants,
commit hashes, SHA256 of each output file, and the emulation-based disclaimers.
Smoke mode writes under <out-dir>-smoke/ instead and marks the output non-citable.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from foundations.config_loader import load_config
from ssta.monte_carlo import run_branching_monte_carlo
from data_generation.graph_generator import (
    GeneratedGraph,
    generate_dataset,
    generate_random_dag,
)
from timing.graph import TimingGraph, Gate

# ---------------------------------------------------------------------------
# Frozen process constants (identical to run_stage6a.py / generate_ood.py)
# ---------------------------------------------------------------------------
GENERATION_SEED = 42
SPLITS_SEED = 42
MC_N_SAMPLES = 10_000
MC_SEED = 42
N_GATES_RANGE = (6, 14)
MIN_RECONVERGENCE = 1
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

# OOD companion constants (generate_ood.py)
OOD_N_GRAPHS = 100
OOD_N_GATES_MIN, OOD_N_GATES_MAX = 15, 25
OOD_MIN_RECONVERGENCE = 2
OOD_GENERATION_SEED = 12345

PREFIX_IDS = [f"graph_{i:06d}" for i in range(2000)]  # the pissta-2k extent
PICKLE_PROTOCOL = 4  # readable by Python >= 3.4; pinned for byte stability

VARIANTS = {"pissta-2k": 2000, "pissta-5k": 5000, "pissta-10k": 10_000}

GENERATOR_COMMIT = "d6970218c6bd93061f85af00a1bb81985ff052f9"
GENERATOR_COMMIT_DATE = "2026-09-15"
STRUCTURE_BUILD_NOTE_2K = (
    "pissta-2k structures were generated 2026-08-17 (commit 99a70cf-era working "
    "tree). Topology, coordinates and the stratified split reproduce bit-exactly "
    "under current code (verified 2026-09-17, all 2000 graphs). The gate-load "
    "value->gate pairing was drawn in Python set-iteration order during that "
    "build (PYTHONHASHSEED-dependent) and is frozen as part of the artifact; "
    "load VALUES are hash-order independent and bit-exact."
)

TOY_MODEL_DISCLAIMER = (
    "All numerical parameters (alpha=1.3, Vdd=1.0 V, L_nom=45 nm, W_nom=90 nm, "
    "Vth_nom=0.40 V, etc.) are deliberately simple demonstration parameters for "
    "software/methodology validation. They are NOT silicon-calibrated and must "
    "never be presented as real technology results."
)
EMULATION_DISCLAIMER = (
    "Labels are Monte-Carlo EMULATIONS produced by the repository's own "
    "analytical-variation sampler (inter-die + spatial + Pelgrom mismatch), not "
    "measurements of physical circuits or SPICE simulations of a real process. "
    "MC labels use N=10,000 per graph (Stage 6A convention, seed 42) — a "
    "tail-noisy grade, distinct from the locked N=100,000 Stage 3 reference. "
    "Label noise (seed-to-seed, N=10k): mean ~0.05%, std ~0.7-0.8%."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath(path: Path) -> str:
    """Repo-relative POSIX path for display/manifest use.

    Never emits an absolute path: paths outside the repo are reported
    relative to the CWD when possible, else as "<external>/<name>".
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        pass
    try:
        return p.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return f"<external>/{p.name}"


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def canonicalize_loads(graph: GeneratedGraph) -> GeneratedGraph:
    """Re-assign gate-load values to gates in a process-independent order.

    The RNG draws (the value multiset) are hash-order independent; only their
    assignment to gate names was set-iteration-order dependent in the original
    build. Canonical rule: ascending load value -> lexicographic gate name.
    Consumes no RNG state, so all downstream graphs are unaffected.
    """
    names = sorted(graph.gates)
    values = sorted(graph.gate_loads.values())
    loads = dict(zip(names, values))
    gates = {
        n: dataclasses.replace(graph.gates[n], load_ff=loads[n]) for n in names
    }
    return dataclasses.replace(graph, gates=gates, gate_loads=loads)


def to_timing_graph(graph: GeneratedGraph) -> TimingGraph:
    gates = {
        name: Gate(name=name, load_ff=g.load_ff, x=g.x, y=g.y)
        for name, g in graph.gates.items()
    }
    return TimingGraph(gates=gates, successors=graph.successors)


def graph_to_dict(graph: GeneratedGraph) -> dict:
    return {
        "graph_id": graph.graph_id,
        "source": graph.source,
        "sink": graph.sink,
        "successors": graph.successors,
        "gates": {
            name: {"load_ff": g.load_ff, "x": g.x, "y": g.y}
            for name, g in graph.gates.items()
        },
        "coordinates": {name: list(c) for name, c in graph.coordinates.items()},
        "gate_loads": graph.gate_loads,
        "reconvergence_points": graph.reconvergence_points,
    }


def label_graph(graph: GeneratedGraph, config) -> dict:
    """MC label one graph: N=10k, seed 42, name-aligned sampler."""
    coords = {name: (g.x, g.y) for name, g in graph.gates.items()}
    variation_params = dataclasses.replace(config.variation_params, gate_coords=coords)
    timing_params = dataclasses.replace(
        config.timing_params, gate_loads=dict(graph.gate_loads)
    )
    result = run_branching_monte_carlo(
        n_samples=MC_N_SAMPLES,
        seed=MC_SEED,
        variation_params=variation_params,
        gate_params=timing_params,
        graph=to_timing_graph(graph),
    )
    cpd = result["critical_path_delay"]
    return {
        "mean": float(np.mean(cpd)),
        "std": float(np.std(cpd, ddof=1)),
    }


def compute_physics_features(graph: GeneratedGraph, config, variation_params=None) -> dict:
    """Per-gate linearized sensitivities + analytical SSTA (same as run_stage6a)."""
    from timing.delay import compute_delay_moments, delay_partials
    from variation.analytical import compute_process_moments
    from data_generation.analytical_ssta_arbitrary import compute_analytical_ssta_arbitrary

    if variation_params is None:
        variation_params = config.variation_params

    timing_graph = to_timing_graph(graph)
    analytical = compute_analytical_ssta_arbitrary(
        timing_graph, config.timing_params, variation_params
    )

    process_moments = compute_process_moments(variation_params)
    graph_gate_loads = {name: g.load_ff for name, g in graph.gates.items()}
    delay_moments = compute_delay_moments(
        config.timing_params, variation_params, process_moments, gate_loads=graph_gate_loads
    )

    sensitivities = {}
    var_d_per_load_ff_sq = {}
    for name in timing_graph.topological_order():
        gate = timing_graph.gates[name]
        partials = delay_partials(
            load_ff=gate.load_ff,
            gate_params=config.timing_params,
            vth_nom=variation_params.vth_nom_v,
            l_nom=variation_params.l_nom_nm,
            w_nom=variation_params.w_nom_nm,
        )
        sensitivities[name] = partials
        idx = process_moments["idx"][name]
        var_L = process_moments["var_l"][idx]
        var_W = process_moments["var_w"][idx]
        var_Vth = process_moments["var_vth"][idx]
        load_ff = gate.load_ff
        if load_ff > 1e-12:
            var_d = (
                partials["vth"] ** 2 * var_Vth
                + partials["l"] ** 2 * var_L
                + partials["w"] ** 2 * var_W
            )
            var_d_per_load_ff_sq[name] = var_d / (load_ff ** 2)
        else:
            var_d_per_load_ff_sq[name] = 0.0

    return {
        "analytical_ssta": analytical,
        "sensitivities": sensitivities,
        "var_d_per_load_ff_sq": var_d_per_load_ff_sq,
    }


def make_splits(dataset: dict) -> dict:
    """Frozen stratified split procedure (run_stage6a.py, seed 42, 70/15/15)."""
    graph_ids = list(dataset.keys())  # insertion order = graph_000000.. (frozen)
    rng = np.random.default_rng(SPLITS_SEED)

    by_recon: dict[int, list[str]] = {}
    for gid in graph_ids:
        nrecon = len(dataset[gid]["graph"]["reconvergence_points"])
        by_recon.setdefault(nrecon, []).append(gid)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for group in by_recon.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(TRAIN_FRAC * n)
        n_val = int(VAL_FRAC * n)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def dist_by_recon(dataset: dict) -> dict:
    out: dict[str, int] = {}
    for entry in dataset.values():
        k = str(len(entry["graph"]["reconvergence_points"]))
        out[k] = out.get(k, 0) + 1
    return {k: out[k] for k in sorted(out)}


def label_noise_check(graphs: list[GeneratedGraph], config) -> dict:
    """Seed-to-seed label noise on the first 20 graphs (run_stage6a procedure)."""
    noise = []
    for graph in graphs[:20]:
        labels = []
        for seed in (0, 1):
            coords = {name: (g.x, g.y) for name, g in graph.gates.items()}
            vp = dataclasses.replace(config.variation_params, gate_coords=coords)
            tp = dataclasses.replace(
                config.timing_params, gate_loads=dict(graph.gate_loads)
            )
            result = run_branching_monte_carlo(
                n_samples=MC_N_SAMPLES, seed=seed,
                variation_params=vp, gate_params=tp, graph=to_timing_graph(graph),
            )
            cpd = result["critical_path_delay"]
            labels.append((float(np.mean(cpd)), float(np.std(cpd, ddof=1))))
        noise.append({
            "graph_id": graph.graph_id,
            "mean_noise": abs(labels[0][0] - labels[1][0]) / max(labels[0][0], 1e-9),
            "std_noise": abs(labels[0][1] - labels[1][1]) / max(labels[0][1], 1e-9),
        })
    means = [r["mean_noise"] for r in noise]
    stds = [r["std_noise"] for r in noise]
    return {
        "mean_noise_mean": float(np.mean(means)),
        "mean_noise_std": float(np.std(means, ddof=1)),
        "std_noise_mean": float(np.mean(stds)),
        "std_noise_std": float(np.std(stds, ddof=1)),
        "per_graph": noise,
    }


def verify_prefix_labels(seed_entries: dict, config, n_total: int) -> dict:
    """Recompute MC labels for the frozen prefix and compare bit-exactly."""
    mismatches = []
    t0 = time.time()
    for i, gid in enumerate(PREFIX_IDS):
        if i >= n_total:
            break
        entry = seed_entries[gid]
        graph = reconstruct_graph(entry["graph"], gid)
        recomputed = label_graph(graph, config)
        stored = entry["mc_labels"]
        if (recomputed["mean"] != stored["mean"]) or (recomputed["std"] != stored["std"]):
            mismatches.append({
                "graph_id": gid,
                "stored": stored,
                "recomputed": recomputed,
            })
        if (i + 1) % 250 == 0:
            print(f"  prefix label verify {i + 1}/{min(len(PREFIX_IDS), n_total)} "
                  f"({time.time() - t0:.0f}s)")
    return {
        "n_checked": min(len(PREFIX_IDS), n_total),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:10],
        "bit_exact": len(mismatches) == 0,
        "elapsed_s": time.time() - t0,
    }


def reconstruct_graph(g: dict, gid: str) -> GeneratedGraph:
    from data_generation.graph_generator import GeneratedGate

    gates = {
        name: GeneratedGate(name=name, load_ff=d["load_ff"], x=d["x"], y=d["y"])
        for name, d in g["gates"].items()
    }
    return GeneratedGraph(
        graph_id=gid,
        gates=gates,
        successors=g["successors"],
        coordinates={k: tuple(v) for k, v in g["coordinates"].items()},
        gate_loads=dict(g["gate_loads"]),
        reconvergence_points=list(g["reconvergence_points"]),
        source=g["source"],
        sink=g["sink"],
    )


def structures_match(graph: GeneratedGraph, stored: dict) -> bool:
    return (
        graph.successors == stored["successors"]
        and graph.coordinates == {k: tuple(v) for k, v in stored["coordinates"].items()}
        and {n: g.load_ff for n, g in graph.gates.items()} == stored["gate_loads"]
        and len(graph.gates) == len(stored["gates"])
    )


def write_manifest(out_dir: Path, manifest: dict) -> None:
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def hash_outputs(out_dir: Path, names: list[str]) -> dict:
    files = {}
    for name in names:
        p = out_dir / name
        if p.exists():
            files[name] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
    return files


def base_manifest(version: str, role: str, out_dir: Path, smoke: bool) -> dict:
    config = load_config(str(REPO / "foundations" / "stage3_config.json"))
    return {
        "dataset": version,
        "role": role,
        "smoke": smoke,
        "smoke_note": "smoke output: pipeline test only, non-citable" if smoke else None,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator_commit": GENERATOR_COMMIT,
        "generator_commit_date": GENERATOR_COMMIT_DATE,
        "config_file": "foundations/stage3_config.json",
        "config_snapshot": json.loads(
            (REPO / "foundations" / "stage3_config.json").read_text()
        ),
        "process_constants": {
            "generation_seed": GENERATION_SEED,
            "splits_seed": SPLITS_SEED,
            "mc_n_samples": MC_N_SAMPLES,
            "mc_seed": MC_SEED,
            "n_gates_range": list(N_GATES_RANGE),
            "min_reconvergence": MIN_RECONVERGENCE,
            "split_fractions": [TRAIN_FRAC, VAL_FRAC, round(1 - TRAIN_FRAC - VAL_FRAC, 2)],
            "sampler": "run_branching_monte_carlo (name-aligned variation columns, "
                       "ssta/monte_carlo.py @ 2873f73, 2026-08-21)",
            "delay_model": "alpha-power (k=1.0, alpha=1.3) with per-gate load and "
                           "2D spatial coordinates",
        },
        "disclaimers": {
            "toy_model": TOY_MODEL_DISCLAIMER,
            "emulation": EMULATION_DISCLAIMER,
        },
        "output_dir": relpath(out_dir),
        "_config_obj": config,
    }


# ---------------------------------------------------------------------------
# Mode: pissta-2k (verify frozen artifact + publish)
# ---------------------------------------------------------------------------
def build_pissta_2k(seed_pkl: Path, out_dir: Path, smoke: bool) -> None:
    print(f"=== pissta-2k: verifying frozen paper-exact artifact ===")
    config = load_config(str(REPO / "foundations" / "stage3_config.json"))

    with open(seed_pkl, "rb") as f:
        seed_entries = pickle.load(f)
    if set(seed_entries) != set(PREFIX_IDS):
        sys.exit(f"ABORT: seed pickle has {len(seed_entries)} graphs, expected {len(PREFIX_IDS)}")

    # 1) Structure regeneration check (all 2000 graphs).
    graphs = generate_dataset(
        n_graphs=2000, n_gates_range=N_GATES_RANGE,
        min_reconvergence=MIN_RECONVERGENCE, seed=GENERATION_SEED,
    )
    n_struct_bad = sum(
        1 for g in graphs
        if g.successors != seed_entries[g.graph_id]["graph"]["successors"]
        or g.coordinates != {k: tuple(v) for k, v in seed_entries[g.graph_id]["graph"]["coordinates"].items()}
    )
    print(f"structure regeneration: {2000 - n_struct_bad}/2000 match "
          f"({n_struct_bad} mismatches)")

    # 2) Full MC label re-verification (all 2000 graphs, bit-exact).
    label_check = verify_prefix_labels(seed_entries, config, n_total=2000)
    print(f"label verification: {label_check['n_checked']} graphs, "
          f"{label_check['n_mismatches']} mismatches, bit_exact={label_check['bit_exact']}")

    # 3) Splits regeneration check.
    with open(seed_pkl.parent / "splits.json") as f:
        stored_splits = json.load(f)
    regen_splits = make_splits(seed_entries)
    splits_ok = all(regen_splits[k] == stored_splits[k] for k in ("train", "val", "test"))
    print(f"splits regeneration: {'match' if splits_ok else 'MISMATCH'}")

    # 4) Publish: copy the frozen bytes verbatim (dataset.pkl/splits.json/summary).
    target = out_dir / "pissta-2k"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.pkl", "splits.json", "summary_stats.json"):
        shutil.copy2(seed_pkl.parent / name, target / name)

    manifest = base_manifest("pissta-2k", "paper-exact (Stage 6A, frozen 2026-08-21)", out_dir, smoke)
    del manifest["_config_obj"]
    label_noise = json.loads((seed_pkl.parent / "manifest.json").read_text())["label_noise"]
    with open(target / "dataset.pkl", "rb") as f:
        n_graphs = len(pickle.load(f))
    manifest.update({
        "n_graphs": n_graphs,
        "n_train": len(stored_splits["train"]),
        "n_val": len(stored_splits["val"]),
        "n_test": len(stored_splits["test"]),
        "n_generated": 2000,
        "skip_reasons": {"mc_error": 0},
        "label_noise": label_noise,
        "structure_build_note": STRUCTURE_BUILD_NOTE_2K,
        "physics_features_provenance": (
            "computed 2026-08-21 with pre-2026-08-22 variation/analytical.py "
            "process moments; MC labels unaffected (verified bit-exact)"
        ),
        "regeneration": {
            "mode": "verify-only (frozen artifact)",
            "structure_match": n_struct_bad == 0,
            "labels_bit_exact": label_check["bit_exact"],
            "labels_n_checked": label_check["n_checked"],
            "splits_match": splits_ok,
            "note": "full-dataset MC label re-verification; the load-value "
                    "pairing is frozen (see structure_build_note)",
        },
        "files": hash_outputs(target, ["dataset.pkl", "splits.json", "summary_stats.json"]),
    })
    write_manifest(target, manifest)
    print(f"published {relpath(target)} (dataset.pkl sha256 "
          f"{manifest['files']['dataset.pkl']['sha256'][:16]}...)")


# ---------------------------------------------------------------------------
# Mode: pissta-5k / pissta-10k (nested extension builds)
# ---------------------------------------------------------------------------
def build_extension(version: str, n_total: int, seed_pkl: Path, out_dir: Path, smoke: bool) -> None:
    print(f"=== {version}: nested extension build ({n_total} graphs) ===")
    config = load_config(str(REPO / "foundations" / "stage3_config.json"))

    with open(seed_pkl, "rb") as f:
        seed_entries = pickle.load(f)
    if set(seed_entries) != set(PREFIX_IDS):
        sys.exit(f"ABORT: seed pickle must be the frozen pissta-2k (got {len(seed_entries)} graphs)")

    dataset: dict = {}
    for gid in PREFIX_IDS:
        dataset[gid] = seed_entries[gid]  # verbatim byte-level copy of the entry
    print(f"prefix: {len(dataset)} graphs copied verbatim from pissta-2k")

    n_tail = n_total - len(PREFIX_IDS)
    if smoke:
        n_tail = 50
        n_total = len(PREFIX_IDS) + n_tail
        print(f"SMOKE MODE: tail truncated to {n_tail} graphs")

    # Re-verify prefix labels bit-exactly (sampled in smoke mode).
    label_check = verify_prefix_labels(seed_entries, config, n_total=60 if smoke else 2000)
    print(f"prefix label verification: {label_check['n_checked']} graphs, "
          f"{label_check['n_mismatches']} mismatches, bit_exact={label_check['bit_exact']}")
    if not label_check["bit_exact"]:
        sys.exit("ABORT: prefix labels do not reproduce; refusing to build on a broken base.")

    # Regenerate the full graph sequence (prefix property: graph i is identical
    # across any n_total >= i+1 because generation is sequential from seed 42).
    graphs = generate_dataset(
        n_graphs=n_total, n_gates_range=N_GATES_RANGE,
        min_reconvergence=MIN_RECONVERGENCE, seed=GENERATION_SEED,
    )

    # Prefix structure must match the frozen artifact.
    bad = [g.graph_id for g in graphs[:2000]
           if g.successors != seed_entries[g.graph_id]["graph"]["successors"]
           or g.coordinates != {k: tuple(v) for k, v in seed_entries[g.graph_id]["graph"]["coordinates"].items()}]
    if bad:
        sys.exit(f"ABORT: prefix structure mismatch for {bad[:5]}... — generator drift.")

    # MC labels + physics features for extension graphs (canonical load pairing).
    skip_reasons = {"mc_error": 0}
    t0 = time.time()
    for i in range(2000, n_total):
        graph = canonicalize_loads(graphs[i])
        graph_t0 = time.time()
        try:
            coords = {name: (g.x, g.y) for name, g in graph.gates.items()}
            graph_variation_params = dataclasses.replace(
                config.variation_params, gate_coords=coords
            )
            mc_labels = label_graph(graph, config)
            physics = compute_physics_features(graph, config, variation_params=graph_variation_params)
            dataset[graph.graph_id] = {
                "graph": graph_to_dict(graph),
                "mc_labels": mc_labels,
                "physics_features": physics,
                "wall_time_s": time.time() - graph_t0,
            }
        except Exception as e:  # noqa: BLE001 — mirror run_stage6a skip behaviour
            skip_reasons["mc_error"] += 1
            print(f"  MC failed for {graph.graph_id}: {e}")
            continue
        if (i + 1 - 2000) % 250 == 0:
            print(f"  extension graphs {i + 1 - 2000}/{n_total - 2000} "
                  f"({time.time() - t0:.0f}s)")
    print(f"extension labeling done: {n_total - 2000 - skip_reasons['mc_error']} ok, "
          f"{skip_reasons['mc_error']} skipped ({time.time() - t0:.0f}s)")

    # Splits + summary (frozen procedures, full dataset).
    splits = make_splits(dataset)
    means = [d["mc_labels"]["mean"] for d in dataset.values()]
    stds = [d["mc_labels"]["std"] for d in dataset.values()]
    sizes = [len(d["graph"]["gates"]) for d in dataset.values()]
    reconv = [len(d["graph"]["reconvergence_points"]) for d in dataset.values()]
    summary = {
        "mean_delay": {"mean": float(np.mean(means)), "std": float(np.std(means, ddof=1))},
        "std_delay": {"mean": float(np.mean(stds)), "std": float(np.std(stds, ddof=1))},
        "n_graphs": len(dataset),
        "gates_per_graph": {"min": int(min(sizes)), "max": int(max(sizes)),
                            "mean": float(np.mean(sizes))},
        "reconvergence_points_per_graph": {"min": int(min(reconv)), "max": int(max(reconv)),
                                           "mean": float(np.mean(reconv))},
    }

    tail_noise = label_noise_check(graphs[2000:2020], config)

    # Publish.
    target = out_dir / version
    target.mkdir(parents=True, exist_ok=True)
    with open(target / "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f, protocol=PICKLE_PROTOCOL)
    with open(target / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    with open(target / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = base_manifest(version, "companion (nested extension of pissta-2k)", out_dir, smoke)
    del manifest["_config_obj"]
    manifest.update({
        "n_graphs": len(dataset),
        "n_generated": n_total,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "skip_reasons": skip_reasons,
        "nested_from": {
            "base": "pissta-2k",
            "prefix_ids": "graph_000000..graph_001999 (byte-identical entries, "
                          "MC labels included)",
            "prefix_structure_build_note": STRUCTURE_BUILD_NOTE_2K,
        },
        "extension_provenance": {
            "graph_ids": f"graph_002000..graph_{n_total - 1:06d}",
            "load_pairing": "canonical: ascending load value -> lexicographic "
                            "gate name (process-independent)",
            "physics_features": (
                "computed at release time with current code "
                f"({GENERATOR_COMMIT[:12]}); note the 2k prefix carries its "
                "original 2026-08-21-era physics features (pre-2026-08-22 "
                "process moments). MC labels are unaffected and bit-consistent "
                "across the whole dataset. For physics-feature work, either "
                "restrict to MC-label-only training or recompute prefix "
                "features with current code."
            ),
            "label_noise": tail_noise,
        },
        "splits_note": "graph sets are strictly nested across versions; splits "
                       "are regenerated per version by the frozen stratified "
                       "procedure and therefore differ across versions",
        "regeneration": {
            "mode": "bit-exact from this repo + frozen pissta-2k pickle",
            "command": f"python zenodo/scripts/generate_pissta.py --version {version}",
            "prefix_labels_bit_exact": label_check["bit_exact"],
        },
        "files": hash_outputs(target, ["dataset.pkl", "splits.json", "summary_stats.json"]),
    })
    write_manifest(target, manifest)
    print(f"published {relpath(target)} (dataset.pkl sha256 "
          f"{manifest['files']['dataset.pkl']['sha256'][:16]}...)")


# ---------------------------------------------------------------------------
# Mode: pissta-ood100 (frozen companion publish)
# ---------------------------------------------------------------------------
def build_ood(out_dir: Path, smoke: bool) -> None:
    print("=== pissta-ood100: verifying + publishing frozen OOD companion ===")
    config = load_config(str(REPO / "foundations" / "stage3_config.json"))
    src = REPO / "data_generation" / "data"

    with open(src / "ood_dataset.pkl", "rb") as f:
        entries = pickle.load(f)
    ood_manifest_src = json.loads((src / "ood_manifest.json").read_text())

    # Structure regeneration from seed 12345 (loads: value multiset only —
    # the original pairing was set-iteration-order dependent).
    rng = np.random.default_rng(OOD_GENERATION_SEED)
    n_struct_bad = 0
    load_multiset_bad = 0
    for i in range(OOD_N_GRAPHS):
        n_gates = int(rng.integers(OOD_N_GATES_MIN, OOD_N_GATES_MAX + 1))
        graph = generate_random_dag(
            n_gates=n_gates, graph_id=f"ood_{i:06d}", rng=rng,
            min_reconvergence=OOD_MIN_RECONVERGENCE,
        )
        stored = entries[graph.graph_id]["graph"]
        if (graph.successors != stored["successors"]
                or graph.coordinates != {k: tuple(v) for k, v in stored["coordinates"].items()}):
            n_struct_bad += 1
        if sorted(graph.gate_loads.values()) != sorted(stored["gate_loads"].values()):
            load_multiset_bad += 1
    print(f"structure regeneration: {OOD_N_GRAPHS - n_struct_bad}/{OOD_N_GRAPHS} match; "
          f"load value multisets: {OOD_N_GRAPHS - load_multiset_bad}/{OOD_N_GRAPHS} match")

    # MC label spot-check (every 10th graph, bit-exact expectation).
    mismatches = 0
    for i in range(0, OOD_N_GRAPHS, 10):
        gid = f"ood_{i:06d}"
        graph = reconstruct_graph(entries[gid]["graph"], gid)
        recomputed = label_graph(graph, config)
        if recomputed != entries[gid]["mc_labels"]:
            mismatches += 1
    print(f"label spot-check (10 graphs): {mismatches} mismatches")

    target = out_dir / "pissta-ood100"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "ood_dataset.pkl", target / "ood_dataset.pkl")
    shutil.copy2(src / "ood_manifest.json", target / "ood_manifest.json")

    manifest = base_manifest("pissta-ood100",
                             "companion (OOD topology eval, Stage 7 Step 6)", out_dir, smoke)
    del manifest["_config_obj"]
    manifest["process_constants"].update({
        "generation_seed": OOD_GENERATION_SEED,
        "n_gates_range": [OOD_N_GATES_MIN, OOD_N_GATES_MAX],
        "min_reconvergence": OOD_MIN_RECONVERGENCE,
        "splits": None,
        "splits_note": "evaluation-only set; no train/val/test split",
    })
    manifest.update({
        "n_graphs": len(entries),
        "n_generated": ood_manifest_src["n_generated"],
        "nrecon_distribution": ood_manifest_src["nrecon_distribution"],
        "provenance": (
            "frozen artifact copied verbatim from the repo (2026-08-31-era "
            "build). Structures regenerate bit-exactly from seed 12345; MC "
            "labels are bit-exact (N=10k, seed 42). Gate-load pairing is "
            "frozen (PYTHONHASHSEED-era build); load VALUES are bit-exact."
        ),
        "schema_note": "entries carry graph + mc_labels only (no physics_features)",
        "regeneration": {
            "mode": "verify-only (frozen artifact)",
            "structure_match": n_struct_bad == 0,
            "load_multiset_match": load_multiset_bad == 0,
            "label_spotcheck_mismatches": mismatches,
        },
        "files": hash_outputs(target, ["ood_dataset.pkl", "ood_manifest.json"]),
    })
    write_manifest(target, manifest)
    print(f"published {relpath(target)}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--version", required=True,
                    choices=["pissta-2k", "pissta-5k", "pissta-10k", "pissta-ood100"])
    ap.add_argument("--seed-2k", type=Path, default=REPO / "data_generation" / "data" / "dataset.pkl",
                    help="path to the frozen pissta-2k dataset.pkl (prefix source)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "zenodo" / "data")
    ap.add_argument("--smoke", action="store_true",
                    help="truncate extension to 50 graphs, write to <version>-smoke, "
                         "mark non-citable (repo N2 hardening convention)")
    args = ap.parse_args()

    out_dir = args.out_dir
    if args.smoke:
        out_dir = out_dir.with_name(out_dir.name + "-smoke")

    if args.version == "pissta-2k":
        build_pissta_2k(args.seed_2k, out_dir, args.smoke)
    elif args.version == "pissta-ood100":
        build_ood(out_dir, args.smoke)
    else:
        build_extension(args.version, VARIANTS[args.version], args.seed_2k, out_dir, args.smoke)


if __name__ == "__main__":
    main()
