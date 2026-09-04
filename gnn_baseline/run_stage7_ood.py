"""
Stage 7 Step 6 — OOD conformal evaluation.

Tests whether conformal coverage holds on genuinely out-of-distribution
topologies (n_gates 15-25, nrecon ≥2) using the same calibration quantile
(q_hat) from the ID calibration set.

Design choices:
  - Same q_hat as ID (no recalibration — tests whether ID calibration
    generalizes to OOD)
  - Uses the frozen splits_stage7.json (unchanged)
  - Normalization from train split only (same as ID — OOD features may
    exceed training range; this is reported explicitly)
  - Smoke-tests GraphSAGE on single OOD graph before full batch
  - Reports OOD feature range vs training normalization range (Step 2 of
    user's feedback: distinguishes "exchangeability broke" from "features
    out of range for normalizer")
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "gnn_baseline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "gnn_baseline"))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from dataset import GraphDataset
from eval import evaluate_model
from model import VanillaDAGGNNSage, MaxBiasedDAGGNNSage
from conformal import (
    conformal_interval,
    coverage_by_group,
    empirical_coverage,
    interval_width_stats,
    split_conformal_quantile,
    studentized_score,
)
from run_stage7 import build_model, carve_cal_eval

DATA_DIR = REPO_ROOT / "data_generation" / "data"
RESULTS_DIR = REPO_ROOT / "gnn_baseline" / "results"
OOD_DATA = DATA_DIR / "ood_dataset.pkl"
OOD_MANIFEST = DATA_DIR / "ood_manifest.json"
STAGE7_SPLITS = RESULTS_DIR / "splits_stage7.json"
STAGE7_RESULTS = RESULTS_DIR / "stage7_calibration_results.json"
CHECKPOINT_DIR = REPO_ROOT / "gnn_baseline" / "checkpoints"

CONFIGS = ["vanilla", "maxbias_cm"]
SEEDS = [42, 123, 999]


def load_ood_dataset() -> dict:
    """Load the OOD dataset (same format as dataset.pkl)."""
    if not OOD_DATA.exists():
        print(f"ERROR: OOD dataset not found at {OOD_DATA}")
        print("Run: python data_generation/generate_ood.py")
        sys.exit(1)
    with open(OOD_DATA, "rb") as f:
        return pickle.load(f)


def load_id_calibration_results() -> dict:
    """Load Stage 7 ID results for q_hat and calibration predictions."""
    with open(STAGE7_RESULTS) as f:
        return json.load(f)


def check_ood_feature_range(ood_dataset: dict, train_dataset: GraphDataset) -> dict:
    """Report OOD feature values against training normalization range.

    This is the normalization diagnostic from user feedback #2: if OOD
    features are wildly outside the training range, a coverage drop won't
    cleanly indicate "exchangeability broke" — it could mean "the features
    were out of domain for the normalizer."
    """
    # Compute OOD feature ranges.
    ood_loads, ood_xs, ood_ys = [], [], []
    ood_means, ood_stds = [], []
    for gid, entry in ood_dataset.items():
        gates = entry["graph"]["gates"]
        for name, gate in gates.items():
            ood_loads.append(gate["load_ff"])
            ood_xs.append(gate["x"])
            ood_ys.append(gate["y"])
        ood_means.append(entry["mc_labels"]["mean"])
        ood_stds.append(entry["mc_labels"]["std"])

    ood_loads = np.array(ood_loads)
    ood_xs = np.array(ood_xs)
    ood_ys = np.array(ood_ys)

    # Get training normalization stats.
    feature_stats, target_stats = train_dataset._compute_normalization()

    def _check_range(name, ood_vals, train_mean, train_std, n_sigma=4.0):
        """Check how many OOD values fall outside n_sigma of training mean."""
        z = (ood_vals - train_mean) / max(train_std, 1e-9)
        n_outside = int(np.sum(np.abs(z) > n_sigma))
        pct_outside = n_outside / len(z) * 100
        return {
            "name": name,
            "ood_min": float(np.min(ood_vals)),
            "ood_max": float(np.max(ood_vals)),
            "ood_mean": float(np.mean(ood_vals)),
            "train_mean": train_mean,
            "train_std": train_std,
            "n_outside_4sigma": n_outside,
            "pct_outside_4sigma": pct_outside,
        }

    checks = [
        _check_range("load_ff", ood_loads, feature_stats["load_mean"], feature_stats["load_std"]),
        _check_range("x", ood_xs, feature_stats["x_mean"], feature_stats["x_std"]),
        _check_range("y", ood_ys, feature_stats["y_mean"], feature_stats["y_std"]),
        _check_range("mc_mean", np.array(ood_means), target_stats["mean_mean"], target_stats["mean_std"]),
        _check_range("mc_std", np.array(ood_stds), target_stats["std_mean"], target_stats["std_std"]),
    ]

    return {
        "checks": checks,
        "any_extrapolated": any(c["n_outside_4sigma"] > 0 for c in checks),
    }


def smoke_test_model(model: torch.nn.Module, ood_dataset: dict, feature_stats: Dict,
                     target_stats: Dict, device: torch.device) -> bool:
    """Smoke-test GraphSAGE forward pass on a single OOD graph.

    Required gate before full batch (user feedback #5): validates the model
    handles OOD-sized graphs (15-25 gates vs training 6-14) without shape
    errors or NaN outputs.
    """
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    gid = next(iter(ood_dataset))
    entry = ood_dataset[gid]
    gates = entry["graph"]["gates"]
    successors = entry["graph"]["successors"]

    node_names = sorted(gates.keys())
    n_nodes = len(node_names)
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    x = np.zeros((n_nodes, 3), dtype=np.float32)
    for name in node_names:
        gate = gates[name]
        idx = name_to_idx[name]
        x[idx, 0] = (gate["load_ff"] - feature_stats["load_mean"]) / feature_stats["load_std"]
        x[idx, 1] = (gate["x"] - feature_stats["x_mean"]) / feature_stats["x_std"]
        x[idx, 2] = (gate["y"] - feature_stats["y_mean"]) / feature_stats["y_std"]

    edge_list = []
    for src, succs in successors.items():
        for dst in succs:
            edge_list.append([name_to_idx[src], name_to_idx[dst]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(0)
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=y,
        graph_id=gid,
        n_nodes=n_nodes,
        nrecon=len(entry["graph"]["reconvergence_points"]),
        original_mean=0.0,
        original_std=0.0,
    )

    loader = DataLoader([data], batch_size=1)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)

    ok = pred.shape == (1, 2) and torch.isfinite(pred).all()
    if not ok:
        print(f"  SMOKE TEST FAILED: shape={pred.shape}, finite={torch.isfinite(pred).all().item()}")
    return ok


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Load data.
    ood_dataset = load_ood_dataset()
    id_results = load_id_calibration_results()

    print(f"OOD dataset: {len(ood_dataset)} graphs")
    print()

    # Load training dataset for normalization stats.
    train_dataset = GraphDataset(split="train", data_dir=DATA_DIR, physics_mode="vanilla")
    feature_stats, target_stats = train_dataset._compute_normalization()
    physics_stats = train_dataset._compute_physics_normalization()

    # === Normalization range check (user feedback #2) ===
    print("=== OOD Feature Range vs Training Normalization ===")
    norm_check = check_ood_feature_range(ood_dataset, train_dataset)
    for c in norm_check["checks"]:
        flag = f"WARNING: {c['n_outside_4sigma']} values outside 4σ" if c["n_outside_4sigma"] > 0 else "OK"
        print(f"  {c['name']:12s}: ood=[{c['ood_min']:.4f}, {c['ood_max']:.4f}] "
              f"train=μ{c['train_mean']:.4f} σ{c['train_std']:.4f} → {flag}")
    if norm_check["any_extrapolated"]:
        print("\n  NOTE: Some OOD features exceed training normalization range.")
        print("  Coverage findings below are ambiguous between exchangeability")
        print("  failure and feature extrapolation — frame accordingly.")
    else:
        print("\n  All OOD features within training normalization range.")
        print("  Coverage findings below cleanly test exchangeability.")
    print()

    # Build OOD DataLoader.
    ood_ids = list(ood_dataset.keys())
    ood_graph_dataset = GraphDataset(split="test", data_dir=DATA_DIR, physics_mode="vanilla")
    ood_graph_dataset.graph_ids = ood_ids
    # Patch the dataset to use OOD data instead of test data.
    ood_graph_dataset.dataset = ood_dataset

    from torch_geometric.loader import DataLoader
    ood_loader = DataLoader(
        ood_graph_dataset.get_data(feature_stats, target_stats, physics_stats),
        batch_size=16, shuffle=False,
    )

    # === Smoke test (user feedback #5) ===
    print("=== Smoke test ===")
    smoke_gid = next(iter(ood_dataset))
    smoke_n = len(ood_dataset[smoke_gid]["graph"]["gates"])
    for config in CONFIGS:
        model = build_model(config, device)
        ckpt = CHECKPOINT_DIR / f"best_model_{config}_seed42.pt"
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        ok = smoke_test_model(model, ood_dataset, feature_stats, target_stats, device)
        status = "PASS" if ok else "FAIL"
        print(f"  {config}: forward pass on {smoke_n}-gate OOD graph: {status}")
        if not ok:
            print(f"  ABORT: smoke test failed for {config}")
            sys.exit(1)
    print()

    # === OOD conformal evaluation ===
    print("=== OOD Conformal Evaluation ===")
    ood_results = {}

    for config in CONFIGS:
        config_results = []
        for seed in SEEDS:
            # Load trained model.
            model = build_model(config, device)
            ckpt = CHECKPOINT_DIR / f"best_model_{config}_seed{seed}.pt"
            if not ckpt.exists():
                print(f"  Skipping {config} seed={seed}: checkpoint not found")
                continue
            model.load_state_dict(torch.load(ckpt, weights_only=True))

            # Get q_hat from ID calibration (same seed).
            id_seed = [s for s in id_results["per_seed"][config] if s["seed"] == seed][0]
            q_hat = id_seed["q_hat"]

            # Evaluate on OOD.
            eval_metrics = evaluate_model(model, ood_loader, device, target_stats, dataset=ood_loader.dataset)

            # Form intervals using ID q_hat (no recalibration).
            ev_y = np.array([g["mc_mean"] for g in eval_metrics["per_graph"]])
            ev_mean = np.array([g["pred_mean"] for g in eval_metrics["per_graph"]])
            ev_std = np.array([g["pred_std"] for g in eval_metrics["per_graph"]])
            intervals = conformal_interval(ev_mean, ev_std, q_hat)
            lo, hi = intervals[:, 0], intervals[:, 1]
            pooled_cov = empirical_coverage(lo, hi, ev_y)

            groups = coverage_by_group(lo, hi, ev_y, np.array([g["nrecon"] for g in eval_metrics["per_graph"]]))
            width = interval_width_stats(lo, hi, np.array([g["nrecon"] for g in eval_metrics["per_graph"]]))

            config_results.append({
                "seed": seed,
                "config": config,
                "q_hat_from_id": q_hat,
                "pooled_coverage": pooled_cov,
                "coverage_by_nrecon": groups,
                "width_by_nrecon": width,
                "eval_mean_mae": float(eval_metrics["mean_mae"]),
                "n_ood_graphs": len(ev_y),
            })

        ood_results[config] = config_results

    # Aggregate.
    agg = {}
    for config in CONFIGS:
        if not ood_results[config]:
            continue
        covs = [r["pooled_coverage"] for r in ood_results[config]]
        mae = [r["eval_mean_mae"] for r in ood_results[config]]
        agg[config] = {
            "mean_pooled_coverage": float(np.mean(covs)),
            "pooled_coverage_per_seed": covs,
            "eval_mean_mae": float(np.mean(mae)),
            "eval_mean_mae_per_seed": mae,
        }

    output = {
        "status": "OOD conformal evaluation (Step 6), same q_hat as ID, no recalibration",
        "ood_manifest": str(OOD_MANIFEST),
        "id_results": str(STAGE7_RESULTS),
        "normalization_check": norm_check,
        "configs": CONFIGS,
        "seeds": SEEDS,
        "per_seed": ood_results,
        "aggregate": agg,
    }
    out_path = RESULTS_DIR / "stage7_ood_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary.
    print("\n===== OOD SUMMARY (90% nominal, same q_hat as ID) =====")
    for config in CONFIGS:
        if config not in agg:
            continue
        a = agg[config]
        print(f"\n{config.upper()}:")
        print(f"  pooled coverage   : {[f'{c:.3f}' for c in a['pooled_coverage_per_seed']]} "
              f"(mean {a['mean_pooled_coverage']:.3f})")
        print(f"  eval mean MAE     : {[f'{m:.4f}' for m in a['eval_mean_mae_per_seed']]} "
              f"(mean {a['eval_mean_mae']:.4f})")
        # Per-nrecon on first seed.
        if ood_results[config]:
            r0 = ood_results[config][0]
            print("  per-nrecon coverage (seed 42):")
            for g in sorted(r0["coverage_by_nrecon"], key=lambda k: int(k)):
                cov = r0["coverage_by_nrecon"][g]
                w = r0["width_by_nrecon"][g]
                flag = "OK" if cov["coverage"] >= 0.9 else "BELOW"
                print(f"    nrecon={g} (n={cov['n']}): "
                      f"cov={cov['coverage']:.3f} {flag} | "
                      f"width mean={w['mean_width']:.3f} med={w['median_width']:.3f}")

    # ID vs OOD comparison.
    print("\n===== ID vs OOD COMPARISON =====")
    for config in CONFIGS:
        if config not in agg:
            continue
        id_agg = id_results["aggregate"][config]
        ood_agg = agg[config]
        print(f"\n{config.upper()}:")
        print(f"  ID  pooled coverage: {id_agg['mean_pooled_coverage']:.3f}  "
              f"width: {id_agg.get('mean_pooled_interval_width', 0):.3f}")
        print(f"  OOD pooled coverage: {ood_agg['mean_pooled_coverage']:.3f}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
