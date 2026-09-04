"""
OOD extrapolation disambiguation analysis.

Separates the two hypotheses for OOD coverage collapse:
  H1: Exchangeability failure (topology is OOD, features are fine)
  H2: Feature extrapolation (normalizer out of domain)

Method: flag each OOD graph by whether ANY node feature exceeds 4σ of
training range, then compute conformal coverage separately for "in-range"
and "extrapolated" subsets. Cross-tabulate by nrecon.

No new MC runs needed — reuses existing OOD dataset + trained models.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
from conformal import conformal_interval, coverage_by_group, empirical_coverage, interval_width_stats
from run_stage7 import build_model

DATA_DIR = REPO_ROOT / "data_generation" / "data"
RESULTS_DIR = REPO_ROOT / "gnn_baseline" / "results"
OOD_DATA = DATA_DIR / "ood_dataset.pkl"
STAGE7_RESULTS = RESULTS_DIR / "stage7_calibration_results.json"
CHECKPOINT_DIR = REPO_ROOT / "gnn_baseline" / "checkpoints"

CONFIGS = ["vanilla", "maxbias_cm"]
SEEDS = [42, 123, 999]


def load_ood_dataset() -> dict:
    with open(OOD_DATA, "rb") as f:
        return pickle.load(f)


def compute_graph_extrapolation_flags(
    ood_dataset: dict,
    feature_stats: Dict[str, float],
    threshold_sigma: float = 4.0,
) -> Dict[str, dict]:
    """Per-graph extrapolation flag.

    A graph is "extrapolated" if ANY of its nodes has ANY feature
    (load_ff, x, or y) outside threshold_sigma of the training mean.

    Returns dict mapping graph_id → {
        "extrapolated": bool,
        "n_extrapolated_nodes": int,
        "max_z_feature": str,
        "max_z_value": float,
        "n_nodes": int,
        "frac_extrapolated": float,
    }
    """
    flags = {}
    for gid, entry in ood_dataset.items():
        gates = entry["graph"]["gates"]
        n_nodes = len(gates)
        n_extrap = 0
        worst_z = 0.0
        worst_feat = ""

        for name, gate in gates.items():
            for feat, val, fmean, fstd in [
                ("load_ff", gate["load_ff"], feature_stats["load_mean"], feature_stats["load_std"]),
                ("x", gate["x"], feature_stats["x_mean"], feature_stats["x_std"]),
                ("y", gate["y"], feature_stats["y_mean"], feature_stats["y_std"]),
            ]:
                z = abs((val - fmean) / max(fstd, 1e-9))
                if z > threshold_sigma:
                    n_extrap += 1
                if z > worst_z:
                    worst_z = z
                    worst_feat = feat

        flags[gid] = {
            "extrapolated": n_extrap > 0,
            "n_extrapolated_nodes": n_extrap,
            "max_z_feature": worst_feat,
            "max_z_value": float(worst_z),
            "n_nodes": n_nodes,
            "frac_extrapolated": n_extrap / max(n_nodes, 1),
        }

    return flags


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ood_dataset = load_ood_dataset()
    with open(STAGE7_RESULTS) as f:
        id_results = json.load(f)

    # Load training normalization stats.
    train_dataset = GraphDataset(split="train", data_dir=DATA_DIR, physics_mode="vanilla")
    feature_stats, target_stats = train_dataset._compute_normalization()

    # Compute extrapolation flags.
    ext_flags = compute_graph_extrapolation_flags(ood_dataset, feature_stats)
    n_extrap = sum(1 for f in ext_flags.values() if f["extrapolated"])
    n_inrange = len(ext_flags) - n_extrap
    print(f"Extrapolation flags ({len(ood_dataset)} graphs, 4σ threshold):")
    print(f"  In-range (no extrapolated nodes): {n_inrange}")
    print(f"  Extrapolated (≥1 node outside 4σ): {n_extrap}")
    print()

    # Build OOD DataLoader (same as run_stage7_ood.py).
    ood_ids = list(ood_dataset.keys())
    ood_graph_dataset = GraphDataset(split="test", data_dir=DATA_DIR, physics_mode="vanilla")
    ood_graph_dataset.graph_ids = ood_ids
    ood_graph_dataset.dataset = ood_dataset

    from torch_geometric.loader import DataLoader
    ood_loader = DataLoader(
        ood_graph_dataset.get_data(feature_stats, target_stats, train_dataset._compute_physics_normalization()),
        batch_size=16, shuffle=False,
    )

    # Run inference + conformal for each config × seed, with per-graph flags.
    all_results = {}  # config → seed → {per_graph_data, coverage_by_group}

    for config in CONFIGS:
        config_results = {}
        for seed in SEEDS:
            model = build_model(config, device)
            ckpt = CHECKPOINT_DIR / f"best_model_{config}_seed{seed}.pt"
            if not ckpt.exists():
                print(f"  Skipping {config} seed={seed}: no checkpoint")
                continue
            model.load_state_dict(torch.load(ckpt, weights_only=True))

            # q_hat from ID calibration.
            id_seed = [s for s in id_results["per_seed"][config] if s["seed"] == seed][0]
            q_hat = id_seed["q_hat"]

            # Evaluate.
            eval_metrics = evaluate_model(model, ood_loader, device, target_stats, dataset=ood_loader.dataset)

            # Per-graph conformal intervals + coverage + extrapolation flag.
            per_graph = []
            for g in eval_metrics["per_graph"]:
                gid = g["graph_id"]
                mc = g["mc_mean"]
                pm = g["pred_mean"]
                ps = g["pred_std"]
                score = abs(mc - pm) / max(ps, 1e-9)
                lo = pm - q_hat * ps
                hi = pm + q_hat * ps
                covered = lo <= mc <= hi
                ext = ext_flags.get(gid, {"extrapolated": False, "n_extrapolated_nodes": 0,
                                           "max_z_feature": "", "max_z_value": 0.0,
                                           "frac_extrapolated": 0.0})
                per_graph.append({
                    "graph_id": gid,
                    "nrecon": g["nrecon"],
                    "mc_mean": mc,
                    "pred_mean": pm,
                    "pred_std": ps,
                    "q_hat": q_hat,
                    "interval_lo": lo,
                    "interval_hi": hi,
                    "covered": covered,
                    "score": score,
                    "extrapolated": ext["extrapolated"],
                    "n_extrapolated_nodes": ext["n_extrapolated_nodes"],
                    "max_z_feature": ext["max_z_feature"],
                    "max_z_value": ext["max_z_value"],
                    "frac_extrapolated": ext["frac_extrapolated"],
                })

            config_results[seed] = {
                "q_hat": q_hat,
                "per_graph": per_graph,
            }

        all_results[config] = config_results

    # === Cross-tabulation: coverage by extrapolation flag × nrecon ===
    print("=" * 70)
    print("COVERAGE BY EXTRAPOLATION FLAG × NRECON")
    print("=" * 70)

    summary_tables = {}

    for config in CONFIGS:
        if config not in all_results:
            continue
        print(f"\n--- {config.upper()} ---")

        # Aggregate across seeds (pooled first).
        for flag_label, flag_val in [("ALL", None), ("IN-RANGE", False), ("EXTRAPOLATED", True)]:
            all_covs = []
            all_widths = []
            nrecon_data = {}  # nrecon → list of (covered, width)

            for seed in SEEDS:
                if seed not in all_results[config]:
                    continue
                pg = all_results[config][seed]["per_graph"]
                q_hat = all_results[config][seed]["q_hat"]

                for g in pg:
                    if flag_val is not None and g["extrapolated"] != flag_val:
                        continue
                    all_covs.append(g["covered"])
                    w = g["interval_hi"] - g["interval_lo"]
                    all_widths.append(w)
                    nr = g["nrecon"]
                    if nr not in nrecon_data:
                        nrecon_data[nr] = {"covered": [], "width": []}
                    nrecon_data[nr]["covered"].append(g["covered"])
                    nrecon_data[nr]["width"].append(w)

            if not all_covs:
                print(f"\n  {flag_label}: no graphs in this subset")
                continue

            pooled_cov = np.mean(all_covs)
            pooled_width = np.mean(all_widths)
            n_graphs = len(all_covs)
            print(f"\n  {flag_label} (n={n_graphs}): pooled coverage = {pooled_cov:.3f}, mean width = {pooled_width:.3f}")

            # Per-nrecon breakdown.
            nrecon_rows = []
            for nr in sorted(nrecon_data):
                d = nrecon_data[nr]
                cov = np.mean(d["covered"])
                w = np.mean(d["width"])
                n = len(d["covered"])
                flag = "OK" if cov >= 0.9 else "BELOW"
                print(f"    nrecon={nr} (n={n}): cov={cov:.3f} {flag}, width={w:.3f}")
                nrecon_rows.append({"nrecon": nr, "n": n, "coverage": cov, "width": w})

            summary_tables[(config, flag_label)] = {
                "n_graphs": n_graphs,
                "pooled_coverage": pooled_cov,
                "mean_width": pooled_width,
                "per_nrecon": nrecon_rows,
            }

    # === Save per-graph results for auditability ===
    output = {
        "extrapolation_threshold": "4σ of training feature mean",
        "n_inrange": n_inrange,
        "n_extrapolated": n_extrap,
        "per_graph_flags": ext_flags,
        "per_config_seed": all_results,
        "summary_tables": {f"{k[0]}|{k[1]}": v for k, v in summary_tables.items()},
    }
    out_path = RESULTS_DIR / "stage7_ood_extrapolation_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # === Final disambiguation verdict ===
    print("\n" + "=" * 70)
    print("DISAMBIGUATION VERDICT")
    print("=" * 70)

    for config in CONFIGS:
        in_range_key = (config, "IN-RANGE")
        extrap_key = (config, "EXTRAPOLATED")
        all_key = (config, "ALL")

        if in_range_key not in summary_tables:
            continue

        ir = summary_tables[in_range_key]
        ex = summary_tables.get(extrap_key, {"pooled_coverage": None, "n_graphs": 0})
        al = summary_tables[all_key]

        print(f"\n{config.upper()}:")
        print(f"  ALL OOD:           {al['pooled_coverage']:.3f} (n={al['n_graphs']})")
        print(f"  In-range topology: {ir['pooled_coverage']:.3f} (n={ir['n_graphs']})")
        if ex["pooled_coverage"] is not None:
            print(f"  Extrapolated:      {ex['pooled_coverage']:.3f} (n={ex['n_graphs']})")

        if ir["pooled_coverage"] is not None and ir["pooled_coverage"] < 0.85:
            print(f"  → H1 SUPPORTED: coverage collapses even on in-range-topology OOD graphs")
            print(f"    Exchangeability failure is the dominant mechanism.")
        elif ir["pooled_coverage"] is not None and ir["pooled_coverage"] >= 0.85:
            print(f"  → H2 SUPPORTED: in-range-topology OOD coverage is near-nominal")
            print(f"    Feature extrapolation is the dominant mechanism.")
        else:
            print(f"  → INCONCLUSIVE: insufficient data in in-range subset")

    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
