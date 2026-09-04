"""
Stage 7 — Split conformal calibration of the DAG-GNN surrogate.

Closes the "calibrated uncertainty" half of the central claim. The MAX-bias
GNN (capacity-matched, h=54) and vanilla GNN each output a per-graph critical-
path delay (mean, std). Split conformal calibrates those point/scale estimates
into prediction intervals with a *guaranteed* (under exchangeability) marginal
coverage rate — without assuming the residual distribution is Gaussian.

Method (per user scope decision):
  * Split conformal, nonconformity score = studentized residual
        score = |y_true - mean_pred| / std_pred
  * Calibration set carved from the test split, stratified by nrecon, so the
    Stage 7 coverage is asserted on a held-out EVAL slice of test rather than
    the same graphs whose full-test MAE Stage 6C already reported.
  * q_hat = split-conformal quantile of calibration scores at level alpha.
  * Interval on eval = mean_pred +/- q_hat * std_pred.
  * Check: empirical coverage on eval vs nominal (alpha), pooled and per-nrecon.

Honesty constraints ($2): coverage is exchangeability-guaranteed only within
the same topology family; DISAGGREGATED per-nrecon coverage is the make-or-
break number (a model well-calibrated on average but miscalibrated on high
nrecon / complex topologies is a real failure the pooled number hides). We do
NOT assert coverage under distribution shift to unseen topology families unless
that is separately tested (see OOD note in the report).
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import GraphDataset
from eval import evaluate_model
from model import VanillaDAGGNNSage, MaxBiasedDAGGNNSage
from train import train_model
from conformal import (
    conformal_interval,
    coverage_by_group,
    empirical_coverage,
    split_conformal_quantile,
    studentized_score,
)
from run_stage6c import _to_serializable, set_seed

SMOKE = os.environ.get("VLSI_SMOKE", "0") == "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

script_dir = Path(__file__).resolve().parent
results_dir = script_dir / "results"
data_dir = REPO_ROOT / "data_generation" / "data"
checkpoint_dir = script_dir / "checkpoints"
results_dir.mkdir(parents=True, exist_ok=True)

CARVE_SEED = 7
ALPHA = 0.10  # 90% nominal coverage
SEEDS = [42, 123, 999]
CONFIGS = ["vanilla", "maxbias_cm"]


def _nrecon_of(dataset: dict, gid: str) -> int:
    return len(dataset[gid]["graph"]["reconvergence_points"])


def carve_cal_eval(data_dir: Path, carve_seed: int) -> Tuple[List[str], List[str]]:
    """Carve test split into cal/eval, stratified by nrecon, deterministic.

    Stratification by nrecon keeps both slices topologically representative so
    calibration is not biased toward simple topologies. Returns (cal, eval)
    graph-id lists. Deterministic on carve_seed for auditability.
    """
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)
    with open(data_dir / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    test_ids = splits["test"]
    groups: Dict[int, List[str]] = {}
    for gid in test_ids:
        groups.setdefault(_nrecon_of(dataset, gid), []).append(gid)

    rng = random.Random(carve_seed)
    cal: List[str] = []
    eval_ids: List[str] = []
    for nrecon in sorted(groups):
        ids = groups[nrecon]
        rng.shuffle(ids)
        half = len(ids) // 2
        # If a bucket has an odd count the extra goes to eval (keep cal balanced).
        cal.extend(ids[:half])
        eval_ids.extend(ids[half:])

    return cal, eval_ids


def build_model(config: str, device: torch.device) -> torch.nn.Module:
    if config == "vanilla":
        return VanillaDAGGNNSage(
            num_node_features=3, hidden_dim=64, num_layers=3,
            dropout=0.15, num_outputs=2,
        ).to(device)
    if config == "maxbias_cm":
        return MaxBiasedDAGGNNSage(
            num_node_features=3, hidden_dim=54, num_layers=3,
            dropout=0.15, num_outputs=2,
        ).to(device)
    raise ValueError(f"Unknown config: {config}")


def train_and_evaluate(
    config: str,
    seed: int,
    device: torch.device,
    feature_stats: Dict,
    target_stats: Dict,
    physics_stats: Dict,
    cal_dataset: GraphDataset,
    eval_dataset: GraphDataset,
) -> Dict:
    """Train one seed on the FULL train split, then evaluate on cal and eval.

    Both vanilla and maxbias_cm are trained fresh with the identical frozen
    protocol (same train split, same hyperparameters) so the calibration
    comparison between them is fair and reproducible.
    """
    set_seed(seed)
    train_dataset = GraphDataset(split="train", data_dir=data_dir, physics_mode="vanilla")
    val_dataset = GraphDataset(split="val", data_dir=data_dir, physics_mode="vanilla")

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset.get_data(feature_stats, target_stats, physics_stats),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset.get_data(feature_stats, target_stats, physics_stats),
                            batch_size=32, shuffle=False)

    model = build_model(config, device)
    max_epochs = 10 if SMOKE else 200
    ckpt_path = str(checkpoint_dir / f"best_model_{config}_seed{seed}.pt")
    train_result = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, lr=1e-3, max_epochs=max_epochs, patience=20,
        checkpoint_path=ckpt_path,
    )
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    cal_loader = DataLoader(cal_dataset.get_data(feature_stats, target_stats, physics_stats),
                            batch_size=32, shuffle=False)
    eval_loader = DataLoader(eval_dataset.get_data(feature_stats, target_stats, physics_stats),
                             batch_size=32, shuffle=False)

    cal_metrics = evaluate_model(model, cal_loader, device, target_stats, dataset=cal_loader.dataset)
    eval_metrics = evaluate_model(model, eval_loader, device, target_stats, dataset=eval_loader.dataset)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "seed": seed,
        "config": config,
        "n_params": n_params,
        "best_epoch": train_result["best_epoch"],
        "best_val_loss": float(train_result["best_val_loss"]),
        "eval_train_loss": _eval_mode_train_loss(model, train_loader, device),
        "cal_metrics": cal_metrics,
        "eval_metrics": eval_metrics,
    }


def _eval_mode_train_loss(model, train_loader, device) -> float:
    model.eval()
    total, nb = 0.0, 0
    with torch.no_grad():
        from train import eval_epoch
        loss, _, _ = eval_epoch(model, train_loader, device)
    return float(loss)


def _conformal_run(seed_result: Dict) -> Dict:
    """Fit q_hat on the seed's cal metrics; evaluate coverage on eval metrics.

    Uses the studentized-residual score on the per-graph MEAN delay predictions.
    """
    cal = seed_result["cal_metrics"]["per_graph"]
    ev = seed_result["eval_metrics"]["per_graph"]

    cal_y = np.array([g["mc_mean"] for g in cal])
    cal_mean = np.array([g["pred_mean"] for g in cal])
    cal_std = np.array([g["pred_std"] for g in cal])
    scores = studentized_score(cal_y, cal_mean, cal_std)
    q_hat = split_conformal_quantile(scores, ALPHA)

    ev_y = np.array([g["mc_mean"] for g in ev])
    ev_mean = np.array([g["pred_mean"] for g in ev])
    ev_std = np.array([g["pred_std"] for g in ev])
    intervals = conformal_interval(ev_mean, ev_std, q_hat)
    lo, hi = intervals[:, 0], intervals[:, 1]
    pooled_cov = empirical_coverage(lo, hi, ev_y)

    groups = coverage_by_group(lo, hi, ev_y, np.array([g["nrecon"] for g in ev]))

    return {
        "seed": seed_result["seed"],
        "config": seed_result["config"],
        "alpha": ALPHA,
        "nominal_coverage": 1.0 - ALPHA,
        "n_cal": len(cal),
        "n_eval": len(ev),
        "q_hat": q_hat,
        "pooled_coverage": pooled_cov,
        "coverage_by_nrecon": groups,
        "eval_mean_mae": float(seed_result["eval_metrics"]["mean_mae"]),
        "eval_std_mae": float(seed_result["eval_metrics"]["std_mae"]),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Freeze the calibration split (deterministic, auditable).
    cal_ids, eval_ids = carve_cal_eval(data_dir, CARVE_SEED)
    split_info = {
        "carve_seed": CARVE_SEED,
        "n_cal": len(cal_ids),
        "n_eval": len(eval_ids),
        "cal_ids": sorted(cal_ids),
        "eval_ids": sorted(eval_ids),
    }
    print(f"Carve: cal={len(cal_ids)} eval={len(eval_ids)}")
    with open(results_dir / "splits_stage7.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Normalization stats from train only (frozen protocol).
    train_dataset = GraphDataset(split="train", data_dir=data_dir, physics_mode="vanilla")
    feature_stats, target_stats = train_dataset._compute_normalization()
    physics_stats = train_dataset._compute_physics_normalization()

    # Subclass GraphDataset to accept explicit graph-id lists (cal/eval).
    cal_dataset = GraphDataset(split="test", data_dir=data_dir, physics_mode="vanilla")
    eval_dataset = GraphDataset(split="test", data_dir=data_dir, physics_mode="vanilla")
    cal_dataset.graph_ids = cal_ids
    eval_dataset.graph_ids = eval_ids

    config_results: Dict[str, List[Dict]] = {c: [] for c in CONFIGS}
    for config in CONFIGS:
        for seed in SEEDS:
            print(f"\n===== {config} seed={seed} =====")
            r = train_and_evaluate(
                config, seed, device, feature_stats, target_stats, physics_stats,
                cal_dataset, eval_dataset,
            )
            config_results[config].append(r)

    conformal_results = {c: [] for c in CONFIGS}
    for config in CONFIGS:
        for r in config_results[config]:
            conformal_results[config].append(_conformal_run(r))

    # Aggregate across seeds.
    agg = {}
    for config in CONFIGS:
        covs = [c["pooled_coverage"] for c in conformal_results[config]]
        qs = [c["q_hat"] for c in conformal_results[config]]
        mae = [c["eval_mean_mae"] for c in conformal_results[config]]
        agg[config] = {
            "mean_pooled_coverage": float(np.mean(covs)),
            "pooled_coverage_per_seed": covs,
            "mean_q_hat": float(np.mean(qs)),
            "q_hat_per_seed": qs,
            "eval_mean_mae": float(np.mean(mae)),
            "eval_mean_mae_per_seed": mae,
        }

    output = {
        "status": (
            f"Stage 7 split conformal, alpha={ALPHA}, carved cal/eval from test "
            f"(carve_seed={CARVE_SEED}), configs={CONFIGS}, seeds={SEEDS}"
        ),
        "split": split_info,
        "alpha": ALPHA,
        "nominal_coverage": 1.0 - ALPHA,
        "configs": CONFIGS,
        "seeds": SEEDS,
        "per_seed": conformal_results,
        "aggregate": agg,
    }
    out_path = results_dir / "stage7_calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(_to_serializable(output), f, indent=2)

    print("\n===== STAGE 7 SUMMARY (90% nominal) =====")
    for config in CONFIGS:
        a = agg[config]
        print(f"\n{config.upper()}:")
        print(f"  q_hat per seed    : {[f'{q:.4f}' for q in a['q_hat_per_seed']]}")
        print(f"  pooled coverage   : {[f'{c:.3f}' for c in a['pooled_coverage_per_seed']]} "
              f"(mean {a['mean_pooled_coverage']:.3f})")
        print(f"  eval mean MAE     : {[f'{m:.4f}' for m in a['eval_mean_mae_per_seed']]} "
              f"(mean {a['eval_mean_mae']:.4f})")
        # Per-nrecon coverage on the first seed's eval for a quick look.
        c0 = conformal_results[config][0]
        print("  per-nrecon coverage (seed 42):")
        for g, s in sorted(c0["coverage_by_nrecon"].items(), key=lambda kv: int(kv[0])):
            flag = "OK" if s["coverage"] >= 0.9 else "BELOW"
            print(f"    nrecon={g} (n={s['n']}): cov={s['coverage']:.3f} {flag}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
