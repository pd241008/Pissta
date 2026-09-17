"""
Stage 6C combined ablation — MAX-biased aggregation (capacity-matched) +
redesigned Tier A+B per-node physics features (vs vanilla, frozen protocol).

The two levers that each produced results on their own:
  * architectural MAX-bias (Section 7 / ADR-008 addendum #2): mean+max fanin
    aggregation, capacity-matched h=54 — the arc's first positive result
    (Delta -0.062 vs vanilla, 7 seeds).
  * redesigned Tier A+B features (per-node var_d/load_ff^2 + AT_mean/AT_var,
    6-dim node features, physics_mode="tier_ab"): the closest-to-tied feature
    variant (ns, -0.007 on 3 seeds).

This harness runs BOTH TOGETHER on the same frozen architecture/hyperparameters/
splits/seeds protocol as maxbias_cm, to test whether the physics features add
anything on top of the positive aggregation architecture (the Stage 8
combined-ablation follow-up).

Combined config = MaxBiasedDAGGNNSage(num_node_features=6, hidden_dim=54)
  (capacity matched to vanilla's 29698 params).

Runs configs ["vanilla", "maxbias_cm_tierab"] over the given seeds (default
[42, 123, 999]) with the frozen hyperparameters/splits/dataset of the verified
run, then:
  - lockstep-verifies vanilla against the committed 6B baseline,
  - computes the paired per-graph cluster-bootstrap CI vs vanilla,
  - reports param-count and training-time (avg_train_time_s, per-seed train_time)
    so the Stage 8 cost accounting has real numbers for this variant.

Writes gnn_baseline/results/stage6c_results_maxbias_cm_tierab.json; leaves the
authoritative stage6c_results.json and the maxbias artifacts untouched.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
SMOKE = os.environ.get("VLSI_SMOKE", "0") == "1"

from dataset import create_dataloaders, GraphDataset
from run_stage6c import (
    run_single_seed, paired_cluster_bootstrap_ci, compute_lockstep_verification,
    _to_serializable, preflight, check_batch_shape, measure_physics_feature_time,
)

script_dir = Path(__file__).resolve().parent
results_dir = script_dir / "results"
data_dir = REPO_ROOT / "data_generation" / "data"
checkpoint_dir = script_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
seeds = [int(s) for s in os.environ.get("VLSI_SEEDS", "42,123,999").split(",")]
configs = ["vanilla", "maxbias_cm_tierab"]
PHYSICS_MODE = {"vanilla": "vanilla", "maxbias_cm_tierab": "tier_ab"}
NODE_FEATS = {"vanilla": 3, "maxbias_cm_tierab": 6}


def _agg(res):
    return {
        "mean_mae": float(sum(r['test_metrics']['mean_mae'] for r in res) / len(res)),
        "std_mae": float(sum(r['test_metrics']['std_mae'] for r in res) / len(res)),
        "mean_rel": float(sum(r['test_metrics']['mean_relative'] for r in res) / len(res)),
        "std_rel": float(sum(r['test_metrics']['std_relative'] for r in res) / len(res)),
        "avg_train_time_s": float(np.mean([r["train_result"]["train_time"] for r in res])),
        "avg_inference_ms_per_graph": float(np.mean([r["test_metrics"]["avg_inference_time_ms"] for r in res])),
    }


def main() -> None:
    checks = preflight(data_dir, "all")
    if not checks["ok"]:
        print(f"Preflight failed: {checks['errors']}"); sys.exit(1)

    all_results = {}
    per_config_stats = {}
    for config in configs:
        print(f"\n{'#'*60}\n# Configuration: {config}\n{'#'*60}")
        pm = PHYSICS_MODE[config]
        train_loader, val_loader, test_loader, fs, ts, ps = create_dataloaders(
            data_dir=data_dir, batch_size=32, physics_mode=pm,
        )
        per_config_stats[config] = {
            "feature_stats": {k: float(v) for k, v in fs.items()},
            "target_stats": {k: float(v) for k, v in ts.items()},
            "physics_stats": {k: float(v) for k, v in ps.items()},
        }

        sample_batch = next(iter(train_loader))
        check_batch_shape(sample_batch, train_loader.batch_size, NODE_FEATS[config])
        print(f"Batch OK: x={sample_batch.x.shape}, y={sample_batch.y.shape}")

        test_dataset = GraphDataset(split="test", data_dir=data_dir, physics_mode=pm)
        train_dataset = GraphDataset(split="train", data_dir=data_dir, physics_mode=pm)

        cfg_results = []
        physics_t = 0.0
        if config != "vanilla":
            physics_t = measure_physics_feature_time(test_dataset, config="tier_ab", n_samples=100)
            print(f"Physics feature time (analytical SSTA + sensitivities): {physics_t:.2f} ms/graph")
        for seed in seeds:
            r = run_single_seed(
                seed=seed, train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, test_dataset=test_dataset,
                feature_stats=fs, target_stats=ts, device=device,
                checkpoint_dir=checkpoint_dir, config_name=config,
                physics_feature_time_ms=physics_t,
            )
            cfg_results.append(r)
        all_results[config] = cfg_results

    vanilla_results = all_results["vanilla"]
    combined_results = all_results["maxbias_cm_tierab"]

    # Lockstep verification of the fresh vanilla run against committed 6B
    print("\n--- Lockstep Verification (6B vanilla vs fresh 6C vanilla) ---")
    lockstep = compute_lockstep_verification(vanilla_results, script_dir)
    if lockstep.get("status") == "verified":
        print(f"  status=verified all_match={lockstep['all_match']} "
              f"max_mean_diff={lockstep['max_mean_diff']} max_std_diff={lockstep['max_std_diff']}")
    else:
        print(f"  {lockstep.get('status')}: {lockstep.get('reason')}")

    # Paired cluster-bootstrap CI vs vanilla
    print("\n--- Paired Per-Graph Significance vs Vanilla ---")
    sig = paired_cluster_bootstrap_ci(vanilla_results, combined_results, tier_name="maxbias_cm_tierab")
    print(f"  MAXBIAS_CM_TIERAB:")
    print(f"    Mean MAE delta (combined - vanilla): {sig['mean_delta']:+.4f}  win_rate={sig['mean_win_rate']:.1%}  95% CI: [{sig['mean_ci_low']:+.4f}, {sig['mean_ci_high']:+.4f}]  sig={'Yes' if (sig['mean_ci_low']>0 or sig['mean_ci_high']<0) else 'No'}")
    print(f"    Std  MAE delta (combined - vanilla): {sig['std_delta']:+.4f}  win_rate={sig['std_win_rate']:.1%}  95% CI: [{sig['std_ci_low']:+.4f}, {sig['std_ci_high']:+.4f}]  sig={'Yes' if (sig['std_ci_low']>0 or sig['std_ci_high']<0) else 'No'}")
    print(f"    Per-seed mean deltas: {[f'{d:+.4f}' for d in sig['per_seed_mean_deltas']]}")

    vb = _agg(vanilla_results); cb = _agg(combined_results)
    nv = vanilla_results[0]['n_params']; nc = combined_results[0]['n_params']
    print("\nPoint estimates (fresh run):")
    print(f"  Vanilla            : mean_mae={vb['mean_mae']:.4f}  std_mae={vb['std_mae']:.4f}  mean_rel={vb['mean_rel']:.2%}  train_time={vb['avg_train_time_s']:.1f}s  params={nv}")
    print(f"  MAX-bias CM + TierAB: mean_mae={cb['mean_mae']:.4f}  std_mae={cb['std_mae']:.4f}  mean_rel={cb['mean_rel']:.2%}  train_time={cb['avg_train_time_s']:.1f}s  params={nc}")
    print(f"  NOTE: capacity-matched — combined (h=54) carries {nc} params vs vanilla {nv} "
          f"({100*(nc-nv)/nv:+.1f}%).")

    output = {
        "status": (
            f"maxbias_cm_tierab combined standalone | lockstep={lockstep.get('status')} "
            f"(all_match={lockstep.get('all_match')}) | paired cluster CI "
            f"maxbias_cm_tierab mean_delta={sig['mean_delta']:+.4f} "
            f"CI=[{sig['mean_ci_low']:+.4f},{sig['mean_ci_high']:+.4f}] | "
            f"param_delta={nc-nv:+d} | "
            f"maxbias_cm_tierab avg_train_time_s={cb['avg_train_time_s']:.1f}"
        ),
        "configs": configs,
        "seeds": seeds,
        "results": {
            cfg: [
                {
                    "seed": r["seed"], "config": r["config"], "n_params": r["n_params"],
                    "best_epoch": r["train_result"]["best_epoch"],
                    "best_val_loss": float(r["train_result"]["best_val_loss"]),
                    "train_time": float(r["train_result"]["train_time"]),
                    "history": r["train_result"].get("history", {}),
                    "eval_train_loss": float(r["eval_train_loss"]),
                    "test_metrics": r["test_metrics"],
                    "physics_feature_time_ms": r.get("physics_feature_time_ms"),
                    "total_inference_ms": r.get("total_inference_ms"),
                }
                for r in all_results[cfg]
            ]
            for cfg in configs
        },
        "stability_summary": {cfg: _agg(all_results[cfg]) for cfg in configs},
        "paired_significance_vs_vanilla": sig,
        "lockstep_verification": lockstep,
        "physics_stats": per_config_stats["maxbias_cm_tierab"]["physics_stats"],
    }
    out_path = results_dir / "stage6c_results_maxbias_cm_tierab.json"
    with open(out_path, "w") as f:
        json.dump(_to_serializable(output), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()