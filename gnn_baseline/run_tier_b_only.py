"""
Step 4 follow-up — Tier-B-ONLY ablation vs vanilla (frozen protocol).

Runs configs ["vanilla", "tier_b_only"] over seeds [42, 123, 999] with the exact
same frozen hyperparameters/splits/dataset as the verified redesigned run, then:
  - lockstep-verifies vanilla against the committed 6B baseline,
  - computes the paired per-graph cluster-bootstrap CI vs vanilla
    (reusing paired_cluster_bootstrap_ci, bit-identical to the verified run),
  - compares tier_b_only point estimate against the already-verified tier_ab.

Hypothesis being tested: if [load,x,y,AT_mean,AT_var] (B only) beats vanilla,
then Tier A (var_d/load_ff²) was noise and B is the real signal; if it only ties,
then B's signal is already recovered by message-passing over structure.

Writes results to gnn_baseline/results/stage6c_results_tier_b_only.json so the
authoritative stage6c_results.json (A/A+B) is left untouched.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
SMOKE = os.environ.get("VLSI_SMOKE", "0") == "1"

from dataset import create_dataloaders, GraphDataset
from run_stage6c import (
    run_single_seed, paired_cluster_bootstrap_ci, compute_lockstep_verification,
    _to_serializable, preflight,
)

script_dir = Path(__file__).resolve().parent
results_dir = script_dir / "results"
data_dir = REPO_ROOT / "data_generation" / "data"
checkpoint_dir = script_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
seeds = [42, 123, 999]
configs = ["vanilla", "tier_b_only"]

def main() -> None:
    checks = preflight(data_dir, "all")
    if not checks["ok"]:
        print(f"Preflight failed: {checks['errors']}"); sys.exit(1)

    all_results = {}
    per_config_stats = {}
    for config in configs:
        print(f"\n{'#'*60}\n# Configuration: {config}\n{'#'*60}")
        c = preflight(data_dir, config)
        if not c["ok"]:
            print(f"Preflight failed for {config}: {c['errors']}"); sys.exit(1)

        train_loader, val_loader, test_loader, fs, ts, ps = create_dataloaders(
            data_dir=data_dir, batch_size=32, physics_mode=config,
        )
        per_config_stats[config] = {
            "feature_stats": {k: float(v) for k, v in fs.items()},
            "target_stats": {k: float(v) for k, v in ts.items()},
            "physics_stats": {k: float(v) for k, v in ps.items()},
        }

        sample_batch = next(iter(train_loader))
        expected = {"vanilla": 3, "tier_b_only": 5}[config]
        from run_stage6c import check_batch_shape
        check_batch_shape(sample_batch, train_loader.batch_size, expected)
        print(f"Batch OK: x={sample_batch.x.shape}, y={sample_batch.y.shape}")

        test_dataset = GraphDataset(split="test", data_dir=data_dir, physics_mode=config)
        train_dataset = GraphDataset(split="train", data_dir=data_dir, physics_mode=config)

        cfg_results = []
        physics_t = None
        if config != "vanilla":
            from run_stage6c import measure_physics_feature_time
            physics_t = measure_physics_feature_time(test_dataset, config="tier_ab", n_samples=100)
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
    tier_b_results = all_results["tier_b_only"]

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
    sig = paired_cluster_bootstrap_ci(vanilla_results, tier_b_results, tier_name="tier_b_only")
    print(f"  TIER_B_ONLY:")
    print(f"    Mean MAE delta (B - vanilla): {sig['mean_delta']:+.4f}  win_rate={sig['mean_win_rate']:.1%}  95% CI: [{sig['mean_ci_low']:+.4f}, {sig['mean_ci_high']:+.4f}]  sig={'Yes' if (sig['mean_ci_low']>0 or sig['mean_ci_high']<0) else 'No'}")
    print(f"    Std  MAE delta (B - vanilla): {sig['std_delta']:+.4f}  win_rate={sig['std_win_rate']:.1%}  95% CI: [{sig['std_ci_low']:+.4f}, {sig['std_ci_high']:+.4f}]  sig={'Yes' if (sig['std_ci_low']>0 or sig['std_ci_high']<0) else 'No'}")
    print(f"    Per-seed mean deltas: {[f'{d:+.4f}' for d in sig['per_seed_mean_deltas']]}")

    # Aggregate table
    def agg(res):
        return {
            "mean_mae": float(sum(r['test_metrics']['mean_mae'] for r in res)/len(res)),
            "std_mae": float(sum(r['test_metrics']['std_mae'] for r in res)/len(res)),
            "mean_rel": float(sum(r['test_metrics']['mean_relative'] for r in res)/len(res)),
            "std_rel": float(sum(r['test_metrics']['std_relative'] for r in res)/len(res)),
        }
    vb = agg(vanilla_results); tb = agg(tier_b_results)
    print("\n3-Way Points (fresh run + verified A/A+B from stage6c_results.json):")
    print(f"  Vanilla      : mean_mae={vb['mean_mae']:.4f}  std_mae={vb['std_mae']:.4f}  mean_rel={vb['mean_rel']:.2%}  std_rel={vb['std_rel']:.2%}")
    print(f"  Tier B only  : mean_mae={tb['mean_mae']:.4f}  std_mae={tb['std_mae']:.4f}  mean_rel={tb['mean_rel']:.2%}  std_rel={tb['std_rel']:.2%}")
    try:
        old = json.load(open(results_dir / "stage6c_results.json"))
    except Exception:
        old = None
    if old:
        for cfg in ["tier_a", "tier_ab"]:
            r = old["results"].get(cfg)
            if r:
                a = agg(r)
                print(f"  Tier {'A ' if cfg=='tier_a' else 'A+B'}      : mean_mae={a['mean_mae']:.4f}  std_mae={a['std_mae']:.4f}  mean_rel={a['mean_rel']:.2%}  std_rel={a['std_rel']:.2%}")

    output = {
        "status": (
            f"tier_b_only standalone | lockstep={lockstep.get('status')} "
            f"(all_match={lockstep.get('all_match')}) | paired cluster CI "
            f"tier_b_only mean_delta={sig['mean_delta']:+.4f} "
            f"CI=[{sig['mean_ci_low']:+.4f},{sig['mean_ci_high']:+.4f}]"
        ),
        "configs": configs,
        "seeds": seeds,
        "results": {
            cfg: [
                {
                    "seed": r["seed"], "config": r["config"], "n_params": r["n_params"],
                    "best_epoch": r["train_result"]["best_epoch"],
                    "best_val_loss": float(r["train_result"]["best_val_loss"]),
                    "eval_train_loss": float(r["eval_train_loss"]),
                    "test_metrics": r["test_metrics"],
                    "physics_feature_time_ms": r.get("physics_feature_time_ms"),
                    "total_inference_ms": r.get("total_inference_ms"),
                }
                for r in all_results[cfg]
            ]
            for cfg in configs
        },
        "stability_summary": {cfg: agg(all_results[cfg]) for cfg in configs},
        "paired_significance_vs_vanilla": sig,
        "lockstep_verification": lockstep,
        "physics_stats": per_config_stats["tier_b_only"]["physics_stats"],
    }
    out_path = results_dir / "stage6c_results_tier_b_only.json"
    with open(out_path, "w") as f:
        json.dump(_to_serializable(output), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
