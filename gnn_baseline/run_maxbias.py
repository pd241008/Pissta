"""
Stage 6C architectural variant — MAX-biased aggregation vs vanilla (frozen protocol).

Runs configs ["vanilla", "maxbias"] (or ["vanilla", "maxbias_cm"] when
MAXBIAS_CM=1) over seeds [42, 123, 999] with the exact same frozen
hyperparameters/splits/dataset as the verified run, then:
  - lockstep-verifies vanilla against the committed 6B baseline,
  - computes the paired per-graph cluster-bootstrap CI vs vanilla,
  - reports the param-count delta.

Two modes, distinct output files:
  * default (full-capacity): maxbias hidden 64 -> 41986 params vs vanilla 29698.
    The mean+max fanin aggregation costs parameters, so the caveat is stated:
    an improvement conflates architecture effect with capacity effect.
  * MAXBIAS_CM=1 (capacity-matched control): maxbias hidden 54 -> 30026 params,
    i.e. ~vanilla's 29698. Near-equal capacity isolates the aggregation change
    from the capacity bump. This is the decisive test for whether MAX-bias
    aggregation helps per se, versus the full-capacity result merely being
    "more parameters help". Writes stage6c_results_maxbias_cm.json, leaving
    stage6c_results_maxbias.json (full-capacity artifact) intact.

hypothesis being tested: does biasing message-passing aggregation toward a
MAX-like combination (Clark-MAX faithful at reconvergence) beat the pure-mean
aggregation, on the SAME 3-dim features? Feature injection tied/failed across
four variants; this is the architectural lever. Default writes
gnn_baseline/results/stage6c_results_maxbias.json so the authoritative
stage6c_results.json (A/A+B) is left untouched.
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
# MAXBIAS_CM=1 -> capacity-matched control: reduce the mean+max model's hidden
# dim (54) so its params (~30026) match vanilla h=64 (29698), isolating the
# aggregation change from the capacity bump that the h=64 maxbias run (41986)
# carries. Both variants are pure architectural changes on the same 3-dim
# features. Distinct output file keeps the full-capacity artifact intact.
CM = os.environ.get("MAXBIAS_CM", "0") == "1"

CONFIG_MODE = "maxbias_cm" if CM else "maxbias"
OUTFILE = "stage6c_results_maxbias_cm.json" if CM else "stage6c_results_maxbias.json"
HIDDEN = 54 if CM else 64

from dataset import create_dataloaders, GraphDataset
from run_stage6c import (
    run_single_seed, paired_cluster_bootstrap_ci, compute_lockstep_verification,
    _to_serializable, preflight, check_batch_shape,
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
configs = ["vanilla", CONFIG_MODE]


def _agg(res):
    return {
        "mean_mae": float(sum(r['test_metrics']['mean_mae'] for r in res) / len(res)),
        "std_mae": float(sum(r['test_metrics']['std_mae'] for r in res) / len(res)),
        "mean_rel": float(sum(r['test_metrics']['mean_relative'] for r in res) / len(res)),
        "std_rel": float(sum(r['test_metrics']['std_relative'] for r in res) / len(res)),
    }


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

        # maxbias is an architectural variant of vanilla: SAME 3-dim features.
        physics_mode = "vanilla"
        train_loader, val_loader, test_loader, fs, ts, ps = create_dataloaders(
            data_dir=data_dir, batch_size=32, physics_mode=physics_mode,
        )
        per_config_stats[config] = {
            "feature_stats": {k: float(v) for k, v in fs.items()},
            "target_stats": {k: float(v) for k, v in ts.items()},
            "physics_stats": {k: float(v) for k, v in ps.items()},
        }

        sample_batch = next(iter(train_loader))
        expected = 3
        check_batch_shape(sample_batch, train_loader.batch_size, expected)
        print(f"Batch OK: x={sample_batch.x.shape}, y={sample_batch.y.shape}")

        test_dataset = GraphDataset(split="test", data_dir=data_dir, physics_mode=physics_mode)
        train_dataset = GraphDataset(split="train", data_dir=data_dir, physics_mode=physics_mode)

        cfg_results = []
        physics_t = 0.0
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
    maxbias_results = all_results[CONFIG_MODE]

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
    sig = paired_cluster_bootstrap_ci(vanilla_results, maxbias_results, tier_name=CONFIG_MODE)
    label = "maxbias_cm" if CM else "maxbias"
    print(f"  {label.upper()}:")
    print(f"    Mean MAE delta ({label} - vanilla): {sig['mean_delta']:+.4f}  win_rate={sig['mean_win_rate']:.1%}  95% CI: [{sig['mean_ci_low']:+.4f}, {sig['mean_ci_high']:+.4f}]  sig={'Yes' if (sig['mean_ci_low']>0 or sig['mean_ci_high']<0) else 'No'}")
    print(f"    Std  MAE delta ({label} - vanilla): {sig['std_delta']:+.4f}  win_rate={sig['std_win_rate']:.1%}  95% CI: [{sig['std_ci_low']:+.4f}, {sig['std_ci_high']:+.4f}]  sig={'Yes' if (sig['std_ci_low']>0 or sig['std_ci_high']<0) else 'No'}")
    print(f"    Per-seed mean deltas: {[f'{d:+.4f}' for d in sig['per_seed_mean_deltas']]}")

    vb = _agg(vanilla_results); mb = _agg(maxbias_results)
    nv = vanilla_results[0]['n_params']; nm = maxbias_results[0]['n_params']
    print("\nPoint estimates (fresh run):")
    print(f"  Vanilla       : mean_mae={vb['mean_mae']:.4f}  std_mae={vb['std_mae']:.4f}  mean_rel={vb['mean_rel']:.2%}  std_rel={vb['std_rel']:.2%}  params={nv}")
    print(f"  {label:<13}: mean_mae={mb['mean_mae']:.4f}  std_mae={mb['std_mae']:.4f}  mean_rel={mb['mean_rel']:.2%}  std_rel={mb['std_rel']:.2%}  params={nm}")
    if CM:
        print(f"  NOTE: capacity-matched control — {label} (h=54) carries {nm} params vs vanilla {nv} "
              f"({100*(nm-nv)/nv:+.1f}%). Near-equal capacity: any remaining difference is attributable "
              f"to the mean+max aggregation itself, not added parameters.")
    else:
        print(f"  CAVEAT: maxbias carries {nm-nv} more params than vanilla (+{100*(nm-nv)/nv:.1f}%) — "
              "any gain conflates architecture effect with capacity effect.")

    output = {
        "status": (
            f"{CONFIG_MODE} {'capacity-matched' if CM else 'full-capacity'} standalone | lockstep={lockstep.get('status')} "
            f"(all_match={lockstep.get('all_match')}) | paired cluster CI "
            f"{CONFIG_MODE} mean_delta={sig['mean_delta']:+.4f} "
            f"CI=[{sig['mean_ci_low']:+.4f},{sig['mean_ci_high']:+.4f}] | "
            f"param_delta={nm-nv:+d}"
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
        "stability_summary": {cfg: _agg(all_results[cfg]) for cfg in configs},
        "paired_significance_vs_vanilla": sig,
        "lockstep_verification": lockstep,
        "physics_stats": per_config_stats["vanilla"]["physics_stats"],
    }
    out_path = results_dir / OUTFILE
    with open(out_path, "w") as f:
        json.dump(_to_serializable(output), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
