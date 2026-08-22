"""Bulk verifier for Stage 6C artifacts: recomputes aggregates + sha256 digests
from per-graph arrays without bulk transmission.

Usage:
    python verify_stage6c.py                       # full-run artifact
    python verify_stage6c.py gnn_baseline/results/stage6c_results_smoke.json
    python verify_stage6c.py gnn_baseline/results/stage6c_results_interim.json
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT = REPO_ROOT / "gnn_baseline" / "results" / "stage6c_results.json"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT

d = json.load(open(path))
print(f"file: {path}")
print("first key:", next(iter(d)))
print("top-level keys:", list(d.keys()))
print("status:", json.dumps(d.get("status", "<none>")))
if "environment" in d:
    print("environment:", json.dumps(d["environment"]))


def digest(pg):
    return hashlib.sha256(json.dumps(pg, sort_keys=True).encode()).hexdigest()[:16]


for cfg, runs in d.get("results", {}).items():
    for r in runs:
        pg = r["test_metrics"]["per_graph"]
        m = np.array([p["mean_mae"] for p in pg])
        s = np.array([p["std_mae"] for p in pg])
        tm = r["test_metrics"]
        hist = r.get("history") or r.get("train_result", {}).get("history", {})
        n_ep = len(hist.get("val_loss", [])) if isinstance(hist, dict) else 0
        print(f"{cfg}/seed{r['seed']}: n={len(pg)} mean_mae={m.mean():.10f} (json {tm['mean_mae']:.10f}) "
              f"std_mae={s.mean():.10f} (json {tm['std_mae']:.10f}) per_graph_sha256={digest(pg)} "
              f"n_params={r.get('n_params')} best_epoch={r.get('best_epoch')} "
              f"epochs={n_ep} train_time={r.get('train_time', -1):.2f}s")
    maes = [r["test_metrics"]["mean_mae"] for r in runs]
    print(f"  -> {cfg} aggregate: {np.mean(maes):.4f} ± {np.std(maes, ddof=1):.4f}")

ls = d.get("lockstep_verification", {})
if ls:
    print(f"\nlockstep: status={ls.get('status')} all_match={ls.get('all_match')} "
          f"max_mean_diff={ls.get('max_mean_diff')} max_std_diff={ls.get('max_std_diff')}")

sig = d.get("paired_significance_vs_vanilla", {})
for tier, s in sig.items():
    print(f"significance {tier}: mean_delta={s['mean_delta']:+.4f} CI=[{s['mean_ci_low']:+.4f},{s['mean_ci_high']:+.4f}] "
          f"| std_delta={s['std_delta']:+.4f} CI=[{s['std_ci_low']:+.4f},{s['std_ci_high']:+.4f}]")

ng = d.get("no_gnn_baseline", {})
if ng:
    print(f"\nno_gnn: mlp={ng.get('mean_mae_mean'):.4f}±{ng.get('mean_mae_std'):.4f} "
          f"ols_floor={ng.get('ols_floor', {}).get('mean_mae'):.4f} "
          f"slope_sink_mean={ng.get('ols_floor', {}).get('slope_sink_mean', float('nan')):.4f} "
          f"gates={json.dumps(ng.get('convergence_gates', {}))}")

ta = d.get("tier_a_diagnostics", {})
if ta:
    mi = ta.get("magnitude_identities", {})
    print(f"\ntier_a_diagnostics: corr(vth,l vs load)={ta.get('corr_vth_load')}/{ta.get('corr_l_load')}/{ta.get('corr_w_load')}")
    for key in ("vth_identity", "l_identity", "w_identity"):
        b = mi.get(key, {})
        print(f"  {key}: {b.get('mean', float('nan')):+.6f} ± {b.get('std', float('nan')):.2e}")
    print(f"  cross vth/l: {mi.get('d_free_cross_ratio_vth_over_l_mean')} (expected {mi.get('expected_vth_over_l_alpha_l_over_vdd_minus_vth')})")
    print(f"  cross w/l:   {mi.get('d_free_cross_ratio_w_over_l_mean')} (expected {mi.get('expected_w_over_l_minus_l_over_2w')})")
    print(f"  k={mi.get('k_value')} discriminative={mi.get('k_placement_discriminative')}")
