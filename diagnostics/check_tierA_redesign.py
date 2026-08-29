"""
Step 1 — Redundancy & degeneracy check for REDESIGNED Tier A feature: var_d/load_ff^2.

The old Tier A used raw linearized sensitivities (d/dVth, d/dL, d/dW), which are
algebraically proportional to load_ff (fixed global ratios). That made them carry
zero information beyond load_ff.

This diagnostic computes the NEW per-node feature var_d/load_ff^2 using the exact
same formula production uses (data_generation/run_stage6a.py:94):

    var_d = (d/dVth)^2 * var_Vth + (d/dL)^2 * var_L + (d/dW)^2 * var_W
    feature = var_d / load_ff^2

where var_Vth/var_L/var_W are position-dependent (spatial covariance diagonal +
Pelgrom d_eff), so the feature is NOT expected to be a fixed linear function of
load_ff the way the old sensitivities were.

Checks (all cheap, no training):
  (1) OLS regression of feature vs load_ff  -> R^2, beta  (the exact test that
      killed the old Tier A at R^2 ≈ 1.0)
  (2) Correlations vs load_ff, x, y
  (3) Variance/range/spread across nodes (flag near-constant = degenerate)

Gate: proceed only if R^2 vs load_ff is meaningfully below the old feature's
(≈0.99+), ideally < 0.3-0.5.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from foundations.config_loader import load_config
from variation.analytical import compute_process_moments

DATA = REPO / "data_generation" / "data" / "dataset.pkl"


def compute_feature_map(entry: dict):
    """Return {gate_name: var_d/load_ff^2} from the DATASET (the exact feature the
    model consumes in training)."""
    return entry["physics_features"]["var_d_per_load_ff_sq"]


def main() -> None:
    dataset = pickle.load(open(DATA, "rb"))

    feats, loads, xs, ys = [], [], [], []
    hash_ctx = hashlib.sha256()
    n_graphs = 0
    # deterministic order for the feature-function hash on a fixed graph
    sorted_gids = sorted(dataset.keys())
    hash_gid = sorted_gids[0]
    hash_entry = dataset[hash_gid]
    hash_map = compute_feature_map(hash_entry)
    hash_ctx.update(json.dumps(
        {k: round(v, 12) for k, v in sorted(hash_map.items())}, sort_keys=True
    ).encode())

    for gid in sorted_gids:
        entry = dataset[gid]
        fmap = compute_feature_map(entry)
        for name, val in fmap.items():
            gate = entry["graph"]["gates"][name]
            feats.append(val)
            loads.append(gate["load_ff"])
            xs.append(gate["x"])
            ys.append(gate["y"])
        n_graphs += 1

    feats = np.array(feats)
    loads = np.array(loads)
    xs = np.array(xs)
    ys = np.array(ys)

    def _ols_r2(y, X):
        Xd = np.column_stack([X, np.ones((len(y), 1))])
        coef = np.linalg.lstsq(Xd, y, rcond=None)[0]
        resid = y - Xd @ coef
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _corr(a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    # Regression against load_ff ALONE (the exact test that killed old Tier A).
    r2_load = _ols_r2(feats, loads.reshape(-1, 1))
    # Honest full-redundancy test: against the COMPLETE vanilla node-feature set
    # (load_ff, x, y) that the model already receives. If the new feature is
    # linearly predictable from these, it adds ~nothing.
    r2_vanilla_set = _ols_r2(feats, np.column_stack([loads, xs, ys]))

    result = {
        "n_nodes": len(feats),
        "n_graphs": n_graphs,
        "feature_mean": float(feats.mean()),
        "feature_std": float(feats.std()),
        "feature_min": float(feats.min()),
        "feature_max": float(feats.max()),
        "feature_cv": float(feats.std() / max(abs(feats.mean()), 1e-12)),
        "collapse_fraction": float(
            np.mean(np.abs(feats - feats.mean()) < 1e-9)
        ),
        "ols": {
            "R2_vs_load_ff": r2_load,
            "R2_vs_full_vanilla_set_load_xy": r2_vanilla_set,
        },
        "corr_vs_load_ff": _corr(feats, loads),
        "corr_vs_x": _corr(feats, xs),
        "corr_vs_y": _corr(feats, ys),
        "feature_function_hash_sha256": hash_ctx.hexdigest(),
        "hash_graph_id": hash_gid,
        "comparison": "old Tier A sensitivities: corr(load_ff) & |R2| ~ 1.0 (killed).",
    }

    print(json.dumps(result, indent=2))

    status = "PASS (low redundancy vs load_ff)"
    if r2_load >= 0.5:
        status = "FAIL (still collinear with load_ff)"
    print(f"\nGATE R2(load_ff)      = {r2_load:.4f}  ->  {status}")
    status2 = (
        "WARNING: largely redundant with the x coordinate / full vanilla set"
        if r2_vanilla_set >= 0.5
        else "OK: not linearly predictably from full vanilla set"
    )
    print(f"GATE R2(full vanilla set=load,x,y) = {r2_vanilla_set:.4f}  ->  {status2}")


if __name__ == "__main__":
    main()
