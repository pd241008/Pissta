"""
Step 2 — Per-node leakage check for REDESIGNED Tier B (per-node AT_mean/AT_var).

Old Tier B was graph-level sink_mean/sink_std and leaked ~linearly in the target.
Redesign: expose per-node analytical AT_mean and AT_var as node-level features.

Inherent design: the SINK node's AT = analytical prediction of the sink MC delay,
so a strong sink-vs-target relation is EXPECTED and acceptable (that is the
physics-informed point). The concern is whether EVERY node leaks -- i.e. whether
non-sink nodes' AT features are also near-linearly proportional to the sink MC
target, which would let the model read the label anywhere instead of actually
learning the DAG-to-delay mapping.

Only the graph-level MC label (sink delay mean/std) is available, so the check is:
how well does each node's analytical AT_mean / AT_var predict the sink MC target?
Expected healthy profile: sink is the strongest predictor; leakage falls off for
shallow/non-sink nodes. Flag: if non-sink AT_mean/AT_var still hit high R^2 vs
the target (e.g. >= 0.5) across many nodes -> every node leaks.

Gate: report per-node R^2 distribution; flag if the median non-sink node R^2 is
high (>= 0.5), which would signal pervasive leakage.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA = REPO / "data_generation" / "data" / "dataset.pkl"


def main() -> None:
    dataset = pickle.load(open(DATA, "rb"))

    at_mean_sink, at_var_sink = [], []
    at_mean_non, at_var_non = [], []
    tgt_mean = []  # graph MC label mean, repeated per node for sink alignment
    tgt_std = []
    sink_row_idx = []  # which target each sink sample belongs to
    nonsink_row_idx = []  # which target each non-sink sample belongs to

    for gid in sorted(dataset.keys()):
        entry = dataset[gid]
        asta = entry["physics_features"]["analytical_ssta"]
        mc = entry["mc_labels"]
        sink = entry["graph"]["sink"]
        tgt_mean.append(float(mc["mean"]))
        tgt_std.append(float(mc["std"]))
        for name, am in asta["AT_mean"].items():
            av = asta["AT_var"][name]
            if name == sink:
                at_mean_sink.append(float(am))
                at_var_sink.append(float(av))
                sink_row_idx.append(len(tgt_mean) - 1)
            else:
                at_mean_non.append(float(am))
                at_var_non.append(float(av))
                nonsink_row_idx.append(len(tgt_mean) - 1)

    at_mean_sink = np.array(at_mean_sink)
    at_var_sink = np.array(at_var_sink)
    at_mean_non = np.array(at_mean_non)
    at_var_non = np.array(at_var_non)
    tgt_mean = np.array(tgt_mean)
    tgt_std = np.array(tgt_std)
    sink_row_idx = np.array(sink_row_idx)
    nonsink_row_idx = np.array(nonsink_row_idx)
    tgt_mean_sink = tgt_mean[sink_row_idx]
    tgt_std_sink = tgt_std[sink_row_idx]
    tgt_mean_non = tgt_mean[nonsink_row_idx]
    tgt_std_non = tgt_std[nonsink_row_idx]

    def _ols_r2(y, X):
        Xd = np.column_stack([X, np.ones(len(y))])
        coef = np.linalg.lstsq(Xd, y, rcond=None)[0]
        resid = y - Xd @ coef
        ss = float(np.sum(resid**2))
        sst = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss / sst if sst > 0 else 0.0

    def _corr(a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    # pooled per-node R^2 across all non-sink nodes -> one number per node column
    r2_nonsink_mean_vs_target = _ols_r2(tgt_mean_non, at_mean_non)
    r2_nonsink_var_vs_target = _ols_r2(tgt_mean_non, at_var_non)
    corr_nonsink_mean_vs_target = _corr(at_mean_non, tgt_mean_non)
    corr_nonsink_var_vs_target = _corr(at_var_non, tgt_mean_non)

    # SINK (expected leak) for contrast
    r2_sink_mean_vs_target = _ols_r2(tgt_mean, at_mean_sink)
    r2_sink_var_vs_target = _ols_r2(tgt_mean, at_var_sink)
    corr_sink_mean_vs_target = _corr(at_mean_sink, tgt_mean)
    corr_sink_var_vs_target = _corr(at_var_sink, tgt_mean)

    result = {
        "n_graphs": len(dataset),
        "n_sink_samples": len(at_mean_sink),
        "n_nonsink_samples": len(at_mean_non),
        "SINK (expected leakage, acceptable)": {
            "corr(AT_mean, target)": corr_sink_mean_vs_target,
            "R2(AT_mean, target)": r2_sink_mean_vs_target,
            "corr(AT_var, target)": corr_sink_var_vs_target,
            "R2(AT_var, target)": r2_sink_var_vs_target,
        },
        "NON-SINK (flag if high)": {
            "corr(AT_mean, target)": corr_nonsink_mean_vs_target,
            "R2(AT_mean, target)": r2_nonsink_mean_vs_target,
            "corr(AT_var, target)": corr_nonsink_var_vs_target,
            "R2(AT_var, target)": r2_nonsink_var_vs_target,
        },
    }

    print(json.dumps(result, indent=2))

    # Per-node-position profile: R^2 of node AT_mean vs target by topological
    # depth (how close to sink the node is). Expected: deeper nodes leak more.
    # We approximate depth by x-coordinate (nodes placed along x = pipeline depth).
    xs = []
    r2s = []
    edges_depth_buckets = {"shallow(x<2)": [], "mid(2<=x<4)": [], "deep(x>=4)": []}
    for gid in sorted(dataset.keys()):
        entry = dataset[gid]
        asta = entry["physics_features"]["analytical_ssta"]
        tgt = float(entry["mc_labels"]["mean"])
        sink = entry["graph"]["sink"]
        for name, am in asta["AT_mean"].items():
            if name == sink:
                continue
            x = entry["graph"]["coordinates"][name][0]
            b = "shallow(x<2)" if x < 2 else ("mid(2<=x<4)" if x < 4 else "deep(x>=4)")
            edges_depth_buckets[b].append((am, tgt))
    print("\nNon-sink AT_mean -> target R^2, bucketed by node depth (x):")
    for b, pairs in edges_depth_buckets.items():
        if not pairs:
            print(f"  {b}: n=0")
            continue
        arr = np.array(pairs)
        r2 = _ols_r2(arr[:, 1], arr[:, 0])
        print(f"  {b}: n={len(pairs):5d}  R2={r2:.4f}  corr={_corr(arr[:,0],arr[:,1]):.4f}")

    r2ns = r2_nonsink_var_vs_target
    flag = "EVERY NODE LEAKS (non-sink R2 >= 0.5): per-node Tier B is effectively reading the target" if r2ns >= 0.5 else "OK: non-sink leakage is low; sink is the dominant leak"
    print(f"\nGATE non-sink AT_var R2 vs target = {r2ns:.4f}  ->  {flag}")


if __name__ == "__main__":
    main()
