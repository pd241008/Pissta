"""
Step 9 — Independent verification of Stage 7 conformal coverage numbers.

Recomputes coverage from raw per-graph predictions + calibration quantile,
without importing gnn_baseline/conformal.py (separate code path to catch
bugs in the module being verified).

Checks performed:
  1. Finite-sample quantile index matches formula ceil((n+1)(1-alpha))/n
  2. Studentized scores match |y_true - mean_pred| / max(std_pred, eps)
  3. Intervals are mean_pred ± q_hat * std_pred
  4. Coverage is fraction of true values within [lo, hi]
  5. Per-nrecon coverage matches reported values
  6. All numbers agree with stage7_calibration_results.json

Run after stage7 run to independently confirm no silent miscomputation.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "gnn_baseline" / "results" / "stage7_calibration_results.json"
_EPS = 1e-9


def _studentized_score(y_true, mean_pred, std_pred):
    """Reimplemented from scratch — does NOT import conformal.py."""
    y_true = np.asarray(y_true, dtype=float)
    mean_pred = np.asarray(mean_pred, dtype=float)
    std_pred = np.asarray(std_pred, dtype=float)
    scale = np.maximum(np.abs(std_pred), _EPS)
    return np.abs(y_true - mean_pred) / scale


def _split_conformal_quantile(scores, alpha):
    """Reimplemented from scratch — does NOT import conformal.py.

    Uses the finite-sample-corrected index: ceil((n+1)(1-alpha))/n.
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    sorted_scores = np.sort(np.asarray(scores, dtype=float))
    q_index = int(math.ceil((n + 1) * (1.0 - alpha)))
    q_index = min(max(q_index, 1), n)
    return float(sorted_scores[q_index - 1])


def _conformal_interval(mean_pred, std_pred, q_hat):
    """Reimplemented from scratch."""
    mean_pred = np.asarray(mean_pred, dtype=float)
    std_pred = np.asarray(std_pred, dtype=float)
    half_width = q_hat * np.abs(std_pred)
    lo = mean_pred - half_width
    hi = mean_pred + half_width
    return lo, hi


def _empirical_coverage(lo, hi, y_true):
    """Reimplemented from scratch."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def _coverage_by_group(lo, hi, y_true, groups):
    """Reimplemented from scratch."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    groups = np.asarray(groups)
    out = {}
    for g in sorted(set(groups)):
        mask = groups == g
        n = int(mask.sum())
        if n == 0:
            continue
        cov = float(np.mean((y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask])))
        out[str(g)] = {"n": n, "coverage": cov}
    return out


def verify_seed(seed_result: dict) -> list:
    """Verify one seed's conformal results. Returns list of (check, status) tuples."""
    checks = []
    config = seed_result["config"]
    seed = seed_result["seed"]
    alpha = seed_result["alpha"]
    n_cal = seed_result["n_cal"]
    n_eval = seed_result["n_eval"]

    # --- Extract raw per-graph data ---
    if "cal_per_graph" not in seed_result or "eval_per_graph" not in seed_result:
        return [(
            f"[{config} seed={seed}] SKIP: per_graph data not in results JSON",
            "N/A (run_stage7.py must be re-run with per_graph persistence)",
        )]
    cal_pg = seed_result["cal_per_graph"]
    eval_pg = seed_result["eval_per_graph"]

    cal_y = np.array([g["mc_mean"] for g in cal_pg])
    cal_mean = np.array([g["pred_mean"] for g in cal_pg])
    cal_std = np.array([g["pred_std"] for g in cal_pg])
    cal_nrecon = np.array([g["nrecon"] for g in cal_pg])

    ev_y = np.array([g["mc_mean"] for g in eval_pg])
    ev_mean = np.array([g["pred_mean"] for g in eval_pg])
    ev_std = np.array([g["pred_std"] for g in eval_pg])
    ev_nrecon = np.array([g["nrecon"] for g in eval_pg])

    # --- Check 1: Verify cal_per_graph scores match recomputed ---
    reported_scores = np.array([g["score"] for g in cal_pg])
    recomputed_scores = _studentized_score(cal_y, cal_mean, cal_std)
    score_maxdiff = float(np.max(np.abs(reported_scores - recomputed_scores)))
    checks.append((
        f"[{config} seed={seed}] Check 1: cal scores match recompute",
        "PASS" if score_maxdiff < 1e-12 else f"FAIL (max diff {score_maxdiff:.2e})",
    ))

    # --- Check 2: Verify q_hat from reported scores ---
    recomputed_qhat = _split_conformal_quantile(reported_scores, alpha)
    qhat_diff = abs(recomputed_qhat - seed_result["q_hat"])
    checks.append((
        f"[{config} seed={seed}] Check 2: q_hat matches recompute",
        "PASS" if qhat_diff < 1e-12 else f"FAIL (diff {qhat_diff:.2e})",
    ))

    # --- Check 3: Verify finite-sample correction index ---
    q_index_expected = int(math.ceil((n_cal + 1) * (1.0 - alpha)))
    q_index_expected = min(max(q_index_expected, 1), n_cal)
    sorted_scores = np.sort(reported_scores)
    q_from_index = float(sorted_scores[q_index_expected - 1])
    index_match = abs(q_from_index - recomputed_qhat) < 1e-12
    checks.append((
        f"[{config} seed={seed}] Check 3: q_hat index ceil(({n_cal}+1)*(1-{alpha}))/{n_cal} = {q_index_expected}",
        "PASS" if index_match else f"FAIL (q from index {q_from_index:.6f} != q_hat {recomputed_qhat:.6f})",
    ))

    # --- Check 4: Verify eval intervals ---
    reported_lo = np.array([g["interval_lo"] for g in eval_pg])
    reported_hi = np.array([g["interval_hi"] for g in eval_pg])
    recomputed_lo, recomputed_hi = _conformal_interval(ev_mean, ev_std, recomputed_qhat)
    lo_maxdiff = float(np.max(np.abs(reported_lo - recomputed_lo)))
    hi_maxdiff = float(np.max(np.abs(reported_hi - recomputed_hi)))
    checks.append((
        f"[{config} seed={seed}] Check 4: eval intervals match recompute",
        "PASS" if lo_maxdiff < 1e-12 and hi_maxdiff < 1e-12
        else f"FAIL (lo maxdiff={lo_maxdiff:.2e}, hi maxdiff={hi_maxdiff:.2e})",
    ))

    # --- Check 5: Verify eval coverage ---
    recomputed_cov = _empirical_coverage(reported_lo, reported_hi, ev_y)
    cov_diff = abs(recomputed_cov - seed_result["pooled_coverage"])
    checks.append((
        f"[{config} seed={seed}] Check 5: pooled coverage matches recompute",
        "PASS" if cov_diff < 1e-12 else f"FAIL (diff {cov_diff:.2e})",
    ))

    # --- Check 6: Verify per-nrecon coverage ---
    recomputed_nrecon = _coverage_by_group(reported_lo, reported_hi, ev_y, ev_nrecon)
    reported_nrecon = seed_result["coverage_by_nrecon"]
    nrecon_ok = True
    nrecon_diffs = []
    for key in reported_nrecon:
        if key in recomputed_nrecon:
            d = abs(reported_nrecon[key]["coverage"] - recomputed_nrecon[key]["coverage"])
            if d > 1e-12:
                nrecon_ok = False
                nrecon_diffs.append(f"nrecon={key}: diff={d:.2e}")
        else:
            nrecon_ok = False
            nrecon_diffs.append(f"nrecon={key}: missing in recompute")
    checks.append((
        f"[{config} seed={seed}] Check 6: per-nrecon coverage matches",
        "PASS" if nrecon_ok else f"FAIL ({'; '.join(nrecon_diffs)})",
    ))

    # --- Check 7: Verify reported covered flag matches recomputed ---
    reported_covered = np.array([g["covered"] for g in eval_pg])
    recomputed_covered = (ev_y >= reported_lo) & (ev_y <= reported_hi)
    covered_match = np.array_equal(reported_covered, recomputed_covered)
    checks.append((
        f"[{config} seed={seed}] Check 7: per-graph covered flags match",
        "PASS" if covered_match else "FAIL (covered flags disagree)",
    ))

    # --- Check 8: Cross-check pooled coverage = mean(covered) ---
    mean_covered = float(np.mean(reported_covered))
    cross_check = abs(mean_covered - seed_result["pooled_coverage"])
    checks.append((
        f"[{config} seed={seed}] Check 8: mean(covered) == pooled_coverage",
        "PASS" if cross_check < 1e-12 else f"FAIL (diff {cross_check:.2e})",
    ))

    return checks


def main() -> None:
    print("=== Stage 7 Independent Verification ===")
    print(f"Reading: {RESULTS}")
    print()

    with open(RESULTS) as f:
        data = json.load(f)

    all_checks = []
    for config in data["configs"]:
        for seed_result in data["per_seed"][config]:
            checks = verify_seed(seed_result)
            all_checks.extend(checks)

    # Print results
    n_pass = 0
    n_fail = 0
    for check, status in all_checks:
        marker = "✓" if status == "PASS" else "✗"
        print(f"  {marker} {check}: {status}")
        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

    print()
    print(f"Results: {n_pass} passed, {n_fail} failed out of {n_pass + n_fail} checks.")
    if n_fail == 0:
        print("ALL CHECKS PASSED — Stage 7 numbers are independently verified.")
    else:
        print("SOME CHECKS FAILED — investigate before trusting the reported numbers.")
        sys.exit(1)


if __name__ == "__main__":
    main()
