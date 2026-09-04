"""
Step 1 — Calibration split integrity check for Stage 7.

Verifies that the cal/eval carve from the test split:
  (a) has no overlap with train/val (no contamination of already-reported MAE)
  (b) has no overlap between cal and eval (calibration and test are disjoint)
  (c) is deterministic (same carve on re-run)
  (d) reports sizes and nrecon distribution for auditability

Run before any conformal work to confirm the split is clean.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "data_generation" / "data"
SPLITS_JSON = DATA_DIR / "splits.json"
STAGE7_SPLITS = REPO / "gnn_baseline" / "results" / "splits_stage7.json"


def _nrecon_of(dataset: dict, gid: str) -> int:
    return len(dataset[gid]["graph"]["reconvergence_points"])


def main() -> None:
    # Load splits
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    with open(STAGE7_SPLITS) as f:
        stage7 = json.load(f)

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])
    cal_ids = set(stage7["cal_ids"])
    eval_ids = set(stage7["eval_ids"])

    # Load dataset for nrecon distribution
    with open(DATA_DIR / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    # Check (a): cal/eval disjoint from train/val
    cal_train_overlap = cal_ids & train_ids
    cal_val_overlap = cal_ids & val_ids
    eval_train_overlap = eval_ids & train_ids
    eval_val_overlap = eval_ids & val_ids

    print("=== Stage 7 Calibration Split Integrity ===")
    print(f"carve_seed: {stage7['carve_seed']}")
    print(f"n_cal: {stage7['n_cal']}, n_eval: {stage7['n_eval']}")
    print()

    # Sizes
    print(f"Original splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Stage 7 carve:   cal={len(cal_ids)}, eval={len(eval_ids)}")
    print(f"cal + eval = {len(cal_ids) + len(eval_ids)}, test = {len(test_ids)}")
    print()

    # Overlap checks
    ok = True
    checks = [
        ("cal ∩ train", cal_train_overlap),
        ("cal ∩ val", cal_val_overlap),
        ("eval ∩ train", eval_train_overlap),
        ("eval ∩ val", eval_val_overlap),
    ]
    for label, overlap in checks:
        status = "PASS (empty)" if len(overlap) == 0 else f"FAIL ({len(overlap)} overlap)"
        if len(overlap) > 0:
            ok = False
        print(f"  {label}: {status}")

    # Check (b): cal and eval disjoint
    cal_eval_overlap = cal_ids & eval_ids
    status = "PASS (empty)" if len(cal_eval_overlap) == 0 else f"FAIL ({len(cal_eval_overlap)} overlap)"
    if len(cal_eval_overlap) > 0:
        ok = False
    print(f"  cal ∩ eval: {status}")
    print()

    # Check (c): cal + eval = test (complete coverage of test split)
    cal_plus_eval = cal_ids | eval_ids
    missing_from_test = cal_plus_eval - test_ids
    extra_in_test = test_ids - cal_plus_eval
    if len(missing_from_test) == 0 and len(extra_in_test) == 0:
        print("  cal ∪ eval == test: PASS (exact match)")
    else:
        ok = False
        if missing_from_test:
            print(f"  cal ∪ eval == test: FAIL ({len(missing_from_test)} IDs in cal/eval but not in test)")
        if extra_in_test:
            print(f"  cal ∪ eval == test: FAIL ({len(extra_in_test)} IDs in test but not in cal/eval)")
    print()

    # Check (d): nrecon distribution per split
    print("=== nrecon distribution ===")
    splits_nrecon = {}
    for label, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids),
                        ("cal", cal_ids), ("eval", eval_ids)]:
        dist = {}
        for gid in ids:
            nrecon = _nrecon_of(dataset, gid)
            dist[nrecon] = dist.get(nrecon, 0) + 1
        splits_nrecon[label] = dist

    # Print table
    all_nrecon = sorted(set(n for d in splits_nrecon.values() for n in d))
    header = f"{'nrecon':>6}"
    for label in ["train", "val", "test", "cal", "eval"]:
        header += f" {label:>8}"
    print(header)
    print("-" * len(header))
    for n in all_nrecon:
        row = f"{n:>6}"
        for label in ["train", "val", "test", "cal", "eval"]:
            count = splits_nrecon[label].get(n, 0)
            row += f" {count:>8}"
        print(row)
    row = f"{'total':>6}"
    for label in ["train", "val", "test", "cal", "eval"]:
        row += f" {sum(splits_nrecon[label].values()):>8}"
    print(row)
    print()

    # Determinism check: re-run carve and compare
    print("=== Determinism check ===")
    sys.path.insert(0, str(REPO / "gnn_baseline"))
    from run_stage7 import carve_cal_eval
    cal_new, eval_new = carve_cal_eval(DATA_DIR, stage7["carve_seed"])
    cal_new_set = set(cal_new)
    eval_new_set = set(eval_new)
    if cal_new_set == cal_ids and eval_new_set == eval_ids:
        print("  Re-run carve matches saved split: PASS")
    else:
        ok = False
        print("  Re-run carve matches saved split: FAIL (split is not deterministic or was modified)")
        print(f"    cal diff: {len(cal_new_set - cal_ids)} new, {len(cal_ids - cal_new_set)} missing")
        print(f"    eval diff: {len(eval_new_set - eval_ids)} new, {len(eval_ids - eval_new_set)} missing")

    print()
    if ok:
        print("ALL CHECKS PASSED — calibration split is clean.")
    else:
        print("SOME CHECKS FAILED — see above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
