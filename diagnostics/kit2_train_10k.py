"""
Kit 2 step 3 — Fresh training at scale on pissta-10k.

Trains vanilla and maxbias_cm backbones on the pissta-10k TRAIN split
(6,997 graphs) with the identical frozen Stage 6B/7 protocol:
  * GraphSAGE skeleton, hidden 64 (vanilla) / 54 (maxbias_cm, capacity-matched),
    3 layers, dropout 0.15, Adam lr=1e-3, MSE, grad-clip 1.0,
    early stopping patience 20, max 200 epochs, batch 32
  * physics_mode="vanilla" (3-dim node features: load_ff, x, y) — same as every
    reported 6B/6C/7 result; the (D1-defected) stored physics_features are NOT
    consumed in this mode
  * normalization from the 10k train split only
  * seeds 42/123/999 (the Stage 7 seed set), deterministic kernels

Evaluates on the pissta-10k TEST split (1,505 graphs) and exports per-graph
records in the stage6c_results_*.json format (test_metrics.per_graph).

Checkpoint-skip: if the per-(config,seed) result JSON exists, the run is
skipped, so the driver is resumable.

Run:  python diagnostics/kit2_train_10k.py [--configs vanilla maxbias_cm]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gnn_baseline"))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from dataset import GraphDataset  # noqa: E402
from eval import evaluate_model  # noqa: E402
from train import train_model  # noqa: E402
from run_stage6c import set_seed, _to_serializable  # noqa: E402
from run_stage7 import build_model  # noqa: E402

DATA_DIR = REPO / "zenodo" / "data" / "pissta-10k"
CHECKPOINT_DIR = REPO / "gnn_baseline" / "checkpoints"
RESULTS_DIR = REPO / "gnn_baseline" / "results"

SEEDS = [42, 123, 999]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["vanilla", "maxbias_cm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data:   {DATA_DIR}")

    # Normalization from the 10k train split only (frozen protocol).
    train_dataset = GraphDataset(data_dir=DATA_DIR, split="train", physics_mode="vanilla")
    val_dataset = GraphDataset(data_dir=DATA_DIR, split="val", physics_mode="vanilla")
    test_dataset = GraphDataset(data_dir=DATA_DIR, split="test", physics_mode="vanilla")
    feature_stats, target_stats = train_dataset._compute_normalization()
    physics_stats = {}  # physics_mode="vanilla" consumes none

    print(f"train/val/test: {len(train_dataset.graph_ids)}/"
          f"{len(val_dataset.graph_ids)}/{len(test_dataset.graph_ids)}")

    train_loader = DataLoader(
        train_dataset.get_data(feature_stats, target_stats, physics_stats),
        batch_size=32, shuffle=True)
    val_loader = DataLoader(
        val_dataset.get_data(feature_stats, target_stats, physics_stats),
        batch_size=32, shuffle=False)
    test_data = test_dataset.get_data(feature_stats, target_stats, physics_stats)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    for config in args.configs:
        for seed in args.seeds:
            tag = f"{config}_seed{seed}_10k"
            out_path = RESULTS_DIR / f"kit2_10k_results_{config}_seed{seed}.json"
            if out_path.exists():
                print(f"[skip] {tag}: {out_path.name} exists")
                continue

            print(f"\n===== {config} seed={seed} (pissta-10k fresh training) =====")
            set_seed(seed)
            model = build_model(config, device)
            ckpt_path = str(CHECKPOINT_DIR / f"best_model_{tag}.pt")

            train_result = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                device=device, lr=1e-3, max_epochs=200, patience=20,
                checkpoint_path=ckpt_path,
            )
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))

            test_metrics = evaluate_model(model, test_loader, device, target_stats,
                                          dataset=test_loader.dataset)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            result = {
                "status": "Kit 2: fresh training on pissta-10k train split (frozen 6B/7 protocol)",
                "data_dir": str(DATA_DIR),
                "config": config,
                "seed": seed,
                "n_params": n_params,
                "best_epoch": train_result["best_epoch"],
                "best_val_loss": train_result["best_val_loss"],
                "train_time_s": train_result["train_time"],
                "checkpoint": ckpt_path,
                "test_metrics": test_metrics,
            }
            with open(out_path, "w") as f:
                json.dump(_to_serializable(result), f, indent=2)
            print(f"  best_epoch={train_result['best_epoch']} "
                  f"best_val={train_result['best_val_loss']:.6f} "
                  f"train={train_result['train_time']:.0f}s")
            print(f"  test mean_mae={test_metrics['mean_mae']:.4f} "
                  f"(n={test_metrics['n_graphs']})")
            print(f"  saved {out_path.name}")

    print("\nKit 2 fresh training complete.")


if __name__ == "__main__":
    main()
