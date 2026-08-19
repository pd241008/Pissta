"""
Stage 6B — Vanilla DAG-GNN Baseline

Trains and evaluates a GraphSAGE-based GNN on the Stage 6A dataset.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage6b.dataset import create_dataloaders, GraphDataset
from stage6b.model import VanillaDAGGNNSage
from stage6b.train import train_model
from stage6b.eval import evaluate_model, compute_analytical_baseline_metrics, compare_model_vs_analytical


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_seed(
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    feature_stats: Dict,
    target_stats: Dict,
    device: torch.device,
    checkpoint_dir: Path,
) -> Dict:
    """Run training and evaluation for a single seed."""
    print(f"\n{'='*60}")
    print(f"Running seed {seed}")
    print(f"{'='*60}")

    set_seed(seed)

    # Create model
    model = VanillaDAGGNNSage(
        num_node_features=3,
        hidden_dim=64,
        num_layers=3,
        dropout=0.15,
        num_outputs=2,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Train
    checkpoint_path = str(checkpoint_dir / f"best_model_seed{seed}.pt")
    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        max_epochs=200,
        patience=20,
        checkpoint_path=checkpoint_path,
    )

    print(f"Best epoch: {train_result['best_epoch']}")
    print(f"Best val loss: {train_result['best_val_loss']:.6f}")
    print(f"Train time: {train_result['train_time']:.1f}s")

    # Load best model for evaluation
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_metrics = evaluate_model(model, test_loader, device, target_stats)
    print(f"Test graphs: {test_metrics['n_graphs']}")
    print(f"Mean delay MAE: {test_metrics['mean_mae']:.4f}")
    print(f"Mean delay relative error: {test_metrics['mean_relative']:.2%}")
    print(f"Std delay MAE: {test_metrics['std_mae']:.4f}")
    print(f"Std delay relative error: {test_metrics['std_relative']:.2%}")
    print(f"Avg inference time: {test_metrics['avg_inference_time_ms']:.2f} ms")

    # Analytical baseline
    print("\n--- Analytical SSTA Baseline ---")
    test_dataset = GraphDataset(split="test")
    analytical_baseline = test_dataset.get_analytical_baseline()
    analytical_metrics = compute_analytical_baseline_metrics(analytical_baseline)
    print(f"Analytical mean MAE: {analytical_metrics['mean_mae']:.4f}")
    print(f"Analytical mean relative: {analytical_metrics['mean_relative']:.2%}")
    print(f"Analytical std MAE: {analytical_metrics['std_mae']:.4f}")
    print(f"Analytical std relative: {analytical_metrics['std_relative']:.2%}")

    # Comparison
    comparison = compare_model_vs_analytical(test_metrics, analytical_metrics)
    print(f"\nGNN beats analytical on mean: {comparison['beats_analytical_mean']}")
    print(f"GNN beats analytical on std: {comparison['beats_analytical_std']}")
    print(f"Mean MAE ratio (GNN/Analytical): {comparison['mean_mae_ratio']:.2f}x")
    print(f"Std MAE ratio (GNN/Analytical): {comparison['std_mae_ratio']:.2f}x")

    # Nrecon breakdown
    print("\n--- Error by Reconvergence Count ---")
    for nrecon in sorted(test_metrics["nrecon_breakdown"].keys()):
        stats = test_metrics["nrecon_breakdown"][nrecon]
        print(f"  nrecon={nrecon} (n={stats['count']}): mean_mae={stats['mean_mae']:.4f}, "
              f"mean_rel={stats['mean_relative']:.2%}, std_mae={stats['std_mae']:.4f}, "
              f"std_rel={stats['std_relative']:.2%}")

    return {
        "seed": seed,
        "train_result": train_result,
        "test_metrics": test_metrics,
        "analytical_metrics": analytical_metrics,
        "comparison": comparison,
        "n_params": n_params,
    }


def run_trivial_baseline(test_loader: DataLoader, target_stats: Dict) -> Dict:
    """Compute trivial baseline: predict training set mean for every graph."""
    # Get training set mean targets
    train_dataset = GraphDataset(split="train")
    all_mean = []
    all_std = []
    for gid in train_dataset.graph_ids:
        entry = train_dataset.dataset[gid]
        all_mean.append(entry["mc_labels"]["mean"])
        all_std.append(entry["mc_labels"]["std"])

    train_mean_mean = np.mean(all_mean)
    train_mean_std = np.mean(all_std)

    # Evaluate on test set
    target_means = []
    target_stds = []
    for batch in test_loader:
        target_means.append(batch.y[:, 0].numpy() * target_stats["mean_std"] + target_stats["mean_mean"])
        target_stds.append(batch.y[:, 1].numpy() * target_stats["std_std"] + target_stats["std_mean"])

    target_means = np.concatenate(target_means)
    target_stds = np.concatenate(target_stds)

    mean_mae = np.mean(np.abs(train_mean_mean - target_means))
    mean_relative = np.mean(np.abs(train_mean_mean - target_means) / np.maximum(np.abs(target_means), 1e-9))
    std_mae = np.mean(np.abs(train_mean_std - target_stds))
    std_relative = np.mean(np.abs(train_mean_std - target_stds) / np.maximum(np.abs(target_stds), 1e-9))

    return {
        "train_mean_mean": float(train_mean_mean),
        "train_mean_std": float(train_mean_std),
        "mean_mae": float(mean_mae),
        "mean_relative": float(mean_relative),
        "std_mae": float(std_mae),
        "std_relative": float(std_relative),
    }


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = Path("stage6b/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, feature_stats, target_stats = create_dataloaders(
        data_dir="data_generation/data",
        batch_size=32,
        use_undirected=False,
    )
    print(f"Train: {len(train_loader.dataset)} graphs, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print(f"Feature stats: {feature_stats}")
    print(f"Target stats: {target_stats}")

    # Trivial baseline
    print("\n" + "="*60)
    print("Trivial Baseline (predict training set mean)")
    print("="*60)
    trivial = run_trivial_baseline(test_loader, target_stats)
    print(f"Mean MAE: {trivial['mean_mae']:.4f}, relative: {trivial['mean_relative']:.2%}")
    print(f"Std MAE: {trivial['std_mae']:.4f}, relative: {trivial['std_relative']:.2%}")

    # Train 3 seeds
    seeds = [42, 123, 999]
    results = []
    for seed in seeds:
        result = run_single_seed(
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            feature_stats=feature_stats,
            target_stats=target_stats,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        results.append(result)

    # Aggregate results across seeds
    print("\n" + "="*60)
    print("3-Seed Stability Summary")
    print("="*60)

    mean_maes = [r["test_metrics"]["mean_mae"] for r in results]
    mean_relatives = [r["test_metrics"]["mean_relative"] for r in results]
    std_maes = [r["test_metrics"]["std_mae"] for r in results]
    std_relatives = [r["test_metrics"]["std_relative"] for r in results]

    print(f"Mean delay MAE: {np.mean(mean_maes):.4f} ± {np.std(mean_maes, ddof=1):.4f}")
    print(f"Mean delay relative error: {np.mean(mean_relatives):.2%} ± {np.std(mean_relatives, ddof=1):.2%}")
    print(f"Std delay MAE: {np.mean(std_maes):.4f} ± {np.std(std_maes, ddof=1):.4f}")
    print(f"Std delay relative error: {np.mean(std_relatives):.2%} ± {np.std(std_relatives, ddof=1):.2%}")

    # Compare to analytical baseline (using first seed's analytical metrics)
    analytical = results[0]["analytical_metrics"]
    comparison = results[0]["comparison"]

    print(f"\nAnalytical baseline MAE: mean={analytical['mean_mae']:.4f}, std={analytical['std_mae']:.4f}")
    print(f"GNN beats analytical on mean: {comparison['beats_analytical_mean']}")
    print(f"GNN beats analytical on std: {comparison['beats_analytical_std']}")

    # Check overfitting
    print("\n" + "="*60)
    print("Overfitting Check")
    print("="*60)
    for r in results:
        train_result = r.get("train_result", {})
        history = train_result.get("history", {})
        train_losses = history.get("train_loss", [])
        val_losses = history.get("val_loss", [])
        if train_losses and val_losses:
            best_train = min(train_losses)
            best_val = min(val_losses)
            print(f"Seed {r['seed']}: best_train={best_train:.6f}, best_val={best_val:.6f}, ratio={best_val/max(best_train, 1e-9):.2f}")
        else:
            print(f"Seed {r['seed']}: no history available")

    # Save results
    results_dir = Path("stage6b/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable_results = []
    for r in results:
        train_result = r.get("train_result", {})
        history = train_result.get("history", {})
        serializable_history = {}
        for k, v in history.items():
            if isinstance(v, list):
                serializable_history[k] = [float(x) for x in v]
            else:
                serializable_history[k] = float(v) if hasattr(v, 'item') else v
        
        serializable_results.append({
            "seed": r["seed"],
            "n_params": r["n_params"],
            "best_epoch": train_result.get("best_epoch", 0),
            "best_val_loss": float(train_result.get("best_val_loss", 0)) if hasattr(train_result.get("best_val_loss", 0), 'item') else train_result.get("best_val_loss", 0),
            "train_time": float(train_result.get("train_time", 0)) if hasattr(train_result.get("train_time", 0), 'item') else train_result.get("train_time", 0),
            "history": serializable_history,
            "test_metrics": r["test_metrics"],
            "analytical_metrics": r["analytical_metrics"],
            "comparison": r["comparison"],
        })

    output = {
        "seeds": seeds,
        "results": serializable_results,
        "stability_summary": {
            "mean_mae_mean": float(np.mean(mean_maes)),
            "mean_mae_std": float(np.std(mean_maes, ddof=1)),
            "mean_relative_mean": float(np.mean(mean_relatives)),
            "mean_relative_std": float(np.std(mean_relatives, ddof=1)),
            "std_mae_mean": float(np.mean(std_maes)),
            "std_mae_std": float(np.std(std_maes, ddof=1)),
            "std_relative_mean": float(np.mean(std_relatives)),
            "std_relative_std": float(np.std(std_relatives, ddof=1)),
        },
        "trivial_baseline": trivial,
        "analytical_baseline": analytical,
        "comparison": comparison,
        "feature_stats": {k: float(v) for k, v in feature_stats.items()},
        "target_stats": {k: float(v) for k, v in target_stats.items()},
    }

    with open(results_dir / "stage6b_results.json", "w") as f:
        json.dump(_to_serializable(output), f, indent=2)

    print(f"\nResults saved to {results_dir / 'stage6b_results.json'}")


if __name__ == "__main__":
    main()
