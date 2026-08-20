"""
Stage 6C — Physics-Informed DAG-GNN Baseline

Runs 3-way ablation: Vanilla (6B) → Tier A (node sensitivities) → Tier A+B (node + graph-level analytical)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import create_dataloaders, GraphDataset
from model import VanillaDAGGNNSage, PhysicsInformedDAGGNNSage
from train import train_model
from eval import evaluate_model, compute_analytical_baseline_metrics, compare_model_vs_analytical


def _to_serializable(obj: Any) -> Any:
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


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_batch_shape(batch: Any, batch_size: int, expected_node_features: int) -> None:
    """Verify PyG batching invariants on one batch."""
    assert batch.x.dim() == 2, f"Expected 2D node features, got {batch.x.dim()}D"
    assert batch.x.shape[1] == expected_node_features, f"Expected {expected_node_features} node features, got {batch.x.shape[1]}"
    assert batch.y.dim() == 2, f"Expected 2D targets [num_graphs, 2], got {batch.y.dim()}D {batch.y.shape}"
    assert batch.y.shape[1] == 2, f"Expected 2 target dims, got {batch.y.shape[1]}"
    num_nodes = batch.x.shape[0]
    num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else (batch.batch.max().item() + 1)
    assert num_graphs == batch_size, f"Expected {batch_size} graphs, got {num_graphs}"
    assert batch.edge_index.max().item() < num_nodes, "Edge index exceeds node count"
    assert batch.edge_index.min().item() >= 0, "Negative edge index"


@torch.no_grad()
def eval_mode_train_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute eval-mode loss over the train set for overfitting check."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        batch = batch.to(device)
        kwargs = {"x": batch.x, "edge_index": batch.edge_index, "batch": batch.batch}
        if hasattr(batch, "graph_physics"):
            kwargs["graph_physics"] = batch.graph_physics
        pred = model(**kwargs)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def run_single_seed(
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_dataset: GraphDataset,
    feature_stats: Dict,
    target_stats: Dict,
    physics_stats: Dict,
    device: torch.device,
    checkpoint_dir: Path,
    config_name: str,
    model_fn: Any,
) -> Dict:
    """Run training and evaluation for a single seed and config."""
    print(f"\n{'='*60}")
    print(f"Running {config_name} — seed {seed}")
    print(f"{'='*60}")

    set_seed(seed)

    # Create model
    if config_name == "vanilla":
        model = VanillaDAGGNNSage(
            num_node_features=3,
            hidden_dim=64,
            num_layers=3,
            dropout=0.15,
            num_outputs=2,
        ).to(device)
    elif config_name == "tier_a":
        model = PhysicsInformedDAGGNNSage(
            tier="a",
            hidden_dim=64,
            num_layers=3,
            dropout=0.15,
            num_outputs=2,
        ).to(device)
    elif config_name == "tier_ab":
        model = PhysicsInformedDAGGNNSage(
            tier="ab",
            hidden_dim=64,
            num_layers=3,
            dropout=0.15,
            num_outputs=2,
        ).to(device)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Train
    checkpoint_path = str(checkpoint_dir / f"best_model_{config_name}_seed{seed}.pt")
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
    test_metrics = evaluate_model(model, test_loader, device, target_stats, dataset=test_loader.dataset)
    print(f"Test graphs: {test_metrics['n_graphs']}")
    print(f"Mean delay MAE: {test_metrics['mean_mae']:.4f}")
    print(f"Mean delay relative error: {test_metrics['mean_relative']:.2%}")
    print(f"Std delay MAE: {test_metrics['std_mae']:.4f}")
    print(f"Std delay relative error: {test_metrics['std_relative']:.2%}")
    print(f"Avg inference time: {test_metrics['avg_inference_time_ms']:.2f} ms")

    # Analytical baseline
    print("\n--- Analytical SSTA Baseline ---")
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
        if stats["count"] >= 5:
            print(f"  nrecon={nrecon} (n={stats['count']}): mean_mae={stats['mean_mae']:.4f}, "
                  f"mean_rel={stats['mean_relative']:.2%}, std_mae={stats['std_mae']:.4f}, "
                  f"std_rel={stats['std_relative']:.2%}")
        else:
            print(f"  nrecon={nrecon} (n={stats['count']}): not-interpretable (n<5)")

    # Eval-mode train loss for overfitting check
    eval_train_loss = eval_mode_train_loss(model, train_loader, device)

    return {
        "seed": seed,
        "config": config_name,
        "train_result": train_result,
        "eval_train_loss": eval_train_loss,
        "test_metrics": test_metrics,
        "analytical_metrics": analytical_metrics,
        "comparison": comparison,
        "n_params": n_params,
    }


def compute_tier_b_leakage(test_dataset: GraphDataset, physics_stats: Dict) -> Dict:
    """Compute raw correlation between analytical features and MC labels (Tier B leakage check)."""
    sink_means = []
    sink_stds = []
    mc_means = []
    mc_stds = []

    for gid in test_dataset.graph_ids:
        entry = test_dataset.dataset[gid]
        ana = entry["physics_features"]["analytical_ssta"]
        mc = entry["mc_labels"]

        sink_means.append((ana["sink_mean"] - physics_stats["sink_mean_mean"]) / physics_stats["sink_mean_std"])
        sink_stds.append((ana["sink_std"] - physics_stats["sink_std_mean"]) / physics_stats["sink_std_std"])
        mc_means.append((mc["mean"] - physics_stats["sink_mean_mean"]) / physics_stats["sink_mean_std"])
        mc_stds.append((mc["std"] - physics_stats["sink_std_mean"]) / physics_stats["sink_std_std"])

    sink_means = np.array(sink_means)
    sink_stds = np.array(sink_stds)
    mc_means = np.array(mc_means)
    mc_stds = np.array(mc_stds)

    mean_corr = float(np.corrcoef(sink_means, mc_means)[0, 1])
    std_corr = float(np.corrcoef(sink_stds, mc_stds)[0, 1])

    mean_mae = float(np.mean(np.abs(sink_means - mc_means)))
    std_mae = float(np.mean(np.abs(sink_stds - mc_stds)))

    return {
        "mean_correlation": mean_corr,
        "std_correlation": std_corr,
        "mean_mae_normalized": mean_mae,
        "std_mae_normalized": std_mae,
    }


def main():
    script_dir = Path(__file__).resolve().parent
    checkpoint_dir = script_dir / "checkpoints"
    results_dir = script_dir / "results"
    data_dir = script_dir.parent / "data_generation" / "data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Configurations to run
    configs = ["vanilla", "tier_a", "tier_ab"]
    seeds = [42, 123, 999]

    # Run all configurations
    all_results = {}
    for config in configs:
        print(f"\n{'#'*60}")
        print(f"# Configuration: {config}")
        print(f"{'#'*60}")

        train_loader, val_loader, test_loader, feature_stats, target_stats, physics_stats = create_dataloaders(
            data_dir=data_dir,
            batch_size=32,
            physics_mode=config if config != "vanilla" else "vanilla",
        )

        # Batch shape check
        sample_batch = next(iter(train_loader))
        expected_features = 3 if config == "vanilla" else 6
        check_batch_shape(sample_batch, train_loader.batch_size, expected_features)
        print(f"Batch OK: x={sample_batch.x.shape}, y={sample_batch.y.shape}, "
              f"num_graphs={sample_batch.num_graphs if hasattr(sample_batch, 'num_graphs') else sample_batch.batch.max().item()+1}")

        test_dataset = GraphDataset(split="test", physics_mode=config if config != "vanilla" else "vanilla")
        train_dataset = GraphDataset(split="train", physics_mode=config if config != "vanilla" else "vanilla")

        config_results = []
        for seed in seeds:
            result = run_single_seed(
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                test_dataset=test_dataset,
                feature_stats=feature_stats,
                target_stats=target_stats,
                physics_stats=physics_stats,
                device=device,
                checkpoint_dir=checkpoint_dir,
                config_name=config,
                model_fn=None,
            )
            config_results.append(result)

        all_results[config] = config_results

    # Aggregate results
    print("\n" + "="*60)
    print("3-Way Ablation Summary")
    print("="*60)

    # Analytical baseline (same for all configs)
    analytical_baseline = test_dataset.get_analytical_baseline()
    analytical_metrics = compute_analytical_baseline_metrics(analytical_baseline)

    # Compute vanilla stats (frozen baseline)
    vanilla_results = all_results["vanilla"]
    vanilla_mean_maes = [r["test_metrics"]["mean_mae"] for r in vanilla_results]
    vanilla_mean_relatives = [r["test_metrics"]["mean_relative"] for r in vanilla_results]
    vanilla_std_maes = [r["test_metrics"]["std_mae"] for r in vanilla_results]
    vanilla_std_relatives = [r["test_metrics"]["std_relative"] for r in vanilla_results]

    # Compute Tier A stats
    tier_a_results = all_results["tier_a"]
    tier_a_mean_maes = [r["test_metrics"]["mean_mae"] for r in tier_a_results]
    tier_a_mean_relatives = [r["test_metrics"]["mean_relative"] for r in tier_a_results]
    tier_a_std_maes = [r["test_metrics"]["std_mae"] for r in tier_a_results]
    tier_a_std_relatives = [r["test_metrics"]["std_relative"] for r in tier_a_results]

    # Compute Tier A+B stats
    tier_ab_results = all_results["tier_ab"]
    tier_ab_mean_maes = [r["test_metrics"]["mean_mae"] for r in tier_ab_results]
    tier_ab_mean_relatives = [r["test_metrics"]["mean_relative"] for r in tier_ab_results]
    tier_ab_std_maes = [r["test_metrics"]["std_mae"] for r in tier_ab_results]
    tier_ab_std_relatives = [r["test_metrics"]["std_relative"] for r in tier_ab_results]

    # Print ablation table
    print(f"\n{'Model':<35} {'Mean MAE':<12} {'Mean Rel':<12} {'Std MAE':<12} {'Std Rel':<12} {'Params':<10}")
    print("-" * 95)
    print(f"{'Analytical SSTA':<35} {analytical_metrics['mean_mae']:.4f}{'':>8} {analytical_metrics['mean_relative']:.2%}{'':>9} {analytical_metrics['std_mae']:.4f}{'':>8} {analytical_metrics['std_relative']:.2%}{'':>9} {'—':>10}")
    print(f"{'Vanilla DAG-GNN (6B, locked)':<35} {np.mean(vanilla_mean_maes):.4f}±{np.std(vanilla_mean_maes, ddof=1):.4f} {np.mean(vanilla_mean_relatives):.2%}±{np.std(vanilla_mean_relatives, ddof=1):.2%} {np.mean(vanilla_std_maes):.4f}±{np.std(vanilla_std_maes, ddof=1):.4f} {np.mean(vanilla_std_relatives):.2%}±{np.std(vanilla_std_relatives, ddof=1):.2%} {vanilla_results[0]['n_params']:>10,}")
    print(f"{'Physics-informed (Tier A)':<35} {np.mean(tier_a_mean_maes):.4f}±{np.std(tier_a_mean_maes, ddof=1):.4f} {np.mean(tier_a_mean_relatives):.2%}±{np.std(tier_a_mean_relatives, ddof=1):.2%} {np.mean(tier_a_std_maes):.4f}±{np.std(tier_a_std_maes, ddof=1):.4f} {np.mean(tier_a_std_relatives):.2%}±{np.std(tier_a_std_relatives, ddof=1):.2%} {tier_a_results[0]['n_params']:>10,}")
    print(f"{'Physics-informed (Tier A+B)':<35} {np.mean(tier_ab_mean_maes):.4f}±{np.std(tier_ab_mean_maes, ddof=1):.4f} {np.mean(tier_ab_mean_relatives):.2%}±{np.std(tier_ab_mean_relatives, ddof=1):.2%} {np.mean(tier_ab_std_maes):.4f}±{np.std(tier_ab_std_maes, ddof=1):.4f} {np.mean(tier_ab_std_relatives):.2%}±{np.std(tier_ab_std_relatives, ddof=1):.2%} {tier_ab_results[0]['n_params']:>10,}")

    # Statistical significance check
    print("\n--- Statistical Significance vs Vanilla ---")
    for metric_name, vanilla_vals, tier_a_vals, tier_ab_vals in [
        ("mean_mae", vanilla_mean_maes, tier_a_mean_maes, tier_ab_mean_maes),
        ("std_mae", vanilla_std_maes, tier_a_std_maes, tier_ab_std_maes),
    ]:
        vanilla_mean = np.mean(vanilla_vals)
        vanilla_std = np.std(vanilla_vals, ddof=1)
        tier_a_mean = np.mean(tier_a_vals)
        tier_a_std = np.std(tier_a_vals, ddof=1)
        tier_ab_mean = np.mean(tier_ab_vals)
        tier_ab_std = np.std(tier_ab_vals, ddof=1)

        # Simple z-test for difference in means
        pooled_std_a = np.sqrt((vanilla_std**2 + tier_a_std**2) / 3)
        z_a = (vanilla_mean - tier_a_mean) / pooled_std_a if pooled_std_a > 0 else 0
        pooled_std_ab = np.sqrt((vanilla_std**2 + tier_ab_std**2) / 3)
        z_ab = (vanilla_mean - tier_ab_mean) / pooled_std_ab if pooled_std_ab > 0 else 0

        print(f"  {metric_name}:")
        print(f"    Tier A: {tier_a_mean:.4f} vs vanilla {vanilla_mean:.4f}, z={z_a:.2f} ({'significant' if abs(z_a) > 1.96 else 'not significant'})")
        print(f"    Tier A+B: {tier_ab_mean:.4f} vs vanilla {vanilla_mean:.4f}, z={z_ab:.2f} ({'significant' if abs(z_ab) > 1.96 else 'not significant'})")

    # Tier B leakage check
    print("\n--- Tier B Leakage Check ---")
    leakage = compute_tier_b_leakage(test_dataset, physics_stats)
    print(f"  Analytical sink_mean correlation with MC mean: {leakage['mean_correlation']:.4f}")
    print(f"  Analytical sink_std correlation with MC std: {leakage['std_correlation']:.4f}")
    print(f"  Analytical mean MAE (normalized): {leakage['mean_mae_normalized']:.4f}")
    print(f"  Analytical std MAE (normalized): {leakage['std_mae_normalized']:.4f}")

    tier_ab_mean_mae_normalized = np.mean([r["test_metrics"]["mean_mae"] for r in tier_ab_results])
    tier_ab_std_mae_normalized = np.mean([r["test_metrics"]["std_mae"] for r in tier_ab_results])
    print(f"  Tier A+B mean MAE (normalized): {tier_ab_mean_mae_normalized:.4f}")
    print(f"  Tier A+B std MAE (normalized): {tier_ab_std_mae_normalized:.4f}")

    if leakage["mean_correlation"] > 0.9:
        print("  WARNING: Analytical sink_mean already highly correlated with MC mean (>0.9).")
        print("  Tier B gains may reflect 'GNN lightly perturbs analytical' rather than learned physics.")
    if leakage["std_correlation"] > 0.9:
        print("  WARNING: Analytical sink_std already highly correlated with MC std (>0.9).")
        print("  Tier B gains may reflect 'GNN lightly perturbs analytical' rather than learned physics.")

    # Overfitting check for all configs
    print("\n--- Overfitting Check (eval-mode train pass) ---")
    for config in configs:
        config_results = all_results[config]
        for r in config_results:
            train_result = r.get("train_result", {})
            history = train_result.get("history", {})
            val_losses = history.get("val_loss", [])
            eval_train = r.get("eval_train_loss", float("nan"))
            best_val = min(val_losses) if val_losses else float("nan")
            ratio = best_val / max(eval_train, 1e-9) if not np.isnan(eval_train) else float("nan")
            print(f"  {config} seed {r['seed']}: eval_train={eval_train:.6f}, best_val={best_val:.6f}, ratio={ratio:.2f}")

    # nrecon breakdown comparison
    print("\n--- nrecon Breakdown Comparison ---")
    for config in configs:
        config_results = all_results[config]
        print(f"\n  {config}:")
        for r in config_results:
            print(f"    seed {r['seed']}:")
            for nrecon in sorted(r["test_metrics"]["nrecon_breakdown"].keys()):
                stats = r["test_metrics"]["nrecon_breakdown"][nrecon]
                if stats["count"] >= 5:
                    print(f"      nrecon={nrecon} (n={stats['count']}): mean_mae={stats['mean_mae']:.4f}, "
                          f"mean_rel={stats['mean_relative']:.2%}, std_mae={stats['std_mae']:.4f}, "
                          f"std_rel={stats['std_relative']:.2%}")
                else:
                    print(f"      nrecon={nrecon} (n={stats['count']}): not-interpretable (n<5)")

    # Build output JSON
    output = {
        "configs": configs,
        "seeds": seeds,
        "results": {
            config: [
                {
                    "seed": r["seed"],
                    "config": r["config"],
                    "n_params": r["n_params"],
                    "best_epoch": r["train_result"]["best_epoch"],
                    "best_val_loss": float(r["train_result"]["best_val_loss"]),
                    "train_time": float(r["train_result"]["train_time"]),
                    "eval_train_loss": float(r["eval_train_loss"]),
                    "test_metrics": r["test_metrics"],
                    "analytical_metrics": r["analytical_metrics"],
                    "comparison": r["comparison"],
                }
                for r in results
            ]
            for config, results in all_results.items()
        },
        "stability_summary": {
            config: {
                "mean_mae_mean": float(np.mean([r["test_metrics"]["mean_mae"] for r in results])),
                "mean_mae_std": float(np.std([r["test_metrics"]["mean_mae"] for r in results], ddof=1)),
                "mean_relative_mean": float(np.mean([r["test_metrics"]["mean_relative"] for r in results])),
                "mean_relative_std": float(np.std([r["test_metrics"]["mean_relative"] for r in results], ddof=1)),
                "std_mae_mean": float(np.mean([r["test_metrics"]["std_mae"] for r in results])),
                "std_mae_std": float(np.std([r["test_metrics"]["std_mae"] for r in results], ddof=1)),
                "std_relative_mean": float(np.mean([r["test_metrics"]["std_relative"] for r in results])),
                "std_relative_std": float(np.std([r["test_metrics"]["std_relative"] for r in results], ddof=1)),
            }
            for config, results in all_results.items()
        },
        "analytical_baseline": analytical_metrics,
        "tier_b_leakage_check": leakage,
        "feature_stats": {k: float(v) for k, v in feature_stats.items()},
        "target_stats": {k: float(v) for k, v in target_stats.items()},
        "physics_stats": {k: float(v) for k, v in physics_stats.items()},
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_geometric": __import__("torch_geometric").__version__,
        },
        "protocol": {
            "loss": "MSE (mean reduction)",
            "gradient_clip": "max_norm=1.0",
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "early_stopping_patience": 20,
            "dropout": 0.15,
            "hidden_dim": 64,
            "num_layers": 3,
            "readout": "mean_pool",
            "edge_type": "directed (successors)",
            "target_normalization": "train-set mean/std",
            "feature_normalization": "train-set mean/std",
            "physics_normalization": "train-set mean/std",
        },
    }

    output_path = results_dir / "stage6c_results.json"
    with open(output_path, "w") as f:
        json.dump(_to_serializable(output), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
