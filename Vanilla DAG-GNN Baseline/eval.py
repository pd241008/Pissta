"""
Stage 6B — Evaluation utilities for vanilla DAG-GNN baseline.

Computes MAE, relative error, breakdown by reconvergence count,
and comparison against analytical SSTA baseline.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch

from model import VanillaDAGGNNSage


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_stats: Dict[str, float],
) -> Dict:
    """Evaluate model on a dataset and compute all metrics."""
    model.eval()

    all_preds = []
    all_targets = []
    all_gids = []
    all_nrecons = []
    all_original_mean = []
    all_original_std = []
    inference_times = []

    for batch in loader:
        batch = batch.to(device)

        # Time inference
        start = time.time()
        pred = model(batch.x, batch.edge_index, batch.batch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_times.append(time.time() - start)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

        # Store graph-level info
        num_graphs = batch.batch.max().item() + 1 if batch.batch is not None else 1
        
        # Handle graph_id - may be list of strings or tensor
        if hasattr(batch, 'graph_id'):
            gid_data = batch.graph_id
            if isinstance(gid_data, list):
                gids = gid_data
            elif gid_data.dim() == 0:
                gids = [str(gid_data.item())] * num_graphs
            else:
                gids = [str(g.item()) for g in gid_data]
        else:
            gids = [""] * num_graphs
            
        # Handle nrecon - may be tensor or list
        if hasattr(batch, 'nrecon'):
            nrecon_data = batch.nrecon
            if isinstance(nrecon_data, list):
                nrecons = nrecon_data
            elif nrecon_data.dim() == 0:
                nrecons = [int(nrecon_data.item())] * num_graphs
            else:
                nrecons = [int(n.item()) for n in nrecon_data]
        else:
            nrecons = [0] * num_graphs
            
        # Handle original_mean and original_std similarly
        if hasattr(batch, 'original_mean'):
            mean_data = batch.original_mean
            if isinstance(mean_data, list):
                orig_means = mean_data
            elif mean_data.dim() == 0:
                orig_means = [float(mean_data.item())] * num_graphs
            else:
                orig_means = [float(m.item()) for m in mean_data]
        else:
            orig_means = [0.0] * num_graphs
            
        if hasattr(batch, 'original_std'):
            std_data = batch.original_std
            if isinstance(std_data, list):
                orig_stds = std_data
            elif std_data.dim() == 0:
                orig_stds = [float(std_data.item())] * num_graphs
            else:
                orig_stds = [float(s.item()) for s in std_data]
        else:
            orig_stds = [0.0] * num_graphs
        
        for i in range(num_graphs):
            all_gids.append(gids[i] if i < len(gids) else "")
            all_nrecons.append(nrecons[i] if i < len(nrecons) else 0)
            all_original_mean.append(orig_means[i] if i < len(orig_means) else 0.0)
            all_original_std.append(orig_stds[i] if i < len(orig_stds) else 0.0)

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # Un-normalize predictions and targets
    pred_mean = preds[:, 0] * target_stats["mean_std"] + target_stats["mean_mean"]
    pred_std = preds[:, 1] * target_stats["std_std"] + target_stats["std_mean"]

    target_mean = targets[:, 0] * target_stats["mean_std"] + target_stats["mean_mean"]
    target_std = targets[:, 1] * target_stats["std_std"] + target_stats["std_mean"]

    # Overall metrics
    mean_mae = np.mean(np.abs(pred_mean - target_mean))
    mean_relative = np.mean(np.abs(pred_mean - target_mean) / np.maximum(np.abs(target_mean), 1e-9))
    std_mae = np.mean(np.abs(pred_std - target_std))
    std_relative = np.mean(np.abs(pred_std - target_std) / np.maximum(np.abs(target_std), 1e-9))

    # Per-graph errors
    per_graph = []
    for i in range(len(all_gids)):
        per_graph.append({
            "graph_id": all_gids[i],
            "nrecon": all_nrecons[i],
            "mc_mean": all_original_mean[i],
            "mc_std": all_original_std[i],
            "pred_mean": float(pred_mean[i]),
            "pred_std": float(pred_std[i]),
            "mean_mae": float(np.abs(pred_mean[i] - target_mean[i])),
            "mean_relative": float(np.abs(pred_mean[i] - target_mean[i]) / max(abs(target_mean[i]), 1e-9)),
            "std_mae": float(np.abs(pred_std[i] - target_std[i])),
            "std_relative": float(np.abs(pred_std[i] - target_std[i]) / max(abs(target_std[i]), 1e-9)),
        })

    # Breakdown by nrecon
    nrecon_breakdown = {}
    for item in per_graph:
        nrecon = item["nrecon"]
        if nrecon not in nrecon_breakdown:
            nrecon_breakdown[nrecon] = {
                "count": 0,
                "mean_mae_sum": 0.0,
                "mean_relative_sum": 0.0,
                "std_mae_sum": 0.0,
                "std_relative_sum": 0.0,
            }
        nrecon_breakdown[nrecon]["count"] += 1
        nrecon_breakdown[nrecon]["mean_mae_sum"] += item["mean_mae"]
        nrecon_breakdown[nrecon]["mean_relative_sum"] += item["mean_relative"]
        nrecon_breakdown[nrecon]["std_mae_sum"] += item["std_mae"]
        nrecon_breakdown[nrecon]["std_relative_sum"] += item["std_relative"]

    for nrecon, stats in nrecon_breakdown.items():
        stats["mean_mae"] = stats["mean_mae_sum"] / stats["count"]
        stats["mean_relative"] = stats["mean_relative_sum"] / stats["count"]
        stats["std_mae"] = stats["std_mae_sum"] / stats["count"]
        stats["std_relative"] = stats["std_relative_sum"] / stats["count"]

    # Runtime
    avg_inference_time_ms = np.mean(inference_times) * 1000 if inference_times else 0.0

    return {
        "mean_mae": float(mean_mae),
        "mean_relative": float(mean_relative),
        "std_mae": float(std_mae),
        "std_relative": float(std_relative),
        "per_graph": per_graph,
        "nrecon_breakdown": nrecon_breakdown,
        "avg_inference_time_ms": float(avg_inference_time_ms),
        "n_graphs": len(all_gids),
    }


def compute_analytical_baseline_metrics(baseline: Dict) -> Dict:
    """Compute aggregate analytical SSTA baseline metrics."""
    mean_maes = [v["mean_mae"] for v in baseline.values()]
    mean_relatives = [v["mean_relative"] for v in baseline.values()]
    std_maes = [v["std_mae"] for v in baseline.values()]
    std_relatives = [v["std_relative"] for v in baseline.values()]

    return {
        "mean_mae": float(np.mean(mean_maes)),
        "mean_relative": float(np.mean(mean_relatives)),
        "std_mae": float(np.mean(std_maes)),
        "std_relative": float(np.mean(std_relatives)),
    }


def compare_model_vs_analytical(
    model_metrics: Dict,
    analytical_metrics: Dict,
) -> Dict:
    """Compare GNN performance against analytical SSTA baseline."""
    return {
        "model_mean_mae": model_metrics["mean_mae"],
        "analytical_mean_mae": analytical_metrics["mean_mae"],
        "mean_mae_ratio": model_metrics["mean_mae"] / max(analytical_metrics["mean_mae"], 1e-9),
        "model_std_mae": model_metrics["std_mae"],
        "analytical_std_mae": analytical_metrics["std_mae"],
        "std_mae_ratio": model_metrics["std_mae"] / max(analytical_metrics["std_mae"], 1e-9),
        "beats_analytical_mean": model_metrics["mean_mae"] < analytical_metrics["mean_mae"],
        "beats_analytical_std": model_metrics["std_mae"] < analytical_metrics["std_mae"],
    }
