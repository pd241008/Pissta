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
from torch_geometric.loader import DataLoader

from model import VanillaDAGGNNSage


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_stats: Dict[str, float],
    dataset: List | None = None,
) -> Dict:
    """Evaluate model on a dataset and compute all metrics."""
    model.eval()

    all_preds = []
    all_targets = []
    inference_times = []

    # Warm-up pass (lazy init, cuBLAS kernels, etc.)
    warmup_loader = DataLoader(loader.dataset, batch_size=min(loader.batch_size, 4))
    for batch in warmup_loader:
        batch = batch.to(device)
        kwargs = {"x": batch.x, "edge_index": batch.edge_index, "batch": batch.batch}
        if hasattr(batch, "graph_physics"):
            kwargs["graph_physics"] = batch.graph_physics
        _ = model(**kwargs)
        break

    for batch in loader:
        batch = batch.to(device)

        # Time inference with perf_counter
        start = time.perf_counter()
        kwargs = {"x": batch.x, "edge_index": batch.edge_index, "batch": batch.batch}
        if hasattr(batch, "graph_physics"):
            kwargs["graph_physics"] = batch.graph_physics
        pred = model(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

        # Per-graph timing
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else (batch.batch.max().item() + 1 if batch.batch is not None else 1)
        inference_times.append(elapsed / batch_size)

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

    # Per-graph errors with metadata from dataset
    per_graph = []
    if dataset is not None:
        for idx, data in enumerate(dataset):
            if idx < len(pred_mean):
                nrecon = data.nrecon if hasattr(data, 'nrecon') else 0
                per_graph.append({
                    "graph_id": data.graph_id if hasattr(data, 'graph_id') else "",
                    "nrecon": nrecon,
                    "mc_mean": float(data.original_mean) if hasattr(data, 'original_mean') else 0.0,
                    "mc_std": float(data.original_std) if hasattr(data, 'original_std') else 0.0,
                    "pred_mean": float(pred_mean[idx]),
                    "pred_std": float(pred_std[idx]),
                    "mean_mae": float(np.abs(pred_mean[idx] - target_mean[idx])),
                    "mean_relative": float(np.abs(pred_mean[idx] - target_mean[idx]) / max(abs(target_mean[idx]), 1e-9)),
                    "std_mae": float(np.abs(pred_std[idx] - target_std[idx])),
                    "std_relative": float(np.abs(pred_std[idx] - target_std[idx]) / max(abs(target_std[idx]), 1e-9)),
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

    # Runtime (per-graph, after warmup)
    avg_inference_time_ms = float(np.mean(inference_times) * 1000) if inference_times else 0.0

    return {
        "mean_mae": float(mean_mae),
        "mean_relative": float(mean_relative),
        "std_mae": float(std_mae),
        "std_relative": float(std_relative),
        "per_graph": per_graph,
        "nrecon_breakdown": nrecon_breakdown,
        "avg_inference_time_ms": avg_inference_time_ms,
        "n_graphs": len(per_graph),
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
