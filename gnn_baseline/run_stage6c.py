"""
Stage 6C — Physics-Informed DAG-GNN Baseline

Runs 3-way ablation: Vanilla (6B) → Tier A (node sensitivities) → Tier A+B (node + graph-level analytical)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import create_dataloaders, GraphDataset
from foundations.config_loader import load_config, TimingParams
from model import VanillaDAGGNNSage, PhysicsInformedDAGGNNSage, MaxBiasedDAGGNNSage
from train import train_model
from eval import evaluate_model, compute_analytical_baseline_metrics, compare_model_vs_analytical
from timing.graph import TimingGraph, Gate
from timing.delay import delay_partials
from data_generation.analytical_ssta_arbitrary import compute_analytical_ssta_arbitrary


SMOKE = os.environ.get("VLSI_SMOKE", "0") == "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


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
    torch.use_deterministic_algorithms(True)


def preflight(data_dir: Path, config: str) -> Dict:
    """Pre-flight checks before running a config."""
    checks = {"ok": True, "warnings": [], "errors": []}
    dataset_pkl = data_dir / "dataset.pkl"
    splits_json = data_dir / "splits.json"
    if not dataset_pkl.exists():
        checks["errors"].append(f"Missing {dataset_pkl}")
        checks["ok"] = False
    if not splits_json.exists():
        checks["errors"].append(f"Missing {splits_json}")
        checks["ok"] = False
    if config != "vanilla":
        try:
            cfg = load_config(str(REPO_ROOT / "foundations" / "stage3_config.json"))
            vdd = cfg.timing_params.vdd_v
            vth = cfg.variation_params.vth_nom_v
            if vdd <= vth:
                checks["errors"].append(f"Vdd ({vdd}) <= Vth ({vth})")
                checks["ok"] = False
        except Exception as exc:
            checks["warnings"].append(f"Config load failed: {exc}")
    return checks


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
    device: torch.device,
    checkpoint_dir: Path,
    config_name: str,
    physics_feature_time_ms: float | None = None,
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
    elif config_name == "vanilla_nocoor":
        # ADR-008 §10 item 4 diagnostic: vanilla-minus-coordinates. Same
        # architecture, but only load_ff (no x,y); tests whether vanilla is
        # implicitly reconstructing geometry from raw features.
        model = VanillaDAGGNNSage(
            num_node_features=1,
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
            num_node_features=4,
        ).to(device)
    elif config_name == "tier_b_only":
        model = PhysicsInformedDAGGNNSage(
            tier="b",
            hidden_dim=64,
            num_layers=3,
            dropout=0.15,
            num_outputs=2,
            num_node_features=5,
        ).to(device)
    elif config_name == "tier_ab":
        model = PhysicsInformedDAGGNNSage(
            tier="ab",
            hidden_dim=64,
            num_layers=3,
            dropout=0.15,
            num_outputs=2,
            num_node_features=6,
        ).to(device)
    elif config_name == "maxbias":
        model = MaxBiasedDAGGNNSage(
            num_node_features=3,
            hidden_dim=64,
            num_layers=3,
            dropout=0.15,
            num_outputs=2,
        ).to(device)
    elif config_name == "maxbias_cm":
        # Capacity-matched control: reduce hidden dim so the mean+max hybrid
        # (2h->h per conv) carries ~the same params as vanilla h=64 (29698).
        # h=54 -> 30026 params (+1.1%) - isolates the aggregation change from
        # the capacity change that the h=64 maximas run conflates (41986).
        model = MaxBiasedDAGGNNSage(
            num_node_features=3,
            hidden_dim=54,
            num_layers=3,
            dropout=0.15,
            num_outputs=2,
        ).to(device)
    elif config_name == "maxbias_cm_tierab":
        # Combined ablation: MAX-biased aggregation (capacity-matched h=54, the
        # Section 7 positive architecture) + redesigned Tier A+B per-node
        # physics features (load_ff, x, y, var_d/load_ff^2, AT_mean, AT_var).
        # Same capacity-matched hidden dim; only the node-feature semantics
        # differ from maxbias_cm (6-dim tier_ab features, physics_mode="tier_ab").
        model = MaxBiasedDAGGNNSage(
            num_node_features=6,
            hidden_dim=54,
            num_layers=3,
            dropout=0.15,
            num_outputs=2,
        ).to(device)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Train
    max_epochs = 10 if SMOKE else 200
    checkpoint_path = str(checkpoint_dir / f"best_model_{config_name}_seed{seed}.pt")
    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        max_epochs=max_epochs,
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

    # Physics feature computation time (Tier A+B only; measured for Tier A, 0 for vanilla/maxbias)
    if config_name in ("vanilla", "vanilla_nocoor", "maxbias", "maxbias_cm"):
        physics_feature_time_ms = 0.0
        total_inference_ms = test_metrics['avg_inference_time_ms']
    else:
        if physics_feature_time_ms is None:
            physics_feature_time_ms = measure_physics_feature_time(test_dataset, config=config_name, n_samples=100)
        total_inference_ms = test_metrics['avg_inference_time_ms'] + physics_feature_time_ms
        print(f"Physics feature time: {physics_feature_time_ms:.2f} ms/graph")
        print(f"Total inference cost (GNN + physics): {total_inference_ms:.2f} ms/graph")

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
        "physics_feature_time_ms": physics_feature_time_ms,
        "total_inference_ms": total_inference_ms,
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


def measure_physics_feature_time(dataset: GraphDataset, config: str = "tier_ab", n_samples: int = 100) -> float:
    """Measure per-graph physics feature computation time (ms).

    Times the actual analytical SSTA + sensitivity computations, not dict lookups.
    TimingGraph construction is excluded (defensible: graph is already parsed).
    """
    from foundations.config_loader import TimingParams, load_config
    from timing.graph import TimingGraph, Gate
    from timing.delay import delay_partials
    from data_generation.analytical_ssta_arbitrary import compute_analytical_ssta_arbitrary

    config_obj = load_config(str(REPO_ROOT / "foundations" / "stage3_config.json"))
    timing_params = config_obj.timing_params
    variation_params = config_obj.variation_params

    graph_ids = dataset.graph_ids[:n_samples]
    times = []
    for gid in graph_ids:
        entry = dataset.dataset[gid]
        graph = entry["graph"]

        gates = {
            name: Gate(name=name, load_ff=gate_data["load_ff"], x=gate_data["x"], y=gate_data["y"])
            for name, gate_data in graph["gates"].items()
        }
        timing_graph = TimingGraph(gates=gates, successors=graph["successors"])

        gate_loads = {name: gate_data["load_ff"] for name, gate_data in graph["gates"].items()}
        graph_timing_params = TimingParams(
            k=timing_params.k,
            alpha=timing_params.alpha,
            vdd_v=timing_params.vdd_v,
            gate_loads=gate_loads,
        )
        graph_variation_params = replace(
            variation_params,
            gate_coords={name: (gate_data["x"], gate_data["y"]) for name, gate_data in graph["gates"].items()},
        )

        start = time.perf_counter()

        if config == "tier_a":
            for name in timing_graph.topological_order():
                gate = timing_graph.gates[name]
                delay_partials(
                    load_ff=gate.load_ff,
                    gate_params=graph_timing_params,
                    vth_nom=variation_params.vth_nom_v,
                    l_nom=variation_params.l_nom_nm,
                    w_nom=variation_params.w_nom_nm,
                )
        else:
            compute_analytical_ssta_arbitrary(
                timing_graph, graph_timing_params, graph_variation_params
            )
            for name in timing_graph.topological_order():
                gate = timing_graph.gates[name]
                delay_partials(
                    load_ff=gate.load_ff,
                    gate_params=graph_timing_params,
                    vth_nom=variation_params.vth_nom_v,
                    l_nom=variation_params.l_nom_nm,
                    w_nom=variation_params.w_nom_nm,
                )

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000.0)

    return float(np.mean(times)) if times else 0.0


def compute_tier_a_diagnostics(train_dataset: GraphDataset) -> Dict:
    """Compute diagnostics for Tier A (node-level var_d/load_ff²) feature.

    Checks:
    (a) Sign sanity of underlying raw sensitivities: d/dL>0, d/dW<0, d/dVth>0 across all gates
    (b) Redundancy check: correlation of var_d/load_ff² with load_ff, x, y
    (c) Physics_stats spreads (std/mean) of var_d/load_ff²
    (d) Magnitude identities on every gate sample (using raw sensitivities)
    (e) d-free cross-ratios
    """
    all_vth = []
    all_l = []
    all_w = []
    all_load = []
    all_x = []
    all_y = []
    all_var_d = []

    for gid in train_dataset.graph_ids:
        entry = train_dataset.dataset[gid]
        gates = entry["graph"]["gates"]
        sensitivities = entry["physics_features"]["sensitivities"]
        var_d_map = entry["physics_features"]["var_d_per_load_ff_sq"]
        for name, gate in gates.items():
            sens = sensitivities[name]
            all_vth.append(sens["vth"])
            all_l.append(sens["l"])
            all_w.append(sens["w"])
            all_load.append(gate["load_ff"])
            all_x.append(gate["x"])
            all_y.append(gate["y"])
            all_var_d.append(var_d_map.get(name, 0.0))

    all_vth = np.array(all_vth)
    all_l = np.array(all_l)
    all_w = np.array(all_w)
    all_load = np.array(all_load)
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_var_d = np.array(all_var_d)

    sign_vth = float(np.mean(all_vth > 0))
    sign_l = float(np.mean(all_l > 0))
    sign_w = float(np.mean(all_w < 0))

    corr_var_d_load = float(np.corrcoef(all_var_d, all_load)[0, 1]) if len(all_var_d) > 1 else 0.0
    corr_var_d_x = float(np.corrcoef(all_var_d, all_x)[0, 1]) if len(all_var_d) > 1 else 0.0
    corr_var_d_y = float(np.corrcoef(all_var_d, all_y)[0, 1]) if len(all_var_d) > 1 else 0.0

    spreads = {
        "var_d_spread": float(np.std(all_var_d) / max(abs(np.mean(all_var_d)), 1e-9)),
        "load_ff_spread": float(np.std(all_load) / max(abs(np.mean(all_load)), 1e-9)),
    }

    # Magnitude identities on every gate sample
    cfg = load_config(str(REPO_ROOT / "foundations" / "stage3_config.json"))
    vdd = cfg.timing_params.vdd_v
    k = cfg.timing_params.k
    alpha = cfg.timing_params.alpha
    l_nom = cfg.variation_params.l_nom_nm
    w_nom = cfg.variation_params.w_nom_nm
    vth_nom = cfg.variation_params.vth_nom_v

    vth_ids = []
    l_ids = []
    w_ids = []
    cross_ratios = []
    wl_cross_ratios = []
    for gid in train_dataset.graph_ids:
        entry = train_dataset.dataset[gid]
        sensitivities = entry["physics_features"]["sensitivities"]
        for name, gate in entry["graph"]["gates"].items():
            sens = sensitivities[name]
            d_nom = gate["load_ff"] * vdd / (k * (vdd - vth_nom) ** alpha)
            if d_nom > 1e-12:
                vth_ids.append(sens["vth"] * (vdd - vth_nom) / (alpha * d_nom))
                l_ids.append(sens["l"] * l_nom / d_nom)
                w_ids.append(sens["w"] * (2.0 * w_nom) / d_nom)
                if abs(sens["l"]) > 1e-12:
                    cross_ratios.append(sens["vth"] / sens["l"])
                    wl_cross_ratios.append(sens["w"] / sens["l"])

    magnitudes = {
        "vth_identity": {
            "mean": float(np.mean(vth_ids)) if vth_ids else 0.0,
            "std": float(np.std(vth_ids, ddof=1)) if len(vth_ids) > 1 else 0.0,
            "min": float(np.min(vth_ids)) if vth_ids else 0.0,
            "max": float(np.max(vth_ids)) if vth_ids else 0.0,
        },
        "l_identity": {
            "mean": float(np.mean(l_ids)) if l_ids else 0.0,
            "std": float(np.std(l_ids, ddof=1)) if len(l_ids) > 1 else 0.0,
            "min": float(np.min(l_ids)) if l_ids else 0.0,
            "max": float(np.max(l_ids)) if l_ids else 0.0,
        },
        "w_identity": {
            "mean": float(np.mean(w_ids)) if w_ids else 0.0,
            "std": float(np.std(w_ids, ddof=1)) if len(w_ids) > 1 else 0.0,
            "min": float(np.min(w_ids)) if w_ids else 0.0,
            "max": float(np.max(w_ids)) if w_ids else 0.0,
        },
        "n_samples": len(vth_ids),
        "expected_vth": 1.0,
        "expected_l": 1.0,
        "expected_w": -1.0,
        "d_free_cross_ratio_vth_over_l_mean": float(np.mean(cross_ratios)) if cross_ratios else 0.0,
        "d_free_cross_ratio_vth_over_l_std": float(np.std(cross_ratios, ddof=1)) if len(cross_ratios) > 1 else 0.0,
        "d_free_cross_ratio_n_samples": len(cross_ratios),
        "expected_vth_over_l_alpha_l_over_vdd_minus_vth": alpha * l_nom / (vdd - vth_nom),
        "d_free_cross_ratio_w_over_l_mean": float(np.mean(wl_cross_ratios)) if wl_cross_ratios else 0.0,
        "d_free_cross_ratio_w_over_l_std": float(np.std(wl_cross_ratios, ddof=1)) if len(wl_cross_ratios) > 1 else 0.0,
        "d_free_cross_ratio_w_over_l_n_samples": len(wl_cross_ratios),
        "expected_w_over_l_minus_l_over_2w": -l_nom / (2.0 * w_nom),
        "k_value": k,
        "k_placement_discriminative": bool(abs(k - 1.0) > 1e-12),
    }

    redundancy_ok = (
        abs(corr_var_d_load) < 0.3 and
        abs(corr_var_d_x) < 0.3 and
        abs(corr_var_d_y) < 0.3
    )
    redundancy_statement = (
        f"var_d/load_ff² is NOT redundant with existing features: "
        f"corr(load_ff)={corr_var_d_load:.3f}, corr(x)={corr_var_d_x:.3f}, corr(y)={corr_var_d_y:.3f}. "
        "This feature provides new position-dependent signal."
    ) if redundancy_ok else (
        f"WARNING: var_d/load_ff² is still correlated with existing features: "
        f"corr(load_ff)={corr_var_d_load:.3f}, corr(x)={corr_var_d_x:.3f}, corr(y)={corr_var_d_y:.3f}. "
        "Check if position-dependent variance is already captured."
    )

    return {
        "sign_vth_positive_rate": sign_vth,
        "sign_l_positive_rate": sign_l,
        "sign_w_negative_rate": sign_w,
        "spreads": spreads,
        "corr_var_d_load": corr_var_d_load,
        "corr_var_d_x": corr_var_d_x,
        "corr_var_d_y": corr_var_d_y,
        "magnitude_identities": magnitudes,
        "redundancy_statement": redundancy_statement,
    }


def run_no_gnn_baseline(train_dataset: GraphDataset, val_dataset: GraphDataset, test_dataset: GraphDataset, physics_stats: Dict, seeds: List[int] = [42, 123, 999]) -> Dict:
    """No-GNN residual baseline: MLP on [sink_mean, sink_std, n_gates_norm] → predicts (mean, std).

    Uses mini-batch training, frozen val split, early stopping, and OLS floor.
    """
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader as TorchDataLoader

    class ResidualMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(3, 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(32, 2),
            )

        def forward(self, x):
            return self.mlp(x)

    train_n_gates = []
    for gid in train_dataset.graph_ids:
        entry = train_dataset.dataset[gid]
        train_n_gates.append(len(entry["graph"]["gates"]))
    n_gates_mean = float(np.mean(train_n_gates))
    n_gates_std = float(np.std(train_n_gates)) if len(train_n_gates) > 1 else 1.0

    def _build_features(dataset_graph_ids, dataset):
        xs, ys = [], []
        for gid in dataset_graph_ids:
            entry = dataset.dataset[gid]
            ana = entry["physics_features"]["analytical_ssta"]
            mc = entry["mc_labels"]
            n_gates = len(entry["graph"]["gates"])
            xs.append([
                (ana["sink_mean"] - physics_stats["sink_mean_mean"]) / physics_stats["sink_mean_std"],
                (ana["sink_std"] - physics_stats["sink_std_mean"]) / physics_stats["sink_std_std"],
                (n_gates - n_gates_mean) / max(n_gates_std, 1e-6),
            ])
            ys.append([
                (mc["mean"] - physics_stats["sink_mean_mean"]) / physics_stats["sink_mean_std"],
                (mc["std"] - physics_stats["sink_std_mean"]) / physics_stats["sink_std_std"],
            ])
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

    train_x, train_y = _build_features(train_dataset.graph_ids, train_dataset)
    val_x, val_y = _build_features(val_dataset.graph_ids, val_dataset)
    test_x, test_y = _build_features(test_dataset.graph_ids, test_dataset)

    test_mean_orig = test_y[:, 0] * physics_stats["sink_mean_std"] + physics_stats["sink_mean_mean"]
    test_std_orig = test_y[:, 1] * physics_stats["sink_std_std"] + physics_stats["sink_std_mean"]

    train_ds = TensorDataset(train_x, train_y)
    train_loader_mlp = TorchDataLoader(train_ds, batch_size=32, shuffle=True)

    # OLS floor (deterministic, seedless)
    Xtr_ols = torch.cat([train_x, torch.ones(len(train_x), 1)], dim=1).numpy()
    beta = np.linalg.lstsq(Xtr_ols, train_y.numpy(), rcond=None)[0]
    Xte_ols = torch.cat([test_x, torch.ones(len(test_x), 1)], dim=1).numpy()
    ols_pred = Xte_ols @ beta
    ols_pred_mean = torch.tensor(ols_pred[:, 0]) * physics_stats["sink_mean_std"] + physics_stats["sink_mean_mean"]
    ols_pred_std = torch.tensor(ols_pred[:, 1]) * physics_stats["sink_std_std"] + physics_stats["sink_std_mean"]
    ols_mean_mae = float(torch.mean(torch.abs(ols_pred_mean - test_mean_orig)).item())
    ols_std_mae = float(torch.mean(torch.abs(ols_pred_std - test_std_orig)).item())
    ols_slope_mean = float(beta[0, 0])
    ols_slope_std = float(beta[0, 1])

    seed_results = []
    for seed in seeds:
        set_seed(seed)
        model = ResidualMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val = float("inf")
        best_state = None
        patience_counter = 0

        max_epochs = 50 if SMOKE else 1000
        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_loader_mlp:
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = float(F.mse_loss(val_pred, val_y).item())

            if epoch % 100 == 0:
                print(f"    ResidualMLP seed{seed} epoch {epoch}: val={val_loss:.6f}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

        model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            pred = model(test_x)
            pred_mean = pred[:, 0] * physics_stats["sink_mean_std"] + physics_stats["sink_mean_mean"]
            pred_std = pred[:, 1] * physics_stats["sink_std_std"] + physics_stats["sink_std_mean"]

        mean_mae = float(torch.mean(torch.abs(pred_mean - test_mean_orig)).item())
        std_mae = float(torch.mean(torch.abs(pred_std - test_std_orig)).item())
        mean_relative = float(torch.mean(torch.abs(pred_mean - test_mean_orig) / torch.maximum(torch.abs(test_mean_orig), torch.tensor(1e-9))).item())
        std_relative = float(torch.mean(torch.abs(pred_std - test_std_orig) / torch.maximum(torch.abs(test_std_orig), torch.tensor(1e-9))).item())

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        with torch.no_grad():
            train_pred = model(train_x)
            train_loss = float(F.mse_loss(train_pred, train_y).item())

        seed_results.append({
            "seed": seed,
            "mean_mae": mean_mae,
            "mean_relative": mean_relative,
            "std_mae": std_mae,
            "std_relative": std_relative,
            "n_params": n_params,
            "best_val_loss": best_val,
            "n_gates_mean": n_gates_mean,
            "n_gates_std": n_gates_std,
            "final_train_loss": train_loss,
            "train_val_ratio": train_loss / max(best_val, 1e-9),
        })
        print(f"    seed{seed}: mean_mae={mean_mae:.4f}, std_mae={std_mae:.4f}, params={n_params}, train/val={train_loss/max(best_val,1e-9):.3f}")

    mean_maes = [r["mean_mae"] for r in seed_results]
    std_maes = [r["std_mae"] for r in seed_results]
    mean_relatives = [r["mean_relative"] for r in seed_results]
    std_relatives = [r["std_relative"] for r in seed_results]

    mlp_mean = float(np.mean(mean_maes))
    ols_mean = ols_mean_mae
    mlp_le_ols = mlp_mean <= ols_mean + 0.01
    slope_near_1 = abs(ols_slope_mean - 1.0) <= 0.1

    return {
        "seeds": seeds,
        "seed_results": seed_results,
        "mean_mae_mean": mlp_mean,
        "mean_mae_std": float(np.std(mean_maes, ddof=1)),
        "mean_relative_mean": float(np.mean(mean_relatives)),
        "mean_relative_std": float(np.std(mean_relatives, ddof=1)),
        "std_mae_mean": float(np.mean(std_maes)),
        "std_mae_std": float(np.std(std_maes, ddof=1)),
        "std_relative_mean": float(np.mean(std_relatives)),
        "std_relative_std": float(np.std(std_relatives, ddof=1)),
        "n_params": seed_results[0]["n_params"] if seed_results else 0,
        "n_gates_mean": n_gates_mean,
        "n_gates_std": n_gates_std,
        "ols_floor": {
            "mean_mae": ols_mean_mae,
            "std_mae": ols_std_mae,
            "slope_sink_mean": ols_slope_mean,
            "slope_sink_std": ols_slope_std,
        },
        "convergence_gates": {
            "mlp_le_ols_plus_001": mlp_le_ols,
            "mlp_mean": mlp_mean,
            "ols_mean": ols_mean,
            "ols_slope_within_0p1_of_1": slope_near_1,
            "ols_slope_sink_mean": ols_slope_mean,
        },
    }


def compute_lockstep_verification(vanilla_6c_results: List[Dict], script_dir: Path) -> Dict:
    """Compare 6B vanilla results (disk) with 6C vanilla run (in-memory) for lockstep verification."""
    candidates = [
        script_dir / "results" / "vanilla_dag_gnn_results.json",
        script_dir / "results" / "vanilla_dag_gnn_results_rerun.json",
    ]
    results_6b_path = next((p for p in candidates if p.exists()), None)

    if results_6b_path is None:
        return {"status": "skipped", "reason": "6B results file not found"}

    with open(results_6b_path) as f:
        data_6b = json.load(f)

    vanilla_6b = data_6b["results"]
    vanilla_6c = vanilla_6c_results

    vanilla_6b_sorted = sorted(vanilla_6b, key=lambda r: r["seed"])
    vanilla_6c_sorted = sorted(vanilla_6c, key=lambda r: r["seed"])

    # Match by SEED value, not list position: extended-seed runs (5-7 seeds)
    # carry vanilla seeds beyond the committed 6 seeds, and positional zip()
    # would misalign them (e.g. 6B's 999 vs 6C's 777 if 777 sorts between 123
    # and 999). Only the seeds present in BOTH runs are comparable.
    comparison = []
    all_match = True
    max_mean_diff = 0.0
    max_std_diff = 0.0
    by_seed_6b = {r["seed"]: r for r in vanilla_6b_sorted}
    for r6c in vanilla_6c_sorted:
        r6b = by_seed_6b.get(r6c["seed"])
        if r6b is None:
            # No committed 6B reference for this seed: it is an independent
            # extended-seed run and not lockstep-comparable. Record but skip.
            comparison.append({
                "seed": r6c["seed"],
                "source_6b": None,
                "source_6c": "in-memory 6C vanilla",
                "mean_mae_6b": None,
                "mean_mae_6c": r6c["test_metrics"]["mean_mae"],
                "mean_mae_match": None,
                "std_mae_6b": None,
                "std_mae_6c": r6c["test_metrics"]["std_mae"],
                "std_mae_match": None,
                "best_val_loss_6b": None,
                "best_val_loss_6c": r6c["train_result"]["best_val_loss"],
                "n_params_6b": None,
                "n_params_6c": r6c["n_params"],
            })
            continue
        mean_diff = abs(r6b["test_metrics"]["mean_mae"] - r6c["test_metrics"]["mean_mae"])
        std_diff = abs(r6b["test_metrics"]["std_mae"] - r6c["test_metrics"]["std_mae"])
        max_mean_diff = max(max_mean_diff, mean_diff)
        max_std_diff = max(max_std_diff, std_diff)
        match = mean_diff < 1e-5 and std_diff < 1e-5
        all_match = all_match and match

        comparison.append({
            "seed": r6b["seed"],
            "source_6b": str(results_6b_path),
            "source_6c": "in-memory 6C vanilla",
            "mean_mae_6b": r6b["test_metrics"]["mean_mae"],
            "mean_mae_6c": r6c["test_metrics"]["mean_mae"],
            "mean_mae_match": match,
            "std_mae_6b": r6b["test_metrics"]["std_mae"],
            "std_mae_6c": r6c["test_metrics"]["std_mae"],
            "std_mae_match": match,
            "best_val_loss_6b": r6b.get("best_val_loss") if "best_val_loss" in r6b else r6b.get("train_result", {}).get("best_val_loss"),
            "best_val_loss_6c": r6c["train_result"]["best_val_loss"],
            "n_params_6b": r6b.get("n_params", sum(p.numel() for p in VanillaDAGGNNSage().parameters())),
            "n_params_6c": r6c["n_params"],
        })

    return {
        "status": "verified",
        "all_match": all_match,
        "max_mean_diff": max_mean_diff,
        "max_std_diff": max_std_diff,
        "comparison": comparison,
    }


def compute_nrecon_reconciliation(data_dir: Path) -> Dict:
    """Raw recount of nrecon distribution from dataset.pkl + splits.json."""
    import pickle

    with open(data_dir / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    with open(data_dir / "splits.json", "r") as f:
        splits = json.load(f)

    reconciliation = {}
    for split_name, split_ids in splits.items():
        nrecon_counts = {}
        n_gates_list = []
        for gid in split_ids:
            entry = dataset[gid]
            nrecon = len(entry["graph"]["reconvergence_points"])
            n_gates_list.append(len(entry["graph"]["gates"]))
            nrecon_counts[nrecon] = nrecon_counts.get(nrecon, 0) + 1

        reconciliation[split_name] = {
            "n_graphs": len(split_ids),
            "nrecon_distribution": dict(sorted(nrecon_counts.items())),
            "mean_nrecon": float(np.mean([len(dataset[gid]["graph"]["reconvergence_points"]) for gid in split_ids])),
            "mean_n_gates": float(np.mean(n_gates_list)),
            "corr_nrecon_n_gates": float(np.corrcoef(
                [len(dataset[gid]["graph"]["reconvergence_points"]) for gid in split_ids],
                n_gates_list
            )[0, 1]) if len(split_ids) > 1 else 0.0,
        }

    return reconciliation


def compute_analytical_mae_per_nrecon(test_dataset: GraphDataset, physics_stats: Dict) -> Dict:
    """Compute analytical SSTA MAE per nrecon bucket in normalized and original units."""
    buckets = {}
    for gid in test_dataset.graph_ids:
        entry = test_dataset.dataset[gid]
        nrecon = len(entry["graph"]["reconvergence_points"])
        ana = entry["physics_features"]["analytical_ssta"]
        mc = entry["mc_labels"]

        ana_mean_norm = (ana["sink_mean"] - physics_stats["sink_mean_mean"]) / physics_stats["sink_mean_std"]
        mc_mean_norm = (mc["mean"] - physics_stats["sink_mean_mean"]) / physics_stats["sink_mean_std"]
        ana_std_norm = (ana["sink_std"] - physics_stats["sink_std_mean"]) / physics_stats["sink_std_std"]
        mc_std_norm = (mc["std"] - physics_stats["sink_std_mean"]) / physics_stats["sink_std_std"]

        mean_mae_norm = abs(ana_mean_norm - mc_mean_norm)
        std_mae_norm = abs(ana_std_norm - mc_std_norm)
        mean_mae_orig = abs(ana["sink_mean"] - mc["mean"])
        std_mae_orig = abs(ana["sink_std"] - mc["std"])

        if nrecon not in buckets:
            buckets[nrecon] = {
                "mean_maes_norm": [], "std_maes_norm": [],
                "mean_maes_orig": [], "std_maes_orig": [],
                "count": 0
            }

        buckets[nrecon]["mean_maes_norm"].append(mean_mae_norm)
        buckets[nrecon]["std_maes_norm"].append(std_mae_norm)
        buckets[nrecon]["mean_maes_orig"].append(mean_mae_orig)
        buckets[nrecon]["std_maes_orig"].append(std_mae_orig)
        buckets[nrecon]["count"] += 1

    result = {}
    for nrecon, data in sorted(buckets.items()):
        mean_maes_norm = np.array(data["mean_maes_norm"])
        std_maes_norm = np.array(data["std_maes_norm"])
        mean_maes_orig = np.array(data["mean_maes_orig"])
        std_maes_orig = np.array(data["std_maes_orig"])
        result[nrecon] = {
            "count": data["count"],
            "mean_mae_mean_norm": float(np.mean(mean_maes_norm)),
            "mean_mae_std_norm": float(np.std(mean_maes_norm, ddof=1)) if len(mean_maes_norm) > 1 else 0.0,
            "std_mae_mean_norm": float(np.mean(std_maes_norm)),
            "std_mae_std_norm": float(np.std(std_maes_norm, ddof=1)) if len(std_maes_norm) > 1 else 0.0,
            "mean_mae_mean_orig": float(np.mean(mean_maes_orig)),
            "mean_mae_std_orig": float(np.std(mean_maes_orig, ddof=1)) if len(mean_maes_orig) > 1 else 0.0,
            "std_mae_mean_orig": float(np.mean(std_maes_orig)),
            "std_mae_std_orig": float(np.std(std_maes_orig, ddof=1)) if len(std_maes_orig) > 1 else 0.0,
        }
    return result


def paired_cluster_bootstrap_ci(vanilla_results: List[Dict], tier_results: List[Dict], tier_name: str = "tier") -> Dict:
    """Paired per-graph cluster bootstrap CI (tier minus vanilla), per graph_id,
    clustered across the 3 seeds. Uses RandomState(42) + 10000 resamples, matching
    the historical Stage 6C methodology exactly."""
    by_graph_mean = defaultdict(list)
    by_graph_std = defaultdict(list)
    for vr, tr in zip(vanilla_results, tier_results):
        for vp, tp in zip(vr["test_metrics"]["per_graph"], tr["test_metrics"]["per_graph"]):
            gid = vp["graph_id"]
            assert tp["graph_id"] == gid
            by_graph_mean[gid].append(tp["mean_mae"] - vp["mean_mae"])
            by_graph_std[gid].append(tp["std_mae"] - vp["std_mae"])

    graph_mean_deltas = np.array([np.mean(v) for v in by_graph_mean.values()])
    graph_std_deltas = np.array([np.mean(v) for v in by_graph_std.values()])
    mean_delta = float(np.mean(graph_mean_deltas))
    std_delta = float(np.mean(graph_std_deltas))
    mean_win_rate = float(np.mean(graph_mean_deltas < 0))
    std_win_rate = float(np.mean(graph_std_deltas < 0))

    rng = np.random.RandomState(42)
    boot_mean_means = []
    boot_std_means = []
    for _ in range(10000):
        idx = rng.randint(0, len(graph_mean_deltas), len(graph_mean_deltas))
        boot_mean_means.append(np.mean(graph_mean_deltas[idx]))
        boot_std_means.append(np.mean(graph_std_deltas[idx]))
    mean_ci_low = float(np.percentile(boot_mean_means, 2.5))
    mean_ci_high = float(np.percentile(boot_mean_means, 97.5))
    std_ci_low = float(np.percentile(boot_std_means, 2.5))
    std_ci_high = float(np.percentile(boot_std_means, 97.5))

    per_seed_mean = []
    per_seed_std = []
    for vr, tr in zip(vanilla_results, tier_results):
        seed_mean_deltas = [tp["mean_mae"] - vp["mean_mae"] for vp, tp in zip(vr["test_metrics"]["per_graph"], tr["test_metrics"]["per_graph"])]
        seed_std_deltas = [tp["std_mae"] - vp["std_mae"] for vp, tp in zip(vr["test_metrics"]["per_graph"], tr["test_metrics"]["per_graph"])]
        per_seed_mean.append(float(np.mean(seed_mean_deltas)))
        per_seed_std.append(float(np.mean(seed_std_deltas)))

    return {
        "tier": tier_name,
        "mean_delta": mean_delta,
        "mean_win_rate": mean_win_rate,
        "mean_ci_low": mean_ci_low,
        "mean_ci_high": mean_ci_high,
        "std_delta": std_delta,
        "std_win_rate": std_win_rate,
        "std_ci_low": std_ci_low,
        "std_ci_high": std_ci_high,
        "n_graphs": len(graph_mean_deltas),
        "per_seed_mean_deltas": per_seed_mean,
        "per_seed_std_deltas": per_seed_std,
        "bootstrap_method": "cluster by graph_id",
    }


def main():
    script_dir = Path(__file__).resolve().parent
    checkpoint_dir = script_dir / ("checkpoints_smoke" if SMOKE else "checkpoints")
    results_dir = script_dir / "results"
    interim_name = "stage6c_results_interim_smoke.json" if SMOKE else "stage6c_results_interim.json"
    final_name = "stage6c_results_smoke.json" if SMOKE else "stage6c_results.json"
    data_dir = script_dir.parent / "data_generation" / "data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checks = preflight(data_dir, "all")
    if not checks["ok"]:
        print(f"Preflight failed: {checks['errors']}")
        sys.exit(1)
    if checks["warnings"]:
        for w in checks["warnings"]:
            print(f"Preflight warning: {w}")

    # Configurations to run
    configs = ["vanilla", "tier_a", "tier_ab"]
    seeds = [42, 123, 999]

    # Run all configurations
    all_results = {}
    per_config_stats: Dict[str, Dict] = {}
    for config in configs:
        print(f"\n{'#'*60}")
        print(f"# Configuration: {config}")
        print(f"{'#'*60}")

        checks = preflight(data_dir, config)
        if not checks["ok"]:
            print(f"Preflight failed for {config}: {checks['errors']}")
            sys.exit(1)
        if checks["warnings"]:
            for w in checks["warnings"]:
                print(f"Preflight warning: {w}")

        train_loader, val_loader, test_loader, feature_stats, target_stats, physics_stats = create_dataloaders(
            data_dir=data_dir,
            batch_size=32,
            physics_mode=config,
        )
        per_config_stats[config] = {
            "feature_stats": {k: float(v) for k, v in feature_stats.items()},
            "target_stats": {k: float(v) for k, v in target_stats.items()},
            "physics_stats": {k: float(v) for k, v in physics_stats.items()},
        }

        # Batch shape check
        sample_batch = next(iter(train_loader))
        expected_node_feats = {"vanilla": 3, "tier_a": 4, "tier_b_only": 5, "tier_ab": 6}[config]
        check_batch_shape(sample_batch, train_loader.batch_size, expected_node_feats)
        print(f"Batch OK: x={sample_batch.x.shape}, y={sample_batch.y.shape}, "
              f"num_graphs={sample_batch.num_graphs if hasattr(sample_batch, 'num_graphs') else sample_batch.batch.max().item()+1}")

        test_dataset = GraphDataset(split="test", data_dir=data_dir, physics_mode=config)
        train_dataset = GraphDataset(split="train", data_dir=data_dir, physics_mode=config)

        config_results = []
        physics_feature_time_ms = None
        if config != "vanilla":
            physics_feature_time_ms = measure_physics_feature_time(test_dataset, config=config, n_samples=100)
        for seed in seeds:
            result = run_single_seed(
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                test_dataset=test_dataset,
                feature_stats=feature_stats,
                target_stats=target_stats,
                device=device,
                checkpoint_dir=checkpoint_dir,
                config_name=config,
                physics_feature_time_ms=physics_feature_time_ms,
            )
            config_results.append(result)

        all_results[config] = config_results

        interim = {
            "status": "trainings_complete_stats_pending",
            "completed_config": config,
            "results": {c: all_results[c] for c in all_results},
        }
        with open(results_dir / interim_name, "w") as f:
            json.dump(_to_serializable(interim), f, indent=2)

    physics_stats = per_config_stats["tier_ab"]["physics_stats"]

    # Tier A mechanism diagnostics
    print("\n--- Tier A Mechanism Diagnostics ---")
    tier_a_diag = compute_tier_a_diagnostics(train_dataset)
    print(f"  Sign sanity (raw sensitivities):")
    print(f"    d/dVth > 0: {tier_a_diag['sign_vth_positive_rate']:.1%} of gates")
    print(f"    d/dL > 0: {tier_a_diag['sign_l_positive_rate']:.1%} of gates")
    print(f"    d/dW < 0: {tier_a_diag['sign_w_negative_rate']:.1%} of gates")
    print(f"  Redundancy check (var_d/load_ff² vs existing features):")
    print(f"    corr(var_d/load_ff², load_ff): {tier_a_diag['corr_var_d_load']:.4f}")
    print(f"    corr(var_d/load_ff², x): {tier_a_diag['corr_var_d_x']:.4f}")
    print(f"    corr(var_d/load_ff², y): {tier_a_diag['corr_var_d_y']:.4f}")
    print(f"  Physics spreads (std/mean):")
    for k, v in tier_a_diag["spreads"].items():
        print(f"    {k}: {v:.4f}")
    print(f"  Magnitude identities (n={tier_a_diag['magnitude_identities']['n_samples']}):")
    print(f"    vth_identity: {tier_a_diag['magnitude_identities']['vth_identity']['mean']:.4f} ± {tier_a_diag['magnitude_identities']['vth_identity']['std']:.4f}  (expected +1.0)")
    print(f"    l_identity:   {tier_a_diag['magnitude_identities']['l_identity']['mean']:.4f} ± {tier_a_diag['magnitude_identities']['l_identity']['std']:.4f}  (expected +1.0)")
    print(f"    w_identity:   {tier_a_diag['magnitude_identities']['w_identity']['mean']:.4f} ± {tier_a_diag['magnitude_identities']['w_identity']['std']:.4f}  (expected -1.0)")
    mid = tier_a_diag["magnitude_identities"]
    print(f"    cross-ratio vth/l: {mid['d_free_cross_ratio_vth_over_l_mean']:.4f} ± {mid['d_free_cross_ratio_vth_over_l_std']:.2e}  (expected {mid['expected_vth_over_l_alpha_l_over_vdd_minus_vth']:.4f} = alpha*L_nom/(Vdd-Vth))")
    print(f"    cross-ratio w/l:   {mid['d_free_cross_ratio_w_over_l_mean']:.4f} ± {mid['d_free_cross_ratio_w_over_l_std']:.2e}  (expected {mid['expected_w_over_l_minus_l_over_2w']:.4f} = -L_nom/(2*W_nom))")
    if mid["k_placement_discriminative"]:
        print(f"    k-placement test DISCRIMINATIVE (k={mid['k_value']}): identities=1 confirms k sits in the delay denominator")
    else:
        print(f"    k-placement test non-discriminative at k={mid['k_value']} (identities=1 consistent with either placement)")
    print(f"  {tier_a_diag['redundancy_statement']}")

    # No-GNN residual baseline (Tier B-only, no graph structure)
    print("\n--- No-GNN Residual Baseline (MLP on analytical + n_gates) ---")
    no_gnn = run_no_gnn_baseline(
        train_dataset=GraphDataset(split="train", data_dir=data_dir, physics_mode="vanilla"),
        val_dataset=GraphDataset(split="val", data_dir=data_dir, physics_mode="vanilla"),
        test_dataset=GraphDataset(split="test", data_dir=data_dir, physics_mode="vanilla"),
        physics_stats=physics_stats,
        seeds=seeds,
    )
    print(f"  ResidualMLP mean MAE: {no_gnn['mean_mae_mean']:.4f} ± {no_gnn['mean_mae_std']:.4f}")
    print(f"  ResidualMLP mean relative: {no_gnn['mean_relative_mean']:.2%} ± {no_gnn['mean_relative_std']:.2%}")
    print(f"  ResidualMLP std MAE: {no_gnn['std_mae_mean']:.4f} ± {no_gnn['std_mae_std']:.4f}")
    print(f"  ResidualMLP std relative: {no_gnn['std_relative_mean']:.2%} ± {no_gnn['std_relative_std']:.2%}")
    print(f"  ResidualMLP params: {no_gnn['n_params']}")
    print(f"  n_gates mean/std (normalizer): {no_gnn['n_gates_mean']:.2f} / {no_gnn['n_gates_std']:.2f}")
    print(f"  OLS slope beta on normalized sink_mean: {no_gnn['ols_floor']['slope_sink_mean']:.4f} "
          f"(gate |beta-1|<=0.1: {'PASS' if no_gnn['convergence_gates']['ols_slope_within_0p1_of_1'] else 'FAIL'}; "
          f"expected ~ corr(analytical, MC) ~ 0.97)")

    tier_ab_mean_mae = float(np.mean([r["test_metrics"]["mean_mae"] for r in all_results["tier_ab"]]))
    print(f"  Tier A+B GNN mean MAE: {tier_ab_mean_mae:.4f}")
    rel = (no_gnn["mean_mae_mean"] - tier_ab_mean_mae) / max(tier_ab_mean_mae, 1e-9)
    if rel > 0.05:
        print(f"  GNN beats no-GNN residual MLP by {rel:.1%} — graph structure contributes beyond scalar residual correction")
    elif rel < -0.05:
        print(f"  no-GNN MLP BEATS Tier A+B by {-rel:.1%} — headline must be 'learned residual correction of analytical SSTA'")
    else:
        print("  no-GNN MLP matches Tier A+B within 5% — framing: residual correction; GNN adds little")

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
    print(f"\n{'Model':<35} {'Mean MAE':<12} {'Mean Rel':<12} {'Std MAE':<12} {'Std Rel':<12} {'Params':<8} {'Inference':<12} {'Physics':<10}")
    print("-" * 115)
    print(f"{'Analytical SSTA':<35} {analytical_metrics['mean_mae']:.4f}{'':>8} {analytical_metrics['mean_relative']:.2%}{'':>9} {analytical_metrics['std_mae']:.4f}{'':>8} {analytical_metrics['std_relative']:.2%}{'':>9} {'—':>8} {'—':>12} {'—':>10}")
    vanilla_inf = vanilla_results[0]['test_metrics']['avg_inference_time_ms']
    print(f"{'Vanilla DAG-GNN (6B, locked)':<35} {np.mean(vanilla_mean_maes):.4f}±{np.std(vanilla_mean_maes, ddof=1):.4f} {np.mean(vanilla_mean_relatives):.2%}±{np.std(vanilla_mean_relatives, ddof=1):.2%} {np.mean(vanilla_std_maes):.4f}±{np.std(vanilla_std_maes, ddof=1):.4f} {np.mean(vanilla_std_relatives):.2%}±{np.std(vanilla_std_relatives, ddof=1):.2%} {vanilla_results[0]['n_params']:>8,} {vanilla_inf:>10.2f} {'—':>10}")
    tier_a_inf = tier_a_results[0]['test_metrics']['avg_inference_time_ms']
    tier_a_phys = tier_a_results[0].get('physics_feature_time_ms', '—')
    tier_a_phys_str = f"{tier_a_phys:.2f}" if isinstance(tier_a_phys, (int, float)) else str(tier_a_phys)
    print(f"{'Physics-informed (Tier A)':<35} {np.mean(tier_a_mean_maes):.4f}±{np.std(tier_a_mean_maes, ddof=1):.4f} {np.mean(tier_a_mean_relatives):.2%}±{np.std(tier_a_mean_relatives, ddof=1):.2%} {np.mean(tier_a_std_maes):.4f}±{np.std(tier_a_std_maes, ddof=1):.4f} {np.mean(tier_a_std_relatives):.2%}±{np.std(tier_a_std_relatives, ddof=1):.2%} {tier_a_results[0]['n_params']:>8,} {tier_a_inf:>10.2f} {tier_a_phys_str:>10}")
    tier_ab_inf = tier_ab_results[0]['test_metrics']['avg_inference_time_ms']
    tier_ab_phys = tier_ab_results[0].get('physics_feature_time_ms', '—')
    tier_ab_phys_str = f"{tier_ab_phys:.2f}" if isinstance(tier_ab_phys, (int, float)) else str(tier_ab_phys)
    print(f"{'Physics-informed (Tier A+B)':<35} {np.mean(tier_ab_mean_maes):.4f}±{np.std(tier_ab_mean_maes, ddof=1):.4f} {np.mean(tier_ab_mean_relatives):.2%}±{np.std(tier_ab_mean_relatives, ddof=1):.2%} {np.mean(tier_ab_std_maes):.4f}±{np.std(tier_ab_std_maes, ddof=1):.4f} {np.mean(tier_ab_std_relatives):.2%}±{np.std(tier_ab_std_relatives, ddof=1):.2%} {tier_ab_results[0]['n_params']:>8,} {tier_ab_inf:>10.2f} {tier_ab_phys_str:>10}")
    print("  Note: Inference/Physics columns show seed-42 values; MAE/Rel columns show mean ± std across 3 seeds.")

    # Paired per-graph statistical significance (cluster bootstrap CI)
    print("\n--- Paired Per-Graph Significance vs Vanilla ---")
    significance_results = {}
    for tier_name, tier_results in [("tier_a", tier_a_results), ("tier_ab", tier_ab_results)]:
        significance_results[tier_name] = paired_cluster_bootstrap_ci(
            vanilla_results, tier_results, tier_name=tier_name
        )

        print(f"  {tier_name.upper()}:")
        print(f"    Mean MAE delta (tier - vanilla): {significance_results[tier_name]['mean_delta']:+.4f}  win_rate={significance_results[tier_name]['mean_win_rate']:.1%}  95% CI: [{significance_results[tier_name]['mean_ci_low']:+.4f}, {significance_results[tier_name]['mean_ci_high']:+.4f}]  sig={'Yes' if (significance_results[tier_name]['mean_ci_low'] > 0 or significance_results[tier_name]['mean_ci_high'] < 0) else 'No'}")
        print(f"    Std  MAE delta (tier - vanilla): {significance_results[tier_name]['std_delta']:+.4f}  win_rate={significance_results[tier_name]['std_win_rate']:.1%}  95% CI: [{significance_results[tier_name]['std_ci_low']:+.4f}, {significance_results[tier_name]['std_ci_high']:+.4f}]  sig={'Yes' if (significance_results[tier_name]['std_ci_low'] > 0 or significance_results[tier_name]['std_ci_high'] < 0) else 'No'}")
        print(f"    Per-seed mean deltas: {[f'{d:+.4f}' for d in significance_results[tier_name]['per_seed_mean_deltas']]}")
        print(f"    Per-seed std deltas:  {[f'{d:+.4f}' for d in significance_results[tier_name]['per_seed_std_deltas']]}")

    tier_a_sig = significance_results["tier_a"]["mean_ci_low"] > 0 or significance_results["tier_a"]["mean_ci_high"] < 0
    tier_ab_sig = significance_results["tier_ab"]["mean_ci_low"] > 0 or significance_results["tier_ab"]["mean_ci_high"] < 0

    print("\n  Headline (per-variant, driven by CI bounds):")
    tier_order = ["tier_a", "tier_ab"]
    tier_labels = {"tier_a": "Tier A", "tier_ab": "Tier A+B"}
    tier_summary = []
    for tier_name in tier_order:
        s = significance_results[tier_name]
        lo, hi = s["mean_ci_low"], s["mean_ci_high"]
        delta = s["mean_delta"]
        if hi < 0:
            verdict = "significantly BETTER than vanilla"
        elif lo > 0:
            verdict = "significantly WORSE than vanilla"
        else:
            verdict = "not statistically distinguishable from vanilla"
        tier_summary.append(f"{tier_labels[tier_name]} {verdict}")
        print(f"  - {tier_labels[tier_name]}: {verdict} "
              f"(delta={delta:+.4f}, 95% CI=[{lo:+.4f}, {hi:+.4f}])")
    n_sig = sum(1 for tn in tier_order if significance_results[tn]["mean_ci_low"] > 0
                or significance_results[tn]["mean_ci_high"] < 0)
    print(f"  Summary: {n_sig} of {len(tier_order)} physics tiers have a mean-MAE CI excluding 0; "
          f"verdicts: {tier_summary[0]}; {tier_summary[1]}.")

    # Tier B leakage check
    print("\n--- Tier B Leakage Check ---")
    leakage = compute_tier_b_leakage(test_dataset, physics_stats)
    print(f"  Analytical sink_mean correlation with MC mean: {leakage['mean_correlation']:.4f}")
    print(f"  Analytical sink_std correlation with MC std: {leakage['std_correlation']:.4f}")
    print(f"  Analytical mean MAE (normalized units): {leakage['mean_mae_normalized']:.4f}")
    print(f"  Analytical std MAE (normalized units): {leakage['std_mae_normalized']:.4f}")

    tier_ab_mean_mae = float(np.mean([r["test_metrics"]["mean_mae"] for r in tier_ab_results]))
    tier_ab_std_mae = float(np.mean([r["test_metrics"]["std_mae"] for r in tier_ab_results]))
    print(f"  Tier A+B mean MAE (original (toy) units): {tier_ab_mean_mae:.4f}")
    print(f"  Tier A+B std MAE (original (toy) units): {tier_ab_std_mae:.4f}")
    print(f"  Note: Analytical MAE in normalized units cannot be directly compared to GNN MAE in original (toy) units.")
    print(f"  Normalized Tier A+B mean MAE (approx): {tier_ab_mean_mae / physics_stats['sink_mean_std']:.4f}")
    print(f"  Normalized Tier A+B std MAE (approx): {tier_ab_std_mae / physics_stats['sink_std_std']:.4f}")

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

    print("  Note: eval_train uses eval-mode (dropout disabled), unlike 6B's train-active ratio 0.71–0.77.")

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

    # Lockstep verification (6B vanilla vs 6C vanilla)
    print("\n--- Lockstep Verification (6B vanilla vs 6C vanilla) ---")
    lockstep = compute_lockstep_verification(all_results["vanilla"], script_dir)
    if lockstep.get("status") == "verified":
        print(f"  {'Seed':<6} {'mean_mae_6b':<12} {'mean_mae_6c':<12} {'match':<8} {'std_mae_6b':<12} {'std_mae_6c':<12} {'match':<8} {'best_val_6b':<14} {'best_val_6c':<14}")
        print("  " + "-" * 100)
        for comp in lockstep["comparison"]:
            print(f"  {comp['seed']:<6} {comp['mean_mae_6b']:<12.6f} {comp['mean_mae_6c']:<12.6f} {str(comp['mean_mae_match']):<8} {comp['std_mae_6b']:<12.6f} {comp['std_mae_6c']:<12.6f} {str(comp['std_mae_match']):<8} {comp['best_val_loss_6b']:<14.6f} {comp['best_val_loss_6c']:<14.6f}")
    else:
        print(f"  Lockstep verification skipped: {lockstep.get('reason', 'unknown')}")

    # B5 nrecon reconciliation
    print("\n--- B5 Nrecon Reconciliation ---")
    nrecon_recon = compute_nrecon_reconciliation(data_dir)
    for split_name, recon in nrecon_recon.items():
        print(f"  {split_name}: n_graphs={recon['n_graphs']}, mean_nrecon={recon['mean_nrecon']:.2f}, mean_n_gates={recon['mean_n_gates']:.2f}")
        print(f"    nrecon_distribution: {recon['nrecon_distribution']}")
        print(f"    corr(nrecon, n_gates)={recon['corr_nrecon_n_gates']:.3f}")

    # Analytical MAE per nrecon bucket (S3 support)
    print("\n--- Analytical MAE Per Nrecon Bucket (normalized + original units) ---")
    analytical_per_nrecon = compute_analytical_mae_per_nrecon(test_dataset, physics_stats)
    print(f"  {'nrecon':<8} {'count':<8} {'mean_mae_mean_norm':<18} {'std_mae_mean_norm':<18} {'mean_mae_mean_orig':<18} {'std_mae_mean_orig':<18}")
    print("  " + "-" * 90)
    for nrecon, stats in sorted(analytical_per_nrecon.items()):
        print(f"  {nrecon:<8} {stats['count']:<8} {stats['mean_mae_mean_norm']:<18.4f} {stats['std_mae_mean_norm']:<18.4f} {stats['mean_mae_mean_orig']:<18.4f} {stats['std_mae_mean_orig']:<18.4f}")
    print("  Note: _norm columns are normalized units; _orig columns are original (toy) units.")

    # Build output JSON
    ls_verified = lockstep.get("status") == "verified"
    ls_exact = bool(lockstep.get("all_match")) if ls_verified else False
    phys_t = tier_ab_results[0].get("physics_feature_time_ms")
    phys_str = f"{phys_t:.2f} ms/graph" if isinstance(phys_t, (int, float)) else "not_measured"
    s2_word = ("gnn_beyond_scalar_residual" if rel > 0.05
               else ("residual_correction" if rel >= -0.05 else "mlp_beats_tier_ab"))
    mode_tag = "SMOKE" if SMOKE else "full"
    vs_vanilla_str = (
        f"vs_vanilla: tier_a={'sig' if tier_a_sig else 'ns'} "
        f"(delta={significance_results['tier_a']['mean_delta']:+.4f}, "
        f"CI=[{significance_results['tier_a']['mean_ci_low']:+.4f}, {significance_results['tier_a']['mean_ci_high']:+.4f}]), "
        f"tier_ab={'sig' if tier_ab_sig else 'ns'} "
        f"(delta={significance_results['tier_ab']['mean_delta']:+.4f}, "
        f"CI=[{significance_results['tier_ab']['mean_ci_low']:+.4f}, {significance_results['tier_ab']['mean_ci_high']:+.4f}])"
    )
    status = (
        f"{mode_tag} run | "
        f"lockstep={'exact' if ls_exact else ('mismatch' if ls_verified else 'skipped')} "
        f"(max_mean_diff={lockstep.get('max_mean_diff')}) | "
        f"cluster-bootstrap CIs computed (n={significance_results['tier_a']['n_graphs']}) | "
        f"physics timing tier_ab={phys_str} | "
        f"{vs_vanilla_str} | "
        f"S2 no-gnn rel={rel:+.1%} -> {s2_word} | "
        f"s2_convergence mlp_le_ols_plus_001={no_gnn['convergence_gates']['mlp_le_ols_plus_001']} | "
        f"B5 reconciliation persisted"
    )

    output = {
        "status": status,
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
                    "history": r["train_result"].get("history", {}),
                    "eval_train_loss": float(r["eval_train_loss"]),
                    "test_metrics": r["test_metrics"],
                    "analytical_metrics": r["analytical_metrics"],
                    "comparison": r["comparison"],
                    "physics_feature_time_ms": r.get("physics_feature_time_ms", "not_measured"),
                    "total_inference_ms": r.get("total_inference_ms", "not_measured"),
                }
                for r in results
            ]
            for config, results in all_results.items()
        },
        "no_gnn_baseline": no_gnn,
        "s2_convergence_decision": {
            "mlp_le_ols_plus_001": no_gnn.get("convergence_gates", {}).get("mlp_le_ols_plus_001", False),
            "mlp_mean_mae": no_gnn.get("mean_mae_mean", float("nan")),
            "ols_mean_mae": no_gnn.get("ols_floor", {}).get("mean_mae", float("nan")),
            "decision": "MLP beats OLS within 1%" if no_gnn.get("convergence_gates", {}).get("mlp_le_ols_plus_001", False) else "OLS competitive or better",
            "ols_slope_sink_mean": no_gnn.get("ols_floor", {}).get("slope_sink_mean", float("nan")),
            "ols_slope_within_0p1_of_1": no_gnn.get("convergence_gates", {}).get("ols_slope_within_0p1_of_1", False),
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
                "avg_train_time_s": float(np.mean([r["train_result"]["train_time"] for r in results])),
            }
            for config, results in all_results.items()
        },
        "paired_significance_vs_vanilla": significance_results,
        "analytical_baseline": analytical_metrics,
        "tier_b_leakage_check": leakage,
        "lockstep_verification": lockstep,
        "nrecon_reconciliation": nrecon_recon,
        "analytical_mae_per_nrecon": analytical_per_nrecon,
        "tier_a_diagnostics": tier_a_diag,
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
            "units": "original (toy) units (dimensionless, not ps)",
        },
    }

    output_path = results_dir / final_name
    with open(output_path, "w") as f:
        json.dump(_to_serializable(output), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
