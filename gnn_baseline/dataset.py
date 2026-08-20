"""
Stage 6C — Data loading pipeline for physics-informed DAG-GNN baseline.

Extends Stage 6B dataset with physics features:
- Tier A: per-node sensitivities (vth, l, w) appended to node features
- Tier B: graph-level analytical SSTA (sink_mean, sink_std) concatenated to pooled embedding
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GraphDataset:
    """Loads Stage 6A dataset and creates PyG Data objects."""

    def __init__(
        self,
        data_dir: str | Path = "data_generation/data",
        split: str = "train",
        physics_mode: str = "vanilla",
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.physics_mode = physics_mode

        # Load dataset and splits
        with open(self.data_dir / "dataset.pkl", "rb") as f:
            self.dataset = pickle.load(f)
        with open(self.data_dir / "splits.json", "r") as f:
            self.splits = json.load(f)

        self.graph_ids = self.splits[split]

    def _compute_normalization(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute mean/std for features and targets from train set only."""
        all_load = []
        all_x = []
        all_y = []
        all_mean = []
        all_std = []

        for gid in self.graph_ids:
            entry = self.dataset[gid]
            gates = entry["graph"]["gates"]
            for name, gate in gates.items():
                all_load.append(gate["load_ff"])
                all_x.append(gate["x"])
                all_y.append(gate["y"])
            all_mean.append(entry["mc_labels"]["mean"])
            all_std.append(entry["mc_labels"]["std"])

        feature_stats = {
            "load_mean": float(np.mean(all_load)),
            "load_std": float(np.std(all_load, ddof=0) + 1e-8),
            "x_mean": float(np.mean(all_x)),
            "x_std": float(np.std(all_x, ddof=0) + 1e-8),
            "y_mean": float(np.mean(all_y)),
            "y_std": float(np.std(all_y, ddof=0) + 1e-8),
        }

        target_stats = {
            "mean_mean": float(np.mean(all_mean)),
            "mean_std": float(np.std(all_mean, ddof=0) + 1e-8),
            "std_mean": float(np.mean(all_std)),
            "std_std": float(np.std(all_std, ddof=0) + 1e-8),
        }

        return feature_stats, target_stats

    def _compute_physics_normalization(self) -> Dict[str, float]:
        """Compute mean/std for physics features from train set only."""
        if self.physics_mode == "vanilla":
            return {}

        all_vth = []
        all_l = []
        all_w = []
        all_sink_mean = []
        all_sink_std = []

        for gid in self.graph_ids:
            entry = self.dataset[gid]
            sensitivities = entry["physics_features"]["sensitivities"]
            analytical = entry["physics_features"]["analytical_ssta"]

            for gate_data in sensitivities.values():
                all_vth.append(gate_data["vth"])
                all_l.append(gate_data["l"])
                all_w.append(gate_data["w"])

            all_sink_mean.append(analytical["sink_mean"])
            all_sink_std.append(analytical["sink_std"])

        physics_stats = {
            "vth_mean": float(np.mean(all_vth)),
            "vth_std": float(np.std(all_vth, ddof=0) + 1e-8),
            "l_mean": float(np.mean(all_l)),
            "l_std": float(np.std(all_l, ddof=0) + 1e-8),
            "w_mean": float(np.mean(all_w)),
            "w_std": float(np.std(all_w, ddof=0) + 1e-8),
            "sink_mean_mean": float(np.mean(all_sink_mean)),
            "sink_mean_std": float(np.std(all_sink_mean, ddof=0) + 1e-8),
            "sink_std_mean": float(np.mean(all_sink_std)),
            "sink_std_std": float(np.std(all_sink_std, ddof=0) + 1e-8),
        }

        return physics_stats

    def _create_data_object(
        self,
        gid: str,
        entry: dict,
        feature_stats: Dict[str, float],
        target_stats: Dict[str, float],
        physics_stats: Dict[str, float] | None = None,
    ) -> Data:
        """Create a PyG Data object from a dataset entry."""
        gates = entry["graph"]["gates"]
        successors = entry["graph"]["successors"]

        node_names = sorted(gates.keys())
        n_nodes = len(node_names)
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        if self.physics_mode == "vanilla":
            num_node_features = 3
            x = np.zeros((n_nodes, num_node_features), dtype=np.float32)
            for name in node_names:
                gate = gates[name]
                idx = name_to_idx[name]
                x[idx, 0] = (gate["load_ff"] - feature_stats["load_mean"]) / feature_stats["load_std"]
                x[idx, 1] = (gate["x"] - feature_stats["x_mean"]) / feature_stats["x_std"]
                x[idx, 2] = (gate["y"] - feature_stats["y_mean"]) / feature_stats["y_std"]
            graph_physics = None
        else:
            num_node_features = 6
            x = np.zeros((n_nodes, num_node_features), dtype=np.float32)
            sensitivities = entry["physics_features"]["sensitivities"]
            for name in node_names:
                gate = gates[name]
                idx = name_to_idx[name]
                sens = sensitivities[name]
                x[idx, 0] = (gate["load_ff"] - feature_stats["load_mean"]) / feature_stats["load_std"]
                x[idx, 1] = (gate["x"] - feature_stats["x_mean"]) / feature_stats["x_std"]
                x[idx, 2] = (gate["y"] - feature_stats["y_mean"]) / feature_stats["y_std"]
                x[idx, 3] = (sens["vth"] - physics_stats["vth_mean"]) / physics_stats["vth_std"]
                x[idx, 4] = (sens["l"] - physics_stats["l_mean"]) / physics_stats["l_std"]
                x[idx, 5] = (sens["w"] - physics_stats["w_mean"]) / physics_stats["w_std"]

            if self.physics_mode == "tier_ab":
                analytical = entry["physics_features"]["analytical_ssta"]
                graph_physics = torch.tensor([
                    (analytical["sink_mean"] - physics_stats["sink_mean_mean"]) / physics_stats["sink_mean_std"],
                    (analytical["sink_std"] - physics_stats["sink_std_mean"]) / physics_stats["sink_std_std"],
                ], dtype=torch.float32).unsqueeze(0)  # Shape [1, 2] so batching gives [batch_size, 2]
            else:
                graph_physics = None

        # Edge index from successors (directed)
        edge_list = []
        for src, succs in successors.items():
            for dst in succs:
                edge_list.append([name_to_idx[src], name_to_idx[dst]])

        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Targets: normalized mean and std
        mean_val = entry["mc_labels"]["mean"]
        std_val = entry["mc_labels"]["std"]
        y = torch.tensor([
            (mean_val - target_stats["mean_mean"]) / target_stats["mean_std"],
            (std_val - target_stats["std_mean"]) / target_stats["std_std"],
        ], dtype=torch.float32).unsqueeze(0)

        # Store graph-level info for evaluation
        nrecon = len(entry["graph"]["reconvergence_points"])

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            y=y,
            graph_id=gid,
            n_nodes=n_nodes,
            nrecon=nrecon,
            original_mean=mean_val,
            original_std=std_val,
        )

        if graph_physics is not None:
            data.graph_physics = graph_physics

        return data

    def get_data(
        self,
        feature_stats: Dict[str, float],
        target_stats: Dict[str, float],
        physics_stats: Dict[str, float] | None = None,
    ) -> List[Data]:
        """Get list of PyG Data objects for this split."""
        data_list = []
        for gid in self.graph_ids:
            entry = self.dataset[gid]
            data = self._create_data_object(gid, entry, feature_stats, target_stats, physics_stats)
            data_list.append(data)

        return data_list

    def get_analytical_baseline(self) -> Dict:
        """Get analytical SSTA baseline errors for this split."""
        baseline = {}
        for gid in self.graph_ids:
            entry = self.dataset[gid]
            analytical = entry["physics_features"]["analytical_ssta"]
            mc_mean = entry["mc_labels"]["mean"]
            mc_std = entry["mc_labels"]["std"]

            ana_mean = analytical["sink_mean"]
            ana_std = analytical["sink_std"]

            baseline[gid] = {
                "analytical_mean": ana_mean,
                "analytical_std": ana_std,
                "mc_mean": mc_mean,
                "mc_std": mc_std,
                "mean_mae": abs(ana_mean - mc_mean),
                "mean_relative": abs(ana_mean - mc_mean) / max(abs(mc_mean), 1e-9),
                "std_mae": abs(ana_std - mc_std),
                "std_relative": abs(ana_std - mc_std) / max(abs(mc_std), 1e-9),
            }

        return baseline


def create_dataloaders(
    data_dir: str | Path = "data_generation/data",
    batch_size: int = 32,
    physics_mode: str = "vanilla",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict, Dict, Dict]:
    """Create train/val/test DataLoaders with proper normalization.

    Args:
        data_dir: Path to Stage 6A dataset
        batch_size: Batch size for DataLoaders
        physics_mode: 'vanilla' (3 features), 'tier_a' (6 node features), or 'tier_ab' (6 node + 2 graph features)

    Returns:
        train_loader, val_loader, test_loader, feature_stats, target_stats, physics_stats
    """
    train_dataset = GraphDataset(data_dir=data_dir, split="train", physics_mode=physics_mode)
    feature_stats, target_stats = train_dataset._compute_normalization()
    physics_stats = train_dataset._compute_physics_normalization()

    train_data = train_dataset.get_data(feature_stats, target_stats, physics_stats)
    val_data = GraphDataset(data_dir=data_dir, split="val", physics_mode=physics_mode).get_data(feature_stats, target_stats, physics_stats)
    test_data = GraphDataset(data_dir=data_dir, split="test", physics_mode=physics_mode).get_data(feature_stats, target_stats, physics_stats)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, feature_stats, target_stats, physics_stats
