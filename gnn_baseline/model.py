"""
Stage 6B/6C — DAG-GNN model variants.

VanillaDAGGNNSage: Stage 6B baseline, 3-dim node features (load_ff, x, y)
PhysicsInformedDAGGNNSage: Stage 6C physics-informed variants
  - tier_a: 6-dim node features (adds vth, l, w sensitivities)
  - tier_ab: 6-dim node features + 2 graph-level analytical SSTA features
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, global_mean_pool, BatchNorm


class VanillaDAGGNNSage(torch.nn.Module):
    """Vanilla DAG-GNN using GraphSAGE layers (Stage 6B baseline, frozen)."""

    def __init__(
        self,
        num_node_features: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.15,
        num_outputs: int = 2,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.input_proj = torch.nn.Linear(num_node_features, hidden_dim)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, num_layers=1))
            self.bns.append(BatchNorm(hidden_dim))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_outputs),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is None:
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            x = global_mean_pool(x, batch)

        x = self.mlp(x)
        return x


class PhysicsInformedDAGGNNSage(torch.nn.Module):
    """Physics-informed DAG-GNN with position-varying delay-variance feature (Stage 6C).

    Args:
        tier: 'a' for node-level var_d/load_ff² only (4-dim node features),
              'ab' for node-level + graph-level analytical SSTA features
        num_node_features: dimension of per-node input features (4 for tier_a, 4 for tier_ab)
    """

    def __init__(
        self,
        tier: Literal["a", "ab"] = "a",
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.15,
        num_outputs: int = 2,
        num_node_features: int = 4,
    ):
        super().__init__()

        assert tier in ("a", "ab"), f"tier must be 'a' or 'ab', got {tier}"

        self.tier = tier
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Tier A: 4 node features (load_ff, x, y, var_d_per_load_ff_sq)
        # Tier B: same node features + 2 graph-level features
        graph_physics_dim = 2 if tier == "ab" else 0

        self.input_proj = torch.nn.Linear(num_node_features, hidden_dim)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, num_layers=1))
            self.bns.append(BatchNorm(hidden_dim))

        mlp_input_dim = hidden_dim + graph_physics_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_outputs),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        graph_physics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is None:
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            x = global_mean_pool(x, batch)

        if self.tier == "ab" and graph_physics is not None:
            x = torch.cat([x, graph_physics], dim=1)

        x = self.mlp(x)
        return x
