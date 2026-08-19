"""
Stage 6B — Vanilla DAG-GNN baseline model.

Architecture:
- 3 GraphSAGE layers with batch normalization and dropout
- Mean pooling for graph-level readout
- 2-layer MLP head for dual regression (mean, std)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, global_mean_pool, BatchNorm


class VanillaDAGGNNSage(torch.nn.Module):
    """Vanilla DAG-GNN using GraphSAGE layers."""

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

        # Input projection
        self.input_proj = torch.nn.Linear(num_node_features, hidden_dim)

        # GraphSAGE layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, num_layers=1))
            self.bns.append(BatchNorm(hidden_dim))

        # Output head
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
        """Forward pass.

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph-level predictions [batch_size, num_outputs]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Message passing layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        if batch is None:
            # Single graph
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            x = global_mean_pool(x, batch)

        # Output head
        x = self.mlp(x)

        return x
