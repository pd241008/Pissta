"""
Stage 6B — Training loop for vanilla DAG-GNN baseline.

Includes early stopping, logging, and checkpoint management.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from model import VanillaDAGGNNSage


class EarlyStopping:
    """Early stopping with patience and best checkpoint restoration."""

    def __init__(self, patience: int = 20, delta: float = 0.0, checkpoint_path: str = "best_model.pt"):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Check if training should stop. Returns True if should stop."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                # Load best checkpoint
                model.load_state_dict(torch.load(self.checkpoint_path, weights_only=True))
        return self.early_stop


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(pred, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate for one epoch. Returns loss, predictions, targets."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(pred, batch.y)

        total_loss += loss.item()
        num_batches += 1

        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.vstack(all_preds) if all_preds else np.empty((0, 2))
    all_targets = np.vstack(all_targets) if all_targets else np.empty((0, 2))

    return avg_loss, all_preds, all_targets


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 20,
    checkpoint_path: str = "best_model.pt",
) -> Dict:
    """Train model with early stopping.

    Returns:
        Dict with training history and best epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch": [],
    }

    start_time = time.time()
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, _, _ = eval_epoch(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch"].append(epoch)

        if early_stopping(val_loss, model):
            print(f"  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time
    best_epoch = np.argmin(history["val_loss"])

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": min(history["val_loss"]),
        "train_time": train_time,
    }
