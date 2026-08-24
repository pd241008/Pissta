import os, sys, json, hashlib, shutil
from pathlib import Path

os.environ["VLSI_SMOKE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
torch.use_deterministic_algorithms(True)

from gnn_baseline.run_stage6b import set_seed, run_single_seed
from dataset import create_dataloaders, GraphDataset

def hash_dict(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

data_dir = REPO_ROOT / "data_generation" / "data"
device = torch.device("cpu")

print("=== Run 1 ===")
set_seed(42)
train_loader, val_loader, test_loader, feature_stats, target_stats, physics_stats = create_dataloaders(
    data_dir=data_dir, batch_size=32, physics_mode="vanilla"
)
test_dataset = GraphDataset(split="test", data_dir=data_dir)
result1 = run_single_seed(42, train_loader, val_loader, test_loader, test_dataset, feature_stats, target_stats, device, REPO_ROOT / "gnn_baseline" / "checkpoints")

print("=== Run 2 ===")
set_seed(42)
train_loader2, val_loader2, test_loader2, feature_stats2, target_stats2, physics_stats2 = create_dataloaders(
    data_dir=data_dir, batch_size=32, physics_mode="vanilla"
)
test_dataset2 = GraphDataset(split="test", data_dir=data_dir)
result2 = run_single_seed(42, train_loader2, val_loader2, test_loader2, test_dataset2, feature_stats2, target_stats2, device, REPO_ROOT / "gnn_baseline" / "checkpoints")

h1 = hash_dict(result1)
h2 = hash_dict(result2)
print(f"\nRun1 hash: {h1}")
print(f"Run2 hash: {h2}")
print(f"Bit-identical: {h1 == h2}")
