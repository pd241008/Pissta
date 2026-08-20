# ADR-007: Stage 6B GNN Baseline Architecture

> **Status:** Decided  
> **Date:** August 19, 2026  
> **Last updated:** August 20, 2026

## Context

Stage 6B required a vanilla (no physics-informed features) GNN baseline to:
- Establish a competent lower bound for Stage 6C comparison
- Validate that the Stage 6A dataset is learnable by standard message-passing
- Provide a reproducible reference implementation for ablation studies

Key constraints:
- Must use only raw graph structure and node features (load_ff, x, y)
- Must be trained and evaluated on the Stage 6A dataset
- Must produce metrics comparable to analytical SSTA and trivial baselines
- Must be reproducible across 3 seeds with stable results

## Options Considered

1. **GCN (Kipf & Welling)** — Classic spectral GNN, but requires normalized Laplacian and struggles with directed edges
2. **GAT (Veličković et al.)** — Attention-based, more expressive, but heavier and less stable on small graphs
3. **GraphSAGE (Hamilton et al.)** — Inductive, handles directed edges naturally, mean pooling readout, stable training

## Decision

We use **GraphSAGE** as the backbone with the following architecture:

| Component | Specification |
|-----------|---------------|
| Message passing | 3 × GraphSAGE layers (each: SAGEConv → BatchNorm1d → ReLU → Dropout) |
| Hidden dimension | 64 |
| Dropout | 0.15 (applied after input proj, each conv layer, and MLP head) |
| Readout | Mean pooling (`global_mean_pool`) |
| Output head | 2-layer MLP (64 → 64 → 2) |
| Total parameters | 29,698 |
| Optimizer | Adam (lr=1e-3) |
| Loss | MSE with mean reduction on train-normalized targets |
| Gradient clipping | max_norm=1.0 |
| Early stopping | Patience=20 on validation loss |
| Edge type | Directed edges following successors (source→sink) |

## Reasoning

### GraphSAGE over GCN/GAT
- **Directed edges**: GraphSAGE's neighbor aggregation works naturally with directed edges. GCN requires symmetric normalization. GAT is overkill for this dataset size.
- **Inductive**: GraphSAGE was designed for inductive learning on large graphs. Our test graphs are unseen during training.
- **Stability**: GraphSAGE trains reliably with default hyperparameters. GAT can diverge without careful attention head tuning.

### Architecture Details
- **3 layers**: With mean pooling, 3 layers give a receptive field of 3 hops. Sink nodes see ≤3 hops directly upstream — sufficient for our 6–14 gate graphs.
- **BatchNorm1d**: Applied after each SAGEConv. Stabilizes training by normalizing across the batch dimension for each feature.
- **Dropout at 4 sites**: Input projection, each of 3 conv layers, and MLP head. This is more aggressive than typical but prevents overfitting on small graphs.
- **Mean pooling**: Simple, deterministic, no learned parameters. Chosen over max/sum pooling for smooth gradient flow.

### Normalization Strategy
- **Per-target normalization**: mean and std are normalized separately using train-set statistics. This gives each target unit variance, preventing the ~760× real-unit weighting of std vs mean errors that would occur without normalization.
- **Feature normalization**: load_ff, x, y normalized with train-set mean/std. No physics-derived features (reserved for Stage 6C).

### Loss Function
- **MSE with mean reduction**: Default PyTorch behavior. Sum reduction would weight larger graphs more heavily; mean reduction gives equal weight per graph.
- **Frozen definition**: Stage 6C must use identical normalization, loss, and gradient clipping for fair comparison.

### Training Recipe
- **Adam lr=1e-3**: Standard choice, no learning rate scheduling needed with early stopping.
- **Gradient clipping max_norm=1.0**: Prevents gradient explosions during early training on noisy labels.
- **Early stopping patience=20**: ~20 epochs of no improvement before stopping. Combined with best checkpoint restoration, this prevents overfitting.

### Edge Direction
- **Directed (successors)**: Source nodes aggregate from their predecessors. This respects causality — a gate's delay depends on its inputs, not its loads.
- **Undirected ablation deferred**: With directed-only edges, source nodes receive no neighbor information and mean pooling dilutes sink embeddings. This is a known limitation documented for Stage 6C.

## Final Model State

| Property | Value |
|----------|-------|
| Architecture | 3× GraphSAGE (SAGEConv → BN → ReLU → Dropout) + MLP head |
| Parameters | 29,698 |
| Train / Val / Test | 1,397 / 296 / 307 graphs |
| Best epoch (mean) | 97.0 (0-based: 91–108) |
| Train time | ~40–85s per seed (CPU) |
| Mean delay MAE | 0.687 ± 0.004 (4.04% relative) |
| Std delay MAE | 0.0337 ± 0.0002 (4.67% relative) |
| Inference time | 0.08 ms/graph |
| vs Analytical SSTA | 0.95× mean MAE, 0.37× std MAE (3/3 seeds) |
| vs Trivial baseline | 79.7% mean improvement, 72.6% std improvement |

## Consequences

- **Stage 6C comparison target**: The 0.37× std MAE ratio vs analytical SSTA is the number Stage 6C must beat or match.
- **Physics feature gap**: Error degrades on high-nrecon graphs (nrecon≥5), suggesting physics-informed features (Stage 6C) could help with complex multi-reconvergence topologies.
- **Reproducibility**: All hyperparameters, normalization stats, and environment versions are frozen in `protocol` and `environment` fields of the results JSON.
- **Checkpoint provenance**: Checkpoints contain `.module.` BN keys from DataParallel-style training. The current model.py produces identical keys by design (BatchNorm wrapped in ModuleList). Round-trip verified.
- **Metadata provenance**: Per-graph metadata (nrecon, graph_id, mc_mean, mc_std) is read directly from `test_loader.dataset` inside `evaluate_model()`, not from a separately-constructed list. This eliminates the risk of silent misalignment between predictions and metadata if construction paths diverge.

## Alternatives Rejected

| Alternative | Reason |
|-------------|--------|
| GCN | Requires symmetric adjacency; directed edges would need conversion |
| GAT | Higher variance across seeds; attention overhead not justified for small graphs |
| Sum/max pooling | Sum pooling biases toward large graphs; max pooling loses information |
| Undirected edges | Would change receptive field semantics; deferred to ablation study |
| Learning rate scheduling | Early stopping already handles convergence; adds complexity |
