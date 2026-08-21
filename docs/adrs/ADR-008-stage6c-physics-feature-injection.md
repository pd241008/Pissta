# ADR-008: Stage 6C Physics-Informed DAG-GNN Feature Injection Strategy

> **Status:** Decided  
> **Date:** August 20, 2026  
> **Last updated:** August 20, 2026

## Context

Stage 6C needed to test whether injecting physics features into the Stage 6B vanilla GNN baseline improves delay prediction accuracy. The key requirements were:
- Maintain identical training protocol (frozen hyperparameters, splits, seeds)
- Isolate the effect of physics features from architecture changes
- Produce an honest, defensible claim about whether "physics-informed" earns its name

Key constraint: The only thing that changes between runs is the input feature set. If architecture size changes simultaneously, any improvement could be attributed to "more parameters" rather than physics informativeness.

## Options Considered

1. **Architectural physics constraints** — Modify the GNN architecture to enforce physical laws (e.g., monotonicity, causality). Powerful but complex, hard to ablate.
2. **Loss-based physics penalties** — Add physics-derived terms to the loss function. Flexible but introduces new hyperparameters and tuning burden.
3. **Feature-level physics injection** — Append physics-computed features to node/graph inputs. Fastest to test, cleanest ablation, minimal code change.

## Decision

We use **feature-level physics injection** with a 3-way ablation:

| Tier | Features | Description |
|------|----------|-------------|
| Vanilla | 3-dim: load_ff, x, y | Stage 6B baseline (locked) |
| Tier A | 6-dim: + vth_sens, l_sens, w_sens | Per-gate linearized delay sensitivities |
| Tier A+B | 6-dim node + 2-dim graph | + analytical_ssta.sink_mean, sink_std |

## Reasoning

### Why Tier A first
- **Fastest to test**: No architecture changes needed, just widen the input projection from 3→6.
- **Cleanest ablation**: Identical architecture, only input features change.
- **Genuinely physics-informed**: The GNN sees local delay sensitivity to process variations, not just geometry.

### Why Tier B separately
- **Different claim**: Tier B is closer to "GNN corrects the analytical baseline" than "GNN learns physics." This is a weaker but still useful claim — worth testing separately so we know which one is doing the work.
- **Graph-level, not node-level**: analytical_ssta results are global properties, not per-gate features. Concatenating to the pooled embedding (after mean pooling) is the correct placement.

### Why not architectural/loss-based constraints yet
- The user's plan explicitly states: "implement feature-level physics injection first (fastest to test, cleanest ablation), treat architectural/loss-based physics constraints as follow-ups only if #1 shows a real effect."
- Tier A showed no significant effect. Tier A+B showed large effect, but primarily driven by the graph-level analytical feature (0.98 correlation with label). This suggests the "physics learning" is happening at the graph level, not the node level.

### Normalization strategy
- All physics features normalized with train-set mean/std, same discipline as raw features.
- Sensitivities have very different scales than load_ff/x/y (vth: 3.4–6.7, l: 0.03–0.07, w: -0.02 to -0.008). Skipping normalization would let one feature dominate purely by scale.

### Model design
- **VanillaDAGGNNSage**: Kept exactly as Stage 6B (frozen baseline, untouched).
- **PhysicsInformedDAGGNSSage**: New class with `tier` parameter.
  - Tier A: `num_node_features=6`, same architecture as vanilla.
  - Tier A+B: `num_node_features=6`, `mlp_input_dim=hidden_dim+2`, graph_physics concatenated after pooling.
- This ensures Stage 6B checkpoints and code remain exactly as verified.

## Final Model State

| Config | Parameters | Mean MAE | Mean Rel. | Std MAE | Std Rel. | vs Vanilla |
|--------|-----------|----------|-----------|---------|----------|------------|
| Vanilla | 29,698 | 0.687 ± 0.004 | 4.04% ± 0.03% | 0.0337 ± 0.0002 | 4.67% ± 0.06% | — |
| Tier A | 29,890 | 0.685 ± 0.005 | 3.98% ± 0.05% | 0.0341 ± 0.0009 | 4.72% ± 0.14% | Not significant (bootstrap CI [-0.025, +0.020]) |
| Tier A+B | 30,018 | 0.542 ± 0.008 | 3.39% ± 0.06% | 0.0321 ± 0.0003 | 4.53% ± 0.04% | Significant (bootstrap CI [-0.192, -0.100]) |
| No-GNN Residual MLP | ~1K | 0.943 | 6.27% | 0.0431 | 6.21% | Far worse than Tier A+B |

## Consequences

- **Tier A is redundant, not insufficient**: Mechanism diagnostics reveal ∂d/∂Vth is perfectly correlated (r=1.00) with the existing load_ff feature. Tier A adds no new information. Future physics features must provide signal not already captured by geometry.
- **Tier B works, and the GNN contributes meaningfully**: The no-GNN residual baseline (MLP on sink_mean, sink_std, n_gates) achieves mean MAE 0.943 vs Tier A+B's 0.542. Graph structure matters — this is not just residual correction.
- **Tier B's gain comes with a caveat**: The analytical sink_mean feature is 97.7% correlated with the MC mean label. The improvement is partly "the analytical baseline was already decent" rather than "the GNN learned physics from node features." This is legitimate but should be stated honestly.
- **nrecon-dependent improvement**: Tier A+B's improvement is larger on complex topologies (36–47% for nrecon≥5 vs 10% for nrecon=2). This is the most interesting finding — the GNN is correcting analytical SSTA's linearization error where it compounds most. Caveat: nrecon confounds with n_gates, and n=10 at nrecon=6 limits confidence.
- **Stage 6C comparison target**: The 0.542 mean MAE is the new number to beat. The 21% improvement over vanilla is substantial but comes with the analytical feature caveat.
- **Lockstep verification**: 6C vanilla run matches 6B baseline exactly (per-seed mean_mae identical to 1e-6), confirming the frozen baseline is untouched.

## Alternatives Rejected

| Alternative | Reason |
|-------------|--------|
| Architectural constraints | Deferred to follow-up; feature injection is cleaner first test |
| Loss-based penalties | Adds hyperparameters; feature injection isolates the physics effect |
| Node-level analytical features (AT_mean per gate) | Too correlated with labels at node level; would create similar leakage |
| Removing analytical features entirely | Would test "pure physics learning" but loses the legitimate correction capability |
