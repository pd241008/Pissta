# ADR-008: Stage 6C Physics-Informed DAG-GNN Feature Injection Strategy

> **Status:** Decided  
> **Date:** August 20, 2026  
> **Last updated:** August 24, 2026

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

**Important:** Tier A+B's graph-level analytical features are highly correlated (r ≈ 0.976) with MC labels. The ablation tests whether the GNN adds value *on top of* these near-perfect features, not whether physics features are useful in isolation.

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
| Vanilla | 29,698 | 0.4263 ± 0.0063 | 2.57% ± 0.04% | 0.0194 ± 0.0003 | 2.64% ± 0.04% | — |
| Tier A | 29,890 | 0.4340 ± 0.0299 | 2.59% ± 0.17% | 0.0197 ± 0.0012 | 2.64% ± 0.14% | Not significant (bootstrap CI [-0.0113, +0.0272]) |
| Tier A+B | 30,018 | 0.4307 ± 0.0147 | 2.66% ± 0.10% | 0.0199 ± 0.0007 | 2.72% ± 0.10% | Not significant (bootstrap CI [-0.0450, +0.0523]) |
| No-GNN Residual MLP | 194 | 0.5786 ± 0.0010 | 3.68% ± 0.01% | 0.0331 ± 0.0001 | 4.60% ± 0.01% | Far worse than all GNN variants |

**Key finding:** On the corrected dataset (post-B1 fix), neither Tier A nor Tier A+B is statistically distinguishable from vanilla. The point estimates actually slightly favor vanilla (Tier A +0.0077, Tier A+B +0.0044 mean MAE delta), but both CIs cross zero — physics feature injection has no significant effect in either direction.

The earlier significant improvement (Tier A+B -21%) was an artifact of the mislabeled dataset (B1) plus its different graph-complexity mix.

## Consequences

- **Tier A is redundant, not insufficient**: Mechanism diagnostics reveal ∂d/∂Vth is perfectly correlated (r=1.00) with the existing load_ff feature. Tier A adds no new information. Future physics features must provide signal not already captured by geometry.
- **Neither tier shows statistically significant improvement on corrected data**: The honest headline is a null result — vanilla remains the best point estimate and requires no extra features to achieve it. The CIs are wide enough that meaningful effects in either direction cannot be ruled out with only 3 seeds.
- **Tier A+B's analytical features are highly predictive but not learnable by the GNN**: analytical_ssta.sink_mean is ~97.6% correlated with MC mean label. The GNN cannot extract additional signal from this near-perfect feature in the current architecture.
- **No-GNN baseline establishes the ceiling**: The ResidualMLP (0.5786 ± 0.0010) and OLS floor (~0.58) show that scalar methods plateau around 0.58. All GNN variants beat this substantially (0.426–0.434), proving graph structure matters — but the physics features don't push the GNN beyond what vanilla already achieves.
- **Lockstep verification**: 6C vanilla run matches 6B baseline exactly (per-seed mean_mae identical to 1e-6), confirming the frozen baseline is untouched.

## Alternatives Rejected

| Alternative | Reason |
|-------------|--------|
| Architectural constraints | Deferred to follow-up; feature injection is cleaner first test |
| Loss-based penalties | Adds hyperparameters; feature injection isolates the physics effect |
| Node-level analytical features (AT_mean per gate) | Too correlated with labels at node level; would create similar leakage |
| Removing analytical features entirely | Would test "pure physics learning" but loses the legitimate correction capability |
