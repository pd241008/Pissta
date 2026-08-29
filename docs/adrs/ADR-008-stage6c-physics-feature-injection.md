# ADR-008: Stage 6C Physics-Informed DAG-GNN Feature Injection Strategy

> **Status:** Decided  
> **Date:** August 20, 2026  
> **Last updated:** August 28, 2026

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
- On the verified hardened run, feature-level injection was significantly *detrimental* to both tiers (see Final Model State). This strongly motivates the originally-deferred architectural physics-constraint route as the next attempt, rather than additional feature-level iterations.

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
| Vanilla | 29,698 | 0.4082 ± 0.0055 | 2.46% ± 0.04pp | 0.0188 ± 0.0004 | 2.56% ± 0.05pp | — |
| Tier A | 29,890 | 0.4478 ± 0.0220 | 2.69% ± 0.14pp | 0.0204 ± 0.0011 | 2.75% ± 0.15pp | Significantly **worse** (cluster CI [+0.0241, +0.0557]) |
| Tier A+B | 30,018 | 0.4714 ± 0.0095 | 2.97% ± 0.07pp | 0.0216 ± 0.0003 | 2.99% ± 0.04pp | Significantly **worse** (cluster CI [+0.0147, +0.1123]) |
| No-GNN Residual MLP | 194 | 0.6553 ± 0.0031 | 4.21% ± 0.01pp | 0.0340 ± 0.0001 | 4.73% ± 0.01pp | Far worse than all GNN variants |

**Key finding (verified hardened run, 3 seeds, N=304 test graphs):** On the corrected dataset (post-B1 fix), **both physics tiers are significantly WORSE than vanilla** on mean MAE:
- Tier A: +0.040, cluster-bootstrap CI [+0.0241, +0.0557] (excludes 0)
- Tier A+B: +0.063, cluster-bootstrap CI [+0.0147, +0.1123] (excludes 0)

Values here are the aggregates recomputed from the committed `gnn_baseline/results/stage6c_results.json` and independently re-derived by `verify_stage6c.py` (per-graph sha256-consistent, lockstep exact, max diff 0.0). This supersedes the earlier "corrected re-run" table below (which read as a null because its CIs crossed zero at the interim dataset). Note the earlier significant improvement (Tier A+B -21%) was an artifact of the mislabeled dataset (B1) plus its different graph-complexity mix.

**Earlier corrected re-run (interim, superseded — kept for the record):** the first post-B1-numbers re-run gave Vanilla 0.4263, Tier A 0.4340 (CI [-0.0113, +0.0272]), Tier A+B 0.4307 (CI [-0.0450, +0.0523]) — point estimates slightly favoring vanilla, CIs crossing zero. That interim dataset was regenerated concurrently with the hardening work and is not graph-for-graph identical to the final one; the hardened full run (table above) is the authoritative result.

## Consequences

- **Tier A is redundant, not insufficient**: Mechanism diagnostics reveal ∂d/∂Vth, ∂d/∂L, ∂d/∂W are all algebraically proportional to the existing `load_ff` feature (fixed global ratios; cross-ratios verified: ∂Vth/∂L = 97.5, ∂W/∂L = −0.25 to ~14 decimal places; corr(r=1.00/1.00/−1.00)). Tier A carries zero information beyond `load_ff` by construction for any graph. Future physics features must provide signal not already captured by geometry.
- **Feature-level physics injection is significantly detrimental on the verified run**: both tiers land materially *worse* than vanilla (CI excludes 0). Even the null-framing of the interim re-run has been overtaken by the hardened result. Feature injection as designed does not help; a redesign is required before any further feature-level attempt (§9 of the context file), or a pivot to architectural constraints (option 2 in the original plan).
- **Tier A+B's analytical features are highly predictive but not learnable by the GNN**: analytical_ssta.sink_mean is ~97.6% correlated with MC mean label (verified OLS slope β=0.9267 within the |β−1|≤0.1 gate). The GNN cannot extract additional signal from this near-perfect feature in the current architecture.
- **No-GNN baseline establishes the ceiling**: The ResidualMLP (0.6553 ± 0.0031) and OLS floor (~0.65) show scalar methods plateau around 0.65. All GNN variants beat this substantially (0.408–0.471), proving graph structure matters — but the physics features don't push the GNN beyond what vanilla already achieves.
- **Lockstep verification**: 6C vanilla run matches 6B baseline exactly (per-seed mean_mae identical to 0.0 max diff across all 3 seeds), confirming the frozen baseline is untouched and that the determination is now real (post-hardening CUDA determinism).
- **Diagnostic for the next round**: train vanilla-GNN-minus-coordinates (drop x,y, keep load_ff only) to test whether the vanilla GNN is already implicitly reconstructing geometry/physics from raw features — if accuracy barely degrades, the null/worse result is mechanistically explained (explicit physics features are redundant); if it degrades a lot, further investigation is warranted.

## Alternatives Rejected

| Alternative | Reason |
|-------------|--------|
| Architectural constraints | Deferred to follow-up; feature injection is cleaner first test |
| Loss-based penalties | Adds hyperparameters; feature injection isolates the physics effect |
| Node-level analytical features (AT_mean per gate) | Too correlated with labels at node level; would create similar leakage |
| Removing analytical features entirely | Would test "pure physics learning" but loses the legitimate correction capability |

---

## ADR-008 Addendum (2026-08-29): Redesigned features, still no headroom

**Redesign:** Tier A replaced [∂d/∂Vth, ∂d/∂L, ∂d/∂W] (algebraically collinear
with load_ff, R²≈1.0) with var_d/load_ff² sourced from the dataset's stored
delay_var (NOT recomputed from the post-08-22 process-moment formula, which
would silently mismatch what the model was trained on — caught in Step 1).
R²(vs load_ff, x, y) = 0.088 — genuinely non-redundant.

Tier B moved from one post-pool sink-level (mean,std) scalar pair to
per-node AT_mean/AT_var, injected as node features so they pass through
message-passing instead of bypassing it.

**Result (3 seeds, paired cluster-bootstrap CI, n=304 graphs × 3 seeds):**

| Variant | Mean MAE | Δ vs vanilla | CI | Verdict |
|---|---|---|---|---|
| Vanilla | 0.4082 | — | — | — |
| Tier A (redesigned) | 0.4562 | +0.048 | [+0.019, +0.078] | sig. worse |
| Tier A+B (redesigned) | 0.4010 | −0.007 | [−0.047, +0.032] | ns (wide) |
| Tier B only | 0.4166 | +0.009 | [−0.027, +0.043] | ns (wide) |

Per-seed Tier-B-only deltas (−0.014, −0.000, +0.039) disagree in sign,
independent evidence against a real effect beyond the wide CI.

**Conclusion:** fixing collinearity (Tier A) and leakage (Tier B) improved
the *diagnostic* cleanliness of the features but did not produce a result
distinguishable from vanilla in either direction. Tier A alone still hurts
(non-redundant ≠ useful — it can still just be optimizer-costing noise).
Tier B alone and A+B are statistically tied with vanilla, with CIs too wide
(3-seed) to rule out modest effects either way.

**Caveat on the "ns (wide)" verdicts:** all three CIs are wide (0.078–0.10),
which stems from only 3 seeds. These are genuine null results but
low-precision ones — the data is compatible with anything from "meaningfully
worse" to "moderately better" for Tier A+B and Tier B-only. Read "ns" as
"not proven different," not "proven equivalent." Tier A alone remains
sig. worse (its CI excludes 0).

**Status:** feature-level physics injection, across four attempts now
(original Tier A/A+B, redesigned Tier A/A+B, Tier B only), has not
produced a result better than vanilla. Supersedes ADR-008's original
2026-08-28 update. Next candidate per next-steps: architectural
MAX-biased aggregation (item 3), not further feature redesign.
