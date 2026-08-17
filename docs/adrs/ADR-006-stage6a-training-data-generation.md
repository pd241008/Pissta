# ADR-006: Stage 6A Training Data Generation Architecture

> **Status:** Decided  
> **Date:** August 2026  
> **Last updated:** August 17, 2026

## Context

Stage 6A needed a validated dataset of (DAG structure → MC-labeled delay statistics) pairs for GNN training. The key requirements were:
- Structurally diverse but valid timing DAGs
- Clean train/val/test split by graph
- MC labels with physics-derived features
- Generalizable pipeline that works with arbitrary DAGs, not just hardcoded Stage 3 structures

## Options Considered

1. **Reuse Stage 3's hardcoded G1–G6 graphs with perturbation** — Simple but limits structural diversity; GNN would only learn to interpolate between 6 topologies
2. **Random graph generation with post-hoc filtering** — Maximum diversity but risks generating invalid/physically implausible graphs
3. **Structured random generation with validation at each step** — Balanced approach: procedural generation ensures validity, randomization ensures diversity

## Decision

We use **structured random generation with validation at each step**:
- `graph_generator.py`: Recursive DAG construction with immediate validation
- `monte_carlo.py` generalized: Accepts arbitrary `TimingGraph` instead of hardcoded structure
- `analytical_ssta_arbitrary.py`: Iterative Clark MAX for arbitrary DAGs
- Physical safeguards: Vth clipping, Pelgrom distance capping, alpha-power delay floor

## Reasoning

- **Generator design**: Start with source→sink, repeatedly subdivide edges or add split-reconverge blocks. Validate with topological sort after each modification. This ensures every generated graph is a valid timing DAG.
- **Binary splits only**: Matches Stage 3 structure; N-way splits deferred to later extension
- **Per-graph coordinates**: Spatial covariance model requires meaningful 2D layout. We assign coordinates based on topological level × branch position.
- **Physical bounds**: Vth sampling can produce values near Vdd, causing alpha-power delay singularity. We cap Pelgrom distance at 5μm, clip Vth to `[vth_nom - 5σ, Vdd - 50mV]`, and set alpha-power floor at 0.1V.
- **Skip invalid graphs**: Eliminated by construction. The generator forces the first operation to be a split-reconverge on `source→sink`, guaranteeing the sink has ≥2 predecessors for every generated graph. All 2,000 generated graphs are valid — 0 skips, no downstream filter needed.
- **Label noise**: N=10,000 samples gives mean noise 0.05%, std noise 0.75% — well below the signal variance across graphs.
- **Timing consistency**: `total_generation_time_s` measures only the per-graph MC+physics loop (not graph generation overhead), so it divided by `mean_time_per_graph_s` equals exactly 2000.

## Final Dataset State

| Property | Value |
|----------|-------|
| Total generated | 2,000 |
| Valid graphs | 2,000 (0 skips) |
| Train / Val / Test | 1,397 / 296 / 307 |
| Gates per graph | 6–14 (mean 9.97) |
| Reconvergence points | 2–8 (mean 2.85) |
| Mean delay | 7.95–28.94 (median 15.63) |
| Std delay | 0.37–1.15 (median 0.71) |
| Label noise (mean) | 0.05% ± 0.04% |
| Label noise (std) | 0.75% ± 0.55% |
| Generation time | ~33s total (~16ms/graph) |
| Timing consistency | `total / mean = 2000` exactly |

## Consequences

- **Downstream impact**: Stage 6B (vanilla GNN) and 6C (physics-informed) inherit this dataset. Splits are frozen in `splits.json` for fair comparison.
- **Physics features**: Analytical SSTA and per-gate sensitivities computed for all graphs, stored but not used by vanilla GNN
- **Future extension**: N-way splits, P99.87 labels, and larger graphs (n_gates > 14) can be added by modifying `graph_generator.py` parameters
