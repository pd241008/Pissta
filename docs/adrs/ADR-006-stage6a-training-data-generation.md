# ADR-006: Stage 6A Training Data Generation Architecture

> **Status:** Decided  
> **Date:** August 2026

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
- **Skip invalid graphs**: 652/2000 graphs skipped (sink with < 2 predecessors). These are pure chains with no MAX operation to learn — not useful for GNN training.
- **Label noise**: N=10,000 samples gives mean noise 0.06%, std noise 0.62% — well below the signal variance across graphs.

## Consequences

- **Dataset size**: 1,348 valid graphs from 2,000 generated (70/15/15 split: 943/202/203)
- **Generation cost**: ~26s total (~19ms/graph) — cheap at this scale
- **Downstream impact**: Stage 6B (vanilla GNN) and 6C (physics-informed) inherit this dataset. Splits are frozen in `splits.json` for fair comparison.
- **Physics features**: Analytical SSTA and per-gate sensitivities computed for all graphs, stored but not used by vanilla GNN
- **Future extension**: N-way splits, P99.87 labels, and larger graphs (n_gates > 12) can be added by modifying `graph_generator.py` parameters
