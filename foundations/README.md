# foundations

Core theory, parameters, and frozen configuration for the VLSI SSTA project.

## Contents

| File | Description |
|------|-------------|
| `README.md` | This file — overview of foundations |
| `stage3_config.json` | Frozen config: DAG topology, variation params, timing constants |
| `stage3_config_asymmetric.json` | Asymmetric DAG config for sanity checks |

## Key Concepts

### Process Variation Model

Each gate's process parameters (L, W, Vth) are modeled as the sum of three zero-mean components:

```
X_i = X_nom + ΔX_inter + ΔX_spatial,i + ΔX_random,i
```

- **Inter-die:** One global scalar per Monte Carlo sample, applied to all gates equally
- **Spatial intra-die:** Correlated via distance-based exponential covariance Σ_ij = σ² exp(-d_ij/λ)
- **Random mismatch:** Independent per-gate Pelgrom-style noise for Vth

### Timing Model

Alpha-power delay with geometry sensitivity:

```
d = C_load · Vdd / [k · (Vdd - Vth)^α] · (L/L_nom) / √(W/W_nom)
```

### DAG Topology

```
              ┌── G2 ── G4 ──┐
Input → G1 ──┤                ├── G6 → Output
              └── G3 ── G5 ──┘
```

## Usage

All modules load parameters from `stage3_config.json` via `config_loader.py`. Do not hardcode parameters in individual modules.
