# timing

Timing graph definition and delay models for VLSI SSTA.

## Contents

| File | Description |
|------|-------------|
| `README.md` | This file — overview of timing models |
| `__init__.py` | Package exports |
| `graph.py` | DAG topology, topological sort, gate coordinates |
| `delay.py` | Alpha-power delay model with geometry sensitivity |
| `clark_max.py` | Clark's Gaussian moment-matched MAX approximation |

## Interfaces

### `graph.py` — Timing Graph

```python
from timing.graph import TimingGraph, build_branching_graph_from_config

graph = build_branching_graph_from_config()
order = graph.topological_order()
```

### `delay.py` — Alpha-Power Delay

```python
from timing.delay import nominal_delay, delay_partials, compute_delay_moments

d_nom = nominal_delay(load_ff, gate_params, vth_nom, l_nom, w_nom)
partials = delay_partials(load_ff, gate_params, vth_nom, l_nom, w_nom)
moments = compute_delay_moments(timing_params, variation_params, process_moments)
```

### `clark_max.py` — Clark MAX Approximation

```python
from timing.clark_max import clark_max

mu_max, var_max = clark_max(mu1, var1, mu2, var2, rho)
```

## Delay Model

Alpha-power with geometry sensitivity:

```
d = C_load · Vdd / [k · (Vdd - Vth)^α] · (L/L_nom) / √(W/W_nom)
```

First-order Taylor partials:
- ∂d/∂Vth = d_nom · α / (Vdd − Vth_nom)
- ∂d/∂L = d_nom / L_nom
- ∂d/∂W = −d_nom / (2 · W_nom)

## Dependencies

- `config_loader` — loads timing and variation parameters
- `numpy` — numerical operations
