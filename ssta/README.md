# ssta

SSTA analysis engines — both Monte Carlo reference and analytical approximations.

## Contents

| File | Description |
|------|-------------|
| `README.md` | This file — overview of SSTA engines |
| `__init__.py` | Package exports |
| `monte_carlo.py` | Vectorized Monte Carlo timing analysis |
| `analytical_ssta.py` | Full analytical SSTA pipeline (Stage 4) |
| `statistical_sum.py` | Gaussian sum propagation for path delays |
| `statistics.py` | Summary statistics (mean, std, percentiles) |

## Interfaces

### `monte_carlo.py` — Monte Carlo Reference

```python
from ssta.monte_carlo import run_branching_monte_carlo

results = run_branching_monte_carlo(
    n_samples=100_000,
    seed=42,
    variation_params=config.variation_params,
    gate_params=config.timing_params,
    graph=graph,
)
# Returns: {"L_nm", "W_nm", "Vth_v", "delays", "arrival_times",
#           "critical_path_delay", "path_labels", "gate_names"}
```

### `analytical_ssta.py` — Analytical Pipeline

```python
from ssta.analytical_ssta import compute_analytical_ssta

result = compute_analytical_ssta(config)
# Returns: {"mean", "std", "p95", "p99", "p99_87", "path_moments",
#           "clark_max", "final_sum"}
```

### `statistical_sum.py` — Gaussian Sums

```python
from ssta.statistical_sum import gaussian_sum, propagate_path, covariance_between_paths

mu_z, var_z = gaussian_sum(mu_x, var_x, mu_y, var_y, cov_xy)
mu_path, var_path = propagate_path(path, mean_d, var_d, cov_d, idx)
cov = covariance_between_paths(path_a, path_b, cov_d, idx)
```

## Analysis Flow

1. **Monte Carlo:** Sample L/W/Vth → compute delays → propagate AT → MAX at reconvergence
2. **Analytical:** Compute moments → linearize delays → sum along paths → Clark MAX → final sum

## Dependencies

- `variation` — process variation models
- `timing` — timing graph and delay models
- `config_loader` — configuration
