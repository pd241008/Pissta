# variation

Process variation modeling for VLSI SSTA.

## Contents

| File | Description |
|------|-------------|
| `README.md` | This file — overview of variation modeling |
| `__init__.py` | Package exports |
| `sampler.py` | Monte Carlo correlated variation sampler |
| `analytical.py` | Closed-form analytical variation moments |

## Interfaces

### `sampler.py` — Monte Carlo Sampling

```python
from variation.sampler import sample_correlated_process

samples = sample_correlated_process(
    n_samples=10_000,
    params=variation_params,
    rng=np.random.default_rng(42),
)
# Returns: {"L_nm": (N,6), "W_nm": (N,6), "Vth_v": (N,6)}
```

### `analytical.py` — Closed-Form Moments

```python
from variation.analytical import compute_process_moments

moments = compute_process_moments(variation_params)
# Returns: {"mean_l", "mean_w", "mean_vth", "var_l", "var_w", "var_vth",
#           "cov_l", "cov_w", "cov_vth", "names", "idx"}
```

## Variation Components

1. **Inter-die:** Global scalar shift per sample (σ_inter_L, σ_inter_W, σ_inter_Vth)
2. **Spatial intra-die:** Distance-based exponential covariance Σ_ij = σ² exp(-d_ij/λ)
3. **Random mismatch:** Per-gate Pelgrom-style noise (σ²_Vth = A²/(WL) + S²D²)

## Dependencies

- `config_loader` — loads `stage3_config.json` parameters
- `numpy` — linear algebra and random sampling
