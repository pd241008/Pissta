# experiments

Experiment orchestration scripts, reproducibility checks, and sanity tests.

## Contents

| File | Description |
|------|-------------|
| `README.md` | This file — overview of experiments |
| `__init__.py` | Package exports |
| `run_stage3.py` | Reference MC run (N=100k, seed 42) |
| `run_stage4.py` | Analytical SSTA pipeline execution |
| `run_stage4b_hybrid.py` | Empirical+Clark gap decomposition |
| `reproducibility_check.py` | Multi-seed stability validation |
| `asymmetric_sanity_check.py` | AT/argmax logic validation |
| `sanity_checks.py` | Diagnostic N=10k comparison vs chain |

## Running Experiments

```bash
# Reference MC run (N=100k, seed 42) — locks ground truth
python -m experiments.run_stage3

# Analytical SSTA (Stage 4)
python -m experiments.run_stage4

# Hybrid empirical+Clark decomposition (Stage 4b)
python -m experiments.run_stage4b_hybrid

# Reproducibility check across seeds 42, 123, 999
python -m experiments.reproducibility_check

# Asymmetric sanity check (extra gate on one branch)
python -m experiments.asymmetric_sanity_check

# Diagnostic 10k comparison (not ground truth)
python -m experiments.sanity_checks
```

## Naming Convention

- `run_stage{N}.py` — Main orchestration for stage N
- `run_stage{N}b_*.py` — Sub-experiments or variants of stage N
- `*_check.py` — Validation and sanity checks
- `*_reproducibility.py` — Multi-seed or multi-config stability tests

## Dependencies

- `ssta` — Analysis engines
- `variation` — Variation models
- `timing` — Timing graph and delay
- `config_loader` — Configuration loading
