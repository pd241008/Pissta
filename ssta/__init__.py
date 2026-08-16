"""
ssta — SSTA analysis engines.

Exports:
- monte_carlo: Vectorized Monte Carlo timing analysis
- analytical_ssta: Full analytical SSTA pipeline
- statistical_sum: Gaussian sum propagation
- statistics: Summary statistics utilities
"""

from .monte_carlo import run_branching_monte_carlo
from .analytical_ssta import compute_analytical_ssta
from .statistical_sum import gaussian_sum, propagate_path, covariance_between_paths
from .statistics import summarize_delay, critical_path_split

__all__ = [
    "run_branching_monte_carlo",
    "compute_analytical_ssta",
    "gaussian_sum",
    "propagate_path",
    "covariance_between_paths",
    "summarize_delay",
    "critical_path_split",
]
