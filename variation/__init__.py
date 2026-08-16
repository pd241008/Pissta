"""
variation — Process variation modeling.

Exports:
- sampler: Monte Carlo correlated variation sampler
- analytical: Closed-form analytical variation moments
"""

from .sampler import sample_correlated_process
from .analytical import compute_process_moments

__all__ = ["sample_correlated_process", "compute_process_moments"]
