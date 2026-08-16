"""
timing — Timing graph and delay models.

Exports:
- graph: DAG topology, topological sort, gate coordinates
- delay: Alpha-power delay model with geometry sensitivity
- clark_max: Clark's Gaussian moment-matched MAX approximation
- tail_aware_max: Skew-normal 3-moment matched MAX approximation
"""

from .graph import TimingGraph, Gate, build_branching_graph_from_config
from .delay import nominal_delay, delay_partials, compute_delay_moments
from .clark_max import clark_max
from .tail_aware_max import (
    skew_normal_from_moments,
    skew_normal_pdf,
    skew_normal_cdf,
    skew_normal_ppf,
    sample_skew_normal,
    tail_aware_max,
)

__all__ = [
    "TimingGraph",
    "Gate",
    "build_branching_graph_from_config",
    "nominal_delay",
    "delay_partials",
    "compute_delay_moments",
    "clark_max",
    "skew_normal_from_moments",
    "skew_normal_pdf",
    "skew_normal_cdf",
    "skew_normal_ppf",
    "sample_skew_normal",
    "tail_aware_max",
]
