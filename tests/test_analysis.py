import numpy as np
import pytest

from ssta.monte_carlo import run_branching_monte_carlo
from timing.graph import build_branching_graph_from_config
from foundations.config_loader import load_config


def test_run_branching_monte_carlo_shape():
    config = load_config("foundations/stage3_config.json")
    graph = build_branching_graph_from_config(
        timing_params=config.timing_params,
        dag=config.dag.successors,
        variation_params=config.variation_params,
    )
    result = run_branching_monte_carlo(graph=graph, n_samples=100)
    assert "critical_path_delay" in result
    assert result["critical_path_delay"].shape == (100,)
    assert result["critical_path_delay"].mean() > 0.0


def test_run_branching_monte_carlo_reproducibility():
    config = load_config("foundations/stage3_config.json")
    graph = build_branching_graph_from_config(
        timing_params=config.timing_params,
        dag=config.dag.successors,
        variation_params=config.variation_params,
    )
    r1 = run_branching_monte_carlo(graph=graph, n_samples=100, seed=42)
    r2 = run_branching_monte_carlo(graph=graph, n_samples=100, seed=42)
    assert np.isclose(r1["critical_path_delay"].mean(), r2["critical_path_delay"].mean())
    assert np.isclose(r1["critical_path_delay"].std(), r2["critical_path_delay"].std())
