import numpy as np
import pytest

from timing.graph import Gate, TimingGraph, alpha_power_delay
from foundations.config_loader import TimingParams, VariationParams


def test_alpha_power_delay_positive():
    tp = TimingParams()
    vp = VariationParams()
    vth = np.array([0.5])
    load = np.array([1.0])
    l = np.array([45.0])
    w = np.array([90.0])
    delay = alpha_power_delay(vth, load, tp, l, w, vp.l_nom_nm, vp.w_nom_nm)
    assert delay[0] > 0.0


def test_timing_graph_sources_sinks():
    gates = {
        "g0": Gate("g0", 1.0),
        "g1": Gate("g1", 1.0),
        "g2": Gate("g2", 1.0),
    }
    successors = {"g0": ["g1", "g2"], "g1": [], "g2": []}
    graph = TimingGraph(gates=gates, successors=successors)
    assert graph.sources() == ["g0"]
    assert sorted(graph.sinks()) == ["g1", "g2"]


def test_timing_graph_topological_order():
    gates = {
        "g0": Gate("g0", 1.0),
        "g1": Gate("g1", 1.0),
        "g2": Gate("g2", 1.0),
    }
    successors = {"g0": ["g1"], "g1": ["g2"], "g2": []}
    graph = TimingGraph(gates=gates, successors=successors)
    order = graph.topological_order()
    assert order == ["g0", "g1", "g2"]
