"""D1 regression tests for compute_analytical_ssta_arbitrary (2026-09-29 fix).

The corpus pipeline read its delay-moment arrays with topological indices
while compute_delay_moments lays them out in gate_coords key order — every
gate received another gate's (mean, var, cov) entries whenever the two
orders diverged (0/50 sampled ID graphs had them coincide). These tests pin
the name-keyed readback contract: results must be invariant to the ORDER of
the gate_coords dict and per-gate values must follow the gate NAME.
"""

import pytest

from data_generation.analytical_ssta_arbitrary import compute_analytical_ssta_arbitrary
from foundations.config_loader import TimingParams, VariationParams
from ssta.statistical_sum import propagate_path
from timing.delay import compute_delay_moments, nominal_delay
from timing.graph import Gate, TimingGraph
from variation.analytical import compute_process_moments

# Reconvergent DAG exercising both SUM and MAX propagation: g3 fans back in.
SUCCESSORS = {"g0": ["g1", "g2"], "g1": ["g3"], "g2": ["g3"], "g3": ["g4"], "g4": []}
TOPO_ORDER = ["g0", "g1", "g2", "g3", "g4"]
PERMUTED_ORDER = ["g3", "g0", "g4", "g2", "g1"]

LOADS = {"g0": 1.0, "g1": 1.1, "g2": 1.2, "g3": 1.3, "g4": 1.4}
COORDS = {"g0": (0.0, 0.0), "g1": (1.0, 2.0), "g2": (3.0, -1.0), "g3": (4.0, 2.0), "g4": (5.0, 0.0)}


def _compute(coords_order):
    """Run the pipeline with gate_coords (and its dict order) as given."""
    gates = {n: Gate(name=n, load_ff=LOADS[n], x=COORDS[n][0], y=COORDS[n][1]) for n in TOPO_ORDER}
    vp = VariationParams(gate_coords={n: COORDS[n] for n in coords_order})
    tp = TimingParams(gate_loads=dict(LOADS))
    tg = TimingGraph(gates=gates, successors=dict(SUCCESSORS))
    return compute_analytical_ssta_arbitrary(tg, tp, vp)


def test_sink_and_per_gate_moments_invariant_to_gate_coords_ordering():
    """Core D1 property: values depend on gate names, never on dict order.

    Under the buggy topological readback, permuting gate_coords permutes the
    array layout while reads stay at topological positions, so sink/per-gate
    moments change. Name-keyed readback must reproduce them bit-exactly.
    """
    assert TOPO_ORDER != PERMUTED_ORDER  # premise: the two orders genuinely diverge
    a = _compute(TOPO_ORDER)
    b = _compute(PERMUTED_ORDER)
    for key in ("sink_mean", "sink_var", "sink_std", "AT_mean", "AT_var", "delay_mean", "delay_var"):
        assert a[key] == b[key], key


def test_per_gate_delay_moments_follow_the_gate_name():
    """With orders mismatched, each gate's moments must be its OWN, not the
    topological slot's. Means are checked against an independent computation
    (nominal_delay); variances against the name-keyed slot of the shared
    moment builder."""
    vp = VariationParams(gate_coords={n: COORDS[n] for n in PERMUTED_ORDER})
    tp = TimingParams(gate_loads=dict(LOADS))
    res = _compute(PERMUTED_ORDER)

    pm = compute_process_moments(vp)
    dm = compute_delay_moments(tp, vp, pm, gate_loads=dict(LOADS))
    for name in TOPO_ORDER:
        expected_mean = nominal_delay(
            load_ff=LOADS[name],
            gate_params=tp,
            vth_nom=vp.vth_nom_v,
            l_nom=vp.l_nom_nm,
            w_nom=vp.w_nom_nm,
        )
        assert res["delay_mean"][name] == pytest.approx(expected_mean, rel=1e-12)
        assert res["delay_var"][name] == pytest.approx(float(dm["var_d"][dm["idx"][name]]), rel=1e-12)


def test_two_gate_chain_matches_propagate_path_closed_form():
    """Order-coincidence case (coords given in topological order): the fix
    must be value-neutral vs the old topological readback, and the 2-gate
    chain result must match the all-pairs closed form from propagate_path."""
    loads = {"g0": 1.0, "g1": 1.2}
    vp = VariationParams(gate_coords={"g0": (0.0, 0.0), "g1": (1.0, 1.0)})
    tp = TimingParams(gate_loads=dict(loads))
    tg = TimingGraph(
        gates={"g0": Gate("g0", 1.0, 0.0, 0.0), "g1": Gate("g1", 1.2, 1.0, 1.0)},
        successors={"g0": ["g1"], "g1": []},
    )
    res = compute_analytical_ssta_arbitrary(tg, tp, vp)

    pm = compute_process_moments(vp)
    dm = compute_delay_moments(tp, vp, pm, gate_loads=dict(loads))
    mu, var = propagate_path(["g0", "g1"], dm["mean_d"], dm["var_d"], dm["cov_d"], dm["idx"])
    assert res["AT_mean"]["g1"] == pytest.approx(mu, rel=1e-12)
    assert res["AT_var"]["g1"] == pytest.approx(var, rel=1e-12)
