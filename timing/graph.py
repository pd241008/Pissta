"""
Timing graph definition and gate delay model for Stage 3.

Defines:
- Gate coordinates (matching the spatial variation model).
- DAG adjacency (topology only).
- Alpha-power gate delay with geometry sensitivity.
"""

from __future__ import annotations

from foundations.config_loader import TimingParams, VariationParams
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class GateParams:
    k: float = 1.0
    alpha: float = 1.3
    vdd_v: float = 1.0


@dataclass(frozen=True)
class Gate:
    name: str
    load_ff: float
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class TimingGraph:
    gates: Dict[str, Gate]
    successors: Dict[str, List[str]]

    def sources(self) -> List[str]:
        preds: Set[str] = set()
        for succs in self.successors.values():
            preds.update(succs)
        return [g for g in self.gates if g not in preds]

    def sinks(self) -> List[str]:
        return [g for g in self.gates if not self.successors.get(g)]

    def topological_order(self) -> List[str]:
        in_degree: Dict[str, int] = {g: 0 for g in self.gates}
        reverse: Dict[str, List[str]] = {g: [] for g in self.gates}
        for g, succs in self.successors.items():
            for s in succs:
                in_degree[s] += 1
                reverse[s].append(g)
        queue = [g for g in self.gates if in_degree[g] == 0]
        order: List[str] = []
        while queue:
            g = queue.pop(0)
            order.append(g)
            for s in self.successors.get(g, []):
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)
        if len(order) != len(self.gates):
            raise ValueError("TimingGraph contains a cycle.")
        return order


def build_branching_graph() -> TimingGraph:
    return build_branching_graph_from_config()


def build_branching_graph_from_config(
    timing_params: TimingParams | None = None,
    dag: Dict[str, List[str]] | None = None,
    variation_params=None,
) -> TimingGraph:
    if timing_params is None or dag is None or variation_params is None:
        from foundations.config_loader import load_config
        cfg = load_config()
        if timing_params is None:
            timing_params = cfg.timing_params
        if dag is None:
            dag = cfg.dag.successors
        if variation_params is None:
            variation_params = cfg.variation_params

    all_names = set(dag.keys())
    for succs in dag.values():
        all_names.update(succs)

    coords = variation_params.gate_coords
    gates = {
        name: Gate(name=name, load_ff=float(timing_params.gate_loads[name]), x=coords[name][0], y=coords[name][1])
        for name in all_names
        if name in timing_params.gate_loads and name in coords
    }
    missing = all_names - set(gates.keys())
    if missing:
        raise ValueError(f"Missing gate definitions for: {missing}")
    return TimingGraph(gates=gates, successors=dag)


def alpha_power_delay(
    vth_v: np.ndarray,
    load_ff: np.ndarray,
    gate_params: GateParams,
    l_nm: np.ndarray,
    w_nm: np.ndarray,
    l_nom_nm: float,
    w_nom_nm: float,
) -> np.ndarray:
    vdd = gate_params.vdd_v
    denominator = gate_params.k * np.power(
        np.maximum(vdd - vth_v, 0.1),
        gate_params.alpha,
    )
    delay = load_ff * vdd / denominator

    l_ratio = l_nm / l_nom_nm
    w_ratio = w_nm / w_nom_nm
    geometry_factor = l_ratio / np.sqrt(np.maximum(w_ratio, 1e-6))
    return delay * geometry_factor
