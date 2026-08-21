"""
Config loader for Stage 3.

Reads stage3_config.json and provides typed access to variation,
timing, DAG, and experiment parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class VariationParams:
    l_nom_nm: float = 45.0
    l_sigma_nm: float = 2.0
    w_nom_nm: float = 90.0
    w_sigma_nm: float = 3.0
    vth_nom_v: float = 0.40
    vth_sigma_v: float = 0.02
    inter_die_sigma_l: float = 0.5
    inter_die_sigma_w: float = 1.0
    inter_die_sigma_vth: float = 0.005
    spatial_sigma_l: float = 1.0
    spatial_sigma_w: float = 1.5
    spatial_sigma_vth: float = 0.01
    spatial_lambda: float = 3.0
    vth_pelgrom_A_v_um: float = 1e-3
    vth_pelgrom_S_v_um: float = 1e-2
    l_random_sigma_nm: float = 0.5
    w_random_sigma_nm: float = 1.0
    vdd_v: float = 1.0
    gate_coords: Dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "G1": (0.0, 0.0),
            "G2": (0.0, 2.0),
            "G3": (0.0, -2.0),
            "G4": (2.0, 2.0),
            "G5": (2.0, -2.0),
            "G6": (4.0, 0.0),
        }
    )


@dataclass(frozen=True)
class TimingParams:
    k: float = 1.0
    alpha: float = 1.3
    vdd_v: float = 1.0
    gate_loads: Dict[str, float] = field(
        default_factory=lambda: {
            "G1": 1.0,
            "G2": 1.1,
            "G3": 1.1,
            "G4": 1.2,
            "G5": 1.2,
            "G6": 1.4,
        }
    )


@dataclass(frozen=True)
class DagSpec:
    successors: Dict[str, List[str]]


@dataclass(frozen=True)
class ExperimentConfig:
    n_samples: int = 100000
    seeds: List[int] = field(default_factory=lambda: [42, 123, 999])
    reference_seed: int = 42
    output_dir: str = "results"
    raw_data_file: str = "results/stage3_raw.npz"
    summary_file: str = "results/stage3_summary.json"
    sanity_check_file: str = "results/stage3_sanity_checks.json"
    reproducibility_file: str = "results/stage3_reproducibility_N100k.json"
    asymmetric_check_file: str = "results/stage3_asymmetric_check.json"


@dataclass(frozen=True)
class Stage3Config:
    variation_params: VariationParams
    timing_params: TimingParams
    dag: DagSpec
    experiment: ExperimentConfig


def _coords_to_tuples(raw: dict) -> Dict[str, tuple[float, float]]:
    return {name: (float(v["x"]), float(v["y"])) for name, v in raw.items()}


def load_config(path: str | Path = Path(__file__).resolve().parent / "stage3_config.json") -> Stage3Config:
    with open(path, "r") as f:
        data = json.load(f)

    vp = VariationParams(
        **{k: v for k, v in data["variation_params"].items() if k != "gate_coords"},
        gate_coords=_coords_to_tuples(data["variation_params"]["gate_coords"]),
    )
    tp = TimingParams(**data["timing_params"])
    dag = DagSpec(**data["dag"])
    exp = ExperimentConfig(**data["experiment"])

    return Stage3Config(variation_params=vp, timing_params=tp, dag=dag, experiment=exp)
