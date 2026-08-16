"""
Simple timing statistics utilities.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np


def summarize_delay(delay: np.ndarray) -> Dict[str, float]:
    if delay.ndim != 1 or delay.size == 0:
        raise ValueError("delay must be a non-empty 1-D array.")
    return {
        "mean": float(np.mean(delay)),
        "std": float(np.std(delay, ddof=1)),
        "p95": float(np.quantile(delay, 0.95)),
        "p99": float(np.quantile(delay, 0.99)),
        "p99_87": float(np.quantile(delay, 0.9987)),
        "min": float(np.min(delay)),
        "max": float(np.max(delay)),
    }


def critical_path_split(
    path_labels: Union[Dict[str, np.ndarray], None] = None,
    path1_critical: Union[np.ndarray, None] = None,
    path2_critical: Union[np.ndarray, None] = None,
) -> Dict[str, float]:
    if path_labels is not None:
        n = float(next(iter(path_labels.values())).size)
        return {
            name: float(np.count_nonzero(arr) / n)
            for name, arr in path_labels.items()
        }
    if path1_critical is not None and path2_critical is not None:
        n = float(path1_critical.size)
        return {
            "path1_fraction": float(np.count_nonzero(path1_critical) / n),
            "path2_fraction": float(np.count_nonzero(path2_critical) / n),
        }
    raise ValueError("Must provide either path_labels or both path1_critical and path2_critical.")
