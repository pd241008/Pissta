"""
Deterministic augmentation of the Stage 6A dataset with the REDESIGNED Tier A
feature (var_d/load_ff^2).

The on-disk dataset.pkl was generated 2026-08-21 with a version of
variation/analytical.py that was subsequently changed (analytical.py was last
committed 2026-08-22, one day LATER than the dataset build). Recommputing the
feature from current process moments does NOT round-trip the stored
analytical_ssta delays (delay_var differs), so recomputing would make Tier A's
var_d use different process variances than the dataset's own analytical features.

Robust choice: derive var_d_per_load_ff^2 from the ALREADY-STORED per-gate
delay_var (analytical_ssta['delay_var']), which encodes whatever process moments
were used to build the dataset:
    var_d_per_load_ff_sq[name] = delay_var[name] / load_ff^2
This is algebraically identical to run_stage6a's formula (var_d/d^2 + ... +
w^2*var_W over load_ff^2) and is guaranteed consistent with the dataset's own
analytical_ssta (used by redesigned Tier B), because it IS that delay_var.

Fidelity guarantees:
  - mc_labels, graph structure, splits, and all existing physics_features are
    preserved bit-for-bit.
  - Only adds the missing 'var_d_per_load_ff_sq' key.
  - Feature value is independently verifiable from stored delay_var + load_ff.

Writes dataset.pkl in place (backed up first).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA = REPO / "data_generation" / "data"
DATA_PKL = DATA / "dataset.pkl"

def main() -> None:
    with open(DATA_PKL, "rb") as f:
        dataset = pickle.load(f)

    shutil.copy2(DATA_PKL, DATA / "dataset.pkl.pre-redesign.bak")
    print(f"Backed up to {DATA / 'dataset.pkl.pre-redesign.bak'}")

    hash_ctx = hashlib.sha256()
    for gid in sorted(dataset.keys()):
        entry = dataset[gid]
        phys = entry["physics_features"]
        if "var_d_per_load_ff_sq" in phys:
            print(f"ABORT: {gid} already has var_d_per_load_ff_sq; refusing to overwrite.")
            sys.exit(1)
        delay_var = phys["analytical_ssta"]["delay_var"]
        loads = entry["graph"]["gates"]
        fmap = {}
        for name, dv in delay_var.items():
            load_ff = loads[name]["load_ff"]
            fmap[name] = dv / (load_ff**2) if load_ff > 1e-12 else 0.0
        phys["var_d_per_load_ff_sq"] = fmap
        hash_ctx.update(json.dumps(
            {k: round(v, 12) for k, v in sorted(fmap.items())}, sort_keys=True
        ).encode())

    with open(DATA_PKL, "wb") as f:
        pickle.dump(dataset, f)

    sample = dataset[list(dataset.keys())[0]]["physics_features"]
    print("Augmented dataset saved.")
    print("physics_features keys:", list(sample.keys()))
    print("feature function sha256:", hash_ctx.hexdigest())

if __name__ == "__main__":
    main()
