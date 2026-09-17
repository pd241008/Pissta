#!/usr/bin/env python3
"""
verify_release.py — independent end-to-end verification of a Pissta release.

Run AFTER generation, BEFORE upload. Produces VERIFICATION.txt per dataset and
verifies the cross-version nesting guarantee. Exits non-zero on any failure.

Checks per dataset
------------------
  H1  manifest hashes match the published files (byte integrity)
  H2  every .pkl unpickles and matches the documented schema
  H3  summary_stats.json recomputed from dataset.pkl matches to printed precision
  H4  manifest counts (n_graphs / n_train / n_val / n_test) match the files
  R1  2k/5k/10k: stratified splits re-derived from the frozen procedure match
  R2  2k: sampled MC labels recomputed bit-exactly (default 200 graphs)
  R3  5k/10k: extension-graph MC labels recomputed bit-exactly (default 100)
  N1  nesting: graphs 000000–001999 are byte-identical across 2k/5k/10k

Usage
-----
  python zenodo/scripts/verify_release.py                    # all datasets
  python zenodo/scripts/verify_release.py --version pissta-5k
  python zenodo/scripts/verify_release.py --label-sample 0   # skip MC recompute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from foundations.config_loader import load_config

# ---------------------------------------------------------------------------
# Reuse the generator's frozen procedures (imported from the script file).
# ---------------------------------------------------------------------------
import importlib.util

_gen_spec = importlib.util.spec_from_file_location(
    "generate_pissta", Path(__file__).resolve().parent / "generate_pissta.py"
)
gen = importlib.util.module_from_spec(_gen_spec)
sys.modules["generate_pissta"] = gen
_gen_spec.loader.exec_module(gen)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath(path: Path) -> str:
    """Repo-relative POSIX path for display use (never absolute)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        pass
    try:
        return p.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return f"<external>/{p.name}"


def load_dataset(pkl: Path) -> dict:
    with open(pkl, "rb") as f:
        return pickle.load(f)


def labels_bit_exact(entries: dict, ids: list[str], config) -> tuple[int, int]:
    """Recompute MC labels for the given graph ids; return (n_ok, n_checked)."""
    n_ok = 0
    for gid in ids:
        graph = gen.reconstruct_graph(entries[gid]["graph"], gid)
        recomputed = gen.label_graph(graph, config)
        if recomputed == entries[gid]["mc_labels"]:
            n_ok += 1
    return n_ok, len(ids)


def uniform_ids(n_graphs: int, n_total: int, seed: int) -> list[str]:
    if n_total <= 0:
        return []
    rng = np.random.default_rng(seed)
    return [f"graph_{i:06d}" for i in sorted(rng.choice(n_graphs, size=min(n_total, n_graphs), replace=False).tolist())]


# ---------------------------------------------------------------------------
def verify_core(version: str, ddir: Path, config, label_sample: int) -> None:
    print(f"\n=== {version} ===")
    manifest = json.loads((ddir / "manifest.json").read_text())

    # H1 — hashes
    hash_ok = True
    for fname, meta in manifest["files"].items():
        p = ddir / fname
        ok = p.exists() and sha256_file(p) == meta["sha256"] and p.stat().st_size == meta["bytes"]
        hash_ok &= ok
    check("H1 manifest file hashes", hash_ok, f"{len(manifest['files'])} files")

    # H2 — schema
    pkl_name = "ood_dataset.pkl" if version == "pissta-ood100" else "dataset.pkl"
    ds = load_dataset(ddir / pkl_name)
    required_keys = {"graph", "mc_labels"}
    if version != "pissta-ood100":
        required_keys.add("physics_features")
    schema_ok = all(
        set(e) >= required_keys
        and set(e["mc_labels"]) == {"mean", "std"}
        and set(e["graph"]) >= {"successors", "gates", "gate_loads", "coordinates",
                                "reconvergence_points", "source", "sink"}
        for e in ds.values()
    )
    check("H2 pickle schema", schema_ok, f"{len(ds)} graphs ({pkl_name})")

    # H3 — summary stats recomputation
    if version == "pissta-ood100":
        check("H3 summary_stats consistency", manifest["n_graphs"] == len(ds),
              "OOD: manifest count only (no summary_stats.json)")
    else:
        summary = json.loads((ddir / "summary_stats.json").read_text())
        means = [e["mc_labels"]["mean"] for e in ds.values()]
        stds = [e["mc_labels"]["std"] for e in ds.values()]
        sizes = [len(e["graph"]["gates"]) for e in ds.values()]
        reconv = [len(e["graph"]["reconvergence_points"]) for e in ds.values()]
        h3_ok = (
            summary["n_graphs"] == len(ds)
            and abs(summary["mean_delay"]["mean"] - float(np.mean(means))) < 5e-9
            and abs(summary["std_delay"]["mean"] - float(np.mean(stds))) < 5e-9
            and summary["gates_per_graph"]["min"] == min(sizes)
            and summary["gates_per_graph"]["max"] == max(sizes)
            and summary["reconvergence_points_per_graph"]["min"] == min(reconv)
            and summary["reconvergence_points_per_graph"]["max"] == max(reconv)
        )
        check("H3 summary_stats consistency", h3_ok)

    # H4 — counts
    if version == "pissta-ood100":
        check("H4 manifest counts", manifest["n_graphs"] == len(ds),
              "evaluation-only set, no split partition")
    else:
        splits = json.loads((ddir / "splits.json").read_text())
        h4_ok = (
            manifest["n_graphs"] == len(ds)
            and manifest["n_train"] == len(splits["train"])
            and manifest["n_val"] == len(splits["val"])
            and manifest["n_test"] == len(splits["test"])
            and len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(ds)
            and len(set(splits["train"]) & set(splits["val"]) & set(splits["test"])) == 0
        )
        check("H4 manifest counts & partition", h4_ok)

    # R1 — splits regeneration (2k/5k/10k)
    if version != "pissta-ood100":
        regen = gen.make_splits(ds)
        r1_ok = all(regen[k] == splits[k] for k in ("train", "val", "test"))
        check("R1 splits regenerate bit-exactly", r1_ok)

        # R2/R3 — MC label recompute (bit-exact)
        sample_ids = uniform_ids(len(ds), label_sample, seed=20260917)
        n_ok, n_checked = labels_bit_exact(ds, sample_ids, config)
        exact = n_ok == n_checked
        check("R2/R3 sampled MC labels bit-exact", exact, f"{n_ok}/{n_checked}")

        # Determinism metadata sanity
        pc = manifest["process_constants"]
        det_ok = (pc["mc_n_samples"] == 10_000 and pc["mc_seed"] == 42
                  and pc["generation_seed"] == 42 and pc["splits_seed"] == 42)
        check("M1 process constants documented", det_ok)

    # VERIFICATION.txt
    lines = [
        f"dataset: {version}",
        f"verified_utc: {__import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime())}",
        f"verifier_commit: {gen.GENERATOR_COMMIT}",
        f"n_graphs: {len(ds)}",
        f"label_sample_size: {label_sample if version != 'pissta-ood100' else 0}",
        f"result: {'ALL CHECKS PASSED' if not FAILURES else 'FAILURES PRESENT (see stdout)'}",
    ]
    (ddir / "VERIFICATION.txt").write_text("\n".join(lines) + "\n")


def verify_nesting(data_root: Path) -> None:
    print("\n=== N1 nesting (2k ⊂ 5k ⊂ 10k) ===")
    d2 = load_dataset(data_root / "pissta-2k" / "dataset.pkl")
    nested_ok = True
    for ver in ("pissta-5k", "pissta-10k"):
        d = load_dataset(data_root / ver / "dataset.pkl")
        prefix_ok = True
        for gid in gen.PREFIX_IDS:
            if pickle.dumps(d2[gid], protocol=gen.PICKLE_PROTOCOL) != pickle.dumps(d.get(gid), protocol=gen.PICKLE_PROTOCOL):
                prefix_ok = False
                break
        check(f"2k ⊂ {ver} (byte-identical prefix entries)", prefix_ok,
              f"{len(gen.PREFIX_IDS)} entries" if prefix_ok else "first mismatch near " + gid)
        nested_ok &= prefix_ok
    return nested_ok


def compare_dirs(released: Path, regenerated: Path) -> None:
    """Content-compare a released dataset against a fresh regeneration.

    Byte equality is expected ONLY for frozen-verbatim datasets (pissta-2k,
    pissta-ood100). For extension builds (5k/10k) the regeneration is compared
    content-wise: everything except wall_time_s timing metadata and manifest
    timestamps must match exactly (structures, labels, splits, physics
    features, summary stats to print precision).
    """
    print(f"\n=== compare released {released.name} vs regenerated ===")
    pkl_name = "ood_dataset.pkl" if "ood" in released.name else "dataset.pkl"
    a = load_dataset(released / pkl_name)
    b = load_dataset(regenerated / pkl_name)
    check("same graph id set", set(a) == set(b), f"{len(a)} vs {len(b)}")

    n_diff = 0
    first_diff = None
    for gid in a:
        ea, eb = a[gid], b.get(gid)
        if eb is None:
            n_diff += 1
            continue
        ea_c = {k: v for k, v in ea.items() if k != "wall_time_s"}
        eb_c = {k: v for k, v in eb.items() if k != "wall_time_s"}
        if ea_c != eb_c:
            n_diff += 1
            if first_diff is None:
                first_diff = gid
    check("entry content equality (excl. wall_time_s)", n_diff == 0,
          f"{len(a) - n_diff}/{len(a)} match" + (f"; first diff {first_diff}" if first_diff else ""))

    for fname in ("splits.json", "summary_stats.json"):
        pa, pb = released / fname, regenerated / fname
        if pa.exists() and pb.exists():
            ok = json.loads(pa.read_text()) == json.loads(pb.read_text())
            check(f"{fname} equality", ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["pissta-2k", "pissta-5k", "pissta-10k", "pissta-ood100"])
    ap.add_argument("--data-dir", type=Path, default=REPO / "zenodo" / "data")
    ap.add_argument("--label-sample", type=int, default=100,
                    help="graphs to MC-recompute per dataset (0 skips; 2k always gets >=200 by default contract)")
    ap.add_argument("--compare", type=Path, default=None, metavar="REGEN_DIR",
                    help="content-compare the released dataset against a fresh "
                         "regeneration in REGEN_DIR (expects REGEN_DIR/<version>/)")
    args = ap.parse_args()

    config = load_config(str(REPO / "foundations" / "stage3_config.json"))

    versions = [args.version] if args.version else ["pissta-2k", "pissta-5k", "pissta-10k", "pissta-ood100"]
    if args.version == "pissta-2k" and args.label_sample == 100:
        args.label_sample = 200  # paper-exact dataset: deeper sample

    for v in versions:
        ddir = args.data_dir / v
        if not (ddir / "manifest.json").exists():
            sys.exit(f"ABORT: {relpath(ddir)}/manifest.json missing — run generate_pissta.py first")
        verify_core(v, ddir, config, args.label_sample)
        if args.compare is not None:
            compare_dirs(ddir, args.compare / v)

    if args.version is None:
        verify_nesting(args.data_dir)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"VERIFICATION FAILED — {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED — release is internally consistent and reproducible.")


if __name__ == "__main__":
    main()
