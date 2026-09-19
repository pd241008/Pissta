"""
Package the three collaboration kits into deliverables/ for handoff:

  deliverables/
    kit1_ood100/          ood_dataset.pkl (schema-verified copy) + per-graph
                          GNN predictions + analytical baseline + paired test
    kit2_pissta10k/       pissta-10k dataset.pkl/splits.json + fresh-training
                          per-seed results + zero-shot results + verdict
    kit3_stage4_audit/    audit verdict + report + the audited Stage 4 code
                          and stored outputs (self-contained handoff)

Every file is recorded with sha256 + size in deliverables/MANIFEST.json.
The OOD pickle is schema-verified before copying (100 graphs, graph+mc_labels
keys, 15-25 gates, nrecon>=2).

Run:  python diagnostics/package_deliverables.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "deliverables"
OUT_DIR = REPO / "diagnostics" / "out"
DATA_DIR = REPO / "data_generation" / "data"
P10K_DIR = REPO / "zenodo" / "data" / "pissta-10k"
RESULTS_DIR = REPO / "gnn_baseline" / "results"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_ood_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        ds = pickle.load(f)
    n = len(ds)
    required_graph = {"successors", "gates", "gate_loads", "coordinates",
                      "reconvergence_points", "source", "sink"}
    schema_ok = all(
        set(e) >= {"graph", "mc_labels"}
        and set(e["mc_labels"]) == {"mean", "std"}
        and set(e["graph"]) >= required_graph
        for e in ds.values()
    )
    gates = [len(e["graph"]["gates"]) for e in ds.values()]
    nrecon = [len(e["graph"]["reconvergence_points"]) for e in ds.values()]
    checks = {
        "n_graphs": n,
        "schema_ok": schema_ok,
        "n_gates_min": min(gates), "n_gates_max": max(gates),
        "n_gates_in_15_25": min(gates) >= 15 and max(gates) <= 25,
        "nrecon_min": min(nrecon),
        "nrecon_ge_2_all": min(nrecon) >= 2,
        "nrecon_distribution": {str(k): nrecon.count(k) for k in sorted(set(nrecon))},
    }
    ok = (n == 100 and schema_ok and checks["n_gates_in_15_25"] and checks["nrecon_ge_2_all"])
    checks["verified"] = ok
    return checks


KIT1_FILES = [
    (DATA_DIR / "ood_dataset.pkl", "ood_dataset.pkl"),
    (OUT_DIR / "kit1_ood_gnn_pergraph.json", "kit1_ood_gnn_pergraph.json"),
    (OUT_DIR / "kit1_ood_analytical_pergraph.json", "kit1_ood_analytical_pergraph.json"),
    (OUT_DIR / "kit1_ood_verdict.json", "kit1_ood_verdict.json"),
    (RESULTS_DIR / "kit1_ood100_results.json", "kit1_ood100_results.json"),
    (REPO / "results" / "stage9_ood_crossmethod_report.md", "stage9_ood_crossmethod_report.md"),
]

KIT2_FILES = [
    (P10K_DIR / "dataset.pkl", "pissta-10k_dataset.pkl"),
    (P10K_DIR / "splits.json", "pissta-10k_splits.json"),
    (P10K_DIR / "summary_stats.json", "pissta-10k_summary_stats.json"),
    (P10K_DIR / "manifest.json", "pissta-10k_manifest.json"),
    (OUT_DIR / "kit2_10k_verdict.json", "kit2_10k_verdict.json"),
    (OUT_DIR / "kit2_10k_zeroshot_results.json", "kit2_10k_zeroshot_results.json"),
    (OUT_DIR / "kit2_10k_analytical_pergraph.json", "kit2_10k_analytical_pergraph.json"),
    (RESULTS_DIR / "kit2_10k_results.json", "kit2_10k_results.json"),
    (REPO / "results" / "stage10_scale_test_report.md", "stage10_scale_test_report.md"),
] + [
    (RESULTS_DIR / f"kit2_10k_results_{c}_seed{s}.json",
     f"kit2_10k_results_{c}_seed{s}.json")
    for c in ("vanilla", "maxbias_cm") for s in (42, 123, 999)
]

KIT3_FILES = [
    (OUT_DIR / "kit3_stage4_audit_verdict.json", "kit3_stage4_audit_verdict.json"),
    (OUT_DIR / "kit3_stage4_audit_report.json", "kit3_stage4_audit_report.json"),
    (REPO / "results" / "stage11_stage4_audit_report.md", "stage11_stage4_audit_report.md"),
    (REPO / "ssta" / "analytical_ssta.py", "code/ssta_analytical_ssta.py"),
    (REPO / "experiments" / "run_stage4b_hybrid.py", "code/run_stage4b_hybrid.py"),
    (REPO / "results" / "stage4_analytical_ssta.json", "stored/stage4_analytical_ssta.json"),
    (REPO / "results" / "stage4b_hybrid_run.json", "stored/stage4b_hybrid_run.json"),
    (REPO / "results" / "stage3_summary.json", "stored/stage3_summary.json"),
    (REPO / "foundations" / "stage3_config.json", "stored/stage3_config.json"),
]


def copy_all(files, target_dir: Path) -> list:
    manifest = []
    for src, rel in files:
        if not src.exists():
            raise FileNotFoundError(f"missing deliverable source: {src}")
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".pkl":
            shutil.copyfile(src, dst)
        else:
            shutil.copy2(src, dst)
        manifest.append({"file": rel, "sha256": sha256_file(dst),
                         "bytes": dst.stat().st_size, "source": str(src.relative_to(REPO))})
    return manifest


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    # --- Kit 1 ---
    print("=== kit1_ood100 ===")
    ood_checks = verify_ood_pkl(DATA_DIR / "ood_dataset.pkl")
    print(f"  ood_dataset.pkl schema verification: {'PASS' if ood_checks['verified'] else 'FAIL'}")
    if not ood_checks["verified"]:
        raise SystemExit("ABORT: OOD pickle failed schema verification")
    m1 = copy_all(KIT1_FILES, OUT_ROOT / "kit1_ood100")
    (OUT_ROOT / "kit1_ood100" / "schema_verification.json").write_text(
        json.dumps(ood_checks, indent=2))

    # --- Kit 2 ---
    print("=== kit2_pissta10k ===")
    m2 = copy_all(KIT2_FILES, OUT_ROOT / "kit2_pissta10k")

    # --- Kit 3 ---
    print("=== kit3_stage4_audit ===")
    m3 = copy_all(KIT3_FILES, OUT_ROOT / "kit3_stage4_audit")
    shutil.copy2(REPO / "diagnostics" / "out" / "kit3_audit.log",
                 OUT_ROOT / "kit3_stage4_audit" / "audit_console_log.txt")

    # --- Root MANIFEST + README ---
    manifest = {
        "purpose": "Collaboration-kit handoff: OOD-100 comparison, pissta-10k scale test, Stage 4 audit",
        "created_by": "diagnostics/package_deliverables.py",
        "kits": {
            "kit1_ood100": {"files": m1,
                            "ood_schema_verification": ood_checks},
            "kit2_pissta10k": {"files": m2},
            "kit3_stage4_audit": {"files": m3},
        },
    }
    (OUT_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    n_files = len(m1) + len(m2) + len(m3)
    total_mb = sum(f["bytes"] for f in m1 + m2 + m3) / 1e6
    print(f"\nPackaged {n_files} files ({total_mb:.1f} MB) under {OUT_ROOT.relative_to(REPO)}/")
    print(f"MANIFEST: {OUT_ROOT / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
