"""
Kit 1 — OOD-100 deliverable: GNN per-graph predictions + validated analytical
baseline + paired significance test on the 100 OOD graphs.

Closes the counterpart asks on the SAME 100 OOD graphs (n_gates 15–25):

  1. ood_dataset.pkl (raw graphs + MC labels) — packaged separately by
     package_deliverables.py (schema-verified copy).

  2. GNN per-graph predictions on the 100 OOD graphs in the SAME format as
     the stage6c_results_*.json per_graph records (graph_id, nrecon, mc_mean,
     mc_std, pred_mean, pred_std, mean_mae, ...), for BOTH backbones
     (vanilla, maxbias_cm) x 3 seeds (42/123/999). The checkpointed models are
     exactly the Stage 7 ones (identical filenames, weights_only load).

  3. Analytical baseline: the VALIDATED reimplementation (name-correct
     readback + capped Pelgrom moments — the "fixed_capped" variant whose
     0.1129 MAE reversed the Stage 8 ordering), run on the 100 OOD graphs.
     New data, zero new method code. Also exports the buggy_precap variant
     for reference, mirroring the diagnostics matrix.

  4. Paired per-graph cluster bootstrap (tier minus analytical, clustered by
     graph_id across the 3 GNN seeds; RandomState(42), 10,000 resamples — the
     exact Stage 6C methodology) + nrecon-stratified breakdown. Both GNNs are
     compared against the same deterministic analytical per-graph values.

Outputs (diagnostics/out/):
  kit1_ood_gnn_pergraph.json         GNN per-graph records, both backbones x 3 seeds
  kit1_ood_analytical_pergraph.json  analytical per-graph values, all variants
  kit1_ood_verdict.json              summary + paired bootstrap + nrecon breakdown
  (merged) gnn_baseline/results/kit1_ood100_results.json

Run:  python diagnostics/kit1_ood_export.py
"""

from __future__ import annotations

import inspect
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gnn_baseline"))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from foundations.config_loader import load_config  # noqa: E402
from timing.graph import TimingGraph, Gate  # noqa: E402
from timing.delay import compute_delay_moments  # noqa: E402
from timing.clark_max import clark_max  # noqa: E402
import variation.analytical as current_analytical  # noqa: E402

from dataset import GraphDataset  # noqa: E402
from eval import evaluate_model  # noqa: E402
from model import VanillaDAGGNNSage, MaxBiasedDAGGNNSage  # noqa: E402
from run_stage6c import _to_serializable, set_seed  # noqa: E402

DATA_DIR = REPO / "data_generation" / "data"
OOD_PKL = DATA_DIR / "ood_dataset.pkl"
CHECKPOINT_DIR = REPO / "gnn_baseline" / "checkpoints"
RESULTS_DIR = REPO / "gnn_baseline" / "results"
OUT_DIR = REPO / "diagnostics" / "out"
SEEDS = [42, 123, 999]
CONFIGS = ["vanilla", "maxbias_cm"]


# ---------------------------------------------------------------------------
# Analytical baseline — the VALIDATED reimplementation
# ---------------------------------------------------------------------------
def build_pre_cap_moments_fn():
    """compute_process_moments with ba994f3's cap edit textually reverted
    (the 2873f73 state: vth_random_sigma uses d_um, uncapped). Identical to
    diagnostics/verify_analytical_baseline.py."""
    src = inspect.getsource(current_analytical)
    marker = "d_eff = min(d_um, 5.0)"
    if marker not in src:
        raise RuntimeError("cap line not found in current variation/analytical.py; "
                           "the textual reversion premise is broken")
    reverted = src.replace(marker, "d_eff = d_um  # PRE-CAP (2873f73 state)")
    if marker in reverted:
        raise RuntimeError("multiple cap lines; reversion ambiguous")
    ns = {}
    exec(compile(reverted, "<pre_cap_analytical>", "exec"), ns)
    return ns["compute_process_moments"]


def analytical_variant(tg: TimingGraph, timing_params, variation_params,
                       moments_fn, name_correct: bool):
    """Original propagation algorithm of compute_analytical_ssta_arbitrary;
    the ONLY deviation is readback alignment (name-correct when True).
    Returns (sink_mean, sink_var). Copied verbatim from the validated
    diagnostics/verify_analytical_baseline.py implementation."""
    order = tg.topological_order()
    topo_idx = {name: i for i, name in enumerate(order)}

    pm = moments_fn(variation_params)            # arrays in gate_coords key order
    pm_idx = pm["idx"]                           # name -> array index (correct)
    ridx = pm_idx if name_correct else topo_idx  # topo_idx reproduces the bug

    loads = {name: tg.gates[name].load_ff for name in tg.gates}
    dm = compute_delay_moments(timing_params, variation_params, pm, gate_loads=loads)

    dmean = {n: float(dm["mean_d"][ridx[n]]) for n in order}
    dvar = {n: float(dm["var_d"][ridx[n]]) for n in order}
    cov = lambda a, b: float(dm["cov_d"][ridx[a], ridx[b]])

    preds = {n: [] for n in order}
    for g, succs in tg.successors.items():
        for s in succs:
            preds[s].append(g)

    AT_mean, AT_var = {}, {}
    for name in order:
        ps = preds[name]
        if not ps:
            AT_mean[name], AT_var[name] = dmean[name], dvar[name]
        else:
            mu = AT_mean[ps[0]] + dmean[name]
            var = AT_var[ps[0]] + dvar[name] + 2.0 * cov(ps[0], name)
            for p in ps[1:]:
                mu_p = AT_mean[p] + dmean[name]
                var_p = AT_var[p] + dvar[name] + 2.0 * cov(p, name)
                cov_cp = 0.0
                for q in ps[: ps.index(p) + 1]:
                    cov_cp += cov(q, name)
                rho = cov_cp / (math.sqrt(max(var, 1e-18)) * math.sqrt(max(var_p, 1e-18)))
                rho = max(-1.0, min(1.0, rho))
                mu, var = clark_max(mu, var, mu_p, var_p, rho)
            AT_mean[name], AT_var[name] = mu, var

    sink = tg.sinks()[0]
    return AT_mean[sink], AT_var[sink]


# ---------------------------------------------------------------------------
# GNN inference on the OOD graphs
# ---------------------------------------------------------------------------
def build_ood_loader(feature_stats, target_stats, physics_stats):
    """OOD DataLoader mirroring run_stage7_ood.py (patched GraphDataset)."""
    with open(OOD_PKL, "rb") as f:
        ood_dataset = pickle.load(f)
    ood_ids = list(ood_dataset.keys())

    ds = GraphDataset(split="test", data_dir=DATA_DIR, physics_mode="vanilla")
    ds.graph_ids = ood_ids
    ds.dataset = ood_dataset
    loader = DataLoader(ds.get_data(feature_stats, target_stats, physics_stats),
                        batch_size=16, shuffle=False)
    return ood_dataset, ood_ids, loader


def export_gnn_per_graph(ood_dataset, ood_ids, loader, feature_stats, target_stats,
                         physics_stats, device):
    """Load Stage 7 checkpoints, run inference, export stage6c-format records."""
    records = {}
    for config in CONFIGS:
        per_seed = []
        for seed in SEEDS:
            set_seed(seed)
            if config == "vanilla":
                model = VanillaDAGGNNSage(num_node_features=3, hidden_dim=64,
                                          num_layers=3, dropout=0.15, num_outputs=2)
            else:
                model = MaxBiasedDAGGNNSage(num_node_features=3, hidden_dim=54,
                                            num_layers=3, dropout=0.15, num_outputs=2)
            model = model.to(device)
            ckpt = CHECKPOINT_DIR / f"best_model_{config}_seed{seed}.pt"
            model.load_state_dict(torch.load(ckpt, weights_only=True))

            metrics = evaluate_model(model, loader, device, target_stats,
                                     dataset=loader.dataset)
            per_graph = metrics["per_graph"]
            assert len(per_graph) == len(ood_ids) == 100, \
                f"{config} seed={seed}: got {len(per_graph)} graphs, expected 100"

            # Order-normalize per_graph to ood_ids (dataloader is unshuffled).
            by_id = {r["graph_id"]: r for r in per_graph}
            ordered = [by_id[gid] for gid in ood_ids]
            per_seed.append({"seed": seed, "config": config, "per_graph": ordered})
            print(f"  {config} seed={seed}: mean_mae={metrics['mean_mae']:.4f} "
                  f"over {len(ordered)} OOD graphs")

        records[config] = per_seed
    return records


# ---------------------------------------------------------------------------
# Paired cluster bootstrap — Stage 6C methodology (RandomState(42), 10k resamples)
# ---------------------------------------------------------------------------
def paired_cluster_bootstrap_ci(baseline_by_graph: dict, tier_by_graph: dict,
                                tier_name: str = "tier") -> dict:
    """Paired per-graph cluster bootstrap CI (tier minus baseline), per graph_id,
    clustered across the 3 seeds. Uses RandomState(42) + 10000 resamples,
    matching the historical Stage 6C methodology exactly.

    baseline_by_graph: gid -> [per-seed baseline mean_mae] (length 3; the
                       analytical baseline is deterministic so the list holds
                       the same value 3x — clustering is a no-op for it)
    tier_by_graph:     gid -> [per-seed GNN mean_mae] (length 3)
    """
    by_graph_mean = defaultdict(list)
    for gid in baseline_by_graph:
        assert len(baseline_by_graph[gid]) == len(tier_by_graph[gid]) == len(SEEDS)
        for b, t in zip(baseline_by_graph[gid], tier_by_graph[gid]):
            by_graph_mean[gid].append(t - b)

    graph_mean_deltas = np.array([np.mean(v) for v in by_graph_mean.values()])
    mean_delta = float(np.mean(graph_mean_deltas))
    mean_win_rate = float(np.mean(graph_mean_deltas < 0))

    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(10000):
        idx = rng.randint(0, len(graph_mean_deltas), len(graph_mean_deltas))
        boot_means.append(np.mean(graph_mean_deltas[idx]))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    per_seed = []
    for si in range(len(SEEDS)):
        per_seed.append(float(np.mean([tier_by_graph[g][si] - baseline_by_graph[g][si]
                                       for g in baseline_by_graph])))
    return {
        "tier": tier_name,
        "mean_delta": mean_delta,           # GNN minus analytical (negative = GNN better)
        "mean_win_rate": mean_win_rate,
        "mean_ci_low": ci_low,
        "mean_ci_high": ci_high,
        "n_graphs": len(graph_mean_deltas),
        "per_seed_mean_deltas": per_seed,
        "bootstrap_method": "cluster by graph_id",
    }


def nrecon_breakdown(baseline_by_graph: dict, tier_by_graph: dict,
                     nrecon_by_graph: dict) -> dict:
    """Per-nrecon mean MAE for baseline vs GNN, pooled across seeds, plus
    per-seed detail. Mirrors the stage6c_report.md nrecon table structure."""
    buckets = defaultdict(lambda: {"n": 0, "ana": [], "gnn": []})
    for gid, nrecon in nrecon_by_graph.items():
        b = buckets[nrecon]
        b["n"] += 1
        b["ana"].append(np.mean(baseline_by_graph[gid]))
        b["gnn"].append(np.mean(tier_by_graph[gid]))
    out = {}
    for k in sorted(buckets):
        b = buckets[k]
        out[str(k)] = {
            "n": b["n"],
            "analytical_mean_mae": float(np.mean(b["ana"])),
            "gnn_mean_mae": float(np.mean(b["gnn"])),
        }
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cfg = load_config(str(REPO / "foundations" / "stage3_config.json"))

    with open(OOD_PKL, "rb") as f:
        ood_dataset = pickle.load(f)
    assert len(ood_dataset) == 100, f"expected 100 OOD graphs, got {len(ood_dataset)}"
    print(f"OOD dataset: {len(ood_dataset)} graphs")

    # ------------------------------------------------------------------
    # Part A: analytical baseline on the 100 OOD graphs (validated impl)
    # ------------------------------------------------------------------
    print("\n=== A. Analytical baseline (validated: name-correct readback + capped moments) ===")
    pre_cap_fn = build_pre_cap_moments_fn()
    capped_fn = current_analytical.compute_process_moments

    ana = {f"{r}_{m}": {} for r in ("buggy", "fixed") for m in ("precap", "capped")}
    nrecon_by_graph = {}
    n_gates_by_graph = {}
    for gid, entry in ood_dataset.items():
        g = entry["graph"]
        tg = TimingGraph(
            gates={n: Gate(name=n, load_ff=gd["load_ff"], x=gd["x"], y=gd["y"])
                   for n, gd in g["gates"].items()},
            successors=g["successors"],
        )
        topo = tg.topological_order()
        loads = {n: gd["load_ff"] for n, gd in g["gates"].items()}
        gp = replace(cfg.variation_params,
                     gate_coords={n: g["coordinates"][n] for n in g["gates"]})
        gt = replace(cfg.timing_params, gate_loads=loads)

        row = {}
        for m_label, m_fn in (("precap", pre_cap_fn), ("capped", capped_fn)):
            v_bug, _ = analytical_variant(tg, gt, gp, m_fn, name_correct=False)
            v_fix, _ = analytical_variant(tg, gt, gp, m_fn, name_correct=True)
            ana[f"buggy_{m_label}"][gid] = v_bug
            ana[f"fixed_{m_label}"][gid] = v_fix
        nrecon_by_graph[gid] = len(g["reconvergence_points"])
        n_gates_by_graph[gid] = len(g["gates"])

    # Invariance check on a sample (fixed readback must not depend on
    # gate_coords ordering — the validated property from the ID audit).
    invariance_maxdiff = 0.0
    for gid in list(ood_dataset)[:20]:
        g = ood_dataset[gid]["graph"]
        tg = TimingGraph(
            gates={n: Gate(name=n, load_ff=gd["load_ff"], x=gd["x"], y=gd["y"])
                   for n, gd in g["gates"].items()},
            successors=g["successors"],
        )
        topo = tg.topological_order()
        loads = {n: gd["load_ff"] for n, gd in g["gates"].items()}
        gp_topo = replace(cfg.variation_params,
                          gate_coords={n: g["coordinates"][n] for n in topo})
        gt = replace(cfg.timing_params, gate_loads=loads)
        v_fix2, _ = analytical_variant(tg, gt, gp_topo, capped_fn, name_correct=True)
        invariance_maxdiff = max(invariance_maxdiff, abs(ana["fixed_capped"][gid] - v_fix2))
    print(f"  fixed-readback ordering invariance (20-graph sample): max|d|={invariance_maxdiff:.3e}")

    ana_per_graph = {
        "mc_mean": {gid: ood_dataset[gid]["mc_labels"]["mean"] for gid in ood_dataset},
        "mc_std": {gid: ood_dataset[gid]["mc_labels"]["std"] for gid in ood_dataset},
        "nrecon": nrecon_by_graph,
        "n_gates": n_gates_by_graph,
        "variants": {k: {gid: float(v) for gid, v in d.items()} for k, d in ana.items()},
    }
    with open(OUT_DIR / "kit1_ood_analytical_pergraph.json", "w") as f:
        json.dump(ana_per_graph, f, indent=1)

    for variant in sorted(ana):
        diffs = [abs(ana[variant][gid] - ood_dataset[gid]["mc_labels"]["mean"])
                 for gid in ood_dataset]
        print(f"  {variant}: mean MAE vs MC = {np.mean(diffs):.4f}")

    # ------------------------------------------------------------------
    # Part B: GNN per-graph inference on the OOD graphs
    # ------------------------------------------------------------------
    print("\n=== B. GNN per-graph inference (Stage 7 checkpoints, 3 seeds x 2 backbones) ===")
    train_dataset = GraphDataset(split="train", data_dir=DATA_DIR, physics_mode="vanilla")
    feature_stats, target_stats = train_dataset._compute_normalization()
    physics_stats = train_dataset._compute_physics_normalization()

    ood_ids = list(ood_dataset.keys())
    ds = GraphDataset(split="test", data_dir=DATA_DIR, physics_mode="vanilla")
    ds.graph_ids = ood_ids
    ds.dataset = ood_dataset
    loader = DataLoader(ds.get_data(feature_stats, target_stats, physics_stats),
                        batch_size=16, shuffle=False)

    gnn_records = export_gnn_per_graph(ood_dataset, ood_ids, loader,
                                       feature_stats, target_stats, physics_stats, device)

    gnn_out = {
        "status": "Kit 1: GNN per-graph predictions on the 100 OOD graphs (Stage 7 checkpoints)",
        "checkpoint_note": ("identical weights to Stage 7 (best_model_{config}_seed{seed}.pt, "
                            "trained on the pissta-2k train split, frozen 6B/7 protocol)"),
        "ood_dataset": "data_generation/data/ood_dataset.pkl",
        "n_graphs": len(ood_ids),
        "configs": CONFIGS,
        "seeds": SEEDS,
        "results": gnn_records,
    }
    with open(OUT_DIR / "kit1_ood_gnn_pergraph.json", "w") as f:
        json.dump(_to_serializable(gnn_out), f, indent=1)

    # ------------------------------------------------------------------
    # Part C: paired significance test + nrecon breakdown
    # ------------------------------------------------------------------
    print("\n=== C. Paired cluster bootstrap: GNN vs analytical (100 graphs) ===")
    verdict = {
        "n_graphs": len(ood_ids),
        "n_gates_range": [int(min(n_gates_by_graph.values())), int(max(n_gates_by_graph.values()))],
        "nrecon_distribution": {str(k): sum(1 for v in nrecon_by_graph.values() if v == k)
                                for k in sorted(set(nrecon_by_graph.values()))},
        "analytical_variant_used": "fixed_capped (name-correct readback + capped Pelgrom moments)",
        "fixed_readback_invariance_sample20_max_absdiff": invariance_maxdiff,
        "mae": {},
        "paired_bootstrap": {},
        "nrecon_breakdown": {},
    }

    fixed_capped = ana["fixed_capped"]
    for config in CONFIGS:
        per_seed_mae = [float(np.mean([r["mean_mae"] for r in seed_rec["per_graph"]]))
                        for seed_rec in gnn_records[config]]
        pooled_gnn = np.mean(per_seed_mae)

        # analytical MAE per graph (deterministic; identical across seeds)
        ana_mae = {gid: abs(fixed_capped[gid] - ood_dataset[gid]["mc_labels"]["mean"])
                   for gid in ood_dataset}
        ana_mae_mean = float(np.mean(list(ana_mae.values())))

        baseline_by_graph = {gid: [ana_mae[gid]] * len(SEEDS) for gid in ood_dataset}
        tier_by_graph = {gid: [next(r["mean_mae"] for r in seed_rec["per_graph"] if r["graph_id"] == gid)
                               for seed_rec in gnn_records[config]]
                         for gid in ood_ids}

        boot = paired_cluster_bootstrap_ci(baseline_by_graph, tier_by_graph,
                                           tier_name=f"{config}_vs_analytical")

        verdict["mae"][config] = {
            "gnn_pooled_mean_mae": float(pooled_gnn),
            "gnn_per_seed_mean_mae": per_seed_mae,
            "analytical_mean_mae": ana_mae_mean,
        }
        verdict["paired_bootstrap"][config] = boot
        verdict["nrecon_breakdown"][config] = nrecon_breakdown(
            baseline_by_graph, tier_by_graph, nrecon_by_graph)

        print(f"\n{config}: GNN pooled MAE={pooled_gnn:.4f} vs analytical {ana_mae_mean:.4f}")
        print(f"  paired delta (GNN-analytical) = {boot['mean_delta']:+.4f} "
              f"CI [{boot['mean_ci_low']:+.4f}, {boot['mean_ci_high']:+.4f}] "
              f"(win rate {boot['mean_win_rate']:.2f}, n={boot['n_graphs']})")
        print("  nrecon breakdown (n, analytical, gnn):")
        for k, v in verdict["nrecon_breakdown"][config].items():
            print(f"    nrecon={k} (n={v['n']}): ana={v['analytical_mean_mae']:.4f} "
                  f"gnn={v['gnn_mean_mae']:.4f}")

    verdict["conclusion"] = (
        "Sign direction: negative delta = GNN better than analytical; positive = "
        "analytical better. Significant iff CI excludes 0."
    )

    with open(OUT_DIR / "kit1_ood_verdict.json", "w") as f:
        json.dump(_to_serializable(verdict), f, indent=2)

    # Merged result file in gnn_baseline/results (convenient single artifact)
    merged = {
        "status": "Kit 1: OOD-100 — GNN per-graph predictions + validated analytical baseline + paired bootstrap",
        "n_graphs": len(ood_ids),
        "configs": CONFIGS,
        "seeds": SEEDS,
        "gnn_per_graph": gnn_records,
        "analytical": {
            "variant_used": "fixed_capped",
            "per_graph": {gid: float(fixed_capped[gid]) for gid in ood_ids},
            "mae_per_graph": {gid: float(abs(fixed_capped[gid] - ood_dataset[gid]["mc_labels"]["mean"]))
                              for gid in ood_ids},
            "variants_all": {k: {gid: float(v) for gid, v in d.items()} for k, d in ana.items()},
        },
        "verdict": verdict,
    }
    with open(RESULTS_DIR / "kit1_ood100_results.json", "w") as f:
        json.dump(_to_serializable(merged), f, indent=1)

    print(f"\nSaved: {OUT_DIR / 'kit1_ood_verdict.json'}")
    print(f"Saved: {RESULTS_DIR / 'kit1_ood100_results.json'}")


if __name__ == "__main__":
    main()
