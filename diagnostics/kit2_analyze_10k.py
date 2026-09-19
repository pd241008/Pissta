"""
Kit 2 — pissta-10k analysis: zero-shot + analytical baseline + paired tests.

Steps (complements diagnostics/kit2_train_10k.py which produced the fresh-
trained per-graph results):

  1. Zero-shot generalization: the existing Stage 7 MAX-bias-CM and vanilla
     models (trained on pissta-2k) run directly on the pissta-10k TEST split,
     using the 2k train-split normalizer (the normalizer the deployed model
     was trained with). 3 seeds x 2 backbones.

  2. Analytical baseline: the VALIDATED reimplementation (name-correct
     readback + capped Pelgrom moments — "fixed_capped") on the same 1,505
     test graphs, plus per-graph inference timing (practicality-at-scale ask).

  3. Paired per-graph cluster bootstrap (Stage 6C methodology:
     RandomState(42), 10,000 resamples, clustered by graph_id across 3 seeds):
       - fresh-10k GNN vs analytical
       - zero-shot GNN vs analytical
       - fresh-10k GNN vs zero-shot GNN (does 5x data help?)

  4. Disaggregation by nrecon and by n_gates bucket.

Outputs (diagnostics/out/):
  kit2_10k_analytical_pergraph.json
  kit2_10k_zeroshot_results.json
  kit2_10k_verdict.json
  (merged) gnn_baseline/results/kit2_10k_results.json

Run:  python diagnostics/kit2_analyze_10k.py [--skip-zeroshot]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
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

# Reuse the validated analytical implementation + bootstrap from Kit 1
# (single source of truth; kit1 has a __main__ guard so import is side-effect free).
_spec = importlib.util.spec_from_file_location("kit1_ood_export", REPO / "diagnostics" / "kit1_ood_export.py")
kit1 = importlib.util.module_from_spec(_spec)
sys.modules["kit1_ood_export"] = kit1
_spec.loader.exec_module(kit1)

from dataset import GraphDataset  # noqa: E402
from eval import evaluate_model  # noqa: E402
from model import VanillaDAGGNNSage, MaxBiasedDAGGNNSage  # noqa: E402
from run_stage6c import _to_serializable  # noqa: E402

DATA_2K = REPO / "data_generation" / "data"
DATA_10K = REPO / "zenodo" / "data" / "pissta-10k"
CHECKPOINT_DIR = REPO / "gnn_baseline" / "checkpoints"
RESULTS_DIR = REPO / "gnn_baseline" / "results"
OUT_DIR = REPO / "diagnostics" / "out"
SEEDS = [42, 123, 999]
CONFIGS = ["vanilla", "maxbias_cm"]

N_GATES_BUCKETS = [(6, 8), (9, 11), (12, 14)]


def load_module_results(config: str, seed: int) -> dict:
    with open(RESULTS_DIR / f"kit2_10k_results_{config}_seed{seed}.json") as f:
        return json.load(f)


def bucket_key(n: int, buckets) -> str:
    for lo, hi in buckets:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return f">={buckets[-1][1]}"


def paired_bootstrap_from_lists(tier_by_graph: dict, base_by_graph: dict,
                                tier_name: str) -> dict:
    """Wrapper over kit1.paired_cluster_bootstrap_ci with (tier - baseline)
    orientation; both dicts gid -> [per-seed values]."""
    return kit1.paired_cluster_bootstrap_ci(base_by_graph, tier_by_graph, tier_name)


def disaggregate(per_seed_recs: list, ana_mae: dict, nrecon_by_graph: dict,
                 n_gates_by_graph: dict) -> dict:
    """Per-bucket mean MAE (pooled across seeds) for GNN and analytical."""
    out = {"by_nrecon": {}, "by_n_gates": {}}
    for label, keyfn in (("by_nrecon", lambda gid: str(nrecon_by_graph[gid])),
                         ("by_n_gates", lambda gid: bucket_key(n_gates_by_graph[gid], N_GATES_BUCKETS))):
        groups = defaultdict(lambda: {"gnn": [], "ana": []})
        for seed_rec in per_seed_recs:
            src = seed_rec["test_metrics"]["per_graph"] if "test_metrics" in seed_rec else seed_rec["per_graph"]
            for r in src:
                gid = r["graph_id"]
                g = groups[keyfn(gid)]
                g["gnn"].append(r["mean_mae"])
                g["ana"].append(ana_mae[gid])
        table = {}
        for k in sorted(groups, key=lambda s: (len(s), s)):
            g = groups[k]
            table[k] = {
                "n_graphs": sum(1 for gid in nrecon_by_graph if keyfn(gid) == k),
                "gnn_mean_mae": float(np.mean(g["gnn"])),
                "analytical_mean_mae": float(np.mean(g["ana"])),
            }
        out[label] = table
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-zeroshot", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cfg = load_config(str(REPO / "foundations" / "stage3_config.json"))

    # ------------------------------------------------------------------
    # Load pissta-10k
    # ------------------------------------------------------------------
    with open(DATA_10K / "dataset.pkl", "rb") as f:
        dataset_10k = pickle.load(f)
    with open(DATA_10K / "splits.json") as f:
        splits_10k = json.load(f)
    test_ids = splits_10k["test"]
    nrecon_by_graph = {gid: len(dataset_10k[gid]["graph"]["reconvergence_points"]) for gid in test_ids}
    n_gates_by_graph = {gid: len(dataset_10k[gid]["graph"]["gates"]) for gid in test_ids}
    print(f"pissta-10k test split: {len(test_ids)} graphs "
          f"(n_gates {min(n_gates_by_graph.values())}-{max(n_gates_by_graph.values())}, "
          f"nrecon {min(nrecon_by_graph.values())}-{max(nrecon_by_graph.values())})")

    # Generation-config confirmation (Kit 2 step 2, apples-to-apples check)
    manifest_10k = json.loads((DATA_10K / "manifest.json").read_text())
    pc = manifest_10k["process_constants"]
    gen_config_check = {
        "generation_seed": pc["generation_seed"],
        "mc_n_samples": pc["mc_n_samples"],
        "mc_seed": pc["mc_seed"],
        "n_gates_range": pc["n_gates_range"],
        "min_reconvergence": pc["min_reconvergence"],
        "sampler": pc["sampler"],
        "same_as_2k": (pc["n_gates_range"] == [6, 14] and pc["min_reconvergence"] == 1
                       and pc["mc_n_samples"] == 10000 and pc["mc_seed"] == 42),
        "load_pairing_note": manifest_10k["extension_provenance"]["load_pairing"],
    }
    print(f"generation config same as pissta-2k: {gen_config_check['same_as_2k']}")

    # ------------------------------------------------------------------
    # Step A: analytical baseline on the 10k test split (+ timing)
    # ------------------------------------------------------------------
    print("\n=== A. Analytical baseline on 1,505 test graphs (validated fixed_capped) ===")
    import variation.analytical as current_analytical
    pre_cap_fn = kit1.build_pre_cap_moments_fn()
    capped_fn = current_analytical.compute_process_moments

    ana = {f"{r}_{m}": {} for r in ("buggy", "fixed") for m in ("precap", "capped")}
    ana_times_ms = []
    invariance_maxdiff = 0.0
    t_start = time.time()
    for i, gid in enumerate(test_ids):
        g = dataset_10k[gid]["graph"]
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

        for m_label, m_fn in (("precap", pre_cap_fn), ("capped", capped_fn)):
            v_bug, _ = kit1.analytical_variant(tg, gt, gp, m_fn, name_correct=False)
            ana[f"buggy_{m_label}"][gid] = v_bug
            if m_label == "capped":
                t0 = time.perf_counter()
                v_fix, _ = kit1.analytical_variant(tg, gt, gp, m_fn, name_correct=True)
                ana_times_ms.append((time.perf_counter() - t0) * 1000.0)
                ana[f"fixed_capped"][gid] = v_fix
            else:
                v_fix, _ = kit1.analytical_variant(tg, gt, gp, m_fn, name_correct=True)
                ana[f"fixed_precap"][gid] = v_fix

        # ordering-invariance sample
        if i < 50:
            gp_topo = replace(cfg.variation_params,
                              gate_coords={n: g["coordinates"][n] for n in topo})
            v_fix2, _ = kit1.analytical_variant(tg, gt, gp_topo, capped_fn, name_correct=True)
            invariance_maxdiff = max(invariance_maxdiff, abs(ana["fixed_capped"][gid] - v_fix2))
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(test_ids)} ({time.time() - t_start:.0f}s)")

    ana_mae = {gid: abs(ana["fixed_capped"][gid] - dataset_10k[gid]["mc_labels"]["mean"])
               for gid in test_ids}
    ana_mae_mean = float(np.mean(list(ana_mae.values())))
    print(f"  fixed_capped mean MAE vs MC = {ana_mae_mean:.4f}")
    print(f"  per-graph analytical time: mean={np.mean(ana_times_ms):.3f}ms "
          f"p50={np.percentile(ana_times_ms, 50):.3f} p95={np.percentile(ana_times_ms, 95):.3f} "
          f"max={np.max(ana_times_ms):.3f}")
    print(f"  ordering-invariance sample (50 graphs): max|d|={invariance_maxdiff:.3e}")

    # timing by n_gates bucket (linearity check)
    timing_by_size = {}
    for lo, hi in N_GATES_BUCKETS:
        ts = [t for gid, t in zip(test_ids, ana_times_ms) if lo <= n_gates_by_graph[gid] <= hi]
        timing_by_size[f"{lo}-{hi}"] = {"n": len(ts), "mean_ms": float(np.mean(ts))}

    with open(OUT_DIR / "kit2_10k_analytical_pergraph.json", "w") as f:
        json.dump({
            "mc_mean": {gid: dataset_10k[gid]["mc_labels"]["mean"] for gid in test_ids},
            "nrecon": nrecon_by_graph,
            "n_gates": n_gates_by_graph,
            "variants": {k: {gid: float(v) for gid, v in d.items()} for k, d in ana.items()},
            "timing": {
                "note": "fixed_capped analytical_variant per-graph wall time, "
                        "CPU single-run, excludes TimingGraph construction",
                "mean_ms": float(np.mean(ana_times_ms)),
                "p50_ms": float(np.percentile(ana_times_ms, 50)),
                "p95_ms": float(np.percentile(ana_times_ms, 95)),
                "max_ms": float(np.max(ana_times_ms)),
                "by_n_gates": timing_by_size,
                "total_corpus_s": float(np.sum(ana_times_ms) / 1000.0),
            },
        }, f, indent=1)

    # ------------------------------------------------------------------
    # Step B: zero-shot (2k checkpoints -> 10k test, 2k train normalizer)
    # ------------------------------------------------------------------
    zeroshot_records = {}
    if not args.skip_zeroshot:
        print("\n=== B. Zero-shot: 2k-trained models on the 10k test split ===")
        train_2k = GraphDataset(split="train", data_dir=DATA_2K, physics_mode="vanilla")
        feature_stats_2k, target_stats_2k = train_2k._compute_normalization()
        physics_stats_2k = train_2k._compute_physics_normalization()

        ds_10k_test = GraphDataset(split="test", data_dir=DATA_10K, physics_mode="vanilla")
        loader_zs = DataLoader(
            ds_10k_test.get_data(feature_stats_2k, target_stats_2k, physics_stats_2k),
            batch_size=32, shuffle=False)

        for config in CONFIGS:
            per_seed = []
            for seed in SEEDS:
                if config == "vanilla":
                    model = VanillaDAGGNNSage(num_node_features=3, hidden_dim=64,
                                              num_layers=3, dropout=0.15, num_outputs=2)
                else:
                    model = MaxBiasedDAGGNNSage(num_node_features=3, hidden_dim=54,
                                                num_layers=3, dropout=0.15, num_outputs=2)
                model = model.to(device)
                ckpt = CHECKPOINT_DIR / f"best_model_{config}_seed{seed}.pt"
                model.load_state_dict(torch.load(ckpt, weights_only=True))
                metrics = evaluate_model(model, loader_zs, device, target_stats_2k,
                                         dataset=loader_zs.dataset)
                assert metrics["n_graphs"] == len(test_ids)
                per_seed.append({"seed": seed, "config": config, "per_graph": metrics["per_graph"]})
                print(f"  {config} seed={seed}: zero-shot 10k-test MAE={metrics['mean_mae']:.4f}")
            zeroshot_records[config] = per_seed

        with open(OUT_DIR / "kit2_10k_zeroshot_results.json", "w") as f:
            json.dump(_to_serializable({
                "status": "Kit 2: zero-shot pissta-2k checkpoints on pissta-10k test split",
                "normalizer": "2k train-split stats (the stats the deployed model was trained with)",
                "configs": CONFIGS, "seeds": SEEDS, "results": zeroshot_records,
            }), f, indent=1)

    # ------------------------------------------------------------------
    # Step C: fresh-training results + paired bootstraps
    # ------------------------------------------------------------------
    print("\n=== C. Paired cluster bootstraps (1,505 graphs) ===")
    fresh_records = {c: [load_module_results(c, s) for s in SEEDS] for c in CONFIGS}
    for c in CONFIGS:
        maes = [r["test_metrics"]["mean_mae"] for r in fresh_records[c]]
        print(f"  fresh {c}: per-seed MAE {[round(m, 4) for m in maes]} (mean {np.mean(maes):.4f})")

    def by_graph(per_seed_recs, field="mean_mae"):
        out = defaultdict(list)
        for rec in per_seed_recs:
            src = rec["test_metrics"]["per_graph"] if "test_metrics" in rec else rec["per_graph"]
            for r in src:
                out[r["graph_id"]].append(r[field])
        return dict(out)

    baseline_by_graph = {gid: [ana_mae[gid]] * len(SEEDS) for gid in test_ids}
    verdict = {
        "n_test_graphs": len(test_ids),
        "generation_config_check": gen_config_check,
        "analytical": {
            "variant_used": "fixed_capped",
            "mean_mae": ana_mae_mean,
            "per_seed_mean_mae": [ana_mae_mean] * len(SEEDS),
            "ordering_invariance_sample50_max_absdiff": invariance_maxdiff,
            "timing": {
                "mean_ms_per_graph": float(np.mean(ana_times_ms)),
                "p95_ms_per_graph": float(np.percentile(ana_times_ms, 95)),
                "total_corpus_s": float(np.sum(ana_times_ms) / 1000.0),
                "by_n_gates": timing_by_size,
            },
        },
        "gnn": {},
        "paired_bootstrap": {},
        "disaggregation": {},
    }

    for config in CONFIGS:
        fresh_bg = by_graph(fresh_records[config])
        gnn_maes = [r["test_metrics"]["mean_mae"] for r in fresh_records[config]]
        verdict["gnn"][config] = {
            "fresh_10k": {
                "per_seed_mean_mae": gnn_maes,
                "pooled_mean_mae": float(np.mean(gnn_maes)),
                "avg_inference_ms_per_graph": float(np.mean(
                    [r["test_metrics"]["avg_inference_time_ms"] for r in fresh_records[config]])),
            },
        }

        boot_fresh = paired_bootstrap_from_lists(fresh_bg, baseline_by_graph,
                                                 f"{config}_fresh10k_vs_analytical")
        verdict["paired_bootstrap"][f"{config}_fresh10k_vs_analytical"] = boot_fresh
        print(f"\n  {config} fresh-10k vs analytical: delta={boot_fresh['mean_delta']:+.4f} "
              f"CI [{boot_fresh['mean_ci_low']:+.4f}, {boot_fresh['mean_ci_high']:+.4f}]")

        verdict["disaggregation"][config] = disaggregate(
            fresh_records[config], ana_mae, nrecon_by_graph, n_gates_by_graph)

        if not args.skip_zeroshot:
            zs_bg = by_graph(zeroshot_records[config])
            zs_maes = [np.mean([r["mean_mae"] for r in rec["per_graph"]]) for rec in zeroshot_records[config]]
            verdict["gnn"][config]["zero_shot_2k_on_10k"] = {
                "per_seed_mean_mae": [float(m) for m in zs_maes],
                "pooled_mean_mae": float(np.mean(zs_maes)),
            }
            boot_zs = paired_bootstrap_from_lists(zs_bg, baseline_by_graph,
                                                  f"{config}_zeroshot_vs_analytical")
            verdict["paired_bootstrap"][f"{config}_zeroshot_vs_analytical"] = boot_zs
            print(f"  {config} zero-shot vs analytical: delta={boot_zs['mean_delta']:+.4f} "
                  f"CI [{boot_zs['mean_ci_low']:+.4f}, {boot_zs['mean_ci_high']:+.4f}]")

            # fresh vs zero-shot: does 5x data help the GNN?
            boot_fz = paired_bootstrap_from_lists(fresh_bg, zs_bg,
                                                  f"{config}_fresh10k_vs_zeroshot")
            verdict["paired_bootstrap"][f"{config}_fresh10k_vs_zeroshot"] = boot_fz
            print(f"  {config} fresh-10k vs zero-shot: delta={boot_fz['mean_delta']:+.4f} "
                  f"CI [{boot_fz['mean_ci_low']:+.4f}, {boot_fz['mean_ci_high']:+.4f}]")

        print("  disaggregation (nrecon):")
        for k, v in verdict["disaggregation"][config]["by_nrecon"].items():
            print(f"    nrecon={k} (n={v['n_graphs']}): ana={v['analytical_mean_mae']:.4f} "
                  f"gnn={v['gnn_mean_mae']:.4f}")
        print("  disaggregation (n_gates):")
        for k, v in verdict["disaggregation"][config]["by_n_gates"].items():
            print(f"    gates={k} (n={v['n_graphs']}): ana={v['analytical_mean_mae']:.4f} "
                  f"gnn={v['gnn_mean_mae']:.4f}")

    verdict["sign_convention"] = ("negative delta = first-listed method better; "
                                  "significant iff CI excludes 0")

    with open(OUT_DIR / "kit2_10k_verdict.json", "w") as f:
        json.dump(_to_serializable(verdict), f, indent=2)

    # Merged artifact
    merged = {
        "status": "Kit 2: pissta-10k scale test — fresh training + zero-shot + validated analytical + paired bootstrap",
        "n_test_graphs": len(test_ids),
        "generation_config_check": gen_config_check,
        "fresh_results_files": [f"kit2_10k_results_{c}_seed{s}.json" for c in CONFIGS for s in SEEDS],
        "verdict": verdict,
    }
    if not args.skip_zeroshot:
        merged["gnn_per_graph"] = {
            "fresh": {c: [{"seed": r["seed"], "test_metrics_per_graph": r["test_metrics"]["per_graph"]}
                          for r in fresh_records[c]] for c in CONFIGS},
            "zero_shot": zeroshot_records,
        }
        merged["analytical"] = {
            "variant_used": "fixed_capped",
            "per_graph": {gid: float(ana["fixed_capped"][gid]) for gid in test_ids},
            "mae_per_graph": {gid: float(ana_mae[gid]) for gid in test_ids},
        }
    with open(RESULTS_DIR / "kit2_10k_results.json", "w") as f:
        json.dump(_to_serializable(merged), f, indent=1)

    print(f"\nSaved: {OUT_DIR / 'kit2_10k_verdict.json'}")
    print(f"Saved: {RESULTS_DIR / 'kit2_10k_results.json'}")


if __name__ == "__main__":
    main()
