"""Analysis and statistical reporting for Package G (G1 & G3) tail screening.

Loads:
- research/data/jgen/campaign/B/g/strain_results.json
- research/data/jgen/campaign/B/g/<arm>/records/*.json
- research/data/jgen/campaign/B/g/evaluation_contract_g1_g3.json

Computes:
- Top-1 and floor strain (E_unrel - E_rel) per unit and median across panel
- delta_strain relative to control
- delta_E_unrel and floor delta_E_unrel
- Obscene rates (> 200 kJ/mol)
- Parent-clustered bootstrap 95% CIs (B=2000 resamples over parents)
- Guardrail checks against contract
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[4]
BASE_DIR = REPO / "research/data/jgen/campaign/B/g"


def cluster_bootstrap(
    diffs_by_parent: Dict[str, List[float]],
    n_resamples: int = 2000,
    seed: int = 42,
    stat_fn=np.median,
) -> Tuple[float, float, float]:
    """Parent-clustered bootstrap for clustered differences (e.g. 3 seeds per parent)."""
    rng = np.random.default_rng(seed)
    parents = list(diffs_by_parent.keys())
    n_p = len(parents)

    all_vals = [v for vals in diffs_by_parent.values() for v in vals]
    point_est = float(stat_fn(all_vals))

    boot_stats = []
    for _ in range(n_resamples):
        sampled_parents = rng.choice(parents, size=n_p, replace=True)
        boot_vals = []
        for p in sampled_parents:
            boot_vals.extend(diffs_by_parent[p])
        boot_stats.append(float(stat_fn(boot_vals)))

    ci_low = float(np.percentile(boot_stats, 2.5))
    ci_high = float(np.percentile(boot_stats, 97.5))
    return point_est, ci_low, ci_high


def main():
    strain_file = BASE_DIR / "strain_results.json"
    contract_file = BASE_DIR / "evaluation_contract_g1_g3.json"
    out_file = BASE_DIR / "g1_g3_summary_analysis.json"

    strain_data = json.loads(strain_file.read_text())["results"]
    contract = json.loads(contract_file.read_text())

    tail_parents = contract["panel_and_seeds"]["parents"]
    seeds = [str(s) for s in contract["panel_and_seeds"]["seeds"]]
    arms = list(contract["arms"].keys())

    # Load records for solver warnings and dE
    records = {arm: {} for arm in arms}
    for arm in arms:
        rec_dir = BASE_DIR / arm / "records"
        for p in tail_parents:
            records[arm][p] = {}
            for s in seeds:
                f = rec_dir / f"{p}_{s}.json"
                if f.exists():
                    records[arm][p][s] = json.loads(f.read_text())

    analysis = {
        "arms": {},
        "comparisons": {},
        "guardrail_checks": {},
    }

    # Per-arm metrics
    for arm in arms:
        top1_strains = []
        floor_strains = []
        top1_dEs = []
        floor_dEs = []
        wall_solves = []
        infeasible_replaced = 0
        replay_fallbacks = 0
        obscene_count = 0
        total_units = 0

        parent_top1_strain = {p: [] for p in tail_parents}
        parent_top1_dE = {p: [] for p in tail_parents}
        parent_floor_dE = {p: [] for p in tail_parents}

        for p in tail_parents:
            for s in seeds:
                st = strain_data[arm][p][s]
                rec = records[arm][p][s]

                t1_str = st["top1"]["strain_kj"]
                fl_str = st["floor"]["strain_kj"]
                top1_strains.append(t1_str)
                floor_strains.append(fl_str)
                parent_top1_strain[p].append(t1_str)

                t1_dE = rec["top1_dE"]
                fl_dE = rec["floor_dE"]
                top1_dEs.append(t1_dE)
                floor_dEs.append(fl_dE)
                parent_top1_dE[p].append(t1_dE)
                parent_floor_dE[p].append(fl_dE)

                wall_solves.append(rec["wall_solve_s"])
                infeasible_replaced += rec.get("infeasible_warnings", {}).get("infeasible_replaced", 0)
                replay_fallbacks += rec.get("replay_fallbacks", 0)
                if t1_dE > 200.0:
                    obscene_count += 1
                total_units += 1

        analysis["arms"][arm] = {
            "median_top1_strain_kj": float(np.median(top1_strains)),
            "mean_top1_strain_kj": float(np.mean(top1_strains)),
            "median_floor_strain_kj": float(np.median(floor_strains)),
            "mean_floor_strain_kj": float(np.mean(floor_strains)),
            "median_top1_dE": float(np.median(top1_dEs)),
            "mean_top1_dE": float(np.mean(top1_dEs)),
            "median_floor_dE": float(np.median(floor_dEs)),
            "mean_floor_dE": float(np.mean(floor_dEs)),
            "obscene_units": obscene_count,
            "obscene_rate": float(obscene_count / total_units),
            "mean_wall_solve_s": float(np.mean(wall_solves)),
            "total_infeasible_replaced": infeasible_replaced,
            "total_replay_fallbacks": replay_fallbacks,
            "parent_top1_strain": parent_top1_strain,
            "parent_top1_dE": parent_top1_dE,
            "parent_floor_dE": parent_floor_dE,
        }

    # Comparisons against control
    ctrl = analysis["arms"]["control"]
    for arm in arms:
        if arm == "control":
            continue

        # Per-unit pairwise differences
        d_top1_strain = {p: [] for p in tail_parents}
        d_top1_dE = {p: [] for p in tail_parents}
        d_floor_dE = {p: [] for p in tail_parents}

        for p in tail_parents:
            for s_idx in range(len(seeds)):
                d_top1_strain[p].append(analysis["arms"][arm]["parent_top1_strain"][p][s_idx] - ctrl["parent_top1_strain"][p][s_idx])
                d_top1_dE[p].append(analysis["arms"][arm]["parent_top1_dE"][p][s_idx] - ctrl["parent_top1_dE"][p][s_idx])
                d_floor_dE[p].append(analysis["arms"][arm]["parent_floor_dE"][p][s_idx] - ctrl["parent_floor_dE"][p][s_idx])

        # Bootstraps
        p_strain, ci_s_lo, ci_s_hi = cluster_bootstrap(d_top1_strain, stat_fn=np.median)
        p_dE, ci_dE_lo, ci_dE_hi = cluster_bootstrap(d_top1_dE, stat_fn=np.median)
        p_fl, ci_fl_lo, ci_fl_hi = cluster_bootstrap(d_floor_dE, stat_fn=np.mean)

        # Max parent top1 degradation
        parent_mean_dE_diffs = {p: float(np.mean(d_top1_dE[p])) for p in tail_parents}
        max_parent_deg = max(parent_mean_dE_diffs.values())
        max_parent_deg_name = max(parent_mean_dE_diffs, key=parent_mean_dE_diffs.get)

        # Mean floor degradation
        mean_floor_deg = float(np.mean([v for vals in d_floor_dE.values() for v in vals]))

        # Guardrails check
        gr_parent_pass = bool(max_parent_deg <= 25.0)
        gr_floor_pass = bool(mean_floor_deg <= 0.0)
        gr_infeas_pass = bool(analysis["arms"][arm]["total_infeasible_replaced"] <= ctrl["total_infeasible_replaced"])
        gr_replay_pass = bool(analysis["arms"][arm]["total_replay_fallbacks"] == 0)

        all_gr_pass = gr_parent_pass and gr_floor_pass and gr_infeas_pass and gr_replay_pass

        analysis["comparisons"][arm] = {
            "delta_median_top1_strain_kj": {
                "estimate": p_strain,
                "ci_95": [ci_s_lo, ci_s_hi],
            },
            "delta_median_top1_dE_kj": {
                "estimate": p_dE,
                "ci_95": [ci_dE_lo, ci_dE_hi],
            },
            "delta_mean_floor_dE_kj": {
                "estimate": p_fl,
                "ci_95": [ci_fl_lo, ci_fl_hi],
            },
            "delta_obscene_rate": analysis["arms"][arm]["obscene_rate"] - ctrl["obscene_rate"],
            "max_parent_top1_degradation_kj": {
                "parent": max_parent_deg_name,
                "value": max_parent_deg,
                "passed": gr_parent_pass,
            },
            "mean_floor_degradation_kj": {
                "value": mean_floor_deg,
                "passed": gr_floor_pass,
            },
            "guardrails": {
                "parent_top1_unrelaxed_le_25": gr_parent_pass,
                "mean_floor_unrelaxed_le_0": gr_floor_pass,
                "no_infeasible_increase": gr_infeas_pass,
                "replay_fallback_zero": gr_replay_pass,
                "overall_passed": all_gr_pass,
            },
            "recommendation": "PROMOTE to cal panel (seeds 11-13)" if all_gr_pass and (p_strain < 0 or p_dE < 0) else ("REJECT / ARCHIVE" if not all_gr_pass else "NEUTRAL / INCONCLUSIVE"),
        }

    out_file.write_text(json.dumps(analysis, indent=2))
    print(f"Summary analysis written to {out_file}")

    # Print summary tables
    print("\n" + "=" * 80)
    print("PACKAGE G: SCREENING RESULTS TABLE (Tail Seeds 11-13, 30 units/arm)")
    print("=" * 80)
    print(f"{'Arm':<20} | {'Med Strain':<10} | {'dStrain [95% CI]':<22} | {'Med dE':<8} | {'Floor dE':<8} | {'Obscene':<7} | {'GR Pass'}")
    print("-" * 80)
    for arm in arms:
        med_str = f"{analysis['arms'][arm]['median_top1_strain_kj']:.1f}"
        med_dE = f"{analysis['arms'][arm]['median_top1_dE']:.1f}"
        fl_dE = f"{analysis['arms'][arm]['mean_floor_dE']:.1f}"
        obs = f"{analysis['arms'][arm]['obscene_rate']*100:.1f}%"
        if arm == "control":
            print(f"{arm:<20} | {med_str:<10} | {'[reference control]':<22} | {med_dE:<8} | {fl_dE:<8} | {obs:<7} | {'REF'}")
        else:
            cmp = analysis["comparisons"][arm]
            ds_str = f"{cmp['delta_median_top1_strain_kj']['estimate']:+.1f} [{cmp['delta_median_top1_strain_kj']['ci_95'][0]:+.1f}, {cmp['delta_median_top1_strain_kj']['ci_95'][1]:+.1f}]"
            gr = "PASS" if cmp["guardrails"]["overall_passed"] else "FAIL"
            print(f"{arm:<20} | {med_str:<10} | {ds_str:<22} | {med_dE:<8} | {fl_dE:<8} | {obs:<7} | {gr}")
    print("=" * 80)


if __name__ == "__main__":
    main()
