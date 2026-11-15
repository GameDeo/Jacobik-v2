#!/usr/bin/env python3
"""Package C Confirmation Analysis Script.

Computes parent-clustered bootstrap 95% CIs (10,000 resamples), unit-level CIs,
win/loss/tie counts, catastrophic (> 500 kJ/mol) and obscene (> 200 kJ/mol) counts,
floor deltas, ring-parent subset analysis on cal, and markdown summary tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CAL_RING_PARENTS = ["AWONOR", "USAPOU", "XOJHIN"]


def parent_bootstrap_ci(parent_deltas: Dict[str, float], n_resamples: int = 10000, seed: int = 42) -> Tuple[float, float]:
    """10,000 parent-clustered bootstrap resamples over parent-mean deltas."""
    rng = np.random.default_rng(seed)
    parents = list(parent_deltas.keys())
    vals = np.array([parent_deltas[p] for p in parents], dtype=float)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(vals[idx])
    return (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def unit_bootstrap_ci(unit_deltas: List[float], n_resamples: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.array(unit_deltas, dtype=float)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(vals[idx])
    return (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def analyze_panel(panel_dir: Path, seeds: List[int], arm_name: str, ctrl_name: str = "control") -> dict:
    ctrl_dir = panel_dir / ctrl_name / "records"
    cand_dir = panel_dir / arm_name / "records"

    ctrl_files = list(ctrl_dir.glob("*_*.json"))
    parents = sorted(list({f.stem.split("_")[0] for f in ctrl_files}))

    unit_records = []
    p_deltas_top1 = {}
    p_deltas_floor = {}
    p_ctrl_top1 = {}
    p_cand_top1 = {}

    ctrl_catastrophic = 0
    cand_catastrophic = 0
    ctrl_obscene = 0
    cand_obscene = 0
    total_fallbacks = 0

    unit_d_top1 = []
    unit_d_floor = []

    for code in parents:
        c_top_list, cand_top_list = [], []
        c_flr_list, cand_flr_list = [], []

        for s in seeds:
            cp = ctrl_dir / f"{code}_{s}.json"
            ap = cand_dir / f"{code}_{s}.json"
            if not cp.exists() or not ap.exists():
                continue
            c_rec = json.loads(cp.read_text())
            a_rec = json.loads(ap.read_text())

            c_top = c_rec.get("top1_dE")
            a_top = a_rec.get("top1_dE")
            c_flr = c_rec.get("floor_dE")
            a_flr = a_rec.get("floor_dE")

            fb = a_rec.get("replay_fallbacks", 0)
            total_fallbacks += fb

            if c_top is not None:
                if c_top > 500.0:
                    ctrl_catastrophic += 1
                if c_top > 200.0:
                    ctrl_obscene += 1
            if a_top is not None:
                if a_top > 500.0:
                    cand_catastrophic += 1
                if a_top > 200.0:
                    cand_obscene += 1

            if c_top is not None and a_top is not None:
                d_top = a_top - c_top
                d_flr = a_flr - c_flr if (a_flr is not None and c_flr is not None) else 0.0
                c_top_list.append(c_top)
                cand_top_list.append(a_top)
                c_flr_list.append(c_flr)
                cand_flr_list.append(a_flr)
                unit_d_top1.append(d_top)
                unit_d_floor.append(d_flr)
                unit_records.append({
                    "code": code,
                    "seed": s,
                    "ctrl_top1": c_top,
                    "cand_top1": a_top,
                    "delta_top1": d_top,
                    "ctrl_floor": c_flr,
                    "cand_floor": a_flr,
                    "delta_floor": d_flr,
                })

        if c_top_list and cand_top_list:
            p_deltas_top1[code] = float(np.mean(np.array(cand_top_list) - np.array(c_top_list)))
            p_deltas_floor[code] = float(np.mean(np.array(cand_flr_list) - np.array(c_flr_list)))
            p_ctrl_top1[code] = float(np.mean(c_top_list))
            p_cand_top1[code] = float(np.mean(cand_top_list))

    # Parent wins, losses, ties (tolerance 0.05 kJ/mol)
    p_wins = sum(1 for d in p_deltas_top1.values() if d < -0.05)
    p_losses = sum(1 for d in p_deltas_top1.values() if d > 0.05)
    p_ties = sum(1 for d in p_deltas_top1.values() if abs(d) <= 0.05)

    u_wins = sum(1 for d in unit_d_top1 if d < -0.05)
    u_losses = sum(1 for d in unit_d_top1 if d > 0.05)
    u_ties = sum(1 for d in unit_d_top1 if abs(d) <= 0.05)

    p_ci_top1 = parent_bootstrap_ci(p_deltas_top1)
    p_ci_floor = parent_bootstrap_ci(p_deltas_floor)
    u_ci_top1 = unit_bootstrap_ci(unit_d_top1)

    mean_d_top1 = float(np.mean(unit_d_top1)) if unit_d_top1 else 0.0
    median_d_top1 = float(np.median(unit_d_top1)) if unit_d_top1 else 0.0
    mean_p_d_top1 = float(np.mean(list(p_deltas_top1.values()))) if p_deltas_top1 else 0.0
    mean_d_floor = float(np.mean(unit_d_floor)) if unit_d_floor else 0.0
    median_d_floor = float(np.median(unit_d_floor)) if unit_d_floor else 0.0

    return dict(
        arm=arm_name,
        n_parents=len(p_deltas_top1),
        n_units=len(unit_d_top1),
        mean_delta_top1=mean_d_top1,
        median_delta_top1=median_d_top1,
        mean_parent_delta_top1=mean_p_d_top1,
        parent_bootstrap_ci_95=p_ci_top1,
        unit_bootstrap_ci_95=u_ci_top1,
        mean_delta_floor=mean_d_floor,
        median_delta_floor=median_d_floor,
        parent_bootstrap_ci_floor_95=p_ci_floor,
        parent_w_l_t=(p_wins, p_losses, p_ties),
        unit_w_l_t=(u_wins, u_losses, u_ties),
        ctrl_catastrophic_gt500=ctrl_catastrophic,
        cand_catastrophic_gt500=cand_catastrophic,
        ctrl_obscene_gt200=ctrl_obscene,
        cand_obscene_gt200=cand_obscene,
        total_fallbacks=total_fallbacks,
        parent_deltas_top1=p_deltas_top1,
        parent_deltas_floor=p_deltas_floor,
        unit_records=unit_records,
    )


def analyze_ring_subset(panel_analysis: dict, ring_parents: List[str] = CAL_RING_PARENTS) -> dict:
    p_d_top = {p: panel_analysis["parent_deltas_top1"][p] for p in ring_parents if p in panel_analysis["parent_deltas_top1"]}
    p_d_flr = {p: panel_analysis["parent_deltas_floor"][p] for p in ring_parents if p in panel_analysis["parent_deltas_floor"]}
    u_recs = [u for u in panel_analysis["unit_records"] if u["code"] in ring_parents]

    u_top = [u["delta_top1"] for u in u_recs]
    u_flr = [u["delta_floor"] for u in u_recs]

    mean_top = float(np.mean(u_top)) if u_top else 0.0
    median_top = float(np.median(u_top)) if u_top else 0.0
    mean_flr = float(np.mean(u_flr)) if u_flr else 0.0
    ci_top = parent_bootstrap_ci(p_d_top) if p_d_top else (0.0, 0.0)

    per_parent = {}
    for p in ring_parents:
        p_units = [u for u in u_recs if u["code"] == p]
        if p_units:
            per_parent[p] = {
                "mean_delta_top1": float(np.mean([u["delta_top1"] for u in p_units])),
                "mean_delta_floor": float(np.mean([u["delta_floor"] for u in p_units])),
                "seeds": {u["seed"]: {"delta_top1": u["delta_top1"], "delta_floor": u["delta_floor"]} for u in p_units}
            }

    return dict(
        n_parents=len(p_d_top),
        n_units=len(u_recs),
        mean_delta_top1=mean_top,
        median_delta_top1=median_top,
        mean_delta_floor=mean_flr,
        parent_bootstrap_ci_95=ci_top,
        per_parent=per_parent,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--seeds", default="14,15,16")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    base = Path(args.base_dir)

    res_ring = analyze_panel(base, seeds, "ring_0.5")
    res_comb = analyze_panel(base, seeds, "combined")

    out_payload = {
        "ring_0.5": res_ring,
        "combined": res_comb,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out_payload, indent=2))
        print(f"Results written to {args.out}")
