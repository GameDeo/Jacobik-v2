#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

ctrl_dir = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/confirmation/tail/b1_confirm_control/records")
cand_dir = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/confirmation/tail/b1_confirm_bend_release_0.1_100/records")

codes = ["DAMVAQ", "HOXJUZ", "IFUROQ", "JEGPUJ", "NEPNEF", "SOBHAS", "USAPOU", "XEYTUT", "YUCZUS", "ZELTAP"]
seeds = [14, 15, 16]

per_parent = {}
all_deltas = []
all_floor_deltas = []
all_bend_prior_deltas = []
ctrl_top1s = []
cand_top1s = []
ctrl_floors = []
cand_floors = []

for c in codes:
    per_parent[c] = {
        "dE": [],
        "floor_dE": [],
        "cand_dE": [],
        "ctrl_dE": [],
        "cand_floor": [],
        "ctrl_floor": [],
        "top1_bp_ctrl": [],
        "top1_bp_cand": [],
    }
    for s in seeds:
        f_ctrl = ctrl_dir / f"{c}_{s}.json"
        f_cand = cand_dir / f"{c}_{s}.json"
        if not f_ctrl.exists() or not f_cand.exists():
            continue
        d_ctrl = json.loads(f_ctrl.read_text())
        d_cand = json.loads(f_cand.read_text())

        t_ctrl = d_ctrl["top1_dE"]
        t_cand = d_cand["top1_dE"]
        fl_ctrl = d_ctrl["floor_dE"]
        fl_cand = d_cand["floor_dE"]
        bp_ctrl = d_ctrl.get("top1_bend_prior")
        bp_cand = d_cand.get("top1_bend_prior")

        delta = t_cand - t_ctrl
        delta_floor = fl_cand - fl_ctrl
        all_deltas.append(delta)
        all_floor_deltas.append(delta_floor)
        ctrl_top1s.append(t_ctrl)
        cand_top1s.append(t_cand)
        ctrl_floors.append(fl_ctrl)
        cand_floors.append(fl_cand)

        per_parent[c]["dE"].append(delta)
        per_parent[c]["floor_dE"].append(delta_floor)
        per_parent[c]["ctrl_dE"].append(t_ctrl)
        per_parent[c]["cand_dE"].append(t_cand)
        per_parent[c]["ctrl_floor"].append(fl_ctrl)
        per_parent[c]["cand_floor"].append(fl_cand)
        if bp_ctrl is not None and bp_cand is not None:
            per_parent[c]["top1_bp_ctrl"].append(bp_ctrl)
            per_parent[c]["top1_bp_cand"].append(bp_cand)
            all_bend_prior_deltas.append(bp_cand - bp_ctrl)

all_deltas = np.array(all_deltas)
all_floor_deltas = np.array(all_floor_deltas)
ctrl_top1s = np.array(ctrl_top1s)
cand_top1s = np.array(cand_top1s)

print(f"Total units evaluated: {len(all_deltas)} / {len(codes)*len(seeds)}")
if len(all_deltas) == 0:
    exit(0)

print("=== TAIL FRESH SEEDS (14, 15, 16) CONFIRMATION SUMMARY ===")
print(f"Control Mean Top-1 dE:   {ctrl_top1s.mean():.2f} kJ/mol")
print(f"Candidate Mean Top-1 dE: {cand_top1s.mean():.2f} kJ/mol")
print(f"Mean Delta Top-1 dE:     {all_deltas.mean():.2f} kJ/mol")
print(f"Median Delta Top-1 dE:   {np.median(all_deltas):.2f} kJ/mol")
print(f"Std Delta Top-1 dE:      {all_deltas.std():.2f} kJ/mol")
print(f"Units Better / Worse / Neutral: {(all_deltas < -0.01).sum()} / {(all_deltas > 0.01).sum()} / {(np.abs(all_deltas) <= 0.01).sum()}")
print(f"Units > 500 kJ/mol:      Candidate {(cand_top1s > 500).sum()} vs Control {(ctrl_top1s > 500).sum()} (out of {len(all_deltas)})")
print(f"Control Mean Floor dE:   {np.mean(ctrl_floors):.2f} kJ/mol")
print(f"Candidate Mean Floor dE: {np.mean(cand_floors):.2f} kJ/mol")
print(f"Mean Delta Floor dE:     {all_floor_deltas.mean():.2f} kJ/mol")
print(f"Median Delta Floor dE:   {np.median(all_floor_deltas):.2f} kJ/mol")
if all_bend_prior_deltas:
    print(f"Mean Delta Top-1 Bend Prior: {np.mean(all_bend_prior_deltas):.4f}")

# Bootstrap 95% CI
np.random.seed(42)
boot_means = [np.random.choice(all_deltas, size=len(all_deltas), replace=True).mean() for _ in range(10000)]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"Bootstrap 95% CI for Mean Delta Top-1 dE: [{ci_low:.2f}, {ci_high:.2f}] kJ/mol")

# Exclude JEGPUJ and USAPOU
excl_deltas = [d for c in codes if c not in ("JEGPUJ", "USAPOU") for d in per_parent[c]["dE"]]
excl_deltas = np.array(excl_deltas)
if len(excl_deltas) > 0:
    print("\n=== CONCENTRATION CHECK: EXCLUDING JEGPUJ AND USAPOU ===")
    print(f"Evaluated Units: {len(excl_deltas)}")
    print(f"Mean Delta Top-1 dE (excl):   {excl_deltas.mean():.2f} kJ/mol")
    print(f"Median Delta Top-1 dE (excl): {np.median(excl_deltas):.2f} kJ/mol")
    print(f"Units Better / Worse / Neutral: {(excl_deltas < -0.01).sum()} / {(excl_deltas > 0.01).sum()} / {(np.abs(excl_deltas) <= 0.01).sum()}")

print("\n=== PER-PARENT CONFIRMATION BREAKDOWN ===")
for c in codes:
    deltas = per_parent[c]["dE"]
    if not deltas:
        print(f"{c:8s}: (pending)")
        continue
    s_strs = ", ".join(f"{d:+.2f}" for d in deltas)
    mean_d = np.mean(deltas)
    mean_fl = np.mean(per_parent[c]["floor_dE"])
    bp_c = np.mean(per_parent[c]["top1_bp_ctrl"]) if per_parent[c]["top1_bp_ctrl"] else None
    bp_cand_m = np.mean(per_parent[c]["top1_bp_cand"]) if per_parent[c]["top1_bp_cand"] else None
    bp_str = f"top1_bp: ctrl={bp_c:.3f}, cand={bp_cand_m:.3f}" if bp_c is not None else "no bends"
    print(f"{c:8s}: mean dE = {mean_d:+7.2f} kJ/mol | floor dE = {mean_fl:+7.2f} kJ/mol | seeds [14,15,16]: [{s_strs}] | {bp_str}")
