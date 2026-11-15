#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

ctrl_dir = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/confirmation/cal/b1_confirm_control/records")
cand_dir = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/confirmation/cal/b1_confirm_bend_release_0.1_100/records")

cand_files = sorted(list(cand_dir.glob("*.json")))

units = []
all_deltas = []
all_floor_deltas = []
all_bp_deltas = []
ctrl_top1s = []
cand_top1s = []
ctrl_floors = []
cand_floors = []
better = worse = neutral = 0

per_parent = {}

for f_cand in cand_files:
    f_ctrl = ctrl_dir / f_cand.name
    if not f_ctrl.exists():
        continue
    d_cand = json.loads(f_cand.read_text())
    d_ctrl = json.loads(f_ctrl.read_text())

    code = d_cand["code"]
    seed = d_cand["seed"]

    if d_cand["status"] != "ok" or d_ctrl["status"] != "ok":
        continue

    t_ctrl = d_ctrl["top1_dE"]
    t_cand = d_cand["top1_dE"]
    fl_ctrl = d_ctrl["floor_dE"]
    fl_cand = d_cand["floor_dE"]
    bp_ctrl = d_ctrl.get("top1_bend_prior")
    bp_cand = d_cand.get("top1_bend_prior")

    delta = t_cand - t_ctrl
    delta_fl = fl_cand - fl_ctrl

    all_deltas.append(delta)
    all_floor_deltas.append(delta_fl)
    ctrl_top1s.append(t_ctrl)
    cand_top1s.append(t_cand)
    ctrl_floors.append(fl_ctrl)
    cand_floors.append(fl_cand)

    if bp_ctrl is not None and bp_cand is not None:
        all_bp_deltas.append(bp_cand - bp_ctrl)

    if delta < -0.01:
        better += 1
    elif delta > 0.01:
        worse += 1
    else:
        neutral += 1

    if code not in per_parent:
        per_parent[code] = []
    per_parent[code].append(delta)

all_deltas = np.array(all_deltas)
all_floor_deltas = np.array(all_floor_deltas)
ctrl_top1s = np.array(ctrl_top1s)
cand_top1s = np.array(cand_top1s)

print(f"Total Paired Units Evaluated: {len(all_deltas)} / 360")
print("=== CAL PANEL (120 PARENTS × SEEDS 11,12,13) CONFIRMATION SUMMARY ===")
print(f"Control Mean Top-1 dE:   {ctrl_top1s.mean():.2f} kJ/mol")
print(f"Candidate Mean Top-1 dE: {cand_top1s.mean():.2f} kJ/mol")
print(f"Mean Delta Top-1 dE:     {all_deltas.mean():.2f} kJ/mol")
print(f"Median Delta Top-1 dE:   {np.median(all_deltas):.2f} kJ/mol")
print(f"Std Delta Top-1 dE:      {all_deltas.std():.2f} kJ/mol")
print(f"Units Better / Worse / Neutral: {better} / {worse} / {neutral} ({better/len(all_deltas)*100:.1f}% / {worse/len(all_deltas)*100:.1f}% / {neutral/len(all_deltas)*100:.1f}%)")
print(f"Units > 500 kJ/mol:      Candidate {(cand_top1s > 500).sum()} vs Control {(ctrl_top1s > 500).sum()} (out of {len(all_deltas)})")
print(f"Control Mean Floor dE:   {np.mean(ctrl_floors):.2f} kJ/mol")
print(f"Candidate Mean Floor dE: {np.mean(cand_floors):.2f} kJ/mol")
print(f"Mean Delta Floor dE:     {all_floor_deltas.mean():.2f} kJ/mol")
print(f"Median Delta Floor dE:   {np.median(all_floor_deltas):.2f} kJ/mol")
if all_bp_deltas:
    print(f"Mean Delta Top-1 Bend Prior: {np.mean(all_bp_deltas):.4f}")

# Bootstrap 95% CI on mean Delta top-1 dE
np.random.seed(42)
boot_means = [np.random.choice(all_deltas, size=len(all_deltas), replace=True).mean() for _ in range(10000)]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"Bootstrap 95% CI for Mean Delta Top-1 dE: [{ci_low:.2f}, {ci_high:.2f}] kJ/mol")

# Non-inferiority bound test: upper bound of 95% CI vs +2.0 kJ/mol (or 0 kJ/mol)
print(f"Upper 95% CI bound: {ci_high:.2f} kJ/mol")

# Parent-level win rate
parent_means = {c: np.mean(ds) for c, ds in per_parent.items()}
p_better = sum(1 for m in parent_means.values() if m < -0.01)
p_worse = sum(1 for m in parent_means.values() if m > 0.01)
p_neutral = sum(1 for m in parent_means.values() if abs(m) <= 0.01)
print(f"Parent-level Win/Loss/Tie: {p_better} better / {p_worse} worse / {p_neutral} neutral (out of {len(parent_means)} parents)")

# Top 5 most improved and top 5 most degraded parents on cal
sorted_parents = sorted(parent_means.items(), key=lambda x: x[1])
print("\nTop 5 Most Improved Parents on Cal:")
for c, m in sorted_parents[:5]:
    print(f"  {c:8s}: mean Delta dE = {m:+7.2f} kJ/mol")

print("\nTop 5 Most Degraded Parents on Cal:")
for c, m in sorted_parents[-5:]:
    print(f"  {c:8s}: mean Delta dE = {m:+7.2f} kJ/mol")
