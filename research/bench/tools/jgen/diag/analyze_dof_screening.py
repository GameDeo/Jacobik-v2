#!/usr/bin/env python3
"""Analyze Checkpoint 2.3 dof_step_scale tail screening results."""
import json
from pathlib import Path
import numpy as np

BASE = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/screening")
CTRL_DIR = BASE / "b1_screen_control/records"

ARMS = [
    ("b1_screen_dof_rigid_1.5", "rigid=1.5"),
    ("b1_screen_dof_torsion_1.25", "torsion=1.25"),
    ("b1_screen_dof_ring_0.5", "ring=0.5"),
]

CODES = ["DAMVAQ", "HOXJUZ", "IFUROQ", "JEGPUJ", "NEPNEF", "SOBHAS", "USAPOU", "XEYTUT", "YUCZUS", "ZELTAP"]
SEEDS = [11, 12, 13]


def bootstrap_parent_ci(per_parent_deltas, n_boot=10000, seed=42):
    """Parent-clustered bootstrap: resample whole parents."""
    rng = np.random.default_rng(seed)
    parents = list(per_parent_deltas.keys())
    means = []
    for _ in range(n_boot):
        sample_parents = rng.choice(parents, size=len(parents), replace=True)
        sample_deltas = [d for p in sample_parents for d in per_parent_deltas[p]]
        means.append(np.mean(sample_deltas))
    return np.percentile(means, [2.5, 97.5])


def main():
    print("=" * 100)
    print("CHECKPOINT 2.3: dof_step_scale TAIL SCREENING (seeds 11, 12, 13 vs same-VM Control)")
    print("=" * 100)

    for arm_name, desc in ARMS:
        cand_dir = BASE / arm_name / "records"
        if not cand_dir.exists():
            print(f"[{arm_name}] Pending...")
            continue

        all_deltas = []
        all_floor_deltas = []
        ctrl_top1s = []
        cand_top1s = []
        per_parent = {c: [] for c in CODES}
        per_parent_floor = {c: [] for c in CODES}
        better = worse = neutral = 0
        fallbacks = 0

        for c in CODES:
            for s in SEEDS:
                fc = cand_dir / f"{c}_{s}.json"
                fb = CTRL_DIR / f"{c}_{s}.json"
                if not fc.exists() or not fb.exists():
                    continue
                dc = json.loads(fc.read_text())
                db = json.loads(fb.read_text())

                fallbacks += dc.get("replay_fallbacks", 0)

                t_ctrl = db["top1_dE"]
                t_cand = dc["top1_dE"]
                fl_ctrl = db["floor_dE"]
                fl_cand = dc["floor_dE"]

                delta = t_cand - t_ctrl
                delta_fl = fl_cand - fl_ctrl

                all_deltas.append(delta)
                all_floor_deltas.append(delta_fl)
                ctrl_top1s.append(t_ctrl)
                cand_top1s.append(t_cand)

                per_parent[c].append(delta)
                per_parent_floor[c].append(delta_fl)

                if delta < -0.01:
                    better += 1
                elif delta > 0.01:
                    worse += 1
                else:
                    neutral += 1

        all_deltas = np.array(all_deltas)
        all_floor_deltas = np.array(all_floor_deltas)
        ctrl_top1s = np.array(ctrl_top1s)
        cand_top1s = np.array(cand_top1s)

        if len(all_deltas) == 0:
            print(f"[{arm_name}] No units found.")
            continue

        ci_parent_low, ci_parent_high = bootstrap_parent_ci(per_parent)

        print(f"\nArm: {arm_name} ({desc}) | Units: {len(all_deltas)}/30 | Fallbacks: {fallbacks}")
        print(f"  Mean Delta top-1 dE:   {all_deltas.mean():+8.2f} kJ/mol (median: {np.median(all_deltas):+8.2f}, std: {all_deltas.std():.2f})")
        print(f"  Parent-clustered 95% CI: [{ci_parent_low:+8.2f}, {ci_parent_high:+8.2f}] kJ/mol")
        print(f"  Units Better/Worse/Tie: {better} / {worse} / {neutral} (out of {len(all_deltas)})")
        print(f"  Units > 500 kJ/mol:    Cand {(cand_top1s > 500).sum()} vs Ctrl {(ctrl_top1s > 500).sum()}")
        print(f"  Mean Delta Floor dE:   {all_floor_deltas.mean():+8.2f} kJ/mol")

        # Exclude JEGPUJ and USAPOU
        excl_deltas = [d for c in CODES if c not in ("JEGPUJ", "USAPOU") for d in per_parent[c]]
        excl_deltas = np.array(excl_deltas)
        if len(excl_deltas) > 0:
            print(f"  Excl JEGPUJ/USAPOU:    Mean Delta dE: {excl_deltas.mean():+8.2f} kJ/mol (median: {np.median(excl_deltas):+8.2f})")

        print("  Per-Parent Breakdown (mean Delta top-1 dE):")
        for c in CODES:
            ds = per_parent[c]
            ds_str = ", ".join(f"{d:+7.1f}" for d in ds) if ds else "none"
            m = np.mean(ds) if ds else 0.0
            print(f"    {c:8s}: {m:+7.2f} kJ/mol | seeds: [{ds_str}]")


if __name__ == "__main__":
    main()
