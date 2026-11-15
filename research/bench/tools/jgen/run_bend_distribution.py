"""Measure bend column distribution at delivery for control vs g3_bend_free_1.0 (Package G).

Runs tail panel across seeds 11, 12, 13.
Extracts delivered bend angles from solver state thetas across all conformers and ligands.
Reports fraction of bend DOFs outside [-1°, 1°] and [-2°, 2°].
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/bench"))

from tools.jgen import inputs
from jacobik.core.env import build_builder_and_inputs
from tools.jdev.budget.common import stats_from_db
import jacobik.core.profile as profile
from jacobik.ik import angle_bends as AB
from jacobik.ik.fk import ForwardKinematics


def main():
    parser = argparse.ArgumentParser(description="Bend column distribution measurement")
    parser.add_argument("--out", default=str(REPO / "research/data/jgen/campaign/B/g/bend_distribution_results.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tail_parents = ["DAMVAQ", "HOXJUZ", "IFUROQ", "JEGPUJ", "NEPNEF", "SOBHAS", "USAPOU", "XEYTUT", "YUCZUS", "ZELTAP"]
    seeds = [11, 12, 13]
    arms = [
        {"arm": "control", "bend_free_deg": 2.0},
        {"arm": "g3_bend_free_1.0", "bend_free_deg": 1.0},
    ]

    shards = inputs.frame_shards()
    stats = stats_from_db(inputs.DB_PATH)

    results = {}

    for arm_cfg in arms:
        arm_name = arm_cfg["arm"]
        bf_deg = arm_cfg["bend_free_deg"]
        print(f"\n--- Running arm: {arm_name} (bend_free_deg={bf_deg}) on {args.device} ---")

        profile._VALUES["bend_free_deg"] = bf_deg
        profile.OVERRIDES["bend_free_deg"] = bf_deg
        AB.FREE_DEG = bf_deg

        arm_angles_all = []
        arm_angles_top1 = []
        parent_stats = {}

        t0 = time.perf_counter()
        for code in tail_parents:
            sh, ds = shards[code]
            mol = inputs.load_mol(code, sh, ds)
            builder, _, err = build_builder_and_inputs(mol)
            if builder is None:
                print(f"Skipping {code}: {err}")
                continue

            # Update fk bend free zones if any
            # (handled inside builder.solve via bend_free_deg)

            parent_angles_all = []
            parent_angles_top1 = []

            # Precompute FK metadata for each ligand
            fks = [ForwardKinematics(mol, lig, args.device, angle_bends=True) for lig in mol.ligands]

            for seed in seeds:
                res = builder.solve(
                    seed=seed,
                    stats=stats,
                    angle_bends=True,
                    device=args.device,
                    n_conformers=16,
                    bend_free_deg=bf_deg,
                )

                for rank, conf in enumerate(res.conformers):
                    thetas = conf.solver_state.get("thetas", [])
                    # Extract from each ligand
                    conf_angles = []
                    for li, th in enumerate(thetas):
                        fk = fks[li]
                        n0 = 6 + fk.n_joints
                        nb = getattr(fk, "n_angle_bends", 0)
                        if nb > 0:
                            b_rad = np.asarray(th).reshape(-1)[n0 : n0 + nb]
                            b_deg = np.rad2deg(b_rad)
                            conf_angles.extend(b_deg.tolist())

                    parent_angles_all.extend(conf_angles)
                    if rank == 0:
                        parent_angles_top1.extend(conf_angles)

            arm_angles_all.extend(parent_angles_all)
            arm_angles_top1.extend(parent_angles_top1)

            arr_p = np.abs(parent_angles_all)
            parent_stats[code] = {
                "n_bends_sampled": len(arr_p),
                "frac_gt_1deg": float(np.mean(arr_p > 1.0)) if len(arr_p) > 0 else 0.0,
                "frac_gt_2deg": float(np.mean(arr_p > 2.0)) if len(arr_p) > 0 else 0.0,
                "mean_abs_deg": float(np.mean(arr_p)) if len(arr_p) > 0 else 0.0,
                "max_abs_deg": float(np.max(arr_p)) if len(arr_p) > 0 else 0.0,
            }
            print(f"  [{code}] bends sampled: {len(arr_p)}, >1°: {parent_stats[code]['frac_gt_1deg']*100:.1f}%, >2°: {parent_stats[code]['frac_gt_2deg']*100:.1f}%")

        arr_all = np.abs(arm_angles_all)
        arr_top1 = np.abs(arm_angles_top1)
        results[arm_name] = {
            "bend_free_deg": bf_deg,
            "wall_s": time.perf_counter() - t0,
            "all_conformers": {
                "total_bends_sampled": len(arr_all),
                "frac_gt_1deg": float(np.mean(arr_all > 1.0)) if len(arr_all) > 0 else 0.0,
                "frac_gt_2deg": float(np.mean(arr_all > 2.0)) if len(arr_all) > 0 else 0.0,
                "mean_abs_deg": float(np.mean(arr_all)) if len(arr_all) > 0 else 0.0,
                "median_abs_deg": float(np.median(arr_all)) if len(arr_all) > 0 else 0.0,
                "max_abs_deg": float(np.max(arr_all)) if len(arr_all) > 0 else 0.0,
            },
            "top1_conformers": {
                "total_bends_sampled": len(arr_top1),
                "frac_gt_1deg": float(np.mean(arr_top1 > 1.0)) if len(arr_top1) > 0 else 0.0,
                "frac_gt_2deg": float(np.mean(arr_top1 > 2.0)) if len(arr_top1) > 0 else 0.0,
                "mean_abs_deg": float(np.mean(arr_top1)) if len(arr_top1) > 0 else 0.0,
                "median_abs_deg": float(np.median(arr_top1)) if len(arr_top1) > 0 else 0.0,
                "max_abs_deg": float(np.max(arr_top1)) if len(arr_top1) > 0 else 0.0,
            },
            "parent_breakdown": parent_stats,
        }

    # Reset profile to default
    profile._VALUES["bend_free_deg"] = 2.0
    profile.OVERRIDES["bend_free_deg"] = 2.0
    AB.FREE_DEG = 2.0

    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved bend distribution results to {out_file}")

    print("\nSummary Comparison (All conformers):")
    print(f"Control (2.0°): >1° = {results['control']['all_conformers']['frac_gt_1deg']*100:.2f}%, >2° = {results['control']['all_conformers']['frac_gt_2deg']*100:.2f}%")
    print(f"G3 (1.0°):      >1° = {results['g3_bend_free_1.0']['all_conformers']['frac_gt_1deg']*100:.2f}%, >2° = {results['g3_bend_free_1.0']['all_conformers']['frac_gt_2deg']*100:.2f}%")

    print("\nSummary Comparison (Top-1 conformers):")
    print(f"Control (2.0°): >1° = {results['control']['top1_conformers']['frac_gt_1deg']*100:.2f}%, >2° = {results['control']['top1_conformers']['frac_gt_2deg']*100:.2f}%")
    print(f"G3 (1.0°):      >1° = {results['g3_bend_free_1.0']['top1_conformers']['frac_gt_1deg']*100:.2f}%, >2° = {results['g3_bend_free_1.0']['top1_conformers']['frac_gt_2deg']*100:.2f}%")


if __name__ == "__main__":
    main()
