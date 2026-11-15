"""S-P1: Beam torsion-mode coverage measurement (Package G).

Runs on the calibration panel ('cal') with seed 11.
For each parent with rotatable bonds (fk.n_joints_base > 0):
- Examines fitted Q5 VMM modes with mixture weight >= 0.05.
- Checks if every supported mode receives at least one beam row within 0.2 rad in beam_torsions (width 16).
- Flags parent if any supported mode has 0 beam rows within 0.2 rad.
- Reports overall percentage of parents with unfilled supported modes (gating beam_target_aware at >= 10%).
- Also evaluates beam width 48 on tail panel (seed 11) vs width 16.
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
from tools.jgen.bench import panels
from jacobik.core.env import build_builder_and_inputs
from tools.jdev.budget.common import stats_from_db
from jacobik.ik.fk import ForwardKinematics
from jacobik.ik.beam import BeamSearch
from jacobik.ik.geometry import GeometryExtractor
from jacobik.ik.scorer import Scorer


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def measure_panel_coverage(panel_name: str, beam_width: int, seed: int = 11, device: str = "cpu") -> dict:
    manifest = json.load(open(inputs.FRAME_V1 / "manifest.json"))
    tasks = panels.select(manifest, panel_name)
    units = tasks["units"]
    unique_codes = sorted({u["code"] for u in units})
    shards = inputs.frame_shards()
    stats = stats_from_db(inputs.DB_PATH)

    print(f"\nEvaluating panel '{panel_name}' ({len(unique_codes)} parents) at beam_width={beam_width}, seed={seed} on {device}...")

    total_parents_with_joints = 0
    parents_with_unfilled = 0
    total_supported_modes = 0
    total_unfilled_modes = 0

    parent_details = {}

    for idx, code in enumerate(unique_codes, start=1):
        if code not in shards:
            continue
        sh, ds = shards[code]
        try:
            mol = inputs.load_mol(code, sh, ds)
        except Exception:
            continue

        builder, _, err = build_builder_and_inputs(mol)
        if builder is None:
            continue

        parent_has_joints = False
        parent_unfilled_modes = []

        X_full = torch.tensor(mol.positions(), dtype=torch.float32, device=device)

        for li, lig in enumerate(mol.ligands):
            fk = ForwardKinematics(mol, lig, device, angle_bends=True)
            J = fk.n_joints_base
            if J == 0:
                continue

            parent_has_joints = True
            ge_li = GeometryExtractor(mol, device)
            scorer_li = Scorer(
                mol,
                ge_li,
                stats,
                device=device,
                quantity_weights=builder.quantity_weights if hasattr(builder, "quantity_weights") else None,
                orientation_model="q3q4",
            )

            rows, offs = fk.q5_alignment(ge_li, X_full)
            if scorer_li._q5_means.shape[0] == 0:
                continue

            has = rows >= 0
            r0 = rows.clamp(min=0)
            q5_m = wrap_angle(scorer_li._q5_means[r0] + offs[:, None])
            q5_k = scorer_li._q5_kappas[r0].clone()
            q5_w = scorer_li._q5_weights[r0].clone()
            if not bool(has.all()):
                miss = ~has
                q5_m[miss] = 0.0
                q5_m[miss, 0] = offs[miss]
                q5_k[miss] = 1.0
                q5_w[miss] = 0.0
                q5_w[miss, 0] = 1.0

            beam = BeamSearch(fk, beam_width=beam_width, device=device)
            beam_torsions, _ = beam.search(
                X_ref=X_full,
                base_theta=torch.zeros(1, 6, device=device),
                q5_means=q5_m,
                q5_kappas=q5_k,
                q5_weights=q5_w,
                q6_mu=scorer_li._q6_mu,
                q6_sigma=scorer_li._q6_sigma,
                q6_pairs=(ge_li._q6_a, ge_li._q6_b) if getattr(ge_li, "n_q6", 0) > 0 else None,
            )

            # Check coverage for each joint and each mode with weight >= 0.05
            for j in range(J):
                w_j = q5_w[j].detach().cpu().numpy()
                m_j = q5_m[j].detach().cpu().numpy()
                b_j = beam_torsions[:, j].detach().cpu().numpy()

                for c in range(len(w_j)):
                    if w_j[c] >= 0.05:
                        total_supported_modes += 1
                        target_angle = m_j[c]
                        # Compute angular distance to all beam rows
                        diff = np.abs(np.arctan2(np.sin(b_j - target_angle), np.cos(b_j - target_angle)))
                        min_dist = float(np.min(diff))
                        if min_dist > 0.2:
                            total_unfilled_modes += 1
                            parent_unfilled_modes.append({
                                "ligand": li,
                                "joint": j,
                                "mode": c,
                                "weight": float(w_j[c]),
                                "target_rad": float(target_angle),
                                "min_dist_rad": min_dist,
                            })

        if parent_has_joints:
            total_parents_with_joints += 1
            if len(parent_unfilled_modes) > 0:
                parents_with_unfilled += 1
                parent_details[code] = {
                    "unfilled_count": len(parent_unfilled_modes),
                    "unfilled_modes": parent_unfilled_modes,
                }

        if idx % 20 == 0 or idx == len(unique_codes):
            print(f"  Processed {idx}/{len(unique_codes)} parents... Parents with unfilled modes: {parents_with_unfilled}/{total_parents_with_joints} ({parents_with_unfilled/max(1, total_parents_with_joints)*100:.1f}%)", flush=True)

    frac_parents = parents_with_unfilled / max(1, total_parents_with_joints)
    frac_modes = total_unfilled_modes / max(1, total_supported_modes)

    res = {
        "panel": panel_name,
        "beam_width": beam_width,
        "seed": seed,
        "total_parents_evaluated": len(unique_codes),
        "total_parents_with_joints": total_parents_with_joints,
        "parents_with_unfilled_modes": parents_with_unfilled,
        "pct_parents_with_unfilled_modes": float(frac_parents * 100),
        "total_supported_modes": total_supported_modes,
        "total_unfilled_modes": total_unfilled_modes,
        "pct_unfilled_modes": float(frac_modes * 100),
        "gate_exceeded_10pct": bool(frac_parents >= 0.10),
        "unfilled_parent_details": parent_details,
    }
    return res


def main():
    parser = argparse.ArgumentParser(description="S-P1: Beam torsion-mode coverage")
    parser.add_argument("--out", default=str(REPO / "research/data/jgen/campaign/B/g/sp1_coverage_results.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Primary evaluation: cal panel, width 16, seed 11
    cal_res = measure_panel_coverage("cal", beam_width=16, seed=11, device=args.device)

    # 2. Secondary evaluation: tail panel width 16 vs width 48, seed 11
    tail_w16 = measure_panel_coverage("tail", beam_width=16, seed=11, device=args.device)
    tail_w48 = measure_panel_coverage("tail", beam_width=48, seed=11, device=args.device)

    final_results = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cal_width_16": cal_res,
        "tail_width_16": tail_w16,
        "tail_width_48": tail_w48,
    }

    out_file.write_text(json.dumps(final_results, indent=2))
    print(f"\nAll results saved to {out_file}")

    print("\n" + "=" * 60)
    print("S-P1 SUMMARY RESULTS:")
    print(f"Calibration panel ('cal', W=16):")
    print(f"  Parents with joints: {cal_res['total_parents_with_joints']}")
    print(f"  Parents with unfilled modes: {cal_res['parents_with_unfilled_modes']} ({cal_res['pct_parents_with_unfilled_modes']:.1f}%)")
    print(f"  Supported modes: {cal_res['total_supported_modes']}")
    print(f"  Unfilled modes: {cal_res['total_unfilled_modes']} ({cal_res['pct_unfilled_modes']:.1f}%)")
    print(f"  Gate threshold >= 10%: {'TRIGGERED (>=10%) -> beam_target_aware justified' if cal_res['gate_exceeded_10pct'] else 'NOT TRIGGERED (<10%)'}")
    print("\nTail panel:")
    print(f"  W=16: {tail_w16['parents_with_unfilled_modes']}/{tail_w16['total_parents_with_joints']} parents ({tail_w16['pct_parents_with_unfilled_modes']:.1f}%), {tail_w16['total_unfilled_modes']}/{tail_w16['total_supported_modes']} modes ({tail_w16['pct_unfilled_modes']:.1f}%)")
    print(f"  W=48: {tail_w48['parents_with_unfilled_modes']}/{tail_w48['total_parents_with_joints']} parents ({tail_w48['pct_parents_with_unfilled_modes']:.1f}%), {tail_w48['total_unfilled_modes']}/{tail_w48['total_supported_modes']} modes ({tail_w48['pct_unfilled_modes']:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
