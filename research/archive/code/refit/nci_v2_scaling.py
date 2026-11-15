"""Jacobik NCI v2 Pilot Scaling & Joint Support Feasibility Evaluation.

Step 1 implementation per Jacobik Roadmap v2 Sections 3.2, 3.9, 5.1, and 7.1.
Evaluates:
1. Admission occurrence scaling O_{p, f} across the 172 development/fit complexes
   (12 pilot parents + 160 surface_fit group representatives, strictly 0 validation leakage).
2. Per-admission-class distribution statistics (min, median, p75, p90, p95, max, mean)
   for ALL admission classes specified in spec.json, including BOTH activated_ch_acceptor
   and sp2_ch_acceptor.
3. Verification against Step 1 scaling thresholds: median <= 100, p95 <= 300 per class.
4. Continuous cubic B-spline basis joint support feasibility across angular intervals 3 and 2.
5. Embeds full cryptographic provenance (spec, frame, code commit/diff, constraints DB, MACE model).
6. Writes data/refit/nci_v2/pilot_v1/reports/pilot_scaling_and_support.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "research" / "refit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from jacobik.stats.mapping import assemble
from research.archive.code.interactions.nci_v2.nci_v2 import (
    bspline_basis_1d,
    enumerate_opportunities,
    features_torch,
    make_clamped_knots,
    DEFAULT_RADIAL_WINDOWS,
)
from jacobik.fit.parser import (
    TMQMParser,
    _assemble_raw,
    _attach_qm_properties,
    _iter_bo_blocks,
    _iter_xyz_blocks,
)
from jacobik import profile as _P
from nci_v2_provenance import code_state_sha256

DEFAULT_SPEC = REPO_ROOT / "data/refit/nci_v2/pilot_v1/spec.json"
DEFAULT_FRAME = REPO_ROOT / "data/refit/nci_v2/pilot_v1/frame.json"
DEFAULT_CLEAN = REPO_ROOT / "data/refit/clean"
DEFAULT_REPORT = REPO_ROOT / "data/refit/nci_v2/pilot_v1/reports/pilot_scaling_and_support.json"
CONSTRAINTS_PATH = REPO_ROOT / "data/refit/final/constraints_2026-10-18.json"
MACE_MODEL_PATH = REPO_ROOT / "research/bench/tools/jdev/MACE-POLAR-1-S.model"
# Keys under spec admission_rules.<family> that are settings, not admission classes
NON_CLASS_RULE_KEYS = frozenset({"explicit_exclusions", "eligible_rings", "ring_elements"})


def sha256_of_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_provenance(spec_path: Path = DEFAULT_SPEC, frame_path: Path = DEFAULT_FRAME, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    spec_p = Path(spec_path)
    frame_p = Path(frame_path)
    constraints_p = repo_root / "data/refit/final/constraints_2026-10-18.json"
    mace_p = repo_root / "research/bench/tools/jdev/MACE-POLAR-1-S.model"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    dirty_sha = code_state_sha256(repo_root)

    return {
        "spec_sha256": sha256_of_file(spec_p),
        "frame_sha256": sha256_of_file(frame_p),
        "code_git_commit": commit,
        "code_dirty_diff_sha256": dirty_sha,
        "constraints_db_sha256": sha256_of_file(constraints_p),
        "mace_model_sha256": sha256_of_file(mace_p),
        "full_resolved_profile": _P.snapshot(),
    }


def run_scaling(
    spec_path: Path = DEFAULT_SPEC,
    frame_path: Path = DEFAULT_FRAME,
    clean_dir: Path = DEFAULT_CLEAN,
    report_path: Path = DEFAULT_REPORT,
    surface_fit_only: Optional[bool] = None,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    spec_path = Path(spec_path)
    frame_path = Path(frame_path)
    clean_dir = Path(clean_dir)
    report_path = Path(report_path)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    frame = json.loads(frame_path.read_text(encoding="utf-8"))

    admission_rules = spec.get("admission_rules", {})

    # Collect all admission class names declaratively from spec (skipping the shared ring/exclusion settings)
    all_admission_classes: List[str] = []
    for fam in ("bond_atom", "bond_ring", "ring_ring"):
        for c_name in admission_rules.get(fam, {}):
            if c_name not in NON_CLASS_RULE_KEYS and c_name not in all_admission_classes:
                all_admission_classes.append(c_name)
    if "sp2_5_6_ring_pair" in all_admission_classes and "ring_ring" not in all_admission_classes:
        all_admission_classes.append("ring_ring")

    # 1. Identify Development / Fit complexes ONLY
    pilot_parents = frame.get("pilot_cohort", {}).get("parents", [])
    surface_fit_groups = frame.get("main_allocation", {}).get("splits", {}).get("surface_fit", [])

    if surface_fit_only is None:
        surface_fit_only = (len(surface_fit_groups) == 1200)

    fit_complexes: List[Dict[str, Any]] = []
    if not surface_fit_only:
        for p in pilot_parents:
            fit_complexes.append({
                "code": p["code"],
                "group_id": p.get("group_id", p["code"]),
                "shard": p["shard"],
                "dataset": p["dataset"],
                "source": "pilot_cohort",
                "metal_row": p.get("metal_row", "unknown"),
                "charge_class": p.get("charge_class", "unknown"),
            })

    for g in surface_fit_groups:
        fit_complexes.append({
            "code": g["representative_code"],
            "group_id": g["group_id"],
            "shard": g["representative_shard"],
            "dataset": g["dataset"],
            "source": "surface_fit",
            "metal_row": g.get("metal_row", "unknown"),
            "charge_class": g.get("charge_class", "unknown"),
        })

    print(
        f"Evaluating {len(fit_complexes)} Fit/Dev complexes "
        f"({0 if surface_fit_only else len(pilot_parents)} pilot + {len(surface_fit_groups)} surface_fit)...",
        flush=True,
    )

    parsers = {
        "tmQM": TMQMParser(clean_dir / "tmQM"),
        "lnQM": TMQMParser(clean_dir / "lnQM"),
    }
    by_shard: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in fit_complexes:
        data_kind = "lnQM" if c["dataset"] == "LNf" else "tmQM"
        by_shard[(data_kind, c["shard"])].append(c)

    occurrence_records: List[Dict[str, Any]] = []
    feature_records_by_family: Dict[str, List[Dict[str, Any]]] = {
        "bond_atom": [],
        "bond_ring": [],
        "ring_ring": [],
    }

    t0 = time.perf_counter()
    for (data_kind, shard), c_list in by_shard.items():
        parent_dir = clean_dir / data_kind
        parser = parsers[data_kind]
        target_map = {c["code"]: c for c in c_list}

        xp = parent_dir / "xyz" / f"tmQM_{shard}.xyz"
        bp = parent_dir / "bo" / f"tmQM_{shard}.BO"

        for (comment, atoms), (code, bonds) in zip(_iter_xyz_blocks(xp), _iter_bo_blocks(bp)):
            if code in target_map:
                c_info = target_map[code]
                csv_row = parser.csv_index.get(code)
                raw = _assemble_raw(comment, atoms, code, bonds, csv_row)
                mol = assemble(raw)
                if csv_row:
                    _attach_qm_properties(mol, csv_row)

                n_heavy = sum(1 for a in mol.atoms if a.symbol != "H")
                batch = enumerate_opportunities(mol, schema={"admission_rules": admission_rules})

                class_counts: Dict[str, int] = defaultdict(int)
                for fam, o_list in batch.opportunities.items():
                    for o in o_list:
                        ac = o.get("admission_class", fam)
                        class_counts[ac] += 1
                if "sp2_5_6_ring_pair" in class_counts:
                    class_counts["ring_ring"] = class_counts["sp2_5_6_ring_pair"]

                rec = {
                    "code": code,
                    "group_id": c_info["group_id"],
                    "source": c_info["source"],
                    "metal": mol.metal_symbol,
                    "metal_row": c_info["metal_row"],
                    "charge": mol.charge,
                    "n_atoms": len(mol.atoms),
                    "n_heavy": n_heavy,
                    "total_opportunities": batch.total_count,
                    "class_counts": dict(class_counts),
                }
                occurrence_records.append(rec)

                if batch.total_count > 0:
                    coords = np.array([a.position for a in mol.atoms], dtype=np.float64)
                    X_torch = torch.from_numpy(coords).unsqueeze(0)
                    fb = features_torch(X_torch, batch, schema=spec)

                    for fam in ("bond_atom", "bond_ring", "ring_ring"):
                        feats = fb.features.get(fam)
                        if feats is not None and feats.shape[1] > 0:
                            f_np = feats[0].detach().cpu().numpy()
                            cands = batch.opportunities.get(fam, [])
                            for i_cand, cand in enumerate(cands):
                                feature_records_by_family[fam].append({
                                    "code": code,
                                    "group_id": c_info["group_id"],
                                    "family": fam,
                                    "admission_class": cand.get("admission_class", fam),
                                    "features": f_np[i_cand].tolist(),
                                })

    print(
        f"Parsed and enumerated {len(occurrence_records)} complexes in {time.perf_counter() - t0:.2f}s.",
        flush=True,
    )

    # 2. Admission Scaling Distribution Statistics (Per Admission Class)
    scaling_stats: Dict[str, Any] = {}
    thresholds = spec.get("pilot_thresholds", {}).get("occurrence_scaling", {
        "max_median_opf_per_class": 100,
        "max_p95_opf_per_class": 300,
    })
    max_median_allowed = float(thresholds["max_median_opf_per_class"])
    max_p95_allowed = float(thresholds["max_p95_opf_per_class"])

    all_comply = True

    for ac in all_admission_classes:
        counts_fit = [r["class_counts"].get(ac, 0) for r in occurrence_records]
        counts_pilot = [r["class_counts"].get(ac, 0) for r in occurrence_records if r["source"] == "pilot_cohort"]

        arr = np.array(counts_fit, dtype=np.float64)
        med = float(np.median(arr)) if len(arr) > 0 else 0.0
        p75 = float(np.percentile(arr, 75)) if len(arr) > 0 else 0.0
        p90 = float(np.percentile(arr, 90)) if len(arr) > 0 else 0.0
        p95 = float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0
        max_val = int(np.max(arr)) if len(arr) > 0 else 0
        min_val = int(np.min(arr)) if len(arr) > 0 else 0
        mean_val = float(np.mean(arr)) if len(arr) > 0 else 0.0
        nonzero_count = int(np.sum(arr > 0))

        complies_median = med <= max_median_allowed
        complies_p95 = p95 <= max_p95_allowed
        class_complies = complies_median and complies_p95
        if not class_complies:
            all_comply = False

        scaling_stats[ac] = {
            "admission_class": ac,
            "sample_size": len(counts_fit),
            "nonzero_complexes_count": nonzero_count,
            "nonzero_fraction": round(nonzero_count / max(len(counts_fit), 1), 4),
            "distribution": {
                "min": min_val,
                "median": med,
                "p75": p75,
                "p90": p90,
                "p95": p95,
                "max": max_val,
                "mean": round(mean_val, 2),
            },
            "pilot_cohort_counts": counts_pilot,
            "threshold_compliance": {
                "median_threshold": max_median_allowed,
                "p95_threshold": max_p95_allowed,
                "median_passes": complies_median,
                "p95_passes": complies_p95,
                "status": "PASS" if class_complies else "FAIL",
            },
        }

    # 3. Continuous B-Spline Basis Joint Support Evaluation
    support_evaluation: Dict[str, Any] = {}

    for fam in ("bond_atom", "bond_ring", "ring_ring"):
        rows = feature_records_by_family[fam]
        if not rows:
            support_evaluation[fam] = {"status": "no_observations"}
            continue

        X_all = np.array([r["features"] for r in rows], dtype=np.float64)
        groups = [r["group_id"] for r in rows]
        unique_groups = sorted(list(set(groups)))
        D = X_all.shape[1]

        r_win = DEFAULT_RADIAL_WINDOWS.get(fam, (1.1, 1.35, 6.0, 8.0))
        r_domain = (r_win[0], r_win[3])

        results_by_resolution = {}
        for ang_intervals in (3, 2):
            knots_r = make_clamped_knots(r_domain, 6, degree=3)
            knots_x_list = []
            for k in range(D - 1):
                x_min = 0.0 if (fam in ("bond_ring", "ring_ring") and k >= 1) else -1.0
                x_max = 2.0 if (fam == "ring_ring" and k == 1) else 1.0
                knots_x_list.append(make_clamped_knots((x_min, x_max), ang_intervals, degree=3))

            X_torch = torch.from_numpy(X_all)
            b_r = bspline_basis_1d(X_torch[:, 0], knots_r).numpy()
            b_x = [bspline_basis_1d(X_torch[:, k + 1], knots_x_list[k]).numpy() for k in range(D - 1)]

            Kr = b_r.shape[1]
            Kx = [b.shape[1] for b in b_x]
            grid_shape = tuple([Kr] + Kx)
            total_basis_functions = int(np.prod(grid_shape))

            v_ref = np.zeros((len(unique_groups),) + grid_shape, dtype=np.float64)
            for g_idx, g in enumerate(unique_groups):
                g_mask = np.array([grp == g for grp in groups])
                br_g = b_r[g_mask]
                if D == 2:
                    prod = np.einsum("i j, i k -> j k", br_g, b_x[0][g_mask])
                elif D == 4:
                    prod = np.einsum(
                        "i j, i k, i l, i m -> j k l m",
                        br_g,
                        b_x[0][g_mask],
                        b_x[1][g_mask],
                        b_x[2][g_mask],
                    )
                else:
                    prod = br_g.sum(axis=0)
                v_ref[g_idx] = prod

            w_bar_grp = len(rows) / max(len(unique_groups), 1)
            tau = 1e-3 * w_bar_grp

            N_ref = (v_ref > tau).sum(axis=0)
            sum_v_ref = v_ref.sum(axis=0)
            sum_v2_ref = (v_ref**2).sum(axis=0)
            ESS_ref = np.where(sum_v2_ref > 1e-12, (sum_v_ref**2) / (sum_v2_ref + 1e-12), 0.0)

            supported_mask = (N_ref >= 10) & (ESS_ref >= 10.0)
            active_basis_functions = int(np.sum(sum_v_ref > 0))
            supported_basis_functions = int(np.sum(supported_mask))

            if D == 2:
                obs_support = np.einsum("i j, i k, j k -> i", b_r, b_x[0], supported_mask.astype(np.float64))
            elif D == 4:
                obs_support = np.einsum(
                    "i j, i k, i l, i m, j k l m -> i",
                    b_r,
                    b_x[0],
                    b_x[1],
                    b_x[2],
                    supported_mask.astype(np.float64),
                )
            else:
                obs_support = np.einsum("i j, j -> i", b_r, supported_mask.astype(np.float64))

            mean_obs_support = float(np.mean(obs_support))
            unsupported_obs_frac = float(np.mean(obs_support < 0.1))

            results_by_resolution[f"angular_{ang_intervals}"] = {
                "angular_intervals": ang_intervals,
                "grid_shape": list(grid_shape),
                "total_basis_functions": total_basis_functions,
                "active_basis_functions": active_basis_functions,
                "supported_basis_functions_min10_groups": supported_basis_functions,
                "mean_observation_supported_mass": round(mean_obs_support, 4),
                "unsupported_observation_fraction": round(unsupported_obs_frac, 4),
                "grid_fragmentation_gt_50pct": unsupported_obs_frac > 0.50,
            }

        support_evaluation[fam] = {
            "family": fam,
            "observations_count": len(rows),
            "unique_groups_count": len(unique_groups),
            "resolutions": results_by_resolution,
        }

    coarsen_needed = any(
        res.get("resolutions", {}).get("angular_3", {}).get("grid_fragmentation_gt_50pct", False)
        for res in support_evaluation.values()
    )

    recommended_angular_intervals = 2 if coarsen_needed else 3
    resolution_decision = {
        "recommended_angular_intervals": recommended_angular_intervals,
        "coarsening_required_by_step1_rule": coarsen_needed,
        "data_split_utilized": "surface_fit_and_pilot_only",
        "reference_only_preliminary_feasibility": True,
    }

    # 4. Provenance
    provenance = get_provenance(spec_path=spec_path, frame_path=frame_path, repo_root=REPO_ROOT)

    # 5. Compile Final Certified Report
    is_scaled = "scaled" in spec.get("version", "") or "scaled" in str(report_path)
    report = {
        "report_schema": "nci_v2.0_scaled_scaling_and_support" if is_scaled else "nci_v2.0_pilot_scaling_and_support",
        "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "spec_version": spec.get("version", "2.0.0-pilot.2"),
        "provenance": provenance,
        "evaluated_dataset": {
            "total_complexes_evaluated": len(occurrence_records),
            "pilot_complexes_count": 0 if surface_fit_only else len(pilot_parents),
            "surface_fit_complexes_count": len(surface_fit_groups),
            "calibration_complexes_count": 0,
            "motion_val_complexes_count": 0,
            "validation_leakage_prevented": True,
        },
        "step_1_admission_scaling": {
            "thresholds": {
                "max_median_opf_per_class": max_median_allowed,
                "max_p95_opf_per_class": max_p95_allowed,
            },
            "overall_compliance": "PASS" if all_comply else "FAIL",
            "per_admission_class_statistics": scaling_stats,
        },
        "joint_support_feasibility": {
            "evaluation_methodology": "continuous_cubic_bspline_basis_group_weights_partition_of_unity",
            "families": support_evaluation,
            "resolution_decision": resolution_decision,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp = report_path.with_suffix(report_path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(report_path)

    print(
        f"Wrote certified scaling report -> {report_path} "
        f"({time.perf_counter() - t_start:.2f}s total). "
        f"Overall Compliance: {report['step_1_admission_scaling']['overall_compliance']}",
        flush=True,
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="spec JSON path")
    p.add_argument("--frame", type=Path, default=DEFAULT_FRAME, help="frame JSON path")
    p.add_argument("--clean", type=Path, default=DEFAULT_CLEAN, help="clean shards dir")
    p.add_argument("--out", type=Path, default=DEFAULT_REPORT, help="output report JSON path")
    p.add_argument("--surface-fit-only", action="store_true", default=None, help="Evaluate only surface_fit groups (excludes pilot)")
    p.add_argument("--include-pilot", action="store_true", help="Include pilot parents in fit sample")
    args = p.parse_args()

    surf_fit_only = None
    if args.surface_fit_only:
        surf_fit_only = True
    elif args.include_pilot:
        surf_fit_only = False

    run_scaling(
        spec_path=args.spec,
        frame_path=args.frame,
        clean_dir=args.clean,
        report_path=args.out,
        surface_fit_only=surf_fit_only,
    )


if __name__ == "__main__":
    main()
