"""Extract strict native positives and matched u1+repulsive-Q6 HB opportunities.

Per-parent compressed records are written under /tmp. No production scorer or
parameter database is read for mutation or changed by this research stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from nci_ctx1_cohort import DEFAULT_SCAN, _load_cases
from nci_hb_context import SUPPORT_MAX_HA, features, select_high_confidence

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_COHORT = DEFAULT_SCAN / "cohort"
DEFAULT_MACE = DEFAULT_COHORT / "mace_polar1s_float64_top32"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    p.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--mace", type=Path, default=DEFAULT_MACE)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=("calibration", "validation"),
                   default=("calibration", "validation"))
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    scan, cohort, mace = a.scan.resolve(), a.cohort.resolve(), a.mace.resolve()
    out = (a.out or cohort / "strict_hb_opportunities_v1").resolve()
    selection_path = cohort / "selection.json"
    selection = json.loads(selection_path.read_text())
    selected = {s: selection["selected_rows"][s] for s in a.splits}
    jobs = [(s, row) for s in a.splits for row in selected[s]]
    if a.limit:
        jobs = jobs[:a.limit]
        selected = {s: [row for split, row in jobs if split == s] for s in a.splits}

    cases = _load_cases(scan, selected)
    out.mkdir(parents=True, exist_ok=True)
    all_counts = Counter()
    parent_summaries = []
    native_rows = []
    errors = []
    for n, (split, row) in enumerate(jobs, 1):
        code = row["code"]
        pool_path = cohort / "pools/q6_u1rep" / split / f"{code}_s0.npz"
        mace_path = mace / split / f"{code}_s0.npz"
        target = out / "opportunities" / split / f"{code}_s0.npz"
        try:
            if not pool_path.is_file() or not mace_path.is_file():
                raise FileNotFoundError(f"missing baseline or MACE pool: {pool_path} / {mace_path}")
            native_count = len(row["contacts"])
            for contact in row["contacts"]:
                native_rows.append({"code": code, "split": split, "dataset": row["dataset"],
                                    "donor": contact["donor"], "hydrogen": contact["hydrogen"],
                                    "acceptor": contact["acceptor"], "keys": contact["keys"],
                                    "r_ha": contact["r_ha"],
                                    "cos_dha": float(np.cos(np.deg2rad(contact["dha_deg"]))),
                                    "cos_acceptor": contact["acceptor_cos"],
                                    "acceptor_axis_defined": contact["acceptor_axis_defined"],
                                    "partners": contact["partners"]})
            if target.exists():
                zold = np.load(target, allow_pickle=False)
                parent_summaries.append(json.loads(str(zold["summary"].item())))
                all_counts.update(json.loads(str(zold["counts"].item())))
                continue

            mol = cases[code].tmqm_mol
            from nci_hb_chemistry import from_training
            chemistry = from_training(mol)
            baseline_features = []
            pool = np.load(pool_path, allow_pickle=False)
            mace_z = np.load(mace_path, allow_pickle=False)
            coords = np.asarray(pool["X_raw"], dtype=np.float64)
            indices = np.asarray(mace_z["pool_indices"], dtype=np.int64)
            if not np.array_equal(np.asarray(mace_z["symbols"]), np.asarray(pool["ref_symbols"])):
                raise ValueError("MACE/pool reference atom order mismatch")
            if not np.array_equal(np.sort(np.asarray(pool["symbols"])), np.sort(np.asarray(pool["ref_symbols"]))):
                raise ValueError("solver/reference elemental composition mismatch")
            by_pose = []
            for rank, pool_index in enumerate(indices):
                fts = features(mol, coords[pool_index], chemistry=chemistry)
                strict = {(x.donor, x.hydrogen, x.acceptor)
                          for x in select_high_confidence(fts)}
                pose = []
                for x in fts:
                    if x.r_ha > SUPPORT_MAX_HA:
                        continue
                    pose.append((x, (x.donor, x.hydrogen, x.acceptor) in strict))
                by_pose.append(pose)

            flat = [(rank, x, is_strict) for rank, pose in enumerate(by_pose) for x, is_strict in pose]
            m = len(flat)
            strict_pose_hits = sum(any(is_strict for _, is_strict in pose) for pose in by_pose)
            opportunity_pose_hits = sum(bool(pose) for pose in by_pose)
            key_levels = np.asarray([json.dumps(x.keys, separators=(",", ":")) for _, x, _ in flat], dtype="U512")
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                code=code, split=split, dataset=row["dataset"], seed=0,
                pool_sha256=sha256(pool_path), mace_sha256=sha256(mace_path),
                selection_sha256=sha256(selection_path), chemistry_version="hb_ctx1",
                support_max_ha=SUPPORT_MAX_HA,
                pool_rank=np.asarray([rank for rank, _, _ in flat], dtype=np.int16),
                donor=np.asarray([x.donor for _, x, _ in flat], dtype=np.int32),
                hydrogen=np.asarray([x.hydrogen for _, x, _ in flat], dtype=np.int32),
                acceptor=np.asarray([x.acceptor for _, x, _ in flat], dtype=np.int32),
                key_levels=key_levels,
                r_ha=np.asarray([x.r_ha for _, x, _ in flat], dtype=np.float32),
                cos_dha=np.asarray([x.cos_dha for _, x, _ in flat], dtype=np.float32),
                cos_acceptor=np.asarray([x.cos_acceptor for _, x, _ in flat], dtype=np.float32),
                acceptor_axis_defined=np.asarray([x.acceptor_axis_defined for _, x, _ in flat], dtype=np.bool_),
                strict_selected=np.asarray([flag for _, _, flag in flat], dtype=np.bool_),
                pool_index=indices.astype(np.int16),
                dE_kJ_mol=np.asarray(mace_z["dE_kJ_mol"], dtype=np.float64),
                summary=json.dumps({"code": code, "split": split, "dataset": row["dataset"],
                                    "pool_poses": len(indices), "opportunities": m,
                                    "native_strict_contacts": native_count,
                                    "baseline_strict_contacts": int(sum(x[2] for x in flat)),
                                    "baseline_poses_with_strict_contact": int(strict_pose_hits),
                                    "baseline_poses_with_any_typed_opportunity": int(opportunity_pose_hits)}),
                counts=json.dumps({"parents": 1, "native_strict_contacts": native_count,
                                   "baseline_poses": len(indices), "baseline_candidates_le_3p5": m,
                                   "baseline_strict_selected_contacts": int(sum(x[2] for x in flat)),
                                   "poses_with_strict_contact": int(strict_pose_hits),
                                   "poses_with_any_typed_opportunity": int(opportunity_pose_hits)}),
            )

            counts = {"parents": 1, "native_strict_contacts": native_count,
                      "baseline_poses": len(indices), "baseline_candidates_le_3p5": m,
                      "baseline_strict_selected_contacts": int(sum(x[2] for x in flat)),
                      "poses_with_strict_contact": int(strict_pose_hits),
                      "poses_with_any_typed_opportunity": int(opportunity_pose_hits)}
            all_counts.update(counts)
            summary = {"code": code, "split": split, "dataset": row["dataset"],
                       "pool_poses": len(indices), "opportunities": m,
                       "native_strict_contacts": native_count,
                       "baseline_strict_contacts": counts["baseline_strict_selected_contacts"],
                       "baseline_poses_with_strict_contact": counts["poses_with_strict_contact"],
                       "baseline_poses_with_any_typed_opportunity": counts["poses_with_any_typed_opportunity"]}
            parent_summaries.append(summary)
            print(f"{n}/{len(jobs)} {split} {code}: native={native_count}, baseline opportunities={m}, "
                  f"strict contacts={counts['baseline_strict_selected_contacts']}", flush=True)
        except Exception as exc:
            errors.append({"code": code, "split": split,
                           "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
            print(f"{n}/{len(jobs)} {split} {code}: ERROR {type(exc).__name__}: {str(exc)[:250]}", flush=True)

    (out / "native_positive_contacts.json").write_text(
        json.dumps(native_rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "schema": "nci_ctx1_strict_hb_opportunities_v1",
        "scope": "strict native positives matched to top MACE-measured poses from frozen u1 + repulsive-Q6 solves",
        "support_radius_A": SUPPORT_MAX_HA,
        "selection_sha256": sha256(selection_path),
        "scan_summary_sha256": sha256(scan / "summary.json"),
        "mace_summary_sha256": sha256(mace / "summary.json"),
        "native_positive_count": len(native_rows), "baseline_counts": dict(all_counts),
        "parents": parent_summaries, "errors": errors,
        "weighting_rule_for_fitting": "equal parent mass; split parent mass over its donor-H groups; within each donor-H average over the fixed baseline pose sample and its typed acceptor opportunities",
        "out": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"parents": len(parent_summaries), "native_strict_contacts": len(native_rows),
                      "baseline_counts": dict(all_counts), "errors": errors,
                      "summary": str(out / "summary.json")}, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
