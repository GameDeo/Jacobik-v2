"""Freeze geometry-only Step 4 motions from calibration centers, before labels."""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

import numpy as np
import torch

from research.archive.code.interactions.nci_v2.nci_v2 import enumerate_opportunities
from research.archive.code.interactions.nci_v2.nci_v2_assignment import native_build
from research.archive.code.interactions.nci_v2.nci_v2_controls import health_metrics
from research.archive.code.interactions.nci_v2.nci_v2_extract import unique_opportunities
from research.archive.code.interactions.nci_v2.nci_v2_native_motions import native_dofs
from research.archive.code.interactions.nci_v2.nci_v2_response import AMPLITUDES, generate_motions
from research.archive.code.interactions.nci_v2.nci_v2_s2b_parity import safe_load_mol, safe_smiles_index
from research.archive.code.interactions.nci_v2.nci_v2_poolio import file_sha256


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _seam_diagnostic(X, pairs, d0):
    if len(pairs) == 0:
        return np.zeros(len(X), dtype=float)
    pairs = np.asarray(pairs, dtype=np.int64)
    d0 = np.asarray(d0, dtype=float)
    r = np.linalg.norm(X[:, pairs[:, 0]] - X[:, pairs[:, 1]], axis=-1) - d0[None, :]
    return np.square(r).sum(axis=1)


def _beta_state(center_npz, definitions):
    theta = np.asarray(center_npz["theta"], dtype=float) if "theta" in center_npz else None
    offsets = np.asarray(center_npz["thetas_lig_offsets"], dtype=int) if "thetas_lig_offsets" in center_npz else None
    states = {}
    for d in definitions:
        if d["kind"] != "bend":
            continue
        lig, col = int(d["definition"]["ligand"]), int(d["definition"]["theta_column"])
        value = float(theta[offsets[lig] + col]) if theta is not None and offsets is not None and offsets[lig] + col < len(theta) else 0.0
        states[f"{lig}:{col}"] = value
    return states


def select(frame_path: Path, spec_path: Path, clean_root: Path, centers_dir: Path,
           out_path: Path, paths_dir: Path, policy_path: Path):
    torch.set_num_threads(1)
    frame = json.loads(frame_path.read_text())
    spec = json.loads(spec_path.read_text())
    policy = json.loads(policy_path.read_text())
    parents = frame["main_allocation"]["splits"]["calibration"]
    smiles = safe_smiles_index(clean_root / "tmQM" / "tmQM_y.csv")
    output = []
    counters = Counter()
    rejected = Counter()
    omitted = Counter()
    seam_flagged = []
    counts_by_family = defaultdict(Counter)
    counts_by_kind = defaultdict(Counter)

    for parent in parents:
        code = parent["representative_code"]
        ref = safe_load_mol(code, parent["representative_shard"], parent["dataset"], clean_root, smiles)
        if ref is None:
            output.append(dict(code=code, status="reference_missing"))
            rejected["reference_missing"] += 1
            continue
        native = native_build(ref, 11)
        opps = unique_opportunities(enumerate_opportunities(ref, spec))
        Xref = np.asarray([a.position for a in ref.atoms], dtype=np.float32)
        center_inputs = [("reference", f"{code}_reference.npz")]
        for seed in (11, 12, 13):
            center_inputs.append(("baseline", f"{code}_{seed}_baseline.npz"))
        baseline_seam = None
        for label, filename in center_inputs:
            path = centers_dir / filename
            if not path.exists():
                if label == "baseline":
                    output.append(dict(code=code, center=label, pool=filename, status="no_health_passing_pose"))
                    counters["baseline_pool_without_health_passing_center"] += 1
                else:
                    output.append(dict(code=code, center=label, status="missing_reference_center"))
                continue
            z = np.load(path, allow_pickle=False)
            X = np.asarray(z["X"], dtype=np.float32)
            if label == "baseline" and baseline_seam is None:
                baseline_seam = (np.asarray(z["seam_pairs"], dtype=np.int64),
                                 np.asarray(z["seam_d0"], dtype=float))
            if label == "reference":
                if baseline_seam is None:
                    # Load the first available seed's solver closure pair table.
                    for seed in (11, 12, 13):
                        cp = centers_dir / f"{code}_{seed}_baseline.npz"
                        if cp.exists():
                            with np.load(cp, allow_pickle=False) as cz:
                                baseline_seam = (np.asarray(cz["seam_pairs"], dtype=np.int64),
                                                 np.linalg.norm(Xref[cz["seam_pairs"][:, 0]] - Xref[cz["seam_pairs"][:, 1]], axis=1))
                            break
                pairs = baseline_seam[0] if baseline_seam is not None else np.empty((0, 2), dtype=np.int64)
                d0 = np.linalg.norm(Xref[pairs[:, 0]] - Xref[pairs[:, 1]], axis=1) if len(pairs) else np.empty(0)
            else:
                pairs = np.asarray(z["seam_pairs"], dtype=np.int64)
                d0 = np.asarray(z["seam_d0"], dtype=float)

            dofs, omitted_ligands = native_dofs(ref, native, X)
            for item in omitted_ligands:
                omitted[f"macro_joints_{item['macro_joints']}"] += item["macro_joints"]

            def validity(Y):
                h, _, _, _ = health_metrics(ref, Xref, Y.detach().cpu().numpy(), policy)
                return {"valid": h["healthy"], "bad_bond": h["bad_bond"],
                        "overlap": h["gross_overlap"], "coord_intact": h["coord_intact"]}

            records = generate_motions(X, opps, group_id=parent["group_id"], complex_id=code,
                                       dofs=dofs, center=label, validity=validity,
                                       schema=spec, wider_multiplier=4.0, max_local=3)
            beta = _beta_state(z, dofs) if label == "baseline" else _beta_state({}, dofs)
            path_rows = []
            for rec in records:
                if "positions" in rec:
                    pos = np.asarray(rec["positions"], dtype=np.float32)
                    seam = _seam_diagnostic(pos, pairs, d0)
                    delta = np.abs(seam - seam[2])
                    rec["seam_ss_A2"] = seam.tolist()
                    rec["seam_ss_max_delta_A2"] = float(delta.max(initial=0.))
                    rec["seam_ss_change_flag"] = bool(np.any(delta > 1e-3))
                    if rec["seam_ss_change_flag"]:
                        seam_flagged.append(dict(code=code, center=label, motion_id=rec["motion_id"],
                                                  max_delta_A2=rec["seam_ss_max_delta_A2"]))
                    bpath = {k: [v] * len(rec["t"]) for k, v in beta.items()}
                    if rec.get("definition", {}).get("kind") == "bend":
                        d = rec["definition"]
                        key = f"{int(d['ligand'])}:{int(d['theta_column'])}"
                        b0 = beta.get(key, 0.)
                        bpath[key] = [b0 + float(t) * AMPLITUDES["bend"] for t in rec["t"]]
                    rec["bend_beta_path_rad"] = bpath
                    rec["seam_pairs_source"] = "S2b FK closure_pairs mapped to reference atom order" if len(pairs) else "no FK closure_pairs"
                    path_rows.append(rec)
                    if rec.get("selected"):
                        counters["selected_paths"] += 1
                        counts_by_kind[label][rec.get("kind", "unsupported") ] += 1
                        if rec.get("family"):
                            counts_by_family[rec["family"]][label] += 1
                if not rec.get("valid", False):
                    for reason in rec.get("rejection_reasons", []):
                        rejected[reason] += 1
            center_report = dict(code=code, group_id=parent["group_id"], center=label,
                                 pool=filename if label == "baseline" else None,
                                 pool_sha256=str(z["pool_sha256"]) if "pool_sha256" in z else None,
                                 geometry_sha256=str(z["geometry_sha256"]) if "geometry_sha256" in z else None,
                                 candidate_dofs=len(dofs), selected=sum(bool(r.get("selected")) for r in records),
                                 rejected=sum(not bool(r.get("valid")) for r in records),
                                 macro_joints_omitted=sum(x["macro_joints"] for x in omitted_ligands),
                                 fk_closure_pairs=len(pairs), seam_flagged=sum(bool(r.get("seam_ss_change_flag")) for r in path_rows),
                                 health_center="passed" if label == "reference" else "solver-artifact-controls.2 hash-selected pose")
            output.append(center_report)
            counts_by_kind[label]["centers"] += 1
            paths_dir.mkdir(parents=True, exist_ok=True)
            (paths_dir / f"{code}_{label}_{path.stem}.json").write_text(json.dumps(_jsonable({"center": center_report, "motions": path_rows}), separators=(",", ":")) + "\n")
            counters["candidate_paths"] += len(records)
            counters["centers"] += 1

    report = dict(schema="nci_v2_step4_motion_selection_v1", split="calibration",
                  groups=len(parents), centers=counts_by_kind, motion_counts_by_family=counts_by_family,
                  motion_counts_by_type=counts_by_kind, rejected=dict(rejected), omitted_macrocycle_joints=dict(omitted),
                  seam_ss_change_threshold_A2=1e-3, seam_ss_flagged_paths=seam_flagged,
                  counts=dict(counters), center_records=output,
                  selection_policy=dict(local_max=3, t=[-1, -0.5, 0, 0.5, 1], amplitude_rad_or_A=AMPLITUDES,
                                        wider_multiplier=4., ranking="geometry feature displacement only",
                                        baseline_center="SHA256 little-endian float32 FK geometry among health-passing poses"),
                  input_policy="calibration split only; safe CSV reader accesses only CSD_code and SMILES; no QM properties or energies")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--frame", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--clean-root", type=Path, required=True)
    p.add_argument("--centers", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--paths", type=Path, required=True)
    a = p.parse_args()
    result = select(a.frame, a.spec, a.clean_root, a.centers, a.out, a.paths, a.policy)
    print(json.dumps({"centers": result["counts"].get("centers"), "paths": result["counts"].get("candidate_paths")}, sort_keys=True))
