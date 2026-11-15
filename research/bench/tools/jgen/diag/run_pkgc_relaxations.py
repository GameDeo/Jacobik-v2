#!/usr/bin/env python3
"""Overlapped GFN2 relaxation audit for Package C confirmation (directive_jgen_B_session2 §3.5).

Runs concurrent CPU GFN2 relaxations (xtb --opt crude, -P 1, OMP_NUM_THREADS=1, max 3 workers)
on parents whose unrelaxed top-1 changed by > 10 kJ/mol on any seed (or parent mean).
Evaluates MACE float64 single-point energies on all relaxed xtbopt.xyz structures.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[5]
for p in (str(REPO / "src"), str(REPO), str(REPO / "research" / "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)
XTB_BIN = shutil.which("xtb") or "/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.pixi/envs/linux/bin/xtb"
EV_TO_KJ_MOL = 96.48530749925793


def parse_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    n = int(lines[0])
    syms = []
    coords = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        syms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return syms, np.array(coords, float)


def write_xyz(path: Path, syms: list[str], coords: np.ndarray, comment: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{len(syms)}\n{comment}\n")
        for s, (x, y, z) in zip(syms, coords):
            f.write(f"{s:<2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")


def compute_rmsd(c1: np.ndarray, c2: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((c1 - c2) ** 2, axis=-1))))


def run_xtb_opt(work_dir: Path, xyz_in: Path, chrg: int, uhf: int) -> tuple[bool, Optional[Path], str]:
    """Run xtb --opt crude in work_dir. Returns (success, xtbopt_path, error_str)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    target_xyz = work_dir / "input.xyz"
    shutil.copyfile(xyz_in, target_xyz)

    cmd = [
        XTB_BIN,
        str(target_xyz),
        "--opt", "crude",
        "--chrg", str(chrg),
        "--uhf", str(uhf),
        "-P", "1"
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    res = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, env=env)
    xtbopt = work_dir / "xtbopt.xyz"
    if res.returncode != 0 and not xtbopt.exists():
        return False, None, f"xtb failed: {res.stderr[-200:]}"
    if not xtbopt.exists():
        return False, None, "xtbopt.xyz not found"
    return True, xtbopt, ""


def find_parents_to_audit(base_dir: Path, seeds: list[int], threshold: float = 10.0) -> list[str]:
    """Identify parents where any candidate top-1 changed by > threshold kJ/mol vs control."""
    ctrl_dir = base_dir / "control/records"
    comb_dir = base_dir / "combined/records"
    ring_dir = base_dir / "ring_0.5/records"

    if not ctrl_dir.exists():
        return []

    ctrl_files = list(ctrl_dir.glob("*_*.json"))
    parents = sorted(list({f.stem.split("_")[0] for f in ctrl_files}))
    audit_parents = []

    for code in parents:
        max_delta = 0.0
        for s in seeds:
            cp = ctrl_dir / f"{code}_{s}.json"
            if not cp.exists():
                continue
            c_rec = json.loads(cp.read_text())
            c_top = c_rec.get("top1_dE")
            if c_top is None:
                continue

            for arm_dir in [comb_dir, ring_dir]:
                ap = arm_dir / f"{code}_{s}.json"
                if ap.exists():
                    a_rec = json.loads(ap.read_text())
                    a_top = a_rec.get("top1_dE")
                    if a_top is not None:
                        delta = abs(a_top - c_top)
                        if delta > max_delta:
                            max_delta = delta

        if max_delta >= threshold:
            audit_parents.append(code)

    return audit_parents


def run_audit(panel: str, base_dir: Path, seeds: list[int], max_workers: int = 3) -> dict:
    """Run full GFN2 relaxation audit for panel."""
    from tools.jgen import inputs
    from jacobik.core.env import MACE_MODEL_PATH
    from jacobik.core.energy import MACEPolarEvaluator

    shards = inputs.frame_shards()
    parents = find_parents_to_audit(base_dir, seeds, threshold=10.0)
    print(f"[{panel}] Identified {len(parents)} parents for relaxed audit (|Δtop-1| >= 10 kJ/mol): {parents}", flush=True)

    out_dir = base_dir / "relaxations"
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz_dir = base_dir / "xyz"

    # Tasks: (code, seed, arm_label, xyz_path, work_dir, chrg, uhf)
    tasks = []
    parent_mols = {}

    for code in parents:
        shard, ds = shards[code]
        mol = inputs.load_mol(code, shard, ds)
        if mol is None:
            continue
        parent_mols[code] = mol
        chrg = int(mol.charge)
        uhf = int(mol.spin_state) - 1

        # Relax crystal reference (once per parent)
        ref_work = out_dir / code / "ref"
        ref_xyz = ref_work / "ref.xyz"
        ref_sym = [x.symbol for x in mol.atoms]
        ref_pos = np.array([x.position for x in mol.atoms], float)
        write_xyz(ref_xyz, ref_sym, ref_pos, f"{code} crystal reference")
        tasks.append((code, None, "ref", ref_xyz, ref_work, chrg, uhf))

        # Relax top-1 control and top-1 combined (and ring_0.5 if it differed from control)
        for s in seeds:
            # Control top-1
            ctrl_xyz = xyz_dir / code / f"{code}_b1_pkgc_control_s{s}_top1.xyz"
            if ctrl_xyz.exists():
                c_work = out_dir / code / f"s{s}_ctrl"
                tasks.append((code, s, "control", ctrl_xyz, c_work, chrg, uhf))

            # Combined top-1
            comb_xyz = xyz_dir / code / f"{code}_b1_pkgc_combined_s{s}_top1.xyz"
            if comb_xyz.exists():
                comb_work = out_dir / code / f"s{s}_combined"
                tasks.append((code, s, "combined", comb_xyz, comb_work, chrg, uhf))

            # Ring_0.5 top-1 (only if different from control)
            ring_rec_path = base_dir / "ring_0.5/records" / f"{code}_{s}.json"
            ctrl_rec_path = base_dir / "control/records" / f"{code}_{s}.json"
            if ring_rec_path.exists() and ctrl_rec_path.exists():
                r_dE = json.loads(ring_rec_path.read_text()).get("top1_dE")
                c_dE = json.loads(ctrl_rec_path.read_text()).get("top1_dE")
                if r_dE is not None and c_dE is not None and abs(r_dE - c_dE) > 1e-4:
                    ring_xyz = xyz_dir / code / f"{code}_b1_pkgc_ring_0.5_s{s}_top1.xyz"
                    if ring_xyz.exists():
                        r_work = out_dir / code / f"s{s}_ring_0.5"
                        tasks.append((code, s, "ring_0.5", ring_xyz, r_work, chrg, uhf))

    print(f"[{panel}] Launching {len(tasks)} GFN2 relaxations (at most {max_workers} concurrent workers)...", flush=True)

    opt_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_xtb_opt, work_dir, xyz_in, chrg, uhf): (code, s, arm, work_dir)
            for (code, s, arm, xyz_in, work_dir, chrg, uhf) in tasks
        }
        for fut in as_completed(futures):
            code, s, arm, work_dir = futures[fut]
            try:
                ok, xtbopt, err = fut.result()
                opt_results[(code, s, arm)] = (ok, xtbopt, err)
                tag = f"{code}:{s}:{arm}" if s is not None else f"{code}:ref"
                print(f"  [xtb] {tag} {'OK' if ok else 'FAILED'}", flush=True)
            except Exception as exc:
                opt_results[(code, s, arm)] = (False, None, str(exc))

    # Evaluate MACE float64 on all relaxed geometries
    print(f"[{panel}] Evaluating MACE float64 on all relaxed xtbopt structures...", flush=True)
    mace = MACEPolarEvaluator(model="polar-1-s", device="cuda", dtype="float64", model_path=str(MACE_MODEL_PATH),
                              mode="energy-only", threads=1).load()
    mace._batch_atoms = 900

    structures = []
    struct_keys = []
    for (code, s, arm), (ok, xtbopt, err) in opt_results.items():
        if ok and xtbopt and xtbopt.exists():
            mol = parent_mols[code]
            syms, coords = parse_xyz(xtbopt)
            structures.append((syms, coords, int(mol.charge), int(mol.spin_state)))
            struct_keys.append((code, s, arm))

    energies_kj = {}
    if structures:
        e_ev = mace.energies(structures)
        e_kj = e_ev * EV_TO_KJ_MOL
        for k, e in zip(struct_keys, e_kj):
            energies_kj[k] = float(e)

    # Compute relaxed ΔE and regression stats
    audit_records = []
    flagged_parents = []

    for code in parents:
        ref_e = energies_kj.get((code, None, "ref"))
        mol = parent_mols.get(code)
        if ref_e is None or mol is None:
            continue

        p_regressions = []
        for s in seeds:
            ctrl_e = energies_kj.get((code, s, "control"))
            comb_e = energies_kj.get((code, s, "combined"))
            ring_e = energies_kj.get((code, s, "ring_0.5"))

            # Unrelaxed dE from records
            c_rec = json.loads((base_dir / "control/records" / f"{code}_{s}.json").read_text())
            comb_rec = json.loads((base_dir / "combined/records" / f"{code}_{s}.json").read_text())
            unrel_c_dE = c_rec.get("top1_dE")
            unrel_comb_dE = comb_rec.get("top1_dE")

            rel_c_dE = (ctrl_e - ref_e) if ctrl_e is not None else None
            rel_comb_dE = (comb_e - ref_e) if comb_e is not None else None
            diff_rel = (rel_comb_dE - rel_c_dE) if (rel_comb_dE is not None and rel_c_dE is not None) else None

            if diff_rel is not None:
                p_regressions.append(diff_rel)

            row = dict(
                panel=panel,
                code=code,
                seed=s,
                unrel_ctrl_top1_dE=unrel_c_dE,
                unrel_comb_top1_dE=unrel_comb_dE,
                diff_unrel=(unrel_comb_dE - unrel_c_dE) if (unrel_comb_dE is not None and unrel_c_dE is not None) else None,
                rel_ctrl_top1_dE=rel_c_dE,
                rel_comb_top1_dE=rel_comb_dE,
                diff_rel=diff_rel,
            )
            audit_records.append(row)

        # Flag if consistent relaxed regression > 20.0 kJ/mol across all 3 seeds
        if len(p_regressions) == len(seeds) and all(r > 20.0 for r in p_regressions):
            flagged_parents.append(dict(code=code, regressions=p_regressions, mean=float(np.mean(p_regressions))))

    result = dict(
        panel=panel,
        n_audited_parents=len(parents),
        audited_parents=parents,
        flagged_parents=flagged_parents,
        records=audit_records,
    )
    res_path = out_dir / "relaxed_audit_results.json"
    res_path.write_text(json.dumps(result, indent=2))
    print(f"[{panel}] Audit finished! Saved to {res_path}. Flagged parents: {flagged_parents}", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=["tail", "cal"])
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--seeds", default="11,12,13")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    run_audit(args.panel, Path(args.base_dir), seeds, max_workers=args.workers)
