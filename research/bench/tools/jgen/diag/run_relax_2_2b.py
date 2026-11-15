#!/usr/bin/env python3
"""Step 2.2b Relaxed-Energy Check (Directive Checkpoint 2.2 audit).

GFN2 relaxations of candidate and control top-1 conformers against the relaxed reference
for the 4 biggest winners and 4 worst parents on cal (seeds 11-13).
"""
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from tools.jgen import inputs

REPO = Path(__file__).resolve().parents[5]
XTB_BIN = "/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.pixi/envs/linux/bin/xtb"
CAL_DIR = REPO / "research/data/jgen/campaign/B/confirmation/cal"
OUT_BASE = REPO / "research/data/jgen/campaign/B/relax_2_2b"

WINNERS = ["JEGPUJ", "NUWFAM", "USAPOU", "WEYJIT"]
WORST = ["QEZVEY", "XACMOE", "BOHWUT", "AWONOR"]
PARENTS = WINNERS + WORST
SEEDS = [11, 12, 13]
HARTREE_TO_KJ = 2625.499638


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


def run_xtb_opt(work_dir: Path, xyz_in: Path, chrg: int, uhf: int) -> tuple[float, float]:
    """Run xtb --opt crude in work_dir. Returns (energy_hartree, rmsd_A)."""
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
    if res.returncode != 0:
        raise RuntimeError(f"xtb failed in {work_dir}: {res.stderr[-300:]}")

    # Parse optimized energy
    e_hartree = None
    xtbopt = work_dir / "xtbopt.xyz"
    if xtbopt.exists():
        # xtbopt comment line usually contains energy: -123.4567
        lines = xtbopt.read_text().splitlines()
        if len(lines) >= 2:
            comment = lines[1]
            for part in comment.split():
                try:
                    e_hartree = float(part)
                    break
                except ValueError:
                    pass

    if e_hartree is None:
        for line in reversed(res.stdout.splitlines()):
            if "TOTAL ENERGY" in line:
                parts = line.split()
                e_hartree = float(parts[3])
                break

    if e_hartree is None:
        raise RuntimeError(f"could not parse energy from xtb in {work_dir}")

    # Compute RMSD
    _, c_in = parse_xyz(target_xyz)
    _, c_opt = parse_xyz(xtbopt)
    rmsd = compute_rmsd(c_in, c_opt)
    return e_hartree, rmsd


def main():
    shards = inputs.frame_shards()
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print("=== Step 2.2b: Relaxed-Energy Check (GFN2 xtb --opt crude) ===")
    print(f"Parents ({len(PARENTS)}): Winners: {WINNERS} | Worst: {WORST}")
    print(f"Seeds: {SEEDS}")

    # 1. Relax references first
    ref_energies = {}
    for code in PARENTS:
        shard, ds = shards[code]
        mol = inputs.load_mol(code, shard, ds)
        chrg = mol.charge
        uhf = mol.spin_state - 1
        ref_dir = OUT_BASE / code / "ref"
        ref_xyz = ref_dir / "ref_orig.xyz"
        syms = [a.symbol for a in mol.atoms]
        coords = np.array([a.position for a in mol.atoms], float)
        write_xyz(ref_xyz, syms, coords, f"{code} tmQM reference")
        
        t0 = time.perf_counter()
        e_ref, r_ref = run_xtb_opt(ref_dir, ref_xyz, chrg, uhf)
        dt = time.perf_counter() - t0
        ref_energies[code] = (e_ref, chrg, uhf)
        print(f"[{code}] Relaxed reference: {e_ref:.6f} Eh (rmsd: {r_ref:.3f} A, {dt:.2f}s)")

    # 2. Build task list for top1 conformers
    tasks = []
    for code in PARENTS:
        chrg = ref_energies[code][1]
        uhf = ref_energies[code][2]
        for s in SEEDS:
            # Control top1
            ctrl_xyz = CAL_DIR / "xyz" / code / f"{code}_b1_confirm_control_s{s}_top1.xyz"
            ctrl_rec = json.loads((CAL_DIR / "b1_confirm_control/records" / f"{code}_{s}.json").read_text())
            ctrl_dir = OUT_BASE / code / f"ctrl_s{s}"
            tasks.append((code, s, "ctrl", ctrl_dir, ctrl_xyz, chrg, uhf, ctrl_rec["top1_dE"]))

            # Candidate top1
            cand_xyz = CAL_DIR / "xyz" / code / f"{code}_b1_confirm_bend_release_0.1_100_s{s}_top1.xyz"
            cand_rec = json.loads((CAL_DIR / "b1_confirm_bend_release_0.1_100/records" / f"{code}_{s}.json").read_text())
            cand_dir = OUT_BASE / code / f"cand_s{s}"
            tasks.append((code, s, "cand", cand_dir, cand_xyz, chrg, uhf, cand_rec["top1_dE"]))

    print(f"\nRunning {len(tasks)} top-1 relaxations with max_workers=4 (OMP_NUM_THREADS=1)...")
    t_start = time.perf_counter()

    results_map = {}
    def worker(task):
        code, s, arm, work_dir, xyz_path, chrg, uhf, unrel_dE = task
        e_h, rmsd = run_xtb_opt(work_dir, xyz_path, chrg, uhf)
        return (code, s, arm), (e_h, rmsd, unrel_dE)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            key, val = f.result()
            results_map[key] = val

    total_time = time.perf_counter() - t_start
    print(f"All relaxations completed in {total_time:.2f}s ({total_time/len(tasks):.2f}s per relaxation)")

    # 3. Analyze results
    report_rows = []
    parent_summaries = {}

    for code in PARENTS:
        e_ref = ref_energies[code][0]
        parent_summaries[code] = {
            "unrel_diffs": [],
            "rel_diffs": [],
            "ctrl_rel_dEs": [],
            "cand_rel_dEs": [],
            "ctrl_rmsds": [],
            "cand_rmsds": []
        }
        for s in SEEDS:
            e_ctrl, r_ctrl, unrel_ctrl = results_map[(code, s, "ctrl")]
            e_cand, r_cand, unrel_cand = results_map[(code, s, "cand")]

            # Relaxed dE vs relaxed reference in kJ/mol
            rel_ctrl_dE = (e_ctrl - e_ref) * HARTREE_TO_KJ
            rel_cand_dE = (e_cand - e_ref) * HARTREE_TO_KJ

            diff_unrel = unrel_cand - unrel_ctrl
            diff_rel = rel_cand_dE - rel_ctrl_dE

            report_rows.append({
                "code": code,
                "seed": s,
                "group": "winner" if code in WINNERS else "worst",
                "unrel_ctrl_dE": unrel_ctrl,
                "unrel_cand_dE": unrel_cand,
                "diff_unrel": diff_unrel,
                "rel_ctrl_dE": rel_ctrl_dE,
                "rel_cand_dE": rel_cand_dE,
                "diff_rel": diff_rel,
                "rmsd_ctrl": r_ctrl,
                "rmsd_cand": r_cand,
            })

            parent_summaries[code]["unrel_diffs"].append(diff_unrel)
            parent_summaries[code]["rel_diffs"].append(diff_rel)
            parent_summaries[code]["ctrl_rel_dEs"].append(rel_ctrl_dE)
            parent_summaries[code]["cand_rel_dEs"].append(rel_cand_dE)
            parent_summaries[code]["ctrl_rmsds"].append(r_ctrl)
            parent_summaries[code]["cand_rmsds"].append(r_cand)

    # Save JSON
    out_json = OUT_BASE / "results_2_2b.json"
    out_json.write_text(json.dumps({
        "rows": report_rows,
        "parent_summaries": {
            c: {k: [float(v) for v in vs] for k, vs in parent_summaries[c].items()}
            for c in PARENTS
        }
    }, indent=2))

    # Print Report
    print("\n" + "="*95)
    print("STEP 2.2b RELAXED-ENERGY AUDIT TABLE (Candidate vs Control Top-1, GFN2 crude)")
    print("="*95)
    print(f"{'Parent':8s} {'Seed':4s} {'Unrel Ctrl':10s} {'Unrel Cand':10s} {'Unrel Diff':10s} | {'Rel Ctrl':10s} {'Rel Cand':10s} {'Rel Diff':10s} | {'RMSD ctrl':9s} {'RMSD cand':9s}")
    print("-"*95)
    for r in report_rows:
        print(f"{r['code']:8s} {r['seed']:<4d} {r['unrel_ctrl_dE']:10.1f} {r['unrel_cand_dE']:10.1f} {r['diff_unrel']:+10.1f} | {r['rel_ctrl_dE']:10.1f} {r['rel_cand_dE']:10.1f} {r['diff_rel']:+10.1f} | {r['rmsd_ctrl']:9.3f} {r['rmsd_cand']:9.3f}")

    print("\n" + "="*95)
    print("PARENT-LEVEL SUMMARY & PROMOTION CRITERION EVALUATION")
    print("="*95)
    print(f"{'Parent':8s} {'Group':8s} {'Mean Unrel Diff':16s} {'Mean Rel Diff':16s} {'Gain Retained %':16s} {'Mean RMSD (ctrl/cand)':22s} {'Criterion':10s}")
    print("-"*95)

    winner_pass = True
    worst_pass = True

    for c in PARENTS:
        grp = "WINNER" if c in WINNERS else "WORST"
        m_unrel = np.mean(parent_summaries[c]["unrel_diffs"])
        m_rel = np.mean(parent_summaries[c]["rel_diffs"])
        m_r_ctrl = np.mean(parent_summaries[c]["ctrl_rmsds"])
        m_r_cand = np.mean(parent_summaries[c]["cand_rmsds"])

        if c in WINNERS:
            # Criterion: relaxed difference <= -0.5 * unrelaxed difference
            # (i.e. retains >= 50% of the energy reduction)
            retained_pct = (m_rel / m_unrel) * 100 if abs(m_unrel) > 1e-3 else 0.0
            crit = "PASS" if m_rel <= 0.5 * m_unrel else "FAIL"
            if crit == "FAIL":
                winner_pass = False
        else:
            # Worst 4: does not get materially worse relaxed (e.g. rel diff <= unrel diff + 15 kJ/mol or rel diff <= 10)
            retained_pct = (m_rel / m_unrel) * 100 if abs(m_unrel) > 1e-3 else 0.0
            crit = "PASS" if m_rel <= max(m_unrel + 15.0, 10.0) else "WARN"
            if crit != "PASS":
                worst_pass = False

        print(f"{c:8s} {grp:8s} {m_unrel:+16.2f} {m_rel:+16.2f} {retained_pct:15.1f}% {m_r_ctrl:6.3f} / {m_r_cand:<6.3f} A   {crit:10s}")

    print("-"*95)
    print(f"Winners Verdict: {'PASS (retains >= 50% gain)' if winner_pass else 'FAIL'}")
    print(f"Worst Verdict:   {'PASS (no material degradation)' if worst_pass else 'CHECK'}")


if __name__ == "__main__":
    main()
