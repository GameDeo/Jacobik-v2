"""GFN2 crude relaxation and MACE strain calculation for G1/G3 screening (Package G).

Evaluates top-1 and floor conformers from all 4 arms (control, g1_lr_final_0.01, g1_lr_final_0.001, g3_bend_free_1.0)
across seeds 11, 12, 13 on the tail panel.
Runs `xtb in.xyz --gfn 2 --opt crude --cycles 200 --chrg Q --uhf M-1 -P 1` with OMP_NUM_THREADS=1,
followed by float64 MACE single-point evaluations on relaxed coordinates.
Calculates strain = E_unrel - E_rel for top-1 and floor.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[4]
HARTREE_KJ = 2625.4996394799
EV_TO_KJ = 96.4853321233
_E_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
_CONV_RE = re.compile(r"GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)\s+ITERATIONS")


def xtb_path() -> str:
    p = os.environ.get("JACOBIK_XTB") or shutil.which("xtb")
    if not p or not os.path.exists(p):
        raise SystemExit("xtb does not resolve (put pixi env on PATH or set JACOBIK_XTB)")
    return os.path.abspath(p)


def xtb_opt(symbols: List[str], X: np.ndarray, charge: int, spin: int, workroot: str, cycles: int = 200) -> dict:
    t0 = time.perf_counter()
    res = dict(status="failed", E_h=None, cycles=None, converged=False, X=None, wall_s=None, stderr_tail="")
    try:
        Path(workroot).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workroot) as td:
            with open(f"{td}/in.xyz", "w") as f:
                f.write(f"{len(symbols)}\n\n" + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(symbols, X)))
            cmd = [xtb_path(), "in.xyz", "--gfn", "2", "--opt", "crude", "--cycles", str(cycles),
                   "--chrg", str(int(charge)), "--uhf", str(int(spin) - 1), "-P", "1"]
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="500M")
            p = subprocess.run(cmd, cwd=td, capture_output=True, text=True, env=env, timeout=1800)
            out = p.stdout
            res["stderr_tail"] = p.stderr[-300:]
            m = _E_RE.findall(out)
            c = _CONV_RE.search(out)
            if c:
                res["cycles"], res["converged"] = int(c.group(1)), True
            elif "FAILED TO CONVERGE" in out:
                res["cycles"] = cycles
            else:
                n = len(re.findall(r"^\s*\d+\s+-?\d+\.\d+\s+-?\d+\.\d+", out, re.M))
                res["cycles"] = n or None
            if m and os.path.exists(f"{td}/xtbopt.xyz"):
                L = open(f"{td}/xtbopt.xyz").read().split("\n")[2:2 + len(symbols)]
                res["X"] = np.array([[float(v) for v in ln.split()[1:4]] for ln in L])
                res["E_h"] = float(m[-1])
                res["status"] = "ok" if res["converged"] else "not_converged"
            else:
                res["status"] = f"no_output(rc={p.returncode})"
    except Exception as exc:
        res["status"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    res["wall_s"] = time.perf_counter() - t0
    return res


def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    lines = path.read_text().strip().split("\n")
    n = int(lines[0])
    symbols = []
    coords = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(coords, dtype=float)


def main():
    import sys
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "research/bench"))

    from tools.jgen import inputs
    from jacobik.core.energy import MACEPolarEvaluator

    parser = argparse.ArgumentParser(description="GFN2 crude relaxation & strain analysis")
    parser.add_argument("--xyz-dir", default=str(REPO / "research/data/jgen/campaign/B/g/xyz"))
    parser.add_argument("--out", default=str(REPO / "research/data/jgen/campaign/B/g/strain_results.json"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    xyz_dir = Path(args.xyz_dir)
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    workroot = str(out_file.parent / "xtb_work")

    tail_parents = ["DAMVAQ", "HOXJUZ", "IFUROQ", "JEGPUJ", "NEPNEF", "SOBHAS", "USAPOU", "XEYTUT", "YUCZUS", "ZELTAP"]
    seeds = [11, 12, 13]
    arms = ["control", "g1_lr_final_0.01", "g1_lr_final_0.001", "g3_bend_free_1.0"]

    shards = inputs.frame_shards()
    mol_info = {}
    for code in tail_parents:
        sh, ds = shards[code]
        mol = inputs.load_mol(code, sh, ds)
        mol_info[code] = dict(charge=mol.charge, spin=mol.spin_state)

    # 1. Collect all jobs
    jobs = []
    job_keys = []
    unique_geoms = {}
    file_map = {}

    for code in tail_parents:
        d_code = xyz_dir / code
        if not d_code.exists():
            continue
        q = mol_info[code]["charge"]
        spin = mol_info[code]["spin"]
        for arm in arms:
            for s in seeds:
                for kind in ("top1", "floor"):
                    f = d_code / f"{code}_{arm}_s{s}_{kind}.xyz"
                    if not f.exists():
                        continue
                    sym, X = read_xyz(f)
                    # Cache key rounded to 4 decimals
                    geom_key = (code, tuple(np.round(X.ravel(), 4)))
                    if geom_key not in unique_geoms:
                        unique_geoms[geom_key] = (sym, X, q, spin, code)
                    file_map[(code, arm, s, kind)] = (geom_key, sym, X)

    print(f"Total files to analyze: {len(file_map)}; Unique geometries to relax: {len(unique_geoms)}")

    # 2. Run GFN2 relaxations on unique geometries using ThreadPoolExecutor
    print(f"Running GFN2 crude relaxations on {len(unique_geoms)} structures across {args.workers} CPU workers...")
    t0 = time.perf_counter()
    geom_results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(xtb_opt, sym, X, q, spin, workroot): k
            for k, (sym, X, q, spin, code) in unique_geoms.items()
        }
        done = 0
        for fut in futs:
            k = futs[fut]
            res = fut.result()
            geom_results[k] = res
            done += 1
            if done % 10 == 0 or done == len(unique_geoms):
                print(f"  Relaxed {done}/{len(unique_geoms)} in {time.perf_counter() - t0:.1f}s (latest status={res['status']})", flush=True)

    print(f"All relaxations finished in {time.perf_counter() - t0:.1f}s.")

    # 3. Evaluate MACE float64 on unrelaxed and relaxed structures
    print("Evaluating MACE float64 single points on unrelaxed and relaxed geometries...")
    mace_model_path = REPO / "research/bench/tools/jdev/MACE-POLAR-1-S.model"
    ev = MACEPolarEvaluator(model="polar-1-s", device="cuda" if torch.cuda.is_available() else "cpu",
                           dtype="float64", model_path=str(mace_model_path), mode="energy-only").load()

    # Batch MACE queries
    mace_structures_unrel = []
    mace_structures_rel = []
    ordered_keys = list(unique_geoms.keys())

    for k in ordered_keys:
        sym, X_unrel, q, spin, code = unique_geoms[k]
        mace_structures_unrel.append((list(sym), X_unrel, int(q), int(spin)))
        X_rel = geom_results[k]["X"] if geom_results[k]["X"] is not None else X_unrel
        mace_structures_rel.append((list(sym), X_rel, int(q), int(spin)))

    E_unrel_all = np.array(ev.energies(mace_structures_unrel), dtype=float) * EV_TO_KJ
    E_rel_all = np.array(ev.energies(mace_structures_rel), dtype=float) * EV_TO_KJ

    mace_energies = {}
    for i, k in enumerate(ordered_keys):
        mace_energies[k] = {
            "E_unrel_kj": float(E_unrel_all[i]),
            "E_rel_kj": float(E_rel_all[i]),
            "strain_kj": float(E_unrel_all[i] - E_rel_all[i]),
            "xtb_conv": geom_results[k]["converged"],
            "xtb_cycles": geom_results[k]["cycles"],
            "xtb_wall_s": geom_results[k]["wall_s"],
        }

    # 4. Map back to units and arms
    results_by_arm = {arm: {} for arm in arms}
    for (code, arm, s, kind), (gk, sym, X) in file_map.items():
        if code not in results_by_arm[arm]:
            results_by_arm[arm][code] = {}
        if str(s) not in results_by_arm[arm][code]:
            results_by_arm[arm][code][str(s)] = {}
        results_by_arm[arm][code][str(s)][kind] = mace_energies[gk]

    # Save summary json
    out_data = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_unique_geometries": len(unique_geoms),
        "total_files": len(file_map),
        "results": results_by_arm,
    }
    out_file.write_text(json.dumps(out_data, indent=2))
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    import torch
    main()
