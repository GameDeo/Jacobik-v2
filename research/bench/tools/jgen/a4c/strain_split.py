"""Direct strain split of pool rows by constrained GFN2 relaxation: which coordinates hold the energy?

    python -m tools.jgen.a4c.strain_split --run SOLVE_RUN --arm control --inputs A4B_INPUTS --out OUT [--jobs 16]
        [--seed 11] [--rows top1] [--codes ...]

Per row (default: the top-1 row of one seed per parent) the unrelaxed geometry X0 is relaxed with groups of internal
coordinates restrained at their X0 values (xtb $constrain, harmonic, force constant 1.0 Eh/bohr^2 as in the embed),
then a plain GFN2 single point is taken on the result, so no restraint energy is counted. Coordinate classes are those
of `a4c.displacement` (one dihedral per bond). Runs, named by what is FREE:

  all        nothing restrained (the row's own minimum)
  internals  ligand bonds and angles free; every torsion and the metal coordinates (M-D, D-M-D, M-D-X) restrained
  solver     torsions and metal coordinates free; ligand bonds and angles restrained
  core       metal coordinates and chelate/donor torsions free; ligand bonds/angles and the rotors/rings restrained
  torsions   every torsion free; ligand bonds/angles and the metal coordinates restrained
  metal      metal coordinates free; everything in the ligands restrained
  md         only the metal-donor distances free (what Q1 sets)
  metal_angles  only D-M-D and M-D-X angles free
  lig_bonds  only ligand bonds free;  lig_angles  only ligand angles free

Released energy of a run = E(X0) - E(result), kJ/mol, GFN2. `internals` + `solver` need not add up to `all` (the
groups are coupled); both are reported with the total. Each row is saved as it finishes (OUT/rows/) and a rerun
resumes; --summary-only prints the table from the saved rows. Never fails the batch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

H2KJ = 2625.4996
LIG = ("B_donor", "B_other", "A_donor", "A_adj", "A_other")
METAL = ("B_MD", "A_DMD_cis", "A_DMD_trans", "A_MDX")
T_CORE = ("T_chel", "T_donor")
T_PERI = ("T_ring", "T_big", "T_small", "T_H")
RESTRAIN = {"all": (), "internals": METAL + T_CORE + T_PERI, "solver": LIG, "core": LIG + T_PERI,
            "torsions": LIG + METAL, "metal": LIG + T_CORE + T_PERI,
            "md": LIG + T_CORE + T_PERI + METAL[1:], "metal_angles": LIG + T_CORE + T_PERI + METAL[:1],
            "lig_bonds": ("A_donor", "A_adj", "A_other") + METAL + T_CORE + T_PERI,
            "lig_angles": ("B_donor", "B_other") + METAL + T_CORE + T_PERI}
_E = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)")


def _xtb(binary, syms, X, q, mult, td, constraints=None, opt=None, cycles=400, timeout=900):
    with open(f"{td}/in.xyz", "w") as f:
        f.write(f"{len(syms)}\n\n" + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(syms, X)))
    for n in ("xtbopt.xyz", "xtbrestart", "c.inp"):
        Path(td, n).unlink(missing_ok=True)
    cmd = [binary, "in.xyz", "--gfn", "2", "--chrg", str(q), "--uhf", str(mult - 1), "-P", "1"]
    if opt:
        cmd += ["--opt", opt, "--cycles", str(cycles)]
    if constraints:
        with open(f"{td}/c.inp", "w") as f:
            f.write("$constrain\n force constant=1.0\n")
            for kind, at in constraints:
                f.write(f" {kind}: {', '.join(str(a + 1) for a in at)}, auto\n")
            f.write("$end\n")
        cmd += ["--input", "c.inp"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="500M")
    p = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=timeout, env=env)
    m = _E.findall(p.stdout)
    Xo = None
    if opt and os.path.exists(f"{td}/xtbopt.xyz"):
        L = open(f"{td}/xtbopt.xyz").read().split("\n")[2:2 + len(syms)]
        Xo = np.array([[float(v) for v in ln.split()[1:4]] for ln in L])
    cyc = re.search(r"GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)", p.stdout)
    return (float(m[-1]) if m else None), Xo, (int(cyc.group(1)) if cyc else None)


def one(args):
    run, arm, code, seed, q, mult, workroot, runs = args
    done = Path(workroot).parent / "rows" / f"{code}_{seed}.json"
    if done.exists():                                                 # resume: a finished row is never recomputed
        return json.loads(done.read_text())
    out = dict(code=code, seed=seed, error=None, runs={})
    t0 = time.perf_counter()
    try:
        from tools.jgen.bench.pool_export import _read_frames
        from tools.jgen.a4c.displacement import coordinates
        from tools.runkit import xtb
        binary = xtb.info(None)["path"]
        mp = Path(run) / arm / "xyz" / code / f"{code}_{arm}_s{seed}_pool.json"
        m = json.loads(mp.read_text())
        g2r = np.asarray(m["gen_to_ref_idx"], int)
        syms = list(m["ref_symbols"])
        top = next(i for i, r in enumerate(m["rows"]) if r["is_top1"])
        frames = [np.asarray(fr[2], float)[g2r] for f in dict.fromkeys(r["file"] for r in m["rows"]) for fr in _read_frames(mp.parent / f)]
        X0 = frames[top]
        coords = coordinates(syms, np.asarray(m["ref_X"], float))
        kind = {"b": "distance", "a": "angle", "t": "dihedral"}
        out.update(row=top, n_atoms=len(syms), n_coords={c: sum(1 for x in coords if x[0] == c) for c in LIG + METAL + T_CORE + T_PERI})
        Path(workroot).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workroot) as td:
            e0, _, _ = _xtb(binary, syms, X0, q, mult, td)
            if e0 is None:
                raise ValueError("single point of the row failed")
            out["E0_h"] = e0
            for name in runs:
                rs = set(RESTRAIN[name])
                cons = [(kind[k], at) for c, k, at in coords if c in rs]
                try:
                    _, Xo, cyc = _xtb(binary, syms, X0, q, mult, td, constraints=cons, opt="normal")
                    if Xo is None:
                        raise ValueError("no optimised geometry")
                    e1, _, _ = _xtb(binary, syms, Xo, q, mult, td)
                    out["runs"][name] = dict(released=(e0 - e1) * H2KJ, cycles=cyc, n_restraints=len(cons),
                                             rmsd_all=float(np.sqrt((((Xo - Xo.mean(0)) - (X0 - X0.mean(0))) ** 2).sum(1).mean())))
                except Exception as exc:
                    out["runs"][name] = dict(error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["wall_s"] = time.perf_counter() - t0
    try:
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(out))
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+"); ap.add_argument("--arm", default="control")
    ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--codes", nargs="*"); ap.add_argument("--runs", nargs="+", default=list(RESTRAIN))
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cm = json.loads((Path(a.inputs) / "charge_mult.json").read_text())
    work = []
    for run in a.run:
        for p in sorted((Path(run) / a.arm / "xyz").iterdir()):
            c = p.name
            if p.is_dir() and c in cm and (not a.codes or c in a.codes) and (p / f"{c}_{a.arm}_s{a.seed}_pool.json").exists():
                work.append((run, a.arm, c, a.seed, int(cm[c]["charge"]), int(cm[c]["mult"]), str(out / "tmp"), a.runs))
    if a.summary_only:
        res = [json.loads(p.read_text()) for p in sorted((out / "rows").glob("*.json"))]
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            res = list(ex.map(one, work, chunksize=1))
    (out / "strain_split.json").write_text(json.dumps(res))
    ok = [r for r in res if not r["error"] and all("released" in r["runs"].get(n, {}) for n in a.runs)]
    print(f"rows {len(res)} complete {len(ok)}; failed {[(r['code'], r['error'] or [n for n in a.runs if 'released' not in r['runs'].get(n, {})]) for r in res if r not in ok][:15]}")
    tot = np.array([r["runs"]["all"]["released"] for r in ok]) if "all" in a.runs else None
    print(f"{'free':10s} {'released kJ/mol: p25':>20s} {'median':>8s} {'p75':>8s} {'mean':>8s} {'share of all (mean ratio of sums)':>36s} {'cycles median':>14s}")
    for n in a.runs:
        v = np.array([r["runs"][n]["released"] for r in ok])
        cy = [r["runs"][n]["cycles"] for r in ok if r["runs"][n]["cycles"]]
        share = v.sum() / tot.sum() if tot is not None else float("nan")
        print(f"{n:10s} {np.percentile(v, 25):20.1f} {np.median(v):8.1f} {np.percentile(v, 75):8.1f} {v.mean():8.1f} {share:36.2f} {np.median(cy) if cy else float('nan'):14.0f}")


if __name__ == "__main__":
    main()
