"""Record the IK solver trajectory (every step, every row) for one parent and seed, CPU (trace_hook disables the CUDA
replay). Writes multi-frame xyz in reference order and a summary with, per solver call:

  * step-0 metal-donor distances of every row (per donor: min / median / max over rows);
  * the bend-column travel of every row over the first 20 steps (max abs change of any angle-bend column, rad),
    and the bend-column travel over the whole run;
  * the final score of every row and the top five rows.

Outputs go to <main checkout>/inspect/solver_trace/<CODE>/ (the owner's inspection folder; the only write outside the
worktree besides inspect/jgen_bench). Read only otherwise. Atom order: the solver's own (shell) order, which the
reference mapping in the xyz files uses.

    python tools/jgen/diag/trace_solve.py CODE SEED [--out DIR]
"""
import argparse
import json
import os
import sys
import time

# Resolve repo root regardless of file location
_cur = os.path.abspath(__file__)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
while _cur and _cur != os.path.dirname(_cur):
    if os.path.exists(os.path.join(_cur, "src", "jacobik")):
        ROOT = _cur
        break
    _cur = os.path.dirname(_cur)

sys.path[:0] = [
    os.path.join(ROOT, "src"),
    ROOT,
    os.path.join(ROOT, "research", "bench"),
    os.path.join(ROOT, "refit"),
    os.path.join(ROOT, "tools", "jdev", "budget"),
]

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("seed", type=int)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from tools.jgen import inputs
    from jacobik.core.env import build_builder_and_inputs
    from tools.jdev.budget.common import stats_from_db
    from jacobik.ik.ik_solver import JointIKSolver
    from jacobik.builder.atom_map import build_atom_mapping
    from jacobik.ik.fk import ForwardKinematics
    from jacobik.topology.decompose import decompose_ligand

    out_dir = a.out or os.path.join(str(inputs.DATA.parent), "inspect", "solver_trace", a.code)
    os.makedirs(out_dir, exist_ok=True)
    shard, ds = inputs.frame_shards()[a.code]
    mol = inputs.load_mol(a.code, shard, ds)
    ref_sym = [x.symbol for x in mol.atoms]
    ref_X = np.array([x.position for x in mol.atoms], float)
    builder, _, err = build_builder_and_inputs(mol)
    if builder is None:
        raise SystemExit(f"unmappable: {err}")
    stats = stats_from_db(inputs.DB_PATH)

    # the solver's shell layout (theta columns: 6 rigid, n_joints torsion, n_angle_bends bend) per ligand
    shell, _, _ = builder._build_shell(False, False, chelate_embed=True)
    for lig in shell.ligands:
        decompose_ligand(shell, lig)
        lig.macrocycles = []
    fks = [ForwardKinematics(shell, lig, torch.device("cpu"), angle_bends=True) for lig in shell.ligands]
    bend_cols = []
    for f in fks:
        n_tot = 6 + f.n_joints + f.n_angle_bends
        bend_cols.append(list(range(6 + f.n_joints, n_tot)))  # per-ligand theta columns

    # shell atom indices are 1-based (shell.atoms[i].idx == i + 1); the solver's coordinate arrays are 0-based
    metal = shell.metal_idx - 1
    donors = [int(s.donor_indices[0]) - 1 for s in shell.coord_sites if len(s.donor_indices) == 1]
    assert shell.atoms[metal].symbol == shell.metal_symbol, "metal index convention"

    calls = []

    def hook(t, d):
        if t == 0:
            calls.append(dict(X=[], score=[], alive=[], thetas=[]))
        c = calls[-1]
        c["X"].append(d["X"].cpu().numpy().astype(np.float32))
        comps = d["comps"]
        s = sum(v for v in comps.values()) if comps else torch.zeros(d["X"].shape[0])
        c["score"].append((s + d["bend_prior"]).cpu().numpy() if torch.is_tensor(s) else np.zeros(d["X"].shape[0]))
        c["alive"].append(None if d["alive"] is None else d["alive"].cpu().numpy())
        # copy: on CPU .numpy() is a view, and the optimiser updates the parameters in place
        c["thetas"].append([np.array(th.detach().cpu().numpy(), copy=True) for th in d["thetas"]])

    JointIKSolver.trace_hook = staticmethod(hook)
    t0 = time.time()
    try:
        res = builder.solve(seed=a.seed, stats=stats, angle_bends=True, device="cpu")
    finally:
        JointIKSolver.trace_hook = None
    print("solve %.0fs, solver calls %d, conformers %d" % (time.time() - t0, len(calls), len(res.conformers)), flush=True)

    conf = res.conformers[0]
    Xg = np.array([x.position for x in conf.complex.atoms])
    am = build_atom_mapping(mol, gen_mol=conf.complex, X_ref=ref_X, X_gen=Xg)
    g2r = np.asarray(am.gen_to_ref_idx, int)

    def wx(path, frames, comments):
        with open(path, "w") as fh:
            for X, c in zip(frames, comments):
                fh.write("%d\n%s\n" % (len(ref_sym), c))
                for s, (x, y, z) in zip(ref_sym, X[g2r]):
                    fh.write("%-2s %12.6f %12.6f %12.6f\n" % (s, x, y, z))

    wx(os.path.join(out_dir, f"{a.code}_reference.xyz"), [ref_X[np.argsort(g2r)]], ["reference"])
    summary = dict(code=a.code, seed=a.seed, bend_columns=bend_cols, calls=[],
                   layout=[dict(n_joints=f.n_joints, n_angle_bends=f.n_angle_bends) for f in fks])
    for ci, c in enumerate(calls):
        T = len(c["X"])
        n_rows = c["X"][0].shape[0]
        last = {}
        for t in range(T):
            al = c["alive"][t]
            idx = np.arange(c["X"][t].shape[0]) if al is None else al
            for j, r in enumerate(idx):
                last[int(r)] = (t, j)
        fin = np.array([c["score"][last[r][0]][last[r][1]] if r in last else -np.inf for r in range(n_rows)])
        order = np.argsort(-fin)[:5]
        # step-0 metal-donor distances per row
        X0 = c["X"][0]
        md = np.array([[np.linalg.norm(X0[r][d] - X0[r][metal]) for d in donors] for r in range(n_rows)])
        # bend travel: the bend columns of each row (rows follow the alive index by position in the batch)
        travel20, travel_all = [], []
        th0 = c["thetas"][0]
        for r in range(n_rows):
            def bend_vec(t):
                al = c["alive"][t]
                idx = np.arange(c["X"][t].shape[0]) if al is None else al
                w = np.where(idx == r)[0]
                if len(w) == 0:
                    return None
                return np.concatenate([th[w[0]][bend_cols[li]] for li, th in enumerate(c["thetas"][t])] or [np.zeros(0)])
            b0 = bend_vec(0)
            if b0 is None or b0.size == 0:
                continue
            tr20 = max((np.max(np.abs(bend_vec(t) - b0)) for t in range(min(20, T)) if bend_vec(t) is not None), default=0.0)
            trall = max((np.max(np.abs(bend_vec(t) - b0)) for t in range(T) if bend_vec(t) is not None), default=0.0)
            travel20.append(float(tr20))
            travel_all.append(float(trall))
        ci_rec = dict(call=ci, steps=T, rows=int(n_rows),
                      step0_metal_donor_A=dict(
                          min=[round(float(x), 3) for x in md.min(0)],
                          median=[round(float(x), 3) for x in np.median(md, 0)],
                          max=[round(float(x), 3) for x in md.max(0)]),
                      bend_travel_first20_rad=dict(max=round(max(travel20, default=0.0), 4),
                                                   median=round(float(np.median(travel20)) if travel20 else 0.0, 4)),
                      bend_travel_all_rad=dict(max=round(max(travel_all, default=0.0), 4),
                                               median=round(float(np.median(travel_all)) if travel_all else 0.0, 4)),
                      top_final=[round(float(fin[r]), 3) for r in order])
        # diagnostic: the full theta row 0 range over the run (all columns), to check the bend columns move at all
        full0 = np.array([np.concatenate([th[0] for th in c["thetas"][t]]) for t in range(T)])
        ci_rec["theta_row0_range_all_columns_rad"] = round(float(np.ptp(full0, axis=0).max()), 4)
        flat_bend = [idx for sub in bend_cols for idx in sub]
        ci_rec["theta_row0_range_bend_columns_rad"] = round(float(np.ptp(full0[:, flat_bend], axis=0).max()), 4) if flat_bend else 0.0
        summary["calls"].append(ci_rec)
        wx(os.path.join(out_dir, f"{a.code}_s{a.seed}_call{ci}_all_starts.xyz"), [X0[r] for r in range(n_rows)],
           [f"call {ci} row {r} start score {fin[r]:.3f}" for r in range(n_rows)])
        for rank, r in enumerate(order):
            frames, com = [], []
            for t in range(T):
                al = c["alive"][t]
                idx = np.arange(c["X"][t].shape[0]) if al is None else al
                w = np.where(idx == r)[0]
                if len(w) == 0:
                    break
                frames.append(c["X"][t][w[0]])
                com.append("call %d row %d step %d score %.3f" % (ci, r, t, c["score"][t][w[0]]))
            wx(os.path.join(out_dir, f"{a.code}_s{a.seed}_call{ci}_rank{rank}_row{r}_traj.xyz"), frames, com)
    for k, cf in enumerate(res.conformers[:5]):
        wx(os.path.join(out_dir, f"{a.code}_s{a.seed}_final_conf{k}.xyz"),
           [np.array([x.position for x in cf.complex.atoms])], ["final conformer %d score %.3f" % (k, cf.score)])
    with open(os.path.join(out_dir, f"{a.code}_s{a.seed}_trace_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary["calls"], default=str)[:800])


if __name__ == "__main__":
    main()
