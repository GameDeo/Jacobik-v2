"""Which solver DOF moves the ring angles? Column ablation on the stored S2b rows (CPU, read-only).

For one parent and seed, the stored S2b state holds each pool row's final DOF vector (thetas_flat, per ligand,
offsets thetas_lig_offsets). The rebuild uses the CPU embed of the same ligand (multiseed.prepare, which runs the
GFN2 refinement when xtb is on PATH; the pixi bin is put first on PATH by the runner), and the solver's own FK
(fk.apply, which includes the angle-bend columns).

Steps
  1. Rebuild each stored row: X_rec = fk_lig.apply(theta_lig, X) applied ligand by ligand from the embed.
     Check against the stored generated-order coordinates X (max abs difference over ligand atoms, A).
  2. Ring deviation (RMS over the ring's internal angles from the regular polygon angle, deg) of each ring in the
     rebuilt and in the stored rows. Rings come from the embed's own bonded heavy graph (shell order).
  3. Ablation: for every theta column c of every ligand, set that column to zero (the embed value for bend
     columns, the reference torsion for torsions) and recompute the ring deviations. Report the columns whose
     zeroing moves any ring by more than RING_TOL_DEG on the top-1 or the floor row, with the ring affected.
  4. Zeroing all bend columns together (the embed angles of every bend) gives the ring deviation that the
     torsion and base joints alone produce.

Usage (no GPU; CPU embed):
    python tools/jgen/diag/d16_ring_dof_ablation.py --code HOXJUZ --seed 11 --out-json <path>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxio  # noqa: E402
from d15_hoxjuz_stage_trace import heavy_graph, ring_list, ring_dev  # noqa: E402

RING_TOL_DEG = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="HOXJUZ")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    import torch

    idx = ctxio.frame_index()
    mol = ctxio.load_parent(a.code, idx)
    builder, ctx, asg_w, err = ctxio.load_ctx(mol, a.seed)
    if ctx is None:
        print(ctxio.to_json(dict(code=a.code, error=err)))
        return
    shell = ctx.mol
    E = np.array([at.position for at in shell.atoms], float)
    sym = [at.symbol for at in shell.atoms]
    rings = ring_list(heavy_graph(E, sym))

    from tools.jdev import multiseed
    shared, _ = multiseed.seed_rows(builder, ctx, asg_w[0], asg_w[1], [a.seed])
    fks = shared[0]
    lig_atoms = [list(getattr(lig, "atom_indices", [])) for lig in shell.ligands]

    z = np.load(os.path.join(str(ctxio.SCALED), "s2b_eval_state_v1", "pools", "%s_%d_control.npz" % (a.code, a.seed)),
                allow_pickle=True)
    th = np.asarray(z["thetas_flat"], float)
    off = np.asarray(z["thetas_lig_offsets"], int)
    Xst = np.asarray(z["X"], float)
    score = np.asarray(z["score"], float)
    top, flo = int(np.argmax(score)), None
    m = json.load(open(os.path.join(str(ctxio.SCALED), "pools", "%s_%d_control.mace64.json" % (a.code, a.seed))))
    de = np.asarray(m["energies_kj"], float) - float(m["ref_energy_kj"])
    flo = int(np.argmin(de))
    n_lig = len(off) - 1                       # thetas_lig_offsets holds the start of each ligand plus the end
    widths = [int(off[i + 1] - off[i]) for i in range(n_lig)]

    def rebuild(thetas_row):
        X = torch.tensor(E, dtype=torch.float32)[None]
        for li in range(n_lig):
            seg = thetas_row[off[li]: off[li] + widths[li]]
            X = fks[li].apply(torch.tensor(seg, dtype=torch.float32)[None], X[0])
        return X[0].double().numpy()

    out = dict(code=a.code, seed=a.seed, rings=[str(r) for r in rings], widths=widths, rows=int(th.shape[0]),
               top1=top, floor=flo, n_bend_columns=[int(getattr(f, "n_angle_bends", 0)) for f in fks])
    # 1. reproduction
    recon_err = []
    for r in range(th.shape[0]):
        Xr = rebuild(th[r])
        recon_err.append(float(np.abs(Xr - Xst[r]).max()))
    out["reconstruction_max_abs_err_A"] = [round(x, 4) for x in recon_err]
    # 2. ring deviations, rebuilt (R) and stored (S)
    out["ring_dev_stored"] = {str(r): [round(ring_dev(Xst[k], r), 2) for k in range(th.shape[0])] for r in rings}
    base_dev = {str(r): [round(ring_dev(rebuild(th[k]), r), 2) for k in range(th.shape[0])] for r in rings}
    out["ring_dev_rebuilt"] = base_dev
    # 3. column ablation for the top-1 and the floor row
    effects = []
    for li in range(n_lig):
        for c in range(widths[li]):
            for label, row in (("top1", top), ("floor", flo)):
                t = th[row].copy()
                t[off[li] + c] = 0.0
                X = rebuild(t)
                d = {str(r): round(ring_dev(X, r) - ring_dev(rebuild(th[row]), r), 2) for r in rings}
                big = {k: v for k, v in d.items() if abs(v) > RING_TOL_DEG}
                if big:
                    effects.append(dict(lig=li, col=c, row=label, delta_ring_dev_deg=big))
    out["ablation_effects"] = effects
    # 3b. invariants of the rigid-motion claim (angle_bends.py header): bond lengths and ring internals.
    import networkx as nx
    Gh = heavy_graph(E, sym)
    edges = list(Gh.edges())
    d_e = {e: np.linalg.norm(E[e[0]] - E[e[1]]) for e in edges}
    inv = {}
    for label, row in (("top1", top), ("floor", flo)):
        X = rebuild(th[row])
        changes = sorted(((abs(np.linalg.norm(X[u] - X[v]) - d_e[(u, v)]), u, v) for u, v in edges), reverse=True)
        inv[label] = dict(max_heavy_bond_change_A=round(float(changes[0][0]), 4),
                          top5_edges=[dict(i=int(u), j=int(v), sym=[sym[u], sym[v]], d_embed=round(float(d_e[(u, v)]), 3),
                                           change_A=round(float(c), 3)) for c, u, v in changes[:5]],
                          n_edges_changed_over_0p05=int(sum(1 for c, _, _ in changes if c > 0.05)),
                          n_edges=len(edges))
    out["invariants_rebuilt_rows"] = inv
    # 3c. for each bend column of ligand 0 with a stored non-zero value: which atoms it displaces, and whether it
    # displaces part of a ring (a ring with some atoms displaced and some not).
    moves = []
    for li in range(n_lig):
        for c in range(widths[li]):
            vals = th[:, off[li] + c]
            if not np.any(np.abs(vals) > 1e-9):
                continue
            t_zero = th[top].copy()
            t_zero[off[li] + c] = 0.0
            dX = rebuild(th[top]) - rebuild(t_zero)
            moved = np.where(np.linalg.norm(dX, axis=1) > 1e-4)[0]
            partial = []
            for r in rings:
                mv = [a in set(moved.tolist()) for a in r]
                if any(mv) and not all(mv):
                    partial.append([int(a) for a in r])
            moves.append(dict(lig=li, col=c, stored_abs_max_rad=round(float(np.abs(vals).max()), 4),
                              n_moved_atoms=int(len(moved)), moved_atoms=[int(i) for i in moved],
                              partial_rings=partial))
    out["bend_column_displacements_top1"] = moves
    # 4. all bend columns zeroed (torsion and base joints only): which ring deviation remains
    nb_zeroed = {}
    for label, row in (("top1", top), ("floor", flo)):
        t = th[row].copy()
        for li, f in enumerate(fks):
            nb = int(getattr(f, "n_angle_bends", 0))
            if nb:
                t[off[li] + widths[li] - nb: off[li] + widths[li]] = 0.0
        X = rebuild(t)
        nb_zeroed[label] = {str(r): round(ring_dev(X, r), 2) for r in rings}
    out["all_bends_zeroed_ring_dev"] = nb_zeroed
    out["top1_dE"] = round(float(de[top]), 1)
    out["floor_dE"] = round(float(de[flo]), 1)
    print(ctxio.to_json(out))
    if a.out_json:
        with open(a.out_json, "w") as fh:
            fh.write(ctxio.to_json(out))


if __name__ == "__main__":
    main()
