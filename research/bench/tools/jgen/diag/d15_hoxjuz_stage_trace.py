"""HOXJUZ ring-angle puzzle: at which stage does the ring crush appear? (CPU only, read-only)

Question. In HOXJUZ the ligand is rigid (no rotatable bonds, no macrocycle), each pool has one embed, and yet the
bonded angles of the ring atoms move between rows (d13: 8 moving ring angles). Is the distortion born in the
embed (E), in the beam and Kabsch starting poses (S), or in the joint solve (F, the stored final rows)?

Stages, seed 11 (same ligand atoms in every stage):
  E  the free-ligand embed the solver starts from: ctx.mol.atoms after multiseed.prepare. The solver takes
     X_full = positions of ctx.mol.atoms (engine.py:2012) for every candidate row.
  S  the starting pose of every beam row: fk.apply(theta_init, X_full) for the ligand that owns the ring, where
     theta_init is the per-row output of _beam_search_all_candidates (torsion seeds plus Kabsch base pose,
     engine.py:2160-2185). These are the rows multiseed.seed_rows produces before the IK solve.
  F  the stored final rows of the seed-11 control pool (effective pool, poolio.py), in reference order
     (X[:, gen_to_ref_idx]).
  R  the reference geometry, reference order.

Metric. For each five- or six-membered ring of a ligand (rings from the minimum cycle basis of the bonded heavy
graph of the stage's own coordinates), the internal angles theta_k in the ring's cyclic order. The regular
polygon angle is 180(n-2)/n. The deviation RMS over the ring's angles needs no atom mapping, and each ring is
compared inside its own stage. Summaries are min / median / max over rows. Ring angles are not compared
atom by atom across stages, because the index spaces differ (shell, seed rows and pool are different orders).

Rigidity check. Max over ligand heavy-atom pairs (pairs in one ligand only, ligands = connected components of the
reference heavy graph) of the spread of their distance across the stored rows, in A. The metal and other ligands
are excluded, because the whole complex is not rigid.

Usage (from the worktree; no GPU):
    python tools/jgen/diag/d15_hoxjuz_stage_trace.py --code HOXJUZ --seed 11 --out-json <path>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxio  # noqa: E402
from poolio import Repairs, load_effective  # noqa: E402


def heavy_graph(X, sym, cov_scale=1.20):
    import networkx as nx
    G = nx.Graph()
    idx = [i for i, s in enumerate(sym) if s in ctxio.LIGHT and s != "H"]
    G.add_nodes_from(idx)
    for ii, i in enumerate(idx):
        for j in idx[ii + 1:]:
            if np.linalg.norm(X[i] - X[j]) < cov_scale * (ctxio.COV[sym[i]] + ctxio.COV[sym[j]]):
                G.add_edge(i, j)
    return G


def ring_list(G):
    import networkx as nx
    return [list(c) for c in nx.minimum_cycle_basis(G) if len(c) in (5, 6)] if G.number_of_edges() else []


def ring_dev(X, ring):
    """RMS deviation (deg) of the internal angles of one ring from the regular polygon angle."""
    n = len(ring)
    dev = []
    for k in range(n):
        a, b, c = ring[k - 1], ring[k], ring[(k + 1) % n]
        u, v = X[a] - X[b], X[c] - X[b]
        cosv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        dev.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))) - 180.0 * (n - 2) / n)
    return float(np.sqrt(np.mean(np.square(dev))))


def stats(vals):
    v = np.asarray(vals, float)
    return dict(min=round(float(v.min()), 2), median=round(float(np.median(v)), 2), max=round(float(v.max()), 2),
                rows=int(v.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="HOXJUZ")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--pools", default=os.path.join(str(ctxio.SCALED), "pools"))
    a = ap.parse_args()

    idx = ctxio.frame_index()
    mol = ctxio.load_parent(a.code, idx)
    builder, ctx, asg_w, err = ctxio.load_ctx(mol, a.seed)
    if ctx is None:
        print(ctxio.to_json(dict(code=a.code, error=err)))
        return
    asg, W = asg_w
    shell = ctx.mol
    E = np.array([at.position for at in shell.atoms], float)
    sym_e = [at.symbol for at in shell.atoms]
    rings_e = ring_list(heavy_graph(E, sym_e))
    out = dict(code=a.code, seed=a.seed, n_atoms_shell=len(E), n_ligands=len(shell.ligands))

    # E: embed.
    out["E"] = {str(r): round(ring_dev(E, r), 2) for r in rings_e}

    # S: starting poses of every beam row, per ligand (fks from _beam_search_all_candidates).
    from tools.jdev import multiseed
    import torch
    shared, per_seed = multiseed.seed_rows(builder, ctx, asg, W, [a.seed])
    fks, _, _, _ = shared
    th_list, X_seed, _, cand = per_seed[0]
    X_full = torch.tensor(E, dtype=torch.float32)
    S_dev = {}
    for li, th in enumerate(th_list):
        Xl = fks[li].apply(th, X_full).detach().cpu().numpy()          # (B, N, 3)
        lig_atoms = set()
        for r in rings_e:
            lig_atoms.update(r)
        comp = None
        for r in rings_e:
            # ring belongs to ligand li if the ligand's apply moved its atoms (checked by coordinate change)
            moved = np.linalg.norm(Xl[0][r] - E[r], axis=1).max() > 1e-6
            if not moved:
                continue
            S_dev.setdefault(str(r), []).extend([ring_dev(Xl[k], r) for k in range(Xl.shape[0])])
    out["S"] = {k: stats(v) for k, v in S_dev.items()}
    out["S_candidates"] = int(len(set(np.asarray(cand).tolist())))

    # F and R: stored pool (effective) and reference.
    rep = Repairs(str(ctxio.SCALED))
    pool = load_effective(a.pools, rep, a.code, a.seed)
    ref = np.asarray(pool["ref_X"], float)
    sym_r = list(pool["ref_symbols"])
    rings_r = ring_list(heavy_graph(ref, sym_r))
    F = pool["X"][:, pool["gen_to_ref"], :]
    de = pool["dE"]
    top, flo = int(np.argmax(pool["score"])), int(np.argmin(de))
    out["R"] = {str(r): round(ring_dev(ref, r), 2) for r in rings_r}
    out["F"] = {str(r): stats([ring_dev(F[k], r) for k in range(F.shape[0])]) for r in rings_r}
    out["F_top1_ring_dev"] = {str(r): round(ring_dev(F[top], r), 2) for r in rings_r}
    out["F_floor_ring_dev"] = {str(r): round(ring_dev(F[flo], r), 2) for r in rings_r}
    out["pool_rows"] = int(F.shape[0])
    out["top1_dE"] = round(float(de[top]), 1)
    out["floor_dE"] = round(float(de[flo]), 1)

    # Rigidity: intra-ligand heavy pair distance spread across stored rows.
    import networkx as nx
    Gr = heavy_graph(ref, sym_r)
    comps = [sorted(c) for c in nx.connected_components(Gr)]
    spreads = []
    for c in comps:
        if len(c) < 2:
            continue
        Fc = F[:, c, :]
        D = np.linalg.norm(Fc[:, :, None, :] - Fc[:, None, :, :], axis=-1)
        spreads.append(float((D.max(0) - D.min(0)).max()))
    out["intra_ligand_pair_spread_A"] = [round(s, 3) for s in spreads]
    out["ligand_components_heavy"] = [len(c) for c in comps]
    print(ctxio.to_json(out))
    if a.out_json:
        with open(a.out_json, "w") as fh:
            fh.write(ctxio.to_json(out))


if __name__ == "__main__":
    main()
