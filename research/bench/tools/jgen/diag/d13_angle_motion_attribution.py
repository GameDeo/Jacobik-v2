"""Which bonded angles actually move within a pool, and how the top-1 vs floor angle error splits (CPU only).

Uses the S2b per-pose pools (one embed per pool: embed_idx = -1 on every row, verified), which store the
generated coordinates X (rows) and the reference geometry (ref_X, ref_symbols). For each bonded non-metal
angle i-j-k (bonds from the reference graph, covalent radii rule of d03):
  spread_deg   = max over rows of theta minus min over rows of theta
  moving       = spread_deg > 0.1
  role         = 'donor-centred' if the centre j is bonded to a metal in the reference, 'ring' if j and both
                 neighbours lie in one reference ring (networkx minimum cycle basis), else 'other'
  err_top1 / err_floor = theta(top-1 row) - theta_ref and theta(floor row) - theta_ref (deg)
Per pool, counts of moving angles by role and the RMS over moving angles of top-1 and floor error.

What this does NOT do: map each moving angle to a DOF (joint type). That needs the FK joint lists of the
ligand (ik/fk.py, angle_bends.py bend specs, ring-opening joints); see the D-T3 specification in
research/archive/docs/roadmap/roadmap_generation_v1.md. Any moving angle that such a mapping cannot explain is a bug to be reported.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d13_angle_motion_attribution.py \
        --s2b <.../s2b_eval_state_v1/pools> --codes HOXJUZ,ZELTAP,YUCZUS,SOBHAS
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d03_bond_distortion import COV, NONMETAL  # noqa: E402

import networkx as nx  # noqa: E402


def angles_deg(X, tri):
    u = X[tri[:, 0]] - X[tri[:, 1]]
    v = X[tri[:, 2]] - X[tri[:, 1]]
    c = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2b", required=True)
    ap.add_argument("--codes", required=True)
    a = ap.parse_args()
    for code in a.codes.split(","):
        for seed in (11, 12, 13):
            base = os.path.join(a.s2b, "%s_%d_control" % (code, seed))
            z = np.load(base + ".npz", allow_pickle=True)
            if "error" in z.files:
                print(code, seed, "error pool")
                continue
            m = json.load(open(base + ".mace64.json"))
            de = np.asarray(m["energies_kj"], float) - float(m["ref_energy_kj"])
            sym = [str(s) for s in z["ref_symbols"]]
            Xref = np.asarray(z["ref_X"], float)
            X = np.asarray(z["X"], float)
            if "gen_to_ref_idx" in z.files:
                X = X[:, np.asarray(z["gen_to_ref_idx"], int), :]
            nm = [i for i, s in enumerate(sym) if s in NONMETAL]
            met = [i for i, s in enumerate(sym) if s not in NONMETAL]
            G = nx.Graph()
            G.add_nodes_from(nm)
            for ii, i in enumerate(nm):
                for j in nm[ii + 1:]:
                    if np.linalg.norm(Xref[i] - Xref[j]) < 1.25 * (COV[sym[i]] + COV[sym[j]]) + 0.25:
                        G.add_edge(i, j)
            donor = {i for i in nm if any(np.linalg.norm(Xref[i] - Xref[m]) <= 1.15 * (
                COV.get(sym[m], 1.4) + COV[sym[i]]) for m in met)}
            rings = [set(c) for c in nx.minimum_cycle_basis(G)] if G.number_of_edges() else []
            tri = [(i, j, k) for j in nm for i in G.neighbors(j) for k in G.neighbors(j) if i < k]
            if not tri:
                continue
            tri = np.array(tri, int)
            th = np.stack([angles_deg(Xr, tri) for Xr in X])          # (rows, n_angles)
            th_ref = angles_deg(Xref, tri)
            spread = th.max(0) - th.min(0)
            moving = spread > 0.1
            roles = []
            for (i, j, k) in tri:
                if j in donor:
                    roles.append("donor-centred")
                elif any({i, j, k} <= r for r in rings):
                    roles.append("ring")
                else:
                    roles.append("other")
            roles = np.array(roles)
            score = np.asarray(z["score"], float)
            top, flo = int(np.argmax(score)), int(np.argmin(de))
            e_top = th[top] - th_ref
            e_flo = th[flo] - th_ref
            out = dict(code=code, seed=seed, rows=int(th.shape[0]), angles=int(len(tri)), moving=int(moving.sum()))
            for role in ("donor-centred", "ring", "other"):
                msk = moving & (roles == role)
                out["moving_" + role.replace("-", "_")] = int(msk.sum())
                if msk.any():
                    out["rms_top1_" + role.replace("-", "_")] = round(float(np.sqrt(np.mean(e_top[msk] ** 2))), 2)
                    out["rms_floor_" + role.replace("-", "_")] = round(float(np.sqrt(np.mean(e_flo[msk] ** 2))), 2)
            for role in ("donor-centred", "ring", "other"):
                msk = moving & (roles == role)
                key = role.replace("-", "_")
                out["ss_top1_" + key] = round(float(np.sum(e_top[msk] ** 2)), 1)
                out["ss_floor_" + key] = round(float(np.sum(e_flo[msk] ** 2)), 1)
            out["fixed_angles_rms_top1"] = round(float(np.sqrt(np.mean(e_top[~moving] ** 2))), 3) if (~moving).any() else None
            out["top1_dE"] = round(float(de[top]), 1)
            out["floor_dE"] = round(float(de[flo]), 1)
            print(json.dumps(out))


if __name__ == "__main__":
    main()
