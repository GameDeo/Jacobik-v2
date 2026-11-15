"""S5 step 1 (research/archive/docs/roadmap/roadmap_generation_v1.md 2.6, S5): planarity of the chelate core and the C-O-C angle of JEGPUJ.

Core = the metal, the donor atoms, and every heavy atom on a ligand-graph path between two donors of one ligand
(the chelate rings and their bridges; substituents excluded). Reported for: the reference (CSD), the free-ligand
embed of the solver shell (legacy and with metallacycle_bite_bound), and the exported top-1 xyz files of each arm.
Measures: RMS of the core from its best plane (A); angles between the best planes of the two aryl-containing chelate
rings (deg); the donors' distance from the metal's donor plane (A); the ether C-O-C angle (deg). CPU only.

    python tools/jgen/diag/d29_s5_planarity.py [--xyz-dir DIR] [--arms control,nofold]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT, os.path.join(ROOT, "refit"), os.path.join(ROOT, "tools", "jdev", "budget")]


def best_plane(X):
    c = X.mean(0)
    _, s, vt = np.linalg.svd(X - c)
    n = vt[2]
    rms = float(np.sqrt(np.mean(((X - c) @ n) ** 2)))
    return c, n, rms


def angle_between(n1, n2):
    v = abs(float(np.dot(n1, n2)) / np.linalg.norm(n1) / np.linalg.norm(n2))
    return float(np.degrees(np.arccos(np.clip(v, 0, 1))))


def core_indices(graph_nodes_paths, donors, metal):
    core = {metal, *donors}
    for path in graph_nodes_paths:
        core |= set(path)
    return sorted(core)


def measures(X, sym, metal, donors, core, ether_O, ring_sets):
    Xc = X[core]
    c, n, rms = best_plane(Xc)
    planes = []
    for ring in ring_sets:
        idx = [i for i in ring if i in core]
        if len(idx) >= 3:
            planes.append(best_plane(X[idx])[1])
    ang = [round(angle_between(planes[i], planes[j]), 1) for i in range(len(planes)) for j in range(i + 1, len(planes))]
    # donors out of the donor plane (metal and donors only)
    dpl = best_plane(X[[metal] + list(donors)])
    dev = [round(float(abs(np.dot(X[d] - dpl[0], dpl[1]))), 3) for d in donors]
    cco = None
    if ether_O is not None:
        nb = [j for j in range(len(sym)) if sym[j] == "C" and np.linalg.norm(X[j] - X[ether_O]) < 1.7]
        if len(nb) == 2:
            u, w = X[nb[0]] - X[ether_O], X[nb[1]] - X[ether_O]
            cco = round(float(np.degrees(np.arccos(np.clip(np.dot(u, w) / np.linalg.norm(u) / np.linalg.norm(w), -1, 1)))), 1)
    return dict(core_atoms=len(core), core_rms_A=round(rms, 3), ring_plane_angles_deg=ang,
                donor_out_of_plane_A=dev, C_O_C_deg=cco)


def read_xyz(path):
    lines = open(path).read().split("\n")
    n = int(lines[0])
    sym, X = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        sym.append(p[0])
        X.append([float(v) for v in p[1:4]])
    return sym, np.array(X)


def topology(smiles_atoms_bonds, donors_sets, metal_idx):
    import networkx as nx
    G = smiles_atoms_bonds
    paths = []
    for a in range(len(donors_sets)):
        for b in range(a + 1, len(donors_sets)):
            for da in donors_sets[a]:
                for db in donors_sets[b]:
                    try:
                        paths.append(nx.shortest_path(G, da, db))
                    except nx.NetworkXNoPath:
                        pass
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="JEGPUJ")
    ap.add_argument("--xyz-dir", default="")
    ap.add_argument("--arms", default="control,nofold")
    a = ap.parse_args()
    import networkx as nx
    from tools.jgen import inputs
    from jacobik.core.env import build_builder_and_inputs

    shard, ds = inputs.frame_shards()[a.code]
    mol = inputs.load_mol(a.code, shard, ds)
    sym = [x.symbol for x in mol.atoms]
    ref = np.array([x.position for x in mol.atoms], float)
    metal = sym.index(next(s for s in sym if s not in ("C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se")))
    G = nx.Graph()
    for b in mol.bonds:
        if metal not in (b.i, b.j):
            G.add_edge(b.i, b.j)
    # donors from the reference: heavy atoms bonded to the metal
    donors = [i for i in range(len(sym)) if i != metal and sym[i] not in ("H",) and
              np.linalg.norm(ref[i] - ref[metal]) < 2.6]
    donor_sets = [[d] for d in donors]
    paths = topology(G, donor_sets, metal)
    core = core_indices(paths, donors, metal)
    ether = [i for i in range(len(sym)) if sym[i] == "O" and i not in (metal,) and
             sum(1 for j in range(len(sym)) if sym[j] == "C" and np.linalg.norm(ref[j] - ref[i]) < 1.7) == 2]
    ether_O = ether[0] if ether else None
    # aryl-containing chelate rings: rings of the ligand graph that touch the core
    rings = [list(c) for c in nx.cycle_basis(G)]
    ring_sets = [r for r in rings if len(r) >= 5]
    out = dict(code=a.code, n_donors=len(donors), core_atoms=len(core), reference=measures(ref, sym, metal, donors, core,
                                                                                          ether_O, ring_sets))
    b, _, err = build_builder_and_inputs(mol)
    if b is not None:
        from jacobik.builder import engine as E
        for flag in (False, True):
            E.METALLACYCLE_BITE_BOUND = flag
            try:
                shell, _, _ = b._build_shell(False, False, chelate_embed=True)
                X = np.array([x.position for x in shell.atoms], float)
                ssym = [x.symbol for x in shell.atoms]
                sm = shell.metal_idx
                Gs = nx.Graph()
                for bd in shell.bonds:
                    if sm not in (bd.i, bd.j):
                        Gs.add_edge(bd.i, bd.j)
                sdon = [[int(site.donor_indices[0])] for site in shell.coord_sites
                        if not site.is_haptic and len(site.donor_indices) == 1]
                sp = topology(Gs, sdon, sm)
                score = core_indices(sp, [d[0] for d in sdon], sm)
                sO = [i for i in range(len(ssym)) if ssym[i] == "O" and sum(
                    1 for j in range(len(ssym)) if ssym[j] == "C" and np.linalg.norm(X[j] - X[i]) < 1.7) == 2]
                rings_s = [list(c) for c in nx.cycle_basis(Gs) if len(c) >= 5]
                out[f"embed_bite_{flag}"] = measures(X, ssym, sm, [d[0] for d in sdon], score,
                                                     sO[0] if sO else None, rings_s)
                out[f"embed_bite_{flag}"]["status"] = getattr(shell, "bite_bound_status", "legacy")
            except Exception as exc:
                out[f"embed_bite_{flag}"] = dict(error=f"{type(exc).__name__}: {str(exc)[:160]}")
        E.METALLACYCLE_BITE_BOUND = False
    if a.xyz_dir:
        for arm in a.arms.split(","):
            for f in sorted(glob.glob(os.path.join(a.xyz_dir, a.code, f"{a.code}_{arm}_s*_top1.xyz"))):
                s2, X2 = read_xyz(f)
                if s2 != sym:
                    out[os.path.basename(f)] = dict(error="symbol order differs from reference")
                    continue
                out[os.path.basename(f)] = measures(X2, sym, metal, donors, core, ether_O, ring_sets)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
