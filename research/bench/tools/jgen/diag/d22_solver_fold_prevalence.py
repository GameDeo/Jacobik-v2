"""Fold-hinge prevalence on the SOLVER's own molecule (CPU, read-only, no IK solve).

Why not the reference molecule: load_mol() returns the reference complex with its own atom order and its own
donor sets (ref HOXJUZ lig 0 donors [2, 3, 4]). The solver builds a shell from the SMILES embed
(engine.py _build_shell -> _build_complex_shell, then classify(), engine.py ~1333) whose donors are re-derived
(HOXJUZ lig 0 donors [16, 9, 2]). bend_specs reads donor_atoms, the ligand atom set and the bond graph, so on the
reference molecule it sees a different graph and finds no fold hinge (HOXJUZ: 0 on the reference, 6 on the shell).

This script rebuilds the shell with builder._build_shell (no beam, no IK), runs decompose_ligand on each ligand
exactly as the solver does before the bend plan (topology/decompose.py:392), and counts the hinge specs from
angle_bends.bend_specs whose moving set splits a 5- or 6-membered ring of the shell graph. Reported per label
(d14 v3 labels) and per ring size and conjugation class (all sp2 = every ring atom has a bond of order >= 1.4 in
the shell; sp3-containing otherwise).

    python tools/jgen/diag/d22_solver_fold_prevalence.py --labels <d14 json> --out-json <path>
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxio  # noqa: E402


def shell_for(mol):
    from jacobik.core.env import build_builder_and_inputs
    builder, _, err = build_builder_and_inputs(mol)
    if builder is None:
        return None, err
    shell, rdmols, donor_idx_full = builder._build_shell(fast_embed=True, debug=False)
    return shell, None


def _dihedral(p0, p1, p2, p3):
    import numpy as np
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def fold_angle_ref(code, shell, piv, oth, mvs, ring):
    """Fold angle (deg, 0 = planar) of a fold hinge on the REFERENCE geometry, via the pool's gen_to_ref map.

    Shell (generated) index of atom k is k-1 (shell.atoms are 1-based). gen_to_ref[r] = generated index of reference
    atom r, so the reference index of generated index g is the inverse permutation. Returns None if the pool is
    missing or the ring atoms are not bonded in the reference order."""
    import numpy as np
    from poolio import Repairs, load_effective
    p = load_effective(os.path.join(str(ctxio.SCALED), "pools"), Repairs(str(ctxio.SCALED)), code, 11)
    if p is None:
        return None
    inv = np.argsort(np.asarray(p["gen_to_ref"], int))
    ref = np.asarray(p["ref_X"], float)
    pos = lambda k: ref[inv[k - 1]]
    n = len(ring)
    if piv not in ring or oth not in ring:
        return None   # the split ring is not the fold's own ring (the fold axis lies on another ring)
    k = list(ring).index(piv)
    nb = [ring[(k - 1) % n], ring[(k + 1) % n]]
    p1 = [x for x in nb if x in mvs]
    p2 = [x for x in nb if x not in mvs]
    if not p1 or not p2:
        return None
    return round(180.0 - abs(_dihedral(pos(p1[0]), pos(piv), pos(oth), pos(p2[0]))), 2)


def main():
    import networkx as nx
    from jacobik.ik import angle_bends as AB
    from jacobik.topology.decompose import decompose_ligand
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    labs = {p["code"]: p["label"] for p in json.load(open(a.labels))["parents"]}
    idx = ctxio.frame_index()
    res = {}
    for code in labs:
        mol = ctxio.load_parent(code, idx)
        if mol is None:
            res[code] = dict(label=labs[code], note="not loadable")
            continue
        try:
            shell, err = shell_for(mol)
        except Exception as exc:
            res[code] = dict(label=labs[code], note="shell error %s: %s" % (type(exc).__name__, str(exc)[:120]))
            continue
        if shell is None:
            res[code] = dict(label=labs[code], note="builder error: %s" % err)
            continue
        found = []
        for lig in shell.ligands:  # the ligand id is the list position
            try:
                decompose_ligand(shell, lig)
                specs = AB.bend_specs(shell, lig)
            except Exception as exc:
                res[code] = dict(label=labs[code], note="bend_specs error %s" % type(exc).__name__)
                break
            G = AB._lig_graph(shell, lig)
            rings = [list(c) for c in nx.minimum_cycle_basis(G) if len(c) in (5, 6)]
            for sp in specs:
                if not (len(sp) > 4 and sp[4][0] == "hinge"):
                    continue
                mvs = set(sp[3])
                for r in rings:
                    rs = set(r)
                    if (rs & mvs) and len(rs & mvs) < len(rs):
                        ring_bonds = [b for b in shell.bonds if b.i in rs and b.j in rs]
                        n_multi = sum(1 for b in ring_bonds if b.order >= 1.4)
                        atoms_multi = {x for b in ring_bonds if b.order >= 1.4 for x in (b.i, b.j)}
                        conj = "all_sp2" if rs <= atoms_multi else "contains_sp3_or_nonconj"
                        haptic_atoms = set()
                        for hg in (getattr(lig, "haptic_groups", None) or []):
                            haptic_atoms |= set(hg.donor_indices)
                        fold_deg = fold_angle_ref(code, shell, sp[0], sp[1], mvs, r)
                        found.append(dict(lig=shell.ligands.index(lig),
                                          piv=int(sp[0]), oth=int(sp[1]), ring=[int(x) for x in r],
                                          size=len(r), conj=conj, n_multi_ring_bonds=n_multi,
                                          haptic_ring=bool(rs & haptic_atoms), fold_deg_ref=fold_deg))
                        break
        res[code] = dict(label=labs[code], note="ok", fold_split_rings=len(found), folds=found[:12])
    counts = collections.Counter()
    for code, v in res.items():
        if v.get("note") == "ok":
            counts[(v["label"], v["fold_split_rings"] > 0)] += 1
    summary = {"%s|%s" % k: n for k, n in sorted(counts.items(), key=str)}
    print(ctxio.to_json(dict(summary=summary)))
    if a.out_json:
        with open(a.out_json, "w") as fh:
            fh.write(ctxio.to_json(dict(summary=summary, parents=res)))


if __name__ == "__main__":
    main()
