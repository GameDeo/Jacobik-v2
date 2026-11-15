"""Per-parent structural features from the production data model (geometry, bond orders and SMILES only; no QM columns are touched).

Only these TMComplex fields are read: atoms (symbol, position), metal_idx, metal_symbol, charge, spin_state, coord_geom,
coordination_number(), ligands (atom_indices, donor_atoms, haptic, macrocycles), donor_labels (family, subclass).
Tags are derived from the features in tags.py, so a rule change does not need the structures again."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import geom, policy


def candidate_groups(frame_path, splits=("surface_fit", "calibration")):
    """Groups of the frame in the given splits whose every member is outside the locked exclusions.
    Returns (list of dicts with code, shard, dataset, split, stratum), dict of counts)."""
    pol = policy.DataPolicy()
    fr = json.load(open(frame_path))["main_allocation"]["splits"]
    out, n_all, n_drop = [], 0, 0
    for sp in splits:
        for g in fr[sp]:
            n_all += 1
            members = set(g.get("member_codes", [])) | {g["representative_code"]}
            if {policy.parent(c) for c in members} & pol.excluded:
                n_drop += 1
                continue
            pol.assert_allowed([g["representative_code"]])
            out.append(dict(code=g["representative_code"], shard=g["representative_shard"], dataset=g["dataset"], split=sp,
                            stratum=g["stratum"]))
    return out, dict(groups=n_all, dropped_locked=n_drop, kept=len(out), policy_sha256=pol.sha256)


def _graph(mol):
    sym = [a.symbol for a in mol.atoms]
    X = np.array([a.position for a in mol.atoms], float)
    metal = mol.metal_idx - 1
    adj = geom.ligand_adjacency(X, sym, metal)
    return sym, X, metal, adj


def _bfs_dist(adj, allowed, src):
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.nonzero(adj[u])[0]:
                v = int(v)
                if v in allowed and v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def extract(mol) -> dict:
    sym, X, metal, adj = _graph(mol)
    labels = {int(k) - 1: (v.family, v.subclass) for k, v in mol.donor_labels.items()}
    ligs = []
    for L in mol.ligands:
        atoms = {int(i) - 1 for i in L.atom_indices}
        donors = [int(d) - 1 for d in L.donor_atoms]
        hap = L.haptic
        rec = dict(n_atoms=len(atoms), donors=[sym[d] for d in donors], subclasses=sorted({labels.get(d, ("", ""))[1] for d in donors}),
                   haptic_order=int(getattr(hap, "haptic_order", 0)) if hap is not None else 0,
                   haptic_donors=list(getattr(hap, "donor_symbols", [])) if hap is not None else [],
                   macrocycle=bool(L.macrocycles), rings=[], phosphine=[], carbene=[])
        if hap is None and len(donors) >= 2:
            sizes = set()
            for i, d in enumerate(donors):
                dist = _bfs_dist(adj, atoms, d)
                for e in donors[i + 1:]:
                    if e in dist and dist[e] > 0:
                        sizes.add(dist[e] + 2)            # d..e path edges + the two M-donor bonds = atoms in the metallacycle
            rec["rings"] = sorted(sizes)
        for d in donors:
            nb = [int(v) for v in np.nonzero(adj[d])[0]]
            if sym[d] == "P":
                heavy_extra = [sum(1 for w in np.nonzero(adj[v])[0] if int(w) != d and sym[int(w)] != "H") for v in nb if sym[v] == "C"]
                rec["phosphine"].append(dict(n_c=len(heavy_extra), branching=sorted(heavy_extra)))
            if sym[d] == "C":
                nN = sum(1 for v in nb if sym[v] == "N")
                nH = sum(1 for v in nb if sym[v] == "H")
                rec["carbene"].append(dict(n_N=nN, n_H=nH, n_nb=len(nb)))
        ligs.append(rec)
    return dict(n_atoms=len(sym), metal=mol.metal_symbol, charge=int(mol.charge), cn=int(mol.coordination_number()),
                coord_geom=mol.coord_geom, ligands=ligs, donor_subclasses=sorted({v[1] for v in labels.values()}),
                n_ligands=len(ligs))


def load_features(group: dict, mol):
    f = extract(mol)
    f.update({k: group[k] for k in ("code", "split", "stratum", "dataset")})
    return f


def _worker(args):
    shard, dataset, groups = args
    from tools.jgen import inputs
    from jacobik.topology.mapping import assemble
    from fit.parser import _assemble_raw, _iter_bo_blocks, _iter_xyz_blocks
    parent_dir = inputs.CLEAN / ("lnQM" if dataset == "LNf" else "tmQM")
    xp, bp = parent_dir / "xyz" / f"tmQM_{shard}.xyz", parent_dir / "bo" / f"tmQM_{shard}.BO"
    smiles = inputs.tmqm_smiles(parent_dir)
    want = {g["code"]: g for g in groups}
    out = []
    if not xp.exists() or not bp.exists():
        return [dict(code=c, error="no shard files") for c in want]
    for (comment, atoms), (c, bonds) in zip(_iter_xyz_blocks(xp), _iter_bo_blocks(bp)):
        if c in want:
            try:
                raw = _assemble_raw(comment, atoms, c, bonds, inputs._Row(smiles.get(c)))
                mol = assemble(raw)
                mol.smiles = smiles.get(c)
                out.append(load_features(want.pop(c), mol))
            except Exception as exc:
                out.append(dict(code=c, error=f"{type(exc).__name__}: {str(exc)[:150]}"))
            if not want:
                break
    out += [dict(code=c, error="not in shard") for c in want]
    return out


def collect(groups, workers: int = 4) -> list:
    """Features of all groups, one pass per shard, at most `workers` (<= 2 now, auditor 2026-10-28) processes."""
    import multiprocessing as mp
    by = defaultdict(list)
    for g in groups:
        by[(g["shard"], g["dataset"])].append(g)
    jobs = [(s, d, gs) for (s, d), gs in sorted(by.items())]
    res = []
    if workers <= 1:
        for j in jobs:
            res.extend(_worker(j))
    else:                                    # fork: the parent has not loaded torch; spawn cannot re-import a stdin __main__
        with mp.get_context("fork").Pool(min(workers, 2)) as p:
            for r in p.imap_unordered(_worker, jobs):
                res.extend(r)
    return sorted(res, key=lambda r: r["code"])
