"""Research-only ligand embed arms for F-2. No production modules are modified."""
from __future__ import annotations

import itertools
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from jacobik.builder import embed as ce
from jacobik.builder.engine import _smiles_to_conformer
from jacobik.ik.targets import IDEAL_GEOMETRIES

XTB = os.environ.get("JACOBIK_XTB", "xtb")
BOHR_PER_ANGSTROM = 1.8897261254578281


def _xtb_optimize(symbols, xyz, charge, *, method="gfn2", fixed=(), constraints=(), point_charges=(), timeout=180):
    """Optimize coordinates with xTB. All indices in this helper are zero-based and converted for xcontrol."""
    import shutil
    from rdkit import Chem

    if shutil.which(XTB) is None:
        raise FileNotFoundError(f"xtb executable not found: {XTB}")
    xyz = np.asarray(xyz, float)
    if len(symbols) != len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("xTB input has nonfinite coordinates or symbol/coordinate count mismatch")
    uhf = max(0, (sum(Chem.GetPeriodicTable().GetAtomicNumber(s) for s in symbols) - charge) % 2)
    with tempfile.TemporaryDirectory(prefix="jgen_f2_xtb_") as td:
        xyz_path = Path(td) / "lig.xyz"
        xyz_path.write_text(f"{len(symbols)}\nF-2 embed arm\n" + "".join(
            f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n" for s, p in zip(symbols, xyz)))
        xcontrol = []
        if fixed:
            xcontrol += ["$fix", " atoms: " + ",".join(str(int(i) + 1) for i in fixed), "$end"]
        if constraints:
            xcontrol += ["$constrain", " force constant=1.0"]
            xcontrol += [f" distance: {i+1}, {j+1}, {d:.6f}" for i, j, d in constraints]
            xcontrol += ["$end"]
        if point_charges:
            (Path(td) / "pcharge").write_text("".join(
                f"{q:.6f} {p[0]*BOHR_PER_ANGSTROM:.8f} {p[1]*BOHR_PER_ANGSTROM:.8f} {p[2]*BOHR_PER_ANGSTROM:.8f} 999999\n"
                for q, p in point_charges))
            xcontrol += ["$embedding", " input=pcharge", "$end"]
        if xcontrol:
            (Path(td) / "xcontrol.inp").write_text("\n".join(xcontrol) + "\n")
        gfn_args = ["--gfnff"] if method == "gfnff" else ["--gfn", "2"]
        cmd = [XTB, xyz_path.name, *gfn_args, "--opt", "crude", "--chrg", str(int(charge)), "--uhf", str(uhf)]
        if xcontrol:
            cmd += ["--input", "xcontrol.inp"]
        result = subprocess.run(cmd, cwd=td, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
        opt = Path(td) / "xtbopt.xyz"
        if result.returncode != 0 or not opt.exists():
            tail = (result.stdout or "")[-1200:]
            raise RuntimeError(f"xtb {method} failed (rc={result.returncode}): {tail}")
        lines = opt.read_text().splitlines()
        out = np.asarray([[float(v) for v in line.split()[1:4]] for line in lines[2:2+len(symbols)]], float)
        if out.shape != xyz.shape or not np.isfinite(out).all():
            raise RuntimeError("xtb returned malformed/nonfinite xtbopt.xyz")
        return out


def _target_sites(geometry, donors, positions, base):
    dirs = IDEAL_GEOMETRIES.get(geometry)
    if dirs is None or len(donors) > len(dirs):
        return None
    centers, radii = [], []
    for group in donors:
        centers.append(np.mean(positions[np.asarray(group, int)], axis=0))
        syms = [base.GetAtomWithIdx(int(i)).GetSymbol() for i in group]
        radii.append(float(np.mean([ce.R_MD.get(s, ce.R_MD_DEFAULT) for s in syms])))
    best = None
    for sites in itertools.permutations(range(len(dirs)), len(donors)):
        tgt = np.asarray([dirs[j] * radii[i] for i, j in enumerate(sites)], float)
        src = np.asarray(centers, float)
        a, b = src.mean(0), tgt.mean(0)
        u, _, vt = np.linalg.svd((src-a).T @ (tgt-b))
        rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        aligned = (positions-a) @ rot + b
        err = float(np.sum((aligned[np.asarray([g[0] for g in donors])] - tgt) ** 2))
        # Haptic site centers use their full centroid in the assignment score.
        aligned_centers = np.asarray([aligned[np.asarray(g, int)].mean(0) for g in donors])
        err = float(np.sum((aligned_centers - tgt) ** 2))
        pair_targets = []
        for x in range(len(donors)):
            for y in range(x+1, len(donors)):
                pair_targets.append((x, y, float(np.linalg.norm(tgt[x]-tgt[y]))))
        if best is None or err < best[0]:
            best = (err, aligned, pair_targets, tgt, sites)
    return best


def _production(smiles, donors, seed, ez):
    rd, xyz, _ = _smiles_to_conformer(smiles, embed_seed=seed, donors=donors,
                                        chelate_embed=True, embed="gfn2", ez=ez)
    return rd, np.asarray(xyz, float), "ok"


def _donor_restraint(smiles, donors, geometry, seed, ez):
    from rdkit import Chem
    rd, xyz, _ = _smiles_to_conformer(smiles, embed_seed=seed, donors=donors,
                                        chelate_embed=True, embed="mmff", ez=ez)
    targets = _target_sites(geometry, donors, xyz, Chem.MolFromSmiles(smiles))
    if targets is None or not targets[2]:
        rd, out, _ = _production(smiles, donors, seed, ez)
        return rd, out, "not_applicable"
    constraints = [(donors[i][0], donors[j][0], d) for i, j, d in targets[2]
                   if len(donors[i]) == 1 and len(donors[j]) == 1]
    if not constraints:
        raise ValueError("donor restraint arm has no kappa donor-pair constraints")
    out = _xtb_optimize([a.GetSymbol() for a in rd.GetAtoms()], xyz,
                        Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)), constraints=constraints)
    conf = Chem.Conformer(rd.GetNumAtoms())
    for i, p in enumerate(out): conf.SetAtomPosition(i, p.tolist())
    rd.RemoveAllConformers(); rd.AddConformer(conf, assignId=True)
    return rd, out, "ok"


def _gfnff_preopt(smiles, donors, seed, ez):
    from rdkit import Chem
    rd, xyz, _ = _smiles_to_conformer(smiles, embed_seed=seed, donors=donors,
                                        chelate_embed=True, embed="etkdg", ez=ez)
    base = Chem.MolFromSmiles(smiles)
    xyz = _xtb_optimize([a.GetSymbol() for a in rd.GetAtoms()], xyz,
                        Chem.GetFormalCharge(base), method="gfnff")
    constraints = [(i, j, float(np.linalg.norm(xyz[i]-xyz[j])))
                   for g in donors for x, i in enumerate(g) for j in g[x+1:]]
    out = _xtb_optimize([a.GetSymbol() for a in rd.GetAtoms()], xyz,
                        Chem.GetFormalCharge(base), constraints=constraints)
    conf = Chem.Conformer(rd.GetNumAtoms())
    for i, p in enumerate(out): conf.SetAtomPosition(i, p.tolist())
    rd.RemoveAllConformers(); rd.AddConformer(conf, assignId=True)
    return rd, out, "ok"


def _bound_state(smiles, donors, geometry, metal, charge, seed, ez):
    from rdkit import Chem
    rd, xyz, _ = _smiles_to_conformer(smiles, embed_seed=seed, donors=donors,
                                        chelate_embed=True, embed="mmff", ez=ez)
    targets = _target_sites(geometry, donors, xyz, Chem.MolFromSmiles(smiles))
    if targets is None:
        raise ValueError(f"no template site set for {geometry} and {len(donors)} donor groups")
    placed = targets[1]
    all_xyz = np.vstack((np.zeros((1,3)), placed))
    symbols = [metal] + [a.GetSymbol() for a in rd.GetAtoms()]
    out = _xtb_optimize(symbols, all_xyz, Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)),
                        fixed=(0,))
    lig_xyz = out[1:]
    conf = Chem.Conformer(rd.GetNumAtoms())
    for i, p in enumerate(lig_xyz): conf.SetAtomPosition(i, p.tolist())
    rd.RemoveAllConformers(); rd.AddConformer(conf, assignId=True)
    return rd, lig_xyz, "ok"


def _li_cap_start(smiles, donors, seed, ez):
    """Embed with the production H-cap, then replace each temporary H cap by an explicit Li+ at the same site."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    base = Chem.MolFromSmiles(smiles)
    capped, caps = ce.cap_anions(base, donors)
    cap_h = [i for i in caps if capped.GetAtomWithIdx(i).GetAtomicNum() == 1]
    if not cap_h:
        return None
    mh = Chem.AddHs(capped)
    if ez: ce.set_ez(mh, ez)
    params = AllChem.ETKDGv3(); params.randomSeed = int(seed)
    pairs = ce.donor_pair_bounds(base, donors)
    if pairs:
        from rdkit.DistanceGeometry import DoTriangleSmoothing
        bm = AllChem.GetMoleculeBoundsMatrix(mh)
        for i, j, lo, hi in pairs:
            a,b=max(i,j),min(i,j); bm[b,a]=min(bm[b,a],hi); bm[a,b]=max(bm[a,b],lo)
        if DoTriangleSmoothing(bm): params.SetBoundsMat(bm); params.useRandomCoords=True
    if AllChem.EmbedMolecule(mh, params) == -1:
        raise ValueError("Li-cap ETKDG failed")
    if AllChem.MMFFHasAllMoleculeParams(mh): AllChem.MMFFOptimizeMolecule(mh, maxIters=1000)
    pp = mh.GetConformer().GetPositions().astype(float)
    n = base.GetNumAtoms(); orig = Chem.AddHs(base)
    xyz = np.zeros((orig.GetNumAtoms(),3)); xyz[:n]=pp[:n]
    capset=set(caps); pools={}
    for at in mh.GetAtoms():
        if at.GetAtomicNum()==1 and at.GetIdx() not in capset and at.GetIdx()>=n:
            parent=at.GetNeighbors()[0].GetIdx()
            if parent<n: pools.setdefault(parent,[]).append(at.GetIdx())
    for at in orig.GetAtoms():
        if at.GetAtomicNum()==1:
            parent=at.GetNeighbors()[0].GetIdx(); xyz[at.GetIdx()]=pp[pools[parent].pop(0)]
    li_xyz=np.asarray([pp[i] for i in cap_h])
    return orig, xyz, li_xyz


def _li_cap(smiles, donors, seed, ez):
    from rdkit import Chem
    built = _li_cap_start(smiles, donors, seed, ez)
    if built is None:
        return _production(smiles, donors, seed, ez)[0], _production(smiles, donors, seed, ez)[1], "not_applicable"
    rd, xyz, li = built
    symbols = [a.GetSymbol() for a in rd.GetAtoms()] + ["Li"]*len(li)
    start = np.vstack((xyz, li))
    q = Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)) + len(li)
    out = _xtb_optimize(symbols, start, q)[...,:]
    out_lig = out[:len(xyz)]
    conf = Chem.Conformer(rd.GetNumAtoms())
    for i,p in enumerate(out_lig): conf.SetAtomPosition(i,p.tolist())
    rd.RemoveAllConformers(); rd.AddConformer(conf, assignId=True)
    return rd, out_lig, "ok"


ARMS = {"production": _production, "bound_state": _bound_state,
        "donor_restrained": _donor_restraint, "gfnff_preopt": _gfnff_preopt, "li_cap": _li_cap}


def embed_arm(arm, smiles, donors, *, seed=42, ez=None, geometry=None, metal="Zn", charge=0):
    start = time.perf_counter()
    if arm == "production": result = _production(smiles, donors, seed, ez)
    elif arm == "bound_state": result = _bound_state(smiles, donors, geometry, metal, charge, seed, ez)
    elif arm == "donor_restrained": result = _donor_restraint(smiles, donors, geometry, seed, ez)
    elif arm == "gfnff_preopt": result = _gfnff_preopt(smiles, donors, seed, ez)
    elif arm == "li_cap": result = _li_cap(smiles, donors, seed, ez)
    else: raise ValueError(f"unknown F-2 arm {arm}")
    rd, xyz, status = result
    return rd, np.asarray(xyz,float), status, time.perf_counter()-start
