"""Production-path embed geometry census. CPU only; tmQM property columns are never loaded."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "bench"))

from tools.jgen import inputs  # noqa: E402

COV = {"H": .31, "B": .84, "C": .76, "N": .71, "O": .66, "F": .57, "Si": 1.11, "P": 1.07,
       "S": 1.05, "Cl": 1.02, "Se": 1.20, "Br": 1.20, "I": 1.39, "As": 1.19}


class FallbackGuard(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.events: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if "GFN2 embed refinement failed" in msg:
            self.events.append(msg)


def angle_deg(x, y, z):
    a, b = np.asarray(x) - y, np.asarray(z) - y
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / max(den, 1e-12), -1, 1))))


def dihedral_deg(p0, p1, p2, p3):
    p0, p1, p2, p3 = [np.asarray(p, float) for p in (p0, p1, p2, p3)]
    b0, b1, b2 = -(p1-p0), p2-p1, p3-p2
    b1 = b1 / max(np.linalg.norm(b1), 1e-12)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _reference_by_rdkit(mol, group, rdkit_mol):
    """Map heavy atoms via the ligand-group isomorphism; map H locally by parent and stable order."""
    xyz = {a.idx: np.asarray(a.position, float) for a in mol.atoms}
    heavy_map = group["rdkit_to_xyz"]
    out = np.full((rdkit_mol.GetNumAtoms(), 3), np.nan)
    for ri, xi in heavy_map.items():
        if ri < rdkit_mol.GetNumAtoms() and xi in xyz:
            out[ri] = xyz[xi]
    bond_h = {}
    for b in mol.bonds:
        if b.order < .5:
            continue
        a, c = b.i, b.j
        sa = mol.atoms[a - 1].symbol if 0 < a <= len(mol.atoms) else ""
        sc = mol.atoms[c - 1].symbol if 0 < c <= len(mol.atoms) else ""
        if sa == "H" and c in heavy_map.values():
            bond_h.setdefault(c, []).append(a)
        if sc == "H" and a in heavy_map.values():
            bond_h.setdefault(a, []).append(c)
    for vals in bond_h.values():
        vals.sort()
    for atom in rdkit_mol.GetAtoms():
        ri = atom.GetIdx()
        if atom.GetAtomicNum() != 1:
            continue
        parent = atom.GetNeighbors()[0].GetIdx()
        xi = heavy_map.get(parent)
        hs = bond_h.get(xi, []) if xi is not None else []
        same = sorted(n.GetIdx() for n in rdkit_mol.GetAtomWithIdx(parent).GetNeighbors()
                      if n.GetAtomicNum() == 1)
        rank = same.index(ri) if ri in same else 0
        if rank < len(hs) and hs[rank] in xyz:
            out[ri] = xyz[hs[rank]]
    return out


def _symmetry_align_reference(smiles, embedded, reference, bend_pivots=None, max_automorphisms=4096):
    """Choose the stereochemistry-preserving graph automorphism with the lowest aligned heavy-atom RMS."""
    import itertools
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    base = Chem.MolFromSmiles(smiles)
    full = Chem.AddHs(base)
    # Enumerate heavy-atom symmetries only. Full-H matches explode by 3! per methyl group.
    matches = base.GetSubstructMatches(base, uniquify=False, useChirality=True, maxMatches=max_automorphisms)
    if not matches:
        matches = [tuple(range(base.GetNumAtoms()))]
    emb = np.asarray(embedded, float)
    ref = np.asarray(reference, float)
    probe, target = Chem.Mol(base), Chem.Mol(base)
    for mol, coords in ((probe, emb), (target, ref)):
        conf = Chem.Conformer(base.GetNumAtoms())
        for i in range(base.GetNumAtoms()):
            conf.SetAtomPosition(i, np.nan_to_num(coords[i], nan=0.0).tolist())
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
    atom_maps = [[(i, int(j)) for i, j in enumerate(match)] for match in matches]
    rms, _, heavy_pairs = rdMolAlign.GetBestAlignmentTransform(
        probe, target, map=atom_maps, maxMatches=max_automorphisms, numThreads=1, reflect=False)
    heavy_map = dict(heavy_pairs)
    atom_map = [-1] * full.GetNumAtoms()
    mapped = np.full_like(ref, np.nan)
    for i, j in heavy_map.items():
        atom_map[i] = j
        mapped[i] = ref[j]
    # Map equivalent H atoms on each already-mapped heavy atom using its local frozen bonds and angles.
    for parent in range(base.GetNumAtoms()):
        target_parent = int(heavy_map[parent])
        query_h = sorted(n.GetIdx() for n in full.GetAtomWithIdx(parent).GetNeighbors()
                         if n.GetAtomicNum() == 1)
        target_h = sorted(n.GetIdx() for n in full.GetAtomWithIdx(target_parent).GetNeighbors()
                          if n.GetAtomicNum() == 1)
        if len(query_h) != len(target_h):
            continue
        best_perm, best_cost = None, float("inf")
        for perm in itertools.permutations(target_h):
            trial = mapped.copy()
            for qi, rj in zip(query_h, perm):
                trial[qi] = ref[rj]
            c = 0.0
            for qi in query_h:
                dr = np.linalg.norm(emb[qi] - emb[parent]) - np.linalg.norm(trial[qi] - trial[parent])
                c += (dr / .15) ** 2
            nbr = [n.GetIdx() for n in full.GetAtomWithIdx(parent).GetNeighbors()]
            for x in range(len(nbr)):
                for y in range(x + 1, len(nbr)):
                    a, b = nbr[x], nbr[y]
                    if (a not in query_h and b not in query_h) or not np.isfinite(trial[[a, parent, b]]).all():
                        continue
                    da = angle_deg(emb[a], emb[parent], emb[b])
                    db = angle_deg(trial[a], trial[parent], trial[b])
                    c += ((da - db) / 15.0) ** 2
            if c < best_cost:
                best_perm, best_cost = perm, c
        for qi, rj in zip(query_h, best_perm or ()):
            atom_map[qi] = rj
            mapped[qi] = ref[rj]
    return mapped, {"automorphisms_considered": len(matches),
        "automorphism_search_truncated": len(matches) >= max_automorphisms,
        "aligned_heavy_rms_A": float(rms), "query_to_reference_atom": atom_map}


def _ligand_metrics(smiles, donor_groups, embedded, reference, bend_pivots=None):
    from rdkit import Chem

    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError("RDKit could not parse ligand SMILES")
    full = Chem.AddHs(base)
    emb = np.asarray(embedded, float)
    if emb.shape != (full.GetNumAtoms(), 3):
        raise ValueError(f"embed atom count mismatch: {emb.shape} vs {full.GetNumAtoms()}")
    reference, symmetry = _symmetry_align_reference(smiles, emb, reference, bend_pivots)
    bonds = []
    bonded_anomalies = []
    for b in full.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if np.isfinite(reference[[i, j]]).all():
            er = float(np.linalg.norm(emb[i] - emb[j]))
            rr = float(np.linalg.norm(reference[i] - reference[j]))
            si, sj = full.GetAtomWithIdx(i).GetSymbol(), full.GetAtomWithIdx(j).GetSymbol()
            threshold = 1.25 * (COV.get(si, .8) + COV.get(sj, .8))
            if er > threshold:
                bonded_anomalies.append({"atoms": [i, j], "distance_A": er, "threshold_A": threshold})
            bonds.append({"atoms": [i, j], "type": [full.GetAtomWithIdx(i).GetSymbol(),
                                                         full.GetAtomWithIdx(j).GetSymbol(),
                                                         b.GetBondTypeAsDouble()],
                          "embed_A": er, "reference_A": rr, "delta_A": er - rr})
    angles = []
    bend_pivots = set(bend_pivots or ())
    for center in full.GetAtoms():
        i = center.GetIdx()
        nbr = sorted(a.GetIdx() for a in center.GetNeighbors())
        for p in range(len(nbr)):
            for q in range(p + 1, len(nbr)):
                j, k = nbr[p], nbr[q]
                if np.isfinite(reference[[i, j, k]]).all():
                    angles.append({"atoms": [j, i, k], "type": [full.GetAtomWithIdx(j).GetSymbol(),
                                   center.GetSymbol(), full.GetAtomWithIdx(k).GetSymbol()],
                                   "contains_H": any(full.GetAtomWithIdx(a).GetAtomicNum() == 1 for a in (i,j,k)),
                                   "solver_dof": i in bend_pivots,
                                   "embed_deg": angle_deg(emb[j], emb[i], emb[k]),
                                   "reference_deg": angle_deg(reference[j], reference[i], reference[k])})
    donor_flat = sorted({int(i) for g in donor_groups for i in g})
    donor_pairs = []
    for p, i in enumerate(donor_flat):
        for j in donor_flat[p + 1:]:
            if i < len(emb) and j < len(emb) and np.isfinite(reference[[i, j]]).all():
                donor_pairs.append({"atoms": [i, j], "embed_A": float(np.linalg.norm(emb[i]-emb[j])),
                                    "reference_A": float(np.linalg.norm(reference[i]-reference[j]))})
    nonbond = []
    bonded = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in base.GetBonds()}
    for i in range(base.GetNumAtoms()):
        for j in range(i + 1, base.GetNumAtoms()):
            if (i, j) in bonded or not np.isfinite(reference[[i, j]]).all():
                continue
            si, sj = base.GetAtomWithIdx(i).GetSymbol(), base.GetAtomWithIdx(j).GetSymbol()
            radius_sum = COV.get(si, .8) + COV.get(sj, .8)
            d = float(np.linalg.norm(emb[i] - emb[j]))
            if d < .8 * radius_sum:
                nonbond.append({"atoms": [i, j], "type": [si, sj], "distance_A": d,
                                "threshold_A": .8 * radius_sum})
    ring_rows = []
    for ring in base.GetRingInfo().AtomRings():
        ring = list(ring)
        if len(ring) < 4 or not np.isfinite(reference[ring]).all():
            continue
        def ring_shape(coords):
            xx = coords[ring]
            centered = xx - xx.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            height = centered @ vt[-1]
            rms = float(np.sqrt(np.mean(height ** 2)))
            if rms < .10:
                cls = "planar"
            elif len(ring) == 5:
                order = np.argsort(np.abs(height))[::-1]
                cls = "envelope" if abs(height[order[0]]) > 1.5 * max(abs(height[order[1]]), 1e-6) else "twist_or_envelope"
            elif len(ring) == 6:
                cls = "chair_like" if np.count_nonzero(height > 0) in (2, 4) and np.max(np.abs(height)) < 0.8 else "boat_twist_or_halfchair"
            else:
                cls = "puckered"
            return cls, rms
        ecls, erms = ring_shape(emb)
        rcls, rrms = ring_shape(reference)
        ring_torsions = []
        for k in range(len(ring)):
            a, b, c, d = (ring[(k + j) % len(ring)] for j in range(4))
            et, rt = dihedral_deg(emb[a], emb[b], emb[c], emb[d]), dihedral_deg(reference[a], reference[b], reference[c], reference[d])
            delta = (et - rt + 180.0) % 360.0 - 180.0
            ring_torsions.append({"atoms": [a, b, c, d], "embed_deg": et, "reference_deg": rt,
                                  "delta_deg": delta})
        ring_rows.append({"atoms": ring, "size": len(ring), "embed_pucker_class": ecls,
                          "reference_pucker_class": rcls, "pucker_class_match": ecls == rcls,
                          "embed_plane_rms_A": erms, "reference_plane_rms_A": rrms,
                          "plane_rms_delta_A": erms - rrms, "torsions": ring_torsions})
    impropers, planarity = [], []
    for atom in base.GetAtoms():
        i = atom.GetIdx()
        nbr = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(nbr) != 3 or not np.isfinite(reference[[i] + nbr]).all():
            continue
        def height(coords):
            pts = coords[nbr]
            cen = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - cen, full_matrices=False)
            return float(np.dot(coords[i] - cen, vt[-1]))
        eh, rh = height(emb), height(reference)
        mean_bond = max(float(np.mean([np.linalg.norm(emb[i] - emb[j]) for j in nbr])), 1e-8)
        mean_ref_bond = max(float(np.mean([np.linalg.norm(reference[i] - reference[j]) for j in nbr])), 1e-8)
        eang = float(np.degrees(np.arcsin(np.clip(eh / mean_bond, -1, 1))))
        rang = float(np.degrees(np.arcsin(np.clip(rh / mean_ref_bond, -1, 1))))
        impropers.append({"center": i, "atoms": nbr + [i], "embed_height_A": eh,
                          "reference_height_A": rh, "delta_A": eh - rh,
                          "embed_improper_deg": eang, "reference_improper_deg": rang,
                          "improper_delta_deg": eang - rang,
                          "planarity_abs_delta_deg": abs(eang) - abs(rang)})
        if atom.GetHybridization() == Chem.HybridizationType.SP2 or atom.GetIsAromatic():
            planarity.append({"center": i, "atoms": nbr + [i], "embed_abs_deg": abs(eang),
                              "reference_abs_deg": abs(rang),
                              "abs_delta_deg": abs(eang) - abs(rang)})
    def stereo_tags(coords=None):
        m = Chem.AddHs(Chem.Mol(base))
        if coords is not None:
            conf = Chem.Conformer(m.GetNumAtoms())
            for i, xyz in enumerate(coords):
                conf.SetAtomPosition(i, xyz.tolist())
            m.AddConformer(conf, assignId=True)
            Chem.AssignStereochemistryFrom3D(m, confId=0, replaceExistingTags=True)
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
        atoms = {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}
        bonds = {}
        for b in m.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE and b.GetStereo() != Chem.BondStereo.STEREONONE:
                bonds[b.GetIdx()] = str(b.GetStereo())
        return atoms, bonds
    i_atoms, i_bonds = stereo_tags()
    e_atoms, e_bonds = stereo_tags(emb)
    if np.isfinite(reference).all():
        r_atoms, r_bonds = stereo_tags(reference)
    else:
        r_atoms, r_bonds = {}, {}
    stereo = {"input_tetrahedral": i_atoms, "embed_tetrahedral": e_atoms, "reference_tetrahedral": r_atoms,
              "input_ez": i_bonds, "embed_ez": e_bonds, "reference_ez": r_bonds,
              "input_tetrahedral_lost": [i for i in i_atoms if e_atoms.get(i) != i_atoms[i]],
              "input_ez_lost": [i for i in i_bonds if e_bonds.get(i) != i_bonds[i]],
              "tetrahedral_mismatches": [i for i in set(e_atoms) & set(r_atoms) if e_atoms[i] != r_atoms[i]],
              "ez_mismatches": [i for i in set(e_bonds) & set(r_bonds) if e_bonds[i] != r_bonds[i]]}
    return {"bonds": bonds, "angles": angles, "rings": ring_rows, "impropers": impropers,
            "planarity": planarity, "symmetry_mapping": symmetry,
            "stereo": stereo, "bonded_graph_anomalies": bonded_anomalies, "donor_pairs": donor_pairs,
            "nonbonded_clashes": nonbond,
            "n_missing_reference_atoms": int((~np.isfinite(reference).all(axis=1)).sum())}


def measure_parent(code: str, seed: int = 42, ligand_indices=None):
    from jacobik.builder.engine import _smiles_to_conformer
    from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol
    from jacobik.builder.embed import ez_variants

    shard, dataset = inputs.frame_shards()[code]
    frame = json.load(open(inputs.FRAME_V1 / "frame.json"))
    split_by_code = {g["representative_code"]: split for split, groups in
                     frame["main_allocation"]["splits"].items() for g in groups}
    if split_by_code.get(code) in ("motion_validation", "reserve"):
        raise ValueError(f"excluded split {split_by_code[code]} for {code}")
    mol = inputs.load_mol(code, shard, dataset)
    if mol is None:
        return {"code": code, "status": "failed", "error": "reference parent not found"}
    groups = _build_ligand_groups(mol, mol.smiles)
    ligands = ligand_inputs_from_mol(mol, groups)
    if ligands is None:
        return {"code": code, "status": "failed", "error": "ligand input mapping failed"}
    guard = FallbackGuard()
    logger = logging.getLogger("jacobik.builder.engine")
    logger.addHandler(guard)
    rows = []
    try:
        selected = None if ligand_indices is None else set(map(int, ligand_indices))
        for idx, (group, lig) in enumerate(zip(groups, ligands)):
            if selected is not None and idx not in selected:
                continue
            variants = ez_variants(lig.smiles, lig.donors)
            if not variants:
                variants = [[]]
            for variant_idx, ezfix in enumerate(variants):
                start = time.perf_counter()
                try:
                    n_fallbacks = len(guard.events)
                    rd, coords, _ = _smiles_to_conformer(lig.smiles, embed_seed=seed, donors=lig.donors,
                                                          chelate_embed=True, embed="gfn2", ez=ezfix)
                    if len(guard.events) > n_fallbacks:
                        raise RuntimeError(guard.events[-1])
                    ref = _reference_by_rdkit(mol, group, rd)
                    bend_pivots = set()
                    try:
                        from jacobik.ik.angle_bends import bend_specs
                        xyz_to_rdkit = {int(x): int(r) for r, x in group["rdkit_to_xyz"].items()}
                        bend_pivots = {xyz_to_rdkit[int(sp[0])] for sp in bend_specs(mol, mol.ligands[idx])
                                       if int(sp[0]) in xyz_to_rdkit}
                    except Exception:
                        bend_pivots = set()
                    metric = _ligand_metrics(lig.smiles, lig.donors, coords, ref, bend_pivots)
                    rows.append({"ligand_idx": idx, "variant_idx": variant_idx, "n_ez_variants": len(variants),
                                 "ez_fix": ezfix, "smiles": lig.smiles, "donor_groups": lig.donors,
                                 "n_atoms": len(coords), "n_heavy": sum(a.GetAtomicNum() > 1 for a in rd.GetAtoms()),
                                 "ligand_class": ("haptic" if any(len(g) > 1 for g in lig.donors) else
                                                  "multidentate_kappa" if len(lig.donors) > 1 else "monodentate_kappa"),
                                 "donor_elements": [[rd.GetAtomWithIdx(int(j)).GetSymbol() for j in g] for g in lig.donors],
                                 "embed_seconds": time.perf_counter() - start, "status": "ok", **metric})
                except Exception as exc:
                    rows.append({"ligand_idx": idx, "variant_idx": variant_idx, "n_ez_variants": len(variants),
                                 "ez_fix": ezfix, "smiles": lig.smiles, "donor_groups": lig.donors,
                                 "embed_seconds": time.perf_counter() - start, "status": "failed",
                                 "error": f"{type(exc).__name__}: {exc}"})
    finally:
        logger.removeHandler(guard)
    return {"code": code, "split": split_by_code.get(code), "shard": shard, "dataset": dataset,
            "status": "ok" if all(x["status"] == "ok" for x in rows) else "partial",
            "embed_seed": seed, "n_ligands": len(rows), "ligands": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, help="Comma-separated parent codes; no val parents")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ligand-indices", default=None, help="Optional comma-separated local ligand indices")
    a = ap.parse_args()
    codes = [c.strip().upper() for c in a.codes.split(",") if c.strip()]
    frame = json.load(open(inputs.FRAME_V1 / "frame.json"))
    split_by_code = {g["representative_code"]: split for split, groups in
                     frame["main_allocation"]["splits"].items() for g in groups}
    forbidden = {c for c in codes if split_by_code.get(c) in ("motion_validation", "reserve")}
    if forbidden:
        ap.error("excluded val/reserve parents requested: " + ",".join(sorted(forbidden)))
    recs = []
    ligand_indices = None if a.ligand_indices is None else [int(x) for x in a.ligand_indices.split(",") if x]
    for code in codes:
        try:
            recs.append(measure_parent(code, a.seed, ligand_indices))
        except Exception as exc:
            recs.append({"code": code, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("\n".join(json.dumps(x, default=str) for x in recs) + "\n")
    print(json.dumps({"parents": len(recs), "ok": sum(x["status"] == "ok" for x in recs),
                      "partial": sum(x["status"] == "partial" for x in recs),
                      "failed": sum(x["status"] == "failed" for x in recs), "output": str(a.output)}))


if __name__ == "__main__":
    main()
