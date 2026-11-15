"""Source-only D3 gate input audit. No generated structures or arm results."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TYPING_FIELDS = ("HAPTO_BOND_GATE", "HAPTO_LABEL_CANON", "HAPTO_CHELATE_EXCLUSION")
# These are review triggers, never rules for choosing a donor or excluding a case.
NEAR = 1.24
STRONG_BO = 0.25
GEOMETRY_GAP = 0.025
REVIEW_HISTORY = {"NUHPEL", "MIDQEY", "UMUBAF", "NUWFAM"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _smiles(path, wanted):
    """Access CSD_code and SMILES only; the other CSV fields are discarded."""
    with path.open(newline="") as f:
        rows = csv.reader(f, delimiter=";")
        header = next(rows)
        ci, si = header.index("CSD_code"), header.index("SMILES")
        result = {}
        for row in rows:
            if len(row) > max(ci, si) and row[ci].strip() in wanted:
                code = row[ci].strip()
                if code in result:
                    raise ValueError(f"duplicate SMILES for {code}")
                result[code] = row[si].strip() or None
        return result


def freeze(root: Path, panel: Path, out: Path):
    """Index the frozen public panel source blocks once. No chemical analysis."""
    from fit.parser import _iter_xyz_blocks, _iter_bo_blocks, _parse_xyz_comment

    holdout = root / "research/data/jgen/campaign/eta2/holdout"
    draw_files = [holdout / name for name in ("ETA2_HOLDOUT.json", "ETA2_HOLDOUT_D.json")]
    stage1 = _read(panel)["codes"]
    stage2 = [code for path in draw_files for draw in _read(path)["draw"].values() for code in draw]
    if len(stage1) != 120 or len(stage2) != 110 or len(set(stage1 + stage2)) != 230:
        raise ValueError("expected the original disjoint 120/110 gate panels")
    # Public policy file only; never touch auditor custody files.
    ex = _read(root / "research/bench/benchmarks/locked_exclusions.json")
    excluded = {code for rows in ex.values() if isinstance(rows, list) for code in rows}
    if excluded.intersection(stage1 + stage2):
        raise ValueError("frozen panel intersects a protected population")
    frame_path = root / "data/refit/frame_v1/frame.json"
    if hashlib.sha256(frame_path.read_bytes()).hexdigest() != _read(panel)["frame_sha256"]:
        raise ValueError("frozen frame hash differs from stage 1 registration")
    frame = _read(frame_path)["main_allocation"]["splits"]
    wanted = set(stage1 + stage2)
    locations = {}
    for split in frame.values():
        for row in split:
            code = row["representative_code"]
            if code in wanted:
                locations[code] = (row["dataset"], row["representative_shard"])
    # A missing frame location is retained as a failed input audit, never silently dropped.
    shards = defaultdict(set)
    for code, location in locations.items():
        shards[location].add(code)
    source_dir = out / "source_blocks"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_files = {}
    missing = {code: "missing frame location" for code in wanted - locations.keys()}
    ref = root / "data/ref"
    for (dataset, shard), codes in sorted(shards.items()):
        raw_dir = ref / ("lnQM" if dataset == "LNf" else "tmQM")
        xyz = raw_dir / "xyz" / f"tmQM_{shard}.xyz"
        bo = raw_dir / "bo" / f"tmQM_{shard}.BO"
        smiles = _smiles(raw_dir / "tmQM_y.csv", codes)
        by_xyz, by_bo = {}, {}
        for comment, lines in _iter_xyz_blocks(xyz):
            code = _parse_xyz_comment(comment)[0]
            if code in codes:
                if code in by_xyz:
                    raise ValueError(f"duplicate XYZ source for {code}")
                by_xyz[code] = (comment, lines)
        for code, lines in _iter_bo_blocks(bo):
            if code in codes:
                if code in by_bo:
                    raise ValueError(f"duplicate BO source for {code}")
                by_bo[code] = lines
        for code in sorted(codes):
            if code not in by_xyz or code not in by_bo:
                missing[code] = "missing XYZ or BO source block"
                continue
            comment, atoms = by_xyz[code]
            source = {"code": code, "smiles": smiles.get(code), "xyz_comment": comment,
                      "xyz_atoms": atoms, "bo_lines": by_bo[code], "dataset": dataset, "shard": shard}
            path = source_dir / f"{code}.json"
            _write(path, source)
            source_files[code] = {"source_file": str(path.resolve()), "source_sha256": _digest(source)}
    rows = [{"code": code, "stage": 1 if code in stage1 else 2,
             "stage1_primary": code in stage1 and code not in {"FEKKAJ", "NEPNEF"},
             **source_files.get(code, {"source_error": missing.get(code)})} for code in sorted(wanted)]
    manifest = {"schema": 1, "parents": rows,
                "panel_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (panel, *draw_files, frame_path)}}
    _write(out / "manifest.json", manifest)
    return len(rows)


def _source(entry):
    if entry.get("source_error"):
        raise ValueError(entry["source_error"])
    source = _read(entry["source_file"])
    if source["code"] != entry["code"] or _digest(source) != entry["source_sha256"]:
        raise ValueError("audit source identity mismatch")
    return source


def _raw(source):
    from fit.parser import _assemble_raw

    return _assemble_raw(source["xyz_comment"], source["xyz_atoms"], source["code"],
                         source["bo_lines"], SimpleNamespace(smiles=source["smiles"]))


def _typing(source, enabled=None, overlay=None):
    """Run the actual reference loader, recording its mapping, donor and fallback decisions."""
    import numpy as np
    from jacobik.topology import mapping
    from jacobik.stats import keys as key_builder
    from jacobik.topology import donor_classifier

    raw = _raw(source)
    metal = mapping._find_metal(raw.atoms)
    trace = {"mapping_path": "fallback", "smiles_xyz": {}, "h_mapping": {},
             "smi_declared_xyz": [], "loader_declared_xyz": [], "align_error": None}
    original_align = mapping._align_components
    original_h = mapping._assign_hydrogens
    original_smiles_path = mapping._apply_smiles_path

    def align(smi_graph, bo_graph):
        got = original_align(smi_graph, bo_graph)
        trace["smiles_xyz"] = dict(got)
        trace["unmatched_heavy_xyz"] = sorted(set(bo_graph) - set(got.values()))
        return got

    def hydrogens(full_map, mol_h, bo_graph):
        got = original_h(full_map, mol_h, bo_graph)
        trace["h_mapping"] = dict(got)
        trace["h_count_disagreements"] = [xyz for i, xyz in full_map.items()
            if sum(a.GetSymbol() == "H" for a in mol_h.GetAtomWithIdx(i).GetNeighbors()) !=
            sum(bo_graph.nodes[j]["symbol"] == "H" for j in bo_graph.neighbors(xyz))]
        return got

    def smiles_path(atoms, bonds, metal_idx, smiles):
        try:
            bo_updates, donors = original_smiles_path(atoms, bonds, metal_idx, smiles)
        except Exception as exc:
            trace["align_error"] = f"{type(exc).__name__}: {exc}"
            raise
        mol, _ = mapping._smiles_to_rdkit(smiles)
        _, _, declared = mapping._build_smiles_heavy_graph(mol)
        full_map = {**trace["smiles_xyz"], **trace["h_mapping"]}
        trace["smi_declared_xyz"] = sorted(full_map[i] for i in declared if i in full_map)
        trace["unmapped_smi_donors"] = sorted(set(declared) - full_map.keys())
        trace["loader_declared_xyz"] = sorted(donors)
        trace["mapping_path"] = "smiles"
        if overlay is not None:
            # Audit-only proof for proposed donor-set overlays. Does not change production inputs.
            donors = set(overlay)
        return bo_updates, donors

    with ExitStack() as stack:
        if enabled is not None:
            for name in TYPING_FIELDS:
                stack.enter_context(patch.object(donor_classifier, name, enabled))
            stack.enter_context(patch.object(key_builder, "HAPTO_LABEL_CANON", enabled))
        stack.enter_context(patch.object(mapping, "_align_components", align))
        stack.enter_context(patch.object(mapping, "_assign_hydrogens", hydrogens))
        stack.enter_context(patch.object(mapping, "_apply_smiles_path", smiles_path))
        mol = mapping.assemble(raw)
        sites = [{"atoms": sorted(site.donor_indices), "eta": int(site.haptic_order),
                  "label": site.label, "subclass": key_builder._site_subclass(site, mol),
                  "centroid_distance_a": round(float(np.linalg.norm(site.position - metal.position)), 4)}
                 for site in mol.coord_sites]
    return mol, {"sites": sorted(sites, key=lambda s: s["atoms"]), "geometry": mol.coord_geom,
                 "geometry_alt": mol.coord_geom_alt, "geometry_gap": mol.coord_geom_gap,
                 "oxidation_state": mol.oxidation_state, "oxidation_source": mol.oxidation_source,
                 "ligands": [{"atoms": sorted(lig.atom_indices), "donor_atoms": sorted(lig.donor_atoms),
                              "dropped_donors": sorted(lig.dropped_donors)} for lig in mol.ligands],
                 "mapping": trace}


def production_target(entry):
    """tools.run diagnostic task: the target exactly as the bench builds it (``mapping.assemble`` on the frozen source
    block, import-time production flags, no patched hooks). Used to re-check the audited targets after a typing fold-in
    (2026-11-12: `_typing` patches mapping internals that the DS donor_assignment path no longer uses the same way)."""
    from jacobik.core import profile
    from jacobik.topology import mapping
    from jacobik.stats import keys as key_builder

    source = _source(entry)
    mol = mapping.assemble(_raw(source))
    sites = [{"atoms": sorted(s.donor_indices), "eta": int(s.haptic_order), "label": s.label,
              "subclass": key_builder._site_subclass(s, mol)} for s in mol.coord_sites]
    return {"code": source["code"], "stage": entry["stage"], "fingerprint": profile.snapshot()["fingerprint"],
            "sites": sorted(sites, key=lambda s: s["atoms"]), "geometry": mol.coord_geom}


def audit_imported(entry):
    """Pilot parity check using the import-time flags directly."""
    _, typed = _typing(_source(entry))
    return typed


def _molecular_evidence(raw, mol, modes):
    import numpy as np
    from rdkit import Chem
    from fit.sanity import charge_is_plausible
    from jacobik.core.chemistry import covalent_radius as _covalent_radius

    smi = Chem.MolFromSmiles(raw.smiles) if raw.smiles else None
    explicit = Chem.AddHs(smi) if smi is not None else None
    xyz_elements = Counter(a.symbol for a in raw.atoms)
    smi_elements = Counter(a.GetSymbol() for a in explicit.GetAtoms()) if explicit is not None else {}
    smi_charge = Chem.GetFormalCharge(smi) if smi is not None else None
    smi_metals = [a for a in smi.GetAtoms() if a.GetSymbol() == mol.metal_symbol] if smi is not None else []
    formal = smi_metals[0].GetFormalCharge() if len(smi_metals) == 1 else None
    stereo = {"atom_tags": [a.GetIdx() for a in smi.GetAtoms()
                            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED],
              "bond_tags": [b.GetIdx() for b in smi.GetBonds()
                            if b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)]} if smi is not None else None
    by_xyz = {a.idx: a for a in raw.atoms}
    pos = by_xyz[mol.metal_idx].position
    metal_radius = _covalent_radius(mol.metal_symbol)
    raw_neighbors = defaultdict(dict)
    for bond in raw.bonds:
        raw_neighbors[bond.i][bond.j] = bond.order
        raw_neighbors[bond.j][bond.i] = bond.order
    typed_neighbors = defaultdict(dict)
    for bond in mol.bonds:
        if mol.metal_idx not in (bond.i, bond.j) and bond.order >= 0.5:
            typed_neighbors[bond.i][bond.j] = bond.order
            typed_neighbors[bond.j][bond.i] = bond.order
    source_donors = set(modes["production"]["mapping"]["smi_declared_xyz"])
    parsed_donors = {a for mode in modes.values() for s in mode["sites"] for a in s["atoms"]}
    haptic_atoms = {a for mode in modes.values() for s in mode["sites"] if s["eta"] for a in s["atoms"]}
    contacts = {}
    for i, a in sorted(by_xyz.items()):
        if i == mol.metal_idx:
            continue
        d = float(np.linalg.norm(a.position - pos))
        ratio = d / (metal_radius + _covalent_radius(a.symbol))
        bo = raw_neighbors[mol.metal_idx].get(i, 0.0)
        if ratio > 1.50 and bo == 0 and i not in source_donors | parsed_donors:
            continue
        contacts[i] = {"element": a.symbol, "distance_a": round(d, 4), "ratio": round(ratio, 4),
                       "metal_bo": bo, "smiles_declared": i in source_donors,
                       "neighbors": [{"atom": j, "element": by_xyz[j].symbol,
                                       "distance_a": round(float(np.linalg.norm(a.position - by_xyz[j].position)), 4),
                                       "bond_order": order, "raw_bo": raw_neighbors[i].get(j, 0.0)}
                                      for j, order in sorted(typed_neighbors[i].items())],
                       "is_haptic_either_mode": i in haptic_atoms}
    return {"metal": mol.metal_symbol, "metal_xyz": mol.metal_idx, "xyz_elements": dict(xyz_elements),
            "smiles_elements": dict(smi_elements), "xyz_charge": raw.charge, "xyz_spin_multiplicity": raw.spin_state,
            "smiles_total_charge": smi_charge, "smiles_metal_formal_charge": formal,
            "metal_formal_charge_plausible": charge_is_plausible(mol.metal_symbol, formal),
            "specified_stereo": stereo, "contacts": contacts}


def _flags(code, evidence, modes):
    p, d = modes["production"], modes["t1d"]
    flags = defaultdict(set)
    contacts = evidence["contacts"]
    p_atoms = {i for site in p["sites"] for i in site["atoms"]}
    d_atoms = {i for site in d["sites"] for i in site["atoms"]}
    mp = p["mapping"]
    if mp["mapping_path"] != "smiles":
        flags["fallback_mapping"]
    if evidence["xyz_elements"] != evidence["smiles_elements"]:
        flags["composition_disagrees"]
    if evidence["smiles_total_charge"] != evidence["xyz_charge"]:
        flags["total_charge_disagrees"]
    if not evidence["metal_formal_charge_plausible"]:
        flags["implausible_formal_charge"]
    if mp.get("unmatched_heavy_xyz"):
        flags["unmatched_heavy_xyz"].update(mp["unmatched_heavy_xyz"])
    if mp.get("h_count_disagreements"):
        flags["hydrogen_parent_disagreement"].update(mp["h_count_disagreements"])
    if mp.get("unmapped_smi_donors"):
        flags["unmapped_smiles_donor"]
    if mp["smi_declared_xyz"] != mp["loader_declared_xyz"]:
        flags["mapped_donor_remap"].update(set(mp["smi_declared_xyz"]) ^ set(mp["loader_declared_xyz"]))
    dropped = set(mp["smi_declared_xyz"]) - p_atoms
    if dropped:
        flags["smiles_donor_absent_from_production"].update(dropped)
    if p["sites"] != d["sites"] or p["geometry"] != d["geometry"]:
        flags["typing_difference"].update(p_atoms ^ d_atoms)
    if p["geometry_gap"] is not None and p["geometry_gap"] <= GEOMETRY_GAP:
        flags["near_geometry_boundary"]
    if p["geometry"] in (None, "unknown") or not p["sites"]:
        flags["missing_geometry_or_sites"]
    if evidence["specified_stereo"] and any(evidence["specified_stereo"].values()):
        flags["specified_stereo_review"]
    if code in REVIEW_HISTORY:
        flags["previously_reported_input_issue"]
    for i, row in contacts.items():
        ratio, bo, element = row["ratio"], row["metal_bo"], row["element"]
        if i in p_atoms and (ratio > 1.31 or (ratio > NEAR and bo < 0.15)):
            flags["long_typed_contact"].add(i)
        if i not in p_atoms and ratio <= NEAR and bo >= STRONG_BO:
            if element != "H" or bo > max((n["raw_bo"] for n in row["neighbors"]), default=0):
                flags["untyped_short_contact"].add(i)
    for site in p["sites"]:
        if site["eta"] != 2:
            continue
        pair = set(site["atoms"])
        for i in pair:
            row = contacts[i]
            if row["ratio"] > 1.21:
                flags["eta2_past_bond_ratio"].add(i)
            if row["element"] != "C":
                continue
            for n in row["neighbors"]:
                j = n["atom"]
                other = contacts.get(j)
                if j in pair or other is None or j in p_atoms:
                    continue
                if (n["element"] == "C" and len(other["neighbors"]) <= 3 and other["ratio"] <= 1.33
                        and (n["bond_order"] >= 1.25 or n["raw_bo"] >= 1.2)
                        and other["metal_bo"] >= 0.08):
                    flags["eta2_possible_extended_face"].update((i, j))
    return {name: sorted(atoms) for name, atoms in sorted(flags.items())}


def audit_parent(entry):
    """tools.run diagnostic task. Does not embed, solve, or access arm outputs."""
    source = _source(entry)
    p_mol, p = _typing(source, False)
    _, d = _typing(source, True)
    modes = {"production": p, "t1d": d}
    evidence = _molecular_evidence(_raw(source), p_mol, modes)
    return {"code": source["code"], "stage": entry["stage"], "source_sha256": entry["source_sha256"],
            "dataset": source["dataset"], "shard": source["shard"], "smiles": source["smiles"],
            "evidence": evidence, "typing": modes, "review_flags": _flags(source["code"], evidence, modes)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    for name in ("root", "panel", "out"):
        f.add_argument(f"--{name}", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "freeze":
        print(f"Frozen {freeze(args.root, args.panel, args.out)} input-only parents")


if __name__ == "__main__":
    main()
