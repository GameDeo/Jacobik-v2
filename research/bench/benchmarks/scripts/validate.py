"""Check the real catalog: stoichiometry, donors, parser agreement, and source hashes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib

from catalog import ROOT, load_catalog, make_input, read_json


def validate(chemistry=False):
    cases, ligands = load_catalog()
    errors = []
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        errors.append("Duplicate case IDs")
    sources = {entry["id"]: entry for entry in read_json(ROOT / "data/sources.json")["sources"]}
    molecules = {}
    ob = None
    if chemistry:
        from rdkit import Chem
        from openbabel import openbabel as ob

    for key, ligand in ligands.items():
        donors = ligand["donor_indices"]
        if not donors or len(set(donors)) != len(donors) or min(donors) < 0:
            errors.append(f"{key}: empty, negative, or duplicate donor indices")
        if len(donors) != len(ligand["donor_elements"]):
            errors.append(f"{key}: donor element count mismatch")
        if chemistry:
            mol = Chem.MolFromSmiles(ligand["smiles"])
            if mol is None:
                errors.append(f"{key}: RDKit rejected SMILES")
                continue
            molecules[key] = mol
            if Chem.GetFormalCharge(mol) != ligand["formal_charge"]:
                errors.append(f"{key}: RDKit formal charge mismatch")
            conversion = ob.OBConversion()
            conversion.SetInFormat("smi")
            obmol = ob.OBMol()
            if not conversion.ReadString(obmol, ligand["smiles"]):
                errors.append(f"{key}: Open Babel rejected SMILES")
                continue
            if obmol.GetTotalCharge() != ligand["formal_charge"]:
                errors.append(f"{key}: Open Babel formal charge mismatch")
            # Architector indexes Open Babel atoms; compare complete heavy-atom order and bonds.
            rd_atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
            ob_atoms = [a.GetAtomicNum() for a in ob.OBMolAtomIter(obmol)]
            rd_edges = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
            ob_edges = {tuple(sorted((b.GetBeginAtomIdx()-1, b.GetEndAtomIdx()-1))) for b in ob.OBMolBondIter(obmol)}
            if rd_atoms != ob_atoms or rd_edges != ob_edges:
                errors.append(f"{key}: RDKit/Open Babel atom order or graph mismatch")
            for index, element in zip(donors, ligand["donor_elements"]):
                if index >= mol.GetNumAtoms() or mol.GetAtomWithIdx(index).GetSymbol() != element:
                    errors.append(f"{key}: incorrect donor {index} ({element})")
    for case in cases:
        try:
            entries = [(ligands[e["ligand_id"]], e["count"]) for e in case["ligands"]]
        except KeyError as exc:
            errors.append(f"{case['id']}: unknown ligand {exc}")
            continue
        if any(not isinstance(count, int) or count <= 0 for _, count in entries):
            errors.append(f"{case['id']}: invalid ligand count")
        cn = sum(len(lig["donor_indices"]) * count for lig, count in entries)
        charge = case["oxidation_state"] + sum(lig["formal_charge"] * count for lig, count in entries)
        if cn != case["coordination_number"]:
            errors.append(f"{case['id']}: donor count {cn} != CN {case['coordination_number']}")
        if charge != case["charge"]:
            errors.append(f"{case['id']}: formal charge sum {charge} != {case['charge']}")
        if not isinstance(case["unpaired_electrons"], int) or case["unpaired_electrons"] < 0:
            errors.append(f"{case['id']}: invalid unpaired electron count")
        for source_id in case["source_ids"]:
            if source_id not in sources:
                errors.append(f"{case['id']}: unknown source {source_id}")
        if case["origin"] == "literature" and not case["source_ids"]:
            errors.append(f"{case['id']}: literature case has no source")
        if chemistry and all(lig["id"] in molecules for lig, _ in entries):
            # Include implicit ligand hydrogen electrons, then remove the net complex charge.
            pt = Chem.GetPeriodicTable()
            electrons = pt.GetAtomicNumber(case["metal"])
            for ligand, count in entries:
                full = Chem.AddHs(molecules[ligand["id"]])
                electrons += count * sum(a.GetAtomicNum() for a in full.GetAtoms())
            electrons -= case["charge"]
            if (electrons - case["unpaired_electrons"]) % 2:
                errors.append(f"{case['id']}: total-electron/spin parity mismatch")
        actual_input = make_input(case, ligands)
        if sum(len(lig["coordList"]) for lig in actual_input["ligands"]) != cn:
            errors.append(f"{case['id']}: expanded input donor count mismatch")
    manifest = read_json(ROOT / "sources/manifest.json")
    for source in manifest["files"]:
        path = ROOT / "sources" / source["file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            errors.append(f"Source missing or changed: {source['file']}")
    for source in sources.values():
        path = ROOT / "sources" / source["cif_file"]
        if not path.is_file():
            errors.append(f"Missing literature CIF: {source['id']}")
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if "_atom_site_fract_x" not in content or source["doi"].lower() not in content.lower():
            errors.append(f"{source['id']}: CIF coordinates or expected paper DOI absent")
    references = read_json(ROOT / "data/references.json")
    prepared = {ref["case_id"]: ref for ref in references["references"]}
    script_hash = hashlib.sha256((ROOT / "scripts/prepare_references.py").read_bytes()).hexdigest()
    if references["script_sha256"] != script_hash:
        errors.append("Reference extraction script changed; regenerate reference artifacts")
    for case in cases:
        if case["origin"] != "literature":
            continue
        ref = prepared.get(case["id"], {})
        if ref.get("status") != case["reference_status"] or ref.get("status") != "heavy_atom_reference_prepared":
            errors.append(f"{case['id']}: reference is not prepared")
            continue
        for file_key, hash_key in [("xyz_file", "xyz_sha256"), ("metadata_file", "metadata_sha256")]:
            path = ROOT / "data" / ref[file_key]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ref[hash_key]:
                errors.append(f"{case['id']}: changed reference {file_key}")
        metadata = read_json(ROOT / "data" / ref["metadata_file"])
        source_path = ROOT / "sources" / sources[case["source_ids"][0]]["cif_file"]
        if metadata["source_sha256"] != hashlib.sha256(source_path.read_bytes()).hexdigest():
            errors.append(f"{case['id']}: source CIF changed since extraction")
        if len(metadata["metal_donor_distances"]) != case["coordination_number"]:
            errors.append(f"{case['id']}: reference CN mismatch")
        if metadata.get("published_bond_distance_checks") != case["coordination_number"]:
            errors.append(f"{case['id']}: donor bonds not checked against published CIF distances")
        if chemistry:
            from ase.io import read
            import numpy as np
            atoms = read(str(ROOT / "data" / ref["xyz_file"]))
            expected = Counter({case["metal"]: 1})
            for entry in case["ligands"]:
                for atom in molecules[entry["ligand_id"]].GetAtoms():
                    expected[atom.GetSymbol()] += entry["count"]
            if Counter(atoms.get_chemical_symbols()) != expected:
                errors.append(f"{case['id']}: reference heavy-atom formula mismatch")
            if not np.isfinite(atoms.positions).all():
                errors.append(f"{case['id']}: nonfinite reference coordinates")
            donor_indices = [i for lig in metadata["ligand_instances"] for i in lig["donor_reference_indices"]]
            actual_distances = np.linalg.norm(atoms.positions[donor_indices] - atoms.positions[0], axis=1)
            if not np.allclose(actual_distances, metadata["metal_donor_distances"], atol=1e-7):
                errors.append(f"{case['id']}: reference donor coordinates disagree with metadata")
    if errors:
        raise SystemExit("Validation failed:\n" + "\n".join(errors))
    print(f"PASS: {len(cases)} cases; {len(ligands)} ligands; {len(sources)} crystallographic sources")
    print("Origins:", dict(sorted(Counter(c["origin"] for c in cases).items())))
    print("Coordination numbers:", dict(sorted(Counter(c["coordination_number"] for c in cases).items())))
    print("Metals:", ", ".join(sorted({c["metal"] for c in cases})))
    print("Chemistry:", "RDKit + Open Babel graphs, donors, charges, electron parity" if chemistry else "not checked; use --chemistry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemistry", action="store_true")
    parser.add_argument("--include-challenge", action="store_true")
    options = parser.parse_args()
    validate(options.chemistry)
    if options.include_challenge:
        from validate_challenge import validate as validate_challenge
        validate_challenge(options.chemistry)
