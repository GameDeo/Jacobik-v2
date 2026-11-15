"""Validate challenge chemistry, explicit binding groups, stereo and export policies."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib

from catalog import ROOT, UnsupportedInput, load_catalog, make_input, read_json, write_json


def validate(chemistry=False, report=False):
    document = read_json(ROOT / "data/challenge/cases.json")
    ligand_document = read_json(ROOT / "data/challenge/ligands.json")
    cases, ligands = load_catalog("challenge")
    sources = {source["id"]: source for source in read_json(ROOT / "data/challenge/sources.json")["sources"]}
    errors, molecules = [], {}
    if len({case["id"] for case in cases}) != len(cases):
        errors.append("Duplicate case IDs")
    if len(ligands) != len(ligand_document["ligands"]):
        errors.append("Duplicate ligand IDs")
    if chemistry:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        from openbabel import openbabel as ob
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        conversion = ob.OBConversion()
        if not conversion.SetInAndOutFormats("smi", "can"):
            raise SystemExit("Open Babel SMILES plugin unavailable; run through pixi run.")
    for key, ligand in ligands.items():
        donors = ligand["donor_indices"]
        grouped = [index for group in ligand["binding_groups"] for index in group["atom_indices"]]
        if not donors or len(set(donors)) != len(donors) or min(donors) < 0:
            errors.append(f"{key}: invalid donor list")
        if sorted(grouped) != sorted(donors) or len(grouped) != len(set(grouped)):
            errors.append(f"{key}: binding groups must partition the coordinating atoms")
        if len(ligand["donor_elements"]) != len(donors):
            errors.append(f"{key}: donor element count mismatch")
        for group in ligand["binding_groups"]:
            if group["kind"] == "kappa" and len(group["atom_indices"]) != 1:
                errors.append(f"{key}: kappa contacts are stored one atom per group")
            elif group["kind"] == "eta" and group.get("hapticity") != len(group["atom_indices"]):
                errors.append(f"{key}: hapticity mismatch")
            elif group["kind"] not in {"kappa", "eta"}:
                errors.append(f"{key}: unknown binding kind")
        if not chemistry:
            continue
        mol = Chem.MolFromSmiles(ligand["smiles"], parser)
        if mol is None:
            errors.append(f"{key}: invalid RDKit SMILES")
            continue
        molecules[key] = mol
        if Chem.GetFormalCharge(mol) != ligand["formal_charge"]:
            errors.append(f"{key}: RDKit formal charge mismatch")
        if rdMolDescriptors.CalcMolFormula(mol) != ligand["formula"]:
            errors.append(f"{key}: formula metadata mismatch")
        if sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()) != 0:
            errors.append(f"{key}: accidental ligand radicals")
        obmol = ob.OBMol()
        if not conversion.ReadString(obmol, ligand["smiles"]):
            errors.append(f"{key}: invalid Open Babel SMILES")
            continue
        if obmol.GetTotalCharge() != ligand["formal_charge"]:
            errors.append(f"{key}: Open Babel formal charge mismatch")
        ob_atoms = [atom.GetAtomicNum() for atom in ob.OBMolAtomIter(obmol)]
        edges = lambda molecule: {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in molecule.GetBonds()}
        ob_edges = {tuple(sorted((bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1))) for bond in ob.OBMolBondIter(obmol)}
        if ob_atoms != [atom.GetAtomicNum() for atom in mol.GetAtoms()] or edges(mol) != ob_edges:
            errors.append(f"{key}: parser atom-order or connectivity mismatch")
        for index, symbol in zip(donors, ligand["donor_elements"]):
            if index >= mol.GetNumAtoms() or mol.GetAtomWithIdx(index).GetSymbol() != symbol:
                errors.append(f"{key}: wrong donor index {index}")
        if ligand["fixed_stereocenters"]:
            roundtrip = Chem.MolFromSmiles(conversion.WriteString(obmol), parser)
            if roundtrip is None or Chem.MolToSmiles(roundtrip, isomericSmiles=True) != Chem.MolToSmiles(mol, isomericSmiles=True):
                errors.append(f"{key}: Open Babel round trip changed fixed ligand stereochemistry")
    for case in cases:
        try:
            expanded = [ligands[entry["ligand_id"]] for entry in case["ligands"] for _ in range(entry["count"])]
        except KeyError as exc:
            errors.append(f"{case['id']}: missing ligand {exc}")
            continue
        if any(not isinstance(entry["count"], int) or entry["count"] <= 0 for entry in case["ligands"]):
            errors.append(f"{case['id']}: invalid ligand count")
        contacts = sum(len(ligand["donor_indices"]) for ligand in expanded)
        groups = [group for ligand in expanded for group in ligand["binding_groups"]]
        has_pi = any(group["kind"] == "eta" for group in groups)
        if case["metal_contact_count"] != contacts or case["binding_site_count"] != len(groups):
            errors.append(f"{case['id']}: contact count or binding-site count mismatch")
        expected_cn = None if has_pi else contacts
        if case["coordination_number"] != expected_cn:
            errors.append(f"{case['id']}: pi contacts must not be labeled as ordinary atomic CN")
        charge = case["oxidation_state"] + sum(ligand["formal_charge"] for ligand in expanded)
        if charge != case["charge"]:
            errors.append(f"{case['id']}: charge does not match formal oxidation/ligand charges")
        if any(source_id not in sources for source_id in case["source_ids"]):
            errors.append(f"{case['id']}: unknown source")
        if not case["failure_modes"] or not case["challenge"]:
            errors.append(f"{case['id']}: missing challenge rationale")
        status = case["architector_support"]["status"]
        if status not in {"direct", "assembly_only", "requires_adapter"}:
            errors.append(f"{case['id']}: unknown support status")
        if status == "requires_adapter" and case["architector_core_cn"] is not None:
            errors.append(f"{case['id']}: unvalidated adapter must not invent an Architector core CN")
        if chemistry:
            pt = Chem.GetPeriodicTable()
            electrons = pt.GetAtomicNumber(case["metal"]) - case["charge"]
            for ligand in expanded:
                electrons += sum(atom.GetAtomicNum() for atom in Chem.AddHs(molecules[ligand["id"]]).GetAtoms())
            if (electrons - case["unpaired_electrons"]) % 2:
                errors.append(f"{case['id']}: electron/spin parity mismatch")
        for final_relax in [True, False]:
            expected_block = ("unsupported_representation" if status == "requires_adapter" else
                              "requires_no_final_relax" if status == "assembly_only" and final_relax else None)
            try:
                specification = make_input(case, ligands, final_relax=final_relax)
            except UnsupportedInput as exc:
                if exc.status != expected_block:
                    errors.append(f"{case['id']}: unexpected export block {exc.status}")
                continue
            if expected_block:
                errors.append(f"{case['id']}: unsupported representation was exported")
                continue
            slots = sum(3 if item.get("ligType") == "sandwich" else len(item["coordList"]) for item in specification["ligands"])
            if slots != case["architector_core_cn"] or specification["core"]["coreCN"] != slots:
                errors.append(f"{case['id']}: Architector slot count would add fill ligands or over-coordinate")
            if specification["parameters"]["xtb_solvent"] != case["environment"]["xtb_solvent"]:
                errors.append(f"{case['id']}: solvent lost during export")
            if chemistry:
                for ligand in expanded:
                    pi = [group for group in ligand["binding_groups"] if group["kind"] == "eta"]
                    if not pi:
                        continue
                    molecule = molecules[ligand["id"]]
                    rings = [set(ring) for ring in molecule.GetRingInfo().AtomRings()]
                    if len(pi) != 1 or set(pi[0]["atom_indices"]) not in rings:
                        errors.append(f"{case['id']}: direct eta group is not exactly one ring")
                    if not all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in pi[0]["atom_indices"]):
                        errors.append(f"{case['id']}: direct sandwich ring is not aromatic")
    manifest = read_json(ROOT / "sources/manifest.json")
    for file in manifest["files"]:
        path = ROOT / "sources" / file["file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != file["sha256"]:
            errors.append(f"Source archive changed: {file['file']}")
    for source in sources.values():
        path = ROOT / "sources" / source["metadata_file"]
        if not path.is_file():
            errors.append(f"Missing source metadata: {source['id']}")
        elif read_json(path)["message"]["DOI"].lower() != source["doi"].lower():
            errors.append(f"Wrong source DOI: {source['id']}")
    reference_manifest = read_json(ROOT / "data/challenge/references.json")
    for file, key in [(ROOT / "data/challenge/ligands.json", "ligands_sha256"),
                      (ROOT / "scripts/prepare_challenge_references.py", "extractor_sha256"),
                      (ROOT / "scripts/prepare_references.py", "shared_cif_extractor_sha256")]:
        if hashlib.sha256(file.read_bytes()).hexdigest() != reference_manifest[key]:
            errors.append(f"Reference inputs changed: {file.name}; regenerate challenge references")
    for reference in reference_manifest["references"]:
        for file_key, hash_key in [("xyz_file", "xyz_sha256"), ("metadata_file", "metadata_sha256")]:
            path = ROOT / "data/challenge" / reference[file_key]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != reference[hash_key]:
                errors.append(f"Changed reference artifact: {reference[file_key]}")
        metadata = read_json(ROOT / "data/challenge" / reference["metadata_file"])
        source_path = ROOT / "sources" / reference["source_file"]
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != metadata["source_sha256"]:
            errors.append(f"Changed reference source: {reference['source_file']}")
        if chemistry:
            from ase.io import read
            import numpy as np
            atoms = read(str(ROOT / "data/challenge" / reference["xyz_file"]))
            case = next(case for case in cases if case["id"] == reference["case_id"])
            expected = Counter({case["metal"]: 1})
            for entry in case["ligands"]:
                for atom in molecules[entry["ligand_id"]].GetAtoms():
                    if atom.GetSymbol() != "H":
                        expected[atom.GetSymbol()] += entry["count"]
            if Counter(atoms.get_chemical_symbols()) != expected or not np.isfinite(atoms.positions).all():
                errors.append(f"{reference['xyz_file']}: formula or coordinate mismatch")
            metadata = read_json(ROOT / "data/challenge" / reference["metadata_file"])
            indices = [i for instance in metadata["ligand_instances"] for i in instance["donor_reference_indices"]]
            if len(indices) != case["metal_contact_count"]:
                errors.append(f"{reference['xyz_file']}: donor count mismatch")
            distances = np.linalg.norm(atoms.positions[indices]-atoms.positions[0], axis=1)
            if not np.allclose(distances, metadata["metal_donor_distances"], atol=1e-7):
                errors.append(f"{reference['xyz_file']}: donor mapping mismatch")
    if errors:
        raise SystemExit("Challenge validation failed:\n" + "\n".join(errors))
    summary = {"dataset_id": document["dataset_id"], "case_count": len(cases), "ligand_count": len(ligands),
               "chemistry_checked": chemistry, "source_count": len(sources),
               "origins": dict(Counter(case["origin"] for case in cases)),
               "adapter_status": dict(Counter(case["architector_support"]["status"] for case in cases)),
               "prepared_coordinate_sets": dict(Counter(ref["kind"] for ref in reference_manifest["references"])),
               "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [ROOT / "data/challenge/cases.json", ROOT / "data/challenge/ligands.json",
                                             ROOT / "data/challenge/sources.json", ROOT / "sources/manifest.json"]},
               "checks": ["binding-group partition", "metal contacts versus template slots", "formal charges",
                          "export policy", "source hashes and DOI metadata"]}
    if chemistry:
        summary["checks"] += ["RDKit/Open Babel atom order and connectivity", "ligand radicals",
                              "fixed stereochemistry round trip", "electron parity", "aromatic pi-group topology"]
    if report:
        write_json(ROOT / "data/challenge/validation_report.json", summary)
    print(f"PASS: {len(cases)} challenge cases, {len(ligands)} ligands, {len(sources)} papers")
    print("Adapter status:", summary["adapter_status"])
    print("Origins:", summary["origins"])
    print("Chemistry:", "checked with RDKit and Open Babel" if chemistry else "not checked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemistry", action="store_true")
    parser.add_argument("--report", action="store_true")
    options = parser.parse_args()
    validate(options.chemistry, options.report)
