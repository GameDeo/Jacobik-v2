"""Prepare the archived experimental HOPO and published computed bite-angle sets."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from ase.data import covalent_radii
from ase.io import read
from ase.neighborlist import neighbor_list
from rdkit import Chem

from catalog import ROOT, load_catalog, read_json, write_json
from prepare_references import extract, topology


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_xyz(path, symbols, positions, comment):
    lines = [str(len(symbols)), comment]
    lines += [f"{symbol:2s} {x: .9f} {y: .9f} {z: .9f}" for symbol, (x,y,z) in zip(symbols, positions)]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def ordered_computed_reference(atoms, case, ligands):
    heavy = atoms[[i for i, atom in enumerate(atoms) if atom.symbol != "H"]]
    metal_index = next(i for i, atom in enumerate(heavy) if atom.symbol == case["metal"])
    other = heavy[[i for i in range(len(heavy)) if i != metal_index]]
    first, second = neighbor_list("ij", other, 1.20 * covalent_radii[other.numbers])
    graph = topology(other.numbers, [(int(i),int(j)) for i,j in zip(first, second) if i < j])
    numbers, bonds, instances = [], [], []
    for entry in case["ligands"]:
        ligand = ligands[entry["ligand_id"]]
        mol = Chem.MolFromSmiles(ligand["smiles"])
        for _ in range(entry["count"]):
            offset = len(numbers)
            numbers += [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            bonds += [(offset+bond.GetBeginAtomIdx(), offset+bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
            instances.append({"ligand_id": ligand["id"], "reference_indices": list(range(offset+1, len(numbers)+1)),
                              "donor_reference_indices": [offset+1+i for i in ligand["donor_indices"]]})
    query = topology(numbers, bonds)
    if query.GetNumAtoms() != graph.GetNumAtoms() or query.GetNumBonds() != graph.GetNumBonds():
        raise ValueError("Published XYZ does not have the expected complete ligand graph")
    match = graph.GetSubstructMatch(query)
    if not match:
        raise ValueError("Published XYZ ligand connectivity differs from authored input")
    positions = np.vstack([heavy.positions[metal_index], other.positions[list(match)]]) - heavy.positions[metal_index]
    symbols = [case["metal"]] + [other[i].symbol for i in match]
    donor_indices = [i for instance in instances for i in instance["donor_reference_indices"]]
    phosphorus = [i for i in donor_indices if symbols[i] == "P"]
    v1, v2 = positions[phosphorus]
    angle = float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2), -1, 1))))
    return symbols, positions, {"ligand_instances": instances, "P_Pd_P_angle_deg": angle,
                                "source_nonmetal_heavy_indices_in_input_order": list(match),
                                "metal_donor_distances": [float(np.linalg.norm(positions[i])) for i in donor_indices],
                                "validation": "all ligand atoms, edges and donor identities matched"}


def main():
    cases, ligands = load_catalog("challenge")
    lookup = {case["id"]: case for case in cases}
    directory = ROOT / "data/challenge"
    output = directory / "references"
    output.mkdir(exist_ok=True)
    manifest = {"schema_version": 2, "ligands_sha256": digest(directory / "ligands.json"),
                "extractor_sha256": digest(Path(__file__)),
                "shared_cif_extractor_sha256": digest(ROOT / "scripts/prepare_references.py"), "references": []}
    source = {"id": "HOPO", "cif_file": "challenge/HOPO-bc5b00572_si_002.cif", "block": "15lcf1h",
              "metal_donor_labels": [f"O{i}" for i in range(1,9)]}
    symbols, positions, metadata = extract(lookup["D03"], source, ligands)
    metadata.update({"reference_kind": "experimental_crystal_heavy_atoms",
                     "donor_filter_basis": "O1-O8 are the eight Zr-O bonds in the published CIF geometry table"})
    xyz_path, metadata_path = output / "D03.xyz", output / "D03.json"
    write_xyz(xyz_path, symbols, positions, "Zr(3,4,3-LI(1,2-HOPO)); experimental heavy atoms; 10.1021/acs.bioconjchem.5b00572")
    write_json(metadata_path, metadata)
    manifest["references"].append({"case_id": "D03", "kind": "experimental_crystal_heavy_atoms", "xyz_file": "references/D03.xyz",
                                    "source_file": source["cif_file"],
                                    "metadata_file": "references/D03.json", "xyz_sha256": digest(xyz_path),
                                    "metadata_sha256": digest(metadata_path)})
    print(f"D03: experimental reference, {len(symbols)} heavy atoms, 8 published Zr-O bonds matched")
    raw = ROOT / "sources/challenge/XLXANT-om9b00141_si_002.xyz"
    lines = [line for line in raw.read_text().splitlines() if line.strip()]
    comments, cursor = [], 0
    while cursor < len(lines):
        count = int(lines[cursor])
        comments.append(lines[cursor+1])
        cursor += count+2
    frames = read(str(raw), index=":", format="xyz")
    if len(frames) != len(comments):
        raise ValueError("Published XYZ frame/comment count mismatch")
    frame_index = []
    for index, (atoms, comment) in enumerate(zip(frames, comments)):
        carbon_count = sum(atom.symbol == "C" for atom in atoms)
        case_id = "B03" if carbon_count == 39 else "B05" if carbon_count == 51 else None
        frame_index.append({"frame_index": index, "comment": comment, "formula": atoms.get_chemical_formula(),
                            "case_id": case_id, "kind": "published_computed_geometry"})
        if case_id is None:
            continue
        symbols, positions, metadata = ordered_computed_reference(atoms, lookup[case_id], ligands)
        metadata.update({"case_id": case_id, "reference_kind": "published_computed_geometry", "source_id": "XLXANT",
                         "source_sha256": digest(raw), "source_frame_index": index, "source_comment": comment,
                         "stationarity_or_frequencies_independently_verified": False, "population_reference": False})
        stem = f"{case_id}_frame_{index:02d}"
        xyz_path, metadata_path = output / f"{stem}.xyz", output / f"{stem}.json"
        write_xyz(xyz_path, symbols, positions, comment+"; computed, not experimental; 10.1021/acs.organomet.9b00141")
        write_json(metadata_path, metadata)
        manifest["references"].append({"case_id": case_id, "kind": "published_computed_geometry", "frame_index": index,
                                        "source_file": str(raw.relative_to(ROOT / "sources")).replace("\\", "/"),
                                        "xyz_file": f"references/{stem}.xyz", "metadata_file": f"references/{stem}.json",
                                        "xyz_sha256": digest(xyz_path), "metadata_sha256": digest(metadata_path)})
        print(f"{case_id} frame {index}: graph matched; P-Pd-P {metadata['P_Pd_P_angle_deg']:.1f} deg; {comment}")
    write_json(directory / "computed_source_frames.json", {"source_file": str(raw.relative_to(ROOT / "sources")).replace("\\", "/"),
                                                           "sha256": digest(raw), "frames": frame_index})
    write_json(directory / "references.json", manifest)
    document = read_json(directory / "cases.json")
    for case in document["cases"]:
        case["reference_status"] = ("experimental_heavy_atom_reference_prepared" if case["id"] == "D03" else
                                    "published_computed_geometry_set_prepared" if case["id"] in {"B03", "B05"} else
                                    "no_coordinate_reference_archived")
    write_json(directory / "cases.json", document)


if __name__ == "__main__":
    main()
