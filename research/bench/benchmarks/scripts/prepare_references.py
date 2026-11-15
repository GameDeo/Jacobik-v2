"""Extract and graph-map experimental heavy-atom complexes from the archived CIFs."""
from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic
from ase.io.cif import CIFBlock, parse_cif
from ase.neighborlist import neighbor_list
from rdkit import Chem

from catalog import ROOT, load_catalog, read_json, write_json


def topology(atomic_nums, edges):
    mol = Chem.RWMol()
    for number in atomic_nums:
        mol.AddAtom(Chem.Atom(int(number)))
    for i, j in sorted({tuple(sorted(edge)) for edge in edges}):
        mol.AddBond(i, j, Chem.BondType.SINGLE)
    mol.UpdatePropertyCache(strict=False)
    return mol.GetMol()


def extract(case, source, ligands):
    blocks = list(parse_cif(str(ROOT / "sources" / source["cif_file"])))
    block = next(block for block in blocks if block.name == source["block"])
    # ASE 3.23 rejects these expanded monoclinic names when the IT number is
    # absent. Normalize only the names; use the explicit operations in the CIF.
    tags = dict(block.items())
    normalization = []
    aliases = {"C 1 2/c 1": "C 2/c", "P 1 21/c 1": "P 21/c"}
    symbol = tags.get("_space_group_name_h-m_alt")
    if symbol in aliases:
        tags["_space_group_name_h-m_alt"] = aliases[symbol]
        normalization.append(f"ASE space-group name: {symbol} -> {aliases[symbol]}")
    if block._get_sitesym():
        # These redundant crystal-system hints trigger misleading ASE warnings
        # for non-rhombohedral groups; supplied symmetry operations are retained.
        for key in ("_space_group_crystal_system", "_symmetry_cell_setting"):
            tags.pop(key, None)
    block = CIFBlock(block.name, tags)
    atoms = block.get_atoms(store_tags=True)
    center_index = next(i for i, atom in enumerate(atoms) if atom.symbol == case["metal"])
    center = atoms.positions[center_index]
    symbols = {"B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
    keep = [i for i, atom in enumerate(atoms) if atom.symbol in symbols]
    heavy = atoms[keep]
    cell = np.asarray(heavy.cell)
    vectors, distances = find_mic(heavy.positions - center, cell, pbc=True)
    radius = covalent_radii[atomic_numbers[case["metal"]]]
    donors = {i for i, atom in enumerate(heavy)
              if atom.symbol in {"N", "O", "P", "S", "F", "Cl", "Br", "I"}
              and distances[i] < 1.30 * (radius + covalent_radii[atom.number])}
    if source.get("metal_donor_labels"):
        # Crowded high-CN structures can have nearby secondary-sphere O atoms
        # inside the generous radius screen. Use the source's published bond
        # table to nominate donor labels, then still validate all distances.
        labels = block["_atom_site_label"]
        source_kinds = atoms.arrays["spacegroup_kinds"]
        allowed = set(source["metal_donor_labels"])
        donors = {i for i in donors if labels[source_kinds[keep[i]]] in allowed}
    if len(donors) != case["coordination_number"]:
        raise ValueError(f"Found {len(donors)} metal neighbours, expected {case['coordination_number']}")
    adj = {i: [] for i in range(len(heavy))}
    first, second, shifts = neighbor_list("ijS", heavy, 1.20 * covalent_radii[heavy.numbers])
    for i, j, shift in zip(first, second, shifts):
        adj[int(i)].append((int(j), np.asarray(shift, dtype=int)))
    components = []
    visited = set()
    for donor in sorted(donors):
        if donor in visited:
            continue
        initial = np.rint((vectors[donor] - (heavy.positions[donor] - center)) @ np.linalg.inv(cell)).astype(int)
        offsets = {donor: initial}
        queue = deque([donor])
        while queue:
            i = queue.popleft()
            for j, shift in adj[i]:
                candidate = offsets[i] + shift
                if j in offsets:
                    if not np.array_equal(offsets[j], candidate):
                        raise ValueError("Periodic covalent component; not an isolated ligand")
                else:
                    offsets[j] = candidate
                    queue.append(j)
        indices = sorted(offsets)
        visited.update(indices)
        local = {index: i for i, index in enumerate(indices)}
        edges = [(local[i], local[j]) for i in indices for j, _ in adj[i] if j in local and i < j]
        graph = topology(heavy.numbers[indices], edges)
        components.append({"indices": indices, "offsets": offsets, "graph": graph,
                           "donors": donors.intersection(indices)})
    source_labels = block["_atom_site_label"]
    occupancies = block.get("_atom_site_occupancy", [1.0] * len(source_labels))
    kinds = atoms.arrays["spacegroup_kinds"]
    selected_symbols = [case["metal"]]
    positions = [[0.0, 0.0, 0.0]]
    mapping = [{"reference_index": 0, "element": case["metal"],
                "cif_label": source_labels[kinds[center_index]], "cell_shift": [0, 0, 0]}]
    instances = []
    for entry in case["ligands"]:
        ligand = ligands[entry["ligand_id"]]
        mol = Chem.MolFromSmiles(ligand["smiles"])
        query = topology([atom.GetAtomicNum() for atom in mol.GetAtoms()],
                         [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()])
        for _ in range(entry["count"]):
            chosen = None
            for index, component in enumerate(components):
                if (query.GetNumAtoms() != component["graph"].GetNumAtoms()
                        or query.GetNumBonds() != component["graph"].GetNumBonds()):
                    continue
                for match in component["graph"].GetSubstructMatches(query, uniquify=False, maxMatches=100000):
                    mapped_donors = {component["indices"][match[i]] for i in ligand["donor_indices"]}
                    if mapped_donors == component["donors"]:
                        chosen = (index, component, match)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                raise ValueError(f"No full graph and donor match for ligand {ligand['id']}")
            index, component, match = chosen
            components.pop(index)
            start = len(positions)
            for ligand_atom, component_atom in enumerate(match):
                heavy_index = component["indices"][component_atom]
                original_index = keep[heavy_index]
                kind = kinds[original_index]
                if float(occupancies[kind]) != 1.0:
                    raise ValueError(f"Disordered selected site: {source_labels[kind]}")
                offset = component["offsets"][heavy_index]
                coordinate = heavy.positions[heavy_index] + offset @ cell - center
                positions.append(coordinate.tolist())
                selected_symbols.append(heavy[heavy_index].symbol)
                mapping.append({"reference_index": len(positions)-1, "element": heavy[heavy_index].symbol,
                                "ligand_atom_index": ligand_atom, "cif_label": source_labels[kind],
                                "cell_shift": offset.tolist()})
            instances.append({"ligand_id": ligand["id"], "reference_indices": list(range(start, len(positions))),
                              "donor_reference_indices": [start+i for i in ligand["donor_indices"]]})
    if components:
        raise ValueError("Unassigned metal-bound components remain")
    metal_label = source_labels[kinds[center_index]]
    published = {}
    for label1, label2, distance in zip(block.get("_geom_bond_atom_site_label_1", []),
                                       block.get("_geom_bond_atom_site_label_2", []),
                                       block.get("_geom_bond_distance", [])):
        if label1 == metal_label:
            published.setdefault(label2, []).append(float(distance))
        elif label2 == metal_label:
            published.setdefault(label1, []).append(float(distance))
    checked = 0
    for instance in instances:
        for index in instance["donor_reference_indices"]:
            label = mapping[index]["cif_label"]
            distance = float(np.linalg.norm(positions[index]))
            candidates = published.get(label, [])
            if not candidates or min(abs(distance - value) for value in candidates) > 0.005:
                raise ValueError(f"Metal-donor distance disagrees with CIF geometry table: {label} {distance:.5f}")
            checked += 1
    result = {"case_id": case["id"], "source_id": source["id"], "datablock": source["block"],
              "source_sha256": hashlib.sha256((ROOT / "sources" / source["cif_file"]).read_bytes()).hexdigest(),
              "reader_normalizations": normalization,
              "donor_label_filter": source.get("metal_donor_labels"),
              "units": "angstrom", "hydrogens": "omitted", "metal_reference_index": 0,
              "atom_mapping": mapping, "ligand_instances": instances,
              "metal_donor_distances": [float(np.linalg.norm(positions[i])) for lig in instances for i in lig["donor_reference_indices"]],
              "published_bond_distance_checks": checked,
              "validation": "complete ligand connectivity and donor sets matched to catalog; selected sites have occupancy 1",
              "limitations": "Bond order, proton positions, spin, and absolute stereochemical labels are not independently assigned."}
    return selected_symbols, positions, result


def main():
    cases, ligands = load_catalog()
    lookup = {case["id"]: case for case in cases}
    output = ROOT / "data/references"
    output.mkdir(exist_ok=True)
    manifest = {"schema_version": 1, "method": "Covalent radii + periodic unwrapping + full ligand graph/donor matching",
                "ligand_distance_factor": 1.20, "metal_distance_factor": 1.30,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "references": []}
    for source in read_json(ROOT / "data/sources.json")["sources"]:
        case = lookup[source["id"]]
        try:
            symbols, positions, metadata = extract(case, source, ligands)
        except Exception as exc:
            manifest["references"].append({"case_id": case["id"], "status": "requires_manual_review", "reason": str(exc)})
            print(f"{case['id']}: REVIEW: {exc}")
            continue
        xyz = output / f"{case['id']}.xyz"
        lines = [str(len(symbols)), f"{case['name']}; heavy atoms; {source['doi']}; crystal datablock {source['block']}"]
        lines.extend(f"{symbol:2s} {x: .9f} {y: .9f} {z: .9f}" for symbol, (x,y,z) in zip(symbols, positions))
        xyz.write_text("\n".join(lines)+"\n", encoding="utf-8")
        metadata_path = output / f"{case['id']}.json"
        write_json(metadata_path, metadata)
        manifest["references"].append({"case_id": case["id"], "status": "heavy_atom_reference_prepared",
                                        "xyz_file": str(xyz.relative_to(ROOT / "data")).replace("\\", "/"),
                                        "metadata_file": str(metadata_path.relative_to(ROOT / "data")).replace("\\", "/"),
                                        "xyz_sha256": hashlib.sha256(xyz.read_bytes()).hexdigest(),
                                        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                                        "heavy_atom_count": len(symbols)})
        print(f"{case['id']}: {len(symbols)} heavy atoms; CN={len(metadata['metal_donor_distances'])}; graph + donors matched")
    write_json(ROOT / "data/references.json", manifest)
    return 1 if any(ref["status"] != "heavy_atom_reference_prepared" for ref in manifest["references"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
