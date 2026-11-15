"""S7 trace (research/archive/docs/roadmap/roadmap_generation_v1.md S7 step 1) for the parents the ring-closure fix does not reach: for each SMILES
dative donor, where it goes in _apply_smiles_path (heavy isomorphism, H assignment, no-bond proximity remap, H guard,
donor set). Reads SMILES and geometry only (tools/jgen/inputs.py). Writes JSON to stdout.

    python -m tools.jgen diag d34_s7_lost_trace CODE [CODE ...] > out.json
"""
import json
import sys

import networkx as nx


def main():
    from jacobik.topology import mapping as M
    from fit.parser import _assemble_raw, _iter_bo_blocks, _iter_xyz_blocks
    from jacobik.topology.graph import build_heavy_ligand_graph
    from tools.jgen import inputs

    shards = inputs.frame_shards()
    smiles = inputs.tmqm_smiles(inputs.CLEAN / "tmQM")
    out = []
    for code in sys.argv[1:]:
        rec = dict(code=code)
        try:
            shard, ds = shards[code]
            parent = inputs.CLEAN / "tmQM"
            for (comment, atoms), (c, bonds) in zip(_iter_xyz_blocks(parent / "xyz" / f"tmQM_{shard}.xyz"),
                                                    _iter_bo_blocks(parent / "bo" / f"tmQM_{shard}.BO")):
                if c == code:
                    break
            raw = _assemble_raw(comment, atoms, c, bonds, inputs._Row(smiles.get(code)))
            atoms, bonds = raw.atoms, raw.bonds
            smi = raw.smiles
            metal_idx = M._find_metal(atoms).idx
            mol, mol_h = M._smiles_to_rdkit(smi)
            G_smi, _, donor_smi_map = M._build_smiles_heavy_graph(mol)
            G_bo_heavy = build_heavy_ligand_graph(atoms, bonds, metal_idx, bo_threshold=M._LIGAND_BO_THRESHOLD)
            G_bo_full = M._build_bo_full_graph(atoms, bonds, metal_idx)
            rec["n_smiles_donors"] = len(donor_smi_map)
            try:
                heavy = M._align_components(G_smi, G_bo_heavy)
            except ValueError as exc:
                rec["drop"] = f"_align_components raises (whole SMILES path falls back): {str(exc)[:120]}"
                out.append(rec)
                continue
            full = {**heavy, **M._assign_hydrogens(heavy, mol_h, G_bo_full)}
            raw_bo = {(min(b.i, b.j), max(b.i, b.j)): b.order for b in bonds}
            sym = {a.idx: a.symbol for a in atoms}
            donors = []
            taken = set()
            for sd, sm in donor_smi_map.items():
                x = full.get(sd)
                d = dict(smi_idx=sd, metal_pair=sm, xyz=x, symbol_smiles_atom=mol.GetAtomWithIdx(sd).GetSymbol())
                if x is None:
                    d["drop"] = "not in full_map (no heavy/H match)"
                    donors.append(d)
                    continue
                bo = raw_bo.get((min(metal_idx, x), max(metal_idx, x)), 0.0)
                d["mapped_bo"] = bo
                best = x
                if bo < 0.15:  # local constant _COORD_BO_SUPPLEMENT in _apply_smiles_path
                    comp = nx.node_connected_component(G_bo_heavy, x) if x in G_bo_heavy else {x}
                    bb = bo
                    for cand in comp:
                        cb = raw_bo.get((min(metal_idx, cand), max(metal_idx, cand)), 0.0)
                        if cb > bb:
                            bb, best = cb, cand
                d["remap_atom"] = best
                d["remap_symbol"] = sym.get(best)
                d["remap_bo"] = raw_bo.get((min(metal_idx, best), max(metal_idx, best)), 0.0)
                if sym.get(best) == "H":
                    d["drop"] = "remap target is H (H guard)"
                elif best in taken:
                    d["drop"] = "remap target already a donor (set collision)"
                else:
                    taken.add(best)
                    d["kept"] = best
                donors.append(d)
            rec["donors"] = donors
            rec["n_kept"] = len(taken)
        except Exception as exc:  # recorded, never raised
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        out.append(rec)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
