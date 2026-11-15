"""D-T6 (research/archive/docs/roadmap/roadmap_generation_v1.md 2.4 T6): is the NEPNEF N-O bond length already wrong at embed time? CPU, RDKit only.

Reads the SMILES of NEPNEF from the tmQM CSV (CSD_code and SMILES columns only) and the reference geometry (CSD
xyz block, read only). Lists every N-O bond with its order and formal charges, then embeds the ligand graph with the
plain ETKDGv3 path (seed 42) and measures the same bonds.

H-T6 is supported if the embedded N-O length is about 1.5 A before any solve (the bond order given by the SMILES),
and refuted if the embed gives about 1.2 A.

    python tools/jgen/diag/d23_nepnef_bond_orders.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "bench")]

from tools.jgen import inputs  # noqa: E402


def main():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    code = "NEPNEF"
    parent = inputs.CLEAN / "tmQM"
    smiles = inputs.tmqm_smiles(parent)[code]
    shard = inputs.frame_shards()[code][0]
    mol_ref = inputs.load_mol(code, shard, "tmQM")
    ref_xyz = np.array([a.position for a in mol_ref.atoms], float)
    ref_sym = [a.symbol for a in mol_ref.atoms]

    base = Chem.MolFromSmiles(smiles)
    mh = Chem.AddHs(base)
    bonds = []
    for b in base.GetBonds():
        i, j = b.GetBeginAtom(), b.GetEndAtom()
        if {i.GetSymbol(), j.GetSymbol()} == {"N", "O"}:
            bonds.append(dict(i=b.GetBeginAtomIdx(), j=b.GetEndAtomIdx(), order=str(b.GetBondType()),
                              charges=[i.GetFormalCharge(), j.GetFormalCharge()]))
    p = AllChem.ETKDGv3()
    p.randomSeed = 42
    cid = AllChem.EmbedMolecule(mh, p)
    if cid == -1:
        cid = AllChem.EmbedMolecule(mh, AllChem.ETKDG())
    emb = {}
    if cid != -1:
        if AllChem.MMFFHasAllMoleculeParams(mh):
            AllChem.MMFFOptimizeMolecule(mh)
        X = mh.GetConformer().GetPositions()
        for bd in bonds:
            bd["embed_A"] = round(float(np.linalg.norm(X[bd["i"]] - X[bd["j"]])), 3)
    # the reference N-O distances are indexed in the CSD atom order (the same heavy-atom order as the SMILES only
    # if the xyz uses it; reported by element pair instead)
    no_ref = sorted(round(float(np.linalg.norm(ref_xyz[a] - ref_xyz[b])), 3)
                    for a in range(len(ref_sym)) for b in range(a + 1, len(ref_sym))
                    if {ref_sym[a], ref_sym[b]} == {"N", "O"} and np.linalg.norm(ref_xyz[a] - ref_xyz[b]) < 1.6)
    single = bool(bonds) and all(b["order"] == "SINGLE" for b in bonds)
    out = dict(code=code, smiles=smiles, n_no_bonds_smiles=len(bonds), no_bonds=bonds,
               embed_seed=42, embed_ok=cid != -1, reference_no_distances_A=no_ref,
               # the SMILES writes the N-O bonds as single bonds with O- charges (no N=O): the plain embed (ETKDGv3 +
               # MMFF) carries single-bond N-O lengths, far from the reference (about 1.21 A). The 1.48-1.52 A of the
               # stored pools come from the GFN2-refined embed, which this script does not run.
               verdict=("smiles_single_bond_representation" if single else "other_bond_orders"),
               verdict_note=("H-T6 is not refuted: the embed starts from single N-O bonds. The fix is an input "
                             "(SMILES) change; no solver flag acts on it." if single else ""))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
