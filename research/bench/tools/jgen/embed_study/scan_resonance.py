"""Scan allowed tmQM SMILES for charge-separated and resonance-group patterns.

Only the CSD_code and SMILES columns are read. No reference coordinates or QM property columns are accessed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "bench")]

from tools.jgen import inputs  # noqa: E402
from tools.jgen.bench.panels import TAIL  # noqa: E402
from jacobik.core.chemistry import METALS  # noqa: E402
from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol  # noqa: E402

SEED = 20261028
MAIN = ROOT.parents[2]
LOCKED_EXCLUSIONS = MAIN / "research/bench/benchmarks/locked_exclusions.json"


def _metal_symbol(complex_smiles):
    from rdkit import Chem

    m = Chem.MolFromSmiles(complex_smiles or "")
    if m is None:
        return None
    found = [a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() in METALS]
    return found[0] if found else None


def motifs(smiles):
    """Return candidate resonance motifs based solely on graph, bond order and formal charge."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], "unparseable"
    out = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        ons = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
        if sym == "N" and len(ons) in (1, 2, 3):
            orders = sorted(float(mol.GetBondBetweenAtoms(atom.GetIdx(), o.GetIdx()).GetBondTypeAsDouble()) for o in ons)
            charges = sorted(int(o.GetFormalCharge()) for o in ons)
            ncharge = int(atom.GetFormalCharge())
            carbon_nbrs = sum(n.GetSymbol() == "C" for n in atom.GetNeighbors())
            if len(ons) == 2:
                if orders == [1.0, 1.0] and charges == [-1, -1] and ncharge == 0:
                    kind = "nitro_malformed_N0_Ominus2_single"
                elif ncharge == 1 and sorted(charges) == [-1, 0] and max(orders) >= 2:
                    kind = "nitro_charge_separated"
                else:
                    kind = "nitro_other"
            elif len(ons) == 3:
                kind = "nitrate_charge_separated" if ncharge == 1 and -1 in charges and max(orders) >= 2 else "nitrate_other"
            elif len(ons) == 1:
                if ncharge == 1 and charges == [-1] and carbon_nbrs >= 2:
                    kind = "N_oxide_charge_separated"
                elif mol.GetNumHeavyAtoms() <= 3 or len(atom.GetNeighbors()) <= 2:
                    kind = "nitrosyl_or_NO_other"
                else:
                    kind = "N_O_other"
            out.append({"kind": kind, "center": atom.GetIdx(), "oxygen_atoms": [o.GetIdx() for o in ons],
                        "N_or_center_charge": ncharge, "O_charges": charges, "bond_orders": orders})
        elif sym in ("C", "P", "S", "Se", "As") and len(ons) >= 2:
            orders = sorted(float(mol.GetBondBetweenAtoms(atom.GetIdx(), o.GetIdx()).GetBondTypeAsDouble()) for o in ons)
            charges = sorted(int(o.GetFormalCharge()) for o in ons)
            if any(charges) or orders.count(1.0) >= 2:
                out.append({"kind": f"other_resonance_{sym}O{len(ons)}", "center": atom.GetIdx(),
                            "oxygen_atoms": [o.GetIdx() for o in ons], "N_or_center_charge": int(atom.GetFormalCharge()),
                            "O_charges": charges, "bond_orders": orders})
    return out, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-size", type=int, default=300)
    args = ap.parse_args()
    frame = json.load(open(inputs.FRAME_V1 / "frame.json"))
    exclusion = json.load(open(LOCKED_EXCLUSIONS))
    locked = set(exclusion["reserve_all_codes"]) | set(exclusion["motion_validation_codes"])
    code_rec = {}
    for split, groups in frame["main_allocation"]["splits"].items():
        for g in groups:
            code = g["representative_code"]
            code_rec[code] = dict(split=split, shard=g["representative_shard"], dataset=g["dataset"])
    cal_codes = {c for c, r in code_rec.items() if r["split"] == "calibration"} - locked
    tail_codes = set(TAIL) - locked - {c for c, r in code_rec.items() if r["split"] == "motion_validation"}
    # `tail` overlaps calibration by design; keep parent codes once.
    selected_codes = cal_codes | {c for c in tail_codes if c in cal_codes}
    surface_codes = [c for c, r in code_rec.items() if r["split"] == "surface_fit" and c not in locked]

    def ligands_for(code):
        rec = code_rec[code]
        mol = inputs.load_mol(code, rec["shard"], rec["dataset"])
        if mol is None or not mol.smiles:
            return []
        groups = _build_ligand_groups(mol, mol.smiles)
        ligands = ligand_inputs_from_mol(mol, groups)
        return [lig.smiles for lig in ligands] if ligands is not None else []

    cohort_ligands = []
    failed_codes = []
    for code in sorted(selected_codes):
        ligs = ligands_for(code)
        if not ligs:
            failed_codes.append(code)
        cohort_ligands.extend((code, i, s, code_rec[code]["split"]) for i, s in enumerate(ligs))

    surface_ligands = []
    for code in sorted(surface_codes):
        ligs = ligands_for(code)
        if not ligs:
            failed_codes.append(code)
        surface_ligands.extend((code, i, s, "surface_fit") for i, s in enumerate(ligs))
    rng = random.Random(SEED)
    if len(surface_ligands) > args.sample_size:
        surface_sample = rng.sample(surface_ligands, args.sample_size)
    else:
        surface_sample = surface_ligands

    def summarize(records):
        class_instances = Counter()
        class_parents = defaultdict(set)
        invalid = 0
        detailed = []
        for code, ligidx, smi, split in records:
            found, status = motifs(smi)
            if status != "ok":
                invalid += 1
                continue
            for m in found:
                class_instances[m["kind"]] += 1
                class_parents[m["kind"]].add(code)
                detailed.append({"code": code, "ligand_idx": ligidx, "split": split, "smiles": smi, **m})
        return {"n_ligand_instances": len(records), "motif_occurrences": dict(class_instances),
                "parents_by_class": {k: len(v) for k, v in class_parents.items()}, "unparseable_ligands": invalid,
                "details": detailed}

    out = {"seed": SEED, "sample_size_requested": args.sample_size,
           "exclusions": {"locked_codes_removed": True, "motion_validation_codes_removed": True,
                          "reserve_codes_removed": True},
           "cal_tail": summarize(cohort_ligands),
           "surface_fit_sample": summarize(surface_sample),
           "surface_fit_sample_manifest": [{"code": c, "ligand_idx": i, "smiles": s}
                                            for c, i, s, _ in surface_sample],
           "parents_failed_to_fragment": sorted(set(failed_codes))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"cal_tail_parents": len(selected_codes), "cal_tail_ligands": len(cohort_ligands),
                      "surface_fit_population_ligands": len(surface_ligands),
                      "surface_fit_sample_ligands": len(surface_sample),
                      "failed_parent_fragmentations": len(set(failed_codes)), "output": str(args.output)}))


if __name__ == "__main__":
    main()
