"""Build F-5's corrected Li-cap footprint and classify donors removed by local charge."""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from rdkit import Chem
from tools.jgen.bench.panels import TAIL


def donor_class(mol, idx):
    a = mol.GetAtomWithIdx(idx)
    z = a.GetSymbol()
    ns = list(a.GetNeighbors())
    # Name the chemically important cases from the audit explicitly.
    if z == "C" and any(n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 3 for n in ns):
        return "CO-like [C-]#[O+]"
    if z == "C" and any(n.GetSymbol() == "N" and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 3 for n in ns):
        return "isocyanide-like [C-]#[N+]"
    if z == "O" and any(n.GetSymbol() == "N" and n.GetFormalCharge() > 0 for n in ns):
        return "O− bonded to N+ (nitro/N-oxide-like)"
    if z == "C" and any(n.GetFormalCharge() > 0 for n in ns):
        return "ylidic/zwitterionic C− adjacent to positive atom"
    if z == "O":
        return "other O−"
    if z == "N":
        return "other N−"
    return f"other {z}− environment"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    census = [json.loads(x) for x in a.census.read_text().splitlines() if x.strip()]
    old_by_parent, new_by_parent = defaultdict(list), defaultdict(list)
    changed = Counter(); changed_ligands = Counter(); removed_detail = []
    for p in census:
        code = p["code"]
        if p.get("cohort") != "cal_tail":
            continue
        unique = {}
        for lig in p.get("ligands", []): unique.setdefault(int(lig["ligand_idx"]), lig)
        for ix, lig in unique.items():
            if lig.get("status") != "ok": continue
            mol = Chem.MolFromSmiles(lig["smiles"])
            if mol is None: continue
            old, new = [], []
            for group in lig.get("donor_groups", []):
                if len(group) != 1: continue
                di = int(group[0]); atom = mol.GetAtomWithIdx(di)
                if atom.GetFormalCharge() >= 0: continue
                old.append(di)
                local_q = int(atom.GetFormalCharge()) + sum(int(n.GetFormalCharge()) for n in atom.GetNeighbors())
                if local_q < 0:
                    new.append(di)
                else:
                    cls = donor_class(mol, di)
                    changed[cls] += 1; changed_ligands[cls] += 1
                    removed_detail.append({"code":code,"ligand_idx":ix,"donor_idx":di,"class":cls,
                                           "smiles":lig["smiles"],"local_charge":local_q})
            if old: old_by_parent[code].append({"ligand_idx":ix,"donors":old})
            if new: new_by_parent[code].append({"ligand_idx":ix,"donors":new})
    oldset, newset = set(old_by_parent), set(new_by_parent)
    removed_parents = oldset - newset
    # A parent is CO-only removed if every old-only candidate donor is classified CO-like.
    old_only_donors = defaultdict(list)
    for d in removed_detail: old_only_donors[d["code"]].append(d)
    co_only = sorted(c for c in removed_parents if old_only_donors[c] and all(d["class"] == "CO-like [C-]#[O+]" for d in old_only_donors[c]))
    out={"source":str(a.census),"old_parent_count":len(oldset),"new_parent_count":len(newset),
         "retained_parents":len(oldset & newset),"removed_parents":sorted(removed_parents),
         "parents_only_because_of_CO_like_donors":co_only,
         "counts_by_changed_donor_class":dict(changed),
         "old_ligand_instances":sum(map(len,old_by_parent.values())),
         "new_ligand_instances":sum(map(len,new_by_parent.values())),
         "new_parent_codes":sorted(newset),"new_ligands_by_parent":dict(new_by_parent),
         "removed_donors":removed_detail,"tail_overlap":sorted(newset & set(TAIL))}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:out[k] for k in ("old_parent_count","new_parent_count","retained_parents","parents_only_because_of_CO_like_donors","counts_by_changed_donor_class","old_ligand_instances","new_ligand_instances","tail_overlap","output") if k!="output"},indent=2))
    print(f"output={a.output}")
if __name__=="__main__": main()
