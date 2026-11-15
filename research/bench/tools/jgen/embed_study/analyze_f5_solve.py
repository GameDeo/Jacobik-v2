"""Analyze paired F-5 solves, averaging three seeds within each parent."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from rdkit import Chem

SEEDS = (11, 12, 13)

def load(path):
    return {str(r["code"]): r for r in json.loads(path.read_text())["codes"]}

def get(row, seed):
    x = row.get("seeds", {}).get(str(seed))
    return x if x and x.get("status") == "ok" else None

def ci(values, rng, nboot):
    values = np.asarray(values, dtype=float)
    if not len(values): return None
    ix = rng.integers(0, len(values), size=(nboot, len(values)))
    return [float(x) for x in np.quantile(values[ix].mean(axis=1), [0.025, 0.975])]

def top1_capped_donors(row, seed_record):
    """Intersect the selected top-1 assignment with donors that receive Li under F-5."""
    assigned = seed_record.get("top1_assignment", [])
    assigned_ligands = {int(x["ligand_idx"]) for x in assigned}
    assignment_by_ligand = {}
    for item in assigned:
        assignment_by_ligand.setdefault(int(item["ligand_idx"]), []).append(item)
    used = []
    for donor in row.get("anionic_kappa_donors", []):
        li = int(donor["ligand_idx"])
        if li not in assigned_ligands:
            continue
        mol = Chem.MolFromSmiles(donor["smiles"])
        if mol is None:
            continue
        atom = mol.GetAtomWithIdx(int(donor["atom_idx"]))
        local_charge = int(atom.GetFormalCharge()) + sum(int(n.GetFormalCharge()) for n in atom.GetNeighbors())
        if local_charge < 0:
            used.append({"ligand_idx": li, "atom_idx_local_smiles": int(donor["atom_idx"]),
                         "formal_charge": int(atom.GetFormalCharge()), "local_charge": local_charge,
                         "smiles": donor["smiles"],
                         "top1_assigned_groups": assignment_by_ligand[li]})
    return used

def summarize(control, arm, nboot=10000, boot_seed=20261029):
    common = sorted(set(control) & set(arm))
    pairs = {}
    missing = {}
    for code in common:
        seed_pairs = {}
        for seed in SEEDS:
            c, a = get(control[code], seed), get(arm[code], seed)
            if c is not None and a is not None: seed_pairs[seed] = (c, a)
        if len(seed_pairs) == len(SEEDS): pairs[code] = seed_pairs
        else: missing[code] = {"paired_seeds": sorted(seed_pairs), "missing": sorted(set(SEEDS)-set(seed_pairs))}
    rng = np.random.default_rng(boot_seed)
    metrics = ("top1_dE_unrel", "floor_dE_unrel", "obscene_rate")
    pooled = {}
    per_seed = {}
    for metric in metrics:
        deltas = {code: float(np.mean([a[metric]-c[metric] for c,a in sd.values()])) for code,sd in pairs.items()}
        vals = list(deltas.values())
        pooled[metric] = {"n_parents": len(vals), "parent_mean_delta": float(np.mean(vals)) if vals else None,
                          "parent_median_delta": float(np.median(vals)) if vals else None,
                          "parent_bootstrap_95_ci_mean": ci(vals,rng,nboot)}
        per_seed[metric] = {}
        for seed in SEEDS:
            v = [float(a[metric]-c[metric]) for sd in pairs.values() for c,a in [sd[seed]]]
            per_seed[metric][str(seed)] = {"n_parents":len(v),"mean_delta":float(np.mean(v)) if v else None,
                                           "median_delta":float(np.median(v)) if v else None,
                                           "parent_bootstrap_95_ci_mean":ci(v,rng,nboot)}
    breaches=[]
    for code,sd in pairs.items():
        ds=[float(a["top1_dE_unrel"]-c["top1_dE_unrel"]) for c,a in sd.values()]
        if sum(x>25 for x in ds)>=2:
            used=[]
            for seed,(c,a) in sd.items():
                if a["top1_dE_unrel"]-c["top1_dE_unrel"]>25:
                    used.append({"seed":seed,"delta_kj_mol":a["top1_dE_unrel"]-c["top1_dE_unrel"],
                                 "assignment_idx":a.get("top1_assignment_idx"),
                                 "assignment":a.get("top1_assignment",[]),
                                 "li_capped_donors_in_used_assignment":top1_capped_donors(arm[code],a)})
            breaches.append({"code":code,"seed_deltas_kj_mol":dict(zip(map(str,SEEDS),ds)),
                             "breach_seeds":used})
    return {"seeds":list(SEEDS),"complete_paired_parents":len(pairs),"footprint_parents":len(common),
            "missing_or_partial":missing,"pooled_by_parent":pooled,"per_seed":per_seed,
            "guard_breaches_gt_25_on_at_least_2_of_3_seeds":breaches}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--control",type=Path,required=True)
    ap.add_argument("--arm",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--bootstraps",type=int,default=10000); ap.add_argument("--boot-seed",type=int,default=20261029)
    a=ap.parse_args(); out=summarize(load(a.control),load(a.arm),a.bootstraps,a.boot_seed)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
