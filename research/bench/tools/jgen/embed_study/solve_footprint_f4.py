"""Build the solve-level cal/tail parent footprints from the frozen F-2 census."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from tools.jgen.bench.panels import TAIL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--transitions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    census = [json.loads(line) for line in a.census.read_text().splitlines() if line.strip()]
    parent_rows = {p["code"]: p for p in census}
    li_parents = defaultdict(list)
    for p in census:
        if p.get("cohort") != "cal_tail":
            continue
        unique = {}
        for lig in p.get("ligands", []):
            unique.setdefault(int(lig["ligand_idx"]), lig)
        for ix, lig in unique.items():
            if lig.get("status") != "ok":
                continue
            mol = Chem.MolFromSmiles(lig["smiles"])
            if mol is None:
                continue
            atoms = sorted({int(g[0]) for g in lig.get("donor_groups", [])
                            if len(g) == 1 and mol.GetAtomWithIdx(int(g[0])).GetFormalCharge() < 0})
            if atoms:
                li_parents[p["code"]].append({"ligand_idx": ix, "anionic_kappa_donors": atoms})
    transitions = json.loads(a.transitions.read_text())
    guard_all = set(transitions["trigger_parents"]["guard"])
    guard_cal = sorted(code for code in guard_all if parent_rows.get(code, {}).get("cohort") == "cal_tail")
    li_codes = sorted(li_parents)
    out = {"source": str(a.census), "guard": guard_cal, "li_cap": li_codes,
           "combined": sorted(set(guard_cal) | set(li_codes)),
           "li_cap_ligands_by_parent": {c: li_parents[c] for c in li_codes},
           "tail_overlap": sorted(c for c in li_codes if c in set(TAIL))}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"guard_parents": len(out["guard"]), "li_cap_parents": len(out["li_cap"]),
                      "combined_parents": len(out["combined"]), "tail_overlap": out["tail_overlap"],
                      "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
