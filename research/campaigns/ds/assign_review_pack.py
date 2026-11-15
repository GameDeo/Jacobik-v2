"""Write review packs (xyz + donor table) for parents where the rule chain and the evidence-ranked assignment disagree."""
import csv, sys, time, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from jacobik.topology import mapping
from jacobik.topology import donor_classifier as dc
from fit.parser import _iter_xyz_blocks, _iter_bo_blocks, _parse_xyz_comment, _assemble_raw

ROOT = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/data/ref/tmQM")

def load(codes):
    smiles = {}
    with (ROOT / "tmQM_y.csv").open(newline="") as f:
        rows = csv.reader(f, delimiter=";"); head = next(rows); ci, si = head.index("CSD_code"), head.index("SMILES")
        for r in rows:
            if r and r[ci].strip() in codes: smiles[r[ci].strip()] = r[si].strip() or None
    out = {}
    for xyzf in sorted((ROOT / "xyz").glob("*.xyz")):
        found = {}
        for comment, lines in _iter_xyz_blocks(xyzf):
            c = _parse_xyz_comment(comment)[0]
            if c in codes: found[c] = (comment, lines)
        if found:
            bo = {c: l for c, l in _iter_bo_blocks(ROOT / "bo" / (xyzf.stem + ".BO")) if c in found}
            for c, (comment, lines) in found.items():
                out[c] = dict(code=c, xyz_comment=comment, xyz_atoms=lines, bo_lines=bo[c], smiles=smiles.get(c))
        if len(out) == len(codes): break
    return out

def sites(src, proto):
    for m in (mapping, dc): m.DONOR_IDENTITY_RECONCILE = True
    dc.HAPTO_CHELATE_EXCLUSION = True; dc.HAPTO_BOND_GATE = True
    mapping.DONOR_ASSIGNMENT = proto
    dc.DONOR_ASSIGNMENT = proto
    t = time.perf_counter()
    raw = _assemble_raw(src["xyz_comment"], src["xyz_atoms"], src["code"], src["bo_lines"], SimpleNamespace(smiles=src["smiles"]))
    mol = mapping.assemble(raw)
    return [tuple(sorted(s.donor_indices)) for s in mol.coord_sites], time.perf_counter() - t

def main(codes, out):
    out.mkdir(parents=True, exist_ok=True)
    srcs = load(set(codes)); table = ["| parent | metal | rule chain | assignment | only rule chain | only assignment |", "|---|---|---|---|---|---|"]
    detail = []; times = []
    for code in codes:
        s = srcs[code]
        atoms = [l.split() for l in s["xyz_atoms"]]
        sym = {i + 1: a[0] for i, a in enumerate(atoms)}; pos = {i + 1: np.array(a[1:4], float) for i, a in enumerate(atoms)}
        metal = 1 + next(i for i, a in enumerate(atoms) if a[0] not in "H C N O F P S Cl Br I B Si As Se".split())
        mline = next(l for l in s["bo_lines"] if l.split()[0] == str(metal)).split()
        bo = {int(mline[k + 1]): float(mline[k + 2]) for k in range(3, len(mline) - 2, 3)}
        A, ta = sites(s, False); B, tb = sites(s, True); times.append((code, ta, tb))
        fa = {i for st in A for i in st}; fb = {i for st in B for i in st}
        lab = lambda i: f"{sym[i]}{i} ({np.linalg.norm(pos[i]-pos[metal]):.2f} Å, BO {bo.get(i,0):.2f})"
        grp = lambda S: "; ".join("+".join(f"{sym[i]}{i}" for i in st) for st in S)
        table.append(f"| {code} | {sym[metal]}{metal} | {grp(A)} | {grp(B)} | {', '.join(lab(i) for i in sorted(fa-fb)) or '—'} | {', '.join(lab(i) for i in sorted(fb-fa)) or '—'} |")
        n = len(atoms)
        (out / f"{code}.xyz").write_text(f"{n}\n{code} metal={sym[metal]}{metal} rulechain_donors={sorted(fa)} assignment_donors={sorted(fb)} (atom numbers are 1-based line numbers)\n" + "\n".join(s["xyz_atoms"]) + "\n")
        detail.append(dict(code=code, metal=f"{sym[metal]}{metal}", rule_chain=[list(x) for x in A], assignment=[list(x) for x in B],
                           metal_bo={f"{sym[i]}{i}": bo[i] for i in sorted(bo)},
                           dist={f"{sym[i]}{i}": round(float(np.linalg.norm(pos[i]-pos[metal])), 3) for i in sorted(fa | fb)}))
    (out / "table.md").write_text("\n".join(table) + "\n")
    (out / "detail.json").write_text(json.dumps(detail, indent=1))
    print("\n".join(table)); print()
    for c, a, b in times: print(f"{c}: rule chain {a*1000:.0f} ms, assignment {b*1000:.0f} ms")
    print("total", sum(a for _, a, _ in times)*1000, "ms vs", sum(b for _, _, b in times)*1000, "ms")

if __name__ == "__main__":
    main(sys.argv[2:], Path(sys.argv[1]))
