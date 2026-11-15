"""Reference-free input-plausibility status from the SMILES alone, compared with the reference label (CPU, read-only).

This is the candidate production status of research/archive/docs/roadmap/roadmap_generation_v1.md section 2.0. It never reads the reference
geometry, never rejects an input, and never changes the solver. For each pool parent it reports:

  hard  hypervalent_S           S/Se/Te with valence 4, neutral, no O neighbour (RDKit valence after sanitising)
  hard  absurd_light_charge     formal charge outside the element window (ctxio.LIGHT_QWIN)
  hard  absurd_metal_charge     formal charge above the metal's highest common oxidation state (ctxio.MAX_OX)
  hard  charge_imbalance_N      neutral N with >= 2 O- neighbours
  soft  quinoid_pattern         a 5/6 ring of C/N atoms with an exocyclic double bond on a ring atom and fewer than
                                floor(n/2) ring-internal double bonds, in the SMILES (no geometry)
  soft  sanitize_failed         RDKit sanitisation of the metal-free ligand graph failed

Status: 'implausible' if any hard flag; 'suspect' if any soft flag; otherwise 'plausible'. Ranking is never changed.
Compared with the reference label of d14 (evaluation only) and with the energy failure (top-1 above 200 kJ/mol) on
the effective pools. The AUC is the probability that a random failing pool has a higher count of flags than a
random non-failing pool (ties one half).

    python tools/jgen/diag/d19_smiles_only_status.py --labels <d14 band json> --out-json <path>
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxio  # noqa: E402
from poolio import Repairs, load_effective  # noqa: E402
from ctxio import absurd_light, absurd_metal  # noqa: E402



def smiles_flags(smi):
    from rdkit import Chem
    from rdkit.Chem import rdchem
    hard, soft = set(), set()
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return {"unparseable"}, set()
    m.UpdatePropertyCache(strict=False)
    if any(absurd_metal(a.GetSymbol(), a.GetFormalCharge()) for a in m.GetAtoms()
           if a.GetSymbol() not in ctxio.LIGHT and a.GetSymbol() != "H"):
        hard.add("absurd_metal_charge")
    rw = Chem.RWMol(m)
    for a in sorted(rw.GetAtoms(), key=lambda x: -x.GetIdx()):
        if a.GetSymbol() not in ctxio.LIGHT:
            rw.RemoveAtom(a.GetIdx())
    lig = rw.GetMol()
    try:
        Chem.SanitizeMol(lig)
    except Exception:
        soft.add("sanitize_failed")
    for a in lig.GetAtoms():
        if absurd_light(a.GetSymbol(), a.GetFormalCharge()):
            hard.add("absurd_light_charge")
        if a.GetSymbol() in ("S", "Se", "Te") and a.GetTotalValence() == 4 and a.GetFormalCharge() == 0 \
                and not any(n.GetSymbol() == "O" for n in a.GetNeighbors()):
            hard.add("hypervalent_S")
        if a.GetSymbol() == "N" and a.GetFormalCharge() == 0 and \
                sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "O" and n.GetFormalCharge() == -1) >= 2:
            hard.add("charge_imbalance_N")
    try:
        Chem.Kekulize(lig, clearAromaticFlags=True)
    except Exception:
        pass
    ri = lig.GetRingInfo()
    for r in ri.AtomRings():
        r = set(r)
        n = len(r)
        if n not in (5, 6) or not all(lig.GetAtomWithIdx(i).GetSymbol() in ("C", "N") for i in r):
            continue
        ring_double = exo = 0
        for b in lig.GetBonds():
            if b.GetBondType() != rdchem.BondType.DOUBLE:
                continue
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if u in r and v in r:
                ring_double += 1
            elif u in r or v in r:
                exo += 1
        if exo >= 1 and ring_double < n // 2:
            soft.add("quinoid_pattern")
    return hard, soft


def auc_counts(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return round(float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                       / (len(pos) * len(neg))), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    from poolio import control_tasks
    lab = {p["code"]: p["label"] for p in json.load(open(a.labels))["parents"]}
    idx = ctxio.frame_index()
    status = {}
    for code in lab:
        mol = ctxio.load_parent(code, idx)
        if mol is None:
            status[code] = dict(status="plausible", hard=[], soft=["not_loadable"])
            continue
        hard, soft = smiles_flags(mol.smiles)
        st = "implausible" if hard else ("suspect" if soft else "plausible")
        status[code] = dict(status=st, hard=sorted(hard), soft=sorted(soft))
    rep = Repairs(str(ctxio.SCALED))
    rows = []
    for code, seed, rec in control_tasks(os.path.join(str(ctxio.SCALED), "manifest.json")):
        if code not in lab:
            continue
        p = load_effective(os.path.join(str(ctxio.SCALED), "pools"), rep, code, seed)
        if p is None:
            continue
        rows.append(dict(code=code, seed=seed, label=lab[code], status=status[code]["status"],
                         n_hard=len(status[code]["hard"]), n_soft=len(status[code]["soft"]),
                         fail=bool(p["dE"][int(np.argmax(p["score"]))] > 200)))
    table = collections.Counter((r["label"], r["status"]) for r in rows)
    cross = {f"{k[0]}|{k[1]}": v for k, v in sorted(table.items())}
    by_status = {}
    for s in ("plausible", "suspect", "implausible"):
        rs = [r for r in rows if r["status"] == s]
        by_status[s] = dict(pools=len(rs), fail_share=round(float(np.mean([r["fail"] for r in rs])), 3) if rs else None)
    fail_flags = [r["n_hard"] + 0.5 * r["n_soft"] for r in rows if r["fail"]]
    ok_flags = [r["n_hard"] + 0.5 * r["n_soft"] for r in rows if not r["fail"]]
    out = dict(pools=len(rows), crosstab_label_status=cross, by_status=by_status,
               auc_flag_count_failure=auc_counts(fail_flags, ok_flags),
               auc_implausible_failure=auc_counts([1 if r["status"] == "implausible" else 0 for r in rows if r["fail"]],
                                                  [1 if r["status"] == "implausible" else 0 for r in rows if not r["fail"]]),
               flag_counts=dict(collections.Counter(f for s in status.values() for f in s["hard"] + s["soft"])),
               status_by_code={c: s["status"] for c, s in status.items()})
    print(ctxio.to_json(out))
    if a.out_json:
        with open(a.out_json, "w") as fh:
            fh.write(ctxio.to_json(out))


if __name__ == "__main__":
    main()
