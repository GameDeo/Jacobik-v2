"""Estimate Q1 key contamination from inputs (CPU, read-only). ESTIMATE, not a measurement of the DB.

The production constraints DB (data/refit/final/constraints_2026-10-18.json) has no per-key provenance, so the
contamination of its Q1 keys cannot be read from the DB. This script estimates it from the inputs instead:

  1. For every parent of a split (default surface_fit), run the donor classifier on the reference complex
     (jacobik.topology.donor_classifier.classify, smiles_path=True; no solve) and record the donor subclass labels
     (mol.donor_labels). The Q1 key contains the donor subclass (roadmap_generation_v1 section 1.5, engine.py
     Q1 tuple), so a subclass is the unit of contamination here.
  2. Attach each parent's d14 label (--labels json, produced for the same split).
  3. For each subclass: the number of training parents (split) with that subclass, the number of them that are
     inconsistent, and the share. Each inconsistent parent lists its subclasses.

The estimate says how many training parents with a given subclass have a defective SMILES. It does not say which
DB key a parent contributed to, nor how much weight it had.

    python tools/jgen/diag/d21_q1_subclass_contamination.py --labels <d14 split json> --split surface_fit --out-json <path>
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxio  # noqa: E402


def donor_subclasses(mol):
    from jacobik.topology.donor_classifier import classify
    classify(mol, smiles_path=True)
    out = []
    for idx, lab in sorted(getattr(mol, "donor_labels", {}).items()):
        out.append("%s:%s" % (getattr(lab, "family", "?"), getattr(lab, "subclass", "?")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", default="surface_fit")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    labs = {p["code"]: p["label"] for p in json.load(open(a.labels))["parents"]}
    idx = ctxio.frame_index()
    codes = [c for c, r in idx.items() if r["split"] == a.split and c in labs]
    rows = []
    for code in codes:
        mol = ctxio.load_parent(code, idx)
        if mol is None:
            continue
        try:
            subs = sorted(set(donor_subclasses(mol)))
        except Exception as exc:  # record, do not fail
            subs = ["classify_error:%s" % type(exc).__name__]
        rows.append(dict(code=code, label=labs[code], subclasses=subs))
    per_sub = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        for s in r["subclasses"]:
            per_sub[s][r["label"]] += 1
            per_sub[s]["parents"] += 1
    table = {}
    for s, c in sorted(per_sub.items(), key=lambda kv: -kv[1]["parents"]):
        n = c["parents"]
        table[s] = dict(parents=n, inconsistent=c["inconsistent"], ambiguous=c["ambiguous"],
                        share_inconsistent=round(c["inconsistent"] / n, 3) if n else None)
    bad = [r for r in rows if r["label"] == "inconsistent"]
    out = dict(split=a.split, parents=len(rows), inconsistent=len(bad),
               inconsistent_parents=[dict(code=r["code"], subclasses=r["subclasses"]) for r in bad],
               subclass_table=table)
    print(ctxio.to_json(dict(split=a.split, parents=len(rows), inconsistent=len(bad),
                             top_subclasses=dict(list(table.items())[:25]))))
    if a.out_json:
        with open(a.out_json, "w") as fh:
            fh.write(ctxio.to_json(out))


if __name__ == "__main__":
    main()
