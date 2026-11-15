"""S2 measurement (research/archive/docs/roadmap/roadmap_generation_v1.md 2.5): how often a native top-1 row has a metal-hydrogen contact below
2.2 A where the reference has that hydrogen farther than 3 A from the metal, per metal. CPU only, stored effective pools.

Rows are mapped to reference order with gen_to_ref (X[:, gen_to_ref] is reference order, as in the orchestrator).
Metal: every non-H atom of the reference with the metal symbol (the complex has one metal).

    python tools/jgen/diag/d24_mh_contacts.py --root <frame_v1> [--codes ...]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poolio import Repairs, load_effective  # noqa: E402

METALS = {"Li", "Na", "K", "Be", "Mg", "Ca", "Sr", "Ba", "Al", "Ga", "In", "Sn", "Pb", "Bi", "Tl", "Zn", "Cd", "Hg",
          "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
          "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
          "Er", "Tm", "Yb", "Lu", "Th", "U"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--codes", default="")
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()
    pools = os.path.join(a.root, "pools")
    rep = Repairs(a.root)
    m = json.load(open(os.path.join(a.root, "manifest.json")))
    units = sorted({(r["code"], int(r["seed"])) for r in m["tasks"].values()
                    if r.get("status") == "success" and r.get("split") in ("calibration", "motion_validation")})
    only = set(x for x in a.codes.split(",") if x)
    per_metal = defaultdict(lambda: dict(units=0, top1_contact_units=0, rows=0, contact_rows=0, parents=set()))
    details = []
    for code, seed in units:
        if only and code not in only:
            continue
        e = load_effective(pools, rep, code, seed)
        if e is None or e["gen_to_ref"] is None:
            continue
        X = e["X"][:, e["gen_to_ref"], :]
        ref = e["ref_X"]
        sym = e["ref_symbols"]
        mi = [i for i, s in enumerate(sym) if s in METALS]
        if len(mi) != 1:
            continue
        mi = mi[0]
        hs = [i for i, s in enumerate(sym) if s == "H"]
        if not hs:
            continue
        metal = sym[mi]
        pm = per_metal[metal]
        pm["units"] += 1
        pm["parents"].add(code)
        top = int(np.argmax(e["score"]))
        d_ref = np.linalg.norm(ref[hs] - ref[mi], axis=1)
        flagged_top = []
        for r in range(len(X)):
            d_gen = np.linalg.norm(X[r][hs] - X[r][mi], axis=1)
            bad = np.where((d_gen < 2.2) & (d_ref > 3.0))[0]
            pm["rows"] += 1
            if len(bad):
                pm["contact_rows"] += 1
            if r == top and len(bad):
                flagged_top = [dict(H=int(hs[k]), d_gen=round(float(d_gen[k]), 3), d_ref=round(float(d_ref[k]), 3))
                               for k in bad]
        if flagged_top:
            pm["top1_contact_units"] += 1
            details.append(dict(code=code, seed=seed, metal=metal, top1_contacts=flagged_top))
    summary = {k: dict(units=v["units"], parents=len(v["parents"]), top1_contact_units=v["top1_contact_units"],
                       top1_contact_share=round(v["top1_contact_units"] / v["units"], 3) if v["units"] else None,
                       rows=v["rows"], contact_row_share=round(v["contact_rows"] / v["rows"], 4) if v["rows"] else None)
               for k, v in sorted(per_metal.items())}
    out = dict(criterion="top-1 row: any H with d(M,H) < 2.2 A and reference d(M,H) > 3.0 A", per_metal=summary,
               details=details[:200])
    txt = json.dumps(out, indent=1)
    print(txt)
    if a.out_json:
        open(a.out_json, "w").write(txt)


if __name__ == "__main__":
    main()
