"""Per-parent inspection bundle for A4b/A4t pools (coverage package, RC_DIAGNOSIS 5.2 (f)).

Every file is in REFERENCE atom order: pool frames are builder order and the manifest `gen_to_ref_idx` maps
builder -> reference by `X_ref = X_builder[gen_to_ref_idx]` (`a4b.relax_run.to_ref_order`).  The v1 bundle wrote
`X[argsort(gen_to_ref_idx)]`, which scrambles atoms.  SUMMARY.csv takes RMSD and dE from the v2 `rows2.csv`.

    python -m tools.jgen.a4t.bundle --a4b <root> --a4t <root> --rows2 <rows2.csv> --out <dir> --parents A,B
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.jgen.a4b.relax_run import to_ref_order
from tools.jgen.bench.pool_export import _read_frames, xyz_text

SEEDS = (11, 12, 13)


def top1_ref_order(manifest_path):
    """(symbols_ref, X_ref, header) of the pool's top-1 row, in reference order."""
    mp = Path(manifest_path)
    man = json.loads(mp.read_text())
    g2r = man["gen_to_ref_idx"]
    for header, _, X in _read_frames(mp.parent / man["rows"][0]["file"]):
        if header.get("is_top1"):
            return list(man["ref_symbols"]), to_ref_order(X, g2r), header
    raise ValueError("no top-1 row in manifest")


def export(a4b, a4t, rows2, out, parents):
    a4b, a4t, out = Path(a4b), Path(a4t), Path(out)
    refs = json.loads((a4t / "relax" / "references.json").read_text())
    rows = {}
    if rows2 and Path(rows2).exists():
        for r in csv.DictReader(open(rows2)):
            rows[(r["code"], int(r["seed"]), int(r["row_index"]))] = r
    summary = []
    for code in parents:
        d = out / code
        d.mkdir(parents=True, exist_ok=True)
        ref = refs[code]
        for s in SEEDS:
            mp = a4b / "xyz" / code / f"{code}_control_s{s}_pool.json"
            sy, X, h = top1_ref_order(mp)
            (d / f"top1_unrelaxed_s{s}.xyz").write_text(xyz_text(sy, X, dict(code=code, seed=s, row=h["row_index"], score=h["score"], order="reference")))
            rj = a4t / "relax" / f"{code}_{s}.json"
            frames = []
            if rj.exists():
                for i, r in sorted(json.loads(rj.read_text()).get("rows", {}).items(), key=lambda kv: int(kv[0])):
                    if r.get("status") != "ok" or r.get("X") is None:
                        continue
                    rr = rows.get((code, s, int(i)), {})
                    frames.append(xyz_text(sy, np.asarray(r["X"], float), dict(
                        code=code, seed=s, row=int(i), dE_mace=rr.get("dE_mace"), dE_gfn2=rr.get("dE_gfn2"),
                        rmsd_proper=rr.get("rmsd_proper"), rmsd_mirror=rr.get("rmsd_mirror"))))
            (d / f"pool_relaxed_s{s}.xyz").write_text("".join(frames))
            man = json.loads(mp.read_text())
            (d / "ref_crystal.xyz").write_text(xyz_text(sy, np.asarray(man["ref_X"], float), dict(code=code, kind="tmQM reference")))
        if ref.get("X") is not None:
            (d / "ref_relaxed.xyz").write_text(xyz_text(sy, np.asarray(ref["X"], float), dict(code=code, kind="GFN2-relaxed reference")))
        for s in SEEDS:
            rr = [r for (c, sd, _), r in rows.items() if c == code and sd == s and r.get("rmsd_any") not in ("", None)]
            if rr:
                b = min(rr, key=lambda r: float(r["rmsd_any"]))
                summary.append(dict(code=code, seed=s, best_row=b["row_index"], best_rmsd_proper=b["rmsd_proper"],
                                    best_rmsd_any=b["rmsd_any"], dE_mace=b["dE_mace"], dE_gfn2=b["dE_gfn2"]))
    with (out / "SUMMARY.csv").open("w", newline="") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a4b", required=True)
    ap.add_argument("--a4t", required=True)
    ap.add_argument("--rows2", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parents", required=True)
    a = ap.parse_args()
    print(len(export(a.a4b, a.a4t, a.rows2, a.out, a.parents.split(","))), "summary rows")


if __name__ == "__main__":
    main()
