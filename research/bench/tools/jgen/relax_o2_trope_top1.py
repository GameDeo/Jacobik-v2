"""At the end of O2, relax exactly one seed-11 top-1 structure per trope parent."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.runkit import mace, resources, xtb
from tools.runkit.trope.suite import load_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-dir", required=True, help="Best arm's unrelaxed suite output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected", type=int, default=9)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    xtb.require()
    rows, missing = [], []
    for p in load_suite()["parents"]:
        code = p["code"]
        path = Path(args.screen_dir) / f"{code}.json"
        if not path.exists():
            missing.append({"parent": code, "reason": "screen record missing"})
            continue
        rec = json.loads(path.read_text())
        seed = rec.get("per_seed", {}).get("11", {})
        if rec.get("status") != "ok" or seed.get("status") != "ok" or "top1_X" not in seed:
            missing.append({"parent": code, "reason": f"top-1 unavailable: {rec.get('status')} / {seed.get('status')}"})
            continue
        rows.append({"parent": code, "symbols": seed["atom_symbols"], "X": seed["top1_X"],
                     "charge": seed["charge"], "spin_state": seed["spin_state"],
                     "score": seed["top1_score"], "dE_unrel_kj": seed["dE_unrel"]})
    if len(rows) != args.expected:
        raise SystemExit(f"expected {args.expected} top-1 structures; available={len(rows)}, missing={len(missing)}")
    codes = [r["parent"] + ":seed11:top1" for r in rows]
    jobs = [(r["symbols"], r["X"], r["charge"], r["spin_state"]) for r in rows]
    out_path = Path(args.out)
    opt = xtb.opt_many(jobs, workers=max(1, min(args.workers, 4)),
                       workroot=str(out_path.resolve().parent / "o2_trope_top1_xtb_work"),
                       cycles=200, level="crude",
                       failures=str(out_path.resolve().parent / "o2_trope_top1_xtb_failures.jsonl"), codes=codes)
    valid, valid_idx = [], []
    result_rows = []
    for i, (row, relax) in enumerate(zip(rows, opt)):
        item = {"parent": row["parent"], "score": row["score"], "dE_unrel_kj": row["dE_unrel_kj"],
                "xtb_status": relax.get("status"), "xtb_converged": relax.get("converged"),
                "xtb_cycles": relax.get("cycles"), "xtb_wall_s": relax.get("wall_s"),
                "xtb_stderr_tail": relax.get("stderr_tail")}
        result_rows.append(item)
        if relax.get("status") == "ok" and relax.get("X") is not None:
            valid.extend([(row["symbols"], row["X"], row["charge"], row["spin_state"]),
                          (row["symbols"], relax["X"], row["charge"], row["spin_state"])])
            valid_idx.append(i)
    if valid:
        with resources.gpu_lock("CodexB"):
            ev = mace.Evaluator(device="cuda", dtype="float64", threads=1)
            energies = ev.energies_kj(valid)
        for j, i in enumerate(valid_idx):
            result_rows[i]["E_unrel_kj"] = float(energies[2*j])
            result_rows[i]["E_relaxed_kj"] = float(energies[2*j+1])
            result_rows[i]["strain_kj"] = float(energies[2*j] - energies[2*j+1])
    payload = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "screen_dir": args.screen_dir, "relaxed_structures_attempted": len(rows),
               "expected": args.expected, "missing": missing, "results": result_rows,
               "note": "Only the nine seed-11 top-1 rows were submitted to xTB; failed relaxations remain failed."}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"xTB structures attempted={len(rows)}; successful={len(valid_idx)}; missing={len(missing)}")


if __name__ == "__main__":
    main()
