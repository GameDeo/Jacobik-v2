"""Relax only the top-1 rows selected by the GPU MACE unrelaxed screen."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from tools.runkit import mace, resources, xtb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="run_o1_g2.py --skip-xtb --save-structures output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold-kj", type=float, default=20.0)
    ap.add_argument("--max-structures", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    xtb.require()
    src = json.loads(Path(args.screen).read_text())
    selected = []
    for parent, pdata in src.get("parents", {}).items():
        top = next((r for r in pdata.get("rows", []) if r.get("rank") == 1 and r.get("status") == "ok"), None)
        if top is None or top.get("delta_E_unrel_kj") is None:
            continue
        if abs(float(top["delta_E_unrel_kj"])) > float(args.threshold_kj):
            selected.append((parent, top))
    selected.sort(key=lambda pr: abs(float(pr[1]["delta_E_unrel_kj"])), reverse=True)
    nrows = max(0, int(args.max_structures) // 2)
    selected = selected[:nrows]

    run = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "screen": str(args.screen), "criterion": f"abs(top1 delta_E_unrel_kj) > {args.threshold_kj}",
           "max_structures": int(args.max_structures), "selected_parent_rows": [], "results": [],
           "failures": {}}
    jobs, owners, codes = [], [], []
    for parent, row in selected:
        if row.get("symbols_order") != "builder/conf.complex.atoms":
            run["failures"][parent] = "unverified atom order; repair the saved export before using it"
            continue
        symbols = row.get("atom_symbols")
        X0, X1 = row.get("X_init_A"), row.get("X_star_A")
        if not symbols or X0 is None or X1 is None:
            run["failures"][parent] = "screen output lacks stored structures"
            continue
        q, spin = int(row["charge"]), int(row["spin_state"])
        run["selected_parent_rows"].append({"parent": parent, "rank": 1,
            "key": row["key"], "delta_E_unrel_kj": row["delta_E_unrel_kj"]})
        jobs.extend([(symbols, np.asarray(X0, dtype=np.float64), q, spin),
                     (symbols, np.asarray(X1, dtype=np.float64), q, spin)])
        owners.append((parent, row))
        codes.extend([f"{parent}:top1:init", f"{parent}:top1:star"])

    results = xtb.opt_many(jobs, workers=max(1, min(int(args.workers), 4)),
                           workroot=str(Path(args.out).resolve().parent / "o1_selected_xtb_work"),
                           cycles=200, level="crude", failures=str(Path(args.out).resolve().parent / "o1_selected_xtb_failures.jsonl"),
                           codes=codes) if jobs else []

    valid_structures, valid_map = [], []
    for i, (parent, row) in enumerate(owners):
        rel0, rel1 = results[2 * i], results[2 * i + 1]
        item = {"parent": parent, "rank": 1, "key": row["key"],
                "delta_score": row["delta_score"],
                "delta_E_unrel_kj": row["delta_E_unrel_kj"],
                "xtb_init": {k: rel0.get(k) for k in ("status", "converged", "cycles", "wall_s", "E_h", "stderr_tail")},
                "xtb_star": {k: rel1.get(k) for k in ("status", "converged", "cycles", "wall_s", "E_h", "stderr_tail")}}
        run["results"].append(item)
        if rel0.get("status") == "ok" and rel1.get("status") == "ok" and rel0.get("X") is not None and rel1.get("X") is not None:
            symbols, q, spin = row["atom_symbols"], int(row["charge"]), int(row["spin_state"])
            valid_structures.extend([(symbols, rel0["X"], q, spin), (symbols, rel1["X"], q, spin)])
            valid_map.append((item, float(row["E_unrel_init_kj"]), float(row["E_unrel_star_kj"])))
        else:
            item["strain_status"] = "failed: xTB relaxation did not converge"

    if valid_structures:
        try:
            with resources.gpu_lock("CodexB"):
                evaluator = mace.Evaluator(device="cuda", dtype="float64", threads=1)
                erel = evaluator.energies_kj(valid_structures)
            for j, (item, e0, e1) in enumerate(valid_map):
                s0, s1 = float(e0 - erel[2*j]), float(e1 - erel[2*j+1])
                item["strain_init_kj"] = s0
                item["strain_star_kj"] = s1
                item["delta_strain_kj"] = s1 - s0
                if item["delta_score"] <= 0.05:
                    item["classification"] = "c_barely_moves"
                elif item["delta_strain_kj"] < -5:
                    item["classification"] = "a_optimizer_problem"
                else:
                    item["classification"] = "b_score_minimum_strained"
        except Exception as exc:
            for item, _e0, _e1 in valid_map:
                item["strain_status"] = f"failed: {type(exc).__name__}: {exc}"
                item["traceback"] = traceback.format_exc()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(run, indent=2))
    print(f"selected {len(owners)} top-1 rows / {2*len(owners)} structures; results={len(run['results'])}; failures={len(run['failures'])}")


if __name__ == "__main__":
    main()
