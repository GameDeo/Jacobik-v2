"""Production-solve score reproduction smoke suite for Package G-2 task 2a."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / "src"), str(REPO), str(REPO / "research/bench")]

from tools.jgen import inputs
from tools.jgen.score_repro import CaptureSession
from jacobik.core.env import build_builder_and_inputs
from tools.jdev.budget.common import stats_from_db
from tools.runkit import xtb


CASES = {
    "single_embed": "HEDJOP",
    "ez_variants": "NUWFAM",
    "multiple_assignments": "QEZVEY",
    "macrocycle": "USAPOU",
}


def run(device: str = "cpu", only: str | None = None) -> dict:
    shards = inputs.frame_shards()
    stats = stats_from_db(inputs.DB_PATH)
    out = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "device": device, "cases": {}, "all_rows_pass": True}
    cases = {k: v for k, v in CASES.items() if only is None or k == only}
    for kind, code in cases.items():
        try:
            sh, ds = shards[code]
            mol = inputs.load_mol(code, sh, ds)
            builder, _, err = build_builder_and_inputs(mol)
            if builder is None:
                raise RuntimeError(err)
            with xtb.FallbackWatch(raise_on_exit=True):
                with CaptureSession(builder) as capture:
                    result = builder.solve(seed=11, stats=stats, angle_bends=True,
                                           device=device, n_conformers=16)
            capture.attach_conformers(result.conformers)
            rows = []
            for conf in result.conformers:
                cap = capture.for_conformer(conf)
                row = int(conf.trial_idx)
                score32 = float(cap.scores[row].detach().cpu())
                check32 = {"key": [conf.embed_idx, int(conf.assignment_idx), row],
                           "score_calc": score32, "score_stored": float(conf.score),
                           "abs_diff": abs(score32 - float(conf.score)),
                           "pass": abs(score32 - float(conf.score)) < 1e-4}
                theta64 = [np.asarray(t, dtype=np.float64) for t in conf.solver_state["thetas"]]
                check64 = dict(check32)
                check64["score_calc"] = float(cap.score(theta64, row).detach().cpu())
                check64["abs_diff"] = abs(check64["score_calc"] - float(conf.score))
                check64["pass"] = check64["abs_diff"] < 1e-4
                check64["float32_float64_delta"] = check64["score_calc"] - check32["score_calc"]
                rows.append({"float32": check32, "float64": check64})
            n_assign = int((result.assignment_search or {}).get("n_candidates", 1))
            variant_count = len(capture.captures)
            expected = {"single_embed": 1, "ez_variants": 2, "multiple_assignments": 2,
                        "macrocycle": 1}[kind]
            structural_pass = (variant_count >= expected if kind == "ez_variants" else True)
            if kind == "multiple_assignments":
                structural_pass = structural_pass and n_assign > 1
            out["cases"][kind] = {"code": code, "rows": rows, "captures": variant_count,
                                  "n_assignments": n_assign, "structural_pass": structural_pass,
                                  "all_rows_pass": all(x["float64"]["pass"] for x in rows)}
            out["all_rows_pass"] &= out["cases"][kind]["structural_pass"] and out["cases"][kind]["all_rows_pass"]
        except Exception as exc:
            out["all_rows_pass"] = False
            out["cases"][kind] = {"code": code, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc()}
            print(f"[{code}] score reproduction failed: {exc}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", choices=tuple(CASES))
    ap.add_argument("--out", default=str(REPO / "research/data/jgen/campaign/B/g/score_repro_smoke.json"))
    args = ap.parse_args()
    result = run(args.device, args.only)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"all_rows_pass": result["all_rows_pass"],
                      "case_status": {k: v.get("all_rows_pass", False) for k, v in result["cases"].items()},
                      "output": str(path)}, indent=2))
    if not result["all_rows_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
