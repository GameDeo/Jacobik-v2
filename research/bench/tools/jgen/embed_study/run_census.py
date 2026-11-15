"""Measure current production embeds over the contracted F cohort (three CPU workers, no GPU)."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "bench")]

from tools.jgen import inputs  # noqa: E402
from tools.jgen.bench.panels import TAIL  # noqa: E402
from tools.jgen.embed_study.measure_embeds import measure_parent  # noqa: E402

MAIN = ROOT.parents[2]
LOCKED = MAIN / "research/bench/benchmarks/locked_exclusions.json"


def _work(arg):
    code, seed, ligand_indices = arg
    try:
        return measure_parent(code, seed, ligand_indices)
    except Exception as exc:
        return {"code": code, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True, help="resonance_scan.json (defines sampled ligand IDs)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.workers < 1 or args.workers > 3:
        ap.error("the directive caps local CPU concurrency at 3")
    selection = json.load(open(args.selection))
    frame = json.load(open(inputs.FRAME_V1 / "frame.json"))
    ex = json.load(open(LOCKED))
    excluded = set(ex["reserve_all_codes"]) | set(ex["motion_validation_codes"])
    split_codes = {split: {g["representative_code"] for g in gs} for split, gs in
                   frame["main_allocation"]["splits"].items()}
    cal_tail = (split_codes["calibration"] | set(TAIL)) - excluded
    cal_tail = {c for c in cal_tail if c in split_codes["calibration"]}
    if set(cal_tail) & excluded:
        raise RuntimeError("val/reserve/locked code entered calibration cohort")
    sample_keys = selection["surface_fit_sample_manifest"]
    by_code = {}
    for rec in sample_keys:
        by_code.setdefault(rec["code"], []).append(int(rec["ligand_idx"]))
    if any(c in excluded for c in by_code):
        raise RuntimeError("val/reserve/locked code entered surface_fit sample")
    if any(c not in split_codes["surface_fit"] for c in by_code):
        raise RuntimeError("non-surface_fit code entered surface_fit sample")

    jobs = [(c, args.seed, None) for c in sorted(cal_tail)]
    jobs.extend((c, args.seed, sorted(ix)) for c, ix in sorted(by_code.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, open(args.output, "w") as out:
        futures = [pool.submit(_work, j) for j in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            rec["cohort"] = "surface_fit_sample" if rec["code"] in by_code else "cal_tail"
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
    print(json.dumps({"cal_tail_parents": len(cal_tail), "surface_fit_sample_parents": len(by_code),
                      "surface_fit_sample_ligands": len(sample_keys), "jobs": len(jobs),
                      "workers": args.workers, "output": str(args.output)}))


if __name__ == "__main__":
    main()
