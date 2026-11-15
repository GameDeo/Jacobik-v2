"""Run the owner-directed unrelaxed O2 gate on all nine trope parents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.runkit.trope import suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--only", default="", help="Optional comma-separated parent subset for a smoke run")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--agent", default="CodexB")
    args = ap.parse_args()
    root = Path(args.out_root)
    foot_path = Path(__file__).resolve().parents[3] / "campaigns/g/o2_trope_footprint.json"
    foot = json.loads(foot_path.read_text())
    only = {c for c in args.only.split(",") if c} or None
    arms = [("control", {}), ("adam_metric", {"adam_metric": True}),
            ("fire_metric", {"fire_metric": True})]
    for name, flags in arms:
        out = root / f"{name}_unrel"
        print(f"RUN {name}: unrelaxed-only, CUDA MACE, seed 11, {len(foot['parents']) if only is None else len(only)} parents", flush=True)
        suite.run_suite(name, flags, foot, out, only=only, procs=args.procs, agent=args.agent,
                        unrelaxed_only=True, mace_device="cuda", seeds_override=[11])
    for name, _flags in arms[1:]:
        compare_foot = json.loads(foot_path.read_text())
        report = suite.compare(root / "control_unrel", root / f"{name}_unrel", compare_foot)
        path = root / f"compare_{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1))
        print(f"COMPARE {name}: outside-identical={report['outside_ok']} rows={len(report['rows'])} -> {path}", flush=True)


if __name__ == "__main__":
    main()
