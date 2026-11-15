"""Benchmark worker configurations for Jacobik NCI v2 Directive T3.
Measures groups/h for workers = 1, 2, 3 on the 12 pilot parents, with and without MACE.
Extracts per-stage timing from solve_seeds info: embed / seed_rows / solver / dedup / MACE / other.
Projects the scaled run from measured rates only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "research/bench/tools/jdev/budget"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research.archive.code.refit.nci_v2_orchestrator import run_orchestrator, DEFAULT_SPEC, DEFAULT_FRAME

PILOT_12_CODES = [
    "HESBAM", "OMAXUV", "QOCKUO", "JIXWEW", "ICEYAQ", "SUDMEM",
    "WIVVED", "KOTJIP", "RIGZUC", "DAFFEZ", "HAMYUP", "YUKLIA",
]


def run_benchmark(
    workers_list: List[int] = (1, 2, 3),
    mace_modes: List[bool] = (False, True),
    seeds: List[int] = (11, 12, 13),
    out_base: Path = REPO_ROOT / "data/refit/nci_v2/t3_benchmark",
) -> Dict[str, Any]:
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    for eval_mace in mace_modes:
        mace_tag = "mace" if eval_mace else "no_mace"
        results[mace_tag] = {}

        for w in workers_list:
            run_tag = f"w{w}_{mace_tag}"
            run_dir = out_base / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=======================================================")
            print(f"Benchmarking Workers={w}, MACE={eval_mace} on 12 Pilot Parents")
            print(f"=======================================================")

            # Check load average before run
            load1, load5, load15 = os.getloadavg()
            print(f"Pre-run load average: 1m={load1:.2f}, 5m={load5:.2f}, 15m={load15:.2f}")

            t0 = time.perf_counter()
            manifest = run_orchestrator(
                spec_path=DEFAULT_SPEC,
                frame_path=DEFAULT_FRAME,
                cohort="pilot",
                codes=PILOT_12_CODES,
                seeds=seeds,
                arms=["control"],
                workers=w,
                eval_mace=eval_mace,
                out_dir=run_dir,
                force_recompute=True,
                research_arm=True,
            )
            total_wall_s = time.perf_counter() - t0

            # Compute statistics across the 12 parents
            # Each parent has 3 seeds in tasks dictionary
            # Read stage timing from task summaries
            pools_dir = run_dir / "pools"
            parent_wall_times = []
            embed_times = []
            seed_rows_times = []
            solver_times = []
            dedup_times = []
            mace_times = []
            other_times = []

            for code in PILOT_12_CODES:
                # Seed 11 summary contains the group-level solve timings
                summ_p = pools_dir / f"{code}_11_control.json"
                if summ_p.exists():
                    data = json.loads(summ_p.read_text(encoding="utf-8"))
                    wall = data.get("wall_s", 0.0)
                    parent_wall_times.append(wall)
                    embed_times.append(data.get("stage_embed_s", 0.0))
                    seed_rows_times.append(data.get("stage_seed_rows_s", 0.0))
                    solver_times.append(data.get("stage_solver_s", 0.0))
                    dedup_times.append(data.get("stage_dedup_s", 0.0))
                    mace_times.append(data.get("stage_mace_s", 0.0))
                    other_times.append(data.get("stage_other_s", 0.0))

            n_groups = len(parent_wall_times)
            groups_per_hour = (n_groups / total_wall_s) * 3600.0 if total_wall_s > 0 else 0.0
            mean_s_per_group = total_wall_s / n_groups if n_groups > 0 else 0.0

            run_summary = {
                "workers": w,
                "eval_mace": eval_mace,
                "n_groups": n_groups,
                "total_wall_s": total_wall_s,
                "groups_per_hour": groups_per_hour,
                "mean_wall_s_per_group": mean_s_per_group,
                "p95_wall_s_per_group": float(np.percentile(parent_wall_times, 95)) if parent_wall_times else 0.0,
                "per_parent_wall_s": dict(zip(PILOT_12_CODES, parent_wall_times)),
                "stage_timing_mean_s": {
                    "embed": float(np.mean(embed_times)) if embed_times else 0.0,
                    "seed_rows": float(np.mean(seed_rows_times)) if seed_rows_times else 0.0,
                    "solver": float(np.mean(solver_times)) if solver_times else 0.0,
                    "dedup": float(np.mean(dedup_times)) if dedup_times else 0.0,
                    "mace": float(np.mean(mace_times)) if mace_times else 0.0,
                    "other": float(np.mean(other_times)) if other_times else 0.0,
                },
                "stage_timing_p95_s": {
                    "embed": float(np.percentile(embed_times, 95)) if embed_times else 0.0,
                    "seed_rows": float(np.percentile(seed_rows_times, 95)) if seed_rows_times else 0.0,
                    "solver": float(np.percentile(solver_times, 95)) if solver_times else 0.0,
                    "dedup": float(np.percentile(dedup_times, 95)) if dedup_times else 0.0,
                    "mace": float(np.percentile(mace_times, 95)) if mace_times else 0.0,
                    "other": float(np.percentile(other_times, 95)) if other_times else 0.0,
                },
                "peak_vram_mb": manifest.get("peak_vram_mb", 0.0),
                "peak_rss_mb": manifest.get("peak_rss_mb", 0.0),
            }

            results[mace_tag][f"workers_{w}"] = run_summary
            print(f"Results for workers={w}, MACE={eval_mace}:")
            print(f"  Wall time: {total_wall_s:.2f}s for {n_groups} groups")
            print(f"  Throughput: {groups_per_hour:.2f} groups/h ({mean_s_per_group:.2f}s/group)")
            print(f"  Stage breakdown (mean): embed={run_summary['stage_timing_mean_s']['embed']:.2f}s, "
                  f"seed_rows={run_summary['stage_timing_mean_s']['seed_rows']:.2f}s, "
                  f"solver={run_summary['stage_timing_mean_s']['solver']:.2f}s, "
                  f"dedup={run_summary['stage_timing_mean_s']['dedup']:.2f}s, "
                  f"mace={run_summary['stage_timing_mean_s']['mace']:.2f}s, "
                  f"other={run_summary['stage_timing_mean_s']['other']:.2f}s")

    # Projections
    report_file = out_base / "t3_timing_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport written to {report_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=str, default="1,2,3")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/refit/nci_v2/t3_benchmark")
    args = parser.parse_args()
    w_list = [int(w.strip()) for w in args.workers.split(",") if w.strip()]
    run_benchmark(workers_list=w_list, out_base=args.out_dir)
