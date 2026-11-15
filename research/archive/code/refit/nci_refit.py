"""Scoped NCI regeneration: real process workers, resume guards, separate candidate DB.

    pixi run python research/archive/code/refit/nci_refit.py pilot --workers 8 --limit 100
    pixi run python research/archive/code/refit/nci_refit.py run --workers 8

Reads existing cleaned shards. Does not extract/refit other quantities, install
the DB, enable NCI or run solver benchmarks. Census uses references/stored picks.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "research" / "refit"), str(ROOT / "research" / "bench")]
DEFAULT_OUT = ROOT / "data/refit/nci/2026-10-18_bound_fix"
BASE = ROOT / "data/refit/final/constraints_2026-10-17.json"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.update(MKL_THREADING_LAYER="SEQUENTIAL", PYTHONIOENCODING="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


@lru_cache(maxsize=2)
def resources(data):
    import torch
    from rdkit import RDLogger
    from jacobik.fit.parser import TMQMParser
    from refit.refit_lib import sibling_weights
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(1)
    return TMQMParser(data), sibling_weights(Path(data))


def hist_task(task):
    from collections import Counter
    import logging
    import numpy as np
    import psutil
    from research.archive.code.interactions.nci import HistAccumulator
    from tools.jdev.budget.nci_census import TYPES, census

    start = time.perf_counter()
    warning_counts, warning_examples = Counter(), []

    class CountWarnings(logging.Handler):
        def emit(self, record):
            warning_counts[f"{record.name}:{record.levelname}"] += 1
            if len(warning_examples) < 5:
                warning_examples.append(record.getMessage())

    logger = logging.getLogger()
    logger.handlers = [CountWarnings()]
    logger.setLevel(logging.WARNING)
    parser, weights = resources(task["dir"])
    parser.n_failed = 0
    shard, data = task["shard"], Path(task["dir"])
    codes = re.findall(r"CSD_code = ([^ |\r\n]+)", (data / "xyz" / f"tmQM_{shard}.xyz").read_text())
    exclude = set(task["exclude"])
    chosen = set(sorted((c for c in codes if c not in exclude),
                        key=lambda c: zlib.crc32(f'nci-census-2026-10-18:{c}'.encode()))[:task["census_limit"]])
    acc = HistAccumulator()
    n = excluded = 0
    census_rows = []
    errors = []
    for mol in parser.stream_split(shard, limit=task["limit"]):
        n += 1
        if mol.csd_code in exclude:
            excluded += 1
            continue
        weight = weights.get(mol.csd_code, 1.0)
        try:
            acc.add(mol, weight)
            if mol.csd_code in chosen:
                census_rows.append(dict(code=mol.csd_code, weight=weight, **census(mol)))
        except Exception as exc:
            errors.append(dict(code=mol.csd_code, error=f"{type(exc).__name__}: {exc}"))
        if n % 250 == 0:
            write_json(Path(task["out"]) / f"progress_{shard}.json",
                       dict(shard=shard, inputs=n, elapsed_s=time.perf_counter()-start, pid=os.getpid(),
                            rss_mb=psutil.Process().memory_info().rss/2**20, errors=len(errors)))
    path = Path(task["out"]) / f"ncihist_{shard}.npz"
    np.savez_compressed(path, **{"h:" + k: v for k, v in acc.h.items()})
    report = dict(dataset=task["dataset"], shard=shard, complexes=n, excluded=excluded, used=n-excluded,
                  parse_failed=parser.n_failed, errors=errors, keys=len(acc.h), census=census_rows,
                  expected_inputs=len(codes) if task["limit"] is None else min(len(codes), task["limit"]),
                  elapsed_s=time.perf_counter()-start, pid=os.getpid(), rss_mb=psutil.Process().memory_info().rss/2**20,
                  warning_counts=dict(warning_counts), warning_examples=warning_examples,
                  sha256=digest(path), counts={t: float(sum(v.sum() for k, v in acc.h.items() if k.startswith(t+"|")))
                                             for t in ("hb", "pp", "chp")})
    write_json(path.with_suffix(".json"), report)
    return report


def regenerate(out, workers, limit=None, pilot=False):
    from refit.refit_lib import _loky_unordered, benchmark_codes
    from tools.jdev.env import tree_hash
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shards = json.loads((ROOT / "data/refit/reports/contacts/shards.json").read_text())
    if pilot:
        # Same 16 representative tasks for both worker counts, including LnQM.
        tm = [s for s in shards if s["dataset"] != "LNf"]
        ln = [s for s in shards if s["dataset"] == "LNf"]
        shards = tm[::2][:12] + ln[:4]
    inputs = []
    for sh in shards:
        data = Path(sh["dir"])
        for p in (data / "xyz" / f'tmQM_{sh["shard"]}.xyz', data / "bo" / f'tmQM_{sh["shard"]}.BO', data / "tmQM_y.csv"):
            inputs.append((str(p), p.stat().st_size, p.stat().st_mtime_ns))
    identity = dict(source=tree_hash(), driver_sha256=digest(__file__),
                    census_sha256=digest(ROOT / "research/bench/tools/jdev/budget/nci_census.py"), inputs=sorted(set(inputs)),
                    exclusions=benchmark_codes(), limit=limit, census_per_shard=100)
    # JSON normalizes tuples to lists; normalize before comparison on resume.
    identity = json.loads(json.dumps(identity))
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["identity"] == identity, "inputs/source changed across resume"
    else:
        manifest = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), identity=identity, workers=workers,
                        pilot=pilot, status="running", input_files_sha256={p: digest(p) for p, _, _ in sorted(set(inputs))})
        write_json(manifest_path, manifest)
    tasks = []
    for sh in shards:
        path = out / f'ncihist_{sh["shard"]}.npz'
        report = path.with_suffix(".json")
        if path.exists() and report.exists():
            r = json.loads(report.read_text())
            assert not r["errors"] and not r["parse_failed"] and r["complexes"] == r["expected_inputs"]
            assert digest(path) == r["sha256"]
            continue
        tasks.append({**sh, "out": str(out), "exclude": identity["exclusions"], "limit": limit, "census_limit": 100})
    start = time.perf_counter()
    task_fn = importlib.import_module("research.archive.code.refit.nci_refit").hist_task
    for i, result in enumerate(_loky_unordered(task_fn, tasks, workers), 1):
        print(f'{i}/{len(tasks)} {result["shard"]}: {result["complexes"]} inputs, {result["elapsed_s"]:.1f}s, '
              f'PID {result["pid"]}, RSS {result["rss_mb"]:.0f} MB, errors {len(result["errors"])}', flush=True)
        assert not result["errors"] and not result["parse_failed"], result
        assert result["complexes"] == result["expected_inputs"], result
    assert tree_hash() == identity["source"], "source changed during regeneration"
    results = [json.loads((out / f'ncihist_{s["shard"]}.json').read_text()) for s in shards]
    manifest.update(status="histograms_complete", finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                    wall_s=time.perf_counter()-start, worker_pids=sorted({r["pid"] for r in results}),
                    worker_peak_rss_mb=max(r["rss_mb"] for r in results), complexes=sum(r["complexes"] for r in results),
                    excluded=sum(r["excluded"] for r in results), used=sum(r["used"] for r in results), shards=len(results))
    write_json(manifest_path, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("identity", "input_files_sha256")}), flush=True)
    return results


def fit(out, results):
    from research.archive.code.interactions import nci
    start = time.perf_counter()
    base_hash = digest(BASE)
    hist = nci.load_hist([out / f'ncihist_{r["shard"]}.npz' for r in results])
    entries = nci.fit_tables(hist)
    assert entries, "no fitted NCI tables"
    raw = json.loads(BASE.read_text())
    unchanged = {k: v for k, v in raw.items() if not k.startswith("NCI:")}
    candidate = {**unchanged, **entries}
    target = out / "constraints_nci_2026-10-18.json"
    write_json(target, candidate)
    reread = json.loads(target.read_text())
    assert {k: v for k, v in reread.items() if not k.startswith("NCI:")} == unchanged
    meta = json.loads(BASE.with_suffix(".meta.json").read_text())
    meta["nci_regeneration"] = dict(rule="exclude both D/A primary metal-bound", base=str(BASE), base_sha256=base_hash,
                                    source_manifest=str(out / "manifest.json"), enabled_by_default=False)
    write_json(target.with_suffix(".meta.json"), meta)
    assert digest(BASE) == base_hash
    summary = dict(base=str(BASE), base_sha256=base_hash, candidate=str(target), candidate_sha256=digest(target),
                   old_nci_tables=sum(k.startswith("NCI:") for k in raw), new_nci_tables=len(entries),
                   non_nci_entries_unchanged=len(unchanged), elapsed_s=time.perf_counter()-start,
                   tables={k: dict(n=v["n"], maximum=max(map(max, v["s"])), minimum=min(map(min, v["s"]))) for k, v in entries.items()})
    write_json(out / "fit_summary.json", summary)
    print(json.dumps(summary), flush=True)


def dev_task(item):
    import numpy as np
    from tools.jdev.budget.nci_census import census
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry
    from tools.jdev.budget.common import pool_path
    code, name, set_name = item
    case = get_case(name)
    pool = pool_path("etkdg_u1_os", set_name, code, 11)
    with np.load(pool) as z:
        with starting_geometry(case, "full_oracle"):
            mol, _, _ = case.make_builder()._build_shell(False, False)
        assert [a.symbol for a in mol.atoms] == z["symbols"].tolist(), f"atom order mismatch: {code}"
        pick = census(mol, z["X"][int(np.argmax(z["score"]))])
    return dict(code=code, set=set_name, reference=census(case.tmqm_mol), pick=pick)


def report_census(out, results, workers):
    from refit.refit_lib import _loky_unordered
    from tools.jdev.budget.pools import set_cases
    from tools.jdev.budget.nci_census import TYPES
    valid = json.loads((ROOT / "data/benchmark_runs/2026-10-16_post_refit/donor_atoms.json").read_text())
    tasks = [(c, name, s) for s in ("dev32", "n48") for c, name in set_cases(s)
             if not isinstance(valid.get(c), dict) or valid[c].get("valid_atoms", True)]
    rows = list(_loky_unordered(importlib.import_module("research.archive.code.refit.nci_refit").dev_task, tasks, workers))
    training = {}
    for dataset in sorted({r["dataset"] for r in results}):
        sample = [c for r in results if r["dataset"] == dataset for c in r["census"]]
        weight = sum(c["weight"] for c in sample)
        training[dataset] = dict(complexes=len(sample), effective_weight=weight,
                                presence={t: dict(count=sum(c[t] > 0 for c in sample),
                                                  share=sum(c[t] > 0 for c in sample)/max(len(sample), 1),
                                                  weighted_share=sum(c["weight"] for c in sample if c[t] > 0)/max(weight, 1e-12))
                                          for t in TYPES})
    dev = {t: dict(references_with_it=sum(r["reference"][t] > 0 for r in rows),
                   picks_missing=sum(r["reference"][t] > 0 and r["pick"][t] < r["reference"][t] for r in rows),
                   picks_extra=sum(r["pick"][t] > r["reference"][t] for r in rows)) for t in TYPES}
    write_json(out / "census.json", dict(arm="etkdg_u1_os", seed=11, dev_complexes=len(rows), dev=dev, training=training,
                                        rows=rows, scope="geometric inventory; no energy evaluation or solver run"))
    print("census complete", len(rows), "dev cases", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("pilot", "run", "v2-registry", "v2-frame", "v2-pilot", "v2-scaling", "v2-orchestrate", "orchestrate"),
                   help="Pipeline stage: legacy 'pilot'/'run', or v2 subcommands 'v2-registry', 'v2-frame', 'v2-pilot', 'v2-scaling', 'v2-orchestrate'")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for legacy stage")
    p.add_argument("--workers", type=int, default=8, help="worker process count for legacy stage")
    p.add_argument("--limit", type=int, default=100, help="pilot inputs per shard for legacy stage")
    p.add_argument("--spec", type=Path, default=ROOT / "data/refit/nci_v2/pilot_v1/spec.json",
                   help="v2 specification JSON path (default: data/refit/nci_v2/pilot_v1/spec.json)")
    p.add_argument("--frame", type=Path, default=ROOT / "data/refit/nci_v2/pilot_v1/frame.json",
                   help="v2 corpus frame JSON path (default: data/refit/nci_v2/pilot_v1/frame.json)")
    p.add_argument("--constraints-db", type=Path, default=ROOT / "data/refit/final/constraints_2026-10-18.json",
                   help="constraints DB JSON path")
    p.add_argument("--mace-model", type=Path, default=ROOT / "research/bench/tools/jdev/MACE-POLAR-1-S.model",
                   help="MACE model path")
    p.add_argument("--cohort", type=str, default="pilot",
                   choices=("pilot", "surface_fit", "calibration", "motion_validation", "main", "all", "custom"),
                   help="cohort to solve in v2-orchestrate")
    p.add_argument("--codes", type=str, default="", help="comma-separated list of CSD codes to solve")
    p.add_argument("--seeds", type=str, default="11,12,13", help="comma-separated integer seeds")
    p.add_argument("--arms", type=str, default="control", help="comma-separated arms (control, incumbent, candidate)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/refit/nci_v2/orchestrator_runs",
                   help="output directory for v2-orchestrate pools and manifest")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"), help="device for solve")
    p.add_argument("--timeout", type=float, default=1800.0, help="per-solve timeout in seconds")
    p.add_argument("--batch-size", type=int, default=16, help="batch size for task queue")
    p.add_argument("--force-recompute", action="store_true", help="force recompute completed tasks")
    p.add_argument("--no-mace", action="store_true", help="disable MACE energy evaluation")
    p.add_argument("--quick", action="store_true", help="quick verification mode for v2-pilot")
    a = p.parse_args()

    if a.stage in ("pilot", "run"):
        if a.workers < 1:
            p.error("workers must be positive")
        results = regenerate(a.out, a.workers, a.limit if a.stage == "pilot" else None, pilot=a.stage == "pilot")
        if a.stage == "run":
            fit(a.out, results)
            report_census(a.out, results, a.workers)
            manifest = json.loads((a.out / "manifest.json").read_text())
            manifest["status"] = "complete"
            write_json(a.out / "manifest.json", manifest)
        return

    if a.stage == "v2-registry":
        from refit import nci_v2_registry
        exp_dir = ROOT / "data" / "refit" / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        index_path = exp_dir / "index.json"
        readme_path = exp_dir / "README.md"
        dep_audit_path = exp_dir / "dependency_audit.json"

        print(f"v2-registry: Building historical experiment records -> {index_path}...")
        records = nci_v2_registry.build_historical_records()
        write_json(index_path, records)

        print(f"v2-registry: Rendering README.md -> {readme_path}...")
        md = nci_v2_registry.render_markdown(records)
        readme_path.write_text(md, encoding="utf-8")

        print(f"v2-registry: Auditing tools/jdev/budget dependencies -> {dep_audit_path}...")
        dep_audit = nci_v2_registry.audit_budget_dependencies(repo_root=ROOT)
        write_json(dep_audit_path, dep_audit)

        print(f"v2-registry: Verifying {index_path} schema and checksums...")
        errors = nci_v2_registry.validate_registry(records, repo_root=ROOT)
        if errors:
            print(f"FAILED: Found {len(errors)} validation errors:")
            for err in errors:
                print(f"  - {err}")
            raise SystemExit(1)
        print("v2-registry: All records verified successfully.")
        return

    if a.stage == "v2-frame":
        from refit import nci_v2_frame
        clean_dir = ROOT / "data" / "refit" / "clean"
        out_frame = a.frame
        print(f"v2-frame: Building and validating frame -> {out_frame}...")
        frame_data = nci_v2_frame.build_nci_v2_frame(clean_dir, out_frame)
        print(f"v2-frame: Complete. Shards audited: {len(frame_data['shard_audit']['shards'])}, pilot parents: {len(frame_data['pilot_cohort']['parents'])}")
        return

    if a.stage == "v2-pilot":
        from refit import nci_v2_pilot
        print(f"v2-pilot: Running Step 1 repeatability & budget pilot...")
        report = nci_v2_pilot.run_pilot(
            spec_path=a.spec,
            frame_path=a.frame,
            quick=a.quick,
        )
        print(f"v2-pilot: Pilot complete. Status: {report['repeatability']['status']}")
        return

    if a.stage == "v2-scaling":
        from refit import nci_v2_scaling
        print("v2-scaling: Running Step 1 admission scaling & support feasibility evaluation...")
        report = nci_v2_scaling.run_scaling(
            spec_path=a.spec,
            frame_path=a.frame,
        )
        print(f"v2-scaling: Complete. Overall compliance: {report['step_1_admission_scaling']['overall_compliance']}")
        return

    if a.stage in ("orchestrate", "v2-orchestrate"):
        from refit import nci_v2_orchestrator
        codes_list = [c.strip() for c in a.codes.split(",") if c.strip()] if a.codes else None
        seeds_list = [int(s.strip()) for s in a.seeds.split(",") if s.strip()] if a.seeds else [11, 12, 13]
        arms_list = [arm.strip() for arm in a.arms.split(",") if arm.strip()] if a.arms else ["control"]
        out_d = a.out_dir if a.out_dir else ROOT / "data/refit/nci_v2/orchestrator_runs"

        print("v2-orchestrate: Launching production solve orchestrator...")
        manifest = nci_v2_orchestrator.run_orchestrator(
            spec_path=a.spec,
            frame_path=a.frame,
            constraints_db=a.constraints_db,
            mace_model=a.mace_model,
            cohort=a.cohort,
            codes=codes_list,
            seeds=seeds_list,
            arms=arms_list,
            out_dir=out_d,
            device=a.device,
            eval_mace=not a.no_mace,
            timeout=a.timeout,
            batch_size=a.batch_size,
            force_recompute=a.force_recompute,
            workers=a.workers if a.workers in (1, 2, 3) else 2,
        )
        print(f"v2-orchestrate: Complete. Succeeded: {manifest['succeeded_tasks']}/{manifest['total_tasks']}")
        return


if __name__ == "__main__":
    main()
