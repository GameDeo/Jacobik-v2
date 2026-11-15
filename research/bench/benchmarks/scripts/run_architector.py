"""Run isolated, time-limited Architector jobs on the local catalog; never downloads data."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from catalog import ROOT, UnsupportedInput, catalog_directory, load_catalog, make_input, read_json, write_json


def environment():
    versions = {}
    for name in ("architector", "xtb", "ase", "numpy", "openbabel", "pynauty", "mendeleev", "rdkit"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


def scalar_fields(row):
    """Preserve upstream scalar energies and timings without serializing ASE objects."""
    scalars = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, TypeError):
                continue
        if value is None or isinstance(value, (str, bool, int)):
            if key not in ("mol2string", "init_mol2string"):
                scalars[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            scalars[key] = value
    return scalars


def worker(input_path):
    jobdir = input_path.parent
    start = time.perf_counter()
    result = {"status": "exception", "structures": [], "environment": environment()}
    try:
        from architector import build_complex
        result["import_seconds"] = time.perf_counter() - start
        specification = read_json(input_path)
        build_start = time.perf_counter()
        output = build_complex(specification)
        result["build_seconds"] = time.perf_counter() - build_start
        if not isinstance(output, dict):
            raise TypeError(f"Expected Architector dictionary, received {type(output).__name__}")
        for rank, (key, row) in enumerate(output.items()):
            stem = f"structure_{rank:03d}"
            if row.get("mol2string"):
                (jobdir / f"{stem}.mol2").write_text(row["mol2string"], encoding="utf-8")
            if row.get("init_mol2string"):
                (jobdir / f"{stem}_initial.mol2").write_text(row["init_mol2string"], encoding="utf-8")
            atoms = row.get("ase_atoms")
            if atoms is not None:
                atoms.write(str(jobdir / f"{stem}.xyz"), format="xyz")
            result["structures"].append({"rank": rank, "upstream_key": str(key), "files_stem": stem,
                                         "scalars": scalar_fields(row)})
        result["status"] = "returned_structures" if output else "empty_output"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    result["worker_seconds"] = time.perf_counter() - start
    write_json(jobdir / "result.json", result)
    return 0 if result["status"] == "returned_structures" else 1


def run(args):
    cases, ligands = load_catalog(args.suite)
    known = {case["id"]: case for case in cases}
    selected_ids = list(known) if args.all else list(dict.fromkeys(args.case or []))
    if not selected_ids:
        raise SystemExit("Select --case S01 (repeatable) or --all. Use --dry-run to export inputs only.")
    unknown = set(selected_ids) - known.keys()
    if unknown:
        raise SystemExit(f"Unknown case IDs: {sorted(unknown)}")
    if not 1 <= args.n_conformers <= 50 or args.timeout <= 0:
        raise SystemExit("Use 1..50 conformers and a positive timeout in seconds.")
    if not args.dry_run:
        try:
            importlib.metadata.version("architector")
        except importlib.metadata.PackageNotFoundError:
            raise SystemExit("Architector is absent. Use the Linux default environment; Windows audit supports --dry-run.")
    seeds = list(dict.fromkeys(args.seed or [17, 29, 43]))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    track = "unrelaxed" if args.no_final_relax else "relaxed"
    run_dir = ROOT / "results" / args.suite / track / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    fingerprints = {}
    data_directory = catalog_directory(args.suite)
    for path in [data_directory / "cases.json", data_directory / "ligands.json", data_directory / "sources.json",
                 ROOT / "sources/manifest.json", ROOT / "pixi.lock", Path(__file__), ROOT / "scripts/catalog.py"]:
        fingerprints[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    runinfo = {"run_id": run_id, "suite": args.suite, "dry_run": args.dry_run, "case_ids": selected_ids, "seeds": seeds,
               "n_conformers": args.n_conformers, "timeout_seconds": args.timeout,
               "mode": "no_final_relax" if args.no_final_relax else "full_gfn2_xtb",
               "threads": 1, "geometry_input": "donor_topology_and_template_slot_count", "environment": environment(),
               "sha256": fingerprints, "jobs": []}
    write_json(run_dir / "run.json", runinfo)
    any_failure = False
    for case_id in selected_ids:
        for seed in seeds:
            jobdir = run_dir / case_id / f"seed_{seed}"
            jobdir.mkdir(parents=True)
            scratch = jobdir / "scratch"
            scratch.mkdir()
            definitions = {entry["ligand_id"]: ligands[entry["ligand_id"]]
                           for entry in known[case_id]["ligands"]}
            write_json(jobdir / "target.json", {"schema_version": 2, "suite": args.suite,
                                                "case": known[case_id], "ligand_definitions": definitions})
            record = {"case_id": case_id, "seed": seed, "directory": str(jobdir.relative_to(run_dir))}
            try:
                specification = make_input(known[case_id], ligands, seed, args.n_conformers,
                                           not args.no_final_relax, scratch)
            except UnsupportedInput as exc:
                record.update({"status": exc.status, "reason": str(exc)})
                write_json(jobdir / "capability.json", record)
                runinfo["jobs"].append(record)
                write_json(run_dir / "run.json", runinfo)
                any_failure = True
                print(f"{case_id} seed={seed}: {record['status']}", flush=True)
                continue
            input_path = jobdir / "input.json"
            write_json(input_path, specification)
            if args.dry_run:
                record["status"] = "input_exported"
            else:
                child_env = os.environ.copy()
                child_env.update({name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                         "NUMEXPR_NUM_THREADS", "OMP_MAX_ACTIVE_LEVELS")})
                child_env.update({name: str(scratch) for name in ("TMPDIR", "TMP", "TEMP")})
                start = time.perf_counter()
                with (jobdir / "worker.log").open("w", encoding="utf-8") as logfile:
                    try:
                        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(input_path)],
                                                   cwd=jobdir, env=child_env, stdout=logfile, stderr=subprocess.STDOUT,
                                                   timeout=args.timeout, check=False)
                        record["exit_code"] = completed.returncode
                        if (jobdir / "result.json").is_file():
                            result = read_json(jobdir / "result.json")
                            record.update({"status": result["status"], "structure_count": len(result["structures"])})
                        else:
                            record["status"] = "worker_crash"
                    except subprocess.TimeoutExpired:
                        record["status"] = "timeout"
                        write_json(jobdir / "result.json", {"status": "timeout", "structures": [],
                                                            "timeout_seconds": args.timeout})
                record["process_wall_seconds"] = time.perf_counter() - start
                any_failure |= record["status"] != "returned_structures"
            print(f"{case_id} seed={seed}: {record['status']}", flush=True)
            runinfo["jobs"].append(record)
            write_json(run_dir / "run.json", runinfo)
    print("Outcomes:", dict(Counter(job["status"] for job in runinfo["jobs"])))
    print(f"Results: {run_dir}")
    return 1 if any_failure and not args.dry_run else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["baseline", "challenge"], default="challenge")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--case", action="append", help="Case ID, repeatable")
    choice.add_argument("--all", action="store_true")
    parser.add_argument("--seed", action="append", type=int, help="Default: 17, 29, 43")
    parser.add_argument("--n-conformers", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-final-relax", action="store_true", help="Still uses xTB during assembly; separate comparison track")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
