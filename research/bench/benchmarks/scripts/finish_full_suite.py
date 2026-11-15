"""Resume and finish the complete Architector benchmark suite.

Monitors the checkpointed pilot/full-run manifests from the active setup, runs
only case/seed pairs that are still missing, and adds the required assembly-
only challenge S03 jobs. All outputs stay in fresh timestamped result dirs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PIX = Path.home() / ".pixi" / "bin" / "pixi"
SEEDS = (17, 29, 43)
TIMEOUT = 600
WORKERS = 6
MONITORED = {
    "baseline": (
        "20261020T083727853783Z",
        "20261020T084406274970Z",
        "20261020T084406284540Z",
        "20261020T084406289821Z",
    ),
    "challenge": (
        "20261020T083727855300Z",
        "20261020T084406279113Z",
        "20261020T084406291275Z",
        "20261020T084406296099Z",
    ),
}
ORCHESTRATION_ID = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
ORCHESTRATION_DIR = ROOT / "results" / "orchestration" / ORCHESTRATION_ID
ORCHESTRATION_DIR.mkdir(parents=True, exist_ok=False)
STATE_PATH = ORCHESTRATION_DIR / "state.json"
LOG_PATH = ORCHESTRATION_DIR / "orchestrator.log"


def log(message: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    line = f"{stamp} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def manifest_paths(suite: str) -> list[Path]:
    return [ROOT / "results" / suite / "relaxed" / run_id / "run.json"
            for run_id in MONITORED[suite]]


def status_map(suite: str, track: str = "relaxed") -> dict[tuple[str, int], dict]:
    records: dict[tuple[str, int], dict] = {}
    paths = manifest_paths(suite) if track == "relaxed" else list(
        (ROOT / "results" / suite / track).glob("*/run.json"))
    for path in paths:
        data = read_manifest(path)
        if not data:
            continue
        for job in data.get("jobs", []):
            if "case_id" in job and "seed" in job and "status" in job:
                records[(job["case_id"], int(job["seed"]))] = {
                    **job, "run_manifest": str(path.relative_to(ROOT))
                }
    return records


def active_dirs(suite: str) -> list[Path]:
    dirs = []
    for run_id in MONITORED[suite]:
        base = ROOT / "results" / suite / "relaxed" / run_id
        if not base.exists():
            continue
        for input_path in base.glob("*/*/input.json"):
            jobdir = input_path.parent
            if not (jobdir / "result.json").exists() and not (jobdir / "capability.json").exists():
                dirs.append(jobdir)
    return dirs


def monitor_existing() -> None:
    """Let already-running batch commands finish, or resume after interruption."""
    while True:
        pending = []
        now = time.time()
        for suite in MONITORED:
            for jobdir in active_dirs(suite):
                newest = max((p.stat().st_mtime for p in jobdir.iterdir() if p.is_file()), default=now)
                if now - newest < TIMEOUT + 90:
                    pending.append((suite, jobdir))
        if not pending:
            return
        log(f"Waiting for {len(pending)} already-started case/seed jobs to checkpoint")
        time.sleep(30)


def case_ids(suite: str) -> list[str]:
    path = ROOT / "data" / ("challenge" if suite == "challenge" else "") / "cases.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row["id"] for row in data["cases"]]


def execute_one(suite: str, case_id: str, seed: int, no_final_relax: bool = False) -> dict:
    out_dir = ORCHESTRATION_DIR / "logs" / suite / case_id / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "runner.log"
    command = [str(PIX), "run", "--environment", "default", "python",
               "scripts/run_architector.py", "--suite", suite, "--case", case_id,
               "--seed", str(seed), "--timeout", str(TIMEOUT)]
    if no_final_relax:
        command.append("--no-final-relax")
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, check=False)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    run_dir = None
    for line in reversed(text.splitlines()):
        if line.startswith("Results: "):
            run_dir = Path(line.removeprefix("Results: ").strip())
            break
    record = {"suite": suite, "case_id": case_id, "seed": seed,
              "no_final_relax": no_final_relax, "exit_code": completed.returncode,
              "runner_log": str(log_path.relative_to(ROOT))}
    if run_dir and (run_dir / "run.json").exists():
        record["run_dir"] = str(run_dir.relative_to(ROOT))
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if data.get("jobs"):
            record.update(data["jobs"][-1])
    else:
        record["status"] = "orchestrator_error"
    return record


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def audit_relaxed_structures() -> dict:
    """Catch malformed coordinates and catastrophic overlaps in saved XYZ files."""
    # Covalent radii in angstroms; this deliberately conservative screen is
    # only intended to catch grossly collapsed geometries, not judge chemistry.
    radii = {"H": .31, "B": .85, "C": .76, "N": .71, "O": .66, "F": .57,
             "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
             "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
             "Mn": 1.39, "Cr": 1.39, "V": 1.53, "Ti": 1.60, "Mo": 1.54,
             "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
             "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
             "Au": 1.36, "Hg": 1.32, "Y": 1.90, "La": 2.07, "Ce": 2.04,
             "Gd": 1.96, "U": 1.96}
    # Keep each suite's relaxed track as the audit boundary.
    files = sorted(path for suite in ("baseline", "challenge")
                   for path in (ROOT / "results" / suite / "relaxed").glob("*/*/structure_*.xyz"))
    checked, malformed, overlaps = [], [], []
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            count = int(lines[0].strip())
            atoms = []
            for line in lines[2:2 + count]:
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError("short coordinate row")
                symbol = parts[0]
                xyz = tuple(float(value) for value in parts[1:4])
                if symbol not in radii or not all(map(math.isfinite, xyz)):
                    raise ValueError(f"unsupported element or non-finite coordinates: {symbol}")
                atoms.append((symbol, xyz))
            if len(atoms) != count:
                raise ValueError(f"expected {count} atoms, found {len(atoms)}")
            record = {"file": str(path.relative_to(ROOT)), "atoms": count}
            for i, (sym_a, xyz_a) in enumerate(atoms):
                for j in range(i + 1, len(atoms)):
                    sym_b, xyz_b = atoms[j]
                    distance = math.sqrt(sum((xyz_a[k] - xyz_b[k]) ** 2 for k in range(3)))
                    threshold = .5 * (radii[sym_a] + radii[sym_b])
                    if distance < threshold:
                        overlaps.append({**record, "atom_i": i, "atom_j": j,
                                         "elements": [sym_a, sym_b], "distance_A": round(distance, 4),
                                         "threshold_A": round(threshold, 4)})
            checked.append(record)
        except Exception as exc:
            malformed.append({"file": str(path.relative_to(ROOT)), "error": str(exc)})
    report = {"description": "Basic XYZ/finite-coordinate and severe-overlap screen; not a chemistry or energy validation.",
              "structures_checked": len(checked), "malformed_or_nonfinite": malformed,
              "severe_overlaps": overlaps}
    (ORCHESTRATION_DIR / "relaxed-geometry-screen.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    log(f"Monitoring checkpointed runs; concurrency={WORKERS}, timeout={TIMEOUT}s/job")
    monitor_existing()

    baseline_ids = case_ids("baseline")
    challenge_ids = case_ids("challenge")
    missing = []
    for suite, ids in (("baseline", baseline_ids), ("challenge", challenge_ids)):
        known = status_map(suite)
        for case_id in ids:
            for seed in SEEDS:
                if (case_id, seed) not in known:
                    missing.append((suite, case_id, seed, False))

    challenge_unrelaxed = status_map("challenge", "unrelaxed")
    for seed in SEEDS:
        if ("S03", seed) not in challenge_unrelaxed:
            missing.append(("challenge", "S03", seed, True))

    state = {"orchestration_id": ORCHESTRATION_ID, "seeds": SEEDS,
             "workers": WORKERS, "timeout_seconds": TIMEOUT,
             "initial_missing_jobs": len(missing), "completed_jobs": [],
             "source_manifests": {suite: [str(p.relative_to(ROOT)) for p in manifest_paths(suite)]
                                  for suite in MONITORED}}
    save_state(state)
    log(f"Starting {len(missing)} missing case/seed attempts")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(execute_one, suite, case_id, seed, no_relax):
                   (suite, case_id, seed, no_relax)
                   for suite, case_id, seed, no_relax in missing}
        for future in as_completed(futures):
            requested = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # retain scheduler errors rather than dropping a job
                record = {"suite": requested[0], "case_id": requested[1],
                          "seed": requested[2], "no_final_relax": requested[3],
                          "status": "orchestrator_error", "error": repr(exc)}
            state["completed_jobs"].append(record)
            save_state(state)
            log(f"{record['suite']} {record['case_id']} seed={record['seed']}: {record.get('status')}")

    log("All missing generation jobs are checkpointed")
    audit = audit_relaxed_structures()
    log(f"Relaxed geometry screen: checked={audit['structures_checked']}, "
        f"malformed={len(audit['malformed_or_nonfinite'])}, severe_overlaps={len(audit['severe_overlaps'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
