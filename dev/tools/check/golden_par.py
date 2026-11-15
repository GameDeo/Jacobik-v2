"""Golden check in parallel: one process per (profile x complex group), each with the 8-part CPU layout.

Same arrays, tolerances and reference files as dev/tools/jdev/golden.py (which is not modified). The
groups partition the complexes of golden.RESCORE and golden.SOLVE; a complex's arrays are computed in one process,
so the result does not depend on how many processes run at once.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .common import REPO, base_env

#: complex groups (each code in RESCORE/SOLVE appears in exactly one group). Fine-grained so the slowest solves
#: (aqua_La3_8H2O, YAVFEG, SOKVET) run beside the rest instead of serialising one process.
GROUPS = {
    "aqua": dict(solve=["aqua_La3_8H2O"], rescore=[]),
    "yav": dict(solve=["YAVFEG"], rescore=[]),
    "sok": dict(solve=["SOKVET"], rescore=["SOKVET"]),
    "cbt": dict(solve=["CBTYCO"], rescore=["CBTYCO"]),
    "kiq": dict(solve=["KIQYEQ"], rescore=["KIQYEQ"]),
    "a": dict(solve=["OJIMIF", "HUYNEU"], rescore=["OJIMIF", "HUYNEU"]),
    "d1": dict(solve=[], rescore=["APEBIJ", "IYONIW", "PEWBOL"]),
    "d2": dict(solve=[], rescore=["XAQRUC", "DEVFUG", "COPYIP"]),
}
#: rough per-group wall (s, one process, workstation) used only to start the longest groups first
COST = {"aqua": 200, "yav": 120, "sok": 90, "cbt": 70, "kiq": 40, "a": 35, "d1": 40, "d2": 40}
PROFILES = ("production", "u1")
#: Differences that are EXPECTED and reported, not failures, until the owner re-records the reference. Each entry must
#: match exactly (key and shape change); anything else is a failure. Empty: research/data/golden/ was re-recorded on the
#: solver fix 959b4bb (2026-11-08), so CBTYCO's 13 -> 30 change is in the reference.
KNOWN_PENDING: dict = {}
TOL_SCORE, TOL_X = 1e-6, 1e-5


def _golden_module():
    sys.path.insert(0, str(REPO / "dev/tools/jdev"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("golden_ref", REPO / "dev/tools/jdev/golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module level only: defines constants, does not run the check
    return mod


def _child(profile: str, group: str, out: Path) -> int:
    """Child process: compute one profile x group and save the arrays (the profile is read at import time)."""
    from threadpoolctl import threadpool_limits
    g = _golden_module()
    import os
    os.environ.update(g.golden_environment({}, 8))
    t0 = time.time()
    got, per_code = {}, {}
    with threadpool_limits(limits=1):
        for kind in ("rescore", "solve"):
            for code in GROUPS[group][kind]:
                g.RESCORE, g.SOLVE = ([code], []) if kind == "rescore" else ([], [code])
                t1 = time.time()
                got.update(g._compute(profile))
                per_code[f"{kind}|{code}"] = round(time.time() - t1, 1)
    np.savez_compressed(out, **{k.replace("|", "__"): v for k, v in got.items()})
    print(json.dumps(dict(profile=profile, group=group, arrays=len(got), wall_s=time.time() - t0,
                          per_code_s=per_code)), flush=True)
    return 0


def _group_of(key: str) -> str | None:
    code = key.split("|")[1] if "|" in key else None
    for name, spec in GROUPS.items():
        if code in spec["rescore"] or code in spec["solve"]:
            return name
    return None


def _compare(profile: str, group: str, got: dict, ref_npz) -> dict:
    ref = {k.replace("__", "|"): v for k, v in ref_npz.items()}
    ref = {k: v for k, v in ref.items() if k.startswith(("rescore|", "solve|")) and _group_of(k) == group}
    bad = []
    for k in sorted(set(ref) | set(got)):
        if k not in ref or k not in got:
            bad.append(f"{k}: {'missing now' if k not in got else 'new'}")
            continue
        a, b = ref[k], got[k]
        tol = TOL_X if k.endswith("|X") else TOL_SCORE
        if a.shape != b.shape:
            bad.append(f"{k}: shape {a.shape} -> {b.shape}")
        elif not np.allclose(a, b, atol=tol, rtol=0, equal_nan=True):
            bad.append(f"{k}: max |diff| {float(np.nanmax(np.abs(a - b))):.3g} (tol {tol})")
    return dict(arrays_ref=len(ref), differing=bad)


def run_parallel(golden_dir: Path, jobs: int, groups: list[str], out_dir: Path, profiles=PROFILES) -> dict:
    """Run every (profile, group) child with `jobs` concurrent processes; compare to golden_dir/<profile>.npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(((p, g) for p in profiles for g in groups), key=lambda t: -COST.get(t[1], 0))

    def one(task):
        profile, group = task
        out = out_dir / f"{profile}__{group}.npz"
        env = base_env({"JACOBIK_PROFILE": profile, "JACOBIK_GOLDEN_DIR": str(golden_dir),
                        "JACOBIK_NUM_WORKERS": "8", "JACOBIK_WORKER_MODE": "thread",
                        "MKL_THREADING_LAYER": "SEQUENTIAL"})
        log = out_dir / f"{profile}__{group}.log"
        with log.open("w") as fh:
            rc = subprocess.run([sys.executable, "-m", "tools.check.golden_par", "child", profile, group, str(out)],
                                cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        return dict(profile=profile, group=group, rc=rc, out=str(out), log=str(log))

    results = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for r in pool.map(one, tasks):
            results.append(r)
    # compare against the reference files
    details, arrays_ref, differing, failed = [], 0, [], []
    for profile in profiles:
        ref_path = golden_dir / f"{profile}.npz"
        if not ref_path.exists():
            failed.append(f"{profile}: reference missing {ref_path}")
            continue
        ref_npz = np.load(ref_path)
        for group in groups:
            r = next(x for x in results if x["profile"] == profile and x["group"] == group)
            if r["rc"] != 0:
                failed.append(f"{profile}/{group}: child exit {r['rc']} (log {r['log']})")
                continue
            got = {k.replace("__", "|"): v for k, v in np.load(r["out"]).items()}
            c = _compare(profile, group, got, {k: ref_npz[k] for k in ref_npz.files})
            arrays_ref += c["arrays_ref"]
            differing += [f"{profile}/{group}: {x}" for x in c["differing"]]
            details.append(dict(profile=profile, group=group, arrays=c["arrays_ref"], differing=len(c["differing"])))
    known = [d for d in differing if _is_known(d)]
    unexpected = [d for d in differing if not _is_known(d)]
    return dict(arrays_ref=arrays_ref, differing=differing, unexpected=unexpected, known_pending=known,
                failed=failed, groups=details, children=results)


def _is_known(line: str) -> bool:
    # line: "<profile>/<group>: <key>: <what>"
    try:
        _pg, key, what = line.split(": ", 2)
    except ValueError:
        return False
    return KNOWN_PENDING.get(key) == what


def record(golden_dir: Path, jobs: int = 4, groups: list[str] | None = None) -> dict:
    """Record reference files <profile>.npz into an empty golden_dir (refuses to overwrite). Same layout as golden.py."""
    golden_dir = Path(golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)
    groups = groups or list(GROUPS)
    clash = [p for p in PROFILES if (golden_dir / f"{p}.npz").exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite existing references in {golden_dir}: {clash}")
    tmp = golden_dir / "_parts"
    tasks = [(p, g) for p in PROFILES for g in groups]

    def one(task):
        profile, group = task
        out = tmp / f"{profile}__{group}.npz"
        env = base_env({"JACOBIK_PROFILE": profile, "JACOBIK_NUM_WORKERS": "8", "JACOBIK_WORKER_MODE": "thread",
                        "MKL_THREADING_LAYER": "SEQUENTIAL"})
        tmp.mkdir(parents=True, exist_ok=True)
        log = tmp / f"{profile}__{group}.log"
        with log.open("w") as fh:
            rc = subprocess.run([sys.executable, "-m", "tools.check.golden_par", "child", profile, group, str(out)],
                                cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        return rc, profile, group, out, log

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(one, tasks))
    bad = [(p, g, rc, str(l)) for rc, p, g, _o, l in results if rc != 0]
    if bad:
        raise SystemExit(f"child failures (not recorded): {bad}")
    for profile in PROFILES:
        merged = {}
        for rc, p, g, out, _l in results:
            if p == profile:
                merged.update({k: v for k, v in np.load(out).items()})
        np.savez_compressed(golden_dir / f"{profile}.npz", **merged)
    return dict(dir=str(golden_dir), arrays={p: len(np.load(golden_dir / f"{p}.npz").files) for p in PROFILES})


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "child":
        sys.exit(_child(sys.argv[2], sys.argv[3], Path(sys.argv[4])))
    if len(sys.argv) == 3 and sys.argv[1] == "record":
        print(json.dumps(record(Path(sys.argv[2]))))
        raise SystemExit(0)
    raise SystemExit("usage: python -m tools.check.golden_par record <empty-dir>")
