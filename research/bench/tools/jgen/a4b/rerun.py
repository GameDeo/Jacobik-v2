"""A4b + A4t rerun on Perlmutter, split by hardware.

1. Solves (GPU, free login-node A100), through the unified runner; rerun the same command until every parent is done
   (the runner resumes finished parents):
     python -m tools.jgen.a4b.rerun codes --inputs DIR > codes.txt
     ~/scripts/jacobik/jk_run.sh a4r-solve panel --codes $(cat codes.txt) \
         --arms-spec '[{"arm":"control","flags":{}}]' --seeds 11,12,13 --mps --no-pack
2. Everything after the solves (CPU node; $SCRATCH is mounted there, nothing is copied):
     python -m tools.jgen.a4b.rerun post --solve $SCRATCH/jacobik/runs/a4r-solve --inputs DIR --out OUT --workers 128
   = A4b relaxations (GFN2 crude, <= 5 rows per unit; --select a4c (default): class-aware rule a4c.relax_select;
   --select a4b: the frozen A4b rule) + MACE on CPU across processes -> A4b tables and report -> A4c evaluation
   (OUT/a4c: parents.json, summary.json; a4c only) -> A4t parents (cal parents missed on every seed: C-MISS with
   a4c, R-MISS with a4b) -> A4t full-pool relaxations (A4b rows reused) + MACE -> A4t analysis -> one tar of OUT.

DIR holds the frozen A4b inputs: charge_mult.json, vm_split.json (cal/tail membership), fallback_codes.json.
The records must carry the current production fingerprint (JACOBIK_A4_FINGERPRINT, set by `post` from jacobik.core.profile).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

INPUT_FILES = ("charge_mult.json", "vm_split.json", "fallback_codes.json")
#: unit seeds; the P2R fold-in gate runs fresh seeds 31-33 (JACOBIK_A4_SEEDS="31,32,33")
SEEDS = tuple(int(x) for x in os.environ.get("JACOBIK_A4_SEEDS", "11,12,13").split(","))


def _locked() -> set[str]:
    """Reserve / motion_validation codes: never run (A4b 2026-11-01 predates the lock and included 5 of them)."""
    from tools.run.__main__ import locked_codes
    return set(locked_codes())


def panel_codes(inputs: Path) -> list[str]:
    split = json.loads((inputs / "vm_split.json").read_text())
    return sorted((set(split["cal_parents"]) | set(split["tail_parents"])) - _locked())


def cmd_codes(a) -> int:
    print(",".join(panel_codes(Path(a.inputs))))
    return 0


def _log(msg, t0):
    print(f"[a4-rerun {time.perf_counter() - t0:7.1f} s] {msg}", flush=True)


def stage_data(solve: Path, inputs: Path, data: Path, fingerprint: str, arm_name: str = "control") -> dict:
    """data/ = A4b layout: records/ and xyz/ linked from the runner's control arm, plus the frozen inputs."""
    arm = solve / arm_name
    if not (arm / "records").is_dir():
        raise SystemExit(f"{arm}/records missing: run the solve stage first (arm directory {arm_name!r})")
    data.mkdir(parents=True, exist_ok=True)
    for name in ("records", "xyz"):
        link = data / name
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to((arm / name).resolve(), target_is_directory=True)
    for f in INPUT_FILES:
        shutil.copy2(inputs / f, data / f)
    # panel membership without locked codes (the tables use vm_split.json as the panel contract)
    split = json.loads((inputs / "vm_split.json").read_text())
    locked = _locked()
    split["cal_parents"] = [c for c in split["cal_parents"] if c not in locked]
    split["tail_parents"] = [c for c in split["tail_parents"] if c not in locked]
    split["locked_excluded"] = sorted(locked & (set(json.loads((inputs / "vm_split.json").read_text())["cal_parents"])
                                                | set(json.loads((inputs / "vm_split.json").read_text())["tail_parents"])))
    (data / "vm_split.json").write_text(json.dumps(split, indent=1))
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[5]), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    if commit == "unknown" and (Path(__file__).resolve().parents[5] / "COMMIT").exists():
        commit = (Path(__file__).resolve().parents[5] / "COMMIT").read_text().strip()
    import hashlib
    db = os.environ.get("JACOBIK_DB", "")
    db_sha = hashlib.sha256(Path(db).read_bytes()).hexdigest() if db and Path(db).is_file() else "unknown"
    meta = dict(commit=commit[:12], fingerprint=fingerprint, db=db, db_sha256=db_sha, vm_type="Perlmutter login A100-PCIE-40GB (solve); CPU node "
                "(GFN2 + MACE)", seeds=list(SEEDS), solve_dir=str(solve))
    (data / "run_meta.json").write_text(json.dumps(meta, indent=1))
    codes = panel_codes(inputs)
    recs = {p.stem for p in (data / "records").glob("*.json")}
    missing = [f"{c}_{s}" for c in codes for s in SEEDS if f"{c}_{s}" not in recs]
    return dict(codes=len(codes), units_expected=3 * len(codes), units_found=len(recs), missing=missing)


def a4t_parents(data: Path) -> list[str]:
    """Cal parents whose relaxed bucket is R-MISS on every seed (the A4t population; A4t 2026-11-02 used 35)."""
    split = json.loads((data / "vm_split.json").read_text())
    cal = set(split["cal_parents"])
    by = defaultdict(list)
    with open(data / "units.csv") as fh:
        for u in csv.DictReader(fh):
            if u.get("panel") == "cal" and u["code"] in cal and u.get("status") == "ok":
                by[u["code"]].append(u.get("r_bucket"))
    return sorted(c for c, b in by.items() if len(b) == len(SEEDS) and all(x == "R-MISS" for x in b))


def a4c_missed_parents(a4c_dir: Path) -> list[str]:
    """Cal parents whose A4c bucket is C-MISS on every seed (the A4t population under the class-aware metric)."""
    from tools.jgen.a4c.summary import joined
    _, units, _, _ = joined(str(Path(a4c_dir).parent / "a4b"), str(a4c_dir))
    by = defaultdict(list)
    for u in units:
        if u["panel"] == "cal":
            by[u["code"]].append(u["bucket"])
    return sorted(c for c, b in by.items() if len(b) == len(SEEDS) and all(x == "C-MISS" for x in b))


def cmd_post(a) -> int:
    t0 = time.perf_counter()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    import jacobik.core.profile as profile
    fp = a.fingerprint or profile.snapshot()["fingerprint"]
    os.environ["JACOBIK_A4_FINGERPRINT"] = fp
    # imported after the fingerprint is set: the A4 modules read it at import
    from tools.jgen.a4b import relax_run as a4b_relax, tables, report
    from tools.jgen.a4t import relax_run as a4t_relax, analyse as a4t_analyse
    from tools.runkit import mace

    solve, inputs, out = Path(a.solve), Path(a.inputs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "a4b"
    timing = {}
    st = stage_data(solve, inputs, data, fp, a.arm)
    _log(f"staged: fingerprint {fp}, {st['units_found']}/{st['units_expected']} unit records "
         f"({len(st['missing'])} missing)", t0)
    (out / "stage.json").write_text(json.dumps(st, indent=1))
    n_match = 0
    for p in (data / "records").glob("*.json"):
        try:
            r = json.loads(p.read_text())
            n_match += r.get("status") == "ok" and r.get("fingerprint") == fp
        except Exception:
            pass
    if n_match == 0:   # e.g. a flag arm posted without --fingerprint: every unit would be rejected
        raise SystemExit(f"no ok record carries fingerprint {fp}: pass the arm's --fingerprint")
    _log(f"{n_match} ok records carry fingerprint {fp}", t0)
    # MACE float64 either way; cuda (a login-node A100) is ~100x faster than the CPU pool (energies equal to ~1e-6 kJ/mol)
    ev = mace.Evaluator(device="cuda") if a.mace_device == "cuda" else mace.PoolEvaluator(a.workers)

    # A4b: frozen selection (top-1, floor, up to 3 rows within 1.5 A) per unit
    t = time.perf_counter()
    only = set(a.codes.split(",")) if a.codes else None
    manifests = sorted(str(p) for p in (data / "xyz").rglob("*_pool.json") if only is None or p.parent.name in only)
    cm = json.loads((data / "charge_mult.json").read_text())
    prev = Path(a.resume_from) if a.resume_from else None
    if prev is not None and prev.resolve() == out.resolve():
        raise SystemExit("--resume-from must be a different directory than --out")
    m1 = a4b_relax.process(manifests, data / "relax", cm, nproc=a.workers + 1, evaluator=ev,
                           resume=(prev / "a4b" / "relax") if prev else None, selector=a.select)
    timing["a4b_relax_s"] = time.perf_counter() - t
    _log(f"A4b relax: {json.dumps({k: v for k, v in m1.items() if k != 'mace'})}", t0)

    t = time.perf_counter()
    tables.build(str(data))
    report.build_report(str(data), str(Path(__file__).resolve().parents[5]), str(out / "A4B_REPORT.md"))
    timing["a4b_tables_s"] = time.perf_counter() - t
    _log("A4b tables and report written", t0)

    if a.select == "a4c":
        from tools.jgen.a4c import baseline as a4c_baseline
        t = time.perf_counter()
        s4 = a4c_baseline.evaluate(data, out / "a4c", workers=a.workers, codes=sorted(only) if only else None)
        timing["a4c_eval_s"] = time.perf_counter() - t
        _log("A4c evaluation: " + json.dumps({p: (s4[p].get("miss") or {}).get("mean") for p in ("cal", "tail") if p in s4}), t0)

    if a.a4t_parents:        # paired gate: the control's A4t population, not this arm's own R-MISS set
        parents = [c for c in json.loads(Path(a.a4t_parents).read_text()) if only is None or c in only]
    elif a.select == "a4c":
        parents = [c for c in a4c_missed_parents(out / "a4c") if only is None or c in only]
    else:
        parents = [c for c in a4t_parents(data) if only is None or c in only]
    (out / "a4t_parents.json").write_text(json.dumps(parents))
    _log(f"A4t population: {len(parents)} cal parents missed on all seeds ({a.select})", t0)
    if parents and not a.skip_a4t:
        t = time.perf_counter()
        m2 = a4t_relax.run(str(data), str(out / "a4t" / "relax"), parents, nproc=a.workers + 1, device="cpu",
                           resume=(prev / "a4t" / "relax") if prev else None,
                           evaluator=ev)
        timing["a4t_relax_s"] = time.perf_counter() - t
        _log(f"A4t relax: {json.dumps({k: v for k, v in m2.items() if k not in ('mace', 'parents')})}", t0)
        t = time.perf_counter()
        a4t_analyse.analyse(str(data), str(out / "a4t"), str(out / "a4t" / "analysis"), parents=parents, workers=a.workers)
        timing["a4t_analyse_s"] = time.perf_counter() - t
        _log("A4t analysis written", t0)
        if a.analysis2:
            from tools.jgen.a4t import analyse2 as a4t_analyse2
            t = time.perf_counter()
            a4t_analyse2.analyse2(str(data), str(out / "a4t"), str(out / "a4t" / "analysis_v2"), parents, workers=a.workers)
            timing["a4t_analyse2_s"] = time.perf_counter() - t
            _log("A4t v2 analysis written", t0)
    timing["total_s"] = time.perf_counter() - t0
    timing["mace"] = ev.stats
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    if not a.no_pack:
        tar = out.with_suffix(".tar")
        with tarfile.open(tar, "w") as tf:
            tf.add(out, arcname=out.name)
            tf.add(solve, arcname=out.name + "/solve")
        _log(f"packed {tar}", t0)
    _log("done", t0)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.jgen.a4b.rerun", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("codes", help="print the A4b panel (cal + tail) as a comma list")
    c.add_argument("--inputs", required=True)
    q = sub.add_parser("post", help="relaxations, MACE, tables, A4t (CPU node)")
    q.add_argument("--solve", required=True, help="tools.run output dir of the solve stage")
    q.add_argument("--inputs", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    q.add_argument("--skip-a4t", action="store_true")
    q.add_argument("--codes", default="", help="smoke test: only these parents")
    q.add_argument("--fingerprint", default="", help="records' fingerprint (default: the current production one)")
    q.add_argument("--no-pack", action="store_true")
    q.add_argument("--resume-from", default="", help="a previous post OUT: reuse its relaxed rows/references and their "
                   "MACE energies (xtb + MACE only for what it lacks)")
    q.add_argument("--arm", default="control", help="arm directory inside --solve (the chair arm is solved as its own run)")
    q.add_argument("--a4t-parents", default="", help="JSON list of A4t parents (paired gate: the control's 36); default: this arm's R-MISS set")
    q.add_argument("--mace-device", choices=("cpu", "cuda"), default="cpu",
                   help="cuda: MACE single points on the GPU (login-node dev loop); cpu: process pool (CPU node)")
    q.add_argument("--select", choices=("a4c", "a4b"), default="a4c",
                   help="relax-set rule: a4c (class-aware, P2R Step 2) or a4b (frozen A4b rule; historical reruns)")
    q.add_argument("--analysis2", action="store_true", help="also run the v2 A4t analysis (a4t/analyse2.py) into a4t/analysis_v2")
    a = p.parse_args(argv)
    return dict(codes=cmd_codes, post=cmd_post)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
