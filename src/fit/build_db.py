"""build_db.py
===========
One pipeline from the raw tmQM/lnQM xyz files to a self-describing constraints DB.

    python -m fit.build_db --profile d2 --workers 64 \\
        --ref-dir data/ref --out-dir OUT --locked-path research/bench/benchmarks/locked_exclusions.json \\
        --fit-frame-path data/refit/frame_v1/frame.json [--target-db DB --check-reproduction] [--quick]
    python -m fit.build_db compare A.json B.json [--exact]

Stages (each logged with wall time, peak RSS and, on the pool, parallel efficiency; see ``fit.pipeline``):

    extract   raw xyz -> observation parquet (Q1-Q6, QO) + Q6 histograms, one task per shard, merged per split
    census    charge / oxidation-state census of every parent (per shard)
    filter    whole-parent exclusions (charge-implausible, locked) when the profile asks for them
    expand    ancestor expansion per quantity
    fit       Q1..Q6, QO, Q6P: one task list over one shared pool (heavy quantities first)
    ref_ll    reference log-likelihoods
    bends     Q7 crystal bend-angle table from the frozen fit frame
    assemble  container: entries + ``_meta`` + ``_tables`` (bends) -> constraints.json and the lossless NPZ
    reproduce optional entry-by-entry comparison with ``--target-db``
    rescore   optional 60-pool sanity check

Flags come from one place: ``--profile NAME`` (a ``jacobik.core.profile.PROFILES`` entry) and optionally ``--flags FILE.json``
set JACOBIK_PROFILE / JACOBIK_PROFILE_OVERRIDES for this process and every worker; the resolved ``profile.snapshot()`` is
stored in the DB. Inputs are ``data/ref`` plus the locked-exclusion list and the frozen fit frame (Q7); each is hashed
into the metadata. Results do not depend on ``--workers`` or on completion order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Library imports below are lazy: the profile environment must be installed before ``jacobik`` is first imported.
# (``jacobik.core.env`` itself does not import the generator; it only locates the checkout.)
from fit.pipeline.compare import PRODUCTION_QUANTITIES
from jacobik.core.env import REPO_ROOT  # the source checkout, or None for an installed package

log = logging.getLogger("build_db")

ALL_STAGE_QUANTITIES = PRODUCTION_QUANTITIES + ["Q7"]
OBS_FITTED = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO")
#: descriptive model names stored in ``_meta`` (equal to data/db/constraints_2026-11-01.meta.json)
MODEL_LABELS = {"q1_model": "gmm_pooled", "q2_model": "tmix_pooled_pairclass", "q6_model": "pure_repulsive_n1",
                "oxidation_keyed": True,
                "ladders": "jacobik.stats.ladders", "pooling": {"Q1": 4.0, "Q2": 4.0}}
SCHEMA = "jacobik.constraints_container.1"


# kept importable from here (tests, tools): the comparison lives in fit.pipeline.compare
def __getattr__(name):
    if name in ("compare_databases", "_align_mixture_components", "compare_exact", "load_container"):
        from fit.pipeline import compare

        return getattr(compare, name)
    if name == "rescore_60_pools":
        return _rescore_driver()
    raise AttributeError(name)


def _rescore_driver():
    """The optional 60-pool sanity check is a bench driver (it uses ``tools.jdev``): it needs the dev tools on the path."""
    try:
        from tools.jdev.rescore import rescore_60_pools
    except ImportError as exc:
        raise SystemExit("--rescore-pools-dir needs the dev tools (tools.jdev.rescore, with dev and research/bench on PYTHONPATH): "
                         f"{exc}") from exc
    return rescore_60_pools


def _git(*args: str) -> str:
    if REPO_ROOT is None:
        return ""
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _code_commit() -> str:
    """HEAD of the checkout, or the ``COMMIT`` file of a clean snapshot (Perlmutter runs from ``git archive`` copies, no .git)."""
    head = _git("rev-parse", "HEAD")
    if head:
        return head
    try:
        return (Path(REPO_ROOT or ".") / "COMMIT").read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _hash_task(p: dict) -> tuple:
    return p["rel"], _sha_file(Path(p["path"]))


def _hash_files(pool, files: List[tuple]) -> Dict[str, str]:
    order = sorted(files, key=lambda rp: -Path(rp[1]).stat().st_size)
    res = pool.map(_hash_task, [dict(rel=r, path=str(p)) for r, p in order])
    return dict(sorted(res))


def build(args: argparse.Namespace) -> int:
    from fit.pipeline import extract as runtime
    from fit.pipeline.extract import StageLog

    env = runtime.profile_env(args.profile, args.flags)
    runtime.pin_threads()
    runtime.apply_profile_env(env)
    workers = args.workers or os.cpu_count() or 1

    from jacobik.core import profile
    from jacobik.core import container as dbio
    from fit import q1, q5
    from jacobik.core.container import save_lossless_npz
    from fit.pipeline import compare as cmp
    from fit.pipeline import extract as ex
    from fit.pipeline import fitting as fs
    from fit.pipeline.fitting import TaskPool
    from fit.sanity import excluded_charge_codes, excluded_structure_codes, locked_codes

    t_start = time.perf_counter()
    selected = [q.upper() for q in args.quantities] if args.quantities else list(ALL_STAGE_QUANTITIES)
    unknown = sorted(set(selected) - set(ALL_STAGE_QUANTITIES))
    if unknown:
        raise SystemExit(f"unknown quantities {unknown}; choose from {ALL_STAGE_QUANTITIES}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir) if args.work_dir else out_dir / ".build_work"
    shard_dir, obs_dir = work / "shards", work / "obs"
    filtered_dir, expanded_dir = work / "filtered", work / "expanded"
    snap = profile.snapshot()
    log.info("build: profile %s (fingerprint %s), overrides %s, %d workers%s", snap["profile"], snap["fingerprint"],
             dict(profile.OVERRIDES), workers, ", QUICK subset" if args.quick else "")
    slog = StageLog(workers)
    failures: List[dict] = []
    ref_dir = Path(args.ref_dir)
    obs_q = [q for q in OBS_FITTED if q in selected]

    with TaskPool(workers, env) as pool:
        # ---- extract (or reuse merged observations)
        shards = ex.list_shards(ref_dir)
        use_shards, limit = ex.select_shards(shards, args.quick)
        if args.only_shards:  # development: a chosen subset of whole shards
            missing = sorted(set(args.only_shards) - {s.name for s in shards})
            if missing:
                raise SystemExit(f"unknown shards {missing}")
            use_shards, limit = [s for s in shards if s.name in set(args.only_shards)], None
        exclude = ex.benchmark_codes()
        extraction: Dict[str, Any] = {}
        merge_info: Dict[str, Any] = {}
        hist_dir: Path
        qo_obs_dir: Optional[Path] = None
        if args.obs_dir:
            obs_dir = Path(args.obs_dir)
            hist_dir = Path(args.hist_dir) if args.hist_dir else obs_dir
            qo_obs_dir = Path(args.qo_obs_dir) if args.qo_obs_dir else None
            log.info("using merged observations from %s (extract skipped)", obs_dir)
        else:
            chunk_plan = ex.plan_chunks(use_shards, limit)
            with slog.stage("extract", items=sum(len(v) for v in chunk_plan.values()), pool=pool):
                extraction = ex.run_extract(pool, ref_dir, shard_dir, use_shards, chunk_plan, exclude)
            bad = {k: v for k, v in extraction.items()
                   if v["n_errors"] or v.get("n_parse_failed") or any(v["quantity_errors"].values())}
            for k, v in sorted(bad.items()):
                failures.append({"stage": "extract", "shard": k, "n_errors": v["n_errors"],
                                 "n_parse_failed": v.get("n_parse_failed", 0), "quantity_errors": v["quantity_errors"]})
            with slog.stage("merge", pool=pool):
                merge_info = ex.run_merge(pool, shard_dir, obs_dir, use_shards, exclude, chunk_plan)
            hist_dir = obs_dir
        eligibility_report = _ledger_report(obs_dir)
        if args.qo_obs_dir and not qo_obs_dir:
            qo_obs_dir = Path(args.qo_obs_dir)
        if qo_obs_dir:  # QO rows from their own directory (e.g. a separate eta2 extraction)
            (work / "qo_in").mkdir(parents=True, exist_ok=True)
            for f in qo_obs_dir.glob("observations_QO_*.parquet"):
                dst = work / "qo_in" / f.name
                if not dst.exists():
                    os.symlink(f.resolve(), dst)

        if args.stop_after == "merge":
            log.info("stopping after merge (development option); observations in %s", obs_dir)
            slog_summary = slog.summary()
            (out_dir / "stop_after_merge.json").write_text(json.dumps(
                {"failures": failures, "extraction": extraction, "merge": merge_info, "eligibility": eligibility_report,
                 "stage_timings": slog_summary},
                indent=2, default=str) + "\n")
            return int(bool(failures))
        # ---- census + filter
        with slog.stage("census", items=len(use_shards), pool=pool):
            census = fs.run_census(pool, ref_dir, use_shards)
        charge_bad = excluded_charge_codes(census)
        structure_bad = excluded_structure_codes(census) if profile.get("structure_sanity") else set()
        charge_bad |= structure_bad          # whole-parent input exclusions share one filter (and its os_sanity switch)
        locked = locked_codes(Path(args.locked_path))
        by_metal: Dict[str, dict] = {}
        for row in census:
            item = by_metal.setdefault(row["metal"], {"parents": 0, "implausible_charge": 0, "unknown_os": 0,
                                                      "os_values": {}})
            item["parents"] += 1
            item["implausible_charge"] += int(not row["charge_plausible"])
            item["unknown_os"] += int(row["oxidation_state_used"] is None)
            k = str(row["oxidation_state_used"])
            item["os_values"][k] = item["os_values"].get(k, 0) + 1
        with slog.stage("filter", items=len(obs_q), pool=pool):
            row_counts, filter_info = _filter(pool, fs, obs_dir, filtered_dir,
                                              [q for q in obs_q if not (qo_obs_dir and q == "QO")], charge_bad, locked)
            if qo_obs_dir and "QO" in obs_q:
                rc, fi = _filter(pool, fs, work / "qo_in", filtered_dir, ["QO"], charge_bad, locked)
                row_counts.update(rc), filter_info.update(fi)
        flags_on = {"os_sanity": bool(profile.get("os_sanity")), "fit_exclude_locked": bool(profile.get("fit_exclude_locked"))}
        log.info("filter: %s; census %d charge-implausible (of which %d broken-structure), %d locked parents listed; parents removed per quantity: %s",
                 flags_on, len(charge_bad), len(structure_bad), len(locked),
                 {q: (r["parents_removed_charge_implausible"], r["parents_removed_locked"]) for q, r in filter_info.items()})
        # ---- expand
        with slog.stage("expand", pool=pool):
            expand_q = [q for q in obs_q if q not in ("QO",)]
            expand_info = fs.run_expand(pool, filtered_dir, expanded_dir, expand_q) if expand_q else {}
        # ---- fit
        fit_q = [q for q in selected if q in PRODUCTION_QUANTITIES]
        with slog.stage("fit", pool=pool) as rec:
            db, fit_failures, key_counts = fs.fit_stage(pool, expanded_dir, filtered_dir, hist_dir, fit_q)
            rec["keys"] = key_counts
        for q, k, e in fit_failures:
            failures.append({"stage": "fit", "quantity": q, "key": k, "error": e})
        row_counts.update({f"{k}": v for k, v in key_counts.items() if k.endswith("_histograms")})
        # ---- reference log-likelihoods
        with slog.stage("ref_ll", pool=pool):
            ref_ll_counts = fs.run_reference_ll(pool, db)
        db = fs.sort_db(db)
        # ---- bends
        tables: Dict[str, Any] = {}
        bends_report = None
        if "Q7" in selected:
            with slog.stage("bends", pool=pool) as rec:
                shards_q = [s.name for s in use_shards] if args.quick else None
                payload = fs.run_bends(pool, Path(args.fit_frame_path), ref_dir, min_n=3, shards=shards_q,
                                       max_parents=40 if args.quick else None)
                tables["bends"] = payload
                bends_report = payload["provenance"]["extraction_report"]
                rec["entries"] = len(payload["entries"])
        # ---- input hashes
        with slog.stage("hash_inputs", pool=pool):
            ref_files = [(str(p.relative_to(ref_dir)).replace("\\", "/"), p) for p in sorted(ref_dir.rglob("*"))
                         if p.is_file()]
            input_hashes = _hash_files(pool, ref_files)
            obs_files = [(p.name, p) for p in sorted(obs_dir.glob("*.parquet")) + sorted(obs_dir.glob("*.npz"))]
            observation_hashes = _hash_files(pool, obs_files)

    # ---- assemble
    with slog.stage("assemble"):
        frame_sha = _sha_file(Path(args.fit_frame_path)) if "Q7" in selected else None
        entries_json = dbio.dumps_entries(db)
        tables_json = dbio.dumps_tables(tables)
        entries_sha = hashlib.sha256((entries_json + tables_json).encode()).hexdigest()
        dirty = bool(_git("status", "--porcelain", "--", "src", "tools", "research/bench"))
        meta: Dict[str, Any] = {
            "schema": SCHEMA,
            # runtime model choices (read by stats.load_db / the scorer)
            "q1_variant": q1.DEFAULT_VARIANT, "q5_model": q5.MODEL_NAME, **MODEL_LABELS,
            "benchmark_holdout": ex.benchmark_codes(),
            # provenance
            "code_commit": _code_commit(), "code_dirty": dirty,
            "profile": snap["profile"], "fingerprint": snap["fingerprint"], "profile_overrides": dict(profile.OVERRIDES),
            "flags": snap["fields"], "differs_from_profile": snap["differs_from_profile"],
            "input_hashes": input_hashes,
            "inputs": {
                "ref": "data/ref (tmQM, lnQM xyz + BO + y.csv; every file hashed in input_hashes)",
                "locked_exclusions_sha256": _sha_file(Path(args.locked_path)),
                "locked_parent_count": len(locked),
                "benchmark_holdout_sha256": _sha_file(ex.HOLDOUT_JSON),
                "fit_frame_sha256": frame_sha,
                "fit_frame_note": "frozen 1,200-parent fit frame (Q7 table only); not regenerable from data/ref",
                "observation_source": str(args.obs_dir) if args.obs_dir else "extracted from data/ref by this build",
                "observation_files": observation_hashes,
            },
            "quick": bool(args.quick),
            "eligibility": eligibility_report,
            "observation_identity": "observation.1" if profile.get("stable_ancestor_sampling") else "legacy",
            "observation_identity_quantities": ["Q1", "Q2", "Q3", "Q4", "Q5", "QO"]
            if profile.get("stable_ancestor_sampling") else [],
            "ancestor_sampling": "hash_priority.1" if profile.get("stable_ancestor_sampling") else "algorithm_r",
            "quantities": [q for q in selected],
            "row_counts": row_counts, "ref_ll_entries": ref_ll_counts,
            "charge_os_census_sha256": hashlib.sha256(
                json.dumps(census, separators=(",", ":"), sort_keys=True).encode()).hexdigest(),
            "charge_implausible_count": len(charge_bad),
            "filter": {"flags": flags_on, "census_charge_implausible_parents": len(charge_bad),
                       "locked_parents_listed": len(locked), "per_quantity": filter_info},
            "tables": {name: {"entries": len(t.get("entries", ())), "min_n": t["provenance"].get("min_n"),
                              "sha256": hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()}
                       for name, t in tables.items()},
            "fit_failures": len([f for f in failures if f["stage"] == "fit"]),
            "entries": len(db), "entries_sha256": entries_sha, "seed": args.seed,
            # volatile (differs between two builds of identical content)
            "build_run": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "workers": workers,
                          "host": platform.node(), "python": platform.python_version(),
                          "total_wall_s": round(time.perf_counter() - t_start, 3), "stages": slog.stages},
        }
        out_json = out_dir / "constraints.json"
        out_json.write_text(dbio.compose_container(meta, tables_json, entries_json))
        npz_size = save_lossless_npz({"_meta": meta, "_tables": tables, **db}, out_dir / "constraints.npz",
                                     header_meta=meta)
        # the NPZ must be the same DB as the JSON: decode it and compare entry for entry (it once dropped ref_ll/ref_sd)
        back, _, back_tables = cmp.load_container(out_dir / "constraints.npz")
        npz_check = cmp.compare_exact(back, db, back_tables, tables)
        del back
        if not npz_check["identical"]:
            failures.append({"stage": "assemble", "error": "NPZ container differs from the JSON", "detail": npz_check})
            log.error("NPZ container differs from the JSON: %s", {k: npz_check[k] for k in ("n_different", "n_only_in_a")})
        human = dict(meta, _note="human-readable copy of the container's embedded _meta; no code reads this file")
        (out_dir / "constraints.meta.json").write_text(json.dumps(human, indent=2, sort_keys=True) + "\n")
        (out_dir / "charge_os_census.json").write_text(json.dumps(census, indent=2) + "\n")
        log.info("saved %d entries (%.1f MB NPZ), tables %s", len(db), npz_size / 2 ** 20, sorted(tables))

    # ---- reproduce
    rep_report = None
    gate_failed = False
    target = None
    if args.target_db:
        with slog.stage("reproduce"):
            target = json.loads(Path(args.target_db).read_text())
            target.pop("_meta", None), target.pop("_tables", None)
            only = [q for q in selected if q in PRODUCTION_QUANTITIES]
            rep_report = cmp.compare_databases(db, target, only_quantities=only)
        print(json.dumps(rep_report, indent=2, sort_keys=True))
        gate_failed = bool(args.check_reproduction) and any(
            s["different"] or s["missing"] or s["extra"] for s in rep_report.values())

    # ---- rescore (optional)
    rescore = None
    if args.rescore_pools_dir:
        rescore_60_pools = _rescore_driver()

        with slog.stage("rescore"):
            rescore = rescore_60_pools(db, target, pools_dir=Path(args.rescore_pools_dir),
                                       frame_path=Path(args.fit_frame_path), manifest_path=Path(args.rescore_manifest),
                                       ref_dir=ref_dir)
        print(json.dumps({"rescore": rescore}, indent=2, sort_keys=True))
    del target

    (out_dir / "build_report.json").write_text(json.dumps({
        "reproduction": rep_report, "rescore": rescore, "row_counts": row_counts,
        "ref_ll_entries": ref_ll_counts, "charge_os_census_summary": by_metal,
        "charge_implausible_codes": sorted(charge_bad), "locked_parent_count": len(locked),
        "bends_extraction": bends_report, "filter": meta["filter"], "failures": failures, "extraction": extraction, "merge": merge_info,
        "expand": expand_info, "npz_roundtrip": npz_check, "stage_timings": slog.summary(),
    }, indent=2, sort_keys=True, default=str) + "\n")
    if not args.keep_work and not args.obs_dir:
        shutil.rmtree(work, ignore_errors=True)
    log.info("build finished in %.1f s (main peak RSS %.0f MB); failures recorded: %d", time.perf_counter() - t_start,
             runtime.peak_rss_mb(), len(failures))
    return int(gate_failed or any(f["stage"] in ("fit", "assemble") for f in failures))


def _ledger_report(obs_dir: Path) -> dict:
    """Policy version and reason counts of the merged extraction ledger (empty when the ledger is off)."""
    import pandas as pd
    from fit import sanity as eligibility

    files = sorted(Path(obs_dir).glob("ledger_*.parquet"))
    if not files:
        return {"policy_version": eligibility.POLICY_VERSION, "ledger": None}
    ledger = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rep = eligibility.reason_counts(ledger, eligibility.load_review())
    rep["ledger_files"] = {f.name: _sha_file(f) for f in files}
    return rep


def _filter(pool, fs, obs_dir, filtered_dir, quantities, charge_bad, locked):
    return fs.run_filter(pool, Path(obs_dir), Path(filtered_dir), quantities, charge_bad, locked) if quantities else ({}, {})


def compare_cmd(args: argparse.Namespace) -> int:
    from fit.pipeline import compare as cmp

    a, meta_a, tab_a = cmp.load_container(args.a)
    b, meta_b, tab_b = cmp.load_container(args.b)
    if args.exact:
        report = cmp.compare_exact(a, b, tab_a, tab_b)
        report["entries_sha256"] = {"a": meta_a.get("entries_sha256"), "b": meta_b.get("entries_sha256")}
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["identical"] else 1
    report = cmp.compare_databases(a, b, tolerance=args.tolerance)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(any(s["different"] or s["missing"] or s["extra"] for s in report.values()))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        ap = argparse.ArgumentParser(prog="fit.build_db compare")
        ap.add_argument("cmd")
        ap.add_argument("a", type=Path)
        ap.add_argument("b", type=Path)
        ap.add_argument("--exact", action="store_true", help="entry-for-entry equality (canonical JSON), meta ignored")
        ap.add_argument("--tolerance", type=float, default=1e-4)
        return compare_cmd(ap.parse_args(argv))
    if argv and argv[0] == "build":
        argv = argv[1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", type=Path, required=True, help="raw reference directory (data/ref: tmQM, lnQM)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--locked-path", type=Path, required=True, help="locked_exclusions.json")
    ap.add_argument("--fit-frame-path", type=Path, required=True, help="frozen fit frame JSON (Q7 surface_fit parents)")
    ap.add_argument("--profile", default="production", help="jacobik.core.profile.PROFILES entry (production | d2 | ...)")
    ap.add_argument("--flags", type=Path, default=None, help="JSON object of extra profile overrides")
    ap.add_argument("--workers", type=int, default=0, help="pool size (default: os.cpu_count())")
    ap.add_argument("--quick", action="store_true", help="development subset: ~2%% of the complexes (4 shards, 750 each)")
    ap.add_argument("--only-shards", nargs="*", default=None, help="development: extract only these whole shards")
    ap.add_argument("--stop-after", choices=["merge"], default=None, help="development: stop after extract + merge")
    ap.add_argument("--quantities", nargs="*", default=None, help=f"subset of {ALL_STAGE_QUANTITIES} (default: all)")
    ap.add_argument("--target-db", type=Path, default=None, help="reference DB for the reproduce stage")
    ap.add_argument("--check-reproduction", action="store_true", help="exit non-zero on any difference to --target-db")
    ap.add_argument("--work-dir", type=Path, default=None, help="intermediates (default: OUT/.build_work)")
    ap.add_argument("--keep-work", action="store_true", help="keep shard/merged/filtered/expanded intermediates")
    ap.add_argument("--obs-dir", type=Path, default=None, help="merged observations to use instead of extracting")
    ap.add_argument("--hist-dir", type=Path, default=None, help="histograms for --obs-dir (default: the same folder)")
    ap.add_argument("--qo-obs-dir", type=Path, default=None, help="QO observations from a separate folder")
    ap.add_argument("--rescore-pools-dir", type=Path, default=None, help="60-pool sanity check: pool folder")
    ap.add_argument("--rescore-manifest", type=Path, default=None, help="60-pool sanity check: manifest json")
    ap.add_argument("--seed", type=int, default=42, help="recorded; every fit seeds from its key")
    args = ap.parse_args(argv)
    if args.rescore_pools_dir and not (args.target_db and args.rescore_manifest):
        ap.error("--rescore-pools-dir needs --target-db and --rescore-manifest")
    if args.check_reproduction and not args.target_db:
        ap.error("--check-reproduction needs --target-db")
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
