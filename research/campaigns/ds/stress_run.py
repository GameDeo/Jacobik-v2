"""Step 5b: registered stress design for one resolved key on a corrected base DB, in two stages.

    refit  (any machine with the observation tables): null refit must reproduce the base DB entry exactly, then every registered trial is refitted in a
           worker pool and written to ENTRIES.jsonl
    python stress_run.py refit PIPELINE_DIR ENTRIES.jsonl --case LUQVOK --quantity Q1 --key "<resolved key>" [--base D-final] [--workers N]
                         [--replicates 32] [--limit K] [--scenarios observed_input_error parent_removal geometry_0.02A]
    score  (needs the saved D3 pools and the frozen manifest paths): every refitted entry is scored on the fixed seed pools, the severity rule is applied
           and the registry summary is written
    python stress_run.py score PIPELINE_DIR ENTRIES.jsonl RECORDS.jsonl --case LUQVOK --quantity Q1 --key "<resolved key>" [--base D-final]

`observed_input_error` replaces the affected parents' rows by the same parents' rows of the uncorrected extraction (E_off), `parent_removal` deletes them,
`geometry_0.02A` re-extracts the affected parents from coordinates perturbed by at most 0.02 A (see `stress_geometry.py`). A trial that cannot run is a failed
record (never skipped). The registered design is deterministic from the support parents and the seed, so both stages rebuild it identically.
"""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

# the build profile must be installed before the first jacobik import (the sampler flag is resolved at import)
FLAGS = {"donor_assignment": True, "donor_identity_reconcile": True, "stable_ancestor_sampling": True, "structure_sanity": True}
os.environ["JACOBIK_PROFILE"] = "d2"
os.environ["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(FLAGS, sort_keys=True, separators=(",", ":"))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
MAIN = Path(os.environ.get("JK_MAIN", "/mnt/e/Dev/Projects/jacobik_2026-10-02_full"))
from fit.pipeline import confidence
registry = stress = stress_driver = confidence
from fit.sanity import filter_fit_frame, locked_codes

G = {}


def _frames(P, quantity, bad, locked, extraction):
    files = sorted((P / extraction / "work/obs").glob(f"observations_{quantity}_*.parquet"))
    return pd.concat([filter_fit_frame(pd.read_parquet(f), charge_bad=bad, locked=locked, os_sanity=True, exclude_locked=True) for f in files],
                     ignore_index=True)


def _setup(a):
    P, q = a.pipeline, a.quantity
    census = json.loads((P / a.base / "charge_os_census.json").read_text())
    bad = {r["csd_code"] for r in census if not r["charge_plausible"] or (a.base == "D-final" and not r.get("structure_plausible", True))}
    locked = locked_codes(MAIN / "research/bench/benchmarks/locked_exclusions.json")
    raw = _frames(P, q, bad, locked, "E_assign"); rep = _frames(P, q, bad, locked, "E_off")
    key = a.key if q != "QO" else tuple(json.loads(a.key))
    if q == "QO":
        from jacobik.stats import orientation as O
        fr = O.fit_rows(raw); fr = fr[~(fr["hb_min"] < O.HB_CUT)]
        support = fr.loc[(fr["k"] == key[0]) & (fr["ring"].astype(int) == key[1]), "csd_code"].astype(str).tolist()
    else:
        support = raw.loc[raw["key"] == key, "csd_code"].astype(str).tolist()
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    inputs = {"base_db": sha(P / a.base / "constraints.json"), "base_meta": sha(P / a.base / "constraints.meta.json"),
              "extraction_E_assign": sha(P / "E_assign.obs.sha256"), "extraction_E_off": sha(P / "E_off.obs.sha256")}
    reg = registry.register({"quantity": q, "resolved_key": a.key, "parent_code": a.case}, support, inputs, scenarios=tuple(a.scenarios),
                            replicates=a.replicates)
    G.update(raw=raw, rep=rep, quantity=q, key=key, pipeline=P, base=a.base)
    if any(t["scenario"].startswith("geometry") for t in reg["trials"]) and a.stage == "refit":
        from stress_geometry import build_block_cache
        from jacobik.stats.mixture import structure_id
        need = {t_ for tr in reg["trials"] if tr["scenario"].startswith("geometry") for t_ in tr["affected_parents"]}
        codes = [c for c in raw["csd_code"].astype(str).unique() if structure_id(c) in need]
        t0 = time.time(); G["blocks"] = build_block_cache(a.ref_dir, codes)
        print(f"source blocks of {len(codes)} parents cached in {time.time() - t0:.0f}s", flush=True)
    print(f"{a.case} {q} {key}: {len(raw)} rows, support {len(set(support))} parents, {len(reg['trials'])} registered trials", flush=True)
    return reg, [t for t in reg["trials"]][:a.limit] if a.limit else reg["trials"], key


def _null(_):
    q, key = G["quantity"], G["key"]
    return stress.refit_qo(G["raw"], key) if q == "QO" else stress.refit_resolved(G["raw"], q, key, stable=True)


def _worker(trial):
    t0 = time.time()
    try:
        replacements = G["rep"]
        if trial["scenario"].startswith("geometry"):
            from stress_geometry import geometry_replacements       # same directory
            replacements = geometry_replacements(G, trial)
        entry = stress_driver.refit_trial(G["raw"], G["quantity"], G["key"], trial, trial["scenario"], stable=True, replacements=replacements)
        return dict(entry=entry, seconds=time.time() - t0)
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {str(exc)[:300]}", seconds=time.time() - t0)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def refit(a):
    reg, trials, key = _setup(a)
    db = json.loads((a.pipeline / a.base / "constraints.json").read_text()); db = db.get("entries", db)
    ref = db[f"{a.quantity}:{key}"]
    t0 = time.time()
    with ProcessPoolExecutor(1, mp_context=get_context("fork")) as one:
        clean_entry = one.submit(_null, 0).result()
    same = bool(np.allclose(np.sort(np.ravel(clean_entry["means"])), np.sort(np.ravel(ref["means"])), rtol=0, atol=1e-9))
    print(f"null refit {time.time() - t0:.0f}s reproduces the base DB entry: {same}", flush=True)
    if not same:
        a.out.write_text(json.dumps(dict(status="null_refit_mismatch", ref_means=_jsonable(ref["means"]), got_means=_jsonable(clean_entry["means"]))))
        sys.exit(3)
    done = 0
    with ProcessPoolExecutor(a.workers, mp_context=get_context("fork")) as ex, a.out.open("w") as fh:
        for t, r in zip(trials, ex.map(_worker, trials, chunksize=1)):
            rec = dict(trial_id=t["trial_id"], scenario=t["scenario"], replicate=t["replicate"], affected_count=t["affected_count"],
                       actual_fraction=t["actual_fraction"], seconds=r["seconds"])
            rec.update(error=r["error"]) if "error" in r else rec.update(entry=_jsonable(r["entry"]))
            fh.write(json.dumps(rec) + "\n"); fh.flush(); done += 1
            if done % 20 == 0:
                print(f"{done}/{len(trials)} refits, {time.time() - t0:.0f}s", flush=True)
    print(f"refit stage done: {done} trials, mean {np.mean([json.loads(l)['seconds'] for l in a.out.open()]):.0f}s", flush=True)


def score(a):
    reg, trials, key = _setup(a)
    from tools.jgen.ds import stress_trial as ST
    score_fn, clean, info = ST.prepare(ST.manifest_row(MAIN, a.case, "D"), resolved_key=key if a.quantity == "QO" else eval(key), quantity=a.quantity,
                                       score_db=str(a.pipeline / a.base / "constraints.json"))
    print(info, flush=True)
    entries = {json.loads(l)["trial_id"]: json.loads(l) for l in a.entries.open()}
    records = []
    for t in trials:
        e = entries.get(t["trial_id"])
        base = dict(trial_id=t["trial_id"], scenario=t["scenario"], replicate=t["replicate"], affected_count=t["affected_count"],
                    actual_fraction=t["actual_fraction"], support_count=reg["support_count"], broken_coordination_rule=reg["broken_coordination_rule"],
                    refit_seconds=None if e is None else e["seconds"])
        if e is None or "error" in e:
            records.append(dict(base, status="failed", reason="missing refit record" if e is None else e["error"])); continue
        try:
            records.append(dict(base, status="ok", outcome=confidence.evaluate(clean, score_fn(stress_driver.fitdist_kwargs(e["entry"]))),
                                means=np.ravel(e["entry"]["means"]).tolist()[:6]))
        except Exception as exc:
            records.append(dict(base, status="failed", reason=f"scoring {type(exc).__name__}: {str(exc)[:300]}"))
    a.out.write_text("".join(json.dumps(r, default=str) + "\n" for r in records))
    ok = [r for r in records if r["status"] == "ok"]
    print(f"scored: {len(ok)} ok, {len(records) - len(ok)} failed; severe {sum(r['outcome']['severe'] for r in ok)}; "
          f"single-seed {sum(r['outcome']['single_seed_severe'] for r in ok)}", flush=True)
    complete = len(records) == len(reg["trials"])
    summary = confidence.summarize_registered(reg, records, level="fixed_pool") if complete else None
    a.out.with_suffix(".summary.json").write_text(json.dumps(dict(summary=summary, reconciliation=registry.reconcile(reg, records)), default=str, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["refit", "score"]); ap.add_argument("pipeline", type=Path); ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--case", required=True); ap.add_argument("--quantity", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--base", default="D-final"); ap.add_argument("--ref-dir", type=Path, default=MAIN / "data/ref"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--replicates", type=int, default=32); ap.add_argument("--limit", type=int)
    ap.add_argument("--scenarios", nargs="*", default=list(confidence.SCENARIOS))
    a = ap.parse_args()
    if a.stage == "refit":
        a.out = a.files[0]; refit(a)
    else:
        a.entries, a.out = a.files[0], a.files[1]; score(a)


if __name__ == "__main__":
    main()
