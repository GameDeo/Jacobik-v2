"""Stages ``census``/``filter``, ``expand``, ``fit``, ``prune``/``ref_ll`` and the Q7 ``bends`` table.

Every task here is a module-level function over one picklable payload, run on the shared pool. Determinism does not rest
on scheduling: every key's random state is derived from the key (crc32 of its string), blocks are only a transport unit,
and results are written back by key before anything is sorted and serialised.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("build_db")

#: seconds for one key of n rows, from the 2026-11-07 cost survey (power law t = t1k * (n / 1000) ** b), used only to
#: size and order the task blocks
_COST = {"Q1": (0.62, 0.81), "Q2": (1.6, 0.88), "Q3": (0.26, 0.86), "Q4": (0.044, 0.84), "Q5": (0.44, 1.08),
         "QO": (0.1, 0.41)}
_FLOOR = {"Q1": 0.004, "Q2": 0.01, "Q3": 0.003, "Q4": 0.002, "Q5": 0.01, "QO": 0.9}
BLOCK_SECONDS = 4.0


def est_seconds(quantity: str, n: int) -> float:
    t1k, b = _COST[quantity]
    return _FLOOR[quantity] + t1k * (max(n, 1) / 1000.0) ** b


def make_blocks(quantity: str, sizes: Sequence[int], budget: float = BLOCK_SECONDS) -> List[List[int]]:
    """Group item indices (given in the order to run: largest first) into blocks of about ``budget`` estimated seconds."""
    blocks, cur, acc = [], [], 0.0
    for i, n in enumerate(sizes):
        c = est_seconds(quantity, n)
        if cur and acc + c > budget:
            blocks.append(cur)
            cur, acc = [], 0.0
        cur.append(i)
        acc += c
    if cur:
        blocks.append(cur)
    return blocks


# ---------------------------------------------------------------------------------------------------------------------
# census + filter
def census_task(p: dict) -> list:
    from fit.sanity import census_file

    return census_file(Path(p["ref_dir"]), p["dataset"], Path(p["xyz"]))


def run_census(pool, ref_dir: Path, shards) -> list:
    """Census rows of the given xyz shards (tmQM before lnQM, shard names sorted: the serial census order)."""
    names = {(s.dataset, s.xyz.name) for s in shards}
    from fit.sanity import census_files

    files = [(ds, p) for ds, p in census_files(ref_dir) if (ds, p.name) in names]
    order = sorted(range(len(files)), key=lambda i: -files[i][1].stat().st_size)
    futs = {i: pool.submit(census_task, dict(ref_dir=str(ref_dir), dataset=files[i][0], xyz=str(files[i][1])))
            for i in order}
    rows: list = []
    for i in range(len(files)):
        rows.extend(pool.collect([futs[i]])[0])
    return rows


def filter_task(p: dict) -> dict:
    """Whole-parent exclusions (charge-implausible, locked) under the build profile, splits concatenated in X1, X2, X3,
    LNf order (the expander's reservoir stream depends on row order)."""
    import pandas as pd
    from fit.sanity import filter_fit_frame

    from fit import sanity

    parts, n, n_in = [], 0, 0
    codes: set = set()
    for f in p["files"]:
        if not Path(f).exists():
            continue
        raw = pd.read_parquet(f)
        n_in += len(raw)
        codes.update(raw["csd_code"].astype(str).unique())
        df = filter_fit_frame(raw, charge_bad=set(p["charge_bad"]), locked=set(p["locked"]))
        n += len(df)
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no observation tables for {p['quantity']}")
    pd.concat(parts, ignore_index=True).to_parquet(p["out"], index=False)
    # parents of this table removed by each rule (a rule that is off removes none)
    charge = sorted(codes & set(p["charge_bad"])) if sanity.OS_SANITY else []
    locked = sorted(codes & set(p["locked"])) if sanity.FIT_EXCLUDE_LOCKED else []
    return {"quantity": p["quantity"], "rows": n, "rows_in": n_in, "parents_in": len(codes),
            "parents_removed_charge_implausible": len(charge), "parents_removed_locked": len(locked),
            "parents_removed_either": len(set(charge) | set(locked))}


def expand_task(p: dict) -> dict:
    from fit.expander import expand_quantity

    return {p["quantity"]: expand_quantity(p["in_dir"], p["out_dir"], quantities=[p["quantity"]]).get(p["quantity"])}


def run_filter(pool, obs_dir: Path, filtered_dir: Path, quantities: Sequence[str], charge_bad, locked):
    """(rows kept per quantity, per-quantity filter record: rows/parents in and parents removed by each rule)."""
    filtered_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(quantities, key=lambda q: -sum(f.stat().st_size for f in obs_dir.glob(f"observations_{q}_*.parquet")))
    futs = {q: pool.submit(filter_task, dict(
        quantity=q, charge_bad=sorted(charge_bad), locked=sorted(locked), out=str(filtered_dir / f"observations_{q}.parquet"),
        files=[str(obs_dir / f"observations_{q}_{s}.parquet") for s in ("X1", "X2", "X3", "LNf")])) for q in order}
    info = {q: pool.collect([futs[q]])[0] for q in quantities}
    return {q: r["rows"] for q, r in info.items()}, info


def run_expand(pool, filtered_dir: Path, expanded_dir: Path, quantities: Sequence[str]) -> Dict[str, Any]:
    expanded_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(quantities, key=lambda q: -(filtered_dir / f"observations_{q}.parquet").stat().st_size)
    futs = {q: pool.submit(expand_task, dict(quantity=q, in_dir=str(filtered_dir), out_dir=str(expanded_dir)))
            for q in order}
    out: Dict[str, Any] = {}
    for q in quantities:
        out.update(pool.collect([futs[q]])[0])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# fit tasks
def fit_keys_task(p: dict) -> dict:
    """Q3/Q4/Q5: independent fit of each (key, values, weights) of a block with the quantity module's own routine."""
    from fit import q3, q4, q5

    mod = {"Q3": q3, "Q4": q4, "Q5": q5}[p["quantity"]]
    entries, failures = {}, []
    for key, vals, w in p["items"]:
        try:
            entries.update(mod.fit_from_observations({key: (vals, w)}, n_min=mod.N_MIN))
        except Exception as exc:
            failures.append((key, f"{type(exc).__name__}: {exc}"[:300]))
    return {"entries": entries, "failures": failures, "keys": len(p["items"])}


def pooled_fit_task(p: dict) -> dict:
    """Q1/Q2: independent mixture fit of each work item of a block (fit.pooling)."""
    from fit import pooling as PL

    fits, failures = [], []
    for item in p["items"]:
        key, fit, err = PL.fit_one_checked(item)
        fits.append((key, fit))
        if err:
            failures.append((key, err))
    return {"fits": fits, "failures": failures, "keys": len(p["items"])}


def qo_fit_task(p: dict) -> dict:
    from fit import qo as QO

    meta = p["meta"]
    fits, failures = [], []
    for idx, G, seed in p["items"]:
        try:
            fits.append((idx, QO._fit_one(G, seed, meta["k_max"], meta["cov_floor"], meta["w_min"])))
        except Exception as exc:
            failures.append((idx, f"{type(exc).__name__}: {exc}"[:300]))
    return {"fits": fits, "failures": failures, "keys": len(p["items"])}


def _submit_blocks(pool, task, quantity: str, items: list, sizes: List[int], extra: Optional[dict] = None):
    """Submit ``items`` (already ordered largest first) as blocks; returns the futures in block order."""
    blocks = make_blocks(quantity, sizes)
    return [pool.submit(task, dict(quantity=quantity, items=[items[i] for i in b], **(extra or {}))) for b in blocks]


def fit_stage(pool, expanded_dir: Path, filtered_dir: Path, hist_dir: Path, quantities: Sequence[str]) -> tuple:
    """Fit Q1..Q6, QO and Q6P. Returns (db, failures, per-quantity key counts).

    One task list over the whole pool (heavy quantities first, each ordered largest key first). The main process plans
    (loads, groups) and runs the cheap serial parts (Q6, Q6P, the Q1/Q2 pooling pass) while workers fit.
    """
    from fit import q1, q2, q3, q4, q5, q6, q6p
    from fit import fitter as FT
    from jacobik.stats import orientation as O
    from fit import qo as QO
    from fit import pooling as PL
    import pandas as pd

    selected = set(quantities)
    pending: Dict[str, Any] = {}
    failures: List[Tuple[str, str, str]] = []
    counts: Dict[str, int] = {}

    # --- plan + submit, heaviest first
    if "Q5" in selected:
        obs = FT._load_observations(expanded_dir, "Q5", q5.N_MIN)
        keys = sorted(obs, key=lambda k: (-len(obs[k][0]), k))
        items = [(k, obs[k][0], obs[k][1]) for k in keys]
        pending["Q5"] = _submit_blocks(pool, fit_keys_task, "Q5", items, [len(i[1]) for i in items])
        counts["Q5"] = len(items)
    pooled: Dict[str, Any] = {}
    for q, mod in (("Q2", q2), ("Q1", q1)):
        if q not in selected:
            continue
        df = PL.load_expanded(expanded_dir, q)
        pf = PL.PooledFitter(q, df, n_indep=mod.N_INDEP)
        del df
        items = pf.independent_items()
        pooled[q] = pf
        pending[q] = _submit_blocks(pool, pooled_fit_task, q, items, [len(i[1]) for i in items])
        counts[q] = len(items)
    qo_items: list = []
    qo_meta: dict = {}
    if "QO" in selected:
        df = pd.read_parquet(filtered_dir / "observations_QO.parquet")
        groups = QO.training_groups(QO.fit_rows(df), n_min=O.N_MIN, hb_cut=O.HB_CUT)
        del df
        order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), O.key_str(kv[0])))
        qo_items = [(i, np.ascontiguousarray(G), QO.group_seed(key)) for i, (key, G) in enumerate(order)]
        qo_keys = [key for key, _ in order]
        qo_meta = {"n_min": O.N_MIN, "hb_cut": O.HB_CUT, "k_max": O.K_MAX, "cov_floor": O.COV_FLOOR, "w_min": O.W_MIN}
        pending["QO"] = _submit_blocks(pool, qo_fit_task, "QO", qo_items, [len(i[1]) for i in qo_items],
                                       extra={"meta": qo_meta})
        counts["QO"] = len(qo_items)
    for q, mod in (("Q3", q3), ("Q4", q4)):
        if q not in selected:
            continue
        obs = FT._load_observations(expanded_dir, q, mod.N_MIN)
        keys = sorted(obs, key=lambda k: (-len(obs[k][0]), k))
        items = [(k, obs[k][0], obs[k][1]) for k in keys]
        pending[q] = _submit_blocks(pool, fit_keys_task, q, items, [len(i[1]) for i in items])
        counts[q] = len(items)

    # --- serial parts while the pool works
    db: Dict[str, Any] = {}
    if "Q6" in selected:
        obs = FT._load_observations(expanded_dir, "Q6", q6.N_MIN)
        db.update(q6.fit_from_observations(obs, n_min=q6.N_MIN))
        counts["Q6"] = len(obs)
    if "Q6P" in selected:
        contact = sorted(p for p in hist_dir.glob("q6hist_*.npz") if not p.name.startswith("q6hist_m13_"))
        m13 = sorted(hist_dir.glob("q6hist_m13_*.npz"))
        db.update(q6p.fit_from_observations(q6p.load_hist(contact), q6p.load_hist(m13)))
        counts["Q6P_histograms"] = len(contact) + len(m13)

    # --- collect
    for q in ("Q5", "Q3", "Q4"):
        if q in pending:
            for r in pool.collect(pending[q]):
                db.update(r["entries"])
                failures += [(q, k, e) for k, e in r["failures"]]
    for q, mod in (("Q2", q2), ("Q1", q1)):
        if q not in pending:
            continue
        pf = pooled[q]
        for r in pool.collect(pending[q]):
            for k, f in r["fits"]:
                pf.indep[k] = f
            failures += [(q, k, e) for k, e in r["failures"]]
        if q == "Q1":
            entries = pf.entries(r=q1.DEFAULT_POOLING_R, variant=q1.DEFAULT_VARIANT)
        else:
            entries = pf.entries(r=q2.DEFAULT_POOLING_R)
        db.update({f"{q}:{k}": e for k, e in entries.items()})
        del pooled[q]
    if "QO" in pending:
        fitted: Dict[int, dict] = {}
        for r in pool.collect(pending["QO"]):
            fitted.update(dict(r["fits"]))
            failures += [("QO", O.key_str(qo_keys[i]), e) for i, e in r["failures"]]
        done = [i for i in range(len(qo_items)) if i in fitted]
        db.update(QO.assemble_orientation([(qo_keys[i], None) for i in done], [fitted[i] for i in done], **qo_meta))
    return db, failures, counts


def sort_db(db: Dict[str, Any]) -> Dict[str, Any]:
    """Entries in key order (the container is written sorted, independent of completion order)."""
    return {k: db[k] for k in sorted(db)}


# ---------------------------------------------------------------------------------------------------------------------
# prune + reference likelihoods
def reflik_task(items: list) -> list:
    from fit.quantity import entry_reference_ll

    out = []
    for key, entry in items:
        got = entry_reference_ll(key, entry)
        if got is not None:
            out.append((key, got[1], got[2]))
    return out


def run_reference_ll(pool, db: Dict[str, Any], block: int = 1500) -> Dict[str, int]:
    """ref_ll / ref_sd on every supported 1-D entry, in place."""
    keys = [k for k in db if isinstance(k, str) and k.split(":", 1)[0] in ("Q1", "Q2", "Q3", "Q4", "Q5")]
    futs = [pool.submit(reflik_task, [(k, db[k]) for k in keys[i:i + block]]) for i in range(0, len(keys), block)]
    counts: Dict[str, int] = {}
    for res in pool.collect(futs):
        for key, ll, sd in res:
            db[key]["ref_ll"], db[key]["ref_sd"] = ll, sd
            q = key.split(":", 1)[0]
            counts[q] = counts.get(q, 0) + 1
    return counts


# ---------------------------------------------------------------------------------------------------------------------
# Q7 bends table
def bends_task(p: dict) -> dict:
    """Crystal bend-angle observations of the surface_fit parents of one shard block (one pass over the shard)."""
    from fit.bends import extract_bound_angle_observations
    from fit.parser import load_raw_complexes

    obs, failures = {}, []
    wanted = {c: None for c in p["codes"]}
    try:
        for code, mol in load_raw_complexes(p["codes"], p["shard"], p["dataset"], Path(p["ref_dir"])):
            wanted[code] = mol
    except Exception as exc:
        failures.extend({"code": c, "error": f"{type(exc).__name__}: {exc}"[:400]} for c in p["codes"])
        return {"obs": obs, "failures": failures}
    for code, mol in wanted.items():
        try:
            if isinstance(mol, Exception):
                raise mol
            if mol is None:
                raise FileNotFoundError(f"no raw geometry found for {code} ({p['shard']}, {p['dataset']})")
            obs[code] = extract_bound_angle_observations(mol, code)
        except Exception as exc:
            failures.append({"code": code, "error": f"{type(exc).__name__}: {exc}"[:400]})
    return {"obs": obs, "failures": failures}


def run_bends(pool, fit_frame_path: Path, ref_dir: Path, *, min_n: int = 3, shards: Optional[Sequence[str]] = None,
              max_parents: Optional[int] = None) -> Dict[str, Any]:
    """The Q7 table payload (the former ``constraints.bends.json``), built on the pool from the frozen fit frame."""
    import hashlib

    from fit.bends import build_bound_angle_table

    frame_bytes = Path(fit_frame_path).read_bytes()
    frame = json.loads(frame_bytes)
    rows = frame.get("main_allocation", {}).get("splits", {}).get("surface_fit", [])
    parents: Dict[str, dict] = {}
    for row in rows:
        code = row.get("code") or row.get("representative_code")
        if code:
            parents.setdefault(str(code), row)
    if shards is not None:
        parents = {c: r for c, r in parents.items() if (r.get("shard") or r.get("representative_shard")) in set(shards)}
    codes_sorted = sorted(parents)
    if max_parents:
        codes_sorted = codes_sorted[:max_parents]
    by_shard: Dict[Tuple[str, str], List[str]] = {}
    for code in codes_sorted:
        r = parents[code]
        by_shard.setdefault((r.get("shard") or r.get("representative_shard"), r.get("dataset")), []).append(code)
    # at most ~40 parents per task so one big shard does not serialise the stage
    tasks = []
    for (shard, dataset), codes in sorted(by_shard.items(), key=lambda kv: -len(kv[1])):
        for i in range(0, len(codes), 40):
            tasks.append(dict(shard=shard, dataset=dataset, codes=codes[i:i + 40], ref_dir=str(ref_dir)))
    results = pool.map(bends_task, tasks)
    observations_by_code: Dict[str, list] = {}
    failures: list = []
    for r in results:
        observations_by_code.update(r["obs"])
        failures.extend(r["failures"])
    failures.sort(key=lambda f: f["code"])
    observations = [o for code in codes_sorted for o in observations_by_code.get(code, ())]  # sorted-code order
    table = build_bound_angle_table(observations, min_n=min_n)
    report = {"frame_split": "surface_fit", "parents_requested": len(codes_sorted), "parents_failed": len(failures),
              "n_observations": len(observations), "n_entries": len(table), "failures": failures}
    if len(failures) / max(len(codes_sorted), 1) > 0.05:
        raise RuntimeError(f"Q7 bound-angle extraction failed for {len(failures)}/{len(codes_sorted)} surface_fit "
                           f"parents (>5%); details: {failures[:10]}")
    return {
        "schema": "q7_bound_angles_v1",
        "provenance": {"source": "fit-frame surface_fit crystal geometries",
                       "fit_frame_sha256": hashlib.sha256(frame_bytes).hexdigest(),
                       "fit_frame_sha256_method": "raw_bytes", "extraction_report": report, "min_n": min_n},
        "entries": [{"key": list(key), **value} for key, value in sorted(table.items(), key=lambda it: repr(it[0]))],
    }


# ---------------------------------------------------------------------------
# Persistent process pool
# ---------------------------------------------------------------------------

from .extract import THREAD_VARS, peak_rss_mb


def _init_worker(env: Dict[str, str]) -> None:
    for var in THREAD_VARS:
        os.environ[var] = "1"
    for k, v in env.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    import warnings

    warnings.filterwarnings("ignore")
    try:  # preload the fit stack once per worker
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        import fit.pipeline.fitting  # noqa: F401
        import fit.pipeline.extract  # noqa: F401
    except Exception:  # pragma: no cover - a task reports its own import error
        pass


def _call(fn: Callable, payload: Any):
    t = time.perf_counter()
    result = fn(payload)
    return result, time.perf_counter() - t, peak_rss_mb()


class _Done:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class TaskPool:
    def __init__(self, workers: int, env: Dict[str, str]):
        self.workers = max(1, int(workers))
        self.env = dict(env)
        self.busy_s = 0.0
        self.max_worker_rss_mb = 0.0
        self.n_tasks = 0
        self._ex = None
        if self.workers > 1:
            self._ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"),
                                           initializer=_init_worker, initargs=(self.env,))

    def submit(self, fn: Callable, payload: Any):
        self.n_tasks += 1
        if self._ex is None:
            return _Done(_call(fn, pickle.loads(pickle.dumps(payload, protocol=5))))
        return self._ex.submit(_call, fn, payload)

    def collect(self, futures: Sequence) -> List[Any]:
        """Results in submission order."""
        out = []
        for f in futures:
            result, seconds, rss = f.result()
            self.busy_s += seconds
            self.max_worker_rss_mb = max(self.max_worker_rss_mb, rss)
            out.append(result)
        return out

    def map(self, fn: Callable, payloads: Sequence) -> List[Any]:
        return self.collect([self.submit(fn, p) for p in payloads])

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
