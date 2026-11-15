"""Stage ``extract``: raw tmQM/lnQM xyz -> observation parquet + Q6 histograms, by chunk, then merged per split.

A shard (4,000 complexes) is cut into blocks of CHUNK_BLOCKS complexes; each chunk is one pool task, an
``fit.extractor.extract_split`` call under the build profile (the typing flags reach it through the
environment), exactly the production extraction. Parsing and assembling a complex costs ~10x its key extraction, so this
is what keeps the stage short on a wide node. Q1-Q5 and QO rows and the histograms concatenate/sum exactly over chunks;
the Q6 sample is order-dependent (a seeded reservoir per shard), so chunks emit their Q6 observations uncapped and
``replay_q6`` runs them, in file order, through the shard's reservoir: the single-pass shard sample, bit for bit.
The merge step concatenates the
shards of a split in sorted shard order, drops the benchmark hold-out codes from every quantity, and re-caps Q6 per
element-pair key with a generator seeded by the split (the cap is per split, as in a single-process extraction).
Histograms (Q6 PMF, metal 1-3) are summed per split; the hold-out codes were left out at extraction.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import resource
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from jacobik.core.profile import get as profile_get

log = logging.getLogger("build_db")

SPLITS = ("X1", "X2", "X3", "LNf")
OBS_QUANTITIES = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO")
#: seed of the per-split Q6 re-cap (refit_lib.SEED of the 2026-09 refit; changing it changes the Q6 sample)
Q6_RECAP_SEED = 20260920
#: ``--quick``: the first QUICK_LIMIT complexes of one shard per split (about 2 % of the data), for development
QUICK_SHARDS = ("X1s000", "X2s000", "X3s000", "LNfs000")
QUICK_LIMIT = 750
#: molecule blocks per extraction task (a 4,000-complex shard is 10 tasks); exact for any value: see run_extract
CHUNK_BLOCKS = 400
HOLDOUT_JSON = Path(__file__).resolve().parents[1] / "data" / "benchmark_holdout_codes.json"
HIST_PREFIXES = ("q6hist", "q6hist_m13")


@dataclass(frozen=True)
class Shard:
    name: str      # X1s000, LNfs003 ...
    split: str     # X1 | X2 | X3 | LNf
    dataset: str   # tmQM | lnQM (directory under data/ref)
    xyz: Path
    size: int


def benchmark_codes(path: Path = HOLDOUT_JSON) -> List[str]:
    """CSD codes of every benchmark complex (N48 screen, RING10, DEV32); never fitted on."""
    d = json.loads(Path(path).read_text())
    return sorted({c for k, v in d.items() if not k.startswith("_") for c in v})


def list_shards(ref_dir: Path) -> List[Shard]:
    out = []
    for split in SPLITS:
        dataset = "lnQM" if split == "LNf" else "tmQM"
        for p in sorted((Path(ref_dir) / dataset / "xyz").glob(f"tmQM_{split}s*.xyz")):
            out.append(Shard(p.stem[len("tmQM_"):], split, dataset, p, p.stat().st_size))
    return out


def select_shards(shards: Sequence[Shard], quick: bool) -> tuple[List[Shard], Optional[int]]:
    if not quick:
        return list(shards), None
    names = {s.name: s for s in shards}
    missing = [n for n in QUICK_SHARDS if n not in names]
    if missing:
        raise FileNotFoundError(f"--quick needs shards {missing} under data/ref")
    return [names[n] for n in QUICK_SHARDS], QUICK_LIMIT


def sibling_weights(csv_path: Path) -> Dict[str, float]:
    """{code: 1 / siblings} for codes that share an independent structure (LnQM ligand sets); empty for tmQM."""
    from collections import Counter

    import pandas as pd
    from jacobik.stats.mixture import structure_id

    if not Path(csv_path).exists():
        return {}
    codes = pd.read_csv(csv_path, sep=";", usecols=["CSD_code"])["CSD_code"].astype(str).tolist()
    sid = {c: structure_id(c) for c in codes}
    n = Counter(sid.values())
    return {c: 1.0 / n[s] for c, s in sid.items() if n[s] > 1}


_WEIGHTS: Dict[str, Dict[str, float]] = {}


def _weights(csv_path: Path) -> Dict[str, float]:
    key = str(csv_path)
    if key not in _WEIGHTS:
        _WEIGHTS[key] = sibling_weights(csv_path)
    return _WEIGHTS[key]


def count_blocks(xyz: Path) -> int:
    """Number of molecule blocks of an xyz shard (the parser's own block iterator)."""
    from fit.parser import _iter_xyz_blocks

    return sum(1 for _ in _iter_xyz_blocks(Path(xyz)))


def plan_chunks(shards: Sequence[Shard], limit: Optional[int], chunk_blocks: int = CHUNK_BLOCKS) -> Dict[str, List[tuple]]:
    """{shard: [(chunk index, lo, hi), ...]}: file-block ranges that concatenate to the shard (or to its first
    ``limit`` blocks in quick mode). The partition depends only on the data, never on the worker count."""
    plan = {}
    for s in shards:
        n = count_blocks(s.xyz)
        if limit is not None:
            n = min(n, limit)
        plan[s.name] = [(i, lo, min(lo + chunk_blocks, n)) for i, lo in enumerate(range(0, n, chunk_blocks))]
    return plan


def chunk_task(p: dict) -> dict:
    """One block range of a shard through ``extract_split`` (Q1-Q5, QO, histograms; Q6 uncapped, in add order).
    Molecule-level failures are counted, not raised."""
    from fit.extractor import extract_split

    data_dir = Path(p["ref_dir"]) / p["dataset"]
    with contextlib.redirect_stdout(io.StringIO()):
        st = extract_split(data_dir, p["shard"], p["out_dir"], progress=False, hypothesis=False,
                           q6_hist_exclude=set(p["exclude"]), q6_hist_weight=_weights(data_dir / "tmQM_y.csv"),
                           block_range=(p["lo"], p["hi"]), q6_uncapped=True, out_tag=p["tag"])
    return st


def q6_replay_task(p: dict) -> dict:
    """The shard's Q6 sample: its chunks' uncapped streams through the seeded reservoir, in chunk order."""
    from fit.extractor import replay_q6

    return {"shard": p["shard"], "rows": replay_q6([Path(f) for f in p["files"]], Path(p["out"]))}


def chunk_tag(shard: str, idx: int) -> str:
    return f"{shard}.c{idx:03d}"


def run_extract(pool, ref_dir: Path, shard_dir: Path, shards: Sequence[Shard], plan: Dict[str, List[tuple]],
                exclude: Sequence[str]) -> Dict[str, dict]:
    """Extract every shard as chunk tasks; per-shard statistics (summed over the chunks) are returned by shard name.

    The chunk files stay in ``shard_dir`` (``observations_<Q>_<shard>.c<idx>.parquet``, ``<hist>_<shard>.c<idx>.npz``);
    the shard Q6 sample is ``observations_Q6_<shard>.parquet``.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    by_name = {s.name: s for s in shards}
    jobs = [(name, c) for name in plan for c in plan[name]]
    # all chunks have about the same cost; the biggest shards' chunks go first
    jobs.sort(key=lambda j: (-(j[1][2] - j[1][1]), -by_name[j[0]].size, j[0], j[1][0]))
    futs = [pool.submit(chunk_task, dict(ref_dir=str(ref_dir), dataset=by_name[n].dataset, shard=n, out_dir=str(shard_dir),
                                         exclude=list(exclude), lo=c[1], hi=c[2], tag=chunk_tag(n, c[0])))
            for n, c in jobs]
    results = pool.collect(futs)
    stats: Dict[str, dict] = {n: {"chunks": len(plan[n]), "n_complexes": 0, "n_errors": 0, "n_parse_failed": 0,
                                  "elapsed_s": 0.0, "quantity_errors": {}} for n in plan}
    for (n, _), st in zip(jobs, results):
        t = stats[n]
        t["n_complexes"] += st["n_complexes"]
        t["n_errors"] += st["n_errors"]
        t["n_parse_failed"] += st.get("n_parse_failed", 0)
        t["elapsed_s"] += st["elapsed_s"]
        for k, v in st["quantity_errors"].items():
            t["quantity_errors"][k] = t["quantity_errors"].get(k, 0) + v
    replays = pool.map(q6_replay_task, [
        dict(shard=n, out=str(shard_dir / f"observations_Q6_{n}.parquet"),
             files=[str(shard_dir / f"observations_Q6_{chunk_tag(n, c[0])}.parquet") for c in plan[n]])
        for n in sorted(plan)])
    for r in replays:
        stats[r["shard"]]["Q6_rows_after_shard_reservoir"] = r["rows"]
    return stats


# ---------------------------------------------------------------------------------------------------------------------
def merge_task(p: dict) -> dict:
    """observations_<Q>_<split>.parquet from the shard files of that split (benchmark codes dropped; Q6 re-capped)."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fit.extractor import observation_schema, Q6_DEFAULT_CAP, Q6_GLOBAL_CAP, QO_SCHEMA

    q, split = p["quantity"], p["split"]
    schema = QO_SCHEMA if q == "QO" else observation_schema(q)
    # every chunk is cast to the declared schema (never inferred from a chunk) before the concat
    tabs = [pq.read_table(f).select(schema.names).cast(schema) for f in p["files"]]
    t = pa.concat_tables(tabs)
    excl = set(p["exclude"])
    if excl and "csd_code" in t.column_names and t.num_rows:
        keep = ~pd.Series(t.column("csd_code").to_pylist()).isin(excl).to_numpy()
        if not keep.all():
            t = t.filter(pa.array(keep))
    if q == "Q6" and t.num_rows:
        rng = np.random.default_rng(Q6_RECAP_SEED + zlib.crc32(split.encode()))
        uk = pd.Series(t.column("key").to_pylist())
        cap = {k: Q6_GLOBAL_CAP.get(tuple(k[2:-2].split("', '")), Q6_DEFAULT_CAP) for k in uk.unique()}
        r = pd.DataFrame({"key": uk, "r": rng.random(len(uk))})
        rank = r.groupby("key")["r"].rank(method="first")
        capv = uk.map(lambda k: cap[k] if cap[k] is not None else 10 ** 12)
        t = t.filter(pa.array((rank <= capv).to_numpy()))
    pq.write_table(t, p["out"], compression="zstd")
    return {"quantity": q, "split": split, "rows": t.num_rows}


def ledger_merge_task(p: dict) -> dict:
    """ledger_<split>.parquet from the chunk ledgers of that split; benchmark codes dropped; dispositions assigned."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fit.extractor import LEDGER_SCHEMA
    from fit import sanity as eligibility

    tabs = [pq.read_table(f).select(LEDGER_SCHEMA.names).cast(LEDGER_SCHEMA) for f in p["files"]]
    df = pa.concat_tables(tabs).to_pandas()
    df = df[~df["csd_code"].isin(set(p["exclude"]))].reset_index(drop=True)
    df["disposition"] = [eligibility.disposition(k) for k in df["kind"]]
    pq.write_table(pa.Table.from_pandas(df, schema=LEDGER_SCHEMA, preserve_index=False), p["out"], compression="zstd")
    return {"ledger": p["split"], "rows": len(df)}


def hist_sum_task(p: dict) -> dict:
    """Sum the shard histograms of one split (arrays ``h:<label>``; the Q6 distance grid is stored alongside)."""
    import numpy as np
    from fit import q6p as q6pmf

    h = q6pmf.load_hist([Path(f) for f in p["files"]])
    np.savez_compressed(p["out"], edges=q6pmf.EDGES, **{"h:" + k: v for k, v in h.items()})
    return {"prefix": p["prefix"], "split": p["split"], "labels": len(h)}


def run_merge(pool, shard_dir: Path, obs_dir: Path, shards: Sequence[Shard], exclude: Sequence[str],
              chunks: Dict[str, List[tuple]]) -> dict:
    """Per split: observations concatenated in (shard, chunk) order; Q6 from the shard samples; histograms summed."""
    obs_dir.mkdir(parents=True, exist_ok=True)
    by_split: Dict[str, List[str]] = {}
    for s in shards:
        by_split.setdefault(s.split, []).append(s.name)
    jobs = []
    for split in SPLITS:
        names = sorted(by_split.get(split, ()))
        if not names:
            continue
        tags = [chunk_tag(n, c[0]) for n in names for c in chunks[n]]
        for q in OBS_QUANTITIES:
            files = ([shard_dir / f"observations_Q6_{n}.parquet" for n in names] if q == "Q6"
                     else [shard_dir / f"observations_{q}_{t}.parquet" for t in tags])
            jobs.append((merge_task, dict(quantity=q, split=split, exclude=list(exclude), files=[str(f) for f in files],
                                          out=str(obs_dir / f"observations_{q}_{split}.parquet"))))
        if profile_get("donor_assignment"):
            jobs.append((ledger_merge_task, dict(split=split, exclude=list(exclude),
                                                 files=[str(shard_dir / f"ledger_{t}.parquet") for t in tags],
                                                 out=str(obs_dir / f"ledger_{split}.parquet"))))
        for prefix in HIST_PREFIXES:
            jobs.append((hist_sum_task, dict(prefix=prefix, split=split,
                                             files=[str(shard_dir / f"{prefix}_{t}.npz") for t in tags],
                                             out=str(obs_dir / f"{prefix}_{split}.npz"))))
    # the biggest tables (Q5, Q2, Q1) first
    weight = {"Q5": 0, "Q2": 1, "Q1": 2, "Q3": 3, "Q4": 4, "Q6": 5, "QO": 6}
    order = sorted(range(len(jobs)), key=lambda i: (weight.get(jobs[i][1].get("quantity"), 7), i))
    futs = {i: pool.submit(*jobs[i]) for i in order}
    res = [pool.collect([futs[i]])[0] for i in range(len(jobs))]
    rows = {f"{r['quantity']}_{r['split']}": r["rows"] for r in res if "quantity" in r}
    labels = {f"{r['prefix']}_{r['split']}": r["labels"] for r in res if "prefix" in r}
    ledgers = {r["ledger"]: r["rows"] for r in res if "ledger" in r}
    return {"rows": rows, "hist_labels": labels, **({"ledger_rows": ledgers} if ledgers else {})}


# ---------------------------------------------------------------------------
# Runtime environment, pinning & stage telemetry
# ---------------------------------------------------------------------------


THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def pin_threads() -> None:
    """One thread per numerical library in this process and every child it spawns."""
    for var in THREAD_VARS:
        os.environ[var] = "1"


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # kB on Linux


def profile_env(profile: str = "production", flags_path: Optional[Path] = None) -> Dict[str, str]:
    """Environment that makes ``jacobik.core.profile`` resolve to this build profile in every process.

    ``profile`` is a name in ``jacobik.core.profile.PROFILES``; ``flags_path`` a JSON object of extra overrides. This is
    the only way flags reach a stage or a worker.
    """
    overrides = json.loads(Path(flags_path).read_text()) if flags_path else {}
    return {"JACOBIK_PROFILE": profile,
            "JACOBIK_PROFILE_OVERRIDES": json.dumps(overrides, sort_keys=True, separators=(",", ":")) if overrides else ""}


def apply_profile_env(env: Dict[str, str]) -> None:
    """Install ``env`` and check the profile module (resolved at import) agrees with it."""
    for k, v in env.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    from jacobik.core import profile

    want = json.loads(env.get("JACOBIK_PROFILE_OVERRIDES") or "{}")
    if profile.ACTIVE_NAME != env["JACOBIK_PROFILE"] or dict(profile.OVERRIDES) != want:
        raise RuntimeError(
            f"jacobik.core.profile was imported as {profile.ACTIVE_NAME!r}+{dict(profile.OVERRIDES)} before the build profile "
            f"{env['JACOBIK_PROFILE']!r}+{want} could be set; start the build with `python -m fit.build_db`")


class StageLog:
    """Wall time, CPU time and peak RSS per stage; for pool stages also the parallel efficiency."""

    def __init__(self, workers: int):
        self.workers = workers
        self.stages: list[dict[str, Any]] = []
        self.t0 = time.perf_counter()

    @contextlib.contextmanager
    def stage(self, name: str, *, items: Optional[int] = None, pool=None):
        rec: dict[str, Any] = {"stage": name}
        if items is not None:
            rec["items"] = items
        busy0 = pool.busy_s if pool is not None else 0.0
        t, c = time.perf_counter(), time.process_time()
        try:
            yield rec
        finally:
            wall = time.perf_counter() - t
            rec["wall_s"] = round(wall, 3)
            rec["main_cpu_s"] = round(time.process_time() - c, 3)
            rec["main_peak_rss_mb"] = round(peak_rss_mb(), 1)
            if pool is not None and pool.workers > 1:
                busy = pool.busy_s - busy0
                rec["workers"] = pool.workers
                rec["worker_busy_s"] = round(busy, 3)
                rec["worker_peak_rss_mb"] = round(pool.max_worker_rss_mb, 1)
                rec["parallel_efficiency"] = round(busy / (wall * pool.workers), 4) if wall > 0 else None
            self.stages.append(rec)
            log.info("stage %-10s %8.1f s  main RSS %.0f MB%s", name, wall, rec["main_peak_rss_mb"],
                     f"  efficiency {rec['parallel_efficiency']:.2f}" if rec.get("parallel_efficiency") else "")

    def summary(self) -> dict[str, Any]:
        return {"workers": self.workers, "total_wall_s": round(time.perf_counter() - self.t0, 3),
                "stages": self.stages}
