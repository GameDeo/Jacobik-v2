"""
extractor.py
============
Step 4 — Geometry extractor.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pandas as _pd
import pyarrow.parquet as pq
from fit.sanity import STABLE_ANCESTOR_SAMPLING, observation_id
from jacobik.core import profile as _profile
from fit.sanity import LEDGER_VERSION

#: write the extraction ledger (source corrections / uncertainty events) beside the observations
DONOR_ASSIGNMENT = _profile.get("donor_assignment")
LEDGER_SCHEMA = pa.schema([pa.field(n, pa.string()) for n in
                           ("ledger_version", "csd_code", "dataset", "metal")]
                          + [pa.field("event_idx", pa.int32())]
                          + [pa.field(n, pa.string()) for n in ("kind", "disposition", "before", "after", "detail", "dependent")])


def ledger_rows(mol, dataset, all_kv, qo_keys):
    """Ledger rows of one complex: one per source event (disposition is assigned at merge by fit.sanity), each with the quantities/keys that depend on the donor set.

    A donor-count or coordination-geometry change can invalidate every distance, angle and orientation row of the
    complex, so the dependency is the complex's whole Q1-Q5 and QO key set (counts and distinct keys)."""
    import json

    events = list(getattr(mol, "donor_identity_events", None) or ())
    if not events:
        return []
    syms = {a.idx: a.symbol for a in mol.atoms}
    dep = {}
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        kv = all_kv.get(q, ())
        dep[q] = {"n": len(kv), "keys": sorted({repr(k) for k, _ in kv})}
    dep["QO"] = {"n": len(qo_keys), "keys": sorted(set(qo_keys))}
    dep_json = json.dumps(dep, separators=(",", ":"), default=str)

    def atoms(v):
        return json.dumps([[i, syms.get(i, "?")] for i in (v or ())], default=str) if v is not None else None

    out = []
    for n, ev in enumerate(events):
        kind = ev.get("kind", "unknown")
        before = ev.get("before", ev.get("face"))
        after = ev.get("after", ev.get("added", ev.get("atoms")))
        detail = {k: v for k, v in ev.items() if k not in ("kind", "before", "after")}
        out.append({"ledger_version": LEDGER_VERSION, "csd_code": mol.csd_code, "dataset": dataset,
                    "metal": mol.metal_symbol or "", "event_idx": n, "kind": kind,
                    "disposition": "", "before": atoms(before), "after": atoms(after),
                    "detail": json.dumps(detail, separators=(",", ":"), default=str), "dependent": dep_json})
    return out

log = logging.getLogger(__name__)


QUANTITIES = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
FLUSH_EVERY = 100  # row-group flush interval for Q1–Q5

_SCHEMA = pa.schema(
    [
        pa.field("csd_code", pa.string()),
        pa.field("key", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("metal", pa.string()),
        pa.field("coord_geom", pa.string()),
    ]
)

#: observations_QO_*.parquet: one declared schema, never inferred from a chunk (a chunk without eta faces has all-null
#: raw_* columns, which pandas/pyarrow would infer as a plain null/double column and concat_tables would refuse)
QO_SCHEMA = pa.schema(
    [
        pa.field("csd_code", pa.large_string()), pa.field("hb_min", pa.float64()), pa.field("ring", pa.int64()),
        pa.field("sub", pa.large_string()), pa.field("metal", pa.large_string()), pa.field("geom", pa.large_string()),
        pa.field("anion", pa.int64()), pa.field("k", pa.int64()), pa.field("sub_elems", pa.large_string()),
    ]
    + [pa.field(f"g{i}", pa.float64()) for i in range(6)]
    + [pa.field("raw_face", pa.list_(pa.float64())), pa.field("raw_face_elems", pa.list_(pa.string())),
       pa.field("raw_closed", pa.bool_()), pa.field("raw_subs", pa.list_(pa.float64()))]
)
if STABLE_ANCESTOR_SAMPLING:
    QO_SCHEMA = QO_SCHEMA.append(pa.field("obs_id", pa.string()))


def observation_schema(quantity):
    if STABLE_ANCESTOR_SAMPLING and quantity in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        return _SCHEMA.append(pa.field("obs_id", pa.string()))
    return _SCHEMA

# Global cap on Q6 observations per element-pair key per split.
# Q6_DEFAULT_CAP applies to any key not explicitly listed in Q6_GLOBAL_CAP.
# Set to None to keep unlisted keys in full (not recommended with H included).
Q6_DEFAULT_CAP: Optional[int] = 1_500

Q6_GLOBAL_CAP: Dict[Tuple[str, str], int] = {
    ("C", "C"): 5_000,
    ("C", "H"): 4_000,
    ("H", "H"): 2_000,
    ("C", "N"): 4_000,
    ("C", "O"): 4_000,
    ("C", "P"): 3_000,
    ("C", "S"): 3_000,
    ("N", "N"): 2_500,
    ("H", "N"): 3_000,
    ("H", "O"): 3_000,
    ("H", "P"): 2_000,
    ("H", "S"): 2_000,
}


# ---------------------------------------------------------------------------
# Q6 reservoir (Vitter's Algorithm R)
# ---------------------------------------------------------------------------


@dataclass
class _Q6Reservoir:
    """
    Maintains a fixed-size uniform random sample per Q6 key across the
    entire split.  Feed observations in with .add(); drain at the end with
    .to_rows().
    """

    # reservoir[key] = list of (csd_code, key_str, value, metal, coord_geom)
    _reservoir: Dict[str, List[Tuple]] = field(default_factory=dict)
    # count of total items *seen* per key (including rejected ones)
    _seen: Dict[str, int] = field(default_factory=dict)
    # seeded so a re-run reproduces the same reservoir
    _rng: random.Random = field(default_factory=lambda: random.Random(20260101))
    # uncapped keys accumulate here without a size limit
    _uncapped: Dict[str, List[Tuple]] = field(default_factory=dict)

    def add(
        self,
        key_str: str,
        value: float,
        csd_code: str,
        metal: str,
        coord_geom: str,
    ) -> None:
        # Parse the key tuple from its repr to look up the cap.
        # key_str is always "('X', 'Y')" — two sorted element symbols.
        # Fast path: slice out the two symbols directly.
        try:
            inner = key_str[2:-2]  # strip "('" and "')"
            a, b = inner.split("', '")  # split on "', '"
            cap_key = (a, b)
        except Exception:
            cap_key = None

        cap = Q6_GLOBAL_CAP.get(cap_key, Q6_DEFAULT_CAP) if cap_key else Q6_DEFAULT_CAP

        row = (csd_code, key_str, value, metal, coord_geom)

        if cap is None:
            # Explicitly uncapped — accumulate freely
            if key_str not in self._uncapped:
                self._uncapped[key_str] = []
            self._uncapped[key_str].append(row)
            return

        # Vitter's Algorithm R
        n = self._seen.get(key_str, 0) + 1
        self._seen[key_str] = n

        if n <= cap:
            if key_str not in self._reservoir:
                self._reservoir[key_str] = []
            self._reservoir[key_str].append(row)
        else:
            # Replace a random slot with probability cap/n
            j = self._rng.randint(1, n)
            if j <= cap:
                self._reservoir[key_str][j - 1] = row

    def size(self) -> int:
        """Current number of rows held (reservoir + uncapped)."""
        return sum(len(v) for v in self._reservoir.values()) + sum(
            len(v) for v in self._uncapped.values()
        )

    def to_rows(self) -> List[Tuple]:
        """Return all retained rows as a flat list."""
        out: List[Tuple] = []
        for rows in self._reservoir.values():
            out.extend(rows)
        for rows in self._uncapped.values():
            out.extend(rows)
        return out


class _Q6Stream(_Q6Reservoir):
    """Keeps every Q6 observation in add order (a chunk of a shard); ``replay_q6`` feeds the chunks of a shard, in
    order, through the shard's seeded reservoir, which reproduces a single-pass extraction of the whole shard."""

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[Tuple] = []

    def add(self, key_str, value, csd_code, metal, coord_geom) -> None:
        self._rows.append((csd_code, key_str, value, metal, coord_geom))

    def size(self) -> int:
        return len(self._rows)

    def to_rows(self) -> List[Tuple]:
        return self._rows


def replay_q6(chunk_files, out_file: Path, compression: str = "zstd") -> int:
    """Shard-level Q6 table from the uncapped Q6 streams of its chunks (given in file order)."""
    reservoir = _Q6Reservoir()
    for f in chunk_files:
        t = pq.read_table(str(f)).to_pydict()
        for code, key, value, metal, geom in zip(t["csd_code"], t["key"], t["value"], t["metal"], t["coord_geom"]):
            reservoir.add(key, value, code, metal, geom)
    writer = pq.ParquetWriter(str(out_file), _SCHEMA, compression=compression)
    try:
        return _flush_q6_reservoir(writer, reservoir)
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_writers(
    out_dir: Path,
    split: str,
    compression: str = "zstd",
) -> Dict[str, pq.ParquetWriter]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        q: pq.ParquetWriter(
            str(out_dir / f"observations_{q}_{split}.parquet"),
            observation_schema(q),
            compression=compression,
        )
        for q in QUANTITIES
    }


def _flush_q1_q5(
    writers: Dict[str, pq.ParquetWriter],
    buffers: Dict[str, Dict[str, List]],
) -> None:
    """Write one row group per Q1–Q5 quantity and clear buffers."""
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        buf = buffers[q]
        n = len(buf["csd_code"])
        if n == 0:
            continue
        writers[q].write_table(
            pa.table(
                {
                    "csd_code": pa.array(buf["csd_code"], type=pa.string()),
                    "key": pa.array(buf["key"], type=pa.string()),
                    "value": pa.array(buf["value"], type=pa.float64()),
                    "metal": pa.array(buf["metal"], type=pa.string()),
                    "coord_geom": pa.array(buf["coord_geom"], type=pa.string()),
                    **({"obs_id": pa.array(buf["obs_id"], type=pa.string())} if STABLE_ANCESTOR_SAMPLING else {}),
                },
                schema=observation_schema(q),
            )
        )
        for lst in buf.values():
            lst.clear()


def _flush_q6_reservoir(
    writer: pq.ParquetWriter,
    reservoir: _Q6Reservoir,
) -> int:
    """Write the entire Q6 reservoir as a single row group. Returns row count."""
    rows = reservoir.to_rows()
    if not rows:
        return 0
    csd_codes, keys, values, metals, geoms = zip(*rows)
    writer.write_table(
        pa.table(
            {
                "csd_code": pa.array(list(csd_codes), type=pa.string()),
                "key": pa.array(list(keys), type=pa.string()),
                "value": pa.array(list(values), type=pa.float64()),
                "metal": pa.array(list(metals), type=pa.string()),
                "coord_geom": pa.array(list(geoms), type=pa.string()),
            },
            schema=_SCHEMA,
        )
    )
    return len(rows)


pd_NA = _pd.NA


def _qo_cell(v):
    if v is None or v is pd_NA or (isinstance(v, float) and v != v):
        return None
    return v.tolist() if hasattr(v, "tolist") else v


def _qo_table(qo_rows, cols) -> pa.Table:
    """QO rows -> a table of the declared QO_SCHEMA, column by column (a chunk may have no eta face at all)."""
    arrays = []
    for name in cols:
        field = QO_SCHEMA.field(name)
        arrays.append(pa.array([_qo_cell(r.get(name)) for r in qo_rows], type=field.type))
    return pa.Table.from_arrays(arrays, schema=QO_SCHEMA)


def _empty_buffers() -> Dict[str, Dict[str, List]]:
    return {
        q: {"csd_code": [], "key": [], "value": [], "metal": [], "coord_geom": [],
            **({"obs_id": []} if STABLE_ANCESTOR_SAMPLING else {})}
        for q in ("Q1", "Q2", "Q3", "Q4", "Q5")
    }


HIST_PASSES = ("q6hist", "q6hist_m13")


def _hist_accumulators() -> Dict:
    from fit import q6p

    return {"q6hist": q6p.HistAccumulator(), "q6hist_m13": q6p.HistAccumulator(records=q6p.m13_records)}


def extract_split(
    data_dir: str | Path,
    split: str,
    out_dir: str | Path,
    limit: Optional[int] = None,
    flush_every: int = FLUSH_EVERY,
    compression: str = "zstd",
    progress: bool = True,
    progress_file: Optional[Path] = None,
    orientation: bool = True,
    hypothesis: bool = True,
    q6_hist: bool = True,
    q6_hist_exclude: Optional[set] = None,
    q6_hist_weight: Optional[Dict[str, float]] = None,
    block_range: Optional[Tuple[int, int]] = None,
    q6_uncapped: bool = False,
    out_tag: Optional[str] = None,
) -> Dict:
    """
    Extract Q1–Q6 observations for every complex in *split* and write them
    to per-quantity Parquet files under *out_dir*.


    Parameters
    ----------
    data_dir      : tmQM dataset root (xyz/ bo/ subdirs + tmQM_y.csv)
    split         : "X1", "X2", "X3", etc.
    out_dir       : directory for output parquet files
    limit         : stop after this many complexes (None = full split)
    flush_every   : Q1–Q5 row-group flush interval (complexes)
    compression   : parquet codec ("zstd", "snappy", "none")
    progress      : tqdm bar (CLI/direct mode only)
    progress_file : per-split file for batch progress lines (notebook mode)
    orientation   : also write observations_QO_<split>.parquet, one row per kappa
                    donor with the joint-orientation features (stats/orientation.py)
    hypothesis    : with orientation, also write the QG / QP / QB hypothesis-layer tables (the DB builder skips them)
    q6_hist       : also write the uncapped histograms (HIST_PASSES): q6hist_<split>.npz (Q6 contact distances per
                    key, bond-separation class and H-bond angle class: q6p.py) and q6hist_m13_<split>.npz (metal
                    1-3 contacts: q6p.m13_records)
    q6_hist_exclude : CSD codes left out of the histograms (the benchmark sets; histograms cannot be filtered later)
    q6_hist_weight  : {code: weight} for the histograms (LnQM siblings: 1 / siblings of the ligand set); default 1
    block_range     : (lo, hi) file-block range of the shard (a chunk); the shard is the concatenation of its chunks
    q6_uncapped     : write every Q6 observation in add order (chunk mode) instead of the seeded reservoir sample;
                      ``replay_q6`` turns the chunks of a shard into exactly the single-pass shard sample
    out_tag         : file-name suffix of the outputs (default: the split name), e.g. "X1s000.c003" for a chunk

    Returns
    -------
    dict: split, n_complexes, n_errors, elapsed_s, ms_per_mol, Q1_rows…Q6_rows

    """
    from fit.parser import TMQMParser

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    parser = TMQMParser(data_dir)
    tag = out_tag or split
    writers = _open_writers(out_dir, tag, compression=compression)
    buffers = _empty_buffers()
    reservoir = _Q6Stream() if q6_uncapped else _Q6Reservoir()
    hists = _hist_accumulators() if q6_hist else {}
    hist_errors = {name: 0 for name in hists}

    row_counts: Dict[str, int] = {q: 0 for q in QUANTITIES}
    qo_rows: List[dict] = []
    ledger: List[dict] = []
    qo_errors = 0
    hyp_rows: Dict[str, List[dict]] = {"QG": [], "QP": [], "QB": []}
    hyp_errors = 0
    n_complexes = 0
    n_errors = 0
    q_errors: Dict[str, int] = {q: 0 for q in QUANTITIES}
    t_start = time.perf_counter()
    t_batch = t_start

    if progress_file is not None:
        progress_file = Path(progress_file)
        progress_file.write_text("")

    stream = parser.stream_split(split, limit=limit, block_range=block_range)
    if progress:
        try:
            from tqdm import tqdm

            stream = tqdm(stream, desc=f"split={split}", unit="mol", dynamic_ncols=True)
        except ImportError:
            pass

    try:
        for mol in stream:
            try:
                all_kv = all_keys(mol, errors=q_errors)
            except Exception as exc:
                log.error("%s: all_keys failed: %s", mol.csd_code, exc)
                n_errors += 1
                n_complexes += 1
                continue

            code = mol.csd_code
            metal = mol.metal_symbol or ""
            geom = mol.coord_geom or "unknown"

            # Q1–Q5: straight into the flush buffer
            for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
                buf = buffers[q]
                occurrence = Counter()
                for key_tuple, value in all_kv.get(q, ()):
                    key = repr(key_tuple)
                    if STABLE_ANCESTOR_SAMPLING:
                        buf["obs_id"].append(observation_id(Path(data_dir).name, code, q, key, occurrence[key]))
                        occurrence[key] += 1
                    buf["csd_code"].append(code)
                    buf["key"].append(repr(key_tuple))
                    buf["value"].append(float(value))
                    buf["metal"].append(metal)
                    buf["coord_geom"].append(geom)

            # Q6: into the reservoir sampler
            for key_tuple, value in all_kv.get("Q6", ()):
                reservoir.add(repr(key_tuple), float(value), code, metal, geom)

            # uncapped histograms (Q6 PMF, metal 1-3): benchmark codes left out here, they cannot be filtered later
            if hists and not (q6_hist_exclude and code in q6_hist_exclude):
                wt = (q6_hist_weight or {}).get(code, 1.0)
                for name, acc in hists.items():
                    try:
                        acc.add(mol, wt)
                    except Exception as exc:
                        hist_errors[name] += 1
                        log.debug("%s: %s histogram failed: %s", code, name, exc)

            # QO: joint donor orientation rows (a few per complex; kept in memory, written once)
            if orientation:
                try:
                    from fit.qo import donor_rows

                    rows = donor_rows(mol)
                    if STABLE_ANCESTOR_SAMPLING:
                        occurrence = Counter()
                        for row in rows:
                            key = repr(tuple(row[k] for k in ("k", "ring", "sub", "metal", "geom", "anion")))
                            row["obs_id"] = observation_id(Path(data_dir).name, code, "QO", key, occurrence[key])
                            occurrence[key] += 1
                    qo_rows.extend(rows)
                except Exception as exc:
                    qo_errors += 1
                    rows = []
                    log.debug("%s: QO rows failed: %s", code, exc)
            if DONOR_ASSIGNMENT:
                try:
                    qo_keys = [repr(tuple(r[k] for k in ("k", "ring", "sub", "metal", "geom", "anion")))
                               for r in (rows if orientation else ())]
                    ledger.extend(ledger_rows(mol, Path(data_dir).name, all_kv, qo_keys))
                except Exception as exc:  # a ledger problem never drops a complex
                    log.warning("%s: ledger rows failed: %s", code, exc)

            # QG / QP / QB: rows for the hypothesis-layer priors (stats/hypothesis_tables.py, B2.5)
            if orientation and hypothesis:
                try:
                    from jacobik.stats import hypothesis_tables as HT

                    g = HT.geometry_row(mol)
                    if g is not None:
                        hyp_rows["QG"].append(g)
                    hyp_rows["QP"].extend(HT.partition_rows(mol))
                    hyp_rows["QB"].extend(HT.candidate_rows(mol))
                except Exception as exc:
                    hyp_errors += 1
                    log.debug("%s: hypothesis-table rows failed: %s", code, exc)

            n_complexes += 1

            if n_complexes % flush_every == 0:
                batch_obs = {q: len(buffers[q]["csd_code"]) for q in ("Q1", "Q2", "Q3", "Q4", "Q5")}
                for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
                    row_counts[q] += batch_obs[q]
                _flush_q1_q5(writers, buffers)

                t_now = time.perf_counter()
                batch_s = t_now - t_batch
                t_batch = t_now
                elapsed_so_far = t_now - t_start
                q6_res_size = reservoir.size()

                obs_str = "  ".join(f"{q}={batch_obs[q]:,}" for q in ("Q1", "Q2", "Q3", "Q4", "Q5"))
                line = (
                    f"[{split}] mol {n_complexes:>6}  "
                    f"{batch_s / flush_every * 1000:4.1f} ms/mol  "
                    f"batch: {obs_str}  Q6_reservoir={q6_res_size:,}  "
                    f"elapsed: {elapsed_so_far:.1f}s"
                )
                if progress_file is not None:
                    with open(progress_file, "a") as pf:
                        pf.write(line + "\n")
                else:
                    print(line, flush=True)

    finally:
        # Flush remaining Q1–Q5 buffer
        for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            row_counts[q] += len(buffers[q]["csd_code"])
        _flush_q1_q5(writers, buffers)

        # Write Q6 reservoir in one shot
        row_counts["Q6"] = _flush_q6_reservoir(writers["Q6"], reservoir)

        for w in writers.values():
            w.close()

        for name, acc in hists.items():
            acc.save(out_dir / f"{name}_{tag}.npz")
            if hist_errors[name]:
                log.warning("split=%s: %s failed for %d complexes", split, name, hist_errors[name])

        if DONOR_ASSIGNMENT:
            cols = LEDGER_SCHEMA.names
            pq.write_table(pa.table({c: [r[c] for r in ledger] for c in cols}, schema=LEDGER_SCHEMA),
                           out_dir / f"ledger_{tag}.parquet", compression="zstd")

        if orientation:
            import pandas as pd

            from jacobik.stats import hypothesis_tables as HT

            comp = None if compression == "none" else compression
            # raw_* = haptic face geometry (B2.2; kappa rows leave them empty) -- needed by the per-substituent fit
            cols = (["csd_code", "hb_min", "ring", "sub", "metal", "geom", "anion", "k", "sub_elems"]
                    + [f"g{i}" for i in range(6)] + ["raw_face", "raw_face_elems", "raw_closed", "raw_subs"])
            if STABLE_ANCESTOR_SAMPLING:
                cols.append("obs_id")
            pq.write_table(_qo_table(qo_rows, cols), out_dir / f"observations_QO_{tag}.parquet",
                           compression=comp or "NONE")
            for q, c in (("QG", HT.QG_COLS), ("QP", HT.QP_COLS), ("QB", HT.QB_COLS)):
                if hypothesis:
                    pd.DataFrame(hyp_rows[q], columns=c).to_parquet(out_dir / f"observations_{q}_{tag}.parquet",
                                                                    compression=comp)

    elapsed = time.perf_counter() - t_start
    ms_per = (elapsed / n_complexes * 1000) if n_complexes else 0.0

    if progress_file is not None:
        with open(progress_file, "a") as pf:
            pf.write(
                f"[{split}] DONE  complexes={n_complexes}  errors={n_errors}  "
                f"elapsed={elapsed:.1f}s  {ms_per:.1f}ms/mol\n"
            )

    log.info(
        "split=%s  complexes=%d  errors=%d  elapsed=%.1fs  %.1fms/mol",
        split,
        n_complexes,
        n_errors,
        elapsed,
        ms_per,
    )
    result = {
        "split": split,
        "n_complexes": n_complexes,
        "n_errors": n_errors,
        "n_parse_failed": parser.n_failed,
        "elapsed_s": elapsed,
        "ms_per_mol": ms_per,
    }
    result.update({f"{q}_rows": row_counts[q] for q in QUANTITIES})
    if DONOR_ASSIGNMENT:
        result["ledger_events"] = len(ledger)
        result["ledger_parents"] = len({r["csd_code"] for r in ledger})
    result["quantity_errors"] = dict(q_errors)
    if orientation:
        result["QO_rows"] = len(qo_rows)
        result["quantity_errors"]["QO"] = qo_errors
        for q in ("QG", "QP", "QB"):
            result[f"{q}_rows"] = len(hyp_rows[q])
        result["quantity_errors"]["QG/QP/QB"] = hyp_errors
    return result

# Convenience: produce all keys for a complex in one call
# ---------------------------------------------------------------------------


def all_keys(
    mol,
    errors: Optional[Dict[str, int]] = None,
) -> Dict[str, List[tuple]]:
    """
    Return a dict mapping quantity name to its list of (key, value) pairs.

    A failing quantity yields an empty list for that complex.  Failures are
    never silent: they are logged and, if ``errors`` is given, counted per
    quantity (``errors["Q3"] += 1``) so the extractor can report them.
    """
    import zlib
    from fit import q1, q2, q3, q4, q5, q6

    result: Dict[str, List[KeyValue]] = {}
    seed = zlib.crc32(str(mol.csd_code).encode())
    for name, fn in (
        ("Q1", q1.q1_bond_lengths),
        ("Q2", q2.q2_dmd_angles),
        ("Q3", q3.q3_approach_angles),
        ("Q4", q4.q4_pyramidalization),
        ("Q5", q5.q5_torsions),
        ("Q6", lambda m: q6.q6_steric_contacts(m, seed=seed)),
    ):
        try:
            result[name] = fn(mol)
        except Exception as exc:
            log.error("%s: %s failed: %s", mol.csd_code, name, exc)
            result[name] = []
            if errors is not None:
                errors[name] = errors.get(name, 0) + 1
    return result
