"""
expander.py
===========
Step 4b — Ancestor key expander.
"""

from __future__ import annotations

import ast
import logging
import random
import time
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from jacobik.stats import ladders as _L
from jacobik.stats.ladders import TRANS_NONE
from fit.sanity import STABLE_ANCESTOR_SAMPLING, priority

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

#: Maximum observations any single L0 subclass may contribute to one ancestor
#: node per split.  L0 rows are never capped.
N_ANCESTOR_CAP: int = 500

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


_OUT_SCHEMA = pa.schema(
    [
        pa.field("csd_code", pa.string()),
        pa.field("key", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("metal", pa.string()),
        pa.field("coord_geom", pa.string()),
        pa.field("split", pa.string()),
        pa.field("level", pa.int16()),
        pa.field("is_l0", pa.bool_()),
    ]
)
if STABLE_ANCESTOR_SAMPLING:
    _OUT_SCHEMA = _OUT_SCHEMA.append(pa.field("obs_id", pa.string()))

# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------


def _parse_key(key_str: str) -> tuple:
    """Parse a repr() key string back to a Python tuple (strings + ints)."""
    return ast.literal_eval(key_str)


# ---------------------------------------------------------------------------
# Ancestor generators: ONE definition shared with the runtime lookup
# ---------------------------------------------------------------------------

_ANCESTOR_FN = _L.ANCESTOR_FN
_l0_subclass = _L.l0_bucket

# ---------------------------------------------------------------------------
# Ancestor reservoir (Vitter's Algorithm R, keyed by ancestor+subclass)
# ---------------------------------------------------------------------------

# Row tuple stored in reservoir: (csd_code, key_str, value, metal, coord_geom, split, level)
_ReservoirRow = Tuple[str, str, float, str, str, str, int]


@dataclass
class _AncestorReservoir:
    """
    Keeps at most N_ANCESTOR_CAP rows per (ancestor_key_str, l0_subclass_str)
    bucket across a full split, using Vitter's Algorithm R.
    """

    cap: int = N_ANCESTOR_CAP

    # L0 rows: always kept in full
    _l0_rows: List[_ReservoirRow] = field(default_factory=list)

    # Ancestor rows: reservoir per bucket_key
    _reservoir: Dict[Tuple[str, str], List[_ReservoirRow]] = field(default_factory=dict)
    _seen: Dict[Tuple[str, str], int] = field(default_factory=dict)
    # seeded so that a re-run reproduces the same reservoir
    _rng: random.Random = field(default_factory=lambda: random.Random(20260101))
    _priorities: dict = field(default_factory=dict)
    _input_ids: set = field(default_factory=set)

    def add_l0(self, row: _ReservoirRow) -> None:
        self._l0_rows.append(row)

    def add_ancestor(
        self,
        ancestor_key_str: str,
        l0_subclass_str: str,
        row: _ReservoirRow,
    ) -> None:
        bk = (ancestor_key_str, l0_subclass_str)
        if STABLE_ANCESTOR_SAMPLING:
            identity = row[7]
            if not identity:
                raise ValueError("stable ancestor sampling needs extraction-time obs_id")
            heap = self._priorities.setdefault(bk, [])
            item = (-priority(bk, identity), identity, row)
            if len(heap) < self.cap:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            return
        n = self._seen.get(bk, 0) + 1
        self._seen[bk] = n

        if n <= self.cap:
            if bk not in self._reservoir:
                self._reservoir[bk] = []
            self._reservoir[bk].append(row)
        else:
            j = self._rng.randint(1, n)
            if j <= self.cap:
                self._reservoir[bk][j - 1] = row

    def to_rows(self) -> List[_ReservoirRow]:
        if STABLE_ANCESTOR_SAMPLING:
            out = sorted(self._l0_rows, key=lambda row: row[7] or "")
            for bucket in sorted(self._priorities):
                out.extend(item[2] for item in sorted(self._priorities[bucket], key=lambda item: (-item[0], item[1])))
            return out
        out = list(self._l0_rows)
        for rows in self._reservoir.values():
            out.extend(rows)
        return out

    def l0_count(self) -> int:
        return len(self._l0_rows)

    def ancestor_count(self) -> int:
        if STABLE_ANCESTOR_SAMPLING:
            return sum(len(heap) for heap in self._priorities.values())
        return sum(len(v) for v in self._reservoir.values())


# ---------------------------------------------------------------------------
# Core expansion logic (fills reservoir; no direct parquet writes)
# ---------------------------------------------------------------------------


def _expand_into_reservoir(
    quantity: str,
    batch: pa.RecordBatch,
    reservoir: _AncestorReservoir,
) -> None:
    """
    Process one input batch: emit L0 rows unconditionally, feed ancestor rows
    through the reservoir per (ancestor_key, l0_subclass) bucket.

    Q1 rows without a trans partner have no meaningful 6-tuple L0 (its last
    field would be the literal 'none'); they are not emitted as L0 and instead
    feed the 5-tuple 'geom' level (and everything below) through the reservoir.
    """
    ancestor_fn = _ANCESTOR_FN.get(quantity)

    csd_codes = batch.column("csd_code").to_pylist()
    identities = [None] * len(csd_codes)
    if STABLE_ANCESTOR_SAMPLING and quantity != "Q6":
        if batch.schema.get_field_index("obs_id") < 0:
            raise ValueError("stable ancestor sampling requires fresh observations with obs_id")
        identities = batch.column("obs_id").to_pylist()
        if any(not identity for identity in identities):
            raise ValueError("missing stable observation identity")
        if len(set(identities)) != len(identities) or reservoir._input_ids.intersection(identities):
            raise ValueError("duplicate extraction observation identity")
        reservoir._input_ids.update(identities)
    key_strs = batch.column("key").to_pylist()
    values = batch.column("value").to_pylist()
    metals = batch.column("metal").to_pylist()
    geoms = batch.column("coord_geom").to_pylist()
    splits_col = (
        batch.column("split").to_pylist()
        if batch.schema.get_field_index("split") >= 0
        else [""] * batch.num_rows
    )

    for csd, key_str, val, metal, geom, split, identity in zip(
        csd_codes, key_strs, values, metals, geoms, splits_col, identities
    ):
        suffix = (identity,) if STABLE_ANCESTOR_SAMPLING else ()
        if ancestor_fn is None:  # Q6: L0 only
            reservoir.add_l0((csd, key_str, val, metal, geom, split, 0) + suffix)
            continue

        key_tuple = _parse_key(key_str)

        l0_emitted = True
        l0_level = 0
        if quantity == "Q1" and len(key_tuple) > 5 and key_tuple[5] == TRANS_NONE:
            l0_emitted, l0_level = False, -1
        if l0_emitted:
            reservoir.add_l0((csd, key_str, val, metal, geom, split, 0) + suffix)

        try:
            anc_rows = ancestor_fn(key_tuple)
        except Exception as exc:
            log.warning("Ancestor expansion failed for %s key %r: %s", quantity, key_str, exc)
            continue

        l0_sub = _l0_subclass(quantity, key_tuple)
        for level, anc_key in anc_rows:
            if level <= l0_level:
                continue
            anc_key_str = repr(anc_key)
            if l0_emitted and anc_key_str == key_str:
                continue
            reservoir.add_ancestor(
                anc_key_str, l0_sub, (csd, anc_key_str, val, metal, geom, split, level) + suffix
            )


def _flush_reservoir(
    reservoir: _AncestorReservoir,
    writer: pq.ParquetWriter,
    batch_size: int = 50_000,
) -> int:
    """
    Write all rows from the reservoir to parquet in batches.
    Returns total row count written.
    """
    all_rows = reservoir.to_rows()
    total = len(all_rows)

    for start in range(0, total, batch_size):
        chunk = all_rows[start : start + batch_size]
        # _ReservoirRow = (csd, key, value, metal, coord_geom, split, level)
        csd_col = [r[0] for r in chunk]
        key_col = [r[1] for r in chunk]
        val_col = [r[2] for r in chunk]
        metal_col = [r[3] for r in chunk]
        geom_col = [r[4] for r in chunk]
        split_col = [r[5] for r in chunk]
        lvl_col = [r[6] for r in chunk]
        l0_boundary = reservoir.l0_count()
        l0_col = [(start + idx) < l0_boundary for idx in range(len(chunk))]

        writer.write_batch(
            pa.record_batch(
                {
                    "csd_code": pa.array(csd_col, type=pa.string()),
                    "key": pa.array(key_col, type=pa.string()),
                    "value": pa.array(val_col, type=pa.float64()),
                    "metal": pa.array(metal_col, type=pa.string()),
                    "coord_geom": pa.array(geom_col, type=pa.string()),
                    "split": pa.array(split_col, type=pa.string()),
                    "level": pa.array(lvl_col, type=pa.int16()),
                    "is_l0": pa.array(l0_col, type=pa.bool_()),
                    **({"obs_id": pa.array([r[7] for r in chunk], type=pa.string())}
                       if STABLE_ANCESTOR_SAMPLING else {}),
                },
                schema=_OUT_SCHEMA,
            )
        )

    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_quantity(
    in_dir: str | Path,
    out_dir: str | Path,
    quantities: Optional[List[str]] = None,
    compression: str = "zstd",
    batch_size: int = 50_000,
    n_ancestor_cap: int = N_ANCESTOR_CAP,
) -> Dict[str, Dict[str, int]]:
    """
    Expand aggregated observation parquets (one file per quantity) into
    ancestor-expanded parquets.  This is the primary entry point when the
    aggregator has already been run.


    Parameters
    ----------
    in_dir          : directory containing observations_<Q>.parquet
    out_dir         : directory to write expanded_<Q>.parquet
    quantities      : which quantities to expand (default: all Q1-Q6)
    compression     : parquet compression codec
    batch_size      : rows per I/O batch
    n_ancestor_cap  : max observations per (ancestor_key, l0_subclass) bucket

    Returns
    -------
    dict mapping quantity -> {"in_rows": N, "out_rows": M,
                              "l0_rows": L, "ancestor_rows": A}

    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if quantities is None:
        quantities = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]

    stats: Dict[str, Dict[str, int]] = {}

    for q in quantities:
        in_path = in_dir / f"observations_{q}.parquet"
        out_path = out_dir / f"expanded_{q}.parquet"

        if not in_path.exists():
            log.warning("Input not found, skipping: %s", in_path)
            continue

        t0 = time.perf_counter()
        in_rows = 0
        reservoir = _AncestorReservoir(cap=n_ancestor_cap)
        pf = pq.ParquetFile(str(in_path))

        for batch in pf.iter_batches(batch_size=batch_size):
            in_rows += len(batch)
            _expand_into_reservoir(q, batch, reservoir)

        writer = pq.ParquetWriter(str(out_path), _OUT_SCHEMA, compression=compression)
        try:
            out_rows = _flush_reservoir(reservoir, writer, batch_size=batch_size)
        finally:
            writer.close()

        elapsed = time.perf_counter() - t0
        l0_n = reservoir.l0_count()
        anc_n = reservoir.ancestor_count()
        expansion = out_rows / in_rows if in_rows else 0.0

        log.info(
            "%s  in=%d  l0=%d  ancestor=%d  out=%d  expansion=%.1fx  %.1fs",
            q,
            in_rows,
            l0_n,
            anc_n,
            out_rows,
            expansion,
            elapsed,
        )
        stats[q] = {
            "in_rows": in_rows,
            "out_rows": out_rows,
            "l0_rows": l0_n,
            "ancestor_rows": anc_n,
        }

    return stats


def expand_split(
    in_dir: str | Path,
    split: str,
    out_dir: str | Path,
    quantities: Optional[List[str]] = None,
    compression: str = "zstd",
    batch_size: int = 50_000,
    n_ancestor_cap: int = N_ANCESTOR_CAP,
) -> Dict[str, Dict[str, int]]:
    """
    Expand one split's raw parquets into ancestor-expanded parquets.

    Parameters
    ----------
    in_dir          : directory containing observations_<Q>_<split>.parquet
    split           : split name (e.g. "train", "test", "val")
    out_dir         : directory to write expanded_<Q>_<split>.parquet
    quantities      : which quantities to expand (default: all Q1-Q6)
    compression     : parquet compression codec
    batch_size      : rows per parquet batch when reading input / writing output
    n_ancestor_cap  : max observations per (ancestor_key, l0_subclass) bucket

    Returns
    -------
    dict mapping quantity -> {"in_rows": N, "out_rows": M,
                              "l0_rows": L, "ancestor_rows": A}
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if quantities is None:
        quantities = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]

    stats: Dict[str, Dict[str, int]] = {}

    for q in quantities:
        in_path = in_dir / f"observations_{q}_{split}.parquet"
        out_path = out_dir / f"expanded_{q}_{split}.parquet"

        if not in_path.exists():
            log.warning("Input not found, skipping: %s", in_path)
            continue

        t0 = time.perf_counter()
        in_rows = 0
        reservoir = _AncestorReservoir(cap=n_ancestor_cap)
        pf = pq.ParquetFile(in_path)

        # Pass 1: stream input, fill reservoir
        for batch in pf.iter_batches(batch_size=batch_size):
            in_rows += len(batch)
            _expand_into_reservoir(q, batch, reservoir)

        # Pass 2: flush reservoir to parquet
        writer = pq.ParquetWriter(str(out_path), _OUT_SCHEMA, compression=compression)
        try:
            out_rows = _flush_reservoir(reservoir, writer, batch_size=batch_size)
        finally:
            writer.close()

        elapsed = time.perf_counter() - t0
        l0_n = reservoir.l0_count()
        anc_n = reservoir.ancestor_count()
        expansion = out_rows / in_rows if in_rows else 0.0

        log.info(
            "%s  split=%s  in=%d  l0=%d  ancestor=%d  out=%d  expansion=%.1fx  %.1fs",
            q,
            split,
            in_rows,
            l0_n,
            anc_n,
            out_rows,
            expansion,
            elapsed,
        )
        stats[q] = {
            "in_rows": in_rows,
            "out_rows": out_rows,
            "l0_rows": l0_n,
            "ancestor_rows": anc_n,
        }

    return stats
