"""
expander.py
===========
Step 4b — Ancestor key expander.
"""

from __future__ import annotations

import argparse
import ast
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from jacobik.stats.key_builder import (
    METAL_ANY,
    TRANS_NONE,
    metal_period_group,
)
from jacobik.topology.donor_hierarchy import (
    ELEMENT_ANY,
    full_path,
    is_known,
)

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

_IN_SCHEMA = pa.schema(
    [
        pa.field("csd_code", pa.string()),
        pa.field("key", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("metal", pa.string()),
        pa.field("coord_geom", pa.string()),
        pa.field("split", pa.string()),  # present in aggregated files
    ]
)

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

# ---------------------------------------------------------------------------
# Geometry coarsening helpers
# ---------------------------------------------------------------------------

_GEOM_TO_CN: Dict[str, str] = {
    "linear": "CN2",
    "t_shaped": "CN3",
    "trigonal": "CN3",
    "square_plan": "CN4",
    "see_saw": "CN4",
    "tetrahedral": "CN4",
    "square_pyr": "CN5",
    "trig_bipyr": "CN5",
    "octahedral": "CN6",
    "trig_pris": "CN6",
    "pent_bipyr": "CN7",
    "square_anti": "CN8",
    # "other" intentionally absent -- jumps to GEOM_ANY directly
}

GEOM_ANY = "geom_any"


def _geom_coarsen_chain(geom: str) -> List[str]:
    """
    Return the coarsening chain for a geometry label, NOT including the
    label itself: [CN_only, geom_any] or [geom_any] for "other".
    """
    cn = _GEOM_TO_CN.get(geom)
    if cn is not None:
        return [cn, GEOM_ANY]
    return [GEOM_ANY]


# ---------------------------------------------------------------------------
# Q5 atom-type coarsening
# ---------------------------------------------------------------------------

_H = "H"
_HYBRIDISED_PREFIXES = ("C_", "N_", "O_", "S_", "P_")


def _q5_type_coarsen_chain(token: str) -> List[str]:
    """
    Coarsening chain for a Q5 local bond atom type, NOT including the token.
    """
    if token == "any":
        return []
    if token == "heavy":
        return ["any"]
    for prefix in _HYBRIDISED_PREFIXES:
        if token.startswith(prefix):
            elem = prefix[:-1]
            return [elem, "heavy", "any"]
    if token == _H:
        return ["any"]
    return ["heavy", "any"]


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------


def _parse_key(key_str: str) -> tuple:
    """Parse a repr() key string back to a Python tuple (strings + ints)."""
    return ast.literal_eval(key_str)


# ---------------------------------------------------------------------------
# L0 subclass extraction (for reservoir bucket key)
# ---------------------------------------------------------------------------


def _l0_subclass(quantity: str, key_tuple: tuple) -> str:
    """
    Extract the most specific subclass token from an L0 key tuple.
    This is the dimension that varies between children of an ancestor node
    and is used as the second component of the reservoir bucket key.
    """
    try:
        if quantity == "Q1":
            return str(key_tuple[3])
        if quantity == "Q2":
            return str(key_tuple[3])
        if quantity in ("Q3", "Q4"):
            return str(key_tuple[0])
        if quantity == "Q5":
            return str(key_tuple[0])
    except (IndexError, TypeError):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Per-quantity ancestor generators
# ---------------------------------------------------------------------------


def _q1_ancestors(key: tuple) -> List[Tuple[int, tuple]]:
    """
    Return (level, ancestor_key) list for Q1.
    Input: (metal, spin, charge, donor_subclass, coord_geom, trans_partner)
    L0 row is NOT included; caller handles it.
    """
    metal, spin, charge, donor_sub, geom, trans = key

    donor_full = full_path(donor_sub) if is_known(donor_sub) else [donor_sub, ELEMENT_ANY]
    donor_bucket = donor_full[1] if len(donor_full) > 1 else ELEMENT_ANY
    donor_family = donor_full[-2] if len(donor_full) >= 2 else ELEMENT_ANY
    period_group = metal_period_group(metal)

    rows: List[Tuple[int, tuple]] = []

    def _add(level: int, k: tuple) -> None:
        rows.append((level, k))

    if trans != TRANS_NONE:
        _add(1, (metal, spin, charge, donor_sub, geom))
    # else: input is already the L1-floor; caller emits it, we start at L2

    _add(2, (metal, spin, charge, donor_sub))
    _add(3, (metal, spin, charge, donor_bucket))
    if donor_bucket != donor_family:
        _add(4, (metal, spin, charge, donor_family))
    _add(5, (metal, charge, donor_family))
    _add(6, (metal, donor_family))
    _add(7, (period_group, donor_family))
    if period_group != METAL_ANY:
        _add(8, (METAL_ANY, donor_family))
    _add(9, (donor_family,))
    _add(10, (ELEMENT_ANY,))

    return rows


def _q2_ancestors(key: tuple) -> List[Tuple[int, tuple]]:
    """
    Return (level, ancestor_key) list for Q2.
    Input: (coord_geom, graph_dist, metal, donor_A_subclass, donor_B_subclass)
    """
    geom, dist, metal, sub_a, sub_b = key

    def _fp(s: str) -> List[str]:
        return full_path(s) if is_known(s) else [s, ELEMENT_ANY]

    path_a = _fp(sub_a)
    path_b = _fp(sub_b)
    bkt_a = path_a[1] if len(path_a) > 1 else ELEMENT_ANY
    bkt_b = path_b[1] if len(path_b) > 1 else ELEMENT_ANY
    fam_a = path_a[-2] if len(path_a) >= 2 else ELEMENT_ANY
    fam_b = path_b[-2] if len(path_b) >= 2 else ELEMENT_ANY

    geom_chain = _geom_coarsen_chain(geom)
    cn_or_any = geom_chain[0]

    rows: List[Tuple[int, tuple]] = []

    def _add(level: int, k: tuple) -> None:
        rows.append((level, k))

    _add(1, (geom, dist, metal, bkt_a, bkt_b))
    _add(2, (geom, dist, metal, fam_a, fam_b))
    _add(3, (geom, dist, fam_a, fam_b))
    _add(4, (geom, dist))
    _add(5, (cn_or_any, dist))
    _add(6, (geom,))
    _add(7, (cn_or_any,))
    _add(8, (GEOM_ANY,))

    # Deduplicate (handles "other" geom where cn_or_any == GEOM_ANY,
    # and shallow donors where bkt == fam)
    seen: set = set()
    deduped = []
    for level, k in rows:
        if k not in seen:
            seen.add(k)
            deduped.append((level, k))
    return deduped


def _q3_ancestors(key: tuple) -> List[Tuple[int, tuple]]:
    """
    Return (level, ancestor_key) list for Q3.
    Input: (donor_subclass, metal, coord_geom)
    """
    donor_sub, metal, geom = key

    donor_full = full_path(donor_sub) if is_known(donor_sub) else [donor_sub, ELEMENT_ANY]
    donor_bucket = donor_full[1] if len(donor_full) > 1 else ELEMENT_ANY
    donor_family = donor_full[-2] if len(donor_full) >= 2 else ELEMENT_ANY
    period_group = metal_period_group(metal)

    rows: List[Tuple[int, tuple]] = []

    def _add(level: int, k: tuple) -> None:
        rows.append((level, k))

    _add(1, (donor_sub, metal))
    _add(2, (donor_sub, period_group))
    _add(3, (donor_sub, METAL_ANY))
    _add(4, (donor_sub,))
    _add(5, (donor_bucket,))
    _add(6, (donor_family,))
    _add(7, (ELEMENT_ANY,))

    seen: set = set()
    deduped = []
    for level, k in rows:
        if k not in seen:
            seen.add(k)
            deduped.append((level, k))
    return deduped


def _q4_ancestors(key: tuple) -> List[Tuple[int, tuple]]:
    """
    Return (level, ancestor_key) list for Q4.
    Input: (donor_subclass, coord_geom)
    Monotonic: geom dropped at L1, never restored.
    """
    donor_sub, geom = key

    donor_full = full_path(donor_sub) if is_known(donor_sub) else [donor_sub, ELEMENT_ANY]
    donor_bucket = donor_full[1] if len(donor_full) > 1 else ELEMENT_ANY
    donor_family = donor_full[-2] if len(donor_full) >= 2 else ELEMENT_ANY

    rows: List[Tuple[int, tuple]] = []

    def _add(level: int, k: tuple) -> None:
        rows.append((level, k))

    _add(1, (donor_sub,))
    _add(2, (donor_bucket,))
    _add(3, (donor_family,))
    _add(4, (ELEMENT_ANY,))

    seen: set = set()
    deduped = []
    for level, k in rows:
        if k not in seen:
            seen.add(k)
            deduped.append((level, k))
    return deduped


def _q5_ancestors(key: tuple) -> List[Tuple[int, tuple]]:
    """
    Return (level, ancestor_key) list for Q5.
    Input: (type_j, type_k, type_i, type_l, ring_ctx)
    """
    type_j, type_k, type_i, type_l, ring_ctx = key

    elem_j_chain = _q5_type_coarsen_chain(type_j)
    elem_k_chain = _q5_type_coarsen_chain(type_k)

    def _elem(token: str, chain: List[str]) -> str:
        """First step in chain, but only if it's an element symbol."""
        if not chain:
            return token
        c0 = chain[0]
        return token if c0 in ("heavy", "any") else c0

    rows: List[Tuple[int, tuple]] = []

    def _add(level: int, k: tuple) -> None:
        rows.append((level, k))

    _add(1, (type_j, type_k, type_i, type_l))
    _add(2, (type_j, type_k, type_i))
    _add(3, (type_j, type_k))
    _add(4, (_elem(type_j, elem_j_chain), _elem(type_k, elem_k_chain)))
    _add(5, (ELEMENT_ANY,))

    seen: set = set()
    deduped = []
    for level, k in rows:
        if k not in seen:
            seen.add(k)
            deduped.append((level, k))
    return deduped


# ---------------------------------------------------------------------------
# Ancestor dispatch
# ---------------------------------------------------------------------------

_ANCESTOR_FN = {
    "Q1": _q1_ancestors,
    "Q2": _q2_ancestors,
    "Q3": _q3_ancestors,
    "Q4": _q4_ancestors,
    "Q5": _q5_ancestors,
}

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

    def add_l0(self, row: _ReservoirRow) -> None:
        self._l0_rows.append(row)

    def add_ancestor(
        self,
        ancestor_key_str: str,
        l0_subclass_str: str,
        row: _ReservoirRow,
    ) -> None:
        bk = (ancestor_key_str, l0_subclass_str)
        n = self._seen.get(bk, 0) + 1
        self._seen[bk] = n

        if n <= self.cap:
            if bk not in self._reservoir:
                self._reservoir[bk] = []
            self._reservoir[bk].append(row)
        else:
            j = random.randint(1, n)
            if j <= self.cap:
                self._reservoir[bk][j - 1] = row

    def to_rows(self) -> List[_ReservoirRow]:
        out = list(self._l0_rows)
        for rows in self._reservoir.values():
            out.extend(rows)
        return out

    def l0_count(self) -> int:
        return len(self._l0_rows)

    def ancestor_count(self) -> int:
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
    """
    ancestor_fn = _ANCESTOR_FN.get(quantity)

    csd_codes = batch.column("csd_code").to_pylist()
    key_strs = batch.column("key").to_pylist()
    values = batch.column("value").to_pylist()
    metals = batch.column("metal").to_pylist()
    geoms = batch.column("coord_geom").to_pylist()
    # split column may be absent in legacy per-split files
    splits_col = (
        batch.column("split").to_pylist()
        if batch.schema.get_field_index("split") >= 0
        else [""] * batch.num_rows
    )

    for csd, key_str, val, metal, geom, split in zip(
        csd_codes, key_strs, values, metals, geoms, splits_col
    ):
        # Q6: pass through as L0, no ancestors
        if ancestor_fn is None:
            reservoir.add_l0((csd, key_str, val, metal, geom, split, 0))
            continue

        key_tuple = _parse_key(key_str)

        # Q1: floor level depends on whether trans_partner is meaningful
        if quantity == "Q1":
            trans = key_tuple[5] if len(key_tuple) > 5 else TRANS_NONE
            l0_level = 0 if trans != TRANS_NONE else 1
        else:
            l0_level = 0

        # L0 row always kept in full
        reservoir.add_l0((csd, key_str, val, metal, geom, split, l0_level))

        # Compute ancestor rows
        try:
            anc_rows = ancestor_fn(key_tuple)
        except Exception as exc:
            log.debug("Ancestor expansion failed for %s key %r: %s", quantity, key_str, exc)
            continue

        l0_sub = _l0_subclass(quantity, key_tuple)

        for level, anc_key in anc_rows:
            if level <= l0_level:
                continue
            anc_key_str = repr(anc_key)
            row = (csd, anc_key_str, val, metal, geom, split, level)
            reservoir.add_ancestor(anc_key_str, l0_sub, row)


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Expand observation parquets with ancestor keys.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-dir", required=True, help="Directory of input parquets")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument(
        "--mode",
        default="aggregated",
        choices=["aggregated", "split"],
        help="'aggregated': read observations_<Q>.parquet (default); "
        "'split': read observations_<Q>_<split>.parquet",
    )
    p.add_argument("--split", default=None, help="Split name (required when --mode=split)")
    p.add_argument(
        "--quantities", nargs="+", default=None, help="Quantities to expand (default: all Q1-Q6)"
    )
    p.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "none"])
    p.add_argument("--batch-size", type=int, default=50_000)
    p.add_argument(
        "--n-ancestor-cap",
        type=int,
        default=N_ANCESTOR_CAP,
        help="Max obs per (ancestor_key, subclass) bucket",
    )
    from jacobik.stats.cli_utils import add_logging_args

    add_logging_args(p)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    from jacobik.stats.cli_utils import setup_logging

    setup_logging(args)

    common = dict(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        quantities=args.quantities,
        compression=args.compression,
        batch_size=args.batch_size,
        n_ancestor_cap=args.n_ancestor_cap,
    )

    if args.mode == "aggregated":
        stats = expand_quantity(**common)
    else:
        if args.split is None:
            print("ERROR: --split is required when --mode=split", file=sys.stderr)
            return 1
        stats = expand_split(split=args.split, **common)

    for q, s in stats.items():
        expansion = s["out_rows"] / s["in_rows"] if s["in_rows"] else 0.0
        print(
            f"  {q}: {s['in_rows']:>10,} in  "
            f"l0={s['l0_rows']:>10,}  anc={s['ancestor_rows']:>10,}  "
            f"out={s['out_rows']:>10,}  ({expansion:.1f}x)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
