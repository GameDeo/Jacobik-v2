"""
extractor.py
============
Step 4 — Geometry extractor.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
            j = random.randint(1, n)
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
            _SCHEMA,
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
                },
                schema=_SCHEMA,
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


def _empty_buffers() -> Dict[str, Dict[str, List]]:
    return {
        q: {"csd_code": [], "key": [], "value": [], "metal": [], "coord_geom": []}
        for q in ("Q1", "Q2", "Q3", "Q4", "Q5")
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_split(
    data_dir: str | Path,
    split: str,
    out_dir: str | Path,
    limit: Optional[int] = None,
    flush_every: int = FLUSH_EVERY,
    compression: str = "zstd",
    progress: bool = True,
    progress_file: Optional[Path] = None,
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

    Returns
    -------
    dict: split, n_complexes, n_errors, elapsed_s, ms_per_mol, Q1_rows…Q6_rows

    """
    from jacobik.stats.parser import TMQMParser
    from jacobik.topology import key_builder

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    parser = TMQMParser(data_dir)
    writers = _open_writers(out_dir, split, compression=compression)
    buffers = _empty_buffers()
    reservoir = _Q6Reservoir()

    row_counts: Dict[str, int] = {q: 0 for q in QUANTITIES}
    n_complexes = 0
    n_errors = 0
    t_start = time.perf_counter()
    t_batch = t_start

    if progress_file is not None:
        progress_file = Path(progress_file)
        progress_file.write_text("")

    stream = parser.stream_split(split, limit=limit)
    if progress:
        try:
            from tqdm import tqdm

            stream = tqdm(stream, desc=f"split={split}", unit="mol", dynamic_ncols=True)
        except ImportError:
            pass

    try:
        for mol in stream:
            try:
                all_kv = key_builder.all_keys(mol)
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
                for key_tuple, value in all_kv.get(q, ()):
                    buf["csd_code"].append(code)
                    buf["key"].append(repr(key_tuple))
                    buf["value"].append(float(value))
                    buf["metal"].append(metal)
                    buf["coord_geom"].append(geom)

            # Q6: into the reservoir sampler
            for key_tuple, value in all_kv.get("Q6", ()):
                reservoir.add(repr(key_tuple), float(value), code, metal, geom)

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
        "elapsed_s": elapsed,
        "ms_per_mol": ms_per,
    }
    result.update({f"{q}_rows": row_counts[q] for q in QUANTITIES})
    return result


# ---------------------------------------------------------------------------
# Module-level worker — picklable under fork AND spawn
# ---------------------------------------------------------------------------


@dataclass
class WorkerArgs:
    """
    Plain dataclass carrying all arguments for one worker process.
    Dataclasses pickle cleanly under both fork and spawn.
    """

    data_dir: str
    split: str
    out_dir: str
    limit: Optional[int]
    flush_every: int
    compression: str
    extra_paths: List[str]
    progress_file: str


def _worker_fn(args: WorkerArgs) -> Dict:
    for p in args.extra_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] worker={args.split} %(name)s — %(message)s",
        stream=sys.stderr,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    error_file = out_dir / f"{args.split}.error"

    try:
        return extract_split(
            data_dir=args.data_dir,
            split=args.split,
            out_dir=args.out_dir,
            limit=args.limit,
            flush_every=args.flush_every,
            compression=args.compression,
            progress=False,
            progress_file=Path(args.progress_file),
        )
    except Exception:
        tb = traceback.format_exc()
        error_file.write_text(tb)
        logging.getLogger(__name__).error("Worker for split=%s crashed:\n%s", args.split, tb)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract Q1–Q6 geometry observations from one tmQM split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--flush-every", type=int, default=FLUSH_EVERY)
    p.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "none"])
    p.add_argument("--no-progress", action="store_true")
    from jacobik.stats.cli_utils import add_logging_args

    add_logging_args(p)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    from jacobik.stats.cli_utils import setup_logging

    setup_logging(args)
    stats = extract_split(
        data_dir=args.data_dir,
        split=args.split,
        out_dir=args.out_dir,
        limit=args.limit,
        flush_every=args.flush_every,
        compression=args.compression,
        progress=not args.no_progress,
    )
    print(
        f"split={stats['split']}  complexes={stats['n_complexes']}  "
        f"errors={stats['n_errors']}  elapsed={stats['elapsed_s']:.1f}s  "
        f"{stats['ms_per_mol']:.1f}ms/mol"
    )
    for q in QUANTITIES:
        print(f"  {q}: {stats[f'{q}_rows']:,} rows")
    return 0 if stats["n_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
