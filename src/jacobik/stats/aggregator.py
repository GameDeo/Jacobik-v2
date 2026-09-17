"""
aggregator.py
=============
Post-extraction step: merges per-split observation files into one
combined file per quantity.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTITIES = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")

_COMBINED_SCHEMA = pa.schema(
    [
        pa.field("csd_code", pa.string()),
        pa.field("key", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("metal", pa.string()),
        pa.field("coord_geom", pa.string()),
        pa.field("split", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def aggregate(
    out_dir: str | Path,
    splits: Sequence[str],
    quantities: Sequence[str] = QUANTITIES,
    compression: str = "zstd",
    overwrite: bool = True,
) -> Dict[str, Dict]:
    """
    Merge per-split parquet files into one combined file per quantity.

    Parameters
    ----------
    out_dir     : directory that contains the per-split files and receives
                  the combined outputs
    splits      : split names to merge, e.g. ["X1", "X2", "test"]
    quantities  : subset of quantities to aggregate (default: all six)
    compression : parquet compression for the combined files
    overwrite   : if False, skip quantities whose combined file already exists

    Returns
    -------
    dict[quantity -> {rows, missing_splits, elapsed_s, size_mb}]
    """
    out_dir = Path(out_dir)
    results: Dict[str, Dict] = {}

    for q in quantities:
        combined_path = out_dir / f"observations_{q}.parquet"

        if combined_path.exists() and not overwrite:
            log.info("%s: combined file exists, skipping (overwrite=False)", q)
            results[q] = {"skipped": True}
            continue

        t0 = time.perf_counter()
        total_rows = 0
        missing: List[str] = []

        # Collect source files that actually exist
        sources: List[tuple[str, Path]] = []
        for split in splits:
            src = out_dir / f"observations_{q}_{split}.parquet"
            if src.exists():
                sources.append((split, src))
            else:
                log.warning("%s: missing source file for split=%s (%s)", q, split, src)
                missing.append(split)

        if not sources:
            log.error("%s: no source files found — skipping", q)
            results[q] = {"rows": 0, "missing_splits": missing, "elapsed_s": 0.0}
            continue

        writer = pq.ParquetWriter(str(combined_path), _COMBINED_SCHEMA, compression=compression)
        try:
            for split, src_path in sources:
                pf = pq.ParquetFile(str(src_path))
                for batch in pf.iter_batches():
                    n = batch.num_rows
                    if n == 0:
                        continue
                    # Add split column
                    split_col = pa.array([split] * n, type=pa.string())
                    tbl = pa.table(
                        {
                            "csd_code": batch.column("csd_code"),
                            "key": batch.column("key"),
                            "value": batch.column("value"),
                            "metal": batch.column("metal"),
                            "coord_geom": batch.column("coord_geom"),
                            "split": split_col,
                        },
                        schema=_COMBINED_SCHEMA,
                    )
                    writer.write_table(tbl)
                    total_rows += n
                log.info("%s/%s: merged", q, split)
        finally:
            writer.close()

        elapsed = time.perf_counter() - t0
        size_mb = combined_path.stat().st_size / 1e6
        log.info("%s: %d rows total  %.1fs  %.1f MB on disk", q, total_rows, elapsed, size_mb)
        results[q] = {
            "rows": total_rows,
            "missing_splits": missing,
            "elapsed_s": elapsed,
            "size_mb": size_mb,
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate per-split observation parquets into one file per quantity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", required=True, help="Directory with per-split parquets")
    p.add_argument("--splits", nargs="+", default=["X1", "X2", "test"])
    p.add_argument("--quantities", nargs="+", default=list(QUANTITIES))
    p.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "none"])
    p.add_argument("--no-overwrite", action="store_true")
    from jacobik.stats.cli_utils import add_logging_args

    add_logging_args(p)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    from jacobik.stats.cli_utils import setup_logging

    setup_logging(args)
    results = aggregate(
        out_dir=args.out_dir,
        splits=args.splits,
        quantities=args.quantities,
        compression=args.compression,
        overwrite=not args.no_overwrite,
    )
    print(f"\n{'Quantity':<8}  {'Rows':>10}  {'Size MB':>8}  {'Time s':>7}  Missing splits")
    print("-" * 60)
    for q, r in results.items():
        if r.get("skipped"):
            print(f"  {q:<6}  {'(skipped)':>10}")
        else:
            missing_str = ", ".join(r["missing_splits"]) or "—"
            print(
                f"  {q:<6}  {r['rows']:>10,}  {r.get('size_mb', 0):>8.1f}  "
                f"{r['elapsed_s']:>7.1f}  {missing_str}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
