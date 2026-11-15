"""reader.py
=========
Strict reference data reader for tmQM and lnQM datasets.

Enforces strict 3-column schema constraints:
- CSD_code
- SMILES
- Geometry (.xyz coordinates and .BO bond orders)

QM property columns are strictly never read, requested, or attached.
Yields parsed TMComplex structures for constraint extraction and fitting.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Union

from jacobik.core.types import TMComplex
from jacobik.fit.parser import TMQMParser, _CSVRow

log = logging.getLogger(__name__)


def load_strict_csv_index(csv_path: Union[str, Path]) -> Dict[str, _CSVRow]:
    """Load reference CSV index reading ONLY CSD_code and SMILES columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {csv_path}")

    index: Dict[str, _CSVRow] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            code = row.get("CSD_code", "").strip()
            if not code:
                continue
            smiles_raw = row.get("SMILES", "").strip()
            index[code] = _CSVRow(
                csd_code=code,
                smiles=smiles_raw if smiles_raw else None,
            )
    log.info("Loaded %d reference rows from %s (CSD_code and SMILES only)", len(index), csv_path)
    return index


class ReferenceReader:
    """Streams parsed TMComplex molecules from tmQM/lnQM directory structures.

    Parameters
    ----------
    data_dir : Union[str, Path, Sequence[Union[str, Path]]]
        Directory or list of directories containing xyz/, bo/, and tmQM_y.csv.
    """

    def __init__(self, data_dir: Union[str, Path, Sequence[Union[str, Path]]]) -> None:
        if isinstance(data_dir, (str, Path)):
            self.data_dirs = [Path(data_dir)]
        else:
            self.data_dirs = [Path(d) for d in data_dir]

        self._parsers: Dict[Path, TMQMParser] = {}
        for d in self.data_dirs:
            if not d.exists():
                raise FileNotFoundError(f"Reference directory does not exist: {d}")
            p = TMQMParser(d)
            # Ensure parser's CSV index uses our strict reader
            csv_path = d / "tmQM_y.csv"
            if csv_path.exists():
                p._csv_index = load_strict_csv_index(csv_path)
            self._parsers[d] = p

    def get_splits(self, data_dir: Optional[Path] = None) -> List[str]:
        """Discover available split/shard identifiers in data directory."""
        dirs = [data_dir] if data_dir is not None else self.data_dirs
        splits: List[str] = []
        for d in dirs:
            xyz_dir = d / "xyz"
            if not xyz_dir.exists():
                continue
            for f in sorted(xyz_dir.glob("tmQM_*.xyz")):
                # Extract split identifier, e.g. "X1s000" from "tmQM_X1s000.xyz"
                name = f.stem
                if name.startswith("tmQM_"):
                    splits.append(name[5:])
        return splits

    get_shards = get_splits

    def stream_split(
        self,
        split: str,
        limit: Optional[int] = None,
        data_dir: Optional[Path] = None,
    ) -> Generator[TMComplex, None, None]:
        """Stream complexes for a specific split across matching directories."""
        dirs = [data_dir] if data_dir is not None else self.data_dirs
        count = 0
        for d in dirs:
            parser = self._parsers[d]
            xyz_path = d / "xyz" / f"tmQM_{split}.xyz"
            if not xyz_path.exists():
                continue
            for mol in parser.stream_split(split):
                yield mol
                count += 1
                if limit is not None and count >= limit:
                    return

    def stream_all(
        self,
        limit: Optional[int] = None,
    ) -> Generator[TMComplex, None, None]:
        """Stream complexes across all discovered splits and directories."""
        count = 0
        for d in self.data_dirs:
            parser = self._parsers[d]
            splits = self.get_splits(d)
            for split in splits:
                for mol in parser.stream_split(split):
                    yield mol
                    count += 1
                    if limit is not None and count >= limit:
                        return


TMQMReader = ReferenceReader


def iter_reference_complexes(
    data_dirs: Union[str, Path, Sequence[Union[str, Path]]],
    limit: Optional[int] = None,
    split: Optional[str] = None,
) -> Generator[TMComplex, None, None]:
    """Generator interface yielding parsed TMComplex objects for fitting.

    Parameters
    ----------
    data_dirs : Union[str, Path, Sequence[Union[str, Path]]]
        Directory or list of directories containing reference data (e.g. data/ref/tmQM, data/ref/lnQM).
    limit : Optional[int]
        Maximum number of complexes to yield.
    split : Optional[str]
        If specified, yield only complexes matching this split identifier.
    """
    reader = ReferenceReader(data_dirs)
    if split is not None:
        yield from reader.stream_split(split, limit=limit)
    else:
        yield from reader.stream_all(limit=limit)
