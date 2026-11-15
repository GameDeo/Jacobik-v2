"""Check unified extraction against independent histogram passes in temporary scratch.

No production observations, histograms or DBs are written. Use a small shard or
--limit for a bounded diagnostic. --exercise-exclusion adds one diagnostic holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO
from pools import setup


def check(data, split, limit=None, exercise_exclusion=False):
    from fit import extractor, nci, q6pmf
    from fit.parser import TMQMParser
    from fit.pipeline.extract import benchmark_codes, sibling_weights

    started = time.perf_counter()
    parser = TMQMParser(data)
    mols = list(parser.stream_split(split, limit=limit))
    if not mols or parser.n_failed:
        raise RuntimeError(f"no inputs or parse failures: {len(mols)}, {parser.n_failed}")
    exclude = set(benchmark_codes())
    diagnostic_holdout = next((m.csd_code for m in mols if m.csd_code not in exclude), None) if exercise_exclusion else None
    if diagnostic_holdout:
        exclude.add(diagnostic_holdout)
    weights = sibling_weights(Path(data) / "tmQM_y.csv")
    factories = {"q6hist": q6pmf.HistAccumulator,
                 "q6hist_m13": lambda: q6pmf.HistAccumulator(records=q6pmf.m13_records),
                 "ncihist": nci.HistAccumulator}
    independent = {}
    # Independent full passes, as the retired separate tools did, with identical inputs/rules.
    for name, factory in factories.items():
        acc = factory()
        for mol in mols:
            if mol.csd_code not in exclude:
                acc.add(mol, weights.get(mol.csd_code, 1.0))
        independent[name] = acc.h
    results = {}
    with tempfile.TemporaryDirectory(prefix="jacobik_hist_parity_") as scratch:
        extraction = extractor.extract_split(data, split, scratch, limit=limit, progress=False, orientation=False,
                                             q6_hist_exclude=exclude, q6_hist_weight=weights)
        assert extraction["n_complexes"] == len(mols) and extraction["n_errors"] == 0
        assert extraction["n_parse_failed"] == 0
        for name, ref in independent.items():
            with np.load(Path(scratch) / f"{name}_{split}.npz") as z:
                got = {k[2:]: z[k] for k in z.files if k.startswith("h:")}
            missing, extra = sorted(ref.keys() - got.keys()), sorted(got.keys() - ref.keys())
            different = [k for k in ref.keys() & got.keys() if not np.array_equal(ref[k], got[k])]
            results[name] = dict(keys=len(ref), missing=missing, extra=extra, different_bins=different,
                                 weighted_count=float(sum(h.sum() for h in ref.values())))
            assert not (missing or extra or different), results[name]
    return dict(data=str(data), split=split, limit=limit, complexes=len(mols),
                excluded=sum(m.csd_code in exclude for m in mols), diagnostic_holdout=diagnostic_holdout,
                fractional_weights=sum(m.csd_code not in exclude and weights.get(m.csd_code, 1.0) != 1.0 for m in mols),
                extraction=extraction, passes=results, elapsed_s=time.perf_counter()-started,
                status="exact parity", scope="temporary diagnostic; no production histogram regeneration or fit")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=REPO / "data/refit/clean/tmQM")
    p.add_argument("--split", default="X2s009")
    p.add_argument("--limit", type=int)
    p.add_argument("--exercise-exclusion", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    setup()
    report = check(a.data, a.split, a.limit, a.exercise_exclusion)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
