"""Diagnostic replay on retained raw rows, with IDs assigned before filtering.

This explicit backfill audits the sampler only. Production stable builds
require fresh extraction-time IDs and reject legacy input files.
"""
from collections import Counter
import json
from pathlib import Path


def run(main, output):
    import pandas as pd
    import pyarrow as pa
    from fit.sanity import excluded_charge_codes, locked_codes
    from fit.expander import _AncestorReservoir, _expand_into_reservoir
    from fit.sanity import STABLE_ANCESTOR_SAMPLING, observation_id
    from jacobik.stats.ladders import q1_candidates
    from tools.jgen.ds.fit_trace import O_KEY, C_KEY

    if not STABLE_ANCESTOR_SAMPLING:
        raise ValueError("replay requires the stable_ancestor_sampling flag")
    evidence = main / "research/data/audit/2026-11-10_DS"
    watched = {repr(k) for key in (O_KEY,C_KEY) for k in q1_candidates(key)}
    locked = locked_codes(main / "research/bench/benchmarks/locked_exclusions.json")
    bad = excluded_charge_codes(json.loads((evidence / "fit_inputs/charge_os_census.json").read_text()))

    class Capture(_AncestorReservoir):
        def add_l0(self,row):
            pass

        def add_ancestor(self,key,sub,row):
            if key in watched:
                self._seen[key,sub] = self._seen.get((key,sub),0) + 1
                super().add_ancestor(key,sub,row)

    before, after = Capture(), Capture()
    for split in ("X1","X2","X3","LNf"):
        frame = pd.read_parquet(evidence / f"fit_inputs/d2_obs/observations_Q1_{split}.parquet")
        occurrence = Counter()
        ids = []
        for row in frame.itertuples():
            bucket = row.csd_code,row.key
            ids.append(observation_id("lnQM" if split=="LNf" else "tmQM",row.csd_code,"Q1",row.key,occurrence[bucket]))
            occurrence[bucket] += 1
        frame["obs_id"] = ids
        for capture, excluded in ((before,locked),(after,locked|bad)):
            table=pa.Table.from_pandas(frame.loc[~frame.csd_code.isin(excluded)],preserve_index=False)
            for batch in table.to_batches(max_chunksize=50_000):
                _expand_into_reservoir("Q1",batch,capture)
    buckets=[]
    for bucket in sorted(before._priorities.keys() | after._priorities.keys()):
        a={item[1] for item in before._priorities.get(bucket,[])}
        b={item[1] for item in after._priorities.get(bucket,[])}
        unchanged = before._seen.get(bucket,0)==after._seen.get(bucket,0)
        if unchanged and a!=b:
            raise AssertionError(f"unrelated filtering changed bucket {bucket}")
        buckets.append(dict(key=bucket,identical_eligible_rows=unchanged,sample_removed=len(a-b)))
    result=dict(sampler="hash_priority.1",identity_source="explicit diagnostic backfill before exclusions",
                buckets=buckets,unchanged_buckets=sum(b["identical_eligible_rows"] for b in buckets),
                collateral_changes=sum(b["identical_eligible_rows"] and b["sample_removed"]>0 for b in buckets))
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result
