"""Source-observation census with review triggers and no automatic exclusions."""
from collections import Counter
import ast
from pathlib import Path


def retained_q1(directory):
    import pandas as pd
    from jacobik.stats.mixture import structure_id

    parents, rows, long_carboxylate = set(), 0, []
    types = Counter()
    missing_ids = 0
    paths = sorted(Path(directory).glob("observations_Q1_*.parquet"))
    if not paths:
        raise ValueError("no retained Q1 observations")
    for path in paths:
        frame = pd.read_parquet(path)
        rows += len(frame)
        parents.update(frame.csd_code.astype(str))
        missing_ids += len(frame) if "obs_id" not in frame else int(frame.obs_id.isna().sum())
        keys = {key:ast.literal_eval(key) for key in frame.key.unique()}
        for key, size in frame.groupby("key",sort=False).size().items():
            typed = keys[key]
            if len(typed) > 3:
                types[str(typed[3])] += int(size)
        mask = frame.key.map(lambda key:len(keys[key]) > 3 and keys[key][3] == "O_carboxylate")
        for row in frame.loc[mask & (frame.value > 2.4)].itertuples():
            long_carboxylate.append(dict(code=str(row.csd_code), key=row.key, distance_A=float(row.value)))
    return dict(files=[str(p) for p in paths], rows=rows, source_parent_codes=len(parents),
                independent_parent_count=len({structure_id(code) for code in parents}),
                missing_observation_ids=missing_ids, donor_subclass_counts=dict(types),
                carboxylate_above_2_4A=long_carboxylate,
                trigger_note="Distance is a review trigger. Secondary coordination remains chemically possible.")
