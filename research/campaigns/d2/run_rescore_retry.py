"""Reproduce the D2 rescore with a logged xTB retry; never overwrite prior results."""
import json
import time
from pathlib import Path
from fit.pipeline import compare
from tools.jdev.rescore import rescore_60_pools
root = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
db, _, _ = compare.load_container(root / 'research/data/d2/db_d2/constraints_2026-11-09.npz')
target = json.loads((root / 'data/db/constraints_2026-11-01.json').read_text())
t = time.time()
r = rescore_60_pools(db, target, pools_dir=root / 'data/refit/frame_v1/pools',
    frame_path=root / 'data/refit/frame_v1/frame.json',
    manifest_path=root / 'data/refit/frame_v1/manifest.json', ref_dir=root / 'data/ref')
r['wall_s'] = time.time() - t
(root / 'research/data/d2/db_d2/rescore_60_per_pool.json').write_text(json.dumps(r, indent=2, sort_keys=True))
print(json.dumps({k: v for k, v in r.items() if k != 'per_pool'}))
