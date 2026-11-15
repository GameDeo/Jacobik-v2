"""Capture concrete lookup keys and resolved fits for the D2 rescore (no QM properties)."""
import dataclasses
import json
import os
from pathlib import Path
from fit.pipeline import compare
from tools.jdev.rescore import rescore_60_pools
from jacobik.stats import stats
from tools.jgen import inputs
root = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
root = Path(os.environ.get('D2_KEY_ROOT', str(root)))
db_path = Path(os.environ.get('D2_KEY_DB', str(root / 'research/data/d2/db_d2/constraints_2026-11-09.npz')))
out_path = Path(os.environ.get('D2_KEY_OUT', str(root / 'research/data/d2/db_d2/rescore_changed_keys.json')))
state = {'code': None, 'next': 0}
rows = {}
original_load = inputs.load_mol
original_init = stats.StatsLookup.__init__
original_dist = stats.StatsLookup.dist

def load(code, *args, **kwargs):
    state['code'] = code
    return original_load(code, *args, **kwargs)

def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._capture_name = ('target', 'test')[state['next'] % 2]
    state['next'] += 1

def dist(self, quantity, key):
    value = original_dist(self, quantity, key)
    code = state['code']
    if code and quantity in ('Q1', 'Q2', 'Q3', 'Q5'):
        identity = json.dumps([quantity, key], default=str, sort_keys=True)
        rows.setdefault(code, {}).setdefault(identity, {})[self._capture_name] = {
            'resolved_key': self._last_resolved_key, 'distribution': dataclasses.asdict(value)}
    return value
inputs.load_mol = load
stats.StatsLookup.__init__ = init
stats.StatsLookup.dist = dist
db, _, _ = compare.load_container(db_path)
target = json.loads((root / 'data/db/constraints_2026-11-01.json').read_text())
r = rescore_60_pools(db, target, pools_dir=root / 'data/refit/frame_v1/pools',
    frame_path=root / 'data/refit/frame_v1/frame.json',
    manifest_path=root / 'data/refit/frame_v1/manifest.json', ref_dir=root / 'data/ref')
changed = {}
for code, keys in rows.items():
    changed[code] = [{'quantity_key': json.loads(k), **v} for k, v in sorted(keys.items())
                     if 'target' in v and 'test' in v and v['target'] != v['test']]
out = out_path
out.write_text(json.dumps({'note': 'Lookup fit differences, not per-key score contributions',
                           'aggregates': {k:v for k,v in r.items() if k != 'per_pool'},
                           'changed': changed}, indent=2, default=str))
print(f'Wrote {out}: {sum(map(len, changed.values()))} changed lookups')
