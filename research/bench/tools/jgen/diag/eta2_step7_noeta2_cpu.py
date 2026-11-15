"""eta2 fold-in, CPU check (contract amendment 2): parents without an eta2 face score bit-identically under
eta2_face_plane + DB v5. Rescores every exported ctl pool row of holdout stratum C (ranking score, per term) on CPU.

  python eta2_step7_noeta2_cpu.py OUT.json        run once per configuration in its own process:
      production:  (no overrides, JACOBIK_DB unset)
      candidate:   JACOBIK_PROFILE_OVERRIDES='{"eta2_face_plane": true}' JACOBIK_DB=<DB v5>
  python eta2_step7_noeta2_cpu.py compare A.json B.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('research/data/jgen/campaign/eta2/holdout')


def read_xyz(p):
    lines = Path(p).read_text().splitlines()
    return np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2:2 + int(lines[0])]])


def run(out):
    import torch
    from jacobik.stats import stats
    from tools.jdev import multiseed
    from tools.jdev.budget.pools import rank_terms
    from tools.jgen import inputs
    from jacobik.core.env import build_builder_and_inputs

    lookup = stats.StatsLookup(stats.load_db(inputs.DB_PATH))
    shards = inputs.frame_shards()
    res = {'db': str(inputs.DB_PATH)}
    for code in json.load(open(ROOT / 'ETA2_HOLDOUT.json'))['draw']['C']:
        try:
            builder, _, err = build_builder_and_inputs(inputs.load_mol(code, *shards[code]))
            ctx, asg, W, _ = multiseed.prepare(builder, 11, stats=lookup, device='cpu', angle_bends=True)
            (_, _, _, sc), _ = multiseed.seed_rows(builder, ctx, asg, W, [11])
            per = {}
            for pm in sorted((ROOT / 'ctl' / 'xyz' / code).glob(f'{code}_ctl_s*_pool.json')):
                P = json.load(open(pm))
                X = np.stack([read_xyz(pm.parent / r['file']) for r in P['rows']])
                rg = [min(max(int(r['row_key'][1] or 0), 0), len(asg) - 1) for r in P['rows']]
                tot, terms = rank_terms(sc, torch.tensor(X, dtype=torch.float32), torch.as_tensor(rg, dtype=torch.long))
                per[pm.stem] = dict(total=tot.detach().double().tolist(),
                                    terms={k: v.detach().double().tolist() for k, v in terms.items()})
            res[code] = per
        except Exception as exc:  # a loop over complexes never fails the batch
            res[code] = {'error': f'{type(exc).__name__}: {exc}'}
    Path(out).write_text(json.dumps(res) + '\n')


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    rep, worst = {}, 0.0
    for code in A:
        if code == 'db':
            continue
        if not A[code] or 'error' in A[code] or 'error' in B.get(code, {'error': 1}):
            rep[code] = 'error or no pool'
            continue
        d = max(float(np.max(np.abs(np.subtract(A[code][p]['total'], B[code][p]['total'])))) for p in A[code])
        rep[code] = d
        worst = max(worst, d)
    print(json.dumps(dict(db_a=A['db'], db_b=B['db'], max_abs_diff=worst, per_parent=rep), indent=1))


if __name__ == '__main__':
    compare(*sys.argv[2:4]) if sys.argv[1] == 'compare' else run(sys.argv[1])
