"""Capture per-torsion Q5 score deltas for the named low-rho pools."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fit.pipeline import compare
from jacobik.stats import stats
from tools.jgen.a4t.analyse import _load_parent
from tools.jgen import inputs
from tools.jdev import multiseed
from tools.jdev.budget.pools import rank_terms

root = Path(os.environ['D2_KEY_ROOT'])
inputs.FRAME_V1 = root / 'data/refit/frame_v1'
# frame_shards' default is bound at definition time; explicitly bind this run's staged frame.
_frame_shards = inputs.frame_shards
inputs.frame_shards = lambda: _frame_shards(inputs.FRAME_V1)
out = Path(os.environ['D2_KEY_OUT'])
db, _, _ = compare.load_container(Path(os.environ['D2_KEY_DB']))
target = json.loads((root/'data/db/constraints_2026-11-01.json').read_text())
lookups = [stats.StatsLookup(stats.load_db(x)) for x in (target, db)]
frame = json.loads((root/'data/refit/frame_v1/frame.json').read_text())
meta = {}
for row in frame['main_allocation']['splits']['surface_fit']:
    code = row.get('code') or row.get('representative_code')
    if code: meta[code] = (row.get('shard') or row.get('representative_shard') or 'unknown', row.get('dataset') or 'unknown')

def components(sc, x):
    obs = sc._ge.q5_torsions(x)
    logp = sc._q5_kappas.unsqueeze(0)*torch.cos(obs.unsqueeze(-1)-sc._q5_means.unsqueeze(0)) - sc._q5_log_norm.unsqueeze(0) + sc._q5_log_w.unsqueeze(0)
    return (torch.logsumexp(logp, dim=-1) / sc._agg_norm()).detach().cpu().numpy()

result = {}
for code in ('AJIXUN', 'AFOWIB'):
    mol, builder, _ = _load_parent(code)
    scorers=[]
    for lookup in lookups:
        ctx, asg, w, _ = multiseed.prepare(builder, 11, stats=lookup, device='cpu', angle_bends=True)
        (_, _, _, sc), _ = multiseed.seed_rows(builder, ctx, asg, w, [11])
        scorers.append(sc)
    keys = [list(k) for k in scorers[0]._ge._q5_keys]
    assert keys == [list(k) for k in scorers[1]._ge._q5_keys]
    result[code] = {}
    for seed in (11,12,13):
        with np.load(root/f'data/refit/frame_v1/pools/{code}_{seed}_control.npz', allow_pickle=True) as pool:
            x = torch.tensor(pool['X'], dtype=torch.float32); groups = torch.as_tensor(pool['asg'], dtype=torch.long)
        scores=[]; terms=[]; parts=[]
        for sc in scorers:
            total, bd = rank_terms(sc, x, groups)
            scores.append(total.detach().cpu().numpy());terms.append(bd['Q5'].detach().cpu().numpy())
            parts.append(components(sc,x))
        delta = parts[1]-parts[0]
        if not np.allclose(delta.sum(axis=1), terms[1]-terms[0], atol=1e-4, rtol=0):
            raise RuntimeError(f'{code}/{seed}: per-key sum does not match rank_terms')
        old,new = int(np.argmax(scores[0])),int(np.argmax(scores[1])); bykey={}
        for i,key in enumerate(keys):
            k=json.dumps(key);bykey.setdefault(k,np.zeros(len(x)));bykey[k]+=delta[:,i]
        result[code][str(seed)] = [{'key':json.loads(k),'old_top_delta':float(v[old]),'new_top_delta':float(v[new]),
                                   'max_abs_delta':float(np.max(np.abs(v)))} for k,v in sorted(bykey.items())]
out.write_text(json.dumps(result,indent=2))
print(f'wrote {out}')
