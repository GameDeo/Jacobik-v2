"""E-2 step 0.2: angle-tail quadratic extra cost on cal crystal Q2 pairs, CPU/no solve."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from tools.jgen import inputs
from tools.jdev.budget.common import stats_from_db
from jacobik.stats.keys import q2_pair_keys
from jacobik.ik.scorer import _q2_table,_floored,_cos_mix_logp
from jacobik.ik import scorer as SCM
ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
ap=argparse.ArgumentParser();ap.add_argument('--db',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
DB=Path(args.db)
CODES=(ROOT/'research/data/jgen/campaign/A/a4/codes_cal.txt').read_text().split()
stats=stats_from_db(DB)
rows=[];failures=[]
old=SCM.ANGLE_TAIL_QUADRATIC
try:
 for code in CODES:
  try:
   mol=inputs.load_mol(code,*inputs.frame_shards()[code]); metal=np.asarray(mol.get_atom(mol.metal_idx).position,float)
   coords=[np.asarray(s.position,float) for s in mol.coord_sites]
   for i,j,key in q2_pair_keys(mol):
    mu,var,w,nu=_q2_table(stats,key)
    obs=float(np.dot(coords[i]-metal,coords[j]-metal)/np.linalg.norm(coords[i]-metal)/np.linalg.norm(coords[j]-metal))
    means=torch.tensor(mu,dtype=torch.float32).reshape(1,1,-1)
    vars_=torch.tensor(var,dtype=torch.float32).reshape(1,1,-1)
    vars_=_floored('Q2',vars_,means.reshape(1,-1),'thermal')
    logw=torch.log(torch.tensor(w,dtype=torch.float32).clamp_min(1e-30)).reshape(1,1,-1)
    nu_t=torch.tensor([[nu]],dtype=torch.float32)
    obs_t=torch.tensor([[obs]],dtype=torch.float32)
    zmin=float(np.min(np.abs(obs-np.asarray(mu))/np.sqrt(vars_.numpy().reshape(-1))))
    SCM.ANGLE_TAIL_QUADRATIC=False
    off=float(_cos_mix_logp(obs_t,means,vars_,logw,nu_t,has_t=math.isfinite(nu)).item())
    SCM.ANGLE_TAIL_QUADRATIC=True
    on=float(_cos_mix_logp(obs_t,means,vars_,logw,nu_t,has_t=math.isfinite(nu)).item())
    rows.append({'code':code,'pair_sites':[i,j],'key':list(key),'nu':nu,'angle_deg':math.degrees(math.acos(float(np.clip(obs,-1,1)))),
                 'nearest_component_abs_z':zmin,'beyond_z0':bool(math.isfinite(nu) and zmin>2.0),
                 'extra_cost_nats':max(0.0,off-on)})
  except Exception as e: failures.append({'code':code,'error':f'{type(e).__name__}: {e}'})
finally: SCM.ANGLE_TAIL_QUADRATIC=old
costs=np.array([r['extra_cost_nats'] for r in rows],float)
beyond=np.array([r['beyond_z0'] for r in rows],bool)
out={'db':str(DB),'parents_requested':len(CODES),'pairs':len(rows),'failures':failures,'z0':2.0,
     'share_pairs_beyond_z0':float(beyond.mean()) if len(beyond) else None,
     'median_extra_cost_nats':float(np.median(costs)) if len(costs) else None,
     'p95_extra_cost_nats':float(np.percentile(costs,95)) if len(costs) else None,
     'pairs_with_positive_extra_cost':int((costs>1e-8).sum()),'parents_with_pairs':len(set(r['code'] for r in rows)),
     'definition':'per crystal Q2 pair, minimum standardized cos-space residual over table components > 2; per-pair cost is logp(flag off)-logp(flag on), in nats'}
P=Path(args.out);P.parent.mkdir(parents=True,exist_ok=True);json.dump({'summary':out,'pairs':rows},open(P,'w'),indent=2)
print(json.dumps(out,indent=2))
