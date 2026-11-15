"""Compare identical-start eager/replay runs, including an eager-repeat noise check."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--steps',type=int,default=200)
ap.add_argument('--nondeterministic',action='store_true')
ap.add_argument('--control',action='store_true')
ap.add_argument('--out',type=Path,default=Path('research/data/jgen/campaign/H/h3b/replay_deterministic.json'))
a=ap.parse_args()
flags={'nci':False,'deterministic_embed':True}
if not a.control:flags.update(embed_haptic_li_cap=True,q6_donor_pairs={'hh':.95,'ho':.90,'oo':.80})
os.environ['JACOBIK_PROFILE']='u1'
os.environ['JACOBIK_PROFILE_OVERRIDES']=json.dumps(flags)
os.environ['OMP_NUM_THREADS']='3'
os.environ['JACOBIK_NUM_WORKERS']='1'
if not a.nondeterministic:os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from tools.jgen.diag import ctxio
from jacobik.core.env import build_builder_and_inputs
from jacobik.core import profile
from jacobik.stats.stats import StatsLookup,load_db
from jacobik.ik import solver_replay as SR
from jacobik.ik.ik_solver import IKSolverConfig

torch.use_deterministic_algorithms(not a.nondeterministic)
rows=[]
meta={'steps':a.steps,'deterministic_algorithms':not a.nondeterministic,'flags':flags,
      'fingerprint':profile.snapshot()['fingerprint'],'fast_fk':SR.FAST_FK,'cases':rows}
stats=StatsLookup(load_db(str(ctxio.CONSTRAINTS)))

def digest(parts):
    h=hashlib.sha256()
    def add(value):
        if torch.is_tensor(value):
            arr=value.detach().cpu().numpy();h.update(str((arr.dtype,arr.shape)).encode());h.update(arr.tobytes())
        elif isinstance(value,dict):
            for key in sorted(value):h.update(str(key).encode());add(value[key])
        elif isinstance(value,(list,tuple)):
            for item in value:add(item)
    add(parts);return h.hexdigest()

def relative(x,y):
    if x.shape!=y.shape:return {'max':None,'median':None,'p99':None,'shape_match':False}
    d=np.abs(x-y)/np.maximum(np.abs(x),1.)
    return {'max':float(d.max()),'median':float(np.median(d)),'p99':float(np.percentile(d,99)),'shape_match':True}

for code in ['FEKKAJ','NUWFAM','QUTWOT']:
    row={'code':code}
    try:
        def solve(replay):
            b,_,err=build_builder_and_inputs(ctxio.load_parent(code))
            if b is None:raise RuntimeError(err)
            raw=[];starts=[];original=b._run_joint_ik
            def observed(*args,**kwargs):
                starts.append({'theta':digest(args[5]),'reference':digest(args[4]),'scorer':digest(vars(args[2]))})
                result=original(*args,**kwargs);raw.append(result[2].detach().cpu().numpy().copy());return result
            b._run_joint_ik=observed
            SR.ENABLED=True;SR.ReplaySolver.replay=replay
            t=time.monotonic()
            result=b.solve(stats=stats,seed=11,angle_bends=True,n_conformers=8,n_ik_steps=a.steps,
                           device='cuda',ik_config=IKSolverConfig(cull_at=(),cull_keep=()))
            return result,np.concatenate(raw),starts,time.monotonic()-t,dict(SR.LAST)
        eager,er,ei,ew,_=solve(False)
        repeat,nr,ni,nw,_=solve(False)
        replay,rr,ri,rw,last=solve(True)
        noise=relative(er,nr);diff=relative(er,rr)
        es=np.asarray([c.score for c in eager.conformers]);rs=np.asarray([c.score for c in replay.conformers])
        same_starts=ei==ni==ri
        returned=(es.shape==rs.shape and np.allclose(es,rs,rtol=1e-6,atol=1e-6)
                  and [c.trial_idx for c in eager.conformers]==[c.trial_idx for c in replay.conformers])
        passed=(same_starts and diff['shape_match'] and diff['max']<=1e-6 and returned)
        row.update(status='ok',parity=bool(passed),identical_starts=same_starts,starts=[ei,ni,ri],
                   eager_repeat_relative=noise,eager_replay_relative=diff,returned_parity=bool(returned),
                   eager_scores=es.tolist(),replay_scores=rs.tolist(),raw_rows=len(er),
                   mode=last.get('mode'),replayed=last.get('replayed'),replay_error=last.get('error'),
                   wall_s={'eager':ew,'eager_repeat':nw,'replay':rw})
    except Exception as exc:row.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    rows.append(row);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(meta,indent=2)+'\n')
    print({k:v for k,v in row.items() if k not in ('starts','traceback','eager_scores','replay_scores')},flush=True)
    torch.cuda.empty_cache()
