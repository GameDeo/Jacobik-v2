"""Registered CPU direction check: do relaxed A4b top-1 backbone angles move toward bound-table means?"""
from __future__ import annotations
import json, math, io, logging, shutil
from pathlib import Path
import numpy as np
import torch
from tools.jgen import inputs
from tools.jdev import multiseed
from tools.jdev.budget.common import stats_from_db
from jacobik.core.env import build_builder_and_inputs
from jacobik.ik import angle_bends as AB
from fit.bends import bound_angle_key

ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
A4B=ROOT/'research/data/jgen/campaign/A4b'
PANEL=Path(__file__).resolve().parents[4]/'research/bench/benchmarks/panels/tmqm_cal_tail_120.json'
LOCK=Path(__file__).resolve().parents[4]/'research/bench/benchmarks/locked_exclusions.json'
TABLE=ROOT/'research/data/jgen/campaign/D/q7_bound_angles.bends.json'
OUT=ROOT/'research/data/jgen/campaign/D/q7_a4b_check.json'

def angle_deg(X,a,p,b):
    u=X[a]-X[p];v=X[b]-X[p];den=np.linalg.norm(u)*np.linalg.norm(v)
    return math.degrees(math.acos(float(np.clip(np.dot(u,v)/max(den,1e-12),-1,1))))

def main():
    torch.set_num_threads(1)
    if shutil.which('xtb') is None: raise RuntimeError('xtb is not on PATH; refusing to use fallback embed shells')
    table=json.loads(TABLE.read_text())
    means={tuple([*e['key'][:2],tuple(e['key'][2]) if isinstance(e['key'][2],list) else e['key'][2],*e['key'][3:]]):e
           for e in table.get('entries',[])}
    panel=json.loads(PANEL.read_text()); lock=json.loads(LOCK.read_text())
    excluded=set(lock.get('reserve_all_codes',[]))|set(lock.get('locked_codes',[]))
    codes=[c for c in panel['codes'] if c not in excluded]
    shards=inputs.frame_shards(); stats=stats_from_db(inputs.DB_PATH)
    records=[]; failures=[]
    for ix,code in enumerate(codes,1):
      try:
        shard,dataset=shards[code]; mol=inputs.load_mol(code,shard,dataset)
        builder,_,err=build_builder_and_inputs(mol)
        if builder is None: raise RuntimeError(f'builder failed: {err}')
        logstream=io.StringIO(); handler=logging.StreamHandler(logstream); logger=logging.getLogger('jacobik.builder.engine'); logger.addHandler(handler)
        try: ctx,assign,bw,_=multiseed.prepare(builder,11,device='cpu',angle_bends=True,stats=stats)
        finally: logger.removeHandler(handler)
        if 'GFN2 embed refinement failed' in logstream.getvalue():
            raise RuntimeError('GFN2 embed refinement failed; refusing an MMFF fallback shell')
        (_,_,ge,_),seeds=multiseed.seed_rows(builder,ctx,assign,bw,(11,12,13))
        trip=AB.angle_triplets(mol); atom={a.idx:a for a in mol.atoms}; idx={a.idx:i for i,a in enumerate(mol.atoms)}
        obs=[]
        for si,seed in enumerate((11,12,13)):
            mp=A4B/'xyz'/code/f'{code}_control_s{seed}_pool.json'
            rp=A4B/'relax'/f'{code}_{seed}.json'
            if not mp.exists() or not rp.exists(): continue
            man=json.loads(mp.read_text()); relaxed=json.loads(rp.read_text()).get('rows',{})
            top=next((r for r in man.get('rows',[]) if r.get('is_top1')),None)
            if top is None: continue
            ri=int(top['row_index']); rr=relaxed.get(str(ri),{})
            if rr.get('status')!='ok' or rr.get('X') is None: continue
            theta,x_ref,_,_=seeds[si]
            trial=int(top['row_key'][2])
            xref=np.asarray(x_ref.detach().cpu() if torch.is_tensor(x_ref) else x_ref)
            if xref.ndim==3: xref=xref[min(trial,xref.shape[0]-1)]
            xrel=np.asarray(rr['X'],float)
            for a,p,b in trip:
                hyb=AB.pivot_hybridisation(mol,p)
                key=bound_angle_key(atom[p].symbol,hyb,[atom[a].symbol,atom[b].symbol],None,None)
                # This exact generic key is the row materialized by the sidecar's shared fallback builder.
                ent=means.get(key)
                if ent is None: continue
                th0=angle_deg(xref,idx[a],idx[p],idx[b]); thr=angle_deg(xrel,idx[a],idx[p],idx[b]); thb=float(ent['mean_deg'])
                gap=thb-th0
                if abs(gap)<1e-4: continue
                obs.append({'seed':seed,'row_index':ri,'triplet':[a,p,b],'key':list(key),'embed_deg':th0,'relaxed_deg':thr,'bound_mean_deg':thb,
                    'sign_agree':bool(gap*(thr-th0)>0),'gap_closed':1-abs(thb-thr)/abs(gap)})
        records.append({'code':code,'n_covered':len(obs),'observations':obs})
      except Exception as e:
        failures.append({'code':code,'error':f'{type(e).__name__}: {e}'[:400]})
      if ix%10==0: print(f'{ix}/{len(codes)} parents; covered obs={sum(r["n_covered"] for r in records)}; failures={len(failures)}',flush=True)
    vals=[o for r in records for o in r['observations']]
    summary={'parents_requested':len(codes),'parents_ok':len(records),'failures':failures,'covered_parent_seed_angles':len(vals),
      'sign_agreement':float(np.mean([o['sign_agree'] for o in vals])) if vals else None,
      'mean_gap_closed':float(np.mean([o['gap_closed'] for o in vals])) if vals else None,
      'median_gap_closed':float(np.median([o['gap_closed'] for o in vals])) if vals else None,
      'records':records}
    OUT.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
if __name__=='__main__': main()
