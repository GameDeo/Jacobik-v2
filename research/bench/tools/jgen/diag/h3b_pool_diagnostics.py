"""CPU score terms, geminal/close-contact geometry and saved MACE energy for each pool row."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--directory',type=Path,required=True,help='One arm directory containing records/ and xyz/')
ap.add_argument('--codes',required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();codes=a.codes.split(',')
sample=next(json.loads(p.read_text()) for p in sorted((a.directory/'records').glob('*.json')) if json.loads(p.read_text()).get('status')=='ok')
flags={'nci':False,'deterministic_embed':True,**sample['flags']}
os.environ['JACOBIK_PROFILE']='u1';os.environ['JACOBIK_PROFILE_OVERRIDES']=json.dumps(flags)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import torch
from tools.jgen.diag import ctxio
from tools.jgen.diag.h3_guard_score_terms import _q3_override
from tools.jgen.embedding_validity import EmbeddingValidity
from tools.jdev import multiseed
from jacobik.core.env import build_builder_and_inputs
from jacobik.builder import engine as E
from jacobik.stats.stats import StatsLookup,load_db

stats=StatsLookup(load_db(str(ctxio.CONSTRAINTS)));rows_out=[];failures=[]
validity=EmbeddingValidity(E).install()
for code in codes:
    try:
        validity.parent=code
        builder,_,err=build_builder_and_inputs(ctxio.load_parent(code))
        if builder is None:raise RuntimeError(err)
        ctx,assignments,width,_=multiseed.prepare(builder,11,device='cpu',angle_bends=True,stats=stats)
        (fks,_,ge,scorer),seed_rows=multiseed.seed_rows(builder,ctx,assignments,width,(11,12,13))
        validity.check(code)
        if E.CLASS_BUDGET=='split':
            keep=builder._class_budget_rows(ctx,assignments,width)
            if keep is not None:
                kt=torch.as_tensor(keep,dtype=torch.long)
                seed_rows=[([t[kt] for t in th],X[kt] if X.dim()==3 else X,ideal[kt],cand[keep]) for th,X,ideal,cand in seed_rows]
    except Exception as exc:
        failures.append({'code':code,'stage':'prepare','error':str(exc)});continue
    for si,seed in enumerate((11,12,13)):
        try:
            rec=json.loads((a.directory/'records'/f'{code}_{seed}.json').read_text())
            if rec['status']!='ok':raise RuntimeError(rec.get('reason','not ok'))
            files=sorted((a.directory/'xyz'/code).glob(f'{code}_{rec["arm"]}_s{seed}_row*.xyz'))
            if len(files)!=rec['n_rows']:raise RuntimeError(f'pool count {len(files)} != record {rec["n_rows"]}')
            headers=[];positions=[];symbols=None
            for path in files:
                lines=path.read_text().splitlines();headers.append(json.loads(lines[1]))
                positions.append([[float(v) for v in line.split()[1:4]] for line in lines[2:]])
                if symbols is None:symbols=[line.split()[0] for line in lines[2:]]
            X=torch.as_tensor(np.asarray(positions),dtype=torch.float32)
            theta,x_ref,_,candidate_idx=seed_rows[si]
            trials=[int(h['row_key'][2]) for h in headers];groups=[int(h['row_key'][1]) for h in headers]
            if any(t>=len(candidate_idx) or int(candidate_idx[t])!=g for t,g in zip(trials,groups)):
                raise RuntimeError('row keys do not match reconstructed initial rows')
            with torch.no_grad():
                modes=_q3_override(scorer,ge,ctx,fks,theta,x_ref,None)
                override=None if modes is None else modes[torch.as_tensor(trials)]
                stat,bd=scorer.score(X,row_groups=torch.as_tensor(groups,dtype=torch.long),q3_mode_override=override)
                angles=torch.rad2deg(ge.q2_dmd_angles(X)).numpy()
            geminal=set()
            for fk in fks:
                plan=fk._bend_plan
                if plan is None:continue
                by_pivot={}
                for i,sp in enumerate(fk._bend_specs):
                    if len(sp)>4 and sp[4][0]=='haptic':
                        pivot=int(plan.spec_piv[i]);other=int(plan.spec_oth[i])
                        if symbols[other]=='H':by_pivot.setdefault(pivot,set()).add(other)
                for pivot,hydrogens in by_pivot.items():
                    for h1,h2 in itertools.combinations(sorted(hydrogens),2):geminal.add((pivot,h1,h2))
            hs=[i for i,s in enumerate(symbols) if s=='H']
            for i,(header,pos) in enumerate(zip(headers,np.asarray(positions))):
                distance=np.linalg.norm(pos[:,None]-pos[None,:],axis=-1);np.fill_diagonal(distance,np.inf)
                hh=distance[np.ix_(hs,hs)] if len(hs)>1 else np.asarray([])
                gs=[]
                for pivot,h1,h2 in geminal:
                    u=pos[h1]-pos[pivot];v=pos[h2]-pos[pivot]
                    gs.append(float(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))))
                known=rec.get('top1_bend_prior') if header['is_top1'] else rec.get('floor_bend_prior') if header['is_floor'] else None
                row={'code':code,'seed':seed,'arm':rec['arm'],'row_index':header['row_index'],
                     'is_top1':header['is_top1'],'is_floor':header['is_floor'],'score_recorded':header['score'],
                     'stat_score_recomputed':float(stat[i]),'implied_added_prior':header['score']-float(stat[i]),
                     'known_bend_prior':known,'rescore_residual_with_known_prior':None if known is None else float(stat[i])+known-header['score'],
                     'mace_dE_kj':header['mace_energy_kj']-rec['ref_energy_kj'],
                     'min_interatomic_A':float(distance.min()),'min_HH_A':float(hh.min()) if hh.size else None,
                     'min_haptic_geminal_angle_deg':min(gs) if gs else None,
                     'donor_angle_min_deg':float(angles[i].min()) if angles.shape[1] else None,
                     'donor_angle_max_deg':float(angles[i].max()) if angles.shape[1] else None}
                row.update({f'Q{j}':float(bd[f'Q{j}'][i]) for j in range(1,7)})
                rows_out.append(row)
        except Exception as exc:failures.append({'code':code,'seed':seed,'stage':'rescore','error':str(exc)})
    print(code,'rows',sum(r['code']==code for r in rows_out),flush=True)
validity.close()
a.out.parent.mkdir(parents=True,exist_ok=True)
if rows_out:
    with a.out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows_out[0]));writer.writeheader();writer.writerows(rows_out)
a.out.with_suffix('.failures.json').write_text(json.dumps(failures,indent=2)+'\n')
print(json.dumps({'rows':len(rows_out),'failures':failures}),flush=True)
