"""48 O1 row identities: true continuous Adam state, constrained objective.

Replay is not used: this is CPU diagnostic instrumentation. T=250 with cull at
60 preserves the first 200 production steps. Schedule drops start at 200.
Run against the recorded pre-merge source snapshot to preserve O1 embeddings.
"""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from jacobik.core import profile
from jacobik.ik import ik_solver as IKM,finish_schedule as FSM
from tools.jgen import inputs
from tools.jgen.run_o1_g2 import PARENTS
from tools.jgen.atom_export import symbols_for_conformer
from tools.jgen.score_repro import CaptureSession
from tools.jdev.budget.common import stats_from_db
from tools.runkit import xtb
from jacobik.core.env import build_builder_and_inputs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--parents',nargs='+',default=PARENTS)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--max-concurrent',type=int,default=5)
    args=ap.parse_args();torch.set_num_threads(1);assert 1<=args.max_concurrent<=5
    assert profile.snapshot()['fingerprint']=='615ae97d2350','use pre-merge runtime snapshot for original O1 initialization'
    class LimitedExecutor(ThreadPoolExecutor):
        def __init__(self,max_workers=None,**kw):super().__init__(max_workers=min(args.max_concurrent,max_workers or args.max_concurrent),**kw)
    IKM.ThreadPoolExecutor=LimitedExecutor
    original_schedule=FSM.FinishSchedule
    class DiagnosticSchedule(original_schedule):
        def __init__(self,theta,steps,start,mode,**kw):super().__init__(theta,steps,200,mode,**kw)
    FSM.FinishSchedule=DiagnosticSchedule
    root=Path(__file__).resolve().parents[4]/'research/data/jgen/campaign/B/g'
    old=json.loads((root/'o1_g2_cap30_unrel_gpu_results.json').read_text())
    out=Path(args.out);result=json.loads(out.read_text()) if args.resume and out.exists() else {'parents':{},'failures':{},'calibration_trials':[{'zigzag':.5,'patience':4,'delta':.001}],
        'runtime_source':'9ff8655 src snapshot','production_fingerprint':'615ae97d2350','diagnostic':'T250, cull60, bend-release100, schedule starts200; native CPU constrained objective, continuous theta/m/v/dual state; all 48 original row identities',
        'dtype':'float32 (native production CPU)','library':IKM.__file__}
    stats=stats_from_db(inputs.DB_PATH);xtb.require()
    try:
        for code in args.parents:
            os.environ['JACOBIK_NUM_WORKERS']='8' if code in ('QEZVEY','USAPOU') else '3'
            result['parents'].setdefault(code,{'arms':{}})
            for mode in ('control','zigzag','plateau'):
                if mode in result['parents'][code]['arms']:continue
                started=time.monotonic();context={};identities={};records={};digests={}
                for row in old['parents'][code]['rows']:
                    variant,assignment,index=row['key'];key=(0 if variant is None else variant,index)
                    digests[key]=hashlib.sha256()
                    records[key]={'key':row['key'],'old_delivered_score':row['score64_theta0'],'best_before_200':-float('inf'),'trace':[]}
                try:
                    sh,ds=inputs.frame_shards()[code];mol=inputs.load_mol(code,sh,ds)
                    builder,_,err=build_builder_and_inputs(mol)
                    if builder is None:raise RuntimeError(err)
                    original_joint=builder._run_joint_ik
                    def joint(ctx,fks,scorer,ge,X,theta,*rest,**kw):
                        context.update(variant=len(session.captures),ptr=theta[0].untyped_storage().data_ptr(),offset=theta[0].storage_offset(),stride=theta[0].stride(0),candidate=rest[-1])
                        return original_joint(ctx,fks,scorer,ge,X,theta,*rest,**kw)
                    builder._run_joint_ik=joint
                    def observe(solver,t,alive,theta,m,v,g,score,finish):
                        sid=id(solver)
                        if t==0:
                            assert theta[0].untyped_storage().data_ptr()==context['ptr']
                            offset=theta[0].storage_offset()-context['offset'];assert offset%context['stride']==0
                            offset//=context['stride'];variant=context['variant'];selected=[]
                            for (vi,row),rec in records.items():
                                if vi==variant and offset<=row<offset+len(theta[0]):
                                    assert int(context['candidate'][row])==rec['key'][1]
                                    selected.append((row-offset,rec,digests[(vi,row)]))
                            identities[sid]=selected
                        for local,rec,hasher in identities[sid]:
                            found=torch.tensor([local]) if alive is None else (alive==local).nonzero().flatten()
                            if len(found)!=1:
                                if t>=200:raise RuntimeError(f'original O1 row culled: {rec["key"]}')
                                continue
                            i=int(found[0]);value=float(score[i])
                            if t<200:
                                rec['best_before_200']=max(rec['best_before_200'],value)
                                for state in (theta,m,v,g):
                                    for tensor in state:hasher.update(tensor[i].detach().contiguous().numpy().tobytes())
                                hasher.update(score[i].detach().numpy().tobytes())
                            else:
                                rec['trace'].append({'t':t,'score':value,'best':float(finish.best[i]),'z':float(finish.z[i]),'lr':float(finish.lr[i]),'n_flat_or_zigzag':int(finish.count[i])})
                    IKM.FINISH_SCHEDULE=None if mode=='control' else mode;IKM.FINISH_EARLY_STOP=False;IKM.POLISH=None;IKM.OPTIMIZER=None
                    IKM.JointIKSolver.finish_hook=staticmethod(observe)
                    with xtb.FallbackWatch(raise_on_exit=True),CaptureSession(builder) as session:
                        solved=builder.solve(seed=11,stats=stats,angle_bends=True,device='cpu',n_conformers=16,n_ik_steps=200 if mode=='control' else 250,
                            ik_config=IKM.IKSolverConfig(cull_at=(.3 if mode=='control' else .24,),cull_keep=(.35,)))
                    assert solved.conformers
                    symbols=symbols_for_conformer(solved.conformers[0])
                    rows=[]
                    for key,rec in records.items():
                        assert len(rec['trace'])==(0 if mode=='control' else 50),rec['key']
                        rec['first200_trajectory_sha256']=digests[key].hexdigest()
                        rec['pre200_best_score_error']=rec['best_before_200']-rec['old_delivered_score']
                        if mode!='control':
                            reference=next(r for r in result['parents'][code]['arms']['control']['rows'] if r['key']==rec['key'])
                            assert rec['first200_trajectory_sha256']==reference['first200_trajectory_sha256'],('pre200 state trajectory differs',code,rec['key'])
                            assert rec['best_before_200']==reference['best_before_200']
                            rec['first200_trajectory_matches_control']=True
                        drops=[r['t'] for r in rec['trace'] if r['lr']<.1-1e-8]
                        rec.update(first_halving=min(drops) if drops else None,at_floor=bool(rec['trace']) and rec['trace'][-1]['lr']<=.01+1e-8,symbols=symbols,symbols_order='builder/conf.complex.atoms')
                        rows.append(rec)
                    result['parents'][code]['arms'][mode]={'rows':rows,'wall_s':time.monotonic()-started}
                except Exception:result['failures'][f'{code}/{mode}']=traceback.format_exc()
                finally:IKM.JointIKSolver.finish_hook=None;IKM.FINISH_SCHEDULE=None
                out.write_text(json.dumps(result,indent=2));print(code,mode,len(result['parents'][code]['arms'].get(mode,{}).get('rows',[])),round(time.monotonic()-started,2),result['failures'].get(f'{code}/{mode}',''),flush=True)
        result['summary']={}
        for mode in ('zigzag','plateau'):
            rows=[r for p in result['parents'].values() for r in p['arms'].get(mode,{}).get('rows',[])]
            result['summary'][mode]={'rows':len(rows),'criterion_fired':sum(r['first_halving'] is not None for r in rows),'at_floor':sum(r['at_floor'] for r in rows),'max_raw_vs_delivered_score_difference':max([abs(r['pre200_best_score_error']) for r in rows],default=None),'all_first200_trajectories_identical':bool(rows) and all(r.get('first200_trajectory_matches_control') for r in rows)}
        out.write_text(json.dumps(result,indent=2));print(result['summary'],flush=True)
    finally:FSM.FinishSchedule=original_schedule;IKM.JointIKSolver.finish_hook=None;IKM.FINISH_SCHEDULE=None

if __name__=='__main__':main()
