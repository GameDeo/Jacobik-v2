"""48 fixed O1 endpoints: prescribed late-Adam trigger calibration, no energy tuning."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from jacobik.ik import ik_solver as IKM
from jacobik.ik.finish_schedule import FinishSchedule
from jacobik.ik.kernels import wrap_angle
from tools.jgen.run_g3_sanity import capture_parent, functions
from tools.jgen.run_o1_g2 import PARENTS
from tools.jgen.atom_export import symbols_for_conformer
from tools.jgen import inputs
from tools.jdev.budget.common import stats_from_db


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--parents',nargs='+',default=PARENTS)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1)
    class LimitedExecutor(ThreadPoolExecutor):
        def __init__(self,max_workers=None,**kw):super().__init__(max_workers=min(3,max_workers or 3),**kw)
    IKM.ThreadPoolExecutor=LimitedExecutor
    root=Path(__file__).resolve().parents[4]/'research/data/jgen/campaign/B/g'
    prior=json.loads((root/'o1_g2_cap30_unrel_gpu_results.json').read_text())
    out=Path(args.out);result=json.loads(out.read_text()) if args.resume and out.exists() else {'parents':{},'failures':{},'calibration_trials':[{'zigzag':.5,'patience':4,'delta':.001}],'diagnostic_steps':50,'description':'200-step delivered theta; inherited last available moments and bias count; 50 diagnostic continuation steps; distinct from full T=200 solves'}
    stats=stats_from_db(inputs.DB_PATH)
    for code in args.parents:
        if code in result['parents']:continue
        t0=time.monotonic();endpoints=[]
        def endpoint(th,m,v,ts):endpoints.append(([x.detach().clone() for x in th],[x.clone() for x in m],[x.clone() for x in v],ts.clone()))
        try:
            os.environ['JACOBIK_NUM_WORKERS']='8' if code in ('QEZVEY','USAPOU') else '3'
            IKM.JointIKSolver.finish_endpoint_hook=staticmethod(endpoint)
            session,solved=capture_parent(code,stats)
            IKM.JointIKSolver.finish_endpoint_hook=None
            rows=[]
            for old in prior['parents'][code]['rows']:
                variant,assignment,row=old['key'];cap=session.captures[0 if variant is None else variant]
                assert int(cap.candidate_idx_per_b[row])==assignment
                conf=next(c for c in solved.conformers if (c.embed_idx,c.assignment_idx,c.trial_idx)==tuple(old['key']))
                th0=[t[row:row+1].detach().double() for t in cap.thetas_final]
                score,coords,_=functions(cap,row)
                assert abs(float(score(th0))-old['score64_theta0'])<1e-4
                assert np.max(np.abs(coords(th0)[0].detach().numpy()-np.array(old['X_init_A'])))<1e-4
                matched=[]
                for th,m,v,ts in endpoints:
                    if len(th)!=len(th0) or any(x.shape[1]!=y.shape[1] for x,y in zip(th,th0)):continue
                    err=sum((x.double()-y).abs().amax(-1) for x,y in zip(th,th0))
                    for idx in (err<1e-6).nonzero().flatten():matched.append(([x[idx:idx+1].double() for x in m],[x[idx:idx+1].double() for x in v],int(ts[idx])))
                assert matched,'missing inherited moments'
                m0,v0,count=matched[0]
                rec={'key':old['key'],'symbols':symbols_for_conformer(conf),'symbols_order':'builder/conf.complex.atoms','score0':float(score(th0)),'adam_bias_count':count,'arms':{}}
                for mode in ('zigzag','plateau'):
                    th=[x.clone().requires_grad_(True) for x in th0];m=[x.clone() for x in m0];v=[x.clone() for x in v0]
                    state=FinishSchedule(th,50,0,mode,record=True);trace=[]
                    for t in range(50):
                        val=score(th);g=torch.autograd.grad(-val.sum(),th)
                        with torch.no_grad():
                            lr=state.update(torch.tensor([t]),val,m,g)
                            trace.append({'t':200+t,'score':float(val),'best':float(state.best[0]),'z':float(state.z[0]),'lr':float(state.lr[0]),'n_flat_or_zigzag':int(state.count[0])})
                            for li in range(len(th)):
                                m[li]=.9*m[li]+.1*g[li];v[li]=.999*v[li]+.001*g[li].square()
                                step=lr*(m[li]/(1-.9**(count+t+1)))/((v[li]/(1-.999**(count+t+1))).sqrt()+1e-8)
                                th[li].sub_(step)
                                if th[li].shape[1]>6:th[li][:,6:]=wrap_angle(th[li][:,6:])
                                th[li][:,3:6]=IKM._renorm_axisangle(th[li][:,3:6])
                    rec['arms'][mode]={'trace':trace,'first_halving':int(state.first_drop[0])+200 if state.first_drop[0]>=0 else None,'at_floor':bool(state.lr[0]<=.01),'final_score':float(score(th).detach())}
                rows.append(rec)
            result['parents'][code]={'rows':rows,'wall_s':time.monotonic()-t0}
        except Exception:result['failures'][code]=traceback.format_exc()
        finally:IKM.JointIKSolver.finish_endpoint_hook=None
        out.write_text(json.dumps(result,indent=2));print(code,len(result['parents'].get(code,{}).get('rows',[])),time.monotonic()-t0,result['failures'].get(code,''),flush=True)
    rows=[r for p in result['parents'].values() for r in p['rows']]
    result['summary']={mode:{'rows':len(rows),'criterion_fired':sum(r['arms'][mode]['first_halving'] is not None for r in rows),'at_floor':sum(r['arms'][mode]['at_floor'] for r in rows)} for mode in ('zigzag','plateau')}
    out.write_text(json.dumps(result,indent=2));print(result['summary'],flush=True)

if __name__=='__main__':main()
