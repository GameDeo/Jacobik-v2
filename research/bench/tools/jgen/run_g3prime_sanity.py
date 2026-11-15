"""Exact O1-row iRprop+ sensitivity using captured last Adam moments."""
from tools.jgen.atom_export import symbols_for_conformer
import argparse
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tools.jgen.run_g3_sanity import capture_parent, functions
from tools.jgen.run_o1_g2 import PARENTS
from tools.jgen import inputs
from tools.jdev.budget.common import stats_from_db
from jacobik.ik import polish, ik_solver as IKM


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--parents',nargs='+',default=PARENTS)
    args=ap.parse_args();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    class LimitedExecutor(ThreadPoolExecutor):
        def __init__(self,max_workers=None,**kw): super().__init__(max_workers=min(3,max_workers or 3),**kw)
    IKM.ThreadPoolExecutor=LimitedExecutor
    root=Path(__file__).resolve().parents[4]/'research/data/jgen/campaign/B/g'
    old=json.loads((root/'o1_g2_cap30_unrel_gpu_results.json').read_text())
    stats=stats_from_db(inputs.DB_PATH)
    result={'parents':{},'failures':{},'delta0_source':'last Adam effective step, by row including culled archives','started':time.time()}
    original=polish.optimise
    for code in args.parents:
        t0=time.monotonic();captured=[]
        def observe(th,score_fn,*a,delta0=None,**kw):
            captured.append(([t.detach().clone() for t in th],[d.detach().clone() for d in delta0]))
            return th,score_fn(th).detach()
        try:
            os.environ['JACOBIK_NUM_WORKERS']='8' if code in ('QEZVEY','USAPOU') else '3'
            IKM.POLISH='irprop';polish.optimise=observe
            session,solved=capture_parent(code,stats)
            polish.optimise=original;IKM.POLISH=None
            rows=[]
            for prior in old['parents'][code]['rows']:
                variant,assignment,row=prior['key'];cap=session.captures[0 if variant is None else variant]
                conf = next(c for c in solved.conformers if (c.embed_idx,c.assignment_idx,c.trial_idx) == tuple(prior['key']))
                assert int(cap.candidate_idx_per_b[row]) == assignment
                th0=[t[row:row+1].detach().double() for t in cap.thetas_final]
                score,coords,metric=functions(cap,row);s0=float(score(th0).detach())
                assert abs(s0-prior['score64_theta0'])<1e-4,(s0,prior['score64_theta0'])
                assert np.max(np.abs(coords(th0)[0].detach().numpy()-np.asarray(prior['X_init_A'])))<1e-4
                matches=[]
                for th,ds in captured:
                    if len(th)!=len(th0) or any(t.shape[1]!=u.shape[1] for t,u in zip(th,th0)): continue
                    err=sum((t.double()-u).abs().amax(-1) for t,u in zip(th,th0))
                    for idx in (err<1e-6).nonzero().flatten(): matches.append([d[idx:idx+1].double() for d in ds])
                assert matches,'missing Adam handoff'
                delta0=matches[0]
                rec={'key':prior['key'],'score0':s0,'lbfgs_gain':prior['delta_score'],
                     'symbols':symbols_for_conformer(conf),'charge':prior['charge'],'spin':prior['spin_state'],
                     'X0_A':prior['X_init_A'],'delta0':[d.tolist() for d in delta0],'arms':{}}
                for name,method,scale in [('P2','adam_small',1),('R1','irprop',1),('R1_x01','irprop',.1),('R1_x10','irprop',10)]:
                    diag={};start=time.monotonic()
                    best,val=original(th0,score,coords,metric,method=method,delta0=[d*scale for d in delta0],diagnostics=diag)
                    gain=float(val)-s0
                    rec['arms'][name]={'score':float(val),'gain':gain,'fraction':gain/prior['delta_score'],
                          'wall_s':time.monotonic()-start,'X_A':coords(best)[0].detach().tolist(),**diag}
                rows.append(rec)
            result['parents'][code]={'rows':rows,'wall_s':time.monotonic()-t0}
        except Exception:
            result['failures'][code]=traceback.format_exc()
        finally: polish.optimise=original;IKM.POLISH=None
        out.write_text(json.dumps(result,indent=2));print(code,'rows',len(result['parents'].get(code,{}).get('rows',[])),result['failures'].get(code,''),flush=True)

if __name__=='__main__':main()
