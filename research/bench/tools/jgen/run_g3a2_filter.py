"""Paired full solves for the G-3a-double-prime O1 filter, without relaxation."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from tools.jgen.run_o1_g2 import PARENTS
from tools.jgen import inputs
from tools.jgen.atom_export import symbols_for_conformer
from tools.jdev.budget.common import stats_from_db
from tools.runkit import xtb
from jacobik.core.env import build_builder_and_inputs
from jacobik.core import profile
from jacobik.ik import ik_solver as IKM

ARMS={'C0':(200,None,None),'C1':(250,None,None),'P2':(200,'adam_small',None),
      'P2prime':(200,None,'late_cosine'),'P2a':(200,None,'zigzag'),'P2p':(200,None,'plateau')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--parents',nargs='+',default=PARENTS);ap.add_argument('--arms',nargs='+',choices=ARMS,default=['C0','P2prime','P2a','P2p'])
    ap.add_argument('--resume',action='store_true');ap.add_argument('--max-concurrent',type=int,default=3)
    args=ap.parse_args();assert 1<=args.max_concurrent<=5
    torch.set_num_threads(1);os.environ['JACOBIK_NUM_WORKERS']='8'
    class LimitedExecutor(ThreadPoolExecutor):
        def __init__(self,max_workers=None,**kw):super().__init__(max_workers=min(args.max_concurrent,max_workers or args.max_concurrent),**kw)
    IKM.ThreadPoolExecutor=LimitedExecutor;xtb.require()
    stats=stats_from_db(inputs.DB_PATH);out=Path(args.out)
    result=json.loads(out.read_text()) if args.resume and out.exists() else {'parents':{},'failures':{},'seed':11,'layout_workers':8,'device':'cpu','production_fingerprint':profile.snapshot()['fingerprint'],'calibration':{'zigzag':.5,'patience':4,'delta':.001}}
    for code in args.parents:
        result['parents'].setdefault(code,{})
        for arm in args.arms:
            if arm in result['parents'][code]:continue
            start=time.monotonic();chunks={}
            def observe(solver,t,alive,theta,m,v,g,score,finish):
                if finish is None:return
                # Retain only row-summary diagnostics, not large raw-pool trajectories.
                key=id(solver)
                chunks[key]={'rows_alive':len(finish.lr),'t':t,'fired':int(finish.dropped.sum()),
                             'at_floor':int((finish.lr<=.01).sum()),'first_halvings':finish.first_drop.tolist(),
                             'lr':finish.lr.tolist()}
            try:
                sh,ds=inputs.frame_shards()[code];mol=inputs.load_mol(code,sh,ds)
                builder,_,err=build_builder_and_inputs(mol)
                if builder is None:raise RuntimeError(err)
                steps,IKM.POLISH,IKM.FINISH_SCHEDULE=ARMS[arm]
                IKM.JointIKSolver.finish_hook=staticmethod(observe)
                with xtb.FallbackWatch(raise_on_exit=True):
                    solved=builder.solve(seed=11,stats=stats,angle_bends=True,device='cpu',n_conformers=16,n_ik_steps=steps)
                assert solved.conformers
                rec={'top1_score':float(solved.conformers[0].score),'rows':len(solved.conformers),'wall_s':time.monotonic()-start,
                     'reference':{'symbols':[a.symbol for a in mol.atoms],'X_A':[a.position.tolist() for a in mol.atoms],
                                  'charge':int(mol.charge),'spin':int(mol.spin_state)},'conformers':[],'chunks':list(chunks.values())}
                for c in solved.conformers:
                    symbols=symbols_for_conformer(c);assert symbols==[a.symbol for a in c.complex.atoms]
                    X=[a.position.tolist() for a in c.complex.atoms];assert len(symbols)==len(X)
                    rec['conformers'].append({'symbols':symbols,'X_A':X,'score':float(c.score),'key':[c.embed_idx,int(c.assignment_idx),int(c.trial_idx)],'symbols_order':'builder/conf.complex.atoms'})
                result['parents'][code][arm]=rec
            except Exception:result['failures'][f'{code}/{arm}']=traceback.format_exc()
            finally:IKM.JointIKSolver.finish_hook=None;IKM.POLISH=None;IKM.FINISH_SCHEDULE=None
            out.write_text(json.dumps(result,indent=2));print(code,arm,result['parents'][code].get(arm,{}).get('top1_score'),round(time.monotonic()-start,2),result['failures'].get(f'{code}/{arm}',''),flush=True)

if __name__=='__main__':main()
