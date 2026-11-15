"""CPU whole-run iRprop sanity, paired default-layout C0; no panel evaluation."""
import json
import os
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from tools.jgen.run_o1_g2 import PARENTS
from tools.jgen import inputs
from tools.jdev.budget.common import stats_from_db
from jacobik.ik import ik_solver as IKM


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--parents', nargs='+', default=PARENTS)
    ap.add_argument('--max-concurrent', type=int, default=2)
    ap.add_argument('--save-structures', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--arms', nargs='+', choices=('C0','R2','R2_minus'), default=['C0','R2'])
    args=ap.parse_args()
    if not 1 <= args.max_concurrent <= 5: ap.error('max concurrency must be 1..5')
    torch.set_num_threads(1);os.environ['JACOBIK_NUM_WORKERS']='8'
    class LimitedExecutor(ThreadPoolExecutor):
        def __init__(self,max_workers=None,**kw):super().__init__(max_workers=min(args.max_concurrent,max_workers or args.max_concurrent),**kw)
    IKM.ThreadPoolExecutor=LimitedExecutor
    stats=stats_from_db(inputs.DB_PATH);out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    result=json.loads(out.read_text()) if args.resume and out.exists() else {'parents':{},'failures':{}}
    prior_path=Path(__file__).resolve().parents[4]/'research/data/jgen/campaign/B/g/g3prime_r2.json'
    prior=json.loads(prior_path.read_text())
    from tools.jgen.atom_export import symbols_for_conformer
    result.update(layout_workers=8,max_concurrent=args.max_concurrent,structures_saved=args.save_structures)
    from jacobik.core.env import build_builder_and_inputs
    from tools.runkit import xtb
    xtb.require()
    for code in args.parents:
        result['parents'].setdefault(code,{})
        for arm in args.arms:
            if args.resume and arm in result['parents'][code] and (not args.save_structures or 'conformers' in result['parents'][code][arm]):continue
            t0=time.monotonic()
            try:
                sh,ds=inputs.frame_shards()[code];mol=inputs.load_mol(code,sh,ds)
                builder,_,err=build_builder_and_inputs(mol)
                if builder is None:raise RuntimeError(err)
                IKM.OPTIMIZER={'C0':None,'R2':'irprop','R2_minus':'irprop_minus'}[arm]
                with xtb.FallbackWatch(raise_on_exit=True):
                    solved=builder.solve(seed=11,stats=stats,angle_bends=True,device='cpu',n_conformers=16,n_ik_steps=200 if arm=='C0' else 250)
                result['parents'][code][arm]={'top1_score':float(solved.conformers[0].score),'rows':len(solved.conformers),'wall_s':time.monotonic()-t0}
                record=result['parents'][code][arm]
                record['prior_top1_score']=prior['parents'][code].get(arm,{}).get('top1_score')
                record['score_reproduction']=(None if record['prior_top1_score'] is None else
                    abs(record['top1_score']-record['prior_top1_score'])<=1e-6)
                if args.save_structures:
                    record['reference']={'symbols':[a.symbol for a in mol.atoms],
                        'X_A':[a.position.tolist() for a in mol.atoms],
                        'charge':int(mol.charge),'spin':int(mol.spin_state)}
                    record['conformers']=[]
                    for conf in solved.conformers:
                        symbols=symbols_for_conformer(conf)
                        assert symbols == [a.symbol for a in conf.complex.atoms]
                        X=[a.position.tolist() for a in conf.complex.atoms]
                        assert len(symbols)==len(X)
                        record['conformers'].append({'symbols':symbols,'X_A':X,'score':float(conf.score),
                            'key':[conf.embed_idx,int(conf.assignment_idx),int(conf.trial_idx)],
                            'symbols_order':'builder/conf.complex.atoms'})
                if record['score_reproduction'] is False:raise RuntimeError('top1 score did not reproduce prior R2/C0 sanity')

            except Exception:result['failures'][f'{code}/{arm}']=traceback.format_exc()
            out.write_text(json.dumps(result,indent=2));print(code,arm,{k:v for k,v in result['parents'][code].get(arm,{}).items() if k not in ('conformers','reference')},result['failures'].get(f'{code}/{arm}',''),flush=True)
    IKM.OPTIMIZER=None

if __name__=='__main__':main()
