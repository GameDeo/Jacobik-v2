"""Run canonical golden checks with original chunk layout and bounded concurrency."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',choices=('production','u1'),required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--max-concurrent',type=int,default=3)
    args=ap.parse_args()
    from tools.jdev import golden as G
    os.environ.pop('JACOBIK_PROFILE_OVERRIDES',None)
    os.environ.update(JACOBIK_PROFILE=args.profile,JACOBIK_NUM_WORKERS='8',OMP_NUM_THREADS='1',
                      MKL_THREADING_LAYER='SEQUENTIAL',PYTHONIOENCODING='utf-8')
    budget=G.REPO/'research/archive/data/energy_scale_refit'
    if not budget.exists(): budget=G.DATA/'energy_scale_refit'
    db=G.REPO/'research/archive/data/refit/final/constraints_2026-10-17.json'
    if not db.exists(): db=G.DATA/'refit/final/constraints_2026-10-17.json'
    os.environ.setdefault('JACOBIK_BUDGET_OUT',str(budget))
    os.environ.setdefault('JACOBIK_BUDGET_DB',str(db))
    import jacobik.ik.ik_solver as IKM
    from concurrent.futures import ThreadPoolExecutor
    class LimitedExecutor(ThreadPoolExecutor):
        def __init__(self,max_workers=None,**kwargs):
            super().__init__(max_workers=min(args.max_concurrent,max_workers or args.max_concurrent),**kwargs)
    IKM.ThreadPoolExecutor=LimitedExecutor
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic();rc=0
    with out.with_suffix('.log').open('w') as log, contextlib.redirect_stdout(log):
        try:
            G._run(args.profile,'check')
        except SystemExit as e:
            rc=int(e.code or 0)
    record={'profile':args.profile,'layout_workers':8,'max_concurrent_workers':args.max_concurrent,
            'exit_code':rc,'wall_s':time.monotonic()-t0,'log':str(out.with_suffix('.log'))}
    out.write_text(json.dumps(record,indent=2));print(record,flush=True)
    raise SystemExit(rc)

if __name__=='__main__':
    main()
