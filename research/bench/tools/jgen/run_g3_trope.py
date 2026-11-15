"""G-3 CPU trope filter with compute-matched controls; strain disabled."""
import argparse
import json
from pathlib import Path
from tools.runkit.trope import suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root',required=True)
    ap.add_argument('--arms',nargs='+',choices=('C0','C1','A1','P1','P2'),required=True)
    ap.add_argument('--only',default='')
    ap.add_argument('--procs',type=int,default=3)
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,12,13])
    ap.add_argument('--dt',type=float,default=0.038974768870482596)
    ap.add_argument('--no-strain',action='store_true',default=True,help='Always enabled for G-3 screens')
    args = ap.parse_args()
    root=Path(args.out_root)
    foot=json.loads((Path(__file__).resolve().parents[3]/'campaigns/g/o2_trope_footprint.json').read_text())
    choices={'C0':({},200),'C1':({},250),'A1':({'adam_metric':True},200),
             'P1':({'polish':'fire','polish_dt':args.dt},200),'P2':({'polish':'adam_small'},200)}
    for name in args.arms:
        flags,steps=choices[name]
        suite.run_suite(name,flags,foot,root/name,only=set(filter(None,args.only.split(','))) or None,
                        procs=args.procs,agent='CodexB',unrelaxed_only=True,mace_device='cuda',
                        seeds_override=args.seeds,solve_steps=steps)
    for name in args.arms:
        if name in ('C0','C1'):
            continue
        control='C0' if name=='A1' else 'C1'
        if (root/control).exists():
            result=suite.compare(root/control,root/name,foot)
            (root/f'compare_{name}.json').write_text(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
