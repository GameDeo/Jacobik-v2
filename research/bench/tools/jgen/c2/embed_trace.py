"""Capture actual xTB failure diagnostics without altering its input/output."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from .rings import population


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--code',required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert a.code in population()[0];a.out.mkdir(parents=True,exist_ok=True)
    from tools.jgen.bench.run import _setup_env
    _setup_env({})
    from tools.jgen.a4t.analyse import _load_parent
    from tools.jdev.budget.common import stats_from_db
    from tools.jgen import inputs
    from jacobik.ik.ik_solver import IKSolverConfig
    import torch
    torch.set_num_threads(1)
    original=subprocess.run;calls=[]
    def run(args,*pargs,**kwargs):
        if isinstance(args,list) and Path(args[0]).name=='xtb' and 'l.xyz' in args:
            directory=a.out/f'call_{len(calls)}';directory.mkdir()
            cwd=Path(kwargs['cwd'])
            for name in ['l.xyz','c.inp']:
                if (cwd/name).exists():shutil.copy2(cwd/name,directory/name)
            with (directory/'xtb.log').open('w') as log:
                kwargs.update(stdout=log,stderr=subprocess.STDOUT)
                result=original(args,*pargs,**kwargs)
            calls.append(dict(args=args,returncode=result.returncode,optimized=(cwd/'xtbopt.xyz').exists()))
            return result
        return original(args,*pargs,**kwargs)
    result=dict(code=a.code)
    try:
        subprocess.run=run
        _,builder,_=_load_parent(a.code)
        solved=builder.solve(seed=11,stats=stats_from_db(inputs.DB_PATH),device='cpu',n_ik_steps=2,
            angle_bends=True,ik_config=IKSolverConfig(n_workers=1,cull_at=(),cull_keep=()))
        result.update(status='degraded_return',n_conformers=len(solved.conformers))
    except Exception as exc:result.update(status='raised',reason=repr(exc))
    finally:subprocess.run=original
    result['calls']=calls
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
