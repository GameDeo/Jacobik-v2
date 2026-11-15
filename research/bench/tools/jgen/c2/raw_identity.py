"""Full-precision geometry supplement to the --cpu-identity bench check."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--baseline-engine',action='store_true')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    from tools.jgen.bench.run import _setup_env
    _setup_env({})
    from jacobik.builder import engine as E
    if a.baseline_engine:
        source=subprocess.check_output(['git','show','b0134a2:src/jacobik/engine.py'],text=True)
        exec(compile(source,'<b0134a2-engine>','exec'),E.__dict__)
    from .phase0 import prepare_parent
    from .rings import population
    from jacobik.ik.ik_solver import IKSolverConfig
    codes=['BABNAV','MIVXAR','UYOVUA'];assert set(codes)<=set(population()[0])
    for code in codes:
        try:
            _,builder,ctx,assignments,W,T=prepare_parent(code,'cpu')
            ctx.ik_config=IKSolverConfig(n_workers=1,cull_at=(),cull_keep=())
            result=builder._solve_for_assignments(ctx,assignments,W,T)
            X=np.asarray([[atom.position for atom in c.complex.atoms] for c in result.conformers])
            scores=np.asarray([c.score for c in result.conformers])
            np.savez_compressed(a.out/f'{code}.npz',X=X,scores=scores)
            status=dict(code=code,status='ok',n_rows=len(scores),seed=11,
                settings=dict(n_workers=1,cull_at=[],cull_keep=[]),energy_evaluation='not performed: geometry-only identity supplement')
        except Exception as exc:
            status=dict(code=code,status='failed',reason=repr(exc))
        (a.out/f'{code}.json').write_text(json.dumps(status))
        print(status,flush=True)


if __name__=='__main__':main()
