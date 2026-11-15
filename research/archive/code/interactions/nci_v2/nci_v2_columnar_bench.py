"""One-process nice-19 synthetic columnar scaling measurement (one size per run)."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from pathlib import Path
import time
import numpy as np
from research.archive.code.interactions.nci_v2.nci_v2_columnar import write_columns,RootRows,support_mass,PreparedRows,ChunkedProblem,host_budget
from research.archive.code.interactions.nci_v2.nci_v2_devio import Run,write_json,load_json,digest
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import BudgetPause

ROOT=('bond_atom','polar_h_acceptor',('N','H','O'),(0,0,0),'single',None,(False,False,False),False,True)
CFG=dict(interior_knots=4,lambda_r=.1,lambda_z=.1,lambda_0=.01)


def synthetic_rows(n,groups=80):
    rng=np.random.default_rng(17)
    for start in range(0,n//2,8192):
        host_budget();count=min(8192,n//2-start);u=rng.random((count,4))
        X=np.c_[1.1+6.9*u[:,0],2*u[:,1]-1,2*u[:,2]-1,u[:,3]].astype(np.float32)
        for j,x in enumerate(X):
            g=(start+j)%groups;ctx=(ROOT,(ROOT,2),((ROOT,2),'Fe' if g%2 else 'Co','octahedral'))
            for source in (0,1):
                yield dict(features=x.tolist(),context=ctx,family='bond_atom',group_id=str(g),
                    y=source,weight=float(np.exp(.6*u[j,0]+.1*u[j,1])) if source else 1.)


def run(spec_path,n,out,reuse_columns=None):
    if os.nice(0)<19:raise ValueError('nice -n 19 required')
    host_budget(start=True);spec=load_json(spec_path)
    inputs=[Path(reuse_columns)/'dataset.json',*sorted(Path(reuse_columns).glob('root_*.npz'))] if reuse_columns else []
    run=Run('phase2-columnar-scaling',spec_path,inputs,options=dict(rows=n,groups=80,dimension=4,config=CFG,seed=17,reuse_columns=str(reuse_columns)),out=out)
    if run.resumed:return run.path
    checkpoint=run.path/'storage_checkpoint.json'
    if reuse_columns:
        storage=0.
    elif checkpoint.exists():
        saved=load_json(checkpoint)
        if digest(run.path/'columns/dataset.json')!=saved['dataset_sha256']:raise ValueError('columnar checkpoint mismatch')
        storage=saved['storage_wall_s']
    else:
        t=time.perf_counter();meta=write_columns(synthetic_rows(n),run.path/'columns',spec);storage=time.perf_counter()-t
        write_json(checkpoint,dict(storage_wall_s=storage,dataset_sha256=digest(run.path/'columns/dataset.json')))
    columns=Path(reuse_columns) if reuse_columns else run.path/'columns'
    rows=RootRows(columns);
    if rows.n!=n:raise ValueError('requested rows do not match verified source columns')
    t=time.perf_counter();mass=support_mass(rows,spec)
    prep=PreparedRows(rows,spec,mass_cache=mass);support=time.perf_counter()-t
    problem=ChunkedProblem(prep,CFG);t=time.perf_counter();initial=np.zeros(problem.p*problem.blocks)
    v,g=problem.objective(initial);onepass=time.perf_counter()-t
    t=time.perf_counter();result=problem.fit();fitting=time.perf_counter()-t
    report=dict(input_rows=n,stored_rows=rows.n,storage_wall_s=storage,support_wall_s=support,
        objective_wall_s=onepass,fit_wall_s=fitting,iterations=result.nit,success=bool(result.success),
        nfev=result.nfev,coefficient_count=len(result.x),correction_blocks=problem.blocks-1,
        max_gradient=float(np.max(np.abs(result.jac))),normalized_loss=float(result.fun),
        bytes_on_disk=sum(p.stat().st_size for p in columns.glob('*') if p.is_file()),
        estimated_480_fit_s=480*fitting,estimated_20M_rows_480_fit_s=480*fitting*20000000/n,
        caution='Linear pass extrapolation only, excludes root-specific convergence/support/CV differences and warm-start savings.',
        all_groups_independent=80,approximate_reductions=False)
    write_json(run.path/'reports/scaling.json',report)
    write_json(run.path/'surfaces/surface.json',problem.surface(result.x))
    run.finish(dict(input_rows=n,stored_rows=rows.n,objective_evaluations=result.nfev),
        ['Float32 feature quantization is declared; exact comparisons use the same stored precision.',
         'Outside-domain boundary mass is retained; discarding it is not equivalent to the dense estimator.'])
    return run.path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--reuse-columns',type=Path);p.add_argument('--rows',type=int,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    try:print(run(a.spec,a.rows,a.out_dir,a.reuse_columns),flush=True)
    except BudgetPause as e:print(f'PAUSED: {e}',flush=True);raise SystemExit(75)

if __name__=='__main__':main()
