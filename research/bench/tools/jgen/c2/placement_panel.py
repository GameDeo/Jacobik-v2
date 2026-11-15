"""Panel-wide C/K/P/F placement using paired RNG and shared torsion seeds.

Reuses the measured C construction, fits the same conformer three ways, and
uses actual FK to measure output. All native E/Z configurations are included.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time
from pathlib import Path
from .phase0 import prepare_parent, seeded_rows, quantiles
from .rings import population


def measure(code,out):
    started=time.monotonic();values=[];failures=[]
    from tools.jgen.bench.run import _setup_env
    _setup_env({})
    import torch
    from jacobik.ik.fk_compose import ComplexFK
    from jacobik.ik.kabsch import kabsch_init,compensate_fk_pivot
    from tools.jgen.a4t.analyse import _load_parent
    try:
        _,probe,_=_load_parent(code);variants=probe._ez_combinations()
        for vi in range(len(variants)):
            try:
                _,builder,ctx,assignments,W,T=prepare_parent(code,'cpu',ez_variant=vi)
                for seed in [11,12,13]:
                    rows,caps=seeded_rows(builder,ctx,assignments,W,seed)
                    fks,theta,dpi,ge,_,X,_,_=rows;new={k:[] for k in ['K','P','F']}
                    state=torch.get_rng_state()
                    for fk,th,(kw,rng) in zip(fks,theta,caps):
                        internal=th.clone();internal[:,:6]=0
                        with torch.no_grad():own=fk.apply(internal,X)
                        for name in new:
                            torch.set_rng_state(rng)
                            base=kabsch_init(**dict(kw,X_ref=own,diversity_scale=0. if name=='F' else 1.))
                            if name!='K':base=compensate_fk_pivot(base,fk,X,own,ge,kw['donor_pos_idx'])
                            new[name].append(torch.cat([base,th[:,6:]],-1))
                    torch.set_rng_state(state)
                    new['C']=theta
                    offsets={}
                    for name,th in new.items():
                        with torch.no_grad():coords=ComplexFK(fks).apply(th,X)
                        errors=torch.cat([(ge.donor_positions(coords)[:,di]-kw['donor_targets']).norm(dim=-1)
                            for di,(kw,_) in zip(dpi,caps)],1).max(1).values.numpy()
                        offsets[name]=dict(quantiles=quantiles(errors),n=len(errors),row_max=errors.tolist())
                    values.append(dict(variant=vi,seed=seed,offsets=offsets))
            except Exception as exc:failures.append(dict(variant=vi,error=repr(exc)))
        result=dict(code=code,n_variants=len(variants),records=values,failures=failures,status='failed' if failures else 'ok')
    except Exception as exc:result=dict(code=code,status='failed',reason=repr(exc),records=values)
    result['wall_s']=time.monotonic()-started
    Path(out,f'{code}.json').write_text(json.dumps(result)+'\n')
    return {k:v for k,v in result.items() if k!='records'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--only',default='');ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    assert 1<=a.workers<=3;a.out.mkdir(parents=True,exist_ok=True)
    codes=a.only.split(',') if a.only else population()[0];assert set(codes)<=set(population()[0])
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(measure,c,str(a.out)) for c in codes]
        for job in as_completed(jobs):
            try:result=job.result()
            except Exception as exc:result=dict(status='worker_failed',reason=repr(exc))
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
