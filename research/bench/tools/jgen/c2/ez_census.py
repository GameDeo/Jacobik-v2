"""Complete Phase 0 placement over native E/Z variants, with pool atom maps."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time
from pathlib import Path
from .phase0 import prepare_parent,seeded_rows,placement,large_rings,aromatic_attachments,AROMATIC
from .rings import MAIN,population


def measure(code,out,flags):
    start=time.monotonic();values=[];failures=[];bundles={};original=None;aromatic=None
    from tools.jgen.bench.run import _setup_env
    _setup_env(flags)
    from tools.jgen.a4t.analyse import _load_parent
    try:
        original,probe,_=_load_parent(code);variants=probe._ez_combinations()
        selected_seed=None;selected_variant=0
        if code in AROMATIC:
            root=MAIN/'research/data/jgen/campaign'
            saved=json.loads((root/f'C2/rings/{code}.json').read_text());selected_seed=saved['best_seed']
            manifest=json.loads((root/f'A4b/xyz/{code}/{code}_control_s{selected_seed}_pool.json').read_text())
            selected_variant=int(manifest['rows'][saved['best_row']]['row_key'][0] or 0)
        for vi in range(len(variants)):
            try:
                _,builder,ctx,assignments,W,T=prepare_parent(code,'cpu',flags=flags,ez_variant=vi)
                for seed in (11,12,13):
                    rows,captures=seeded_rows(builder,ctx,assignments,W,seed)
                    values.append(dict(seed=seed,variant=vi,placement=placement(rows,captures,ctx)))
                    if seed==selected_seed:bundles[vi]=(ctx,rows)
            except Exception as exc:
                failures.append(dict(variant=vi,error=repr(exc)))
        if selected_seed is not None:
            ctx,rows=bundles[selected_variant]
            aromatic=aromatic_attachments(code,original,ctx,rows,pool_bundles=list(bundles.values()))
            aromatic['source_embed_variant']=selected_variant
        data=dict(code=code,status='ok' if not failures else 'failed',seeds=values,
                  large_rings=large_rings(original),aromatic=aromatic,W=W,T=T,flags=flags,
                  n_ez_variants=len(variants),variant_failures=failures)
        if failures:data['error']=repr(failures)
    except Exception as exc:
        import traceback
        data=dict(code=code,status='failed',error=repr(exc),traceback=traceback.format_exc())
    data['wall_s']=time.monotonic()-start
    Path(out,f'{code}.json').write_text(json.dumps(data))
    return dict(code=code,status=data['status'],variants=data.get('n_ez_variants'),wall_s=data['wall_s'],error=data.get('error'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--flags',default='{}');ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args();assert 1<=a.workers<=3
    scope=json.loads((MAIN/'research/data/jgen/campaign/C2/ez_scope.json').read_text())
    codes=sorted({r['code'] for r in scope if r.get('n_variants',1)>1}|set(AROMATIC))
    assert set(codes)<=set(population()[0]);a.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(measure,c,str(a.out),json.loads(a.flags)) for c in codes]
        for j in as_completed(jobs):print(json.dumps(j.result()),flush=True)


if __name__=='__main__':main()
