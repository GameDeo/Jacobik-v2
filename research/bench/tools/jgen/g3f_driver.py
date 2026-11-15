"""Paired portable G-3f bench driver; timing instrumentation only, no arithmetic edits."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--only',required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--seeds',default='11,12,13');ap.add_argument('--profile',action='store_true')
    ap.add_argument('--max-wall-s',type=int,default=0)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    from tools.jgen.bench import run as bench
    bench._setup_env({})
    import torch
    from tools.jdev import multiseed
    from jacobik.builder import engine as E
    from tools.jgen.embedding_validity import EmbeddingValidity
    from tools.runkit.resources import gpu_lock
    from contextlib import nullcontext
    if a.device=='cuda':
        (a.out/'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    arms=json.loads(Path('research/campaigns/g/G3F_MANIFEST.json').read_text())['arms']
    spec=a.out/'arms.json';spec.write_text(json.dumps([dict(x,out=str(a.out/x['arm'])) for x in arms],indent=2))
    times={};builders={};state={'code':None,'arm':None,'in_ez':False};stops=[]
    def sync():
        if a.device=='cuda':torch.cuda.synchronize()
    def add(code,key,value):times.setdefault(code,{}).setdefault(key,[]).append(value)
    old_prefetch=bench._prefetch_parent
    def prefetch(code,*args,**kw):
        t=time.perf_counter();result=old_prefetch(code,*args,**kw);add(code,'prefetch_s',time.perf_counter()-t)
        if result[1] is not None:builders[id(result[1])]=code
        return result
    bench._prefetch_parent=prefetch
    for name in ['prepare','seed_rows']:
        original=getattr(multiseed,name)
        def wrapped(builder,*args,_name=name,_original=original,**kw):
            code=builders[id(builder)];state['code']=code;sync();t=time.perf_counter()
            try:return _original(builder,*args,**kw)
            finally:
                sync();key=('per_arm_'+state['arm']+'_' if state['in_ez'] else '')+_name+'_s'
                add(code,key,time.perf_counter()-t)
        setattr(multiseed,name,wrapped)
    old_solve_seeds=multiseed.solve_seeds
    def solve_seeds(*args,**kw):
        previous=state['in_ez'];state['in_ez']=True
        try:return old_solve_seeds(*args,**kw)
        finally:state['in_ez']=previous
    multiseed.solve_seeds=solve_seeds
    old_flags=bench._set_finish_flags
    def flags(value):state['arm']='IA-k' if value.get('finish_phase') else 'C1';return old_flags(value)
    bench._set_finish_flags=flags
    # Enforce the carry-over hard stops, including failures raised from cache.
    old_check=EmbeddingValidity.check
    def check(item,parent):
        if parent in item.invalid and parent!='NEPNEF':
            (a.out/'STOP.json').write_text(json.dumps(dict(parent=parent,kind='embed_fallback')));raise SystemExit(86)
        return old_check(item,parent)
    EmbeddingValidity.check=check
    import logging
    class Guard(logging.Handler):
        def emit(self,r):
            message=r.getMessage()
            if 'step recording failed at step' in message or ('GFN2 embed refinement failed' in message and state['code']!='NEPNEF'):
                (a.out/'STOP.json').write_text(json.dumps(dict(parent=state['code'],message=message)));raise SystemExit(86)
    logging.getLogger().addHandler(Guard(logging.WARNING))
    profiles=[]
    if a.profile:
        from tools.jgen.g3f_profile import install
        install(profiles,state)
    t=time.perf_counter();exit_code=None
    if a.max_wall_s:
        import signal
        def timeout(*_):raise SystemExit(124)
        signal.signal(signal.SIGALRM,timeout);signal.alarm(a.max_wall_s)
    try:
        with gpu_lock('CodexB') if a.device=='cuda' else nullcontext():
            torch.use_deterministic_algorithms(True)
            exit_code=bench.main(['--arms-spec',str(spec),'--panel','cal','--only',a.only,'--device',a.device,'--seeds',a.seeds,'--xyz-compression','xz'])
    finally:
        (a.out/'timing.json').write_text(json.dumps(dict(parent_phases=times,profiles=profiles,wall_s=time.perf_counter()-t,exit_code=exit_code),indent=2))
    # Per-arm E/Z preparation is already separately timed by prepare/seed_rows.
    # Ordinary parents share those calls; allocate same full cost to each arm.
    for code,values in times.items():
        pre=sum(values.get('prefetch_s',[]));prep=sum(values.get('prepare_s',[]));seed=sum(values.get('seed_rows_s',[]))
        # E/Z native solve_seeds calls prepare once per variant per arm. Split
        # by captured arm tags below rather than a pooled arbitrary fraction.
        for arm in arms:
            arm_prep=prep+sum(values.get('per_arm_'+arm['arm']+'_prepare_s',[]))
            arm_seed=seed+sum(values.get('per_arm_'+arm['arm']+'_seed_rows_s',[]))
            for p in (a.out/arm['arm']/'records').glob(f'{code}_*.json'):
                r=json.loads(p.read_text())
                r['g3f_timing']=dict(shared_prefetch_s=pre,prepare_s=arm_prep,seed_rows_s=arm_seed,
                    end_to_end_s=pre+arm_prep+arm_seed+r.get('wall_solve_s',0.),partial_timing=r.get('status')!='ok',
                    allocation='full shared preparation to each arm; per-arm E/Z costs separated in timing ledger')
                if r.get('status')!='ok':
                    if code=='NEPNEF':r['embed_fallback']=True
                    p.write_text(json.dumps(r))
                    continue
                p.write_text(json.dumps(r))
    if exit_code:raise SystemExit(exit_code)
if __name__=='__main__':main()
