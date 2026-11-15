"""C-2 paired bench invocation, with preparation/beam/row-step provenance."""
from __future__ import annotations
import argparse
import hashlib
from contextlib import nullcontext
import json
import subprocess
import time
from pathlib import Path
from .rings import population


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--only',default='')
    ap.add_argument('--arms',default='C,K,U,S1');ap.add_argument('--seeds',default='11,12,13')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--cpu-identity',action='store_true')
    ap.add_argument('--confirmation-after-audit',action='store_true',help='Only after owner/auditor clears the contract-freeze checkpoint')
    ap.add_argument('--baseline-engine',action='store_true')
    ap.add_argument('--combined-flags',default='{}')
    ap.add_argument('--combined-seed-budget',choices=['three_w16','single_w48'],default='three_w16')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    codes,population_meta=population()
    if a.only:
        requested=a.only.split(',');assert set(requested)<=set(codes);codes=requested
    definitions=dict(C={},K={'start_kabsch':True},S1={})
    definitions['P']={'start_kabsch':True,'start_kabsch_pivot':True}
    definitions['F']=dict(definitions['P'],start_kabsch_diversity=0.)
    definitions['M']=json.loads(a.combined_flags)
    seed_list=list(map(int,a.seeds.split(',')))
    selected_seeds=set(seed_list)
    if a.confirmation_after_audit:
        if seed_list!=[21,22,23] or a.only or a.baseline_engine or a.cpu_identity or a.device!='cuda':
            ap.error('C-2b confirmation requires seeds 21–23 and the full frozen population')
    elif not selected_seeds <= {11,12,13}:
        ap.error('C-2b development driver only permits seen seeds 11–13; confirmation requires audited freeze')
    names=a.arms.split(',');assert names[0]=='C'
    spec=[dict(arm=n,flags=definitions[n],out=str(a.out/n),
        seed_budget=a.combined_seed_budget if n=='M' else 'single_w48' if n=='S1' else 'three_w16') for n in names]
    if a.confirmation_after_audit:
        frozen_root=Path('research/campaigns/c2')
        frozen=json.loads((frozen_root/'C2B_FINALISTS.json').read_text())
        panel=json.loads((frozen_root/'C2_PANEL.json').read_text())
        if frozen['status']!='frozen':ap.error('C-2b contract is not frozen')
        actual=[{k:arm[k] for k in ['arm','flags','seed_budget']} for arm in spec]
        expected=[{k:arm[k] for k in ['arm','flags','seed_budget']} for arm in frozen['arms']]
        if actual!=expected or codes!=panel['codes']:
            ap.error('confirmation arms/settings/population differ from the frozen contract')
        from tools.jgen import inputs
        if (population_meta['exclusions_sha256']!=panel['exclusions_sha256'] or
            hashlib.sha256((inputs.FRAME_V1/'frame.json').read_bytes()).hexdigest()!=panel['frame_sha256']):
            ap.error('confirmation frame/exclusion hashes differ from frozen population')
        if hashlib.sha256(inputs.DB_PATH.read_bytes()).hexdigest()!=frozen['constraints_db_sha256']:
            ap.error('confirmation DB differs from the frozen production DB')
        sidecar=inputs.DB_PATH.with_name(inputs.DB_PATH.stem+'.meta.json')
        if hashlib.sha256(sidecar.read_bytes()).hexdigest()!=frozen['constraints_meta_sha256']:
            ap.error('confirmation DB sidecar differs from frozen production')
        from jacobik.core.env import MACE_MODEL_PATH
        if hashlib.sha256(Path(MACE_MODEL_PATH).read_bytes()).hexdigest()!=frozen['mace_model_sha256']:
            ap.error('confirmation MACE model differs from the frozen model')
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in frozen['source_sha256'].items()):
            ap.error('confirmation source differs from the contract-freeze source')
    path=a.out/'arms.json';path.write_text(json.dumps(spec,indent=2))
    from tools.jgen.bench import run as bench
    bench._setup_env({})
    import torch
    from jacobik.builder import engine as E
    if a.baseline_engine:
        source=subprocess.check_output(['git','show','b0134a2:src/jacobik/engine.py'],text=True)
        exec(compile(source,'<b0134a2-engine>','exec'),E.__dict__)
    from tools.jdev import multiseed
    from tools.runkit.resources import gpu_lock
    from tools.jgen.g3c0_distortion import start_digests
    if a.device=='cuda':
        (a.out/'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    times={};builders={};state={'arm':None};row_steps={};digests={};beam_times={}
    def sync():
        if a.device=='cuda':torch.cuda.synchronize()
    def add(code,key,value):times.setdefault(code,{}).setdefault(key,[]).append(value)
    original_prefetch=bench._prefetch_parent
    def prefetch(code,*args,**kw):
        t=time.monotonic();result=original_prefetch(code,*args,**kw)
        add(code,'prefetch',time.monotonic()-t)
        if result[1] is not None:builders[id(result[1])]=code
        return result
    bench._prefetch_parent=prefetch
    original_flags=bench._set_finish_flags
    def flags(value):
        state['arm']=next(n for n in names if definitions[n]==value)
        # C and S1 have identical flags; the explicit arm context below is
        # supplied by the bench's optional observation hook.
        return original_flags(value)
    bench._set_finish_flags=flags
    bench._c2_arm_observer=lambda name:state.update(arm=name)
    for name in ['prepare','seed_rows']:
        original=getattr(multiseed,name)
        def wrapped(builder,*args,_name=name,_original=original,**kw):
            code=builders[id(builder)];sync();t=time.monotonic()
            try:
                result=_original(builder,*args,**kw)
                if _name=='seed_rows':
                    key=state['arm'] or 'shared'
                    digests.setdefault(code,{})[key]=start_digests(result[1])
                return result
            finally:
                sync();add(code,(_name if state['arm'] is None else state['arm']+'_'+_name),time.monotonic()-t)
        setattr(multiseed,name,wrapped)
    original_beam=E.ComplexBuilder._beam_search_all_candidates
    def beam(builder,*args,**kw):
        result=original_beam(builder,*args,**kw)
        code=builders[id(builder)];key=state['arm'] or 'shared'
        beam_times.setdefault(code,{}).setdefault(key,[]).append(getattr(builder,'_c2_beam_wall_s',0.))
        return result
    E.ComplexBuilder._beam_search_all_candidates=beam
    # An explicit observer sets arm identity even when two arms share flags.
    original_joint=E.ComplexBuilder._run_joint_ik
    # Signature positions are checked against the method, never inferred from
    # a solve result: ctx,fks,scorer,ge,X,theta,dpi,ideal,sym,q3,W,T,candidate.
    def joint(builder,*args,**kw):
        code=builders[id(builder)];B=int(args[5][0].shape[0]);T=int(args[11])
        result=original_joint(builder,*args,**kw)
        row_steps.setdefault(code,{}).setdefault(state['arm'],[]).append(dict(B=B,T=T,row_steps=B*T))
        return result
    E.ComplexBuilder._run_joint_ik=joint
    argv=['--arms-spec',str(path),'--panel','named','--only',','.join(codes),'--device',a.device,
          '--seeds',a.seeds,'--xyz-compression','xz']
    if a.cpu_identity:argv.append('--cpu-identity')
    started=time.monotonic()
    try:
        with gpu_lock('CodexB') if a.device=='cuda' else nullcontext():
            torch.use_deterministic_algorithms(True)
            result=bench.main(argv)
    finally:
        (a.out/'timing.json').write_text(json.dumps(dict(parent_phases=times,row_steps=row_steps,
            initial_digests=digests,beam_times=beam_times,wall_s=time.monotonic()-started),indent=2))
    for code,phases in times.items():
        prep=sum(phases.get('prefetch',[]))+sum(phases.get('prepare',[]))
        for arm in spec:
            name=arm['arm'];seedtime=sum(phases.get(name+'_seed_rows',[]))
            if name in ('C','T'):seedtime+=sum(phases.get('seed_rows',[]))
            for p in (a.out/name/'records').glob(f'{code}_*.json'):
                r=json.loads(p.read_text());count=len(a.seeds.split(','))
                ownprep=sum(phases.get(name+'_prepare',[]))
                total=seedtime+r.get('wall_solve_s',0.)
                is_wide=arm['seed_budget']=='single_w48'
                e2e=prep+ownprep+total/count if is_wide else (prep+ownprep+total)/count
                r['c2_timing']=dict(shared_prepare_s=prep,seed_rows_s=seedtime,end_to_end_s=e2e,
                    beam_wall_s=sum(beam_times.get(code,{}).get(name,beam_times.get(code,{}).get('shared',[]))),
                    allocation='S1 full prepare per independent wide seed; C/ordinary prepare once per three-seed experiment')
                r['c2_identity']=dict(seed_budget=arm['seed_budget'],W=48 if is_wide else 16,
                    initial_digest=digests.get(code,{}).get(name,digests.get(code,{}).get('shared')),
                    row_steps=row_steps.get(code,{}).get(name,[]))
                p.write_text(json.dumps(r))
        if code in row_steps:
            steps=row_steps[code]
            nseeds=len(a.seeds.split(','))
            c=sum(x['row_steps'] for x in steps.get('C',[]))*3/nseeds
            for arm in spec:
                if arm['seed_budget']!='single_w48':continue
                wide=sum(x['row_steps'] for x in steps.get(arm['arm'],[]))/nseeds
                if c and wide:assert abs(wide/c-1)<=.1, (code,c,wide)
    if result:raise SystemExit(result)


if __name__=='__main__':main()
