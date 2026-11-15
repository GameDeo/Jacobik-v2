"""One persistent process per arm; strict parent-first lockstep and isolated import state."""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time


def enrich_record(path):
    import numpy as np
    from tools.jgen.bench.pool_export import _read_frames
    from tools.jgen.a4t.metrics import shell_rmsd
    from tools.jgen.a4t.analyse import _load_parent
    from tools.jgen.diag.eta2_step4_analyze import faces_of, row_measures
    r=json.loads(path.read_text())
    if r['status']!='ok':return
    mp=Path(r['pool_manifest']);m=json.loads(mp.read_text())
    frames=list(_read_frames(mp.parent/m['rows'][0]['file']))
    X=np.asarray([f[2][m['gen_to_ref_idx']] for f in frames],float)
    sym=m['ref_symbols'];ref=np.asarray(m['ref_X']);mol,_,donors=_load_parent(r['code'])
    shells=[shell_rmsd(x,ref,sym,mol.metal_idx-1,donors) for x in X]
    r['c2_metrics']=dict(minimum_shell_rmsd_A=min(shells),shell_rmsd_A=shells)
    metal,faces=faces_of(sym,ref)
    measures=[row_measures(sym,x,metal,faces) for x in X]
    r['d3_geometry']=dict(edge_on_rows=sum(t is not None and t<.5 for g,t in measures),
                         geminal_under30_rows=sum(g<30 for g,t in measures),n_faces=len(faces))
    path.write_text(json.dumps(r))


class ArmRuntime:
    """DB/model/imports once; geometry and scorer necessarily rebuilt for each parent."""
    def __init__(self, plan, name, out):
        started=time.monotonic()
        self.plan,self.name,self.out=plan,name,out
        arm=plan['identities'][name]
        self.arm=arm
        os.environ['JACOBIK_DB']=arm['db']
        os.environ['JACOBIK_PROFILE_OVERRIDES']=json.dumps(arm['flags'])
        from tools.jgen.bench import run as bench
        bench._setup_env(arm['flags'])
        from jacobik.builder import engine as E
        from jacobik.core import profile as P
        from jacobik.core import energy
        from tools.jdev.budget import common
        from tools.jgen import embedding_validity
        from tools.jdev import multiseed
        from .arms import verify_runtime_identity,sha
        from jacobik.core.env import MACE_MODEL_PATH
        import torch
        if P.snapshot()['fingerprint']!=arm['fingerprint']:raise ValueError('runtime fingerprint mismatch')
        verify_runtime_identity(arm)
        if sha(MACE_MODEL_PATH)!=arm['mace_sha256']:raise ValueError('remote MACE identity mismatch')
        bundle=Path('COLAB_BUNDLE.json')
        if bundle.exists() and json.loads(bundle.read_text())['commit']!=arm['code_commit']:
            raise ValueError('remote code commit mismatch')
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        self.torch,self.bench,self.engine=torch,bench,E
        self.stats=common.stats_from_db(Path(arm['db']))
        self.mace=energy.MACEPolarEvaluator(model='polar-1-s',device='cuda',dtype='float64',
            model_path=str(MACE_MODEL_PATH),mode='energy-only',threads=1).load()
        self.model_loads=1;self.db_loads=1
        self.validity=embedding_validity.EmbeddingValidity(E)
        energy.MACEPolarEvaluator=lambda *args,**kwargs:self.mace
        def cached_stats(db):
            if Path(db).resolve()!=Path(arm['db']).resolve():raise ValueError('arm attempted another DB')
            return self.stats
        common.stats_from_db=cached_stats
        embedding_validity.EmbeddingValidity=lambda engine:self.validity
        gfn2=bench._assert_gfn2()
        bench._assert_gfn2=lambda:gfn2
        self.phases={};self.steps=[]
        for key in ('prepare','seed_rows'):
            original=getattr(multiseed,key)
            def timed(*args,_key=key,_original=original,**kwargs):
                torch.cuda.synchronize();t=time.monotonic()
                try:return _original(*args,**kwargs)
                finally:
                    torch.cuda.synchronize()
                    self.phases[_key]=self.phases.get(_key,0)+time.monotonic()-t
            setattr(multiseed,key,timed)
        prefetch=bench._prefetch_parent
        def timed_prefetch(*args,**kwargs):
            t=time.monotonic()
            try:return prefetch(*args,**kwargs)
            finally:self.phases['prefetch']=self.phases.get('prefetch',0)+time.monotonic()-t
        bench._prefetch_parent=timed_prefetch
        original_joint=E.ComplexBuilder._run_joint_ik
        def joint(builder,*args,**kwargs):
            B=int(args[5][0].shape[0]);T=int(args[11])
            self.steps.append(dict(B=B,T=T,row_steps=B*T))
            return original_joint(builder,*args,**kwargs)
        E.ComplexBuilder._run_joint_ik=joint
        self.startup_s=time.monotonic()-started

    def run_parent(self, code):
        self.phases.clear();self.steps.clear()
        self.torch.cuda.reset_peak_memory_stats()
        started=time.monotonic();root=logging.getLogger();handlers=list(root.handlers)
        original_refine=self.engine._gfn2_refine
        error=None
        try:
            self.bench.main(['--arm',self.name,'--flags',json.dumps(self.arm['flags']),
                '--out',str(self.out/self.name),'--panel','named','--only',code,'--device','cuda',
                '--seeds',','.join(map(str,self.plan['seeds'])),'--xyz-compression','xz'])
        except Exception as exc:
            error=repr(exc)
        finally:
            self.engine._gfn2_refine=original_refine
            for handler in list(root.handlers):
                if handler not in handlers:root.removeHandler(handler)
            ledger=self.out/'embedding_validity.json'
            if ledger.exists():ledger.replace(self.out/f'embedding_validity_{self.name}.json')
        elapsed=time.monotonic()-started
        for seed in self.plan['seeds']:
            path=self.out/self.name/'records'/f'{code}_{seed}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            r=json.loads(path.read_text()) if path.exists() else dict(status='failed',
                reason=error or 'worker completed without record',code=code,seed=seed)
            r.update(d3_identity=self.arm,d3_label=self.plan['label'],
                     d3_worker=dict(pid=os.getpid(),model_loads=self.model_loads,db_loads=self.db_loads))
            r['c2_timing']=dict(end_to_end_s=(sum(self.phases.values())+r.get('wall_solve_s',0))/len(self.plan['seeds']),
                phases=dict(self.phases),process_wall_s=elapsed,startup_s=self.startup_s,
                allocation='persistent arm: C-2b prefetch+prepare+seed_rows+solve_assembly divided across seeds; startup once')
            r['c2_identity']=dict(row_steps=list(self.steps),seed_budget='three_w16',W=16)
            path.write_text(json.dumps(r))
            try:enrich_record(path)
            except Exception as exc:
                r=json.loads(path.read_text());r.update(status='failed',reason='enrichment: '+repr(exc));path.write_text(json.dumps(r))
        peak=self.torch.cuda.max_memory_allocated()
        gc.collect();self.torch.cuda.empty_cache()
        return dict(code=code,pid=os.getpid(),parent_wall_s=time.monotonic()-started,
                    gpu_peak_allocated_bytes=peak,gpu_resident_bytes=self.torch.cuda.memory_allocated(),
                    model_loads=self.model_loads,db_loads=self.db_loads,error=error)


class ArmProcess:
    """Line protocol keeps model output on stderr and acknowledgements on stdout."""
    def __init__(self, command, log):
        self.log=log.open('w')
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                                      text=True,bufsize=1)
        self.ready=None

    def receive(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError(f'worker {self.process.pid} exited {self.process.poll()}')
        message=json.loads(line)
        if message.get('error') and not message.get('code'):raise RuntimeError(message['error'])
        return message

    def start(self):
        self.ready=self.receive()
        if not self.ready.get('ready'):raise RuntimeError('worker not ready')
        return self.ready

    def run_parent(self, code):
        if self.ready is None:raise RuntimeError('worker failed initialization')
        self.process.stdin.write(json.dumps(dict(code=code))+'\n');self.process.stdin.flush()
        result=self.receive()
        if result.get('code')!=code:raise RuntimeError('parent acknowledgement mismatch')
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"stop": true}\n');self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError,subprocess.TimeoutExpired):
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()


def drive_lockstep(workers, codes, failure, event):
    for code in codes:
        for name,worker in workers.items():
            event('parent_start',arm=name,code=code)
            try:
                result=worker.run_parent(code)
                if result.get('error'):failure(name,code,result['error'])
                event('parent_done',arm=name,**result)
            except Exception as exc:
                failure(name,code,repr(exc));event('parent_failed',arm=name,code=code,error=repr(exc))


def serve(plan, name, out):
    protocol=sys.stdout;sys.stdout=sys.stderr
    try:
        runtime=ArmRuntime(plan,name,out)
        protocol.write(json.dumps(dict(ready=True,pid=os.getpid(),startup_s=runtime.startup_s,
                                      model_loads=1,db_loads=1))+'\n');protocol.flush()
    except Exception as exc:
        protocol.write(json.dumps(dict(ready=False,error=repr(exc)))+'\n');protocol.flush();return
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        code=request['code']
        try:result=runtime.run_parent(code)
        except Exception as exc:result=dict(code=code,pid=os.getpid(),error=repr(exc))
        protocol.write(json.dumps(result)+'\n');protocol.flush()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--arm-worker')
    a=parser.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=True)
    from .arms import verify_fk_priors
    verify_fk_priors(plan['identities'])
    if a.arm_worker:return serve(plan,a.arm_worker,a.out)
    (a.out/'manifest.json').write_text(json.dumps(plan,indent=2))
    (a.out/'load.json').write_text(json.dumps(dict(load=os.getloadavg(),time=time.time())))
    (a.out/'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
    if 'L4' not in gpu:raise ValueError('D3 requires an L4 VM')
    def event(kind,**kwargs):
        with (a.out/'worker_events.jsonl').open('a') as f:
            f.write(json.dumps(dict(kind=kind,time=time.time(),**kwargs))+'\n')
    def failure(name,code,reason):
        for seed in plan['seeds']:
            path=a.out/name/'records'/f'{code}_{seed}.json';path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_text(json.dumps(dict(code=code,seed=seed,status='failed',reason=reason,
                d3_label=plan['label'],d3_identity=plan['identities'][name])))
    workers={}
    try:
        for name in plan['arms']:
            workers[name]=ArmProcess([sys.executable,'-u','-m','tools.jgen.d3.worker',
                '--plan',str(a.plan),'--out',str(a.out),'--arm-worker',name],a.out/f'arm_{name}.log')
            try:event('worker_ready',arm=name,**workers[name].start())
            except Exception as exc:event('worker_init_failed',arm=name,error=repr(exc))
        (a.out/'nvidia-smi-workers-ready.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        drive_lockstep(workers,plan['codes'],failure,event)
        (a.out/'nvidia-smi-workers-finished.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    finally:
        for worker in workers.values():worker.close()
    files={str(f.relative_to(a.out)):hashlib.sha256(f.read_bytes()).hexdigest()
           for f in a.out.rglob('*') if f.is_file() and f.name!='completion.json'}
    (a.out/'completion.json').write_text(json.dumps(dict(expected_records=len(plan['codes'])*len(plan['seeds'])*len(plan['arms']),
        actual_records=len(list(a.out.glob('*/records/*.json'))),files=files),indent=2))


if __name__=='__main__':main()
