"""Plan or launch registered D3 stages; unresolved D2 configuration cannot launch."""
import argparse
import importlib.util
import io
import tarfile
import json
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import time
from .arms import CONFIG, validate, sha, verify_fk_priors, verify_runtime_identity
from . import panels

REPO=Path(__file__).resolve().parents[5]
DATA=panels.MAIN/'research/data/jgen/campaign/D3'
ABLATION_ORDER=['far_field','physical_q6','q7_bound_centre','typing_os']


def load_cj():
    spec=importlib.util.spec_from_file_location('d3_cj',REPO/'research/gpuwork/colab/cj.py')
    cj=importlib.util.module_from_spec(spec);spec.loader.exec_module(cj)
    return cj


def estimate():
    root=panels.MAIN/'research/data/jgen/campaign/C2b/confirmation'
    walls=[]
    for code in panels.stage1():
        row=[]
        for name in ('C','P'):
            for seed in (21,22,23):
                r=json.loads((root/name/'records'/f'{code}_{seed}.json').read_text())
                w=r.get('c2_timing',{}).get('end_to_end_s')
                if w:row.append(w)
        if row:walls.append(sum(row)/len(row))
    import numpy as np
    seconds=float(np.mean(walls));sizes=dict(stage1=len(panels.stage1()),stage2=110,stage3=80)
    # One startup per arm/VM, rather than parent x arm. Use the smoke solely
    # for process/MACE/export overhead, not generator performance or scientific gates.
    startup=20.;overhead=0.;measurement=None
    smoke=DATA/'dryrun/smoke_amendment1/d3_output'
    if (smoke/'fetch_check.json').exists() and json.loads((smoke/'fetch_check.json').read_text())['passed']:
        records=[json.loads(p.read_text()) for p in smoke.glob('*/records/*.json')]
        if records and all(r.get('status')=='ok' for r in records):
            events=[json.loads(line) for line in (smoke/'worker_events.jsonl').read_text().splitlines()]
            startup=max(startup,max(e['startup_s'] for e in events if e['kind']=='worker_ready'))
            walls={(r['arm'],r['code']):r['c2_timing']['end_to_end_s'] for r in records}
            extras=[max(0,e['parent_wall_s']-walls[(e['arm'],e['code'])]) for e in events if e['kind']=='parent_done']
            overhead=float(np.mean(extras));measurement=str(smoke)
    return dict(basis='C-2b C/P mean per-seed generator wall on contested 2060; L4 speed factor 1.0 (unmeasured); '
                'persistent startup once per arm/VM, max(20 s, smoke maximum); setup +10 min per VM; '
                'smoke parent-wall minus generator-wall overhead extrapolated per parent/arm; L4 1.9 units/hour',
        observed_per_seed_s=seconds,l4_speed_factor=1.,l4_units_per_hour=1.9,
        startup_per_arm_s=startup,parent_arm_overhead_s=overhead,overhead_measurement=measurement,
        overhead_unmeasured=measurement is None,
        stages={s:dict(parents=n,arms=a,seeds=3,wall_hours=(n*a*(3*seconds+overhead)+a*startup+600)/3600,
                      units=(n*a*(3*seconds+overhead)+a*startup+600)/3600*1.9)
                for (s,n),a in zip(sizes.items(),(4,2,3))})


def remote_plan(plan):
    result=json.loads(json.dumps(plan));targets={}
    for identity in result.get('identities',{}).values():
        target='/content/jk/data/db/'+Path(identity['db']).name
        if target in targets and targets[target]!=identity['db_sha256']:
            raise ValueError('different arm DBs have the same remote basename')
        for path, digest in input_files(identity):
            destination='/content/jk/data/db/'+path.name
            if destination in targets and targets[destination]!=digest:
                raise ValueError('different arm inputs have the same remote basename')
            targets[destination]=digest
        identity['db']=target
    return result


def bundle_plan(bundle, plan, destination):
    """Embed the launch plan before provisioning; no small-plan Colab upload/exec."""
    payload=json.dumps(remote_plan(plan),indent=2).encode()
    with tarfile.open(bundle,'r:gz') as source, tarfile.open(destination,'w:gz',compresslevel=1) as target:
        for member in source:
            if member.name.lstrip('./')=='D3_PLAN.json':continue
            target.addfile(member,source.extractfile(member) if member.isfile() else None)
        info=tarfile.TarInfo('./D3_PLAN.json');info.size=len(payload)
        target.addfile(info,io.BytesIO(payload))
    return str(destination)


def verify_fetch(out):
    completion=json.loads((out/'completion.json').read_text())
    count=len(list(out.glob('*/records/*.json')))
    missing=[p for p,h in completion['files'].items() if not (out/p).is_file() or sha(out/p)!=h]
    ok=count==completion['expected_records']==completion['actual_records'] and not missing
    result=dict(passed=ok,expected_records=completion['expected_records'],actual_records=count,
                expected_files=len(completion['files']),missing_or_changed=missing,time=time.time())
    (out/'fetch_check.json').write_text(json.dumps(result,indent=2))
    if not ok:raise ValueError('fetched record/file count or hash mismatch')
    return result


def input_files(identity):
    db = Path(identity['db'])
    files = [(db, identity['db_sha256'])]
    if identity.get('meta_sha256') != 'embedded':  # a self-describing container needs no sidecar files
        files.append((db.with_suffix('.meta.json'), identity['meta_sha256']))
    if identity.get('flags', {}).get('q7_bound_centre'):
        if not identity.get('bends_sha256'):
            raise ValueError('q7_bound_centre requires bends_sha256')
        if identity['bends_sha256'] != 'embedded':
            files.append((db.with_suffix('.bends.json'), identity['bends_sha256']))
    return files


def ship_inputs(cj, vm, identities, out):
    """Ship each exact input (including bends) and verify its remote SHA before launch."""
    shipped = {}
    for identity in identities.values():
        for path, expected in input_files(identity):
            target = '/content/jk/data/db/' + path.name
            if target in shipped:
                if shipped[target] != expected:raise ValueError('remote input basename collision')
                continue
            if sha(path) != expected:raise ValueError('local input changed: ' + str(path))
            def remote_hash():
                return cj.remote_py(vm, "from pathlib import Path\nimport hashlib\np=Path(%r)\nprint(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else '')\n" % target, timeout=45).strip()
            if remote_hash() != expected:
                upload_dir = out/'upload_inputs';upload_dir.mkdir(exist_ok=True)
                staged = upload_dir/path.name;shutil.copyfile(path, staged)
                cj.push(vm, str(staged), target)
            if remote_hash() != expected:raise ValueError('remote input SHA256 mismatch: ' + target)
            shipped[target] = expected


def launch(plan, out, vm, max_minutes):
    """Always attempt fetch+count before down, including setup/job errors. Own VM only."""
    from types import SimpleNamespace
    if plan.get('identities'):
        verify_fk_priors(plan['identities'])
        for identity in plan['identities'].values():verify_runtime_identity(identity)
    cj=load_cj();out.mkdir(parents=True,exist_ok=False)
    planpath=out/'plan.json';planpath.write_text(json.dumps(plan,indent=2))
    state=dict(vm=vm,started=time.time(),events=[]);owned=False;alarm_handler=None
    def event(kind,**kw):
        state['events'].append(dict(kind=kind,time=time.time(),**kw))
        (out/'lifecycle.json').write_text(json.dumps(state,indent=2))
    try:
        # Record inventory before creation; never stop a pre-existing assignment.
        inventory=cj.colab(['sessions'],timeout=60,retries=1)
        if vm in inventory:raise ValueError(f'{vm} already exists; refusing ownership')
        event('inventory',text=inventory)
        # cj up can leave an assignment after setup errors. Mark ownership only after
        # cj's creation succeeds, before its upload/setup (same helper primitives).
        cj.guard_credits();bundle,commit=cj.build_bundle(str(REPO),'HEAD')
        bundle=bundle_plan(bundle,plan,out/'bundle_with_plan.tgz')
        try:
            cj.colab(['new','-s',vm,'--gpu','L4'],timeout=60,retries=1)
        except BaseException:
            status=cj.colab(['status','-s',vm],timeout=15,retries=1,check=False) or ''
            if vm in status and 'not found' not in status:
                owned=True;event('created_before_provisioning_error',status=status)
            raise
        owned=True;event('created',commit=commit)
        def budget_timeout(signum,frame):raise TimeoutError('VM time budget exhausted, including uploads')
        alarm_handler=signal.signal(signal.SIGALRM,budget_timeout)
        signal.setitimer(signal.ITIMER_REAL,max(1,max_minutes*60))
        cj.push(vm,bundle,'/content/bundle.tgz')
        cj.remote_py(vm,"open('/content/setup.sh','w').write(%r)\nimport subprocess\nsubprocess.Popen('nohup bash /content/setup.sh > /content/setup.log 2>&1 &',shell=True)\n" % cj.SETUP)
        deadline=time.time()+max_minutes*60
        while True:
            text=cj.remote_py(vm,"from pathlib import Path\np=Path('/content/setup.log')\nprint(p.read_text()[-4000:] if p.exists() else '')\n",timeout=45)
            if 'SETUP_DONE' in text:
                if 'FINGERPRINT '+cj.FINGERPRINT not in text:raise ValueError('setup fingerprint mismatch')
                event('ready',log=text);break
            if time.time()>deadline:raise TimeoutError('VM setup budget exhausted')
            time.sleep(15)
        remote=remote_plan(plan)
        ship_inputs(cj, vm, plan['identities'], out)
        (out/'remote_plan.json').write_text(json.dumps(remote,indent=2))
        cmd='python -m tools.jgen.d3.worker --plan /content/jk/D3_PLAN.json --out /content/d3_output'
        cj.cmd_run(SimpleNamespace(name=vm,tag='d3',env=['OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1'],cmd=[cmd]))
        while True:
            text=cj.remote_py(vm,"from pathlib import Path\np=Path('/content/jobs/d3.log')\nprint(p.read_text()[-4000:] if p.exists() else '')\n",timeout=45)
            event('poll',log=text)
            if 'CJ_EXIT ' in text:break
            if time.time()>deadline:raise TimeoutError('VM job budget exhausted')
            time.sleep(15)
    finally:
        if owned:
            signal.setitimer(signal.ITIMER_REAL,0)
            if alarm_handler is not None:signal.signal(signal.SIGALRM,alarm_handler)
            try:
                cj.cmd_get(SimpleNamespace(name=vm,remote='/content/d3_output',local_dir=str(out)))
                event('fetched',check=verify_fetch(out/'d3_output'))
            except BaseException as exc:
                event('fetch_failure',reason=repr(exc),actual_records=len(list(out.glob('d3_output/*/records/*.json'))),
                      expected_records=len(plan['codes'])*len(plan['arms'])*len(plan['seeds']))
            finally:
                cj.cmd_down(SimpleNamespace(name=vm));event('down')


def ablations(config, analysis):
    if analysis.get('stage')!='stage1' or analysis.get('label')!='D3' or analysis.get('reproduction'):
        raise ValueError('ablation requires a real failed stage 1')
    if analysis.get('candidate') in ('D','DS') or not any(not r['qualified'] for n,r in analysis['comparisons'].items() if n in ('D','DS')):
        raise ValueError('ablation enabled only by failed stage 1')
    out=[]
    for group in ABLATION_ORDER:
        change=config['ablation_groups'][group]
        if not change or 'flags' not in change or not change.get('db') or not change.get('fingerprint'):
            raise ValueError(f'unresolved D2 ablation recipe: {group} (DB changes require reproducible ablated DB)')
        out.append(dict(group=group,**change))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stage1','stage2','stage3','ablate','estimate'])
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--plan-only',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--curvature-dir',type=Path,help='directory containing <arm>/summary.json from CPU curvature step');p.add_argument('--stage1-analysis',type=Path);p.add_argument('--codes-file',type=Path);p.add_argument('--codes-sha256')
    a=p.parse_args()
    if a.stage=='estimate':
        result=estimate();print(json.dumps(result,indent=2))
        if a.out:a.out.write_text(json.dumps(result,indent=2))
        return
    config=json.loads(a.config.read_text())
    if a.smoke and a.stage!='stage1':p.error('smoke uses stage1 only')
    if not a.smoke and not a.plan_only and config['status']!='frozen':raise ValueError('contract/config not frozen by auditor')
    if not a.smoke and not a.plan_only:
        contract=Path('research/campaigns/D3_CONTRACT.md')
        if 'DRAFT' in contract.read_text().splitlines()[0]:raise ValueError('contract is still DRAFT')
        if subprocess.check_output(['git','status','--porcelain','--',str(a.config),str(contract),'research/bench/tools/jgen/d3'],text=True):
            raise ValueError('launch configuration/contract/code must be committed')
    if a.stage=='ablate':
        if not a.out or not a.stage1_analysis:p.error('ablate requires --out and --stage1-analysis')
        stage1_result=json.loads(panels.locked_path(a.stage1_analysis).read_text())
        recipes=ablations(config,stage1_result)
        failed=[n for n in ('D','DS') if n in stage1_result['comparisons'] and not stage1_result['comparisons'][n]['qualified']]
        # Each recipe carries its reproducible DB, full flag set and fingerprint.
        for recipe in recipes:
            cfg=json.loads(json.dumps(config));cfg['arms']={k:dict(v) for k,v in config['arms'].items()}
            cfg['arms']['D']={k:recipe[k] for k in ('db','flags','fingerprint')}
            cfg['arms']['DS']=dict(cfg['arms']['D'],flags=dict(recipe['flags'],**config['arms']['S']['flags']))
            cfg['arms']['DS']['fingerprint']=recipe.get('ds_fingerprint')
            identity=validate(cfg)
            plan=dict(stage='ablate',group=recipe['group'],codes=panels.stage1(),seeds=[11,12,13],arms=failed,
                      identities={n:identity[n] for n in failed},label='D3',control_source=str(a.stage1_analysis.parent))
            if a.plan_only:print(json.dumps(plan,indent=2))
            else:
                plan['curvature']={'C':stage1_result.get('curvature',{}).get('C',{})}
                from .analyse import curvature_valid
                if not curvature_valid(plan['curvature']['C']):raise ValueError('ablation requires Q7-inclusive control curvature')
                for n in failed:
                    dest=a.out/'curvature'/recipe['group']/n
                    subprocess.run([sys.executable,'-m','tools.jgen.d3.curvature','--db',identity[n]['db'],
                        '--flags',json.dumps(identity[n]['flags']),'--out',str(dest)],check=True)
                    plan['curvature'][n]=json.loads((dest/'summary.json').read_text())
                location=a.out/recipe['group'];launch(plan,location,'jgD1',360)
                for name in ('C','S'):
                    if (a.stage1_analysis.parent/name).exists():
                        (location/'d3_output'/name).symlink_to((a.stage1_analysis.parent/name).resolve(),target_is_directory=True)
        return
    if a.stage=='stage3':
        if not all((a.codes_file,a.codes_sha256,a.stage1_analysis,a.out)):p.error('stage3 requires codes path/hash, stage1 analysis, locked run output')
        if a.out.resolve().parent!= (panels.MAIN/'research/data/locked_runs').resolve():raise ValueError('stage3 output must be a direct locked_runs child')
        codes,candidate=panels.stage3(a.codes_file,a.codes_sha256,a.stage1_analysis,a.out)
        names=['C','Cprime',candidate]
    else:
        codes=panels.stage1() if a.stage=='stage1' else panels.stage2();names=['C','D','S','DS'] if a.stage=='stage1' else ['C','D']
    if a.smoke:codes=['CAXTAX','FELSAU','JEGPUJ'];names=['C','DS']
    if a.plan_only and not a.smoke:
        print(json.dumps(dict(stage=a.stage,codes=codes,seeds=[11,12,13],arms=names,status=config['status'],D2=config['arms']['D']),indent=2));return
    identity=validate(config,dryrun=a.smoke)
    plan=dict(stage=a.stage,codes=codes,seeds=[11] if a.smoke else [11,12,13],arms=names,
        identities={n:identity[n] for n in names},label='DRYRUN' if a.smoke else 'D3')
    if not a.smoke and a.stage=='stage1' and not a.curvature_dir:
        a.curvature_dir=DATA/'curvature'/identity['C']['code_commit']
        for n in ('C','D','DS'):
            dest=a.curvature_dir/n
            if not (dest/'summary.json').exists():
                subprocess.run([sys.executable,'-m','tools.jgen.d3.curvature','--db',identity[n]['db'],
                    '--flags',json.dumps(identity[n]['flags']),'--out',str(dest)],check=True)
    if a.curvature_dir:
        plan['curvature']={}
        for n in names:
            path=a.curvature_dir/n/'summary.json'
            if not path.exists():continue
            summary=json.loads(path.read_text())
            for key in ('db_sha256','meta_sha256','bends_sha256','fingerprint','fk_prior_settings'):
                if summary.get(key)!=identity[n].get(key):raise ValueError(f'{n} curvature identity mismatch: {key}')
            plan['curvature'][n]=summary
    if not a.smoke and a.stage=='stage1' and not all(n in plan.get('curvature',{}) for n in ('C','D','DS')):
        raise ValueError('stage1 requires registered C, D and DS curvature summaries')
    if not a.smoke and a.stage=='stage1':
        from .analyse import curvature_valid
        if not all(curvature_valid(plan['curvature'][n]) for n in ('C','D','DS')):
            raise ValueError('stage1 requires Scorer Q7 coverage and matching FK prior identities')
    if a.plan_only:print(json.dumps(plan,indent=2));return
    if a.stage=='stage3':
        previous=json.loads(a.stage1_analysis.read_text()).get('identities',{})
        for n in names:
            original=previous.get('C' if n=='Cprime' else n,{})
            if identity[n]!=dict(original,arm=n):raise ValueError(f'{n} identity changed since stage1')
        subprocess.run([sys.executable,'-m','tools.jdev.foldin_check'],check=True)
    out=a.out or DATA/('dryrun/smoke_amendment1' if a.smoke else a.stage)
    if a.stage=='stage3':
        # OWNER_GO lives in the run directory; launch artifacts in a fresh child.
        out=out/'execution'
    launch(plan,out,'jgD3s' if a.smoke else 'jgD1' if a.stage=='stage1' else 'jgD2',12 if a.smoke else 360)

if __name__=='__main__':main()
