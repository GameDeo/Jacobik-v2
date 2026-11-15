"""Own two L4 jobs from one ref; budget-watch, fetch/count, then immediate down."""
import concurrent.futures,importlib.util,json,re,subprocess,threading,time,shutil
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3f')


def main():
    plan=json.loads(Path('research/campaigns/g/G3F_L4_PLAN.json').read_text())
    assert plan['estimated_units']<=10 and set(plan['partitions'])=={'jgB1','jgB2'}
    noise=json.loads(Path('research/campaigns/g/G3F_NOISE_RESULT.json').read_text());assert noise['N0_complete']
    assert json.loads((ROOT/'cpu_smoke/timing.json').read_text())['exit_code']==0
    s=importlib.util.spec_from_file_location('cj','research/gpuwork/colab/cj.py');cj=importlib.util.module_from_spec(s);s.loader.exec_module(cj)
    lock=threading.Lock();abort=threading.Event();owned=set();down=set();started=time.monotonic()
    def note(event,**kw):
        item=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=event,**kw)
        with lock:
            with (ROOT/'colab_journal.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            print(json.dumps(item),flush=True)
    inventory=cj.colab(['sessions']);(ROOT/'sessions_before.txt').write_text(inventory)
    # Session names must be absent or shown stopped; inspect the recorded inventory
    # before starting this controller. The initial no-active condition is required.
    balance,active,raw=cj.balance();assert active==0,'unexpected active VM; inspect ownership before launch'
    (ROOT/'credits_before.txt').write_text(raw);note('credits_before',balance=balance)
    ref=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    (ROOT/'colab_ref.txt').write_text(ref+'\n');cj.build_bundle('.',ref)
    # cj.push uses <local>.parts and removes it after upload. Two VMs need
    # distinct local filenames to prevent concurrent chunk-directory races.
    original_push=cj.push
    def isolated_push(name,local,remote):
        isolated=Path(local).with_name(name+'_'+Path(local).name)
        shutil.copyfile(local,isolated)
        try:return original_push(name,str(isolated),remote)
        finally:isolated.unlink(missing_ok=True)
    cj.push=isolated_push
    original_colab=cj.colab
    def tracked(args,**kw):
        if args and args[0]=='new':
            name=args[args.index('-s')+1]
            with lock:owned.add(name)
        # Keep failed transfer attempts bounded so teardown stays below hard cap.
        if args and args[0]=='download':kw.update(timeout=180,retries=1)
        return original_colab(args,**kw)
    cj.colab=tracked
    def worker(name,parents):
        local=ROOT/'l4'/name;local.mkdir(parents=True,exist_ok=True)
        remote='/content/out/g3f_'+name;tag='g3f_cal';marker=None
        try:
            note('up_start',name=name,parents=len(parents),commit=ref)
            cj.cmd_up(NS(name=name,gpu='L4',repo='.',ref=ref))
            note('ready',name=name)
            if not abort.is_set():
                cj.cmd_run(NS(name=name,tag=tag,env=['OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8'],
                    cmd=['python','-u','-m','tools.jgen.g3f_driver','--out',remote,'--only',','.join(parents),'--max-wall-s','3600']))
                while not abort.is_set():
                    log=cj.colab(['exec','-s',name],timeout=45,retries=1,check=False,
                        stdin=f"from pathlib import Path\np=Path('/content/jobs/{tag}.log')\nprint(p.read_text()[-4000:] if p.exists() else 'PENDING')\n") or ''
                    (local/'latest.log').write_text(log)
                    found=re.search(r'CJ_EXIT (\d+)',log)
                    if found:marker=int(found[1]);note('job_ended',name=name,exit_code=marker);break
                    abort.wait(60)
            else:note('aborted_before_run',name=name)
        except BaseException as exc:
            note('worker_error',name=name,error=f'{type(exc).__name__}: {exc}')
        finally:
            if name in owned:
                # Preserve and count even error/partial output before stopping.
                try:
                    cj.cmd_get(NS(name=name,remote=remote,local_dir=str(local)))
                    cj.cmd_get(NS(name=name,remote=f'/content/jobs/{tag}.log',local_dir=str(local)))
                except BaseException as exc:note('fetch_error',name=name,error=f'{type(exc).__name__}: {exc}')
                directory=local/Path(remote).name;records=list(directory.glob('*/records/*.json'));exports=list(directory.glob('*/xyz/**/*.xz'))
                counts={};ok=0
                for p in records:
                    try:
                        r=json.loads(p.read_text());counts[r['arm']]=counts.get(r['arm'],0)+1;ok+=r['status']=='ok'
                    except Exception as exc:note('partial_record',name=name,path=str(p),error=str(exc))
                note('fetched_counted',name=name,records=len(records),expected=6*len(parents),arms=counts,exports=len(exports),raw_ok_records=ok,exit_code=marker)
                for attempt in range(3):
                    try:
                        cj.cmd_down(NS(name=name));down.add(name);note('down_confirmed',name=name);break
                    except BaseException as exc:note('down_error',name=name,error=str(exc));time.sleep(10)
                if name not in down:raise RuntimeError('owned VM teardown failed: '+name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,n,ps) for n,ps in plan['partitions'].items()]
        while not all(j.done() for j in jobs):
            try:
                current,nactive,raw=cj.balance();spent=balance-current
                (ROOT/'credits_latest.txt').write_text(raw)
                if spent>=10 or time.monotonic()-started>=2.25*3600:
                    note('budget_abort',spent=spent,elapsed_s=time.monotonic()-started);abort.set()
            except BaseException as exc:note('credit_check_error',error=str(exc))
            if all(j.done() for j in jobs):break
            time.sleep(60)
        for job in jobs:job.result()
    final,active,raw=cj.balance();(ROOT/'credits_after.txt').write_text(raw)
    note('complete',units_spent=balance-final,balance_after=final,owned=sorted(owned),down=sorted(down),elapsed_s=time.monotonic()-started)
    assert owned==down and balance-final<=12
if __name__=='__main__':main()
