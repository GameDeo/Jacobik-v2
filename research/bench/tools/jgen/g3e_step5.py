"""Serial cal then tail in one GPU process; reuse only completed paired parents."""
import contextlib,json,shutil,sys,time
from pathlib import Path
from tools.jgen.g3d_gpu_job import main as gpu_job


def main():
    root=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3e')
    plan=json.loads(Path('research/campaigns/g/G3E_STEP5_PLAN.json').read_text())
    assert plan['estimate_s']<=plan['limit_s']
    arms=json.loads(Path('research/campaigns/g/G3E_ARMS_AMEND5.json').read_text())['arms']
    started=time.monotonic();summary=dict(panels={})
    for panel in ['cal','tail']:
        directory=root/panel;completed=[];remaining=[]
        if list(directory.glob('job*/STOP.json')):raise RuntimeError('prior directive stop requires audit; refusing automatic retry')
        for code in plan['panels'][panel]:
            paths=[directory/a['arm']/'records'/f'{code}_{s}.json' for a in arms for s in [11,12,13]]
            try:
                rows=[json.loads(p.read_text()) for p in paths]
                valid=all(r['arm']==a['arm'] and r['code']==code and r['seed']==seed and
                          r['fingerprint']==a['fingerprint'] and r['flags']==a['flags'] and
                          r.get('n_ik_steps',a['n_ik_steps'])==250
                          for (a,seed),r in zip([(a,s) for a in arms for s in [11,12,13]],rows))
            except (FileNotFoundError,json.JSONDecodeError,KeyError):valid=False
            if valid:completed.append(code);continue
            if any(p.exists() for p in paths):
                archive=directory/'interrupted'/f'{code}_{int(time.time())}'
                for a in arms:
                    dest=archive/a['arm'];dest.mkdir(parents=True,exist_ok=True)
                    for s in [11,12,13]:
                        p=directory/a['arm']/'records'/f'{code}_{s}.json'
                        if p.exists():shutil.move(str(p),dest/p.name)
                    xyz=directory/a['arm']/'xyz'/code
                    if xyz.exists():shutil.move(str(xyz),dest/'xyz')
            remaining.append(code)
        summary['panels'][panel]=dict(reused_completed_parents=completed,remaining=remaining)
        (root/'step5_status.json').write_text(json.dumps(summary,indent=2))
        if not remaining:continue
        from tools.jgen.bench.run import _set_finish_flags
        _set_finish_flags({})  # clear prior panel's last arm before metadata/preparation
        attempt=1
        while (directory/f'job_attempt{attempt}').exists():attempt+=1
        argv=sys.argv;sys.argv=['g3d_gpu_job','--spec',str(directory/'arms.json'),'--panel',panel,'--only',','.join(remaining),'--job-dir',str(directory/f'job_attempt{attempt}')]
        try:
            with (directory/f'run_attempt{attempt}.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):gpu_job()
        finally:
            sys.argv=argv
            summary['wall_s']=time.monotonic()-started
            (root/'step5_status.json').write_text(json.dumps(summary,indent=2))
    summary['complete']=True;summary['wall_s']=time.monotonic()-started
    (root/'step5_status.json').write_text(json.dumps(summary,indent=2));print('cal and tail complete',flush=True)

if __name__=='__main__':main()
