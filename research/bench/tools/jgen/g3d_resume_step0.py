"""Resume amendment-3 step 0, retaining completed parents and component caches."""
import json,os,subprocess,sys,time
from pathlib import Path
from tools.jgen.g3d_diagnostics import rule_summary


def main():
    root=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3d')
    previous=[root/'step0_amend2.json',root/'step0_group2.json']
    parts=[json.loads(p.read_text()) for p in previous]
    completed={c:p for part in parts for c,p in part['parents'].items() if p.get('status')=='ok'}
    assert set(completed)=={'HEDJOP','NUWFAM','DAMVAQ'}
    jobs=[];started=time.monotonic()
    for name,parents in [('a3_FEKKAJ',['FEKKAJ']),('a3_QEZVEY',['QEZVEY']),('a3_other',['AJAQAD','USAPOU','NEPNEF'])]:
        path=root/f'{name}.json'
        # A completed job can be reused on restart without recomputing it.
        if path.exists() and all(c in json.loads(path.read_text())['parents'] for c in parents):
            jobs.append((None,path));continue
        log=open(root/f'{name}.log','a')
        proc=subprocess.Popen([sys.executable,'-u','-m','tools.jgen.g3d_diagnostics','--parents',*parents,
                               '--row-manifest','research/campaigns/g/G3D_ROWS_273f2ccbf892.json',
                               '--checkpoint-dir',str(root/'a3_components'),'--out',str(path)],stdout=log,stderr=subprocess.STDOUT)
        jobs.append((proc,path));print(name,proc.pid,flush=True)
    fatal=False
    while any(p is not None and p.poll() is None for p,_ in jobs):
        for _,path in jobs:
            try: part=json.loads(path.read_text())
            except (FileNotFoundError,json.JSONDecodeError):continue
            if part.get('stop_reason')=='nonexempt_embed_fallback':fatal=True
        if fatal:
            for proc,_ in jobs:
                if proc is not None and proc.poll() is None:proc.terminate()
            break
        time.sleep(1)
    for proc,_ in jobs:
        if proc is not None:proc.wait()
    result={**parts[0],'parents':completed,'failures':{}}
    for _,path in jobs:
        if path.exists():
            part=json.loads(path.read_text());result['parents'].update(part['parents']);result['failures'].update(part['failures'])
    result['rules']=rule_summary(result['parents'],False)
    result['amendment3']=dict(stage='50-step aid/source measurements only; cap deferred to 144-step continuation',
                             reused_parents=['HEDJOP','NUWFAM','DAMVAQ'],resumed_wall_s=time.monotonic()-started,
                             nonexempt_fallback_stop=fatal)
    (root/'step0_50_complete.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('complete',list(result['parents']),flush=True)

if __name__=='__main__':main()
