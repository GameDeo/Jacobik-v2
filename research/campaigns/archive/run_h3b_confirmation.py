"""Run the frozen H-3b arms serially, in fresh processes, on the local GPU."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'research/data/jgen/campaign/H/h3b/final')
a=ap.parse_args();out=a.out.resolve()
if out.exists() and any(out.iterdir()):raise SystemExit(f'Refusing to overwrite existing confirmation: {out}')
out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
arms=json.loads((Path(__file__).with_name('H3B_ARMS.json')).read_text())
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
now=lambda:datetime.now(timezone.utc).isoformat()
state={'started':now(),'commit':head,'status':'running','phases':[]}
state_path=out/'state.json'
def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
save()
env=dict(os.environ,JACOBIK_GPU_REPLAY='1',OMP_NUM_THREADS='3',PYTHONUNBUFFERED='1',
         PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ['src','.','research/bench']))
env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
for panel in ['cal','tail']:
    for arm in arms:
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=head:
            state.update(status='stopped_source_changed',ended=now());save();raise SystemExit('Source commit changed')
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if busy:
            state.update(status='stopped_gpu_busy',ended=now(),unexpected_gpu_pids=busy);save();raise SystemExit('GPU already busy')
        target=out/panel/arm['arm'];log=out/'logs'/f"{panel}_{arm['arm']}.out"
        cmd=[sys.executable,'-m','tools.jgen','bench','run','--panel',panel,'--seeds','11,12,13',
             '--arm',arm['arm'],'--flags',json.dumps(arm['flags']),'--device','cuda','--out',str(target)]
        phase={'panel':panel,'arm':arm['arm'],'started':now(),'flags':arm['flags'],'log':str(log)}
        state['phases'].append(phase);save();print('START',panel,arm['arm'],flush=True)
        start=time.monotonic()
        with log.open('w') as f:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            phase['pid']=process.pid;save();rc=process.wait()
        phase.update(returncode=rc,wall_seconds=time.monotonic()-start,ended=now(),records=len(list((target/'records').glob('*.json'))))
        save();print('END',panel,arm['arm'],rc,phase['records'],round(phase['wall_seconds'],1),flush=True)
state.update(status='complete',ended=now());save();print('ALLDONE',flush=True)
