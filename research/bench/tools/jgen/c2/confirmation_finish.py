"""Wait for the sole confirmation at long intervals, then postprocess once."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--wait-seconds',type=float,default=1200.);a=ap.parse_args()
    if a.wait_seconds<=0:ap.error('wait interval must be positive')
    marker=a.root/'postprocess_launch.json'
    with marker.open('x') as out:
        json.dump(dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),
            note='One postprocess chain; no structure generation or rerun'),out,indent=2)
    launch=json.loads((a.root/'launch.json').read_text())
    while not (a.root/'completion.json').exists():time.sleep(a.wait_seconds)
    completion=json.loads((a.root/'completion.json').read_text())
    state=dict(utc=datetime.now(timezone.utc).isoformat(),generation_returncode=completion['returncode'],steps=[])
    status=a.root/'postprocess_status.json'
    def save():status.write_text(json.dumps(state,indent=2)+'\n')
    save()
    if completion['returncode']!=0:
        state['status']='generation_failed_no_retry';save();raise SystemExit(completion['returncode'])
    frozen=Path('research/campaigns/c2')
    manifest=json.loads((frozen/'C2B_FINALISTS.json').read_text())
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in manifest['source_sha256'].items())
    commands=[('finalise',[sys.executable,'-m','tools.jgen.c2.finalise','--root',str(a.root)]),
              ('enrich',[sys.executable,'-m','tools.jgen.c2.enrich','--root',str(a.root)])]
    for arm,route in [('K','B'),('P','D'),('KU','B'),('S1','C')]:
        cmd=[sys.executable,'-m','tools.jgen.c2.c2b_analyse','--root',str(a.root),
            '--arm',arm,'--route',route,'--settings',str(frozen/'C2B_SETTINGS.json'),
            '--out',str(a.root/f'analysis_{arm}.json')]
        if arm=='P':cmd.append('--correctness-proof')
        commands.append(('analyse_'+arm,cmd))
    commands.append(('report',[sys.executable,'-m','tools.jgen.c2.confirmation_report','--root',str(a.root)]))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    for name,cmd in commands:
        print(datetime.now(timezone.utc).isoformat(),name,json.dumps(cmd),flush=True)
        state.update(status='running',current_step=name);save()
        with (a.root/f'postprocess_{name}.log').open('w') as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['steps'].append(dict(name=name,returncode=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat()))
        save()
        if result.returncode:
            state['status']='postprocess_failed_no_retry';save();raise SystemExit(result.returncode)
    state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    print('Postprocessing complete; no generation rerun',flush=True)


if __name__=='__main__':main()
