"""One new C1 replicate per panel; separate processes preserve embed validity."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3f')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    arm=json.loads(Path('research/campaigns/g/G3F_MANIFEST.json').read_text())['arms'][0]
    for panel in ['cal','tail']:
        d=ROOT/'noise'/panel;d.mkdir(parents=True,exist_ok=True)
        spec=d/'arms.json';spec.write_text(json.dumps([dict(arm,out=str(d/'C1'))],indent=2))
        if (d/'job').exists():raise RuntimeError('refusing duplicate invocation')
        with (d/'run.log').open('w') as f:
            r=subprocess.run([sys.executable,'-u','-m','tools.jgen.g3d_gpu_job','--spec',str(spec),'--panel',panel,'--job-dir',str(d/'job')],stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise SystemExit(r.returncode)
        print(panel,'complete',flush=True)
if __name__=='__main__':main()
