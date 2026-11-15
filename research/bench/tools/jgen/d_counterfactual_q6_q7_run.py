"""Rescore the combined D-B Q6 + Q7 arm after 1-4 energy weight was set to zero."""
from __future__ import annotations
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
WT=ROOT/'.claude/worktrees/jgen-d'
DATA=ROOT/'research/data/jgen/campaign/D0/f2'
OUT=ROOT/'research/data/jgen/campaign/D/counterfactual/q6_q7_all'
DB=ROOT/'research/data/jgen/campaign/D/q7_rescore_db/constraints_2026-11-01.json'
F2=DATA/'f2_terms.py'
OUT.mkdir(parents=True,exist_ok=True)
codes=[]
for p in sorted(DATA.glob('*.json')):
    try: rec=json.loads(p.read_text())
    except Exception: continue
    if rec.get('status')=='ok': codes.append(rec['code'])
env=os.environ.copy(); env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env['PYTHONPATH']='src:.:research/bench'; env['JACOBIK_DB']=str(DB)
env['JACOBIK_PROFILE_OVERRIDES']=json.dumps({'q6_physical_all':True,'q6_weight_14':0.0,
                                             'q7_bound_centre':True,'q7_shift_max':10.0})
def run(code):
    p=subprocess.run([sys.executable,str(F2),code,str(OUT)],cwd=WT,env=env,capture_output=True,text=True)
    f=OUT/f'{code}.json'
    if p.returncode or not f.exists(): return code,'runner_failed',p.stderr[-500:]
    r=json.loads(f.read_text()); return code,r.get('status'),r.get('reason','')
print(f'population={len(codes)} arm=q6_q7_all weight14=0 db={DB}',flush=True)
fails=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(run,c) for c in codes]
    for i,f in enumerate(as_completed(futures),1):
        r=f.result()
        if r[1]!='ok': fails.append(r)
        if i%20==0: print(f'completed {i}/{len(codes)} failures={len(fails)}',flush=True)
print(json.dumps({'parents':len(codes),'failures':fails},indent=2))
