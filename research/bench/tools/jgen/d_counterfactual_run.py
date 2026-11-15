from __future__ import annotations
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

root = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
wt = root / '.claude/worktrees/jgen-d'
data = root / 'research/data/jgen/campaign/D0/f2'
out = root / 'research/data/jgen/campaign/D/counterfactual'
script = data / 'f2_terms.py'
out_off, out_on = out / 'control', out / 'q6_physical_all'
out_off.mkdir(parents=True, exist_ok=True); out_on.mkdir(parents=True, exist_ok=True)
codes=[]
for p in sorted(data.glob('*.json')):
    try:
        rec=json.loads(p.read_text())
    except Exception:
        continue
    if rec.get('status') == 'ok': codes.append(rec['code'])

env=os.environ.copy()
env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
env['PYTHONPATH']='src:.:research/bench'
python=sys.executable

def run(code, flag, dst):
    e=env.copy()
    e['JACOBIK_PROFILE_OVERRIDES']=json.dumps({'q6_physical_all':flag,'q6_weight_14':0.0})
    p=subprocess.run([python,str(script),code,str(dst)],cwd=wt,env=e,capture_output=True,text=True)
    target=dst/f'{code}.json'
    if p.returncode or not target.exists(): return code,flag,'runner_failed',p.stderr[-400:]
    try: rec=json.loads(target.read_text())
    except Exception as exc: return code,flag,'invalid_output',str(exc)
    return code,flag,rec.get('status'),rec.get('reason')

def pair(code):
    a=run(code,False,out_off); b=run(code,True,out_on); return a,b

print(f'population={len(codes)} output={out}',flush=True)
fail=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(pair,c) for c in codes]
    for i,f in enumerate(as_completed(futures),1):
        a,b=f.result()
        if a[2]!='ok' or b[2]!='ok': fail.append((a,b))
        if i%10==0: print(f'completed {i}/{len(codes)} failures={len(fail)}',flush=True)
print(json.dumps({'parents':len(codes),'failures':fail},indent=2))
