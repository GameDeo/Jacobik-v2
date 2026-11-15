"""Fixed-row D-B Q7 and combined Q6+Q7 rescoring on stored A4b pools."""
from __future__ import annotations
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
WT = ROOT / '.claude/worktrees/jgen-d'
DATA = ROOT / 'research/data/jgen/campaign/D0/f2'
OUT = ROOT / 'research/data/jgen/campaign/D/counterfactual'
F2 = DATA / 'f2_terms.py'
DB = ROOT / 'research/data/jgen/campaign/D/q7_rescore_db/constraints_2026-11-01.json'
ARMS = {
    'q7_bound_centre': ({'q7_bound_centre': True, 'q7_shift_max': 10.0}, OUT / 'q7_bound_centre'),
    'q6_q7_all': ({'q6_physical_all': True, 'q6_weight_14': 0.0,
                   'q7_bound_centre': True, 'q7_shift_max': 10.0}, OUT / 'q6_q7_all'),
}

for _, dst in ARMS.values():
    dst.mkdir(parents=True, exist_ok=True)
codes = []
for path in sorted(DATA.glob('*.json')):
    try:
        rec = json.loads(path.read_text())
    except Exception:
        continue
    if rec.get('status') == 'ok':
        codes.append(rec['code'])

base_env = os.environ.copy()
base_env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
base_env['PYTHONPATH'] = 'src:.:research/bench'
base_env['JACOBIK_DB'] = str(DB)

def run(code: str, arm: str, flags: dict, dst: Path):
    env = base_env.copy()
    env['JACOBIK_PROFILE_OVERRIDES'] = json.dumps(flags)
    proc = subprocess.run([sys.executable, str(F2), code, str(dst)], cwd=WT, env=env,
                          capture_output=True, text=True)
    path = dst / f'{code}.json'
    if proc.returncode or not path.exists():
        return code, arm, 'runner_failed', proc.stderr[-500:]
    try:
        rec = json.loads(path.read_text())
    except Exception as exc:
        return code, arm, 'invalid_output', str(exc)
    return code, arm, rec.get('status'), rec.get('reason', '')

print(f'population={len(codes)} arms={list(ARMS)} db={DB}', flush=True)
failures = []
jobs = [(code, arm, flags, dst) for code in codes for arm, (flags, dst) in ARMS.items()]
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(run, *job) for job in jobs]
    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if result[2] != 'ok':
            failures.append(result)
        if i % 20 == 0:
            print(f'completed {i}/{len(jobs)} failures={len(failures)}', flush=True)
print(json.dumps({'parents': len(codes), 'runs': len(jobs), 'failures': failures}, indent=2))
