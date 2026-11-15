"""Capture a failing rescore xTB invocation; no rescore results are written."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
OUT = ROOT / 'research/data/d2/db_d2/xtb_diagnostic'
OUT.mkdir(parents=True, exist_ok=True)
original_run = subprocess.run


def capture_run(args, *a, **kw):
    if not isinstance(args, (list, tuple)) or Path(str(args[0])).name != 'xtb':
        return original_run(args, *a, **kw)
    kwargs = dict(kw)
    kwargs['stdout'] = subprocess.PIPE
    kwargs['stderr'] = subprocess.PIPE
    result = original_run(args, *a, **kwargs)
    cwd = Path(kwargs['cwd'])
    if result.returncode != 0 or not (cwd / 'xtbopt.xyz').exists():
        (OUT / 'failure.json').write_text(json.dumps({
            'args': [str(x) for x in args], 'returncode': result.returncode,
            'stdout_tail': result.stdout[-12000:].decode('utf-8', errors='replace'),
            'stderr_tail': result.stderr[-12000:].decode('utf-8', errors='replace'),
        }, indent=2))
        for name in ('l.xyz', 'c.inp'):
            if (cwd / name).exists():
                shutil.copyfile(cwd / name, OUT / name)
    return result


subprocess.run = capture_run
from fit.pipeline import compare
from tools.jdev.rescore import rescore_60_pools

db, _, _ = compare.load_container(ROOT / 'research/data/d2/db_d2/constraints_2026-11-09.npz')
target = json.loads((ROOT / 'data/db/constraints_2026-11-01.json').read_text())
try:
    rescore_60_pools(db, target, pools_dir=ROOT / 'data/refit/frame_v1/pools',
                     frame_path=ROOT / 'data/refit/frame_v1/frame.json',
                     manifest_path=ROOT / 'data/refit/frame_v1/manifest.json', ref_dir=ROOT / 'data/ref')
except RuntimeError as exc:
    print(f'Diagnostic stopped: {exc}')
    if not (OUT / 'failure.json').exists():
        raise
else:
    print('No xTB failure reproduced; no scores saved.')
