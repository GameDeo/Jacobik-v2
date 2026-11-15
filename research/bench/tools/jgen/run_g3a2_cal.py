"""Invoke the established batched runner with explicit G-3a-double-prime arms."""
import argparse
import json
from pathlib import Path

ARMS={'C0':({},200),'C1':({},250),'P2':({'polish':'adam_small'},200),
      'P2prime':({'finish_schedule':'late_cosine'},200),'P2a':({'finish_schedule':'zigzag'},200),
      'P2p':({'finish_schedule':'plateau'},200)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parents',required=True,help='comma separated named calibration parents')
    ap.add_argument('--arms',required=True,help='comma separated frozen arm names; _early suffix for timing companion')
    ap.add_argument('--out',required=True);ap.add_argument('--seeds',default='11,12,13');args=ap.parse_args()
    from jacobik.core import profile
    assert profile.snapshot()['fingerprint']=='6cc586e17ab1','cal must use merged Li-cap production'
    from tools.jgen import inputs
    from tools.jgen.bench.panels import select
    manifest=json.loads((inputs.FRAME_V1/'manifest.json').read_text())
    allowed={t['code'] for t in select(manifest,'cal')['units']};parents=args.parents.split(',')
    assert parents and set(parents)<=allowed,'calibration parents only'
    spec=[]
    for arm in args.arms.split(','):
        base=arm.removesuffix('_early');flags,steps=ARMS[base];flags=dict(flags)
        if arm.endswith('_early'):
            assert base in ('P2prime','P2a','P2p')
            flags['finish_early_stop']=True
        spec.append({'arm':arm,'flags':flags,'n_ik_steps':steps,'out':str(Path(args.out)/arm),'xyz_dir':str(Path(args.out)/arm/'xyz')})
    assert spec[0]['arm']=='C0','C0 initializes the baseline profile before runtime arm resets'
    from tools.runkit import resources
    from tools.jgen.bench.run import main as run
    with resources.gpu_lock('CodexB',timeout=900):
        return run(['--panel','cal','--only',args.parents,'--seeds',args.seeds,'--arms-spec',json.dumps(spec)])

if __name__=='__main__':raise SystemExit(main())
