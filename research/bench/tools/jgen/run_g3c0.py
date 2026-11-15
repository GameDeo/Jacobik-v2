"""Frozen clean24 G-3c0 runner, using the established shared-start bench."""
import argparse
import json
from pathlib import Path

ARMS = {'C1': ({}, 250), 'P2': ({'polish': 'adam_small'}, 200),
        'R1': ({'polish': 'irprop'}, 200), 'R2plus': ({'optimizer': 'irprop'}, 250),
        'R2minus': ({'optimizer': 'irprop_minus'}, 250), 'A0': ({'optimizer': 'adam_score'}, 250)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--parents', help='Optional frozen-panel subset for profiling/sharding')
    args = ap.parse_args()
    contract = json.loads(Path('research/campaigns/g/evaluation_contract_g3c0.json').read_text())
    parents = args.parents.split(',') if args.parents else contract['parents']
    assert parents and len(set(parents)) == len(parents) and set(parents) <= set(contract['parents'])
    from tools.jgen.bench.run import main as run
    from tools.runkit import resources
    spec = [dict(arm=name, flags=flags, n_ik_steps=steps, n_conformers=16, distortion=True,
                 out=str(Path(args.out)/name), xyz_dir=str(Path(args.out)/name/'xyz'))
            for name, (flags, steps) in ARMS.items()]
    with resources.gpu_lock('CodexB', timeout=0):
        return run(['--panel', 'cal', '--only', ','.join(parents), '--seeds', '11,12,13',
                    '--arms-spec', json.dumps(spec)])


if __name__ == '__main__':
    raise SystemExit(main())
