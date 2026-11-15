"""Resolve panel parent charge/multiplicity through the bench geometry loader."""
import json
from pathlib import Path
from tools.jgen import inputs
from tools.jgen.bench import panels

def resolve(out):
    manifest=json.loads((inputs.FRAME_V1/'manifest.json').read_text())
    shards=inputs.frame_shards(); codes=[]
    for panel in ('cal','tail'):
        for u in panels.select(manifest,panel)['units']:
            if u['code'] not in codes: codes.append(u['code'])
    result={}
    for code in codes:
        try:
            shard,ds=shards[code]; mol=inputs.load_mol(code,shard,ds)
            if mol is None: raise ValueError('inputs.load_mol returned None')
            result[code]=dict(charge=int(mol.charge),mult=int(mol.spin_state),shard=shard,dataset=ds)
        except Exception as exc:
            result[code]=dict(status='failed',reason=f'{type(exc).__name__}: {exc}')
    Path(out).write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args(); resolve(a.out)
