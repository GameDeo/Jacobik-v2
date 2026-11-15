"""Strict A4b smoke gate; exits nonzero on any frozen criterion failure."""
import json, sys
from pathlib import Path
import numpy as np

def verify(root, expected=6):
    root=Path(root); records=list((root/'records').glob('*.json')); manifests=list((root/'xyz').rglob('*_pool.json'))
    if len(records)!=expected or len(manifests)!=expected: raise AssertionError(f'expected {expected} records/manifests, got {len(records)}/{len(manifests)}')
    by={(json.loads(p.read_text())['code'],int(json.loads(p.read_text())['seed'])):p for p in records}
    for mp in manifests:
        m=json.loads(mp.read_text()); key=(m['code'],int(m['seed'])); rec=json.loads(by[key].read_text())
        assert rec['status']=='ok' and rec['fingerprint']=='273f2ccbf892',key
        assert rec['n_rows']==m['n_rows']==len(m['rows']),key
        top=next(i for i,r in enumerate(m['rows']) if r['is_top1']); floor=next(i for i,r in enumerate(m['rows']) if r['is_floor'])
        en=np.asarray([r['mace_energy_kj'] for r in m['rows']],float); ref=float(rec['ref_energy_kj'])
        assert abs((en[top]-ref)-rec['top1_dE'])<=1e-6,(key,'top1',en[top]-ref,rec['top1_dE'])
        assert abs((en[floor]-ref)-rec['floor_dE'])<=1e-6,(key,'floor',en[floor]-ref,rec['floor_dE'])
    refs=json.loads((root/'relax'/'references.json').read_text())
    for code in {k[0] for k in by}:
        rec=next(json.loads(p.read_text()) for p in records if json.loads(p.read_text())['code']==code)
        rr=refs[code]; assert rr.get('status') in ('ok','not_converged') and rr.get('E_mace_kj') is not None
        assert abs(rr['E_unrel_mace_kj']-rec['ref_energy_kj'])<=1e-3,(code,rr['E_unrel_mace_kj'],rec['ref_energy_kj'])
    return dict(records=len(records),manifests=len(manifests),references=len(refs),passed=True)

if __name__=='__main__': print(json.dumps(verify(sys.argv[1])))
