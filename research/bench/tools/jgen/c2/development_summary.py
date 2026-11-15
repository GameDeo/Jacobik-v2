"""Compact audited development evidence; requires completed development runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .phase0 import quantiles
from .rings import population


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    codes=population()[0];per_parent={};pooled={k:[] for k in ['C','K','P','F']};failures=[]
    configs=0;changed=0;rows=0
    for code in codes:
        d=json.loads((a.root/'placement_panel'/f'{code}.json').read_text())
        configs+=d.get('n_variants',0)
        if d['status']!='ok':failures.append(dict(code=code,reason=d.get('reason',d.get('failures'))))
        if not d['records']:continue
        p={k:np.concatenate([r['offsets'][k]['row_max'] for r in d['records']]) for k in pooled}
        per_parent[code]={k:quantiles(v) for k,v in p.items()}
        changed+=int(np.any(np.abs(p['K']-p['P'])>.3))
        rows+=len(p['C'])
        for k,v in p.items():pooled[k].extend(v.tolist())
    placement=dict(n_parents=len(codes),n_valid=len(per_parent),n_native_configurations=configs,
        n_rows=rows,failures=failures,pooled={k:quantiles(v) for k,v in pooled.items()},
        parents_with_any_row_max_changed_over_03_A=changed,
        parents_median_over_03_A={k:sum(p[k]['0.5']>.3 for p in per_parent.values()) for k in pooled},
        per_parent=per_parent)
    runs={}
    for name,parents in [('pilot',['BABNAV','HOXJUZ','JEGPUJ','QEZVEY','XACMOE']),
                         ('followup',['JEGPUJ','SIVBUV','FELSAU','CAXTAX','UYOVUA','MIVXAR','LOPCAW'])]:
        root=a.root/name;arms=json.loads((root/'arms.json').read_text());records={};bad=[]
        for c in parents:
            records[c]={}
            for arm in arms:
                ds=[json.loads((root/arm['arm']/'records'/f'{c}_{s}.json').read_text()) for s in [11,12,13]]
                records[c][arm['arm']]=[{k:d.get(k) for k in ['seed','status','reason','top1_dE','floor_dE','n_rows','fingerprint','W','seed_budget','controller_stub']} for d in ds]
                bad.extend(dict(code=c,arm=arm['arm'],seed=d['seed'],reason=d.get('reason')) for d in ds if d['status']!='ok')
        aborted=(root/'controller_abort.json').exists()
        ledger=json.loads((root/('controller_abort.json' if aborted else 'timing.json')).read_text())
        observed_wall=ledger['elapsed_process_s'] if aborted else ledger['wall_s']
        runs[name]=dict(wall_s=observed_wall,aborted=aborted,controller_abort=ledger if aborted else None,
            wall_is_generator_metric=False,timing_context='owner-reported severe host contention; no intrinsic speed inference',
            arms=arms,records=records,failures=bad,
            n_records=len(parents)*len(arms)*3)
    inherited={}
    for arm in ['K','U','S1']:
        d=json.loads((a.root/f'inherited_{arm}.json').read_text())
        inherited[arm]={k:v for k,v in d.items() if k!='per_parent'}
    hashes=json.loads((a.root/'input_hashes.json').read_text())
    unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()}
    result=dict(status='development only; seeds 21–23 unopened',placement=placement,runs=runs,
        inherited=inherited,matched_rho=json.loads((a.root/'matched_rho.json').read_text())['summary'],
        T=json.loads((a.root/'T_reanalysis.json').read_text()),immutable_inputs_unchanged=unchanged,
        variant_count=8,units_spent=0,
        placement_reconstruction_check=json.loads((a.root/'placement_reconstruction_check.json').read_text()),
        default_identity=json.loads((a.root/'raw_identity_comparison.json').read_text()),
        timing_caveat='Owner reports heavy host/device contention; timings are contextual observations, not intrinsic speed measurements.')
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('Placement', {k:v for k,v in placement.items() if k!='per_parent'})
    print('Input hashes',unchanged)


if __name__=='__main__':main()
