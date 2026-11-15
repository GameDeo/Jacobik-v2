"""Add unrelaxed shell and union-pool metrics to paired benchmark records."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .rings import population


def enrich(root, only=None):
    from tools.jgen.bench.pool_export import _read_frames
    from tools.jgen.a4t.metrics import shell_rmsd
    from tools.jgen.a4t.analyse import _load_parent
    from tools.runkit import geom
    codes,_=population()
    if only:assert set(only)<=set(codes);codes=only
    for code in codes:
        try:
            mol,_,donors=_load_parent(code)
            control_X=[];control_scores=[];control_paths=[]
            symbols=None;ref=None
            for directory in root.iterdir():
                if not directory.is_dir():continue
                for path in (directory/'records').glob(f'{code}_*.json'):
                    record=json.loads(path.read_text())
                    if record.get('status')!='ok':continue
                    mp=Path(record['pool_manifest'])
                    manifest=json.loads(mp.read_text())
                    frames=list(_read_frames(mp.parent/manifest['rows'][0]['file']))
                    X=np.asarray([f[2][manifest['gen_to_ref_idx']] for f in frames],float)
                    symbols=manifest['ref_symbols'];ref=np.asarray(manifest['ref_X'],float)
                    shells=[shell_rmsd(x,ref,symbols,mol.metal_idx-1,donors) for x in X]
                    record['c2_metrics']=dict(minimum_shell_rmsd_A=min(shells),shell_rmsd_A=shells)
                    path.write_text(json.dumps(record))
                    if directory.name=='C':
                        control_X.extend(X);control_scores.extend(record['scores']);control_paths.append(path)
            if control_X:
                heavy=np.asarray(symbols)!='H';X=np.asarray(control_X)
                perms,capped=geom.automorphisms(ref,symbols,mol.metal_idx-1)
                keep=[]
                for i in np.argsort(-np.asarray(control_scores),kind='stable'):
                    if not keep or not np.any(geom.sym_rmsd_to_ref(X[keep][:,heavy],X[i,heavy],perms)<=.5):
                        keep.append(int(i))
                for path in control_paths:
                    record=json.loads(path.read_text())
                    record['c2_metrics'].update(control_union_distinct=len(keep),union_symmetry_capped=capped)
                    path.write_text(json.dumps(record))
            print(code,'enriched',flush=True)
        except Exception as exc:
            (root/f'{code}_enrichment_failure.json').write_text(json.dumps(dict(error=repr(exc))))
            print(code,'enrichment failed',repr(exc),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--only',default='')
    ap.add_argument('--watch',action='store_true')
    ap.add_argument('--seeds',default='11,12,13',help='Expected seeds for watch readiness')
    ap.add_argument('--poll-seconds',type=float,default=15.)
    a=ap.parse_args()
    codes=a.only.split(',') if a.only else population()[0]
    if not a.watch:
        enrich(a.root,codes);return
    arms=json.loads((a.root/'arms.json').read_text())
    pending=set(codes)
    while pending:
        ready=[c for c in sorted(pending) if all(
            (a.root/arm['arm']/'records'/f'{c}_{s}.json').exists()
            for arm in arms for s in map(int,a.seeds.split(',')))]
        if ready:
            enrich(a.root,ready);pending.difference_update(ready)
            print('remaining',len(pending),flush=True)
        else:time.sleep(a.poll_seconds)


if __name__=='__main__':main()
