"""All-conformer geometry-only pilot/recert incidence, with frozen thresholds."""
import argparse
from collections import Counter
from pathlib import Path
import re
import numpy as np
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget,health_metrics
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,Run,load_json,write_json
from research.archive.code.interactions.nci_v2.nci_v2_extract import graph_from_record
from research.archive.code.interactions.nci_v2.nci_v2_poolio import mapped_geometry,PoolMappingError


def run(policy_path,graph_path,out):
    budget(start=True);policy=load_json(policy_path);graphs=load_json(graph_path)
    pools={}
    for panel,relative in [('pilot','pilot_v1/pools'),('recert','recert_run_a/pools')]:
        pools[panel]=[p for p in sorted((MAIN/'data/refit/nci_v2'/relative).glob('*.npz'))
            if p.name.split('_')[0] in graphs and (bool(re.fullmatch(r'.+_s[0-9]+\.npz',p.name)) if panel=='pilot' else p.name.endswith('_control.npz'))]
    r=Run('phase2-health-incidence',policy_path,[graph_path,*sum(pools.values(),[])],
        dict(all_conformers=True,energy_fields_accessed=False,overlay_access=False),out)
    if r.resumed:return r.path
    report={}
    for panel,paths in pools.items():
        counts=Counter();records=[]
        for path in paths:
            budget();code=path.name.split('_')[0];mol=graph_from_record(graphs[code]);ref=np.asarray([a.position for a in mol.atoms])
            with np.load(path,allow_pickle=False) as z:
                if 'error' in z.files:records.append(dict(pool=path.name,status='solver_error'));counts['solver_error_pools']+=1;continue
                counts['pools']+=1;counts['raw_poses']+=len(z['X'])
                try:X,info=mapped_geometry(z,pool_path=path)
                except PoolMappingError as exc:
                    counts['excluded_unmapped_poses']+=len(z['X']);counts['excluded_unmapped_pools']+=1
                    records.append(dict(pool=path.name,status='excluded',reason=str(exc),poses=len(z['X'])));continue
                if not np.array_equal(z['ref_symbols'],[a.symbol for a in mol.atoms]) or not np.allclose(z['ref_X'],ref,atol=1e-7):raise ValueError('reference identity mismatch')
                finite=np.isfinite(X).all((1,2));indices=np.flatnonzero(finite);counts['nonfinite']+=int((~finite).sum())
                if not len(indices):continue
                stored=np.asarray(z['coord_intact'])[finite] if 'coord_intact' in z.files else None
                metrics,_,_,_=health_metrics(mol,ref,X[finite],policy,stored)
                flags=dict(stray_atom=metrics['stray_atom'],heavy_overlap=metrics['heavy_overlap'],
                    coord_loss=~metrics['coord_intact'],bad_covalent_bond=metrics['bad_bond'],
                    gross_overlap=metrics['gross_overlap'],unhealthy=~metrics['healthy'])
                if stored is not None:flags['stored_coord_false']=~stored.astype(bool)
                record=dict(pool=path.name,status='mapped',poses=len(X),finite_poses=len(indices),
                    **{k:int(v.sum()) for k,v in flags.items()},
                    flagged_pose_indices={k:indices[v].tolist() for k,v in flags.items()},
                    max_bond_relative=float(metrics['max_bond_relative'].max(initial=0)),
                    min_heavy_nonbonded_angstrom=float(metrics['min_nonbonded_heavy_distance'].min(initial=np.inf)))
                counts['mapped_finite_poses']+=len(indices)
                for k,v in flags.items():counts[k]+=int(v.sum());counts[k+'_pools']+=int(v.any())
                records.append(record)
        report[panel]=dict(counts=dict(counts),pools=records)
        print(panel,dict(counts),flush=True)
    write_json(r.path/'reports/incidence.json',report)
    r.finish(dict(policy_version=policy['version'],panels={p:v['counts'] for p,v in report.items()}))
    return r.path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--graphs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(run(a.spec,a.graphs,a.out))
