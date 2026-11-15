"""Reapply the geometry-only informativeness filter to the floored surface."""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

import numpy as np
import torch

from research.archive.code.interactions.nci_v2.nci_v2 import (RT_KJ_MOL, compute_tapers, enumerate_opportunities,
                                  evaluate_joint_support, features_torch, h_epsilon)
from research.archive.code.interactions.nci_v2.nci_v2_extract import freeze, unique_opportunities
from research.archive.code.interactions.nci_v2.nci_v2_s2b_parity import safe_load_mol, safe_smiles_index
from research.archive.code.interactions.nci_v2.nci_v2_surface import latent, tensor_surface


def run(frame_path, spec_path, clean_root, original_surface_path, floored_surface_path,
        paths_dir, output_dir, report_path, threshold_kj_mol=0.1):
    torch.set_num_threads(1)
    frame=json.loads(Path(frame_path).read_text())
    spec=json.loads(Path(spec_path).read_text())
    original=json.loads(Path(original_surface_path).read_text())
    floored=json.loads(Path(floored_surface_path).read_text())
    parents={p['representative_code']:p for p in frame['main_allocation']['splits']['calibration']}
    smiles=safe_smiles_index(Path(clean_root)/'tmQM'/'tmQM_y.csv')
    artifacts0=original.get('surface_artifacts',original)
    artifacts1=floored.get('surface_artifacts',floored)
    root_data={};tensor_cache={};mol_cache={};summary=Counter();family_counts=defaultdict(Counter)
    for fam,artifact in artifacts0.items():
        for rootstr,surface0 in artifact['surfaces'].items():
            surface1=artifacts1[fam]['surfaces'][rootstr]
            root_data[(fam,rootstr)]={'original':surface0,'floored':surface1}

    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    per_path=[]
    for source_path in sorted(Path(paths_dir).glob('*.json')):
        doc=json.loads(source_path.read_text());code=doc.get('center',{}).get('code')
        parent=parents.get(code)
        if parent is None:
            continue
        group=str(parent['group_id'])
        if code not in mol_cache:
            mol_cache[code]=safe_load_mol(code,parent['representative_shard'],parent['dataset'],Path(clean_root),smiles)
        mol=mol_cache[code]
        if mol is None:
            summary['missing_reference']+=1
            continue
        opps=unique_opportunities(enumerate_opportunities(mol,spec))
        # Root-specific opportunity indices are independent of the motion path.
        indices_by_root={}
        for (fam,rootstr),item in root_data.items():
            indices_by_root[(fam,rootstr)]=[i for i,o in enumerate(opps.opportunities.get(fam,[]))
                                            if str(freeze(o['z_o'])[0])==rootstr]
        rows=[]
        for motion in doc.get('motions',[]):
            if not (motion.get('selected') and motion.get('valid') and 'positions' in motion):
                rows.append(motion);continue
            X=torch.as_tensor(motion['positions'],dtype=torch.float64)
            fb=features_torch(X,opps,spec)
            orig_hits=[];floor_hits=[];root_delta={}
            for (fam,rootstr),item in root_data.items():
                indices=indices_by_root[(fam,rootstr)]
                if not indices or fam not in fb.features:
                    continue
                allf=fb.features[fam];x=allf[:,indices,:].reshape(-1,allf.shape[-1])
                opp=opps.opportunities[fam][indices[0]]
                for variant in ('original','floored'):
                    key=(fam,rootstr,variant)
                    ts=tensor_cache.get(key)
                    if ts is None:
                        ts=tensor_surface(item[variant],x);tensor_cache[key]=ts
                    m=evaluate_joint_support(x,ts['support_basis_spec'],ts['support_a'])
                    tin,tout=compute_tapers(x[:,0],*item[variant]['radial_window'])
                    val=tin*tout*m*h_epsilon(latent(x,item[variant],opp['z_o']))
                    fval=val.reshape(len(X),len(indices)).sum(1).detach().cpu().numpy()
                    delta=(-RT_KJ_MOL*(fval-fval[2])).tolist()
                    root_delta[f'{fam}|{rootstr}|{variant}']=max(map(abs,delta),default=0.)
                    if max(map(abs,delta),default=0.)>threshold_kj_mol:
                        (orig_hits if variant=='original' else floor_hits).append(f'{fam}|{rootstr}')
            kept=bool(floor_hits)
            if orig_hits and not floor_hits:
                reason='only_floored_roots_informative'
                summary['dropped_only_floored_roots']+=1
            elif not orig_hits and not floor_hits:
                reason='no_informative_root'
                summary['dropped_without_original_signal']+=1
            else:
                reason='retained_informative'
                summary['retained_informative']+=1
            family_counts['path_status'][reason]+=1
            summary['selected_paths_before']+=1
            new=dict(motion)
            new.update(selected=kept,placebo_floor_filter_status=reason,
                       informative_roots_original=orig_hits,informative_roots_floored=floor_hits,
                       max_abs_delta_by_root_kj_mol=root_delta)
            rows.append(new)
            per_path.append(dict(code=code,group_id=group,center=doc['center'].get('center'),
                motion_id=motion['motion_id'],selected_before=True,selected_after=kept,status=reason,
                original_informative_roots=orig_hits,floored_informative_roots=floor_hits))
            if kept:
                for root_id in floor_hits:
                    family_counts['retained_by_family'][root_id.split('|',1)[0]]+=1
        out=dict(doc);out['motions']=rows
        (output_dir/source_path.name).write_text(json.dumps(out,separators=(',',':'))+'\n')

    selected_by_family={fam:int(count) for fam,count in family_counts['retained_by_family'].items()}
    result=dict(schema='nci_v2_step4_a2_refilter_after_placebo_floor_v1',split='calibration',
      threshold_kj_mol=threshold_kj_mol,original_surface_sha256=__import__('hashlib').sha256(Path(original_surface_path).read_bytes()).hexdigest(),
      floored_surface_sha256=__import__('hashlib').sha256(Path(floored_surface_path).read_bytes()).hexdigest(),
      source_paths=str(paths_dir),filtered_paths=str(output_dir),
      selected_paths_before=int(summary['selected_paths_before']),retained_paths=int(summary['retained_informative']),
      dropped_only_floored_roots=int(summary['dropped_only_floored_roots']),
      dropped_without_original_signal=int(summary['dropped_without_original_signal']),
      retained_path_incidences_by_family=selected_by_family,missing_reference_motion_files=int(summary['missing_reference']),
      path_rows=per_path,no_energy_labels=True)
    Path(report_path).parent.mkdir(parents=True,exist_ok=True)
    Path(report_path).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for x in ('frame','spec','clean-root','original-surface','floored-surface','paths','out','report'):
        p.add_argument('--'+x,required=True)
    a=p.parse_args()
    r=run(a.frame,a.spec,a.clean_root,a.original_surface,a.floored_surface,a.paths,a.out,a.report)
    print(json.dumps({k:r[k] for k in ('selected_paths_before','retained_paths','dropped_only_floored_roots','dropped_without_original_signal')}))
