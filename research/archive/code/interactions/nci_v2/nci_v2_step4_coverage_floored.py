"""A3 coverage accounting and frozen masks for the placebo-floored surfaces."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

import numpy as np
import torch
from scipy.ndimage import label as connected_components

from research.archive.code.interactions.nci_v2.nci_v2 import enumerate_opportunities, evaluate_joint_support, features_torch
from research.archive.code.interactions.nci_v2.nci_v2_extract import freeze, unique_opportunities
from research.archive.code.interactions.nci_v2.nci_v2_s2b_parity import safe_load_mol, safe_smiles_index


def _tensor_spec(spec):
    return {k:([torch.as_tensor(x,dtype=torch.float64) for x in v] if k=='knots_x'
               else torch.as_tensor(v,dtype=torch.float64)) for k,v in spec.items()}


def run(frame_path,spec_path,clean_root,surface_path,paths_dir,coverage_path,frozen_path):
    torch.set_num_threads(1)
    frame=json.loads(Path(frame_path).read_text());spec=json.loads(Path(spec_path).read_text())
    combined=json.loads(Path(surface_path).read_text())
    frozen=json.loads(json.dumps(combined))
    parents={p['representative_code']:p for p in frame['main_allocation']['splits']['calibration']}
    smiles=safe_smiles_index(Path(clean_root)/'tmQM'/'tmQM_y.csv')
    artifacts=frozen.get('surface_artifacts',frozen)
    root_info={};family_info=defaultdict(set);family_paths=defaultdict(int)
    for fam,artifact in artifacts.items():
        for rootstr,surface in artifact['surfaces'].items():
            mask=np.asarray(surface.get('ref_only_mask',[]),dtype=bool)
            labels,ncomp=connected_components(mask) if mask.size else (np.zeros_like(mask,dtype=np.int32),0)
            components=[]
            a=np.asarray(surface.get('support_a',[]),dtype=float)
            for ci in range(1,ncomp+1):
                cmask=(labels==ci)
                if not np.any(cmask) or not np.any(a[cmask]!=0):continue
                components.append(dict(mask=cmask,groups=set(),cells=int(cmask.sum())))
            root_info[(fam,rootstr)]=dict(surface=surface,components=components,
                informative_groups=set(),path_count=0,component_features=[])

    files_by_code=defaultdict(list)
    for path in sorted(Path(paths_dir).glob('*.json')):
        doc=json.loads(path.read_text());center=doc.get('center',{});code=center.get('code')
        if code in parents:files_by_code[code].append((path,doc))
    for code,(path_docs) in files_by_code.items():
        parent=parents[code];group=str(parent['group_id'])
        mol=safe_load_mol(code,parent['representative_shard'],parent['dataset'],Path(clean_root),smiles)
        if mol is None:continue
        opps=unique_opportunities(enumerate_opportunities(mol,spec))
        indices_by_root={}
        for (fam,rootstr),item in root_info.items():
            if not item['components']:continue
            indices_by_root[(fam,rootstr)]=[i for i,o in enumerate(opps.opportunities.get(fam,[]))
                if str(freeze(o['z_o'])[0])==rootstr]
        for path,doc in path_docs:
            for motion in doc.get('motions',[]):
                if not (motion.get('selected') and motion.get('valid') and 'positions' in motion):continue
                hits=motion.get('informative_roots_floored',[])
                hit_roots=[]
                for value in hits:
                    fam,rootstr=value.split('|',1)
                    key=(fam,rootstr)
                    if key not in root_info:continue
                    item=root_info[key];item['informative_groups'].add(group);item['path_count']+=1
                    family_info[fam].add(group);family_paths[fam]+=1;hit_roots.append(key)
                if not any(root_info[k]['components'] for k in hit_roots):continue
                X=torch.as_tensor(motion['positions'],dtype=torch.float64)
                fb=features_torch(X,opps,spec)
                for key in set(hit_roots):
                    item=root_info[key];fam,rootstr=key;indices=indices_by_root.get(key,[])
                    if not item['components'] or not indices or fam not in fb.features:continue
                    x=fb.features[fam][:,indices,:].reshape(-1,fb.features[fam].shape[-1])
                    item['component_features'].append(x.detach().cpu().numpy())

        # Aggregate all path features for this group, then count traversal once per
        # connected ref-only component instead of once per path.
        for (fam,rootstr),item in root_info.items():
            if not item['components'] or not item['component_features']:continue
            x=torch.as_tensor(np.concatenate(item['component_features'],axis=0),dtype=torch.float64)
            tspec=_tensor_spec(item['surface']['support_basis_spec'])
            for component in item['components']:
                reached=evaluate_joint_support(x,tspec,torch.as_tensor(component['mask'],dtype=x.dtype))
                if bool((reached>0).any()):component['groups'].add(group)
            item['component_features'].clear()

    root_rows=[];family_rows={}
    for (fam,rootstr),item in root_info.items():
        surface=item['surface'];components=item['components'];disabled=np.zeros_like(np.asarray(surface.get('support_a',[]),dtype=float),dtype=bool)
        component_rows=[]
        for idx,component in enumerate(components):
            n=len(component['groups']);passed=n>=5
            component_rows.append(dict(component=idx,active_cells=component['cells'],
                traversing_group_count=n,traversing_groups=sorted(component['groups']),passed=passed,
                action='retain' if passed else 'set_a_j_zero'))
            if not passed:disabled|=component['mask']
        fs=artifacts[fam]['surfaces'][rootstr]
        support=np.asarray(fs['support_a'],dtype=float);refmask=np.asarray(fs.get('ref_only_mask',np.zeros_like(support)),dtype=bool)
        support[disabled]=0.;refmask[disabled]=False
        fs['support_a']=support.tolist();fs['ref_only_mask']=refmask.tolist()
        fs['a3_coverage']={'informative_group_count':len(item['informative_groups']),
            'informative_groups':sorted(item['informative_groups']),'informative_path_incidences':item['path_count'],
            'active':bool(np.any(support>0)),'floored_by_placebo':surface.get('surface_status')=='below_placebo_floor'}
        root_rows.append(dict(family=fam,root_key=surface['root_key'],root=rootstr,
            informative_group_count=len(item['informative_groups']),informative_groups=sorted(item['informative_groups']),
            informative_path_incidences=item['path_count'],ref_only_components=component_rows,
            disabled_ref_only_cells=int(disabled.sum()),retained_ref_only_cells=int(refmask.sum()),
            active=bool(np.any(support>0)),floored_by_placebo=surface.get('surface_status')=='below_placebo_floor'))
    for fam,artifact in artifacts.items():
        groups=sorted(family_info[fam]);active=bool(groups);n=len(groups)
        mode='separate_family_scale' if n>=10 else ('shared_scale' if active else 'inactive_no_supported_response')
        family_rows[fam]=dict(active=active,informative_group_count=n,informative_groups=groups,
            scale_mode=mode,root_count=len(artifact['surfaces']),selected_path_incidences=int(family_paths[fam]),
            active_ref_only_components=sum(sum(c['passed'] for c in r['ref_only_components'])
              for r in root_rows if r['family']==fam))
        artifact.setdefault('metadata',{})['step4_family_coverage']={'active':active,'informative_group_count':n,'scale_mode':mode}
    frozen_path=Path(frozen_path);frozen_path.parent.mkdir(parents=True,exist_ok=True)
    frozen_path.write_text(json.dumps(frozen,sort_keys=True,separators=(',',':'))+'\n')
    frozen_sha=hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    result=dict(schema='nci_v2_step4_a3_floored_surface_v1',split='calibration',
      family_coverage=family_rows,root_coverage=root_rows,threshold_kj_mol=0.1,
      ref_only_group_threshold=5,family_scale_group_threshold=10,
      paths_directory=str(paths_dir),no_energy_labels=True,
      floored_surface_sha256=hashlib.sha256(Path(surface_path).read_bytes()).hexdigest(),
      frozen_surface_path=str(frozen_path),frozen_surface_sha256=frozen_sha)
    Path(coverage_path).parent.mkdir(parents=True,exist_ok=True)
    Path(coverage_path).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for x in ('frame','spec','clean-root','surface','paths','coverage','frozen'):p.add_argument('--'+x,required=True)
    a=p.parse_args();r=run(a.frame,a.spec,a.clean_root,a.surface,a.paths,a.coverage,a.frozen)
    print(json.dumps(r['family_coverage'],sort_keys=True))
