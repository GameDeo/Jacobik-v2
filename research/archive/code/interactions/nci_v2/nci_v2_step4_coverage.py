"""A3 no-label coverage check for frozen Step 4 native motion paths."""
from __future__ import annotations
import json, os
from collections import defaultdict
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np
import torch
from scipy.ndimage import label as connected_components
from research.archive.code.interactions.nci_v2.nci_v2 import (SurfaceArtifact, RT_KJ_MOL, enumerate_opportunities,
                                  features_torch, evaluate_joint_support, compute_tapers, h_epsilon)
from research.archive.code.interactions.nci_v2.nci_v2_extract import unique_opportunities, freeze
from research.archive.code.interactions.nci_v2.nci_v2_s2b_parity import safe_load_mol, safe_smiles_index
from research.archive.code.interactions.nci_v2.nci_v2_surface import latent, tensor_surface


def run(frame_path, spec_path, clean_root, surface_path, paths_dir, out_path):
    torch.set_num_threads(1)
    frame=json.loads(Path(frame_path).read_text()); spec=json.loads(Path(spec_path).read_text())
    surface=json.loads(Path(surface_path).read_text()); parents=frame['main_allocation']['splits']['calibration']
    parents={p['representative_code']:p for p in parents}; smiles=safe_smiles_index(Path(clean_root)/'tmQM'/'tmQM_y.csv')
    fam=surface['root_key'][0]; rootstr=str(freeze(surface['root_key']))
    artifact=SurfaceArtifact(fam,{rootstr:surface},spec,{})
    surfaces={f:None for f in ('bond_atom','bond_ring','ring_ring')}; surfaces[fam]=artifact
    inform=defaultdict(set); path_rows=defaultdict(list); traversals=defaultdict(lambda:defaultdict(set))
    cache={}
    mask=np.asarray(surface.get('ref_only_mask',[]),dtype=bool)
    components,ncomp=connected_components(mask)
    active_components=[components==i for i in range(1,ncomp+1) if np.any(components==i)]
    component_groups=[set() for _ in active_components]
    per_path=[]
    for path_file in sorted(Path(paths_dir).glob('*.json')):
        data=json.loads(path_file.read_text()); center=data.get('center',{}); code=center.get('code')
        if code not in parents: continue
        parent=parents[code]
        if code not in cache:
            mol=safe_load_mol(code,parent['representative_shard'],parent['dataset'],Path(clean_root),smiles)
            if mol is None: continue
            opps=unique_opportunities(enumerate_opportunities(mol,spec))
            indices=[i for i,o in enumerate(opps.opportunities[fam]) if str(freeze(o['z_o'])[0])==rootstr]
            cache[code]=(mol,opps,indices)
        mol,opps,indices=cache[code]
        for motion in data.get('motions',[]):
            if not motion.get('selected') or not motion.get('valid') or 'positions' not in motion: continue
            X=torch.as_tensor(motion['positions'],dtype=torch.float64)
            fb=features_torch(X,opps,spec)
            all_features=fb.features[fam]
            fval=np.zeros(len(X),dtype=float)
            if indices:
                x=all_features[:,indices,:].reshape(-1,all_features.shape[-1])
                root_occ=opps.opportunities[fam][indices[0]]
                ts=tensor_surface(surface,x)
                m=evaluate_joint_support(x,ts['support_basis_spec'],ts['support_a'])
                tin,tout=compute_tapers(x[:,0],*surface['radial_window'])
                v=tin*tout*m*h_epsilon(latent(x,surface,root_occ['z_o']))
                fval=v.reshape(len(X),len(indices)).sum(dim=1).detach().cpu().numpy()
            D=(-RT_KJ_MOL*(fval-fval[2])).tolist()
            max_abs=max(abs(float(v)) for v in D)
            group=str(parent['group_id'])
            if max_abs>0.1: inform[fam].add(group)
            row=dict(code=code,group_id=group,center=center.get('center'),motion_id=motion['motion_id'],
                     family=fam,root=rootstr,D_pk_kj_mol=D,max_abs_D_kj_mol=max_abs)
            per_path.append(row); path_rows[fam].append(row)
            if active_components:
                if indices:
                    x=all_features[:,indices,:].reshape(-1,all_features.shape[-1])
                    sps=surface['support_basis_spec']
                    tensor_spec={k:([torch.as_tensor(vv,dtype=x.dtype) for vv in v] if k=='knots_x' else torch.as_tensor(v,dtype=x.dtype)) for k,v in sps.items()}
                    for ci,cmask in enumerate(active_components):
                        reached=evaluate_joint_support(x,tensor_spec,torch.as_tensor(cmask,dtype=x.dtype))
                        if bool((reached>0).any()): component_groups[ci].add(group)
    informative=len(inform[fam])
    active=informative>0
    scale_mode='separate_family_scale' if informative>=10 else 'shared_scale' if active else 'inactive_no_supported_response'
    refonly=[]
    disabled=[]
    for ci,cmask in enumerate(active_components):
        n=len(component_groups[ci]); active_cell_count=int(cmask.sum())
        row=dict(component=ci,active_cells=active_cell_count,traversing_groups=n,passed=n>=5,action='retain' if n>=5 else 'set_a_j_zero')
        refonly.append(row)
        if n<5: disabled.append(cmask)
    frozen_surface=json.loads(json.dumps(surface))
    support=np.asarray(frozen_surface['support_a'],dtype=float)
    refmask=np.asarray(frozen_surface['ref_only_mask'],dtype=bool)
    disabled_mask=np.zeros_like(refmask)
    for x in disabled: disabled_mask|=x
    support[disabled_mask]=0.; refmask[disabled_mask]=False
    frozen_surface['support_a']=support.tolist(); frozen_surface['ref_only_mask']=refmask.tolist()
    out=dict(schema='nci_v2_step4_a3_coverage_v1',split='calibration',surface_family=fam,surface_root_key=surface['root_key'],
             active=active,scale_mode=scale_mode,informative_group_count=informative,
             family_activation={fam:dict(active=active,informative_groups=informative,scale=scale_mode)},
             unsupported_families={f:'no fitted primary surface in frozen Task 2 artifact' for f in ('bond_atom','bond_ring','ring_ring') if f!=fam},
             informative_definition='a calibration group is informative if any selected valid native path point has |D_pk,f| > 0.1 kJ/mol',
             ref_only_peak_neighbourhoods=refonly,disabled_ref_only_cell_count=int(disabled_mask.sum()),
             retained_ref_only_cell_count=int(refmask.sum()),ref_only_group_rule='retain each active neighbourhood only when at least five distinct group_id values traverse it',
             family_scale_rule='separate family scale at >=10 informative groups; otherwise shared scale',
             D_threshold_kj_mol=0.1,informative_groups={fam:sorted(inform[fam])},path_responses=per_path,
             frozen_surface=frozen_surface,paths_read=len(per_path),no_energy_labels=True)
    Path(out_path).parent.mkdir(parents=True,exist_ok=True);Path(out_path).write_text(json.dumps(out,indent=2)+'\n')
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--frame',required=True);p.add_argument('--spec',required=True);p.add_argument('--clean-root',required=True);p.add_argument('--surface',required=True);p.add_argument('--paths',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=run(a.frame,a.spec,a.clean_root,a.surface,a.paths,a.out);print(json.dumps({k:r[k] for k in ('surface_family','active','scale_mode','informative_group_count','disabled_ref_only_cell_count')}))
