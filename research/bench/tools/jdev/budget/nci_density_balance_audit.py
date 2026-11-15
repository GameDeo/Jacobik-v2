"""Sampling-balance sensitivity, separate from energy/width calibration.

Same independent 300-complex thermal cohort, energy-blind extraction. This is
not a full-corpus refit or deployable DB: weighted mass is matched only to keep
pseudocount strength fixed while comparing the distribution's shape.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO
from pools import setup
from nci_training_calibration import write,digest,contact_indices,scaffold

OUT=REPO/'data/refit/reports/n2_density_balance_2026-10-18'


def run():
    import torch
    from research.archive.code.interactions import nci
    from jacobik.builder.ligands import _build_ligand_groups
    from jacobik.topology.graph import donor_indices_of
    from refit.thermal_widths import _molecules
    from fit.pipeline.extract import benchmark_codes
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    source=REPO/'data/refit/thermal/sample.json';codes=json.loads(source.read_text())['codes']
    assert not(set(codes)&set(benchmark_codes()))
    observations=[];failures=[]
    for mol in _molecules(codes):
        try:
            groups=_build_ligand_groups(mol,mol.smiles);owner={}
            for g in groups:
                core=scaffold(g['canonical_smiles'])
                for i in g['xyz_indices_0based']:owner[mol.atoms[i].idx]=core
            cands=nci.candidates(mol);pos={a.idx:i for i,a in enumerate(mol.atoms)}
            idx=nci.index_tensors(cands,pos,'cpu');x=torch.tensor(np.array([a.position for a in mol.atoms])[None],dtype=torch.float64)
            bound=set(donor_indices_of(mol))
            for typ,entry in idx.items():
                r,a=nci.features(typ,x,entry);r=r[0].numpy();a=a[0].numpy()
                lo,dr,nr,al,da,na=nci._grid(typ)
                for j in np.flatnonzero((r>=lo)&(r<lo+dr*nr)):
                    c=cands[typ][j];atoms=contact_indices(typ,c)
                    if any(i not in owner for i in atoms):raise ValueError('unmapped contact scaffold')
                    family=repr(sorted(set(owner[i] for i in atoms)))
                    both_bound=bool(typ=='pp' and set(c[0])<=bound and set(c[1])<=bound)
                    observations.append(dict(code=mol.csd_code,type=typ,family=family,i=int(np.clip((r[j]-lo)/dr,0,nr-1)),j=int(np.clip((a[j]-al)/da,0,na-1)),both_bound=both_bound))
        except Exception as exc:failures.append(dict(code=mol.csd_code,error=f'{type(exc).__name__}: {exc}'))
    summary={};tables={}
    for typ in nci.TYPES:
        obs=[o for o in observations if o['type']==typ];_,_,nr,_,_,na=nci._grid(typ)
        per_parent=Counter(o['code'] for o in obs);per_group=Counter((o['code'],o['family']) for o in obs)
        parents_by_family=defaultdict(set)
        for o in obs:parents_by_family[o['family']].add(o['code'])
        weights=dict(raw=np.ones(len(obs)),parent_balanced=np.array([1/per_parent[o['code']] for o in obs]),
            scaffold_balanced=np.array([1/per_group[(o['code'],o['family'])]/len(parents_by_family[o['family']]) for o in obs]),
            exclude_cobound_faces=np.array([not o['both_bound'] for o in obs],dtype=float))
        modes={};raw_table=None
        for mode,w in weights.items():
            if w.sum()==0:continue
            w=w*len(obs)/w.sum();h=np.zeros((nr,na))
            np.add.at(h,([o['i'] for o in obs],[o['j'] for o in obs]),w)
            fitted=nci.fit_table(typ,h);tab=np.array(fitted['s']);tables[typ+'_'+mode]=tab
            if mode=='raw':raw_table=tab
            pm=defaultdict(float)
            for o,v in zip(obs,w):pm[o['code']]+=float(v)
            peak=np.unravel_index(np.argmax(tab),tab.shape);lo,dr,_,al,da,_=nci._grid(typ)
            modes[mode]=dict(effective_parents=float(sum(pm.values())**2/sum(v*v for v in pm.values())),
                peak_r=float(lo+dr*(peak[0]+.5)),peak_angle=float(al+da*(peak[1]+.5)),peak_score=float(tab[peak]),
                rms_table_change=float(np.sqrt(np.mean((tab-raw_table)**2))),max_table_change=float(abs(tab-raw_table).max()))
        summary[typ]=dict(contacts=len(obs),parents=len(per_parent),scaffold_families=len(parents_by_family),cobound_face_contacts=sum(o['both_bound'] for o in obs),
            raw_largest_parent_share=max(per_parent.values())/len(obs),raw_largest_family_share=max(Counter(o['family'] for o in obs).values())/len(obs),modes=modes)
    np.savez_compressed(OUT/'tables_diagnostic_only.npz',**tables)
    write(OUT/'summary.json',dict(sample_sha256=digest(source),summary=summary,failures=failures,
        scope='Energy-blind sensitivity on the unchanged, metal-row-balanced 300-complex Q1 Hessian cohort. All contacts within the original radial fitting range. Pooled tables only; no original n_min approval. Parent balance gives each complex equal mass; scaffold balance gives each participating scaffold set equal mass, equal parents within it. Mass restored to the same raw total solely to isolate shape from pseudocount shrinkage; ESS is reported separately. Not representative of the full 125622 structures and not a deployable fit. Co-bound-face removal is a mechanism diagnostic, not an adopted ownership rule.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':setup();run()
