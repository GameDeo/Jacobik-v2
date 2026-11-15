"""Test NCI against existing GFN2 dimer energies, independently of dev-case picks.

No fitting, new energies, ligand embedding, or solver runs. Reuses the N1 probe
geometries/energies (which fitted Q6 repulsion, but never NCI).
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO,stats_from_db
from pools import setup
from nci_targeted_rescore import CANDIDATE


def run():
    import pandas as pd
    import torch
    from rdkit import Chem
    from scipy.stats import spearmanr
    from jacobik.core.types import Atom,Bond
    from research.archive.code.interactions import nci,q6pmf
    from gfn2_dimer_scan import MONOMERS
    from research.archive.code.refit.nci_refit import digest
    torch.set_num_threads(1)
    source=REPO/'data/refit/q6rep/dimers.json';data=json.loads(source.read_text());mons=data['monomers']
    tables=stats_from_db(CANDIDATE).nci_tables()
    groups={}
    for r in data['dimers']:
        a,b=r['a'],r['b']
        if mons[a]['name']!='benzene' and mons[b]['name']!='benzene':continue
        groups.setdefault((a,b),[]).append(r)
    rows=[]
    for (a,b),records in groups.items():
        atomlist=[];bonds=[];parents={};offset=0
        for i in (a,b):
            rd=Chem.AddHs(Chem.MolFromSmiles(MONOMERS[mons[i]['name']]))
            assert [at.GetSymbol() for at in rd.GetAtoms()]==mons[i]['sy']
            atomlist.extend(Atom(offset+j+1,at.GetSymbol(),np.zeros(3)) for j,at in enumerate(rd.GetAtoms()))
            bonds.extend(Bond(offset+bd.GetBeginAtomIdx()+1,offset+bd.GetEndAtomIdx()+1,bd.GetBondTypeAsDouble()) for bd in rd.GetBonds())
            for at in rd.GetAtoms():
                if at.GetAtomicNum()==1:parents[offset+at.GetIdx()+1]=offset+at.GetNeighbors()[0].GetIdx()+1
            offset+=rd.GetNumAtoms()
        mol=SimpleNamespace(atoms=atomlist,bonds=bonds,metal_idx=-1,coord_sites=[])
        rt=nci.build_runtime(mol,tables,'cpu') or {};cands=nci.candidates(mol)
        X=np.array([r['XA']+r['XB'] for r in records]);xt=torch.tensor(X,dtype=torch.float32)
        vals={};extra={}
        for t,e in rt.items():
            cs=[c for c in cands[t] if any(k in tables for k in c[-1])]
            r,angle=nci.features(t,xt,e['idx']);v=nci._bilinear(e['T'],e['row'],r,angle,e['grid']).numpy()
            vals[t]=v.sum(1)
            extra[t+'_positive_contacts']=(v>.1).sum(1)
            if t=='chp':
                half=v.copy();buckets={}
                for j,c in enumerate(cs):
                    h,ring=c[:2];p=parents[h];center=X[:,np.array(ring)-1].mean(1)
                    direction=X[:,p-1]-X[:,h-1];toward=center-X[:,h-1]
                    cosine=np.sum(direction*toward,axis=1)/np.linalg.norm(direction,axis=1)/np.linalg.norm(toward,axis=1)
                    half[:,j]=np.where(cosine<=0,v[:,j],np.minimum(v[:,j],0))
                    buckets.setdefault((p,tuple(ring)),[]).append(j)
                vals['chp_halfspace']=half.sum(1)
                vals['chp_siblingmax']=sum(np.minimum(v[:,ix],0).sum(1)+np.maximum(v[:,ix],0).max(1) for ix in buckets.values())
        nA=len(mons[a]['sy']);syA=mons[a]['sy'];syB=mons[b]['sy']
        pairparams=np.array([q6pmf.rep_pair_params(u,v) for u in syA for v in syB])
        distances=np.linalg.norm(X[:,:nA,None]-X[:,None,nA:],axis=-1).reshape(len(X),-1)
        rep=q6pmf.rep_energy(torch.tensor(distances,dtype=torch.float64),*[torch.tensor(pairparams[:,j]) for j in range(4)]).sum(1).numpy()*q6pmf.RT_KJ
        zero=np.zeros(len(X));pp=vals.get('pp',zero);chp=vals.get('chp',zero)
        for j,r in enumerate(records):
            row=dict(pair=mons[a]['name']+' / '+mons[b]['name'],orient=r['orient'],scale=r['s'],physical=r['E_int'],rep=rep[j],
                pp=float(pp[j]),chp=float(chp[j]),nci=rep[j]-q6pmf.RT_KJ*(pp[j]+chp[j]),
                pp_only=rep[j]-q6pmf.RT_KJ*pp[j],chp_only=rep[j]-q6pmf.RT_KJ*chp[j],
                halfspace=rep[j]-q6pmf.RT_KJ*(pp[j]+vals.get('chp_halfspace',zero)[j]),
                siblingmax=rep[j]-q6pmf.RT_KJ*(pp[j]+vals.get('chp_siblingmax',zero)[j]),
                **{k:int(v[j]) for k,v in extra.items()})
            rows.append(row)
        print(mons[a]['name'],mons[b]['name'],len(records),flush=True)
    df=pd.DataFrame(rows).fillna(0);assert np.isfinite(df.select_dtypes('number')).all().all()
    summary={};radial=[]
    models=['rep','nci','pp_only','chp_only','halfspace','siblingmax']
    for pair,g in df.groupby('pair'):
        summary[pair]={}
        for model in models:
            chosen=g.loc[g[model].idxmin()];physical_window=g[g.physical<50]
            summary[pair][model]=dict(selected_physical=float(chosen.physical),regret=float(chosen.physical-g.physical.min()),
                spearman=float(spearmanr(physical_window[model],physical_window.physical).statistic),
                selected_prediction=float(chosen[model]),selected_pp=float(chosen.pp),selected_chp=float(chosen.chp),
                pp_contacts=int(chosen.get('pp_positive_contacts',0)),chp_contacts=int(chosen.get('chp_positive_contacts',0)))
        for orient,gg in g.groupby('orient'):
            gg=gg.sort_values('scale');physical_slope=np.diff(gg.physical.values);keep=abs(physical_slope)>.25
            for model in models:
                chosen=gg.loc[gg[model].idxmin()];slope=np.diff(gg[model].values)
                radial.append(dict(pair=pair,orient=int(orient),model=model,regret=float(chosen.physical-gg.physical.min()),
                    slope_agreement=float(np.mean((np.sign(slope[keep])==np.sign(physical_slope[keep])))) if keep.any() else None))
    out=REPO/'data/refit/reports/n2_dimer_audit_2026-10-18'
    df.to_csv(out.with_suffix('.csv'),index=False);pd.DataFrame(radial).to_csv(out.with_name(out.name+'_radial').with_suffix('.csv'),index=False)
    radial_summary=pd.DataFrame(radial).groupby('model')[['regret','slope_agreement']].mean().to_dict('index')
    out.with_suffix('.json').write_text(json.dumps(dict(source_sha256=digest(source),candidate_db_sha256=digest(CANDIDATE),
        points=len(df),pairs=len(groups),summary=summary,radial_summary=radial_summary,
        scope='Existing neutral GFN2 probe dimers with benzene, all available orientations/distances. Independent of NCI fitting and dev cases, but these energies originally fitted repulsion; not an independent Q6 validation. No H-bond probes or cap conclusions; no fits. RT converts raw NCI nats according to architecture, not a newly fitted weight.'),indent=2)+'\n')
    print(json.dumps(dict(points=len(df),radial_summary=radial_summary,primary={k:v for k,v in summary.items() if k in ('benzene / benzene','methane / benzene')}),indent=2),flush=True)


if __name__=='__main__':setup();run()
