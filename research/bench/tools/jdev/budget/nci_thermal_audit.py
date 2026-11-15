"""NCI feature thermal compliance from the original independent Q1 Hessian cohort.

Reuses cached GFN2 Hessians; no new Hessians, optimization, density fit or default
change. The same Cartesian atom order is used throughout. Linear/clamped angular
features cannot be assigned a first-order width and are recorded as unavailable.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO
from pools import setup
from nci_training_calibration import write,digest

OUT=REPO/'data/refit/reports/n2_thermal_audit_2026-10-18'


def run():
    import pandas as pd
    import torch
    from research.archive.code.interactions import nci
    from jacobik.topology.graph import donor_indices_of
    from fit.pipeline.extract import benchmark_codes
    from refit.thermal_widths import _molecules,_pinv_hessian
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    sample=REPO/'data/refit/thermal/sample.json';codes=json.loads(sample.read_text())['codes']
    excluded=set(benchmark_codes());assert not(set(codes)&excluded)
    contract=dict(sample_sha256=digest(sample),codes=codes,selection='Reuse unchanged Q1 thermal-width training cohort; no selection by NCI scores or energy.',
        method='C = kT J H+ J^T, J from the native NCI feature implementation in crystal atom order. Original helper drops six rigid and nonpositive/nearzero Hessian modes.',
        thresholds=dict(hb=dict(r=3.2,angle_min=120),pp=dict(r=5.,angle_max=45),chp=dict(r=3.5,angle_max=45)),
        limitations='Cached Hessians inherit historical provenance (no per-file original geometry hash). Dimension, finite values and native input order checked. No angular floor inferred at clamped/linear geometries. Negative-mode sensitivity and independent-parent coverage reported. Widths include coordination/network constraints and do not uniquely price an added NCI term.')
    write(OUT/'contract.json',contract);rows=[];failures=[];hashes={};seen=[]
    for mol in _molecules(codes):
        code=mol.csd_code;seen.append(code);hp=REPO/f'data/refit/thermal/hess/{code}.npy'
        try:
            if not hp.exists():raise ValueError('missing cached Hessian')
            X=np.array([a.position for a in mol.atoms]);H=np.load(hp)
            assert H.shape==(X.size,X.size) and np.isfinite(H).all()
            hashes[code]=dict(hessian_sha256=digest(hp),symbols=[a.symbol for a in mol.atoms],reference=X.tolist())
            cands=nci.candidates(mol);pos={a.idx:i for i,a in enumerate(mol.atoms)};xt=torch.tensor(X[None],dtype=torch.float64)
            idx=nci.index_tensors(cands,pos,'cpu');chosen={};bound=set(donor_indices_of(mol))
            for typ,entry in idx.items():
                r,a=nci.features(typ,xt,entry);r=r[0].numpy();a=a[0].numpy()
                ok=(r<=3.2)&(a>=120) if typ=='hb' else (r<=5.)&(a<=45) if typ=='pp' else (r<=3.5)&(a<=45)
                chosen[typ]=np.flatnonzero(ok).tolist()
            if not any(chosen.values()):continue
            Hp,nneg=_pinv_hessian(H)
            for typ,ids in chosen.items():
                if not ids:continue
                entry=idx[typ]
                def feature(x):
                    r,a=nci.features(typ,x,entry)
                    return torch.stack((r[0,ids],a[0,ids]),dim=1)
                values=feature(xt).numpy()
                J=torch.autograd.functional.jacobian(feature,xt).numpy().reshape(len(ids),2,-1)
                covariance=2.479*np.einsum('nif,fg,njg->nij',J,Hp,J)
                for local,j in enumerate(ids):
                    C=covariance[local];sd=np.sqrt(np.diag(C).clip(min=0));angular_ok=bool(np.linalg.norm(J[local,1])>1e-8 and sd[1]>1e-8)
                    c=cands[typ][j]
                    if typ=='hb':context='one_bound_partner' if c[0] in bound or c[2] in bound else 'outer_sphere'
                    elif typ=='pp':context='bound_face' if set(c[0])<=bound or set(c[1])<=bound else 'unbound_faces'
                    else:context='bound_ring' if set(c[1])<=bound else 'unbound_ring'
                    rows.append(dict(code=code,type=typ,index=j,context=context,r=float(values[local,0]),angle=float(values[local,1]),
                        sigma_r=float(sd[0]),sigma_angle_deg=float(sd[1]) if angular_ok else None,
                        radial_angular_correlation=float(C[0,1]/sd.prod()) if angular_ok and sd[0]>1e-12 else None,
                        n_negative_modes=nneg,atoms=len(X)))
            print(code,'contacts',sum(map(len,chosen.values())),'negative modes',nneg,flush=True)
        except Exception as exc:
            failures.append(dict(code=code,error=f'{type(exc).__name__}: {exc}'));print('FAILED',failures[-1],flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'features.csv',index=False);summary={}
    for typ,g in df.groupby('type'):
        summary[typ]=dict(contacts=len(g),independent_parents=int(g.code.nunique()),angular_unavailable=int(g.sigma_angle_deg.isna().sum()),contexts=g.context.value_counts().to_dict())
        for column in ('sigma_r','sigma_angle_deg'):
            summary[typ][column]={}
            for name,sub in [('all',g),('no_negative_modes',g[g.n_negative_modes==0])]:
                values=sub[column].dropna()
                summary[typ][column][name]=dict(parents=int(sub.loc[sub[column].notna(),'code'].nunique()),median=float(values.median()) if len(values) else None,p10=float(values.quantile(.1)) if len(values) else None,p90=float(values.quantile(.9)) if len(values) else None)
    write(OUT/'summary.json',dict(summary=summary,failures=failures,codes_not_parsed=sorted(set(codes)-set(seen)),
        contract=contract,conclusion='Research widths only. Do not install a thermal floor from under-covered classes or transfer full-complex compliance directly into an extra pair attraction. New density fits must preserve the random-background correction, test second-order angular widths and validate total Q6+NCI depth/curvature.'))
    write(OUT/'input_hashes.json',hashes)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':setup();run()
