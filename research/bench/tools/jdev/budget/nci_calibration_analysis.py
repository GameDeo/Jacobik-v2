"""Read-only numerical analysis of independent NCI probes; write research reports.

Per-probe best scalar fits are falsification diagnostics, never candidate weights.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO
from pools import setup
from nci_calibration import OUT,MONOMERS,CONFIG


def probes():
    import pandas as pd
    from rdkit import Chem
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from research.archive.code.interactions import nci
    KT=2.479
    rows=[];cases={};checks=[];collisions=[]
    fig,axes=plt.subplots(3,3,figsize=(14,12),layout='constrained');hbi=0
    for f in sorted(OUT.glob('hb_*.npz'))+sorted(OUT.glob('pp_*.npz'))+sorted(OUT.glob('chp_*.npz')):
        meta=json.loads(f.with_suffix('.json').read_text());kind=meta['type'];labels=meta['labels']
        with np.load(f) as z:
            x=z['X'];old=z['q6_old'];core=z['q6_core'];score={t:z['score_'+t] if 'score_'+t in z else np.zeros(len(x)) for t in nci.TYPES}
        monA=json.loads((OUT/f"monomer_{meta['a']}.json").read_text());monB=json.loads((OUT/f"monomer_{meta['b']}.json").read_text())
        nA=len(monA['symbols']);geomerr=0
        for sl,m in [(slice(None,nA),monA),(slice(nA,None),monB)]:
            ref=np.array(m['X']);d=np.linalg.norm(ref[:,None]-ref[None],axis=-1)
            geomerr=max(geomerr,float(np.max(abs(np.linalg.norm(x[:,sl,None]-x[:,None,sl],axis=-1)-d))))
        assert geomerr<1e-8,(f,geomerr)
        checks.append(dict(probe=f.stem,rigid_internal_distance_error=geomerr,table_coverage=[meta['q6_pairs_with_table'],meta['q6_pairs']]))
        models=dict(old=old,replacement_core=core,current_all=core-KT*sum(score.values()),
                    component=(core if kind=='hb' else old)-KT*score[kind])
        energies={name:np.array(json.loads(f.with_suffix('.'+ext+'.json').read_text())['interaction']) for name,ext in [('MACE64','mace64'),('GFN2','gfn2')]}
        scans=sorted(set(l['scan'] for l in labels));case={}
        for evaluator,energy in energies.items():
            shifted={k:np.zeros_like(v,dtype=float) for k,v in dict(physical=energy,**models).items()}
            for scan in scans:
                ids=np.array([i for i,l in enumerate(labels) if l['scan']==scan]);far=next(i for i in ids if labels[i]['far'])
                for key,value in dict(physical=energy,**models).items():shifted[key][ids]=value[ids]-value[far]
            # Common physical window; raw high-overlap geometries remain in CSV and selection checks.
            good=(shifted['physical']<50)&np.array([not l['far'] for l in labels])
            base=shifted['replacement_core'] if kind=='hb' else shifted['old']
            raw=KT*score[kind];raw=raw.copy()
            for scan in scans:
                ids=[i for i,l in enumerate(labels) if l['scan']==scan];far=next(i for i in ids if labels[i]['far'])
                raw[ids]-=raw[far]
            target=base-shifted['physical'];den=float(raw[good]@raw[good])
            oracle_scale=max(0.,float(raw[good]@target[good])/den) if den>1e-12 else None
            case[evaluator]=dict(points=int(good.sum()),best_probe_scalar=oracle_scale,models={})
            for model,prediction in models.items():
                p=shifted[model];slope_right=[];radial_regret=[];wrong_curvature=[]
                for scan in scans:
                    ids=sorted([i for i,l in enumerate(labels) if l['scan']==scan and not l['far']],key=lambda i:labels[i]['r'])
                    ep=energy[ids];pp=prediction[ids];dp=np.diff(p[ids]);de=np.diff(shifted['physical'][ids]);mask=abs(de)>.25
                    slope_right.extend((np.sign(dp[mask])==np.sign(de[mask])).tolist())
                    radial_regret.append(float(ep[np.argmin(pp)]-ep.min()))
                    if len(ids)>2:
                        # The distance grids are uniformly spaced within each scan.
                        ek=np.diff(energy[ids],n=2);sk=np.diff(prediction[ids],n=2)
                        valid=(ek>1e-4)&(energy[ids][1:-1]<50)
                        wrong_curvature.extend((sk[valid]<=0).tolist())
                i=int(np.argmin(prediction))
                case[evaluator]['models'][model]=dict(rmse=float(np.sqrt(np.mean((p[good]-shifted['physical'][good])**2))),
                    mae=float(np.mean(abs(p[good]-shifted['physical'][good]))),radial_regret=float(np.mean(radial_regret)),
                    slope_agreement=float(np.mean(slope_right)) if slope_right else None,
                    wrong_curvature_share=float(np.mean(wrong_curvature)) if wrong_curvature else None,
                    selected_physical=float(energy[i]),selected_regret=float(energy[i]-energy.min()))
            if oracle_scale is not None:
                calibrated=base-oracle_scale*raw
                case[evaluator]['best_scalar_rmse']=float(np.sqrt(np.mean((calibrated[good]-shifted['physical'][good])**2)))
            for i,l in enumerate(labels):
                rows.append(dict(probe=f.stem,type=kind,evaluator=evaluator,**l,physical=float(energy[i]),
                    physical_shifted=float(shifted['physical'][i]),q6_old=float(old[i]),q6_core=float(core[i]),
                    removed_Q6_reward=float(core[i]-old[i]),**{t:float(v[i]) for t,v in score.items()},
                    current_all=float(models['current_all'][i]),component=float(models['component'][i])))
        cases[f.stem]=case
        if kind=='hb':
            ids=[i for i,l in enumerate(labels) if l['angle']==180 and not l['far']]
            ax=axes.flat[hbi];hbi+=1;r=[labels[i]['r'] for i in ids]
            for name,color in [('MACE64','#243e63'),('GFN2','#758ba2')]:ax.plot(r,energies[name][ids],color=color,label=name)
            ax.plot(r,old[ids],label='Q6 old',color='#c08b38');ax.plot(r,models['component'][ids],label='core + NCI hb',color='#c35142')
            ax.set_title(meta['a']+' / '+meta['b']);ax.set_xlabel('H...A (A)');ax.set_ylabel('Interaction / surrogate energy (kJ/mol)');ax.set_ylim(bottom=max(-150,min(float(energies['MACE64'][ids].min()),float(energies['GFN2'][ids].min()))-10),top=50)
            if hbi==1:ax.legend(fontsize=7)
            for i in ids:
                if abs(labels[i]['r']-2.)<1e-8:
                    collisions.append(dict(probe=f.stem,primary_distance=2.,primary_angle=180.,
                        MACE64=float(energies['MACE64'][i]),GFN2=float(energies['GFN2'][i]),core=float(core[i]),
                        nci_hb=float(score['hb'][i]),old_q6=float(old[i])))
    fig.savefig(REPO/'data/refit/reports/n2_calibration_hbond_probes_2026-10-18.png',dpi=150);plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'probe_curves.csv',index=False)
    result=dict(cases=cases,checks=checks,matched_hbond_geometries=collisions,
        limitations='Rigid monomer interaction energies are defined by fragment charges; no arbitrary atomic attribution. Far-reference shifts remove offsets for curve errors, not unknown long-range physics. Per-probe scalar fits are diagnostics only, not validated or installed weights. Scans include all runtime contacts, not just the nominal named contact.')
    (OUT/'probe_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Matched H-bond geometry',json.dumps(collisions,indent=2),flush=True)
    for probe,case in cases.items():
        print(probe,json.dumps({e:dict(scale=c['best_probe_scalar'],old_rmse=c['models']['old']['rmse'],component_rmse=c['models']['component']['rmse'],oracle_rmse=c.get('best_scalar_rmse')) for e,c in case.items()}),flush=True)


if __name__=='__main__':setup();probes()
