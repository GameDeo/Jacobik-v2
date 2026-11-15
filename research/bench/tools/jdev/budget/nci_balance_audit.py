"""Full statistical score curvature on cached, identical MACE motions; no new solves."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO,stats_from_db,pool_path
from pools import setup,set_cases
from nci_targeted_rescore import CANDIDATE
from energy_diagnosis import motion_modes,add_contact_modes,moved


def run():
    import torch
    import pandas as pd
    from research.archive.code.interactions import nci
    from tools.jdev.cases import get_case
    from tools.jdev.multiseed import prepare
    torch.set_num_threads(1)
    stats=stats_from_db(CANDIDATE)
    names={c:(n,s) for s in ('dev32','n48') for c,n in set_cases(s)}
    selection=json.loads((REPO/'data/refit/reports/n2_targeted_rescore_2026-10-18.json').read_text())
    roots=[REPO/'data/refit/reports'/r for r in ('n2_energy_diagnosis_2026-10-18','n2_energy_diagnosis_extra_2026-10-18')]
    output=[];checks=[]
    for code in ('XEPKAF','PEWBOL','JEFJUZ','FETSUU','BULGAT','YIYXEJ'):
        name,split=names[code];case=get_case(name);builder=case.make_builder()
        ctx,assignments,W,T=prepare(builder,11,stats=stats,device='cpu',angle_bends=True)
        fks,th,dpi,ge,sc,X,ideal,cand=builder._beam_search_all_candidates(ctx,assignments,W)
        mol=ctx.mol;runtime=nci.build_runtime(mol,stats.nci_tables(),'cpu') or {}
        for seed in (11,12,13):
            selected=next(r for r in selection['pools'] if r['case']==code and r['seed']==seed)
            with np.load(pool_path('etkdg_u1_os',split,code,seed)) as z:
                for kind,key in [('before','base_index'),('after','cand_index')]:
                    label=f's{seed}_{kind}'
                    stem=next((r/code/label for r in roots if (r/code/label).with_suffix('.npz').exists()))
                    meta=json.loads(stem.with_suffix('.json').read_text());h=meta['provenance']['config']['step']
                    with np.load(stem.with_suffix('.npz')) as a:
                        x=a['X'];energies=a['scan_energy'];labels=a['mode_labels'].tolist()
                        modes=motion_modes(mol,x);add_contact_modes(modes,mol,x,runtime,meta['provenance']['config']['max_contact_modes'])
                        assert labels==[m['label'] for m in modes]
                        np.testing.assert_allclose(a['mode_vectors'],[m['v'] for m in modes],atol=1e-7)
                    coords=np.array([x]+[moved(x,m,s) for s in (h,-h,h/2,-h/2) for m in modes])
                    n=len(modes);rg=torch.full((len(coords),),int(z['asg'][selected[key]]),dtype=torch.long)
                    xt=torch.tensor(coords,dtype=torch.float32)
                    values={}
                    for enabled in (False,True):
                        nci.NCI_ON=enabled;sc._nci=runtime if enabled else None
                        with torch.no_grad():total,bd=sc.score(xt,row_groups=rg)
                        Z=float(sc._agg_norm())
                        values[enabled]={k:v.double().numpy()*Z for k,v in bd.items()}
                        values[enabled]['total']=total.double().numpy()*Z
                        if enabled:
                            for t,rt in runtime.items():
                                with torch.no_grad():v=nci.evaluate({t:rt},xt)
                                values[enabled]['NCI_'+t]=v.double().numpy()
                            np.testing.assert_allclose(values[enabled]['NCI'],sum(v for k,v in values[enabled].items() if k.startswith('NCI_')),atol=1e-4)
                    errors={k:abs(values[False][k][0]-float(z['terms_'+k][selected[key]])) for k in values[False] if 'terms_'+k in z}
                    assert max(errors.values())<.02,(code,seed,kind,errors)
                    checks.append(dict(case=code,seed=seed,pose=kind,central_term_errors=errors))
                    for j,m in enumerate(modes):
                        row=dict(case=code,seed=seed,pose=kind,mode=m['label'],kind=m['kind'])
                        for step_index,step in ((0,h),(2,h/2)):
                            tag='wide' if step_index==0 else 'fine'
                            plus=1+step_index*n+j;minus=plus+n
                            row['energy_'+tag]=(energies[plus]+energies[minus]-2*energies[0])/step**2/2.479
                            for enabled in (False,True):
                                prefix='on' if enabled else 'off'
                                for k,v in values[enabled].items():
                                    row[f'{prefix}_{k}_{tag}']=float(-(v[plus]+v[minus]-2*v[0])/step**2)
                                row[f'{prefix}_gradient_{tag}']=float((values[enabled]['total'][plus]-values[enabled]['total'][minus])/(2*step))
                        output.append(row)
        print(code,'curvature cached motions complete',flush=True)
    nci.NCI_ON=False
    out=REPO/'data/refit/reports/n2_balance_audit_2026-10-18'
    df=pd.DataFrame(output).fillna(0);df.to_csv(out.with_suffix('.csv'),index=False)
    summary={}
    for code,g in df.groupby('case'):
        summary[code]={}
        for mode in ('off','on'):
            good=(g.energy_fine>0)&(g[f'{mode}_total_fine']>0)
            stable=good & (abs(g.energy_wide-g.energy_fine)<.25*np.maximum(abs(g.energy_fine),1)) & (abs(g[f'{mode}_total_wide']-g[f'{mode}_total_fine'])<.25*np.maximum(abs(g[f'{mode}_total_fine']),1))
            summary[code][mode]=dict(directions=len(g),positive=int(good.sum()),stable=int(stable.sum()),
                median_ratio=float(np.median(g.loc[good,'energy_fine']/g.loc[good,f'{mode}_total_fine'])),
                stable_median_ratio=float(np.median(g.loc[stable,'energy_fine']/g.loc[stable,f'{mode}_total_fine'])),
                energy_up_score_unstable=int(((g.energy_fine>0)&(g[f'{mode}_total_fine']<=0)).sum()))
        terms=[c for c in df if c.startswith('on_') and c.endswith('_fine') and 'gradient' not in c]
        summary[code]['median_curvatures']={c:float(g[c].median()) for c in terms}
    out.with_suffix('.json').write_text(json.dumps(dict(summary=summary,checks=checks,
        scope='Six selected cases (CBTYCO E/Z omitted), identical cached MACE motions. Assignment-specific statistical terms reproduced at every stored pose. Bend/closure prior cannot be recovered from Cartesian coordinates and is excluded in both objectives, as in forces.py. Not a global calibration or weight fit.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':setup();run()
