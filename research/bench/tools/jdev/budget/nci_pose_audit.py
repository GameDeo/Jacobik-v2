"""Classify PEWBOL's actual geometries and inspect XEPKAF ligand forces."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO,pool_path,pool_energy,stats_from_db
from pools import setup,set_cases
from nci_targeted_rescore import CANDIDATE


def angle(a,b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))


def run():
    import torch
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry
    from nci_census import census
    from research.archive.code.interactions import nci
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    from jacobik.topology.decompose import decompose_ligand
    torch.set_num_threads(1)
    names={c:(n,s) for s in ('dev32','n48') for c,n in set_cases(s)}
    selection=json.loads((REPO/'data/refit/reports/n2_targeted_rescore_2026-10-18.json').read_text())
    out=REPO/'data/refit/reports';report={}
    name,split=names['PEWBOL'];case=get_case(name)
    with starting_geometry(case,'full_oracle'):
        mol,_,_=case.make_builder()._build_shell(False,False)
    for l in mol.ligands:decompose_ligand(mol,l)
    symbols=[a.symbol for a in mol.atoms];pos={a.idx:i for i,a in enumerate(mol.atoms)};metal=pos[mol.metal_idx]
    ligs=[l for l in mol.ligands if any(mol.atoms[pos[a]].symbol=='S' for a in l.atom_indices)]
    ns=[next(pos[a] for a in l.atom_indices if mol.atoms[pos[a]].symbol=='N') for l in ligs]
    ss=[next(pos[a] for a in l.atom_indices if mol.atoms[pos[a]].symbol=='S') for l in ligs]
    def geom(x):
        return dict(N_M_N=angle(x[ns[0]]-x[metal],x[ns[1]]-x[metal]),
                    S_M_S=angle(x[ss[0]]-x[metal],x[ss[1]]-x[metal]),
                    N_S_vector_angle=angle(x[ss[0]]-x[ns[0]],x[ss[1]]-x[ns[1]]),
                    S_S=float(np.linalg.norm(x[ss[0]]-x[ss[1]])),hbond=census(mol,x)['hbond'])
    # _build_shell translates ligands apart; its positions are not a reference complex.
    with np.load(out/'n2_energy_diagnosis_2026-10-18/PEWBOL/reference.npz') as z:
        reference=z['X']
    report['PEWBOL']=dict(reference=geom(reference),pools=[])
    arrays={}
    arms=dict(stored=None,fresh_on='n2_fresh_solve_2026-10-18',fresh_off='n2_fresh_control_2026-10-18',
              isoref_on='n2_pewbol_isoref_on_2026-10-18',isoref_off='n2_pewbol_isoref_off_2026-10-18')
    for arm,directory in arms.items():
        for seed in (11,12,13):
            f=pool_path('etkdg_u1_os',split,'PEWBOL',seed) if arm=='stored' else out/directory/f'PEWBOL_s{seed}.npz'
            if not f.exists() or not f.with_suffix('.mace64.json').exists():continue
            with np.load(f) as z:
                energy=pool_energy(f,'mace64');measure=[geom(x) for x in z['X']]
                picks={'pick':int(z['score'].argmax()),'floor':int(energy.argmin())}
                if arm=='stored':
                    selected=next(r for r in selection['pools'] if r['case']=='PEWBOL' and r['seed']==seed)
                    picks['rerank_on']=selected['cand_index']
                strata={}
                # Fixed angular categories are descriptions, not trained decisions.
                for key,mask in [('N_trans',np.array([m['N_M_N']>140 for m in measure])),
                                 ('N_cis',np.array([m['N_M_N']<=140 for m in measure])),
                                 ('S_opposite',np.array([m['S_M_S']>120 for m in measure])),
                                 ('S_same_side',np.array([m['S_M_S']<=120 for m in measure])),
                                 ('N_trans_SO3_anti',np.array([m['N_M_N']>140 and m['N_S_vector_angle']>150 for m in measure]))]:
                    ix=np.flatnonzero(mask);strata[key]=dict(count=len(ix),floor=float(energy[ix].min()) if len(ix) else None)
                row=dict(arm=arm,seed=seed,n=len(energy),strata=strata,
                         picks={k:dict(index=i,energy=float(energy[i]),score=float(z['score'][i]),**measure[i]) for k,i in picks.items()})
                report['PEWBOL']['pools'].append(row)
                arrays.update({f'{arm}_s{seed}_{k}':z['X'][i] for k,i in picks.items()})
    report['PEWBOL']['scope']='N-M-N describes coordination isomer; S-M-S and N-S vectors describe actual sulfonate orientation. They are measured independently; H-bond counts use the shared co-bound exclusion.'
    np.savez_compressed(out/'n2_pewbol_classified_2026-10-18.npz',**arrays)
    # Physical forces, including their uncancelled internal component, are more interpretable than unique atom energies.
    roots=[out/r for r in ('n2_energy_diagnosis_2026-10-18','n2_energy_diagnosis_extra_2026-10-18')]
    report['XEPKAF']={}
    for seed in (11,12,13):
        for kind in ('before','after'):
            label=f's{seed}_{kind}';stem=next(r/'XEPKAF'/label for r in roots if (r/'XEPKAF'/label).with_suffix('.npz').exists())
            obj=json.loads(stem.with_suffix('.json').read_text())['summary']
            report['XEPKAF'][label]=dict(energy=obj['energy'],force_partition=obj['force_partition'],ligands=obj['ligands'])
    name,split=names['XEPKAF'];case=get_case(name)
    with starting_geometry(case,'full_oracle'):
        xm,_,_=case.make_builder()._build_shell(False,False)
    for l in xm.ligands:decompose_ligand(xm,l)
    stats=stats_from_db(CANDIDATE);tables=stats.nci_tables();rt=nci.build_runtime(xm,tables,'cpu')
    allc=nci.candidates(xm);cands={t:[c for c in cs if any(k in tables for k in c[-1])] for t,cs in allc.items()}
    xp={a.idx:i for i,a in enumerate(xm.atoms)}
    # Donor groups are recorded on inputs; use the haptic ligand's complete atom set, including H.
    cp=set(a for l,li in zip(xm.ligands,case.ligand_inputs) if any(len(g)>2 for g in li.donors) for a in l.atom_indices)
    report['XEPKAF_NCI_coupling']={}
    for seed in (11,12,13):
        selected=next(r for r in selection['pools'] if r['case']=='XEPKAF' and r['seed']==seed)
        poses={}
        with np.load(pool_path('etkdg_u1_os',split,'XEPKAF',seed)) as z:
            poses.update(before=z['X'][selected['base_index']],after=z['X'][selected['cand_index']])
        for tag,directory in [('fresh_on','n2_fresh_solve_2026-10-18'),('fresh_off','n2_fresh_control_2026-10-18')]:
            with np.load(out/directory/f'XEPKAF_s{seed}.npz') as z:poses[tag]=z['X'][z['score'].argmax()]
        for tag,x in poses.items():
            xt=torch.tensor(x[None],dtype=torch.float32,requires_grad=True);parts={}
            for t,e in rt.items():
                r,a=nci.features(t,xt,e['idx']);scores=nci._bilinear(e['T'],e['row'],r,a,e['grid'])[0]
                if t=='pp':mask=[bool(cp.intersection(c[0]) or cp.intersection(c[1])) for c in cands[t]]
                elif t=='chp':mask=[c[0] in cp or bool(cp.intersection(c[1])) for c in cands[t]]
                else:mask=[bool(cp.intersection(c[:3])) for c in cands[t]]
                mask=torch.tensor(mask,dtype=torch.bool)
                total=scores.sum();gradient=torch.autograd.grad(total,xt,retain_graph=True)[0][0].detach().numpy()
                parts[t]=dict(total_nats=float(total.detach()),cp_contacts_nats=float(scores[mask].sum().detach()),
                              cp_gradient_norm=float(np.linalg.norm(gradient[[xp[a] for a in cp]])))
            site_geometry={}
            cp_rows=[xp[a] for a in cp if xm.atoms[xp[a]].symbol!='H']
            m=x[xp[xm.metal_idx]];center=x[cp_rows].mean(0)
            site_geometry['metal_Cp_centroid']=float(np.linalg.norm(center-m))
            for lig,li in zip(xm.ligands,case.ligand_inputs):
                if len(li.donors)!=1 or len(li.donors[0])!=1:continue
                rows=[xp[a] for a in lig.atom_indices];p=next(i for i in rows if xm.atoms[i].symbol=='P')
                neighbors=[xp[b.j if b.i==xm.atoms[p].idx else b.i] for b in xm.bonds if (b.i==xm.atoms[p].idx or b.j==xm.atoms[p].idx) and xm.metal_idx not in (b.i,b.j)]
                site_geometry[f'L{lig.ligand_id}_metal_P']=float(np.linalg.norm(x[p]-m))
                site_geometry[f'L{lig.ligand_id}_metal_P_C_angles']=[angle(m-x[p],x[i]-x[p]) for i in neighbors]
            parts['coordination_geometry']=site_geometry
            report['XEPKAF_NCI_coupling'][f's{seed}_{tag}']=parts
    (out/'n2_pose_audit_2026-10-18.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['PEWBOL'],indent=2),flush=True)


if __name__=='__main__':setup();run()
