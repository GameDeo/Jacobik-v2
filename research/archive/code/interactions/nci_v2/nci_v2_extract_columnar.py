"""Bounded-feature-memory extraction into exact-merged root columns.

S2 pools remain explicitly disabled until audit. This CLI exercises the same
streaming implementation on allowlisted pilot/recert pools only.
"""
import argparse,copy,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2 import enumerate_opportunities,features_torch,FAMILIES
from research.archive.code.interactions.nci_v2.nci_v2_extract import unique_opportunities,deduplicate_poses
from research.archive.code.interactions.nci_v2.nci_v2_columnar import write_columns
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget,health_metrics
from research.archive.code.interactions.nci_v2.nci_v2_poolio import mapped_geometry,certified_assignment_mask
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import reference_stream,BudgetPause
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,DEV,Run,load_json,write_json


def stream_parent_rows(parent,mol,ref,poses,spec):
    """One pose's certified feature arrays at a time, never an all-pose design."""
    opp=unique_opportunities(enumerate_opportunities(mol,spec));reverse=copy.deepcopy(opp);symmetric=False
    for candidates in reverse.opportunities.values():
        for o in candidates:
            if o.get('symmetric_bond'):o['pos_A'],o['pos_B']=o['pos_B'],o['pos_A'];symmetric=True
    poses={s:deduplicate_poses(np.asarray(X),mol=mol) for s,X in poses.items() if len(X)}
    if not poses:return
    wp=float(parent.get('parent_weight',1.))
    for y,seed,XX in [(1,None,np.asarray(ref)[None])]+[(0,s,X) for s,X in poses.items()]:
        for k,x in enumerate(XX):
            budget();xt=torch.tensor(x,dtype=torch.float64)
            f=features_torch(xt,opp,spec);r=features_torch(xt,reverse,spec) if symmetric else f
            for family,oo in opp.opportunities.items():
                for i,o in enumerate(oo):
                    dim=2 if family=='bond_atom' and not o['has_c_neighbors'] else 4
                    view=f.features[family][0,i,:dim].tolist()
                    yield dict(parent_id=parent['code'],group_id=parent['group_id'],family=family,
                        occurrence_id=o['occ_id'],context=o['z_o'],features=view,
                        feature_views=sorted([view,r.features[family][0,i,:dim].tolist()]) if o.get('symmetric_bond') else [view],
                        y=y,weight=wp/len(oo) if y else wp/(len(oo)*len(poses)*len(XX)),seed=seed,pose=k if not y else None)


def run(spec_path,frame_path,pool_dirs,clean_root,out,split='pilot_cohort'):
    budget(start=True);spec=load_json(spec_path);frame=load_json(frame_path)
    pp=frame['pilot_cohort']['parents'] if split=='pilot_cohort' else frame['main_allocation']['splits'][split]
    parents=[dict(p,code=p.get('code',p.get('representative_code')),shard=p.get('shard',p.get('representative_shard'))) for p in pp]
    allowed=[MAIN/'data/refit/nci_v2'/p for p in ('pilot_v1/pools','recert_run_a','recert_arms')]
    paths=[]
    for d in map(Path,pool_dirs):
        if not any(d.resolve().is_relative_to(a.resolve()) for a in allowed):raise ValueError('S2 pools are not authorized; development pools only')
        paths += [p for p in sorted(d.rglob('*.npz')) if p.name.split('_')[0] in {q['code'] for q in parents} and
            ('control' in p.name or re.fullmatch(r'.+_s[0-9]+\.npz',p.name))]
    overlays=spec.get('pool_mapping_overlays',{});certs=spec.get('assignment_certificate_masks',{})
    for p in [*overlays.values(),*certs.values()]:
        if not Path(p).resolve().is_relative_to(DEV.resolve()):raise ValueError('repair/certificate inputs must be under data_dev until S2 audit')
    inputs=[frame_path,*paths,*overlays.values(),*certs.values()]
    for p in parents:
        base=Path(clean_root)/('lnQM' if p['dataset']=='LNf' else 'tmQM');inputs += [base/'xyz'/f"tmQM_{p['shard']}.xyz",base/'bo'/f"tmQM_{p['shard']}.BO",base/'tmQM_y.csv']
    r=Run('extract-columnar',spec_path,inputs,dict(split=split,health_policy=spec['solver_artifact_controls']['version']),out)
    if r.resumed:return r.path
    group_counts=Counter(p['group_id'] for p in parents);coverage=[];pools=[];totals=Counter()
    refs=[dict(p,representative_code=p['code'],representative_shard=p['shard']) for p in parents]
    def rows():
        for parent,mol,error in reference_stream(refs,clean_root):
            budget();code=parent['code'];print('columnar extract',code,flush=True)
            if error:coverage.append(dict(code=code,group_id=parent['group_id'],stratum=parent.get('stratum','unknown'),status='missing_reference',reason=error));continue
            ref=np.asarray([a.position for a in mol.atoms])
            href,_,_,_=health_metrics(mol,ref,ref[None],spec['solver_artifact_controls'])
            if not bool(href['healthy'][0]):
                oo=unique_opportunities(enumerate_opportunities(mol,spec))
                coverage.append(dict(code=code,group_id=parent['group_id'],stratum=parent.get('stratum','unknown'),
                    status='reference_health_failed',counts={f:len(o) for f,o in oo.opportunities.items()},zero_opportunities=oo.total_count==0))
                continue
            sy=np.array([a.symbol for a in mol.atoms]);poses=defaultdict(list)
            for path in paths:
                if not path.name.startswith(code+'_'):continue
                with np.load(path,allow_pickle=False) as z:
                    if 'error' in z.files:pools.append(dict(path=str(path),status='solver_error'));continue
                    try:
                        X,info=mapped_geometry(z,pool_path=path,overlay_path=overlays.get(path.name))
                        if not np.array_equal(z['ref_symbols'],sy) or not np.allclose(z['ref_X'],ref,atol=1e-7):raise ValueError('reference_identity_mismatch')
                        if int(z['charge'])!=mol.charge or int(z['mult'])!=mol.spin_state:raise ValueError('parent_state_mismatch')
                        if path.name not in certs:raise ValueError('missing_assignment_certificate')
                        mask=certified_assignment_mask(certs[path.name],path,len(X));finite=np.isfinite(X).all((1,2));healthy=np.zeros(len(X),bool)
                        if finite.any():
                            h,_,_,_=health_metrics(mol,ref,X[finite],spec['solver_artifact_controls'],z['coord_intact'][finite] if 'coord_intact' in z.files else None)
                            healthy[finite]=h['healthy']
                        keep=mask&healthy&finite
                        seed=int(z['seed']) if 'seed' in z.files else int(path.stem.split('_')[1].lstrip('s'))
                        poses[seed].extend(X[keep]);pools.append(dict(path=str(path),status='mapped',**info,poses=len(X),assignment_matched=int(mask.sum()),healthy=int(healthy.sum()),included=int(keep.sum())))
                    except BudgetPause:raise
                    except (ValueError,KeyError) as exc:pools.append(dict(path=str(path),status='excluded',reason=str(exc),excluded_poses=len(z['X']) if 'X' in z.files else 0))
            opp=unique_opportunities(enumerate_opportunities(mol,spec))
            cc=dict(code=code,group_id=parent['group_id'],stratum=parent.get('stratum','_'.join(str(parent.get(k,'unknown')) for k in ('metal_row','charge_class','cn_class'))),counts={f:len(o) for f,o in opp.opportunities.items()},
                status='matched' if any(len(X) for X in poses.values()) else 'missing_matched_assignment',zero_opportunities=opp.total_count==0)
            coverage.append(cc)
            for row in stream_parent_rows(dict(parent,parent_weight=1/group_counts[parent['group_id']]),mol,ref,poses,spec):
                totals[row['family'],row['y']]+=row['weight'];yield row
    meta=write_columns(rows(),r.path/'opportunities',spec)
    strata={}
    for name in sorted({str(c.get('stratum','unknown')) for c in coverage}):
        cc=[c for c in coverage if str(c.get('stratum','unknown'))==name]
        groups={c['group_id'] for c in cc if 'group_id' in c};matched={c['group_id'] for c in cc if c['status']=='matched'}
        strata[name]=dict(submitted_groups=len(groups),assignment_matched_groups=len(matched),
            fraction=len(matched)/len(groups) if groups else None,zero_opportunity_parents=sum(c.get('zero_opportunities',False) for c in cc))
    write_json(r.path/'reports/coverage.json',dict(parents=coverage,pools=pools,strata=strata,
        excluded_pools=sum(p['status']=='excluded' for p in pools),excluded_poses=sum(p.get('excluded_poses',0) for p in pools),
        raw_family_source_mass={f:{str(y):totals[f,y] for y in (0,1)} for f in FAMILIES},
        normalization='PreparedRows balances training source masses to 1/2 within each independently fitted root/fold',
        physical_rows=meta['physical_input_rows'],stored_rows=sum(p['rows'] for p in meta['roots'])))
    r.finish(dict(parents=len(parents),physical_rows=meta['physical_input_rows'],stored_rows=sum(p['rows'] for p in meta['roots'])),
        ['Development pool allowlist only; S2/repair schema not audited.',
         'One root is loaded for fitting; very large single-root support books still require memory scaling audit.'])
    return r.path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--frame',required=True);p.add_argument('--pools',nargs='+',required=True);p.add_argument('--clean-root',required=True);p.add_argument('--out',required=True);p.add_argument('--split',default='pilot_cohort');a=p.parse_args()
    print(run(a.spec,a.frame,a.pools,a.clean_root,a.out,a.split))
