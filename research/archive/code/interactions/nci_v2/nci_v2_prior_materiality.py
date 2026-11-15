"""Development-only final-score/prior materiality and motion-family inventory."""
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget,health_metrics
from research.archive.code.interactions.nci_v2.nci_v2_poolio import mapped_geometry,verified_pool_labels
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,Run,load_json,write_json
from research.archive.code.interactions.nci_v2.nci_v2_extract import graph_from_record
import argparse,re
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np


def distribution(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return dict(n=0,min=None,median=None,p90=None,p95=None,p99=None,max=None)
    return dict(zip(['n','min','median','p90','p95','p99','max'],[len(x),*map(float,np.quantile(x,[0,.5,.9,.95,.99,1]))]))


def pool_materiality(score,rank):
    s=np.asarray(score,float);r=np.asarray(rank,float)
    if s.ndim!=1 or s.shape!=r.shape or not np.isfinite(s).all() or not np.isfinite(r).all():raise ValueError('invalid score/rank')
    if not len(s):return dict(poses=0)
    gap=np.abs(s-r);spread=float(np.ptp(s));std=float(s.std());eps=2e-6*max(1.,float(np.max(np.abs(s))),float(np.max(np.abs(r))))
    flips=robust=ties=0
    for i in range(len(s)-1):
        a=s[i]-s[i+1:];b=r[i]-r[i+1:]
        flips+=int(np.sum(a*b<0));robust+=int(np.sum((a*b<0)&(np.abs(a)>eps)&(np.abs(b)>eps)))
        ties+=int(np.sum((a==0)!=(b==0)))
    si=int(np.argmax(s));ri=int(np.argmax(r));o1=np.argsort(-s,kind='stable');o2=np.argsort(-r,kind='stable')
    return dict(poses=len(s),score_spread=spread,score_std=std,abs_gap=distribution(gap),
        relative_to_score_range=distribution(gap/spread) if spread>0 else None,
        relative_to_score_std=distribution(gap/std) if std>0 else None,
        exact_order_changed=bool(np.any(o1!=o2)),strict_pair_flips=flips,tie_status_changes=ties,
        numeric_epsilon=eps,robust_pair_flips=robust,robust_order_changed=robust>0,
        exact_top1_changed=si!=ri,top1_sets_disjoint=not bool(set(np.flatnonzero(s==s.max()))&set(np.flatnonzero(r==r.max()))),
        robust_top1_changed=bool(si!=ri and r[ri]-r[si]>eps and s[si]-s[ri]>eps),
        score_top1=si,rank_top1=ri,non_penalty_sign_count=int(np.sum(s-r>eps)))


def run(spec_path,frame_path,graphs_path,motions_path,out):
    budget(start=True);frame=load_json(frame_path);parents={p['code']:p for p in frame['pilot_cohort']['parents']};graphs=load_json(graphs_path)
    pools={panel:[p for p in sorted((MAIN/'data/refit/nci_v2'/part).glob('*.npz')) if p.name.split('_')[0] in parents and
        (re.fullmatch(r'.+_s[0-9]+\.npz',p.name) if panel=='pilot' else p.name.endswith('_control.npz'))]
        for panel,part in [('pilot','pilot_v1/pools'),('recert','recert_run_a/pools')]}
    motion_files=sorted(Path(motions_path).glob('motions/*/*.json'))
    r=Run('followup2-prior-materiality',spec_path,[frame_path,graphs_path,*sum(pools.values(),[]),*motion_files],dict(spread='score max-min',numeric_epsilon='2e-6*max(1,max_abs_score,max_abs_rank)',no_mace=True),out)
    if r.resumed:return r.path
    spec=load_json(spec_path);policy=spec['solver_artifact_controls'];panels={}
    for panel,paths in pools.items():
        records=[];allgap=[];allrel=[];allenergy=[];healthy_gap=[];healthy_rel=[];healthy_energy=[]
        for path in paths:
            budget();code=path.name.split('_')[0];mol=graph_from_record(graphs[code]);ref=np.asarray([a.position for a in mol.atoms])
            with np.load(path,allow_pickle=False) as z:
                if 'error' in z.files:continue
                X,_=mapped_geometry(z,pool_path=path);labels=verified_pool_labels(z,['score','rank','Z'],pool_path=path)
                s=labels['score'];rr=labels['rank'];Z=float(labels['Z']);full=pool_materiality(s,rr)
                h,_,_,_=health_metrics(mol,ref,X,policy,z['coord_intact'] if 'coord_intact' in z.files else None);mask=h['healthy'];hm=pool_materiality(s[mask],rr[mask])
                gap=np.abs(s-rr);energy=2.479*Z*gap;allgap.extend(gap);allenergy.extend(energy)
                if full['score_spread']>0:allrel.extend(gap/full['score_spread'])
                healthy_gap.extend(gap[mask]);healthy_energy.extend(energy[mask])
                if hm.get('score_spread',0)>0:healthy_rel.extend(gap[mask]/hm['score_spread'])
                records.append(dict(pool=path.name,code=code,group_id=parents[code]['group_id'],normalizer=Z,all=full,healthy=hm,abs_raw_nats_gap=distribution(Z*gap),abs_energy_equivalent_kj=distribution(energy)))
        def counts(section):return {key:sum(bool(p[section].get(key,False)) for p in records) for key in ('exact_order_changed','robust_order_changed','exact_top1_changed','top1_sets_disjoint','robust_top1_changed')}
        panels[panel]=dict(pools=records,all=dict(abs_gap=distribution(allgap),relative_to_score_range=distribution(allrel),energy_equivalent_kj=distribution(allenergy),pool_counts=counts('all')),
            healthy=dict(abs_gap=distribution(healthy_gap),relative_to_score_range=distribution(healthy_rel),energy_equivalent_kj=distribution(healthy_energy),pool_counts=counts('healthy')))
        print(panel,panels[panel]['all'],panels[panel]['healthy'],flush=True)
    selected=Counter();candidates=Counter();valid=Counter();motion_rows=[]
    for path in motion_files:
        data=load_json(path)
        for m in data['diagnostics']:
            kind=m.get('definition',{}).get('kind','missing');candidates[kind,m.get('scan','local')]+=1
            if m.get('valid'):valid[kind,m.get('scan','local')]+=1
        for m in data['selected']:
            kind=m['definition']['kind'];selected[kind,m['scan']]+=1
            motion_rows.append(dict(file=str(path),parent=data['center']['code'],center=data['center']['center'],motion=m['motion_id'],kind=kind,scan=m['scan'],
                prior_delta_kj_fixed_beta=0. if kind!='bend' else None,
                interpretation='Parameter-space prior at fixed beta; Cartesian/FK commutation is not certified.'))
    counter=lambda c:[dict(kind=k,scan=s,count=n) for (k,s),n in sorted(c.items())]
    write_json(r.path/'reports/materiality.json',dict(panels=panels,motions=dict(selected=counter(selected),attempted=counter(candidates),valid=counter(valid),records=motion_rows),
        units='score/rank normalized score; raw nats multiply by Z; energy equivalent multiply by RT=2.479',
        caveats=['score-rank may include closure and numerical differences, not exclusively the bend prior.',
                 'Pooled quantiles weight poses, not independent groups; per-pool records are also reported.',
                 'No MACE labels used; fractions of physical reference residual response are unidentified.']))
    r.finish(dict(pools=sum(map(len,pools.values())),selected_motions=len(motion_rows)),['No actual MACE-response fraction can be inferred from score/rank alone.'])
    return r.path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--frame',required=True);p.add_argument('--graphs',required=True);p.add_argument('--motions',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(run(a.spec,a.frame,a.graphs,a.motions,a.out))
