"""P0b geometry-only solver-artifact controls, with frozen development policies.

Never loads score/energy fields. Low-group diagnostic fits are explicitly
unpromotable and cannot stand in for the >=20-group candidate activation gate.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from collections import defaultdict,Counter
import copy
import json
import math
from pathlib import Path
import re
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from research.archive.code.interactions.nci_v2.nci_v2 import OpportunityBatch,FAMILIES,features_torch,enumerate_opportunities
from jacobik.topology.graph import get_ligand_neighbors,donor_indices_of
from jacobik.stats.mapping import _covalent_radius
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,Run,load_json,write_json,digest
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import host_budget,reference_stream,BudgetPause
from research.archive.code.interactions.nci_v2.nci_v2_extract import (graph_record,graph_from_record,unique_opportunities,
    development_assignment_mask,same_pose)
from research.archive.code.interactions.nci_v2.nci_v2_columnar import write_columns,RootRows,PreparedRows,ChunkedProblem
from research.archive.code.interactions.nci_v2.nci_v2_surface import joint_values
from research.archive.code.interactions.nci_v2.nci_v2_poolio import mapped_geometry,certified_assignment_mask

torch.set_num_threads(1)


def budget(start=False):
    available=float(os.environ.get('NCI_V2_MIN_AVAILABLE_GIB',6. if start else 5.))
    return host_budget(min_available_gib=available,max_rss_mib=3900.)


def geometry_catalog(mol):
    pos={a.idx:i for i,a in enumerate(mol.atoms)};symbols=[a.symbol for a in mol.atoms]
    nb=get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx)
    bonds=sorted({tuple(sorted((pos[i],pos[j]))) for i,ns in nb.items() for j in ns})
    angles=[]
    for center,ns in nb.items():
        for j,a in enumerate(sorted(ns)):
            for b in sorted(ns)[j+1:]:angles.append((pos[a],pos[center],pos[b]))
    # All chordless ligand 5/6-membered cycles, including saturated rings.
    heavy={i:[j for j in ns if symbols[pos[j]]!='H'] for i,ns in nb.items() if symbols[pos[i]]!='H' and i!=mol.metal_idx}
    cycles=set()
    def walk(start,path):
        for j in heavy.get(path[-1],[]):
            if j==start and len(path) in (5,6):
                if sum(v in heavy.get(u,[]) for u in path for v in path)==2*len(path):cycles.add(tuple(sorted(path)))
            elif j>start and j not in path and len(path)<6:walk(start,path+[j])
    for a in sorted(heavy):walk(a,[a])
    rings=[tuple(pos[i] for i in r) for r in sorted(cycles)]
    excluded=set(bonds);metal=pos.get(mol.metal_idx)
    if metal is not None:
        excluded|={tuple(sorted((metal,pos[i]))) for i in donor_indices_of(mol)}
    nonbonded=[(i,j) for i in range(len(symbols)) for j in range(i+1,len(symbols)) if (i,j) not in excluded]
    classes=dict(bond_length=['-'.join(sorted((symbols[i],symbols[j]))) for i,j in bonds],
        bond_angle=[symbols[c]+'|'+ '-'.join(sorted((symbols[i],symbols[j]))) for i,c,j in angles],
        ring_pucker=[''.join(f'{s}{n}' for s,n in sorted(Counter(symbols[i] for i in r).items())) for r in rings])
    return dict(symbols=symbols,bonds=bonds,angles=angles,rings=rings,nonbonded=nonbonded,classes=classes,nb=nb,pos=pos)


def geometry_values(X,catalog):
    X=np.asarray(X,dtype=float)
    if X.ndim==2:X=X[None]
    out={}
    for kind,items in (('bond_length',catalog['bonds']),('bond_angle',catalog['angles'])):
        if not items:out[kind]=np.empty((len(X),0));continue
        ii=np.array(items)
        if kind=='bond_length':out[kind]=np.linalg.norm(X[:,ii[:,0]]-X[:,ii[:,1]],axis=-1)
        else:
            a=X[:,ii[:,0]]-X[:,ii[:,1]];b=X[:,ii[:,2]]-X[:,ii[:,1]]
            cosine=np.sum(a*b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-15)
            out[kind]=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    pucker=[]
    for ring in catalog['rings']:
        q=X[:,ring]-X[:,ring].mean(1,keepdims=True)
        pucker.append(np.linalg.svd(q,compute_uv=False)[:,-1]/np.sqrt(len(ring)))
    out['ring_pucker']=np.stack(pucker,axis=1) if pucker else np.empty((len(X),0))
    return out


def health_metrics(mol,ref,X,policy,stored_coord_intact=None):
    from jacobik.benchmark.coordination import coordination_integrity
    catalog=geometry_catalog(mol);refvalues=geometry_values(ref,catalog);values=geometry_values(X,catalog)
    n=len(X);relative=np.abs(values['bond_length']-refvalues['bond_length'])/np.maximum(refvalues['bond_length'],1e-12)
    maxbond=relative.max(1,initial=0.)
    ci=coordination_integrity(catalog['symbols'],ref,X)
    intact=np.asarray(ci['intact'],dtype=bool)
    if stored_coord_intact is not None:intact &= np.asarray(stored_coord_intact,dtype=bool)
    shell=np.maximum.reduce([ci[k] for k in ('sigma_shift_A','haptic_slip_A','haptic_shift_A')])
    pairs=np.asarray(catalog['nonbonded'],dtype=int)
    minimum=np.ones(n)*np.inf
    # Bound pair-displacement memory independently of atom/pose count.
    for i in range(0,len(pairs),1024):
        pp=pairs[i:i+1024];radius=np.array([_covalent_radius(catalog['symbols'][a])+_covalent_radius(catalog['symbols'][b]) for a,b in pp])
        d=np.linalg.norm(X[:,pp[:,0]]-X[:,pp[:,1]],axis=-1)/radius
        minimum=np.minimum(minimum,d.min(1))
    h=policy['health']
    heavy_min=np.full(n,np.inf)
    heavy=[(i,j) for i,j in catalog['nonbonded'] if catalog['symbols'][i]!='H' and catalog['symbols'][j]!='H']
    for offset in range(0,len(heavy),1024):
        pp=np.asarray(heavy[offset:offset+1024],dtype=int)
        heavy_min=np.minimum(heavy_min,np.linalg.norm(X[:,pp[:,0]]-X[:,pp[:,1]],axis=-1).min(1))
    heavy_overlap=heavy_min<h['min_nonbonded_heavy_distance_angstrom']
    stray=np.zeros(n,dtype=bool)
    for k,(i,j) in enumerate(catalog['bonds']):
        if catalog['symbols'][i] in ('H','F') or catalog['symbols'][j] in ('H','F'):
            stray |= values['bond_length'][:,k]>h['max_HF_bond_distance_angstrom']
    gross=minimum<h['min_nonbonded_radius_ratio'];badbond=maxbond>h['max_covalent_bond_relative_deviation']
    rms=lambda a:np.sqrt(np.mean(a*a,axis=1)) if a.shape[1] else np.zeros(n)
    angle=rms(values['bond_angle']-refvalues['bond_angle']);pucker=rms(values['ring_pucker']-refvalues['ring_pucker'])
    healthy=intact&~gross&~badbond&~heavy_overlap&~stray
    severity=maxbond/h['max_covalent_bond_relative_deviation']+shell/h['max_shell_shift_angstrom']+angle/h['angle_score_scale_degrees']+pucker/h['pucker_score_scale_angstrom']
    severity+=np.maximum(0,h['min_nonbonded_radius_ratio']-minimum)/h['min_nonbonded_radius_ratio']+(~healthy)
    return dict(healthy=healthy,severity=severity,max_bond_relative=maxbond,max_shell_shift=shell,
                angle_rms_shift_degrees=angle,pucker_rms_shift_angstrom=pucker,
                gross_overlap=gross|heavy_overlap,heavy_overlap=heavy_overlap,stray_atom=stray,min_nonbonded_heavy_distance=heavy_min,bad_bond=badbond,coord_intact=intact,min_nonbonded_ratio=minimum),catalog,refvalues,values


def placebo_opportunities(mol,ref,spec):
    """Heavy distance pairs; interligand membership frozen on the reference."""
    catalog=geometry_catalog(mol);nb=catalog['nb'];pos=catalog['pos'];sy=catalog['symbols'];bound=set(donor_indices_of(mol))
    actual=unique_opportunities(enumerate_opportunities(mol,spec));excluded=set()
    for family,entries in actual.opportunities.items():
        for o in entries:
            if family=='bond_atom':left=[o['idx_A'],o['idx_B']];right=[o['idx_C']]
            elif family=='bond_ring':left=[o['idx_A'],o['idx_B']];right=o['ring_indices']
            else:left=o['ring_indices_R'];right=o['ring_indices_S']
            excluded|={tuple(sorted((a,b))) for a in left for b in right}
    components={};seen=set()
    for start in sorted(nb):
        if start in seen or start==mol.metal_idx:continue
        queue=[start];seen.add(start)
        for a in queue:
            components[a]=start
            for b in nb.get(a,[]):
                if b not in seen:seen.add(b);queue.append(b)
    heavy=sorted(a.idx for a in mol.atoms if a.symbol!='H' and a.idx!=mol.metal_idx)
    distances={}
    for a in heavy:
        ds={a:0};queue=[a]
        for b in queue:
            if ds[b]>=3:continue
            for c in nb.get(b,[]):
                if c not in ds:ds[c]=ds[b]+1;queue.append(c)
        distances[a]=ds
    band=spec['solver_artifact_controls']['placebo']['interligand_reference_band_angstrom']
    entries=[];counts=Counter()
    for j,a in enumerate(heavy):
        for b in heavy[j+1:]:
            separation=distances[a].get(b)
            family='placebo_1_3' if separation==2 else 'placebo_1_4' if separation==3 else None
            if family is None and components.get(a)!=components.get(b) and (a,b) not in excluded:
                r=np.linalg.norm(ref[pos[a]]-ref[pos[b]])
                if band[0]<=r<=band[1]:family='placebo_interligand'
            if family is None:continue
            # Radial feature is invariant to endpoint order and the dummy A choice.
            B,C=sorted((a,b),key=lambda i:(sy[pos[i]],i in bound,i))
            A=next(iter(sorted(nb.get(B,[]))),B);neighbors=list(nb.get(C,[]))
            root=('bond_atom',family,(sy[pos[B]],sy[pos[C]]),(0,0),'distance_only',None,(B in bound,C in bound),components[B]==components[C],False)
            entries.append(dict(occ_id=f'{family}:{min(a,b)}:{max(a,b)}',family='bond_atom',control_family=family,
                pos_A=pos[A],pos_B=pos[B],pos_C=pos[C],idx_A=A,idx_B=B,idx_C=C,
                neighbor_pos=[pos[x] for x in neighbors],k_C=len(neighbors),has_c_neighbors=bool(neighbors),
                symmetric_bond=False,z_o=(root,)))
            counts[family]+=1
    return OpportunityBatch(mol.csd_code,dict(bond_atom=entries,bond_ring=[],ring_ring=[]),pos,len(entries)),dict(counts)


def weighted_quantile(values,weights,q):
    if not len(values):return None
    order=np.argsort(values,kind='stable');v=np.asarray(values)[order];w=np.asarray(weights)[order]
    return float(np.interp(q,np.cumsum(w)/w.sum(),v))


class Drift:
    def __init__(self):self.data=defaultdict(lambda:dict(ref=[],base=[],weight_ref=[],weight_base=[],delta=[]))
    def add(self,stratum,catalog,ref,base,seed_count,wp):
        for kind,rv in ref.items():
            O=rv.shape[1]
            if not O:continue
            for label in sorted(set(catalog['classes'][kind])):
                idx=np.array([i for i,c in enumerate(catalog['classes'][kind]) if c==label]);e=self.data[(stratum,kind,label)]
                r=rv[0,idx];b=base[kind][:,idx].reshape(-1)
                # Reference is appended per seed at 1/S, preserving equal seed mass.
                e['ref'].extend(r);e['weight_ref'].extend([wp/(O*seed_count)]*len(r))
                e['base'].extend(b);e['weight_base'].extend([wp/(O*seed_count*len(base[kind]))]*len(b))
                e['delta'].extend(np.abs(base[kind][:,idx]-r[None]).reshape(-1))
    def report(self):
        result=[]
        for (stratum,kind,label),e in sorted(self.data.items()):
            r,b=np.array(e['ref']),np.array(e['base']);wr,wb=np.array(e['weight_ref']),np.array(e['weight_base'])
            quant={str(q):weighted_quantile(b,wb,q)-weighted_quantile(r,wr,q) for q in (.5,.95)}
            result.append(dict(stratum=stratum,quantity=kind,element_class=label,median_shift=quant['0.5'],p95_shift=quant['0.95'],
                median_paired_absolute_shift=weighted_quantile(e['delta'],wb,.5),p95_paired_absolute_shift=weighted_quantile(e['delta'],wb,.95),
                wasserstein=float(wasserstein_distance(r,b,wr,wb)),reference_rows=len(r),baseline_rows=len(b),units='degrees' if kind=='bond_angle' else 'Angstrom'))
        return result


def dedup_indices(X,mol):
    keep=[]
    for i,x in enumerate(X):
        if not any(same_pose(x,X[j],mol=mol) for j in keep):keep.append(i)
    return np.array(keep,dtype=int)


def pool_geometry(path,mol):
    # Deliberately request only geometry/identity fields from the NPZ archive.
    ref=np.asarray([a.position for a in mol.atoms]);sy=np.array([a.symbol for a in mol.atoms])
    with np.load(path,allow_pickle=False) as z:
        if 'error' in z.files:return None
        X,mapping_info=mapped_geometry(z,pool_path=path)
        if not np.array_equal(z['ref_symbols'],sy) or not np.allclose(z['ref_X'],ref,atol=1e-7):raise ValueError('reference identity mismatch')
        if int(z['charge'])!=mol.charge or int(z['mult'])!=mol.spin_state:raise ValueError('charge/multiplicity mismatch')
        finite=np.isfinite(X).all((1,2));indices=np.flatnonzero(finite)
        keep=dedup_indices(X[indices],mol);indices=indices[keep];X=X[indices]
        stored=z['coord_intact'][indices] if 'coord_intact' in z.files else None
        return dict(X=X,stored=stored,raw_count=len(finite),nonfinite=int((~finite).sum()),indices=indices)


def feature_rows(parent,mol,poses,wp,spec):
    """Same feature call, physical counts, equal-seed and equal-pose weights."""
    real=unique_opportunities(enumerate_opportunities(mol,spec));placebo,pc=placebo_opportunities(mol,np.array([a.position for a in mol.atoms]),spec)
    real_counts={f:len(v) for f,v in real.opportunities.items()}
    if not poses:return
    for source,seed,coordinates in [(1,None,np.array([a.position for a in mol.atoms])[None])]+[(0,s,X) for s,X in poses.items()]:
        for x in coordinates:
            budget();xt=torch.tensor(x,dtype=torch.float64)
            for opportunities,is_placebo in ((real,False),(placebo,True)):
                features=features_torch(xt,opportunities,spec)
                # Symmetric directional views share one physical occurrence's mass.
                reverse=copy.deepcopy(opportunities);symmetric=False
                for candidates in reverse.opportunities.values():
                    for o in candidates:
                        if o.get('symmetric_bond'):o['pos_A'],o['pos_B']=o['pos_B'],o['pos_A'];symmetric=True
                reversed_features=features_torch(xt,reverse,spec) if symmetric else features
                for family,candidates in opportunities.opportunities.items():
                    for i,o in enumerate(candidates):
                        dim=1 if is_placebo else 2 if family=='bond_atom' and not o['has_c_neighbors'] else 4
                        O=pc[o['control_family']] if is_placebo else real_counts[family]
                        weight=wp/O if source else wp/(O*len(poses)*len(coordinates))
                        forward=features.features[family][0,i,:dim].tolist()
                        views=[forward,reversed_features.features[family][0,i,:dim].tolist()] if o.get('symmetric_bond') else [forward]
                        yield dict(parent_id=parent['code'],group_id=parent['group_id'],family=family,context=o['z_o'],features=forward,feature_views=views,y=source,weight=weight)


def fit_diagnostics(directory,spec,quartile):
    meta=load_json(Path(directory)/'dataset.json')
    if meta.get('fit_only_summary'):raise ValueError('parent attribution requires original columns, never binned fold summaries')
    policy=spec['solver_artifact_controls']['placebo'];records=[];noise=defaultdict(lambda:dict(values=[],weights=[]))
    cfg=policy['fitting']
    for rid,root in enumerate(meta['roots']):
        budget();rows=RootRows(directory,rid);rowparent=np.array([meta['parents'][int(i)] for i in rows.parent]);bad=np.isin(rowparent,list(quartile))
        total=rows.w.sum();record=dict(root=root['root'],family=root['family'],dimension=root['dimension'],stored_rows=rows.n,
            fit_mass_bottom_health_quartile=float(rows.w[bad].sum()/total),candidate_support_policy_unchanged=True,diagnostic_only=True)
        prep=PreparedRows(rows,spec)
        counts=prep.root_counts
        record['independent_groups_by_source']=counts.tolist()
        if min(counts)<policy['diagnostic_min_root_groups']:
            record.update(status='unidentified_low_diagnostic_groups',abs_F_bottom_health_quartile=None);records.append(record);continue
        problem=ChunkedProblem(prep,cfg);fit=problem.fit()
        if not fit.success:raise RuntimeError(f'diagnostic fit nonconverged: {fit.message}')
        sup=prep.support
        gate=(sup['counts_ref']>=policy['diagnostic_local_min_groups'])&(sup['ess_ref']>=policy['diagnostic_local_min_ess']-1e-10)&(sup['counts_base']>=policy['diagnostic_local_min_groups'])&(sup['ess_base']>=policy['diagnostic_local_min_ess']-1e-10)
        numerator=denominator=mass_in_support=0.;values=[];weights=[]
        # Balance class priors for attribution, matching the density fit.
        balanced=prep.weight
        for sl in rows.slices(4096):
            X=rows.coordinates(sl);F=np.abs(problem.logits(fit.x,sl));m=joint_values(X,prep.support_spec,gate.astype(float))
            domain=(rows.clip[sl]&3)==0
            domain &= (X[:,0]>root['domains'][0][0])&(X[:,0]<root['domains'][0][1])
            for j,(lo,hi) in enumerate(root['domains']):domain &= (X[:,j]>=lo)&(X[:,j]<=hi)
            supported=domain&(m>=policy['diagnostic_report_min_reliability'])
            contrib=balanced[sl]*F;denominator+=float(contrib.sum());numerator+=float(contrib[bad[sl]].sum())
            if np.any(supported):values.extend(F[supported]);weights.extend(balanced[sl][supported]);mass_in_support+=float(balanced[sl][supported].sum())
        record.update(status='diagnostic_fit',abs_F_bottom_health_quartile=numerator/denominator if denominator>0 else None,
            normalized_abs_F_mass=denominator,diagnostic_supported_fit_mass=mass_in_support,
            abs_F_median=weighted_quantile(values,weights,.5) if values else None,
            abs_F_p95=weighted_quantile(values,weights,.95) if values else None,abs_F_max=max(values) if values else None)
        label=__import__('ast').literal_eval(root['root'])[1]
        if label.startswith('placebo_') and values:noise[label]['values'].extend(values);noise[label]['weights'].extend(weights)
        records.append(record)
    floors={}
    for label in ('placebo_1_3','placebo_1_4','placebo_interligand'):
        e=noise[label];v,w=e['values'],e['weights']
        floors[label]=dict(median_abs_log_ratio=weighted_quantile(v,w,.5) if v else None,
            p95_abs_log_ratio=weighted_quantile(v,w,.95) if v else None,max_abs_log_ratio=max(v) if v else None,
            status='diagnostic_only_underpowered_for_candidate_activation' if v else 'unidentified_no_diagnostic_common_support')
    return dict(roots=records,placebo_noise_floor=floors,
                caveat='Negative-control diagnostic thresholds are 3 groups/ESS, not candidate activation. Candidate root minimum remains 20.')


def tails(health,fraction):
    order=sorted(health,key=lambda p:(p['health_score'],p['code']))
    return [p['code'] for p in order[:int(math.ceil(fraction*len(order)))]]


def run_controls(spec_path,frame_path,out=None):
    if os.nice(0)<19:raise ValueError('nice -n 19 required')
    budget(start=True);spec=load_json(spec_path);policy=spec['solver_artifact_controls'];frame=load_json(frame_path)
    parents=frame['pilot_cohort']['parents'];clean=MAIN/'data/refit/clean'
    panels=dict(pilot=parents,recert=[p for p in parents if p['code'] in frame['pilot_cohort']['repeatability_parents']])
    source_dirs=dict(pilot=MAIN/'data/refit/nci_v2/pilot_v1/pools',recert=MAIN/'data/refit/nci_v2/recert_run_a/pools')
    pools={}
    for panel,pp in panels.items():
        codes={p['code'] for p in pp}
        pools[panel]=[p for p in sorted(source_dirs[panel].glob('*.npz')) if p.name.split('_')[0] in codes and
            (re.fullmatch(r'.+_s[0-9]+\.npz',p.name) if panel=='pilot' else p.name.endswith('_control.npz'))]
    certificates=spec.get('assignment_certificate_masks',{})
    from research.archive.code.interactions.nci_v2.nci_v2_devio import DEV
    for certificate in certificates.values():
        if not Path(certificate).resolve().is_relative_to(DEV.resolve()):raise ValueError('development certificate inputs must be in data_dev')
    inputs=[frame_path,*sum(pools.values(),[]),*certificates.values()]
    for p in parents:
        base=clean/('lnQM' if p['dataset']=='LNf' else 'tmQM')
        inputs.extend([base/'xyz'/f"tmQM_{p['shard']}.xyz",base/'bo'/f"tmQM_{p['shard']}.BO",base/'tmQM_y.csv'])
    run=Run('phase2-solver-controls',spec_path,inputs,dict(panels=['pilot','recert'],energy_fields_accessed=False),out)
    if run.resumed:return run.path
    refparents=[dict(p,representative_code=p['code'],representative_shard=p['shard']) for p in parents]
    graphs={}
    for p,mol,error in reference_stream(refparents,clean):
        if error:raise ValueError(f"{p['code']}: {error}")
        graphs[p['code']]=graph_record(mol)
    write_json(run.path/'references/graphs.json',graphs)
    panel_results={}
    for panel,pp in panels.items():
        budget();print('controls geometry',panel,flush=True);health=[];cache={};drift=Drift();group_sizes=Counter(p['group_id'] for p in pp)
        for parent in pp:
            budget();code=parent['code'];mol=graph_from_record(graphs[code]);ref=np.asarray([a.position for a in mol.atoms]);seed_summaries=[];cache[code]={}
            files=[p for p in pools[panel] if p.name.startswith(code+'_')]
            loaded=[]
            for path in files:
                budget();data=pool_geometry(path,mol)
                if data is not None:loaded.append((path,data))
            for path,data in loaded:
                X=data['X']
                if not len(X):continue
                metrics,catalog,rv,bv=health_metrics(mol,ref,X,policy,data['stored'])
                if certificates:
                    matched=certified_assignment_mask(certificates[path.name],path,data['raw_count'])[data['indices']]&metrics['coord_intact'] if path.name in certificates else np.zeros(len(X),dtype=bool)
                else:matched=development_assignment_mask(mol,ref,X)
                seed=int(path.stem.split('_')[1].lstrip('s'))
                target=run.path/f'geometry/{panel}/{code}_{seed}.npz';target.parent.mkdir(parents=True,exist_ok=True)
                np.savez(target,X=X,healthy=metrics['healthy'],matched=matched)
                cache[code][seed]=str(target)
                severity=float(np.mean(metrics['severity']))
                seed_summaries.append(dict(seed=seed,raw_poses=data['raw_count'],deduplicated_finite_poses=len(X),
                    nonfinite=data['nonfinite'],healthy=int(metrics['healthy'].sum()),assignment_matched=int(matched.sum()),
                    healthy_assignment_matched=int((metrics['healthy']&matched).sum()),severity=severity,
                    rejected_coord=int((~metrics['coord_intact']).sum()),rejected_bond=int(metrics['bad_bond'].sum()),rejected_overlap=int(metrics['gross_overlap'].sum()),
                    bond_max_relative=float(metrics['max_bond_relative'].max(initial=0)),shell_max_angstrom=float(metrics['max_shell_shift'].max(initial=0))))
                stratum='_'.join(str(parent.get(k,'unknown')) for k in ('metal_row','charge_class','cn_class'))
                drift.add(stratum,catalog,rv,bv,len(loaded),1/group_sizes[parent['group_id']])
            severity=float(np.mean([s['severity'] for s in seed_summaries])) if seed_summaries else None
            nonfinite=sum(s['nonfinite'] for s in seed_summaries)
            score=1/(1+severity) if severity is not None and not nonfinite else 0.
            health.append(dict(code=code,group_id=parent['group_id'],health_score=score,mean_severity=severity,seeds=seed_summaries,
                zero_output=not bool(seed_summaries),no_energy_used=True))
            print(panel,code,'health',score,flush=True)
        decile=tails(health,policy['health']['bottom_parent_decile']);quartile=tails(health,policy['health']['bottom_parent_quartile'])
        variants={}
        for variant in ('raw','healthy','sensitivity'):
            def rows():
                for parent in pp:
                    code=parent['code']
                    if variant=='sensitivity' and code in decile:continue
                    mol=graph_from_record(graphs[code]);seedposes={}
                    for seed,path in cache[code].items():
                        with np.load(path,allow_pickle=False) as z:
                            mask=z['matched']
                            if variant!='raw':mask=mask&z['healthy']
                            X=z['X'][mask]
                            if len(X):seedposes[seed]=X
                    yield from feature_rows(parent,mol,seedposes,1/group_sizes[parent['group_id']],spec)
            destination=run.path/f'columns/{panel}/{variant}'
            metadata=write_columns(rows(),destination,spec,chunk_size=8192)
            diagnostic=fit_diagnostics(destination,spec,set(quartile))
            write_json(run.path/f'reports/{panel}_{variant}_fit.json',diagnostic)
            variants[variant]=dict(physical_rows=metadata['physical_input_rows'],stored_rows=sum(r['rows'] for r in metadata['roots']),
                                  diagnostic_report=str(run.path/f'reports/{panel}_{variant}_fit.json'),noise_floor=diagnostic['placebo_noise_floor'])
        panel_results[panel]=dict(assignment_policy='versioned_native_certificates' if certificates else 'legacy_15_degree_diagnostic',health=health,bottom_decile=decile,bottom_quartile=quartile,drift=drift.report(),variants=variants,
            candidate_activation_qualified_roots=0,interpretation='Only 12 or 6 independent development groups. Low-group controls are diagnostic, never promotable.')
        write_json(run.path/f'reports/{panel}.json',panel_results[panel])
    write_json(run.path/'reports/controls.json',dict(policy=policy,panels=panel_results,energy_fields_accessed=False,production_ready=False))
    run.finish(dict(panels=2,parents=18,source_pool_files=sum(map(len,pools.values()))),
        ['Development diagnostic fits use predeclared 3-group/ESS support; candidate 20-group activation is unchanged.',
         'Assignment matching uses versioned P2 masks.' if certificates else 'Assignment matching uses the legacy 15-degree development heuristic.',
         'No energy/score selection, weighting, or field access; no scaled production pools opened.'])
    return run.path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--frame',type=Path,required=True);p.add_argument('--out-dir',type=Path)
    a=p.parse_args()
    try:print(run_controls(a.spec,a.frame,a.out_dir),flush=True)
    except BudgetPause as e:print(f'PAUSED {e}',flush=True);raise SystemExit(75)

if __name__=='__main__':main()
