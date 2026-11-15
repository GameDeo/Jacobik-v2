"""CPU stored-pool rescoring and descriptive NCI on/off benchmark (cal only)."""
from __future__ import annotations
import copy,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2 import CalibratedArtifact,SurfaceArtifact,build_runtime,evaluate_raw_nci
from research.archive.code.interactions.nci_v2.nci_v2_response import artifact_from_json
from research.archive.code.interactions.nci_v2.nci_v2_step4_label_gpu import safe_load_mol_shard,safe_smiles_index
from tools.jgen.diag import poolio,d14_input_consistency as d14
from tools.jgen.bench import metrics

TAIL={"DAMVAQ","HOXJUZ","IFUROQ","JEGPUJ","NEPNEF","SOBHAS","USAPOU","XEYTUT","YUCZUS","ZELTAP"}

def _energy(runtime,X,batch=128):
    out=[]
    for i in range(0,len(X),batch):
        xx=torch.as_tensor(X[i:i+batch],dtype=torch.float64)
        out.extend(evaluate_raw_nci(xx,runtime).detach().cpu().numpy().tolist())
    return np.asarray(out,dtype=float)

def _score_family(mol,artifact,family,X):
    art=CalibratedArtifact(surface_artifacts={family:artifact.surface_artifacts[family]},
        gamma_shared=artifact.gamma_shared,family_scales=artifact.family_scales,
        selected_mode=artifact.selected_mode,singular_values=artifact.singular_values,
        condition_number=artifact.condition_number,ref_only_traversal_passed=artifact.ref_only_traversal_passed,
        spec=artifact.spec,metadata=artifact.metadata)
    return _energy(build_runtime(mol,art,device='cpu'),X)

def _absolute_summary(records,keys):
    zero={k:dict(code=k[0],seed=k[1],status='ok',top1_dE=0.,floor_dE=0.) for k in keys}
    return metrics.summarize(records,zero,keys)

def run(frame_path,clean_root,pools_root,manifest_path,sidecar_path,labels_out):
    frame=json.loads(Path(frame_path).read_text());manifest=json.loads(Path(manifest_path).read_text())
    artifact=artifact_from_json(json.loads(Path(sidecar_path).read_text()))
    eligibility=artifact.metadata.get('family_eligible',{})
    for family in artifact.family_scales:
        if eligibility and not eligibility.get(family,False):artifact.family_scales[family]=0.
    if artifact.metadata.get('identified') is False:
        artifact.gamma_shared=0.;artifact.family_scales={f:0. for f in artifact.family_scales}
    zero_scales=(artifact.gamma_shared==0. and all(v==0. for v in artifact.family_scales.values()))
    parents=frame['main_allocation']['splits']['calibration'];by_code={p['representative_code']:p for p in parents}
    codes_by_shard=defaultdict(set)
    for code,p in by_code.items():codes_by_shard[(p['dataset'],p['representative_shard'])].add(code)
    tasks=[t for t in manifest['tasks'].values() if t.get('split')=='calibration']
    keys=sorted((t['code'],int(t['seed'])) for t in tasks)
    cal_keys=keys;tail_keys=[k for k in keys if k[0] in TAIL]
    repairs=poolio.Repairs(pools_root)
    smiles=safe_smiles_index(Path(clean_root)/'tmQM'/'tmQM_y.csv')
    cache={};records_off={};records_on={};unit_rows=[];roots_by_unit={};d14_labels={};mol_cache={}
    for code in sorted({c for c,_ in keys}):
        p=by_code.get(code)
        if p is None:continue
        shard_key=(p['dataset'],p['representative_shard'])
        if shard_key not in mol_cache:
            mol_cache[shard_key]=safe_load_mol_shard(codes_by_shard[shard_key],
                p['representative_shard'],p['dataset'],Path(clean_root),smiles)
        mol=mol_cache[shard_key].get(code)
        if mol is None:continue
        d14_labels[code]=d14.classify(mol)
        # Build each family runtime once, then score the pose matrix in chunks.
        family_artifacts={}
        for f in artifact.surface_artifacts:
            family_artifacts[f]=CalibratedArtifact(surface_artifacts={f:artifact.surface_artifacts[f]},
                gamma_shared=artifact.gamma_shared,family_scales=artifact.family_scales,
                selected_mode=artifact.selected_mode,singular_values=artifact.singular_values,
                condition_number=artifact.condition_number,ref_only_traversal_passed=artifact.ref_only_traversal_passed,
                spec=artifact.spec,metadata=artifact.metadata)
        runtimes={} if zero_scales else {f:build_runtime(mol,a,device='cpu') for f,a in family_artifacts.items()}
        full_runtime=None if zero_scales else build_runtime(mol,artifact,device='cpu')
        for _,seed in [k for k in keys if k[0]==code]:
            k=(code,seed);eff=poolio.load_effective(Path(pools_root)/'pools',repairs,code,seed)
            if eff is None:
                rec=dict(code=code,seed=seed,status='failed',reason='stored_error_or_absent',top1_dE=None,floor_dE=None)
                records_off[k]=rec;records_on[k]=rec;continue
            score=np.asarray(eff['score'],float);de=np.asarray(eff['dE'],float)
            finite=np.isfinite(de)
            old=int(np.argmax(np.where(np.isfinite(score),score,-np.inf))) if np.isfinite(score).any() else 0
            Xgen=np.asarray(eff['X'],dtype=np.float64)
            mapping=eff.get('gen_to_ref')
            if mapping is None:
                fail=dict(code=code,seed=seed,status='failed',reason='missing_atom_mapping',top1_dE=None,floor_dE=None)
                records_off[k]=fail;records_on[k]=fail;continue
            Xref=Xgen[:,np.asarray(mapping,dtype=int),:]
            nci=np.zeros_like(score) if zero_scales else _energy(full_runtime,Xref)
            Z=float(eff['Z']);son=score.copy() if zero_scales else score+nci/Z
            new=int(np.argmax(np.where(np.isfinite(son),son,-np.inf))) if np.isfinite(son).any() else 0
            floor=float(np.min(de[finite])) if finite.any() else None
            for target,index,scr in ((records_off,old,score),(records_on,new,son)):
                if finite.size==len(scr) and finite[index]:
                    target[k]=dict(code=code,seed=seed,status='ok',reason='',top1_dE=float(de[index]),floor_dE=floor,
                        score_index=int(index),n_rows=len(de),n_finite=int(finite.sum()))
                else:target[k]=dict(code=code,seed=seed,status='failed',reason='top1_not_finite',top1_dE=None,floor_dE=floor)
            if zero_scales:
                fam_values={f:dict(off=0.,on=0.,delta=0.) for f in artifact.surface_artifacts}
            else:
                fam_values={f:_energy(rt,Xref) for f,rt in runtimes.items()}
                for f,v in fam_values.items():
                    # Exact per-family additive score contribution at old/new top-1.
                    fam_values[f]=dict(off=float(v[old]/Z),on=float(v[new]/Z),delta=float((v[new]-v[old])/Z))
            unit=dict(code=code,seed=seed,d14_label=d14_labels[code]['label'],
                top1_changed=bool(old!=new),old_index=old,new_index=new,
                off_top1_dE=records_off[k].get('top1_dE'),on_top1_dE=records_on[k].get('top1_dE'),
                floor_dE=floor,off_regret=(records_off[k].get('top1_dE')-floor if records_off[k].get('top1_dE') is not None else None),
                on_regret=(records_on[k].get('top1_dE')-floor if records_on[k].get('top1_dE') is not None else None),
                family_nci_score_terms=fam_values)
            unit_rows.append(unit);cache[k]=(mol,artifact,Xref,old,new,Z)

    # Attribute actual ranking flips to family/root score deltas. Keep the top
    # root drivers with signed score changes, enough to audit each flip.
    for unit in unit_rows:
        if not unit['top1_changed']:continue
        k=(unit['code'],unit['seed']);mol,art,Xref,old,new,Z=cache[k]
        pair=Xref[[old,new]];drivers=[]
        for fam,surfart in art.surface_artifacts.items():
            for root,surface in surfart.surfaces.items():
                one=copy.deepcopy(art)
                one.surface_artifacts={fam:SurfaceArtifact(family=fam,surfaces={root:surface},spec=surfart.spec)}
                vals=_energy(build_runtime(mol,one,device='cpu'),pair)/Z
                delta=float(vals[1]-vals[0])
                if abs(delta)>1e-12:drivers.append(dict(family=fam,root=root,score_delta=delta))
        unit['root_drivers']=sorted(drivers,key=lambda x:-abs(x['score_delta']))[:20]

    summaries={}
    for name,subset in (('calibration',cal_keys),('tail_calibration_only',tail_keys)):
        keys_ok=[k for k in subset if k in records_off and k in records_on]
        on_abs=_absolute_summary(records_on,keys_ok);off_abs=_absolute_summary(records_off,keys_ok)
        paired=metrics.summarize(records_on,records_off,keys_ok)
        per_parent={}
        parents_in=sorted({c for c,_ in keys_ok})
        for c in parents_in:
            ks=[k for k in keys_ok if k[0]==c]
            rr=[u for u in unit_rows if u['code']==c]
            def means(arm):
                vals=[arm[k] for k in ks if arm[k]['status']=='ok']
                if not vals:return None
                e=float(np.mean([v['top1_dE'] for v in vals]));f=float(np.mean([v['floor_dE'] for v in vals]));return dict(seeds=len(vals),e=e,f=f,r=e-f)
            per_parent[c]=dict(d14_label=d14_labels.get(c,{}).get('label','unavailable'),
                off=means(records_off),on=means(records_on),top1_changed=any(u['top1_changed'] for u in rr),
                changed_seeds=sum(u['top1_changed'] for u in rr))
        d14_split={}
        for lab in ('consistent','ambiguous','inconsistent'):
            ps={c:v for c,v in per_parent.items() if v['d14_label']==lab}
            def label_mean(arm,key):
                vals=[v[arm][key] for v in ps.values() if v.get(arm) is not None]
                return float(np.mean(vals)) if vals else None
            d14_split[lab]=dict(parents=len(ps),parents_with_flip=sum(v['top1_changed'] for v in ps.values()),
                off={x:label_mean('off',x) for x in ('e','f','r')},
                on={x:label_mean('on',x) for x in ('e','f','r')},
                paired_on_minus_off={x:(label_mean('on',x)-label_mean('off',x)
                    if label_mean('on',x) is not None and label_mean('off',x) is not None else None)
                    for x in ('e','f','r')})
        summaries[name]=dict(status='descriptive_in_sample',off_absolute=off_abs,on_absolute=on_abs,
            paired_on_minus_off=paired,per_parent=per_parent,d14_split=d14_split)
    result=dict(schema='nci_v2_step4_nci_on_off_ranking_v1',status='descriptive_in_sample',
        panel_policy='calibration and calibration-only tail; motion_validation/reserve excluded',
        nci_runtime_skipped_zero_scales=zero_scales,
        tail_codes=sorted(TAIL),calibration_keys=len(cal_keys),tail_keys=len(tail_keys),
        label_key_format='CSD_code:seed',
        labels={'off':{f'{code}:{seed}':v for (code,seed),v in records_off.items()},
                'on':{f'{code}:{seed}':v for (code,seed),v in records_on.items()}},
        summaries=summaries,units=unit_rows,
        d14_labels=d14_labels,known_failure_parents_touched=sorted({u['code'] for u in unit_rows}&{'JEGPUJ','NEPNEF','HOXJUZ','DAMVAQ','BEBQEE'}),
        candidate_sidecar_sha256=hashlib.sha256(Path(sidecar_path).read_bytes()).hexdigest())
    Path(labels_out).parent.mkdir(parents=True,exist_ok=True)
    Path(labels_out).write_text(json.dumps(result,separators=(',',':'))+'\n')
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for x in ('frame','clean-root','pools-root','manifest','sidecar','out'):p.add_argument('--'+x,required=True)
    a=p.parse_args();r=run(a.frame,a.clean_root,a.pools_root,a.manifest,a.sidecar,a.out)
    print(json.dumps({k:r[k] for k in ('calibration_keys','tail_keys','known_failure_parents_touched')}))
