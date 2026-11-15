"""Full-cohort Task 2 runner for audited NCI v2 Step 3.

All source inputs are read-only. Outputs are restricted to data_dev/.
"""
from __future__ import annotations

import argparse
import collections
import os
import resource
import time
from pathlib import Path

import numpy as np

from research.archive.code.interactions.nci_v2.nci_v2_binning import bin_root, supported_error
from research.archive.code.interactions.nci_v2.nci_v2_columnar import (ChunkedProblem, PreparedRows, RootRows,
                                   grouped_cv_root, host_budget, support_mass)
from research.archive.code.interactions.nci_v2.nci_v2_devio import DEV, load_json, write_json
from research.archive.code.interactions.nci_v2.nci_v2_real_step3 import (DATA, MAIN, POLICY_PATH, Step2OccupantAdapter,
                                     SEEDS, extract_columns, max_rss_mib, resolve_cohort,
                                     set_cpu, shard_provenance)
from research.archive.code.interactions.nci_v2.nci_v2_s21_overlay import S21Adapter


def _rss():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)


def _support_table(columns):
    metadata=load_json(Path(columns)/'dataset.json');rows_out=[]
    for rid,root in enumerate(metadata['roots']):
        rows=RootRows(columns,rid)
        groups={str(y):len({rows.dataset['groups'][int(g)]
                           for g in np.unique(rows.g[rows.y==y])}) for y in (0,1)}
        row=dict(id=rid,family=root['family'],root=root['root'],dimension=root['dimension'],
                 rows=rows.n,groups_by_source=groups,
                 min_groups=min(groups.values()),supported_20=min(groups.values())>=20)
        rows_out.append(row)
    rows_out.sort(key=lambda r:(r['min_groups'],r['rows'],r['family'],r['root']),reverse=True)
    return rows_out


def _root_id_for(columns,selected):
    metadata=load_json(Path(columns)/'dataset.json')
    for item in metadata['roots']:
        if (item['family'],item['root'],item['dimension']) == \
           (selected['family'],selected['root'],selected['dimension']):
            return int(item['id'])
    raise RuntimeError(f"root missing from columns: {selected['family']} {selected['root']}")


def _binned_cv(columns,root_id,spec,out,*,workers=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=RootRows(columns,root_id);mass=support_mass(rows,spec)
    bdir=out/'binned';bin_meta=bin_root(rows,bdir,spec,8,4);summary=RootRows(bdir)
    surface,cv=grouped_cv_root(summary,spec,mass_cache=mass,workers=workers)
    write_json(out/'surface.json',surface);write_json(out/'cv.json',cv)
    return rows,summary,surface,cv,bin_meta


def _same_root_binned_surface(columns,root_id,spec,config,out):
    """Fit a held-config sensitivity surface without another hyperparameter search."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=RootRows(columns,root_id);mass=support_mass(rows,spec);bdir=out/'binned'
    meta=bin_root(rows,bdir,spec,8,4);summary=RootRows(bdir)
    prep=PreparedRows(summary,spec,mass_cache=mass);problem=ChunkedProblem(prep,config);fit=problem.fit()
    if not fit.success:raise RuntimeError(f'sensitivity surface fit failed: {fit.message}')
    surface=problem.surface(fit.x);write_json(out/'surface.json',surface)
    return rows,summary,surface,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--stronger-root-probes',type=int,default=3)
    p.add_argument('--resume-sensitivity',action='store_true',
        help='reuse completed primary/placebo extraction and fits after a sensitivity-only interruption')
    a=p.parse_args();set_cpu()
    if os.nice(0)<19:raise ValueError('nice -n 19 required')
    out=a.out.resolve()
    if not out.is_relative_to(DEV.resolve()):raise ValueError('Task 2 output must be under data_dev')
    if not 1<=a.workers<=4:raise ValueError('workers must be between 1 and 4')
    host_budget(start=True)
    frame=load_json(DATA/'frame.json');manifest=load_json(DATA/'manifest.json')
    repair=load_json(DATA/'repair_s2_1/repair_manifest.json');raw_spec=load_json(DATA/'spec.json')
    policy=load_json(POLICY_PATH)
    if policy['version']!='solver-artifact-controls.2':raise RuntimeError('health/placebo policy changed')
    spec=dict(raw_spec,solver_artifact_controls=policy)
    adapter=S21Adapter(DATA,MAIN,allow_audited_production=True)
    parents,decisions=resolve_cohort(frame,manifest,repair)
    restored_codes={d['slot_code'] for d in decisions if d['slot_state']=='restored'}
    if len(restored_codes)!=10:raise RuntimeError(f'expected 10 restored slots, got {len(restored_codes)}')

    columns=out/'primary/columns';placebo_columns=out/'primary/placebo_columns'
    if a.resume_sensitivity:
        primary=load_json(out/'primary/reports/extraction.json')
        supports=_support_table(columns)
        primary_cv=load_json(out/'primary_fit/cv.json')
        surface=load_json(out/'primary_fit/surface.json')
        bin_meta=load_json(out/'primary_fit/binned/dataset.json')
        primary_s=None;main_fit_s=None
    else:
        start=time.perf_counter()
        primary=extract_columns(parents,adapter,spec,out/'primary',focus_one_root=False,include_placebo=True)
        primary_s=time.perf_counter()-start
        supports=_support_table(columns)
    eligible=[r for r in supports if r['min_groups']>=25]
    if not eligible:raise RuntimeError('full cohort has no root with at least 25 groups in each source')
    main_root=eligible[0];main_id=int(main_root['id'])
    rows=RootRows(columns,main_id)
    if a.resume_sensitivity:
        cv=primary_cv;summary=RootRows(out/'primary_fit/binned',0)
    else:
        main_start=time.perf_counter()
        rows,summary,surface,cv,bin_meta=_binned_cv(columns,main_id,spec,out/'primary_fit',workers=a.workers)
        main_fit_s=time.perf_counter()-main_start

    placebo_support=_support_table(placebo_columns)
    placebo_eligible=[r for r in placebo_support if r['min_groups']>=25]
    placebo_result=None
    if a.resume_sensitivity and placebo_eligible:
        placebo_root=placebo_eligible[0]
        pl_summary=RootRows(out/'placebo_fit/binned',0)
        pl_cv=load_json(out/'placebo_fit/cv.json')
        pl_surface=load_json(out/'placebo_fit/surface.json')
        pl_meta=load_json(out/'placebo_fit/binned/dataset.json')
        placebo_result=dict(root=placebo_root,summary_rows=pl_summary.n,
            one_se_choice=pl_cv['selected']['config'],best_cv_mean=pl_cv['best']['mean'],
            cv_wall_s=None,binning=pl_meta['approximation'],surface=pl_surface)
    elif placebo_eligible:
        placebo_root=placebo_eligible[0]
        placebo_id=int(placebo_root['id'])
        _,pl_summary,pl_surface,pl_cv,pl_meta=_binned_cv(
            placebo_columns,placebo_id,spec,out/'placebo_fit',workers=a.workers)
        placebo_result=dict(root=placebo_root,summary_rows=pl_summary.n,
            one_se_choice=pl_cv['selected']['config'],best_cv_mean=pl_cv['best']['mean'],
            cv_wall_s=sum(float(r.get('wall_s',0.)) for r in pl_cv.get('execution',[])),
            binning=pl_meta['approximation'],surface=pl_surface)
    else:
        placebo_result=dict(status='no placebo root has >=25 groups per source',
                            supported_root_count=len(placebo_support))

    # Step 2 occupants only: restored slots read their original Step 2 pools.
    sensitivity_parents,sensitivity_decisions=resolve_cohort(frame,manifest,repair,sensitivity=True)
    missing_step2_codes={code for code in restored_codes if not any(
        manifest['tasks'].get(f'{code}:{seed}:control',{}).get('status')=='success' for seed in SEEDS)}
    sensitivity_adapter=Step2OccupantAdapter(adapter,restored_codes,missing_step2_codes)
    sensitivity_start=time.perf_counter()
    sensitivity=extract_columns(sensitivity_parents,sensitivity_adapter,spec,out/'sensitivity',
        focus_one_root=False,include_placebo=False)
    sensitivity_s=time.perf_counter()-sensitivity_start
    sens_columns=out/'sensitivity/columns';sens_supports=_support_table(sens_columns)
    sens_id=_root_id_for(sens_columns,main_root)
    sens_root=next(r for r in sens_supports if r['id']==sens_id)
    sens_surface_result=None
    if sens_root['min_groups']>=25:
        _,sens_summary,sens_surface,sens_cv,sens_meta=_binned_cv(
            sens_columns,sens_id,spec,out/'sensitivity_fit',workers=a.workers)
        surf_delta=supported_error(rows,surface,sens_surface,seed=37,count=4096)
        sens_surface_result=dict(root=sens_root,summary_rows=sens_summary.n,
            one_se_choice=sens_cv['selected']['config'],best_cv_mean=sens_cv['best']['mean'],
            same_one_se_choice=sens_cv['selected']['config']==cv['selected']['config'],
            supported_surface_delta=surf_delta,binning=sens_meta['approximation'])
    else:
        sens_surface_result=dict(root=sens_root,status='below 25 groups in one source; no sensitivity CV fit')

    # Probe up to the next most-supported roots with binned CV. If one has a
    # materially stronger CV signal than the Task 1 root, perform the requested
    # second exact-versus-binned check there.
    second_candidates=[r for r in eligible if r['id']!=main_id][:a.stronger_root_probes]
    probe_results=[];second_exact=None
    task1_gap=0.0010786329773662118
    for candidate in second_candidates:
        rid=int(candidate['id']);tag=f"root_{rid:04d}"
        _,probe_summary,probe_surface,probe_cv,probe_meta=_binned_cv(
            columns,rid,spec,out/'additional_root_probes'/tag,workers=a.workers)
        rec=dict(root=candidate,binned_best_cv_mean=probe_cv['best']['mean'],
                 binned_one_se_choice=probe_cv['selected']['config'],
                 summary_rows=probe_summary.n)
        probe_results.append(rec)
        if probe_cv['best']['mean'] < np.log(2.)-2*task1_gap:
            exact_rows=RootRows(columns,rid);exact_mass=support_mass(exact_rows,spec)
            exact_surface,exact_cv=grouped_cv_root(exact_rows,spec,mass_cache=exact_mass,workers=a.workers)
            losses=[v['mean'] for v in exact_cv['results']]
            binned_choice=probe_cv['selected']['config']
            under_exact=next(v for v in exact_cv['results'] if v['config']==binned_choice)
            gate=under_exact['mean']<=exact_cv['best']['mean']+exact_cv['best']['se']
            exact_delta=supported_error(exact_rows,exact_surface,probe_surface,seed=41,count=4096)
            second_exact=dict(root=candidate,exact_best_cv_mean=exact_cv['best']['mean'],
                exact_best_se=exact_cv['best']['se'],binned_choice_under_exact_mean=under_exact['mean'],
                binned_choice_within_one_se=gate,exact_one_se_choice=exact_cv['selected']['config'],
                binned_one_se_choice=binned_choice,same_one_se_choice=exact_cv['selected']['config']==binned_choice,
                spearman_cv_loss=float(__import__('scipy.stats',fromlist=['spearmanr']).spearmanr(
                    losses,[v['mean'] for v in probe_cv['results']]).statistic),
                supported_probe_error=exact_delta,exact_cv_wall_s=sum(r.get('wall_s',0.) for r in exact_cv['execution']))
            write_json(out/'additional_root_probes'/tag/'exact_cv.json',exact_cv)
            write_json(out/'additional_root_probes'/tag/'exact_surface.json',exact_surface)
            break

    occupants=dict(collections.Counter(d['origin'] for d in decisions))
    sensitivity_occupants=dict(collections.Counter(d['origin'] for d in sensitivity_decisions))
    counts=dict(parent_slots=len(parents),groups=len({p['group_id'] for p in parents}),
        parents_loaded=primary['parents_loaded'],groups_loaded=primary['groups_loaded'],
        occupants=occupants,sensitivity_occupants=sensitivity_occupants,
        restored_codes=sorted(restored_codes),replacement_slot_count=sum(d['origin']=='step2_replacement' for d in decisions),
        step2_occupant_changed_slots=sum(d1['occupant']!=d2['occupant'] for d1,d2 in zip(decisions,sensitivity_decisions)),
        sensitivity_step2_original_unavailable_codes=sorted(missing_step2_codes))
    result=dict(cohort=counts,primary_extraction_seconds=primary_s,primary_binned_fit_seconds=main_fit_s,
        primary_extraction_wall_time_recovered_from_prior_run=bool(a.resume_sensitivity),
        sensitivity_extraction_seconds=sensitivity_s,main_root=main_root,
        root_support=[r for r in supports if r['min_groups']>=20],all_root_support_count=len(supports),
        family_root_support={fam:sum(r['min_groups']>=20 for r in supports if r['family']==fam)
                             for fam in sorted({r['family'] for r in supports})},
        binned_rows=summary.n,binning=bin_meta['approximation'],binned_one_se_choice=cv['selected']['config'],
        binned_best_cv_mean=cv['best']['mean'],binned_cv_wall_s=main_fit_s,
        placebo=placebo_result,sensitivity=sens_surface_result,
        health_exclusions_by_root=primary['health_exclusions_by_root'],
        unmappable_surface_fit_codes=primary['unmappable_surface_fit_codes'],
        overlay_r3_agreement=primary['overlay_r3_agreement'],
        second_root_binned_probes=probe_results,second_root_exact_comparison=second_exact,
        peak_rss_mib=max(max_rss_mib(),primary['peak_rss_mib'],sensitivity['peak_rss_mib']),
        provenance=dict(policy_sha256=__import__('hashlib').sha256(Path(POLICY_PATH).read_bytes()).hexdigest(),
            spec_sha256=__import__('hashlib').sha256((DATA/'spec.json').read_bytes()).hexdigest(),
            frame_sha256=__import__('hashlib').sha256((DATA/'frame.json').read_bytes()).hexdigest(),
            manifest_sha256=__import__('hashlib').sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
            repair_manifest_sha256=__import__('hashlib').sha256((DATA/'repair_s2_1/repair_manifest.json').read_bytes()).hexdigest(),
            source_shard_hashes=primary['source_shard_hashes'],loader_commit=primary['loader_commit']),
        step4_production_labels='disabled: evaluation pools lack bend-prior state until S2b lands')
    result['process_tree_rss_conservative_upper_bound_mib']=max_rss_mib()+sum(sorted(
        [w['peak_rss_mib'] for w in cv['worker_peak_rss_mib']],reverse=True)[:a.workers])
    write_json(out/'task2_report.json',result)
    write_json(out/'primary_fit/cv_comparison.json',dict(root=main_root,one_se=cv['selected'],best=cv['best'],
        cv_results=cv['results'],cv_workers=a.workers,binned_rows=summary.n))
    print(__import__('json').dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
