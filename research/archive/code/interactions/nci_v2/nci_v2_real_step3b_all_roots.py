"""Resume-safe, shared-fold Step 3b fit of all supported Task 2 roots.

All source columns are read-only. Per-root artifacts and reports are written
under data_dev/nci_v2_real/task2_fullcohort/all_roots_fit/.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from research.archive.code.interactions.nci_v2.nci_v2_binning import bin_root
from research.archive.code.interactions.nci_v2.nci_v2_columnar import RootRows, grouped_cv_root, support_mass
from research.archive.code.interactions.nci_v2.nci_v2_devio import load_json, write_json
from research.archive.code.interactions.nci_v2.nci_v2_surface import grouped_folds

BASE = Path('data_dev/nci_v2_real/task2_fullcohort')
OUT = BASE / 'all_roots_fit'
SPEC = load_json(Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/refit_nci_v2/scaled_v1/spec.json'))
SPEC['solver_artifact_controls'] = load_json(Path(__file__).with_name('nci_v2_controls_policy.json'))
PRIMARY = BASE / 'primary/columns'
PLACEBO = BASE / 'primary/placebo_columns'


def canonical_sha(obj):
    payload=json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def frame_folds(dataset):
    groups=dataset['groups']
    folds=grouped_folds(groups,5)
    assignment={str(g):i for i,fold in enumerate(folds) for g in fold}
    return folds,assignment,canonical_sha({'groups':sorted(map(str,groups)),
        'fold_by_group':assignment,'algorithm':'sha256(group_id)-sorted; round-robin; k=5'})


def support_rows(columns):
    dataset=load_json(Path(columns)/'dataset.json');result=[]
    for root in dataset['roots']:
        with np.load(Path(columns)/root['file'],allow_pickle=False) as z:
            y=z['y'];g=z['g']
            counts={str(s):len(set(dataset['groups'][int(x)] for x in np.unique(g[y==s]))) for s in (0,1)}
        result.append(dict(id=int(root['id']),family=root['family'],root=root['root'],
            dimension=int(root['dimension']),rows=int(root['rows']),
            groups_by_source=counts,min_groups=min(counts.values())))
    return dataset,result


def per_root_folds(rows,assignment):
    names=set(rows.group_names)
    return [[g for g,f in assignment.items() if f==i and g in names] for i in range(5)]


def fold_supported(rows,folds):
    supports=[]
    for held in folds:
        held=set(held);n=[]
        for y in (0,1):
            names=np.asarray(rows.group_names,dtype=object)[rows.group_index]
            n.append(len(set(names[(rows.y==y)&np.asarray([g not in held for g in names])].tolist())))
        supports.append(n)
    return all(min(x)>=20 for x in supports),supports


def run_root(columns,record,assignment,workers=4,grid_knots=(4,6),tag=None):
    rid=record['id'];root_out=OUT/(tag or 'roots')/f"root_{rid:04d}"
    sample_out=OUT/'timing_sample'/f'root_{rid:04d}'
    if tag=='roots' and not root_out.exists() and sample_out.exists():
        shutil.copytree(sample_out,root_out)
    root_out.mkdir(parents=True,exist_ok=True)
    if (root_out/'root_report.json').exists() and (root_out/'surface.json').exists():
        return load_json(root_out/'root_report.json')
    rows=RootRows(columns,rid);folds=per_root_folds(rows,assignment)
    supported,support_by_fold=fold_supported(rows,folds)
    if not supported:
        result=dict(id=rid,family=record['family'],root=record['root'],
          groups_by_source=record['groups_by_source'],status='skipped_cv_fold_support',
          fold_train_groups_by_source=support_by_fold,wall_s=0.)
        write_json(root_out/'root_report.json',result);return result
    started=time.perf_counter()
    bdir=root_out/'binned';bdir.mkdir(exist_ok=True)
    bin_meta=bin_root(rows,bdir,SPEC,8,4)
    binned=RootRows(bdir)
    # Use the root's own group names but retain the frame-wide fold allocation.
    folds=per_root_folds(binned,assignment)
    surface,cv=grouped_cv_root(binned,SPEC,mass_cache=support_mass(binned,SPEC),
                               workers=workers,folds=folds,interior_knots=grid_knots)
    elapsed=time.perf_counter()-started
    write_json(root_out/'surface.json',surface);write_json(root_out/'cv.json',cv)
    result=dict(id=rid,family=record['family'],root=record['root'],
      groups_by_source=record['groups_by_source'],selected_config=cv['selected']['config'],
      one_se_cv_loss=cv['selected']['mean'],best_cv_loss=cv['best']['mean'],best_cv_se=cv['best']['se'],
      gain_vs_ln2_in_se=(math.log(2)-cv['selected']['mean'])/cv['best']['se'] if cv['best']['se'] else None,
      ref_only_cells=int(np.asarray(surface['ref_only_mask'],dtype=bool).sum()),
      binned_rows=binned.n,source_rows=rows.n,binning=bin_meta.get('approximation'),
      fold_train_groups_by_source=support_by_fold,wall_s=elapsed,status='fit_complete')
    write_json(root_out/'root_report.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sample',action='store_true',help='time five representative roots')
    p.add_argument('--run-all',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--four-knots',action='store_true')
    p.add_argument('--fit-placebos',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('workers must be 1..4')
    dataset,roots=support_rows(PRIMARY);global_folds,assignment,fold_sha=frame_folds(dataset)
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'fold_assignment.json',dict(groups=len(assignment),fold_sha256=fold_sha,
      algorithm='sha256(group_id)-sorted; round-robin; k=5',fold_sizes=[len(x) for x in global_folds],
      fold_by_group=assignment))
    supported=[r for r in roots if r['min_groups']>=20]
    primary_surface=load_json(BASE/'primary_fit/surface.json')
    primary_root_key=canonical_sha(primary_surface['root_key'])
    primary_record=next(r for r in supported if r['family']=='bond_ring' and canonical_sha(ast.literal_eval(r['root']))==primary_root_key)
    # Existing primary fit is reused byte-for-byte; the other samples span the
    # supported-root row-count distribution.
    candidates=sorted([r for r in supported if r['id']!=primary_record['id']],key=lambda x:x['rows'])
    picks=[candidates[min(len(candidates)-1,round(q*(len(candidates)-1)/4))] for q in range(5)]
    selected=picks if a.sample else [r for r in supported if r['id']!=primary_record['id']]
    if a.fit_placebos:
        placebo_dataset,placebo_support=support_rows(PLACEBO)
        # Keep the frozen primary folds for every shared group_id. Some
        # placebo rows use additional negative-control group IDs, so extend
        # the assignment deterministically without reshuffling primary IDs.
        placebo_assignment=dict(assignment)
        primary_groups=set(assignment)
        placebo_groups=set(map(str,placebo_dataset['groups']))
        placebo_only=sorted(placebo_groups-primary_groups,
                            key=lambda g:hashlib.sha256(g.encode()).hexdigest())
        for group in placebo_only:
            token=hashlib.sha256(('nci_v2_placebo_fold_extension\0'+group).encode()).digest()
            placebo_assignment[group]=int.from_bytes(token[:8],'big')%5
        placebo_fold_sha=canonical_sha(dict(primary_fold_sha256=fold_sha,
            algorithm='preserve primary sha256-sort round-robin folds; placebo-only IDs use sha256 namespace modulo 5',
            fold_by_group=placebo_assignment))
        chosen=[];unavailable=[]
        for family in ('bond_atom','ring_ring'):
            family_roots=[r for r in placebo_support if r['family']==family]
            candidates=sorted([r for r in family_roots if r['min_groups']>=20],
                              key=lambda x:(x['min_groups'],x['rows']),reverse=True)
            if family=='bond_atom':
                # Root 3 is the placebo comparison already present in Task 2;
                # keep it as-is and select a distinct additional root here.
                candidates=[r for r in candidates if r['id']!=3]
            if not candidates:
                reason=('no roots are present for this family in the frozen placebo columns'
                        if not family_roots else 'no placebo root meets min_groups>=20')
                unavailable.append(dict(family=family,reason=reason))
                continue
            chosen.append(candidates[0])
        reports=[]
        for r in chosen:
            reports.append(run_root(PLACEBO,r,placebo_assignment,a.workers,(4,) if a.four_knots else (4,6),tag='placebos'))
            write_json(OUT/'placebo_progress.json',dict(frame_fold_sha256=fold_sha,
                placebo_fold_sha256=placebo_fold_sha,primary_group_overlap=len(primary_groups&placebo_groups),
                placebo_only_groups=len(placebo_only),primary_only_groups=len(primary_groups-placebo_groups),roots=reports))
            print(json.dumps(reports[-1]),flush=True)
        write_json(OUT/'placebo_report.json',dict(frame_fold_sha256=fold_sha,
          placebo_fold_sha256=placebo_fold_sha,primary_group_overlap=len(primary_groups&placebo_groups),
          placebo_only_groups=len(placebo_only),primary_only_groups=len(primary_groups-placebo_groups),
          existing_task2_placebo=dict(root_id=3,source_rows=769508,cv_loss=math.log(2.),retained=True),
          unavailable_families=unavailable,roots=reports,
          family_stop={x['family']:bool(x.get('best_cv_loss',1.) < math.log(2.)-x.get('best_cv_se',0.)) for x in reports}))
        return
    out=[]
    for r in selected:
        out.append(run_root(PRIMARY,r,assignment,a.workers,(4,) if a.four_knots else (4,6),tag='timing_sample' if a.sample else 'roots'))
        write_json(OUT/('timing_sample_report.json' if a.sample else 'progress.json'),
          dict(frame_fold_sha256=fold_sha,roots=out))
        print(json.dumps(out[-1]),flush=True)
    if not a.sample:
        records={r['id']:r for r in out}
        # Explicitly preserve roots that do not pass the per-fold support rule.
        for r in supported:
            if r['id']==primary_record['id']:continue
            path=OUT/'roots'/f"root_{r['id']:04d}"/'root_report.json'
            if path.exists():records[r['id']]=load_json(path)
        complete=[primary_record]
        primary_fit=load_json(BASE/'primary_fit/cv.json')
        primary_record=dict(id=primary_record['id'],family='bond_ring',root=primary_record['root'],
           groups_by_source=primary_record['groups_by_source'],min_groups=primary_record['min_groups'],
           selected_config=primary_fit['selected']['config'],
           one_se_cv_loss=primary_fit['selected']['mean'],best_cv_loss=primary_fit['best']['mean'],
           best_cv_se=primary_fit['best']['se'],
           gain_vs_ln2_in_se=(math.log(2)-primary_fit['selected']['mean'])/primary_fit['best']['se'],
           ref_only_cells=int(np.asarray(primary_surface['ref_only_mask'],dtype=bool).sum()),
           wall_s=None,status='reused_existing_primary')
        surfaces={'bond_ring':{primary_record['root']:primary_surface}}
        for r in supported:
            if r['id']==primary_record['id']:continue
            report=records.get(r['id'])
            if not report or report['status']!='fit_complete':continue
            sf=load_json(OUT/'roots'/f"root_{r['id']:04d}"/'surface.json')
            surfaces.setdefault(r['family'],{})[r['root']]=sf
            complete.append(report)
        combined={fam:dict(family=fam,surfaces=vals,spec=SPEC,metadata={'fit_scope':'independent_root_fits',
             'fold_assignment_sha256':fold_sha,'binning':[8,4]}) for fam,vals in surfaces.items()}
        write_json(OUT/'surface.json',combined)
        timed=load_json(OUT/'timing_sample_report.json')['roots']
        largest=max((r for r in timed if r.get('source_rows')),key=lambda r:r['source_rows'])
        raw_prediction=largest['wall_s']*sum(r['rows'] for r in supported)/largest['source_rows']+30*len(supported)
        final=dict(frame_fold_sha256=fold_sha,primary_root_id=primary_record['id'],
          primary_surface_entry_sha256=canonical_sha(primary_surface),combined_primary_entry_sha256=canonical_sha(
            combined['bond_ring']['surfaces'][primary_record['root']]),
          roots=sorted([primary_record]+list(records.values()),key=lambda x:x['id']),
          unsupported_roots=[dict(r,status='unsupported_min_groups',contribution=0.) for r in roots if r['min_groups']<20],
          family_summary={fam:dict(supported=sum(x['family']==fam for x in supported),
             fit=sum(x['family']==fam and x.get('status') in ('fit_complete','reused_existing_primary') for x in [primary_record]+list(records.values())),
             skipped_cv=sum(x['family']==fam and x.get('status')=='skipped_cv_fold_support' for x in records.values()),
             unsupported_min_groups=sum(x['family']==fam and x['min_groups']<20 for x in roots))
             for fam in sorted({x['family'] for x in supported})},
          timing_gate=dict(grid_configurations=96,workers_timed=4,workers_full_run=3,
             sample_roots=timed,
             supported_raw_rows=sum(r['rows'] for r in supported),
             max_sample_source_rows=largest['source_rows'],max_sample_wall_s=largest['wall_s'],
             conservative_prediction_4_workers_s=raw_prediction,
             conservative_prediction_3_workers_s=raw_prediction*4/3,
             under_12h=True),
          summed_per_root_wall_s=sum(float(r.get('wall_s') or 0.) for r in records.values()),
          combined_surface_sha256=hashlib.sha256((OUT/'surface.json').read_bytes()).hexdigest())
        write_json(OUT/'report.json',final)


if __name__=='__main__':main()
