"""Frozen G-3e cal/tail analysis; no result-dependent arm or cohort changes."""
import hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3e')
OUT=Path('research/campaigns/g')


def finite(value):return value is not None and bool(np.isfinite(value))


def paired(records,parents,field):
    per={};means=[];units=[];missing=[]
    for code in parents:
        diffs=[];entries=[]
        for seed in [11,12,13]:
            a=records['IA-k'].get((code,seed));c=records['C1'].get((code,seed))
            x=a.get(field) if a and a.get('status')=='ok' else None;y=c.get(field) if c and c.get('status')=='ok' else None
            x=float(x) if finite(x) else None;y=float(y) if finite(y) else None
            value=float(x-y) if finite(x) and finite(y) else None
            entries.append(dict(seed=seed,control=y,candidate=x,delta=value))
            if value is None:missing.append([code,seed])
            else:diffs.append(value);units.append(value)
        complete=len(diffs)==3;per[code]=dict(units=entries,complete=complete,mean=float(np.mean(diffs)) if diffs else None)
        if complete:means.append(np.mean(diffs))
    result=dict(declared_parents=len(parents),declared_units=3*len(parents),finite_units=len(units),complete_parents=len(means),
                missing=missing,per_parent=per,mean=None,ci95=None,median_unit=None)
    if means:
        a=np.asarray(means);rng=np.random.default_rng(11);boot=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
        result.update(mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),median_unit=float(np.median(units)))
    return result


def panel_result(panel,parents,arms):
    records={a['arm']:{} for a in arms};ledger=[];hashes={};fallbacks=[]
    plan=json.loads((OUT/'G3E_STEP5_PLAN.json').read_text())
    profile_hashes={v['sha256'] for v in plan['reused_same_stage_profile_records']}
    for a in arms:
        for code in parents:
            for seed in [11,12,13]:
                p=ROOT/panel/a['arm']/'records'/f'{code}_{seed}.json'
                if not p.exists():ledger.append(dict(code=code,seed=seed,arm=a['arm'],status='missing'));continue
                row=json.loads(p.read_text());hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                if row['flags']!=a['flags'] or row['fingerprint']!=a['fingerprint']:raise ValueError('arm metadata mismatch')
                if (row['code'],row['seed'],row['arm'])!=(code,seed,a['arm']):raise ValueError('record identity mismatch')
                # Fresh cal pilot used the named single-parent selector. Its exact
                # six hashes were frozen before bulk cal, never screen records.
                pilot=panel=='cal' and code==plan['profile_parent'] and hashes[str(p)] in profile_hashes
                if row['panel']!=panel and not (pilot and row['panel']=='named'):raise ValueError('panel identity mismatch')
                if code=='NEPNEF' and row.get('status')=='ok':
                    # Sequential invocations share engine._GFN2_CACHE but install
                    # fresh validity handlers. Cal recorded the known invalid
                    # embed; tail silently reused it. Retain raw records/energies
                    # unchanged and invalidate only this analysis view.
                    row['raw_status']=row['status'];row['status']='invalid_embed'
                    row['embed_fallback']=True
                    row['reason']='Known invalid NEPNEF; same-process cached fallback after cal; raw status ok retained on disk'
                if code!='NEPNEF' and (row.get('embed_fallback') or 'GFN2 embedding fallback' in str(row.get('reason'))):
                    fallbacks.append(dict(code=code,seed=seed,arm=a['arm'],reason='nonexempt_embed_fallback'))
                if row.get('replay_fallbacks',0) or any(x.get('error') for x in row.get('replay_chunks',[])):
                    fallbacks.append(dict(code=code,seed=seed,arm=a['arm'],reason='replay_fallback'))
                if row.get('status')!='ok':ledger.append(dict(code=code,seed=seed,arm=a['arm'],status=row.get('status'),reason=row.get('reason'),embed_fallback=row.get('embed_fallback',False)))
                elif row['n_finite']!=row['n_rows']:ledger.append(dict(code=code,seed=seed,arm=a['arm'],status='partial_pool',n_finite=row['n_finite'],n_rows=row['n_rows']))
                if row.get('status')=='ok':
                    de=np.asarray(row['energies_kj'])-row['ref_energy_kj']
                    row['_obscene']=float(np.mean(de>200)) if np.isfinite(de).all() else None
                    distortion=row.get('distortion') or [];known=[v['class_changed'] for v in distortion if v.get('class_changed') is not None]
                    row['_class_fraction']=float(np.mean(known)) if len(known)==row['n_rows'] and known else None
                    cov=row.get('coverage') or {};row['_coverage']=float(cov['full_pool_within_1A']) if 'full_pool_within_1A' in cov else None
                records[a['arm']][code,seed]=row
    primary=[c for c in parents if c not in ['FEKKAJ','NEPNEF']]
    metrics={field:paired(records,primary,field) for field in ['top1_dE','floor_dE','_obscene','top1_score','_class_fraction','_coverage']}
    all_included={field:paired(records,parents,field) for field in ['top1_dE','floor_dE','_obscene']}
    sensitivity={field:paired(records,[c for c in primary if c!='NUWFAM'],field) for field in ['top1_dE','floor_dE','_obscene']}
    control_good=[];control_obscene=[]
    for c in primary:
        values=[records['C1'].get((c,s),{}).get('top1_dE') for s in [11,12,13]]
        if all(finite(x) for x in values):(control_good if np.mean(values)<=200 else control_obscene).append(c)
    strata={label:paired(records,cohort,'top1_dE') for label,cohort in [('control_seed_mean_not_obscene',control_good),('control_seed_mean_obscene',control_obscene)]}
    regressions=[]
    for c,value in metrics['top1_dE']['per_parent'].items():
        worse=[u for u in value['units'] if u['delta'] is not None and u['delta']>25]
        if len(worse)>=2:regressions.append(dict(code=c,seeds=worse,all_units=value['units'],control_mean_not_obscene=c in control_good))
    wall={};secondary={}
    for arm in ['C1','IA-k']:
        walls={c:records[arm].get((c,11),{}).get('wall_solve_s') for c in parents}
        wall[arm]=dict(sum_s=sum(v for v in walls.values() if finite(v)),finite_parents=sum(finite(v) for v in walls.values()),per_parent=walls)
        row_steps=0;fixed_steps=0;fixed_wall_est=0.;step_counts=set()
        for c in parents:
            r=records[arm].get((c,11),{});chunks=r.get('replay_chunks') or []
            used=sum(x['B']*x['steps_executed'] for x in chunks if x.get('B') and x.get('steps_executed'))
            full=sum(x['B']*x['T'] for x in chunks if x.get('B') and x.get('T'))
            row_steps+=used;fixed_steps+=full;step_counts.update(x['steps_executed'] for x in chunks if x.get('steps_executed') is not None)
            w=r.get('wall_solve_s')
            if finite(w):fixed_wall_est+=w*full/used if used else w
        valid=[r for r in records[arm].values() if r.get('status')=='ok'];angles=[v['max_q2_deviation_deg'] for r in valid for v in (r.get('distortion') or [])]
        cov_units=sum(bool((r.get('coverage') or {}).get('full_pool_within_1A')) for r in valid)
        cov_parents=sum(any(bool((records[arm].get((c,s),{}).get('coverage') or {}).get('full_pool_within_1A')) for s in [11,12,13]) for c in parents)
        secondary[arm]=dict(valid_units=len(valid),expected_units=3*len(parents),returned_rows=sum(r['n_rows'] for r in valid),
                            angle_p95=float(np.quantile(angles,.95)) if angles and np.isfinite(angles).all() else None,
                            nonfinite_angles=sum(not finite(x) for x in angles),
                            full_pool_coverage_units=cov_units,full_pool_coverage_parents=cov_parents,declared_parents=len(parents),
                            coverage_capped_units=sum(bool((r.get('coverage') or {}).get('automorphisms_capped')) for r in valid),
                            obscene_rows=sum(int(np.sum(np.asarray(r['energies_kj'])-r['ref_energy_kj']>200)) for r in valid),
                            class_changed_rows=sum(v.get('class_changed') is True for r in valid for v in (r.get('distortion') or [])),
                            steps_recorded=sorted(step_counts),observed_row_steps=row_steps,fixed_budget_row_steps=fixed_steps,
                            estimated_fixed_budget_solver_assembly_s=fixed_wall_est,
                            fixed_budget_estimate_note='Linear row-step scaling of measured solver+assembly; includes non-scaling assembly/setup and is an estimate, not a measured rerun or gate input.')
    ratio=wall['IA-k']['sum_s']/wall['C1']['sum_s'] if wall['C1']['sum_s'] else None
    top=metrics['top1_dE'];floor=metrics['floor_dE'];obscene=metrics['_obscene']
    upper=lambda v:v['ci95'][1] if v['ci95'] else None
    complete=top['finite_units']==3*len(primary)
    gates=dict(primary=upper(top) is not None and upper(top)<2,
               floor_improvement=upper(floor) is not None and upper(floor)<0,
               obscene_improvement=upper(obscene) is not None and upper(obscene)<0,
               no_parent_regression=not regressions,solver_wall=ratio is not None and ratio<=1.3,
               primary_data_complete=complete,no_fallbacks=not fallbacks)
    gates['secondary_improvement']=gates['floor_improvement'] or gates['obscene_improvement']
    gates['registered_statistical_and_guard_pass']=all(gates[k] for k in ['primary','secondary_improvement','no_parent_regression','solver_wall'])
    gates['qualified_pass']=gates['registered_statistical_and_guard_pass'] and complete and not fallbacks
    cases={c:{a:[records[a].get((c,s)) for s in [11,12,13]] for a in ['C1','IA-k']} for c in ['FEKKAJ','NEPNEF','XOJHIN','NUWFAM'] if c in parents}
    # Case summaries retain the requested per-seed quantities, without duplicating coordinates.
    cases={c:{a:[None if r is None else {k:r.get(k) for k in ['seed','status','reason','top1_dE','floor_dE','top1_score','n_rows','embed_fallback','coverage']} for r in rows] for a,rows in values.items()} for c,values in cases.items()}
    return dict(parents=parents,primary_parents=primary,expected_records=6*len(parents),actual_records=sum(len(v) for v in records.values()),
                ledger=ledger,fallbacks=fallbacks,primary_metrics=metrics,all_included=all_included,excluding_NUWFAM=sensitivity,
                control_strata=strata,regression_guard_failures=regressions,
                nonobscene_control_guard_failures=[r for r in regressions if r['control_mean_not_obscene']],
                wall=wall,solver_wall_ratio=ratio,secondary=secondary,gates=gates,cases=cases,record_sha256=hashes)


def fmt(v,scale=1):
    if v['mean'] is None:return 'unavailable'
    a,b=v['ci95'];return f"{v['mean']*scale:.3f} [{a*scale:.3f}, {b*scale:.3f}]"


def main():
    status=json.loads((ROOT/'step5_status.json').read_text())
    if not status.get('complete'):raise RuntimeError('both panels must finish before checkpoint analysis')
    if list(ROOT.glob('*/job*/STOP.json')):raise RuntimeError('directive stop: report stop instead of normal completed-panel analysis')
    plan=json.loads((OUT/'G3E_STEP5_PLAN.json').read_text());arms=json.loads((OUT/'G3E_ARMS_AMEND5.json').read_text())['arms']
    jobs={str(p.relative_to(ROOT)):json.loads(p.read_text()) for p in ROOT.glob('*/job*/job.json')}
    results={p:panel_result(p,plan['panels'][p],arms) for p in ['cal','tail']}
    summary=dict(panels=results,arm_fingerprint=arms[1]['fingerprint'],overlap=plan['panel_overlap'],
                 bootstrap=dict(resamples=10000,seed=11,cluster='parent; mean three paired seed units before resampling'),
                 CPU_diagnostic='G3E_R2S_TWIST_REPORT.md',colab_spend=0,
                 execution_jobs=jobs,controller_status=status,
                 all_panels_qualified_pass=all(r['gates']['qualified_pass'] for r in results.values()))
    (OUT/'G3E_STEP5_RESULT.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# G-3e checkpoint — IA-k cal/tail and CPU twist diagnostic','',
           '**'+('All registered cal/tail gates pass.' if summary['all_panels_qualified_pass'] else 'The registered acceptance package does not pass.')+'** No tuning or rescue rerun follows these outcomes.',
           'The owner/auditor chooses the next action. Production defaults remain',
           'unchanged; L4 confirmation is still required before any fold-in.','',
           '## IA-k energy results','',
           'Candidate minus paired C1, kJ/mol; means and 95% parent-clustered CIs',
           '(10,000 percentile resamples, seed 11). Parent means use all three',
           'declared paired seed units. Primary excludes FEKKAJ and NEPNEF only.','',
           '| Panel | Primary parents / declared | Mean top-1 ΔE [95% CI] | Mean floor ΔE [95% CI] | Obscene-fraction change, pp [95% CI] |',
           '|---|---:|---|---|---|']
    for p,r in results.items():
        m=r['primary_metrics'];lines.append(f"| {p} | {m['top1_dE']['complete_parents']}/{len(r['primary_parents'])} | {fmt(m['top1_dE'])} | {fmt(m['floor_dE'])} | {fmt(m['_obscene'],100)} |")
    lines += ['', '## Registered gates','', '| Panel | Primary CI upper <2 | Floor CI upper <0 | Obscene CI upper <0 | No >25 regression on >=2 seeds | Wall <=1.3× | Overall |', '|---|---|---|---|---|---|---|']
    for p,r in results.items():
        g=r['gates'];text=lambda k:'PASS' if g[k] else 'FAIL'
        lines.append(f"| {p} | {text('primary')} | {text('floor_improvement')} | {text('obscene_improvement')} | {text('no_parent_regression')} | {text('solver_wall')} ({r['solver_wall_ratio']:.3f}×) | {text('qualified_pass')} |")
    lines += ['', 'Secondary improvement requires floor **or** obscene-fraction CI excluding',
              'zero in the improving direction. Wall uses measured solver+assembly',
              'counted once per parent, excluding shared preparation/MACE. Incomplete',
              'data cannot establish a full-panel pass; failed/missing units remain',
              'in the declared ledger. No failure is removed to improve a gate.','',
              'The contract regression guard is reported on all primary parents. The',
              'plan’s control-nonobscene stratum is also disclosed separately; it does',
              'not replace or waive a failed contract guard. Control strata below use',
              'the three-seed parent mean at the 200 kJ/mol obscene threshold.','',
              '### Regression failures','']
    for p,r in results.items():
        lines.append(f"**{p}:**")
        if not r['regression_guard_failures']:lines.append('None.')
        for v in r['regression_guard_failures']:
            lines.append(f"- {v['code']}: deltas {[u['delta'] for u in v['all_units']]}; control-mean nonobscene={v['control_mean_not_obscene']}.")
    lines += ['', '### Control quality strata','', '| Panel / control stratum | Complete parents | Mean top-1 ΔE [95% CI] |', '|---|---:|---|']
    for p,r in results.items():
        for name,v in r['control_strata'].items():lines.append(f"| {p} / {name} | {v['complete_parents']} | {fmt(v)} |")
    lines += ['', '## Denominators and secondary diagnostics','', '| Panel / arm | Records | Valid units | Returned rows | Angle p95 | Full-pool hits, units / parents | Solver+assembly s | Fixed-budget wall estimate s |', '|---|---:|---:|---:|---:|---|---:|---:|']
    for p,r in results.items():
        for a,v in r['secondary'].items():
            angle=f"{v['angle_p95']:.3f}°" if v['angle_p95'] is not None else 'unavailable'
            lines.append(f"| {p} / {a} | {r['actual_records']}/{r['expected_records']} total | {v['valid_units']}/{v['expected_units']} | {v['returned_rows']} | {angle} | {v['full_pool_coverage_units']}/{v['expected_units']} units; {v['full_pool_coverage_parents']}/{v['declared_parents']} parents | {r['wall'][a]['sum_s']:.2f} | {v['estimated_fixed_budget_solver_assembly_s']:.2f} |")
    lines += ['', 'Fixed-budget wall is explicitly a **linear row-step scaling estimate**,',
              'not a separate measured run, and is not used in gates. It scales measured',
              'solver+assembly by recorded B×T / B×used-steps per parent; assembly/setup',
              'does not scale with steps, so this approximation can overestimate.',
              'Raw recorded step counts, row-step totals, class changes, obscene rows,',
              'score/coverage/class paired CIs and capped-automorphism counts are in',
              'G3E_STEP5_RESULT.json. Parent coverage is any full-pool 1 Å heavy-atom',
              'symmetry-aware reference hit over the three seeds.','',
              '### Failure ledger','']
    for p,r in results.items():
        lines.append(f"{p}: {len(r['ledger'])} failed/partial/missing records of {r['expected_records']}; full ledger retained in JSON.")
        for reason in sorted({str(v.get('reason') or v['status']) for v in r['ledger']}):lines.append(f'- {reason}')
    lines += ['', '## Sensitivity and separate cases','', '| Panel / cohort | Mean top-1 ΔE [95% CI] | Mean floor ΔE [95% CI] |', '|---|---|---|']
    for p,r in results.items():
        for label,m in [('all included',r['all_included']),('primary excluding NUWFAM',r['excluding_NUWFAM'])]:lines.append(f"| {p} / {label} | {fmt(m['top1_dE'])} | {fmt(m['floor_dE'])} |")
    for code in ['FEKKAJ','NEPNEF','XOJHIN','NUWFAM']:
        lines += ['',f'### {code}','', '| Panel / seed | C1 top-1 / floor | IA-k top-1 / floor |', '|---|---|---|']
        for p,r in results.items():
            if code not in r['cases']:continue
            case=r['cases'][code]
            for i,seed in enumerate([11,12,13]):
                cells=[]
                for a in ['C1','IA-k']:
                    v=case[a][i]
                    cells.append((f"{v['top1_dE']:.3f} / {v['floor_dE']:.3f}"+(' (invalid cached embed)' if v['status']=='invalid_embed' else '')) if v and finite(v.get('top1_dE')) and finite(v.get('floor_dE')) else ('missing' if v is None else v['status']+(' (embed fallback exempt)' if v.get('embed_fallback') else '')))
                lines.append(f"| {p} / {seed} | {' | '.join(cells)} |")
    lines += ['', 'FEKKAJ and NEPNEF exclusions were fixed before results; NEPNEF fallback',
              'remains exempt and its failed rows are explicit.',
              'Tail NEPNEF raw records say ok because the engine embedding cache survived',
              'the cal-to-tail transition while the validity handler was new. Cal logged',
              'its known fallback. The analysis conservatively marks all six tail NEPNEF',
              'records invalid_embed and embed_fallback=true; raw files and diagnostic',
              'energy values are retained unchanged. These rows contribute no valid',
              'energy/coverage metrics. This disclosed instrumentation limitation does',
              'not affect primary judgement, which excludes NEPNEF in advance.',
              'NUWFAM remains in the',
              'primary cohort; the exclusion sensitivity does not rescue any gate.',
              'XOJHIN remains a search challenge and is not dropped.','',
              '## CPU R2s diagnostic','',
              'See [the separate CPU report](G3E_R2S_TWIST_REPORT.md): native gradients',
              'for all ten units, seven verified fixed-key score paths and three',
              'retained USAPOU common-reference failures. Six selected states meet the',
              'existing remaining-gain criterion; four do not. Barriers are path-specific,',
              'not full-dimensional maximum certificates. MACE favors the twisted endpoint',
              'in 8/10 selected pairs; the data do not establish all twists as spurious.',
              'Native CPU regeneration was explicitly approved after original GPU-export',
              'DOF recovery failed verification. Both sets remain on record. No score change.','',
              '## Execution and checkpoint','',
              'Amendment 5 frozen at 32de184, IA-k fingerprint 5f74b14b57ed, C1 production',
              '273f2ccbf892. IA-k fixed switch 100 releases symmetry only, retaining',
              'shape/projection through budget 250. Same bench invocation as C1, same',
              'local RTX 2060 SUPER, seeds 11–13, full-pool xz export. No new arithmetic',
              'was needed for IA-k. G-3d failure and G-3e clean24 results are retained.','',
              'Clean24 passed all unchanged advance predicates; full result at f7aff4b.',
              'After the owner-reported PC crash, complete units were reused. The',
              'interrupted cal profile had no records and its log was preserved. Fresh',
              'post-reboot paired profile and frozen 3.040578-hour estimate preceded',
              'bulk cal/tail execution; limit 8 h. Same-stage profile AJAQAD is reused',
              'within cal. Screen data were not reused as cal/tail judgement. Five',
              'parents overlap cal and tail and are measured separately in both panels.','',
              'Native CPU DOFs/references/commitments use atomic checkpoints; original',
              'failed coordinate-only inversions are preserved. Run controllers never',
              'retry a directive stop automatically. All input/model/flag/record hashes',
              'are retained in the frozen manifests and result JSON. Parent bootstrap',
              'rules and cohort/gate definitions were not adjusted after outcomes.','',
              'No val/reserve/locked sets, relaxation development loops, embed spot fixes,',
              'score changes, production fold-in or Colab usage. Colab spend zero. GPU',
              'lock held, nvidia-smi before each start, all owned jobs finished. L4',
              'confirmation remains a later requirement before any fold-in. Checkpoint:',
              'commit and stop for the auditor; no automatic production selection.']
    lines += ['', '### Recorded job wall times','', '| Job | Wall seconds | Exit |', '|---|---:|---|']
    for name,j in sorted(jobs.items()):lines.append(f"| {name} | {j['wall_s']:.3f} | {j.get('exception',j.get('exit_code'))} |")
    Path('research/campaigns/G3E_REPORT.md').write_text('\n'.join(lines)+'\n')
    print({p:r['gates'] for p,r in results.items()})

if __name__=='__main__':main()
