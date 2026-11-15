"""Describe the completed screen without changing frozen advance decisions."""
import hashlib,json
from pathlib import Path
import numpy as np
from tools.jgen.g3d_filter import paired


def main():
    root=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3d/clean24')
    target=Path('research/campaigns/g')
    result=json.loads((target/'G3D_FILTER_RESULT.json').read_text())
    parents=result['parents'];names=['C1','R2s','IA'];records={};hashes={};stats={}
    for name in names:
        rows={}
        for code in parents:
            for seed in [11,12,13]:
                p=root/name/'records'/f'{code}_{seed}.json';r=json.loads(p.read_text())
                assert r['status']=='ok' and r['n_finite']==r['n_rows']
                assert not r['replay_fallbacks'] and all(c['mode']=='replay' for c in r['replay_chunks'])
                assert r['n_rows']==len(r['distortion'])
                assert all(np.isfinite(d['max_q2_deviation_deg']) for d in r['distortion'])
                assert r['coverage']['n_finite_geometry']==r['n_rows']
                hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                r['_obscene_fraction']=float(np.mean(np.asarray(r['energies_kj'])-r['ref_energy_kj']>200))
                r['_class_changes']=sum(d.get('class_changed') is True for d in r['distortion'])
                rows[code,seed]=r
        records[name]=rows
        wall=sum(rows[c,11]['wall_solve_s'] for c in parents)
        total=sum(r['n_rows'] for r in rows.values())
        obscene=sum(int(np.sum(np.asarray(r['energies_kj'])-r['ref_energy_kj']>200)) for r in rows.values())
        stats[name]=dict(returned_rows=total,angle_rows=total,obscene_rows=obscene,pooled_obscene_fraction=obscene/total,
                         class_changed_rows=sum(r['_class_changes'] for r in rows.values()),
                         class_unknown_rows=sum(d.get('class_changed') is None for r in rows.values() for d in r['distortion']),
                         full_pool_coverage_units=sum(bool(r['coverage']['full_pool_within_1A']) for r in rows.values()),
                         full_pool_coverage_parents=sum(any(rows[c,s]['coverage']['full_pool_within_1A'] for s in [11,12,13]) for c in parents),
                         coverage_automorphisms_capped_units=sum(bool(r['coverage']['automorphisms_capped']) for r in rows.values()),
                         solve_assembly_wall_s=wall,
                         steps_recorded=sorted({x['steps_executed'] for r in rows.values() for x in r['replay_chunks']}))
    for name in names[1:]:
        stats[name]['solve_assembly_ratio_C1']=stats[name]['solve_assembly_wall_s']/stats['C1']['solve_assembly_wall_s']
        stats[name]['obscene_fraction_paired']=paired(records[name],records['C1'],parents,'_obscene_fraction')
        stats[name]['top1_score_paired']=paired(records[name],records['C1'],parents,'top1_score')
    cases={c:{name:[dict(seed=s,top1_dE=records[name][c,s]['top1_dE'],floor_dE=records[name][c,s]['floor_dE'],
                             max_angle_p95=float(np.quantile([d['max_q2_deviation_deg'] for d in records[name][c,s]['distortion']],.95)),
                             coverage=records[name][c,s]['coverage']['full_pool_within_1A']) for s in [11,12,13]] for name in names}
           for c in ['NUWFAM','XOJHIN']}
    jobs={stage:json.loads((root.parent/stage/'job/job.json').read_text()) for stage in ['profile_clean24','profile_clean24_QEZVEY','clean24']}
    summary=dict(filter=result,secondary=stats,cases=cases,record_sha256=hashes,jobs=jobs,
                 total_GPU_job_wall_s=sum(j['wall_s'] for j in jobs.values()),
                 verdict='no_arm_advances_stop',colab_spend=0,step0_failed=['FA','FR+'])
    (target/'G3D_SCREEN_FINAL.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# G-3d — clean24 checkpoint, 2026-11-02','',
           '**STOP: neither R2s nor IA advances.** Both pass the registered energy',
           'predicates and fail the maximum-angle-deviation p95 guard. No gate was',
           'retuned or rerun after failure. Cal/tail were not started. FA and FR+',
           'remain failures at step 0, not filter losers. No production choice.','',
           '## Filter, unchanged predicates','',
           '24 fixed parents × seeds 11–13 × C1/R2s/IA: **216/216 valid records**;',
           '72/72 finite paired units per candidate, 24/24 complete parents. No',
           'failures or missing units; no replay or embed fallback. Every returned',
           'row has finite angle diagnostics and reference-coverage geometry.','',
           '| Arm | Median paired top-1 ΔE | Mean paired floor ΔE | Angle p95 | Angle p95 − C1 | Advance |',
           '|---|---:|---:|---:|---:|---|']
    control_angle=result['arms']['R2s']['control_angle']['p95']
    for name,a in result['arms'].items():
        lines.append(f"| {name} | {a['top1']['median']:.3f} | {a['floor']['mean']:.3f} | {a['angle']['p95']:.3f}° | {a['angle']['p95']-control_angle:.3f}° | FAIL: angle guard |")
    lines += ['',f'C1 pooled-row angle p95 is {control_angle:.3f}°; admissible candidate p95 is at most {control_angle+10:.3f}°.',
              'Energy units are kJ/mol, candidate minus paired C1; nonpositive median',
              'top-1 and nonpositive mean floor pass. The angle statistic is the p95',
              'over all returned rows of each row’s maximum deviation from its Q2',
              'target means. Counts: C1 2510 rows, R2s 2507, IA 2402. Both angle',
              'excesses exceed 10°. This guard alone determines both failures.','',
              '## Parent-clustered energy summaries','',
              'Means with 95% percentile CIs: 10,000 parent resamples, RNG seed 11;',
              'average the three paired seed units within each parent first. These',
              'are screen summaries, not the unrun cal/tail acceptance gates.','',
              '| Arm | Mean top-1 ΔE [95% CI] | Mean floor ΔE [95% CI] |',
              '|---|---|---|']
    for name,a in result['arms'].items():
        t=a['top1'];f=a['floor']
        lines.append(f"| {name} | {t['parent_mean']:.3f} [{t['parent_clustered_ci95'][0]:.3f}, {t['parent_clustered_ci95'][1]:.3f}] | {f['parent_mean']:.3f} [{f['parent_clustered_ci95'][0]:.3f}, {f['parent_clustered_ci95'][1]:.3f}] |")
    lines += ['', 'The energy improvements do not override the failed distortion guard.','',
              'Sensitivity excluding FEKKAJ and NUWFAM (FEKKAJ is not in clean24, so',
              'this excludes NUWFAM only): 23 parents, 69 finite pairs per candidate.','',
              '| Arm | Median top-1 ΔE | Mean top-1 ΔE [95% CI] | Mean floor ΔE [95% CI] |',
              '|---|---:|---|---|']
    for name,a in result['arms'].items():
        t=a['excluding_FEKKAJ_NUWFAM']['top1'];f=a['excluding_FEKKAJ_NUWFAM']['floor']
        lines.append(f"| {name} | {t['median']:.3f} | {t['parent_mean']:.3f} [{t['parent_clustered_ci95'][0]:.3f}, {t['parent_clustered_ci95'][1]:.3f}] | {f['parent_mean']:.3f} [{f['parent_clustered_ci95'][0]:.3f}, {f['parent_clustered_ci95'][1]:.3f}] |")
    lines += ['', 'No parent was dropped from the registered advance decision.','',
              '## Secondary diagnostics','',
              '| Arm | Obscene rows (>200 kJ/mol) | Class-changed rows | Pool coverage units /72 | Parent coverage /24 | Solver+assembly wall |',
              '|---|---:|---:|---:|---:|---:|']
    for name,s in stats.items():
        lines.append(f"| {name} | {s['obscene_rows']}/{s['returned_rows']} ({100*s['pooled_obscene_fraction']:.2f}%) | {s['class_changed_rows']}/{s['returned_rows']} | {s['full_pool_coverage_units']} | {s['full_pool_coverage_parents']} | {s['solve_assembly_wall_s']:.2f} s |")
    lines += ['', 'Coverage is full-pool symmetry-aware heavy-atom RMSD <=1 Å, including the',
              'metal; parent coverage is any hit over the three declared seeds, not',
              'top-1/floor-only coverage. Unknown class labels and automorphism-cap',
              'counts, paired top-1 score and paired obscene-fraction CIs are retained',
              'in G3D_SCREEN_FINAL.json. Automorphism enumeration was capped on 21/72',
              'units in every arm. No unknown class labels occurred. Solver/assembly',
              'wall counts batched seeds once',
              'per parent, excluding shared preparation/MACE. Wall ratios are descriptive',
              'here, not additional filter predicates. Nominal budget is 250 steps; some',
              'batches stopped early. Recorded step-count sets are shown below. A',
              'counterfactual fixed-budget wall was not measured; no rescue timing run',
              'was started after the failed filter.','']
    for name in names[1:]:
        s=stats[name];o=s['obscene_fraction_paired']
        lines.append(f"{name}: recorded replay step-count set {s['steps_recorded']}; solver/assembly ratio versus C1 {s['solve_assembly_ratio_C1']:.3f}; paired obscene-fraction change {100*o['parent_mean']:.3f} pp [95% CI {100*o['parent_clustered_ci95'][0]:.3f}, {100*o['parent_clustered_ci95'][1]:.3f}].")
    lines += ['', '## Separate case tables','',
              'FEKKAJ and NEPNEF are not members of the frozen clean24 manifest. Their',
              'step-0 records remain in G3D_STEP0_FINAL.json. NEPNEF remains known invalid',
              'input, embed_fallback=true with no accepted rows, and exempt under',
              'amendment 2. No reserve/val/locked sets were evaluated.']
    for code,arms in cases.items():
        lines += ['',f'### {code}','', '| Seed | C1 top-1 / floor | R2s top-1 / floor | IA top-1 / floor |', '|---|---|---|---|']
        for i,seed in enumerate([11,12,13]):
            vals=[f"{arms[n][i]['top1_dE']:.3f} / {arms[n][i]['floor_dE']:.3f}" for n in names]
            lines.append(f"| {seed} | {' | '.join(vals)} |")
    lines += ['', 'NUWFAM is retained in the main decision and shown in exclusion sensitivity.',
              'XOJHIN remains included as a search challenge; no score-hole exception',
              'is inferred and it is not dropped. Embed/E-Z variant keys remain in the',
              'full record/export row keys.','',
              '## Execution, provenance and checks','',
              'Amendment 4 and runtime/runner fixes: 5bc9fef. FA/FR+ retain the failed',
              'step-0 cap gate (14/36 censored at 144); criterion, horizon and quantile',
              'were not revisited. Their absence is not a missing filter result. The cause',
              'of the step-0 cap failure remains unknown; size is not established as its cause.','',
              'R2s/IA omit an unused cap only in whole-run/fixed-switch modes. Exact',
              '250-step update/state parity against legacy capped modes passed; adaptive',
              'finish still requires a cap. Five light phase tests and two runner',
              'regression checks passed. The runner now sets/resets FINISH_PHASE in both',
              'solver modules and prevents active phases reusing C1 results. Canonical',
              'replay regeneration unchanged; production default fingerprint unchanged.',
              'Production arithmetic at flags-off is unchanged; previous full golden',
              '245/245 per profile and full finish-phase file (31 passes) were not repeated.','',
              'Frozen fingerprints: C1 273f2ccbf892; R2s b2f758e72507; IA 4ad46b7b7722.',
              'Every arm had a 250-step budget. R2s and IA used the resolved thermal widths;',
              'IA fixed',
              'switch 100, R2s score-only throughout. Both release all aids as registered.',
              'Local RTX 2060 SUPER only, seeds 11–13, each candidate in the same bench',
              'invocation as C1. GPU lock held; nvidia-smi captured before every start.',
              'All recorded replay chunks mode replay; zero fallback. No VMs/Colab; spend 0.','',
              'Paired performance profiles AJAQAD and QEZVEY were reused as two fixed',
              'clean24 parents, with record hashes/flags/runner SHA/machine/GPU checked',
              'before the remaining 22-parent run. G3D_CLEAN24_INPUTS.json freezes this',
              'collection before panel-scale results. Original exports are preserved.','',
              'Planning estimate was 7485.6 s (~2.08 h) based on the mean of the two',
              'profiles. Remaining 22-parent job actually took 922.841 s. Including',
              f"both profile jobs, total GPU-job wall was {summary['total_GPU_job_wall_s']:.3f} s ({summary['total_GPU_job_wall_s']/60:.2f} min).",
              'Large profile heterogeneity made the planning estimate conservative.',
              'Full-pool xz XYZ exports, coverage manifests, energies, scores and keys',
              'are preserved in shared research/data/jgen/campaign/B/g3d/. Completion',
              'waits used the owner-requested 20-minute timer or earlier completion.','',
              'G3D_FILTER_RESULT.json holds the exact mechanical decision and denominators;',
              'G3D_SCREEN_FINAL.json adds secondary metrics, case tables and all 216',
              'record SHA256s. No relaxation, rescue rerun, tuning, production change or',
              'fold-in proposal. Step 5 did not start because no arm advanced; its 8-hour',
              'estimate gate and cal/tail gates are not evaluated. L4 confirmation remains',
              'required before any future fold-in. All GPU jobs finished; lock released.',
              'Checkpoint: stop and await audit.']
    Path('research/campaigns/G3D_REPORT.md').write_text('\n'.join(lines)+'\n')
    print({n:{k:v for k,v in s.items() if k not in ['obscene_fraction_paired','top1_score_paired']} for n,s in stats.items()})

if __name__=='__main__':main()
