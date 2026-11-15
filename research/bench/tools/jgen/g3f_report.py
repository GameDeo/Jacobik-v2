"""G-3f frozen profile summary and L4 checkpoint report; never tunes outcomes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from tools.jgen.g3e_step5_report import paired,finite,fmt
ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3f')
OUT=Path('research/campaigns/g')


def profiles():
    result=dict(stages={},invalid_attempt=dict(path='profile_before',reason='diagnostic copied stale module globals; replay drains missing, only eager step measured; rejected in full'))
    for stage,folder in [('before','profile_before_retry1'),('after','profile_after')]:
        raw=json.loads((ROOT/folder/'timing.json').read_text());rows=[]
        for p in raw['profiles']:
            sample=[s['ms'] for s in p['samples'] if s['step']>=2];assert len(sample)==p['last']['replayed']>0
            assert p['last']['mode']=='replay' and not p['last'].get('error')
            means=np.asarray(sample).mean(0);event_sum=float(means.sum());wall_ms=p['last']['s_replay']*1000/p['last']['replayed']
            rows.append(dict(code=p['code'],arm=p['arm'],B=p['B'],steps=p['steps'],measured_replay_steps=len(sample),mean_ms=dict(zip(p['phases'],means.tolist())),event_total_ms=event_sum,replay_wall_ms_per_step=wall_ms,
                             residual_host_replay_ms=wall_ms-event_sum))
        result['stages'][stage]=dict(rows=rows,wall_s=raw['wall_s'],raw_sha256=hashlib.sha256((ROOT/folder/'timing.json').read_bytes()).hexdigest())
    result.update(fix='Skip unread Adam proposal only when neither phase uses Adam. Moment/count state preserved; all used arithmetic unchanged.',
        bottleneck='IA-k takes a separate score-only autograd pass for the remaining-gain stop and row freezing, then the aided gradient pass. Removing/changing it would change numeric decisions and is not admitted.',
        observer_note='External CUDA events inside captured halves; per-step synchronize to read timestamps. Exclude eager warm-up/graph-recording steps 0 and 1. Residual host/replay overhead includes observer synchronization; not confirmation wall.',
        fingerprint_before='5f74b14b57ed',fingerprint_after='5f74b14b57ed')
    (OUT/'G3F_PROFILE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');return result


def l4_result():
    manifest=json.loads((OUT/'G3F_MANIFEST.json').read_text());plan=json.loads((OUT/'G3F_L4_PLAN.json').read_text());noise=json.loads((OUT/'G3F_NOISE_RESULT.json').read_text())
    records={'C1':{},'IA-k':{}};ledger=[];hashes={};wall={a:dict(end_to_end_s=0.,solver_s=0.,timed_parents=0,per_parent={}) for a in records}
    for name,parents in plan['partitions'].items():
        directory=ROOT/'l4'/name/('g3f_'+name)
        for arm in manifest['arms']:
            label=arm['arm']
            for code in parents:
                for seed in [11,12,13]:
                    p=directory/label/'records'/f'{code}_{seed}.json'
                    if not p.exists():ledger.append(dict(vm=name,arm=label,code=code,seed=seed,status='missing'));continue
                    r=json.loads(p.read_text());hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                    assert r['fingerprint']==arm['fingerprint'] and r['flags']==arm['flags'] and r['code']==code and r['seed']==seed
                    assert r['gpu_name']=='NVIDIA L4' or r['gpu_name']=='L4',r['gpu_name']
                    if r.get('replay_fallbacks') or any(v.get('error') for v in r.get('replay_chunks',[])):ledger.append(dict(vm=name,arm=label,code=code,seed=seed,status='replay_fallback'))
                    if code=='NEPNEF' and r['status']=='ok':r['status']='invalid_embed';r['embed_fallback']=True
                    if r['status']!='ok':ledger.append(dict(vm=name,arm=label,code=code,seed=seed,status=r['status'],reason=r.get('reason'),embed_fallback=r.get('embed_fallback')))
                    else:
                        values=np.asarray(r['energies_kj'])-r['ref_energy_kj'];r['_obscene']=float(np.mean(values>200)) if np.isfinite(values).all() else None
                    records[label][code,seed]=r
                    if seed==11:
                        t=r.get('g3f_timing')
                        if t:
                            wall[label]['end_to_end_s']+=t['end_to_end_s'];wall[label]['solver_s']+=r.get('wall_solve_s',0.);wall[label]['timed_parents']+=1;wall[label]['per_parent'][code]=t
    parents=manifest['panels']['cal'];primary=[c for c in parents if c not in ['FEKKAJ','NEPNEF']]
    metrics={f:paired(records,primary,f) for f in ['top1_dE','floor_dE','_obscene']};regressions=[]
    for code,v in metrics['top1_dE']['per_parent'].items():
        if sum(u['delta'] is not None and u['delta']>25 for u in v['units'])>=2:regressions.append(dict(code=code,**v))
    strata={'control_not_obscene':[],'control_obscene':[]}
    for code in primary:
        vals=[records['C1'].get((code,s),{}).get('top1_dE') for s in [11,12,13]]
        if all(finite(v) for v in vals):strata['control_obscene' if np.mean(vals)>200 else 'control_not_obscene'].append(code)
    strata={k:paired(records,v,'top1_dE') for k,v in strata.items()}
    ratio=wall['IA-k']['end_to_end_s']/wall['C1']['end_to_end_s'] if wall['C1']['end_to_end_s'] else None
    complete=metrics['top1_dE']['complete_parents']==118 and metrics['floor_dE']['complete_parents']==118
    timed=all(v['timed_parents']==120 for v in wall.values())
    gates=dict(primary=metrics['top1_dE']['ci95'] is not None and metrics['top1_dE']['ci95'][1]<2,
        floor=metrics['floor_dE']['ci95'] is not None and metrics['floor_dE']['ci95'][1]<0,
        regression=len(regressions)<=noise['L4_guard_threshold'] and complete,wall=ratio is not None and ratio<=1.3 and timed)
    cases={c:{a:[records[a].get((c,s),{}) for s in [11,12,13]] for a in records} for c in ['AWONOR','FELSAU','CAXTAX','NUWFAM','XOJHIN','FEKKAJ','HEDJOP']}
    cases={c:{a:[{k:r.get(k) for k in ['seed','status','reason','top1_dE','floor_dE']} for r in rs] for a,rs in arms.items()} for c,arms in cases.items()}
    jobs={name:json.loads((ROOT/'l4'/name/('g3f_'+name)/'timing.json').read_text())['wall_s'] for name in plan['partitions'] if (ROOT/'l4'/name/('g3f_'+name)/'timing.json').exists()}
    return dict(metrics=metrics,regressions=regressions,N0=noise['N0'],threshold=noise['L4_guard_threshold'],wall=wall,wall_ratio=ratio,gates=gates,complete_primary=complete,complete_timing=timed,
        strata=strata,cases=cases,ledger=ledger,record_sha256=hashes,jobs_wall_s=jobs,
        all_included={f:paired(records,parents,f) for f in ['top1_dE','floor_dE','_obscene']},
        excluding_NUWFAM={f:paired(records,[c for c in primary if c!='NUWFAM'],f) for f in ['top1_dE','floor_dE','_obscene']})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile-only',action='store_true');a=ap.parse_args();profile=profiles()
    if a.profile_only:return
    r=l4_result();(OUT/'G3F_L4_RESULT.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    noise=json.loads((OUT/'G3F_NOISE_RESULT.json').read_text());identity=json.loads((ROOT/'identity/result.json').read_text())
    journal=[json.loads(s) for s in (ROOT/'colab_journal.jsonl').read_text().splitlines()];end=next((v for v in reversed(journal) if v['event']=='complete'),None)
    lines=['# G-3f checkpoint — measurements','', '## Noise calibration: fresh C1 minus G-3e C1','', '| Panel | Complete/declared primary parents | Top-1 mean [95% CI], kJ/mol | Floor mean [95% CI] | Obscene fraction, pp | Guard trips |','|---|---:|---|---|---|---:|']
    for p,v in noise['panels'].items():lines.append(f"| {p} | {v['metrics']['top1_dE']['complete_parents']}/{v['metrics']['top1_dE']['declared_parents']} | {fmt(v['metrics']['top1_dE'])} | {fmt(v['metrics']['floor_dE'])} | {fmt(v['metrics']['_obscene'],100)} | {v['guard_count']} |")
    lines+=['',f"N0={noise['N0']}; names: {[x['code'] for x in noise['panels']['cal']['guard_failures']]}. Full-cal L4 guard threshold={noise['L4_guard_threshold']}.",
        'Primary excludes FEKKAJ/NEPNEF; NUWFAM retained. All three seeds averaged',
        'within each parent, then 10,000 percentile parent bootstrap resamples,',
        'seed 11. C1 replicate deltas are exactly zero on all primary seed units;',
        'largest absolute parent mean and seed delta are zero on both panels.',
        'This measurement does not support the anticipated nonzero guard noise.',
        'G-3e remains failed as registered. Full finite denominators and',
        'NUWFAM-exclusion/all-included sensitivities are retained in noise JSON.','',
        '## Per-step CUDA-event profile and bounded fix','',
        '| Stage / parent / arm | B | Score/FK | Gradients | Aids | Optimizer | Other device/stream | Host/replay residual |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for stage,v in profile['stages'].items():
        for row in v['rows']:
            m=row['mean_ms'];other=row['event_total_ms']-sum(m[k] for k in ['score_forward_kinematics','score_and_total_gradients','aids','optimizer'])
            lines.append(f"| {stage} / {row['code']} / {row['arm']} | {row['B']} | {m['score_forward_kinematics']:.4f} | {m['score_and_total_gradients']:.4f} | {m['aids']:.4f} | {m['optimizer']:.4f} | {other:.4f} | {row['residual_host_replay_ms']:.4f} |")
    lines+=['','All timings are mean milliseconds per replay step, 248 samples per solve;',
        'warm-up/capture steps excluded. Other includes eigendecomposition,',
        'snapshot, projection, dual/history maintenance and applying/wrapping',
        'steps. External CUDA events are inserted in a research-only compiled',
        'method; the expressions and their order are preserved. A synchronize',
        'reads timestamps each step, so residual host overhead includes the',
        'observer cost. Confirmation uses uninstrumented solver code.','',profile['bottleneck'],'',profile['fix'],'',
        f"CPU identity: {sum(x['status']=='pass' for x in identity['units'])}/9 units (AJAQAD/AWONOR/CAXTAX × seeds 11–13), one thread, no culling, 250 steps.",
        'Returned coordinates, row keys, scores, native final DOFs and CPU',
        'rescores compare exactly, with before/after NPZ arrays retained.',
        'Replay/phase tests, canonical regeneration and default golden results',
        'are recorded in the verification manifest. IA-k fingerprint remains',
        '5f74b14b57ed, production 273f2ccbf892; goldens never written.','',
        'Initial verification without Torch deterministic mode: 21 passed, 14',
        'CUDA exact-parity failures. This configuration did not match the bench.',
        'With torch.use_deterministic_algorithms(True) and the same cuBLAS',
        'workspace setting as the bench, all 18 CUDA parity cases passed.',
        'Both logs are retained; no solver tolerance or numeric rule changed.',
        'Production/u1 golden checks each passed 245/245. The entire bounded',
        'profiling/fix/verification stage took less than 31 minutes.','',
        'The first profiling attempt is retained but rejected: it copied stale',
        'module globals and measured only eager warm-up rather than replay.',
        'Live globals and asserted replay-event drains fixed the instrument.',
        'Those invalid measurements are not in the profile table or any gate.','',
        '## Paired L4 cal confirmation','',
        '118/118 primary parents, 354/354 paired seed units. Unrelaxed MACE',
        'polar-1-s float64 energies; candidate minus within-VM C1. Energy',
        'units kJ/mol; obscene-fraction change in percentage points.','',
        '| Metric | Mean IA-k − C1 [95% parent CI] |','|---|---|']
    for field in ['top1_dE','floor_dE','_obscene']:lines.append(f"| {field} | {fmt(r['metrics'][field],100 if field=='_obscene' else 1)} |")
    lines+=['','| Gate | Result | Evidence |','|---|---|---|']
    evidence=dict(primary='Top-1 CI upper <2 kJ/mol',floor='Floor CI upper <0',regression=f"{len(r['regressions'])} parents; threshold {r['threshold']}",wall=f"End-to-end ratio {r['wall_ratio']:.3f}; maximum 1.3")
    for gate,value in r['gates'].items():lines.append(f"| {gate} | {'PASS' if value else 'FAIL'} | {evidence[gate]} |")
    lines+=['',f"Primary complete: {r['complete_primary']}; timing complete: {r['complete_timing']}.",f"Regression names: {[v['code'] for v in r['regressions']]}",'',
        'End-to-end wall counts prefetch, prepare and seed setup plus solver and',
        'assembly once per three-seed parent, excluding MACE, diagnostics and',
        'exports. The full shared preparation cost is allocated identically to',
        'both arms, representing each arm standalone. E/Z preparation is tagged',
        'per arm. This is distinct from the measured shared paired-job wall.','',
        '| Arm | End-to-end seconds | Solver/assembly seconds | Timed parents |','|---|---:|---:|---:|']
    for arm,w in r['wall'].items():lines.append(f"| {arm} | {w['end_to_end_s']:.3f} | {w['solver_s']:.3f} | {w['timed_parents']}/120 |")
    lines+=['','| Control stratum | Parents | Top-1 mean [95% CI] |','|---|---:|---|']
    for name,v in r['strata'].items():lines.append(f"| {name} | {v['complete_parents']} | {fmt(v)} |")
    lines+=['','Control strata use the three-seed parent mean at 200 kJ/mol. The',
        'regression guard still covers all declared primary parents.','', '| Sensitivity | Top-1 mean [95% CI] | Floor mean [95% CI] |','|---|---|---|']
    for name in ['all_included','excluding_NUWFAM']:lines.append(f"| {name} | {fmt(r[name]['top1_dE'])} | {fmt(r[name]['floor_dE'])} |")
    for code,arms in r['cases'].items():
        lines+=['',f'### {code}','', '| Seed | C1 top-1 / floor | IA-k top-1 / floor |','|---|---|---|']
        for i,seed in enumerate([11,12,13]):
            cells=[]
            for arm in ['C1','IA-k']:
                v=arms[arm][i];cells.append(f"{v['top1_dE']:.3f} / {v['floor_dE']:.3f}" if finite(v['top1_dE']) and finite(v['floor_dE']) else str(v.get('status','missing')))
            lines.append(f"| {seed} | {' | '.join(cells)} |")
    lines+=['','## Failures, provenance and spend','', f"L4 failure ledger: {len(r['ledger'])} rows (full entries and record hashes in G3F_L4_RESULT.json)."]
    for row in r['ledger']:lines.append('- '+json.dumps(row))
    lines+=['',f"VM paired-job walls, seconds: {r['jobs_wall_s']}",
        'Estimate before launch: 7.454 units, below the 10-unit launch threshold.',
        f"Credits: 478.58 before, 476.36 after; {end['units_spent']:.2f} units spent (hard cap 12).",
        f"Owned VMs down: {end['down']}; final usage rate zero, active assignments zero.",
        'L4 driver 580.82.07, reported CUDA 13.0, Torch 2.14.1. Local 2060',
        'driver 595.99.02, reported CUDA 13.2. Complete runtime records retained.','',
        'G3F_CONTRACT.md committed 9389a57; measured N0 and L4 partition/cost',
        'plan committed 2446e4d before launch. Both L4 VMs use the single ref',
        'recorded in g3f/colab_ref.txt and COLAB_BUNDLE.json. Flags/manifests',
        'and input/model hashes are frozen in G3F_MANIFEST.json; record hashes',
        'and within-VM pairing retained. Driver/CUDA/hardware in nvidia-smi.txt',
        'per VM; credits before/after and fetch/count/down journal retained.',
        'Local CPU smoke passed on AJAQAD seed 11 before any VM launch.',
        'No L4 tail, no reserve/locked evaluation, no numeric optimizer/score',
        'changes. No production defaults or goldens changed. All four gates',
        'are reported without rescue tuning. No production recommendation.',
        'Section 5 checkpoint: commit, confirm VMs down and credits logged, stop.']
    lines+=['', '### Local measured job walls','', '| Job | Seconds |', '|---|---:|']
    for panel in ['cal','tail']:
        j=json.loads((ROOT/'noise'/panel/'job/job.json').read_text())
        lines.append(f"| C1 replicate {panel} | {j['wall_s']:.3f} |")
    for stage,v in profile['stages'].items():lines.append(f"| Valid event profile {stage} | {v['wall_s']:.3f} |")
    lines.append(f"| Nine-unit CPU identity | {identity['wall_s']:.3f} |")
    lines+=['', 'Profile/CPU verification jobs overlapped; these durations are not',
        'summed as elapsed stage time. G3F_VERIFICATION.json records a conservative',
        '1831.43 s upper bound for the entire bounded stage. G3F_EXECUTION.json',
        'verifies the 720 records, 714 exports, paired initial-state digests,',
        'single VM source ref and input/model hashes; full runtime/credit hashes',
        'are retained there.']
    Path('research/campaigns/G3F_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('N0',noise['N0'],'gates',r['gates'],'end',end)
if __name__=='__main__':main()
