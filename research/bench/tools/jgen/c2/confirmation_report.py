"""Report the once-only C-2b confirmation and descriptive arm-vs-K contrasts.

Never generates structures, changes a frozen route, or retries a run.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .analyse import unit, bootstrap
from .matched_rho import matched_rho

BASE=Path('research/campaigns')
LABELS=[('floor','Floor delta-E (kJ/mol)',1),('top1','Top-1 delta-E (kJ/mol)',1),
 ('hit1','Coverage <=1 A (pp)',100),('hit05','Coverage <=0.5 A (pp)',100),
 ('min_rmsd','Minimum heavy RMSD (A)',1),('min_shell','Minimum shell RMSD (A)',1),
 ('shell03','Shell <=0.3 A (pp)',100),('shell05','Shell <=0.5 A (pp)',100),
 ('distinct','Distinct rows',1),('rho','Score-energy rho',1),('matched_rho','Size-matched rho',1)]


def finite(v):return v is not None and np.isfinite(v)


def descriptive(root, arm, baseline, settings, codes):
    """Paired parent seed means, including failure denominators; no adoption route."""
    values={};failures=[];walls=[];top={}
    for code in codes:
        records={a:[json.loads((root/a/'records'/f'{code}_{s}.json').read_text())
                    for s in settings['seeds']] for a in [arm,baseline]}
        au=[unit(r) for r in records[arm]];bu=[unit(r) for r in records[baseline]]
        bad=any(x['failed'] for x in au+bu)
        if bad:failures.append(code)
        metrics={}
        for key,_,_ in LABELS:
            if key=='matched_rho':continue
            pairs=[(x.get(key),y.get(key)) for x,y in zip(au,bu)]
            valid=[(x,y) for x,y in pairs if finite(x) and finite(y)]
            if bad and key not in ['hit1','hit05','shell03','shell05','distinct']:valid=[]
            metrics[key]=dict(delta=float(np.mean([x-y for x,y in valid])) if valid else None,
                arm=float(np.mean([x for x,y in valid])) if valid else None,
                baseline=float(np.mean([y for x,y in valid])) if valid else None,n_seed_pairs=len(valid))
        matched=[]
        if not bad:
            for a,b in zip(records[arm],records[baseline]):
                v=matched_rho(a['scores'],a['energies_kj'],b['scores'],b['energies_kj'],
                    draws=settings['rho_draws'],seed=settings['rho_seed'])
                if finite(v):matched.append(v)
        metrics['matched_rho']=dict(delta=float(np.mean(matched)) if matched else None,n_seed_pairs=len(matched))
        values[code]=metrics
        if code not in settings['primary_exclusions'] and not bad:
            walls.append((sum(x['wall'] for x in au),sum(x['wall'] for x in bu)))
            if all(finite(x['top1']) for x in au+bu):
                top[code]=np.array([x['top1']-y['top1'] for x,y in zip(au,bu)],float)
    primary=[c for c in codes if c not in settings['primary_exclusions']]
    aggregate={}
    for key in values[codes[0]]:
        aggregate[key]=bootstrap([values[c][key]['delta'] for c in primary if finite(values[c][key]['delta'])],settings)
        if key!='matched_rho':
            for side in ['arm','baseline']:
                v=[values[c][key][side] for c in primary if finite(values[c][key][side])]
                aggregate[key][side+'_mean']=float(np.mean(v)) if v else None
    ratio=None;ci=None
    if walls:
        w=np.array(walls,float);ratio=float(w[:,0].sum()/w[:,1].sum())
        rng=np.random.default_rng(settings['bootstrap_seed'])
        ix=rng.integers(0,len(w),(settings['bootstrap_resamples'],len(w)))
        ci=np.quantile(w[ix,0].sum(1)/w[ix,1].sum(1),[.025,.975]).tolist()
    counts={}
    for label,excluded in [('with_artifacts',[]),('without_artifacts',settings['guard_artifacts'])]:
        use={c:d for c,d in top.items() if c not in excluded}
        worse=[c for c,d in use.items() if (d>settings['regression_threshold_kj']).sum()>=settings['regression_min_seeds']]
        better=[c for c,d in use.items() if (d < -settings['regression_threshold_kj']).sum()>=settings['regression_min_seeds']]
        counts[label]=dict(worse=worse,better=better,n_parents=len(use))
    return dict(arm=arm,baseline=baseline,descriptive_only=True,metrics=aggregate,
        wall_ratio=ratio,wall_ci=ci,severity_counts=counts,failures=failures,per_parent=values)


def number(v,scale=1):return 'undefined' if not finite(v) else f'{v*scale:.3f}'


def estimate(m,scale=1):
    if m['mean'] is None:return 'undefined'
    return number(m['mean'],scale)+' ['+', '.join(number(x,scale) for x in m['ci'])+']'


def metric_table(result, primary, baseline='C'):
    lines=[f'| Metric | {baseline} | {result["arm"]} | Difference [95% CI] | N parents |',
           '|---|---:|---:|---:|---:|']
    for key,label,scale in LABELS:
        m=result['metrics'][key]
        sides={}
        for field in ['arm','control']:
            v=[result['per_parent'][c][key].get(field) for c in primary]
            v=[x for x in v if finite(x)]
            sides[field]=float(np.mean(v)) if v else None
        # Coverage side means are percentages, paired differences are pp.
        control=number(sides['control'],scale) if key!='matched_rho' else 'not reported'
        arm=number(sides['arm'],scale) if key!='matched_rho' else 'not reported'
        lines.append(f'| {label} | {control} | {arm} | {estimate(m,scale)} | {m["n"]} |')
    return lines


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    manifest=json.loads((BASE/'c2/C2B_FINALISTS.json').read_text())
    settings=json.loads((BASE/'c2/C2B_SETTINGS.json').read_text())
    panel=json.loads((BASE/'c2/C2_PANEL.json').read_text());codes=panel['codes']
    primary=[c for c in codes if c not in settings['primary_exclusions']]
    formal={arm['arm']:json.loads((a.root/f'analysis_{arm["arm"]}.json').read_text()) for arm in manifest['arms'][1:]}
    contrasts={arm+'_vs_K':descriptive(a.root,arm,'K',settings,codes) for arm in ['P','KU']}
    failures=[];status={};warnings={};named={};fingerprints={arm['arm']:arm['fingerprint'] for arm in manifest['arms']}
    for arm in fingerprints:
        counts=Counter();warning_totals=Counter()
        for code in codes:
            records=[json.loads((a.root/arm/'records'/f'{code}_{s}.json').read_text()) for s in settings['seeds']]
            for r in records:
                counts[r['status']]+=1
                assert r['fingerprint']==fingerprints[arm]
                if r['status']!='ok':failures.append(dict(arm=arm,code=code,seed=r['seed'],reason=r.get('reason')))
                if code in manifest['named_cases']:
                    named.setdefault(code,{}).setdefault(arm,[]).append({k:r.get(k) for k in ['seed','status','top1_dE','floor_dE','n_rows','reason']})
            # Bench deltas are parent/arm totals repeated in the three records.
            for key in set(k for r in records for k in r.get('infeasible_warnings',{})):
                warning_totals[key]+=max(r.get('infeasible_warnings',{}).get(key,0) for r in records)
        status[arm]=dict(counts);warnings[arm]=dict(warning_totals)
    load=[json.loads(line) for line in (a.root/'load_context.jsonl').read_text().splitlines()]
    completion=json.loads((a.root/'completion.json').read_text())
    launch=json.loads((a.root/'launch.json').read_text())
    events=[json.loads(line) for line in (a.root/'execution_events.jsonl').read_text().splitlines()] if (a.root/'execution_events.jsonl').exists() else []
    checks=json.loads((a.root/'compute_checks.json').read_text())
    qualified=[arm for arm,r in formal.items() if r['qualified']]
    interpretation=('K/KU qualification without P supports deliberate pose perturbation, not shipping the pivot bug. '
        'Adoption requires a documented setting reproducing K bit-for-bit, then the D3 joint gate.'
        if any(x in qualified for x in ['K','KU']) and 'P' not in qualified else
        'P qualifies as the correct start_kabsch implementation; all adoption still awaits the D3 joint gate.'
        if 'P' in qualified else 'No finalist qualifies under its declared route. No route rescue or adoption.')
    result=dict(status='once-only confirmation complete',formal=formal,descriptive=contrasts,
        named_cases=named,record_status=status,failures=failures,parent_arm_warning_totals=warnings,
        compute_checks=checks,load_context=load,launch=launch,completion=completion,
        qualified=qualified,interpretation=interpretation,execution_events=events,units_spent=0,reruns=0)
    (BASE/'c2/C2B_CONFIRMATION.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['## Once-only confirmation — completed','',
        'One invocation, full frozen population, C/K/P/KU/S1, seeds 21–23. No rerun, relaxation, L4 replicate or production fold-in. '
        'Differences are arm minus C; energies kJ/mol, coverage percentage points. '
        'Per-parent seed means, 10,000 parent-cluster resamples, seed 11, percentile 95% CIs. '
        'Undefined values are not imputed; failures remain in coverage/distinct denominators. '
        'Gates use full precision; displayed values are rounded.','',
        '**Qualified:** '+(', '.join(qualified) if qualified else 'none')+'. '+interpretation,'',
        '| Arm / route | Floor difference [CI] | Top-1 difference [CI] | Wall ratio [CI] | Frozen gates | Qualified |',
        '|---|---|---|---|---|---|']
    for arm,r in formal.items():
        gates=', '.join(k+(' PASS' if v else ' FAIL') for k,v in r['gates'].items())
        wall=number(r['wall_ratio'])+' ['+', '.join(number(x) for x in (r['wall_ci'] or []))+']'
        lines.append(f'| {arm} / {r["route"]} | {estimate(r["metrics"]["floor"])} | {estimate(r["metrics"]["top1"])} | {wall} | {gates} | {r["qualified"]} |')
    lines+=['','### Guard counts','',
        'Decision uses the registered artefact-excluded version. Artefacts stay in all other metrics. '
        'Each cell lists trips / mirrors / arm null; C rotation null also reported.','',
        '| Arm | With artefacts | Without artefacts | C null (with / without) | Decision guard |',
        '|---|---|---|---|---|']
    for arm,r in formal.items():
        w=r['guard']['with_artifacts'];o=r['guard']['without_artifacts']
        show=lambda g:f'{g["trips"]} / {g["mirror"]} / {g["null"]}'
        lines.append(f'| {arm} | {show(w)} | {show(o)} | {w["control_null"]} / {o["control_null"]} | {o["passed"]} |')
    for arm,r in formal.items():
        lines+=['',f'### {arm}: all registered metrics','']+metric_table(r,primary)
        g=r['guard']['without_artifacts']
        lines+=['','Artefact-excluded trips: '+(', '.join(g['trip_parents']) or 'none')+
            '; mirrors: '+(', '.join(g['mirror_parents']) or 'none')+'.']
    lines+=['','### P versus K and KU versus K — descriptive only','',
        'These paired contrasts do not define an adoption route or rescue a failed route. '
        'Same primary population, paired seeds and bootstrap; differences are first arm minus K. '
        'Size matching uses the fixed 20 draws / seed 11 procedure on corresponding per-seed pools.','']
    for name,r in contrasts.items():
        lines += [f'#### {name}','', '| Metric | Difference [95% CI] | N parents |','|---|---:|---:|']
        for key,label,scale in LABELS:
            m=r['metrics'][key];lines.append(f'| {label} | {estimate(m,scale)} | {m["n"]} |')
        lines+=['','Observed generator wall ratio versus K: '+number(r['wall_ratio'])+
            ' ['+', '.join(number(x) for x in (r['wall_ci'] or []))+'].']
        for label,g in r['severity_counts'].items():
            lines.append(f'Descriptive >25 kJ/mol on >=2/3 seeds ({label}): worse {len(g["worse"])}; better {len(g["better"])}.')
    lines+=['','### Named cases','', 'Each cell is top-1 / floor kJ/mol; failed results remain explicit.','',
        '| Parent | Seed | C | K | P | KU | S1 |','|---|---:|---|---|---|---|---|']
    for code in manifest['named_cases']:
        for seed in settings['seeds']:
            cells=[]
            for arm in fingerprints:
                r=next(x for x in named[code][arm] if x['seed']==seed)
                cells.append(number(r['top1_dE'])+' / '+number(r['floor_dE']) if r['status']=='ok' else 'FAILED: '+str(r['reason']))
            lines.append('| '+code+' | '+str(seed)+' | '+' | '.join(cells)+' |')
    lines+=['','### Load context and execution','',
        'Start, approximately 20-minute snapshots and completion are preserved in load_context.jsonl and per-snapshot nvidia-smi files. '
        'Samples cannot rule out load changes between them; wall ratios describe the observed environment. '
        'Other jobs were recorded, never used to stop or rerun confirmation.','',
        '| UTC | Elapsed min | Load 1 / 5 / 15 | CPU % | Available RAM GiB | Swap used GiB |',
        '|---|---:|---|---:|---:|---:|']
    for l in load:
        lines.append(f'| {l["utc"]} | {l["elapsed_s"]/60:.1f} | '+ ' / '.join(f'{x:.2f}' for x in l['load_average'])+
            f' | {l["cpu_percent"]:.1f} | {l["ram"]["available"]/2**30:.2f} | {l["swap"]["used"]/2**30:.2f} |')
    observed_jobs={}
    for l in load:
        for process in l['top_cpu']:
            if process['pid'] in [launch['monitor_pid'],launch['driver_pid']]:continue
            if 'fit.typing_reextract' not in process['command'] and 'scripts/run_jobs.py' not in process['command']:continue
            item=observed_jobs.setdefault(process['pid'],dict(command=process['command'],first=l['utc'],last=l['utc'],maximum_cpu=0.))
            item['last']=l['utc'];item['maximum_cpu']=max(item['maximum_cpu'],process['cpu_percent'])
    lines+=['','Other extraction/Architector jobs observed at sampled times:', '',
        '| PID | First / last observed UTC | Maximum sampled process CPU % | Command |','|---:|---|---:|---|']
    for pid,item in observed_jobs.items():
        lines.append(f'| {pid} | {item["first"]} / {item["last"]} | {item["maximum_cpu"]:.1f} | '+item['command'].replace('|','\\|')+' |')
    if any('fit.typing_reextract' in item['command'] for item in observed_jobs.values()):
        lines+=['','T1d extraction appeared after launch. The run continued without interruption, as the audit required. '
            'The host was not quiet throughout; performance ratios have this material limitation. '
            'The frozen numerical wall gate was still applied exactly, without a load correction or rerun.']
    for event in events:
        lines+=['',event['utc']+': '+event['event']+'. '+event['action']+'.']
    lines+=['','Driver return code: '+str(completion['returncode'])+'. Observed process wall (including MACE/export/startup): '+
        number(completion['elapsed_s']/60)+' min. Colab spend: 0 units.','',
        'Record status: '+json.dumps(status,sort_keys=True)+'.',
        'Equal-compute checks: '+str(sum(c['passed'] is True for c in checks))+' pass, '+
        str(sum(c['passed'] is False for c in checks))+' fail, '+str(sum(c['passed'] is None for c in checks))+' unavailable.','',
        'Parent/arm warning totals (not tripled over copied seed metadata): `'+json.dumps(warnings,sort_keys=True)+'`.','',
        'Failure ledger (full reasons also in C2B_CONFIRMATION.json):','']
    lines += [f'- {f["arm"]} {f["code"]} seed {f["seed"]}: {f["reason"]}' for f in failures] or ['- None.']
    report=BASE/'C2B_REPORT.md';old=report.read_text().split('## Once-only confirmation — completed')[0].rstrip()
    old=old.replace('**Stopped at the contract-freeze checkpoint.** Four finalists are declared:',
        '**Once-only confirmation completed after audit clearance.** Development freeze: 5a19ef8. Four finalists were declared:')
    old=old.replace('Confirmation seeds 21–23 have not been run.',
        'Confirmation seeds 21–23 were run once; results follow the development record below.')
    old=old.replace('Audit clearance is required before confirmation.',
        'The audit addendum cleared confirmation; no production adoption or rerun occurred.')
    report.write_text(old+'\n\n'+'\n'.join(lines)+'\n')
    print('Qualified',qualified);print(interpretation)


if __name__=='__main__':main()
