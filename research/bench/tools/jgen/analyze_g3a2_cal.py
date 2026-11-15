"""Frozen parent-clustered cal analysis; also compares preselected 2060 pairs."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def clustered(records,field,stratum='all'):
    grouped=defaultdict(list)
    for r in records:
        if stratum=='all' or r['stratum']==stratum:grouped[r['code']].append(r[field])
    if not grouped:return {'parents':0,'units':0,'mean':None,'median':None,'ci95':None}
    means=np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    rng=np.random.default_rng(11);boot=means[rng.integers(0,len(means),(10000,len(means)))].mean(1)
    return {'parents':len(means),'units':sum(map(len,grouped.values())),'mean':float(means.mean()),
            'median':float(np.median([v for k in sorted(grouped) for v in grouped[k]])),
            'median_parent_mean':float(np.median(means)),'ci95':np.percentile(boot,[2.5,97.5]).tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--roots',nargs='+',type=Path,required=True)
    ap.add_argument('--contract',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();contract=json.loads(args.contract.read_text());rows={}
    for root in args.roots:
        for path in sorted(root.rglob('records/*.json')):
            r=json.loads(path.read_text());key=(r['arm'],r['code'],int(r['seed']))
            assert key not in rows,f'duplicate unit {key}'
            rows[key]=r
    expected={ (c,s) for c in contract['cal']['parents'] for s in contract['cal']['seeds'] }
    arms=contract['admitted_arms'];summary={'arms':{},'expected_units_per_arm':len(expected),'failures':{},'stratum_definition':'control unit top1 >200 is O; otherwise N'}
    collection_complete=all((arm,code,seed) in rows for arm in ['C0','C1']+arms for code,seed in expected)
    summary['collection_complete']=collection_complete
    for arm in ['C0','C1']+arms:
        have=[r for (a,c,s),r in rows.items() if a==arm]
        summary['failures'][arm]={'missing':sorted([list(k) for k in expected if (arm,*k) not in rows]),
                                 'failed':[{'code':r['code'],'seed':r['seed'],'reason':r.get('reason')} for r in have if r['status']!='ok']}
    for arm in arms:
        control=contract['arms'][arm]['control'];pairs=[];guard=defaultdict(list);candidate_only=[];common=[]
        for code,seed in sorted(expected):
            a=rows.get((arm,code,seed));c=rows.get((control,code,seed))
            if c is None or c['status']!='ok' or c.get('n_finite')!=c.get('n_rows'):common.append([code,seed]);continue
            if a is None or a['status']!='ok' or a.get('n_finite')!=a.get('n_rows'):candidate_only.append([code,seed]);continue
            for key in ('gpu_name','machine','mace_model_sha256','bundle_commit','bench_runner_sha256'):
                assert a.get(key)==c.get(key),f'unpaired platform/source {code}/{seed}/{key}'
            obscene=c['top1_dE']>200
            d=a['top1_dE']-c['top1_dE']
            if not obscene and d>25:guard[code].append(seed)
            ae=np.asarray(a['energies_kj'])-a['ref_energy_kj'];ce=np.asarray(c['energies_kj'])-c['ref_energy_kj']
            pairs.append({'code':code,'seed':seed,'stratum':'O' if obscene else 'N','top1':d,'floor':a['floor_dE']-c['floor_dE'],
                          'score':a['top1_score']-c['top1_score'],'obscene_rate':float((ae>200).mean()-(ce>200).mean()),
                          'arm':a,'control':c})
        stats={stratum:{m:clustered(pairs,m,stratum) for m in ('top1','floor','score','obscene_rate')} for stratum in ('all','N','O')}
        diagnostic_without_fekkaj={m:clustered([r for r in pairs if r['code']!='FEKKAJ'],m) for m in ('top1','floor','score','obscene_rate')}
        unique={r['code']:r for r in pairs}
        cw=sum(r['control']['wall_solve_s'] for r in unique.values());aw=sum(r['arm']['wall_solve_s'] for r in unique.values())
        wall=aw/cw if cw>0 else None
        violations={c:seeds for c,seeds in guard.items() if len(seeds)>=2}
        primary=stats['all']['top1']['ci95'];floor=stats['all']['floor']['ci95'];obscene=stats['all']['obscene_rate']['ci95']
        numerical_pass=primary is not None and primary[1]<2 and ((floor is not None and floor[1]<0) or (obscene is not None and obscene[1]<0))
        replay_ok=all(r[name].get('replay_chunks') and all(ch.get('mode')=='replay' and not ch.get('error') for ch in r[name]['replay_chunks']) for r in pairs for name in ('arm','control'))
        passed=bool(collection_complete and numerical_pass and not violations and wall is not None and wall<=1.3 and not candidate_only)
        worst={}
        for name in ('arm','control'):
            if pairs:
                pr=max(pairs,key=lambda r:max(np.asarray(r[name]['energies_kj'])-r[name]['ref_energy_kj']))
                rec=pr[name];es=np.asarray(rec['energies_kj'])-rec['ref_energy_kj'];idx=int(es.argmax())
                worst[name]={'code':pr['code'],'seed':pr['seed'],'energy_kj':float(es[idx]),'row_key':rec.get('row_keys',[None]*len(es))[idx]}
        summary['arms'][arm]={'control':control,'paired_units':len(pairs),'common_control_unavailable':common,'candidate_only_unavailable':candidate_only,
            'statistics':stats,'FEKKAJ_excluded_diagnostic_non_gating':diagnostic_without_fekkaj,'guard_violations':violations,'wall_ratio':wall,'numerical_pass':bool(numerical_pass),'replay_ok':replay_ok,'pass':passed,'worst':worst,
            'paired_rows':[{k:v for k,v in r.items() if k not in ('arm','control')} for r in pairs]}
        companion=arm+'_early'
        early_pairs=[];unchanged=True;saved=0;offtime=ontime=0
        for code,r in unique.items():
            e=rows.get((companion,code,int(r['seed'])))
            if e is not None and e['status']=='ok':
                offtime+=r['arm']['wall_solve_s'];ontime+=e['wall_solve_s']
        for r in pairs:
            e=rows.get((companion,r['code'],int(r['seed'])))
            if e is None or e['status']!='ok':continue
            n_saved=int(e.get('early_stop_saved_steps',0));saved+=n_saved
            if n_saved==0:unchanged &= e['scores']==r['arm']['scores'] and e['energies_kj']==r['arm']['energies_kj']
            early_pairs.append([r['code'],r['seed']])
        if early_pairs:summary['arms'][arm]['early_stop']={'paired_units':len(early_pairs),'time_ratio_on_off':ontime/offtime if offtime else None,'saved_steps_summed_over_records':saved,'no_skipped_steps_outputs_exact':bool(unchanged)}
    p2=summary['arms'].get('P2',{});p2mean=p2.get('statistics',{}).get('all',{}).get('top1',{}).get('mean')
    preferred=[]
    for arm in ('P2p','P2a','P2prime'):
        a=summary['arms'].get(arm)
        if a and a['pass'] and p2mean is not None and a['statistics']['all']['top1']['mean']<=p2mean+3:preferred.append(arm)
    summary['provisional_winner_before_2060']=preferred[0] if preferred else ('P2' if p2.get('pass') else None)
    args.out.write_text(json.dumps(summary,indent=2));print(json.dumps({a:{'pass':r['pass'],'primary':r['statistics']['all']['top1'],'wall_ratio':r['wall_ratio']} for a,r in summary['arms'].items()},indent=2))

if __name__=='__main__':main()
