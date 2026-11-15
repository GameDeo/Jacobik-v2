"""Frozen replicate-minus-G3e C1 analysis and calibrated guard threshold."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from tools.jgen.g3e_step5_report import paired,finite,fmt
ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B')
OUT=Path('research/campaigns/g')


def main():
    manifest=json.loads((OUT/'G3F_MANIFEST.json').read_text());results={}
    for panel,parents in manifest['panels'].items():
        records={'C1':{},'IA-k':{}};ledger=[]
        for label,pathroot in [('C1',ROOT/'g3e'/panel/'C1'),('IA-k',ROOT/'g3f/noise'/panel/'C1')]:
            for code in parents:
                for seed in [11,12,13]:
                    p=pathroot/'records'/f'{code}_{seed}.json';r=json.loads(p.read_text())
                    if label=='C1':assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['baseline_records'][str(p)]
                    assert r['fingerprint']=='273f2ccbf892' and r['flags']=={}
                    if code=='NEPNEF':r['status']='invalid_embed';r['embed_fallback']=True
                    if r['status']!='ok':ledger.append(dict(source=label,code=code,seed=seed,status=r['status'],reason=r.get('reason')))
                    else:
                        values=np.asarray(r['energies_kj'])-r['ref_energy_kj'];r['_obscene']=float(np.mean(values>200)) if np.isfinite(values).all() else None
                    records[label][code,seed]=r
        primary=[c for c in parents if c not in ['FEKKAJ','NEPNEF']]
        metrics={f:paired(records,primary,f) for f in ['top1_dE','floor_dE','_obscene']}
        guard=[]
        for c,v in metrics['top1_dE']['per_parent'].items():
            if sum(u['delta'] is not None and u['delta']>25 for u in v['units'])>=2:guard.append(dict(code=c,**v))
        per=metrics['top1_dE']['per_parent'];worst=max((dict(code=c,mean=v['mean']) for c,v in per.items() if v['mean'] is not None),key=lambda v:abs(v['mean']))
        units=[dict(code=c,**u) for c,v in per.items() for u in v['units'] if u['delta'] is not None]
        worst_unit=max(units,key=lambda u:abs(u['delta']))
        results[panel]=dict(metrics=metrics,guard_count=len(guard),guard_failures=guard,largest_absolute_parent_mean=worst,largest_absolute_seed_delta=worst_unit,ledger=ledger,
            excluding_NUWFAM={f:paired(records,[c for c in primary if c!='NUWFAM'],f) for f in ['top1_dE','floor_dE','_obscene']},
            all_included_finite={f:paired(records,parents,f) for f in ['top1_dE','floor_dE','_obscene']})
    n0=results['cal']['guard_count'];complete=results['cal']['metrics']['top1_dE']['complete_parents']==118
    result=dict(panels=results,N0=n0,N0_complete=complete,L4_declared_primary=118,L4_guard_threshold=math.ceil(n0*118/118)+2,rule='ceil(N0 * declared_L4_primary / 118) + 2',bootstrap_resamples=10000,bootstrap_seed=11)
    (OUT/'G3F_NOISE_RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('N0',n0,'complete',complete,'L4 threshold',result['L4_guard_threshold'])
    for p,r in results.items():print(p,'top1',fmt(r['metrics']['top1_dE']),'floor',fmt(r['metrics']['floor_dE']),'guard',[v['code'] for v in r['guard_failures']])
if __name__=='__main__':main()
