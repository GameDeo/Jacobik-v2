"""Frozen clean24 predicates; all fixed parents retained in the failure ledger."""
import argparse,json
from pathlib import Path
import numpy as np


def paired(arm,control,parents,field):
    values=[];parent_values=[];missing=[]
    for code in parents:
        per=[]
        for seed in [11,12,13]:
            a=arm.get((code,seed));c=control.get((code,seed))
            if not a or not c or a['status']!='ok' or c['status']!='ok':
                missing.append([code,seed]);continue
            x=a.get(field);y=c.get(field)
            if x is None or y is None or not np.isfinite([x,y]).all():
                missing.append([code,seed]);continue
            per.append(x-y);values.append(x-y)
        if len(per)==3:parent_values.append(np.mean(per))
    out=dict(declared_parents=len(parents),declared_units=3*len(parents),finite_pairs=len(values),missing=missing,
             median=float(np.median(values)) if values else None,mean=float(np.mean(values)) if values else None)
    if parent_values:
        a=np.asarray(parent_values);rng=np.random.default_rng(11)
        means=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
        out.update(complete_paired_parents=len(a),parent_mean=float(a.mean()),parent_clustered_ci95=np.quantile(means,[.025,.975]).tolist())
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    parents=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/panel_G3b2_clean24_2026-10-30.txt').read_text().split()
    names=['C1','IA-k'];records={};ledger=[];stops=[]
    for name in names:
        records[name]={}
        for code in parents:
            for seed in [11,12,13]:
                path=args.root/name/'records'/f'{code}_{seed}.json'
                if not path.exists():ledger.append(dict(arm=name,code=code,seed=seed,status='missing'));continue
                row=json.loads(path.read_text());records[name][code,seed]=row
                if row.get('status')!='ok':ledger.append(dict(arm=name,code=code,seed=seed,status=row.get('status'),reason=row.get('reason')))
                if row.get('replay_fallbacks',0) or (code!='NEPNEF' and 'GFN2 embed refinement failed' in (row.get('reason') or '')):
                    stops.append(dict(arm=name,code=code,seed=seed,reason='replay_or_nonexempt_embed_fallback'))
    def angle(name):
        values=[d['max_q2_deviation_deg'] for r in records[name].values() if r.get('status')=='ok' for d in (r.get('distortion') or []) if np.isfinite(d['max_q2_deviation_deg'])]
        return dict(rows=len(values),p95=float(np.quantile(values,.95)) if values else None)
    cangle=angle('C1');arms={}
    for name in names[1:]:
        top=paired(records[name],records['C1'],parents,'top1_dE');floor=paired(records[name],records['C1'],parents,'floor_dE');a=angle(name)
        passed=(top['median'] is not None and floor['mean'] is not None and a['p95'] is not None and cangle['p95'] is not None
                and top['median']<=0 and floor['mean']<=0 and a['p95']<=cangle['p95']+10)
        sensitivity=[p for p in parents if p not in ['FEKKAJ','NUWFAM']]
        arms[name]=dict(top1=top,floor=floor,angle=a,control_angle=cangle,eligible=passed,
                        excluding_FEKKAJ_NUWFAM=dict(top1=paired(records[name],records['C1'],sensitivity,'top1_dE'),floor=paired(records[name],records['C1'],sensitivity,'floor_dE')))
    advanced=[name for name in ['IA-k'] if arms[name]['eligible']][:3]
    result=dict(parents=parents,seeds=[11,12,13],expected_records=len(parents)*3*len(names),actual_records=sum(len(v) for v in records.values()),
                ledger=ledger,fallback_stops=stops,arms=arms,advanced=advanced,
                previous_G3d_failures=dict(step0=['FA','FR+'],clean24=['R2s','IA']))
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('advanced',advanced,flush=True)

if __name__=='__main__':main()
