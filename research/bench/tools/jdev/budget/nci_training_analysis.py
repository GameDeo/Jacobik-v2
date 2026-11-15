"""Case-weighted energy-response calibration with an untouched validation split.

This is a diagnostic, not a DB installer. Descriptor failures in independent
physical probes override an apparent scalar fit. No development-pool tuning.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from nci_training_calibration import OUT, write, digest

KT = 2.479
CONTRACT = dict(
    fit='One nonnegative scalar per type, equal case weights, relative to the unchanged reference; calibration split only.',
    models='hb removes Q6 polar-H wells; pp/chp preserve all old wells. Unselected NCI types excluded.',
    metrics='Per-case relative-energy RMSE and per-mode sampled minimum regret; central gradients and two-step curvature.',
    gate='Require independently adequate descriptors and a strictly positive lower 95% paired case-bootstrap RMSE improvement on validation. No worsening of mean sampled-minimum regret or wrong-sign stable curvature count. Otherwise no candidate gate/arm.',
    uncertainty='10000 paired case bootstrap replicates, seed 20261018; the 16 validation cases are independent parents/scaffold contact pairs, modes are not independent samples.',
    limitations='Rigid training-reference motions and fixed reference keys; no bend/closure prior, solver coverage or off-reference assignment validation. Full-complex MACE response is not uniquely attributable to a selected contact.'
)


def contract():
    p = OUT / 'analysis_contract.json'
    expected = dict(**CONTRACT, selection_sha256=digest(OUT/'selection.json'))
    if p.exists():
        assert json.loads(p.read_text()) == expected, 'analysis contract changed'
    else:
        write(p, expected)


def load():
    selection = json.loads((OUT/'selection.json').read_text())
    assert selection['coverage_pass']
    for key in ('code','scaffolds'):
        sides={s:{v for r in selection['selected'] if r['split']==s for v in (r['scaffolds'] if key=='scaffolds' else [r['code']])} for s in ('calibration','validation')}
        assert not (sides['calibration']&sides['validation']),(key,'split leakage')
    complete=json.loads((OUT/'measurement_complete.json').read_text())
    manifest=json.loads((OUT/'measurement_manifest.json').read_text())
    assert complete['manifest_sha256']==digest(OUT/'measurement_manifest.json')
    data = []
    for r in selection['selected']:
        p = OUT / f"{r['type']}_{r['code']}.npz"
        meta = json.loads(p.with_suffix('.json').read_text())
        assert meta['selection_sha256'] == digest(OUT/'selection.json')
        assert manifest['coordinates_and_scores'][p.name]==digest(p)
        assert complete['energies'][p.with_suffix('.mace64.json').name]==digest(p.with_suffix('.mace64.json'))
        with np.load(p) as z:
            old = -KT*z['off_total'].astype(float)
            # Q6 entries are signed score contributions, hence subtract the
            # signed energy change, not the unsigned repulsion magnitude.
            base = old - KT*(z['on_Q6'].astype(float)-z['off_Q6'].astype(float)) if r['type']=='hb' else old.copy()
            nci = -KT*z['NCI_'+r['type']].astype(float)
            all_on = -KT*z['on_total'].astype(float)
        e = np.array(json.loads(p.with_suffix('.mace64.json').read_text())['relative'])
        assert len(e) == len(old) == 1+len(meta['modes'])*len(meta['steps'])
        shift = lambda x: x-x[0]
        data.append(dict(request=r, meta=meta, energy=e, old=shift(old), base=shift(base), nci=shift(nci), all_on=shift(all_on)))
    return data


def paired_ci(values):
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(20261018)
    draws = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws,[.025,.975])]


def metrics(d, pred):
    from scipy.stats import spearmanr
    e = d['energy']; steps = np.array(d['meta']['steps']); n = len(steps)
    assert np.allclose(steps,[-.16,-.08,-.04,-.02,.02,.04,.08,.16])
    regrets=[]; grads=[]; ks=[]; stable=[]
    for j in range(len(d['meta']['modes'])):
        ids=np.r_[0,1+j*n+np.arange(n)]
        regrets.append(float(e[ids[np.argmin(pred[ids])]]-e[ids].min()))
        def deriv(y,h):
            a=1+j*n+int(np.flatnonzero(np.isclose(steps,-h))[0]); b=1+j*n+int(np.flatnonzero(np.isclose(steps,h))[0])
            return (y[b]-y[a])/(2*h),(y[b]+y[a]-2*y[0])/h**2
        eg,ek=deriv(e,.04);pg,pk=deriv(pred,.04)
        _,ek2=deriv(e,.08)
        if abs(eg)>.25:grads.append(bool(eg*pg>0))
        ks.append((ek,pk))
        # Identical physical directions for every compared model. Conditioning
        # on predicted stability would discard some of the model's own errors.
        stable.append(bool(ek>.1 and abs(ek-ek2)<.25*max(abs(ek),1)))
    k=np.array(ks);mask=np.array(stable)
    pos=(k[:,0]>.1)&(k[:,1]>.1)&mask
    correlation=float(spearmanr(e[1:],pred[1:]).statistic) if np.ptp(pred[1:])>1e-10 and np.ptp(e[1:])>1e-10 else None
    den=float(pred[1:]@pred[1:]);slope=float(pred[1:]@e[1:])/den if den>1e-12 else None
    return dict(rmse=float(np.sqrt(np.mean((e[1:]-pred[1:])**2))),spearman_energy_vs_surrogate=correlation,
        response_energy_per_nat_kj=KT*slope if slope is not None else None,
        sampled_minimum_regret=float(np.mean(regrets)),
        gradient_correct=int(sum(grads)),gradient_tested=len(grads),
        stable_positive_energy_curvatures=int(mask.sum()),wrong_sign_curvatures=int(((k[:,1]<=0)&mask).sum()),
        curvature_energy_over_surrogate=[float(v) for v in k[pos,0]/k[pos,1]])


def analyze():
    contract();data=load();summary={};rows=[]
    for typ in ('hb','pp','chp'):
        group=[d for d in data if d['request']['type']==typ]
        train=[d for d in group if d['request']['split']=='calibration']
        assert len(train)==16 and len(group)==32
        numerator=sum(float(d['nci'][1:]@(d['energy'][1:]-d['base'][1:]))/(len(d['energy'])-1) for d in train)
        denominator=sum(float(d['nci'][1:]@d['nci'][1:])/(len(d['energy'])-1) for d in train)
        coefficient=max(0.,numerator/denominator) if denominator>1e-12 else 0.
        result=dict(frozen_coefficient=coefficient,calibration_numerator=numerator,calibration_denominator=denominator,splits={})
        for split in ('calibration','validation'):
            ds=[d for d in group if d['request']['split']==split];mr=[]
            for d in ds:
                predictions=dict(off=d['old'],all_on=d['all_on'],component=d['base']+d['nci'],calibrated=d['base']+coefficient*d['nci'])
                m={name:metrics(d,p) for name,p in predictions.items()}
                row=dict(code=d['request']['code'],type=typ,split=split,metrics=m);rows.append(row);mr.append(m)
            stats={}
            for name in ('off','all_on','component','calibrated'):
                ratios=[v for m in mr for v in m[name]['curvature_energy_over_surrogate']]
                stats[name]=dict(mean_case_rmse=float(np.mean([m[name]['rmse'] for m in mr])),
                    mean_sampled_minimum_regret=float(np.mean([m[name]['sampled_minimum_regret'] for m in mr])),
                    gradient_correct=sum(m[name]['gradient_correct'] for m in mr),gradient_tested=sum(m[name]['gradient_tested'] for m in mr),
                    stable_positive_energy_curvatures=sum(m[name]['stable_positive_energy_curvatures'] for m in mr),
                    wrong_sign_curvatures=sum(m[name]['wrong_sign_curvatures'] for m in mr),
                    median_curvature_ratio=float(np.median(ratios)) if ratios else None)
                for key in ('spearman_energy_vs_surrogate','response_energy_per_nat_kj'):
                    vals=[m[name][key] for m in mr if m[name][key] is not None]
                    stats[name]['median_'+key]=float(np.median(vals)) if vals else None
                if name!='off':
                    delta=[m['off']['rmse']-m[name]['rmse'] for m in mr]
                    stats[name]['rmse_improvement_ci95']=paired_ci(delta)
                    stats[name]['cases_better_worse_tied']=[int(sum(v>1e-6 for v in delta)),int(sum(v< -1e-6 for v in delta)),int(sum(abs(v)<=1e-6 for v in delta))]
            result['splits'][split]=stats
        val=result['splits']['validation'];v=val['calibrated'];off=val['off']
        # Curvature band was frozen in the physical-probe CONFIG before any
        # measurements. A pooled median is necessary here, never sufficient.
        ratio=v['median_curvature_ratio']
        result['numerical_gate_pass']=bool(coefficient>0 and v['rmse_improvement_ci95'][0]>0 and v['mean_sampled_minimum_regret']<=off['mean_sampled_minimum_regret'] and v['wrong_sign_curvatures']<=off['wrong_sign_curvatures'] and ratio is not None and .7<=ratio<=1.5)
        summary[typ]=result
    write(OUT/'analysis.json',dict(contract=CONTRACT,summary=summary,cases=rows))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    contract() if '--freeze-contract' in sys.argv else analyze()
