"""Consolidate completed measurements; no fitting, new energies or solves."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO
from nci_training_calibration import OUT,write,digest


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((OUT/'analysis.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
    original=json.loads((REPO/'data/refit/nci/2026-10-18_bound_fix/fit_summary.json').read_text())
    assert digest(Path(original['base']))==original['base_sha256'],'default DB changed'
    assert digest(Path(original['candidate']))==original['candidate_sha256'],'candidate DB changed'
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    for ax,typ,title in zip(axes,('hb','pp','chp'),('H-bond replacement','Stacking only','CH–pi only')):
        rows=[r for r in data['cases'] if r['type']==typ and r['split']=='validation']
        for j,(key,label,color) in enumerate([('component','Current unit scale','#718da3'),('calibrated','Calibration-only fit','#b66145')]):
            values=np.array([r['metrics']['off']['rmse']-r['metrics'][key]['rmse'] for r in rows])
            ax.scatter(j+np.linspace(-.08,.08,len(values)),values,color=color,s=16,alpha=.65)
            mean=values.mean();lo,hi=data['summary'][typ]['splits']['validation'][key]['rmse_improvement_ci95']
            ax.errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',capsize=5,color=color,markeredgecolor='black')
        ax.axhline(0,color='#444444',linewidth=.8);ax.set_xticks([0,1],['Unit scale','Fitted scale']);ax.set_title(title)
        ax.set_ylabel('RMSE improvement (kJ/mol); positive = better')
    fig.suptitle('16 scaffold-disjoint validation complexes per type; small motions only')
    target=REPO/'data/refit/reports/n2_calibration_validation_2026-10-18.png'
    fig.savefig(target,dpi=160);plt.close(fig)
    result=dict(status='No component advances; no fixed-pool gate or dev80 arm',dbs_verified_unchanged=dict(base_sha256=original['base_sha256'],candidate_sha256=original['candidate_sha256']),
        calibration=data['summary'],cohort=dict(contact_cases=len(selection['selected']),
            unique_parents=len({r['code'] for r in selection['selected']}),
            poses=sum(len(np.load(p)['X']) for p in OUT.glob('*.npz')),
            parents_disjoint=True,individual_scaffolds_disjoint=True,reused_identical_raw_measurements=16),
        thermal=json.loads((REPO/'data/refit/reports/n2_thermal_audit_2026-10-18/summary.json').read_text())['summary'],
        sampling=json.loads((REPO/'data/refit/reports/n2_density_balance_2026-10-18/summary.json').read_text())['summary'],
        embedding=json.loads((REPO/'data/refit/reports/n2_embedding_validation_2026-10-18/summary.json').read_text()),
        decisions=dict(hb='Small-motion validation signal is positive, but charge-context descriptor collision and scalar-fit curvature failure prevent advancement.',
            pp='Physical dimer signal; scaffold-disjoint whole-complex calibrated improvement CI includes zero.',
            chp='Scalar fit chooses zero; unit component worsens 13 of 16 validation cases and adds wrong-sign curvatures. Direction/sibling probe ablations are partial mechanism evidence, not a supported refit.',
            embedding='Caps rejected; Cp-only restraint release is promising in eight successful independent ligands, broad arene release regresses bond errors in six of eight. No runtime embedding change.'),
        caveats=['First training split incorrectly separated scaffold pairs, allowing five individual cores across splits; marked invalid for validation and replaced by v2.',
                 'Strict v2 selection was a deterministic energy-blind grouping correction. Sixteen identical raw measurements were reused; prior coefficients and validation decisions discarded.',
                 'Scalar fits falsify amplitude-only correction; they do not falsify physically constrained widths or balanced/conditional fitting.',
                 'Cached-Hessian widths and small-cohort density reweighting are diagnostics, not installed floors or full-corpus evidence.'])
    write(REPO/'data/refit/reports/n2_calibration_implementation_2026-10-18.json',result)
    paths=[OUT/'selection.json',OUT/'measurement_manifest.json',OUT/'measurement_complete.json',OUT/'analysis_contract.json',OUT/'analysis.json',target,
        REPO/'data/refit/reports/n2_calibration_implementation_2026-10-18.md',
        REPO/'data/refit/nci/2026-10-18_energy_calibration_v2/manifest.json',
        REPO/'data/refit/nci/2026-10-18_energy_calibration_v2/probe_analysis.json',
        REPO/'data/refit/nci/2026-10-18_energy_calibration_v2/chp_mechanism.json',
        REPO/'data/refit/reports/n2_thermal_audit_2026-10-18/summary.json',
        REPO/'data/refit/reports/n2_density_balance_2026-10-18/summary.json',
        REPO/'data/refit/reports/n2_embedding_validation_2026-10-18/JEFJUZ_factorial.json',
        REPO/'data/refit/reports/n2_embedding_validation_2026-10-18/training_results.json']
    paths.extend(Path(__file__).with_name(name) for name in ('nci_calibration_report.py','nci_calibration.py','nci_calibration_analysis.py','nci_probe_mechanism.py',
        'nci_training_calibration.py','nci_training_analysis.py','nci_embedding_validation.py','nci_thermal_audit.py','nci_density_balance_audit.py'))
    paths.extend(REPO/path for path in ('src/jacobik/profile.py','src/jacobik/stats/nci.py','src/jacobik/stats/q6pmf.py','tests/test_nci_components.py',
        'data/refit/reports/n2_calibration_final_tests_u1_verified_2026-10-18.log','data/refit/reports/n2_calibration_component_tests_legacy_2026-10-18.log'))
    write(REPO/'data/refit/reports/n2_calibration_implementation_manifest_2026-10-18.json',dict(sha256={str(p.relative_to(REPO)):digest(p) for p in paths}))
    print(json.dumps(result['cohort']),flush=True)


if __name__=='__main__':run()
