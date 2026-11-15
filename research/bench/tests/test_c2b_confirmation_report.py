"""Descriptive statistics must preserve pairing and undefined/failure accounting."""
import json
from pathlib import Path
import numpy as np
from tools.jgen.c2.confirmation_report import descriptive


def test_descriptive_parent_means_and_missing_values(tmp_path):
    settings=dict(seeds=[21,22,23],primary_exclusions=[],guard_artifacts=[],
        bootstrap_resamples=100,bootstrap_seed=11,rho_draws=20,rho_seed=11,
        regression_threshold_kj=25.,regression_min_seeds=2)
    for code,delta in [('a',3.),('b',-1.),('c',None)]:
        for arm in ['P','K']:
            path=tmp_path/arm/'records';path.mkdir(parents=True,exist_ok=True)
            for seed in settings['seeds']:
                if delta is None:record=dict(status='failed')
                else:
                    shift=delta if arm=='P' else 0.
                    record=dict(status='ok',floor_dE=5.+shift,top1_dE=10.+shift,n_rows=2,
                        coverage=dict(rmsd_A=[.8,1.2]),c2_metrics=dict(minimum_shell_rmsd_A=.1),
                        c2_timing=dict(end_to_end_s=1.),scores=[2.,1.] if code=='a' else [1.,1.],
                        energies_kj=[10.+shift,5.+shift])
                (path/f'{code}_{seed}.json').write_text(json.dumps(record))
    result=descriptive(tmp_path,'P','K',settings,['a','b','c'])
    assert result['descriptive_only'] and 'qualified' not in result
    assert result['metrics']['floor']['n']==2
    assert np.isclose(result['metrics']['floor']['mean'],1.)
    assert result['metrics']['rho']['n']==1
    assert result['metrics']['hit1']['n']==3
    assert result['failures']==['c']
