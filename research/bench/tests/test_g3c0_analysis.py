from tools.jgen.analyze_g3c0 import paired


def test_parent_bootstrap_and_missing_fixed_parent_are_explicit():
    seeds = [11, 12, 13]
    control = {(p, s): {'top1_dE': 0., 'top1_score': 0.} for p in ['A', 'B', 'FAILED'] for s in seeds}
    records = {(p, s): {'top1_dE': d, 'top1_score': -d} for p,d in [('A', -3.), ('B', 1.)] for s in seeds}
    r = paired(records, control, ['A','B','FAILED'], seeds, 'top1_dE')
    assert r['mean'] == -1.
    assert r['ci95'] == [-3., 1.]
    assert r['n_parents'] == 2 and r['n_paired_units'] == 6
    assert r['parents_improved'] == 1
    assert r['per_parent']['FAILED'] == {'differences': [], 'mean': None, 'complete': False}
    assert r['unavailable'] == [['FAILED', s] for s in seeds]
    score = paired(records, control, ['A','B','FAILED'], seeds, 'top1_score')
    assert score['mean'] == 1. and score['parents_improved'] == 1
