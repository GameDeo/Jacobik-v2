"""Checks of C-2 census metrics, including the atom-map direction."""
import numpy as np
import torch

from tools.jgen.a4t.metrics import ring_share
from tools.jgen.c2.rings import aligned_displacements


def test_ring_displacement_share_matches_a4t_under_permutation():
    ref = np.array([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.],
                    [0., 2., 0.], [0., 0., 3.]])
    symbols = ['Zn', 'C', 'C', 'N', 'O']
    perms = np.array([[0, 1, 2, 3, 4], [0, 2, 1, 3, 4]])
    row = ref[perms[1]].copy()
    row[3] += [0.2, 0.1, 0.3]
    row += [4., -2., 1.]
    ss, mapping = aligned_displacements(row, ref, symbols, perms)
    expected, expected_map = ring_share(row, ref, symbols, perms, {1, 3}, True)
    assert mapping == expected_map
    assert np.isclose((ss[1]+ss[3])/sum(ss.values()), expected)
    assert mapping[1] == 2


def test_alignment_removes_proper_rotation_and_translation():
    ref = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    row = ref @ rotation + [4., 5., 6.]
    ss, _ = aligned_displacements(row, ref, ['Zn', 'C', 'N', 'O'], [np.arange(4)])
    assert sum(ss.values()) < 1e-25


def test_output_window_keeps_finite_rows_within_the_window():
    from jacobik.builder import engine
    scores=np.array([1.,0.,-500.,np.nan,-np.inf])
    assert engine.output_window(scores,1).tolist()==[True,True,False,False,False]


def test_start_kabsch_settings_are_consumed_at_import():
    import os,subprocess,sys,json
    env=dict(os.environ,JACOBIK_PROFILE_OVERRIDES=json.dumps({'start_kabsch':True,'start_kabsch_pivot':True,'start_kabsch_diversity':0.}))
    value=subprocess.check_output([sys.executable,'-c',
        'from jacobik.builder import engine; from jacobik.core import profile; assert engine.START_KABSCH and engine.START_KABSCH_PIVOT; assert engine.START_KABSCH_DIVERSITY==0.; print(profile.snapshot()["fingerprint"])'],
        env=env,text=True)
    assert value.strip()!='5f74b14b57ed'


def test_guard_uses_seed_rotation_null_and_both_failure_conditions():
    from tools.jgen.c2.analyse import guard
    settings=dict(regression_threshold_kj=25.,regression_min_seeds=2,regression_null_margin=2)
    bad=np.tile([100.,100.,0.],(3,1))
    assert not guard(bad,np.zeros_like(bad),settings)['passed']
    noisy=np.tile([0.,100.,200.],(3,1))
    result=guard(noisy,np.zeros_like(noisy),settings)
    assert result['trips']==3 and result['null']==3 and result['passed']


def test_wide_seed_pays_full_native_ez_preparation(tmp_path):
    import json
    from tools.jgen.c2.finalise import finalise
    arms=[dict(arm='C',flags={},seed_budget='three_w16'),
          dict(arm='S1',flags={},seed_budget='single_w48')]
    (tmp_path/'arms.json').write_text(json.dumps(arms))
    ledger=dict(parent_phases={'case':dict(prefetch=[2.],C_prepare=[6.],S1_prepare=[6.],
        C_seed_rows=[3.],S1_seed_rows=[3.])},initial_digests={},beam_times={},
        row_steps={'case':{'C':[dict(row_steps=76800)],'S1':[dict(row_steps=230400)]}})
    (tmp_path/'timing.json').write_text(json.dumps(ledger))
    for name,wall in [('C',3.),('S1',9.)]:
        directory=tmp_path/name/'records';directory.mkdir(parents=True)
        for seed in [11,12,13]:
            (directory/f'case_{seed}.json').write_text(json.dumps(dict(status='ok',wall_solve_s=wall)))
    finalise(tmp_path)
    def wall(name):
        return json.loads((tmp_path/name/'records/case_11.json').read_text())['c2_timing']['end_to_end_s']
    assert np.isclose(wall('C')*3,14.)
    assert wall('S1')==12. # full 8 s preparation + one third of wide solve/seeding
    assert json.loads((tmp_path/'compute_checks.json').read_text())[0]['passed']


def test_seeded_kabsch_translation_uses_original_fk_pivot():
    from types import SimpleNamespace
    from jacobik.ik.kabsch import kabsch_init, compensate_fk_pivot
    from jacobik.ik.fk import axisangle_to_matrix
    original = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    seeded = original[None] + torch.tensor([[[2., 1., 0.]]])
    rotation = axisangle_to_matrix(torch.tensor([[.4, -.7, .9]]))
    target = torch.bmm(seeded, rotation.transpose(1, 2)) + 5.
    ge = SimpleNamespace(metal_pos_idx=0, donor_positions=lambda x: x)
    fk = SimpleNamespace(_donor_pos_t=torch.arange(3))
    base = kabsch_init(ge, seeded, target, torch.arange(3))
    def apply(th):
        c = original.mean(0)
        return torch.bmm(seeded-c, axisangle_to_matrix(th[:, 3:]).transpose(1, 2))+c+th[:,None,:3]
    # Both the production unseeded fit and the incomplete C-2 seeded fit
    # miss when FK's reference pivot differs from the seeded donor centroid.
    unseeded_base = kabsch_init(ge, original, target, torch.arange(3))
    assert (apply(unseeded_base)-target).norm(dim=-1).max() > .5
    assert (apply(base)-target).norm(dim=-1).max() > .5
    fixed = compensate_fk_pivot(base,fk,original,seeded,ge,torch.arange(3))
    torch.testing.assert_close(apply(fixed),target,atol=2e-6,rtol=0)


def test_c2b_routes_use_strict_ci_and_targeted_count_requirements():
    from tools.jgen.c2.c2b_analyse import route_gates
    metrics={k:dict(ci=[-1.,1.]) for k in ['floor','top1','hit1','matched_rho','distinct']}
    g=dict(passed=True,trips=2,mirror=5,null=1)
    assert all(route_gates(metrics,'B',1.3,g).values())
    g['mirror']=3
    assert not route_gates(metrics,'B',1.,g)['targeted']
    assert all(route_gates(metrics,'D',1.,g,proof=True).values())
    assert not route_gates(metrics,'D',1.,g)['correctness_proof']
    metrics['top1']['ci'][1]=2.
    assert not route_gates(metrics,'D',1.,g,proof=True)['top1']


def test_matched_rho_is_size_matched_and_reproducible():
    from tools.jgen.c2.matched_rho import matched_rho
    # Larger arm has perfect opposite ranking; control has positive ranking.
    a=np.arange(20,dtype=float);c=np.arange(5,dtype=float)
    assert np.isclose(matched_rho(a,-a,c,c),-2.)
    assert matched_rho([1.],[2.],[1.],[2.]) is None
    assert matched_rho(a,a[::-1],c,c)==matched_rho(a,a[::-1],c,c)


def test_golden_pins_layout_over_inherited_one_worker_environment():
    from tools.jdev.golden import golden_environment
    env=golden_environment(dict(JACOBIK_NUM_WORKERS='1',OMP_NUM_THREADS='12',OPENBLAS_NUM_THREADS='12'))
    assert env['JACOBIK_NUM_WORKERS']=='8'
    assert env['JACOBIK_WORKER_MODE']=='thread'
    assert env['OMP_NUM_THREADS']==env['MKL_NUM_THREADS']==env['OPENBLAS_NUM_THREADS']=='1'


def test_golden_direct_child_pins_layout_before_compute(tmp_path,monkeypatch):
    import os
    import pytest
    from tools.jdev import golden
    for key in golden.GOLDEN_LAYOUT:monkeypatch.setenv(key,'1')
    monkeypatch.setattr(golden,'OUTD',tmp_path)
    np.savez(tmp_path/'production.npz',probe=np.array([1.]))
    def compute(profile):
        assert profile=='production'
        assert all(os.environ[k]==v for k,v in golden.GOLDEN_LAYOUT.items())
        return dict(probe=np.array([1.]))
    monkeypatch.setattr(golden,'_compute',compute)
    with pytest.raises(SystemExit) as result:golden._run('production','check')
    assert result.value.code==0


def test_confirmation_driver_refuses_an_unfrozen_contract(tmp_path,monkeypatch):
    import json
    import sys
    import pytest
    from tools.jgen.c2 import driver
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(driver,'population',lambda:([],{}))
    root=tmp_path/'research/campaigns/c2';root.mkdir(parents=True)
    (root/'C2B_FINALISTS.json').write_text(json.dumps(dict(status='development')))
    (root/'C2_PANEL.json').write_text(json.dumps(dict(codes=[])))
    out=tmp_path/'never_generated'
    monkeypatch.setattr(sys,'argv',['driver','--out',str(out),'--arms','C,K,P,S1',
        '--seeds','21,22,23','--confirmation-after-audit'])
    with pytest.raises(SystemExit) as result:driver.main()
    assert result.value.code==2
    assert not (out/'arms.json').exists()


def test_confirmation_analysis_refuses_route_switching(tmp_path,monkeypatch):
    import json
    import sys
    import pytest
    from tools.jgen.c2 import c2b_analyse
    monkeypatch.chdir(tmp_path)
    root=tmp_path/'research/campaigns/c2';root.mkdir(parents=True)
    (root/'C2B_FINALISTS.json').write_text(json.dumps(dict(status='frozen',arms=[dict(arm='K',route='B')])))
    settings=tmp_path/'settings.json';settings.write_text('{}')
    out=tmp_path/'never_analysed.json'
    monkeypatch.setattr(sys,'argv',['analyse','--root',str(tmp_path/'no_records'),
        '--arm','K','--route','A','--settings',str(settings),'--out',str(out)])
    with pytest.raises(SystemExit) as result:c2b_analyse.main()
    assert result.value.code==2
    assert not out.exists()
