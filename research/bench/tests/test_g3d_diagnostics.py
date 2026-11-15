"""Small diagnostic accounting tests, independent of panel solves."""
from types import SimpleNamespace
import torch
from tools.jgen.g3d_diagnostics import dof_classes, cap_from_steps, PhaseRecorder


def test_macrocycle_dofs_are_separate_and_cap_is_censor_aware():
    layout=dof_classes(SimpleNamespace(n_joints=5,n_macro_joints=2),13)
    cols=list(range(13))
    assert cols[layout['torsion']]==[6,7,8]
    assert cols[layout['macrocycle_ring_torsion']]==[9,10]
    assert cols[layout['bend']]==[11,12]
    assert cap_from_steps([9,10,11])==16
    assert cap_from_steps([9,None]) is None


def test_delta_and_flip_counts_use_live_rows_and_real_opportunities():
    from jacobik.ik.finish_phase import FinishPhase
    fk=SimpleNamespace(n_joints=2,n_macro_joints=1)
    th=[torch.zeros(2,8)]
    p=FinishPhase(th,[fk],8,0,dict(release=[],cap=8,sigma={k:1 for k in ['translation','rotation','torsion','ring','bend']},
                                optimizer='irprop-',score_only=True),4,1e-5)
    recorder=PhaseRecorder([fk],th)
    p.previous_g[0].fill_(1);p.delta[0].fill_(.001);p.done[1]=True
    g=[torch.full((2,8),-2.)]
    before=recorder.before(p)
    p.update(torch.tensor([0]),g,g,torch.ones(2),None,.1)
    recorder.after(p,g,0,before)
    result=recorder.summary()['last20']
    assert result['macrocycle_ring_torsion']['dofs']==1
    assert result['macrocycle_ring_torsion']['delta_at_min_share']==1
    assert result['macrocycle_ring_torsion']['sign_flip_rate']==1
    assert result['translation']['dofs']==3
    assert result['bend']['dofs']==0


def test_incomplete_or_smoke_inputs_never_select_policy():
    from tools.jgen.g3d_diagnostics import rule_summary
    result=rule_summary({'XOJHIN':{'status':'failed'}},smoke=False)
    assert result['status']=='incomplete_no_policy_selection'
    assert result['proposed_release'] is None
    assert result['sigma_source'] is None
    assert result['proposed_cap'] is None
    assert rule_summary({},smoke=True)['status']=='smoke_only_no_policy_selection'


def test_chunk_offsets_preserve_default_chunk_row_identity(monkeypatch):
    from jacobik.ik import ik_solver as I
    from tools.jgen.g3d_diagnostics import chunk_offsets
    observed=[]
    def fake(self, theta):
        observed.append((state.offset,len(self._g3d_chunk_theta)))
        return theta
    monkeypatch.setattr(I.JointIKSolver,'_solve_impl',fake)
    solver=object.__new__(I.JointIKSolver)
    theta=torch.zeros(12,7)
    with chunk_offsets() as state:
        solver._solve_impl([theta[4:9]])
    assert observed==[(4,5)]
    assert I.JointIKSolver._solve_impl is fake


def test_missing_fixed_row_does_not_discard_other_rows():
    from tools.jgen.g3d_diagnostics import select_switch_rows
    keys=[[None,2,4],[None,2,5],[None,3,6],[1,2,5]]
    rows,ledger,seen=select_switch_rows(keys,0,4,4,[0,2,3],[0,0,0,0,2,2,2,2])
    assert rows==[0]
    assert ledger==[dict(key=keys[1],status='missing_at_switch'),
                    dict(key=keys[2],status='assignment_changed')]
    assert seen==[tuple(k) for k in keys[:3]]


def test_amendment3_cap_uses_all_rows_and_nearest_rank():
    from tools.jgen.g3d_diagnostics import censored_cap
    # For n=36 the 35th order statistic is the p95. One censor is allowed.
    assert censored_cap(list(range(1,36))+[None])==40
    assert censored_cap(list(range(1,35))+[None,None]) is None
    assert censored_cap([144]*36)==144
    assert censored_cap([143]*35+[None])==144
    assert censored_cap([]) is None
