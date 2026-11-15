"""CUDA-event instrument in a research-only compiled method, expression order unchanged."""
import inspect,textwrap,re


def install(results,state):
    from jacobik.ik import solver_replay as SR
    src=textwrap.dedent(inspect.getsource(SR.ReplaySolver._solve_impl))
    def before(needle,code):
        nonlocal src
        if src.count(needle)!=1:raise RuntimeError(('profile anchor',needle,src.count(needle)))
        src=src.replace(needle,code+'\n'+needle)
    before('    def part_a():', '''    _events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(10)]
    for _event in _events: _event.record()
    torch.cuda.synchronize()
    _samples = []
    def _drain(_step):
        torch.cuda.synchronize()
        _samples.append(dict(step=_step, ms=[_events[i].elapsed_time(_events[i+1]) for i in range(9)]))''')
    before('        if cfk is not None:', '        _events[0].record()')
    before('        S.update(X=X_current, donor=all_donor_pos_grad, stat=stat_score, unit=donor_unit_vecs, N=N)', '        _events[1].record()')
    before('        all_donor_pos_det = all_donor_pos_grad.detach()', '        _events[2].record()')
    before('        snap_score = stat_score.detach()', '        _events[3].record()')
    before('        score_grads = None', '        _events[4].record()')
    before('        total_grads: List[torch.Tensor] = []', '        _events[5].record()')
    before('        r = -stat_score + constraint_loss', '        _events[6].record()')
    before('        metric_mass = None', '        _events[7].record()')
    before('            for li, (theta_li, step) in enumerate(zip(thetas, steps)):', '            _events[8].record()')
    before('            tl.add_(1)', '            _events[9].record()')
    src=src.replace('        part_a(); mid(); part_b()                     # step 0 eagerly: also the warm-up','        part_a(); mid(); part_b(); _drain(0)          # measured warm-up')
    src,count=re.subn(r'(?m)^(\s*)g_b\.replay\(\)$',r'\1g_b.replay(); _drain(t)',src)
    if count!=2:raise RuntimeError(('replay drain anchors',count))
    before('    return thetas_final, converged, final_score, solve_trace', '''    _PROFILE_RESULTS.append(dict(code=_PROFILE_STATE['code'],arm=_PROFILE_STATE['arm'],B=B,T=T,
        steps=t,samples=_samples,last=dict(LAST),
        phases=['score_forward_kinematics','eigh','aids','snapshot','score_and_total_gradients','projection','history_duals','optimizer','apply_wrap']))''')
    # Use live module globals: arm switching changes FINISH_PHASE after install.
    namespace=SR.__dict__
    namespace.update(_PROFILE_RESULTS=results,_PROFILE_STATE=state)
    exec(compile(src,'<g3f_cuda_event_profile>','exec'),namespace)
    SR.ReplaySolver._solve_impl=namespace['_solve_impl']
