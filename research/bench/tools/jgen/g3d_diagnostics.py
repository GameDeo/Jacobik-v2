"""CPU-only fixed-row step-0 diagnostics; not an optimizer evaluation runner.

No panel is selected implicitly. --smoke enforces exactly one parent, shortens
all traces, and forbids publishing policy choices. Real runs require a row-key
manifest (the original 48 O1 identities, re-solved on the folded-in base).
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import threading
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F

O1_PARENTS = ('HEDJOP','FEKKAJ','NUWFAM','QEZVEY','DAMVAQ','USAPOU','NEPNEF','AJAQAD')
POLICY_EXCLUSIONS = ('FEKKAJ','NEPNEF')
AIDS = ('shape', 'symmetry', 'projection')
CLASSES = ('translation', 'rotation', 'torsion', 'macrocycle_ring_torsion', 'bend')


def dof_classes(fk, width):
    """FK layout, keeping macrocycle joints separate from ordinary torsions."""
    end = 6 + fk.n_joints
    macro = getattr(fk, 'n_macro_joints', 0)
    return dict(translation=slice(0, 3), rotation=slice(3, 6),
                torsion=slice(6, end - macro), macrocycle_ring_torsion=slice(end - macro, end),
                bend=slice(end, width))


def thermal_widths():
    from jacobik.ik.scorer import THERMAL_SD
    return dict(translation=THERMAL_SD['Q1'], rotation=THERMAL_SD['Q2'],
                torsion=THERMAL_SD['Q5'], ring=THERMAL_SD['Q5'], bend=THERMAL_SD['Q2'])


def distribution(values):
    values = np.asarray(values, float)
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), min=float(values.min()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def cap_from_steps(first_done):
    """Never infer a p95 cap by pretending censored rows converged."""
    if not first_done or any(v is None for v in first_done):
        return None
    return int(np.ceil(np.percentile(first_done, 95) / 8.) * 8)


def censored_cap(first_done):
    """Amendment 3: nearest-rank p95, with censoring above observed values."""
    if not first_done:
        return None
    rank = int(np.ceil(.95 * len(first_done)))
    observed = sorted(v for v in first_done if v is not None)
    if rank > len(observed):
        return None
    return int(np.ceil(observed[rank-1] / 8.) * 8)


def gradients(value, theta, create_graph=False):
    if not value.requires_grad:
        return [torch.zeros_like(th) for th in theta]
    out = torch.autograd.grad(value.sum(), theta, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)
    return [g if g is not None else torch.zeros_like(th) for g, th in zip(out, theta)]


def norm_by_row(grads):
    return sum(g.square().sum(-1) for g in grads).sqrt()


def project(grads, donor_pos, donor_idxs):
    """The native bidentate rotation projection; difference is the aid activity."""
    out = [g.clone() for g in grads]
    for g, idx in zip(out, donor_idxs):
        if len(idx) == 2:
            direction = donor_pos[:, idx[0]] - donor_pos[:, idx[1]]
            direction = F.normalize(direction, dim=-1)
            g[:, 3:6] -= direction * (direction * g[:, 3:6]).sum(-1, keepdim=True)
    return out


class PhaseRecorder:
    """Observe actual phase updates, with live-update denominators and last20 history."""
    def __init__(self, fks, theta):
        self.layout = [dof_classes(fk, th.shape[1]) for fk, th in zip(fks, theta)]
        self.first_done = [None] * len(theta[0])
        self.history = deque(maxlen=20)
        self.per_step = []

    def before(self, phase):
        return ~phase.done.clone(), [v.clone() for v in phase.previous_g]

    def after(self, phase, grads, t, before):
        live, previous = before
        entry = {c: dict(dofs=0, at_min=0, flip_opportunities=0, flips=0) for c in CLASSES}
        for li, (grad, old) in enumerate(zip(grads, previous)):
            bound = torch.where(phase.active[:, None], phase.sigma[li],
                                torch.full_like(phase.sigma[li], phase.search_sigma))
            at_min = phase.delta[li] <= .001 * bound * (1 + 1e-6)
            eligible = live[:, None] & (old != 0) & (grad != 0)
            flips = (old * grad < 0) & eligible
            for cls, sl in self.layout[li].items():
                values = entry[cls]
                values['dofs'] += int(live.sum()) * grad[:, sl].shape[1]
                values['at_min'] += int((at_min[:, sl] & live[:, None]).sum())
                values['flip_opportunities'] += int(eligible[:, sl].sum())
                values['flips'] += int(flips[:, sl].sum())
        for i in torch.nonzero(phase.done).flatten().tolist():
            if self.first_done[i] is None:
                self.first_done[i] = t + 1
        self.history.append(entry)
        self.per_step.append(dict(step=t, classes=self.rates([entry])))

    @staticmethod
    def rates(entries):
        out = {}
        for cls in CLASSES:
            sums = {k: sum(e[cls][k] for e in entries) for k in ('dofs','at_min','flip_opportunities','flips')}
            out[cls] = dict(sums, delta_at_min_share=sums['at_min']/sums['dofs'] if sums['dofs'] else None,
                           sign_flip_rate=sums['flips']/sums['flip_opportunities'] if sums['flip_opportunities'] else None)
        return out

    def summary(self):
        return dict(last20=self.rates(self.history), per_step=self.per_step,
                    first_converged_step=self.first_done,
                    converged_steps=distribution([v for v in self.first_done if v is not None]),
                    censored_rows=sum(v is None for v in self.first_done),
                    cap_p95_rounded_to_8=cap_from_steps(self.first_done),
                    denominator='live row-DOF updates; sign flips require nonzero previous effective and current gradients')


def _copy(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy(v) for v in value)
    return value


def aid_activity(solver, data, rows):
    th = data['thetas']
    score = gradients(data['score_loss'], th)
    shape = gradients(data['shape_loss'], th)
    symmetry = gradients(data['symmetry_loss'], th)
    all_grads = [a+b+c for a,b,c in zip(score,shape,symmetry)]
    donors = solver.ge.donor_positions(data['X']).detach()
    projected = project(all_grads, donors, data['donor_pos_idxs'])
    projection = [a-b for a,b in zip(all_grads,projected)]
    base = norm_by_row(score)[rows]
    result = {}
    for name, g in zip(AIDS, (shape, symmetry, projection)):
        norm = norm_by_row(g)[rows]
        result[name] = dict(active_share=float((norm > 1e-12).float().mean()),
                           gradient_norm=norm.detach().tolist(),
                           relative_to_score=[float(x/y) if y > 1e-12 else None for x,y in zip(norm,base)])
    return result


class Snapshot:
    def __init__(self, solver, data, rows, ctx, variant, candidate):
        self.solver = solver
        self.rows = torch.as_tensor(rows, dtype=torch.long)
        n = len(data['thetas'][0])
        self.data = {}
        for key in ('X_ref','ideal','mu_shape','mu_sym','mu_sym_angle','q3_mode_override','row_groups'):
            value = data[key]
            # A 2-D X_ref is atom x xyz, never row x xyz.
            row_tensor = torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n
            if key == 'X_ref':
                row_tensor = torch.is_tensor(value) and value.ndim == 3
            self.data[key] = _copy(value[self.rows] if row_tensor else value)
        self.theta = [th[self.rows].detach().clone() for th in data['thetas']]
        self.ctx, self.variant = ctx, variant
        original = list(range(n)) if data['alive'] is None else data['alive'].tolist()
        self.keys = [[variant, int(candidate[original[r]]), int(original[r])] for r in rows]
        self.donor_idxs = data['donor_pos_idxs']
        self.sym = [g for g in (data['symmetric_groups'] or []) if len(g) >= 2]
        self.q3sym = data['q3_symmetric_groups'] if data['q3_symmetric_groups'] is not None else self.sym
        self.reference = self.data['X_ref']

    def xyz(self, theta):
        X = self.reference
        if X.ndim == 2:
            X = X.unsqueeze(0).expand(len(theta[0]), -1, -1)
        for fk, th in zip(self.solver.fks, theta):
            X = fk.apply(th, X)
        return X

    def evaluate(self, theta):
        """Frozen switch duals/ideals/Q3 commitments, native score + bend prior."""
        from jacobik.ik import ik_solver as I
        from jacobik.ik.kabsch import _optimal_rotation_nocentroid
        X = self.xyz(theta)
        donors = self.solver.ge.donor_positions(X)
        score, _ = self.solver.scorer.score(X, q3_mode_override=self.data['q3_mode_override'],
                                           row_groups=self.data['row_groups'])
        score = score + self.solver._bend_prior(theta)
        u = F.normalize(donors, dim=-1)
        with torch.no_grad():
            R = _optimal_rotation_nocentroid(self.data['ideal'], u.detach())
        square = (torch.einsum('bij,bnj->bni', R, self.data['ideal']) - u).square().sum(-1)
        deviation = ((torch.logsumexp(I._LSE_TEMP * square, -1) - np.log(square.shape[-1])) / I._LSE_TEMP
                     if square.shape[-1] > 1 else square.mean(-1))
        violation = (deviation - self.solver.shape_tol).clamp(min=0.)
        shape = self.data['mu_shape'] * violation + .5 * I._SHAPE_RHO * violation.square()
        radial = score.new_zeros(len(score)); angular = radial.clone()
        if self.sym:
            norms = donors.norm(dim=-1)
            for group in self.sym:
                d = norms[:, group]
                radial = radial + (d-d.mean(-1,keepdim=True)).square().mean(-1)
            radial = radial / len(self.sym)
            qmap = {site: i for i,site in enumerate(i for i,v in enumerate(self.solver.ge._q3_eligible) if v)}
            groups = [[qmap[s] for s in group if s in qmap] for group in self.q3sym]
            groups = [g for g in groups if len(g) > 1]
            if groups:
                haptic = any(self.solver.ge._donor_is_haptic[s] for group in self.q3sym for s in group if s in qmap)
                q = self.solver.ge.q3_approach_angles(X, haptic=haptic)
                for group in groups:
                    a = q[:, group]
                    angular = angular + (a-a.mean(-1,keepdim=True)).square().mean(-1)
                angular = angular / len(groups)
        rv = (radial-I._SYM_TOL).clamp(min=0.); av = (angular-I._SYM_TOL).clamp(min=0.)
        symmetry = (self.data['mu_sym']*rv + .5*I._SYM_RHO*rv.square() +
                    self.data['mu_sym_angle']*av + .5*I._SYM_RHO*av.square())
        seam = score.new_zeros(len(score))
        for fk in self.solver.fks:
            if getattr(fk,'_cl',None) is not None:
                residual,_ = fk.closure_residual_jacobian(X, fk.seam_d0(self.reference))
                seam += residual.square().sum(-1)
        return X, score, shape, symmetry, score-I._CLOSURE_W*seam

    def classes(self, X):
        from jacobik.builder import engine as E
        from jacobik.topology.coord_geom import classify, best_fit_assignment
        from jacobik.ik.site_enumeration import topological_class_keys
        dirs = self.solver.ge.donor_positions(X).detach().numpy()
        result=[]
        for i, positions in enumerate(X.detach().numpy()):
            mol = E._shallow_copy_mol_for_conformer(self.ctx.mol)
            for atom,pos in zip(mol.atoms,positions):
                atom.position=pos.copy()
            label=classify(mol); assignment=best_fit_assignment(dirs[i],label)
            key=None if assignment is None else topological_class_keys([assignment],self.sym,label)[0]
            result.append(dict(geometry=label,assignment_class=key))
        return result


def finish(snapshot, release, source, steps, optimizer='adam'):
    from jacobik.ik.finish_phase import FinishPhase
    from jacobik.ik.kernels import wrap_angle
    from jacobik.ik.ik_solver import _renorm_axisangle
    theta=[v.clone().requires_grad_(True) for v in snapshot.theta]
    phase=FinishPhase(theta,snapshot.solver.fks,steps,0,
                      dict(release=list(release),sigma=thermal_widths(),sigma_source=source,
                           cap=steps,switch=0,optimizer=optimizer),8,1e-5)
    recorder=PhaseRecorder(snapshot.solver.fks,theta)
    best=[v.detach().clone() for v in theta]
    X,_,_,_,initial=snapshot.evaluate(theta)
    best_score=initial.detach().clone()
    start_classes=snapshot.classes(X)
    initial_X=X.detach().clone()
    curvature_s=0.; invalid=[]; widths=[]
    started=time.perf_counter()
    for t in range(steps):
        X,score,shape,symmetry,rank_score=snapshot.evaluate(theta)
        phase.observe(torch.tensor([t]), -score.detach()+shape.detach()+symmetry.detach())
        gs=gradients(-score,theta,create_graph=source=='curvature')
        if source=='curvature':
            before=time.perf_counter()
            phase.curvature(gs,theta,float(snapshot.solver.scorer._agg_norm()))
            curvature_s+=time.perf_counter()-before
            if t==0:
                invalid=[v.detach().tolist() for v in phase.invalid_curvature]
        if t==0:
            widths=[v.detach().tolist() for v in phase.sigma]
        loss=-score+(0 if 'shape' in release else shape)+(0 if 'symmetry' in release else symmetry)
        grads=gradients(loss,theta)
        if 'projection' not in release:
            grads=project(grads,snapshot.solver.ge.donor_positions(X).detach(),snapshot.donor_idxs)
        before=recorder.before(phase)
        updates=phase.update(torch.tensor([t]),grads,[g*float(snapshot.solver.scorer._agg_norm()) for g in gs],loss.detach(),None,.1)
        recorder.after(phase,grads,t,before)
        with torch.no_grad():
            for li,(th,step,fk) in enumerate(zip(theta,updates,snapshot.solver.fks)):
                if getattr(fk,'_cl',None) is not None:
                    sl=slice(fk._macro_col0,fk._macro_col0+fk.n_macro_joints)
                    step=step.clone(); step[:,sl]=fk.closure_fused_step(X,fk.seam_d0(snapshot.reference),step[:,sl])
                th-=torch.where(phase.done[:,None],0.,step)
                th[:,3:6]=_renorm_axisangle(th[:,3:6]);th[:,6:]=wrap_angle(th[:,6:])
            end_X,_,_,_,end_score=snapshot.evaluate(theta)
            improved=end_score>best_score
            best_score=torch.maximum(best_score,end_score)
            best=[torch.where(improved[:,None],th.detach(),b) for th,b in zip(theta,best)]
        if phase.done.all():
            break
    wall=time.perf_counter()-started
    with torch.no_grad():
        for i,fk in enumerate(snapshot.solver.fks):
            if getattr(fk,'_cl',None) is not None:
                best[i],_=fk.closure_project(best[i],snapshot.reference)
        final_X,_,_,_,final_score=snapshot.evaluate(best)
    final_classes=snapshot.classes(final_X)
    changes=[a!=b for a,b in zip(start_classes,final_classes)]
    return dict(release=list(release),sigma_source=source,optimizer=optimizer,
                score_gain=(final_score-initial).detach().tolist(),class_changed=changes,
                class_change_share=float(np.mean(changes)),start_classes=start_classes,final_classes=final_classes,
                wall_s=wall,curvature_wall_s=curvature_s,steps_executed=t+1,
                sigma_at_switch=widths,invalid_curvature_at_switch=invalid,
                diagnostics=recorder.summary(),
                _initial_X=initial_X,_final_X=final_X.detach())


@contextmanager
def chunk_offsets():
    """Observe global row offsets without changing the default chunk layout."""
    from jacobik.ik import ik_solver as I
    local = threading.local()
    original = I.JointIKSolver._solve_impl
    def wrapped(self, theta, *args, **kwargs):
        local.offset = theta[0].storage_offset() // theta[0].stride(0)
        self._g3d_chunk_theta = theta[0]
        return original(self, theta, *args, **kwargs)
    I.JointIKSolver._solve_impl = wrapped
    try:
        yield local
    finally:
        I.JointIKSolver._solve_impl = original


def select_switch_rows(row_keys, variant, offset, chunk_size, original, candidate):
    """Select surviving fixed keys and ledger each missing or changed row."""
    wanted=[key for key in row_keys if (0 if key[0] is None else key[0])==variant
            and offset <= key[2] < offset+chunk_size]
    rows=[]; ledger=[]
    for key in wanted:
        if key[2]-offset not in original:
            ledger.append(dict(key=key,status='missing_at_switch'))
        elif int(candidate[key[2]]) != key[1]:
            ledger.append(dict(key=key,status='assignment_changed'))
        else:
            rows.append(original.index(key[2]-offset))
    return rows, ledger, [tuple(key) for key in wanted]


def capture_switch(builder, stats, seed, switch, total, row_keys, max_rows, expected_chunks=None):
    """One native search, capturing the live state after exactly `switch` updates.

    Production culling is kept. Missing fixed keys are failures, never replaced.
    Every E/Z call retains its own scorer, ideals, duals and Q3 commitment.
    """
    from jacobik.ik import ik_solver as I
    from tools.runkit import xtb
    snapshots=[]; context={}; activities=[]; call_index=0; ledger=[]; seen=set()
    original_joint=builder._run_joint_ik
    old_diag=I.JointIKSolver.diagnostic_hook
    old_phase=I.FINISH_PHASE

    def wrapped(ctx,fks,scorer,ge,X,theta,didx,ideal,sym,q3sym,W,T,candidate):
        nonlocal call_index
        context.update(ctx=ctx,candidate=np.asarray(candidate),variant=call_index)
        call_index+=1
        return original_joint(ctx,fks,scorer,ge,X,theta,didx,ideal,sym,q3sym,W,T,candidate)

    def hook(solver,t,data):
        if t != switch:
            return
        variant=context['variant']
        offset=offsets.offset
        if expected_chunks is not None:
            expected=next(c['digest'] for c in expected_chunks if c['variant']==variant and c['offset']==offset)
            actual=hashlib.sha256(b''.join(v.detach().contiguous().numpy().tobytes() for v in data['thetas'])).hexdigest()
            if actual!=expected:
                raise ValueError(f'switch-state manifest digest mismatch: variant={variant} offset={offset}')
        original=list(range(len(data['thetas'][0]))) if data['alive'] is None else data['alive'].tolist()
        if row_keys is None:
            rows=list(range(min(max_rows,len(original))))
        else:
            rows, failures, examined = select_switch_rows(
                row_keys,variant,offset,len(solver._g3d_chunk_theta),original,context['candidate'])
            ledger.extend(failures); seen.update(examined)
        if not rows:
            return
        snapshot=Snapshot(solver,data,rows,context['ctx'],variant,context['candidate'][offset:])
        for key in snapshot.keys:
            key[2] += offset
        # Fail loudly if reconstructed frozen-aid surfaces differ from native.
        with torch.no_grad():
            _,score,shape,symmetry,_=snapshot.evaluate(snapshot.theta)
        errors={name:float((calc - data[source].detach()[rows]).abs().max())
                for name,calc,source in [('score',-score,'score_loss'),('shape',shape,'shape_loss'),
                                         ('symmetry',symmetry,'symmetry_loss')]}
        if max(errors.values())>2e-4:
            raise ValueError(f'switch surface reconstruction mismatch: {errors}')
        activities.append(dict(keys=snapshot.keys,at_step=t,
                               aids=aid_activity(solver,data,rows),surface_max_abs_error=errors))
        snapshots.append(snapshot)

    builder._run_joint_ik=wrapped
    I.JointIKSolver.diagnostic_hook=staticmethod(hook)
    I.FINISH_PHASE=None
    started=time.perf_counter()
    try:
        with chunk_offsets() as offsets, xtb.FallbackWatch(raise_on_exit=True):
            solved=builder.solve(seed=seed,stats=stats,angle_bends=True,device='cpu',
                                 n_conformers=16,n_ik_steps=total,
                                 ik_config=I.IKSolverConfig())
        if not solved.conformers:
            ledger.append(dict(status='no_returned_conformers'))
        if row_keys is not None:
            for key in row_keys:
                if tuple(key) not in seen:
                    ledger.append(dict(key=key,status='variant_not_captured'))
    finally:
        builder._run_joint_ik=original_joint
        I.JointIKSolver.diagnostic_hook=old_diag
        I.FINISH_PHASE=old_phase
    return snapshots,activities,time.perf_counter()-started,ledger


def native_irprop_trace(builder,stats,seed,total,finish_steps):
    """Actual in-loop score-only iRprop diagnostic; no energy/arm comparison."""
    from jacobik.ik import finish_phase as FP,ik_solver as I
    from tools.runkit import xtb
    original=FP.FinishPhase; old_config=I.FINISH_PHASE
    traces=[]
    class ObservedPhase(original):
        def __init__(self,theta,fks,*args,**kwargs):
            super().__init__(theta,fks,*args,**kwargs)
            self.observer=PhaseRecorder(fks,theta);traces.append(self.observer)
        def update(self,t,grads,*args,**kwargs):
            before=self.observer.before(self)
            steps=super().update(t,grads,*args,**kwargs)
            self.observer.after(self,grads,int(t.item()),before)
            return steps
    FP.FinishPhase=ObservedPhase
    I.FINISH_PHASE=dict(release=list(AIDS),sigma=thermal_widths(),cap=min(finish_steps,total-100),
                        score_only=True,search='irprop-',optimizer='irprop-',diagnostics=True)
    started=time.perf_counter()
    try:
        with xtb.FallbackWatch(raise_on_exit=True):
            builder.solve(seed=seed,stats=stats,angle_bends=True,device='cpu',
                          n_conformers=16,n_ik_steps=total,ik_config=I.IKSolverConfig(n_workers=1))
    finally:
        FP.FinishPhase=original;I.FINISH_PHASE=old_config
    return dict(wall_s=time.perf_counter()-started,calls=[t.summary() for t in traces],
                scope='native score-only stiffness-derived iRprop- trace, not an evaluation arm')


def native_sigma_timing(builder,stats,seed,total,switch,finish_steps):
    """Paired native thermal/curvature timing; charge every curvature block.

    This correctness/cost probe does not evaluate arm energies. Search states
    must match at the fixed switch. The paired whole-loop wall difference also
    charges extra create_graph work, beyond the explicitly timed HVP blocks.
    """
    from jacobik.ik import finish_phase as FP,ik_solver as I
    from tools.runkit import xtb
    original=FP.FinishPhase;old=I.FINISH_PHASE;results={}
    try:
        for source in ('thermal','curvature'):
            calls=[]
            class TimedPhase(original):
                def __init__(self,theta,*args,**kwargs):
                    super().__init__(theta,*args,**kwargs)
                    self.theta=theta
                    self.timing=dict(curvature_block_s=0.,switch_digest=None,finish_wall_s=None)
                    self.finish_started=None;calls.append(self)
                def observe(self,t,loss):
                    super().observe(t,loss)
                    if int(t.item())==switch:
                        self.timing['switch_digest']=hashlib.sha256(b''.join(
                            v.detach().contiguous().numpy().tobytes() for v in self.theta)).hexdigest()
                        self.finish_started=time.perf_counter()
                def update(self,*args,**kwargs):
                    result=super().update(*args,**kwargs)
                    self.finish_ended=time.perf_counter()
                    return result
                def curvature(self,*args,**kwargs):
                    started=time.perf_counter()
                    try:
                        return super().curvature(*args,**kwargs)
                    finally:
                        self.timing['curvature_block_s']+=time.perf_counter()-started
            FP.FinishPhase=TimedPhase
            I.FINISH_PHASE=dict(release=list(AIDS),sigma=thermal_widths(),cap=finish_steps,
                                switch=switch,sigma_source=source,optimizer='adam')
            started=time.perf_counter()
            with xtb.FallbackWatch(raise_on_exit=True):
                builder.solve(seed=seed,stats=stats,device='cpu',angle_bends=True,n_conformers=16,
                              n_ik_steps=total,ik_config=I.IKSolverConfig(n_workers=1,cull_at=(),cull_keep=()))
            ended=time.perf_counter()
            for phase in calls:
                if phase.finish_started is None:
                    raise ValueError('native timing did not reach switch')
                # A call ends before the next call starts; observe/update records
                # its own last iteration below, rather than charging later E/Zs.
                phase.timing['finish_wall_s']=phase.finish_ended-phase.finish_started
            results[source]=dict(wall_s=ended-started,calls=[c.timing for c in calls])
    finally:
        FP.FinishPhase=original;I.FINISH_PHASE=old
    a,b=results['thermal'],results['curvature']
    if [c['switch_digest'] for c in a['calls']] != [c['switch_digest'] for c in b['calls']]:
        raise ValueError('native sigma timing search states differ')
    finish_wall=sum(c['finish_wall_s'] for c in a['calls'])
    block=sum(c['curvature_block_s'] for c in b['calls'])
    charged=max(block,b['wall_s']-a['wall_s'],0.)
    return dict(results,charged_estimator_cost_s=charged,thermal_finish_wall_s=finish_wall,
                measured_cost_fraction=charged/finish_wall,passes_10_percent=charged<=.1*finish_wall,
                method='same-start no-cull native loops; max(sum all HVP blocks,paired full-loop wall overhead); no MACE')


def run_parent(code,args,stats,mace,row_keys):
    from tools.jgen import inputs
    from jacobik.core.env import build_builder_and_inputs
    shard,dataset=inputs.frame_shards()[code]
    mol=inputs.load_mol(code,shard,dataset)
    if mol is None:
        raise ValueError('no permitted geometry')
    builder,_,err=build_builder_and_inputs(mol)
    if builder is None:
        raise ValueError(err)
    checkpoint_dir=getattr(args,'checkpoint_dir',None)
    cache_path=Path(checkpoint_dir)/f'{code}.json' if checkpoint_dir else None
    cache=json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
    identity=dict(keys=row_keys,switch=args.switch_step,total=args.total_steps,finish=args.finish_steps,base='273f2ccbf892')
    if cache and cache.get('identity')!=identity:
        raise ValueError('checkpoint identity mismatch')
    cache.setdefault('identity',identity);cache.setdefault('variants',[])
    def persist():
        if cache_path:
            cache_path.parent.mkdir(parents=True,exist_ok=True)
            cache_path.write_text(json.dumps(cache,indent=2,allow_nan=False))
    snapshots,activities,search_wall,ledger=capture_switch(builder,stats,args.seed,args.switch_step,args.total_steps,
                                                   row_keys,args.max_rows)
    print(code,'captured',sum(len(s.keys) for s in snapshots),'rows',flush=True)
    variants=[]
    for snapshot in snapshots:
        digest=hashlib.sha256(b''.join(v.contiguous().numpy().tobytes() for v in snapshot.theta)).hexdigest()
        cached=next((v for v in cache['variants'] if v['keys']==snapshot.keys),None)
        if cached:
            if cached['switch_digest']!=digest:
                raise ValueError('checkpoint switch-state digest mismatch')
            variants.append(cached)
            continue
        arms={}
        for name,release in [('keep_all',()),('shape',('shape',)),('symmetry',('symmetry',)),
                             ('projection',('projection',)),('all',AIDS),
                             ('shape+symmetry',('shape','symmetry')),
                             ('shape+projection',('shape','projection')),
                             ('symmetry+projection',('symmetry','projection'))]:
            arms[name]=finish(snapshot,release,'thermal',args.finish_steps)
        for name,thermal_arm in list(arms.items()):
            arms['curvature:'+name]=finish(snapshot,thermal_arm['release'],'curvature',args.finish_steps)
        # Same composition/order as the builder shell; no frame-order labels.
        symbols=[a.symbol for a in snapshot.ctx.mol.atoms]
        positions=[arms['all']['_initial_X']]
        positions.extend(arm['_final_X'] for arm in arms.values())
        structures=[(symbols,X.detach().numpy(),mol.charge,mol.spin_state)
                    for batch in positions for X in batch]
        started=time.perf_counter()
        energy=np.asarray(mace.energies(structures),float)*96.4853321233
        mace_wall=time.perf_counter()-started
        n=len(snapshot.keys); initial_energy=energy[:n]
        for i,(name,arm) in enumerate(arms.items()):
            arm['mace_delta_kj']=(energy[n*(i+1):n*(i+2)]-initial_energy).tolist()
            for private in ('_initial_X','_final_X'):
                del arm[private]
        thermal=arms['all'];curvature=arms['curvature:all']
        # Native curvature executes every search+finish step, not only switches.
        # Charge a conservative projected full-loop cost, explicitly an estimate.
        projected_cost=curvature['curvature_wall_s']/curvature['steps_executed']*args.total_steps
        comparison=dict(thermal_finish_wall_s=thermal['wall_s'],curvature_finish_wall_s=curvature['wall_s'],
                        curvature_measured_finish_block_s=curvature['curvature_wall_s'],
                        projected_native_all_step_cost_s=projected_cost,
                        projected_cost_over_thermal_finish=projected_cost/thermal['wall_s'],
                        native_full_loop_cost_measured=False,
                        rule='curvature only if measured full-loop estimator cost <= 0.1 * finish wall; projection cannot authorize it')
        print(code,'finished fixed-row variant/chunk',snapshot.keys,flush=True)
        variants.append(dict(keys=snapshot.keys,switch_digest=digest,arms=arms,sigma_timing=comparison,mace_wall_s=mace_wall))
        cache['variants'].append(variants[-1]);persist()
    probe=cache.get('native_irprop_dof_trace')
    if probe is None:
        probe=native_irprop_trace(builder,stats,args.seed,args.total_steps,args.finish_steps)
        cache['native_irprop_dof_trace']=probe;persist()
    print(code,'native iRprop trace complete',flush=True)
    native_timing=cache.get('native_sigma_timing')
    if native_timing is None:
        native_timing=native_sigma_timing(builder,stats,args.seed,args.total_steps,args.switch_step,args.finish_steps)
        cache['native_sigma_timing']=native_timing;persist()
    return dict(status='ok',row_failures=ledger,captured_rows=sum(len(s.keys) for s in snapshots),search_wall_s=search_wall,aid_activity_at_switch=activities,variants=variants,
                native_irprop_dof_trace=probe,native_sigma_timing=native_timing)


def rule_summary(parents,smoke):
    cohort={code:p for code,p in parents.items() if code in O1_PARENTS and code not in POLICY_EXCLUSIONS}
    counts={code:sum(len(v['keys']) for v in p.get('variants',[])) for code,p in cohort.items()}
    complete=(set(O1_PARENTS)-set(POLICY_EXCLUSIONS)<=set(cohort) and
              sum(counts.values())>=30 and all(n>=3 for n in counts.values()))
    variants=[v for p in cohort.values() if p.get('status')=='ok' for v in p['variants']]
    timings=[p['native_sigma_timing'] for p in cohort.values() if p.get('native_sigma_timing')]
    native_cost=sum(t['charged_estimator_cost_s'] for t in timings)
    native_finish=sum(t['thermal_finish_wall_s'] for t in timings)
    source=('curvature' if native_cost<=.1*native_finish else 'thermal') if complete and timings else None
    prefix='curvature:' if source=='curvature' else ''
    gains={a:[g for v in variants for g in v['arms'][prefix+a]['score_gain']] for a in AIDS}
    changes={a:[g for v in variants for g in v['arms'][prefix+a]['class_changed']] for a in AIDS}
    proposed=[a for a in AIDS if gains[a] and np.median(gains[a])>=0 and np.mean(changes[a])<=.05]
    joint_name='all' if len(proposed)==3 else '+'.join(proposed) if proposed else 'keep_all'
    chosen=[v['arms'][prefix+joint_name] for v in variants]
    joint_changes=[change for arm in chosen for change in arm['class_changed']]
    first=[step for arm in chosen for step in arm['diagnostics']['first_converged_step']]
    return dict(status='smoke_only_no_policy_selection' if smoke else
                'diagnostic_proposal_pending_audit' if complete else 'incomplete_no_policy_selection',
                policy_exclusions=list(POLICY_EXCLUSIONS),policy_row_counts=counts,policy_row_total=sum(counts.values()),
                expected_parent_identities=list(O1_PARENTS),
                release_rule='median score gain >=0 AND class-change share <=0.05 when released alone',
                release_rule_inputs={a:dict(median_score_gain=float(np.median(gains[a])) if gains[a] else None,
                                           class_change_share=float(np.mean(changes[a])) if changes[a] else None) for a in AIDS},
                proposed_release=proposed if complete and not smoke else None,
                joint_release_class_change_share=float(np.mean(joint_changes)) if joint_changes else None,
                native_curvature_cost_fraction=native_cost/native_finish if native_finish else None,
                sigma_source=source if not smoke else None,
                sigma_rule='measured whole-loop curvature cost <=10% of thermal finish wall, otherwise thermal',
                proposed_cap=cap_from_steps(first) if complete and not smoke else None,
                censored_rows=sum(v is None for v in first),
                convergence_rule='0.5 sum((score-gradient-in-nats * sigma)^2) <0.1; one-based criterion checks; p95 rounded UP to multiple of8; no cap if censored')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parents',nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--checkpoint-dir',type=Path)
    parser.add_argument('--row-manifest',type=Path,help='O1 JSON containing parents[code].rows[].key (identities only)')
    parser.add_argument('--smoke',action='store_true',help='exactly one parent; 104 switch,8 finish,112 total,2 rows/call')
    parser.add_argument('--switch-step',type=int,default=200)
    parser.add_argument('--finish-steps',type=int,default=50)
    parser.add_argument('--total-steps',type=int,default=250)
    parser.add_argument('--max-rows',type=int,default=6)
    args=parser.parse_args(argv)
    if args.smoke:
        if len(args.parents)!=1:
            parser.error('--smoke requires exactly one parent')
        args.switch_step,args.finish_steps,args.total_steps,args.max_rows=104,8,112,2
    elif args.row_manifest is None:
        parser.error('real diagnostics require --row-manifest; smoke is the only implicit row sample')
    if not 100<=args.switch_step<args.total_steps<=250 or args.finish_steps<1:
        parser.error('require 100 <= switch < total <=250 and positive finish steps')
    import os
    os.environ.pop('JACOBIK_NUM_WORKERS',None);torch.set_num_threads(1)
    from tools.jgen import inputs
    from tools.jgen.bench.panels import select
    from tools.runkit.policy import get
    from tools.runkit import xtb
    from tools.jdev.budget.common import stats_from_db
    from jacobik.core import profile
    from jacobik.ik import ik_solver as I
    from jacobik.core.energy import MACEPolarEvaluator
    from jacobik.core.env import MACE_MODEL_PATH
    if any(value is not None for value in (I.FINISH_PHASE,I.OPTIMIZER,I.POLISH,I.FINISH_SCHEDULE)):
        parser.error('baseline finish_phase/optimizer/polish/finish_schedule must be off; diagnostic controls its probes')
    manifest=json.loads((inputs.FRAME_V1/'manifest.json').read_text())
    cal={r['code'] for r in select(manifest,'cal')['units']}
    policy=[]
    for code in args.parents:
        policy.append(get().assert_panel('cal' if code in cal else 'tail',[code],cal_codes=cal))
    xtb.require()
    stats=stats_from_db(inputs.DB_PATH)
    mace=MACEPolarEvaluator(model='polar-1-s',device='cpu',dtype='float64',mode='energy-only',
                            threads=1,model_path=str(MACE_MODEL_PATH)).load()
    row_manifest=json.loads(args.row_manifest.read_text()) if args.row_manifest else None
    result=dict(format='g3d_step0_diagnostics_v1',smoke=args.smoke,seed=args.seed,
                baseline_fingerprint=profile.snapshot()['fingerprint'],
                policy=policy,parents={},failures={},
                thermal_sigma=thermal_widths(),
                thermal_mapping='translation: Q1 A; rotation/bend: Q2 radians; ordinary/macrocycle torsion: Q5 radians; no fitted policy',
                switch_step=args.switch_step,finish_steps=args.finish_steps,total_steps=args.total_steps,
                ablation='same switch rows; frozen native switch duals/ideals/Q3 commitment; fresh finish optimizer; keep_all control',
                row_manifest_sha256=hashlib.sha256(args.row_manifest.read_bytes()).hexdigest() if args.row_manifest else None)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    for code in args.parents:
        try:
            entry=None if row_manifest is None else row_manifest['parents'][code]
            if code=='NEPNEF' and entry and not entry['rows'] and 'GFN2 embed refinement failed' in entry.get('error',''):
                result['parents'][code]=dict(status='invalid_input',embed_fallback=True,rows=[],shortfall=6,
                    reason=entry['error'],source='committed amendment-1 selection ledger; no invalid row accepted')
                result['rules']=rule_summary(result['parents'],args.smoke)
                args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
                print(code,'invalid_input embed_fallback=true (amendment 2)',flush=True)
                continue
            keys=None if entry is None else [r['key'] for r in entry['rows']]
            result['parents'][code]=run_parent(code,args,stats,mace,keys)
        except Exception as exc:
            import traceback
            result['parents'][code]=dict(status='failed',reason=f'{type(exc).__name__}: {exc}')
            result['failures'][code]=traceback.format_exc()
        result['rules']=rule_summary(result['parents'],args.smoke)
        result['wall_s']=time.perf_counter()-started
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
        print(code,result['parents'][code]['status'],flush=True)
        if code!='NEPNEF' and 'GFN2 embed refinement failed' in result['parents'][code].get('reason',''):
            result['stop_reason']='nonexempt_embed_fallback'
            args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
            break
    return int(bool(result['failures']))


if __name__=='__main__':
    raise SystemExit(main())
