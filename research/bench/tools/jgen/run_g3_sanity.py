"""G-3 exact-score polish sanity and two-parent FIRE calibration; no relaxation."""
from __future__ import annotations
from tools.jgen.atom_export import symbols_for_conformer

import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from tools.jgen.run_o1_g2 import PARENTS, xyz
from tools.jgen import inputs
from tools.jgen.score_repro import CaptureSession, conformer_key
from jacobik.core.env import build_builder_and_inputs
from tools.jdev.budget.common import stats_from_db
from tools.runkit import xtb
from jacobik.ik.ik_solver import _normalised_metric_scales
from jacobik.ik.polish import optimise


def capture_parent(code, stats):
    sh, ds = inputs.frame_shards()[code]
    mol = inputs.load_mol(code, sh, ds)
    builder, _, err = build_builder_and_inputs(mol)
    if builder is None:
        raise RuntimeError(err)
    with xtb.FallbackWatch(raise_on_exit=True), CaptureSession(builder) as cap:
        solved = builder.solve(seed=11, stats=stats, angle_bends=True, device='cpu', n_conformers=16)
    cap.attach_conformers(solved.conformers)
    return cap, solved


def functions(cap, row):
    ref = cap.X_full.double()
    if ref.ndim == 3:
        ref = ref[row:row+1]
    def coords(th):
        X = ref
        for fk, t in zip(cap.fks, th):
            X = fk.apply(t, X)
        return X
    def score(th):
        return cap.score([t[0] for t in th], row).reshape(1)
    def metric(th):
        return _normalised_metric_scales([fk.jacobian_column_norm_sq(t, ref) + 1e-3
                                          for fk, t in zip(cap.fks, th)])
    return score, coords, metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parents', nargs='+', default=PARENTS)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--layout-workers',type=int,default=None,help='Original chunk layout; at most three chunks execute concurrently')
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.layout_workers is not None:
        import os
        import jacobik.ik.ik_solver as IKM
        from concurrent.futures import ThreadPoolExecutor
        os.environ['JACOBIK_NUM_WORKERS'] = str(args.layout_workers)
        class LimitedExecutor(ThreadPoolExecutor):
            def __init__(self,max_workers=None,**kwargs):
                super().__init__(max_workers=min(3,max_workers or 3),**kwargs)
        IKM.ThreadPoolExecutor = LimitedExecutor
    xtb.require()
    stats = stats_from_db(inputs.DB_PATH)
    root = Path(__file__).resolve().parents[4]
    old = json.loads((root/'research/data/jgen/campaign/B/g/o1_g2_cap30_unrel_gpu_results.json').read_text())
    result = {'parents':{}, 'failures':{}, 'dt':args.dt, 'calibration':{}, 'started':time.time()}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    calibration = []
    for code in args.parents:
        t0 = time.monotonic()
        try:
            session, solved = capture_parent(code, stats)
            rows = []
            for prior in old['parents'][code]['rows']:
                variant, assignment, row = prior['key']
                conf = next(c for c in solved.conformers if (c.embed_idx,c.assignment_idx,c.trial_idx) == tuple(prior['key']))
                cap = session.captures[0 if variant is None else variant]
                assert int(cap.candidate_idx_per_b[row]) == assignment
                th0 = [t[row:row+1].detach().double() for t in cap.thetas_final]
                score, coords, metric = functions(cap, row)
                s0 = float(score(th0).detach())
                assert abs(s0 - prior['score64_theta0']) < 1e-4, (s0, prior['score64_theta0'], prior['key'])
                initial_xyz = coords(th0)[0].detach().numpy()
                assert np.max(np.abs(initial_xyz - np.asarray(prior['X_init_A']))) < 1e-4, 'O1 initial geometry differs'
                rec = {'key':prior['key'], 'score0':s0, 'theta0':[t.detach().tolist() for t in th0], 'lbfgs_gain':prior['delta_score'], 'arms':{}}
                # Evaluate first-step RMS over a log grid, then interpolate
                # dt using the dt^2 dependence at zero initial velocity.
                th = [t.clone().requires_grad_(True) for t in th0]
                force = torch.autograd.grad(score(th).sum(), th)
                ma = metric(th)
                velocity = [args.dt*g/m for g,m in zip(force,ma)]
                vn = sum(v.square().sum(-1) for v in velocity).sqrt()
                fn = sum(g.square().sum(-1) for g in force).sqrt().clamp_min(1e-20)
                vel = [.75*v+.25*(vn/fn)[:,None]*g for v,g in zip(velocity,force)]
                with torch.no_grad():
                    rms = float((coords([t+args.dt*v for t,v in zip(th,vel)])-coords(th)).square().sum(-1).mean().sqrt())
                calibration.append(rms)
                rec['first_step_rms_A'] = rms
                if not args.calibrate:
                    for method in ('adam_small','fire'):
                        t1 = time.monotonic()
                        best, val = optimise(th0, score, coords, metric, method=method, dt_start=args.dt)
                        gain = float(val)-s0
                        rec['arms'][method] = {'score':float(val),'gain':gain,'fraction':gain/prior['delta_score'],
                            'wall_s':time.monotonic()-t1,'X_A':coords(best)[0].detach().tolist()}
                    rec['symbols'] = symbols_for_conformer(conf); rec['charge'] = prior['charge']; rec['spin'] = prior['spin_state']
                    rec['X0_A'] = prior['X_init_A']
                rows.append(rec)
            result['parents'][code] = {'rows':rows,'wall_s':time.monotonic()-t0}
        except Exception as e:
            result['failures'][code] = traceback.format_exc()
        out.write_text(json.dumps(result,indent=2))
        print(code, 'rows',len(result['parents'].get(code,{}).get('rows',[])), 'seconds',round(time.monotonic()-t0,1),flush=True)
    if calibration:
        med = float(np.median(calibration))
        result['calibration']={'median_first_step_rms_A':med,'proposed_dt':args.dt*np.sqrt(.02/med),'parents':args.parents}
    if not args.calibrate:
        for method in ('adam_small','fire'):
            arm = [r['arms'][method] for p in result['parents'].values() for r in p['rows']]
            result[method] = {'rows':len(arm),'median_gain':float(np.median([r['gain'] for r in arm])),
                             'median_fraction':float(np.median([r['fraction'] for r in arm]))}
    out.write_text(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
