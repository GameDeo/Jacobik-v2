"""Current-production tiny-pool stage trace on the local GPU."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from .phase0 import prepare_parent, seeded_rows, quantiles, TINY
from .rings import population, R_PARENTS


def pair_rmsd(X, symbols, metal, max_pairs):
    from tools.runkit import geom
    X = np.asarray(X, dtype=np.float64); heavy = np.asarray(symbols) != 'H'
    finite = np.isfinite(X).all((1, 2)); X = X[finite]
    if len(X) < 2:
        return dict(n_pairs=0, n_total=0, quantiles={})
    perms, capped = geom.automorphisms(X[0], symbols, metal)
    a, b = np.triu_indices(len(X), 1)
    total = len(a)
    if total > max_pairs:
        sel = np.random.default_rng(11).choice(total, max_pairs, replace=False)
        a, b = a[sel], b[sel]
    values = []
    for i in np.unique(a):
        js = b[a == i]
        values.extend(geom.sym_rmsd_to_ref(X[js][:, heavy], X[i, heavy], perms).tolist())
    return dict(n_pairs=len(values), n_total=total, quantiles=quantiles(values), symmetry_capped=capped)


def trace(code, out, max_pairs):
    import torch
    from jacobik.builder import engine as E
    from jacobik.ik.fk_compose import ComplexFK
    original, builder, ctx, assignments, W, T = prepare_parent(code, 'cuda')
    symbols = [a.symbol for a in ctx.mol.atoms]
    results = []
    for seed in (11, 12, 13):
        rows, _ = seeded_rows(builder, ctx, assignments, W, seed)
        fks, th, dpi, ge, scorer, X, ideal, cand = rows
        keep = builder._class_budget_rows(ctx, assignments, W) if E.CLASS_BUDGET == 'split' else None
        before = len(cand)
        ctx.early_cull = None
        if keep is not None:
            kt = torch.as_tensor(keep, device=ctx.device)
            th = [t[kt] for t in th]
            X = X[kt] if X.dim() == 3 else X
            ideal, cand = ideal[kt], cand[keep]
        sym = ctx.symmetric_donor_groups
        q3 = [g for g in sym if len({ctx.all_donor_groups[d][0] for d in g}) == 1]
        with torch.no_grad():
            start = ComplexFK(fks).apply(th, X).cpu().numpy()
        final, conv, scores, solve_trace = builder._run_joint_ik(
            ctx, fks, scorer, ge, X, th, dpi, ideal, sym, q3,
            len(cand)//ctx.n_coord_samples, T, cand)
        with torch.no_grad():
            solved = ComplexFK(fks).apply(final, X)
        sn = scores.cpu().numpy()
        coords = solved.cpu().numpy()
        finite = np.isfinite(sn) & np.isfinite(coords).all((1, 2))
        z = float(scorer._agg_norm())
        window = E.output_window(sn, z)
        deduper = E.deduper_for(ctx.mol)
        distinct = np.asarray(deduper.select(solved, scores, None), int)
        passing = distinct[window[distinct]]
        cap = ctx.n_conformers if ctx.n_conformers is not None else max(len(cand)//ctx.n_coord_samples, ctx.n_coord_samples)
        selected = np.asarray(deduper.select(solved, scores, cap), int)
        selected = selected[window[selected]]
        np.savez_compressed(out/f'{code}_{seed}_solved.npz', start=start, solved=coords,
                            scores=sn, selected=selected, symbols=np.asarray(symbols), metal=ctx.mol.metal_idx-1)
        result = dict(seed=seed, counts=dict(initial_before_class_budget=before,
            after_class_budget=len(cand), solved_B=len(sn), finite=int(finite.sum()),
            embedding_valid=len(sn), inside_score_window=int(window.sum()),
            distinct_before_window=len(distinct), distinct_inside_window=len(passing),
            after_output_cap=len(selected)),
            filters=['class_budget_split', 'finite_score_output_window', 'finite_geometry_dedup',
                     'global_200_nat_score_window', 'symmetry_RMSD_0.5_A_dedup', 'n_conformers_output_cap'],
            embedding_check='GFN2-valid parent; no row-level embedding-validity filter',
            dofs=[dict(base=f.n_joints_base, bend=f.n_bend_joints, macro=f.n_macro_joints,
                       angle=f.n_angle_bends) for f in fks],
            score_quantiles=quantiles(sn), score_spread=float(np.ptp(sn[finite])) if finite.any() else None,
            start_diversity=pair_rmsd(start, symbols, ctx.mol.metal_idx-1, max_pairs),
            solved_diversity=pair_rmsd(coords, symbols, ctx.mol.metal_idx-1, max_pairs))
        results.append(result)
        print(code, seed, result['counts'], flush=True)
    return dict(code=code, status='ok', seeds=results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--max-pairs', type=int, default=4096)
    ap.add_argument('--repair-metrics', action='store_true',
                    help='Recompute pair distributions in float64 from saved rows; no IK')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    codes, _ = population()
    controls = [c for c in codes if c not in set(TINY+R_PARENTS)][:5]
    selected = a.only.split(',') if a.only else TINY+controls
    assert set(selected) <= set(codes)
    if a.repair_metrics:
        for code in selected:
            path=a.out/f'{code}.json'
            if not path.exists():continue
            data=json.loads(path.read_text())
            if data['status']!='ok':continue
            _,_,ctx,_,_,_=prepare_parent(code,'cpu')
            symbols=[x.symbol for x in ctx.mol.atoms]
            for record in data['seeds']:
                p=a.out/f"{code}_{record['seed']}_solved.npz"
                with np.load(p) as saved:
                    record['start_diversity']=pair_rmsd(saved['start'],symbols,ctx.mol.metal_idx-1,a.max_pairs)
                    record['solved_diversity']=pair_rmsd(saved['solved'],symbols,ctx.mol.metal_idx-1,a.max_pairs)
            data['pair_rmsd_dtype']='float64'
            path.write_text(json.dumps(data,indent=2))
            print(code,'pair metrics recomputed',flush=True)
        return
    (a.out/'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    # The auditor's runtime GPU slot is still named CodexB; C-2 has its
    # explicit exclusive allocation in the directive and AGENTS since Nov 2.
    from tools.runkit.resources import gpu_lock
    with gpu_lock('CodexB'):
        for code in selected:
            t = time.monotonic()
            try:
                data = trace(code, a.out, a.max_pairs)
            except Exception as exc:
                import traceback
                data = dict(code=code, status='failed', error=repr(exc), traceback=traceback.format_exc())
                print(code, 'FAILED', repr(exc), flush=True)
            data['wall_s'] = time.monotonic()-t
            (a.out/f'{code}.json').write_text(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
