"""S6 reachability triage (research/archive/docs/roadmap/roadmap_generation_v1.md S6) for one parent. CPU only.

1. Representability: least-squares fit of the solver's own DOFs (ComplexFK over ForwardKinematics: rigid body,
   torsions, bend columns, ring joints) to the mapped reference, starting from the embed (all DOF zero). Residual RMSD
   overall and per ligand, over non-metal atoms, after aligning the reference to the shell frame (Kabsch, metal and
   donors).
2. Scoring: the u1 solve objective (scorer.score) and the ranking score (scorer.ranking_score) of the fitted pose and
   of the solver's top-1 row, with the per-term breakdown. Both use the same solve (builder.solve), so the scorer and
   the atom order are the solver's own.
3. Basin: a local refinement of the same DOFs on the solve objective, started at the fitted pose (Adam). Reports the
   distance moved from the fit, the RMSD to the top-1 after refinement, and the final objective.

Class: R if the fit residual is above RES_R_A; otherwise compare the fitted pose with the top-1 by the solve objective
and by the basin: S if the fit scores worse (higher objective, lower score than top-1), F if the fit scores better and
stays (moves less than STAY_A), D if the fit scores better but the local refinement drifts away (> STAY_A).

    python tools/jgen/diag/d30_s6_reachability.py --code DAMVAQ [--seed 11]
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT, os.path.join(ROOT, "refit"), os.path.join(ROOT, "tools", "jdev", "budget")]

import numpy as np  # noqa: E402

RES_R_A = 0.5   # residual RMSD above which the fit cannot express the structure (A)
STAY_A = 0.3    # basin: drift above this (A RMSD) counts as leaving the fitted basin


def kabsch(P, Q):
    """Rotation R and translation t minimising |R P + t - Q| (P, Q: (n,3))."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def rmsd(A, B):
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--fit-steps", type=int, default=1500)
    ap.add_argument("--basin-steps", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=6)
    a = ap.parse_args()

    import torch
    from tools.jgen import inputs
    from jacobik.core.env import build_builder_and_inputs
    from jacobik.builder.atom_map import build_atom_mapping
    from jacobik.ik.fk import ForwardKinematics
    from jacobik.ik.fk_compose import ComplexFK
    from jacobik.topology.decompose import decompose_ligand
    from jacobik.topology.decompose import decompose_ligand as _dl  # noqa
    from tools.jdev.budget.common import stats_from_db

    torch.set_default_dtype(torch.float32)  # the solver runs float32 (ComplexFK matrices)
    shard, ds = inputs.frame_shards()[a.code]
    mol = inputs.load_mol(a.code, shard, ds)
    ref = np.array([x.position for x in mol.atoms], float)
    rsym = [x.symbol for x in mol.atoms]
    b, _, err = build_builder_and_inputs(mol)
    if b is None:
        print(json.dumps(dict(code=a.code, error=err)))
        return
    stats = stats_from_db(inputs.DB_PATH)

    # --- the solver's own shell, decomposed as in solve(), and its FK
    shell, _, donor_idx_full = b._build_shell(False, False, chelate_embed=True)
    for lig in shell.ligands:
        decompose_ligand(shell, lig)
        lig.macrocycles = []
    fks = [ForwardKinematics(shell, lig, torch.device("cpu"), angle_bends=True) for lig in shell.ligands]
    cfk = ComplexFK(fks)
    X0 = np.array([x.position for x in shell.atoms], float)
    ssym = [x.symbol for x in shell.atoms]
    metal = shell.metal_idx - 1  # shell atom indices are 1-based; coordinate arrays 0-based
    donors = [int(s.donor_indices[0]) - 1 for s in shell.coord_sites if len(s.donor_indices) == 1]

    # --- reference -> shell order through the solver's atom mapping (the orchestrator's convention)
    from jacobik.builder.atom_map import build_atom_mapping as _bam
    am = _bam(mol, gen_mol=shell, X_ref=ref, X_gen=X0)
    g2r = np.asarray(am.gen_to_ref_idx, int)
    T = np.full_like(X0, np.nan)
    T[g2r] = ref
    ok = np.isfinite(T).all(1)
    # align the reference onto the shell frame by the metal and donors
    idx = sorted(set([metal] + donors) & set(np.where(ok)[0]))
    R, t = kabsch(T[idx], X0[idx])
    Ta = T @ R.T + t
    Ta[~ok] = X0[~ok]
    # --- 1. representability
    n_dof = [6 + f.n_joints + f.n_angle_bends for f in fks]
    Xr = torch.tensor(X0, dtype=torch.float32)
    Tt = torch.tensor(Ta, dtype=torch.float32)
    nm = torch.tensor([i for i in range(len(ssym)) if i != metal])
    nonm_pre = [i for i in range(len(ssym)) if i != metal]
    # the solver's own rigid-body start: kabsch_init of each ligand's donors onto the aligned reference donors
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.ik_solver import kabsch_init
    ge = GeometryExtractor(shell, torch.device("cpu"))
    site_atom = [int(site.donor_indices[0]) - 1 for site in shell.coord_sites if len(site.donor_indices) == 1]
    init_theta = []
    for li, fk in enumerate(fks):
        atoms_li = set(i - 1 for g in donor_idx_full[li] for i in g)
        lig_di = [k for k, a_ in enumerate(site_atom) if a_ in atoms_li]
        th = torch.zeros(1, n_dof[li])
        if lig_di:
            tgt = torch.tensor(Ta[[site_atom[k] for k in lig_di]], dtype=torch.float32)[None]
            base = kabsch_init(ge=ge, X_ref=torch.tensor(X0, dtype=torch.float32),
                               donor_targets=tgt, donor_pos_idx=torch.tensor(lig_di, dtype=torch.long),
                               diversify_orientation=False)
            th[:, :6] = base.reshape(1, 6)
        else:
            # no single-donor site maps to this ligand (kabsch_init needs >= 1 target: its NaN is the S6 NaN):
            # start from the identity rotation, translated so the ligand's atoms sit on the aligned reference centroid
            li_atoms = sorted(i - 1 for g in donor_idx_full[li] for i in g)
            if li_atoms:
                th[0, 3:6] = 0.0
                th[0, 0:3] = torch.tensor(Ta[li_atoms].mean(0) - X0[li_atoms].mean(0), dtype=torch.float32)
        init_theta.append(th)
    _Y0 = cfk.apply(init_theta, torch.tensor(X0, dtype=torch.float32))[0].detach().numpy()
    init_donor_rmsd = rmsd(_Y0[[site_atom[k] for k in range(len(site_atom))]], Ta[[site_atom[k] for k in range(len(site_atom))]])
    rng = np.random.default_rng(0)
    best = None
    for restart in range(a.restarts):
        # restart 0: the embed (all DOF zero); others: random rigid orientation and torsions around the embed
        thetas = [init_theta[li].clone() for li in range(len(fks))]
        if restart:
            for th in thetas:
                th[0, 6:] = torch.tensor(rng.uniform(-np.pi, np.pi, th.shape[1] - 6), dtype=torch.float32)
        thetas = [th.clone().requires_grad_(True) for th in thetas]
        for lr, steps in ((0.01, a.fit_steps // 2), (0.002, a.fit_steps - a.fit_steps // 2)):
            opt = torch.optim.Adam(thetas, lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                Xf = cfk.apply(thetas, Xr)[0]
                loss = ((Xf[nm] - Tt[nm]) ** 2).sum(1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(thetas, 5.0)  # a diverging DOF would otherwise leave the table range
                opt.step()
        with torch.no_grad():
            Xc = cfk.apply(thetas, Xr)[0].numpy()
        res_c = rmsd(Xc[nonm_pre], Ta[nonm_pre])
        if best is None or res_c < best[0]:
            best = (res_c, [th.detach().clone() for th in thetas], restart)
    thetas = [th.clone().requires_grad_(True) for th in best[1]]
    with torch.no_grad():
        Xfit = cfk.apply(thetas, Xr)[0].numpy()
    lig_atoms = {}
    for li, lig in enumerate(shell.ligands):
        ids = sorted({int(i) for g in lig.donor_atoms for i in ([g] if isinstance(g, int) else g)}) if False else None  # noqa
    per_lig = []
    nonm = [i for i in range(len(ssym)) if i != metal]
    if not np.isfinite(Xfit).all():  # never reached after the start fix; recorded, not raised
        print(json.dumps(dict(code=a.code, seed=a.seed, cls="error", error="non-finite fitted pose")))
        return
    res_all = rmsd(Xfit[nonm], Ta[nonm])
    restart_used = best[2]
    # per-fragment: donors' neighbourhood vs the rest (heavy atoms only)
    core = [i for i in nonm if np.linalg.norm(Ta[i] - Ta[metal]) < 2.7]
    rest = [i for i in nonm if i not in core]
    res = dict(overall_rmsd_A=round(res_all, 3), donor_shell_rmsd_A=round(rmsd(Xfit[core], Ta[core]), 3) if core else None,
               periphery_rmsd_A=round(rmsd(Xfit[rest], Ta[rest]), 3) if rest else None,
               embed_rmsd_A=round(rmsd(X0[nonm], Ta[nonm]), 3), n_dof=n_dof, restart_used=restart_used,
               kabsch_start_donor_rmsd_A=round(init_donor_rmsd, 3))
    scored = None
    try:
        # --- 2. scoring: solver's own solve gives the scorer and the top-1
        r = b.solve(seed=a.seed, stats=stats, angle_bends=True, device="cpu", n_ik_steps=2, n_conformers=1)
        sc = r.scorer
        conf = max(r.conformers, key=lambda c: c.score)
        Xt = torch.tensor(np.array([[at.position for at in conf.complex.atoms]], float), dtype=torch.float32)
        rg = torch.zeros(1, dtype=torch.long)

        def objective(X):
            with torch.no_grad():
                s, comps = sc.score(X, row_groups=rg)
                try:
                    rk = sc.ranking_score(X, row_groups=rg)
                except Exception:
                    rk = None
            return float(s.reshape(-1)[0]), (float(rk.reshape(-1)[0]) if rk is not None else None), \
                {k: round(float(v.reshape(-1)[0]), 3) for k, v in comps.items() if torch.is_tensor(v) and v.numel() == 1}

        s_top, rk_top, c_top = objective(Xt)
        s_fit, rk_fit, c_fit = objective(torch.tensor(Xfit[None], dtype=torch.float32))
        sc_out = dict(top1_objective=round(s_top, 3), fit_objective=round(s_fit, 3), top1_rank=rk_top and round(rk_top, 3),
                      fit_rank=rk_fit and round(rk_fit, 3), terms_top1=c_top, terms_fit=c_fit,
                      top1_rmsd_to_fit_A=round(rmsd(np.asarray(Xt[0]), Xfit), 3))
        # --- 3. basin: local refinement of the same DOFs on the solve objective, from the fit
        th2 = [t_.detach().clone().requires_grad_(True) for t_ in thetas]
        opt2 = torch.optim.Adam(th2, lr=0.005)
        X_start = Xfit.copy()
        for _ in range(a.basin_steps):
            opt2.zero_grad()
            Xb = cfk.apply(th2, Xr)[0]
            s, _ = sc.score(Xb[None], row_groups=rg)
            (-s.reshape(-1)[0]).backward()  # ascent on the solve objective (higher is better)
            opt2.step()
        with torch.no_grad():
            Xend = cfk.apply(th2, Xr)[0].numpy()
        s_end, rk_end, _ = objective(torch.tensor(Xend[None], dtype=torch.float32))
        basin = dict(drift_from_fit_A=round(rmsd(Xend[nonm], X_start[nonm]), 3), top1_rmsd_to_reference_A=round(rmsd(np.asarray(Xt[0])[nonm], Ta[nonm]), 3),
                     rmsd_to_top1_after_A=round(rmsd(Xend, np.asarray(Xt[0])), 3),
                     objective_end=round(s_end, 3))
        scored = dict(s_top=s_top, s_fit=s_fit, basin=basin, sc_out=sc_out, rk_top=rk_top)
    except Exception as exc:  # scoring can fail on out-of-table distances of a fitted pose (never hides a result)
        scored = None
        sc_err = f"{type(exc).__name__}: {str(exc)[:160]}"
    if scored is None:
        s_top = s_fit = None
        basin = dict(error=sc_err if 'sc_err' in dir() else 'unknown')
        sc_out = dict(error=basin['error'])
    if res_all > RES_R_A:
        cls = "R"
    elif scored is None:
        cls = "error"
    elif s_fit >= s_top:  # the fit scores at least as well as the top-1 (higher objective is better)
        cls = "F" if basin["drift_from_fit_A"] <= STAY_A else "D"
    else:
        cls = "S"
    sc_out["top1_solver_score"] = round(float(conf.score), 4)
    sc_out["top1_rmsd_to_reference_A"] = round(rmsd(np.asarray(Xt[0])[nonm], Ta[nonm]), 3)
    out = dict(code=a.code, seed=a.seed, residual=res, scoring=sc_out, basin=basin, cls=cls,
               n_solver_rows=len(r.conformers))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
