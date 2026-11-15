"""S2b verification of the per-pose solver state saved in NCI v2 evaluation pools (read-only on frozen data).

Checks, per pool <out>/pools/<CODE>_<SEED>_control.npz:
  4a  FK round trip: rebuild the parent with the orchestrator's builder path (load_mol, build_builder_and_inputs,
      prepare, seed_rows, class-budget row selection; one pass per E/Z variant when variants exist), apply the saved
      thetas of every pose through ComplexFK at its trial row, and compare with the stored X.
  4b  Bend energy: recompute E_bend from the saved thetas and constants (E = scale * sum k (max(0,|beta|-free))^2 / 2)
      and the native angle_bends.energy_prior, and compare with bend_prior_solver.
  4c  Score sanity (summary mode): top-1 score and top-1 / floor dE against the Step 2 pool (numbers only).

Usage (one GPU job at a time; run only when no other solver process is running):
  python -u -m research.archive.code.refit.nci_v2_s2b_verify roundtrip --out-dir DIR --frame FRAME [--codes A,B]
  python -u -m research.archive.code.refit.nci_v2_s2b_verify summary   --out-dir DIR [--step2-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

import numpy as np  # noqa: E402


#: Pool files carry one pre-existing object array (max_rows, None or an int); the S2b keys are checked to be object-free.
def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    s2b_keys = [k for k in out if k in S2B_KEYS]
    bad = [k for k in s2b_keys if out[k].dtype == object]
    if bad:
        raise ValueError(f"{path}: S2b keys with object dtype: {bad}")
    return out


S2B_KEYS = ("state_schema_version", "trial_idx", "assignment_idx", "embed_idx", "thetas_flat", "thetas_lig_offsets",
            "bend_columns", "bend_k", "bend_free_deg", "bend_scale", "bend_per_z", "rt_prior",
            "bend_column_def_sha256", "bend_prior_solver")


def bend_energy_check(pool: dict) -> dict:
    """4b: recomputed E_bend (kJ/mol) and log-prior from saved thetas and constants, versus bend_prior_solver."""
    import torch

    from jacobik.ik import angle_bends as AB

    P = pool["thetas_flat"].shape[0]
    Z = float(pool["Z"])
    per_z = bool(pool["bend_per_z"])
    rt = float(pool["rt_prior"])
    scale = float(pool["bend_scale"])
    zf = Z if per_z else 1.0
    solver = pool["bend_prior_solver"].astype(np.float64)
    cols = pool["bend_columns"]
    if cols.shape[0] == 0:
        e_kj = np.zeros(P)
        recon = np.zeros(P)
        native = np.zeros(P)
    else:
        offs = pool["thetas_lig_offsets"]
        beta = np.stack([pool["thetas_flat"][:, offs[li] + c] for li, c in cols], axis=1)  # (P, K) radians
        k = pool["bend_k"].astype(np.float64)
        free = np.deg2rad(pool["bend_free_deg"].astype(np.float64))
        ex = np.maximum(np.abs(beta) - free[None, :], 0.0)
        e_kj = scale * (0.5 * k[None, :] * ex ** 2).sum(1)
        recon = -e_kj / rt / zf
        native = AB.energy_prior(torch.tensor(beta, dtype=torch.float64),
                                 torch.tensor(k, dtype=torch.float64),
                                 torch.tensor(pool["bend_free_deg"].astype(np.float64)),
                                 scale).numpy() / zf
    e_solver = -solver * rt * zf  # kJ/mol implied by the solver's value
    return dict(
        max_abs_recon_minus_solver=float(np.max(np.abs(recon - solver))) if P else 0.0,
        max_abs_native_minus_solver=float(np.max(np.abs(native - solver))) if P else 0.0,
        max_abs_E_recon_minus_solver_kj=float(np.max(np.abs(e_kj - e_solver))) if P else 0.0,
        max_abs_bend_prior=float(np.max(np.abs(solver))) if P else 0.0,
        n_bend_columns=int(cols.shape[0]),
        per_z=per_z,
        Z=Z,
    )


def group_rows(builder, seeds, solve_kw):
    """Rebuild (fks, rows) for `seeds` exactly as multiseed.solve_seeds does for the builder's current E/Z variant:
    prepare, seed_rows, then the class-budget row selection."""
    import torch

    from jacobik.builder import engine as E
    from tools.jdev import multiseed as MS

    ctx, assignments, W, T = MS.prepare(builder, seeds[0], **solve_kw)
    (fks, dpi, ge, scorer), rows = MS.seed_rows(builder, ctx, assignments, W, seeds)
    ctx.score_norm = float(scorer._agg_norm()) if hasattr(scorer, "_agg_norm") else 0.0
    keep = None
    if E.CLASS_BUDGET == "split":
        keep = builder._class_budget_rows(ctx, assignments, W)
    if keep is not None:
        kt = torch.as_tensor(keep, dtype=torch.long, device=ctx.device)
        rows = [([x[kt] for x in th], X[kt] if X.dim() == 3 else X, ideal[kt], cand[keep])
                for th, X, ideal, cand in rows]
    return fks, rows


def roundtrip_code(builder_cache, code, shard, dataset, seeds, pools, solve_device, stats) -> dict:
    """4a for all seeds of one code. Poses are rebuilt from the pass of their own E/Z variant (embed_idx).
    Returns {seed: {max_abs_dX, worst_pose, per_pose_max_dX, n_poses} or {error}}."""
    import torch

    from jacobik.builder import embed as CE
    from jacobik.ik.fk_compose import ComplexFK
    from refit import nci_v2_orchestrator as O

    if code not in builder_cache:
        O.apply_arm_profile_live("control")
        mol = O.load_mol(code, shard, dataset)
        builder, _lig, err = O.build_builder_and_inputs(mol)
        if builder is None:
            return {s: dict(error=f"rebuild failed: {err}") for s in seeds}
        builder_cache[code] = builder
    builder = builder_cache[code]
    solve_kw = dict(device=solve_device, angle_bends=True, stats=stats)
    n_var = len(builder._ez_combinations()) if hasattr(builder, "_ez_combinations") else 1
    use_var = bool(CE.EZ_ENUMERATE) and n_var > 1
    filled = {s: np.full(pools[s]["X"].shape, np.nan) for s in seeds}
    passes = sorted({int(v) for s in seeds for v in pools[s]["embed_idx"]}) if use_var else [None]
    for vi in passes:
        if use_var:
            fixes = builder._ez_combinations()[vi]
            for lig, f in zip(builder.ligands, fixes):
                lig.ez_fix = f
            builder._ez_inner = True
        try:
            fks, rows = group_rows(builder, seeds, solve_kw)
        finally:
            if use_var:
                builder._ez_inner = False
                for lig in builder.ligands:
                    lig.ez_fix = None
        for si, seed in enumerate(seeds):
            p = pools[seed]
            sel = np.arange(len(p["trial_idx"])) if vi is None else np.where(p["embed_idx"] == vi)[0]
            if sel.size == 0:
                continue
            offs = p["thetas_lig_offsets"]
            X_ref, th0 = rows[si][1], rows[si][0][0]
            tdt, dev = th0.dtype, X_ref.device
            th_all = [torch.tensor(p["thetas_flat"][sel, offs[li]:offs[li + 1]], dtype=tdt, device=dev)
                      for li in range(len(offs) - 1)]
            trials = torch.as_tensor(p["trial_idx"][sel], dtype=torch.long, device=dev)
            X_rows = X_ref[trials] if X_ref.dim() == 3 else X_ref[None].expand(len(sel), -1, -1)
            with torch.no_grad():
                X_fk = ComplexFK(fks).apply(th_all, X_rows.to(tdt)).double().cpu().numpy()
            filled[seed][sel] = X_fk
    out = {}
    for seed in seeds:
        X_fk = filled[seed]
        if np.isnan(X_fk).any():
            out[seed] = dict(error="pose not rebuilt (no variant pass covered it)")
            continue
        dX = np.abs(X_fk - pools[seed]["X"])
        per_pose = dX.reshape(dX.shape[0], -1).max(1)
        out[seed] = dict(max_abs_dX=float(per_pose.max()), worst_pose=int(per_pose.argmax()),
                         per_pose_max_dX=per_pose.tolist(), n_poses=int(len(per_pose)), n_variant_passes=len(passes))
    return out


def roundtrip_main(args) -> None:
    from jacobik import profile as _P  # noqa: F401
    from tools.jdev.budget.common import DB, stats_from_db

    out_dir = Path(args.out_dir)
    frame = json.loads(Path(args.frame).read_text())
    splits = frame["main_allocation"]["splits"]
    item_by_code = {}
    for split in ("calibration", "motion_validation", "surface_fit"):
        for it in splits.get(split, []):
            item_by_code[it.get("representative_code") or it.get("code")] = (it, split)
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else sorted(
        {p.name.split("_")[0] for p in (out_dir / "pools").glob("*_control.npz")})
    constraints = Path(args.constraints_db)
    stats = stats_from_db(constraints) if constraints.exists() else stats_from_db(DB)
    report, builder_cache = {}, {}
    for code in codes:
        it, split = item_by_code[code]
        seeds = sorted({int(p.name.split("_")[1]) for p in (out_dir / "pools").glob(f"{code}_*_control.npz")})
        pools = {s: load_npz(out_dir / "pools" / f"{code}_{s}_control.npz") for s in seeds}
        ok = {s: p for s, p in pools.items() if "X" in p and "thetas_flat" in p}
        if not ok:
            report[code] = dict(split=split, error="no solved pools with solver state")
            continue
        seeds_ok = sorted(ok)
        try:
            fk = roundtrip_code(builder_cache, code, it["representative_shard"], it["dataset"], seeds_ok, ok,
                                args.solve_device, stats)
        except Exception as exc:  # reported, never hidden
            fk = {s: dict(error=f"{type(exc).__name__}: {exc}") for s in seeds_ok}
        bend = {s: bend_energy_check(ok[s]) for s in seeds_ok}
        report[code] = dict(split=split, seeds={str(s): dict(fk=fk[s], bend=bend[s]) for s in seeds_ok})
        print(f"{code} {split}: " + "; ".join(
            f"s{s} dX={fk[s].get('max_abs_dX', fk[s].get('error'))} bend={bend[s]['max_abs_recon_minus_solver']:.2e}"
            for s in seeds_ok), flush=True)
    (out_dir / args.report_name).write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {out_dir / args.report_name}", flush=True)


def summary_main(args) -> None:
    out_dir = Path(args.out_dir)
    step2 = Path(args.step2_dir) if args.step2_dir else None
    rows = []
    for jp in sorted((out_dir / "pools").glob("*_control.json")):
        s = json.loads(jp.read_text())
        if s.get("status") != "success":
            rows.append(dict(code=s.get("code"), seed=s.get("seed"), status=s.get("status"), error=s.get("error")))
            continue
        new = load_npz(out_dir / "pools" / f"{s['code']}_{s['seed']}_control.npz")
        sc = new["score"]
        rec = dict(code=s["code"], seed=s["seed"], status="success", split=s.get("split"), n_confs=int(len(sc)),
                   top1_score=float(sc.max()), top1_dE=s.get("top1_dE_kj"), floor_dE=s.get("floor_dE_kj"),
                   bend_columns=int(new["bend_columns"].shape[0]))
        if step2 is not None:
            old_pool = step2 / "pools" / f"{s['code']}_{s['seed']}_control.npz"
            old_sum = step2 / "pools" / f"{s['code']}_{s['seed']}_control.json"
            if old_pool.exists() and old_sum.exists():
                o = load_npz(old_pool)
                if "score" in o:
                    rec["step2_n_confs"] = int(len(o["score"]))
                    rec["step2_top1_score"] = float(o["score"].max())
                os_ = json.loads(old_sum.read_text())
                ovp = step2 / "repair_s2_1" / "overlay" / f"{s['code']}_{s['seed']}_control.npz"
                ov = load_npz(ovp) if ovp.exists() else {}
                if "top1_dE_kj" in ov:
                    rec["step2_top1_dE"], rec["step2_floor_dE"] = float(ov["top1_dE_kj"]), float(ov["floor_dE_kj"])
                else:
                    rec["step2_top1_dE"], rec["step2_floor_dE"] = os_.get("top1_dE_kj"), os_.get("floor_dE_kj")
        rows.append(rec)
    (out_dir / args.report_name).write_text(json.dumps(rows, indent=1, default=str))
    print(json.dumps(rows, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rt = sub.add_parser("roundtrip")
    rt.add_argument("--out-dir", required=True)
    rt.add_argument("--frame", required=True)
    rt.add_argument("--constraints-db", default=str(REPO / "data/refit/final/constraints_2026-10-18.json"))
    rt.add_argument("--solve-device", default="cuda")
    rt.add_argument("--codes", default="")
    rt.add_argument("--report-name", default="s2b_roundtrip.json")
    sm = sub.add_parser("summary")
    sm.add_argument("--out-dir", required=True)
    sm.add_argument("--step2-dir", default="")
    sm.add_argument("--report-name", default="s2b_scores.json")
    args = ap.parse_args()
    if args.cmd == "roundtrip":
        roundtrip_main(args)
    else:
        summary_main(args)


if __name__ == "__main__":
    main()
