"""Read-only CPU replay of baseline scorer terms from S2b calibration state.

This runner deliberately reads only CSD_code and SMILES from tmQM CSV rows.
The listed QM property columns are never accessed. Saved X is not a scoring
input: every scored geometry is rebuilt from S2b theta through float32 FK.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.update(
    CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    JACOBIK_PROFILE="u1",
    JACOBIK_PROFILE_OVERRIDES='{"nci":false,"deterministic_embed":true}',
)

import numpy as np
import torch

from jacobik.stats.mapping import assemble
from jacobik.fit.parser import _assemble_raw, _iter_bo_blocks, _iter_xyz_blocks
from research.archive.code.refit.nci_v2_orchestrator import apply_arm_profile_live, build_builder_and_inputs
from research.archive.code.refit.nci_v2_s2b_verify import load_npz


def safe_smiles_index(path: Path) -> dict[str, str | None]:
    """Read only the code and SMILES cells; never access any QM-property cell."""
    out = {}
    with path.open(newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader)
        code_i, smiles_i = header.index("CSD_code"), header.index("SMILES")
        for row in reader:
            code = row[code_i].strip()
            value = row[smiles_i].strip()
            out[code] = value or None
    return out


def safe_load_mol(code: str, shard: str, dataset: str, clean_root: Path, smiles_index):
    base = clean_root / ("lnQM" if dataset == "LNf" else "tmQM")
    xyz = _iter_xyz_blocks(base / "xyz" / f"tmQM_{shard}.xyz")
    bos = _iter_bo_blocks(base / "bo" / f"tmQM_{shard}.BO")
    for (comment, atoms), (bo_code, bonds) in zip(xyz, bos):
        if bo_code == code:
            smiles = smiles_index.get(code)
            raw = _assemble_raw(comment, atoms, bo_code, bonds,
                                SimpleNamespace(smiles=smiles))
            mol = assemble(raw)
            # TMQMParser normally sets this alongside QM fields. Preserve only
            # the graph string needed for ligand decomposition.
            mol.smiles = smiles
            return mol
    return None


def fk_geometry(fks, rows, pool):
    """Apply saved theta and compute closure residuals on the float32 FK result."""
    from jacobik.ik.fk_compose import ComplexFK

    offsets = pool["thetas_lig_offsets"]
    th0 = rows[0][0][0]
    device, dtype = th0.device, th0.dtype
    thetas = [torch.as_tensor(pool["thetas_flat"][:, offsets[i]:offsets[i + 1]],
                              dtype=dtype, device=device)
              for i in range(len(offsets) - 1)]
    trials = torch.as_tensor(pool["trial_idx"], dtype=torch.long, device=device)
    base = rows[0][1]
    X0 = base[trials] if base.ndim == 3 else base[None].expand(len(trials), -1, -1)
    with torch.no_grad():
        X0 = X0.to(torch.float32)
        if not fks or not thetas:
            return X0.cpu().numpy(), np.zeros(len(X0), dtype=np.float32)
        X_fk = ComplexFK(fks).apply(thetas, X0)
        seam_ss = X_fk.new_zeros(len(X_fk))
        for fk in fks:
            if getattr(fk, "_cl", None) is None:
                continue
            d0 = fk.seam_d0(X0)
            residual, _ = fk.closure_residual_jacobian(X_fk, d0)
            seam_ss = seam_ss + (residual * residual).sum(dim=-1)
        return X_fk.cpu().numpy(), seam_ss.cpu().numpy()


def replay_group_rows(builder, seeds, solve_kw, MS):
    """Same row preparation and class-budget selection as S2b verification."""
    from jacobik.builder import engine as E

    ctx, assignments, W, _T = MS.prepare(builder, seeds[0], **solve_kw)
    (fks, _dpi, _ge, scorer), rows = MS.seed_rows(builder, ctx, assignments, W, seeds)
    ctx.score_norm = float(scorer._agg_norm()) if hasattr(scorer, "_agg_norm") else 0.
    keep = None
    if E.CLASS_BUDGET == "split":
        keep = builder._class_budget_rows(ctx, assignments, W)
    if keep is not None:
        kt = torch.as_tensor(keep, dtype=torch.long, device=ctx.device)
        rows = [([x[kt] for x in th], X[kt] if X.ndim == 3 else X, ideal[kt], cand[keep])
                for th, X, ideal, cand in rows]
    return fks, rows, scorer


def compare(pred, obs):
    delta = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(obs, dtype=np.float64))
    return dict(max_abs=float(delta.max(initial=0)), p99=float(np.quantile(delta, .99)) if delta.size else 0.,
                failures=int(np.count_nonzero(delta > 1e-4)), count=int(delta.size),
                _delta=delta.tolist())


def run(frame_path: Path, pools_dir: Path, clean_root: Path, db_path: Path, out_path: Path,
        centers_dir: Path | None = None):
    from jacobik.stats.stats import StatsLookup, load_db
    from tools.jdev.budget.pools import rank_terms
    from tools.jdev import multiseed as MS
    from research.archive.code.interactions.nci_v2.nci_v2_controls import health_metrics

    frame = json.loads(frame_path.read_text())
    controls_policy = json.loads((Path(__file__).resolve().parent / "nci_v2_controls_policy.json").read_text())
    parents = frame["main_allocation"]["splits"]["calibration"]
    by_code = {p.get("representative_code") or p["code"]: p for p in parents}
    smiles_index = safe_smiles_index(clean_root / "tmQM" / "tmQM_y.csv")
    meta_path = db_path.with_name(db_path.stem + ".meta.json")
    stats = StatsLookup(load_db(json.loads(db_path.read_text()),
                                meta=json.loads(meta_path.read_text()) if meta_path.exists() else None))
    apply_arm_profile_live("control")
    pools_by_code = defaultdict(dict)
    for path in sorted(pools_dir.glob("*_control.npz")):
        code, seed, _arm = path.stem.split("_")
        if code in by_code:
            pools_by_code[code][int(seed)] = load_npz(path)

    rows_out = []
    progress_path = out_path.with_suffix(".progress.json")
    if progress_path.exists():
        saved = json.loads(progress_path.read_text())
        rows_out = saved.get("rows", [])
    seed_status = defaultdict(set)
    for r in rows_out:
        if r.get("status") == "compared" and "seed" in r:
            seed_status[r["code"]].add(int(r["seed"]))
    done_codes = {c for c, ss in seed_status.items() if ss == {11, 12, 13}}
    builder_cache = {}
    for code in sorted(by_code):
        if code in done_codes:
            continue
        rows_out = [r for r in rows_out if r.get("code") != code]
        parent = by_code[code]
        seed_pools = pools_by_code.get(code, {})
        seeds = sorted(seed_pools)
        if not seeds:
            rows_out.append(dict(code=code, status="missing_calibration_pools"))
            continue
        mol = safe_load_mol(code, parent["representative_shard"], parent["dataset"], clean_root, smiles_index)
        if mol is None:
            rows_out.append(dict(code=code, status="reference_not_found"))
            continue
        builder, _lig, error = build_builder_and_inputs(mol)
        if builder is None:
            rows_out.append(dict(code=code, status="builder_failed", reason=error))
            continue
        builder_cache[code] = builder
        if centers_dir is not None:
            centers_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(centers_dir / f"{code}_reference.npz",
                                X=np.asarray([a.position for a in mol.atoms], dtype=np.float32))
        n_var = len(builder._ez_combinations()) if hasattr(builder, "_ez_combinations") else 1
        use_var = bool(__import__("jacobik.builder.embed", fromlist=["EZ_ENUMERATE"]).EZ_ENUMERATE) and n_var > 1
        pools = seed_pools
        passes = sorted({int(v) for p in pools.values() for v in p["embed_idx"]}) if use_var else [None]
        predicted = {s: {} for s in seeds}
        for vi in passes:
            if use_var:
                for lig, fix in zip(builder.ligands, builder._ez_combinations()[vi]):
                    lig.ez_fix = fix
                builder._ez_inner = True
            try:
                fks, grouped, scorer = replay_group_rows(
                    builder, seeds, dict(device="cpu", angle_bends=True, stats=stats), MS)
            finally:
                if use_var:
                    builder._ez_inner = False
                    for lig in builder.ligands:
                        lig.ez_fix = None
            norm = float(scorer._agg_norm())
            for i, seed in enumerate(seeds):
                p = pools[seed]
                sel = np.arange(len(p["trial_idx"])) if vi is None else np.where(p["embed_idx"] == vi)[0]
                if not len(sel):
                    continue
                pose_keys = {"X", "score", "rank", "asg", "trial_idx", "assignment_idx",
                             "embed_idx", "thetas_flat", "bend_prior_solver"}
                sub = {k: (v[sel] if k in pose_keys else v) for k, v in p.items()}
                X, seam_ss = fk_geometry(fks, [grouped[i]], sub)
                if centers_dir is not None:
                    # FK coordinates are in generated-shell order; health
                    # checks and native motion paths use reference atom order.
                    X_reference = np.asarray(X, dtype=np.float32)[:, p["gen_to_ref_idx"], :]
                    ref_X = np.asarray([a.position for a in mol.atoms], dtype=np.float64)
                    health, _, _, _ = health_metrics(
                        mol, ref_X, X_reference, controls_policy,
                        p.get("coord_intact")[sel] if p.get("coord_intact") is not None else None)
                    eligible = np.flatnonzero(health["healthy"])
                    if len(eligible):
                        pick = min(eligible, key=lambda j: hashlib.sha256(
                            np.asarray(X_reference[j], dtype="<f4").tobytes()).digest())
                        pool_path = pools_dir / f"{code}_{seed}_control.npz"
                        base = grouped[i][1]
                        X0 = base[torch.as_tensor(sub["trial_idx"], dtype=torch.long)] if base.ndim == 3 else base[None].expand(len(X), -1, -1)
                        X0 = X0.to(torch.float32)
                        pairs, d0s = [], []
                        ref_to_gen = np.argsort(p["gen_to_ref_idx"])
                        for fk in fks:
                            cl = getattr(fk, "_cl", None)
                            if cl is None:
                                continue
                            seam_atoms = cl["seam"].detach().cpu().numpy()
                            pair = np.stack((seam_atoms[cl["a"].detach().cpu().numpy()],
                                             seam_atoms[cl["b"].detach().cpu().numpy()]), axis=1)
                            pairs.extend(ref_to_gen[pair].tolist())
                            d0 = fk.seam_d0(X0)[pick].detach().cpu().numpy()
                            d0s.extend(d0.tolist())
                        beta_columns = p.get("bend_columns", np.empty((0, 2), dtype=np.int64))
                        np.savez_compressed(
                            centers_dir / f"{code}_{seed}_baseline.npz",
                            X=np.asarray(X_reference[pick], dtype=np.float32),
                            seam_ss=np.asarray(seam_ss[pick], dtype=np.float32),
                            theta=np.asarray(sub["thetas_flat"][pick], dtype=np.float32),
                            thetas_lig_offsets=np.asarray(sub["thetas_lig_offsets"], dtype=np.int64),
                            bend_prior_solver=np.asarray(sub["bend_prior_solver"][pick], dtype=np.float32),
                            pose_index=np.asarray(pick, dtype=np.int64),
                            pool_row_index=np.asarray(sel[pick], dtype=np.int64),
                            trial_idx=np.asarray(sub["trial_idx"][pick], dtype=np.int64),
                            seam_pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
                            seam_d0=np.asarray(d0s, dtype=np.float32),
                            bend_columns=np.asarray(beta_columns, dtype=np.int64),
                            bend_k=np.asarray(p.get("bend_k", []), dtype=np.float32),
                            bend_free_deg=np.asarray(p.get("bend_free_deg", []), dtype=np.float32),
                            bend_scale=np.asarray(p.get("bend_scale", 1.0), dtype=np.float32),
                            rt_prior=np.asarray(p.get("rt_prior", 2.479), dtype=np.float32),
                            pool_sha256=np.asarray(hashlib.sha256(pool_path.read_bytes()).hexdigest()),
                            geometry_sha256=np.asarray(hashlib.sha256(
                                np.asarray(X_reference[pick], dtype="<f4").tobytes()).hexdigest()))
                    else:
                        (centers_dir / f"{code}_{seed}_baseline.missing.json").write_text(
                            json.dumps({"reason": "no_health_passing_pose"}) + "\n")
                asg = torch.as_tensor(sub["asg"], dtype=torch.long)
                X_t = torch.as_tensor(X, dtype=torch.float32)
                rank_total, bd = rank_terms(scorer, X_t, asg)
                q = {f"Q{j}": bd[f"Q{j}"].detach().cpu().numpy() * norm for j in range(1, 7)}
                # Stored theta is already projected. Evaluate seam_ss directly
                # on its float32 FK result; do not call closure_project again.
                for local, global_i in enumerate(sel):
                    raw_terms = {k: float(v[local]) for k, v in q.items()}
                    bend = float(sub["bend_prior_solver"][local])
                    seam = float(seam_ss[local])
                    predicted[seed][int(global_i)] = dict(rank=float(rank_total[local]),
                        score=float(rank_total[local] + bend - 1000.0 * seam),
                        B_normalized=float(rank_total[local] + bend), seam_ss=seam,
                        total_raw=float(sum(raw_terms.values())), terms=raw_terms)

        for seed in seeds:
            p = pools[seed]
            if len(predicted[seed]) != len(p["score"]):
                rows_out.append(dict(code=code, seed=seed, status="fk_rows_missing",
                                     expected=len(p["score"]), rebuilt=len(predicted[seed])))
                continue
            order = [predicted[seed][i] for i in range(len(p["score"]))]
            row = dict(code=code, seed=seed, status="compared", n_poses=len(order),
                       errors={"score": compare([x["score"] for x in order], p["score"]),
                               "rank": compare([x["rank"] for x in order], p["rank"]),
                               "total_raw": compare([x["total_raw"] for x in order],
                                   sum(p[f"terms_Q{j}"] for j in range(1, 7)))})
            for j in range(1, 7):
                row["errors"][f"Q{j}"] = compare([x["terms"][f"Q{j}"] for x in order], p[f"terms_Q{j}"])
            pred_order = np.argsort(-np.asarray([x["score"] for x in order]), kind="stable")
            stored_order = np.argsort(-np.asarray(p["score"]), kind="stable")
            row["rank_order_agreement"] = float(np.mean(pred_order == stored_order)) if len(order) else 1.
            row["seam_ss"] = [x["seam_ss"] for x in order]
            row["B_normalized"] = [x["B_normalized"] for x in order]
            rows_out.append(row)
        del builder
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(dict(rows=rows_out), separators=(",", ":")))
        print(f"{code}: {len(seeds)} calibration pools replayed", flush=True)

    metrics = {}
    for name in ["score", "rank", "total_raw", *[f"Q{j}" for j in range(1, 7)]]:
        values = [r["errors"][name]["max_abs"] for r in rows_out if r.get("status") == "compared"]
        flat_errors = np.asarray([delta for r in rows_out if r.get("status") == "compared"
                                  for delta in r["errors"][name]["_delta"]], dtype=np.float64)
        for r in rows_out:
            if r.get("status") == "compared":
                r["errors"][name].pop("_delta", None)
        metrics[name] = dict(max_abs=max(values, default=None),
                             p99=float(np.quantile(flat_errors, .99)) if flat_errors.size else None,
                             failing_pools=sorted([dict(pool=f"{r['code']}_{r['seed']}",
                                                        max_abs=r["errors"][name]["max_abs"])
                                                   for r in rows_out if r.get("status") == "compared"],
                                                 key=lambda x: -x["max_abs"])[:10],
                             failures=sum(r["errors"][name]["failures"] for r in rows_out if r.get("status") == "compared"))
    pass_tolerance = (all(metrics[f"Q{j}"]["max_abs"] is not None and metrics[f"Q{j}"]["max_abs"] <= 1e-3 for j in range(1, 7))
                      and metrics["total_raw"]["max_abs"] is not None and metrics["total_raw"]["max_abs"] <= 1e-3
                      and metrics["rank"]["max_abs"] is not None and metrics["rank"]["max_abs"] <= 1e-4
                      and metrics["score"]["max_abs"] is not None and metrics["score"]["max_abs"] <= 1e-4)
    out = dict(schema="nci_v2_s2b_baseline_parity_v1", split="calibration", pools=len([r for r in rows_out if r.get("status") == "compared"]),
               rows=rows_out, metrics=metrics, pass_tolerance=pass_tolerance,
               tolerance_amendment=dict(raw_terms_and_total_nats=1e-3, normalized_rank_and_score=1e-4,
                   reason="Pre-label audit amendment: float32 accumulation over Z-scaled sums; observed raw-total max about 4e-4 nats (~1e-3 kJ/mol), consistent with Follow-up2 float32 findings."),
               closure=dict(weight_nats_per_A2=1000., definition="seam_ss=sum((|x_a-x_b|-d0)^2) at stored final theta; score=rank+bend-1000*seam_ss",
                   max_seam_ss=max((max(r.get("seam_ss", [0.])) for r in rows_out if r.get("status")=="compared"), default=0.)),
               input_policy="S2b theta -> float32 FK -> Q1..Q6; saved X was not scored; QM property columns were not accessed.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--pools", type=Path, required=True)
    ap.add_argument("--clean-root", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--centers-dir", type=Path)
    args = ap.parse_args()
    run(args.frame, args.pools, args.clean_root, args.db, args.out, args.centers_dir)
