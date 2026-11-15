"""Bounded, score-verified O1 local convergence and strain diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / "src"), str(REPO), str(REPO / "research/bench")]
OUT = REPO / "research/data/jgen/campaign/B/g/o1_g2_results.json"
PARENTS = ["HEDJOP", "FEKKAJ", "NUWFAM", "QEZVEY", "DAMVAQ", "USAPOU", "NEPNEF", "AJAQAD"]
EV_TO_KJ = 96.4853321233

from tools.jgen import inputs
from tools.jgen.score_repro import CaptureSession
from jacobik.core.env import build_builder_and_inputs
from tools.jdev.budget.common import stats_from_db
from tools.runkit import mace, resources, xtb


def vectorize(theta):
    shapes = [np.asarray(t).shape for t in theta]
    sizes = [int(np.prod(s)) for s in shapes]
    x = np.concatenate([np.asarray(t, dtype=np.float64).ravel() for t in theta])
    return x, shapes, sizes


def unvectorize(x, shapes, sizes):
    out, pos = [], 0
    for shape, size in zip(shapes, sizes):
        out.append(np.asarray(x[pos:pos + size], dtype=np.float64).reshape(shape))
        pos += size
    return out


def bounds(theta, angle_deg: float = 15.0, translation_A: float = 0.5):
    result = []
    cap_ang = np.deg2rad(float(angle_deg))
    for th in theta:
        for i, val in enumerate(np.asarray(th, dtype=np.float64)):
            cap = float(translation_A) if i < 3 else cap_ang
            result.append((float(val - cap), float(val + cap)))
    return result


def score_and_grad(x, cap, shapes, sizes, trial):
    ts = [torch.tensor(a, dtype=torch.float64, device=cap.X_full.device, requires_grad=True)
          for a in unvectorize(x, shapes, sizes)]
    score = cap.score(ts, trial)
    grads = torch.autograd.grad(score, ts, allow_unused=False)
    grad = np.concatenate([g.detach().cpu().numpy().ravel() for g in grads])
    return -float(score.detach().cpu()), -grad


def xyz(cap, theta, trial):
    Xref = cap.X_full
    X = Xref[trial:trial + 1] if Xref.ndim == 3 else Xref.unsqueeze(0)
    X = X.to(dtype=torch.float64)
    with torch.no_grad():
        for fk, t in zip(cap.fks, theta):
            tt = torch.as_tensor(t, dtype=torch.float64, device=X.device).reshape(1, -1)
            X = fk.apply(tt, X)
    return X[0].detach().cpu().numpy()


def run_parent(code, stats, evaluator, workroot, device, angle_deg=15.0, translation_A=0.5, workers=4,
               run_xtb=True, save_structures=False):
    sh, ds = inputs.frame_shards()[code]
    mol = inputs.load_mol(code, sh, ds)
    builder, _, err = build_builder_and_inputs(mol)
    if builder is None:
        raise RuntimeError(err)
    with xtb.FallbackWatch(raise_on_exit=True):
        with CaptureSession(builder) as capture:
            solved = builder.solve(seed=11, stats=stats, angle_bends=True, device=device, n_conformers=16)
    capture.attach_conformers(solved.conformers)
    rows = []
    pending = []
    for rank, conf in enumerate(solved.conformers[:6], 1):
        row = int(conf.trial_idx)
        cap = capture.for_conformer(conf)
        theta0 = [np.asarray(t, dtype=np.float64) for t in conf.solver_state["thetas"]]
        s32 = float(cap.scores[row].detach().cpu())
        try:
            s0 = float(cap.score(theta0, row).detach().cpu())
        except Exception as exc:
            rows.append({"rank": rank, "key": [conf.embed_idx, conf.assignment_idx, row],
                         "status": "score_repro_failed", "score_error": str(exc)})
            continue
        stored = float(conf.score)
        score_diff = abs(s0 - stored)
        if score_diff >= 1e-4:
            rows.append({"rank": rank, "key": [conf.embed_idx, conf.assignment_idx, row],
                         "score_stored": stored, "score32": s32, "score64_theta0": s0,
                         "score_abs_diff": score_diff, "status": "score_assert_failed"})
            continue

        x0, shapes, sizes = vectorize(theta0)
        opt = minimize(score_and_grad, x0, args=(cap, shapes, sizes, row), method="L-BFGS-B", jac=True,
                       bounds=bounds(theta0, angle_deg, translation_A), options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30})
        theta1 = unvectorize(opt.x, shapes, sizes)
        s1 = float(cap.score(theta1, row).detach().cpu())
        X0, X1 = xyz(cap, theta0, row), xyz(cap, theta1, row)
        disps = np.linalg.norm(X1 - X0, axis=1)
        from tools.jgen.atom_export import symbols_for_conformer
        atom_symbols = symbols_for_conformer(conf)
        assert atom_symbols == [a.symbol for a in conf.complex.atoms]
        assert len(atom_symbols) == len(X0) == len(X1)
        atom_pos = {a.idx: i for i, a in enumerate(conf.complex.atoms)}
        donor_atoms = {atom_pos[int(d)] for lig in conf.complex.ligands for d in lig.donor_atoms if int(d) in atom_pos}
        metal_idx = atom_pos[int(conf.complex.metal_idx)]
        non_donors = [i for i, el in enumerate(atom_symbols) if i != metal_idx and i not in donor_atoms]
        contact = (float(np.min(np.linalg.norm(X1[non_donors] - X1[metal_idx], axis=1)))
                   if non_donors else None)
        flat0 = np.concatenate([t.ravel() for t in theta0]); flat1 = np.asarray(opt.x)
        bds = bounds(theta0, angle_deg, translation_A)
        at_bound = [i for i, (lo, hi) in enumerate(bds)
                    if abs(flat1[i] - lo) < 1e-5 or abs(flat1[i] - hi) < 1e-5]
        class_dof = {"translation": 0, "rotation": 0, "torsion": 0, "ring": 0, "bend": 0}
        disp_class = {k: [] for k in class_dof}
        cursor = 0
        for fk, th in zip(cap.fks, theta0):
            n_base = int(getattr(fk, "n_joints_base", fk.n_joints))
            n_branch_bend = int(getattr(fk, "n_bend_joints", 0))
            n_macro = int(getattr(fk, "n_macro_joints", 0))
            for i in range(len(th)):
                k = ("translation" if i < 3 else "rotation" if i < 6 else
                     "torsion" if i < 6 + n_base else
                     "bend" if i < 6 + n_base + n_branch_bend else
                     "ring" if i < 6 + n_base + n_branch_bend + n_macro else "bend")
                dv = flat1[cursor + i] - flat0[cursor + i]
                disp_class[k].append(float(dv if k == "translation" else np.rad2deg(dv)))
            cursor += len(th)
        dof_disp = {k: {"mean_abs": float(np.mean(np.abs(v))) if v else 0.0,
                        "max_abs": float(np.max(np.abs(v))) if v else 0.0,
                        "n": len(v)} for k, v in disp_class.items()}

        rec = {"rank": rank, "key": [conf.embed_idx, conf.assignment_idx, row],
               "status": "ok", "optimizer_success": bool(opt.success), "optimizer_message": str(opt.message),
               "iterations": int(opt.nit), "score_stored": stored, "score32_theta0": s32,
               "score64_theta0": s0, "score_abs_diff": score_diff, "score_star": s1,
               "delta_score": s1 - s0, "dof_displacement": dof_disp,
               "bound_dofs": at_bound, "n_bound_dofs": len(at_bound),
               "max_atom_displacement_A": float(disps.max()),
               "closest_non_donor_metal_contact_A": contact,
               "metal_contact_lt_1_2A": bool(contact is not None and contact < 1.2)}

        pending.append((rec, atom_symbols, X0, X1, int(mol.charge), int(mol.spin_state)))

    # Batch all valid rows for this parent: 2*rows xTB relaxations can run
    # concurrently, and MACE sees one batched call instead of a pair per row.
    unrel_structures = [item for _, symbols, X0, X1, charge, spin in pending
                        for item in ((symbols, X0, charge, spin), (symbols, X1, charge, spin))]
    mace_unrel = None
    if unrel_structures:
        try:
            mace_unrel = evaluator.energies_kj(unrel_structures)
            for i, (rec, symbols, *_rest) in enumerate(pending):
                e0, e1 = float(mace_unrel[2 * i]), float(mace_unrel[2 * i + 1])
                rec["E_unrel_init_kj"] = e0; rec["E_unrel_star_kj"] = e1
                rec["delta_E_unrel_kj"] = e1 - e0
                if save_structures:
                    rec["atom_symbols"] = symbols
                    rec["symbols_order"] = "builder/conf.complex.atoms"
                    rec["charge"] = int(mol.charge)
                    rec["spin_state"] = int(mol.spin_state)
                    rec["X_init_A"] = pending[i][2].tolist()
                    rec["X_star_A"] = pending[i][3].tolist()
        except Exception as exc:
            for rec, *_rest in pending:
                rec["mace_status"] = f"failed: {type(exc).__name__}: {exc}"

    if not run_xtb:
        for rec, *_rest in pending:
            rec["xtb_init"] = {"status": "skipped_by_unrelaxed_screen"}
            rec["xtb_star"] = {"status": "skipped_by_unrelaxed_screen"}
            rec["strain_status"] = "not measured in unrelaxed screen"
            rec["classification"] = ("c_barely_moves" if rec["delta_score"] <= 0.05
                                      else "awaiting_selected_relaxation")
        rows.extend(rec for rec, *_rest in pending)
        rows.sort(key=lambda r: r["rank"])
        return {"code": code, "n_assignments": (solved.assignment_search or {}).get("n_candidates", 1),
                "capture_count": len(capture.captures), "rows": rows}

    xtb_jobs = [job for _, symbols, X0, X1, charge, spin in pending
                for job in ((symbols, X0, charge, spin), (symbols, X1, charge, spin))]
    xtb_codes = [f"{code}:{rec['rank']}:{label}" for rec, *_rest in pending for label in ("init", "star")]
    xtb_results = xtb.opt_many(
        xtb_jobs, workers=max(1, int(workers)), workroot=workroot, cycles=200, level="crude",
        failures=str(Path(workroot) / "xtb_failures.jsonl"), codes=xtb_codes,
    ) if xtb_jobs else []

    relaxed_structures, relaxed_owners = [], []
    for i, (rec, symbols, _X0, _X1, charge, spin) in enumerate(pending):
        rel0, rel1 = xtb_results[2 * i], xtb_results[2 * i + 1]
        rec["xtb_init"] = {k: rel0.get(k) for k in ("status", "converged", "cycles", "wall_s", "E_h", "stderr_tail")}
        rec["xtb_star"] = {k: rel1.get(k) for k in ("status", "converged", "cycles", "wall_s", "E_h", "stderr_tail")}
        if (rel0.get("status") == "ok" and rel1.get("status") == "ok"
                and rel0.get("X") is not None and rel1.get("X") is not None):
            relaxed_structures.extend([(symbols, rel0["X"], charge, spin), (symbols, rel1["X"], charge, spin)])
            relaxed_owners.append((i, rec, float(mace_unrel[2 * i]), float(mace_unrel[2 * i + 1]))
                                  if mace_unrel is not None else (i, rec, None, None))
        else:
            rec["strain_status"] = "failed: xTB relaxation did not converge"

    if relaxed_structures and mace_unrel is not None:
        try:
            relaxed_energies = evaluator.energies_kj(relaxed_structures)
            for j, (_i, rec, e0, e1) in enumerate(relaxed_owners):
                er0, er1 = float(relaxed_energies[2 * j]), float(relaxed_energies[2 * j + 1])
                rec["strain_init_kj"] = float(e0 - er0)
                rec["strain_star_kj"] = float(e1 - er1)
                rec["delta_strain_kj"] = rec["strain_star_kj"] - rec["strain_init_kj"]
        except Exception as exc:
            for _i, rec, _e0, _e1 in relaxed_owners:
                rec["strain_status"] = f"failed: {type(exc).__name__}: {exc}"
    elif relaxed_structures and mace_unrel is None:
        for _i, rec, _e0, _e1 in relaxed_owners:
            rec["strain_status"] = "failed: unrelaxed MACE evaluation did not complete"

    for rec, *_rest in pending:
        if rec["delta_score"] <= 0.05:
            rec["classification"] = "c_barely_moves"
        elif rec.get("delta_strain_kj") is not None and rec["delta_strain_kj"] < -5:
            rec["classification"] = "a_optimizer_problem"
        elif rec.get("delta_strain_kj") is not None:
            rec["classification"] = "b_score_minimum_strained"
        else:
            rec["classification"] = "unknown_strain_failed"
    rows.extend(rec for rec, *_rest in pending)
    rows.sort(key=lambda r: r["rank"])
    return {"code": code, "n_assignments": (solved.assignment_search or {}).get("n_candidates", 1),
            "capture_count": len(capture.captures), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--parents", nargs="*", default=PARENTS)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--angular-cap-deg", type=float, default=15.0)
    ap.add_argument("--translation-cap-A", type=float, default=0.5)
    ap.add_argument("--mace-device", default=None,
                    help="MACE device (defaults to CUDA when available, else CPU); CUDA holds the CodexB GPU lock")
    ap.add_argument("--skip-xtb", action="store_true",
                    help="run score/MACE/geometry screen only; do not launch xTB relaxations")
    ap.add_argument("--save-structures", action="store_true",
                    help="include initial and optimized coordinates in the output for a selected relaxation stage")
    args = ap.parse_args()
    xtb.require()
    stats = stats_from_db(inputs.DB_PATH)
    mace_device = args.mace_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if mace_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"MACE device {mace_device!r} requested but CUDA is unavailable")
    outpath = Path(args.out); outpath.parent.mkdir(parents=True, exist_ok=True)
    run = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": args.device,
           "mace_device": mace_device,
           "xtb_enabled": not args.skip_xtb, "structures_saved": args.save_structures,
           "angular_cap_deg": args.angular_cap_deg, "translation_cap_A": args.translation_cap_A,
           "parents": {}, "failures": {}}
    workroot = str(Path(args.out).resolve().parent / "o1_xtb_work")
    lock = resources.gpu_lock("CodexB") if mace_device.startswith("cuda") else nullcontext()
    with lock:
        evaluator = mace.Evaluator(device=mace_device, dtype="float64", threads=1)
        for code in args.parents:
            try:
                run["parents"][code] = run_parent(code, stats, evaluator, workroot, args.device,
                                                   args.angular_cap_deg, args.translation_cap_A, args.workers,
                                                   run_xtb=not args.skip_xtb, save_structures=args.save_structures)
            except Exception as exc:
                run["failures"][code] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()}
            outpath.write_text(json.dumps(run, indent=2))
            print(f"[{code}] rows={len(run['parents'].get(code, {}).get('rows', []))} "
                  f"failure={run['failures'].get(code, {}).get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
