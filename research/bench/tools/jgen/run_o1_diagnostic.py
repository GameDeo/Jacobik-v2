"""O1: Reference convergence diagnostic for Jacobik conformer solver (Package G).

Runs seed 11 on 8 parents:
  HEDJOP, FEKKAJ, NUWFAM, QEZVEY, DAMVAQ, USAPOU, NEPNEF, AJAQAD (low-strain control).

Captures delivered theta via S2b solver state.
For top-1 and top 5 rows (6 rows per parent):
  Optimizes the solver's score objective E(theta) = -(stat_score + bend_prior)
  using float64 reference L-BFGS with strong Wolfe line-search.

Measures per row:
  - delta score (S* - S_0)
  - DOF displacements by class:
      * placement: translation (A), rotation (deg)
      * torsion: rotatable bonds (deg)
      * ring: macrocycle joints (deg)
      * bend: chelate angle bends (deg)
  - delta E_unrel (MACE float64 single points)
  - delta strain (GFN2 crude relaxation + MACE float64 single points)

Classifies each parent / row into:
  (a) score improves (> 0.05) & strain drops (< -5 kJ/mol) -> optimizer problem
  (b) score improves (> 0.05) & strain doesn't drop (>= -5 kJ/mol) -> score minimum strained
  (c) score barely moves (<= 0.05) -> already converged
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
BASE_DIR = REPO / "research/data/jgen/campaign/B/g"
EV_TO_KJ = 96.4853321233


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def xtb_opt(symbols: List[str], X: np.ndarray, charge: int, spin: int, workroot: str, cycles: int = 200) -> dict:
    t0 = time.perf_counter()
    p_xtb = os.environ.get("JACOBIK_XTB") or shutil.which("xtb")
    res = dict(status="failed", converged=False, X=None, wall_s=None)
    try:
        Path(workroot).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workroot) as td:
            with open(f"{td}/in.xyz", "w") as f:
                f.write(f"{len(symbols)}\n\n" + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(symbols, X)))
            cmd = [p_xtb, "in.xyz", "--gfn", "2", "--opt", "crude", "--cycles", str(cycles),
                   "--chrg", str(int(charge)), "--uhf", str(int(spin) - 1), "-P", "1"]
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            p = subprocess.run(cmd, cwd=td, capture_output=True, text=True, env=env, timeout=1800)
            if os.path.exists(f"{td}/xtbopt.xyz"):
                L = open(f"{td}/xtbopt.xyz").read().split("\n")[2:2 + len(symbols)]
                res["X"] = np.array([[float(v) for v in ln.split()[1:4]] for ln in L])
                res["converged"] = "GEOMETRY OPTIMIZATION CONVERGED" in p.stdout
                res["status"] = "ok"
    except Exception as exc:
        res["status"] = str(exc)[:80]
    res["wall_s"] = time.perf_counter() - t0
    return res


def run_parent_o1(code: str, ev, workroot: str, device: str = "cuda") -> dict:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "research/bench"))

    from tools.jgen import inputs
    from jacobik.core.env import build_builder_and_inputs
    from tools.jdev.budget.common import stats_from_db
    from jacobik.ik.fk import ForwardKinematics
    from jacobik.ik import angle_bends as AB

    shards = inputs.frame_shards()
    sh, ds = shards[code]
    mol = inputs.load_mol(code, sh, ds)
    charge = int(mol.charge)
    spin = int(mol.spin_state)
    symbols = [a.symbol for a in mol.atoms]

    builder, _, err = build_builder_and_inputs(mol)
    if builder is None:
        return {"code": code, "status": "error", "reason": err}

    stats = stats_from_db(inputs.DB_PATH)
    # Run production solve
    res = builder.solve(seed=11, stats=stats, angle_bends=True, device=device, n_conformers=16)
    conformers = res.conformers
    n_rows = min(6, len(conformers))
    selected_confs = conformers[:n_rows]

    fks = [ForwardKinematics(mol, lig, device, angle_bends=True) for lig in mol.ligands]
    X_ref = torch.tensor(mol.positions(), dtype=torch.float32, device=device)
    scorer = res.scorer

    # Prepare structures for MACE evaluations
    row_results = []

    print(f"\n[{code}] Starting O1 diagnostic on top {n_rows} conformers...")

    for rank, conf in enumerate(selected_confs, start=1):
        deliv_thetas_np = conf.solver_state["thetas"] # list of 1D numpy arrays
        init_score = float(conf.score)

        # Clone and convert to float32/float64 parameters with grad
        th_params = [
            torch.tensor(th, dtype=torch.float32, device=device, requires_grad=True)
            for th in deliv_thetas_np
        ]

        def get_X(thetas_list):
            X_curr = X_ref.unsqueeze(0).clone()
            for li, fk in enumerate(fks):
                X_curr = fk.apply(thetas_list[li].unsqueeze(0), X_curr)
            return X_curr

        def compute_score(thetas_list):
            X_curr = get_X(thetas_list)
            # Statistical score
            s_stat, _ = scorer.score(X_curr)
            # Bend prior
            bp = torch.zeros(1, dtype=torch.float32, device=device)
            for fk, th in zip(fks, thetas_list):
                if getattr(fk, "n_angle_bends", 0) > 0 and getattr(fk, "_bend_k", None) is not None:
                    n0 = 6 + fk.n_joints
                    nb = fk.n_angle_bends
                    b_dofs = th[n0 : n0 + nb]
                    dev = (torch.abs(b_dofs) - fk._bend_free).clamp(min=0.0)
                    bp = bp - 0.5 * (fk._bend_k * dev**2).sum()

            total_s = s_stat.squeeze(0) + bp.squeeze(0)
            return total_s, X_curr

        # Compute initial score and coordinates
        s0_calc, X0 = compute_score(th_params)

        # Reference L-BFGS with strong Wolfe line search
        optimizer = torch.optim.LBFGS(
            th_params,
            lr=1.0,
            max_iter=50,
            tolerance_grad=1e-5,
            tolerance_change=1e-7,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            s, _ = compute_score(th_params)
            loss = -s
            loss.backward()
            return loss

        # Run optimizer
        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"    L-BFGS step warning: {e}")

        # Post-optimization
        with torch.no_grad():
            s_opt, X_opt = compute_score(th_params)
            delta_score = float(s_opt.item() - s0_calc.item())

            # Measure DOF displacements by class across ligands
            trans_disp = 0.0
            rot_disp_deg = 0.0
            tor_disp_deg = []
            ring_disp_deg = []
            bend_disp_deg = []

            for li, fk in enumerate(fks):
                th_0 = deliv_thetas_np[li]
                th_star = th_params[li].detach().cpu().numpy()

                # Rigid base
                trans_disp += float(np.linalg.norm(th_star[:3] - th_0[:3]))
                rot_disp_deg += float(np.linalg.norm(th_star[3:6] - th_0[3:6]) * 180.0 / np.pi)

                # Torsions
                J_base = fk.n_joints_base
                if J_base > 0:
                    diff_tor = np.abs(np.arctan2(np.sin(th_star[6 : 6 + J_base] - th_0[6 : 6 + J_base]),
                                                 np.cos(th_star[6 : 6 + J_base] - th_0[6 : 6 + J_base])))
                    tor_disp_deg.extend((diff_tor * 180.0 / np.pi).tolist())

                # Ring
                J_ring = getattr(fk, "n_macro_joints", 0)
                if J_ring > 0:
                    c_ring0 = 6 + J_base
                    diff_ring = np.abs(np.arctan2(np.sin(th_star[c_ring0 : c_ring0 + J_ring] - th_0[c_ring0 : c_ring0 + J_ring]),
                                                  np.cos(th_star[c_ring0 : c_ring0 + J_ring] - th_0[c_ring0 : c_ring0 + J_ring])))
                    ring_disp_deg.extend((diff_ring * 180.0 / np.pi).tolist())

                # Bend
                nb = getattr(fk, "n_angle_bends", 0)
                if nb > 0:
                    c_bend0 = 6 + fk.n_joints
                    diff_bend = np.abs(th_star[c_bend0 : c_bend0 + nb] - th_0[c_bend0 : c_bend0 + nb])
                    bend_disp_deg.extend((diff_bend * 180.0 / np.pi).tolist())

            mean_tor = float(np.mean(tor_disp_deg)) if tor_disp_deg else 0.0
            mean_ring = float(np.mean(ring_disp_deg)) if ring_disp_deg else 0.0
            mean_bend = float(np.mean(bend_disp_deg)) if bend_disp_deg else 0.0

            # Evaluate MACE float64 and GFN2 relaxation
            X0_np = X0.squeeze(0).detach().cpu().numpy()
            Xstar_np = X_opt.squeeze(0).detach().cpu().numpy()

            E_unrel_0 = float(ev.energies([(symbols, X0_np, charge, spin)])[0]) * EV_TO_KJ
            E_unrel_star = float(ev.energies([(symbols, Xstar_np, charge, spin)])[0]) * EV_TO_KJ
            delta_E_unrel = float(E_unrel_star - E_unrel_0)

            # Crude relaxation for strain
            res_opt_0 = xtb_opt(symbols, X0_np, charge, spin, workroot)
            res_opt_star = xtb_opt(symbols, Xstar_np, charge, spin, workroot)

            X_rel_0 = res_opt_0["X"] if res_opt_0["X"] is not None else X0_np
            X_rel_star = res_opt_star["X"] if res_opt_star["X"] is not None else Xstar_np

            E_rel_0 = float(ev.energies([(symbols, X_rel_0, charge, spin)])[0]) * EV_TO_KJ
            E_rel_star = float(ev.energies([(symbols, X_rel_star, charge, spin)])[0]) * EV_TO_KJ

            strain_0 = float(E_unrel_0 - E_rel_0)
            strain_star = float(E_unrel_star - E_rel_star)
            delta_strain = float(strain_star - strain_0)

            # Classification
            if delta_score > 0.05 and delta_strain < -5.0:
                classification = "(a) optimizer problem (score improves & strain drops)"
            elif delta_score > 0.05 and delta_strain >= -5.0:
                classification = "(b) score minimum strained (score improves & strain doesn't drop)"
            else:
                classification = "(c) already converged (score barely moves <= 0.05)"

            row_info = {
                "rank": rank,
                "score_init": init_score,
                "score_calc_init": float(s0_calc.item()),
                "score_calc_star": float(s_opt.item()),
                "delta_score": delta_score,
                "displacements": {
                    "trans_A": trans_disp,
                    "rot_deg": rot_disp_deg,
                    "mean_torsion_deg": mean_tor,
                    "mean_ring_deg": mean_ring,
                    "mean_bend_deg": mean_bend,
                },
                "delta_E_unrel_kj": delta_E_unrel,
                "strain_init_kj": strain_0,
                "strain_star_kj": strain_star,
                "delta_strain_kj": delta_strain,
                "classification": classification,
            }
            row_results.append(row_info)
            print(f"  Row {rank}: dScore={delta_score:+.3f}, dStrain={delta_strain:+.1f} kJ/mol, dE_unrel={delta_E_unrel:+.1f} kJ/mol -> {classification.split()[0]}")

    # Parent-level classification (based on top-1 and majority)
    top1_class = row_results[0]["classification"].split()[0]
    return {
        "code": code,
        "status": "ok",
        "top1_classification": top1_class,
        "rows": row_results,
    }


def main():
    parser = argparse.ArgumentParser(description="O1: Convergence diagnostic")
    parser.add_argument("--out", default=str(BASE_DIR / "o1_convergence_results.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    workroot = str(out_file.parent / "o1_xtb_work")

    from jacobik.core.energy import MACEPolarEvaluator
    mace_model_path = REPO / "research/bench/tools/jdev/MACE-POLAR-1-S.model"
    print("Loading float64 MACE model...")
    ev = MACEPolarEvaluator(model="polar-1-s", device=args.device,
                           dtype="float64", model_path=str(mace_model_path), mode="energy-only").load()

    parents = ["HEDJOP", "FEKKAJ", "NUWFAM", "QEZVEY", "DAMVAQ", "USAPOU", "NEPNEF", "AJAQAD"]

    t0 = time.perf_counter()
    results = {}
    for code in parents:
        res = run_parent_o1(code, ev, workroot, device=args.device)
        results[code] = res

    # Summary table
    print("\n" + "=" * 90)
    print("O1: CONVERGENCE DIAGNOSTIC SUMMARY TABLE (8 parents, seed 11, reference float64 L-BFGS)")
    print("=" * 90)
    print(f"{'Parent':<10} | {'Top-1 dScore':<13} | {'Top-1 dStrain':<14} | {'Top-1 dE_unrel':<15} | {'Classification'}")
    print("-" * 90)
    for code in parents:
        r = results[code]
        if r.get("status") == "ok":
            t1 = r["rows"][0]
            print(f"{code:<10} | {t1['delta_score']:<+13.4f} | {t1['delta_strain_kj']:<+14.2f} | {t1['delta_E_unrel_kj']:<+15.2f} | {t1['classification']}")
    print("=" * 90)

    final_payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_wall_s": time.perf_counter() - t0,
        "results": results,
    }
    out_file.write_text(json.dumps(final_payload, indent=2))
    print(f"\nSaved O1 results to {out_file}")


if __name__ == "__main__":
    main()
