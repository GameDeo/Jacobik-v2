"""
run_benchmark.py
================
Lightweight tmQM benchmark runner for Jacobik.

Version history
---------------
1.0.0 : Initial release. Streams a tmQM split, runs the engine on each
        complex with pre-loaded stats DB, computes CS/Full/Local RMSD +
        optional GFN2-xTB ΔE for rank-1 and best-RMSD conformers, and
        writes one JSON record per complex to a log file.
        Stores full solve profiling: rotatable bonds, n_atoms, solve time,
        per-beam convergence step, score trace statistics.

Usage
-----
    python run_benchmark.py /path/to/tmQM --split X1 --limit 200 \\
        --out results_X1.jsonl --xtb --beam-width 16

Output
------
One JSON object per line (JSON Lines / JSONL format) in --out file.
Each record is self-contained and can be analysed independently.
The notebook benchmark_analysis.ipynb reads this file directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("run_benchmark")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------


def _count_rotatable_bonds(result) -> Dict:
    """
    Extract per-ligand and total rotatable bond counts from the engine result.
    Uses fk.n_joints (= number of torsion DOFs per ligand).
    """
    total = 0
    per_ligand = []
    for cf in result.conformers[:1]:  # all conformers share same topology
        for lig in cf.complex.ligands:
            # n_joints stored as ligand attribute if engine set it, else 0
            j = getattr(lig, "n_joints", None)
            if j is None:
                j = len(lig.rotatable_bonds)
            per_ligand.append(j)
            total += j
    return {"total_rotatable_bonds": total, "per_ligand": per_ligand}


def _extract_solve_trace(result) -> Dict:
    """
    Extract convergence profiling from result.solve_trace.

    solve_trace keys (T=n_ik_steps, B=beam_width):
      residual   : (T, B)  combined loss per step per beam
      score      : (T, B)  statistical score per step per beam
      anchor_loss: (T, B)  anchor term per step
      grad_norm  : (T, B)  gradient norm per step

    Returns a dict of scalar statistics for JSON serialisation.
    """
    if not result.solve_trace:
        return {}

    tr = result.solve_trace
    score_hist = tr.get("score")       # (T, B) or None
    resid_hist = tr.get("residual")
    grad_hist  = tr.get("grad_norm")

    out: Dict = {}

    if score_hist is not None:
        T, B = score_hist.shape
        out["n_ik_steps_run"] = int(T)
        out["beam_width"]     = int(B)

        # Per-beam: step at which score first plateaued (stopped improving >1e-4)
        # This is the effective convergence step.
        plateau_steps = []
        for b in range(B):
            s = score_hist[:, b]
            plateau = T  # default: never plateaued
            for t in range(1, T):
                if abs(float(s[t]) - float(s[t-1])) < 1e-4:
                    plateau = t
                    break
            plateau_steps.append(int(plateau))

        out["iter_to_plateau_per_beam"] = plateau_steps
        out["iter_to_plateau_mean"]     = float(np.mean(plateau_steps))
        out["iter_to_plateau_min"]      = int(np.min(plateau_steps))
        out["iter_to_plateau_max"]      = int(np.max(plateau_steps))

        # Final score distribution
        final_scores = score_hist[-1].tolist()
        out["final_scores_per_beam"]  = [float(s) for s in final_scores]
        out["final_score_mean"]       = float(np.mean(final_scores))
        out["final_score_std"]        = float(np.std(final_scores))
        out["final_score_best"]       = float(np.max(final_scores))

        # Score improvement: first step vs last step
        out["score_improvement_mean"] = float(
            np.mean(score_hist[-1] - score_hist[0])
        )

    if resid_hist is not None:
        out["final_residual_mean"] = float(resid_hist[-1].mean())
        out["final_residual_min"]  = float(resid_hist[-1].min())

    if grad_hist is not None:
        out["final_grad_norm_mean"] = float(grad_hist[-1].mean())
        out["grad_norm_initial"]    = float(grad_hist[0].mean())

    return out


def _mol_profile(mol) -> Dict:
    """Structural profile of the reference complex."""
    atom_map = {a.idx: a for a in mol.atoms}
    n_heavy  = sum(1 for a in mol.atoms if a.symbol != "H")
    n_h      = len(mol.atoms) - n_heavy

    # Per-ligand heavy atom counts
    lig_heavy = []
    lig_donors = []
    for lig in sorted(mol.ligands, key=lambda l: l.ligand_id):
        lh = sum(1 for idx in lig.atom_indices
                 if atom_map.get(idx) and atom_map[idx].symbol != "H")
        lig_heavy.append(lh)
        ld = len(lig.donor_atoms)
        lig_donors.append(ld)

    return {
        "n_atoms_total":   len(mol.atoms),
        "n_heavy":         n_heavy,
        "n_hydrogen":      n_h,
        "n_ligands":       len(mol.ligands),
        "max_lig_heavy":   max(lig_heavy) if lig_heavy else 0,
        "lig_heavy_atoms": lig_heavy,
        "lig_donor_counts": lig_donors,
        "coord_number":    mol.coordination_number(),
        "coord_geom":      mol.coord_geom or "unknown",
        "metal":           mol.metal_symbol,
        "charge":          mol.charge,
        "spin":            mol.spin_state,
    }


def _rmsd_profile(mol, result, ligand_groups) -> Dict:
    """Compute CS / Full / Local RMSD for rank-1 and best conformers."""
    # Import here to avoid circular deps — benchmark.py must be on path
    from benchmark import (
        _build_matched_arrays, sym_rmsd, coord_sphere_rmsd,
        per_ligand_local_rmsd, kabsch_rmsd,
    )

    out: Dict = {}
    if not result.conformers:
        return out

    rank1      = result.conformers[0]
    conformers = result.conformers

    # Find best by full sym_rmsd
    best_rmsd_val = float("inf")
    best_conf     = rank1
    best_rank     = 1

    all_full_rmsds = []
    all_cs_rmsds   = []
    all_local_rmsds = []
    all_scores     = []

    for ci, cf in enumerate(conformers):
        matched = _build_matched_arrays(mol, cf, ligand_groups)
        if matched is None:
            all_full_rmsds.append(float("nan"))
            all_cs_rmsds.append(float("nan"))
            all_local_rmsds.append(float("nan"))
            all_scores.append(float(cf.score))
            continue

        pm, rm, rxl, prl = matched
        sr   = sym_rmsd(pred_matched=pm, ref_matched=rm,
                        ligand_groups=ligand_groups,
                        heavy_atom_map=rxl, pred_heavy_map=prl,
                        max_perms=512)
        cs_r, _ = coord_sphere_rmsd(mol, cf, ligand_groups, max_bonds=2)

        loc_results  = per_ligand_local_rmsd(mol, cf, ligand_groups)
        valid_locals = [r for _, r, _ in loc_results if r == r]
        loc_r = float(np.mean(valid_locals)) if valid_locals else float("nan")

        all_full_rmsds.append(sr.rmsd)
        all_cs_rmsds.append(cs_r if cs_r == cs_r else float("nan"))
        all_local_rmsds.append(loc_r)
        all_scores.append(float(cf.score))

        if sr.rmsd < best_rmsd_val:
            best_rmsd_val = sr.rmsd
            best_conf     = cf
            best_rank     = ci + 1

    out["rank1_full_rmsd"]   = all_full_rmsds[0]
    out["rank1_cs_rmsd"]     = all_cs_rmsds[0]
    out["rank1_local_rmsd"]  = all_local_rmsds[0]
    out["rank1_score"]       = all_scores[0]
    out["rank1_converged"]   = bool(rank1.converged)

    def _nanmin(arr):
        valid = [x for x in arr if x == x]
        return float(min(valid)) if valid else float("nan")

    out["best_full_rmsd"]    = _nanmin(all_full_rmsds)
    out["best_cs_rmsd"]      = _nanmin(all_cs_rmsds)
    out["best_local_rmsd"]   = _nanmin(all_local_rmsds)
    out["best_rank"]         = best_rank
    out["n_conformers"]      = len(conformers)
    out["n_converged"]       = sum(1 for c in conformers if c.converged)
    out["all_full_rmsds"]    = all_full_rmsds
    out["all_cs_rmsds"]      = all_cs_rmsds
    out["all_local_rmsds"]   = all_local_rmsds
    out["all_scores"]        = all_scores

    # Local shape RMSD (rank-1 only — per-ligand)
    loc_results_r1  = per_ligand_local_rmsd(mol, rank1, ligand_groups)
    out["rank1_local_rmsd_per_lig"] = [
        {"lig_id": lid, "rmsd": (r if r == r else None), "n_atoms": n}
        for lid, r, n in loc_results_r1
    ]

    # Scorer correlation: does rank-1 have the best RMSD?
    out["scorer_selected_best"] = (best_rank == 1)
    if len(all_full_rmsds) > 1:
        finite = [(s, r) for s, r in zip(all_scores, all_full_rmsds) if r == r]
        if len(finite) > 1:
            scores_arr = np.array([s for s, _ in finite])
            rmsds_arr  = np.array([r for _, r in finite])
            # Spearman rank correlation: does higher score → lower RMSD?
            from scipy.stats import spearmanr
            try:
                corr, _ = spearmanr(scores_arr, -rmsds_arr)
                out["score_rmsd_spearman"] = float(corr)
            except Exception:
                out["score_rmsd_spearman"] = None

    return out


def _xtb_profile(mol, result, best_conf) -> Dict:
    """Run GFN2-xTB on rank-1 and best conformer (heavy atoms only)."""
    from benchmark import xtb_energy_gap

    out: Dict = {}
    if not result.conformers:
        return out

    rank1 = result.conformers[0]
    dkj_r1, _, _ = xtb_energy_gap(mol, rank1,     mol.charge, mol.spin_state)
    out["xtb_rank1_delta_kj"] = dkj_r1 if dkj_r1 == dkj_r1 else None

    if best_conf is not rank1:
        dkj_best, _, _ = xtb_energy_gap(mol, best_conf, mol.charge, mol.spin_state)
        out["xtb_best_delta_kj"] = dkj_best if dkj_best == dkj_best else None
    else:
        out["xtb_best_delta_kj"] = out["xtb_rank1_delta_kj"]

    return out


# ---------------------------------------------------------------------------
# Per-complex benchmark
# ---------------------------------------------------------------------------


def _run_one(
    mol,
    preloaded_stats,
    ligand_groups,
    lig_inputs,
    beam_width: int,
    n_ik_steps: int,
    n_coord_samples: int,
    seed: int,
    device: str,
    use_xtb: bool,
) -> Dict:
    """Run the engine on one parsed complex and return a flat result dict."""
    from jacobik.engine import ComplexBuilder

    record: Dict = {
        "csd_code": mol.csd_code,
        "error":    None,
    }
    record.update(_mol_profile(mol))

    geom = mol.coord_geom or "other"

    # ── Engine solve ─────────────────────────────────────────────────────────
    try:
        builder = ComplexBuilder(
            metal    = mol.metal_symbol,
            geometry = geom,
            charge   = mol.charge,
            spin     = mol.spin_state,
            ligands  = lig_inputs,
        )
    except Exception as exc:
        record["error"] = f"builder: {exc}"
        return record

    t0 = time.perf_counter()
    try:
        result = builder.solve(
            stats           = preloaded_stats,
            beam_width      = beam_width,
            n_ik_steps      = n_ik_steps,
            n_coord_samples = n_coord_samples,
            seed            = seed,
            device          = device,
            fast_embed      = True,   # skip MMFF — IK overwrites geometry
        )
    except Exception as exc:
        record["error"] = f"solve: {exc}"
        record["solve_time_s"] = time.perf_counter() - t0
        return record

    record["solve_time_s"] = time.perf_counter() - t0

    if not result.conformers:
        record["error"] = "no_conformers"
        return record

    # ── Solve profiling ───────────────────────────────────────────────────────
    record.update(_extract_solve_trace(result))
    record.update(_count_rotatable_bonds(result))

    # ── RMSD metrics ──────────────────────────────────────────────────────────
    t_rmsd = time.perf_counter()
    rmsd_data = _rmsd_profile(mol, result, ligand_groups)
    record.update(rmsd_data)
    record["rmsd_compute_time_s"] = time.perf_counter() - t_rmsd

    # ── xTB energy gap ────────────────────────────────────────────────────────
    if use_xtb:
        best_rank  = rmsd_data.get("best_rank", 1)
        best_conf  = result.conformers[best_rank - 1]
        t_xtb = time.perf_counter()
        record.update(_xtb_profile(mol, result, best_conf))
        record["xtb_compute_time_s"] = time.perf_counter() - t_xtb

    return record


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    from jacobik.stats.parser import TMQMParser
    from jacobik.stats.stats  import StatsLookup, load_db
    from benchmark import _build_ligand_groups, ligand_inputs_from_mol

    # ── Import tqdm if available ──────────────────────────────────────────────
    try:
        from tqdm import tqdm as _tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Pre-load stats DB once ────────────────────────────────────────────────
    db_path = args.constraints_db
    if db_path is None:
        from importlib import resources
        db_path = resources.files("jacobik") / "constraints.json"
    print(f"Loading constraints DB: {db_path} ...")
    stats = StatsLookup(load_db(db_path))
    print("  ✓ DB loaded\n")

    # ── Stream complexes ──────────────────────────────────────────────────────
    parser   = TMQMParser(data_dir)
    mol_iter = parser.stream_split(args.split, limit=args.limit)

    if use_tqdm:
        pbar = _tqdm(mol_iter, unit="mol", dynamic_ncols=True, smoothing=0.05)
        mol_iter = pbar
    else:
        pbar = None

    n_done = n_err = n_skip = 0
    running_full: List[float] = []
    running_cs:   List[float] = []

    with open(out_path, "w", encoding="utf-8") as fout:
        for mol in mol_iter:
            # ── Skip unparseable complexes ────────────────────────────────────
            if not mol.smiles or not mol.ligands or not mol.coord_sites:
                skip_rec = {
                    "csd_code": mol.csd_code, "error": "no_smiles_or_ligands",
                    "metal": mol.metal_symbol, "coord_geom": mol.coord_geom,
                }
                fout.write(json.dumps(skip_rec) + "\n")
                n_skip += 1
                continue

            # ── Build ligand groups ───────────────────────────────────────────
            try:
                ligand_groups = _build_ligand_groups(mol, mol.smiles)
                lig_inputs    = ligand_inputs_from_mol(mol, ligand_groups)
                if lig_inputs is None:
                    raise ValueError("ligand_inputs_from_mol returned None")
            except Exception as exc:
                err_rec = {
                    "csd_code": mol.csd_code, "error": f"ligand_prep: {exc}",
                    "metal": mol.metal_symbol, "coord_geom": mol.coord_geom,
                }
                fout.write(json.dumps(err_rec) + "\n")
                n_err += 1
                continue

            # ── Run benchmark ─────────────────────────────────────────────────
            try:
                record = _run_one(
                    mol             = mol,
                    preloaded_stats = stats,
                    ligand_groups   = ligand_groups,
                    lig_inputs      = lig_inputs,
                    beam_width      = args.beam_width,
                    n_ik_steps      = args.n_ik_steps,
                    n_coord_samples = args.n_coord_samples,
                    seed            = args.seed,
                    device          = args.device,
                    use_xtb         = args.xtb,
                )
            except Exception as exc:
                record = {
                    "csd_code": mol.csd_code,
                    "error":    f"uncaught: {exc}\n{traceback.format_exc()}",
                    "metal":    mol.metal_symbol,
                    "coord_geom": mol.coord_geom,
                }
                n_err += 1

            # ── Write record ──────────────────────────────────────────────────
            fout.write(json.dumps(record, default=str) + "\n")
            fout.flush()

            if record.get("error"):
                n_err += 1
                log.warning(f"Failed {mol.csd_code}: {record.get('error')}")
            else:
                n_done += 1
                full_r = record.get("rank1_full_rmsd")
                cs_r   = record.get("rank1_cs_rmsd")
                if full_r and full_r == full_r:
                    running_full.append(full_r)
                if cs_r and cs_r == cs_r:
                    running_cs.append(cs_r)
                
                msg = f"Processed {mol.csd_code}"
                if full_r is not None and full_r == full_r:
                    msg += f" (Full: {full_r:.2f}A, CS: {cs_r:.2f}A)"
                log.info(msg)

            # ── Progress bar postfix ──────────────────────────────────────────
            if pbar is not None:
                mean_full = f"{np.mean(running_full):.3f}" if running_full else "—"
                mean_cs   = f"{np.mean(running_cs):.3f}"   if running_cs   else "—"
                pbar.set_postfix({
                    "full": mean_full, "CS": mean_cs,
                    "err":  n_err, "done": n_done,
                }, refresh=False)

    if pbar is not None:
        pbar.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  Benchmark complete: {n_done} valid, {n_err} errors, {n_skip} skipped")
    if running_full:
        print(f"  Mean Full RMSD  : {np.mean(running_full):.3f} Å  "
              f"(median {np.median(running_full):.3f})")
    if running_cs:
        print(f"  Mean CS RMSD    : {np.mean(running_cs):.3f} Å  "
              f"(median {np.median(running_cs):.3f})")
    print(f"  Results written : {out_path}")
    print(f"{'='*56}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast tmQM benchmark runner for Jacobik.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("data_dir",             help="tmQM dataset root directory")
    ap.add_argument("--split",    default="X1",       help="tmQM split")
    ap.add_argument("--out",      default="results.jsonl", help="Output JSONL file")
    ap.add_argument("--constraints-db", default=None, help="Path to constraints.json")
    ap.add_argument("--limit",    type=int, default=None,  help="Max complexes to run")
    ap.add_argument("--beam-width",      type=int, default=8)
    ap.add_argument("--n-ik-steps",      type=int, default=300)
    ap.add_argument("--n-coord-samples", type=int, default=4)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--device",   default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--xtb",      action="store_true", help="Enable GFN2-xTB ΔE")
    ap.add_argument("--log-level",default="INFO",
                    choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    _setup_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()