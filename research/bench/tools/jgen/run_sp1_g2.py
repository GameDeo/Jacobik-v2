"""Recount S-P1 beam coverage by heavy-atom rotor mode and single-joint clashes."""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / "src"), str(REPO), str(REPO / "research/bench")]
OUT = REPO / "research/data/jgen/campaign/B/g/sp1_g2_results.json"

from tools.jgen import inputs
from tools.jgen.bench import panels
from tools.jgen.score_repro import CaptureSession
from jacobik.core.env import build_builder_and_inputs
from tools.jdev.budget.common import stats_from_db
from tools.runkit import xtb


def _neighbors(mol):
    pos = {a.idx: i for i, a in enumerate(mol.atoms)}
    adj = {i: [] for i in range(len(mol.atoms))}
    for b in mol.bonds:
        if b.order >= 0.5 and b.i != mol.metal_idx and b.j != mol.metal_idx:
            i, j = pos[b.i], pos[b.j]
            adj[i].append(j); adj[j].append(i)
    return adj


def _signature(mol, adj, atom, parent, depth=0, seen=None):
    seen = set() if seen is None else set(seen)
    if atom in seen or depth > 5:
        return (mol.atoms[atom].symbol, "cycle")
    seen.add(atom)
    children = [_signature(mol, adj, n, atom, depth + 1, seen) for n in adj[atom] if n != parent]
    return (mol.atoms[atom].symbol, tuple(sorted(children, key=repr)))


def _rotor_period(mol, adj, axis_a, axis_b):
    counts = []
    for atom, parent in ((axis_a, axis_b), (axis_b, axis_a)):
        sigs = [_signature(mol, adj, n, atom) for n in adj[atom]
                if n != parent]
        counts.append(max(Counter(sigs).values(), default=1))
    n = max(counts)
    return n if n in (2, 3) else 1


def _fold_distance(a, b, period):
    d = np.arctan2(np.sin((a - b) * period), np.cos((a - b) * period))
    return abs(float(d)) / period


def _best_initial_row(cap):
    theta = [t.detach() for t in cap.theta_inits]
    B = theta[0].shape[0]
    Xref = cap.X_full
    if Xref.ndim == 2:
        X = Xref.unsqueeze(0).expand(B, -1, -1)
    elif Xref.shape[0] == 1:
        X = Xref.expand(B, -1, -1)
    else:
        X = Xref
    for fk, t in zip(cap.fks, theta):
        X = fk.apply(t, X)
    kwargs = dict(cap.score_kwargs)
    kwargs["row_groups"] = cap.row_groups
    scorer_score, _ = cap.scorer.score(X, **kwargs)
    score = scorer_score
    if cap.bend_prior:
        from jacobik.ik import angle_bends as AB
        from jacobik.ik.ik_solver import _BEND_PER_Z
        bend = score.new_zeros((B,))
        for fk, th in zip(cap.fks, theta):
            nb = getattr(fk, "n_angle_bends", 0)
            if nb:
                start = 6 + fk.n_joints
                free = getattr(fk, "_bend_free", None)
                free = torch.clamp(free, min=float(cap.bend_prior["free_deg"])) if free is not None else cap.bend_prior["free_deg"]
                bend = bend + AB.energy_prior(th[:, start:start + nb], fk._bend_k, free, cap.bend_prior["scale"])
        if _BEND_PER_Z and hasattr(cap.scorer, "_agg_norm"):
            bend = bend / float(cap.scorer._agg_norm())
        score = score + bend
    for fk in cap.fks:
        if getattr(fk, "_cl", None) is not None:
            d0 = fk.seam_d0(Xref)
            residual, _ = fk.closure_residual_jacobian(X, d0)
            score = score - 1000.0 * (residual * residual).sum(dim=-1)
    return score.detach()


def _coords(cap, theta_rows, row):
    Xref = cap.X_full
    X = Xref[row:row + 1] if Xref.ndim == 3 and Xref.shape[0] > 1 else Xref
    if X.ndim == 2:
        X = X.unsqueeze(0)
    for fk, th in zip(cap.fks, theta_rows):
        X = fk.apply(th, X)
    return X


def _single_joint_clash(cap, mol, best_row, li, joint, target):
    baseline_theta = [t[best_row:best_row + 1].clone() for t in cap.theta_inits]
    theta = [t.clone() for t in baseline_theta]
    theta[li][0, 6 + joint] = target
    X = _coords(cap, theta, best_row)
    scorer = cap.scorer
    ge = cap.ge
    max_q6_overlap = 0.0
    baseline_q6_overlap = 0.0
    if getattr(ge, "n_q6", 0) and getattr(scorer, "_q6_mu", None) is not None and scorer._q6_mu.numel():
        a, b = ge._q6_a, ge._q6_b
        d = (X[:, a] - X[:, b]).norm(dim=-1)[0]
        X_base = _coords(cap, baseline_theta, best_row)
        d_base = (X_base[:, a] - X_base[:, b]).norm(dim=-1)[0]
        mu = scorer._q6_mu[:d.numel()]
        max_q6_overlap = float((mu - d).clamp(min=0).max().detach().cpu())
        baseline_q6_overlap = float((mu - d_base).clamp(min=0).max().detach().cpu())
    # Include the metal contact guard even if this pair is not a Q6 feature.
    pos = {a.idx: i for i, a in enumerate(mol.atoms)}
    donors = {pos[int(d)] for lig in mol.ligands for d in lig.donor_atoms if int(d) in pos}
    sym = [a.symbol for a in mol.atoms]
    metal = pos.get(int(mol.metal_idx), int(mol.metal_idx))
    nd = [i for i, e in enumerate(sym) if i != metal and i not in donors and e != "H"]
    metal_contact = float((X[0, nd] - X[0, metal]).norm(dim=-1).min().detach().cpu()) if nd else float("inf")
    delta_overlap = max_q6_overlap - baseline_q6_overlap
    return {"max_q6_overlap_A": max_q6_overlap, "baseline_q6_overlap_A": baseline_q6_overlap,
            "delta_q6_overlap_A": delta_overlap, "closest_non_donor_metal_A": metal_contact,
            "clashing": bool(delta_overlap > 0.5 or metal_contact < 1.2)}


def parent_coverage(code, shard_data, stats, device):
    shard, dataset = shard_data
    mol = inputs.load_mol(code, shard, dataset)
    builder, _, err = build_builder_and_inputs(mol)
    if builder is None:
        raise RuntimeError(err)
    with xtb.FallbackWatch(raise_on_exit=True):
        with CaptureSession(builder) as capture:
            solved = builder.solve(seed=11, stats=stats, angle_bends=True, device=device,
                                   n_conformers=1, n_ik_steps=1)
    adj = _neighbors(mol)
    parent_modes = []
    capture_rows = []
    supported_count = 0
    for cap in capture.captures:
        initial_scores = _best_initial_row(cap)
        candidate_ids = np.asarray(cap.candidate_idx_per_b)
        for assignment in np.unique(candidate_ids):
            assignment_rows = np.flatnonzero(candidate_ids == assignment)
            best_row = int(assignment_rows[int(torch.argmax(initial_scores[assignment_rows]).item())])
            for li, fk in enumerate(cap.fks):
                rows, offsets = fk.q5_alignment(cap.ge, cap.X_full)
                for j in range(int(fk.n_joints_base)):
                    qrow = int(rows[j].item())
                    if qrow < 0:
                        continue
                    q5_i, q5_j, q5_k, q5_l = (int(t[qrow].item()) for t in
                                               (cap.ge._q5_i, cap.ge._q5_j, cap.ge._q5_k, cap.ge._q5_l))
                    if any(mol.atoms[i].symbol == "H" for i in (q5_i, q5_j, q5_k, q5_l)):
                        continue  # heavy-atom torsions only
                    period = _rotor_period(mol, adj, q5_j, q5_k)
                    means = cap.scorer._q5_means[qrow].detach().cpu().numpy()
                    weights = cap.scorer._q5_weights[qrow].detach().cpu().numpy()
                    active = [c for c, w in enumerate(weights) if w >= 0.05]
                    groups = []
                    for c in active:
                        angle = float(means[c] + offsets[j].item())
                        found = next((g for g in groups if _fold_distance(angle, g["angle"], period) < 0.02), None)
                        if found is None:
                            groups.append({"angle": angle, "weight": float(weights[c]), "components": [c]})
                        else:
                            found["weight"] += float(weights[c]); found["components"].append(c)
                    supported_count += len(groups)
                    beam_angles = cap.theta_inits[li][assignment_rows, 6 + j].detach().cpu().numpy()
                    for mode in groups:
                        for tol in (0.2, 0.35):
                            dmin = min(_fold_distance(float(a), mode["angle"], period) for a in beam_angles)
                            if dmin <= tol:
                                continue
                            clash = _single_joint_clash(cap, mol, best_row, li, j, mode["angle"])
                            parent_modes.append({"variant": cap.variant_index,
                                                 "assignment": int(assignment),
                                                 "ligand": li, "joint": j, "period": period,
                                                 "components": mode["components"], "weight": mode["weight"],
                                                 "target_rad": mode["angle"], "tolerance_rad": tol,
                                                 "min_beam_distance_rad": dmin, **clash})
            capture_rows.append({"variant": cap.variant_index, "assignment": int(assignment),
                                 "rows": len(assignment_rows), "best_initial_row": best_row})
    return {"code": code, "capture_rows": capture_rows,
            "n_supported_modes": supported_count,
            "unfilled_by_tolerance": {
                str(t): {"n_unfilled": sum(x["tolerance_rad"] == t for x in parent_modes),
                         "n_reachable": sum(x["tolerance_rad"] == t and not x["clashing"] for x in parent_modes),
                         "n_clashing": sum(x["tolerance_rad"] == t and x["clashing"] for x in parent_modes)}
                for t in (0.2, 0.35)},
            "modes": parent_modes,
            "has_unfilled_by_tolerance": {str(t): any(x["tolerance_rad"] == t for x in parent_modes)
                                          for t in (0.2, 0.35)}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--panel", choices=("cal", "tail"), default="cal")
    ap.add_argument("--codes", nargs="*", help="optional diagnostic subset")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    manifest = json.load(open(inputs.FRAME_V1 / "manifest.json"))
    tasks = panels.select(manifest, args.panel)["units"]
    unique = sorted({t["code"] for t in tasks})
    shards = inputs.frame_shards(); stats = stats_from_db(inputs.DB_PATH)
    if args.codes:
        unique = list(args.codes)
    result = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "panel": args.panel, "seed": 11, "device": args.device,
              "parents": {}, "failures": {}}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(unique, 1):
        try:
            result["parents"][code] = parent_coverage(code, shards[code], stats, args.device)
        except Exception as exc:
            result["failures"][code] = {"error": f"{type(exc).__name__}: {exc}",
                                        "traceback": traceback.format_exc()}
        out.write_text(json.dumps(result, indent=2))
        if i % 5 == 0 or i == len(unique):
            print(f"{args.panel}: {i}/{len(unique)} parents; failures={len(result['failures'])}", flush=True)
    successful = list(result["parents"].values())
    result["summary"] = {
        "n_parents_with_supported_modes": sum(p["n_supported_modes"] > 0 for p in successful),
        "n_supported_modes": sum(p["n_supported_modes"] for p in successful),
        "by_tolerance": {
            str(t): {
                "parents_with_unfilled": sum(p["has_unfilled_by_tolerance"][str(t)] for p in successful),
                "pct_parents_with_unfilled": 100.0 * sum(p["has_unfilled_by_tolerance"][str(t)] for p in successful) / max(1, sum(p["n_supported_modes"] > 0 for p in successful)),
                "unfilled": sum(p["unfilled_by_tolerance"][str(t)]["n_unfilled"] for p in successful),
                "reachable": sum(p["unfilled_by_tolerance"][str(t)]["n_reachable"] for p in successful),
                "clashing": sum(p["unfilled_by_tolerance"][str(t)]["n_clashing"] for p in successful),
            } for t in (0.2, 0.35)
        }
    }
    result["summary"]["gate_exceeded_10pct"] = any(
        result["summary"]["by_tolerance"][str(t)]["pct_parents_with_unfilled"] >= 10.0
        for t in (0.2, 0.35)
    )
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
