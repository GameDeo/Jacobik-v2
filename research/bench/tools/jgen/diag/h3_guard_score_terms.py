"""Recompute Q1-Q6 for the frozen H-3 guard-crossing top-1 rows on CPU.

Reads only the saved H-3 records and exported builder-order XYZ. This is a
score-term attribution diagnostic; it does not run a solver or alter either arm.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / "research/data/jgen/campaign/H/h3"
CASES = {"LOPJEF": "cal", "NUWFAM": "cal", "IFUROQ": "tail"}


def _profile(arm: str) -> None:
    overrides = {"nci": False, "deterministic_embed": True}
    if arm == "combined":
        overrides.update({
            "q6_donor_pairs": {"hh": 0.95, "ho": 0.90, "oo": 0.80},
            "angle_tail_quadratic": True,
            "embed_haptic_li_cap": True,
        })
    os.environ["JACOBIK_PROFILE"] = "u1"
    os.environ["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(overrides)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "2")


def _xyz(path: Path):
    lines = path.read_text().splitlines()
    header = json.loads(lines[1])
    import numpy as np
    positions = np.asarray([[float(v) for v in line.split()[1:4]] for line in lines[2:]], dtype=np.float32)
    return header, positions


def _q3_override(scorer, ge, ctx, fks, theta, x_ref, trial: int | None):
    """Mirror JointIKSolver's initial-pose Q3 mode commitment for one row."""
    import torch
    from jacobik.ik.kabsch import _ref_rows

    symmetric = ctx.symmetric_donor_groups
    q3_groups = [g for g in symmetric if len({ctx.all_donor_groups[d][0] for d in g}) == 1]
    if not q3_groups:
        return None
    site_to_idx = {}
    for site, eligible in enumerate(ge._q3_eligible):
        if eligible:
            site_to_idx[site] = len(site_to_idx)
    means = getattr(scorer, "_q3_means", None)
    variances = getattr(scorer, "_q3_eff_vars", None)
    log_weights = getattr(scorer, "_q3_log_w", None)
    weights = getattr(scorer, "_q3_weights", None)
    if means is None or means.shape[0] == 0 or variances is None or log_weights is None or not site_to_idx:
        return None
    batch = theta[0].shape[0]
    x_init = _ref_rows(x_ref, batch)
    for fk, th in zip(fks, theta):
        x_init = fk.apply(th, x_init)
    cos_init = torch.cos(ge.q3_approach_angles(x_init))
    override = torch.full((batch, means.shape[0]), -1, dtype=torch.long, device=means.device)
    any_group = False
    eps = getattr(__import__("jacobik.ik.scorer", fromlist=["_EPS"]), "_EPS", 1e-8)
    for group in q3_groups:
        idx = [site_to_idx[site] for site in group if site in site_to_idx]
        if len(idx) < 2:
            continue
        any_group = True
        ix = torch.as_tensor(idx, dtype=torch.long, device=means.device)
        obs = cos_init[:, ix]
        mu, var, lw = means[ix], variances[ix], log_weights[ix]
        log_comp = -0.5 * torch.log(var).unsqueeze(0) - 0.5 * (obs.unsqueeze(-1) - mu.unsqueeze(0)) ** 2 / var.unsqueeze(0)
        log_p = log_comp + lw.unsqueeze(0)
        if weights is not None:
            log_p = log_p.masked_fill(~(weights[ix] > eps).unsqueeze(0), float("-inf"))
        mode = log_p.sum(dim=1).argmax(dim=-1)
        override[:, ix] = mode.unsqueeze(-1).expand(-1, len(idx))
    return (override if trial is None else override[trial:trial + 1]) if any_group else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("control", "combined"))
    ap.add_argument("--out", type=Path, default=DATA / "guard_score_terms.json")
    args = ap.parse_args()
    _profile(args.arm)

    import numpy as np
    import torch
    from tools.jgen.diag import ctxio
    from tools.jdev import multiseed
    from jacobik.core.env import build_builder_and_inputs
    from jacobik.builder import engine as E
    from jacobik.stats.stats import StatsLookup, load_db

    rows_out = []
    stats = StatsLookup(load_db(str(ctxio.CONSTRAINTS)))
    for code, panel in CASES.items():
        mol = ctxio.load_parent(code)
        builder, _, err = build_builder_and_inputs(mol)
        if builder is None:
            raise RuntimeError(f"{code}: builder preparation failed: {err}")
        ctx, assignments, beam_width, _ = multiseed.prepare(
            builder, 11, device="cpu", angle_bends=True, stats=stats
        )
        (fks, _, ge, scorer), seed_rows = multiseed.seed_rows(
            builder, ctx, assignments, beam_width, (11, 12, 13)
        )
        if E.CLASS_BUDGET == "split":
            keep = builder._class_budget_rows(ctx, assignments, beam_width)
            if keep is not None:
                keep_t = torch.as_tensor(keep, dtype=torch.long)
                seed_rows = [
                    ([part[keep_t] for part in theta],
                     x_ref[keep_t] if x_ref.dim() == 3 else x_ref,
                     ideal[keep_t], candidate_idx[keep])
                    for theta, x_ref, ideal, candidate_idx in seed_rows
                ]
        for seed_idx, seed in enumerate((11, 12, 13)):
            record_path = DATA / panel / args.arm / "records" / f"{code}_{seed}.json"
            record = json.loads(record_path.read_text())
            if record.get("status") != "ok":
                raise RuntimeError(f"{code}/{seed}/{args.arm}: record is not successful")
            xyz_path = DATA / panel / args.arm / "xyz" / code / f"{code}_{args.arm}_s{seed}_top1.xyz"
            header, xyz = _xyz(xyz_path)
            embed_idx, assignment_idx, trial_idx = header["row_key"]
            theta, _, _, candidate_idx = seed_rows[seed_idx]
            if trial_idx >= theta[0].shape[0] or int(candidate_idx[trial_idx]) != assignment_idx:
                raise RuntimeError(f"{code}/{seed}/{args.arm}: row key does not match reconstructed seed rows")
            q3_override = _q3_override(scorer, ge, ctx, fks, theta, seed_rows[seed_idx][1], trial_idx)
            score, breakdown = scorer.score(
                torch.as_tensor(xyz, dtype=theta[0].dtype).unsqueeze(0),
                q3_mode_override=q3_override,
                row_groups=torch.as_tensor([assignment_idx], dtype=torch.long),
            )
            components = {f"Q{i}": float(breakdown[f"Q{i}"][0].detach().cpu()) for i in range(1, 7)}
            rows_out.append({
                "panel": panel, "code": code, "seed": seed, "arm": args.arm,
                "row_key": header["row_key"], "top1_score_recorded": record["top1_score"],
                "bend_prior_recorded": record.get("top1_bend_prior"),
                "recomputed_stat_score": float(score[0].detach().cpu()),
                "recomputed_total_plus_bend": float(score[0].detach().cpu()) + float(record.get("top1_bend_prior") or 0.0),
                "recomputed_minus_recorded_score": float(score[0].detach().cpu()) + float(record.get("top1_bend_prior") or 0.0) - float(record["top1_score"]),
                "q1_q6": components,
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arm": args.arm, "cases": CASES, "rows": rows_out}, indent=2) + "\n")
    print(json.dumps({"arm": args.arm, "rows": rows_out}, indent=2))


if __name__ == "__main__":
    main()
