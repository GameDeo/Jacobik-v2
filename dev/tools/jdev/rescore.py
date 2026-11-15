"""Rescore stage: the 60-pool sanity check (20 parents x 3 seeds) of a DB against a target DB.

A bench driver (it uses ``tools.jdev.multiseed`` and ``tools.jdev.budget.pools``): moved out of ``src`` so the package
has no dependency on the tools."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.stats


def _pool_metrics(code, seed, score_target, score_test):
    """Compare scores on the same rows; delta is test best minus target best."""
    target_top = int(np.argmax(score_target))
    test_top = int(np.argmax(score_test))
    if len(score_target) > 1 and np.std(score_target) > 1e-12 and np.std(score_test) > 1e-12:
        rho = float(scipy.stats.spearmanr(score_target, score_test).statistic)
    else:
        rho = 1.0
    return {"code": code, "seed": seed, "n_rows": len(score_target),
            "spearman_rho": rho, "top1_unchanged": target_top == test_top,
            "top1_delta_score": float(score_test[test_top] - score_target[target_top]),
            "max_abs_delta_score": float(np.max(np.abs(score_target - score_test)))}


def _rescore_60_pools(
    db_test: Dict[str, Any],
    db_target: Dict[str, Any],
    *, pools_dir: Path, frame_path: Path, manifest_path: Path, ref_dir: Path,
) -> Dict[str, Any]:
    """Score the 60-pool sample (20 parents x 3 seeds) with both DBs."""
    import torch
    from jacobik.stats import stats
    from jacobik.core.env import build_builder_and_inputs
    from tools.jgen.inputs import load_mol
    from tools.jdev import multiseed
    from tools.jdev.budget.pools import rank_terms

    # 1. Identify first 20 valid parents
    with open(manifest_path, "r", encoding="utf-8") as f:
        mf = json.load(f)
    with open(frame_path, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    surface_fit_codes = set()
    for row in frame_data.get("main_allocation", {}).get("splits", {}).get("surface_fit", []):
        c = row.get("code") or row.get("representative_code")
        if c:
            surface_fit_codes.add(c)

    tasks = mf.get("tasks", {})
    task_list = tasks.values() if isinstance(tasks, dict) else tasks
    success_seeds = {}
    for task in task_list:
        if task.get("status") == "success" and task.get("split") == "surface_fit":
            code = task.get("code")
            seed = task.get("seed")
            if code and seed:
                success_seeds.setdefault(code, set()).add(int(seed))

    valid_codes = [
        code for code, seeds in success_seeds.items()
        if {11, 12, 13}.issubset(seeds) and code in surface_fit_codes
    ]
    valid_codes.sort()
    sample_codes = valid_codes[:20]

    code_meta = {}
    for p in frame_data.get("main_allocation", {}).get("splits", {}).get("surface_fit", []):
        c = p.get("code") or p.get("representative_code")
        shard = p.get("shard") or p.get("representative_shard") or "unknown"
        dataset = p.get("dataset") or "unknown"
        if c:
            code_meta[c] = (shard, dataset)

    # 2. Build lookups
    db_tgt_obj = stats.load_db(db_target)
    lookup_tgt = stats.StatsLookup(db_tgt_obj)

    db_test_obj = stats.load_db(db_test)
    lookup_test = stats.StatsLookup(db_test_obj)

    total_pools = 0
    top1_unchanged = 0
    pool_max_diffs = []
    spearman_rhos = []
    per_pool = []

    for code in sample_codes:
        if code not in code_meta:
            continue
        shard, dataset = code_meta[code]
        mol = load_mol(code, shard, dataset, clean_root=ref_dir)
        if mol is None:
            continue
        builder, _, err = build_builder_and_inputs(mol)
        if err or builder is None:
            continue

        ctx_tgt, asg_tgt, W_tgt, _ = multiseed.prepare(builder, 11, stats=lookup_tgt, device="cpu", angle_bends=True)
        (_, _, _, sc_tgt), _ = multiseed.seed_rows(builder, ctx_tgt, asg_tgt, W_tgt, [11])

        ctx_t, asg_t, W_t, _ = multiseed.prepare(builder, 11, stats=lookup_test, device="cpu", angle_bends=True)
        (_, _, _, sc_t), _ = multiseed.seed_rows(builder, ctx_t, asg_t, W_t, [11])

        for seed in [11, 12, 13]:
            pool_file = pools_dir / f"{code}_{seed}_control.npz"
            if not pool_file.exists():
                continue

            pool_data = np.load(pool_file, allow_pickle=True)
            X = pool_data["X"]
            asg = pool_data["asg"]

            Xt = torch.tensor(X, dtype=torch.float32, device="cpu")
            rg = torch.as_tensor(asg, dtype=torch.long, device="cpu")

            tot_tgt, _ = rank_terms(sc_tgt, Xt, rg)
            score_tgt = tot_tgt.detach().cpu().numpy()

            tot_t, _ = rank_terms(sc_t, Xt, rg)
            score_t = tot_t.detach().cpu().numpy()

            row = _pool_metrics(code, seed, score_tgt, score_t)
            pool_max_diffs.append(row["max_abs_delta_score"])
            top1_unchanged += row["top1_unchanged"]
            spearman_rhos.append(row["spearman_rho"])
            per_pool.append(row)
            total_pools += 1

    return {
        "per_pool": sorted(per_pool, key=lambda row: (row["spearman_rho"], row["code"], row["seed"])),
        "total_pools": total_pools,
        "max_abs_delta_score": float(max(pool_max_diffs)) if pool_max_diffs else 0.0,
        "p95_abs_delta_score": float(np.percentile(pool_max_diffs, 95)) if pool_max_diffs else 0.0,
        "fraction_top1_unchanged": float(top1_unchanged / max(total_pools, 1)),
        "spearman_rho_min": float(min(spearman_rhos)) if spearman_rhos else 1.0,
        "spearman_rho_median": float(np.median(spearman_rhos)) if spearman_rhos else 1.0,
    }




class _RejectEmbedFallback(logging.Handler):
    def emit(self, record):
        if "GFN2 embed refinement failed" in record.getMessage():
            raise RuntimeError(record.getMessage())


def rescore_60_pools(
    db_test: Dict[str, Any], db_target: Dict[str, Any], *, pools_dir: Path,
    frame_path: Path, manifest_path: Path, ref_dir: Path,
) -> Dict[str, Any]:
    """Run the rescore, rejecting any invalid MMFF fallback shell."""
    logger = logging.getLogger("jacobik.builder.engine")
    handler = _RejectEmbedFallback()
    logger.addHandler(handler)
    try:
        return _rescore_60_pools(db_test, db_target, pools_dir=pools_dir, frame_path=frame_path,
                                 manifest_path=manifest_path, ref_dir=ref_dir)
    finally:
        logger.removeHandler(handler)
