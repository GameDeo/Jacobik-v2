"""Compare F-4 unrelaxed solve screens with parent-clustered bootstrap CIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, dict]:
    obj = json.loads(path.read_text())
    return {str(row["code"]): row for row in obj["codes"]}


def seed_record(row: dict, seed: int) -> dict | None:
    rec = row.get("seeds", {}).get(str(seed))
    return rec if rec and rec.get("status") == "ok" else None


def interval(values: np.ndarray, rng: np.random.Generator, n_boot: int, statistic: str) -> list[float] | None:
    if not len(values):
        return None
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[draws]
    stats = samples.mean(axis=1) if statistic == "mean" else np.median(samples, axis=1)
    return [float(x) for x in np.quantile(stats, [0.025, 0.975])]


def summarize(control: dict, arm: dict, seed: int, n_boot: int, boot_seed: int) -> dict:
    paired = []
    failed = []
    for code, row in arm.items():
        c = seed_record(control.get(code, {}), seed)
        a = seed_record(row, seed)
        if c is None or a is None:
            failed.append(code)
            continue
        paired.append((code, c, a))

    rng = np.random.default_rng(boot_seed)
    outcomes = {}
    for metric in ("top1_dE_unrel", "floor_dE_unrel", "obscene_rate"):
        deltas = np.asarray([a[metric] - c[metric] for _, c, a in paired], float)
        outcomes[metric] = {
            "n": int(len(deltas)),
            "mean_delta": float(np.mean(deltas)) if len(deltas) else None,
            "median_delta": float(np.median(deltas)) if len(deltas) else None,
            "parent_bootstrap_95_ci_mean": interval(deltas, rng, n_boot, "mean"),
            "parent_bootstrap_95_ci_median": interval(deltas, rng, n_boot, "median"),
        }

    worse = []
    for code, c, a in paired:
        delta = a["top1_dE_unrel"] - c["top1_dE_unrel"]
        if delta <= 25:
            continue
        row = arm[code]
        events = row.get("embed_events", [])
        caused = [e for e in events if e.get("triggered") or e.get("fallback_used") not in (None, "none")]
        donors = sorted({int(d["ligand_idx"]) for d in row.get("anionic_kappa_donors", [])})
        worse.append({
            "code": code, "top1_delta_kj_mol": float(delta),
            "control_top1": c["top1_dE_unrel"], "arm_top1": a["top1_dE_unrel"],
            "guard_events": caused, "li_cap_ligand_indices": donors,
        })

    arm_rows = [seed_record(row, seed) for row in arm.values()]
    arm_rows = [r for r in arm_rows if r is not None]
    return {
        "n_arm_parents": len(arm), "n_paired_ok": len(paired), "failed_or_unpaired": failed,
        "paired_metrics": outcomes,
        "arm_absolute": {
            key: {"mean": float(np.mean([r[key] for r in arm_rows])),
                  "median": float(np.median([r[key] for r in arm_rows]))}
            for key in ("top1_dE_unrel", "floor_dE_unrel", "obscene_rate", "worst_slot_dE_unrel", "top1_rmsd_ref_A")
        } if arm_rows else {},
        "guard_cases_gt_25_kj_mol": worse,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--bootstraps", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=20261028)
    a = ap.parse_args()
    result = summarize(load(a.control), load(a.arm), a.seed, a.bootstraps, a.boot_seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
