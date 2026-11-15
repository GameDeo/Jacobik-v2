"""Summarize the pre-registered paired H-3 control/combined campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from tools.jgen.bench.panels import load_manifest, select


def load_records(root: Path, panel: str, arm: str) -> dict[tuple[str, int], dict]:
    directory = root / panel / arm / "records"
    out = {}
    for path in sorted(directory.glob("*.json")):
        rec = json.loads(path.read_text())
        out[(str(rec["code"]), int(rec["seed"]))] = rec
    return out


def cluster_bootstrap(values: dict[str, list[float]], n: int = 10_000, seed: int = 11):
    parents = sorted(values)
    if not parents or any(not values[p] for p in parents):
        return None
    flattened = [v for p in parents for v in values[p]]
    rng = np.random.default_rng(seed)
    stats = np.empty(n, dtype=float)
    for b in range(n):
        sampled = rng.integers(0, len(parents), size=len(parents))
        stats[b] = np.mean([v for idx in sampled for v in values[parents[idx]]])
    return {"estimate_mean": float(np.mean(flattened)),
            "ci95_percentile": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))],
            "median_unit_difference": float(np.median(flattened)),
            "n_parents": len(parents), "n_pairs": len(flattened),
            "resamples": n, "seed": seed}


def ok(rec):
    return rec is not None and rec.get("status") == "ok" and rec.get("top1_dE") is not None


def summarize_panel(root: Path, panel: str, excluded: set[str], expected_codes: list[str], candidate_arm: str = "combined"):
    control, combined = load_records(root, panel, "control"), load_records(root, panel, candidate_arm)
    keys = [(code, seed) for code in expected_codes for seed in (11, 12, 13)]
    failures, rows = [], []
    excluded_rows = []
    paired: dict[str, list[float]] = {}
    floor_paired: dict[str, list[float]] = {}
    guard_values: dict[str, list[float]] = {}
    obscene = {"control": 0, "combined": 0, "transitions": {"both": 0, "control_only": 0,
                                                                "combined_only": 0, "neither": 0}}
    coverage = {arm: {"top1_hits": 0, "floor_hits": 0, "full_pool_hits": 0, "units": 0}
                for arm in ("control", "combined")}
    wall_by_parent = {"control": {}, "combined": {}}
    spearman = {"control": [], "combined": []}

    for code, seed in keys:
        c, a = control.get((code, seed)), combined.get((code, seed))
        if code in excluded:
            excluded_rows.append({"code": code, "seed": seed,
                                  "control_status": None if c is None else c.get("status"),
                                  "combined_status": None if a is None else a.get("status"),
                                  "reason": "pre-registered invalid input exclusion"})
            continue
        for arm in ("control", "combined"):
            coverage[arm]["units"] += 1
        if not ok(c) or not ok(a):
            failures.append({"code": code, "seed": seed,
                             "control": None if c is None else {"status": c.get("status"), "reason": c.get("reason")},
                             "combined": None if a is None else {"status": a.get("status"), "reason": a.get("reason")}})
            continue
        de = float(a["top1_dE"]) - float(c["top1_dE"])
        floor_de = float(a["floor_dE"]) - float(c["floor_dE"])
        paired.setdefault(code, []).append(de)
        floor_paired.setdefault(code, []).append(floor_de)
        guard_values.setdefault(code, []).append(de)
        row = {"code": code, "seed": seed,
               "control_top1_dE_kj": float(c["top1_dE"]), "combined_top1_dE_kj": float(a["top1_dE"]),
               "paired_top1_delta_kj": de,
               "control_floor_dE_kj": float(c["floor_dE"]), "combined_floor_dE_kj": float(a["floor_dE"]),
               "paired_floor_delta_kj": floor_de}
        rows.append(row)
        for arm, rec in (("control", c), ("combined", a)):
            if rec.get("wall_solve_s") is not None:
                wall = float(rec["wall_solve_s"])
                prior = wall_by_parent[arm].setdefault(code, wall)
                if not np.isclose(prior, wall, rtol=0.0, atol=0.01):
                    raise ValueError(f"{panel}/{arm}/{code}: wall_solve_s differs across seed records")
            cov = rec.get("coverage") or {}
            coverage[arm]["top1_hits"] += int(bool(cov.get("top1_within_1A")))
            coverage[arm]["floor_hits"] += int(bool(cov.get("floor_within_1A")))
            coverage[arm]["full_pool_hits"] += int(bool(cov.get("full_pool_within_1A")))
            scores, energies = rec.get("scores") or [], rec.get("energies_kj") or []
            pairs = [(float(s), float(e)) for s, e in zip(scores, energies)
                     if s is not None and e is not None and np.isfinite(s) and np.isfinite(e)]
            rho = float(spearmanr([p[0] for p in pairs], [-p[1] for p in pairs]).statistic) if len(pairs) >= 2 else None
            spearman[arm].append({"code": code, "seed": seed, "rho": rho, "n_rows": len(pairs)})
        c_obscene, a_obscene = float(c["top1_dE"]) > 200.0, float(a["top1_dE"]) > 200.0
        obscene["control"] += int(c_obscene)
        obscene["combined"] += int(a_obscene)
        label = "both" if c_obscene and a_obscene else "control_only" if c_obscene else "combined_only" if a_obscene else "neither"
        obscene["transitions"][label] += 1

    guards = {code: vals for code, vals in guard_values.items() if sum(v > 25.0 for v in vals) >= 2}
    all_ok = not failures and bool(rows)
    top_boot = cluster_bootstrap(paired) if all_ok else None
    floor_boot = cluster_bootstrap(floor_paired) if all_ok else None
    n = len(rows)
    walls = {arm: float(sum(values.values())) for arm, values in wall_by_parent.items()}
    return {
        "panel": panel, "status": "ok" if all_ok else "failed_missing_pairs",
        "paired_units": n, "parent_count": len(paired), "failures": failures,
        "pre_registered_exclusions": excluded_rows,
        "primary_top1_delta": top_boot, "floor_delta": floor_boot,
        "primary_gate_pass": bool(all_ok and top_boot["ci95_percentile"][1] < 2.0),
        "guard_parents_gt25_two_of_three": guards,
        "named_parents": {name: {"seed_differences_kj": paired.get(name),
                                  "mean_difference_kj": float(np.mean(paired[name])) if name in paired else None,
                                  "improved": bool(name in paired and len(paired[name]) == 3 and np.mean(paired[name]) < 0)}
                          for name in ("FEKKAJ", "NUWFAM", "XOJHIN")},
        "obscene_gt200": {"counts": obscene, "denominator": n,
                          "control_rate": obscene["control"] / n if n else None,
                          "combined_rate": obscene["combined"] / n if n else None,
                          "gate_pass": bool(n and obscene["combined"] <= obscene["control"])},
        "wall_solve_s": {**walls, "aggregation": "deduplicated per parent; each seed record repeats the same multi-seed solve wall",
                         "ratio_combined_over_control": walls["combined"] / walls["control"] if walls["control"] else None,
                         "gate_pass": bool(walls["control"] and walls["combined"] / walls["control"] <= 1.1)},
        "coverage_1A": {arm: {**v, "top1_rate": v["top1_hits"] / v["units"] if v["units"] else None,
                              "floor_rate": v["floor_hits"] / v["units"] if v["units"] else None,
                              "full_pool_rate": v["full_pool_hits"] / v["units"] if v["units"] else None}
                        for arm, v in coverage.items()},
        "score_vs_negative_energy_spearman": spearman,
        "paired_rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research/data/jgen/campaign/H/h3"))
    parser.add_argument("--out", type=Path, default=Path("research/data/jgen/campaign/H/h3/analysis.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/refit/frame_v1/manifest.json"))
    parser.add_argument("--exclude", default="", help="Pre-registered invalid inputs to exclude, e.g. NEPNEF if H-2 remains invalid")
    args = parser.parse_args()
    excluded = {code.strip() for code in args.exclude.split(",") if code.strip()}
    manifest = load_manifest(args.manifest)
    codes = {}
    for panel in ("cal", "tail"):
        codes[panel] = sorted({row["code"] for row in select(manifest, panel)["units"]})
    result = {"exclusions": sorted(excluded),
              "manifest": str(args.manifest),
              "panels": [summarize_panel(args.root, panel, excluded, codes[panel]) for panel in ("cal", "tail")]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"panels": [{k: v for k, v in p.items() if k != "paired_rows"} for p in result["panels"]]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
