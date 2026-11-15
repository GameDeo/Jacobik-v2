"""Analyze the committed E-2 paired cal runs and mapped XYZ outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


def read_records(root: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for p in sorted(root.glob("**/records/*.json")):
        rec = json.loads(p.read_text())
        out[(rec["code"], int(rec["seed"]))] = rec
    return out


def bootstrap(values, stat, n=10_000, seed=11):
    values = np.asarray(values, float)
    if not len(values):
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(values), size=(n, len(values)))
    sims = stat(values[ix], axis=1)
    return [float(stat(values)), float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))]


def read_xyz(p: Path):
    lines = p.read_text().splitlines()
    n = int(lines[0])
    symbols, coords = [], []
    for line in lines[2:2+n]:
        bits = line.split()
        symbols.append(bits[0])
        coords.append([float(x) for x in bits[1:4]])
    if len(coords) != n:
        raise ValueError(f"bad XYZ row count: {p}")
    return symbols, np.asarray(coords, float)


def aggregate_records(control, cand, keys):
    deltas, floor_deltas, paired = {}, {}, []
    for code, seed in keys:
        b, c = control.get((code, seed)), cand.get((code, seed))
        if not b or not c or b.get("status") != "ok" or c.get("status") != "ok":
            continue
        if b.get("top1_dE") is None or c.get("top1_dE") is None:
            continue
        paired.append((code, seed))
        deltas[(code, seed)] = float(c["top1_dE"]) - float(b["top1_dE"])
        floor_deltas[(code, seed)] = float(c["floor_dE"]) - float(b["floor_dE"])

    by_parent, by_parent_floor = {}, {}
    for (code, seed), d in deltas.items():
        by_parent.setdefault(code, []).append(d)
        by_parent_floor.setdefault(code, []).append(floor_deltas[(code, seed)])
    p_top = {k: float(np.mean(v)) for k, v in by_parent.items()}
    p_floor = {k: float(np.mean(v)) for k, v in by_parent_floor.items()}

    def rates(records):
        rows = [records[k] for k in paired]
        return np.asarray([float(float(r["top1_dE"]) > 200.0) for r in rows])

    obscene_b = rates(control)
    obscene_c = rates(cand)
    by_parent_obscene = {}
    for ix, key in enumerate(paired):
        by_parent_obscene.setdefault(key[0], []).append(obscene_c[ix] - obscene_b[ix])
    p_obscene_delta = [float(np.mean(v)) for v in by_parent_obscene.values()]

    spearman_b, spearman_c = {}, {}
    for code, seed in paired:
        for records, dest in ((control, spearman_b), (cand, spearman_c)):
            r = records[(code, seed)]
            s, e = np.asarray(r.get("scores", []), float), np.asarray(r.get("energies_kj", []), float)
            ok = np.isfinite(s) & np.isfinite(e)
            if ok.sum() >= 3 and np.ptp(s[ok]) > 0 and np.ptp(e[ok]) > 0:
                dest.setdefault(code, []).append(float(spearmanr(s[ok], -e[ok]).statistic))
    sp_delta = {p: float(np.mean(spearman_c[p]) - np.mean(spearman_b[p]))
                for p in sorted(spearman_b.keys() & spearman_c.keys())}

    guards = {}
    for code in sorted({x[0] for x in paired}):
        ds = [deltas[(code, s)] for s in (11, 12, 13) if (code, s) in deltas]
        if sum(d > 25.0 for d in ds) >= 2:
            guards[code] = {str(s): round(deltas[(code, s)], 3) for s in (11, 12, 13) if (code, s) in deltas}

    fek = {str(s): deltas.get(("FEKKAJ", s)) for s in (11, 12, 13)}
    wall_b = sum(float(control[k].get("wall_solve_s") or 0) for k in paired)
    wall_c = sum(float(cand[k].get("wall_solve_s") or 0) for k in paired)
    out = {
        "n_panel_units": len(keys), "n_paired_units": len(paired), "n_paired_parents": len(p_top),
        "primary_top1_delta_parent_mean_kj": bootstrap(list(p_top.values()), np.mean),
        "primary_top1_delta_parent_median_kj": bootstrap(list(p_top.values()), np.median),
        "floor_delta_parent_mean_kj": bootstrap(list(p_floor.values()), np.mean),
        "floor_delta_parent_median_kj": bootstrap(list(p_floor.values()), np.median),
        "top1_delta_unit_mean_kj": float(np.mean(list(deltas.values()))) if deltas else None,
        "top1_delta_unit_median_kj": float(np.median(list(deltas.values()))) if deltas else None,
        "parents_improved_top1": sum(v < 0 for v in p_top.values()),
        "obscene_rate_control": float(np.mean(obscene_b)) if len(obscene_b) else None,
        "obscene_rate_candidate": float(np.mean(obscene_c)) if len(obscene_c) else None,
        "obscene_rate_delta_parent_mean": float(np.mean(p_obscene_delta)) if p_obscene_delta else None,
        "obscene_rate_delta_parent_ci95": bootstrap(p_obscene_delta, np.mean),
        "score_energy_spearman_parent_delta_mean": float(np.mean(list(sp_delta.values()))) if sp_delta else None,
        "score_energy_spearman_parent_delta_median": float(np.median(list(sp_delta.values()))) if sp_delta else None,
        "score_energy_spearman_parent_delta_ci95": bootstrap(list(sp_delta.values()), np.mean),
        "score_energy_spearman_parent_deltas": sp_delta,
        "guard_parents_worsened_gt25_on_2plus_seeds": guards,
        "FEKKAJ_top1_deltas_by_seed": fek,
        "FEKKAJ_improved_all_3": all(v is not None and v < 0 for v in fek.values()),
        "solve_wall_s_control": wall_b, "solve_wall_s_candidate": wall_c,
        "solve_wall_ratio_candidate_control": wall_c / wall_b if wall_b else None,
        "unit_top1_deltas_kj": {f"{c}:{s}": d for (c, s), d in deltas.items()},
        "parent_top1_deltas_kj": p_top,
    }
    return out


def coverage(root: Path, codes: list[str]):
    from tools.jgen import inputs
    from tools.runkit.geom import automorphisms, find_metal, sym_rmsd_to_ref

    shards = inputs.frame_shards()
    per_arm = {}
    ref_cache = {}
    for arm in ("control", "donor_pair"):
        # roots are supplied as each arm's main directory; discover both A/B result trees.
        suffix, arm_name = ("control", "control") if arm == "control" else ("candidate", "donor_pair")
        files = sorted(root.glob(f"e2_cal_*_{suffix}/e2_cal_*_{suffix}/xyz/*/*_{arm_name}_s*_top1.xyz"))
        hits, floors, rmsd_rows = {}, {}, {}
        for p in files:
            name = p.name
            code = name.split(f"_{arm_name}_s")[0]
            seed = int(name.split("_s")[-1].split("_")[0])
            if code not in codes:
                continue
            if code not in ref_cache:
                shard, ds = shards[code]
                mol = inputs.load_mol(code, shard, ds)
                ref_sym = [x.symbol for x in mol.atoms]
                ref = np.asarray([x.position for x in mol.atoms], float)
                heavy = np.asarray([i for i, s in enumerate(ref_sym) if s != "H"], int)
                metal = find_metal(ref_sym)
                P, _ = automorphisms(ref, ref_sym, metal)
                ref_cache[code] = (ref_sym, ref, heavy, P)
            ref_sym, ref, heavy, P = ref_cache[code]
            symbols, X = read_xyz(p)
            if symbols != ref_sym:
                raise ValueError(f"XYZ symbol order mismatch: {p}")
            r = float(sym_rmsd_to_ref(X[heavy][None], ref[heavy], P)[0])
            rmsd_rows[(code, seed)] = r
            hits[(code, seed)] = r <= 1.0

        floor_files = sorted(root.glob(f"e2_cal_*_{suffix}/e2_cal_*_{suffix}/xyz/*/*_{arm_name}_s*_floor.xyz"))
        floor_r = {}
        for p in floor_files:
            name = p.name
            code = name.split(f"_{arm_name}_s")[0]
            seed = int(name.split("_s")[-1].split("_")[0])
            if code not in codes:
                continue
            if code not in ref_cache:
                shard, ds = shards[code]
                mol = inputs.load_mol(code, shard, ds)
                ref_sym = [x.symbol for x in mol.atoms]
                ref = np.asarray([x.position for x in mol.atoms], float)
                heavy = np.asarray([i for i, s in enumerate(ref_sym) if s != "H"], int)
                metal = find_metal(ref_sym)
                P, _ = automorphisms(ref, ref_sym, metal)
                ref_cache[code] = (ref_sym, ref, heavy, P)
            ref_sym, ref, heavy, P = ref_cache[code]
            symbols, X = read_xyz(p)
            if symbols != ref_sym:
                raise ValueError(f"XYZ symbol order mismatch: {p}")
            r = float(sym_rmsd_to_ref(X[heavy][None], ref[heavy], P)[0])
            floor_r[(code, seed)] = r
            floors[(code, seed)] = r <= 1.0

        parent_any_top = {c: any(hits.get((c, s), False) for s in (11, 12, 13)) for c in codes}
        parent_any_saved = {c: any(hits.get((c, s), False) or floors.get((c, s), False) for s in (11, 12, 13)) for c in codes}
        per_arm[arm] = {
            "top1_units_within_1A": int(sum(hits.values())), "top1_units": len(hits),
            "floor_units_within_1A": int(sum(floors.values())), "floor_units": len(floors),
            "parents_any_top1_within_1A": int(sum(parent_any_top.values())),
            "parents_any_saved_top1_or_floor_within_1A": int(sum(parent_any_saved.values())),
            "top1_rmsd_A": {f"{c}:{s}": r for (c, s), r in rmsd_rows.items()},
            "floor_rmsd_A": {f"{c}:{s}": r for (c, s), r in floor_r.items()},
        }
    return per_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="research/data/jgen/campaign/E")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    roots = {
        "control": [a.root / f"e2_cal_{s}_control/e2_cal_{s}_control" for s in ("a", "b")],
        "donor_pair": [a.root / f"e2_cal_{s}_candidate/e2_cal_{s}_candidate" for s in ("a", "b")],
    }
    control = {}
    cand = {}
    for r in roots["control"]:
        control.update(read_records(r))
    for r in roots["donor_pair"]:
        cand.update(read_records(r))
    codes = sorted({k[0] for k in control})
    keys = sorted(set(control) | set(cand))
    result = aggregate_records(control, cand, keys)
    result["coverage_saved_xyz"] = coverage(a.root, codes)
    result["run_metadata"] = {
        "panel_codes": len(codes), "seeds": [11, 12, 13], "excluded": ["NEPNEF"],
        "control_fingerprints": sorted({r.get("fingerprint") for r in control.values()}),
        "candidate_fingerprints": sorted({r.get("fingerprint") for r in cand.values()}),
        "records_control": len(control), "records_candidate": len(cand),
        "note": "Coverage is exact A4 symmetry-aware heavy-atom RMSD to the input crystal on exported top1/floor XYZs; the runner does not retain non-selected pool coordinates.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps({k: v for k, v in result.items() if k not in ("unit_top1_deltas_kj", "parent_top1_deltas_kj", "score_energy_spearman_parent_deltas", "coverage_saved_xyz")}, indent=2, default=float))


if __name__ == "__main__":
    main()
