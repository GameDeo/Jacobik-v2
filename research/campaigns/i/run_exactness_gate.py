"""
run_exactness_gate.py
=====================
Execute the Hard Exactness Gate across all 120 cal/tail parents on seeds 11-13.
Directive I (§1.1):
- Identical selections to production code on every cal/tail parent below the cap (107 parents).
- Identical or fewer survivors on the 13 capped parents (M2: XAQCAT, SILROY, NOYTUS, NUHPEL, YOVNEC...).
"""
import os
import sys

# Pin numerical libraries to 1 thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from tools.jgen.a4t import analyse as A
from tools.jgen.bench.pool_export import _read_frames
from jacobik.ik import dedup as dd

A4B_XYZ = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/A4b/xyz")
OUT_JSON = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/I/exactness_gate.json")
OUT_JSONL = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/I/exactness_gate.jsonl")
PANEL_JSON = Path(__file__).resolve().parents[1] / "c2/C2_PANEL.json"

CAPPED_13 = {
    "CIFZEZ", "CIXYUD", "DUHHIX", "EDOBUV", "GOTWET", "HEDJOP",
    "NOYTUS", "NUCKUS", "NUHPEL", "QEZVEY", "SILROY", "XAQCAT", "YOVNEC"
}


def process_parent(code: str) -> dict:
    t0 = time.time()
    rec = {
        "code": code,
        "is_capped": code in CAPPED_13,
        "seeds": {},
        "status": "ok",
        "uncapped_identical": True,
        "capped_le_survivors": True,
    }
    try:
        mol, builder, bd = A._load_parent(code)
        deduper = dd.deduper_for(mol)
        heavy = np.asarray([a.symbol != "H" for a in mol.atoms])
        n_atoms = len(mol.atoms)

        for seed in (11, 12, 13):
            p_path = A4B_XYZ / code / f"{code}_control_s{seed}_pool.json"
            if not p_path.exists():
                rec["seeds"][str(seed)] = {"status": "missing_pool"}
                continue
            man = json.loads(p_path.read_text())
            f_path = A4B_XYZ / code / man["rows"][0]["file"]
            frames = list(_read_frames(Path(f_path)))
            g = np.asarray(man["gen_to_ref_idx"], int)
            scores = torch.tensor([r["score"] for r in man["rows"]])

            X_all = np.empty((len(man["rows"]), n_atoms, 3))
            for r_idx, r in enumerate(man["rows"]):
                X_all[r_idx] = np.asarray(frames[r["row_index"]][2])[g]
            X_t = torch.tensor(X_all)

            # Production baseline
            dd.DEDUP_FACTORED = False
            sel_prod = deduper.select(X_t, scores, k=None, th=0.5)

            # Factored symmetry
            dd.DEDUP_FACTORED = True
            sel_fact = deduper.select(X_t, scores, k=None, th=0.5)
            merges = list(deduper.merges)

            is_cap = deduper.is_capped
            exact_match = (sel_prod == sel_fact)
            le_survivors = (len(sel_fact) <= len(sel_prod)) and set(sel_fact).issubset(set(sel_prod))

            if not is_cap and not exact_match:
                rec["uncapped_identical"] = False
            if is_cap and not le_survivors:
                rec["capped_le_survivors"] = False

            rec["seeds"][str(seed)] = {
                "n_rows": len(man["rows"]),
                "n_prod": len(sel_prod),
                "n_factored": len(sel_fact),
                "n_merged": len(sel_prod) - len(sel_fact),
                "exact_match": exact_match,
                "le_survivors": le_survivors,
                "merges": merges,
            }
    except Exception as exc:
        rec["status"] = f"failed: {exc}"
        rec["uncapped_identical"] = False
        rec["capped_le_survivors"] = False

    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def main():
    import argparse
    global OUT_JSON, OUT_JSONL
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes', default='')
    parser.add_argument('--out', type=Path, default=OUT_JSON)
    args = parser.parse_args()
    OUT_JSON = args.out
    OUT_JSONL = OUT_JSON.with_suffix('.jsonl')
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    panel_data = json.loads(PANEL_JSON.read_text())
    codes = args.codes.split(",") if args.codes else panel_data["codes"]

    # Check for already processed parents in jsonl
    results_map = {}
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    results_map[obj["code"]] = obj
                except Exception:
                    pass

    remaining_codes = [c for c in codes if c not in results_map]
    print(f"Starting Hard Exactness Gate: {len(codes)} total, {len(results_map)} cached, {len(remaining_codes)} to run (2 workers)...")

    t_start = time.time()
    if remaining_codes:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for i, res in enumerate(pool.map(process_parent, remaining_codes), start=1):
                results_map[res["code"]] = res
                code = res["code"]
                is_cap = res["is_capped"]
                cap_str = "CAPPED" if is_cap else "uncapped"
                status_str = "PASS" if (res["uncapped_identical"] and res["capped_le_survivors"]) else "FAIL"
                print(f"[{len(results_map):3d}/{len(codes)}] {code:8s} ({cap_str:8s}): {status_str} ({res['wall_s']:.1f}s)")
                with open(OUT_JSONL, "a") as f:
                    f.write(json.dumps(res) + "\n")

    results = [results_map[c] for c in codes if c in results_map]

    # Aggregates
    n_uncapped = sum(1 for r in results if not r["is_capped"])
    n_capped = sum(1 for r in results if r["is_capped"])
    uncapped_pass = sum(1 for r in results if not r["is_capped"] and r["uncapped_identical"])
    capped_pass = sum(1 for r in results if r["is_capped"] and r["capped_le_survivors"])
    total_dupes_merged = sum(
        s_data.get("n_merged", 0)
        for r in results
        for s_data in r["seeds"].values()
        if isinstance(s_data, dict)
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_parents": len(results),
        "uncapped_parents": n_uncapped,
        "uncapped_exact_pass": uncapped_pass,
        "capped_parents": n_capped,
        "capped_le_pass": capped_pass,
        "total_duplicates_recovered": total_dupes_merged,
        "all_gates_passed": (uncapped_pass == n_uncapped and capped_pass == n_capped),
        "total_wall_s": round(time.time() - t_start, 1),
        "results": results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n=======================================================")
    print(f"HARD EXACTNESS GATE RESULTS:")
    print(f"  Uncapped identical: {uncapped_pass}/{n_uncapped} ({'PASS' if uncapped_pass == n_uncapped else 'FAIL'})")
    print(f"  Capped <= survivors: {capped_pass}/{n_capped} ({'PASS' if capped_pass == n_capped else 'FAIL'})")
    print(f"  Total hidden duplicates recovered: {total_dupes_merged}")
    print(f"  Wrote report to {OUT_JSON}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
