"""
run_stereo_census.py
====================
Compute the stereo census across all 120 cal/tail parents under Scope B.
Directive I (§3.1):
- Record for each parent:
  - unspecified ligand tetrahedral centres
  - binding-created donor centres (N, P, Si, C, S)
  - acyclic E/Z bonds (outside rings <= 8)
  - M-C=X bonds (aminocarbenes, alkenyls)
  - raw combinations
  - mirror-merged classes
  - feasible classes after prefilter
  - baseline assignments and total classes (stereo x assignment)
  - whether uniform sampling triggered under 1.5x cost ceiling
- Save to research/data/jgen/campaign/I/stereo_census_caltail.json
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tools.jgen.a4t import analyse as A
from jacobik.ik import stereo

OUT_JSON = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/I/stereo_census_caltail.json")
PANEL_JSON = Path(__file__).resolve().parents[1] / "c2/C2_PANEL.json"


def census_one(code: str) -> dict:
    t0 = time.time()
    rec = {"code": code, "status": "ok"}
    try:
        mol, builder, bd = A._load_parent(code)
        elements = stereo.perceive_complex_stereo(builder, scope="B")

        # Counts by type
        n_lig_centres = sum(1 for e in elements if e.element_type == stereo.StereoType.LIGAND_TETRAHEDRAL)
        n_bind_centres = sum(1 for e in elements if e.element_type == stereo.StereoType.BINDING_CREATED_DONOR)
        n_ez_bonds = sum(1 for e in elements if e.element_type == stereo.StereoType.DOUBLE_BOND_EZ)
        n_mcx_bonds = sum(1 for e in elements if e.element_type == stereo.StereoType.MCX_ROTAMER)
        n_total_elements = len(elements)

        # Baseline assignment count
        # Hungarian assignment baseline = 1
        baseline_classes = 1

        achiral_env = stereo.is_achiral_geometry(builder.geometry)
        all_variants, _ = stereo.enumerate_stereo_variants(
            elements,
            achiral_environment=achiral_env,
            cost_ceiling_multiplier=1e9,
            baseline_classes=baseline_classes,
            seed=42,
        )
        variants, sampled = stereo.enumerate_stereo_variants(
            elements,
            achiral_environment=achiral_env,
            cost_ceiling_multiplier=1.5,
            baseline_classes=baseline_classes,
            seed=42,
            builder=builder,
        )

        # Feasibility check on variants
        n_feasible = sum(1 for v in all_variants if stereo.check_variant_feasibility(v, builder, elements))

        # Raw combinations without mirror merging
        raw_combinations = (2 ** (n_lig_centres + n_bind_centres + n_ez_bonds + n_mcx_bonds)) if n_total_elements > 0 else 1

        rec.update({
            "geometry": builder.geometry,
            "achiral_geometry": achiral_env,
            "n_elements": n_total_elements,
            "n_lig_centres": n_lig_centres,
            "n_bind_centres": n_bind_centres,
            "n_ez_bonds": n_ez_bonds,
            "n_mcx_bonds": n_mcx_bonds,
            "element_names": [e.name for e in elements],
            "raw_combinations": raw_combinations,
            "mirror_merged_classes": len(all_variants),
            "feasible_classes": n_feasible,
            "selected_classes": len(variants),
            "sampled_above_ceiling": sampled,
            "variants": [v.class_id for v in variants],
        })
    except Exception as exc:
        rec.update({"status": f"failed: {exc}"})
    rec["wall_s"] = round(time.time() - t0, 3)
    return rec


def main():
    panel_data = json.loads(PANEL_JSON.read_text())
    codes = panel_data["codes"]
    print(f"Running stereo census across {len(codes)} parents...")

    results = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i, res in enumerate(pool.map(census_one, codes), start=1):
            results.append(res)

    # Aggregates
    stereo_active = [r for r in results if r.get("n_elements", 0) > 0]
    binding_active = [r for r in results if r.get("n_bind_centres", 0) > 0]
    ez_active = [r for r in results if (r.get("n_ez_bonds", 0) > 0 or r.get("n_mcx_bonds", 0) > 0)]
    sampled_active = [r for r in results if r.get("sampled_above_ceiling", False)]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_parents": len(results),
        "parents_with_stereo_elements": len(stereo_active),
        "parents_with_binding_created_centres": len(binding_active),
        "parents_with_ez_or_mcx_bonds": len(ez_active),
        "parents_sampled_above_ceiling": len(sampled_active),
        "total_wall_s": round(time.time() - t_start, 1),
        "results": results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print("\n=======================================================")
    print("STEREO CENSUS SUMMARY:")
    print(f"  Total parents: {len(results)}")
    print(f"  With stereo elements: {len(stereo_active)} ({len(stereo_active)/len(results)*100:.1f}%)")
    print(f"  With binding-created centres: {len(binding_active)}")
    print(f"  With E/Z or M-C=X bonds: {len(ez_active)}")
    print(f"  Sampled above ceiling: {len(sampled_active)}")
    print(f"  Wrote report to {OUT_JSON}")
    print("=======================================================")


if __name__ == "__main__":
    main()
