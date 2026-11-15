"""Count production embed_bond_check events over an existing census (CPU only)."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "bench")]
TAIL_CODES = {"DAMVAQ", "HOXJUZ", "IFUROQ", "JEGPUJ", "NEPNEF", "SOBHAS", "USAPOU", "XEYTUT", "YUCZUS", "ZELTAP"}


def _scan_parent(parent):
    from jacobik.core import profile
    from jacobik.builder.engine import _smiles_to_conformer, pop_embed_bond_check_events

    if not profile.get("embed_bond_check"):
        raise RuntimeError("run with JACOBIK_PROFILE_OVERRIDES={\"embed_bond_check\":true}")
    rows, errors = [], []
    for ligand in parent.get("ligands", []):
        if ligand.get("status") != "ok":
            continue
        try:
            _rd, _xyz, _has_h = _smiles_to_conformer(
                ligand["smiles"], embed="gfn2", embed_seed=42,
                donors=ligand.get("donor_groups", []), chelate_embed=True,
                ez=ligand.get("ez_fix") or None,
            )
        except Exception as exc:
            errors.append({"ligand_idx": ligand["ligand_idx"], "variant_idx": ligand.get("variant_idx", 0),
                           "error": f"{type(exc).__name__}: {exc}"})
        events = pop_embed_bond_check_events()
        for event in events:
            rows.append({"code": parent["code"], "cohort": parent.get("cohort"), "split": parent.get("split"),
                         "ligand_idx": ligand["ligand_idx"], "variant_idx": ligand.get("variant_idx", 0), **event})
    return {"code": parent["code"], "cohort": parent.get("cohort"), "split": parent.get("split"),
            "events": rows, "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if not 1 <= args.workers <= 3:
        ap.error("F-3 caps CPU workers at 3")
    parents = [json.loads(line) for line in args.census.read_text().splitlines() if line.strip()]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_scan_parent, p) for p in parents]
        for f in as_completed(futures):
            results.append(f.result())
    results.sort(key=lambda r: r["code"])
    all_events = [e for r in results for e in r["events"]]
    trigger_events = [e for e in all_events if e["triggered"]]
    parent_sets = {}
    for key, selected in (
        ("calibration_panel", [r for r in results if r["cohort"] == "cal_tail"]),
        ("tail_subset", [r for r in results if r["cohort"] == "cal_tail" and r["code"] in TAIL_CODES]),
        ("surface_fit_sample", [r for r in results if r["cohort"] == "surface_fit_sample"]),
    ):
        events = [e for r in selected for e in r["events"] if e["triggered"]]
        parent_sets[key] = {"n_parents": len(selected), "trigger_parent_count": len({e["code"] for e in events}),
                            "trigger_parents": sorted({e["code"] for e in events}),
                            "trigger_ligand_variants": len(events)}
    unique_triggers = sorted({e["code"] for e in trigger_events})
    parent_sets["overall_unique"] = {"n_parents": len(results), "trigger_parent_count": len(unique_triggers),
                                     "trigger_parents": unique_triggers,
                                     "trigger_ligand_variants": len(trigger_events)}
    out = {"fingerprint": "embed_bond_check=true", "parents": parent_sets,
           "n_trigger_events": len(trigger_events), "trigger_events": trigger_events,
           "errors": [{"code": r["code"], **e} for r in results for e in r["errors"]], "results": results,
           "workers": args.workers}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"parents": parent_sets, "n_trigger_events": len(trigger_events),
                      "n_errors": sum(len(r["errors"]) for r in results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
