"""Example 1: GPU panel solve on the local GPU (Gemini-style). Illustrative skeleton: solve_all_arms is bench-specific.

    CUDA_VISIBLE_DEVICES unset; PYTHONPATH=src:.:research/bench python research/bench/tools/runkit/examples/panel_solve_local.py
"""
from tools.runkit import mace, pipeline, policy, resources, runprofile as rp, xtb

CODES = ["HOXJUZ", "USAPOU"]                                   # tail panel codes
OUT = resources.RUN_DIR / "examples" / "panel_solve"


def prefetch_parent(code):                                     # CPU work for parent i+1 while the GPU solves parent i
    return {"code": code}                                      # e.g. load_mol + build_builder_and_inputs


def solve_all_arms(builder):                                   # bench-specific; returns structures (symbols, X, charge, spin)
    return []


def main():
    prov = policy.assert_panel("tail", CODES)                   # tail/cal only; reserve and (outside tail) val refused
    xtb.require()                                               # raises if xtb would silently fall back to MMFF
    proj = rp.project(n_items=len(CODES) * 3, per_item_s=40.0, workers=1)   # per_item_s: rp.measure_items(...) on 1-2 parents
    budget = rp.require_budget(proj, max_min=60, record=OUT / "refusals.jsonl")
    if not budget["ok"]:
        raise SystemExit(budget["refusal"])
    tr = rp.Trace()
    with resources.gpu_lock("Gemini", timeout=0), xtb.FallbackWatch() as fw:   # refused unless Gemini owns the GPU and it is free
        ev = mace.Evaluator(device="cuda", batch_atoms=900)
        for code, builder in pipeline.Prefetcher(prefetch_parent, CODES):
            with tr("solve"):
                structures = solve_all_arms(builder)
            with tr("mace"):
                E = ev.energies_kj(structures)                  # duplicates evaluated once, OOM halves the batch
        fw.check()                                              # an MMFF fallback invalidates the run
    OUT.mkdir(parents=True, exist_ok=True)
    tr.write(OUT, extra=dict(manifest=resources.manifest(extra=dict(policy=prov, fallback=fw.hit))))


if __name__ == "__main__":
    main()
