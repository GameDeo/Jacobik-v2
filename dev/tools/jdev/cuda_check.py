"""CUDA self-test for this machine: run it once before a GPU benchmark.

    python -m tools.jdev.cuda_check --db data/refit/final/constraints_2026-10-17.json --mace <MACE-POLAR-1 .model>

It checks, in order, and stops at nothing (each part reports its own failure with the traceback):
  1. the torch build and the visible GPU;
  2. MACE on CUDA: load, energy of a fixture complex on GPU vs CPU (must agree to float32 precision), and the
     time of a 25-structure batch on each device;
  3. the solver on CUDA: the same seeded solve on CPU and on the GPU (time, best score, peak GPU memory). The two
     are different floating-point paths, so scores agree only approximately; an exception here means the solver's
     CUDA path has a device bug -> use ``solver_device="cpu"`` (MACE still on the GPU) and report the traceback.

The last lines say which benchmark layout is faster here:
  device="cuda"                       solver + MACE on the GPU, cases inline
  solver_device="cpu", mace on cuda   CPU solves in parallel worker processes, one MACE model on the GPU
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from .env import FIX, REPO, use_src

CASES = ("XECSON", "AHIGOQ")


def _fixture(code: str):
    from fit.parser import TMQMParser

    for m in TMQMParser(FIX / "tmqm_sample40").stream_split("S"):
        if m.csd_code == code:
            return m
    raise KeyError(code)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--mace", default=None, help="MACE-POLAR-1 checkpoint (omit to skip the MACE part)")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    use_src()
    import numpy as np
    import torch

    ok = True
    print(f"torch {torch.__version__}; built for CUDA {torch.version.cuda}; CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("-> torch cannot use a GPU here. A '+cpu' torch wheel has no CUDA support: install the CUDA build "
              "of torch into this environment (pytorch.org -> your CUDA version), then re-run.")
        return 1
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}, {props.total_memory / 2**30:.1f} GB\n")

    from jacobik.core.energy import EV_TO_KJ_MOL, MACEPolarEvaluator
    from jacobik.stats.stats import StatsLookup, load_db
    from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol
    from jacobik import ComplexBuilder

    mols = {c: _fixture(c) for c in CASES}

    # ---- 2. MACE --------------------------------------------------------------------------------------------
    t_mace = {}
    if a.mace:
        print("== MACE ==")
        m = mols[CASES[0]]
        sy, X = [x.symbol for x in m.atoms], np.array([x.position for x in m.atoms])
        rng = np.random.default_rng(0)
        batch = [(sy, X + rng.normal(0, 0.02, X.shape), m.charge, m.spin_state) for _ in range(25)]
        e = {}
        for dev in ("cuda", "cpu"):
            try:
                t0 = time.perf_counter()
                ev = MACEPolarEvaluator(model_path=a.mace, device=dev, dtype=a.dtype).load()
                t_load = time.perf_counter() - t0
                e[dev] = ev.energy(sy, X, m.charge, m.spin_state)
                ev.energies(batch[:2])                        # warm-up (CUDA kernels, lazy init)
                t0 = time.perf_counter()
                ev.energies(batch)
                t_mace[dev] = time.perf_counter() - t0
                print(f"  {dev:4s}: load {t_load:5.1f}s   25 structures ({len(sy)} atoms) {t_mace[dev]:6.2f}s   "
                      f"E = {e[dev]:.4f} eV   {ev.describe()}")
                del ev
            except Exception:
                ok = False
                print(f"  {dev}: FAILED\n{traceback.format_exc()}")
        if len(e) == 2:
            d = abs(e["cuda"] - e["cpu"]) * EV_TO_KJ_MOL
            print(f"  |E(cuda) - E(cpu)| = {d:.3f} kJ/mol " + ("(ok)" if d < 1.0 else "(LARGER than float32 rounding: check)"))
            ok &= d < 1.0
        torch.cuda.empty_cache()
        print()

    # ---- 3. solver ------------------------------------------------------------------------------------------
    print("== solver ==")
    stats_db = load_db(a.db)
    t_solve = {"cpu": 0.0, "cuda": 0.0}
    solver_ok = True
    for code, m in mols.items():
        lig_in = ligand_inputs_from_mol(m, _build_ligand_groups(m, m.smiles))
        res = {}
        for dev in ("cpu", "cuda"):
            try:
                if dev == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                b = ComplexBuilder(metal=m.metal_symbol, geometry=m.coord_geom, charge=m.charge, spin=m.spin_state,
                                   ligands=lig_in)
                t0 = time.perf_counter()
                r = b.solve(stats=StatsLookup(stats_db), seed=a.seed, angle_bends=True, device=dev)
                dt = time.perf_counter() - t0
                t_solve[dev] += dt
                best = max(c.score for c in r.conformers)
                res[dev] = best
                mem = f"   peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MB" if dev == "cuda" else ""
                print(f"  {code} {dev:4s}: {dt:6.1f}s   best score {best:8.3f}   {len(r.conformers)} conformers{mem}")
            except Exception:
                solver_ok = False
                print(f"  {code} {dev}: FAILED\n{traceback.format_exc()}")
        if len(res) == 2:
            print(f"  {code}: score difference cuda - cpu = {res['cuda'] - res['cpu']:+.3f} "
                  "(different float paths; small differences are expected)")
    print()

    # ---- verdict ---------------------------------------------------------------------------------------------
    if not solver_ok:
        print("VERDICT: the solver FAILED on CUDA (traceback above). Benchmark with the CPU solver and MACE on the "
              "GPU:  bench.run_all(..., solver_device='cpu', mace_device='cuda', jobs=<cores/solver threads>)")
        return 1
    ratio = t_solve["cpu"] / max(t_solve["cuda"], 1e-9)
    print(f"solver: CPU {t_solve['cpu']:.1f}s vs CUDA {t_solve['cuda']:.1f}s over {len(mols)} cases "
          f"(CUDA is {ratio:.2f}x the CPU speed; the first CUDA solve includes kernel warm-up)")
    if t_mace:
        print(f"MACE:   CPU {t_mace.get('cpu', float('nan')):.2f}s vs CUDA {t_mace.get('cuda', float('nan')):.2f}s for 25 structures")
    if ratio >= 1.0:
        print("VERDICT: use device='cuda' (solver + MACE on the GPU, cases inline).")
    else:
        print("VERDICT: the CPU solver is faster per case here, and it also runs several cases in parallel. Use "
              "solver_device='cpu', mace_device='cuda', jobs=N: CPU worker processes solve, one MACE model on the "
              "GPU evaluates.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
