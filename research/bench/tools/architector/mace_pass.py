"""MACE energies for saved Architector structures (GPU pass, written for Amendment 1; NOT run in the CPU stage).

Reads every job's arch_init_k / arch_final_k / relax_init_k / relax_final_k .xyz under run_dir/structures and writes
mace.json beside each job's post.json. Model and charge/spin handling are those of the jgen bench: MACE-POLAR-1-S,
float64, energy-only, charge and multiplicity (not unpaired count) from the job's input.json. Idempotent per job.

Usage (root env, PYTHONPATH=src:.:research/bench): python -m tools.architector.mace_pass --run-dir <run dir> --device cuda
then re-run `python -m tools.architector.post --run-dir <run dir> --tables-only` after filling mace.json to rebuild the
tables (write_tables fills mace_kjmol from each job's mace.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.runkit import mace as rk_mace
from tools.architector.post import read_xyz


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--job-glob", default="*/*/*/seed_*")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    ev = rk_mace.Evaluator(device=a.device, dtype="float64", threads=1)
    for pj in sorted((run / "structures").glob(a.job_glob + "/post.json")):
        out = pj.parent / "mace.json"
        if out.exists():
            continue
        inp = json.loads((pj.parent / "input.json").read_text())
        q = int(inp["parameters"]["full_charge"])
        mult = int(inp["parameters"]["full_spin"]) + 1
        items, keys = [], []
        for f in sorted(pj.parent.glob("*_[0-9][0-9][0-9].xyz")):
            tag = f.name.rsplit("_", 1)[0]  # arch_init, arch_final, relax_init, relax_final
            if tag not in ("arch_init", "arch_final", "relax_init", "relax_final"):
                continue
            sym, X = read_xyz(f)
            items.append((sym, X, q, mult))
            keys.append(f.name[:-4])
        body = {}
        try:
            if items:
                energies = ev.energies_kj(items)
                body = {k: float(e) for k, e in zip(keys, energies)}
            out.write_text(json.dumps(body, indent=1, allow_nan=False) + "\n")
        except Exception as exc:  # record and continue
            out.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
