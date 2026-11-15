"""MACE-POLAR-1-S float64 throughput vs batch size, on recorded pool structures (profile_parent --dump-dedup).

  python -m tools.run.mace_bench DUMP.pt [DUMP.pt ...] [--atoms 900,2700,8100] [--mixed]

Per batch-atom cap: wall, structures per second, and max |dE| (eV) against the recorded per-parent energies (the
bench's own call, 900-atom batches). --mixed evaluates all dumps' structures in one call (parents mixed in batches).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.run.mace_bench")
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--atoms", default="900,2700,8100")
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from jacobik.core.energy import MACEPolarEvaluator
    from jacobik.core.env import MACE_MODEL_PATH
    mace = MACEPolarEvaluator(model="polar-1-s", device=a.device, dtype="float64", model_path=str(MACE_MODEL_PATH),
                              mode="energy-only", threads=1).load()
    ds = [torch.load(p, weights_only=False) for p in a.dumps]
    groups = [(sum((d["structures"] for d in ds), []), np.concatenate([d["energies"] for d in ds]))] if a.mixed else \
        [(d["structures"], d["energies"]) for d in ds]
    mace._batch_size = 10**6
    mace._batch_atoms = 900
    mace.energies(groups[0][0][:4])  # warm-up
    print(f"{len(ds)} dumps, {sum(len(g[0]) for g in groups)} structures, recorded wall {sum(d['wall_s'] for d in ds):.2f} s")
    for cap in [int(x) for x in a.atoms.split(",")]:
        tot, n, worst = 0.0, 0, 0.0
        for st, ref in groups:
            mace._batch_size, mace._batch_atoms = 10**6, cap
            if a.device == "cuda":
                torch.cuda.synchronize()
            t = time.monotonic()
            e = mace.energies(st)
            if a.device == "cuda":
                torch.cuda.synchronize()
            tot += time.monotonic() - t
            n += len(st)
            worst = max(worst, float(np.max(np.abs(e - ref))))
        print(f"batch atoms {cap:6d}: wall {tot:7.2f} s, {n / tot:7.1f} structures/s, max |dE| vs recorded {worst:.2e} eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
