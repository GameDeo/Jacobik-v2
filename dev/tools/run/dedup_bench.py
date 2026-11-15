"""Offline timing and exact-identity check of the output dedup on recorded inputs.

  python -m tools.run.dedup_bench DUMP.pt [DUMP.pt ...] [--variants ref,threads4,par3] [--repeat 1]

Inputs come from `python -m tools.run.profile_parent --dump-dedup` (the deduper object, the rows, scores and k of every
StructureDeduper.select call in a real parent run, with the kept rows the run produced). Every variant must return
exactly the same kept list (same rows, same order) as the recorded run; the table says so per call.
Variants:
  ref        StructureDeduper.select as it is (one torch thread).
  threadsN   the same call with torch.set_num_threads(N) (intra-op threads in the matrix products).
  parN       the calls of one parent (its seeds) in N Python threads at once, one torch thread each (what
             multiseed.assemble_batched does with JACOBIK_DEDUP_SEED_THREADS=N); wall of the whole group.
  cuda       the deduper's tensors on the GPU, float64 (DEDUP_DEVICE="solver").
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import torch


def _load(p):
    return torch.load(p, weights_only=False)


def _call(d, dd=None):
    dd = dd or d["deduper"]
    return dd.select(d["X_all"], d["score"], d["k"], *d.get("args", ()), **d.get("kw", {}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.run.dedup_bench")
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--variants", default="ref,threads4,par3")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    ds = [_load(p) for p in a.dumps]
    print(f"{len(ds)} calls; maps {[d['n_maps'] for d in ds]}; rows {[int(d['X_all'].shape[0]) for d in ds]}; "
          f"kept {[len(d['keep']) for d in ds]}; recorded wall {sum(d['wall_s'] for d in ds):.2f} s")
    ok_all = True
    for v in a.variants.split(","):
        t0 = time.monotonic()
        if v == "ref":
            res = [_call(d) for d in ds]
        elif v.startswith("threads"):
            torch.set_num_threads(int(v[7:]))
            try:
                res = [_call(d) for d in ds]
            finally:
                torch.set_num_threads(1)
        elif v.startswith("par"):
            with ThreadPoolExecutor(max_workers=int(v[3:])) as ex:
                res = list(ex.map(_call, ds))
        elif v == "cuda":
            import copy
            res = []
            for d in ds:
                dd = copy.copy(d["deduper"])
                dev = torch.device("cuda")
                dd.device = dev
                for k in ("maps", "inv_maps", "heavy", "_rad_order"):
                    setattr(dd, k, getattr(dd, k).to(dev))
                dd._rad_groups = [g.to(dev) for g in dd._rad_groups]
                res.append(_call(dict(d, X_all=d["X_all"].to(dev)), dd))
            torch.cuda.synchronize()
        else:
            raise SystemExit(f"unknown variant {v}")
        wall = time.monotonic() - t0
        same = [list(r) == list(d["keep"]) for r, d in zip(res, ds)]
        ok_all &= all(same)
        print(f"{v:10s} wall {wall:8.2f} s   identical kept lists {sum(same)}/{len(same)}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
