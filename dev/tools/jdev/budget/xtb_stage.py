"""Energy-scale study, stage 2 (CPU): GFN2-xTB single points of every pooled conformer, and loose GFN2 relaxations
of a subset (seed-11 pools: top-8 by score + 4 lowest MACE dE; plus the reference) for the strain / basin split.

    python <this> sp   [--workers 10]
    python <this> opt  [--workers 10]
Writes <pool>.xtb.json: {"ref_sp": Eh, "sp": [Eh...], "ref_opt": Eh, "opt": {idx: [Eh, rmsd_moved]}}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
OUT = Path(os.environ.get("JACOBIK_BUDGET_OUT", REPO / "data" / "energy_scale"))
XTB = os.environ.get("JACOBIK_XTB") or shutil.which("xtb") or str(REPO / ".pixi/envs/default/Library/bin/xtb.exe")
TMP = OUT / "_xtbtmp"
_lock = threading.Lock()
E_RE = re.compile(rb"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")


def run_xtb(sy, X, q, mult, mode):
    TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TMP) as td:
        with open(f"{td}/m.xyz", "w") as f:
            f.write(f"{len(sy)}\n\n" + "".join(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(sy, X)))
        args = [XTB, "m.xyz", "--gfn", "2", "--chrg", str(q), "--uhf", str(mult - 1)]
        args += ["--sp"] if mode == "sp" else ["--opt", "loose", "--cycles", "400"]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="500M")
        p = subprocess.run(args, cwd=td, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=1800, env=env)
        m = E_RE.findall(p.stdout)
        if not m:
            return None
        E = float(m[-1])
        if mode == "sp":
            return E
        try:
            L = open(f"{td}/xtbopt.xyz").read().split("\n")[2: 2 + len(sy)]
            Y = np.array([[float(v) for v in ln.split()[1:4]] for ln in L])
            moved = float(np.sqrt(((Y - X) ** 2).sum(1).mean()))  # no alignment needed: xtb keeps the frame
        except Exception:
            Y, moved = None, float("nan")
        return [E, moved, Y]


def pools(seed_filter=None):
    for f in sorted((OUT / "pools").rglob("*.npz")):
        if seed_filter and not f.stem.endswith(f"_s{seed_filter}"):
            continue
        z = np.load(f, allow_pickle=False)
        if "error" in z.files:
            continue
        yield f, z


def save(f, key, val):
    with _lock:
        jf = f.with_suffix(".xtb.json")
        d = json.loads(jf.read_text()) if jf.exists() else {}
        if isinstance(val, tuple):  # (subkey, value) into a dict
            d.setdefault(key, {})[str(val[0])] = val[1]
        else:
            d[key] = val
        jf.write_text(json.dumps(d))


def stage_sp(a):
    jobs = []
    for f, z in pools():
        jf = f.with_suffix(".xtb.json")
        d = json.loads(jf.read_text()) if jf.exists() else {}
        if "sp" in d and "ref_sp" in d:
            continue
        jobs.append((f, [str(s) for s in z["symbols"]], z["X"], [str(s) for s in z["ref_symbols"]], z["ref_X"],
                     int(z["charge"]), int(z["mult"])))

    def work(j):
        f, sy, Xs, rsy, rX, q, m = j
        Es = [run_xtb(sy, X, q, m, "sp") for X in Xs]
        save(f, "sp", Es)
        save(f, "ref_sp", run_xtb(rsy, rX, q, m, "sp"))
        return f

    with ThreadPoolExecutor(a.workers) as ex:
        for k, f in enumerate(ex.map(work, jobs), 1):
            print(f"sp {k}/{len(jobs)} {f.parent.parent.name}/{f.parent.name}/{f.stem}", flush=True)


def stage_opt(a):
    jobs = []
    for f, z in pools(seed_filter=11):
        if f.parent.parent.name != "etkdg":  # oracle pools: the MACE comparison answers their question
            continue
        mf = f.with_suffix(".mace.json")
        jf = f.with_suffix(".xtb.json")
        d = json.loads(jf.read_text()) if jf.exists() else {}
        sy, q, m = [str(s) for s in z["symbols"]], int(z["charge"]), int(z["mult"])
        sel = list(np.argsort(-z["score"])[:48])  # top 48 by score (some pools hold 300+ conformers)
        done = d.get("opt", {})
        for i in sorted(set(int(i) for i in sel)):
            if str(i) not in done:
                jobs.append((f, "opt", i, sy, z["X"][i], q, m))
        if "ref_opt" not in d:
            jobs.append((f, "ref_opt", -1, [str(s) for s in z["ref_symbols"]], z["ref_X"], q, m))

    def work(j):
        f, key, i, sy, X, q, m = j
        r = run_xtb(sy, X, q, m, "opt")
        if r is not None:
            Y = r.pop()
            if Y is not None:
                d = f.parent / "relaxed" / f.stem
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"{i}.npy", Y)
        save(f, key, (i, r) if key == "opt" else r)
        return f, i

    with ThreadPoolExecutor(a.workers) as ex:
        for k, (f, i) in enumerate(ex.map(work, jobs), 1):
            if k % 20 == 0 or k == len(jobs):
                print(f"opt {k}/{len(jobs)} {f.parent.name}/{f.stem} #{i}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sp", "opt"])
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    print("xtb:", XTB, flush=True)
    {"sp": stage_sp, "opt": stage_opt}[a.stage](a)
