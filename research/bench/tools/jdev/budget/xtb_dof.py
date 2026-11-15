"""Energy-scale study: DOF-only GFN2 relaxation (every ligand 1-2 and 1-3 distance held by $constrain, so only
torsions, rigid placement and metal-ligand geometry move) for each seed-11 pool's #1 (score), best-SP and best-basin
conformers. Adds "dof": {idx: [Eh, rms_moved]} to <pool>.xtb.json.

    python <this> [--workers 11]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from xtb_stage import E_RE, OUT, TMP, XTB, pools, save  # noqa: E402

COV = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07, "S": 1.05,
       "Cl": 1.02, "As": 1.19, "Se": 1.20, "Br": 1.20, "Te": 1.38, "I": 1.39, "Al": 1.21, "Ge": 1.20, "Sn": 1.39}


def ligand_pairs(sy, X):
    """1-2 and 1-3 pairs among non-metal atoms (metal = row 0 and any atom without a covalent radius here)."""
    idx = [i for i, s in enumerate(sy) if i != 0 and s in COV]
    P = X[idx]
    r = np.array([COV[sy[i]] for i in idx])
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    B = (D < 1.25 * (r[:, None] + r[None])) & ~np.eye(len(idx), dtype=bool)
    import networkx as nx

    pairs = set()
    nb = [np.flatnonzero(B[a]) for a in range(len(idx))]
    for a in range(len(idx)):
        for b in nb[a]:
            pairs.add((min(a, b), max(a, b)))
            for c in nb[b]:
                if c != a:
                    pairs.add((min(a, c), max(a, c)))
    # 1-4 pairs across every ring bond (non-bridge of the metal-free ligand graph): ring torsions are internal
    # geometry the solver cannot move; acyclic bonds (incl. chelate backbones, acyclic without the metal) stay free
    G = nx.Graph()
    G.add_nodes_from(range(len(idx)))
    G.add_edges_from((a, b) for a in range(len(idx)) for b in nb[a] if a < b)
    bridges = {frozenset(e) for e in nx.bridges(G)}
    for b, c in G.edges():
        if frozenset((b, c)) in bridges:
            continue
        for a in nb[b]:
            for d in nb[c]:
                if a not in (b, c) and d not in (b, c) and a != d:
                    pairs.add((min(a, d), max(a, d)))
    return [(idx[a], idx[b]) for a, b in sorted(pairs)]


def donors(sy, X, rM=1.45, pad=0.5):
    d = np.linalg.norm(X - X[0], axis=1)
    return [i for i, s in enumerate(sy) if i != 0 and s in COV and s != "H" and d[i] < rM + COV[s] + pad]


def run_dof(sy, X, q, mult, coord=False):
    TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TMP) as td:
        with open(f"{td}/m.xyz", "w") as f:
            f.write(f"{len(sy)}\n\n" + "".join(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(sy, X)))
        with open(f"{td}/c.inp", "w") as f:
            f.write("$constrain\n force constant=1.0\n")
            cons = ligand_pairs(sy, X)
            if coord:  # freeze the coordination polyhedron too: metal-donor and donor-donor distances
                dn = donors(sy, X)
                cons += [(0, i) for i in dn] + [(a, b) for k, a in enumerate(dn) for b in dn[k + 1:]]
            for i, j in cons:
                f.write(f" distance: {i + 1}, {j + 1}, auto\n")
            f.write("$end\n")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="500M")
        args = [XTB, "m.xyz", "--gfn", "2", "--chrg", str(q), "--uhf", str(mult - 1), "--opt", "loose",
                "--cycles", "400", "--input", "c.inp"]
        subprocess.run(args, cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800, env=env)
        try:
            L = open(f"{td}/xtbopt.xyz").read().split("\n")[2: 2 + len(sy)]
            Y = np.array([[float(v) for v in ln.split()[1:4]] for ln in L])
        except Exception:
            return None
        # energy of the constrained geometry WITHOUT the restraint energy: a plain single point
        with open(f"{td}/y.xyz", "w") as f:
            f.write(f"{len(sy)}\n\n" + "".join(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(sy, Y)))
        p = subprocess.run([XTB, "y.xyz", "--gfn", "2", "--chrg", str(q), "--uhf", str(mult - 1), "--sp"], cwd=td,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=600, env=env)
        m = E_RE.findall(p.stdout)
        return [float(m[-1]), float(np.sqrt(((Y - X) ** 2).sum(1).mean()))] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--coord", action="store_true", help="also freeze the coordination polyhedron (#1 picks only)")
    a = ap.parse_args()
    jobs = []
    for f, z in pools(seed_filter=11):
        if f.parent.parent.name != "etkdg":
            continue
        jf = f.with_suffix(".xtb.json")
        if not jf.exists():
            continue
        d = json.loads(jf.read_text())
        if "sp" not in d or "opt" not in d:
            continue
        sel = {int(np.argmax(z["score"]))}
        sp = np.array([e if e is not None else np.inf for e in d["sp"]])
        sel.add(int(np.argmin(sp)))
        op = {int(k): v[0] for k, v in d["opt"].items() if v}
        if op:
            sel.add(min(op, key=op.get))
        key = "dofc" if a.coord else "dof"
        if a.coord:
            sel = {int(np.argmax(z["score"]))}  # the #1 pick only
        done = d.get(key, {})
        sy, q, m = [str(s) for s in z["symbols"]], int(z["charge"]), int(z["mult"])
        jobs += [(f, i, sy, z["X"][i], q, m) for i in sorted(sel) if str(i) not in done]

    def work(j):
        f, i, sy, X, q, m = j
        save(f, key, (i, run_dof(sy, X, q, m, coord=a.coord)))
        return f, i

    print(f"dof jobs: {len(jobs)}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (f, i) in enumerate(ex.map(work, jobs), 1):
            if k % 20 == 0 or k == len(jobs):
                print(f"dof {k}/{len(jobs)} {f.parent.name}/{f.stem} #{i}", flush=True)


if __name__ == "__main__":
    main()
