"""Which core feature separates the nearest same-class relaxed basin from the reference? (P2R Step 3 diagnostic)
For each parent: the same-class relaxed row with the smallest core RMSD; every heavy-atom torsion about a bond with both
ends in the core is compared with the reference (minimum over the first 256 automorphisms); torsions are tagged as
in a chelate ring (contains M), in a ligand ring, or acyclic. python -m tools.jgen.a4c.core_diff ROOT CODES... (ROOT =
A4b-layout dir with relax/ rows carrying X)"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import networkx as nx


def dih(X, a, b, c, d):
    b0, b1, b2 = X[a] - X[b], X[c] - X[b], X[d] - X[c]
    b1 /= np.linalg.norm(b1); v = b0 - b0 @ b1 * b1; w = b2 - b2 @ b1 * b1
    return np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))


def run(root, code):
    from tools.jgen.a4c.retro import _pool
    from tools.jgen.a4c.classes import Reference
    root = Path(root)
    m, _, _ = _pool(root, code, 11)
    ref = Reference(m["ref_symbols"], np.asarray(json.loads((root / "relax" / "references.json").read_text())[code]["X"], float),
                    graph_X=np.asarray(m["ref_X"], float))
    best = None
    for s in (11, 12, 13):
        p = root / "relax" / f"{code}_{s}.json"
        if not p.exists():
            continue
        for k, r in json.loads(p.read_text())["rows"].items():
            if r.get("X") is None:
                continue
            v = ref.compare(np.asarray(r["X"], float))
            if v["same_class"] and (best is None or v["core_rmsd_A"] < best[0]):
                best = (v["core_rmsd_A"], s, int(k), np.asarray(r["X"], float), r.get("offset"))
    if best is None:
        return dict(code=code, note="no same-class row")
    H = ref.heavy; G = nx.from_numpy_array(ref.adj.astype(int)); G.add_edges_from((ref.m, int(d)) for d in ref.donors); core = set(int(c) for c in ref.core)
    rings = [set(c) for c in nx.cycle_basis(G)]
    Xr, Xb = ref.Xh, best[3][H]
    tors = []
    for b, c in G.edges():
        if b not in core or c not in core:
            continue
        for a in G[b]:
            for d in G[c]:
                if a != c and d != b and a != d:
                    tors.append((a, b, c, d)); break
            break
    P = ref.perms[:256]
    def diffs(perm):
        Y = Xb[perm]
        out = []
        for t in tors:
            dd = abs((dih(Y, *t) - dih(Xr, *t) + 180) % 360 - 180); out.append(dd)
        return np.array(out)
    D = min((diffs(p) for p in P), key=lambda x: x.sum())
    lab = []
    for (a, b, c, d) in tors:
        rr = [r for r in rings if b in r and c in r]
        kind = "acyclic" if not rr else ("chelate" if any(ref.m in r for r in rr) else "ring")
        lab.append(kind)
    order = np.argsort(-D)[:5]
    agg = {k: float(D[[i for i, l in enumerate(lab) if l == k]].max()) if k in lab else None for k in ("chelate", "ring", "acyclic")}
    sym = [m["ref_symbols"][int(H[i])] for i in range(len(H))]
    top = [(lab[i], "-".join(f"{sym[x]}{int(H[x])}" for x in tors[i]), round(float(D[i]))) for i in order]
    return dict(code=code, core_rmsd=round(best[0], 2), offset=best[4], max_by_kind={k: None if v is None else round(v) for k, v in agg.items()}, top=top)


if __name__ == "__main__":
    for c in sys.argv[2:]:
        try:
            print(json.dumps(run(sys.argv[1], c), default=float))
        except Exception as e:
            print(c, "ERROR", type(e).__name__, e)
