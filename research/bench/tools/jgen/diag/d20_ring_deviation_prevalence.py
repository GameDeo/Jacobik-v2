"""Ring-internal angle deviation of final pool rows, on consistent (and other) inputs (CPU, read-only).

Question (roadmap section 2.4 T3): how often do solved rows carry ring-internal angle distortion, and does the
distortion go with higher MACE energy? The embed or starting pose is not stored in the Step 2 pools, so the
comparison is against the REFERENCE geometry (evaluation only). For each ring of the reference heavy graph
(minimum cycle basis, 5 or 6 members, all light atoms), the internal angles of the row (in reference order,
X[:, gen_to_ref_idx]) are compared with those of the reference. A row's ring deviation is the maximum over its
rings of the RMS over the ring's internal angles of |theta_row - theta_ref| (degrees). Rows are then grouped at
5 and 15 degrees.

Reported per label (d14 labels): parents with any row above 5 deg and above 15 deg; the dE of rows above and at or
below 5 deg, paired within the same parent and seed (mean difference, rows above minus rows below); and the share
of parents with a ring whose reference internal angles are already off the regular polygon by more than 5 deg
(then deviations can be intrinsic to the reference, so these are reported separately).

    python tools/jgen/diag/d20_ring_deviation_prevalence.py --labels <d14 json> --out-json <path>
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxio  # noqa: E402
from poolio import Repairs, control_tasks, load_effective  # noqa: E402


def ring_angles(X, ring):
    n = len(ring)
    out = []
    for k in range(n):
        a, b, c = ring[k - 1], ring[k], ring[(k + 1) % n]
        u, v = X[a] - X[b], X[c] - X[b]
        cosv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        out.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    return np.array(out)


def ref_rings(ref, sym):
    import networkx as nx
    idx = [i for i, s in enumerate(sym) if s in ctxio.LIGHT and s != "H"]
    G = nx.Graph()
    G.add_nodes_from(idx)
    for ii, i in enumerate(idx):
        for j in idx[ii + 1:]:
            if np.linalg.norm(ref[i] - ref[j]) < 1.20 * (ctxio.COV[sym[i]] + ctxio.COV[sym[j]]):
                G.add_edge(i, j)
    out = []
    if G.number_of_edges():
        for c in nx.minimum_cycle_basis(G):
            if len(c) not in (5, 6):
                continue
            # order the cycle so that consecutive atoms are bonded
            order = [c[0]]
            rem = set(c[1:])
            while rem:
                nxt = [x for x in rem if G.has_edge(order[-1], x)]
                if not nxt:
                    break
                order.append(nxt[0])
                rem.discard(nxt[0])
            if len(order) == len(c) and G.has_edge(order[-1], order[0]):
                out.append(order)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    lab = {p["code"]: p["label"] for p in json.load(open(a.labels))["parents"]}
    rep = Repairs(str(ctxio.SCALED))
    pools = os.path.join(str(ctxio.SCALED), "pools")
    per_parent = collections.defaultdict(dict)
    for code, seed, _ in control_tasks(os.path.join(str(ctxio.SCALED), "manifest.json")):
        if code not in lab:
            continue
        p = load_effective(pools, rep, code, seed)
        if p is None:
            continue
        ref = np.asarray(p["ref_X"], float)
        sym = list(p["ref_symbols"])
        rings = ref_rings(ref, sym)
        X = p["X"][:, p["gen_to_ref"], :]
        ref_off = []
        for r in rings:
            th = ring_angles(ref, r)
            reg = 180.0 * (len(r) - 2) / len(r)
            ref_off.append(float(np.sqrt(np.mean((th - reg) ** 2))))
        row_dev = np.zeros(X.shape[0])
        for r in rings:
            th_ref = ring_angles(ref, r)
            for k in range(X.shape[0]):
                d = np.sqrt(np.mean((ring_angles(X[k], r) - th_ref) ** 2))
                row_dev[k] = max(row_dev[k], d)
        per_parent[code][seed] = dict(label=lab[code], row_dev=row_dev, de=p["dE"],
                                      ref_ring_off_gt5=bool(any(x > 5 for x in ref_off)), n_rings=len(rings))
    out = dict(parents=len(per_parent), by_label={})
    for L in ("consistent", "ambiguous", "inconsistent"):
        cs = [c for c in per_parent if per_parent[c][next(iter(per_parent[c]))]["label"] == L]
        any5 = any15 = refoff = 0
        hi_minus_lo = []
        for c in cs:
            devs = np.concatenate([v["row_dev"] for v in per_parent[c].values()])
            if devs.max() > 5:
                any5 += 1
            if devs.max() > 15:
                any15 += 1
            if any(v["ref_ring_off_gt5"] for v in per_parent[c].values()):
                refoff += 1
            for s, v in per_parent[c].items():
                hi = v["row_dev"] > 5
                if hi.any() and (~hi).any():
                    hi_minus_lo.append(float(v["de"][hi].mean() - v["de"][~hi].mean()))
        out["by_label"][L] = dict(parents=len(cs), parents_any_row_gt5=any5, parents_any_row_gt15=any15,
                                  parents_with_ref_ring_off_gt5=refoff,
                                  seeds_with_split=len(hi_minus_lo),
                                  seeds_hi_minus_lo_mean_dE=round(float(np.mean(hi_minus_lo)), 1) if hi_minus_lo else None,
                                  seeds_hi_gt_lo=int(sum(1 for x in hi_minus_lo if x > 0)))
    print(ctxio.to_json(out))
    if a.out_json:
        with open(a.out_json, "w") as fh:
            fh.write(ctxio.to_json(out))


if __name__ == "__main__":
    main()
