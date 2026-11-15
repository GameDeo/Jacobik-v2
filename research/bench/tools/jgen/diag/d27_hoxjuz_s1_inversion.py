"""S1 (research/archive/docs/roadmap/roadmap_generation_v1.md 2.6): is the central S of HOXJUZ inverted, and does any stored row avoid it?

For each stored effective control pool of HOXJUZ (seeds 11-13): the Ni-bonded sulfur whose neighbours are two carbons
and Ni (the central S). Its angle sum (C-S-C, C-S-Ni, C-S-Ni) is compared with the reference (315 deg); a row is
'inverted' when the sum is below 270 deg. Reports every row's score rank, dE and inversion state. CPU only.

    python tools/jgen/diag/d27_hoxjuz_s1_inversion.py --root <frame_v1>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poolio import Repairs, load_effective  # noqa: E402


def angle(a, v, b):
    u, w = a - v, b - v
    return float(np.degrees(np.arccos(np.clip(np.dot(u, w) / np.linalg.norm(u) / np.linalg.norm(w), -1, 1))))


def central_s(X, sym, mi):
    cands = []
    for i, s in enumerate(sym):
        if s != "S":
            continue
        d_ni = np.linalg.norm(X[i] - X[mi])
        if d_ni > 2.8:
            continue
        nb = [j for j, t in enumerate(sym) if t == "C" and np.linalg.norm(X[j] - X[i]) < 2.0]
        if len(nb) == 2:
            cands.append(i)
    return cands


def sum_at(X, i, mi, nb):
    return angle(X[nb[0]], X[i], X[nb[1]]) + angle(X[nb[0]], X[i], X[mi]) + angle(X[nb[1]], X[i], X[mi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    pools = os.path.join(a.root, "pools")
    rep = Repairs(a.root)
    out = {}
    for seed in (11, 12, 13):
        e = load_effective(pools, rep, "HOXJUZ", seed)
        if e is None:
            continue
        X = e["X"][:, e["gen_to_ref"], :]
        ref = e["ref_X"]
        sym = e["ref_symbols"]
        mi = sym.index("Ni")
        ss = central_s(ref, sym, mi)
        if not ss:
            out[seed] = "no central S found in reference"
            continue
        s_ref = ss[0]
        nbC = [j for j, t in enumerate(sym) if t == "C" and np.linalg.norm(ref[j] - ref[s_ref]) < 2.0]
        ref_sum = sum_at(ref, s_ref, mi, nbC)
        rows = []
        order = np.argsort(-e["score"])
        for rank, r in enumerate(order):
            sm = sum_at(X[r], s_ref, mi, nbC)
            rows.append(dict(rank=rank, score=round(float(e["score"][r]), 3), dE=round(float(e["dE"][r]), 1),
                             angle_sum_S=round(sm, 1), inverted=bool(sm < 270.0)))
        out[seed] = dict(central_S=int(s_ref), reference_angle_sum=round(ref_sum, 1), rows=rows,
                         n_rows=len(rows), n_inverted=sum(x["inverted"] for x in rows),
                         best_non_inverted=next((x for x in rows if not x["inverted"]), None))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
