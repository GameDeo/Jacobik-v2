"""S1 companion: central-S angle sum of the exported HOXJUZ top-1 and floor structures (xyz, reference atom order)
of one arm, against the reference (the S1 test of research/archive/docs/roadmap/roadmap_generation_v1.md 2.6). CPU only.

    python tools/jgen/diag/d28_hoxjuz_xyz_inversion.py --dir <inspect/jgen_bench/HOXJUZ> --arm nofold --ref <ref.xyz>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d27_hoxjuz_s1_inversion import angle, central_s, sum_at  # noqa: E402


def read_xyz(path):
    lines = open(path).read().split("\n")
    n = int(lines[0])
    sym, X = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        sym.append(p[0])
        X.append([float(v) for v in p[1:4]])
    return sym, np.array(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref-json", required=True, help="d27 output (reference angle, central S index)")
    a = ap.parse_args()
    ref = json.load(open(a.ref_json))["11"]
    out = {}
    for f in sorted(glob.glob(os.path.join(a.dir, f"HOXJUZ_{a.arm}_s*_*.xyz"))):
        sym, X = read_xyz(f)
        mi = sym.index("Ni")
        s = ref["central_S"]
        nbC = [j for j, t in enumerate(sym) if t == "C" and np.linalg.norm(X[j] - X[s]) < 2.0]
        out[os.path.basename(f)] = dict(angle_sum_S=round(sum_at(X, s, mi, nbC), 1),
                                        inverted=bool(sum_at(X, s, mi, nbC) < 270.0))
    print(json.dumps(dict(reference_angle_sum=ref["reference_angle_sum"], files=out), indent=1))


if __name__ == "__main__":
    main()
