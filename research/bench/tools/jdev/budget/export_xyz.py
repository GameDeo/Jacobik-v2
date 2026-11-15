"""Export pool conformers of chosen complexes as xyz files for visual inspection (2026-10-16).

    python tools/jdev/budget/export_xyz.py FIGHOT TAPWUG [--seed 11] [--top 5] [--out data/inspect]
Writes <out>/<CODE>/: reference.xyz, and per arm (etkdg, etkdg_isoref, full_oracle, ...): <arm>_s<seed>_rank<k>.xyz for
the top-k by score plus <arm>_s<seed>_floor.xyz (lowest MACE dE), each with score / MACE dE / assignment in the
comment line, and all_<arm>_s<seed>.xyz (multi-frame, score order) for viewers that page through frames.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
OUT = Path(os.environ.get("JACOBIK_BUDGET_OUT", REPO / "data" / "energy_scale_refit"))


def xyz(sy, X, comment):
    return f"{len(sy)}\n{comment}\n" + "".join(f"{s} {x:.5f} {y:.5f} {z:.5f}\n" for s, (x, y, z) in zip(sy, X))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("codes", nargs="+")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--out", default=str(REPO / "data" / "inspect"))
    a = p.parse_args()
    for code in a.codes:
        od = Path(a.out) / code
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted((OUT / "pools").glob(f"*/*/{code}_s{a.seed}.npz")):
            arm = f.parent.parent.name
            z = np.load(f, allow_pickle=False)
            if "error" in z.files:
                continue
            mf = f.with_suffix(".mace.json")
            e = np.array(json.loads(mf.read_text())["dE"]) if mf.exists() else np.full(len(z["score"]), np.nan)
            sy = [str(s) for s in z["symbols"]]
            if not (od / "reference.xyz").exists():
                (od / "reference.xyz").write_text(xyz([str(s) for s in z["ref_symbols"]], z["ref_X"],
                                                      f"{code} tmQM reference (GFN2 minimum of the crystal isomer)"))
            order = np.argsort(-z["score"])
            frames = []
            for k, i in enumerate(order):
                c = f"{code} {arm} seed {a.seed} rank {k + 1} score {z['score'][i]:.3f} MACE dE {e[i]:.1f} kJ/mol asg {z['asg'][i]}"
                frames.append(xyz(sy, z["X"][i], c))
                if k < a.top:
                    (od / f"{arm}_s{a.seed}_rank{k + 1}.xyz").write_text(frames[-1])
            if np.isfinite(e).any():
                i = int(np.nanargmin(e))
                (od / f"{arm}_s{a.seed}_floor.xyz").write_text(
                    xyz(sy, z["X"][i], f"{code} {arm} seed {a.seed} FLOOR (lowest MACE dE {e[i]:.1f}) score rank "
                                       f"{int(np.where(order == i)[0][0]) + 1} asg {z['asg'][i]}"))
            (od / f"all_{arm}_s{a.seed}.xyz").write_text("".join(frames))
        print(f"{code}: {od}")


if __name__ == "__main__":
    main()
