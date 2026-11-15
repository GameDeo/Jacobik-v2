"""Panel of the S6 triage: tail parents plus every calibration parent with mean regret above 50 kJ/mol over seeds
11-13 (stored control pools, effective). Prints one JSON list with the regret and the stored top-1 dE per parent.

    python tools/jgen/diag/d31_s6_panel.py --root <frame_v1>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poolio import Repairs, load_effective  # noqa: E402

TAIL = ["DAMVAQ", "HOXJUZ", "IFUROQ", "JEGPUJ", "NEPNEF", "SOBHAS", "USAPOU", "XEYTUT", "YUCZUS", "ZELTAP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    m = json.load(open(os.path.join(a.root, "manifest.json")))
    cal = sorted({t["code"] for t in m["tasks"].values()
                  if t.get("split") == "calibration" and t.get("status") == "success"})
    rep = Repairs(a.root)
    rows = []
    for code in sorted(set(TAIL) | set(cal)):
        regs, tops = [], []
        for seed in (11, 12, 13):
            e = load_effective(os.path.join(a.root, "pools"), rep, code, seed)
            if e is None:
                continue
            t = float(e["dE"][int(e["score"].argmax())])
            regs.append(t - float(e["dE"].min()))
            tops.append(t)
        if not regs:
            continue
        mr = sum(regs) / len(regs)
        if code in TAIL or mr > 50.0:
            rows.append(dict(code=code, tail=code in TAIL, mean_regret=round(mr, 1),
                             mean_top1=round(sum(tops) / len(tops), 1)))
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
