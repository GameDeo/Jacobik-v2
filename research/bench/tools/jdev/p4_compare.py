"""P4 verification: unified DB vs historical production plus the archived add-on DBs.

Exit 1 also reports the 50 known QV/QVR float-order differences; inspect their
quantities rather than treating this as a no-difference regression check.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[4]
F = REPO / "data/refit/final"
new = json.loads((F / "constraints_2026-10-17.json").read_text())
exp = json.loads((F / "constraints_refit_q6p5.json").read_text())
for extra in ("m13", "bvs", "nci"):
    name = f"constraints_refit_q6p5_{extra}.json"
    if (F / name).exists():
        d = json.loads((F / name).read_text())
    else:
        with zipfile.ZipFile(REPO / "data/_archive/refit_addon_dbs_2026-10-18.zip") as z:
            d = json.loads(z.read(f"data/refit/final/{name}"))
    exp.update({k: v for k, v in d.items() if k not in exp})
TOL = 1e-9


def diff(a, b, path, out, budget=[0]):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'missing in new' if k not in a else 'only in new'}")
            else:
                diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(b)} -> {len(a)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=0, abs_tol=TOL) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {b} -> {a}")
    elif a != b:
        out.append(f"{path}: {b!r} -> {a!r}")


missing = [k for k in exp if k not in new]
extra = [k for k in new if k not in exp]
bad = {}
for k in exp:
    if k in new:
        o = []
        diff(new[k], exp[k], "", o)
        if o:
            bad[k] = o
print(f"expected {len(exp)} entries, unified {len(new)}; missing {len(missing)}, extra {len(extra)}, differing {len(bad)}")
print("missing by quantity:", Counter(k.split(':')[0] for k in missing))
print("extra by quantity:", Counter(k.split(':')[0] for k in extra))
print("differing by quantity:", Counter(k.split(':')[0] for k in bad))
for k, o in list(bad.items())[:12]:
    print(" ", k, o[:3])
sys.exit(0 if not (missing or extra or bad) else 1)
