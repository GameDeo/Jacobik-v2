"""Top-1 identity between two record directories (OUT/<arm>): what "same results" means for the runner.

Per (code, seed): status; top-1 row key (argmax of score, as the bench picks it); top-1 score (abs 1e-6); top-1 MACE
dE (abs 1e-3 kJ/mol); top-1 geometry from the exported pool (max |dx| <= 1e-5 A). Non-top-1 pool rows are only counted
(n_rows), never required to match: on a GPU they vary run to run even on the same card.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _top1(r: dict):
    sc = r.get("scores") or []
    if not sc:
        return None
    vals = [x if (x is not None and math.isfinite(x)) else -math.inf for x in sc]
    i = max(range(len(vals)), key=lambda j: vals[j])
    keys = r.get("row_keys") or [None] * len(sc)
    return dict(i=i, key=keys[i] if i < len(keys) else None, score=sc[i], dE=r.get("top1_dE"))


def _top1_xyz(r: dict):
    import numpy as np
    mp = r.get("pool_manifest")
    if not mp or not Path(mp).exists():
        return None
    from tools.jgen.bench.pool_export import _read_frames
    m = json.loads(Path(mp).read_text())
    for row in m.get("rows", []):
        f = Path(mp).parent / row["file"]
        if not f.exists():
            return None
        for fr in _read_frames(f):
            hdr = fr[0] if isinstance(fr[0], dict) else fr[1] if isinstance(fr[1], dict) else {}
            if hdr.get("is_top1"):
                return np.asarray(fr[2], float)
    return None


def compare(new_dir: Path, ref_dir: Path, codes, seeds, arm: str = "", tol_score=1e-6, tol_de=1e-3, tol_x=1e-5) -> dict:
    import numpy as np
    if codes is None:
        codes = sorted({p.stem.rsplit("_", 1)[0] for p in (new_dir / "records").glob("*.json")})
    rows = []
    for c in codes:
        for s in seeds:
            fn, fr = new_dir / "records" / f"{c}_{s}.json", ref_dir / "records" / f"{c}_{s}.json"
            if not fn.exists() or not fr.exists():
                rows.append(dict(code=c, seed=s, identical=False, why="record missing"))
                continue
            a, b = json.loads(fn.read_text()), json.loads(fr.read_text())
            ta, tb = _top1(a), _top1(b)
            why = []
            if a.get("status") != b.get("status"):
                why.append(f"status {b.get('status')} -> {a.get('status')}")
            if (ta is None) != (tb is None):
                why.append("top-1 present in only one")
            elif ta is not None:
                if ta["key"] != tb["key"]:
                    why.append(f"key {tb['key']} -> {ta['key']}")
                if abs(ta["score"] - tb["score"]) > tol_score:
                    why.append(f"score {tb['score']:.6f} -> {ta['score']:.6f}")
                if ta["dE"] is not None and tb["dE"] is not None and abs(ta["dE"] - tb["dE"]) > tol_de:
                    why.append(f"dE {tb['dE']:.3f} -> {ta['dE']:.3f}")
                try:
                    xa, xb = _top1_xyz(a), _top1_xyz(b)
                except Exception as exc:  # geometry is a check, not a crash
                    xa = xb = None
                    why.append(f"xyz unreadable {exc!r}"[:80])
                if xa is not None and xb is not None:
                    dx = float(np.abs(xa - xb).max()) if xa.shape == xb.shape else float("inf")
                    if dx > tol_x:
                        why.append(f"top-1 xyz max|dx| {dx:.2e}")
            rows.append(dict(code=c, seed=s, identical=not why, why="; ".join(why), n_rows_ref=b.get("n_rows"),
                             n_rows_new=a.get("n_rows"), top1_score_ref=(tb or {}).get("score"),
                             top1_score_new=(ta or {}).get("score"), gpu_ref=b.get("gpu_name"), gpu_new=a.get("gpu_name")))
    return dict(n=len(rows), identical=sum(r["identical"] for r in rows),
                pool_rows_equal=sum(r.get("n_rows_ref") == r.get("n_rows_new") for r in rows), rows=rows)


def print_compare(res: dict):
    for r in res["rows"]:
        print(f"{r['code']:<8} s{r['seed']}  {'IDENTICAL' if r['identical'] else 'DIFFERS  '}  "
              f"rows {r.get('n_rows_ref')}->{r.get('n_rows_new')}  {r['why']}")
    print(f"top-1 identical: {res['identical']}/{res['n']}   pool size equal: {res['pool_rows_equal']}/{res['n']}")
