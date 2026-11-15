"""Consequential-change report between two constraint DBs (Step 4): populations, births/deaths, components, means/widths/weights.

    python research/campaigns/ds/db_compare_report.py A.json B.json --out report.json [--md report.md] [--watch KEY ...]

Entries are keyed `Q1:(...)`. "A" is the reference arm, "B" the changed arm. A key present in one DB is a birth/death. For keys in both the entries are
compared by: support n, component count, matched-component mean/var/weight shifts (components matched by sorted mean), and the entry source (`src`).
`consequential` marks a changed key when the component count differs, a matched mean moves by more than the per-quantity `MEAN_TOL` (circular difference for Q5) or a weight by more than `--weight-tol`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fit.pipeline.compare import load_container

#: per-quantity tolerance on a matched component mean (Q1 in Angstrom, Q2-Q5 in degrees, orientation/other features unitless)
MEAN_TOL = {"Q1": 0.02, "Q2": 1.0, "Q3": 1.0, "Q4": 1.0, "Q5": 2.0, "Q6": 0.05, "Q6P": 0.05, "QO": 0.05, "NCI": 0.05}
CIRCULAR = {"Q5"}
DEFAULT_WATCH = ["Q1:('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')", "QO:(2, 4)"]


def _components(e):
    m = e.get("means")
    if not isinstance(m, list) or not m:
        return None
    means = np.asarray([np.asarray(x, float).ravel()[0] for x in m])
    order = np.argsort(means)
    w = np.asarray(e.get("weights") or [1.0 / len(m)] * len(m), float)[order]
    v = e.get("vars")
    var = np.asarray([np.asarray(x, float).ravel()[0] if np.ndim(x) else float(x) for x in v], float)[order] if isinstance(v, list) and len(v) == len(m) else None
    return means[order], var, w


def compare(a: dict, b: dict, *, mean_tol=None, weight_tol=0.05, watch=DEFAULT_WATCH) -> dict:
    by_q = defaultdict(lambda: Counter())
    changed, births, deaths = [], [], []
    for k in sorted(set(a) | set(b)):
        q = k.split(":", 1)[0]
        if k not in b:
            deaths.append(k); by_q[q]["deaths"] += 1; continue
        if k not in a:
            births.append(k); by_q[q]["births"] += 1; continue
        ea, eb = a[k], b[k]
        by_q[q]["shared"] += 1
        row = dict(key=k, n_a=ea.get("n"), n_b=eb.get("n"), src_a=ea.get("src"), src_b=eb.get("src"))
        ca, cb = _components(ea), _components(eb)
        consequential = False
        if ca is not None and cb is not None:
            row["k_a"], row["k_b"] = len(ca[0]), len(cb[0])
            if len(ca[0]) != len(cb[0]):
                consequential = True
                row["component_change"] = f"{len(ca[0])}->{len(cb[0])}"
            else:
                from scipy.optimize import linear_sum_assignment

                diff = np.abs(ca[0][:, None] - cb[0][None, :])
                if q in CIRCULAR:
                    diff = np.minimum(diff, 360.0 - diff)
                r, c = linear_sum_assignment(diff)          # components matched by minimum mean distance (wrap-safe)
                dm = float(diff[r, c].max()); dw = float(np.max(np.abs(ca[2][r] - cb[2][c])))
                row["max_mean_shift"], row["max_weight_shift"] = dm, dw
                consequential = dm > (mean_tol if mean_tol is not None else MEAN_TOL.get(q, 0.05)) or dw > weight_tol
        elif ca is None and cb is None:
            same = json.dumps(ea, sort_keys=True) == json.dumps(eb, sort_keys=True)
            consequential = not same
        else:
            consequential = True
        if (ea.get("n") != eb.get("n")) or consequential:
            row["consequential"] = bool(consequential)
            changed.append(row)
        if consequential:
            by_q[q]["consequential"] += 1
        if ea.get("n") != eb.get("n"):
            by_q[q]["support_changed"] += 1
    watches = {}
    for w in watch:
        watches[w] = dict(in_a=w in a, in_b=w in b,
                          a=_summ(a.get(w)), b=_summ(b.get(w)))
    return dict(entries_a=len(a), entries_b=len(b), by_quantity={q: dict(c) for q, c in sorted(by_q.items())},
                n_births=len(births), n_deaths=len(deaths), births=births[:50], deaths=deaths[:50],
                n_changed=len(changed), changed_top=sorted(changed, key=lambda r: -(r.get("max_mean_shift") or 0))[:50],
                watches=watches, mean_tol=mean_tol, weight_tol=weight_tol)


def _summ(e):
    if e is None:
        return None
    c = _components(e)
    out = dict(n=e.get("n"), src=e.get("src"), level=e.get("level"))
    if c is not None:
        out.update(means=np.round(c[0], 4).tolist(), weights=np.round(c[2], 4).tolist())
    return out


def to_markdown(rep: dict, name_a: str, name_b: str) -> str:
    lines = [f"# DB comparison: {name_a} -> {name_b}", "", f"entries {rep['entries_a']} -> {rep['entries_b']}; births {rep['n_births']}, deaths {rep['n_deaths']}, "
             f"changed (support or fit) {rep['n_changed']}", "", "| quantity | shared | births | deaths | support changed | consequential |", "|---|---:|---:|---:|---:|---:|"]
    for q, c in rep["by_quantity"].items():
        lines.append(f"| {q} | {c.get('shared', 0)} | {c.get('births', 0)} | {c.get('deaths', 0)} | {c.get('support_changed', 0)} | {c.get('consequential', 0)} |")
    lines += ["", "## Watch keys"]
    for k, w in rep["watches"].items():
        lines.append(f"- `{k}`: A {w['a']}  ->  B {w['b']}")
    lines += ["", "## Largest mean shifts"]
    for r in rep["changed_top"][:15]:
        lines.append(f"- `{r['key']}` n {r['n_a']}->{r['n_b']} k {r.get('k_a')}->{r.get('k_b')} mean shift {r.get('max_mean_shift')} weight shift {r.get('max_weight_shift')}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path); ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--md", type=Path)
    ap.add_argument("--mean-tol", type=float, default=None, help="override the per-quantity tolerance table"); ap.add_argument("--weight-tol", type=float, default=0.05)
    ap.add_argument("--watch", nargs="*", default=DEFAULT_WATCH)
    args = ap.parse_args(argv)
    (ea, _, _), (eb, _, _) = load_container(args.a), load_container(args.b)
    rep = compare(ea, eb, mean_tol=args.mean_tol, weight_tol=args.weight_tol, watch=args.watch)
    args.out.write_text(json.dumps(rep, indent=1, default=str))
    if args.md:
        args.md.write_text(to_markdown(rep, args.a.stem, args.b.stem))
    print(to_markdown(rep, args.a.stem, args.b.stem)[:2500])


if __name__ == "__main__":
    main()
