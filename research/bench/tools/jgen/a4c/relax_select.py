"""Relax-set rule for the P2R Step-2 baseline (replaces 'nearest by full RMSD' of `a4b.relax_set.select`, kept as is).

Per unit, at most CAP rows, in this priority:
  1. top-1;  2. floor;
  3. up to 3 same-class rows (vs the unrelaxed reference) nearest by CORE RMSD (<= 1.5 A): only same-class rows can
     make the unit a hit, so they come before anything else;
  4. the best-scored row of every other class present, in score order (classes by greedy matching to class
     representatives with the same descriptors), in the remaining slots.
Rows already chosen are not counted twice. Anything the cap drops is logged (`dropped`), never silent.
Why: A4b relaxed only rows within 1.5 A FULL RMSD, so rows that differ from the reference only in peripheral rotors
(often > 1.5 A full) were never relaxed and the class-aware hit could not see them (retrospective bias).
"""
from __future__ import annotations

import numpy as np

CAP = 5
CORE_NEAR_A = 1.5
MAX_SAME = 3


def select(ref, positions, scores, top, floor, cap=CAP):
    """ref: classes.Reference of the UNRELAXED reference; positions (n, atoms, 3) in reference order; scores (n,) with
    the top-1 row the best. Returns dict(rows=[...], same_class=[...], classes=[[rep, members...]], dropped=[...])."""
    from .classes import Reference
    X = np.asarray(positions, float)
    s = np.asarray(scores, float)
    order = list(np.argsort(s if s[top] <= s.min() else -s, kind="stable"))
    verdict = {i: ref.compare(X[i]) for i in range(len(X))}
    same = sorted((verdict[i]["core_rmsd_A"], i) for i in order
                  if verdict[i]["same_class"] and verdict[i]["core_rmsd_A"] <= CORE_NEAR_A)
    same = [i for _, i in same]
    reps, members = [], {}
    for i in order:
        if verdict[i]["same_class"]:
            continue
        for r in reps:
            if r[1].compare(X[i])["same_class"]:
                members[r[0]].append(int(i)); break
        else:
            reps.append((int(i), Reference(ref.symbols, X[i], graph_X=ref.graph_X, perms=ref.perms, capped=ref.capped)))
            members[int(i)] = [int(i)]
    want = [top, floor] + same[:MAX_SAME] + [r[0] for r in reps]
    rows, dropped = [], []
    for i in want:
        i = int(i)
        if i in rows:
            continue
        (rows if len(rows) < cap else dropped).append(i)
    n_same_any = sum(1 for i in order if verdict[i]["same_class"])
    n_ena = sum(1 for i in order if verdict[i]["enantiomer"])
    return dict(rows=rows, same_class=[int(i) for i in same], classes=[members[r[0]] for r in reps], dropped=dropped,
                cap=cap, n_pool=len(X), n_same_class_any=int(n_same_any), n_enantiomer_rows=int(n_ena),
                n_classes=len(reps) + (1 if n_same_any else 0))


def select_manifest(path):
    """One A4b pool manifest -> (sorted rows to relax, selection log). For a process pool (relax_run selector a4c)."""
    import json
    from pathlib import Path
    from tools.jgen.bench.pool_export import _read_frames
    from .classes import Reference
    mp = Path(path)
    m = json.loads(mp.read_text())
    mapping = np.asarray(m["gen_to_ref_idx"], int)
    frames = []
    for name in dict.fromkeys(r["file"] for r in m["rows"]):
        frames += [np.asarray(X)[mapping] for _, _, X in _read_frames(mp.parent / name)]
    if len(frames) != int(m["n_rows"]):
        raise ValueError("XYZ/manifest row count mismatch")
    top = next(i for i, r in enumerate(m["rows"]) if r.get("is_top1"))
    floor = next(i for i, r in enumerate(m["rows"]) if r.get("is_floor"))
    ref = Reference(m["ref_symbols"], np.asarray(m["ref_X"], float))
    out = select(ref, np.array(frames), np.array([float(r["score"]) for r in m["rows"]]), top, floor)
    out.update(code=m["code"], seed=int(m["seed"]), top=int(top), floor=int(floor), capped_graph=bool(ref.capped))
    return sorted(out["rows"]), out
