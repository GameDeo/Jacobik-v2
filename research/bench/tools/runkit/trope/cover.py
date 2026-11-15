"""Greedy weighted set cover of the trope tags by real parents, rarest tags first."""
from __future__ import annotations

from collections import Counter


def cover(parents: list, max_parents: int = 12, max_atoms: int = 100, atom_penalty: float = 0.004):
    """parents: [{code, n_atoms, tags}]. Returns (chosen list in pick order with the new tags each adds, uncovered tags,
    tag frequency Counter over eligible parents). Score of a parent = sum over its uncovered tags of 1/frequency (rare tags weigh
    more) - atom_penalty * n_atoms (cheap parents win ties). Parents above max_atoms are not eligible."""
    el = [p for p in parents if p["n_atoms"] <= max_atoms and p["tags"]]
    freq = Counter(t for p in el for t in p["tags"])
    unc = set(freq)
    chosen = []
    while unc and len(chosen) < max_parents:
        best, bs = None, 0.0
        for p in el:
            new = set(p["tags"]) & unc
            if not new:
                continue
            s = sum(1.0 / freq[t] for t in new) - atom_penalty * p["n_atoms"]
            if best is None or s > bs or (s == bs and p["code"] < best["code"]):
                best, bs = p, s
        if best is None:
            break
        new = sorted(set(best["tags"]) & unc)
        chosen.append(dict(code=best["code"], n_atoms=best["n_atoms"], new_tags=new, all_tags=sorted(best["tags"])))
        unc -= set(best["tags"])
    return chosen, sorted(unc), freq


def best_cover(parents, max_parents: int = 12):
    """Run `cover` over a fixed grid of (max_atoms, atom_penalty) and keep the solution with the fewest uncovered tags, then the
    fewest parents, then the fewest total atoms (cheapest suite). Deterministic."""
    best = None
    for ma in (80, 100, 130, 160):
        for pen in (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008):
            ch, unc, freq = cover(parents, max_parents, ma, pen)
            key = (len(unc), len(ch), sum(c["n_atoms"] for c in ch), ma, pen)
            if best is None or key < best[0]:
                best = (key, ch, unc, freq, dict(max_atoms=ma, atom_penalty=pen))
    return best[1], best[2], best[3], best[4]
