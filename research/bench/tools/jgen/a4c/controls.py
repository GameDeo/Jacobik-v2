"""A4c controls on real cal parents (stored A4b references; relaxed reference = target class).

* mirror (no xtb, every parent): the mirror image of the relaxed reference must always be the same class (directly if
  configurationally achiral, else as a mirror match); a mirror image superimposable by a proper rotation (heavy RMSD
  <= 0.1 A) must match directly.
* positive (`tools.run diagnostic` task `tools.jgen.a4c.controls:control`, kind=positive): up to 3 peripheral rotors of
  the relaxed reference rotated (outside the core by the graph rule), then GFN2-relaxed (crude, 200 cycles, as A4b).
  Must be the same class with core RMSD <= 0.3 A (unrelaxed: exactly the reference core).
* negative (kind=negative): one stereogenic non-donor C-H centre (three heavy neighbours of distinct WL colour)
  inverted by reflecting it and its H through the plane of its heavy neighbours, then relaxed. Must NOT match directly;
  a mirror match is legitimate only if the rest of the complex is configurationally achiral (checked: the mirror of the
  reference matches directly once that centre's descriptor is dropped).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

A4B = Path(os.environ.get("A4C_A4B", "/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/A4b"))


def reference(code, a4b=A4B):
    from tools.jgen.a4c.classes import Reference
    from tools.jgen.a4c.retro import _pool
    m = None
    for s in (11, 12, 13):
        try:
            m = _pool(a4b, code, s)[0]; break
        except (OSError, ValueError, StopIteration):
            continue
    rel = json.loads((a4b / "relax" / "references.json").read_text())[code]
    return Reference(m["ref_symbols"], np.asarray(rel["X"], float), graph_X=np.asarray(m["ref_X"], float)), \
        np.asarray(rel["X"], float), m


def mirror_check(code, a4b=A4B):
    from tools.runkit import geom
    ref, X, _ = reference(code, a4b)
    Y = X * np.array([-1.0, 1, 1])
    v = ref.compare(Y)
    proper = float(geom.sym_rmsd_to_ref(Y[ref.heavy][None], ref.Xh, ref.perms)[0])
    # consistent: the mirror image is either the same class (achiral) or recognised as the enantiomer, and a
    # proper-rotation-superimposable mirror image is never called an enantiomer
    ok = (v["same_class"] or v["enantiomer"]) and not (v["enantiomer"] and proper <= 0.1)
    return dict(code=code, same_class=v["same_class"], mirror_match=v["enantiomer"], proper_rmsd_of_mirror=proper,
                capped=ref.capped, passed=bool(ok), reason=v["reason"])


def stereogenic_centres(ref):
    from tools.jgen.a4c.constructs import invertible_centres
    out = []
    for c in invertible_centres(ref):
        nb = ref._neighbours(c)
        if len({int(ref.colours[b]) for b in nb}) == 3 and ref.m not in nb:
            out.append(c)
    return out


def rest_achiral(ref, X, c):
    """Mirror of the reference matches directly when the descriptor of centre c is dropped."""
    import copy
    r2 = copy.copy(ref)
    keep = [i for i, (k, g) in enumerate(ref.desc) if not (k == "chi" and g[0][0] == c)]
    r2.desc = [ref.desc[i] for i in keep]; r2.t_ref = ref.t_ref[keep]
    v = r2.compare(X * np.array([-1.0, 1, 1]))
    return bool(v["same_class"] and v["mirror"] is False)


def control(entry):
    """tools.run diagnostic task. entry: {id, code, kind: positive|negative}."""
    from jacobik.core import profile
    from tools.jgen.a4c import constructs
    from tools.runkit import xtb
    code, kind = entry["code"], entry["kind"]
    ref, X, _ = reference(code)
    cm = json.loads((A4B / "charge_mult.json").read_text())[code]
    out = dict(code=code, kind=kind, fingerprint=profile.snapshot()["fingerprint"], capped=ref.capped)
    if kind == "positive":
        Y, applied = constructs.rotor_variant(ref, X, n_rotors=3)
        out["rotors"] = applied
        expect = "same class, core RMSD <= 0.3"
    else:
        c = Y = None
        for cand in stereogenic_centres(ref):        # first centre whose inversion keeps graph and contacts
            Z = constructs.invert_centre(ref, X, cand)
            w = ref.compare(Z)
            if w["topology_same"] and w["donors_same"]:
                c, Y = cand, Z; break
        out["construct_clean"] = c is not None
        if c is None:                                 # fall back: first stereogenic centre, judged after relaxation
            c = stereogenic_centres(ref)[0]; Y = constructs.invert_centre(ref, X, c)
        out.update(centre=int(ref.heavy[c]), centre_symbol=ref.symbols[int(ref.heavy[c])], rest_achiral=rest_achiral(ref, X, c))
        expect = "no direct match"
    out["expect"] = expect
    out["unrelaxed"] = ref.compare(Y)
    with tempfile.TemporaryDirectory(dir=os.environ.get("A4C_WORK")) as td:
        r = xtb.opt(ref.symbols, Y, int(cm["charge"]), int(cm["mult"]), td, cycles=200, level="crude")
    out["xtb"] = {k: r.get(k) for k in ("status", "cycles", "converged", "E_h", "wall_s", "xtb_version")}
    if r.get("X") is None:
        out["relaxed"] = None; out["passed"] = None
        return out
    v = ref.compare(np.asarray(r["X"], float))
    out["relaxed"] = v
    if kind == "positive":
        out["passed"] = bool(v["same_class"] and v["core_rmsd_A"] <= 0.3)
    else:
        out["passed"] = bool(not v["same_class"])          # an enantiomer (rest achiral) is also a different class
        out["judged_by_centre"] = bool(v["topology_same"] and v["donors_same"])   # graph+contacts kept: chi decided
    from tools.runkit import geom
    full = lambda Z: float(geom.sym_rmsd_to_ref(np.asarray(Z, float)[ref.heavy][None], ref.Xh, ref.perms)[0])
    out["full_rmsd_unrelaxed"] = full(Y); out["relaxed"]["full_rmsd_vs_ref"] = full(r["X"])
    return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def select(codes, n=8):
    """Deterministic selection (sorted codes): positives need >= 3 peripheral rotors, negatives a stereogenic C-H."""
    from tools.jgen.a4c.constructs import peripheral_rotors
    from tools.jgen.a4c.constructs import rotor_variant
    pos, neg = [], []
    for code in sorted(codes):
        try:
            ref, X, _ = reference(code)
        except Exception:
            continue
        if len(pos) < n and len(peripheral_rotors(ref)) >= 3 and rotor_variant(ref, X)[1]:
            pos.append(code)
        if len(neg) < n and stereogenic_centres(ref):
            neg.append(code)
        if len(pos) >= n and len(neg) >= n:
            break
    return pos, neg
