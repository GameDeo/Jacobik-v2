"""
hypothesis_tables.py
====================
B2.5: rows and fits for the priors of the coordination-hypothesis layer (plan 2026-09-26, B4). Extracted with the
other observations (extractor.extract_split), fitted at the refit, stored in the constraints DB.

* QG  geometry prior      P(geometry | metal, oxidation state, CN)
      one row per complex: label, runner-up label and margin (coord_geom.classify_with_margin).
      Fit: soft counts -- a complex whose margin is below the CN's band (the 10th percentile of its margins in the
      training data) counts partly for its runner-up, linearly down to 50/50 at margin 0. Ladder (metal, os, cn) ->
      (metal, cn) -> (period-group, cn) -> (cn,). Categorical over SHAPES_BY_CN[cn] (plus any label seen), with a
      Jeffreys pseudo-count (0.5) per label.
* QP  partition prior     P(split pattern | arc pattern, metal)
      one row per arc = connected set of haptic face atoms of one ligand (the faces the B1.4 labeller chose, joined
      by ligand bonds). Arc pattern = (n atoms, context, ring size): context "ring" (the arc is a whole ring), "part"
      (all arc atoms on one ring of size rsize, e.g. COD alkenes, cycloheptatriene) or "chain"; split = face sizes
      along the arc ("4+2"), orientation / ring start chosen to give the smallest split string (then bond string). Bond orders are NOT in the
      key: tmQM draws many bound alkenes with a single C-C (metallacycle SMILES, or Wiberg ~1.2 after back-donation)
      while a user's input has C=C, so a bond-order key would differ between the DB and the solve. The bond string
      ('1' < 1.3, 'a' 1.3-1.7, '2' >= 1.7) is kept as a column for audits only. Ladder (arc, metal) ->
      (arc, period-group) -> (arc,) -> ((n, context),) -> ((n,),); an arc with nothing fitted gets a uniform prior
      over its splits in the hypothesis layer.
* QB  candidate donors    (insurance for the ligands-only mode, not fitted yet)
      one row per heavy ligand atom that could bind: every heteroatom, and every carbon with a multiple / aromatic
      bond or fewer than 4 neighbours; descriptors + bound flag (kappa donor or haptic face atom).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from jacobik.stats.mixture import structure_representative

QG_COLS = ["csd_code", "metal", "os", "cn", "geom", "geom_alt", "geom_gap"]
QP_COLS = ["csd_code", "metal", "n", "ctx", "rsize", "topo", "bo", "split", "n_faces"]
QB_COLS = ["csd_code", "metal", "elem", "heavy_deg", "n_h", "max_bo", "arom", "in_ring", "lig_heavy", "bound", "haptic"]

N_MIN_PRIOR = 20        # rows needed to fit a prior key
PSEUDO = 0.5            # Jeffreys pseudo-count per label
GAP_BAND_Q = 0.10       # soft band = this quantile of the CN's margins


# ---------------------------------------------------------------------------
# Rows (extraction)
# ---------------------------------------------------------------------------
def geometry_row(mol) -> Optional[dict]:
    from jacobik.topology.coord_geom import classify_with_margin

    cn = len(mol.coord_sites)
    if cn == 0:
        return None
    alt, gap = getattr(mol, "coord_geom_alt", None), getattr(mol, "coord_geom_gap", None)
    lab = mol.coord_geom
    if lab is None:
        lab, alt, gap = classify_with_margin(mol)
    return dict(csd_code=mol.csd_code, metal=mol.metal_symbol, os=mol.oxidation_state if mol.oxidation_state is not None else -99,
                cn=cn, geom=lab, geom_alt=alt, geom_gap=float(gap) if gap is not None else float("nan"))


def _bo_char(o: float) -> str:
    return "1" if o < 1.3 else ("a" if o < 1.7 else "2")


def _ligand_graph(mol):
    adj, bo = defaultdict(set), {}
    for b in mol.bonds:
        if mol.metal_idx in (b.i, b.j) or b.order < 0.5:
            continue
        adj[b.i].add(b.j)
        adj[b.j].add(b.i)
        bo[(min(b.i, b.j), max(b.i, b.j))] = b.order
    return adj, bo


def arc_pattern(atoms: Sequence[int], faces: Sequence[Sequence[int]], adj, bo) -> Tuple[int, str, str, str]:
    """(n, topo, bo string, split) for one arc (see module doc)."""
    A = set(atoms)
    face_of = {a: i for i, f in enumerate(faces) for a in f}
    sub = {a: [b for b in adj[a] if b in A] for a in A}
    deg = {a: len(v) for a, v in sub.items()}
    n = len(A)
    ob = lambda x, y: _bo_char(bo.get((min(x, y), max(x, y)), 1.0))

    def walk(start, nxt):
        order, prev, cur = [start], start, nxt
        while cur is not None and cur != start:
            order.append(cur)
            cand = [b for b in sub[cur] if b != prev and b not in order[1:]]
            prev, cur = cur, (cand[0] if cand else None)
        return order

    def encode(order, closed):
        m = len(order)
        edges = [ob(order[i], order[(i + 1) % m]) for i in range(m if closed else m - 1)]
        sizes, run, last = [], 0, None
        for a in order:
            f = face_of[a]
            if f != last and run:
                sizes.append(run)
                run = 0
            run += 1
            last = f
        sizes.append(run)
        if closed and len(sizes) > 1 and face_of[order[0]] == face_of[order[-1]]:
            sizes[0] += sizes.pop()  # the face that wraps around the start
        return "".join(edges), "+".join(str(s) for s in sizes)

    is_path = n >= 2 and all(d <= 2 for d in deg.values()) and sum(1 for d in deg.values() if d == 1) == 2
    is_cycle = n >= 3 and all(d == 2 for d in deg.values())
    if n == 1:
        return 1, "open", "", "1"
    if is_path:
        ends = [a for a, d in deg.items() if d == 1]
        best = None
        for e in ends:
            order = walk(e, sub[e][0])
            if len(order) != n:
                break
            code = encode(order, False)
            best = code if best is None or code[::-1] < best[::-1] else best
        if best is not None:
            return n, "open", best[0], best[1]
    if is_cycle:
        ring = walk(next(iter(A)), sorted(sub[next(iter(A))])[0])
        if len(ring) == n:
            best = None
            for rev in (False, True):
                r = ring[::-1] if rev else ring
                for k in range(n):
                    rot = r[k:] + r[:k]
                    if face_of[rot[0]] == face_of[rot[-1]] and len({face_of[a] for a in rot}) > 1:
                        continue  # start inside a face: that rotation splits it
                    code = encode(rot, True)
                    best = code if best is None or code[::-1] < best[::-1] else best
            if best is not None:
                return n, "ring", best[0], best[1]
    return n, "other", "", "+".join(str(len(f)) for f in sorted(faces, key=len, reverse=True))


def partition_rows(mol) -> List[dict]:
    adj, bo = _ligand_graph(mol)
    out = []
    for lig in mol.ligands:
        faces = [list(h.donor_indices) for h in (getattr(lig, "haptic_groups", None) or [])]
        if not faces:
            continue
        atoms = sorted({a for f in faces for a in f})
        # arcs = connected components of the face atoms over ligand bonds
        seen, comps = set(), []
        for a in atoms:
            if a in seen:
                continue
            stack, comp = [a], []
            seen.add(a)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v in atoms and v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)
        sym = {x.idx: x.symbol for x in mol.atoms}
        heavy = [a for a in lig.atom_indices if sym.get(a) != "H"]
        rings = _ligand_rings(heavy, adj)
        for comp in comps:
            cf = [f for f in faces if set(f) <= set(comp)]
            n, topo, bos, split = arc_pattern(comp, cf, adj, bo)
            ctx, rsize = arc_context(comp, rings)
            out.append(dict(csd_code=mol.csd_code, metal=mol.metal_symbol, n=n, ctx=ctx, rsize=rsize, topo=topo,
                            bo=bos, split=split, n_faces=len(cf)))
    return out


HETERO = frozenset({"N", "O", "P", "S", "As", "Se", "Te", "F", "Cl", "Br", "I", "B", "Si", "Ge", "Sn", "Sb", "Bi", "Al", "Ga"})


def candidate_rows(mol) -> List[dict]:
    adj, bo = _ligand_graph(mol)
    sym = {a.idx: a.symbol for a in mol.atoms}
    out = []
    for lig in mol.ligands:
        bound = set(lig.donor_atoms)
        hapt = {a for h in (getattr(lig, "haptic_groups", None) or []) for a in h.donor_indices}
        heavy = [a for a in lig.atom_indices if sym.get(a) != "H"]
        ring = _ring_atoms(heavy, adj)
        for a in heavy:
            e = sym[a]
            nb = adj[a]
            hd = sum(1 for b in nb if sym.get(b) != "H")
            nh = sum(1 for b in nb if sym.get(b) == "H")
            orders = [bo.get((min(a, b), max(a, b)), 1.0) for b in nb]
            mbo = max(orders) if orders else 0.0
            arom = any(1.3 <= o < 1.7 for o in orders)
            if e == "C" and not (mbo >= 1.3 or len(nb) < 4):
                continue  # saturated carbon: never a donor candidate
            if e != "C" and e not in HETERO:
                continue
            out.append(dict(csd_code=mol.csd_code, metal=mol.metal_symbol, elem=e, heavy_deg=hd, n_h=nh,
                            max_bo=round(float(mbo), 2), arom=bool(arom), in_ring=a in ring, lig_heavy=len(heavy),
                            bound=a in bound or a in hapt, haptic=a in hapt))
    return out


def _ligand_rings(atoms, adj) -> List[set]:
    """Smallest set of smallest rings of the heavy-atom subgraph (as atom sets)."""
    import networkx as nx

    G = nx.Graph()
    A = set(atoms)
    for a in A:
        for b in adj[a]:
            if b in A:
                G.add_edge(a, b)
    try:
        return [set(c) for c in nx.minimum_cycle_basis(G)]
    except Exception:
        return [set(c) for c in nx.cycle_basis(G)]


def arc_context(comp, rings) -> Tuple[str, int]:
    """("ring", n) if the arc is a whole ring, ("part", r) if all its atoms lie on one ring of size r (smallest),
    else ("chain", 0)."""
    C = set(comp)
    hold = [r for r in rings if C <= r]
    if not hold:
        return "chain", 0
    r = min(hold, key=len)
    return ("ring" if len(r) == len(C) else "part"), len(r)


def _ring_atoms(atoms, adj) -> set:
    """Atoms on a cycle of the heavy-atom subgraph (bridges removed = not in a ring)."""
    import networkx as nx

    G = nx.Graph()
    A = set(atoms)
    for a in A:
        for b in adj[a]:
            if b in A:
                G.add_edge(a, b)
    ring = set()
    for c in nx.cycle_basis(G):
        ring |= set(c)
    return ring


# ---------------------------------------------------------------------------
# Fits (refit)
# ---------------------------------------------------------------------------
def _cat_entry(counts: Dict[str, float], labels: Sequence[str], n: int) -> dict:
    labs = sorted(set(labels) | set(counts))
    tot = sum(counts.values()) + PSEUDO * len(labs)
    return {"type": "cat", "n": int(n), "labels": labs, "p": [(counts.get(l, 0.0) + PSEUDO) / tot for l in labs]}


def gap_bands(qg) -> Dict[int, float]:
    """Soft band per CN: GAP_BAND_Q quantile of the margins (data-derived)."""
    out = {}
    for cn, g in qg.dropna(subset=["geom_gap"]).groupby("cn"):
        if len(g) >= N_MIN_PRIOR:
            out[int(cn)] = float(np.quantile(g["geom_gap"].to_numpy(), GAP_BAND_Q))
    return out


def fit_geometry_prior(qg, n_min: int = N_MIN_PRIOR) -> Dict[str, dict]:
    """{"QG:<key>": categorical} over the ladder in the module doc (soft counts near the label boundary)."""
    from jacobik.stats.ladders import metal_period_group
    from jacobik.topology.coord_geom import SHAPES_BY_CN

    bands = gap_bands(qg)
    df = qg.copy()
    band = df["cn"].map(lambda c: bands.get(int(c), 0.0)).to_numpy()
    gap = df["geom_gap"].to_numpy(float)
    has_alt = df["geom_alt"].notna().to_numpy() & np.isfinite(gap) & (band > 0)
    w_main = np.where(has_alt, 0.5 + 0.5 * np.clip(np.nan_to_num(gap) / np.where(band > 0, band, 1.0), 0.0, 1.0), 1.0)
    df["w_main"] = w_main
    df["pg"] = df["metal"].map(metal_period_group)
    df["_rep"] = structure_representative(df["csd_code"]) if "csd_code" in df.columns else True
    out = {}
    for cols in (["metal", "os", "cn"], ["metal", "cn"], ["pg", "cn"], ["cn"]):
        # metal-free levels: one LnQM sibling per ligand set (mixture.structure_representative)
        for key, g in (df if "metal" in cols else df[df["_rep"]]).groupby(cols, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            if len(g) < n_min:
                continue
            counts: Dict[str, float] = defaultdict(float)
            for lab, alt, w in zip(g["geom"], g["geom_alt"], g["w_main"]):
                counts[lab] += w
                if w < 1.0 and isinstance(alt, str):
                    counts[alt] += 1.0 - w
            cn = int(key[-1])
            e = _cat_entry(counts, SHAPES_BY_CN.get(cn, ()), len(g))
            e["level"] = "+".join(cols)
            out[f"QG:{tuple(_py(k) for k in key)!r}"] = e
    return out


def fit_partition_prior(qp, n_min: int = N_MIN_PRIOR) -> Dict[str, dict]:
    from jacobik.stats.ladders import metal_period_group

    df = qp.copy()
    df["pg"] = df["metal"].map(metal_period_group)
    df["_rep"] = structure_representative(df["csd_code"]) if "csd_code" in df.columns else True
    out = {}
    for cols in (["n", "ctx", "rsize", "metal"], ["n", "ctx", "rsize", "pg"], ["n", "ctx", "rsize"], ["n", "ctx"], ["n"]):
        for key, g in (df if "metal" in cols else df[df["_rep"]]).groupby(cols, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            if len(g) < n_min:
                continue
            counts = g["split"].value_counts().to_dict()
            e = _cat_entry({str(k): float(v) for k, v in counts.items()}, [], len(g))
            e["level"] = "+".join(cols)
            out[f"QP:{tuple(_py(k) for k in key)!r}"] = e
    return out


def _py(x):
    return x.item() if hasattr(x, "item") else x
