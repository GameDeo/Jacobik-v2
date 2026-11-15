"""A4t v2 metrics (analysis only, no solves): coverage-package Part A, RC_DIAGNOSIS section 5.2 (a)-(f).

The frozen v1 metrics (`metrics.py`) are untouched.  Everything here works on coordinates in REFERENCE atom order
(`to_ref_order`), a heavy-atom automorphism array `P` (`runkit.geom.automorphisms`, identity first) and a covalent
adjacency from the reference geometry.

(a) `rmsd_proper_mirror`     hit RMSD with and without global inversion
(b) `verdict`                equivalence on MACE and GFN2 energies; disagreement across the threshold = ambiguous (A)
(c) `ring_atoms`, `deviation_shares`, `ring_conformer_diff`, `split_residual`
                             saturated-ring conformer / chelate-macrocycle pucker / aryl-pendant rotor / residual C
(d) `shell_rmsd_auto`        shell isomer test under graph automorphisms (no free donor permutation)
(e) `stereocentres`, `chirality_flips`, `chelate_planarity_flag`
                             signed volumes at P/N/C centres, planar chelate rings that fold
"""
from __future__ import annotations

import math

import numpy as np

from tools.runkit import geom

H2KJ = 2625.4996
HIT_RADIUS = 0.3
EQ_KJ = 5.0
ROTOR_RMS = 0.3          # rms deviation (A) of aryl+pendant atoms that makes a rotor miss
RING_TORS_DEG = 30.0     # internal ring torsion difference that counts as a different ring conformer
PLANAR_REF_DEG = 20.0    # reference chelate ring is planar if every torsion is within this
FOLD_ROW_DEG = 40.0      # ... and the row ring is folded if any torsion exceeds this
VOL_MIN = 0.3            # |signed volume| (A^3) of a reference centre below which it is planar, not a stereocentre

CLASSES2 = ("F", "A", "T", "G", "G*", "S", "Z", "R-sat", "R-chel", "R-rot", "C")


# ---------------------------------------------------------------------------------------------------------------- (a)
def _kabsch_ss(A, B):
    """Minimal sum of squared deviations over (k, n, 3) arrays B (already permuted) against A (n, 3).
    proper = det-fixed; returns (ss_proper, ss_mirror) each (k,)."""
    Ac = A - A.mean(0)
    Bc = B - B.mean(1, keepdims=True)
    H = np.einsum("ni,knj->kij", Ac, Bc)
    u, s, vt = np.linalg.svd(H)
    det = np.sign(np.linalg.det(u @ vt))
    tot = (Ac ** 2).sum() + (Bc ** 2).sum((1, 2))
    s_mirror = s.sum(1)
    s_proper = s_mirror - 2.0 * s[:, -1] * (det < 0)
    return tot - 2.0 * s_proper, tot - 2.0 * s_mirror


def perm_union(P):
    """P together with the inverse permutations.  A complete automorphism group is closed under inversion, but a group
    truncated at `geom.CAP` is not, and the minimum over `b[P]` and over `b[P^-1]` can differ a lot (SILROY: 0.42 vs
    2.41 A).  The union is the better upper bound of the true symmetry-aware RMSD."""
    P = np.asarray(P, int)
    return np.unique(np.vstack([P, geom._inv_perms(P)]), axis=0)


def rmsd_proper_mirror(Xh, Rh, P, graph=None):
    """(rmsd_proper, rmsd_mirror) of one or many heavy-atom coordinate sets Xh (r, n, 3) against Rh (n, 3), minimised over
    the automorphisms P and rigid motion (exact over P; NOT `geom.sym_rmsd_to_ref`, whose capped-parent Hungarian
    refinement ignores elements).  'mirror' is the proper value of the reflected rows (global inversion); the hit rule
    uses min(proper, mirror)."""
    Xh = np.asarray(Xh, float)
    n = Xh.shape[1]
    b = np.asarray(Rh, float) - np.asarray(Rh, float).mean(0)
    bb = float((b ** 2).sum())
    Pu = perm_union(P)
    out = []
    for X in (Xh, Xh * np.array([1.0, 1.0, -1.0])):
        J = X - X.mean(1, keepdims=True)
        ss = geom._min_ss(J, (J ** 2).sum((1, 2)), b, bb, Pu, None)
        res = np.sqrt(np.clip(ss, 0, None) / n)
        if graph is not None and len(P) >= geom.CAP:      # capped group: guarded refinement (within classes, automorphisms only)
            for i in range(len(res)):
                if res[i] > 0.5:
                    res[i] = geom._refine_capped(J[i], b, graph, float(res[i]))
        out.append(res)
    return out[0], out[1]


def best_alignment(rowX, refX, heavy, P):
    """Best (automorphism, proper-or-mirror) alignment of one row onto the reference.

    Returns dict(rmsd, mirror (bool), p (index into heavy: row heavy x <-> ref heavy p[x]), atom_map {row atom: ref atom},
    Xr_ref (row coordinates, reflected when mirror, arranged in reference atom order), delta (per ref-heavy aligned
    deviation vectors, shape (n_heavy, 3), ref-heavy order))."""
    heavy = np.asarray(heavy, int)
    A0 = np.asarray(rowX, float)[heavy]
    B = np.asarray(refX, float)[heavy]
    Pu = perm_union(P)
    Bp = B[Pu]
    best = None
    for mirror in (False, True):
        A = A0 * np.array([1.0, 1.0, -1.0]) if mirror else A0
        ss_p, _ = _kabsch_ss(A, Bp)
        k = int(np.argmin(ss_p))
        if best is None or ss_p[k] < best[0]:
            best = (float(ss_p[k]), k, mirror, A)
    ss, k, mirror, A = best
    p = np.asarray(Pu[k], int)
    bb = B[p]
    Ac = A - A.mean(0)
    u, _, vt = np.linalg.svd(Ac.T @ (bb - bb.mean(0)))
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(u @ vt)
    R = u @ fix @ vt
    delta_row = Ac @ R - (bb - bb.mean(0))            # per row-heavy x, partner ref heavy p[x]
    n = len(heavy)
    delta = np.zeros((n, 3))
    delta[p] = delta_row                                # re-indexed to ref-heavy order
    Xr = np.array(np.asarray(rowX, float))
    if mirror:
        Xr = Xr * np.array([1.0, 1.0, -1.0])
    Xr_ref = Xr.copy()
    for x, a in enumerate(heavy):
        Xr_ref[heavy[p[x]]] = Xr[a]
    return dict(rmsd=float(math.sqrt(max(ss, 0.0) / n)), mirror=bool(mirror), p=p,
                atom_map={int(heavy[x]): int(heavy[p[x]]) for x in range(n)}, Xr_ref=Xr_ref, delta=delta)


# ---------------------------------------------------------------------------------------------------------------- (b)
def verdict(rows, eq_kj=EQ_KJ):
    """rows: dicts with hit (bool, global-inversion rule), dE_mace, dE_gfn2 (kJ/mol vs reference, None if missing).
    Returns dict(F_mace, F_gfn2, status) with status 'F' (both), 'A' (ambiguous: verdicts disagree) or 'miss' (both)."""
    def ok(r, key):
        v = r.get(key)
        return bool(r.get("hit")) or (v is not None and math.isfinite(float(v)) and float(v) <= eq_kj)
    fm = any(ok(r, "dE_mace") for r in rows)
    fg = any(ok(r, "dE_gfn2") for r in rows)
    return dict(F_mace=fm, F_gfn2=fg, status="F" if (fm and fg) else ("A" if (fm or fg) else "miss"))


# ---------------------------------------------------------------------------------------------------------------- (c)
def torsion(X, a, b, c, d):
    b0 = X[a] - X[b]
    b1 = X[c] - X[b]
    b2 = X[d] - X[c]
    n1 = b1 / np.linalg.norm(b1)
    v = b0 - b0 @ n1 * n1
    w = b2 - b2 @ n1 * n1
    return float(np.degrees(np.arctan2(np.cross(n1, v) @ w, v @ w)))


def wrap_deg(x):
    return abs((x + 180.0) % 360.0 - 180.0)


def ring_torsions(X, order):
    n = len(order)
    return [torsion(X, *[order[(k + j) % n] for j in range(4)]) for k in range(n)]


_SAT_DEGREE = {"C": 4, "Si": 4, "N": 3, "P": 3, "O": 2, "S": 2}


def ring_atoms(refX, symbols, metal, donors, pad=0.4, min_size=5):
    """Ring perception on the reference.  Returns dict with
    sat   : list of ordered saturated non-aromatic organic rings (all sp3-like atoms, non-planar), size >= min_size
    chel  : ordered rings through the metal (chelate / macrocycle), size >= min_size
    aryl  : set of atoms in other (planar / unsaturated) organic rings
    adj   : full covalent adjacency (hydrogens included; metal row empty)."""
    import networkx as nx
    n = len(symbols)
    adj = geom.ligand_adjacency(np.asarray(refX, float), symbols, metal, pad)
    heavy = [i for i in range(n) if str(symbols[i]) != "H"]
    G = nx.Graph()
    G.add_nodes_from(heavy)
    for x, i in enumerate(heavy):
        for j in heavy[x + 1:]:
            if adj[i, j]:
                G.add_edge(i, j)
    for d in donors:
        G.add_edge(int(metal), int(d))
    sat, chel, aryl = [], [], set()
    for cyc in nx.minimum_cycle_basis(G):
        if len(cyc) < min_size:
            continue
        try:
            order = [u for u, _ in nx.find_cycle(G.subgraph(cyc))]
        except Exception:
            continue
        if len(order) != len(cyc):
            continue
        if metal in cyc:
            chel.append(order)
            continue
        Y = np.asarray(refX, float)[order]
        Y = Y - Y.mean(0)
        oop = float(np.linalg.svd(Y, compute_uv=False)[-1] / math.sqrt(len(order)))
        dset = set(map(int, donors))
        saturated = all(int(adj[a].sum()) + (a in dset) >= _SAT_DEGREE.get(str(symbols[a]), 99) for a in order)
        if saturated and oop >= 0.1:
            sat.append(order)
        else:
            aryl.update(order)
    return dict(sat=sat, chel=chel, aryl=aryl, adj=adj)


def atom_categories(symbols, metal, donors, rings):
    """Disjoint atom classes (priority shell > chelate > saturated ring > aryl > pendant) for heavy atoms."""
    cat = {}
    chel_atoms = {a for o in rings["chel"] for a in o}
    sat_atoms = {a for o in rings["sat"] for a in o}
    for a, s in enumerate(symbols):
        if str(s) == "H":
            continue
        if a == metal or a in set(map(int, donors)):
            cat[a] = "shell"
        elif a in chel_atoms:
            cat[a] = "chelate"
        elif a in sat_atoms:
            cat[a] = "sat_ring"
        elif a in rings["aryl"]:
            cat[a] = "aryl"
        else:
            cat[a] = "pendant"
    return cat


def deviation_shares(delta, heavy, cat):
    """Per-category share of the squared aligned deviation, atom fraction, enrichment (share / fraction) and rms (A).
    `delta` is in ref-heavy order, `cat` maps atom index -> category."""
    dev2 = (np.asarray(delta, float) ** 2).sum(1)
    tot = float(dev2.sum()) or 1.0
    out = {}
    n = len(heavy)
    for c in ("shell", "chelate", "sat_ring", "aryl", "pendant"):
        idx = [x for x, a in enumerate(heavy) if cat.get(int(a)) == c]
        if not idx:
            out[c] = dict(n=0, share=0.0, frac=0.0, enrich=0.0, rms=0.0)
            continue
        share = float(dev2[idx].sum() / tot)
        frac = len(idx) / n
        out[c] = dict(n=len(idx), share=share, frac=frac, enrich=share / frac, rms=float(math.sqrt(dev2[idx].mean())))
    return out


def ring_conformer_diff(refX, rowX_ref, orders):
    """Per ring: maximal wrapped difference (deg) of the internal ring torsions between reference and row (row
    coordinates in reference atom order, reflected when the alignment is a mirror)."""
    out = []
    for o in orders:
        tr = ring_torsions(refX, o)
        tw = ring_torsions(rowX_ref, o)
        out.append(max(wrap_deg(a - b) for a, b in zip(tr, tw)))
    return out


def split_residual(sat_diff, chel_diff, shares, tors_deg=RING_TORS_DEG, rotor_rms=ROTOR_RMS):
    """Label for a parent that is a miss and passed the T/G/S/Z checks.  Priority: a different saturated-ring conformer
    (embed ring), a different chelate / macrocycle conformer, a rotor miss (aryl + pendant atoms deviate by rotor_rms or
    more), else the residual 'C' (nothing identified)."""
    if sat_diff and max(sat_diff) >= tors_deg:
        return "R-sat"
    if chel_diff and max(chel_diff) >= tors_deg:
        return "R-chel"
    rot = [shares[c] for c in ("aryl", "pendant") if shares[c]["n"]]
    if rot:
        n = sum(s["n"] for s in rot)
        rms = math.sqrt(sum(s["rms"] ** 2 * s["n"] for s in rot) / n)
        if rms >= rotor_rms:
            return "R-rot"
    return "C"


# ---------------------------------------------------------------------------------------------------------------- (d)
def shell_rmsd_auto(rowX, refX, P, heavy, shell, mirror=False):
    """Metal+donor RMSD under graph automorphisms only.  `shell` are atom indices (metal first); P are the heavy-atom
    automorphisms.  mirror=True allows an improper alignment.  Unique shell permutations are taken from P."""
    heavy = list(map(int, heavy))
    pos = {a: x for x, a in enumerate(heavy)}
    ix = [pos[int(a)] for a in shell]
    Pm = np.unique(perm_union(P)[:, ix], axis=0)           # shell images under the automorphisms
    A = np.asarray(rowX, float)[list(map(int, shell))]
    B = np.asarray(refX, float)[heavy][Pm]                      # (k, m, 3): ref atoms matched to the row's shell slots
    ss_p, ss_m = _kabsch_ss(A, B)
    ss = ss_m if mirror else ss_p
    return float(math.sqrt(max(float(ss.min()), 0.0) / len(A)))


def shell_rmsd_auto_any(rowX, refX, P, heavy, shell):
    """(proper, mirror) shell RMSDs under automorphisms."""
    return (shell_rmsd_auto(rowX, refX, P, heavy, shell, False), shell_rmsd_auto(rowX, refX, P, heavy, shell, True))


# ---------------------------------------------------------------------------------------------------------------- (e)
def wl_colours(adj_full, symbols, metal, donors, rounds=8):
    """Colour refinement of the molecular graph (hydrogens included, metal flagged, metal-donor links included).
    Equal colours = candidate symmetry-equivalent atoms (necessary condition; no automorphism cap involved)."""
    n = len(symbols)
    nb = [set(np.nonzero(adj_full[i])[0].tolist()) for i in range(n)]
    for d in donors:
        nb[int(metal)].add(int(d))
        nb[int(d)].add(int(metal))
    col = [(str(symbols[i]), i == metal) for i in range(n)]
    ids = {c: k for k, c in enumerate(sorted(set(col), key=str))}
    col = [ids[c] for c in col]
    for _ in range(rounds):
        sig = [(col[i], tuple(sorted(col[j] for j in nb[i]))) for i in range(n)]
        ids = {c: k for k, c in enumerate(sorted(set(sig), key=str))}
        new = [ids[s] for s in sig]
        if len(set(new)) == len(set(col)):
            col = new
            break
        col = new
    return col, nb


def stereocentres(refX, symbols, metal, donors, adj_full, vol_min=VOL_MIN):
    """Pyramidal C / N / P / Si / As centres of the reference whose (heavy or metal) neighbours are pairwise
    inequivalent.  N counts only with four neighbours (quaternary or metal-bound donor); C needs a tetrahedral
    environment with at least three heavy neighbours; P/As/Si need three.  Returns {centre: (n1, n2, n3)} (the three
    lowest-index neighbours, whose signed volume defines the sense)."""
    col, nb = wl_colours(adj_full, symbols, metal, donors)
    X = np.asarray(refX, float)
    out = {}
    for a, s in enumerate(symbols):
        s = str(s)
        if a == metal or s not in ("C", "N", "P", "As", "Si"):
            continue
        nbrs = sorted(nb[a])
        heavy_n = [j for j in nbrs if str(symbols[j]) != "H"]
        if len(heavy_n) < 3:
            continue
        if s == "N" and len(nbrs) < 4:
            continue
        if s == "C" and len(nbrs) != 4:
            continue
        if len({col[j] for j in heavy_n}) != len(heavy_n):
            continue
        t = tuple(heavy_n[:3])
        if abs(signed_volume(X, a, t)) >= vol_min:
            out[a] = t
    return out


def signed_volume(X, c, nbrs):
    return float(np.linalg.det(np.stack([X[j] - X[c] for j in nbrs])))


def chirality_flips(refX, rowX_ref, centres):
    """Centres whose sign differs between reference and row (row in reference atom order).  Returns
    (flipped, n_centres, n_diastereo) where n_diastereo = min(#flipped, n - #flipped): a global inversion (every centre
    flipped) is the enantiomer and does not count."""
    flipped = [c for c, t in centres.items()
               if np.sign(signed_volume(refX, c, t)) != np.sign(signed_volume(rowX_ref, c, t))]
    n = len(centres)
    return flipped, n, min(len(flipped), n - len(flipped))


def chelate_planarity_flag(refX, rowX_ref, chel_rings, ref_deg=PLANAR_REF_DEG, row_deg=FOLD_ROW_DEG):
    """Planar reference chelate rings (all torsions within ref_deg) that fold in the row (any torsion beyond row_deg).
    Returns the list of folded rings as (ring, ref_max, row_max)."""
    out = []
    for o in chel_rings:
        tr = max(abs(t) for t in ring_torsions(refX, o))
        if tr > ref_deg:
            continue
        tw = max(abs(t) for t in ring_torsions(rowX_ref, o))
        if tw > row_deg:
            out.append((list(o), tr, tw))
    return out


# ---------------------------------------------------------------------------------------------------------------- order
def decision2(status, donor_mismatch, geometry_mismatch, any_shell, z_flag, residual, code=None, known_typing=(),
              geometry_evaluable=True):
    """Ordered list.  status: 'F' / 'A' / 'miss' from `verdict`.  Returns (combined class, class if the parent is a miss).
    F and A are decided first; 'A' parents also carry the class they would have as a miss."""
    if donor_mismatch or code in known_typing:
        miss = "T"
    elif not geometry_evaluable:
        miss = "G*"
    elif geometry_mismatch:
        miss = "G"
    elif not any_shell:
        miss = "S"
    elif z_flag:
        miss = "Z"
    else:
        miss = residual
    return ("F" if status == "F" else ("A" if status == "A" else miss)), miss
