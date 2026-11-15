"""Control constructs on real geometries (A4c): rotate a peripheral rotor (same class) or invert one stereocentre
(different class). Both keep bond lengths; GFN2 relaxation afterwards is done by the caller."""
from __future__ import annotations

import numpy as np


def _side(full_adj, a, b):
    """Atoms reachable from b without crossing the a-b bond (full index space, metal row empty)."""
    seen, stack = {b}, [b]
    while stack:
        x = stack.pop()
        for y in np.flatnonzero(full_adj[x]):
            y = int(y)
            if y != a and y not in seen:
                seen.add(y); stack.append(y)
    return seen


def peripheral_rotors(ref):
    """Single bonds a-b between heavy atoms, not in a ring, b's side outside the core, holding >= 2 heavy atoms and
    no donor; returned as (a, b, side) in FULL atom indices."""
    core_full = {int(ref.heavy[i]) for i in ref.core}
    donors_full = {int(ref.heavy[i]) for i in ref.donors}
    out = []
    A = ref.full_adj
    for i in range(len(ref.heavy)):
        for j in np.flatnonzero(ref.adj[i]):
            a, b = int(ref.heavy[i]), int(ref.heavy[int(j)])
            side = _side(A, a, b)
            if a in side or ref.metal in side or side & donors_full or b in core_full:
                continue
            heavy_side = [x for x in side if ref.symbols[x] != "H"]
            if len(heavy_side) < 2 or side & core_full:
                continue
            out.append((a, b, sorted(side)))
    return out


def rotate(X, a, b, side, degrees):
    X = np.array(X, float)
    k = X[b] - X[a]; k /= np.linalg.norm(k)
    t = np.radians(degrees)
    V = X[side] - X[b]
    V = V * np.cos(t) + np.cross(k, V) * np.sin(t) + np.outer(V @ k, k) * (1 - np.cos(t))
    X[side] = V + X[b]
    return X


def clash_free(X, symbols, moved, full_adj, pad=0.6):
    """No moved atom within r_i + r_j + pad of an unmoved atom it is not bonded to (pad above the 0.4 A bond rule)."""
    from tools.runkit.geom import rcov
    r = np.array([rcov(s) for s in symbols])
    moved = np.asarray(sorted(moved))
    rest = np.setdiff1d(np.arange(len(X)), moved)
    d = np.linalg.norm(X[moved][:, None] - X[rest][None], axis=-1)
    bad = (d < r[moved][:, None] + r[rest][None] + pad) & ~full_adj[np.ix_(moved, rest)]
    return not bad.any()


def rotor_variant(ref, X, n_rotors=3, angles=(120.0, 240.0, 90.0, 180.0, 60.0, 300.0)):
    """Rotate up to n_rotors peripheral rotors (largest sides first). A rotation is kept only if it is clash-free and
    the construct is still the reference class with the same graph and contacts. Returns (X_new, applied list)."""
    rotors = sorted(peripheral_rotors(ref), key=lambda r: -len(r[2]))
    Y = np.array(X, float); applied = []; used = set()
    for a, b, side in rotors:
        if len(applied) >= n_rotors or (set(side) | {a}) & used:
            continue
        for ang in angles:
            Z = rotate(Y, a, b, side, ang)
            if clash_free(Z, ref.symbols, side, ref.full_adj) and ref.compare(Z)["same_class"]:
                Y = Z; applied.append((a, b, ang, len(side))); used |= set(side) | {a, b}
                break
    return Y, applied


def invertible_centres(ref):
    """Non-donor 'chi' centres whose fourth neighbour is a single H (inversion = mirror the centre and that H through
    the plane of the three heavy neighbours; bond lengths to the heavy neighbours are kept)."""
    out = []
    for kind, groups in ref.desc:
        if kind != "chi":
            continue
        c = groups[0][0]
        if c in ref.donors or int(ref.nH[c]) != 1 or len(ref._neighbours(c)) != 3:
            continue
        out.append(c)
    return out


def invert_centre(ref, X, c):
    X = np.array(X, float)
    cf = int(ref.heavy[c])
    nb = [int(ref.heavy[x]) for x in ref._neighbours(c)]
    hs = [int(h) for h in np.flatnonzero(ref.full_adj[cf]) if ref.symbols[int(h)] == "H"]
    p0, p1, p2 = X[nb[0]], X[nb[1]], X[nb[2]]
    n = np.cross(p1 - p0, p2 - p0); n /= np.linalg.norm(n)
    for i in [cf] + hs:
        X[i] = X[i] - 2 * ((X[i] - p0) @ n) * n
    return X
