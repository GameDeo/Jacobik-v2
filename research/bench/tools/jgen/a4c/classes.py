"""Isomer-class descriptors and core RMSD for the class-aware A4b hit (A4c).

All coordinates are in REFERENCE atom order (the A4b relax records and `gen_to_ref_idx`-mapped pool rows are).
The class of a geometry, relative to a reference, is decided by descriptors on reference-labelled atom tuples:

* donor set   - heavy atoms within r_cov(M) + r_cov(d) + 0.6 A of the metal (`runkit.geom.donors_of`, the A4b contact rule).
  A contact CHANGE needs a crossing of that cutoff AND a change of the M-atom distance by more than 0.3 A against the
  reference (contacts within 0.02 A of the cutoff flipped on relaxation noise otherwise: SESWIZ, YISQUM, MEJYIN);
* site arrangement - sites are donor groups (eta groups -> centroid). The labelled unit site vectors of the row are
  Kabsch-aligned onto the reference's (proper rotation; improper for the mirror) and the nearest-site (Hungarian)
  assignment must be the identity. Continuous polyhedral distortions (square planar <-> tetrahedral twists, Berry/
  trigonal twists short of a site swap) keep the identity; a different arrangement (cis/trans, fac/mer, Delta/Lambda,
  any site permutation not undone by an automorphism) cannot. (Angle bands for cis/trans were tried first and
  over-split soft 4-coordinate complexes, AWONOR.)
* centres     - sign of the triple product of three heavy neighbours (metal included) seen from the centre, for every
  4-connected atom and every 3-connected P/As/S/Se (pyramidal, slow inversion); free 3-connected N is excluded (fast
  inversion). Binding-created centres (donor N/P/S/O with the metal as a neighbour) are 4-connected and so included;
  a bound neutral thioether/selenoether (R2S->M) is not a centre: it inverts at 298 K (same rule as production
  `stereo.perceive_binding_created_donors`); bound sulfoxide/sulfonium S is;
  fused/bridged ring configurations are carried by their bridgehead centres;
* double bonds - sign of cos(dihedral) across short (C=C < 1.42, C=N/N=N < 1.32 A) bonds between trigonal atoms, outside
  rings of <= 7 atoms (E/Z).

Centre and E/Z values are ternary with a dead band (centres: sign only, a flattened centre is unknown). Two geometries are the same
class when some graph automorphism p of the reference (heavy-atom VF2, `runkit.geom.automorphisms`) maps the reference
donor set onto the row's donor set and makes every descriptor agree (unknown agrees with anything). The global mirror is
tried (chirality-type descriptors all flip together): a mirror-only match is the ENANTIOMER, a distinct class (owner
2026-11-12), reported as `enantiomer` and never a hit. Core RMSD is the min over the matching automorphisms only (mirror matches against the mirrored row).
Thermally flipping rings are not descriptors (ring C atoms with two H carry no centre).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from tools.runkit import geom

CHI_HI, CHI_LO = 0.30, 0.10
EZ_HI, EZ_LO = 0.50, 0.20
DOUBLE_MAX = {frozenset("C"): 1.42, frozenset("CN"): 1.32, frozenset("N"): 1.32}
PYRAMIDAL_3 = {"P", "As", "S", "Se"}
CHALCOGEN = {"S", "Se", "Te"}
CORE_DEPTH = 2
GUIDED_ROUNDS = 4
CONTACT_SHIFT_A = 0.3


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-9, n, 1.0)


def ternary(v, hi, lo):
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    out[v > hi] = 1.0
    out[v < -hi] = -1.0
    out[np.abs(v) < lo] = 0.0
    return out


def _bfs(adj, start, limit=None, skip_edge=None):
    dist = {start: 0}
    q = deque([start])
    while q:
        a = q.popleft()
        if limit is not None and dist[a] >= limit:
            continue
        for b in np.flatnonzero(adj[a]):
            b = int(b)
            if skip_edge and {a, b} == set(skip_edge):
                continue
            if b not in dist:
                dist[b] = dist[a] + 1
                q.append(b)
    return dist


def _path(adj, a, b):
    prev = {a: None}
    q = deque([a])
    while q:
        x = q.popleft()
        if x == b:
            break
        for y in np.flatnonzero(adj[x]):
            y = int(y)
            if y not in prev:
                prev[y] = x
                q.append(y)
    if b not in prev:
        return []
    out = [b]
    while prev[out[-1]] is not None:
        out.append(prev[out[-1]])
    return out


def wl_colours(adj, labels, rounds=None):
    """Weisfeiler-Lehman colour refinement (colour classes contain the automorphism orbits)."""
    col = list(labels)
    for _ in range(rounds or len(col)):
        sig = [(col[i], tuple(sorted(col[j] for j in np.flatnonzero(adj[i])))) for i in range(len(col))]
        idx = {s: k for k, s in enumerate(sorted(set(sig), key=repr))}
        new = [idx[s] for s in sig]
        if len(set(new)) == len(set(col)):
            col = new
            break
        col = new
    return np.asarray(col)


class Reference:
    """Descriptors of one reference. `graph_X` builds the bond graph and automorphisms (the frozen reference geometry);
    `X` is the geometry whose class is the target (the relaxed reference for relaxed rows, `graph_X` for unrelaxed)."""

    def __init__(self, symbols, X, graph_X=None, perms=None, capped=None, core_depth=CORE_DEPTH):
        self.symbols = [str(s) for s in symbols]
        X = np.asarray(X, float)
        G = np.asarray(graph_X if graph_X is not None else X, float)
        self.graph_X = G
        self.metal = geom.find_metal(self.symbols)
        self.heavy = np.array([i for i, s in enumerate(self.symbols) if s != "H"], int)
        self.hpos = {int(a): x for x, a in enumerate(self.heavy)}
        self.m = self.hpos[self.metal]
        full = geom.ligand_adjacency(G, self.symbols, self.metal)
        self.full_adj = full
        self.adj = full[np.ix_(self.heavy, self.heavy)]
        self.nH = full[:, [i for i, s in enumerate(self.symbols) if s == "H"]].sum(1)[self.heavy]
        if perms is None:
            perms, capped = geom.automorphisms(G, self.symbols, self.metal)
        self.perms, self.capped = np.asarray(perms, int), bool(capped)
        self.colours = wl_colours(self.adj, [self.symbols[i] for i in self.heavy])
        self.Xh = X[self.heavy]
        self.donors = self._donors(X)
        self.cut = np.array([geom.rcov(self.symbols[self.metal]) + geom.rcov(self.symbols[i]) + 0.6 for i in self.heavy])
        self.dref = np.linalg.norm(self.Xh - self.Xh[self.m], axis=1)
        self.sites = self._sites(self.donors)
        self.core = self._core(core_depth)
        self.desc = self._descriptors()
        self.t_ref = self.values(self.Xh[None], np.arange(len(self.heavy))[None])[0]
        keep = ~np.isnan(self.t_ref)
        self.desc = [d for d, k in zip(self.desc, keep) if k]
        self.t_ref = self.t_ref[keep]

    # -- perception ---------------------------------------------------------------------------------------------
    def _donors(self, X):
        return frozenset(self.hpos[int(i)] for i in geom.donors_of(X, self.symbols, self.metal))

    def _sites(self, donors):
        left, out = set(donors), []
        while left:
            a = left.pop(); grp = {a}; q = [a]
            while q:
                x = q.pop()
                for y in np.flatnonzero(self.adj[x]):
                    y = int(y)
                    if y in left:
                        left.discard(y); grp.add(y); q.append(y)
            out.append(tuple(sorted(grp)))
        return sorted(out)

    def _core(self, depth):
        core = {self.m} | set(self.donors)
        for d in self.donors:
            core |= {a for a, k in _bfs(self.adj, d, depth).items()}
        ds = sorted(self.donors)
        for i, a in enumerate(ds):
            for b in ds[i + 1:]:
                core |= set(_path(self.adj, a, b))
        return np.array(sorted(core), int)

    def _neighbours(self, a):
        nb = [int(b) for b in np.flatnonzero(self.adj[a])]
        if a in self.donors:
            nb.append(self.m)
        return nb

    def _descriptors(self):
        out = []
        for a in range(len(self.heavy)):
            if a == self.m:
                continue
            nb = self._neighbours(a)
            total = len(nb) + int(self.nH[a])
            el = self.symbols[self.heavy[a]]
            thioether = el in CHALCOGEN and a in self.donors and len(nb) == 3      # R2S->M: inverts at 298 K
            if len(nb) >= 3 and not thioether and (total == 4 or (total == 3 and el in PYRAMIDAL_3)):
                nb = sorted(nb, key=lambda b: (self.colours[b], b))[:3]
                out.append(("chi", ((a,), (nb[0],), (nb[1],), (nb[2],))))
        for b in range(len(self.heavy)):
            for c in np.flatnonzero(self.adj[b]):
                c = int(c)
                if c <= b or self.m in (b, c):
                    continue
                pair = frozenset((self.symbols[self.heavy[b]], self.symbols[self.heavy[c]]))
                lim = DOUBLE_MAX.get(pair)
                if lim is None or np.linalg.norm(self.Xh[b] - self.Xh[c]) >= lim:
                    continue
                nb_b = [x for x in self._neighbours(b) if x != c]
                nb_c = [x for x in self._neighbours(c) if x != b]
                if len(nb_b) + int(self.nH[b]) != 2 or len(nb_c) + int(self.nH[c]) != 2 or not nb_b or not nb_c:
                    continue
                ring = _bfs(self.adj, b, 6, skip_edge=(b, c)).get(c)
                if ring is not None and ring <= 6:
                    continue
                a = sorted(nb_b, key=lambda x: (self.colours[x], x))[0]
                d = sorted(nb_c, key=lambda x: (self.colours[x], x))[0]
                out.append(("ez", ((a,), (b,), (c,), (d,))))
        return out

    # -- evaluation ---------------------------------------------------------------------------------------------
    def values(self, Xh, P, mirror=False):
        """Ternary descriptor values (k, n_desc) for heavy coordinates Xh (k or 1, nh, 3) under perms P (k, nh)."""
        Xh = np.asarray(Xh, float)
        k = len(P)
        if len(Xh) == 1 and k > 1:
            Xh = np.broadcast_to(Xh, (k,) + Xh.shape[1:])
        rows = np.arange(k)[:, None]
        sign = -1.0 if mirror else 1.0
        out = np.empty((k, len(self.desc)))
        cache = {}

        def pt(g):
            if g not in cache:
                cache[g] = Xh[rows, P[:, list(g)]].mean(1)
            return cache[g]

        for n, (kind, groups) in enumerate(self.desc):
            p = [pt(g) for g in groups]
            if kind == "chi":
                # sign only: a (near-)coplanar value is unknown, never a disagreement (coplanarity of sites is carried
                # by the pair descriptors; a flattened centre is not an inverted one)
                u = [_unit(q - p[0]) for q in p[1:]]
                t = ternary(sign * np.einsum("ki,ki->k", u[0], np.cross(u[1], u[2])), CHI_HI, CHI_LO)
                out[:, n] = np.where(t == 0, np.nan, t)
            else:
                b1, b2, b3 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
                n1, n2 = _unit(np.cross(b1, b2)), _unit(np.cross(b2, b3))
                out[:, n] = ternary((n1 * n2).sum(-1), EZ_HI, EZ_LO)
        return out

    def arrangement_ok(self, Yh, P, mirror=False):
        """(k,) bool: labelled site arrangement of the row under each perm matches the reference (see module doc)."""
        from scipy.optimize import linear_sum_assignment
        ok = np.zeros(len(P), bool)
        if len(self.sites) < 2:
            ok[:] = True
            return ok
        Y = Yh * np.array([-1.0, 1, 1]) if mirror else Yh
        U = _unit(np.array([self.Xh[list(g)].mean(0) for g in self.sites]) - self.Xh[self.m])
        keys = {}
        for i, p in enumerate(P):
            key = tuple(tuple(sorted(p[list(g)].tolist())) for g in self.sites) + (int(p[self.m]),)
            keys.setdefault(key, []).append(i)
        for key, idx in keys.items():
            W = _unit(np.array([Y[list(g)].mean(0) for g in key[:-1]]) - Y[key[-1]])
            H = W.T @ U
            Uu, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(Uu @ Vt)) or 1.0
            Wr = W @ (Uu @ np.diag([1, 1, d]) @ Vt)
            C = ((U[:, None] - Wr[None]) ** 2).sum(-1)
            r, c = linear_sum_assignment(C)
            if np.array_equal(c[np.argsort(r)], np.arange(len(U))):
                ok[idx] = True
        return ok

    def _agree(self, V):
        t = self.t_ref[None]
        return np.all(np.isnan(V) | np.isnan(t) | (V == t), axis=1)

    def topology_same(self, X):
        """Covalent graph (incl. H) unchanged: no bond broken/formed, no H transfer."""
        A = geom.ligand_adjacency(np.asarray(X, float), self.symbols, self.metal)
        return bool(np.array_equal(A, self.full_adj))

    def _guided(self, Yh, mirror, limit=50):
        """Geometry-guided automorphisms for capped groups. A seed correspondence (metal distances within WL colour
        classes, contact status respected) is refined by Kabsch on the core + Hungarian within colours; then VF2 on the
        reference graph searches automorphisms restricted to pairs that are geometrically plausible: same colour, no
        contact change, and for core / near-metal atoms within 1.5 A after alignment. Peripheral atoms are free, so the
        search returns real automorphisms (graph-preserving by construction), up to `limit`."""
        import networkx as nx
        from networkx.algorithms import isomorphism as iso
        from scipy.optimize import linear_sum_assignment
        Y = Yh * np.array([-1.0, 1, 1]) if mirror else Yh.copy()
        n = len(self.heavy)
        drow = np.linalg.norm(Y - Y[self.m], axis=1)
        inside_ref, inside_row = self.dref < self.cut, drow < self.cut
        bad_contact = (inside_ref[:, None] != inside_row[None]) & (np.abs(self.dref[:, None] - drow[None]) > CONTACT_SHIFT_A)
        p = np.empty(n, int)
        for c in np.unique(self.colours):
            mem = np.flatnonzero(self.colours == c)
            C = np.abs(self.dref[mem][:, None] - drow[mem][None]) + 10.0 * bad_contact[np.ix_(mem, mem)]
            r_, c_ = linear_sum_assignment(C)
            p[mem[r_]] = mem[c_]
        R = self.Xh
        core = self.core
        for _ in range(GUIDED_ROUNDS):
            A = Y[p[core]] - Y[p[core]].mean(0); B = R[core] - R[core].mean(0)
            U, _, Vt = np.linalg.svd(A.T @ B)
            d = np.sign(np.linalg.det(U @ Vt)) or 1.0
            Z = (Y - Y[p[core]].mean(0)) @ (U @ np.diag([1, 1, d]) @ Vt) + R[core].mean(0)
            new = np.empty_like(p)
            for c in np.unique(self.colours):
                mem = np.flatnonzero(self.colours == c)
                C = ((R[mem][:, None] - Z[mem][None]) ** 2).sum(-1) + 100.0 * bad_contact[np.ix_(mem, mem)]
                r_, c_ = linear_sum_assignment(C)
                new[mem[r_]] = mem[c_]
            if np.array_equal(new, p):
                break
            p = new
        D = np.linalg.norm(R[:, None] - Z[None], axis=-1)
        constrained = np.zeros(n, bool); constrained[core] = True; constrained |= self.dref < self.cut + 1.0
        allowed = (self.colours[:, None] == self.colours[None]) & ~bad_contact & (~constrained[:, None] | (D < 1.5))
        if not hasattr(self, "_G"):
            self._G = nx.Graph(); self._G.add_nodes_from(range(n))
            self._G.add_edges_from((int(a), int(b)) for a, b in zip(*np.nonzero(np.triu(self.adj))))

        class Matcher(iso.GraphMatcher):
            def semantic_feasibility(self_, a, b):
                return bool(allowed[a, b])

        found = []
        for m in Matcher(self._G, self._G).isomorphisms_iter():
            found.append([m[i] for i in range(n)])
            if len(found) >= limit:
                break
        found += [q for q in self._guided_hungarian(Y) if not any(np.array_equal(q, f) for f in found)]
        return np.array(found, int).reshape(-1, n)

    def _guided_hungarian(self, Y):
        """Kabsch on the core (and on all heavy atoms) + Hungarian within colours, repeated; kept only if the assignment
        preserves the heavy graph. Complements the restricted VF2 search (each finds automorphisms the other misses)."""
        from scipy.optimize import linear_sum_assignment
        R = self.Xh - self.Xh.mean(0)
        found = []
        for start in (self.core, np.arange(len(self.heavy))):
            p = np.arange(len(self.heavy))
            for _ in range(GUIDED_ROUNDS):
                A = Y[p[start]] - Y[p[start]].mean(0); B = R[start] - R[start].mean(0)
                U, _, Vt = np.linalg.svd(A.T @ B)
                d = np.sign(np.linalg.det(U @ Vt)) or 1.0
                Z = (Y - Y[p[start]].mean(0)) @ (U @ np.diag([1, 1, d]) @ Vt) + R[start].mean(0)
                new = np.empty_like(p)
                for c in np.unique(self.colours):
                    mem = np.flatnonzero(self.colours == c)
                    r_, c_ = linear_sum_assignment(((R[mem][:, None] - Z[mem][None]) ** 2).sum(-1))
                    new[mem[r_]] = mem[c_]
                if np.array_equal(new, p):
                    break
                p = new
            if np.array_equal(self.adj, self.adj[np.ix_(p, p)]) and not any(np.array_equal(p, q) for q in found):
                found.append(p)
        return found

    def compare(self, X, energy_offset=None, rmsd_hit=0.3, energy_hit=5.0):
        """Class and core-RMSD verdict for one geometry X (reference order, all atoms)."""
        X = np.asarray(X, float)
        Yh = X[self.heavy]
        topo = self.topology_same(X)
        res = dict(donors_same=None, topology_same=topo, same_class=False, mirror=None, core_rmsd_A=None,
                   n_matching_perms=0, guided=False, energy_offset=energy_offset, reason=None, enantiomer=False)
        best = None
        why = []
        for mirror in (False, True):
            P = self.perms
            if self.capped:
                g = self._guided(Yh, mirror)
                if len(g):
                    P = np.concatenate([P, g]); res["guided"] = True
            drow = np.linalg.norm(Yh - Yh[self.m], axis=1)[P]                     # (k, nh): row atom matched to ref atom
            inside_ref = self.dref < self.cut
            changed = ((drow < self.cut[None]) != inside_ref[None]) & (np.abs(drow - self.dref[None]) > CONTACT_SHIFT_A)
            changed[:, self.m] = False
            dmask = ~changed.any(1)
            if not mirror:
                res["donors_same"] = bool(dmask.any())
            if not dmask.any():
                atoms = sorted({int(self.heavy[j]) for j in np.flatnonzero(changed[changed.sum(1).argmin()])})
                why.append(f"contacts{atoms}")
                continue
            Pm = P[dmask]
            V = self.values(Yh[None], Pm, mirror)
            agree = np.isnan(V) | np.isnan(self.t_ref[None]) | (V == self.t_ref[None])
            ok = agree.all(1)
            if not ok.any():
                bad = agree.sum(1).argmax()
                why.append("descriptors:" + ",".join(sorted({self.desc[j][0] for j in np.flatnonzero(~agree[bad])})))
                continue
            ok[ok] = self.arrangement_ok(Yh, Pm[ok], mirror)
            if not ok.any():
                why.append("arrangement")
                continue
            Pm = Pm[ok]
            r = core_rmsd(Yh, self.Xh, Pm, self.core, mirror)
            if best is None or r < best[0]:
                best = (r, mirror, len(Pm))
            if not mirror:
                break                      # a direct match is preferred; the mirror is only tried without one
        # Enantiomers are distinct classes (owner 2026-11-12): a mirror-only match is reported, never the same class.
        if best is not None and topo:
            res.update(enantiomer=bool(best[1]), same_class=not best[1], core_rmsd_A=float(best[0]), mirror=bool(best[1]),
                       n_matching_perms=int(best[2]))
        elif best is not None:
            res.update(core_rmsd_A=float(best[0]), mirror=bool(best[1]))
        if not res["same_class"]:
            res["reason"] = ("topology" if (best is not None and not topo) else "enantiomer" if res.get("enantiomer")
                             else (why[0] if why else None))
        hit_rmsd = res["same_class"] and res["core_rmsd_A"] <= rmsd_hit
        hit_e = res["same_class"] and energy_offset is not None and energy_offset <= energy_hit
        res.update(hit=bool(hit_rmsd or hit_e), hit_by=("rmsd" if hit_rmsd else "energy" if hit_e else None))
        return res


def core_rmsd(Yh, Rh, P, core, mirror=False):
    """Min over perms P of the Kabsch (proper rotation) RMSD of row core atoms P[:, core] against reference core atoms."""
    Y = Yh * np.array([-1.0, 1, 1]) if mirror else Yh
    b = Rh[core] - Rh[core].mean(0)
    best = np.inf
    for k0 in range(0, len(P), 2048):
        A = Y[P[k0:k0 + 2048][:, core]]
        A = A - A.mean(1, keepdims=True)
        H = np.einsum("kni,nj->kij", A, b)
        sv = np.linalg.svd(H, compute_uv=False)
        sv[:, -1] *= np.sign(np.linalg.det(H))
        ss = (A ** 2).sum((1, 2)) + (b ** 2).sum() - 2 * sv.sum(1)
        best = min(best, float(np.sqrt(max(ss.min(), 0) / len(core))))
    return best
