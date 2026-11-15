"""runkit.geom: the consumer-facing symmetry interface. Code below is A4 geom.py (jgen/a-census, commit 4ad04de, sha256 a0880a68...)
unchanged in behaviour; the factored automorphism pass (PLAN_jgen section 7a-H) will land underneath these names and keep
`automorphisms()` returning (P, capped) with P the explicit array for small groups. Original docstring:

Geometry helpers for A4: reference graph, metal and donor detection, symmetry-aware heavy-atom RMSD, dedup.

Heavy-atom graph from the reference geometry (covalent radii + 0.4 A, metal bonds excluded). Automorphisms by VF2 with
element match, capped at CAP, identity always first. Kabsch on heavy atoms including the metal, vectorised over
(rows x automorphisms), using singular values of the 3x3 covariance (det sign fix).
"""
from __future__ import annotations

import numpy as np

CAP = 4000
_METAL_Z = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81)) | set(range(89, 104))
_PT_CACHE = []


def _pt():
    """RDKit periodic table, imported on first use (importing runkit.geom needs neither rdkit nor networkx)."""
    if not _PT_CACHE:
        from rdkit import Chem
        _PT_CACHE.append(Chem.GetPeriodicTable())
    return _PT_CACHE[0]


def rcov(sym: str) -> float:
    return float(_pt().GetRcovalent(sym))


def is_metal(sym: str) -> bool:
    return _pt().GetAtomicNumber(sym) in _METAL_Z


def find_metal(symbols) -> int:
    """First transition-metal / lanthanide atom (the benchmark parents have one metal centre)."""
    for i, s in enumerate(symbols):
        if is_metal(str(s)):
            return i
    raise ValueError("no metal atom found")


def ligand_adjacency(ref: np.ndarray, symbols, metal: int, pad: float = 0.4) -> np.ndarray:
    """Boolean covalent adjacency of all non-metal atoms (covalent radii + pad); metal row/column empty."""
    n = len(symbols)
    r = np.array([rcov(str(s)) for s in symbols])
    d = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
    adj = d < (r[:, None] + r[None] + pad)
    np.fill_diagonal(adj, False)
    adj[metal, :] = False
    adj[:, metal] = False
    return adj


def donors_of(ref: np.ndarray, symbols, metal: int, pad: float = 0.6) -> np.ndarray:
    """Atoms within r_cov(M) + r_cov(d) + pad of the metal (the bound-donor rule of PLAN_jgen.md section 2)."""
    return np.array([i for i in range(len(symbols)) if i != metal and is_metal(str(symbols[metal]))
                     and np.linalg.norm(ref[i] - ref[metal]) < rcov(str(symbols[metal])) + rcov(str(symbols[i])) + pad
                     and str(symbols[i]) != "H"], int)


def automorphisms(ref: np.ndarray, symbols, metal: int, cap: int = CAP):
    """(P, capped): P is (k, n_heavy) index arrays into the heavy-atom list (identity first); capped flags cap hit."""
    import networkx as nx
    from networkx.algorithms import isomorphism as iso
    heavy = [i for i, s in enumerate(symbols) if str(s) != "H"]
    adj = ligand_adjacency(ref, symbols, metal)
    G = nx.Graph()
    for i in heavy:
        G.add_node(i, el=str(symbols[i]))
    for x, i in enumerate(heavy):
        for j in heavy[x + 1:]:
            if adj[i, j]:
                G.add_edge(i, j)
    pos = {a: x for x, a in enumerate(heavy)}
    ident = np.arange(len(heavy))
    out, seen = [ident], {tuple(ident)}
    capped = False
    gm = iso.GraphMatcher(G, G, node_match=iso.categorical_node_match("el", ""))
    for m in gm.isomorphisms_iter():
        p = np.array([pos[m[i]] for i in heavy])
        t = tuple(p)
        if t in seen:
            continue
        seen.add(t)
        out.append(p)
        if len(out) >= cap:
            capped = True
            break
    return np.array(out), capped


def _min_ss(J: np.ndarray, Jsq: np.ndarray, b: np.ndarray, bb: float, Pinv: np.ndarray, cut: float | None,
            chunk: int = 3_000_000) -> np.ndarray:
    """Minimal sum of squared deviations over the automorphisms (rows of Pinv) for each of the m centred rows J (m, n, 3)
    against the centred reference b (n, 3).  Covariances for all (row, automorphism) pairs are one GEMM; the
    Kabsch sum of singular values is only computed where the cheap bound sum(sigma) <= sqrt(3)*||H||_F cannot rule the
    combination out against `cut` (a sum of squares: cut = n * thr^2; None = no pruning).  Returns (m,) minima
    (inf where every combination was pruned)."""
    m, n = J.shape[0], J.shape[1]
    Jr = J.transpose(0, 2, 1).reshape(m * 3, n)
    best = np.full(m, np.inf)
    kstep = max(1, chunk // (9 * m))
    for k0 in range(0, len(Pinv), kstep):
        Bp = b[Pinv[k0:k0 + kstep]]                         # k, n, 3  (b permuted onto J's atom order)
        k = len(Bp)
        Hm = (Jr @ Bp.transpose(1, 0, 2).reshape(n, k * 3)).reshape(m, 3, k, 3).transpose(0, 2, 1, 3)  # m,k,3,3
        if cut is None:
            sel = np.ones((m, k), bool)
        else:
            fro = np.sqrt((Hm ** 2).sum((-1, -2)))
            sel = (Jsq[:, None] + bb - 2.0 * np.sqrt(3.0) * fro) < cut
        mi, ki = np.nonzero(sel)
        if len(mi) == 0:
            continue
        H = Hm[mi, ki]
        sv = np.linalg.svd(H, compute_uv=False)
        sv[:, -1] *= np.sign(np.linalg.det(H))
        ss = Jsq[mi] + bb - 2.0 * sv.sum(1)
        np.minimum.at(best, mi, ss)
    return best


def _inv_perms(P: np.ndarray) -> np.ndarray:
    Pinv = np.empty_like(P)
    rows = np.arange(len(P))[:, None]
    Pinv[rows, P] = np.arange(P.shape[1])[None, :]
    return Pinv



def perm_union(P: np.ndarray) -> np.ndarray:
    """P with its inverse permutations. A complete automorphism group is closed under inversion; a group truncated at CAP
    is not, and min over b[P] vs b[P^-1] can differ a lot (SILROY 0.42 vs 2.41 A). The union is the better upper bound."""
    P = np.asarray(P, int)
    return np.unique(np.vstack([P, _inv_perms(P)]), axis=0)


def heavy_graph(ref: np.ndarray, symbols, metal: int):
    """(adj, classes) of the heavy-atom graph (same atom order as the heavy list): boolean adjacency and colour-refinement
    classes (element + neighbourhood). Pass as `graph=` to sym_rmsd_to_ref / sym_rmsd_matrix to enable the capped-parent
    refinement, which then permutes only within a class and keeps a permutation only if it is a graph automorphism."""
    heavy = [i for i, s in enumerate(symbols) if str(s) != "H"]
    adj = ligand_adjacency(np.asarray(ref, float), symbols, metal)[np.ix_(heavy, heavy)]
    col = [str(symbols[i]) for i in heavy]
    ids = {c: k for k, c in enumerate(sorted(set(col)))}
    col = [ids[c] for c in col]
    for _ in range(10):
        sig = [(col[i], tuple(sorted(col[j] for j in np.nonzero(adj[i])[0]))) for i in range(len(col))]
        ids = {c: k for k, c in enumerate(sorted(set(sig), key=str))}
        new = [ids[x] for x in sig]
        if len(set(new)) == len(set(col)):
            col = new
            break
        col = new
    return adj, np.asarray(col)


def _refine_capped(Q: np.ndarray, b: np.ndarray, graph, start: float, iters: int = 3) -> float:
    """Alternating assignment + Kabsch from the aligned row Q (centred) against the centred reference b. The assignment is
    done inside each class only, and a permutation is accepted only if it is a graph automorphism (adj[c][:, c] == adj).
    Returns the best RMSD found (never above `start`)."""
    from scipy.optimize import linear_sum_assignment
    adj, classes = graph
    n = len(b)
    best = start
    for _ in range(iters):
        c_full = np.arange(n)
        for cl in np.unique(classes):
            idx = np.nonzero(classes == cl)[0]
            if len(idx) < 2:
                continue
            C = ((b[idx][:, None, :] - Q[idx][None, :, :]) ** 2).sum(-1)
            r_, c_ = linear_sum_assignment(C)
            c_full[idx[r_]] = idx[c_]
        if not (adj[np.ix_(c_full, c_full)] == adj).all():
            break                                     # not a symmetry of the molecule: reject
        Qc = Q[c_full] - Q[c_full].mean(0)
        H = Qc.T @ b
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1, 1, d]) @ Vt
        Q = Qc @ R
        r = float(np.sqrt(((b - Q) ** 2).sum(1).mean()))
        if r < best:
            best = r
        else:
            break
    return best


def sym_rmsd_to_ref(Xh: np.ndarray, Rh: np.ndarray, P: np.ndarray, graph=None) -> np.ndarray:
    """Min over automorphisms (and rotation/translation) of the heavy-atom RMSD. Xh (r, n, 3), Rh (n, 3) -> (r,)."""
    n = Xh.shape[1]
    J = Xh - Xh.mean(1, keepdims=True)
    b = Rh - Rh.mean(0)
    capped_group = len(P) >= CAP
    Pinv = perm_union(P) if capped_group else _inv_perms(P)    # truncated group: use P and P^-1 (was P^-1 only)
    ss = _min_ss(J, (J ** 2).sum((1, 2)), b, float((b ** 2).sum()), Pinv, None)
    res = np.sqrt(np.clip(ss, 0, None) / n)
    if graph is not None and capped_group:
        try:
            from jacobik.core import profile
            if profile.get("dedup_factored"):
                for i in range(len(res)):
                    if res[i] > 0.5:
                        res[i] = _refine_capped(J[i], b, graph, float(res[i]))
        except Exception:
            pass
    return res


def radial_lower_bound(Xh: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """(r, r) lower bound on the symmetry-aware Kabsch RMSD of every pair of rows.

    For any rigid motion and any permutation p that preserves the atom classes, |a_i - R b_p(i)| >= | |a_i| - |b_p(i)| |
    (norms from the heavy-atom centroid, which every permutation keeps). Within a class the best 1-D matching is the sorted
    one, so LB^2 = sum over classes of sum (sorted |a| - sorted |b|)^2 / n. A pair with LB >= thr cannot be linked."""
    r, n = Xh.shape[0], Xh.shape[1]
    C = Xh - Xh.mean(1, keepdims=True)
    rad = np.linalg.norm(C, axis=2)
    out = np.empty_like(rad)
    for c in np.unique(classes):
        idx = np.nonzero(classes == c)[0]
        out[:, idx] = np.sort(rad[:, idx], axis=1)
    diff = out[:, None, :] - out[None, :, :]
    return np.sqrt((diff ** 2).sum(-1) / n)


def sym_rmsd_matrix(Xh: np.ndarray, P: np.ndarray, classes: np.ndarray, thr: float,
                    stats: dict | None = None, graph=None) -> np.ndarray:
    """(r, r) symmetric matrix D with D < thr exactly where the symmetry-aware heavy-atom RMSD is < thr.

    Entries below thr are exact RMSDs (min over the automorphisms P); entries at or above thr are lower bounds (pairs
    ruled out by the radial bound, or by the Frobenius bound on every automorphism, are not evaluated exactly)."""
    r, n = Xh.shape[0], Xh.shape[1]
    LB = radial_lower_bound(Xh, classes)
    D = np.maximum(LB, thr)
    np.fill_diagonal(D, 0.0)
    C = Xh - Xh.mean(1, keepdims=True)
    Csq = (C ** 2).sum((1, 2))
    capped_group = len(P) >= CAP
    Pinv = perm_union(P) if capped_group else _inv_perms(P)
    n_eval = 0
    cut = n * thr ** 2 * 1.0001
    for i in range(r - 1):
        cand = np.nonzero(LB[i, i + 1:] < thr)[0] + i + 1
        if len(cand) == 0:
            continue
        b = C[i]
        best = _min_ss(C[cand], Csq[cand], b, float(Csq[i]), Pinv, cut)
        n_eval += len(cand)
        v = np.sqrt(np.clip(best, 0, None) / n)
        v = np.where(np.isfinite(v), np.maximum(v, thr) if False else v, thr)
        ok = np.isfinite(best)
        D[i, cand[ok]] = D[cand[ok], i] = v[ok]
    if graph is not None and capped_group:
        try:
            from jacobik.core import profile
            if profile.get("dedup_factored"):
                for i in range(r - 1):
                    for j in range(i + 1, r):
                        if D[i, j] >= thr and LB[i, j] < thr:
                            D[i, j] = D[j, i] = _refine_capped(C[j], C[i], graph, float(D[i, j]))
        except Exception:
            pass
    if stats is not None:
        stats.update(pairs=r * (r - 1) // 2, pairs_after_radial=int(((LB < thr).sum() - 0) // 2), pairs_evaluated=n_eval)
    return D


def cluster_single_linkage(D: np.ndarray, thr: float) -> np.ndarray:
    """Labels 0..c-1 of the connected components of D < thr (single linkage)."""
    r = len(D)
    lab = -np.ones(r, int)
    c = 0
    for s in range(r):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = c
        while stack:
            u = stack.pop()
            for v in np.nonzero((D[u] < thr) & (lab < 0))[0]:
                lab[v] = c
                stack.append(int(v))
        c += 1
    return lab


# facade names (stable interface for consumers)
__all__ = ["perm_union", "heavy_graph", "automorphisms", "sym_rmsd_to_ref", "sym_rmsd_matrix", "cluster_single_linkage", "ligand_adjacency",
           "donors_of", "radial_lower_bound", "find_metal", "is_metal", "rcov", "CAP"]
