"""Graph-constrained atom mapping between generated conformers and reference complexes.

In ComplexBuilder, generated conformers use an atom ordering determined by ligand
construction (metal at origin index 0, followed sequentially by each ligand's atoms
as added from RDKit SMILES conformers). The reference crystal structure from CSD /
tmQM typically has a different atom ordering (e.g., all carbons across all ligands
grouped together, followed by all hydrogens).

This module implements robust, graph-constrained generated -> reference atom mapping:
1. Maps the metal atom to the metal atom.
2. Maps each ligand component in the generated complex to the corresponding ligand
   component in the reference complex using donor-metal attachment, element symbols,
   hydrogen counts, and graph isomorphism (via NetworkX VF2).
3. Handles symmetric and identical ligands cleanly; identifies and flags ambiguous
   mappings if any automorphism or symmetry permutation exists.
4. Produces indexing arrays `gen_to_ref_idx` and `ref_to_gen_idx` such that
   `X_gen[..., gen_to_ref_idx, :]` has the exact same atom indexing as `X_ref`
   (and `gen_symbols[gen_to_ref_idx] == ref_symbols`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np

from jacobik.core import profile


@dataclass
class AtomMappingResult:
    """Result of graph-constrained generated -> reference atom mapping.

    Attributes
    ----------
    gen_to_ref_idx : np.ndarray
        Integer array of shape (N,) with gen atom index for each ref atom position.
        Applying this index to generated arrays aligns them to reference ordering:
        `X_gen[..., gen_to_ref_idx, :]` corresponds to `X_ref`.
        `gen_symbols[gen_to_ref_idx]` matches `ref_symbols`.
    ref_to_gen_idx : np.ndarray
        Integer array of shape (N,) with ref atom index for each gen atom position.
        `X_ref[..., ref_to_gen_idx, :]` corresponds to `X_gen`.
        `ref_symbols[ref_to_gen_idx]` matches `gen_symbols`.
    is_ambiguous : bool
        True if any ligand automorphism or permutation symmetry exists.
    num_automorphisms : int
        Total number of valid graph automorphisms / symmetry permutations.
    ligand_matches : List[Dict[str, Any]]
        Diagnostic metadata per ligand component.
    """

    gen_to_ref_idx: np.ndarray
    ref_to_gen_idx: np.ndarray
    is_ambiguous: bool
    num_automorphisms: int
    ligand_matches: List[Dict[str, Any]] = field(default_factory=list)

    def apply_to_positions(self, positions: np.ndarray) -> np.ndarray:
        """Reorder generated positions (N, 3) or (B, N, 3) to match reference atom ordering."""
        pos = np.asarray(positions)
        return pos[..., self.gen_to_ref_idx, :]

    def verify_symbols(self, ref_symbols: Sequence[str], gen_symbols: Sequence[str]) -> bool:
        """Verify that mapped generated symbols match reference symbols element-by-element."""
        ref_s = np.asarray(ref_symbols)
        gen_s = np.asarray(gen_symbols)
        mapped = gen_s[self.gen_to_ref_idx]
        return bool(np.array_equal(mapped, ref_s))

def _extract_ligand_subgraphs(mol: Any) -> List[Dict[str, Any]]:
    """Extract heavy-atom graphs, donor sets, and hydrogen neighbor lists per ligand."""
    ligand_data: List[Dict[str, Any]] = []
    idx_to_0 = {a.idx: i for i, a in enumerate(mol.atoms)}

    for lig in mol.ligands:
        heavy_1idx = [idx for idx in lig.atom_indices if mol.atoms[idx_to_0[idx]].symbol != "H"]
        heavy_set = set(heavy_1idx)
        donor_set = set(lig.donor_atoms)

        # Count bonded hydrogens and record neighbor lists for each heavy atom
        nh = {idx: 0 for idx in heavy_1idx}
        h_neighbors: Dict[int, List[int]] = {idx_to_0[idx]: [] for idx in heavy_1idx}

        for b in mol.bonds:
            if b.order >= 0.5:
                if b.i in heavy_set and mol.atoms[idx_to_0[b.j]].symbol == "H":
                    nh[b.i] += 1
                    h_neighbors[idx_to_0[b.i]].append(idx_to_0[b.j])
                elif b.j in heavy_set and mol.atoms[idx_to_0[b.i]].symbol == "H":
                    nh[b.j] += 1
                    h_neighbors[idx_to_0[b.j]].append(idx_to_0[b.i])

        G = nx.Graph()
        for idx in heavy_1idx:
            u_0 = idx_to_0[idx]
            a = mol.atoms[u_0]
            G.add_node(
                u_0,
                element=a.symbol,
                is_donor=(idx in donor_set),
                nh=nh[idx],
            )

        for b in mol.bonds:
            if b.i in heavy_set and b.j in heavy_set and b.order >= 0.5:
                G.add_edge(idx_to_0[b.i], idx_to_0[b.j])

        elem_counts = tuple(
            sorted(
                [
                    (sym, sum(1 for idx in heavy_1idx if mol.atoms[idx_to_0[idx]].symbol == sym))
                    for sym in set(mol.atoms[idx_to_0[idx]].symbol for idx in heavy_1idx)
                ]
            )
        )

        ligand_data.append({
            "G": G,
            "heavy_0idx": [idx_to_0[idx] for idx in heavy_1idx],
            "h_neighbors": h_neighbors,
            "donors_0idx": [idx_to_0[idx] for idx in lig.donor_atoms],
            "all_0idx": [idx_to_0[idx] for idx in lig.atom_indices],
            "formula": elem_counts,
            "n_heavy": len(heavy_1idx),
            "n_total": len(lig.atom_indices),
        })

    return ligand_data


def _rd_graph(G, nodes, strict):
    """RDKit mol of a ligand heavy-atom graph; atom labels (element, donor flag, H count if strict) in the isotope."""
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    m = Chem.RWMol()
    idx = {}
    for u in nodes:
        d = G.nodes[u]
        a = Chem.Atom(pt.GetAtomicNumber(d.get("element", "")))
        a.SetIsotope(1 + (int(d.get("nh", 0)) if strict else 0) + 64 * int(bool(d.get("is_donor", False))))
        a.SetNoImplicit(True)
        idx[u] = m.AddAtom(a)
    for u, v in G.edges():
        m.AddBond(idx[u], idx[v], Chem.BondType.SINGLE)
    m = m.GetMol()
    m.UpdatePropertyCache(strict=False)
    return m


def _isomorphisms(G_gen, G_ref, strict, node_match, cap):
    """(gen nodes, (K, n) ref node per gen node) of the isomorphisms G_gen -> G_ref, at most cap + 1 of them; None if
    not isomorphic. RDKit substructure matching (graphs of equal size: a match is an isomorphism); networkx VF2 when
    RDKit cannot take the graph."""
    g_nodes = sorted(G_gen.nodes())
    if G_gen.number_of_nodes() != G_ref.number_of_nodes() or G_gen.number_of_edges() != G_ref.number_of_edges():
        return None
    r_nodes = sorted(G_ref.nodes())
    try:
        q = _rd_graph(G_gen, g_nodes, strict)
        t = _rd_graph(G_ref, r_nodes, strict)
        ms = t.GetSubstructMatches(q, uniquify=False, useChirality=False, maxMatches=int(cap) + 1)
        if not ms:
            return None
        return g_nodes, np.asarray([[r_nodes[k] for k in m_] for m_ in ms], dtype=np.int64)
    except Exception:
        GM = isomorphism.GraphMatcher(G_gen, G_ref, node_match=node_match)
        if not GM.is_isomorphic():
            return None
        rows = []
        for iso in GM.isomorphisms_iter():
            rows.append([iso[v] for v in g_nodes])
            if len(rows) > cap:
                break
        return g_nodes, np.asarray(rows, dtype=np.int64)


def build_atom_mapping(
    ref_mol: Any,
    gen_mol: Optional[Any] = None,
    builder: Optional[Any] = None,
    X_ref: Optional[np.ndarray] = None,
    X_gen: Optional[np.ndarray] = None,
    max_automorphisms: Optional[int] = None,
) -> AtomMappingResult:
    """Build a graph-constrained atom mapping from generated complex to reference complex.

    Parameters
    ----------
    ref_mol : TMComplex
        Parsed reference complex (e.g. from load_mol / TMQMParser).
    gen_mol : Optional[TMComplex]
        Generated complex shell or conformer.complex. If None, built from builder.
    builder : Optional[ComplexBuilder]
        ComplexBuilder instance. Used to build gen_mol if gen_mol is None.
    X_ref : Optional[np.ndarray]
        Reference coordinates (N, 3). Defaults to [a.position for a in ref_mol.atoms].
    X_gen : Optional[np.ndarray]
        Generated coordinates (N, 3). Defaults to [a.position for a in gen_mol.atoms].
    max_automorphisms : int
        Maximum number of internal ligand isomorphisms to inspect before truncating.

    Returns
    -------
    AtomMappingResult
        Mapping result containing `gen_to_ref_idx`, `ref_to_gen_idx`, `is_ambiguous`,
        and `num_automorphisms`.

    Notes
    -----
    Mapping convention:
    `X[:, gen_to_ref_idx]` puts generated coordinates in REFERENCE atom order,
    so `symbols[gen_to_ref_idx] == ref_symbols`.
    """
    if max_automorphisms is None:
        max_automorphisms = 100000 if profile.get("dedup_factored") else 512

    if gen_mol is None:
        if builder is None:
            from jacobik.core.env import build_builder_and_inputs
            builder, _, err = build_builder_and_inputs(ref_mol)
            if builder is None:
                raise ValueError(f"Could not build ComplexBuilder for ref_mol: {err}")
        gen_mol, _, _ = builder._build_shell(chelate_embed=True, fast_embed=False, debug=False)

    N_ref = len(ref_mol.atoms)
    N_gen = len(gen_mol.atoms)
    if N_ref != N_gen:
        raise ValueError(f"Atom count mismatch: ref has {N_ref} atoms, gen has {N_gen} atoms")

    # 1. Map metal atom
    ref_m_0idx = next(i for i, a in enumerate(ref_mol.atoms) if a.idx == ref_mol.metal_idx)
    gen_m_0idx = next(i for i, a in enumerate(gen_mol.atoms) if a.idx == gen_mol.metal_idx)
    if ref_mol.atoms[ref_m_0idx].symbol != gen_mol.atoms[gen_m_0idx].symbol:
        raise ValueError(
            f"Metal symbol mismatch: ref is {ref_mol.atoms[ref_m_0idx].symbol}, "
            f"gen is {gen_mol.atoms[gen_m_0idx].symbol}"
        )

    # 2. Extract coordinates for geometric disambiguation
    if X_ref is None:
        X_ref = np.array([a.position for a in ref_mol.atoms], dtype=float)
    if X_gen is None:
        X_gen = np.array([a.position for a in gen_mol.atoms], dtype=float)

    # Center coordinates on respective metal positions
    P_ref = X_ref - X_ref[ref_m_0idx]
    P_gen = X_gen - X_gen[gen_m_0idx]

    # 3. Extract ligand subgraphs
    ref_ligs = _extract_ligand_subgraphs(ref_mol)
    gen_ligs = _extract_ligand_subgraphs(gen_mol)

    if len(ref_ligs) != len(gen_ligs):
        raise ValueError(
            f"Ligand count mismatch: ref has {len(ref_ligs)}, gen has {len(gen_ligs)}"
        )

    # Group ligands by chemical signature (heavy formula, total atom count)
    ref_groups: Dict[Tuple[Any, int], List[int]] = {}
    for i, d in enumerate(ref_ligs):
        sig = (d["formula"], d["n_total"])
        ref_groups.setdefault(sig, []).append(i)

    gen_groups: Dict[Tuple[Any, int], List[int]] = {}
    for j, d in enumerate(gen_ligs):
        sig = (d["formula"], d["n_total"])
        gen_groups.setdefault(sig, []).append(j)

    if set(ref_groups.keys()) != set(gen_groups.keys()):
        raise ValueError(
            f"Ligand chemical signature mismatch between ref and gen: "
            f"ref={set(ref_groups.keys())}, gen={set(gen_groups.keys())}"
        )

    # 4. Map ligands and their internal atoms
    ref_to_gen: Dict[int, int] = {ref_m_0idx: gen_m_0idx}
    total_automorphisms = 1
    ligand_match_meta: List[Dict[str, Any]] = []

    node_match_strict = isomorphism.categorical_node_match(["element", "is_donor", "nh"], ["", False, 0])
    node_match_relaxed = isomorphism.categorical_node_match(["element", "is_donor"], ["", False])

    for sig, r_indices in ref_groups.items():
        g_indices = gen_groups[sig]
        m = len(r_indices)
        if len(g_indices) != m:
            raise ValueError(f"Instance count mismatch for ligand signature {sig}: ref={m}, gen={len(g_indices)}")

        best_perm = None
        best_perm_cost = float("inf")
        best_perm_mapping: Dict[int, int] = {}
        best_perm_autos = 1
        best_perm_details = []

        # Each (ref ligand, gen ligand) pairing is matched once and cached: the permutation loop below only sums the
        # cached pair costs in the same order (identical floats, identical choice). Before 2026-11-18 every permutation
        # of m identical ligands re-ran the full VF2 enumeration of every pair (m! x m enumerations: SILROY 167 s).
        pair_cache: Dict[Tuple[int, int], Tuple[bool, float, Dict[int, int], int, Dict[str, Any]]] = {}

        def h_pair(r_idx, g_idx, u_ref, v_gen, memo):
            """(cost, h_map, valid, matched) of the hydrogens of heavy pair (u_ref, v_gen), as the per-iso loop had it."""
            key = (u_ref, v_gen)
            hit = memo.get(key)
            if hit is not None:
                return hit
            h_r = ref_ligs[r_idx]["h_neighbors"][u_ref]
            h_g = gen_ligs[g_idx]["h_neighbors"][v_gen]
            out = (0.0, {}, True, False)
            if len(h_r) == 1 and len(h_g) == 1:
                dh = P_ref[h_r[0]] - P_gen[h_g[0]]
                out = (float(dh @ dh), {h_r[0]: h_g[0]}, True, True)
            elif len(h_r) > 1 and len(h_r) == len(h_g):
                best_hp = None
                best_hd = float("inf")
                for hp in itertools.permutations(h_g):
                    hd = sum(
                        float((P_ref[hr] - P_gen[hg]) @ (P_ref[hr] - P_gen[hg]))
                        for hr, hg in zip(h_r, hp)
                    )
                    if hd < best_hd:
                        best_hd = hd
                        best_hp = hp
                if best_hp is not None:
                    out = (best_hd, dict(zip(h_r, best_hp)), True, True)
            elif len(h_r) != len(h_g):
                out = (0.0, {}, False, False)
            memo[key] = out
            return out

        def match_pair(r_idx, g_idx):
            key = (r_idx, g_idx)
            if key in pair_cache:
                return pair_cache[key]
            G_ref = ref_ligs[r_idx]["G"]
            G_gen = gen_ligs[g_idx]["G"]
            memo: Dict[Tuple[int, int], Tuple[float, Dict[int, int], bool, bool]] = {}
            mapping: Dict[int, int] = {}
            if G_ref.number_of_nodes() == 1:
                # Single-atom ligand (e.g. halide, aqua, oxo, ammine, methyl)
                u_ref = list(G_ref.nodes())[0]
                v_gen = list(G_gen.nodes())[0]
                mapping[u_ref] = v_gen
                d = P_ref[u_ref] - P_gen[v_gen]
                cost = float(d @ d)
                hc, hm, hv, matched = h_pair(r_idx, g_idx, u_ref, v_gen, memo)
                if not hv:
                    res = (False, cost, mapping, 1, {})
                else:
                    if matched:
                        mapping.update(hm)
                        cost += hc
                    res = (True, cost, mapping, 1, {
                        "ref_ligand_idx": r_idx, "gen_ligand_idx": g_idx, "n_heavy": 1, "n_automorphisms": 1,
                        "cost": cost})
                pair_cache[key] = res
                return res

            # Multi-atom ligand: all isomorphisms gen -> ref (RDKit's C++ matcher; networkx VF2 as the fallback), every
            # one scored at once from the per-atom-pair heavy + hydrogen costs (the hydrogen assignment of a heavy pair
            # does not depend on the rest of the isomorphism, so it is solved once per pair)
            isos = None
            for nm in (node_match_strict, node_match_relaxed):
                isos = _isomorphisms(G_gen, G_ref, nm is node_match_strict, nm, max_automorphisms)
                if isos is not None:
                    break
            if isos is None:
                pair_cache[key] = (False, 0.0, {}, 0, {})
                return pair_cache[key]
            g_nodes, matches = isos
            n_autos = int(matches.shape[0])
            r_nodes = sorted(G_ref.nodes())
            col = {u: k for k, u in enumerate(r_nodes)}
            C = np.full((len(g_nodes), len(r_nodes)), np.inf)
            bad = np.zeros_like(C, dtype=bool)
            used = np.unique(np.stack([np.repeat(np.arange(len(g_nodes))[None], n_autos, 0).ravel(),
                                       matches.ravel()], 1), axis=0)
            for gi, u_ref in used:
                v_gen = g_nodes[gi]
                d = P_ref[u_ref] - P_gen[v_gen]
                hc, _hm, hv, matched = h_pair(r_idx, g_idx, int(u_ref), v_gen, memo)
                C[gi, col[int(u_ref)]] = float(d @ d) + (hc if matched else 0.0)
                bad[gi, col[int(u_ref)]] = not hv
            mcol = np.vectorize(col.get, otypes=[np.int64])(matches) if n_autos else matches
            ar = np.arange(len(g_nodes))[None]
            if n_autos > max_automorphisms:
                mcol = mcol[:max_automorphisms]
            costs = C[ar, mcol].sum(1)
            valid = not bool(bad[ar, mcol].any())
            k_best = int(np.argmin(costs))
            best_iso_cost = float(costs[k_best])
            best_iso = {g_nodes[gi]: int(matches[k_best, gi]) for gi in range(len(g_nodes))}
            h_map: Dict[int, int] = {}
            for v_gen, u_ref in best_iso.items():
                mapping[u_ref] = v_gen
                hc, hm, hv, matched = h_pair(r_idx, g_idx, u_ref, v_gen, memo)
                if matched:
                    h_map.update(hm)
            mapping.update(h_map)
            res = (valid, best_iso_cost, mapping, n_autos, {
                "ref_ligand_idx": r_idx, "gen_ligand_idx": g_idx, "n_heavy": G_ref.number_of_nodes(),
                "n_automorphisms": n_autos, "cost": best_iso_cost})
            pair_cache[key] = res
            return res

        # For identical ligands (m > 1), evaluate pairings against relative coordinates
        for g_perm in itertools.permutations(g_indices):
            perm_cost = 0.0
            perm_mapping: Dict[int, int] = {}
            perm_autos = 1
            perm_details = []
            is_valid_perm = True

            for r_idx, g_idx in zip(r_indices, g_perm):
                ok, cost, mapping, n_autos, detail = match_pair(r_idx, g_idx)
                if not ok:
                    is_valid_perm = False
                    break
                perm_cost += cost
                perm_mapping.update(mapping)
                if ref_ligs[r_idx]["G"].number_of_nodes() > 1:
                    perm_autos *= max(n_autos, 1)
                perm_details.append(detail)

            if is_valid_perm and perm_cost < best_perm_cost:
                best_perm_cost = perm_cost
                best_perm = g_perm
                best_perm_mapping = perm_mapping
                best_perm_autos = perm_autos * (math.factorial(m) if m > 1 else 1)
                best_perm_details = perm_details

        if best_perm is None:
            raise RuntimeError(f"Could not find any isomorphic mapping for ligand group {sig}")

        ref_to_gen.update(best_perm_mapping)
        total_automorphisms *= best_perm_autos
        ligand_match_meta.extend(best_perm_details)

    # Assign any remaining symmetry-equivalent hydrogens attached to mapped heavy parents
    # using minimum distance among that parent's hydrogens.
    for r_idx, d_r in enumerate(ref_ligs):
        for u_ref, h_r_list in d_r["h_neighbors"].items():
            if u_ref in ref_to_gen:
                v_gen = ref_to_gen[u_ref]
                unmapped_hr = [hr for hr in h_r_list if hr not in ref_to_gen]
                if unmapped_hr:
                    g_idx = next(
                        (gi for gi, dg in enumerate(gen_ligs) if v_gen in dg["h_neighbors"]), None
                    )
                    if g_idx is not None:
                        h_g_list = gen_ligs[g_idx]["h_neighbors"][v_gen]
                        mapped_hg_vals = set(ref_to_gen.values())
                        unmapped_hg = [hg for hg in h_g_list if hg not in mapped_hg_vals]
                        if len(unmapped_hr) == len(unmapped_hg):
                            if len(unmapped_hr) == 1:
                                ref_to_gen[unmapped_hr[0]] = unmapped_hg[0]
                            else:
                                best_hp = None
                                best_hd = float("inf")
                                for hp in itertools.permutations(unmapped_hg):
                                    hd = sum(
                                        float((P_ref[hr] - P_gen[hg]) @ (P_ref[hr] - P_gen[hg]))
                                        for hr, hg in zip(unmapped_hr, hp)
                                    )
                                    if hd < best_hd:
                                        best_hd = hd
                                        best_hp = hp
                                if best_hp is not None:
                                    for hr, hg in zip(unmapped_hr, best_hp):
                                        ref_to_gen[hr] = hg

    if len(ref_to_gen) != N_ref:
        unmapped = set(range(N_ref)) - set(ref_to_gen.keys())
        raise RuntimeError(f"Atom mapping incomplete: {len(unmapped)} reference atoms unmapped: {unmapped}")

    gen_to_ref_idx = np.array([ref_to_gen[i] for i in range(N_ref)], dtype=np.int64)
    ref_to_gen_idx = np.zeros(N_ref, dtype=np.int64)
    for r_i, g_i in ref_to_gen.items():
        ref_to_gen_idx[g_i] = r_i

    # Ambiguity is flagged if automorphisms exist or multiple identical ligands exist
    is_ambiguous = (total_automorphisms > 1)

    return AtomMappingResult(
        gen_to_ref_idx=gen_to_ref_idx,
        ref_to_gen_idx=ref_to_gen_idx,
        is_ambiguous=is_ambiguous,
        num_automorphisms=total_automorphisms,
        ligand_matches=ligand_match_meta,
    )
