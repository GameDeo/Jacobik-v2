"""
geometry.py
===========
Differentiable geometric quantity extractor for the IK engine.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import torch

from jacobik.ik.kernels import _safe_normalize, gather_atoms
from jacobik.ik.kernels import dihedral as _kernel_dihedral

from jacobik.core import profile as _profile

if TYPE_CHECKING:
    from jacobik.core.types import TMComplex

log = logging.getLogger(__name__)

#: Q6 donor pairs (production since the Package H fold-in, 2026-10-31): kappa-kappa donor pairs on different ligands join
#: the Q6 pair list with the physical repulsion onset at s * r_m (see stats.q6.build_runtime). Scales by donor-pair class
#: (hh both halide, ho one, oo neither), each the largest s with crystal median <= 0.2 kT and p95 <= 1 kT (E-1b).
Q6_DONOR_PAIR_SCALE = {"hh": 0.95, "ho": 0.90, "oo": 0.80}
Q6_PHYSICAL_ALL = bool(_profile.get("q6_physical_all"))


# ---------------------------------------------------------------------------
# Constants (mirror key_builder values for cross-validation parity)
# ---------------------------------------------------------------------------

#: BO threshold below which a bond is not a real ligand bond.
_BO_THRESHOLD: float = 0.5

#: BO thresholds for hybridisation (Q5 type map, mirrors key_builder).
_BO_AROM = 1.5
_BO_DBL = 2.0
_BO_AROM_TOL = 0.15
_BO_DBL_TOL = 0.15
_HYBRIDISE_ELEMS: FrozenSet[str] = frozenset({"C", "N", "O", "S", "P"})


# ---------------------------------------------------------------------------
# Internal helpers (CPU, called only during __init__)
# ---------------------------------------------------------------------------


def _build_pos_map(mol: "TMComplex") -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build bidirectional mapping between atom.idx (CSD 1-based) and position
    in mol.atoms list (0-based index into X[:, pos_idx, :]).
    """
    idx_to_pos: Dict[int, int] = {}
    pos_to_idx: Dict[int, int] = {}
    for pos, atom in enumerate(mol.atoms):
        idx_to_pos[atom.idx] = pos
        pos_to_idx[pos] = atom.idx
    return idx_to_pos, pos_to_idx


def _build_type_map(mol: "TMComplex") -> Dict[int, str]:
    """Local bond type per atom for Q5 keys: the SAME function the extractor uses."""
    from jacobik.stats.keys import _build_type_map as _shared

    return _shared(mol)


def _q4_valid_pairs(mask: torch.Tensor):
    """Static list of (a, b, pair_valid_float[1, n_q4]) for q4's pair loop, skipping empty pairs."""
    out = []
    max_n = mask.shape[1]
    for a_i in range(max_n):
        for b_i in range(a_i + 1, max_n):
            pv = mask[:, a_i] & mask[:, b_i]
            if bool(pv.any()):
                out.append((a_i, b_i, pv.float()[None, :]))  # (1, n_q4)
    return out


#: halide elements for the donor-pair classes of q6_donor_pairs
_Q6_HALIDES = frozenset({"F", "Cl", "Br", "I"})


def q6_donor_class(e1: str, e2: str) -> str:
    """Donor-pair class for q6_donor_pairs: 'hh' (both halide), 'ho' (one halide), 'oo' (neither)."""
    n = (e1 in _Q6_HALIDES) + (e2 in _Q6_HALIDES)
    return "hh" if n == 2 else ("ho" if n == 1 else "oo")


def _q6_donor_pair_scales(mol: "TMComplex", all_nb: Dict[int, List[int]], scale) -> Dict[Tuple[int, int], float]:
    """{(lo, hi) atom.idx: s} for kappa-kappa donor pairs on different ligands whose elements have repulsion
    parameters. ``scale``: a float (every pair) or a dict class -> s with classes 'hh', 'ho', 'oo' (q6_donor_class).
    Ligands are the connected components of the ligand graph (metal removed). Pairs without parameters are logged and
    left out."""
    from jacobik.stats.steric import rep_pair_params

    comp: Dict[int, int] = {}
    for a0 in all_nb:
        if a0 in comp:
            continue
        comp[a0] = a0
        stack = [a0]
        while stack:
            u = stack.pop()
            for v in all_nb.get(u, ()):
                if v not in comp:
                    comp[v] = a0
                    stack.append(v)
    kappa = [site.donor_indices[0] for site in mol.coord_sites if not site.is_haptic]
    sym = {a.idx: a.symbol for a in mol.atoms}
    out: Dict[Tuple[int, int], float] = {}
    missing = set()
    for k, i in enumerate(kappa):
        for j in kappa[k + 1:]:
            if comp.get(i, i) == comp.get(j, j):
                continue
            s_ij = float(scale[q6_donor_class(sym[i], sym[j])]) if isinstance(scale, dict) else float(scale)
            if rep_pair_params(sym[i], sym[j], s_ij) is None:
                missing.add(tuple(sorted((sym[i], sym[j]))))
                continue
            out[(min(i, j), max(i, j))] = s_ij
    if missing:
        log.info("q6_donor_pairs: element pairs without repulsion parameters left out: %s", sorted(missing))
    return out


def _acos_exact_with_safe_grad(cosine: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Return the exact acos value while differentiating through an interior clamp."""
    exact = torch.acos(cosine.clamp(-1.0, 1.0))
    safe = torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps))
    return safe + (exact - safe).detach()


class _GeometryEvalMixin:
    """Differentiable Q1–Q6 geometric extraction methods inherited by ``GeometryExtractor``."""

    # ------------------------------------------------------------------
    # Internal: chelate bond set (mirrors key_builder._chelate_ring_bonds)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chelate_bonds(
        mol: "TMComplex",
        G: nx.Graph,
    ) -> Set[Tuple[int, int]]:
        """
        Return the set of canonical (min_idx, max_idx) atom.idx bond keys that
        lie on a shortest chelate path between two donors of the same ligand.
        Mirrors key_builder._chelate_ring_bonds exactly.
        """
        _CHELATE_LONG_MAX = 8

        def _site_donor_indices(site) -> List[int]:
            return list(site.donor_indices)

        def _shortest_between(a_indices, b_indices, G):
            best = 10_000
            for a in a_indices:
                for b in b_indices:
                    if not G.has_node(a) or not G.has_node(b):
                        continue
                    if a == b:
                        return 0
                    try:
                        d = nx.shortest_path_length(G, a, b)
                        if d < best:
                            best = d
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
            return best

        ring_bonds: Set[Tuple[int, int]] = set()
        for lig in mol.ligands:
            lig_set = set(lig.atom_indices)
            lig_sites = [s for s in mol.coord_sites if any(d in lig_set for d in s.donor_indices)]
            if len(lig_sites) < 2:
                continue
            for ii in range(len(lig_sites)):
                for jj in range(ii + 1, len(lig_sites)):
                    sa, sb = lig_sites[ii], lig_sites[jj]
                    path_len = _shortest_between(
                        _site_donor_indices(sa),
                        _site_donor_indices(sb),
                        G,
                    )
                    if path_len > _CHELATE_LONG_MAX or path_len == 10_000:
                        continue
                    found = False
                    for a_idx in sa.donor_indices:
                        if found:
                            break
                        for b_idx in sb.donor_indices:
                            if not G.has_node(a_idx) or not G.has_node(b_idx):
                                continue
                            try:
                                path = nx.shortest_path(G, a_idx, b_idx)
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                continue
                            for step in range(len(path) - 1):
                                u, v = path[step], path[step + 1]
                                ring_bonds.add((min(u, v), max(u, v)))
                            found = True
                            break
        return ring_bonds

    # ------------------------------------------------------------------
    # Public differentiable methods
    # ------------------------------------------------------------------

    def donor_positions(self, X: torch.Tensor) -> torch.Tensor:
        """
        Effective donor position for each coordination site.

        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        (B, n_donors, 3) — in coord_sites order
        """
        B = X.shape[0]
        parts: List[torch.Tensor] = []

        kappa_ptr = 0
        haptic_ptr = 0

        for s_i in range(self.n_donors):
            if not self._donor_is_haptic[s_i]:
                pos_idx = self._kappa_pos_indices[kappa_ptr]
                pos = X[:, pos_idx, :]  # (B, 3)
                kappa_ptr += 1
            else:
                face_pos = X.index_select(1, self._haptic_idx(haptic_ptr))  # (B, n_face, 3)
                pos = face_pos.mean(dim=1)  # (B, 3)
                haptic_ptr += 1
            parts.append(pos.unsqueeze(1))  # (B, 1, 3)

        if parts:
            return torch.cat(parts, dim=1)  # (B, n_donors, 3)
        return X.new_zeros(B, 0, 3)

    def q1_distances(
        self, X: torch.Tensor, donor_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Metal–donor distances for all coordination sites.

        Parameters
        ----------
        X : (B, N, 3)
        donor_pos : optional precomputed self.donor_positions(X)

        Returns
        -------
        (B, n_donors)  — Euclidean distances in Å
        """
        metal = X[:, self.metal_pos_idx, :]  # (B, 3)
        donor = donor_pos if donor_pos is not None else self.donor_positions(X)  # (B, n_donors, 3)
        diff = donor - metal[:, None, :]  # (B, n_donors, 3)
        return torch.linalg.norm(diff, dim=-1)  # (B, n_donors)

    def q2_dmd_angles(
        self, X: torch.Tensor, donor_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        All pairwise D–M–D angles in radians.

        Batched: one bmm gives the (n_d x n_d) cosine matrix; the i<j upper
        triangle is extracted with the precomputed _q2_pair_idx.

        Parameters
        ----------
        X : (B, N, 3)
        donor_pos : optional precomputed self.donor_positions(X)

        Returns
        -------
        (B, n_pairs)  where n_pairs = n_donors*(n_donors-1)//2, in i<j order
        """
        n_d = self.n_donors
        if n_d < 2:
            return X.new_zeros(X.shape[0], 0)

        metal = X[:, self.metal_pos_idx, :]  # (B, 3)
        donors = donor_pos if donor_pos is not None else self.donor_positions(X)  # (B, n_d, 3)
        vecs = donors - metal[:, None, :]  # (B, n_d, 3)
        vecs = _safe_normalize(vecs, dim=-1)  # (B, n_d, 3)

        # Full pairwise cosine matrix in one matmul: (B, n_d, n_d)
        cos_all = torch.bmm(vecs, vecs.transpose(1, 2)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angles = torch.acos(cos_all)  # (B, n_d, n_d)

        # Extract upper triangle (i < j) using the precomputed boolean mask
        return angles.reshape(angles.shape[0], n_d * n_d).index_select(1, self._q2_pair_idx)  # (B, n_pairs)

    def q3_approach_angles(self, X: torch.Tensor, haptic: bool = True) -> torch.Tensor:
        """
        Donor approach angles for eligible kappa and haptic donors.

        Parameters
        ----------
        X : (B, N, 3)
        haptic : bool
            When False, skip the haptic face-normal SVD (used when caller only
            consumes kappa Q3 columns, e.g. inside CUDA-graph capture).

        Returns
        -------
        (B, n_q3_eligible)  in radians
        """
        B = X.shape[0]
        n_out = sum(self._q3_eligible)
        out = X.new_zeros(B, n_out)
        if n_out == 0:
            return out

        metal = X[:, self.metal_pos_idx, :]  # (B, 3)

        # --- Kappa Q3 ---
        n_k = self._q3_sub_pos.shape[0]
        if n_k > 0 and self._q3_sub_pos.shape[1] > 0:
            kappa = gather_atoms(X, self._q3_kappa_pos_t)  # (B, n_k, 3)
            sub_pos = gather_atoms(X, self._q3_sub_pos)  # (B, n_k, max_k, 3)
            mask = self._q3_sub_mask.float()  # (n_k, max_k)

            mask_exp = mask[None, :, :, None]  # (1, n_k, max_k, 1)
            count = mask.sum(dim=-1).clamp(min=1.0)  # (n_k,)
            centroid = (sub_pos * mask_exp).sum(dim=2) / count[None, :, None]  # (B, n_k, 3)

            v_M = metal[:, None, :] - kappa  # (B, n_k, 3)
            v_C = centroid - kappa  # (B, n_k, 3)

            v_M = _safe_normalize(v_M, dim=-1)
            v_C = _safe_normalize(v_C, dim=-1)
            cos_t = (v_M * v_C).sum(dim=-1)
            out.scatter_(
                1,
                self._q3_kappa_out_idx.unsqueeze(0).expand(B, -1),
                _acos_exact_with_safe_grad(cos_t),
            )

        # --- Haptic Q3 (Tilt) ---
        n_h = self._q3_haptic_pos.shape[0]
        if haptic and n_h > 0 and self._q3_haptic_pos.shape[1] > 0:
            face_pos = gather_atoms(X, self._q3_haptic_pos)  # (B, n_h, max_f, 3)
            mask = self._q3_haptic_mask.float()  # (n_h, max_f)

            mask_exp = mask[None, :, :, None]  # (1, n_h, max_f, 1)
            count = mask.sum(dim=-1).clamp(min=1.0)  # (n_h,)
            centroid = (face_pos * mask_exp).sum(dim=2) / count[None, :, None]  # (B, n_h, 3)

            centered = face_pos - centroid.unsqueeze(2)  # (B, n_h, max_f, 3)

            # Face normal = right-singular vector of the smallest singular value.
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            normal = Vh[:, :, -1, :]  # (B, n_h, 3)

            mc = centroid - metal[:, None, :]  # (B, n_h, 3)
            mc_norm = _safe_normalize(mc, dim=-1)  # (B, n_h, 3)

            # Orient normal toward metal
            dot_nm = (normal * mc_norm).sum(dim=-1, keepdim=True)  # (B, n_h, 1)
            normal = torch.where(dot_nm < 0, -normal, normal)

            # Tilt angle
            cos_t = (normal * mc_norm).sum(dim=-1)
            out.scatter_(
                1,
                self._q3_haptic_out_idx.unsqueeze(0).expand(B, -1),
                _acos_exact_with_safe_grad(cos_t),
            )

        return out  # (B, n_q3_eligible)

    def q4_pyramidalization(self, X: torch.Tensor) -> torch.Tensor:
        """
        Mean angle over all pairs in S={M, X₁, ..., Xₖ} at donor D for
        eligible donors (k≥2 heavy substituents).

        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        (B, n_q4)  in degrees
        """
        B = X.shape[0]
        if self.n_q4 == 0:
            return X.new_zeros(B, 0)

        # S member positions: (B, n_q4, max_n, 3)
        s_pos = gather_atoms(X, self._q4_s_pos)  # (B, n_q4, max_n, 3)
        donor = gather_atoms(X, self._q4_donor_pos)  # (B, n_q4, 3)
        n_q4 = self.n_q4

        # Vectors from donor to each S member: (B, n_q4, max_n, 3)
        vecs = s_pos - donor[:, :, None, :]

        # Normalise (zero-vectors left as-is; mask handles them)
        norms = torch.linalg.norm(vecs, dim=-1, keepdim=True).clamp(min=1e-8)
        unit = vecs / norms  # (B, n_q4, max_n, 3)

        # Sum all C(max_n, 2) pair angles, masking invalid entries
        sigma_theta = X.new_zeros(B, n_q4)
        # Valid (a_i, b_i) pairs are precomputed (static mask): no host sync per step.
        for a_i, b_i, pair_valid_f in self._q4_pair_valid:
            cos_t = (unit[:, :, a_i, :] * unit[:, :, b_i, :]).sum(dim=-1)
            cos_t = cos_t.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            angle_deg = torch.acos(cos_t) * (180.0 / math.pi)  # (B, n_q4)
            # Zero out invalid pairs
            sigma_theta = sigma_theta + angle_deg * pair_valid_f

        # Mean angle over the C(n,2) pairs.
        n_valid = self._q4_s_mask.float().sum(dim=-1)  # (n_q4,) = n per donor
        n_pairs_ = (n_valid * (n_valid - 1) / 2.0).clamp(min=1.0)  # C(n,2)
        return sigma_theta / n_pairs_[None, :]  # (B, n_q4)

    def q5_torsions(self, X: torch.Tensor) -> torch.Tensor:
        """
        Rotatable bond dihedral angles in radians.

        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        (B, n_q5)  in radians
        """
        if self.n_q5 == 0:
            return X.new_zeros(X.shape[0], 0)

        p_i = gather_atoms(X, self._q5_i)  # (B, n_q5, 3)
        p_j = gather_atoms(X, self._q5_j)
        p_k = gather_atoms(X, self._q5_k)
        p_l = gather_atoms(X, self._q5_l)

        # kernels.dihedral expects (B, 3) per call; reshape to process all joints
        # simultaneously by treating (B * n_q5) as the effective batch.
        B, n_q5 = p_i.shape[0], p_i.shape[1]
        p_i_flat = p_i.reshape(B * n_q5, 3)
        p_j_flat = p_j.reshape(B * n_q5, 3)
        p_k_flat = p_k.reshape(B * n_q5, 3)
        p_l_flat = p_l.reshape(B * n_q5, 3)

        phi_flat = _kernel_dihedral(p_i_flat, p_j_flat, p_k_flat, p_l_flat)
        return phi_flat.reshape(B, n_q5)

    def _haptic_idx(self, i: int) -> torch.Tensor:
        """Face atom indices of haptic site i as a tensor on the solver device (built once; indexing with the
        Python list rebuilt and uploaded an index tensor on every step)."""
        cache = self.__dict__.setdefault("_haptic_pos_t", {})
        t = cache.get(i)
        if t is None:
            t = cache[i] = torch.tensor(self._haptic_pos_lists[i], dtype=torch.long, device=self.device)
        return t

    def q6_contacts(self, X: torch.Tensor) -> torch.Tensor:
        """
        Non-bonded contact distances for all precomputed 1,4+ pairs.

        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        (B, n_q6)  distances in Å
        """
        if self.n_q6 == 0:
            return X.new_zeros(X.shape[0], 0)

        pos_a = gather_atoms(X, self._q6_a)  # (B, n_q6, 3)
        pos_b = gather_atoms(X, self._q6_b)  # (B, n_q6, 3)
        diff = pos_a - pos_b  # (B, n_q6, 3)
        return torch.linalg.norm(diff, dim=-1)  # (B, n_q6)


class GeometryExtractor(_GeometryEvalMixin):
    """
    Precomputes all static structural indices from a TMComplex and exposes
    differentiable methods for extracting Q1–Q6 quantities from a batched
    position tensor X : (B, N, 3).

    Parameters
    ----------
    mol    : TMComplex with populated coord_sites, donor_labels, ligands,
             bonds, and (for Q5) pivot_chains.
    device : torch.device — where to place index tensors and where X will live.

    Attributes (all on ``device``)
    ------------------------------
    metal_pos_idx    : int — position index of metal in mol.atoms list
    n_atoms          : int — total number of atoms
    n_donors         : int — number of coordination sites (kappa + haptic)
    """

    def __init__(self, mol: "TMComplex", device: torch.device) -> None:
        self.device = device

        atom_map = {a.idx: a for a in mol.atoms}
        idx_to_pos, pos_to_idx = _build_pos_map(mol)

        self.n_atoms = len(mol.atoms)
        self.metal_pos_idx = idx_to_pos[mol.metal_idx]

        n_sites = len(mol.coord_sites)
        self.n_donors = n_sites
        self._donor_is_haptic: List[bool] = []
        self._kappa_pos_indices: List[int] = []  # pos_idx for each kappa site
        self._haptic_pos_lists: List[List[int]] = []  # pos_idx lists for haptic
        # Maps from site list position → kappa_list index or haptic_list index
        self._site_to_kappa: Dict[int, int] = {}
        self._site_to_haptic: Dict[int, int] = {}

        for s_i, site in enumerate(mol.coord_sites):
            if site.is_haptic:
                self._donor_is_haptic.append(True)
                face_pos = [idx_to_pos[d] for d in site.donor_indices if d in idx_to_pos]
                h_i = len(self._haptic_pos_lists)
                self._haptic_pos_lists.append(face_pos)
                self._site_to_haptic[s_i] = h_i
            else:
                self._donor_is_haptic.append(False)
                d_idx = site.donor_indices[0]
                k_i = len(self._kappa_pos_indices)
                self._kappa_pos_indices.append(idx_to_pos[d_idx])
                self._site_to_kappa[s_i] = k_i

        from jacobik.topology.graph import get_heavy_ligand_neighbors, get_ligand_neighbors

        heavy_nb = get_heavy_ligand_neighbors(
            mol.atoms, mol.bonds, mol.metal_idx, bo_threshold=_BO_THRESHOLD
        )
        all_nb = get_ligand_neighbors(
            mol.atoms, mol.bonds, mol.metal_idx, bo_threshold=_BO_THRESHOLD
        )
        self._q3_eligible: List[bool] = []

        # Kappa storage
        q3_kappa_sub_lists: List[List[int]] = []
        self._q3_kappa_site_indices: List[int] = []  # index in mol.coord_sites
        self._q3_kappa_donor_pos: List[int] = []  # pos_idx of the donor atom

        # Haptic storage
        q3_haptic_face_lists: List[List[int]] = []
        self._q3_haptic_site_indices: List[int] = []  # index in mol.coord_sites

        for s_i, site in enumerate(mol.coord_sites):
            if site.is_haptic:
                # For haptic, we need the heavy face atoms (no metal, no H)
                face_pos = []
                for d_idx in site.donor_indices:
                    if d_idx != mol.metal_idx:
                        sym = mol.get_atom(d_idx).symbol
                        if sym != "H" and d_idx in idx_to_pos:
                            face_pos.append(idx_to_pos[d_idx])
                if len(face_pos) >= 3:
                    self._q3_eligible.append(True)
                    q3_haptic_face_lists.append(face_pos)
                    self._q3_haptic_site_indices.append(s_i)
                else:
                    self._q3_eligible.append(False)
            else:
                d_idx = site.donor_indices[0]
                nbs = all_nb.get(d_idx, [])
                subs_pos = [idx_to_pos[n] for n in nbs if n in idx_to_pos]
                if len(subs_pos) >= 1:
                    self._q3_eligible.append(True)
                    q3_kappa_sub_lists.append(subs_pos)
                    self._q3_kappa_site_indices.append(s_i)
                    self._q3_kappa_donor_pos.append(idx_to_pos[d_idx])
                else:
                    self._q3_eligible.append(False)

        # -- Build Kappa Tensors --
        n_kappa_q3 = len(q3_kappa_sub_lists)
        max_k_q3 = max((len(s) for s in q3_kappa_sub_lists), default=0)
        if n_kappa_q3 > 0 and max_k_q3 > 0:
            q3_pos = torch.zeros(n_kappa_q3, max_k_q3, dtype=torch.long)
            q3_mask = torch.zeros(n_kappa_q3, max_k_q3, dtype=torch.bool)
            for i, subs in enumerate(q3_kappa_sub_lists):
                for j, p in enumerate(subs):
                    q3_pos[i, j] = p
                    q3_mask[i, j] = True
            self._q3_sub_pos = q3_pos.to(device)
            self._q3_sub_mask = q3_mask.to(device)
            self._q3_kappa_pos_t = torch.tensor(
                self._q3_kappa_donor_pos, dtype=torch.long, device=device
            )
            self._q3_kappa_out_idx = torch.tensor(
                [self._q3_eligible[:si].count(True) for si in self._q3_kappa_site_indices],
                dtype=torch.long,
                device=device,
            )
        else:
            self._q3_sub_pos = torch.zeros(n_kappa_q3, 0, dtype=torch.long, device=device)
            self._q3_sub_mask = torch.zeros(n_kappa_q3, 0, dtype=torch.bool, device=device)
            self._q3_kappa_pos_t = torch.zeros(0, dtype=torch.long, device=device)
            self._q3_kappa_out_idx = torch.zeros(0, dtype=torch.long, device=device)

        # -- Build Haptic Tensors --
        n_haptic_q3 = len(q3_haptic_face_lists)
        max_f_q3 = max((len(f) for f in q3_haptic_face_lists), default=0)
        if n_haptic_q3 > 0 and max_f_q3 > 0:
            q3_h_pos = torch.zeros(n_haptic_q3, max_f_q3, dtype=torch.long)
            q3_h_mask = torch.zeros(n_haptic_q3, max_f_q3, dtype=torch.bool)
            for i, face in enumerate(q3_haptic_face_lists):
                for j, p in enumerate(face):
                    q3_h_pos[i, j] = p
                    q3_h_mask[i, j] = True
            self._q3_haptic_pos = q3_h_pos.to(device)
            self._q3_haptic_mask = q3_h_mask.to(device)
            self._q3_haptic_out_idx = torch.tensor(
                [self._q3_eligible[:si].count(True) for si in self._q3_haptic_site_indices],
                dtype=torch.long,
                device=device,
            )
        else:
            self._q3_haptic_pos = torch.zeros(n_haptic_q3, 0, dtype=torch.long, device=device)
            self._q3_haptic_mask = torch.zeros(n_haptic_q3, 0, dtype=torch.bool, device=device)
            self._q3_haptic_out_idx = torch.zeros(0, dtype=torch.long, device=device)

        q4_donor_pos: List[int] = []
        q4_s_lists: List[List[int]] = []

        for s_i, site in enumerate(mol.coord_sites):
            if site.is_haptic:
                continue
            d_idx = site.donor_indices[0]
            nbs = all_nb.get(d_idx, [])
            if len(nbs) < 2:
                continue
            # S = metal + substituents
            s_pos = [idx_to_pos[mol.metal_idx]] + [idx_to_pos[n] for n in nbs if n in idx_to_pos]
            q4_donor_pos.append(idx_to_pos[d_idx])
            q4_s_lists.append(s_pos)

        self.n_q4 = len(q4_s_lists)
        max_n_q4 = max((len(s) for s in q4_s_lists), default=0)

        if self.n_q4 > 0 and max_n_q4 > 0:
            q4_s = torch.zeros(self.n_q4, max_n_q4, dtype=torch.long)
            q4_m = torch.zeros(self.n_q4, max_n_q4, dtype=torch.bool)
            q4_d = torch.tensor(q4_donor_pos, dtype=torch.long)
            for i, slist in enumerate(q4_s_lists):
                for j, p in enumerate(slist):
                    q4_s[i, j] = p
                    q4_m[i, j] = True
            self._q4_s_pos = q4_s.to(device)
            self._q4_s_mask = q4_m.to(device)
            self._q4_pair_valid = _q4_valid_pairs(self._q4_s_mask)
            self._q4_donor_pos = q4_d.to(device)
        else:
            self._q4_s_pos = torch.zeros(0, 0, dtype=torch.long, device=device)
            self._q4_s_mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
            self._q4_pair_valid = []
            self._q4_donor_pos = torch.zeros(0, dtype=torch.long, device=device)

        type_map = _build_type_map(mol)
        seen_bonds: Set[Tuple[int, int]] = set()
        q5_quads: List[Tuple[int, int, int, int]] = []  # pos indices (i,j,k,l)
        self._q5_keys: List[tuple] = []  # (type_j, type_k, type_i, type_l, ring_ctx)

        from jacobik.topology.graph import build_heavy_ligand_graph
        from jacobik.stats.keys import q5_context, q5_flanks, q5_symmetry_fn

        G_lig = build_heavy_ligand_graph(
            mol.atoms, mol.bonds, mol.metal_idx, bo_threshold=_BO_THRESHOLD
        )
        chelate_bonds = self._build_chelate_bonds(mol, G_lig)

        for lig in mol.ligands:
            for chain in lig.pivot_chains:
                for joint in chain.joints:
                    j_atom = joint.atom_i
                    k_atom = joint.atom_j
                    bond_key = (min(j_atom, k_atom), max(j_atom, k_atom))
                    if bond_key in seen_bonds:
                        continue
                    seen_bonds.add(bond_key)

                    i_atom, l_atom = q5_flanks(
                        j_atom, k_atom, heavy_nb, all_nb, atom_map, type_map
                    )
                    if i_atom is None:
                        continue

                    for a in (i_atom, j_atom, k_atom, l_atom):
                        if a not in idx_to_pos:
                            break
                    else:
                        ring_ctx = q5_context("chelate" if bond_key in chelate_bonds else "free",
                                              j_atom, k_atom, heavy_nb, type_map)
                        sym_i = atom_map[i_atom].symbol if i_atom in atom_map else "?"
                        sym_j = atom_map[j_atom].symbol if j_atom in atom_map else "?"
                        sym_k = atom_map[k_atom].symbol if k_atom in atom_map else "?"
                        sym_l = atom_map[l_atom].symbol if l_atom in atom_map else "?"
                        type_i = type_map.get(i_atom, sym_i)
                        type_j = type_map.get(j_atom, sym_j)
                        type_k = type_map.get(k_atom, sym_k)
                        type_l = type_map.get(l_atom, sym_l)
                        q5_quads.append(
                            (
                                idx_to_pos[i_atom],
                                idx_to_pos[j_atom],
                                idx_to_pos[k_atom],
                                idx_to_pos[l_atom],
                            )
                        )
                        self._q5_keys.append((type_j, type_k, type_i, type_l, ring_ctx))

        # Rotor symmetry numbers per torsion (j side, k side): the flank atom of an n-fold symmetric top is an
        # arbitrary pick among equivalent atoms, so the physical torsion is defined modulo 360/n (Q5 folding). Same
        # function as the extractor's data folding (stats/key_builder.q5_symmetry_fn).
        _sym = q5_symmetry_fn(mol, all_nb, type_map)
        self._q5_sym: List[Tuple[int, int]] = [_sym(pos_to_idx[pj], pos_to_idx[pk]) for (pi, pj, pk, pl) in q5_quads]

        self.n_q5 = len(q5_quads)
        if self.n_q5 > 0:
            q5_t = torch.tensor(q5_quads, dtype=torch.long, device=device)
            self._q5_i = q5_t[:, 0]
            self._q5_j = q5_t[:, 1]
            self._q5_k = q5_t[:, 2]
            self._q5_l = q5_t[:, 3]
        else:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            self._q5_i = self._q5_j = self._q5_k = self._q5_l = empty

        # Shared with stats extraction: pairs within Q6_EXCLUDE_MAX_BONDS bonds on
        # the full graph (H and metal-donor edges included) are topologically
        # constrained and are not scored as non-bonded contacts.
        from jacobik.topology.graph import donor_indices_of, q6_topological_exclusions

        close_pairs = q6_topological_exclusions(
            mol.atoms, mol.bonds, mol.metal_idx, donor_indices_of(mol),
            bo_threshold=_BO_THRESHOLD,
        )

        # Flag q6_donor_pairs: kappa-kappa donor pairs on DIFFERENT ligands (the close set excludes every donor-donor
        # pair) are scored by the repulsion with the scaled onset. Haptic face atoms and same-ligand pairs stay excluded.
        donor_scale_of: Dict[Tuple[int, int], float] = {}
        ligand_sep = None
        if Q6_PHYSICAL_ALL:
            from jacobik.stats.steric import ligand_separation
            ligand_sep = ligand_separation(mol)
        if Q6_DONOR_PAIR_SCALE is not None:
            donor_scale_of = _q6_donor_pair_scales(
                mol, all_nb, Q6_DONOR_PAIR_SCALE if isinstance(Q6_DONOR_PAIR_SCALE, dict) else float(Q6_DONOR_PAIR_SCALE))

        q6_pairs: List[Tuple[int, int]] = []
        q6_scale: List[Optional[float]] = []

        n_all = len(mol.atoms)
        for ii in range(n_all):
            ai = mol.atoms[ii]
            for jj in range(ii + 1, n_all):
                aj = mol.atoms[jj]
                # Canonical pair by atom.idx for close-pair lookup
                lo_idx = ai.idx if ai.idx < aj.idx else aj.idx
                hi_idx = aj.idx if ai.idx < aj.idx else ai.idx
                if (lo_idx, hi_idx) in donor_scale_of:
                    # The physical-all experiment still observes the registered 1-2 / 1-3 exclusion.
                    # The production donor-pair extension normally reopens donor-donor pairs, so guard it here.
                    if (ligand_sep is not None and ligand_sep.get(lo_idx, {}).get(hi_idx, 99) <= 2):
                        continue
                    q6_pairs.append((ii, jj))
                    q6_scale.append(donor_scale_of[(lo_idx, hi_idx)])
                    continue
                if (lo_idx, hi_idx) in close_pairs:
                    continue
                q6_pairs.append((ii, jj))  # store as pos indices
                q6_scale.append(None)

        #: per-pair onset scale of the donor-pair extension (None for every ordinary pair); read by stats.q6.build_runtime
        self.q6_donor_scale: List[Optional[float]] = q6_scale
        self.n_q6_donor = sum(1 for s_ in q6_scale if s_ is not None)
        #: positions of the Q6 pairs that are NOT donor-extension pairs (the beam's partial Q6 walls use these)
        self._q6_wall_idx = (None if self.n_q6_donor == 0 else
                             torch.tensor([p for p, s_ in enumerate(q6_scale) if s_ is None], dtype=torch.long, device=device))
        self.n_q6 = len(q6_pairs)
        if self.n_q6 > 0:
            q6_t = torch.tensor(q6_pairs, dtype=torch.long, device=device)
            self._q6_a = q6_t[:, 0]
            self._q6_b = q6_t[:, 1]
        else:
            self._q6_a = torch.zeros(0, dtype=torch.long, device=device)
            self._q6_b = torch.zeros(0, dtype=torch.long, device=device)

        # Element pair keys for each Q6 pair (for scorer.py lookup)
        # (polar H facing an acceptor -> "Hp"; key_builder.q6_pair_key, shared with the extractor)
        from jacobik.stats.keys import q6_pair_key, q6_polar_h

        _polar = q6_polar_h(mol, all_nb)
        self._q6_elem_keys: List[Tuple[str, str]] = [
            q6_pair_key(mol.atoms[ii].symbol, mol.atoms[jj].symbol,
                        mol.atoms[ii].idx in _polar, mol.atoms[jj].idx in _polar)
            for ii, jj in q6_pairs]

        log.debug(
            "GeometryExtractor: %d atoms, %d donors (%d kappa, %d haptic), "
            "%d Q4, %d Q5, %d Q6 pairs",
            self.n_atoms,
            self.n_donors,
            len(self._kappa_pos_indices),
            len(self._haptic_pos_lists),
            self.n_q4,
            self.n_q5,
            self.n_q6,
        )

        # Static i<j mask for the (n_donors, n_donors) D-M-D angle matrix.
        n_d = self.n_donors
        self._q2_pair_mask = torch.ones(n_d, n_d, dtype=torch.bool, device=device).triu(diagonal=1)
        # The same pairs as flat indices into the (n_d * n_d) matrix, in the mask's row-major order: indexing with
        # the boolean mask runs a data-dependent `nonzero` on every call (a host sync on CUDA, and not recordable).
        self._q2_pair_idx = self._q2_pair_mask.reshape(-1).nonzero().flatten()
