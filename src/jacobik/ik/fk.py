"""
fk.py
=====
Batched, differentiable forward kinematics for the IK engine.

Version history
----------------
1.0.0 : initial release
2.3.0 : Joint list is now built from one canonical BFS over body_adj_for_bfs
        starting at ligand.decomposition.root_body (decompose.py 1.2.1),
        replacing the previous "iterate ligand.pivot_chains in donor order,
        dedupe bonds first-come-first-served" construction. That mechanism
        made joint direction -- and therefore which body ends up as the
        FK tree's un-correctable root -- an accidental function of donor
        list order, with no guarantee a donor would be avoided. See
        decompose.py 1.2.1 for the root-selection rule and the EDTA case
        that surfaced this (a donor atom/group measured moving exactly
        0.0000 A under any combination of the ligand's other torsions).
        Also hoists body_adj_for_bfs construction out of the per-donor
        loop it used to live in -- it doesn't depend on which donor is
        being processed, so rebuilding it once per donor was O(n_donors)
        wasted work for no benefit.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import torch

from jacobik.ik.kernels import axisangle_to_matrix, rodrigues
from jacobik.ik.kernels import dihedral as _dihedral

if TYPE_CHECKING:
    from jacobik.core.types import Ligand, TMComplex

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror geometry.py for consistency)
# ---------------------------------------------------------------------------

_BO_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Internal helpers (CPU, called only during __init__)
# ---------------------------------------------------------------------------


def _build_pos_map(mol: "TMComplex") -> Dict[int, int]:
    """atom.idx (1-based CSD) → position index (0-based in mol.atoms)."""
    return {a.idx: i for i, a in enumerate(mol.atoms)}


def _build_pos_map(mol: "TMComplex") -> Dict[int, int]:
    """atom.idx (1-based CSD) → position index (0-based in mol.atoms)."""
    return {a.idx: i for i, a in enumerate(mol.atoms)}


# ---------------------------------------------------------------------------
# ForwardKinematics
# ---------------------------------------------------------------------------


class ForwardKinematics:
    """
    Batched differentiable FK map for one ligand of a TMComplex.


    Parameters
    ----------
    mol    : fully assembled TMComplex (quantized bonds, decomposed ligands)
    ligand : one Ligand from mol.ligands (must have pivot_chains + decomposition)
    device : torch.device

    """

    def __init__(
        self,
        mol: "TMComplex",
        ligand: "Ligand",
        device: torch.device,
    ) -> None:
        self.device = device

        atom_map = {a.idx: a for a in mol.atoms}
        idx_to_pos = _build_pos_map(mol)
        from jacobik.topology.graph import (
            get_heavy_ligand_neighbors,
            get_ligand_neighbors,
            heaviest_heavy_neighbor,
        )

        heavy_nb = get_heavy_ligand_neighbors(
            mol.atoms, mol.bonds, mol.metal_idx, bo_threshold=_BO_THRESHOLD
        )
        all_nb = get_ligand_neighbors(
            mol.atoms, mol.bonds, mol.metal_idx, bo_threshold=_BO_THRESHOLD
        )

        # ------------------------------------------------------------------
        # Ligand atom position indices
        # ------------------------------------------------------------------
        self._lig_pos_indices: List[int] = [
            idx_to_pos[idx] for idx in ligand.atom_indices if idx in idx_to_pos
        ]
        self._lig_pos_t = torch.tensor(self._lig_pos_indices, dtype=torch.long, device=device)

        # See exp.md for detailed rationale
        donor_atom_idxs: List[int] = []
        for chain in ligand.pivot_chains:
            donor_atom_idxs.extend(chain.donor_atoms)
        # Deduplicate while preserving order
        seen: Set[int] = set()
        donor_atom_idxs_dedup = []
        for idx in donor_atom_idxs:
            if idx not in seen and idx in idx_to_pos:
                seen.add(idx)
                donor_atom_idxs_dedup.append(idx)
        self._donor_pos_indices: List[int] = [idx_to_pos[idx] for idx in donor_atom_idxs_dedup]
        if self._donor_pos_indices:
            self._donor_pos_t = torch.tensor(
                self._donor_pos_indices, dtype=torch.long, device=device
            )
        else:
            self._donor_pos_t = torch.zeros(0, dtype=torch.long, device=device)

        # See exp.md for detailed rationale
        decomp = ligand.decomposition  # _LigandGraph: body_of, pivot_bonds, rigid_bodies

        # body_of: atom.idx → body_index
        body_of: Dict[int, int] = decomp.body_of
        # rigid_bodies: list of sets of atom.idx per body
        body_sets: List[Set[int]] = decomp.rigid_bodies

        # Build body-level adjacency for downstream BFS.
        # body_adj_for_bfs[b] = list of (neighbour_body, atom_in_b, atom_in_neighbour)
        # for every rotatable bond crossing out of body b. Built ONCE.
        #
        # PERF (fk.py 2.3.0): previously rebuilt from scratch inside the
        # per-donor `for chain in ligand.pivot_chains` loop below -- i.e.
        # O(n_donors) redundant reconstructions of the same, donor-independent
        # structure. Hoisted out.
        body_adj_for_bfs: Dict[int, List[Tuple[int, int, int]]] = {}
        for rb in ligand.rotatable_bonds:
            u, v = rb.i, rb.j
            if u not in body_of or v not in body_of:
                continue
            bu, bv = body_of[u], body_of[v]
            if bu != bv:
                body_adj_for_bfs.setdefault(bu, []).append((bv, u, v))
                body_adj_for_bfs.setdefault(bv, []).append((bu, v, u))

        # BUGFIX (fk.py 2.3.0): the joint list (and therefore which body
        # ends up as the un-correctable root -- see decompose.py 1.2.1) used
        # to be built by iterating `ligand.pivot_chains` in donor-list order
        # and deduping bonds first-come-first-served, making the effective
        # root an accidental function of donor order. Replaced with one
        # canonical BFS from the ligand's explicitly-chosen, donor-free
        # root_body, giving an order-independent joint list where every
        # donor sits behind at least one real, position-changing joint.
        root_body: int = getattr(decomp, "root_body", 0)
        ordered_joints: List[Tuple[int, int]] = []  # (j_atom, k_atom) atom.idx pairs
        visited_bodies_root: Set[int] = {root_body}
        queue_root: deque = deque([root_body])
        while queue_root:
            curr = queue_root.popleft()
            for nb_body, j_atom, k_atom in sorted(body_adj_for_bfs.get(curr, [])):
                if nb_body in visited_bodies_root:
                    continue
                visited_bodies_root.add(nb_body)
                queue_root.append(nb_body)
                ordered_joints.append((j_atom, k_atom))

        seen_bonds: Set[Tuple[int, int]] = set()

        # Ordered lists parallel to the joint sequence
        flanking_i_list: List[int] = []
        pivot_j_list: List[int] = []  # pos indices
        pivot_k_list: List[int] = []
        flanking_l_list: List[int] = []
        downstream_lists: List[List[int]] = []  # pos indices per joint

        for j_atom, k_atom in ordered_joints:
            # Skip coordination bonds (metal-ligand, not a torsion DOF)
            if j_atom == mol.metal_idx or k_atom == mol.metal_idx:
                continue

            bond_key = (min(j_atom, k_atom), max(j_atom, k_atom))
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)

            # Flanking atoms — identical convention to geometry.py
            j_nbs = [n for n in heavy_nb.get(j_atom, []) if n != k_atom]
            k_nbs = [n for n in heavy_nb.get(k_atom, []) if n != j_atom]

            if not j_nbs:
                j_nbs = [n for n in all_nb.get(j_atom, []) if n != k_atom]
            if not k_nbs:
                k_nbs = [n for n in all_nb.get(k_atom, []) if n != j_atom]

            if not j_nbs or not k_nbs:
                continue

            fi_atom = heaviest_heavy_neighbor(j_nbs, atom_map)
            fl_atom = heaviest_heavy_neighbor(k_nbs, atom_map)

            for atom in (fi_atom, j_atom, k_atom, fl_atom):
                if atom not in idx_to_pos:
                    break
            else:
                # Downstream bodies: BFS from body_of[k_atom], not
                # allowed to cross back through body_of[j_atom].
                # This correctly handles branching chains.
                if k_atom not in body_of or j_atom not in body_of:
                    continue
                start_body = body_of[k_atom]
                forbid_body = body_of[j_atom]

                visited_bodies: Set[int] = {start_body}
                queue_b = [start_body]
                while queue_b:
                    curr_b = queue_b.pop()
                    for nb_b, _, _ in body_adj_for_bfs.get(curr_b, []):
                        if nb_b not in visited_bodies and nb_b != forbid_body:
                            visited_bodies.add(nb_b)
                            queue_b.append(nb_b)

                down_pos = sorted(
                    idx_to_pos[a_idx]
                    for b_idx in visited_bodies
                    for a_idx in body_sets[b_idx]
                    if a_idx in idx_to_pos
                )

                flanking_i_list.append(idx_to_pos[fi_atom])
                pivot_j_list.append(idx_to_pos[j_atom])
                pivot_k_list.append(idx_to_pos[k_atom])
                flanking_l_list.append(idx_to_pos[fl_atom])
                downstream_lists.append(down_pos)

        self.n_joints = len(pivot_j_list)

        if self.n_joints > 0:
            self._flanking_i = flanking_i_list
            self._pivot_j = pivot_j_list
            self._pivot_k = pivot_k_list
            self._flanking_l = flanking_l_list
            # Downstream index lists — kept as a Python list of tensors (variable length)
            self._downstream = [
                torch.tensor(d, dtype=torch.long, device=device) for d in downstream_lists
            ]
        else:
            self._flanking_i = []
            self._pivot_j = []
            self._pivot_k = []
            self._flanking_l = []
            self._downstream = []

        log.debug(
            "ForwardKinematics: ligand %d, %d atoms, %d joints",
            ligand.ligand_id,
            len(self._lig_pos_indices),
            self.n_joints,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        theta: torch.Tensor,
        X_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the full FK map: base transform then all J torsions.

        Parameters
        ----------
        theta : (B, 6 + J)
            DOF vector.  theta[:, 0:3] = translation, theta[:, 3:6] =
            axis-angle rotation, theta[:, 6:] = absolute torsion angles
            in radians.
        X_ref : (B, N, 3) or (N, 3)
            Reference atom positions.  If (N, 3), broadcast over batch.

        Returns
        -------
        X : (B, N, 3) — updated atom positions (only ligand atoms changed)
        """
        return self._apply_internal(theta, X_ref, self.n_joints)

    def partial_apply(
        self,
        theta: torch.Tensor,
        X_ref: torch.Tensor,
        n_joints: int,
    ) -> torch.Tensor:
        """
        Apply the base transform and only the first n_joints torsions.


        Parameters
        ----------
        theta    : (B, 6 + J)
        X_ref    : (B, N, 3) or (N, 3)
        n_joints : number of torsion joints to apply (0 ≤ n_joints ≤ J)

        Returns
        -------
        X : (B, N, 3)

        """
        n_joints = max(0, min(n_joints, self.n_joints))
        return self._apply_internal(theta, X_ref, n_joints)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _apply_internal(
        self,
        theta: torch.Tensor,
        X_ref: torch.Tensor,
        n_joints: int,
    ) -> torch.Tensor:
        """
        Core FK computation.  Applies base transform then the first
        n_joints torsions.
        """
        B = theta.shape[0]

        # Broadcast X_ref to (B, N, 3) if needed
        if X_ref.dim() == 2:
            X = X_ref.unsqueeze(0).expand(B, -1, -1).clone()
        else:
            X = X_ref.clone()

        if self._lig_pos_t.numel() == 0:
            return X  # empty ligand — nothing to do

        # ------------------------------------------------------------------
        # Step 1: Base rigid-body transform
        # ------------------------------------------------------------------
        X = self._apply_base_transform(theta, X)

        # ------------------------------------------------------------------
        # Step 2: Torsion loop
        # ------------------------------------------------------------------
        for k in range(n_joints):
            X = self._apply_torsion(theta, X, k)

        return X

    def _apply_base_transform(
        self,
        theta: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply theta[:, 0:3] (translation) and theta[:, 3:6] (axis-angle
        rotation) to all ligand atoms.
        """
        B = theta.shape[0]
        transl = theta[:, 0:3]  # (B, 3)
        omega = theta[:, 3:6]  # (B, 3) axis-angle

        # Compute rotation matrices
        R = axisangle_to_matrix(omega)  # (B, 3, 3)

        # Pivot = centroid of donor atoms
        if self._donor_pos_t.numel() > 0:
            donor_pos = X[:, self._donor_pos_t, :]  # (B, n_donors, 3)
            pivot = donor_pos.mean(dim=1)  # (B, 3)
        else:
            pivot = X.mean(dim=1)  # fallback: molecule centroid

        # Extract ligand atoms, translate to pivot frame, rotate, translate back
        lig = X[:, self._lig_pos_t, :]  # (B, n_lig, 3)
        lig_centered = lig - pivot[:, None, :]  # (B, n_lig, 3)

        # Batch matrix multiply: (B, n_lig, 3) × (B, 3, 3)^T
        lig_rotated = torch.bmm(lig_centered, R.transpose(1, 2))  # (B, n_lig, 3)
        lig_new = lig_rotated + pivot[:, None, :] + transl[:, None, :]

        # Scatter back into X (clone not needed — already cloned in _apply_internal)
        X = X.clone()
        X[:, self._lig_pos_t, :] = lig_new
        return X

    def _apply_torsion(
        self,
        theta: torch.Tensor,
        X: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Set joint k to the absolute torsion angle theta[:, 6+k].
        """
        B = theta.shape[0]
        target_phi = theta[:, 6 + k]  # (B,) — absolute target torsion

        fi = self._flanking_i[k]
        pj = self._pivot_j[k]
        pk = self._pivot_k[k]
        fl = self._flanking_l[k]

        # Current dihedral: flanking_i – pivot_j – pivot_k – flanking_l
        # kernels.dihedral expects (B, 3) per atom
        p_i = X[:, fi, :]  # (B, 3)
        p_j = X[:, pj, :]
        p_k = X[:, pk, :]
        p_l = X[:, fl, :]

        phi_cur = _dihedral(p_i, p_j, p_k, p_l)  # (B,)

        # Delta: how much to rotate to reach the target
        delta = target_phi - phi_cur  # (B,)

        # Rotation axis: normalised pivot_j → pivot_k vector
        axis = p_k - p_j  # (B, 3) — rodrigues normalises internally

        # Downstream atoms
        down_idx = self._downstream[k]  # (n_down,)
        if down_idx.numel() == 0:
            return X

        pts = X[:, down_idx, :]  # (B, n_down, 3)
        pivot = p_j  # (B, 3)

        pts_rotated = rodrigues(pts, pivot, axis, delta)  # (B, n_down, 3)

        X = X.clone()
        X[:, down_idx, :] = pts_rotated
        return X