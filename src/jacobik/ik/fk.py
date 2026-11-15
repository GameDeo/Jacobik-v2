"""
fk.py
=====
Batched, differentiable forward kinematics for the IK engine.

Joint list: one canonical BFS over rigid bodies from the ligand's donor-free
root body (one torsion joint per rotatable bond).  Donors that are branch
points additionally get "bend" joints (see ForwardKinematics) after the
n_joints_base torsion joints.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import numpy as np
import torch

from jacobik.ik.kernels import (  # noqa: F401
    axisangle_to_matrix,
    dihedral as _dihedral,
    dihedral_axis_from_quad,
    gather_atoms,
    rodrigues,
    wrap_angle as _wrap_angle,
)

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


# Production resolved configuration (hardcoded values)
_SMALL_RING_OPEN_ADJ = True


def _build_pos_map(mol: "TMComplex") -> Dict[int, int]:
    """atom.idx (1-based CSD) → position index (0-based in mol.atoms)."""
    return {a.idx: i for i, a in enumerate(mol.atoms)}


# ---------------------------------------------------------------------------
# ForwardKinematics
# ---------------------------------------------------------------------------



# Production resolved configuration (hardcoded values)
BRANCH_BENDS = True


class _FKEvalMixin:
    """FK evaluation, alignment, ring-closure projection, and Jacobian methods for ``ForwardKinematics``."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bend_seed_angles(self, X_ref: torch.Tensor) -> torch.Tensor:
        """
        Current dihedral angle of each bend joint in X_ref, used to seed the
        bend-joint columns of theta so they are a no-op at t=0.  Returns an
        empty (0,) tensor if no donor qualified for a bend joint.

        Parameters
        ----------
        X_ref : (N, 3) reference atom positions (single conformer, no batch dim)

        Returns
        -------
        (n_bend_joints,) angles in radians
        """
        if self.n_bend_joints == 0:
            return torch.zeros(0, device=X_ref.device)

        X = X_ref if X_ref.dim() == 2 else X_ref[0]
        angles = []
        for k in range(self.n_joints_base, self.n_joints_base + self.n_bend_joints):
            fi, pj, pk, fl = (
                self._flanking_i[k],
                self._pivot_j[k],
                self._pivot_k[k],
                self._flanking_l[k],
            )
            angles.append(_dihedral(X[fi][None, :], X[pj][None, :], X[pk][None, :], X[fl][None, :])[0])
        return torch.stack(angles, dim=0)

    def q5_alignment(self, ge, X_ref: torch.Tensor):
        """
        Map this ligand's base torsion joints onto the scorer's Q5 rows.

        The Q5 rows (GeometryExtractor, ligand pivot chains, canonical flank atoms) and the FK joints (root-body BFS,
        heaviest-neighbour flanks) enumerate the same rotatable bonds in a different order and may measure the
        dihedral with different flank atoms. Returns (rows, offsets): rows (n_joints_base,) long = Q5 row of the
        same bond (-1 if the bond has no Q5 row), offsets (n_joints_base,) radians = phi_FK - phi_Q5 on X_ref,
        so a Q5 value mu corresponds to the FK joint value mu + offset (the difference between two dihedrals about
        the same bond with different flanks is fixed by the rigid flanking groups).
        """
        J = self.n_joints_base
        X = X_ref if X_ref.dim() == 2 else X_ref[0]
        row_of = {}
        # one device->host transfer of the Q5 index table (was 2 single-element reads per row)
        q5 = torch.stack([ge._q5_i, ge._q5_j, ge._q5_k, ge._q5_l], dim=1).tolist() if int(ge._q5_j.shape[0]) else []
        for r, (_, a, b, _) in enumerate(q5):
            row_of[(min(a, b), max(a, b))] = r
        rows = torch.full((J,), -1, dtype=torch.long, device=X.device)
        offs = torch.zeros(J, dtype=X.dtype, device=X.device)
        for k in range(J):
            fi, pj, pk, fl = self._flanking_i[k], self._pivot_j[k], self._pivot_k[k], self._flanking_l[k]
            key = (min(int(pj), int(pk)), max(int(pj), int(pk)))
            r = row_of.get(key)
            phi_fk = _dihedral(X[fi][None], X[pj][None], X[pk][None], X[fl][None])[0]
            if r is None:
                offs[k] = phi_fk  # no statistics: callers keep the current dihedral
                continue
            qi, qj, qk, ql = q5[r]
            phi_q5 = _dihedral(X[qi][None], X[qj][None], X[qk][None], X[ql][None])[0]
            rows[k] = r
            offs[k] = _wrap_angle(phi_fk - phi_q5)
        return rows, offs

    def macro_seed_angles(self, P: torch.Tensor) -> torch.Tensor:
        """
        Macrocycle-joint dihedrals for conformers, used to seed the trailing
        macro columns of theta.

        P : (C, n_atoms_local, 3) conformer coordinates in ligand-local atom
            order (mol atom.idx - ligand.atom_indices[0]).
        Returns (C, n_macro_joints) radians.
        """
        cols = []
        for fi, pj, pk, fl in self.macro_quads_atom:
            q = [i - self._lig_first for i in (fi, pj, pk, fl)]
            cols.append(_dihedral(P[:, q[0]], P[:, q[1]], P[:, q[2]], P[:, q[3]]))
        if not cols:
            return P.new_zeros(P.shape[0], 0)
        return torch.stack(cols, dim=-1)

    def seam_d0(self, X_ref: torch.Tensor) -> torch.Tensor:
        """Seam distances of each row's own (closed) reference conformer: (B or 1, R)."""
        c = self._cl
        Xr = X_ref if X_ref.dim() == 3 else X_ref[None]
        xs = Xr[:, c["seam"]]
        return (xs[:, c["a"]] - xs[:, c["b"]]).norm(dim=-1)

    def closure_residual_jacobian(self, X: torch.Tensor, d0: torch.Tensor):
        """
        Seam residuals r = |x_a - x_b| - d0 (B, R) and their exact Jacobian with
        respect to the macro-joint dihedrals (B, R, K).

        Joints are internal (dihedral) coordinates, so raising one dihedral by
        d rigidly rotates its downstream atoms about the current joint axis and
        leaves every other dihedral unchanged: dx/dphi = u x (x - p) for
        downstream atoms, 0 otherwise.  No autograd is needed.
        """
        c = self._cl
        xs = X[:, c["seam"]]  # (B, S, 3)
        pj = X[:, c["pj"]]  # (B, K, 3)
        u = X[:, c["pk"]] - pj
        u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rel = xs[:, None, :, :] - pj[:, :, None, :]  # (B, K, S, 3)
        dx = torch.linalg.cross(u[:, :, None, :].expand_as(rel), rel, dim=-1)
        dx = dx * c["mask"][None, :, :, None]
        diff = xs[:, c["a"]] - xs[:, c["b"]]  # (B, R, 3)
        dist = diff.norm(dim=-1).clamp(min=1e-8)
        e = diff / dist[..., None]
        dpair = dx[:, :, c["a"]] - dx[:, :, c["b"]]  # (B, K, R, 3)
        J = torch.einsum("bic,bkic->bik", e, dpair)
        return dist - d0, J

    def closure_fused_step(self, X: torch.Tensor, d0: torch.Tensor, s_m: torch.Tensor) -> torch.Tensor:
        """
        Fold ring closure into an optimiser update without another FK pass.

        X   : (B, N, 3) current positions (already computed for the score);
        s_m : (B, K) proposed decrement of the macro dihedrals (theta -= s_m).
        Returns the adjusted decrement so the linearised seam residual after
        the update is ~0:  r' = r - J s_m + J corr = 0, with the damped
        minimum-norm corr = -J^T (J J^T + lam I)^-1 (r - J s_m).  Exact
        analytic J, fixed shapes, no autograd, no host sync.
        """
        with torch.no_grad():
            r, J = self.closure_residual_jacobian(X.detach(), d0)
            rhs = r - torch.einsum("brk,bk->br", J, s_m)
            eye = torch.eye(r.shape[-1], device=r.device, dtype=r.dtype)
            sol = torch.linalg.solve_ex(J @ J.transpose(-1, -2) + 1e-4 * eye, rhs[..., None], check_errors=False).result
            corr = -(J.transpose(-1, -2) @ sol)[..., 0]
            return s_m - corr.clamp(-0.5, 0.5)

    def closure_project(
        self,
        theta: torch.Tensor,
        X_ref: torch.Tensor,
        iters: int = 4,
        damping: float = 1e-4,
        max_step: float = 0.5,
    ):
        """
        Damped Gauss-Newton projection of the macro-joint dihedrals onto the
        ring-closure manifold: dphi = -J^T (J J^T + lam I)^-1 r, a fixed number
        of iterations, fixed shapes, no host synchronisation.  Returns
        (theta_projected, seam residual sum of squares (B,)).  Non-differentiable.
        """
        if self._cl is None:
            return theta, theta.new_zeros(theta.shape[0])
        c0 = self._macro_col0
        theta = theta.detach().clone()
        eye = torch.eye(self._cl["a"].shape[0], device=theta.device, dtype=theta.dtype)
        with torch.no_grad():
            d0 = self.seam_d0(X_ref)
            for _ in range(iters):
                X = self.apply(theta, X_ref)
                r, J = self.closure_residual_jacobian(X, d0)
                A = J @ J.transpose(-1, -2) + damping * eye
                # solve_ex(check_errors=False): no host-side info check, so no device sync.
                sol = torch.linalg.solve_ex(A, r[..., None], check_errors=False).result
                step = -(J.transpose(-1, -2) @ sol)[..., 0]
                c1 = c0 + self.n_macro_joints   # angle-bend columns (small donor rings) follow the ring joints
                theta[:, c0:c1] = _wrap_angle(theta[:, c0:c1] + step.clamp(-max_step, max_step))
            r, _ = self.closure_residual_jacobian(self.apply(theta, X_ref), d0)
        return theta, (r * r).sum(dim=-1)

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
        n = 6 + self.n_joints
        X = self._apply_internal(theta[:, :n] if theta.shape[1] > n else theta, X_ref, self.n_joints)
        if self.n_angle_bends and theta.shape[1] >= n + self.n_angle_bends:
            from jacobik.ik import angle_bends as AB

            X = self._bend_plan.apply(X, theta[:, n:n + self.n_angle_bends])
        return X

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
        transl = theta[:, 0:3]  # (B, 3)
        omega = theta[:, 3:6]  # (B, 3) axis-angle

        # Compute rotation matrices
        R = axisangle_to_matrix(omega)  # (B, 3, 3)

        # Pivot = centroid of donor atoms
        if self._donor_pos_t.numel() > 0:
            donor_pos = gather_atoms(X, self._donor_pos_t)  # (B, n_donors, 3)
            pivot = donor_pos.mean(dim=1)  # (B, 3)
        else:
            pivot = X.mean(dim=1)  # fallback: molecule centroid

        # Extract ligand atoms, translate to pivot frame, rotate, translate back
        lig = gather_atoms(X, self._lig_pos_t)  # (B, n_lig, 3)
        lig_centered = lig - pivot[:, None, :]  # (B, n_lig, 3)

        # Batch matrix multiply: (B, n_lig, 3) × (B, 3, 3)^T
        lig_rotated = torch.bmm(lig_centered, R.transpose(1, 2))  # (B, n_lig, 3)
        lig_new = lig_rotated + pivot[:, None, :] + transl[:, None, :]

        # In-place write is safe: _apply_internal passes a freshly cloned X.
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
        target_phi = theta[:, 6 + k]  # (B,) — absolute target torsion

        # Current dihedral: flanking_i - pivot_j - pivot_k - flanking_l, plus
        # the unit pivot_j -> pivot_k axis (reused as the rotation axis).
        P = X.index_select(1, self._quad_idx[k])  # (B, 4, 3)
        phi_cur, axis_unit = dihedral_axis_from_quad(P)  # (B,), (B, 3)

        # Delta: how much to rotate to reach the target
        delta = target_phi - phi_cur  # (B,)

        # Downstream atoms
        down_idx = self._downstream[k]  # (n_down,)
        if down_idx.numel() == 0:
            return X

        pts = gather_atoms(X, down_idx)  # (B, n_down, 3)
        pivot = P[:, 1]  # (B, 3) = p_j

        pts_rotated = rodrigues(
            pts, pivot, axis_unit, delta, axis_normalized=True
        )  # (B, n_down, 3)

        # Out-of-place single-kernel equivalent of clone + row overwrite.
        return X.index_copy(1, down_idx, pts_rotated)

    def jacobian_column_norm_sq(self, theta: torch.Tensor, X_ref: torch.Tensor) -> torch.Tensor:
        """Analytic diagonal of JᵀJ for this FK map, refreshed at the supplied pose.

        Cartesian columns use identity translation derivatives, the SO(3) left
        Jacobian for the axis-angle base rotation, and axis × lever-arm for each
        internal joint. This is O(N·D), uses fixed-shape tensor operations, and
        avoids one JVP per generalized coordinate.
        """
        B, width = theta.shape
        out = theta.new_zeros((B, width))
        nlig = int(self._lig_pos_t.numel())
        if nlig == 0:
            return out
        X_ref_b = X_ref.unsqueeze(0).expand(B, -1, -1) if X_ref.dim() == 2 else X_ref

        # The rigid translation affects every ligand atom by one Cartesian unit.
        out[:, :3] = float(nlig)
        Xbase = self._apply_base_transform(theta, X_ref_b.clone())
        X = Xbase
        joint_axes, joint_pivots, joint_down = [], [], []
        for k in range(self.n_joints):
            P = X.index_select(1, self._quad_idx[k])
            _, axis = dihedral_axis_from_quad(P)
            down = self._downstream[k]
            joint_axes.append(axis)
            joint_pivots.append(P[:, 1])
            joint_down.append(down)
            X = self._apply_torsion(theta, X, k)

        n = 6 + self.n_joints
        beta = theta[:, n:n + self.n_angle_bends] if self.n_angle_bends else None
        X_prebend = X
        Xfinal = self._bend_plan.apply(X, beta) if beta is not None and self._bend_plan is not None else X

        # Exact derivative of Exp(omega) applied to the final rigidly carried
        # coordinates: d(Rv)/domega = -[Rv]x J_l(omega).
        if self._donor_pos_t.numel() > 0:
            pivot = gather_atoms(X_ref_b, self._donor_pos_t).mean(dim=1)
        else:
            pivot = X_ref_b.mean(dim=1)
        # Base rotation is an equivariant rigid transform of the whole FK map,
        # including the bend axes built from the transformed coordinates.
        r = gather_atoms(Xfinal, self._lig_pos_t) - pivot[:, None, :] - theta[:, None, :3]
        omega = theta[:, 3:6]
        theta2 = (omega * omega).sum(dim=-1, keepdim=True)
        angle = torch.sqrt(theta2.clamp_min(1e-12))
        A = (1.0 - torch.cos(angle)) / theta2.clamp_min(1e-12)
        Bc = (angle - torch.sin(angle)) / (angle * theta2).clamp_min(1e-12)
        A = torch.where(theta2 < 1e-6, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, A)
        Bc = torch.where(theta2 < 1e-6, 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0, Bc)
        wx, wy, wz = omega.unbind(-1)
        z = torch.zeros_like(wx)
        W = torch.stack([z, -wz, wy, wz, z, -wx, -wy, wx, z], dim=-1).reshape(B, 3, 3)
        eye = torch.eye(3, dtype=theta.dtype, device=theta.device).expand(B, 3, 3)
        Jl = eye + A[..., None] * W + Bc[..., None] * (W @ W)
        rx, ry, rz = r.unbind(-1)
        zr = torch.zeros_like(rx)
        skew_r = torch.stack([zr, -rz, ry, rz, zr, -rx, -ry, rx, zr], dim=-1).reshape(B, nlig, 3, 3)
        jrot = -skew_r @ Jl[:, None, :, :]
        out[:, 3:6] = (jrot * jrot).sum(dim=(1, 2))

        for k, (axis, pv, down) in enumerate(zip(joint_axes, joint_pivots, joint_down)):
            if down.numel() == 0:
                continue
            lever = X_prebend.index_select(1, down) - pv[:, None, :]
            deriv = torch.linalg.cross(axis[:, None, :], lever, dim=-1)
            out[:, 6 + k] = (deriv * deriv).sum(dim=(1, 2))

        if beta is not None and self._bend_plan is not None:
            out[:, n:n + self.n_angle_bends] = self._bend_plan.jacobian_column_norm_sq(X_prebend, beta)
        return out


class ForwardKinematics(_FKEvalMixin):
    """
    Batched differentiable FK map for one ligand of a TMComplex.

    Joints are the ligand's rotatable bonds (n_joints_base torsion joints),
    followed by n_bend_joints "branch bend" joints: for a kappa donor with >= 2
    outgoing rotatable bonds, one joint per outgoing branch rotates only that
    branch's downstream subtree about the donor's incoming (root-ward) bond.
    The donor, its root-ward neighbour and sibling branches stay fixed.
    n_joints_base == n_joints when no donor qualifies.  Bend joints have no
    fitted Q5 statistics.

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
        angle_bends: bool = False,
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
        self._lig_first = min(ligand.atom_indices) if ligand.atom_indices else 0
        self._lig_pos_t = torch.tensor(self._lig_pos_indices, dtype=torch.long, device=device)

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

        decomp = ligand.decomposition  # _LigandGraph: body_of, pivot_bonds, rigid_bodies

        # body_of: atom.idx → body_index
        body_of: Dict[int, int] = decomp.body_of
        # rigid_bodies: list of sets of atom.idx per body
        body_sets: List[Set[int]] = decomp.rigid_bodies

        # body_adj_for_bfs[b] = list of (neighbour_body, atom_in_b, atom_in_neighbour)
        # for every rotatable bond crossing out of body b.
        body_adj_for_bfs: Dict[int, List[Tuple[int, int, int]]] = {}
        for rb in ligand.rotatable_bonds:
            u, v = rb.i, rb.j
            if u not in body_of or v not in body_of:
                continue
            bu, bv = body_of[u], body_of[v]
            if bu != bv:
                body_adj_for_bfs.setdefault(bu, []).append((bv, u, v))
                body_adj_for_bfs.setdefault(bv, []).append((bu, v, u))

        # Canonical BFS from the ligand's donor-free root body: an order-independent
        # joint list where every donor sits behind at least one real joint.
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
        from jacobik.stats.keys import _build_type_map, flank_side as _flank_side, linear_atoms

        _linear = linear_atoms(_build_type_map(mol))

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

            # Flanking atoms: heavy neighbours first, else H. At a linear (sp) atom the neighbours are collinear with
            # the bond and the measured dihedral is rounding noise (XECSON: 1e-6 A input shifts moved atoms 2.8 A), so
            # the flank comes from the first off-axis atom along the linear chain (key_builder.flank_side, shared with
            # the Q5 rows). A joint with no off-axis atom on one side (M-C#C-R, R-C#C-H) is not a torsion relative to
            # anything on that side: its rotation is the ligand's own base rotation about that axis, so it is left out.
            j_nbs = _flank_side(j_atom, k_atom, heavy_nb, all_nb, _linear)
            k_nbs = _flank_side(k_atom, j_atom, heavy_nb, all_nb, _linear)

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

        self.n_joints_base = len(pivot_j_list)

        # Bend joints reuse the parent joint's (flanking_i, pivot_j) and the child
        # joint's (pivot_k, downstream); _apply_torsion treats them like any joint.
        parent_joint_of: Dict[int, int] = {}
        child_joints_of: Dict[int, List[int]] = {}
        for jidx in range(len(pivot_j_list)):
            parent_joint_of.setdefault(pivot_k_list[jidx], jidx)
            child_joints_of.setdefault(pivot_j_list[jidx], []).append(jidx)

        kappa_donor_pos: List[int] = []
        for chain in ligand.pivot_chains:
            if chain.is_haptic:
                continue
            for d_idx in chain.donor_atoms:
                if d_idx in idx_to_pos:
                    kappa_donor_pos.append(idx_to_pos[d_idx])

        for d_pos in (kappa_donor_pos if BRANCH_BENDS else []):
            parent_idx = parent_joint_of.get(d_pos)
            children = child_joints_of.get(d_pos, [])
            if parent_idx is None or len(children) < 2:
                # d_pos is root itself (no parent joint), or not a real
                # branch point (<=1 outgoing branch -- the existing
                # incoming joint already gives it full solo control).
                continue
            for child_idx in children:
                flanking_i_list.append(flanking_i_list[parent_idx])
                pivot_j_list.append(pivot_j_list[parent_idx])
                pivot_k_list.append(d_pos)
                flanking_l_list.append(pivot_k_list[child_idx])
                downstream_lists.append(downstream_lists[child_idx])


        # ------------------------------------------------------------------
        # Macrocycle joints: after base and bend joints.  Each macrocycle is
        # opened at one bond; torsions on the remaining ring bonds (anchor
        # outward along two arms) move the atoms beyond them, and the
        # opening is closed by soft distance restraints (closure_pairs).
        # ------------------------------------------------------------------
        self.n_macro_joints = 0
        self._openings: List[Tuple[int, int]] = []   # closure bond of each jointed ring
        self.macro_quads_atom: List[Tuple[int, int, int, int]] = []
        closure_pairs: List[Tuple[int, int]] = []
        closure_d0: List[float] = []
        n_before_macro = len(pivot_j_list)
        macrocycles = getattr(ligand, "macrocycles", None) or []
        if macrocycles:
            import networkx as nx

            lig_set = set(ligand.atom_indices)
            Gl = nx.Graph()
            Gl.add_nodes_from(lig_set)
            for bnd in mol.bonds:
                if bnd.i in lig_set and bnd.j in lig_set:
                    Gl.add_edge(bnd.i, bnd.j)
            _bo = {}
            for bnd in mol.bonds:
                if bnd.i in lig_set and bnd.j in lig_set:
                    _bo[(min(bnd.i, bnd.j), max(bnd.i, bnd.j))] = float(bnd.order)

            def _single(u, v):
                return abs(_bo.get((min(u, v), max(u, v)), 1.0) - 1.0) < 0.25

            from jacobik.topology.decompose import _RING_JOINT_MIN_SIZE, RING_JOINTS
            min_n = _RING_JOINT_MIN_SIZE if RING_JOINTS else 8
            # bends this ligand gets (angle_bends): (fixed points, moved atoms) of each, to choose a ring opening no
            # bend disturbs
            _bend_moves = []
            if angle_bends:
                from jacobik.ik import angle_bends as _AB

                for sp in _AB.bend_specs(mol, ligand):
                    hinge = len(sp) > 4 and sp[4][0] == "hinge"
                    fixed = {sp[0], sp[1]} if hinge else {sp[0]}
                    moved = set(sp[3])
                    _bend_moves.append((fixed, moved))
            for ring in macrocycles:
                n = len(ring)
                if any(a not in idx_to_pos for a in ring) or n < min_n:
                    continue
                ring_set = set(ring)
                na = (n - 1) // 2
                nb_ = n - 1 - na
                # anchor: most heavy non-ring neighbours, among anchors whose opposite (closure) bond is single
                # (small donor rings may hold double bonds, which stay rigid and can never be the opening)
                cands = sorted(
                    range(n),
                    key=lambda q: -sum(
                        1
                        for nb in Gl.neighbors(ring[q])
                        if nb not in ring_set and atom_map[nb].symbol != "H"
                    ),
                )
                anchor = None
                if _bend_moves:
                    # Open the ring where no bend disturbs the seam: every bend either moves none of the seam atoms
                    # (the opening atoms and their neighbours) or moves all of them except points on its own axis, so
                    # swinging a fold or hinge never changes a seam distance and the closure never undoes it (COD: the
                    # folds the substituent-bend statistics guide). Any ring, any bridge length.
                    def _undisturbed(q):
                        u, v = ring[(q + na) % n], ring[(q + na + 1) % n]
                        if not _single(u, v):
                            return False
                        S = {u, v} | set(Gl.neighbors(u)) | set(Gl.neighbors(v))
                        return all(not (S & mv) or (S - fx) <= mv for fx, mv in _bend_moves)
                    anchor = next((q for q in cands if _undisturbed(q)), None)
                if anchor is None:
                    anchor = next((q for q in cands if _single(ring[(q + na) % n], ring[(q + na + 1) % n])), None)
                if anchor is None:
                    continue
                rr = ring[anchor:] + ring[:anchor]  # rr[0] is the anchor
                cl_u, cl_v = rr[na], rr[na + 1]  # closure bond
                self._openings.append((cl_u, cl_v))
                arm_a = [rr[m] for m in range(0, na + 1)]  # r0..r_na
                arm_b = [rr[0]] + [rr[-m] for m in range(1, nb_ + 1)]  # r0, r-1..
                arm_joints: List[List[Tuple[int, int, int, int]]] = []
                # Small donor rings (< 8 atoms or holding a non-single bond): the bond into each opening atom is a joint
                # too. It turns the opening atom's substituent frame, i.e. where the seam partner can sit; left out, that
                # torsion stays at its embed value and a 6-8 ring has no closure-preserving freedom left, so the ring
                # is locked in its seed fold (COD 2026-09-30: the bound tub unreachable from the free twist-boat).
                small = n < 8 or not all(_single(ring[q], ring[(q + 1) % n]) for q in range(n))
                last = 1 if (small and _SMALL_RING_OPEN_ADJ) else 2
                for arm in (arm_a, arm_b):
                    js: List[Tuple[int, int, int, int]] = []
                    for m in range(len(arm) - last):
                        j_at, k_at = arm[m], arm[m + 1]
                        if not _single(j_at, k_at):
                            continue  # a double bond in the ring stays rigid (no torsion joint)
                        fi = ring[(ring.index(j_at) - 1) % n]
                        if fi == k_at:
                            fi = ring[(ring.index(j_at) + 1) % n]
                        fl = ring[(ring.index(k_at) - 1) % n]
                        if fl == j_at:
                            fl = ring[(ring.index(k_at) + 1) % n]
                        js.append((fi, j_at, k_at, fl))
                    arm_joints.append(js)
                # Interleave the arms so nested (outer) joints follow inner ones.
                ja, jb = arm_joints
                order_ab = []
                for m in range(max(len(ja), len(jb))):
                    if m < len(ja):
                        order_ab.append(ja[m])
                    if m < len(jb):
                        order_ab.append(jb[m])
                for fi, j_at, k_at, fl in order_ab:
                    H = Gl.copy()
                    H.remove_edge(j_at, k_at)
                    H.remove_edge(cl_u, cl_v)
                    comp = nx.node_connected_component(H, k_at)
                    down_pos = sorted(idx_to_pos[a] for a in comp if a in idx_to_pos)
                    flanking_i_list.append(idx_to_pos[fi])
                    pivot_j_list.append(idx_to_pos[j_at])
                    pivot_k_list.append(idx_to_pos[k_at])
                    flanking_l_list.append(idx_to_pos[fl])
                    downstream_lists.append(down_pos)
                    self.macro_quads_atom.append((fi, j_at, k_at, fl))
                # Closure: the opening bond length plus the distance from each opening atom to EVERY other neighbour
                # (H included) of the other one. That fixes the whole relative pose of the two ends except the
                # dihedral about the opening bond. With only the two ring 1-3 distances, each end could still spin
                # about its ring bond, which bends its H (and other substituents) into the opening
                # (TINBIE: C-C-H 17 deg, +1500 kJ/mol).
                seam_pairs = [(cl_u, cl_v)]
                seam_pairs += [(cl_u, x) for x in sorted(Gl.neighbors(cl_v)) if x != cl_u]
                seam_pairs += [(cl_v, y) for y in sorted(Gl.neighbors(cl_u)) if y != cl_v]
                for a_, b_ in seam_pairs:
                    closure_pairs.append((idx_to_pos[a_], idx_to_pos[b_]))
                    closure_d0.append(
                        float(np.linalg.norm(atom_map[a_].position - atom_map[b_].position))
                    )
        self.n_macro_joints = len(pivot_j_list) - n_before_macro
        self.closure_pairs = closure_pairs
        self.closure_d0 = closure_d0

        self.n_joints = len(pivot_j_list)
        self.n_bend_joints = self.n_joints - self.n_joints_base - self.n_macro_joints

        # Static tensors for the closure Jacobian / projection (built once).
        # Seam atoms S, residual pairs as indices into S, and for every macro
        # joint the axis atoms plus a (K, S) mask of seam atoms it moves.
        self._macro_col0 = 6 + self.n_joints_base + self.n_bend_joints
        self._cl = None
        if closure_pairs and self.n_macro_joints > 0:
            seam = sorted({a for pr in closure_pairs for a in pr})
            sidx = {a: i for i, a in enumerate(seam)}
            mk = slice(self.n_joints_base + self.n_bend_joints, self.n_joints)
            down_sets = [set(d) for d in downstream_lists[mk]]
            mask = [[a in ds for a in seam] for ds in down_sets]
            dev = device
            self._cl = dict(
                seam=torch.tensor(seam, dtype=torch.long, device=dev),
                a=torch.tensor([sidx[a] for a, _ in closure_pairs], dtype=torch.long, device=dev),
                b=torch.tensor([sidx[b] for _, b in closure_pairs], dtype=torch.long, device=dev),
                d0=torch.tensor(closure_d0, dtype=torch.float32, device=dev),
                pj=torch.tensor(pivot_j_list[mk], dtype=torch.long, device=dev),
                pk=torch.tensor(pivot_k_list[mk], dtype=torch.long, device=dev),
                mask=torch.tensor(mask, dtype=torch.float32, device=dev),
            )

        if self.n_joints > 0:
            self._flanking_i = flanking_i_list
            self._pivot_j = pivot_j_list
            self._pivot_k = pivot_k_list
            self._flanking_l = flanking_l_list
            # Downstream index lists: variable length, kept as a list of tensors
            self._downstream = [
                torch.tensor(d, dtype=torch.long, device=device) for d in downstream_lists
            ]
            self._quad_idx = [
                torch.tensor([fi, pj, pk, fl], dtype=torch.long, device=device)
                for fi, pj, pk, fl in zip(
                    flanking_i_list, pivot_j_list, pivot_k_list, flanking_l_list
                )
            ]
        else:
            self._flanking_i = []
            self._pivot_j = []
            self._pivot_k = []
            self._flanking_l = []
            self._downstream = []
            self._quad_idx = []

        # Optional bond-angle bends on chelate backbones (ik/angle_bends.py): extra theta columns after
        # every other joint, applied after the normal FK. Not counted in n_joints.
        self._bend_plan = None
        self.n_angle_bends = 0
        if angle_bends:
            from jacobik.ik import angle_bends as AB

            _specs = AB.bend_specs(mol, ligand, openings=self._openings or None)
            if _specs:
                self._bend_specs = _specs
                self._bend_plan = AB.BendPlan(_specs, idx_to_pos, device)
                self._bend_k = torch.tensor(AB.stiffness(mol, _specs), dtype=torch.float32, device=device)
                self._bend_free = torch.tensor(AB.free_zone(_specs), dtype=torch.float32, device=device)
                self.n_angle_bends = self._bend_plan.K

        log.debug(
            "ForwardKinematics: ligand %d, %d atoms, %d joints",
            ligand.ligand_id,
            len(self._lig_pos_indices),
            self.n_joints,
        )
