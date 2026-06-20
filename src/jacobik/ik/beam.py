"""
beam.py
=======
GPU beam search over Q5 VMM torsion modes for IK seed generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from jacobik.ik.kernels import halfgauss_log_prob, vmm_log_prob

if TYPE_CHECKING:
    from jacobik.ik.fk import ForwardKinematics

log = logging.getLogger(__name__)

_EPS = 1e-7


# ---------------------------------------------------------------------------
# BeamSearch
# ---------------------------------------------------------------------------


class BeamSearch:
    """
    GPU beam search over Q5 torsion VMM modes.


    Parameters
    ----------
    fk         : ForwardKinematics for the ligand being searched
    beam_width : W — number of seeds to return (and maximum beam size)
    device     : torch device

    The Q5 and Q6 tensors are supplied per-call to ``search()`` rather than
    at construction time, because the scorer resolves them at its own
    __init__ and beam.py should not re-query stats.

    """

    def __init__(
        self,
        fk: "ForwardKinematics",
        beam_width: int,
        device: torch.device,
    ) -> None:
        self.fk = fk
        self.W = beam_width
        self.device = device
        self.J = fk.n_joints

        # Pre-compute partial Q6 pair sets: for joint k, which atom-index
        # pairs (new_body_pos, fixed_pos) should be clash-checked.
        # These are derived from fk's downstream index lists.
        self._q6_body_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = self._build_body_pairs()

        log.debug(
            "BeamSearch: W=%d, J=%d joints, device=%s",
            self.W,
            self.J,
            self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        X_ref: torch.Tensor,  # (N, 3) reference positions
        base_theta: torch.Tensor,  # (1, 6) or (W, 6) base DOF
        q5_means: torch.Tensor,  # (J, K_max) VMM means (radians)
        q5_kappas: torch.Tensor,  # (J, K_max) VMM kappas
        q5_weights: torch.Tensor,  # (J, K_max) VMM weights
        q6_mu: Optional[torch.Tensor],  # (n_q6,) clash wall positions
        q6_sigma: Optional[torch.Tensor],  # (n_q6,) clash wall widths
        q6_pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # (n_q6,), (n_q6,) atom pos-idx pairs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run beam search and return W seed torsion vectors.

        Parameters
        ----------
        X_ref      : (N, 3) reference atom positions (CPU or device)
        base_theta : (1, 6) or (W, 6) base DOF (translation + axis-angle).
                     If (1, 6), broadcast to all beam candidates.
        q5_means   : (J, K_max) per-joint VMM means in radians.
                     Padding with zeros is fine — weights handle it.
        q5_kappas  : (J, K_max) per-joint VMM concentration parameters.
        q5_weights : (J, K_max) per-joint VMM mixture weights.
                     Rows with fewer than K_max real components are
                     zero-padded; zero-weight components have no effect.
        q6_mu      : (n_q6,) or None — Q6 wall positions (Å).
        q6_sigma   : (n_q6,) or None — Q6 wall widths (Å).
        q6_pairs   : pair of index tensors (idx_a, idx_b), each (n_q6,),
                     giving the position indices of each Q6 pair.
                     If None or q6_mu is None, Q6 partial scoring is skipped.

        Returns
        -------
        beam_torsions : (W, J) — W seed torsion vectors in radians
        beam_scores   : (W,)  — cumulative Q5+Q6 score for each seed
        """
        W = self.W
        J = self.J
        device = self.device

        X_ref = X_ref.to(device)

        # Trivial case: no joints — return empty torsion matrix
        if J == 0:
            log.debug("BeamSearch.search: J=0, returning empty seeds")
            return (
                torch.zeros(W, 0, device=device),
                torch.zeros(W, device=device),
            )

        q5_means = q5_means.to(device)  # (J, K_max)
        q5_kappas = q5_kappas.to(device)
        q5_weights = q5_weights.to(device)

        use_q6 = (
            q6_mu is not None
            and q6_sigma is not None
            and q6_pairs is not None
            and len(self._q6_body_pairs) > 0
        )
        if use_q6:
            q6_mu = q6_mu.to(device)
            q6_sigma = q6_sigma.to(device)
            q6_idx_a = q6_pairs[0].to(device)  # (n_q6,)
            q6_idx_b = q6_pairs[1].to(device)

        # Base theta: broadcast to (W, 6)
        if base_theta.shape[0] == 1:
            base_theta = base_theta.expand(W, -1).to(device)  # (W, 6)
        else:
            base_theta = base_theta[:W].to(device)  # (W, 6)

        # Initialise beam
        beam = torch.zeros(W, J, device=device)  # (W, J) torsion DOF
        scores = torch.zeros(W, device=device)  # (W,)

        with torch.no_grad():
            for k in range(J):
                # Number of active components for this joint
                # (determine from non-negligible weights)
                w_k = q5_weights[k]  # (K_max,)
                K_k = int((w_k > _EPS).sum().item())
                K_k = max(K_k, 1)  # always at least 1

                means_k = q5_means[k, :K_k]  # (K_k,)
                kappas_k = q5_kappas[k, :K_k]  # (K_k,)
                weights_k = w_k[:K_k]  # (K_k,)
                # Renormalise in case of padding artefacts
                weights_k = weights_k / weights_k.sum().clamp(min=_EPS)

                B_exp = W * K_k  # expanded batch size

                # ── Expand ────────────────────────────────────────────
                # beam_exp : (W*K_k, J) — tile W beam rows K_k times
                beam_exp = beam.repeat(K_k, 1)  # (W*K_k, J)

                # Assign joint k in each group of W to the c-th mode mean
                for c in range(K_k):
                    beam_exp[c * W : (c + 1) * W, k] = means_k[c]

                scores_exp = scores.repeat(K_k)  # (W*K_k,)

                # ── Partial FK ────────────────────────────────────────
                # Build full theta: base + torsions (only joints 0..k matter)
                base_exp = base_theta.repeat(K_k, 1)  # (W*K_k, 6)
                theta_exp = torch.cat([base_exp, beam_exp], dim=-1)  # (W*K_k, 6+J)

                X_partial = self.fk.partial_apply(theta_exp, X_ref, n_joints=k + 1)  # (W*K_k, N, 3)

                # ── Q5 score ─────────────────────────────────────────
                phi_k = beam_exp[:, k]  # (W*K_k,)
                q5_score = vmm_log_prob(phi_k, means_k, kappas_k, weights_k)
                # (W*K_k,)

                scores_exp = scores_exp + q5_score

                # ── Partial Q6 ────────────────────────────────────────
                if use_q6:
                    q6_score = self._partial_q6_score(
                        X_partial,
                        k,
                        q6_idx_a,
                        q6_idx_b,
                        q6_mu,
                        q6_sigma,
                    )  # (W*K_k,)
                    scores_exp = scores_exp + q6_score

                # See exp.md for detailed rationale

                # Build a dict: torsion prefix tuple → (score, row)
                # Last-write wins for duplicate prefixes (doesn't matter since
                # identical prefixes have identical scores at this step).
                seen: dict = {}
                for idx in range(B_exp):
                    prefix = tuple(beam_exp[idx, : k + 1].tolist())
                    sc_val = scores_exp[idx].item()
                    if prefix not in seen or sc_val > seen[prefix][0]:
                        seen[prefix] = (sc_val, idx)

                # Sort unique prefixes by score descending, keep top W
                ranked = sorted(seen.values(), key=lambda x: x[0], reverse=True)
                top_w = ranked[:W]
                sel_idx = torch.tensor([r[1] for r in top_w], dtype=torch.long, device=device)

                beam_unique = beam_exp[sel_idx]  # (n_unique ≤ W, J)
                scores_unique = scores_exp[sel_idx]  # (n_unique,)

                # Pad to exactly W if fewer unique prefixes than W
                n_unique = beam_unique.shape[0]
                if n_unique < W:
                    pad = W - n_unique
                    # Repeat top candidates round-robin
                    pad_idx = torch.arange(pad, device=device) % n_unique
                    beam = torch.cat([beam_unique, beam_unique[pad_idx]], dim=0)
                    scores = torch.cat([scores_unique, scores_unique[pad_idx]], dim=0)
                else:
                    beam = beam_unique
                    scores = scores_unique

        log.debug(
            "BeamSearch.search: done. Score range [%.3f, %.3f]",
            scores.min().item(),
            scores.max().item(),
        )
        return beam, scores  # (W, J), (W,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_body_pairs(
        self,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pre-compute partial Q6 pair sets for each joint step.
        """
        fk = self.fk
        J = fk.n_joints

        if J == 0:
            return []

        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # downstream[k] = set of pos-indices downstream of joint k
        # (atoms that move when joint k rotates)
        down_sets: List[set] = [set(fk._downstream[k].tolist()) for k in range(J)]

        for k in range(J):
            # New body: downstream of k but NOT downstream of k-1
            # (i.e. the rigid body that joint k moves for the first time)
            if k == 0:
                new_body = down_sets[0]
            else:
                new_body = down_sets[k] - down_sets[k - 1]

            # Fixed atoms: everything NOT downstream of joint k
            # These are atoms whose positions are set before step k
            all_downstream_k = down_sets[k]
            # All ligand atoms
            all_lig = set(fk._lig_pos_t.tolist())
            fixed = all_lig - all_downstream_k

            new_t = torch.tensor(sorted(new_body), dtype=torch.long)
            fixed_t = torch.tensor(sorted(fixed), dtype=torch.long)
            pairs.append((new_t, fixed_t))

        return pairs

    def _partial_q6_score(
        self,
        X: torch.Tensor,  # (B, N, 3)
        k: int,
        q6_idx_a: torch.Tensor,  # (n_q6,) global Q6 pair index A
        q6_idx_b: torch.Tensor,  # (n_q6,) global Q6 pair index B
        q6_mu: torch.Tensor,  # (n_q6,)
        q6_sigma: torch.Tensor,  # (n_q6,)
    ) -> torch.Tensor:
        """
        Evaluate Q6 clashes between the newly-placed body at step k and all
        previously-fixed atoms.


        Parameters
        ----------
        X        : (B, N, 3) — current atom positions after partial FK
        k        : joint step index
        q6_idx_a, q6_idx_b : (n_q6,) global Q6 pair indices
        q6_mu, q6_sigma     : (n_q6,) halfgauss parameters

        Returns
        -------
        (B,) — summed Q6 penalty for newly-introduced contacts

        """
        B = X.shape[0]
        device = X.device

        if k >= len(self._q6_body_pairs):
            return torch.zeros(B, device=device)

        new_t, fixed_t = self._q6_body_pairs[k]
        new_t = new_t.to(device)
        fixed_t = fixed_t.to(device)

        if new_t.numel() == 0 or fixed_t.numel() == 0:
            return torch.zeros(B, device=device)

        new_set = set(new_t.tolist())
        fixed_set = set(fixed_t.tolist())

        # Filter global Q6 pairs to only those crossing new↔fixed boundary
        a_np = q6_idx_a.tolist()
        b_np = q6_idx_b.tolist()

        mask = torch.tensor(
            [
                (a_np[i] in new_set and b_np[i] in fixed_set)
                or (b_np[i] in new_set and a_np[i] in fixed_set)
                for i in range(len(a_np))
            ],
            dtype=torch.bool,
            device=device,
        )

        if mask.sum() == 0:
            return torch.zeros(B, device=device)

        sel_a = q6_idx_a[mask]  # (n_sel,)
        sel_b = q6_idx_b[mask]
        sel_mu = q6_mu[mask]  # (n_sel,)
        sel_sig = q6_sigma[mask]

        pos_a = X[:, sel_a, :]  # (B, n_sel, 3)
        pos_b = X[:, sel_b, :]
        dists = (pos_a - pos_b).norm(dim=-1)  # (B, n_sel)

        penalty = halfgauss_log_prob(
            dists,
            sel_mu[None, :],
            sel_sig[None, :],
        )  # (B, n_sel)

        return penalty.sum(dim=-1)  # (B,)
