"""
beam.py
=======
GPU beam search over Q5 VMM torsion modes for IK seed generation.

The search covers fk.n_joints_base torsion joints only: bend joints beyond
that have no fitted Q5 statistics and are left to gradient descent.  When no
donor qualifies for a bend joint, n_joints_base == n_joints.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from jacobik.core import gpu
from jacobik.core.env import EPS
from jacobik.ik.kernels import halfgauss_log_prob, vmm_log_prob

cp = gpu.cp

if TYPE_CHECKING:
    from jacobik.ik.fk import ForwardKinematics

log = logging.getLogger(__name__)

_EPS = EPS
# Production resolved configuration (hardcoded value)
BEAM_MODE_SCORE = "density"


class BeamSearch:
    """
    GPU beam search over Q5 torsion VMM modes.

    Parameters
    ----------
    fk         : ForwardKinematics for the ligand being searched
    beam_width : W — number of seeds to return (and maximum beam size)
    device     : device string (CUDA)
    """

    def __init__(
        self,
        fk: "ForwardKinematics",
        beam_width: int,
        device: Any = "cuda",
    ) -> None:
        self.fk = fk
        self.W = beam_width
        self.device = device
        # Bend joints have no Q5 rows: search only the base torsion joints.
        self.J = fk.n_joints_base

        # Pre-compute partial Q6 pair sets: for joint k, which atom-index
        # pairs (new_body_pos, fixed_pos) should be clash-checked.
        self._q6_body_pairs: List[Tuple[np.ndarray, np.ndarray]] = self._build_body_pairs()

        log.debug(
            "BeamSearch: W=%d, J=%d joints, device=%s",
            self.W,
            self.J,
            self.device,
        )

    def search(
        self,
        X_ref: Any,  # (N, 3) reference positions
        base_theta: Any,  # (1, 6) or (W, 6) base DOF
        q5_means: Any,  # (J, K_max) VMM means (radians)
        q5_kappas: Any,  # (J, K_max) VMM kappas
        q5_weights: Any,  # (J, K_max) VMM weights
        q6_mu: Optional[Any],  # (n_q6,) clash wall positions
        q6_sigma: Optional[Any],  # (n_q6,) clash wall widths
        q6_pairs: Optional[Tuple[Any, Any]] = None,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Run beam search and return W seed torsion vectors (W, J) and scores (W,)."""
        W = self.W
        J = self.J

        X_ref = gpu.as_gpu(X_ref, cp.float32)

        if J == 0:
            log.debug("BeamSearch.search: J=0, returning empty seeds")
            return (
                cp.zeros((W, 0), dtype=cp.float32),
                cp.zeros(W, dtype=cp.float32),
            )

        q5_means = gpu.as_gpu(q5_means, cp.float32)
        q5_kappas = gpu.as_gpu(q5_kappas, cp.float32)
        q5_weights = gpu.as_gpu(q5_weights, cp.float32)

        use_q6 = (
            q6_mu is not None
            and q6_sigma is not None
            and q6_pairs is not None
            and len(self._q6_body_pairs) > 0
        )
        if use_q6:
            q6_mu = gpu.as_gpu(q6_mu, cp.float32)
            q6_sigma = gpu.as_gpu(q6_sigma, cp.float32)
            q6_idx_a = gpu.as_gpu(q6_pairs[0], cp.int64)
            q6_idx_b = gpu.as_gpu(q6_pairs[1], cp.int64)
            N_atoms = X_ref.shape[0]
            new_mask = np.zeros((J, N_atoms), dtype=bool)
            fixed_mask = np.zeros((J, N_atoms), dtype=bool)
            for k_idx, (new_t, fixed_t) in enumerate(self._q6_body_pairs):
                if new_t.size > 0:
                    new_mask[k_idx, new_t] = True
                if fixed_t.size > 0:
                    fixed_mask[k_idx, fixed_t] = True
            new_mask_g = cp.asarray(new_mask)
            fixed_mask_g = cp.asarray(fixed_mask)
            step_masks = (new_mask_g[:, q6_idx_a] & fixed_mask_g[:, q6_idx_b]) | (
                new_mask_g[:, q6_idx_b] & fixed_mask_g[:, q6_idx_a]
            )
            step_has_q6 = cp.asnumpy(step_masks.any(axis=1)).tolist()
            pre_q6 = [
                (q6_idx_a[step_masks[k_idx]], q6_idx_b[step_masks[k_idx]], q6_mu[step_masks[k_idx]], q6_sigma[step_masks[k_idx]])
                if step_has_q6[k_idx] else None
                for k_idx in range(J)
            ]
        else:
            pre_q6 = [None] * J

        base_theta = gpu.as_gpu(base_theta, cp.float32)
        if base_theta.shape[0] == 1:
            base_theta = cp.ascontiguousarray(cp.broadcast_to(base_theta, (W, 6)))
        else:
            base_theta = cp.ascontiguousarray(base_theta[:W])

        beam = cp.zeros((W, J), dtype=cp.float32)
        scores = cp.zeros(W, dtype=cp.float32)
        K_ks = cp.asnumpy(cp.maximum((q5_weights > _EPS).sum(axis=-1), 1)).tolist()

        for k in range(J):
            K_k = int(K_ks[k])
            w_k = q5_weights[k]

            means_k = q5_means[k, :K_k]
            kappas_k = q5_kappas[k, :K_k]
            weights_k = w_k[:K_k]
            weights_k = weights_k / cp.maximum(weights_k.sum(), _EPS)

            B_exp = W * K_k

            beam_exp = cp.tile(beam, (K_k, 1))
            beam_exp[:, k] = cp.repeat(means_k, W)
            scores_exp = cp.tile(scores, K_k)

            base_exp = cp.tile(base_theta, (K_k, 1))
            theta_exp = cp.concatenate([base_exp, beam_exp], axis=-1)

            X_partial = self.fk.partial_apply(theta_exp, X_ref, n_joints=k + 1)

            phi_k = beam_exp[:, k]
            if BEAM_MODE_SCORE == "mass":
                q5_score = cp.repeat(cp.log(cp.maximum(weights_k, _EPS)), W)
            else:
                q5_score = vmm_log_prob(phi_k, means_k, kappas_k, weights_k)

            scores_exp = scores_exp + q5_score

            pq6 = pre_q6[k]
            if pq6 is not None:
                sel_a, sel_b, sel_mu, sel_sig = pq6
                dists = cp.linalg.norm(X_partial[:, sel_a, :] - X_partial[:, sel_b, :], axis=-1)
                q6_score = halfgauss_log_prob(dists, sel_mu[None, :], sel_sig[None, :]).sum(axis=-1)
                scores_exp = scores_exp + q6_score

            seen: dict = {}
            _prefixes = cp.asnumpy(beam_exp[:, : k + 1]).tolist()
            _scores = cp.asnumpy(scores_exp).tolist()
            for idx in range(B_exp):
                prefix = tuple(_prefixes[idx])
                sc_val = _scores[idx]
                if prefix not in seen or sc_val > seen[prefix][0]:
                    seen[prefix] = (sc_val, idx)

            ranked = sorted(seen.values(), key=lambda x: x[0], reverse=True)
            top_w = ranked[:W]
            sel_idx = cp.asarray([r[1] for r in top_w], dtype=cp.int64)

            beam_unique = beam_exp[sel_idx]
            scores_unique = scores_exp[sel_idx]

            n_unique = beam_unique.shape[0]
            if n_unique < W:
                pad = W - n_unique
                pad_idx = cp.arange(pad, dtype=cp.int64) % n_unique
                beam = cp.concatenate([beam_unique, beam_unique[pad_idx]], axis=0)
                scores = cp.concatenate([scores_unique, scores_unique[pad_idx]], axis=0)
            else:
                beam = beam_unique
                scores = scores_unique

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "BeamSearch.search: done. Score range [%.3f, %.3f]",
                float(scores.min()),
                float(scores.max()),
            )
        return beam, scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_body_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Pre-compute partial Q6 pair sets for each joint step."""
        fk = self.fk
        J = fk.n_joints_base
        if J == 0:
            return []

        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        down_sets: List[set] = [set(gpu.to_cpu(fk._downstream[k]).tolist()) for k in range(J)]
        all_lig = set(gpu.to_cpu(fk._lig_pos_t).tolist())

        for k in range(J):
            if k == 0:
                new_body = down_sets[0]
            else:
                new_body = down_sets[k] - down_sets[k - 1]
            fixed = all_lig - down_sets[k]
            pairs.append((np.array(sorted(new_body), dtype=np.int64), np.array(sorted(fixed), dtype=np.int64)))

        return pairs
