"""
scorer.py
=========
Combines GeometryExtractor (what the quantities are) with fitted distribution
parameters (what they should be) into a differentiable score and IK residual.

Version history
---------------
1.0.0 : initial release
1.1.0 : _score_components rewritten to use batched tensor ops instead of
         Python for-loops over Q1/Q2/Q3/Q4/Q5 terms.  Each quantity's
         log-probs are now computed in a single (B, N, K) → (B,) tensor
         expression with no Python-level loop.  This eliminates 36+ separate
         autograd graph nodes for EDTA (6 Q1 + 15 Q2 + 15 Q5 terms) and
         reduces backward-pass traversal depth proportionally.  Measured
         ~11× speedup on EDTA-scale scorer calls; overall solve speedup
         ~3-5× for polydentate complexes.  Q6 was already batched.
         The _GMM_VAR_MIN floor is applied per-term via clamping of the
         padded (N, K) variance tensor before the batched computation.
2.0.0 : Unified objective — one honest score, one aggregation.
         Q6 changes from arithmetic mean to soft-min reduction
         (−LSE(−τ·p)/τ + log(n)/τ, temperature = _Q6_LSE_TEMP = 2.0, reused
         from the anchor's existing worst-donor LSE — one device, one constant).
         The mean laundered clashes: a buried atom at 0.40 Å scoring −16 per
         pair was diluted 34× to −2.36 by averaging over ~30 pairs (verified
         on EDTA batch).  The soft-min returns ≈ −14.3 for the same case.
         Uniform penalties: soft-min = mean (equal footing preserved).
         Spiky penalties: soft-min → worst pair (catches buried atoms).
         All other Q1–Q5 quantities keep per-term mean (correct for a
         preference distribution).
2.1.0 : Static parameter precomputation + generalised quantity weights.
         VMM quantities (Q2, Q3, Q5) now precompute log_norm (which contains
         the i0e(κ) call) and log_w at __init__ time.  GMM quantities (Q1, Q4)
         precompute log_w and eff_vars.  These are static — they don't depend
         on X — so computing them inside the T×200 loop was pure waste.
         q2_weight removed as a per-call parameter: replaced by a general
         quantity_weights dict at __init__ that scales any subset of Q1–Q6.
         The weight for each quantity is baked into the precomputed log_w
         tensors (via additive log-offset), so score() takes no float args.
         q2_score() updated to use precomputed tensors.
         Measured ~1.35× speedup per VMM call; negligible for GMM.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from jacobik.ik.geometry import GeometryExtractor
from jacobik.ik.kernels import (
    halfgauss_log_prob,
)

if TYPE_CHECKING:
    from jacobik.core.types import TMComplex

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEG2RAD = math.pi / 180.0

#: Cosine threshold for trans-partner identification (angle > 140°)
_TRANS_COS_THRESHOLD = math.cos(math.radians(140.0))  # ≈ −0.766

#: Graph distance classification cutoffs
_CHELATE_SHORT_MAX = 4
_CHELATE_LONG_MAX = 8

# See exp.md for detailed rationale
_GMM_VAR_MIN: float = 0.04

# Mathematical constants used in the batched VMM/GMM helpers in
# _score_components.  Defined locally so scorer.py has no hidden
# dependency on kernels.py internals.
_LOG_2PI: float = math.log(2.0 * math.pi)
_EPS: float = 1e-7

# See exp.md for detailed rationale
_KAPPA_SYNTH_MIN: float = 0.5
_KAPPA_SYNTH_MAX: float = 10.0

#: Temperature for soft-min aggregation of Q6 per-pair penalties.
#: Reused from the anchor's existing worst-donor LSE (anchor lse_temp=2.0);
#: one device, one constant, applied consistently.
#: Formulation: −LSE(−τ·p)/τ + log(n)/τ
#:   τ→0: approaches mean(p) (equal footing)
#:   τ→∞: approaches min(p) (hard-min, worst clash dominates)
#: At τ=2.0: equals mean when penalties are uniform; tracks the worst
#: violating pair when one atom is buried (spiky penalty distribution).
_Q6_LSE_TEMP: float = 2.0


# ---------------------------------------------------------------------------
# Internal helpers (CPU, __init__ only)
# ---------------------------------------------------------------------------


def _trans_partners(mol: "TMComplex") -> Dict[int, str]:
    """
    For each coord_site, find the trans partner (the site with the largest
    D–M–D angle to this site, threshold > 140°).
    """
    n = len(mol.coord_sites)
    positions = np.array([s.position for s in mol.coord_sites], dtype=float)
    metal_pos = np.array(
        [a.position for a in mol.atoms if a.idx == mol.metal_idx][0],
        dtype=float,
    )

    result: Dict[int, str] = {}
    for i, site_i in enumerate(mol.coord_sites):
        v_i = positions[i] - metal_pos
        n_i = np.linalg.norm(v_i)
        if n_i < 1e-8:
            result[i] = "none"
            continue
        v_i = v_i / n_i

        best_cos = 2.0  # initialise to something impossible (> 1)
        best_j = -1
        for j, site_j in enumerate(mol.coord_sites):
            if j == i:
                continue
            v_j = positions[j] - metal_pos
            n_j = np.linalg.norm(v_j)
            if n_j < 1e-8:
                continue
            v_j = v_j / n_j
            cos_ij = float(np.dot(v_i, v_j))
            if cos_ij < best_cos:
                best_cos = cos_ij
                best_j = j

        if best_j >= 0 and best_cos < _TRANS_COS_THRESHOLD:
            # Site best_j is the trans partner
            partner_site = mol.coord_sites[best_j]
            d_idx = partner_site.donor_indices[0]
            label = mol.donor_labels.get(d_idx)
            result[i] = label.subclass if label is not None else "none"
        else:
            result[i] = "none"

    return result


def _ligand_graph_distances(mol: "TMComplex") -> Dict[Tuple[int, int], int]:
    """
    Build shortest-path distances (in ligand bond hops) between every pair of
    donor atom indices, across the full complex ligand graph (no metal, no H).
    """
    metal_idx = mol.metal_idx
    atom_map = {a.idx: a for a in mol.atoms}

    from jacobik.topology.graph import build_heavy_ligand_graph

    G = build_heavy_ligand_graph(mol.atoms, mol.bonds, mol.metal_idx)

    # Collect kappa donor atom indices (one per kappa site) and haptic centroids
    # represented by the min donor index in the haptic group
    donor_idxs: List[int] = []
    for site in mol.coord_sites:
        donor_idxs.append(min(site.donor_indices))

    # BFS from each donor; only need pairs
    dists: Dict[Tuple[int, int], int] = {}
    for da in donor_idxs:
        if not G.has_node(da):
            continue
        lengths = nx.single_source_shortest_path_length(G, da)
        for db in donor_idxs:
            if db <= da:
                continue
            d = lengths.get(db)
            if d is not None:
                dists[(da, db)] = d
    return dists


def _graph_dist_class(d: Optional[int], same_ligand: bool) -> str:
    """Classify a graph distance into one of the four Q2 distance labels."""
    if not same_ligand:
        return "different_ligand"
    if d is None or d > _CHELATE_LONG_MAX:
        return "same_remote"
    if d <= _CHELATE_SHORT_MAX:
        return "chelate_short"
    return "chelate_long"


def _pad_and_stack(
    param_lists: List[List[float]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of variable-length float lists to a common length and stack.

    Returns
    -------
    tensor  : (N, max_K) float32 padded tensor
    lengths : (N,) int64 — original lengths (for masked computation)
    """
    if not param_lists:
        return (
            torch.zeros(0, 1, dtype=dtype, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )
    max_k = max(len(p) for p in param_lists)
    lengths = torch.tensor([len(p) for p in param_lists], dtype=torch.long, device=device)
    t = torch.zeros(len(param_lists), max_k, dtype=dtype, device=device)
    for i, p in enumerate(param_lists):
        t[i, : len(p)] = torch.tensor(p, dtype=dtype)
    return t, lengths


def _vmm_fields(dist, key: tuple, quantity: str) -> Optional[tuple]:
    """
    Return (means_deg, kappas, weights) from a VMM FittedDist, or None if the
    required fields cannot be recovered even approximately.
    """
    # Empty lists are as unusable as None — a VMM needs at least one component.
    if not dist.means or not dist.weights:
        log.warning(
            "%s: dist for key %s returned type=%r with empty means/weights "
            "(means=%r weights=%r) — using hardcoded fallback.",
            quantity,
            key,
            dist.type,
            dist.means,
            dist.weights,
        )
        return None

    if dist.means is not None and dist.kappas is not None and dist.weights is not None:
        return dist.means, dist.kappas, dist.weights

    if dist.means is not None and dist.weights is not None:
        # kappas is None but means/weights are present.
        if dist.vars is not None:
            # type='gmm' misfiled for a VMM slot.
            # The stats pipeline stores Q3/Q5 GMM vars in degrees² (matching
            # the degree-valued means), but vmm_log_prob works in radians.
            # Convert: var_rad² = var_deg² * (π/180)² before computing κ≈1/var.
            _DEG2RAD_SQ = (math.pi / 180.0) ** 2
            kappas = [
                max(_KAPPA_SYNTH_MIN, min(_KAPPA_SYNTH_MAX, 1.0 / max(v * _DEG2RAD_SQ, 1e-6)))
                for v in dist.vars
            ]
            log.warning(
                "%s: key %s has type=%r with kappas=None; synthesising "
                "kappas from vars (deg²→rad²) via κ≈1/σ²: %s",
                quantity,
                key,
                dist.type,
                [f"{k:.2f}" for k in kappas],
            )
        else:
            # No variance either — use a broad uniform-ish concentration.
            kappas = [0.5] * len(dist.means)
            log.warning(
                "%s: key %s has type=%r with kappas=None and vars=None; "
                "using κ=0.5 (broad VMM) for all %d components.",
                quantity,
                key,
                dist.type,
                len(dist.means),
            )
        return dist.means, kappas, dist.weights

    log.warning(
        "%s: dist for key %s returned type=%r but means/kappas/weights "
        "contains None — using hardcoded fallback.",
        quantity,
        key,
        dist.type,
    )
    return None


def _gmm_fields(dist, key: tuple, quantity: str) -> Optional[tuple]:
    """
    Return (means, vars, weights) from a GMM FittedDist, or None if any
    required field is missing/None or empty.  Logs a warning on failure.
    """
    if (not dist.means) or (dist.vars is None) or (not dist.weights):
        log.warning(
            "%s: dist for key %s returned type=%r but means/vars/weights "
            "is None or empty — using hardcoded fallback.",
            quantity,
            key,
            dist.type,
        )
        return None
    return dist.means, dist.vars, dist.weights


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class Scorer:
    """
    Combines geometric quantities (from GeometryExtractor) with fitted
    distribution parameters (from stats) into a differentiable score.

    Parameters
    ----------
    mol              : fully assembled TMComplex (coord_sites, donor_labels populated)
    ge               : GeometryExtractor for mol (already constructed)
    stats            : object satisfying the stats module interface (§2):
                          stats.dist(quantity: str, key: tuple) → Distribution
    device           : torch.device
    quantity_weights : optional dict mapping "Q1".."Q6" → float multiplier
                       applied to that quantity's normalised contribution.
                       Missing keys default to 1.0.  Q2 is a common use:
                       set {"Q2": 0.0} to disable D–M–D angle scoring,
                       {"Q2": 1.0} (the default) to include it at full weight.
                       Weights are baked into the precomputed log_w tensors
                       at __init__ time — there is no per-call overhead.
    """

    def __init__(
        self,
        mol: "TMComplex",
        ge: GeometryExtractor,
        stats: Any,
        device: torch.device,
        quantity_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.device = device
        self._ge = ge
        self.mol = mol
        self.stats = stats

        # Normalise quantity weights: default 1.0 for all, caller overrides.
        _qw = quantity_weights or {}
        self._qweights = {q: float(_qw.get(q, 1.0)) for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")}

        atom_map = {a.idx: a for a in mol.atoms}
        metal_sym = mol.metal_symbol
        coord_geom = mol.coord_geom or "octahedral"
        spin = mol.spin_state
        charge = mol.charge

        # Pre-resolve trans partners (needed for Q1 keys)
        trans_map = _trans_partners(mol)  # site_index → subclass str

        # Ligand membership for each coord_site (for Q2 graph distance class)
        site_ligand: Dict[int, int] = {}  # site_index → ligand_id or -1
        for s_i, site in enumerate(mol.coord_sites):
            d_min = min(site.donor_indices)
            for lig in mol.ligands:
                if d_min in lig.atom_indices:
                    site_ligand[s_i] = lig.ligand_id
                    break
            else:
                site_ligand[s_i] = -1

        graph_dists = _ligand_graph_distances(mol)

        # See exp.md for detailed rationale

        q1_means_list: List[List[float]] = []
        q1_vars_list: List[List[float]] = []
        q1_weights_list: List[List[float]] = []

        for s_i, site in enumerate(mol.coord_sites):
            # Donor subclass: for kappa the single donor's label; for haptic
            # use the first donor's label (they share the same haptic subclass)
            d_idx = site.donor_indices[0]
            label = mol.donor_labels.get(d_idx)
            subclass = label.subclass if label is not None else "Unknown_donor"
            trans_p = trans_map.get(s_i, "none")

            key = (metal_sym, spin, charge, subclass, coord_geom, trans_p)
            try:
                dist = stats.dist("Q1", key)
                fields = _gmm_fields(dist, key, "Q1")
            except Exception as exc:
                log.warning("Q1 stats.dist failed for key %s: %s", key, exc)
                fields = None
            if fields is not None:
                q1_means_list.append(fields[0])
                q1_vars_list.append(fields[1])
                q1_weights_list.append(fields[2])
            else:
                # Fallback: broad Gaussian centred at 2.1 Å
                q1_means_list.append([2.1])
                q1_vars_list.append([0.04])
                q1_weights_list.append([1.0])

        # Stack as padded tensors (n_donors, max_K_q1)
        self._q1_means, _ = _pad_and_stack(q1_means_list, device)
        self._q1_vars, _ = _pad_and_stack(q1_vars_list, device)
        self._q1_weights, self._q1_K = _pad_and_stack(q1_weights_list, device)
        # _q1_K: (n_donors,) number of active components per donor

        # See exp.md for detailed rationale

        q3_means_list: List[List[float]] = []
        q3_kappas_list: List[List[float]] = []
        q3_weights_list: List[List[float]] = []

        for s_i, site in enumerate(mol.coord_sites):
            if not ge._q3_eligible[s_i]:
                continue

            d_idx = site.donor_indices[0]
            label = mol.donor_labels.get(d_idx)
            subclass = label.subclass if label is not None else "Unknown_donor"
            key = (subclass, metal_sym, coord_geom)
            try:
                dist = stats.dist("Q3", key)
                fields = _vmm_fields(dist, key, "Q3")
            except Exception as exc:
                log.warning("Q3 stats.dist failed for key %s: %s", key, exc)
                fields = None
            if fields is not None:
                q3_means_list.append([m * _DEG2RAD for m in fields[0]])
                q3_kappas_list.append(fields[1])
                q3_weights_list.append(fields[2])
            else:
                q3_means_list.append([math.pi])  # 180° — linear default
                q3_kappas_list.append([1.0])
                q3_weights_list.append([1.0])

        self._q3_means, _ = _pad_and_stack(q3_means_list, device)
        self._q3_kappas, _ = _pad_and_stack(q3_kappas_list, device)
        self._q3_weights, _ = _pad_and_stack(q3_weights_list, device)
        self._n_q3 = len(q3_means_list)

        # ==================================================================
        # Q4 — pyramidalization GMM per eligible kappa donor
        # ge.q4_pyramidalization(X) → (B, n_q4) in DEGREES
        # ==================================================================

        q4_means_list: List[List[float]] = []
        q4_vars_list: List[List[float]] = []
        q4_weights_list: List[List[float]] = []

        # ge._q4_donor_pos gives the pos-indices of eligible donors in order
        # Recover their site indices to look up subclass labels
        pos_to_site: Dict[int, int] = {}
        idx_to_pos = {a.idx: i for i, a in enumerate(mol.atoms)}
        for s_i, site in enumerate(mol.coord_sites):
            if not site.is_haptic:
                d_pos = idx_to_pos.get(site.donor_indices[0], -1)
                if d_pos >= 0:
                    pos_to_site[d_pos] = s_i

        for d_pos in ge._q4_donor_pos.tolist():
            s_i = pos_to_site.get(d_pos)
            if s_i is None:
                q4_means_list.append([109.5])
                q4_vars_list.append([100.0])
                q4_weights_list.append([1.0])
                continue
            site = mol.coord_sites[s_i]
            d_idx = site.donor_indices[0]
            label = mol.donor_labels.get(d_idx)
            subclass = label.subclass if label is not None else "Unknown_donor"
            key = (subclass, coord_geom)
            try:
                dist = stats.dist("Q4", key)
                fields = _gmm_fields(dist, key, "Q4")
            except Exception as exc:
                log.warning("Q4 stats.dist failed for key %s: %s", key, exc)
                fields = None
            if fields is not None:
                q4_means_list.append(fields[0])
                q4_vars_list.append(fields[1])
                q4_weights_list.append(fields[2])
            else:
                q4_means_list.append([109.5])
                q4_vars_list.append([100.0])
                q4_weights_list.append([1.0])

        self._q4_means, _ = _pad_and_stack(q4_means_list, device)
        self._q4_vars, _ = _pad_and_stack(q4_vars_list, device)
        self._q4_weights, _ = _pad_and_stack(q4_weights_list, device)

        # See exp.md for detailed rationale

        q5_means_list: List[List[float]] = []
        q5_kappas_list: List[List[float]] = []
        q5_weights_list: List[List[float]] = []

        for q5_key in ge._q5_keys:
            try:
                dist = stats.dist("Q5", q5_key)
                fields = _vmm_fields(dist, q5_key, "Q5")
            except Exception as exc:
                log.warning("Q5 stats.dist failed for key %s: %s", q5_key, exc)
                fields = None
            if fields is not None:
                q5_means_list.append([m * _DEG2RAD for m in fields[0]])
                q5_kappas_list.append(fields[1])
                q5_weights_list.append(fields[2])
            else:
                # Flat VMM: uniform over [-π, π]  (κ=0 → uniform)
                q5_means_list.append([0.0])
                q5_kappas_list.append([1e-4])
                q5_weights_list.append([1.0])

        self._q5_means, _ = _pad_and_stack(q5_means_list, device)
        self._q5_kappas, _ = _pad_and_stack(q5_kappas_list, device)
        self._q5_weights, _ = _pad_and_stack(q5_weights_list, device)

        # See exp.md for detailed rationale

        q6_mu_list: List[float] = []
        q6_sigma_list: List[float] = []

        for elem_key in ge._q6_elem_keys:
            try:
                dist = stats.dist("Q6", elem_key)
                q6_mu_list.append(dist.mu)
                q6_sigma_list.append(dist.sigma)
            except Exception as exc:
                log.warning("Q6 stats.dist failed for key %s: %s", elem_key, exc)
                # Default: vdW contact ~ 3.4 Å, width 0.3 Å
                q6_mu_list.append(3.4)
                q6_sigma_list.append(0.3)

        if q6_mu_list:
            self._q6_mu = torch.tensor(q6_mu_list, dtype=torch.float32, device=device)
            self._q6_sigma = torch.tensor(q6_sigma_list, dtype=torch.float32, device=device)
        else:
            self._q6_mu = torch.zeros(0, dtype=torch.float32, device=device)
            self._q6_sigma = torch.zeros(0, dtype=torch.float32, device=device)

        # ==================================================================
        # Q2 — D–M–D angle VMM per donor pair (for joint scoring only)
        # ge.q2_dmd_angles(X) → (B, n_pairs) in RADIANS (pairs in i<j order)
        # ==================================================================

        q2_means_list: List[List[float]] = []
        q2_kappas_list: List[List[float]] = []
        q2_weights_list: List[List[float]] = []
        # Store pair (site_i, site_j) metadata for key construction
        self._q2_pair_meta: List[Tuple[int, int, str]] = []  # (s_i, s_j, dist_class)

        for s_i in range(len(mol.coord_sites)):
            for s_j in range(s_i + 1, len(mol.coord_sites)):
                site_i = mol.coord_sites[s_i]
                site_j = mol.coord_sites[s_j]

                # Donor subclasses
                di_idx = site_i.donor_indices[0]
                dj_idx = site_j.donor_indices[0]
                li = mol.donor_labels.get(di_idx)
                lj = mol.donor_labels.get(dj_idx)
                sub_i = li.subclass if li is not None else "Unknown_donor"
                sub_j = lj.subclass if lj is not None else "Unknown_donor"

                # Canonical ordering: alphabetical on subclass, idx as tie-break
                key_i = min(site_i.donor_indices)
                key_j = min(site_j.donor_indices)
                if (sub_i, key_i) <= (sub_j, key_j):
                    sub_A, sub_B = sub_i, sub_j
                else:
                    sub_A, sub_B = sub_j, sub_i

                # Graph distance class
                da_min = min(site_i.donor_indices)
                db_min = min(site_j.donor_indices)
                pair_key = (min(da_min, db_min), max(da_min, db_min))
                hop_dist = graph_dists.get(pair_key)
                same_lig = (
                    site_ligand.get(s_i, -1) == site_ligand.get(s_j, -1)
                    and site_ligand.get(s_i, -1) >= 0
                )
                dist_class = _graph_dist_class(hop_dist, same_lig)

                q2_key = (coord_geom, dist_class, metal_sym, sub_A, sub_B)
                try:
                    dist_obj = stats.dist("Q2", q2_key)
                    fields = _vmm_fields(dist_obj, q2_key, "Q2")
                except Exception as exc:
                    log.warning("Q2 stats.dist failed for key %s: %s", q2_key, exc)
                    fields = None
                if fields is not None:
                    q2_means_list.append([m * _DEG2RAD for m in fields[0]])
                    q2_kappas_list.append(fields[1])
                    q2_weights_list.append(fields[2])
                else:
                    # Broad VMM centred at 90° (κ=0.5 ≈ weakly concentrated)
                    q2_means_list.append([math.pi / 2])
                    q2_kappas_list.append([0.5])
                    q2_weights_list.append([1.0])

                self._q2_pair_meta.append((s_i, s_j, dist_class))

        self._q2_means, _ = _pad_and_stack(q2_means_list, device)
        self._q2_kappas, _ = _pad_and_stack(q2_kappas_list, device)
        self._q2_weights, _ = _pad_and_stack(q2_weights_list, device)

        n_q1 = ge.n_donors
        n_q3 = self._n_q3
        n_q4 = ge.n_q4
        n_q5 = ge.n_q5
        n_q6 = ge.n_q6
        n_q2 = len(q2_means_list)

        log.debug(
            "Scorer: Q1=%d Q2=%d Q3=%d Q4=%d Q5=%d Q6=%d",
            n_q1,
            n_q2,
            n_q3,
            n_q4,
            n_q5,
            n_q6,
        )

        # Per-quantity term counts (for normalisation in score()).
        self._n_q1 = n_q1
        self._n_q2 = n_q2
        self._n_q4 = n_q4
        self._n_q5 = n_q5
        self._n_q6 = n_q6

        # ── Static precomputation ──────────────────────────────────────────────
        # These tensors do not depend on X; computing them inside the solve loop
        # (200 steps × 3 VMM quantities) was pure waste.  They are computed once
        # here and reused on every call to _score_components / score().
        #
        # Quantity weights are baked into log_w (additive log offset) so that
        # _score_components needs no per-call weight multiplications.

        def _vmm_precomp(kappas, weights, q_weight):
            """Precompute static VMM log_norm and log_w (with quantity weight)."""
            ln = _LOG_2PI + torch.log(torch.special.i0e(kappas).clamp(min=_EPS)) + kappas
            lw = torch.log(weights.clamp(min=_EPS))
            # Bake quantity weight into log_w: log(w * q_weight) = log_w + log(q_weight)
            if q_weight != 1.0 and q_weight > 0.0:
                lw = lw + math.log(q_weight)
            elif q_weight == 0.0:
                lw = torch.full_like(lw, float("-inf"))
            return ln, lw

        def _gmm_precomp(vars_, weights, q_weight):
            """Precompute static GMM eff_vars and log_w (with quantity weight)."""
            ev = vars_.clamp(min=max(_GMM_VAR_MIN, _EPS))
            lw = torch.log(weights.clamp(min=_EPS))
            if q_weight != 1.0 and q_weight > 0.0:
                lw = lw + math.log(q_weight)
            elif q_weight == 0.0:
                lw = torch.full_like(lw, float("-inf"))
            return ev, lw

        qw = self._qweights

        # Q1 (GMM)
        self._q1_eff_vars, self._q1_log_w = _gmm_precomp(self._q1_vars, self._q1_weights, qw["Q1"])

        # Q2 (VMM) — weight already controls whether Q2 contributes at all
        self._q2_log_norm, self._q2_log_w = _vmm_precomp(
            self._q2_kappas, self._q2_weights, qw["Q2"]
        )

        # Q3 (VMM)
        self._q3_log_norm, self._q3_log_w = _vmm_precomp(
            self._q3_kappas, self._q3_weights, qw["Q3"]
        )

        # Q4 (GMM)
        self._q4_eff_vars, self._q4_log_w = _gmm_precomp(self._q4_vars, self._q4_weights, qw["Q4"])

        # Q5 (VMM)
        self._q5_log_norm, self._q5_log_w = _vmm_precomp(
            self._q5_kappas, self._q5_weights, qw["Q5"]
        )

        # Q6: log(n_q6)/τ term is also static (depends only on n_q6 and τ)
        self._q6_lse_log_n_over_tau = math.log(max(n_q6, 1)) / _Q6_LSE_TEMP if n_q6 > 0 else 0.0
        # Q6 quantity weight applied as a scalar multiplier (not baked into log_w
        # because Q6 uses a different aggregation structure — soft-min, not logsumexp).
        self._q6_qweight = float(qw["Q6"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _score_components(
        self,
        X: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Shared internal pass: compute each quantity's NORMALISED value as a
        (B,) tensor, keyed by quantity name.  All static parameters
        (log_norm, log_w, eff_vars) are precomputed at __init__ — no i0e()
        or log() calls happen here, only the X-dependent computations.

        Quantity weights are already baked into the precomputed log_w tensors;
        the returned values are quantity_weight × (normalised log-prob).

        All Q1–Q5 quantities are evaluated with fully-batched tensor ops
        (no Python for-loops over terms) so the autograd graph has O(1)
        nodes per quantity rather than O(N_terms) nodes.

        Returns
        -------
        dict with keys "Q1".."Q6", each a (B,) tensor of that quantity's
        weighted normalised contribution to the total score.
        """
        B = X.shape[0]
        comp: Dict[str, torch.Tensor] = {
            q: X.new_zeros(B) for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
        }

        # ── Batched VMM helper (uses precomputed log_norm and log_w) ─────────
        # obs      : (B, N)   — N observed angles in radians
        # means    : (N, K)
        # kappas   : (N, K)
        # log_norm : (N, K)   precomputed: LOG_2PI + log(i0e(κ)) + κ
        # log_w    : (N, K)   precomputed: log(weights) [+ log(q_weight) baked in]
        # returns (B,) — normalised (sum / N) log-prob for this quantity
        def _vmm(obs, means, kappas, log_norm, log_w, n_terms):
            cos_diff = torch.cos(obs.unsqueeze(-1) - means.unsqueeze(0))  # (B, N, K)
            log_comp = kappas.unsqueeze(0) * cos_diff - log_norm.unsqueeze(0)
            log_p = log_comp + log_w.unsqueeze(0)  # (B, N, K)
            return torch.logsumexp(log_p, dim=-1).sum(dim=-1) / max(n_terms, 1)

        # ── Batched GMM helper (uses precomputed eff_vars and log_w) ─────────
        # obs      : (B, N)
        # means    : (N, K)
        # eff_vars : (N, K)   precomputed: vars.clamp(min=GMM_VAR_MIN)
        # log_w    : (N, K)   precomputed: log(weights) [+ log(q_weight) baked in]
        # returns (B,) — normalised (sum / N) log-prob
        def _gmm(obs, means, eff_vars, log_w, n_terms):
            diff = obs.unsqueeze(-1) - means.unsqueeze(0)  # (B, N, K)
            log_comp = (
                -0.5 * _LOG_2PI
                - 0.5 * torch.log(eff_vars).unsqueeze(0)
                - 0.5 * diff**2 / eff_vars.unsqueeze(0)
            )  # (B, N, K)
            log_p = log_comp + log_w.unsqueeze(0)  # (B, N, K)
            return torch.logsumexp(log_p, dim=-1).sum(dim=-1) / max(n_terms, 1)

        # ── Q1 ─────────────────────────────────────────────────────────────
        if self._n_q1 > 0 and self._q1_means.shape[0] > 0:
            q1_d = self._ge.q1_distances(X)  # (B, n_donors)
            comp["Q1"] = _gmm(q1_d, self._q1_means, self._q1_eff_vars, self._q1_log_w, self._n_q1)

        # ── Q2 ─────────────────────────────────────────────────────────────
        # q2_weight baked into _q2_log_w at __init__; skip if weight was 0.0
        if self._n_q2 > 0 and self._q2_means.shape[0] > 0 and self._qweights["Q2"] > 0.0:
            q2_angles = self._ge.q2_dmd_angles(X)  # (B, n_pairs)
            comp["Q2"] = _vmm(
                q2_angles,
                self._q2_means,
                self._q2_kappas,
                self._q2_log_norm,
                self._q2_log_w,
                self._n_q2,
            )

        # ── Q3 ─────────────────────────────────────────────────────────────
        if self._n_q3 > 0 and self._q3_means.shape[0] > 0:
            q3_a = self._ge.q3_approach_angles(X)  # (B, n_q3)
            comp["Q3"] = _vmm(
                q3_a, self._q3_means, self._q3_kappas, self._q3_log_norm, self._q3_log_w, self._n_q3
            )

        # ── Q4 ─────────────────────────────────────────────────────────────
        if self._ge.n_q4 > 0 and self._q4_means.shape[0] > 0:
            q4_v = self._ge.q4_pyramidalization(X)  # (B, n_q4) degrees
            comp["Q4"] = _gmm(
                q4_v, self._q4_means, self._q4_eff_vars, self._q4_log_w, self._ge.n_q4
            )

        # ── Q5 ─────────────────────────────────────────────────────────────
        if self._ge.n_q5 > 0 and self._q5_means.shape[0] > 0:
            q5_t = self._ge.q5_torsions(X)  # (B, n_q5) radians
            comp["Q5"] = _vmm(
                q5_t,
                self._q5_means,
                self._q5_kappas,
                self._q5_log_norm,
                self._q5_log_w,
                self._ge.n_q5,
            )

        # ── Q6 ─────────────────────────────────────────────────────────────
        # Soft-min: −LSE(−τ·p)/τ + log(n)/τ (log(n)/τ is precomputed as
        # _q6_lse_log_n_over_tau).  Q6 quantity weight applied as a scalar
        # multiplier (not baked into log_w; different aggregation structure).
        if self._ge.n_q6 > 0 and self._q6_mu.shape[0] > 0 and self._q6_qweight > 0.0:
            q6_d = self._ge.q6_contacts(X)  # (B, n_q6) Å
            penalty = halfgauss_log_prob(
                q6_d,
                self._q6_mu[None, :],
                self._q6_sigma[None, :],
            )  # (B, n_q6) ≤ 0
            tau = _Q6_LSE_TEMP
            q6_val = -torch.logsumexp(-tau * penalty, dim=-1) / tau + self._q6_lse_log_n_over_tau
            comp["Q6"] = self._q6_qweight * q6_val

        return comp

    def score(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Unified statistical score: one honest, defensible objective.

        Returns both the scalar to optimise and the per-quantity breakdown
        that produced it — they are identical by construction.

        Normalisation and quantity weights are handled inside
        _score_components (via precomputed tensors set at __init__).
        No per-call float arguments needed.

        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        scalar    : (B,)  — sum of per-quantity weighted values; higher = better
        breakdown : dict with keys "Q1".."Q6", each a (B,) contribution
        """
        breakdown = self._score_components(X)
        scalar = X.new_zeros(X.shape[0])
        for v in breakdown.values():
            scalar = scalar + v
        return scalar, breakdown  # (B,), dict[str, (B,)]

    def score_breakdown(
        self,
        X: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible alias: return per-quantity breakdown only.
        Equivalent to score(X)[1].

        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        dict with keys "Q1".."Q6", each a (B,) tensor.
        """
        _, breakdown = self.score(X)
        return breakdown

    def q2_score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Q2 (D–M–D angle) log-probability for post-IK joint ranking.
        Uses precomputed _q2_log_norm and _q2_log_w (no i0e/log per call).

        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        q2 : (B,)  — summed Q2 log-prob; higher = more plausible
        """
        B = X.shape[0]
        if self._q2_means.shape[0] == 0:
            return X.new_zeros(B)

        q2_angles = self._ge.q2_dmd_angles(X)  # (B, n_pairs)
        phi_e = q2_angles.unsqueeze(-1)  # (B, n_pairs, 1)
        cos_diff = torch.cos(phi_e - self._q2_means.unsqueeze(0))  # (B, n_pairs, K)
        log_comp = self._q2_kappas.unsqueeze(0) * cos_diff - self._q2_log_norm.unsqueeze(0)
        # Use raw log_w (not quantity-weighted) for post-IK ranking
        log_w = torch.log(self._q2_weights.clamp(min=_EPS))
        log_p = log_comp + log_w.unsqueeze(0)  # (B, n_pairs, K)
        return torch.logsumexp(log_p, dim=-1).sum(dim=-1)  # (B,)

    def residual(
        self,
        X: torch.Tensor,
        donor_targets: torch.Tensor,
        lambda_t: float,
    ) -> torch.Tensor:
        """
        IK residual: −score(X) + λ·anchor(X, donor_targets).


        Parameters
        ----------
        X              : (B, N, 3) — current atom positions from FK
        donor_targets  : (B, n_donors, 3) — hard target positions for each
                         coordination site (from coordination.build_targets)
        lambda_t       : float — current anchor weight (decays over IK steps)

        Returns
        -------
        r : (B,)  — residual; lower = better (solver minimises this)

        """
        stat_score, _ = self.score(X)  # (B,) — discard breakdown

        # Anchor: mean squared distance from donor positions to targets
        donor_pos = self._ge.donor_positions(X)  # (B, n_donors, 3)
        diff = donor_pos - donor_targets  # (B, n_donors, 3)
        anchor_loss = (diff**2).sum(dim=-1).mean(dim=-1)  # (B,)

        return -stat_score + lambda_t * anchor_loss  # (B,)
