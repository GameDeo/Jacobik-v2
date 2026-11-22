"""
scorer.py
=========
Combines GeometryExtractor (what the quantities are) with fitted distribution
parameters (what they should be) into a differentiable score.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from jacobik.core.env import EPS, LOG_2PI
from jacobik.ik.angle_bends import eta2_plane_normal_t
from jacobik.ik.geometry import GeometryExtractor
from jacobik.ik.kernels import q6_softmin

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

#: Per-quantity GMM variance floor in the quantity's native units (Q1 A^2, Q4 deg^2,
#: Q2/Q3 dimensionless cos-space); bounds gradients from near-delta fitted components.
_GMM_VAR_MIN: Dict[str, float] = {
    "Q1": 0.04,  # sigma floor ~0.2 Å
    "Q4": 4.00,  # sigma floor ~2.0 deg
    "Q2": 0.0025,  # cos-space, sigma ~0.05
    "Q3": 0.0025,
}

# Production resolved configuration (hardcoded u1 values, flags removed)
GMM_FLOOR_MODE: str = "thermal"
RANK_AGG: str = "sum"
Q1_TAIL_SIGMA: float = 0.0
Q5_FOLD_MODE: str = "auto"
SCORE_AGG: str = "sum"
Q6_MODEL: str = "repulsive"


def _q5_fold_for(stats) -> bool:
    if Q5_FOLD_MODE in ("0", "1"):
        return Q5_FOLD_MODE == "1"
    meta = getattr(getattr(stats, "_db", None), "meta", None) or {}
    return str(meta.get("q5_model", "")).endswith("folded")
#: (Removed 2026-10-16, ARCHITECTURE.md ledger: Q6_MODEL="wca" (WCA core; lost 10-14, gates FAIL 10-15 / 10-16) and the
#: SHAPE_CORE / SHAPE_TAIL energy-like shaping of Q1 / Q2 / S_orient (gate FAIL 10-16).)


#: Thermal widths (CHANGELOG 2026-10-16 d, plan U1; solve_floor="thermal"): the sd at 298 K of each scorer feature in
#: one molecule, sigma_T^2 = kT g^T H^+ g (g = the feature's gradient, H^+ = pseudo-inverse of the GFN2 Hessian at the
#: tmQM minimum), median over the training complexes of refit/thermal_widths.py (data/refit/final/
#: thermal_widths.json, data/refit/thermal/report.txt). A table component narrower than this describes the spread of
#: minima across molecules, not where one molecule sits at 298 K, and is floored to it. Measured once offline, like
#: angle_bends.BEND_K; nothing is computed during a solve. Q1 in A; Q2 and the S_orient angles ("SO", also used for Q3)
#: Q1 by donor element ("eta" = haptic centroid): the one descriptor that moves the M-D width by more than sqrt(2)
#: (M-C 0.04 A vs M-N 0.07 A); metal row and coordination number do not (data/refit/thermal/report.txt). Elements
#: with fewer than 15 measured bonds take THERMAL_SD["Q1"].
THERMAL_Q1_BY_DONOR: Dict[str, float] = {"Br": 0.0650, "C": 0.0406, "Cl": 0.0577, "I": 0.0598, "N": 0.0705, "O": 0.0597, "P": 0.0535, "S": 0.0693, "eta": 0.0571}


#: Variance floor sized to precision of geometries (0.002 A, 0.2 deg, 0.002 in cosine space).
VAR_FLOOR: Dict[str, float] = {"Q1": 0.002**2, "Q4": 0.2**2, "Q2": 0.002**2, "Q3": 0.002**2}

#: Q1 one-sided wall (B2.3, hardcoded True in production).
Q1_WALL: bool = True
Q1_WALL_Z: float = 3.0
Q1_WALL_SD_MIN: float = 0.05
#: Score haptic faces per substituent when the DB has per-substituent entries (hardcoded True in production).
HAPTIC_PER_SUB: bool = True
#: Effectively -inf log-weight for GMM components masked out by q3_mode_override.
#: Finite (not -inf) to avoid nan if a masked row's logsumexp sees only masked entries.
_Q3_OVERRIDE_NEG: float = -1.0e30

_EPS: float = EPS


#: Temperature for soft-min aggregation of Q6 per-pair penalties:
#: -LSE(-tau*p)/tau + log(n)/tau  (tau->0: mean, tau->inf: hard min).
_Q6_LSE_TEMP: float = 2.0


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


_LOG_2PI: float = LOG_2PI

#: as an ANGLE sd in rad, converted to cos at each component's own mean (_cos_floor_var); Q4 (pyramidalisation) in deg.
#: Q5 (torsions) in rad, applied as kappa <= 1 / sigma_T^2 on the von Mises wells; bond class (conjugated 12.1 vs
#: single 13.9 deg) moves it by less than sqrt(2), so one value. Haptic S_orient features (tilt / bend / slip) were not
#: measured and keep their fit-time covariance floor.
#: 299 complexes (100 per metal row), 2026-10-16. Q4 (only without S_orient tables) takes the S_orient angle sd in deg.
THERMAL_SD: Dict[str, float] = {"Q1": 0.0585, "Q2": 0.08609, "SO": 0.07385, "Q4": 4.23, "Q5": 0.2402}

#: Clamp range for kappas synthesised from variances of misfiled GMM entries.
_KAPPA_SYNTH_MIN: float = 0.5
_KAPPA_SYNTH_MAX: float = 10.0


# ---------------------------------------------------------------------------
# Internal helpers (CPU, __init__ only)
# ---------------------------------------------------------------------------


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
    lens = [len(p) for p in param_lists]
    max_k = max(lens)
    lengths = torch.tensor(lens, dtype=torch.long, device=device)
    # packed on the host, one transfer (a per-row device write cost 13 s per FUTSOD solve: 555 calls, ~24 ms each);
    # float64 -> float32 rounds as torch.tensor(list, dtype) does
    buf = np.zeros((len(param_lists), max_k), dtype=np.float64)
    for i, p in enumerate(param_lists):
        if lens[i]:
            buf[i, : lens[i]] = np.asarray(p, dtype=np.float64)
    t = torch.from_numpy(buf).to(dtype=dtype).to(device)
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
            # Q5 GMM vars are stored in deg² but the VMM works in radians:
            # convert var_rad = var_deg * (π/180)² before κ≈1/var.
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


_Q2_FALLBACK = ([0.0], [0.5], [1.0], math.inf)  # broad cos-space Gaussian centred at 90 deg


def _q2_table(stats, key) -> Tuple[List[float], List[float], List[float], float]:
    """(means, vars, weights, nu) of one Q2 pair in cos space; nu = inf for a Gaussian mixture ("gmm_cos"), the
    shared degrees of freedom for a Student-t mixture ("tmix_cos", vars = squared scales)."""
    try:
        d = stats.dist("Q2", key)
        f = _gmm_fields(d, key, "Q2")
    except Exception as exc:
        log.warning("Q2 stats.dist failed for key %s: %s", key, exc)
        return _Q2_FALLBACK
    if f is None:
        return _Q2_FALLBACK
    nu = float(d.nu) if (getattr(d, "type", None) == "tmix_cos" and d.nu is not None) else math.inf
    return list(f[0]), list(f[1]), list(f[2]), nu


def _cos_mix_logp(obs, means, eff_vars, log_w, nu, has_t=None):
    """(B, N) log density of a GMM / Student-t mixture in cos space. obs (B, N); means / eff_vars (N, K) or
    (B, N, K); log_w likewise; nu (N,) or (B, N), inf = Gaussian components. ``has_t``: whether any nu is finite,
    when the caller knows (saves a device->host sync per call on CUDA)."""
    mu = means.unsqueeze(0) if means.dim() == 2 else means
    ev = eff_vars.unsqueeze(0) if eff_vars.dim() == 2 else eff_vars
    lw = log_w.unsqueeze(0) if log_w.dim() == 2 else log_w
    nu_b = (nu.unsqueeze(0) if nu.dim() == 1 else nu).unsqueeze(-1)  # (1|B, N, 1)
    z = (obs.unsqueeze(-1) - mu) ** 2 / ev
    lg = -0.5 * _LOG_2PI - 0.5 * torch.log(ev) - 0.5 * z
    if has_t is None:
        has_t = bool(torch.isfinite(nu_b).any())
    if has_t:
        nf = torch.where(torch.isfinite(nu_b), nu_b, torch.full_like(nu_b, 1.0))
        const = torch.lgamma((nf + 1) / 2) - torch.lgamma(nf / 2) - 0.5 * torch.log(nf * math.pi * ev)
        lt = const - (nf + 1) / 2 * torch.log1p(z / nf)
        lg = torch.where(torch.isfinite(nu_b), lt, lg)
    return torch.logsumexp(lg + lw, dim=-1)



def _torsion_tail_logp(obs, means, kappas, log_norm):
    """Periodic von Mises wells with thermal quadratic continuation past 2 sigma."""
    delta = obs.unsqueeze(-1) - means.unsqueeze(0)
    radius = torch.atan2(torch.sin(delta), torch.cos(delta)).abs()
    base = kappas.unsqueeze(0) * torch.cos(delta) - log_norm.unsqueeze(0)
    sigma = THERMAL_SD['Q5']
    r0 = 2 * sigma
    beyond = (radius - r0).clamp_min(0.)
    import math
    continuation = (kappas.unsqueeze(0) * math.cos(r0) - log_norm.unsqueeze(0)
                    - kappas.unsqueeze(0) * math.sin(r0) * beyond - .5 * (beyond / sigma)**2)
    return torch.where(radius <= r0, base, continuation)


def _cos_floor_var(mu_cos, s_ang: float):
    """Thermal variance in cos units of an angle with mean cos mu_cos and angle sd s_ang (rad): to second order
    Var(cos t) = sin^2 t0 s^2 + cos^2 t0 s^4 / 2, which stays non-zero at 0 / 180 deg where d cos / dt = 0."""
    mu = mu_cos.clamp(-1.0, 1.0) if torch.is_tensor(mu_cos) else np.clip(np.asarray(mu_cos, float), -1.0, 1.0)
    return (1.0 - mu ** 2) * s_ang ** 2 + 0.5 * mu ** 2 * s_ang ** 4


def _var_min(q: str, mode: Optional[str] = None) -> float:
    m = mode or GMM_FLOOR_MODE
    if m in ("fit", "thermal"):
        if m == "thermal" and q in ("Q1", "Q4"):
            return max(VAR_FLOOR[q], THERMAL_SD[q] ** 2)
        return VAR_FLOOR[q]  # thermal Q2 / Q3: mean-dependent, see _floored
    return _GMM_VAR_MIN[q]


def _floored(q: str, vars_: torch.Tensor, means: Optional[torch.Tensor], mode: Optional[str] = None,
             row_sd: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Effective variances of a GMM table under the floor mode (GMM_FLOOR_MODE unless given). row_sd (n,): per-row
    thermal sd of a (..., n, K) table (Q1 by donor element), used instead of the scalar under the thermal floor."""
    m = mode or GMM_FLOOR_MODE
    v = vars_.clamp(min=max(_var_min(q, m), EPS))
    if m == "thermal" and row_sd is not None:
        v = torch.maximum(vars_.clamp(min=max(VAR_FLOOR[q], EPS)), (row_sd.to(v) ** 2)[:, None])
    if m == "thermal" and q in ("Q2", "Q3") and means is not None:
        v = torch.maximum(v, _cos_floor_var(means, THERMAL_SD["Q2" if q == "Q2" else "SO"]).to(v.dtype))
    return v


def _loewner_floor(C: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Smallest change to covariance C with C >= diag(sd^2) in the Loewner order: eigenvalues of the sd-whitened
    covariance clipped at 1 (a multivariate component may not be narrower than the floor in any direction)."""
    D = np.diag(sd)
    Di = np.diag(1.0 / sd)
    w, V = np.linalg.eigh(Di @ C @ Di)
    return D @ (V * np.maximum(w, 1.0)) @ V.T @ D


def _q1_wall(means, vars_, weights):
    """(d_hi, sd) of the Q1 wall for one site's mixture (see Q1_WALL)."""
    best = None
    for m, v, w in zip(means, vars_, weights):
        if w <= 0:
            continue
        sd = max(math.sqrt(max(float(v), 0.0)), Q1_WALL_SD_MIN)
        hi = float(m) + Q1_WALL_Z * sd
        if best is None or hi > best[0]:
            best = (hi, sd)
    if best is None:
        best = (2.1 + Q1_WALL_Z * 0.2, 0.2)
    # the slope uses the scorer's own Q1 sd floor: a wall stiffer than the Gaussian it continues stalls convergence
    return best[0], max(best[1], math.sqrt(_var_min("Q1")))


class _ScorerTablesMixin:
    """Q1–Q6 and per-candidate table setup methods inherited by ``Scorer``."""

    def _init_q1_q6_tables(self, mol, ge, stats, device: torch.device) -> None:
        deg2rad = _DEG2RAD
        metal_sym = mol.metal_symbol
        coord_geom = mol.coord_geom or "octahedral"

        from jacobik.stats import keys as _kb

        q1_keys = _kb.q1_site_keys(mol)
        q2_keys = {(i, j): k for i, j, k in _kb.q2_pair_keys(mol)}

        q1_means_list: List[List[float]] = []
        q1_vars_list: List[List[float]] = []
        q1_weights_list: List[List[float]] = []

        for s_i, site in enumerate(mol.coord_sites):
            key = q1_keys[s_i]
            if key is None:
                q1_means_list.append([2.1])
                q1_vars_list.append([0.04])
                q1_weights_list.append([1.0])
                continue
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

        # per-site thermal sd (solve_floor="thermal"): by donor element, haptic sites "eta"
        sym1 = {a.idx: a.symbol for a in mol.atoms}
        self._q1_thermal_sd = torch.tensor(
            [THERMAL_Q1_BY_DONOR.get("eta" if site.is_haptic else sym1.get(site.donor_indices[0], ""), THERMAL_SD["Q1"])
             for site in mol.coord_sites], dtype=torch.float32, device=device)

        # Stack as padded tensors (n_donors, max_K_q1)
        self._q1_means, _ = _pad_and_stack(q1_means_list, device)
        self._q1_vars, _ = _pad_and_stack(q1_vars_list, device)
        self._q1_weights, _ = _pad_and_stack(q1_weights_list, device)

        # Q3: GMM on cos(angle); means/vars are dimensionless cos-space values.
        q3_means_list: List[List[float]] = []
        q3_vars_list: List[List[float]] = []
        q3_weights_list: List[List[float]] = []

        for s_i, site in enumerate(mol.coord_sites):
            if not ge._q3_eligible[s_i]:
                continue

            subclass = _kb._site_subclass(site, mol)
            anion_bucket = _kb.anionic_co_donor_bucket(
                mol, exclude_rep_idx=_kb._site_representative_idx(site)
            )
            key = (subclass, metal_sym, coord_geom, anion_bucket)
            try:
                dist = stats.dist("Q3", key)
                fields = _gmm_fields(dist, key, "Q3")
            except Exception as exc:
                log.warning("Q3 stats.dist failed for key %s: %s", key, exc)
                fields = None
            if fields is not None:
                q3_means_list.append(list(fields[0]))
                q3_vars_list.append(list(fields[1]))
                q3_weights_list.append(list(fields[2]))
            else:
                # Fallback: cos(180°) = -1.
                q3_means_list.append([-1.0])
                q3_vars_list.append([0.1])
                q3_weights_list.append([1.0])

        self._q3_means, _ = _pad_and_stack(q3_means_list, device)
        self._q3_vars, _ = _pad_and_stack(q3_vars_list, device)
        self._q3_weights, _ = _pad_and_stack(q3_weights_list, device)
        self._n_q3 = len(q3_means_list)

        # ==================================================================
        # Q4 — pyramidalization GMM per eligible kappa donor
        # ge.q4_pyramidalization(X) → (B, n_q4) in DEGREES
        # ==================================================================

        q4_means_list: List[List[float]] = []
        q4_vars_list: List[List[float]] = []
        q4_weights_list: List[List[float]] = []

        pos_to_site: Dict[int, int] = {}
        idx_to_pos = {a.idx: i for i, a in enumerate(mol.atoms)}
        for s_i, site in enumerate(mol.coord_sites):
            if not site.is_haptic:
                d_pos = idx_to_pos.get(site.donor_indices[0], -1)
                if d_pos >= 0:
                    pos_to_site[d_pos] = s_i

        for q4_row, d_pos in enumerate(ge._q4_donor_pos.tolist()):
            s_i = pos_to_site.get(d_pos)
            if s_i is None:
                q4_means_list.append([109.5])
                q4_vars_list.append([100.0])
                q4_weights_list.append([1.0])
                continue
            site = mol.coord_sites[s_i]
            subclass = _kb._site_subclass(site, mol)
            anion_bucket = _kb.anionic_co_donor_bucket(
                mol, exclude_rep_idx=_kb._site_representative_idx(site)
            )
            k_sub = int(ge._q4_s_mask[q4_row].sum().item()) - 1
            key = (subclass, coord_geom, k_sub, anion_bucket)
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
                q5_means_list.append([m * deg2rad for m in fields[0]])
                q5_kappas_list.append([max(float(k), 1e-4) for k in fields[1]])
                q5_weights_list.append(fields[2])
            else:
                # Flat VMM: uniform over [-π, π]  (κ=0 → uniform)
                q5_means_list.append([0.0])
                q5_kappas_list.append([1e-4])
                q5_weights_list.append([1.0])

        self._q5_fold = _q5_fold_for(stats)
        if self._q5_fold and getattr(ge, "_q5_sym", None):
            for q, (nj, nk) in enumerate(ge._q5_sym):
                if nj * nk == 1:
                    continue
                sh = [2.0 * math.pi * (r / nk + t / nj) for r in range(nk) for t in range(nj)]
                m0, k0, w0 = q5_means_list[q], q5_kappas_list[q], q5_weights_list[q]
                q5_means_list[q] = [m + d for d in sh for m in m0]
                q5_kappas_list[q] = [k for _ in sh for k in k0]
                q5_weights_list[q] = [w / len(sh) for _ in sh for w in w0]
        self._q5_means, _ = _pad_and_stack(q5_means_list, device)
        self._q5_kappas, _ = _pad_and_stack(q5_kappas_list, device)
        self._q5_weights, _ = _pad_and_stack(q5_weights_list, device)
        if GMM_FLOOR_MODE == "thermal" and self._q5_kappas.shape[0] > 0:
            # no torsion well sharper than the thermal width: kappa <= 1 / sigma_T^2 (von Mises, large kappa)
            self._q5_kappas = self._q5_kappas.clamp(max=1.0 / THERMAL_SD["Q5"] ** 2)

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
        self._q6_pmf = None
        if ge.n_q6 > 0:
            tabs = stats.q6_pmf_tables() if hasattr(stats, "q6_pmf_tables") else {}
            if tabs:
                from jacobik.stats import steric as _q6

                self._q6_pmf = _q6.build_runtime(mol, ge, tabs, self._q6_mu, self._q6_sigma, device)
            else:
                log.warning("q6_model=%s but the DB has no Q6P tables: using the Q6 wall", Q6_MODEL)

        # ==================================================================
        # Q2 — D–M–D angle GMM-on-cos(angle) per donor pair
        # ==================================================================
        q2_tabs = [_q2_table(stats, q2_keys[(s_i, s_j)])
                   for s_i in range(len(mol.coord_sites)) for s_j in range(s_i + 1, len(mol.coord_sites))]
        self._q2_pair_meta: List[Tuple[int, int, str]] = [
            (s_i, s_j, q2_keys[(s_i, s_j)][2])
            for s_i in range(len(mol.coord_sites)) for s_j in range(s_i + 1, len(mol.coord_sites))]
        self._q2_means, _ = _pad_and_stack([t[0] for t in q2_tabs], device)
        self._q2_vars, _ = _pad_and_stack([t[1] for t in q2_tabs], device)
        self._q2_weights, _ = _pad_and_stack([t[2] for t in q2_tabs], device)
        self._q2_nu = torch.tensor([t[3] for t in q2_tabs], dtype=torch.float32, device=device)
        self._q2_has_t = any(math.isfinite(t[3]) for t in q2_tabs)
        self._q2_cand_has_t = None
        q2_means_list = q2_tabs

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

        # ── Static precomputation ──────────────────────────────────────────────
        def _vmm_precomp(kappas, weights, q_weight):
            """Precompute static VMM log_norm and log_w (with quantity weight)."""
            ln = LOG_2PI + torch.log(torch.special.i0e(kappas).clamp(min=EPS)) + kappas
            lw = torch.log(weights.clamp(min=EPS))
            lw = torch.where(weights > 0, lw, torch.full_like(lw, -1e4))
            if q_weight != 1.0 and q_weight > 0.0:
                lw = lw + math.log(q_weight)
            elif q_weight == 0.0:
                lw = torch.full_like(lw, float("-inf"))
            return ln, lw

        def _gmm_precomp(vars_, weights, q_weight, q, means):
            """Precompute static GMM eff_vars (floored per GMM_FLOOR_MODE) and log_w (with quantity weight)."""
            ev = _floored(q, vars_, means, row_sd=self._q1_thermal_sd if q == "Q1" else None)
            lw = torch.log(weights.clamp(min=EPS))
            if q_weight != 1.0 and q_weight > 0.0:
                lw = lw + math.log(q_weight)
            elif q_weight == 0.0:
                lw = torch.full_like(lw, float("-inf"))
            return ev, lw

        qw = self._qweights

        # Q1 wall (per site)
        wl = [_q1_wall(m, v, w) for m, v, w in zip(q1_means_list, q1_vars_list, q1_weights_list)]
        self._q1_wall_hi = torch.tensor([x[0] for x in wl], dtype=torch.float32, device=device)
        self._q1_wall_sd = torch.tensor([x[1] for x in wl], dtype=torch.float32, device=device)

        # Q1 (GMM)
        self._q1_eff_vars, self._q1_log_w = _gmm_precomp(
            self._q1_vars, self._q1_weights, qw["Q1"], "Q1", self._q1_means
        )

        # Q2 (GMM-cos / t-mixture-cos). Padded components are inert.
        self._q2_eff_vars, self._q2_log_w = _gmm_precomp(
            self._q2_vars, self._q2_weights, qw["Q2"], "Q2", self._q2_means
        )
        self._q2_log_w = torch.where(self._q2_weights > 0, self._q2_log_w, torch.full_like(self._q2_log_w, -1e4))

        # Q3 (GMM-cos)
        self._q3_eff_vars, self._q3_log_w = _gmm_precomp(
            self._q3_vars, self._q3_weights, qw["Q3"], "Q3", self._q3_means
        )

        # Q4 (GMM)
        self._q4_eff_vars, self._q4_log_w = _gmm_precomp(
            self._q4_vars, self._q4_weights, qw["Q4"], "Q4", self._q4_means
        )

        # Q5 (VMM)
        self._q5_log_norm, self._q5_log_w = _vmm_precomp(
            self._q5_kappas, self._q5_weights, qw["Q5"]
        )

        # Q6: log(n_q6)/τ term is also static (depends only on n_q6 and τ)
        self._q6_lse_log_n_over_tau = math.log(max(n_q6, 1)) / _Q6_LSE_TEMP if n_q6 > 0 else 0.0
        self._q6_qweight = float(qw["Q6"])

    def _build_q1_per_candidate(self, mol, stats, assignments, donor_groups, ideal, device):
        """(C, n_sites, K) Q1 means / eff. vars / log-weights, trans partner from each candidate assignment."""
        from jacobik.stats import keys as _kb

        sites = mol.coord_sites
        n = len(sites)
        if n <= 2 or mol.coord_geom in _kb._NO_TRANS_GEOMS or len(donor_groups) != n:
            return None
        site_of_group = {frozenset(site.donor_indices): si for si, site in enumerate(sites)}
        try:
            g2s = [site_of_group[frozenset(g)] for g in donor_groups]
        except KeyError:
            log.warning("Q1 per-candidate trans keys: donor groups do not match coordination sites; skipped.")
            return None
        base_keys = _kb.q1_site_keys(mol)
        subs = [_kb._site_subclass(s, mol) for s in sites]
        ideal = np.asarray(ideal, dtype=float)
        tables = []
        for asg in assignments:
            vec = np.zeros((n, 3))
            for d, s_i in enumerate(g2s):
                vec[s_i] = ideal[int(asg[d])]
            rows = []
            for i in range(n):
                key = base_keys[i]
                if key is None:
                    rows.append(([2.1], [0.04], [1.0]))
                    continue
                cos = vec @ vec[i]
                cos[i] = 2.0
                j = int(np.argmin(cos))
                tp = subs[j] if cos[j] < _TRANS_COS_THRESHOLD else _kb.TRANS_NONE
                key = key[:-1] + (tp,)
                try:
                    f = _gmm_fields(stats.dist("Q1", key), key, "Q1")
                except Exception as exc:
                    log.warning("Q1 stats.dist failed for key %s: %s", key, exc)
                    f = None
                rows.append(f if f is not None else ([2.1], [0.04], [1.0]))
            tables.append(rows)
        K = max(len(r[0]) for t in tables for r in t)
        C = len(tables)
        mu = torch.zeros(C, n, K); var = torch.ones(C, n, K); w = torch.zeros(C, n, K)
        for c, t in enumerate(tables):
            for i, (m, v, ww) in enumerate(t):
                k = len(m)
                mu[c, i, :k] = torch.tensor(m, dtype=torch.float32)
                var[c, i, :k] = torch.tensor(v, dtype=torch.float32)
                w[c, i, :k] = torch.tensor(ww, dtype=torch.float32)
        self._q1_cand_var = var.to(device)
        ev = _floored("Q1", var, mu, row_sd=self._q1_thermal_sd)
        lw = torch.where(w > 0, torch.log(w.clamp(min=EPS)), torch.full_like(w, -1e4))
        qw1 = self._qweights["Q1"]
        if qw1 != 1.0 and qw1 > 0.0:
            lw = lw + math.log(qw1)
        wl = [[_q1_wall(*r) for r in t] for t in tables]
        self._q1_cand_wall = (torch.tensor([[x[0] for x in r] for r in wl], dtype=torch.float32, device=device),
                              torch.tensor([[x[1] for x in r] for r in wl], dtype=torch.float32, device=device))
        return mu.to(device), ev.to(device), lw.to(device)

    def _build_q2_per_candidate(self, mol, stats, assignments, donor_groups, ideal, device):
        """(C, n_pairs, K) Q2 tables, the site-pair class of every pair from each candidate assignment (the ideal
        angle between the two sites' assigned vertices), so the class matches the arrangement being solved."""
        from jacobik.stats import keys as _kb
        from jacobik.stats.ladders import pair_class

        sites = mol.coord_sites
        n = len(sites)
        if n < 2 or len(donor_groups) != n:
            return None
        site_of_group = {frozenset(site.donor_indices): si for si, site in enumerate(sites)}
        try:
            g2s = [site_of_group[frozenset(g)] for g in donor_groups]
        except KeyError:
            log.warning("Q2 per-candidate keys: donor groups do not match coordination sites; skipped.")
            return None
        ideal = np.asarray(ideal, dtype=float)
        cache: Dict[tuple, tuple] = {}
        tables = []
        for asg in assignments:
            vec = np.zeros((n, 3))
            for d, s_i in enumerate(g2s):
                vec[s_i] = ideal[int(asg[d])]
            u = vec / np.linalg.norm(vec, axis=1, keepdims=True).clip(min=1e-12)
            ang = np.degrees(np.arccos(np.clip(u @ u.T, -1.0, 1.0)))
            classes = {(i, j): pair_class(ang[i, j]) for i in range(n) for j in range(i + 1, n)}
            rows = []
            for _i, _j, key in _kb.q2_pair_keys(mol, classes):
                if key not in cache:
                    cache[key] = _q2_table(stats, key)
                rows.append(cache[key])
            tables.append(rows)
        K = max(len(r[0]) for t in tables for r in t)
        C, P = len(tables), len(tables[0])
        mu = torch.zeros(C, P, K); var = torch.ones(C, P, K); w = torch.zeros(C, P, K); nu = torch.zeros(C, P)
        for c, t in enumerate(tables):
            for i, (m, v, ww, nn) in enumerate(t):
                k = len(m)
                mu[c, i, :k] = torch.tensor(m, dtype=torch.float32)
                var[c, i, :k] = torch.tensor(v, dtype=torch.float32)
                w[c, i, :k] = torch.tensor(ww, dtype=torch.float32)
                nu[c, i] = nn
        self._q2_cand_var = var.to(device)
        self._q2_cand_has_t = bool(torch.isfinite(nu).any())
        lw = torch.where(w > 0, torch.log(w.clamp(min=EPS)), torch.full_like(w, -1e4))  # inert padding
        qw2 = self._qweights["Q2"]
        if qw2 != 1.0 and qw2 > 0.0:
            lw = lw + math.log(qw2)
        ev = _floored("Q2", var, mu)
        return mu.to(device), ev.to(device), lw.to(device), nu.to(device)


class _SorientMixin:
    """Joint donor-orientation (S_orient) table builder and evaluator inherited by ``Scorer``."""

    def _build_sorient(self, mol, stats, device):
        """Per-k batched mixture tables for every eligible kappa donor (see stats/orientation.py)."""
        from jacobik.stats import orientation as O

        pos = {a.idx: i for i, a in enumerate(mol.atoms)}
        groups: Dict[int, list] = {1: [], 2: [], 3: []}
        for spec in O.site_specs(mol):
            key, fd = stats.orientation_dist(spec["sub"], spec["metal"], spec["geom"], spec["anion"], spec["k"], spec["ring"])
            if fd is None:
                continue
            self._so_levels.append(key)
            groups[spec["k"]].append((pos[spec["donor"]], [pos[x] for x in spec["subs"]], fd))
        for k, rows in groups.items():
            if not rows:
                continue
            dim = O.DIM[k]
            Kmax = max(len(r[2].weights) for r in rows)
            n = len(rows)
            mu = np.zeros((n, Kmax, dim))
            prec = np.tile(np.eye(dim), (n, Kmax, 1, 1))
            logdet = np.zeros((n, Kmax))
            logw = np.full((n, Kmax), -1e9)  # padding components: effectively zero weight
            for i, (_, _, fd) in enumerate(rows):
                for j, (w, m, C) in enumerate(zip(fd.weights, fd.means, fd.covs)):
                    C = np.asarray(C, dtype=float)
                    if GMM_FLOOR_MODE == "thermal":  # features are cosines of the M-D-X / X-D-X angles
                        C = _loewner_floor(C, np.sqrt(_cos_floor_var(np.asarray(m, float), THERMAL_SD["SO"])))
                    mu[i, j] = m
                    prec[i, j] = np.linalg.inv(C)
                    logdet[i, j] = np.linalg.slogdet(C)[1]
                    logw[i, j] = math.log(w)
            t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
            self._so.append(dict(k=k, dim=dim, donor=t([r[0] for r in rows], torch.long),
                                 subs=t([r[1] for r in rows], torch.long), mu=t(mu), prec=t(prec),
                                 logdet=t(logdet), logw=t(logw)))
        sym = {a.idx: a.symbol for a in mol.atoms}
        by_sel: Dict[tuple, list] = {}
        for spec in O.haptic_site_specs(mol):
            if any(a not in pos for a in spec["face"]):
                continue
            fa = [pos[a] for a in spec["face"]]
            subs = [(a, x) for a, x in spec["subs"] if x in pos]
            # eta2 shared face plane (orientation.ETA2_FACE_PLANE): every substituent of the face defines the normal
            plane = []
            if O.ETA2_FACE_PLANE and len(fa) == 2 and all(sum(1 for a_, _ in subs if a_ == a) >= 2 for a in spec["face"]):
                plane = [(pos[a], pos[x]) for a, x in subs]
            if HAPTIC_PER_SUB and subs:
                ps = []
                for a, x in subs:
                    cls = "H" if sym.get(x) == "H" else "X"
                    kk, fdj = stats.orientation_dist(spec["sub"], spec["metal"], spec["geom"], spec["anion"],
                                                     spec["k"] + O.PER_SUB_OFFSET[cls], spec["ring"])
                    if fdj is None or getattr(fdj, "feats", None) is None or len(fdj.means[0]) != len(fdj.feats):
                        ps = None
                        break
                    ps.append((kk, (pos[a], pos[x]), fdj))
                if ps:
                    for kk, ax, fdj in ps:
                        self._so_levels.append(kk)
                        by_sel.setdefault(tuple(fdj.feats), []).append((fa, bool(spec["closed"]), [ax], fdj, 1.0 / len(ps),
                                                                         plane))
                    continue
            key, fd = stats.orientation_dist(spec["sub"], spec["metal"], spec["geom"], spec["anion"], spec["k"], spec["ring"])
            if fd is None:
                continue
            sel = tuple(fd.feats) if getattr(fd, "feats", None) is not None else tuple(range(O.HAPTIC_DIM))
            if len(fd.means[0]) != len(sel):
                continue
            self._so_levels.append(key)
            by_sel.setdefault(sel, []).append((fa, bool(spec["closed"]), [(pos[a], pos[x]) for a, x in subs], fd, 1.0,
                                               plane))
        self._so_hapt = []
        for sel, hrows in by_sel.items():
            n, F = len(hrows), max(len(r[0]) for r in hrows)
            S = max(max(len(r[2]) for r in hrows), 1)
            face = np.zeros((n, F), dtype=np.int64); fmask = np.zeros((n, F))
            nxt = np.zeros((n, F), dtype=np.int64); emask = np.zeros((n, F))  # Newell edge i -> nxt[i]
            sa = np.zeros((n, S), dtype=np.int64); sx = np.zeros((n, S), dtype=np.int64); smask = np.zeros((n, S))
            Kmax = max(len(r[3].weights) for r in hrows)
            dim = len(sel)
            mu = np.zeros((n, Kmax, dim)); prec = np.tile(np.eye(dim), (n, Kmax, 1, 1))
            logdet = np.zeros((n, Kmax)); logw = np.full((n, Kmax), -1e9)
            eta2 = np.zeros(n)
            wrow = np.array([r[4] for r in hrows])
            Pm = max(max(len(r[5]) for r in hrows), 1)
            pa = np.zeros((n, Pm), dtype=np.int64); px = np.zeros((n, Pm), dtype=np.int64); pm = np.zeros((n, Pm))
            pon = np.zeros(n)
            for i, (fa, closed, subs, fd, _w, plane) in enumerate(hrows):
                m = len(fa)
                pa[i], px[i] = fa[0], fa[1 % m]  # padding: a real (nonzero) vector under mask 0
                for j, (a_, x_) in enumerate(plane):
                    pa[i, j], px[i, j], pm[i, j] = a_, x_, 1.0
                pon[i] = 1.0 if plane else 0.0
                face[i, :m] = fa; fmask[i, :m] = 1.0
                eta2[i] = 1.0 if m == 2 else 0.0
                for j in range(m):
                    if j + 1 < m or closed:
                        nxt[i, j] = (j + 1) % m; emask[i, j] = 1.0
                for j, (a_, x_) in enumerate(subs):
                    sa[i, j], sx[i, j], smask[i, j] = a_, x_, 1.0
                for j, (w, mm, C) in enumerate(zip(fd.weights, fd.means, fd.covs)):
                    C = np.asarray(C, dtype=float)
                    mu[i, j] = mm; prec[i, j] = np.linalg.inv(C); logdet[i, j] = np.linalg.slogdet(C)[1]; logw[i, j] = math.log(w)
            t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
            self._so_hapt.append(dict(face=t(face, torch.long), fmask=t(fmask), nxt=t(nxt, torch.long), emask=t(emask),
                                      sa=t(sa, torch.long), sx=t(sx, torch.long), smask=t(smask), eta2=t(eta2),
                                      sel=t(list(sel), torch.long), dim=dim, w=t(wrow),
                                      plane=bool(pon.any()), pa=t(pa, torch.long), px=t(px, torch.long), pm=t(pm), pon=t(pon),
                                      mu=t(mu), prec=t(prec), logdet=t(logdet), logw=t(logw)))
        self._so_n = sum(int(g["donor"].shape[0]) for g in self._so) + int(round(sum(r[4] for rows in by_sel.values() for r in rows)))

    def _sorient(self, X: torch.Tensor, Z: Optional[float] = None) -> torch.Tensor:
        """(B,) mean over eligible donors of log p(g) / dim(g); with Z (summed aggregation): sum of log p(g) / Z."""
        M = X[:, self._ge.metal_pos_idx, :]
        total = X.new_zeros(X.shape[0])
        for g in self._so:
            D = X[:, g["donor"], :]  # (B, n, 3)
            uM = torch.nn.functional.normalize(M[:, None, :] - D, dim=-1)
            us = torch.nn.functional.normalize(X[:, g["subs"], :] - D[:, :, None, :], dim=-1)  # (B, n, k, 3)
            cm = (uM[:, :, None, :] * us).sum(-1)  # (B, n, k)
            cm_s, order = torch.sort(cm, dim=-1)
            us = torch.gather(us, 2, order[..., None].expand_as(us))
            feats = [cm_s]
            k = g["k"]
            if k > 1:
                feats.append(torch.stack([(us[:, :, i, :] * us[:, :, j, :]).sum(-1)
                                          for i in range(k) for j in range(i + 1, k)], dim=-1))
            f = torch.cat(feats, dim=-1)  # (B, n, dim)
            diff = f[:, :, None, :] - g["mu"][None]  # (B, n, K, dim)
            maha = torch.einsum("bnki,nkij,bnkj->bnk", diff, g["prec"], diff)
            logn = -0.5 * (maha + g["logdet"][None] + g["dim"] * LOG_2PI)
            lp = torch.logsumexp(logn + g["logw"][None], dim=-1)  # (B, n)
            total = total + (lp if Z is not None else lp / g["dim"]).sum(-1)
        F = torch.nn.functional
        for h in self._so_hapt:
            P = X[:, h["face"], :]  # (B, n, F, 3)
            fm = h["fmask"][None, :, :, None]
            c = (P * fm).sum(2) / fm.sum(2).clamp(min=1.0)  # (B, n, 3)
            Q = P - c[:, :, None, :]
            Qn = torch.gather(Q, 2, h["nxt"][None, :, :, None].expand(Q.shape[0], -1, -1, 3))
            nrm = (torch.cross(Q, Qn, dim=-1) * h["emask"][None, :, :, None]).sum(2)  # Newell normal (B, n, 3)
            uM = F.normalize(M[:, None, :] - c, dim=-1)
            bvec = F.normalize(P[:, :, 1, :] - P[:, :, 0, :], dim=-1)  # eta2: normal = uM minus its bond component
            n2 = uM - (uM * bvec).sum(-1, keepdim=True) * bvec
            if h["plane"]:  # shared eta2 face plane where it qualifies (orientation.eta2_plane_normal)
                on = h["pon"][None].expand(X.shape[0], -1)
                ns = eta2_plane_normal_t(bvec, X[:, h["pa"], :], X[:, h["px"], :], h["pm"][None].expand(X.shape[0], -1, -1), on)
                n2 = on[..., None] * ns + (1.0 - on[..., None]) * n2
            e2 = h["eta2"][None, :, None]
            nrm = F.normalize(e2 * n2 + (1.0 - e2) * nrm, dim=-1)
            tilt = (nrm * uM).sum(-1)
            sgn = torch.where(tilt < 0, -1.0, 1.0)  # orient the normal towards the metal (no grad through sign)
            nrm, tilt = nrm * sgn[..., None], tilt * sgn
            A, Xs = X[:, h["sa"], :], X[:, h["sx"], :]  # (B, n, S, 3)
            ub = F.normalize(Xs - A, dim=-1)
            sm = h["smask"][None]
            bend = ((ub * nrm[:, :, None, :]).sum(-1) * sm).sum(-1) / sm.sum(-1).clamp(min=1.0)
            fm2 = h["fmask"][None]
            nf = fm2.sum(-1).clamp(min=1.0)
            d = torch.linalg.norm(P - M[:, None, None, :], dim=-1)  # (B, n, F)
            dmean = (d * fm2).sum(-1) / nf
            slip = torch.sqrt((((d - dmean[..., None]) ** 2) * fm2).sum(-1) / nf + 1e-12) / dmean
            f = torch.stack([tilt, bend, slip], dim=-1)[..., h["sel"]]  # (B, n, dim)
            diff = f[:, :, None, :] - h["mu"][None]
            maha = torch.einsum("bnki,nkij,bnkj->bnk", diff, h["prec"], diff)
            logn = -0.5 * (maha + h["logdet"][None] + h["dim"] * LOG_2PI)
            lp = torch.logsumexp(logn + h["logw"][None], dim=-1)
            total = total + (lp * h["w"][None] / (1.0 if Z is not None else h["dim"])).sum(-1)
        return total / (Z if Z is not None else max(self._so_n, 1))


class Scorer(_ScorerTablesMixin, _SorientMixin):
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
                       Missing keys default to 1.0 (0.0 disables a quantity).
                       Weights are baked into the precomputed log_w tensors.
    """

    def __init__(
        self,
        mol: "TMComplex",
        ge: GeometryExtractor,
        stats: Any,
        device: torch.device,
        quantity_weights: Optional[Dict[str, float]] = None,
        assignments: Optional[List[np.ndarray]] = None,
        donor_groups: Optional[List[List[int]]] = None,
        ideal: Optional[np.ndarray] = None,
        orientation_model: str = "auto",
        angle_bend_penalty: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        angle_bend_penalty: (free_deg, sigma_deg, weight) when angle bends are on (ik/angle_bends.py): adds the
        flat-bottom bond-angle penalty "Q7" at every bend pivot. None (default) = no bends, no Q7.
        orientation_model: "auto" (default) uses the joint donor-orientation model S_orient in place of
        Q3 + Q4 whenever the DB has QO fits, else Q3 + Q4; "sorient" requires QO fits; "q3q4" forces the
        separate Q3/Q4 terms.

        assignments / donor_groups / ideal (optional, all three together): the candidate donor-to-site
        assignments of this solve (assignments[c][d] = site index of donor group d; donor_groups[d] = its
        atom indices; ideal = (n_sites, 3) site vectors). When given, each Q1 key's trans partner is read from
        the ASSIGNMENT (the site > 140 deg away, same rule as the extractor), one Q1 table per candidate, and
        score(..., row_groups=candidate index per row) uses each row's own table. Without them the trans
        partner is taken from the current shell geometry (previous behaviour).
        """
        self.device = device
        self._ge = ge
        self.mol = mol
        self.stats = stats

        # Normalise quantity weights: default 1.0 for all, caller overrides.
        _qw = quantity_weights or {}
        self._qweights = {q: float(_qw.get(q, 1.0)) for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")}

        self._init_q1_q6_tables(mol, ge, stats, device)

        self._q1_cand = None
        self._q2_cand = None
        if assignments is not None and donor_groups is not None and ideal is not None:
            self._q1_cand = self._build_q1_per_candidate(mol, stats, assignments, donor_groups, ideal, device)
            self._q2_cand = self._build_q2_per_candidate(mol, stats, assignments, donor_groups, ideal, device)

        # Joint donor orientation (S_orient) replacing Q3 + Q4.
        if orientation_model not in ("auto", "sorient", "q3q4"):
            raise ValueError(f"orientation_model must be 'auto', 'sorient' or 'q3q4', not {orientation_model!r}")
        has_qo = bool(getattr(stats, "has_orientation", False))
        if orientation_model == "sorient" and not has_qo:
            raise ValueError("orientation_model='sorient' but the constraints DB has no QO (orientation) fits; "
                             "refit with the S_orient section or use orientation_model='q3q4'")
        self.orientation_model = "sorient" if (has_qo and orientation_model != "q3q4") else "q3q4"
        self._so: List[dict] = []
        self._so_n = 0
        self._so_hapt: List[dict] = []
        self._so_levels: List[tuple] = []
        if self.orientation_model == "sorient":
            self._build_sorient(mol, stats, device)

        # Angle-bend penalty (Q7), reference angles = the embed's (current shell) angles.
        self._q7 = None
        if angle_bend_penalty is not None:
            from jacobik.ik import angle_bends as AB

            trip = AB.angle_triplets(mol)
            if trip:
                pos = {a.idx: i for i, a in enumerate(mol.atoms)}
                t = torch.tensor([[pos[a], pos[p], pos[b]] for a, p, b in trip], dtype=torch.long, device=device)
                X0 = torch.tensor(np.array([a.position for a in mol.atoms]), dtype=torch.float32, device=device)
                a0 = AB.angles(X0[None], t)[0]
                # The energy-calibrated solver bend prior is the production default and does not create a scorer Q7
                # term; the legacy angle-penalty path (free zone, width, weight) penalises every pivot.
                free7, sig7, weight7 = (float(x) for x in angle_bend_penalty)
                self._q7 = (t, a0, (free7, sig7, weight7))

    def _agg_norm(self) -> float:
        """Number of internal features (Q1 donors, Q2 pairs, S_orient donors or Q3+Q4 terms, Q5 torsions): the one
        constant the summed log-likelihood is divided by (keeps the legacy score scale; ranking = plain sum)."""
        n = getattr(self, "_n_int", None)
        if n is None:
            n3 = self._so_n if self._so_n > 0 else (self._n_q3 + self._ge.n_q4)
            n2 = self._n_q2 if self._qweights["Q2"] > 0.0 else 0
            n = self._n_int = float(max(self._n_q1 + n2 + n3 + self._ge.n_q5, 1))
        return n

    def ranking_score(self, X: torch.Tensor, row_groups: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B,) the score the final ranking uses, per RANK_AGG / Q1_TAIL_SIGMA (module flags), at the fit-time floors.
        RANK_AGG="sum": the total log-likelihood over all features (every contact pair's wall counts), divided by the
        constant _agg_norm(), instead of one mean per quantity. Q1_TAIL_SIGMA > 0: Q1 mixture components broader than
        it (or with a mean below 1.2 A, not a bond) are treated as the outlier share and dropped, weights renormalised
        (the Q5 well rule applied to Q1); a donor whose components would all go keeps its table.
        Call from one thread only: it swaps scorer state temporarily."""
        global SCORE_AGG
        from jacobik.ik.ik_solver import _RANK_FLOOR  # profile field rank_floor_fit (live module attribute)

        old_agg, old_q1c = SCORE_AGG, getattr(self, "_q1_cand", None)
        try:
            if RANK_AGG == "sum":
                SCORE_AGG = "sum"
            # rank with the fit-time floors (legacy) or with the solve's own floors (rank_floor_fit = false: one
            # objective, the thermal floors in both)
            self.set_gmm_floor("fit" if _RANK_FLOOR else GMM_FLOOR_MODE)
            q1c = getattr(self, "_q1_cand", None)
            if Q1_TAIL_SIGMA > 0 and q1c is not None and getattr(self, "_q1_cand_var", None) is not None:
                mu, ev, lw = q1c
                real = lw > -1e3  # padded components carry -1e4 (inert)
                keep = real & ~((self._q1_cand_var.sqrt() > Q1_TAIL_SIGMA) | (mu < 1.2))
                keep = torch.where(keep.any(-1, keepdim=True), keep, real)
                lw2 = torch.where(keep, lw, torch.full_like(lw, -1e30))
                self._q1_cand = (mu, ev, lw2 - torch.logsumexp(lw2, -1, keepdim=True))
            with torch.no_grad():
                s, _ = self.score(X, row_groups=row_groups)
        finally:
            SCORE_AGG = old_agg
            self.set_gmm_floor(GMM_FLOOR_MODE)
            if old_q1c is not None:
                self._q1_cand = (old_q1c[0], self._q1_cand[1], old_q1c[2])
        return s

    def set_gmm_floor(self, mode: str) -> None:
        """Re-derive the GMM effective variances of Q1-Q4 with the "scorer" or "fit" floor (see GMM_FLOOR_MODE).
        Means, weights and the Q1 wall are unchanged."""
        for q in ("Q1", "Q2", "Q3", "Q4"):
            v = getattr(self, f"_{q.lower()}_vars", None)
            if v is not None:
                setattr(self, f"_{q.lower()}_eff_vars", _floored(q, v, getattr(self, f"_{q.lower()}_means", None), mode,
                                                                 row_sd=getattr(self, "_q1_thermal_sd", None) if q == "Q1" else None))
        q1c = getattr(self, "_q1_cand", None)
        if q1c is not None and getattr(self, "_q1_cand_var", None) is not None:
            self._q1_cand = (q1c[0], _floored("Q1", self._q1_cand_var, q1c[0], mode,
                                              row_sd=getattr(self, "_q1_thermal_sd", None)), q1c[2])
        q2c = getattr(self, "_q2_cand", None)
        if q2c is not None and getattr(self, "_q2_cand_var", None) is not None:
            self._q2_cand = (q2c[0], _floored("Q2", self._q2_cand_var, q2c[0], mode), q2c[2], q2c[3])

    def _score_components(
        self,
        X: torch.Tensor,
        q3_mode_override: Optional[torch.Tensor] = None,
        donor_pos: Optional[torch.Tensor] = None,
        row_groups: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute each quantity's normalised (mean over terms) log-prob as a (B,)
        tensor, weighted by its quantity weight.  Static parameters are
        precomputed at __init__; all Q1–Q5 terms are evaluated batched.

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
        agg_sum = SCORE_AGG == "sum"
        Z = self._agg_norm() if agg_sum else None

        def _div(n_terms):
            return Z if agg_sum else max(n_terms, 1)

        def _vmm(obs, means, kappas, log_norm, log_w, n_terms):
            cos_diff = torch.cos(obs.unsqueeze(-1) - means.unsqueeze(0))  # (B, N, K)
            log_comp = kappas.unsqueeze(0) * cos_diff - log_norm.unsqueeze(0)
            log_p = log_comp + log_w.unsqueeze(0)  # (B, N, K)
            per_term = torch.logsumexp(log_p, dim=-1)  # (B, N)
            return per_term.sum(dim=-1) / _div(n_terms)

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
            # log_w is static (N, K), or already batched (B, N, K) for the Q3 override.
            log_w_b = log_w.unsqueeze(0) if log_w.dim() == 2 else log_w  # (B, N, K)
            log_p = log_comp + log_w_b  # (B, N, K)
            per_term = torch.logsumexp(log_p, dim=-1)  # (B, N)
            return per_term.sum(dim=-1) / _div(n_terms)

        # ── Q1 ─────────────────────────────────────────────────────────────
        if self._n_q1 > 0 and self._q1_means.shape[0] > 0:
            q1_d = self._ge.q1_distances(X, donor_pos=donor_pos)  # (B, n_donors)
            q1c = getattr(self, "_q1_cand", None)
            rg = row_groups
            if q1c is not None and (rg is not None or q1c[0].shape[0] == 1):
                # Per-row tables: trans partner from each row's own candidate assignment.
                mu_c, ev_c, lw_c = q1c
                gi = rg.to(mu_c.device) if rg is not None else torch.zeros(B, dtype=torch.long, device=mu_c.device)
                mu_b, ev_b, lw_b = mu_c[gi], ev_c[gi], lw_c[gi]  # (B, n, K)
                diff = q1_d.unsqueeze(-1) - mu_b
                log_p = -0.5 * _LOG_2PI - 0.5 * torch.log(ev_b) - 0.5 * diff**2 / ev_b + lw_b
                lp1 = torch.logsumexp(log_p, dim=-1)
                comp["Q1"] = lp1.sum(dim=-1) / _div(self._n_q1)
                hi, sd = self._q1_cand_wall[0][gi], self._q1_cand_wall[1][gi]  # (B, n)
            else:
                comp["Q1"] = _gmm(q1_d, self._q1_means, self._q1_eff_vars, self._q1_log_w, self._n_q1)
                hi, sd = self._q1_wall_hi[None], self._q1_wall_sd[None]
            if Q1_WALL:
                ex = torch.clamp(q1_d - hi, min=0.0) / sd
                comp["Q1"] = comp["Q1"] - 0.5 * (ex**2).sum(dim=-1) / _div(self._n_q1)

        # ── Q2 ─────────────────────────────────────────────────────────────
        if self._n_q2 > 0 and self._q2_means.shape[0] > 0 and self._qweights["Q2"] > 0.0:
            q2_angles = self._ge.q2_dmd_angles(X, donor_pos=donor_pos)  # (B, n_pairs) radians
            q2_cos = torch.cos(q2_angles)
            q2c = getattr(self, "_q2_cand", None)
            if q2c is not None and (row_groups is not None or q2c[0].shape[0] == 1):
                # per-row tables: site-pair class from each row's own candidate assignment
                gi = (row_groups.to(q2c[0].device) if row_groups is not None
                      else torch.zeros(B, dtype=torch.long, device=q2c[0].device))
                lp2 = _cos_mix_logp(q2_cos, q2c[0][gi], q2c[1][gi], q2c[2][gi], q2c[3][gi],
                                    has_t=self._q2_cand_has_t)
            else:
                lp2 = _cos_mix_logp(q2_cos, self._q2_means, self._q2_eff_vars, self._q2_log_w, self._q2_nu,
                                    has_t=self._q2_has_t)
            comp["Q2"] = lp2.sum(dim=-1) / _div(self._n_q2)

        # ── Q3 ─────────────────────────────────────────────────────────────
        # (not evaluated when S_orient replaces Q3 + Q4 below: the values were discarded, and the haptic
        # approach angle costs a batched SVD per step)
        if self._so_n == 0 and self._n_q3 > 0 and self._q3_means.shape[0] > 0:
            q3_a = self._ge.q3_approach_angles(X)  # (B, n_q3) radians
            q3_cos = torch.cos(q3_a)
            q3_log_w_eff = self._q3_log_w
            # q3_mode_override is (B, n_q3) int64, -1 = no override. Where set,
            # collapse the mixture to the named component (log-weight 0, all
            # others _Q3_OVERRIDE_NEG) so logsumexp is exactly that component's density.
            if q3_mode_override is not None:
                Bq = q3_cos.shape[0]
                Nq, Kq = self._q3_log_w.shape
                mask = q3_mode_override >= 0  # (B, N)
                override_log_w = q3_cos.new_full((Bq, Nq, Kq), _Q3_OVERRIDE_NEG)
                sel = q3_mode_override.clamp(min=0).unsqueeze(-1)  # (B, N, 1) -- safe idx everywhere
                override_log_w.scatter_(-1, sel, torch.zeros_like(sel, dtype=override_log_w.dtype))
                q3_log_w_eff = torch.where(
                    mask.unsqueeze(-1),
                    override_log_w,
                    self._q3_log_w.unsqueeze(0).expand(Bq, -1, -1),
                )
            comp["Q3"] = _gmm(
                q3_cos, self._q3_means, self._q3_eff_vars, q3_log_w_eff, self._n_q3
            )

        # ── Q4 ─────────────────────────────────────────────────────────────
        if self._so_n == 0 and self._ge.n_q4 > 0 and self._q4_means.shape[0] > 0:
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
        if self._q6_pmf is not None and self._q6_qweight > 0.0:
            from jacobik.stats.steric import evaluate as _q6pmf_eval

            comp["Q6"] = -self._q6_qweight * _q6pmf_eval(self._q6_pmf, X, self._ge._q6_a, self._ge._q6_b) / self._agg_norm()
        elif agg_sum and self._ge.n_q6 > 0 and self._q6_mu.shape[0] > 0 and self._q6_qweight > 0.0:
            # total log-likelihood: every pair's one-sided wall term counts (not the soft-max worst contact)
            d6 = (X[:, self._ge._q6_a] - X[:, self._ge._q6_b]).norm(dim=-1)  # (B, M)
            t6 = (self._q6_mu - d6).clamp(min=0.0)
            comp["Q6"] = self._q6_qweight * (-(t6**2) / (2.0 * self._q6_sigma.clamp(min=_EPS) ** 2)).sum(-1) / Z
        elif self._ge.n_q6 > 0 and self._q6_mu.shape[0] > 0 and self._q6_qweight > 0.0:
            q6_val = q6_softmin(
                X,
                self._ge._q6_a,
                self._ge._q6_b,
                self._q6_mu,
                self._q6_sigma,
                _Q6_LSE_TEMP,
            ) + self._q6_lse_log_n_over_tau
            comp["Q6"] = self._q6_qweight * q6_val

        # ── S_orient replaces Q3 + Q4 (its features contain both) ─────────────
        if self._so_n > 0:
            comp["Q3"] = self._qweights["Q3"] * self._sorient(X, Z)
            comp["Q4"] = torch.zeros_like(comp["Q4"])

        # ── Q7: bond-angle bend penalty (only when angle bends are on) ─────────
        if self._q7 is not None:
            from jacobik.ik import angle_bends as AB

            t7, a7, (free7, sig7, w7) = self._q7
            comp["Q7"] = AB.penalty(X, t7, a7, free7, sig7, w7)

        return comp

    def score(
        self,
        X: torch.Tensor,
        q3_mode_override: Optional[torch.Tensor] = None,
        donor_pos: Optional[torch.Tensor] = None,
        row_groups: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Unified statistical score: the sum of the per-quantity weighted values.

        Parameters
        ----------
        X : (B, N, 3)
        q3_mode_override : optional (B, n_q3) int64; >= 0 hard-selects that Q3 GMM
                           component for the (trial, term); -1 = no override
        donor_pos : optional precomputed self._ge.donor_positions(X), shared
                    with Q1/Q2 to avoid recomputation

        Returns
        -------
        scalar    : (B,)  — sum of per-quantity weighted values; higher = better
        breakdown : dict with keys "Q1".."Q6", each a (B,) contribution
        """
        # row_groups is passed through the call, never stored on the scorer: one scorer is shared by
        # the solver's worker threads, each scoring its own chunk of rows.
        breakdown = self._score_components(
            X, q3_mode_override=q3_mode_override, donor_pos=donor_pos, row_groups=row_groups
        )
        scalar = X.new_zeros(X.shape[0])
        for v in breakdown.values():
            scalar = scalar + v
        return scalar, breakdown  # (B,), dict[str, (B,)]
