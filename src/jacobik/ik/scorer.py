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

import cupy as cp
import cupyx.scipy.special as cspec
import numpy as np

from jacobik.core import gpu

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


def _logsumexp(x: cp.ndarray, axis: int = -1, keepdims: bool = False) -> cp.ndarray:
    m = cp.max(x, axis=axis, keepdims=True)
    out = m + cp.log(cp.sum(cp.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else cp.squeeze(out, axis=axis)


def _pad_and_stack(
    param_lists: List[List[float]],
    device: Any = "cuda",
    dtype: Any = cp.float32,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Pad a list of variable-length float lists to a common length and stack.

    Returns
    -------
    tensor  : (N, max_K) float32 padded array
    lengths : (N,) int64 — original lengths (for masked computation)
    """
    if not param_lists:
        return (
            cp.zeros((0, 1), dtype=dtype),
            cp.zeros((0,), dtype=cp.int64),
        )
    lens = [len(p) for p in param_lists]
    max_k = max(lens)
    lengths = cp.asarray(lens, dtype=cp.int64)
    buf = np.zeros((len(param_lists), max_k), dtype=np.float64)
    for i, p in enumerate(param_lists):
        if lens[i]:
            buf[i, : lens[i]] = np.asarray(p, dtype=np.float64)
    t = cp.asarray(buf, dtype=dtype)
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
    (B, N, K); log_w likewise; nu (N,) or (B, N), inf = Gaussian components."""
    mu = means[None] if means.ndim == 2 else means
    ev = eff_vars[None] if eff_vars.ndim == 2 else eff_vars
    lw = log_w[None] if log_w.ndim == 2 else log_w
    nu_b = (nu[None] if nu.ndim == 1 else nu)[..., None]  # (1|B, N, 1)
    z = (obs[..., None] - mu) ** 2 / ev
    lg = -0.5 * _LOG_2PI - 0.5 * cp.log(ev) - 0.5 * z
    if has_t is None:
        has_t = bool(cp.isfinite(nu_b).any())
    if has_t:
        nf = cp.where(cp.isfinite(nu_b), nu_b, cp.ones_like(nu_b))
        const = cspec.gammaln((nf + 1) / 2) - cspec.gammaln(nf / 2) - 0.5 * cp.log(nf * math.pi * ev)
        lt = const - (nf + 1) / 2 * cp.log1p(z / nf)
        lg = cp.where(cp.isfinite(nu_b), lt, lg)
    return _logsumexp(lg + lw, axis=-1)


def _torsion_tail_logp(obs, means, kappas, log_norm):
    """Periodic von Mises wells with thermal quadratic continuation past 2 sigma."""
    delta = obs[..., None] - means[None]
    radius = cp.abs(cp.arctan2(cp.sin(delta), cp.cos(delta)))
    base = kappas[None] * cp.cos(delta) - log_norm[None]
    sigma = THERMAL_SD['Q5']
    r0 = 2 * sigma
    beyond = cp.maximum(radius - r0, 0.0)
    continuation = (kappas[None] * math.cos(r0) - log_norm[None]
                    - kappas[None] * math.sin(r0) * beyond - 0.5 * (beyond / sigma) ** 2)
    return cp.where(radius <= r0, base, continuation)


def _cos_floor_var(mu_cos, s_ang: float):
    """Thermal variance in cos units of an angle with mean cos mu_cos and angle sd s_ang (rad): to second order
    Var(cos t) = sin^2 t0 s^2 + cos^2 t0 s^4 / 2, which stays non-zero at 0 / 180 deg where d cos / dt = 0."""
    mu = cp.clip(mu_cos, -1.0, 1.0) if isinstance(mu_cos, cp.ndarray) else np.clip(np.asarray(mu_cos, float), -1.0, 1.0)
    return (1.0 - mu ** 2) * s_ang ** 2 + 0.5 * mu ** 2 * s_ang ** 4


def _var_min(q: str, mode: Optional[str] = None) -> float:
    m = mode or GMM_FLOOR_MODE
    if m in ("fit", "thermal"):
        if m == "thermal" and q in ("Q1", "Q4"):
            return max(VAR_FLOOR[q], THERMAL_SD[q] ** 2)
        return VAR_FLOOR[q]  # thermal Q2 / Q3: mean-dependent, see _floored
    return _GMM_VAR_MIN[q]


def _floored(q: str, vars_: cp.ndarray, means: Optional[cp.ndarray], mode: Optional[str] = None,
             row_sd: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Effective variances of a GMM table under the floor mode (GMM_FLOOR_MODE unless given). row_sd (n,): per-row
    thermal sd of a (..., n, K) table (Q1 by donor element), used instead of the scalar under the thermal floor."""
    m = mode or GMM_FLOOR_MODE
    v = cp.maximum(vars_, max(_var_min(q, m), EPS))
    if m == "thermal" and row_sd is not None:
        v = cp.maximum(cp.maximum(vars_, max(VAR_FLOOR[q], EPS)), (row_sd.astype(v.dtype) ** 2)[:, None])
    if m == "thermal" and q in ("Q2", "Q3") and means is not None:
        v = cp.maximum(v, _cos_floor_var(means, THERMAL_SD["Q2" if q == "Q2" else "SO"]).astype(v.dtype))
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

    def _init_q1_q6_tables(self, mol, ge, stats, device: Any) -> None:
        deg2rad = _DEG2RAD
        metal_sym = mol.metal_symbol
        coord_geom = mol.coord_geom or "octahedral"

        from jacobik.stats import keys as _kb

        q1_keys = _kb.q1_site_keys(mol)
        q2_keys = {(i, j): k for i, j, k in _kb.q2_pair_keys(mol)}

        def _fetch(qty, keys_iter, extractor, fallback):
            c0, c1, c2 = [], [], []
            for key in keys_iter:
                fields = None
                if key is not None:
                    try:
                        fields = extractor(stats.dist(qty, key), key, qty)
                    except Exception as exc:
                        log.warning("%s stats.dist failed for key %s: %s", qty, key, exc)
                row = fields if fields is not None else fallback
                c0.append(list(row[0])); c1.append(list(row[1])); c2.append(list(row[2]))
            return c0, c1, c2

        q1_means_list, q1_vars_list, q1_weights_list = _fetch("Q1", q1_keys, _gmm_fields, ([2.1], [0.04], [1.0]))
        sym1 = {a.idx: a.symbol for a in mol.atoms}
        self._q1_thermal_sd = cp.asarray(
            [THERMAL_Q1_BY_DONOR.get("eta" if site.is_haptic else sym1.get(site.donor_indices[0], ""), THERMAL_SD["Q1"])
             for site in mol.coord_sites], dtype=cp.float32)
        self._q1_means, _ = _pad_and_stack(q1_means_list, device)
        self._q1_vars, _ = _pad_and_stack(q1_vars_list, device)
        self._q1_weights, _ = _pad_and_stack(q1_weights_list, device)

        q3_keys = [
            (_kb._site_subclass(site, mol), metal_sym, coord_geom,
             _kb.anionic_co_donor_bucket(mol, exclude_rep_idx=_kb._site_representative_idx(site)))
            for s_i, site in enumerate(mol.coord_sites) if ge._q3_eligible[s_i]
        ]
        q3_m, q3_v, q3_w = _fetch("Q3", q3_keys, _gmm_fields, ([-1.0], [0.1], [1.0]))
        self._q3_means, _ = _pad_and_stack(q3_m, device)
        self._q3_vars, _ = _pad_and_stack(q3_v, device)
        self._q3_weights, _ = _pad_and_stack(q3_w, device)
        self._n_q3 = len(q3_m)

        idx_to_pos = {a.idx: i for i, a in enumerate(mol.atoms)}
        pos_to_site = {
            idx_to_pos[site.donor_indices[0]]: s_i
            for s_i, site in enumerate(mol.coord_sites)
            if not site.is_haptic and site.donor_indices[0] in idx_to_pos
        }
        q4_keys = []
        for q4_row, d_pos in enumerate(ge._q4_donor_pos.tolist()):
            s_i = pos_to_site.get(d_pos)
            if s_i is None:
                q4_keys.append(None)
            else:
                site = mol.coord_sites[s_i]
                q4_keys.append((
                    _kb._site_subclass(site, mol), coord_geom,
                    int(ge._q4_s_mask[q4_row].sum().item()) - 1,
                    _kb.anionic_co_donor_bucket(mol, exclude_rep_idx=_kb._site_representative_idx(site)),
                ))
        q4_m, q4_v, q4_w = _fetch("Q4", q4_keys, _gmm_fields, ([109.5], [100.0], [1.0]))
        self._q4_means, _ = _pad_and_stack(q4_m, device)
        self._q4_vars, _ = _pad_and_stack(q4_v, device)
        self._q4_weights, _ = _pad_and_stack(q4_w, device)

        q5_means_list, q5_kappas_list, q5_weights_list = _fetch("Q5", ge._q5_keys, _vmm_fields, ([0.0], [1e-4], [1.0]))
        q5_means_list = [[m * deg2rad for m in row] for row in q5_means_list]
        q5_kappas_list = [[max(float(k), 1e-4) for k in row] for row in q5_kappas_list]

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
            self._q5_kappas = cp.minimum(self._q5_kappas, 1.0 / THERMAL_SD["Q5"] ** 2)

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
            self._q6_mu = cp.asarray(q6_mu_list, dtype=cp.float32)
            self._q6_sigma = cp.asarray(q6_sigma_list, dtype=cp.float32)
        else:
            self._q6_mu = cp.zeros((0,), dtype=cp.float32)
            self._q6_sigma = cp.zeros((0,), dtype=cp.float32)
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
        self._q2_nu = cp.asarray([t[3] for t in q2_tabs], dtype=cp.float32)
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
            ln = LOG_2PI + cp.log(cp.maximum(cspec.i0e(kappas), EPS)) + kappas
            lw = cp.log(cp.maximum(weights, EPS))
            lw = cp.where(weights > 0, lw, cp.full_like(lw, -1e4))
            if q_weight != 1.0 and q_weight > 0.0:
                lw = lw + math.log(q_weight)
            elif q_weight == 0.0:
                lw = cp.full_like(lw, float("-inf"))
            return ln, lw

        def _gmm_precomp(vars_, weights, q_weight, q, means):
            """Precompute static GMM eff_vars (floored per GMM_FLOOR_MODE) and log_w (with quantity weight)."""
            ev = _floored(q, vars_, means, row_sd=self._q1_thermal_sd if q == "Q1" else None)
            lw = cp.log(cp.maximum(weights, EPS))
            if q_weight != 1.0 and q_weight > 0.0:
                lw = lw + math.log(q_weight)
            elif q_weight == 0.0:
                lw = cp.full_like(lw, float("-inf"))
            return ev, lw

        qw = self._qweights

        # Q1 wall (per site)
        wl = [_q1_wall(m, v, w) for m, v, w in zip(q1_means_list, q1_vars_list, q1_weights_list)]
        self._q1_wall_hi = cp.asarray([x[0] for x in wl], dtype=cp.float32)
        self._q1_wall_sd = cp.asarray([x[1] for x in wl], dtype=cp.float32)

        # Q1 (GMM)
        self._q1_eff_vars, self._q1_log_w = _gmm_precomp(
            self._q1_vars, self._q1_weights, qw["Q1"], "Q1", self._q1_means
        )

        # Q2 (GMM-cos / t-mixture-cos). Padded components are inert.
        self._q2_eff_vars, self._q2_log_w = _gmm_precomp(
            self._q2_vars, self._q2_weights, qw["Q2"], "Q2", self._q2_means
        )
        self._q2_log_w = cp.where(self._q2_weights > 0, self._q2_log_w, cp.full_like(self._q2_log_w, -1e4))

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
        mu = cp.zeros((C, n, K), dtype=cp.float32)
        var = cp.ones((C, n, K), dtype=cp.float32)
        w = cp.zeros((C, n, K), dtype=cp.float32)
        for c, t in enumerate(tables):
            for i, (m, v, ww) in enumerate(t):
                k = len(m)
                mu[c, i, :k] = cp.asarray(m, dtype=cp.float32)
                var[c, i, :k] = cp.asarray(v, dtype=cp.float32)
                w[c, i, :k] = cp.asarray(ww, dtype=cp.float32)
        self._q1_cand_var = var
        ev = _floored("Q1", var, mu, row_sd=self._q1_thermal_sd)
        lw = cp.where(w > 0, cp.log(cp.maximum(w, EPS)), cp.full_like(w, -1e4))
        qw1 = self._qweights["Q1"]
        if qw1 != 1.0 and qw1 > 0.0:
            lw = lw + math.log(qw1)
        wl = [[_q1_wall(*r) for r in t] for t in tables]
        self._q1_cand_wall = (cp.asarray([[x[0] for x in r] for r in wl], dtype=cp.float32),
                              cp.asarray([[x[1] for x in r] for r in wl], dtype=cp.float32))
        return mu, ev, lw

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
        mu = cp.zeros((C, P, K), dtype=cp.float32)
        var = cp.ones((C, P, K), dtype=cp.float32)
        w = cp.zeros((C, P, K), dtype=cp.float32)
        nu = cp.zeros((C, P), dtype=cp.float32)
        for c, t in enumerate(tables):
            for i, (m, v, ww, nn) in enumerate(t):
                k = len(m)
                mu[c, i, :k] = cp.asarray(m, dtype=cp.float32)
                var[c, i, :k] = cp.asarray(v, dtype=cp.float32)
                w[c, i, :k] = cp.asarray(ww, dtype=cp.float32)
                nu[c, i] = nn
        self._q2_cand_var = var
        self._q2_cand_has_t = bool(cp.isfinite(nu).any())
        lw = cp.where(w > 0, cp.log(cp.maximum(w, EPS)), cp.full_like(w, -1e4))  # inert padding
        qw2 = self._qweights["Q2"]
        if qw2 != 1.0 and qw2 > 0.0:
            lw = lw + math.log(qw2)
        ev = _floored("Q2", var, mu)
        return mu, ev, lw, nu


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
            t = lambda a, dt=cp.float32: cp.asarray(np.asarray(a), dtype=dt)
            self._so.append(dict(k=k, dim=dim, donor=t([r[0] for r in rows], cp.int64),
                                 subs=t([r[1] for r in rows], cp.int64), mu=t(mu), prec=t(prec),
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
            t = lambda a, dt=cp.float32: cp.asarray(np.asarray(a), dtype=dt)
            self._so_hapt.append(dict(face=t(face, cp.int64), fmask=t(fmask), nxt=t(nxt, cp.int64), emask=t(emask),
                                      sa=t(sa, cp.int64), sx=t(sx, cp.int64), smask=t(smask), eta2=t(eta2),
                                      sel=t(list(sel), cp.int64), dim=dim, w=t(wrow),
                                      plane=bool(pon.any()), pa=t(pa, cp.int64), px=t(px, cp.int64), pm=t(pm), pon=t(pon),
                                      mu=t(mu), prec=t(prec), logdet=t(logdet), logw=t(logw)))
        self._so_n = sum(int(g["donor"].shape[0]) for g in self._so) + int(round(sum(r[4] for rows in by_sel.values() for r in rows)))

    def _sorient_pack(self):
        pk = getattr(self, "_so_triton_pack", None)
        if pk is not None:
            return pk
        so_k = []
        for g in self._so:
            so_k.append(dict(
                donor=cp.ascontiguousarray(g["donor"].astype(cp.int32)),
                subs=cp.ascontiguousarray(g["subs"].astype(cp.int32)),
                mu=cp.ascontiguousarray(g["mu"]),
                prec=cp.ascontiguousarray(g["prec"]),
                const=cp.ascontiguousarray(-0.5 * (g["logdet"] + g["dim"] * LOG_2PI) + g["logw"]),
                n_s=int(g["donor"].shape[0]),
                K_m=int(g["mu"].shape[1]),
                k=int(g["k"]),
                dim=int(g["dim"]),
            ))
        so_h = []
        for h in self._so_hapt:
            sel = h["sel"].tolist()
            dim = int(h["dim"])
            so_h.append(dict(
                face=cp.ascontiguousarray(h["face"].astype(cp.int32)),
                fmask=cp.ascontiguousarray(h["fmask"]),
                nxt=cp.ascontiguousarray(h["nxt"].astype(cp.int32)),
                emask=cp.ascontiguousarray(h["emask"]),
                sa=cp.ascontiguousarray(h["sa"].astype(cp.int32)),
                sx=cp.ascontiguousarray(h["sx"].astype(cp.int32)),
                smask=cp.ascontiguousarray(h["smask"]),
                eta2=cp.ascontiguousarray(h["eta2"]),
                pa=cp.ascontiguousarray(h["pa"].astype(cp.int32)),
                px=cp.ascontiguousarray(h["px"].astype(cp.int32)),
                pm=cp.ascontiguousarray(h["pm"]),
                pon=cp.ascontiguousarray(h["pon"]),
                mu=cp.ascontiguousarray(h["mu"]),
                prec=cp.ascontiguousarray(h["prec"]),
                const=cp.ascontiguousarray(-0.5 * (h["logdet"] + dim * LOG_2PI) + h["logw"]),
                w=cp.ascontiguousarray(h["w"]),
                n_r=int(h["face"].shape[0]),
                F_m=int(h["face"].shape[1]),
                S_m=int(h["sa"].shape[1]),
                Pm_m=int(h["pa"].shape[1]),
                K_m=int(h["mu"].shape[1]),
                dim=dim,
                sel0=int(sel[0]),
                sel1=int(sel[1]) if len(sel) > 1 else 0,
                sel2=int(sel[2]) if len(sel) > 2 else 0,
                has_plane=bool(h["plane"]),
            ))
        pk = (so_k, so_h)
        self._so_triton_pack = pk
        return pk

    def _sorient_triton(self, X: cp.ndarray, Z: Optional[float] = None, scale: float = 1.0,
                        out_val: Optional[cp.ndarray] = None, out_grad_X: Optional[cp.ndarray] = None):
        from jacobik.ik.kernels import _sorient_kappa_kernel, _sorient_haptic_kernel
        B, N, _ = X.shape
        Xc = cp.ascontiguousarray(X)
        if out_val is None:
            out_val = cp.zeros(B, dtype=X.dtype)
        if out_grad_X is None:
            out_grad_X = cp.zeros_like(Xc)
        so_k, so_h = self._sorient_pack()
        metal_idx = int(self._ge.metal_pos_idx)
        so_n = max(self._so_n, 1)
        BLOCK_B = 128
        grid_b = gpu.cdiv(B, BLOCK_B)
        for g in so_k:
            inv_norm = scale / (float(Z) if Z is not None else float(g["dim"] * so_n))
            gpu.launch(
                _sorient_kappa_kernel, (grid_b, g["n_s"]),
                Xc, out_grad_X, out_val,
                g["donor"], g["subs"], g["mu"], g["prec"], g["const"],
                B, N, g["n_s"], g["K_m"], g["k"], metal_idx, inv_norm, BLOCK_B,
            )
        for h in so_h:
            inv_norm = scale / (float(Z) if Z is not None else float(h["dim"] * so_n))
            gpu.launch(
                _sorient_haptic_kernel, (grid_b, h["n_r"]),
                Xc, out_grad_X, out_val,
                h["face"], h["fmask"], h["nxt"], h["emask"],
                h["sa"], h["sx"], h["smask"], h["eta2"],
                h["pa"], h["px"], h["pm"], h["pon"],
                h["mu"], h["prec"], h["const"], h["w"],
                B, N, h["n_r"], h["F_m"], h["S_m"], h["Pm_m"], h["K_m"], h["dim"],
                h["sel0"], h["sel1"], h["sel2"], h["has_plane"], metal_idx, inv_norm, BLOCK_B,
            )
        return out_val, out_grad_X

    def _sorient(self, X: cp.ndarray, Z: Optional[float] = None) -> cp.ndarray:
        if not isinstance(X, cp.ndarray):
            raise RuntimeError("S_orient evaluation requires a CuPy GPU array (GPU-only Triton build)")
        from jacobik.ik.kernels import _HAS_TRITON
        if not _HAS_TRITON:
            raise RuntimeError("Triton is required for CUDA S_orient evaluation")
        if not (self._so or self._so_hapt):
            return cp.zeros(X.shape[0], dtype=X.dtype)
        val, _ = self._sorient_triton(X, Z)
        return val


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
    device           : target GPU device (CuPy)
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
        device: Any = "cuda",
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
                t = cp.asarray([[pos[a], pos[p], pos[b]] for a, p, b in trip], dtype=cp.int64)
                X0 = cp.asarray(np.array([a.position for a in mol.atoms]), dtype=cp.float32)
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

    def ranking_score(self, X: cp.ndarray, row_groups: Optional[cp.ndarray] = None) -> cp.ndarray:
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
                keep = real & ~((cp.sqrt(self._q1_cand_var) > Q1_TAIL_SIGMA) | (mu < 1.2))
                keep = cp.where(keep.any(axis=-1, keepdims=True), keep, real)
                lw2 = cp.where(keep, lw, cp.full_like(lw, -1e30))
                self._q1_cand = (mu, ev, lw2 - _logsumexp(lw2, axis=-1, keepdims=True))
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

    def _q125_pack(self, device: Any = "cuda", row_groups: Optional[cp.ndarray] = None) -> dict:
        agg_sum = SCORE_AGG == "sum"
        Z = self._agg_norm() if agg_sum else None
        q1c = getattr(self, "_q1_cand", None)
        q2c = getattr(self, "_q2_cand", None)
        use_q1c = q1c is not None and (row_groups is not None or q1c[0].shape[0] == 1)
        use_q2c = q2c is not None and (row_groups is not None or q2c[0].shape[0] == 1)

        if use_q1c:
            q1_mu, q1_ev, q1_lw = q1c
            q1_whi, q1_wsd = self._q1_cand_wall
        else:
            q1_mu, q1_ev, q1_lw = self._q1_means[None], self._q1_eff_vars[None], self._q1_log_w[None]
            q1_whi, q1_wsd = self._q1_wall_hi[None], self._q1_wall_sd[None]

        if use_q2c:
            q2_mu, q2_ev, q2_lw, q2_nu = q2c
        else:
            q2_mu, q2_ev, q2_lw, q2_nu = (
                self._q2_means[None], self._q2_eff_vars[None], self._q2_log_w[None], self._q2_nu[None]
            )

        key = (
            device, agg_sum, Z, bool(Q1_WALL), use_q1c, use_q2c,
            id(q1c) if use_q1c else (id(self._q1_means), id(self._q1_eff_vars), id(self._q1_log_w)),
            id(q1c[1]) if use_q1c else 0,
            id(q1c[2]) if use_q1c else 0,
            id(q2c) if use_q2c else (id(self._q2_means), id(self._q2_eff_vars), id(self._q2_log_w), id(self._q2_nu)),
            id(q2c[1]) if use_q2c else 0,
            id(self._q5_means), id(self._q5_kappas), id(self._q5_log_w),
            self._qweights["Q1"], self._qweights["Q2"], self._qweights["Q5"],
        )
        cached = getattr(self, "_q125_triton_pack", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        ge = self._ge
        n_d = ge.n_donors
        n_p = self._n_q2
        n_q5 = ge.n_q5
        inv_div_q1 = (1.0 / (Z if agg_sum else max(self._n_q1, 1))) if (self._n_q1 > 0 and self._q1_means.shape[0] > 0 and self._qweights["Q1"] > 0.0) else 0.0
        inv_div_q2 = (1.0 / (Z if agg_sum else max(n_p, 1))) if (n_p > 0 and self._q2_means.shape[0] > 0 and self._qweights["Q2"] > 0.0) else 0.0
        inv_div_q5 = (1.0 / (Z if agg_sum else max(n_q5, 1))) if (n_q5 > 0 and self._q5_means.shape[0] > 0 and self._qweights["Q5"] > 0.0) else 0.0

        pair_a = cp.asarray([i for i in range(n_d) for j in range(i + 1, n_d)], dtype=cp.int32)
        pair_b = cp.asarray([j for i in range(n_d) for j in range(i + 1, n_d)], dtype=cp.int32)

        if n_p > 0 and self._q2_means.shape[0] > 0:
            nu_b = q2_nu[..., None]
            nf = cp.where(cp.isfinite(nu_b), nu_b, cp.ones_like(nu_b))
            c_g = -0.5 * _LOG_2PI - 0.5 * cp.log(q2_ev) + q2_lw
            c_t = cspec.gammaln((nf + 1) / 2) - cspec.gammaln(nf / 2) - 0.5 * cp.log(nf * math.pi * q2_ev) + q2_lw
            q2_const = cp.ascontiguousarray(cp.where(cp.isfinite(nu_b), c_t, c_g))
        else:
            q2_const = cp.ascontiguousarray(q2_mu)

        q5_const = cp.ascontiguousarray(self._q5_log_w - self._q5_log_norm) if n_q5 > 0 else self._q5_means
        pk = dict(
            q1_mu=cp.ascontiguousarray(q1_mu), q1_ev=cp.ascontiguousarray(q1_ev), q1_lw=cp.ascontiguousarray(q1_lw),
            q1_whi=cp.ascontiguousarray(q1_whi), q1_wsd=cp.ascontiguousarray(q1_wsd),
            C1=int(q1_mu.shape[0]), K1=int(q1_mu.shape[-1]), q1_wall_on=bool(Q1_WALL), inv_div_q1=float(inv_div_q1),
            pair_a=pair_a, pair_b=pair_b,
            q2_mu=cp.ascontiguousarray(q2_mu), q2_ev=cp.ascontiguousarray(q2_ev), q2_lw=cp.ascontiguousarray(q2_lw),
            q2_nu=cp.ascontiguousarray(q2_nu), q2_const=q2_const,
            C2=int(q2_mu.shape[0]), n_p=int(n_p), K2=int(q2_mu.shape[-1]), inv_div_q2=float(inv_div_q2),
            q5_i=cp.ascontiguousarray(cp.asarray(ge._q5_i, dtype=cp.int32)),
            q5_j=cp.ascontiguousarray(cp.asarray(ge._q5_j, dtype=cp.int32)),
            q5_k=cp.ascontiguousarray(cp.asarray(ge._q5_k, dtype=cp.int32)),
            q5_l=cp.ascontiguousarray(cp.asarray(ge._q5_l, dtype=cp.int32)),
            q5_mu=cp.ascontiguousarray(self._q5_means), q5_kap=cp.ascontiguousarray(self._q5_kappas), q5_const=q5_const,
            n_q5=int(n_q5), K5=int(self._q5_means.shape[-1]) if n_q5 > 0 else 0, inv_div_q5=float(inv_div_q5),
            n_d=int(n_d), metal_idx=int(ge.metal_pos_idx),
        )
        self._q125_triton_pack = (key, pk)
        return pk

    def _donor_scatter_pack(self, device: Any = "cuda"):
        pk = getattr(self, "_donor_scatter_cache", None)
        if pk is not None and pk[0] == str(device):
            return pk[1]
        ge = self._ge
        all_kappa = all(not h for h in ge._donor_is_haptic)
        if all_kappa:
            kappa_pos_t = cp.asarray(ge._kappa_pos_indices, dtype=cp.int64)
            res = (True, kappa_pos_t, None, None)
        else:
            d_idxs, a_idxs, ws = [], [], []
            kp = hp = 0
            for s_i in range(ge.n_donors):
                if not ge._donor_is_haptic[s_i]:
                    d_idxs.append(s_i)
                    a_idxs.append(ge._kappa_pos_indices[kp])
                    ws.append(1.0)
                    kp += 1
                else:
                    face = ge._haptic_pos_lists[hp]
                    w = 1.0 / len(face)
                    for a in face:
                        d_idxs.append(s_i)
                        a_idxs.append(a)
                        ws.append(w)
                    hp += 1
            res = (
                False,
                cp.asarray(a_idxs, dtype=cp.int64),
                cp.asarray(d_idxs, dtype=cp.int64),
                cp.asarray(ws, dtype=cp.float32),
            )
        self._donor_scatter_cache = (str(device), res)
        return res

    def _score_components(
        self,
        X: cp.ndarray,
        q3_mode_override: Optional[cp.ndarray] = None,
        donor_pos: Optional[cp.ndarray] = None,
        row_groups: Optional[cp.ndarray] = None,
    ) -> Dict[str, cp.ndarray]:
        """Compute each quantity's normalised weighted log-prob as a (B,) array."""
        _, _, comps = self.score_and_grad(
            X, q3_mode_override=q3_mode_override, donor_pos=donor_pos,
            row_groups=row_groups, return_comps=True,
        )
        assert comps is not None
        return comps

    def score_and_grad(
        self,
        X: cp.ndarray,
        q3_mode_override: Optional[cp.ndarray] = None,
        donor_pos: Optional[cp.ndarray] = None,
        row_groups: Optional[cp.ndarray] = None,
        return_comps: bool = False,
    ) -> Tuple[cp.ndarray, cp.ndarray, Optional[Dict[str, cp.ndarray]]]:
        """
        Compute unified statistical score ``(B,)`` and its coordinate gradient
        ``d(score)/dX`` ``(B, N, 3)`` in a single fused pass without autograd.
        """
        from jacobik.ik.kernels import _HAS_TRITON, _q1_q2_q5_fused_kernel

        if not isinstance(X, cp.ndarray):
            raise RuntimeError("Scorer.score_and_grad requires a CuPy GPU array (GPU-only Triton build)")
        if not _HAS_TRITON:
            raise RuntimeError("Triton is required for CUDA Scorer.score_and_grad")

        B, N, _ = X.shape
        Xc = cp.ascontiguousarray(X)
        dp = cp.ascontiguousarray(donor_pos if donor_pos is not None else self._ge.donor_positions(Xc))
        grad_X = cp.zeros_like(Xc)
        grad_donor = cp.zeros((B, self._ge.n_donors, 3), dtype=X.dtype)
        out_q1 = cp.zeros(B, dtype=X.dtype)
        out_q2 = cp.zeros(B, dtype=X.dtype)
        out_q5 = cp.zeros(B, dtype=X.dtype)
        out_so = cp.zeros(B, dtype=X.dtype)
        out_q6 = cp.zeros(B, dtype=X.dtype)

        pk = self._q125_pack(self.device, row_groups)
        has_rg = row_groups is not None and (pk["C1"] > 1 or pk["C2"] > 1)
        rg_t = cp.ascontiguousarray(cp.asarray(row_groups, dtype=cp.int32)) if has_rg else Xc
        BLOCK_B = 128
        grid_125 = (gpu.cdiv(B, BLOCK_B), 1 + (pk["n_q5"] if pk["inv_div_q5"] > 0.0 else 0))
        gpu.launch(
            _q1_q2_q5_fused_kernel, grid_125,
            Xc, dp, grad_X, grad_donor, Xc, dp, Xc, dp, rg_t, has_rg, False,
            pk["q1_mu"], pk["q1_ev"], pk["q1_lw"], pk["q1_whi"], pk["q1_wsd"],
            pk["C1"], pk["K1"], pk["q1_wall_on"], pk["inv_div_q1"],
            pk["pair_a"], pk["pair_b"], pk["q2_mu"], pk["q2_ev"], pk["q2_lw"], pk["q2_nu"], pk["q2_const"],
            pk["C2"], pk["n_p"], pk["K2"], pk["inv_div_q2"],
            pk["q5_i"], pk["q5_j"], pk["q5_k"], pk["q5_l"],
            pk["q5_mu"], pk["q5_kap"], pk["q5_const"], pk["n_q5"], pk["K5"], pk["inv_div_q5"],
            out_q1, out_q2, out_q5, B, N, pk["n_d"], pk["metal_idx"], BLOCK_B,
        )

        if pk["inv_div_q1"] > 0.0 or pk["inv_div_q2"] > 0.0:
            all_kappa, idx_a, idx_d, w_sc = self._donor_scatter_pack(self.device)
            if all_kappa:
                cp.add.at(grad_X, (slice(None), idx_a, slice(None)), grad_donor)
            else:
                cp.add.at(grad_X, (slice(None), idx_a, slice(None)), grad_donor[:, idx_d, :] * w_sc[None, :, None])

        agg_sum = SCORE_AGG == "sum"
        Z = self._agg_norm() if agg_sum else None
        out_q3 = cp.zeros(B, dtype=X.dtype)
        out_q4 = cp.zeros(B, dtype=X.dtype)
        if self._so or self._so_hapt:
            self._sorient_triton(Xc, Z, scale=float(self._qweights["Q3"]), out_val=out_q3, out_grad_X=grad_X)
        elif self._so_n == 0:
            def _gmm_val_grad(obs, means, eff_vars, log_w, n_terms):
                div = Z if agg_sum else max(n_terms, 1)
                diff = obs[..., None] - means[None]
                log_comp = -0.5 * _LOG_2PI - 0.5 * cp.log(eff_vars)[None] - 0.5 * diff ** 2 / eff_vars[None]
                lw_b = log_w[None] if log_w.ndim == 2 else log_w
                log_p = log_comp + lw_b
                m = cp.max(log_p, axis=-1, keepdims=True)
                e = cp.exp(log_p - m)
                s = cp.sum(e, axis=-1, keepdims=True)
                val = (m.squeeze(-1) + cp.log(s.squeeze(-1))).sum(axis=-1) / div
                g_obs = ((e / s) * (-diff / eff_vars[None])).sum(axis=-1) / div
                return val, g_obs

            if self._n_q3 > 0 and self._q3_means.shape[0] > 0:
                q3_a = self._ge.q3_approach_angles(Xc)
                q3_cos = cp.cos(q3_a)
                q3_lw = self._q3_log_w
                if q3_mode_override is not None:
                    Nq, Kq = self._q3_log_w.shape
                    mask = q3_mode_override >= 0
                    ov = cp.full((B, Nq, Kq), _Q3_OVERRIDE_NEG, dtype=q3_cos.dtype)
                    ov[cp.arange(B, dtype=cp.int64)[:, None], cp.arange(Nq, dtype=cp.int64)[None, :], cp.maximum(q3_mode_override, 0)] = 0.0
                    q3_lw = cp.where(mask[..., None], ov, cp.broadcast_to(self._q3_log_w[None], (B, Nq, Kq)))
                out_q3, g_q3_cos = _gmm_val_grad(q3_cos, self._q3_means, self._q3_eff_vars, q3_lw, self._n_q3)
                grad_X = grad_X + self._ge.q3_approach_angles_vjp(Xc, g_q3_cos * (-cp.sin(q3_a)))

            if self._ge.n_q4 > 0 and self._q4_means.shape[0] > 0:
                s_pos = Xc[:, self._ge._q4_s_pos]
                donor = Xc[:, self._ge._q4_donor_pos]
                vecs = s_pos - donor[:, :, None, :]
                norms = cp.maximum(cp.linalg.norm(vecs, axis=-1, keepdims=True), 1e-8)
                unit = vecs / norms
                n_valid = self._ge._q4_s_mask.astype(Xc.dtype).sum(axis=-1)
                n_pairs_ = cp.maximum(n_valid * (n_valid - 1.0) / 2.0, 1.0)
                sigma_theta = cp.zeros((B, self._ge.n_q4), dtype=Xc.dtype)
                pair_cache = []
                for a_i, b_i, pair_valid_f in self._ge._q4_pair_valid:
                    ua, ub = unit[:, :, a_i, :], unit[:, :, b_i, :]
                    cos_t = cp.clip(cp.sum(ua * ub, axis=-1), -1.0 + 1e-7, 1.0 - 1e-7)
                    sigma_theta = sigma_theta + (cp.arccos(cos_t) * (180.0 / math.pi)) * pair_valid_f
                    pair_cache.append((a_i, b_i, pair_valid_f, ua, ub, cos_t))
                q4_v = sigma_theta / n_pairs_[None, :]
                out_q4, g_q4 = _gmm_val_grad(q4_v, self._q4_means, self._q4_eff_vars, self._q4_log_w, self._ge.n_q4)
                g_s = cp.zeros_like(s_pos)
                g_d = cp.zeros_like(donor)
                for a_i, b_i, pair_valid_f, ua, ub, cos_t in pair_cache:
                    sin_t = cp.sqrt(cp.maximum(1.0 - cos_t * cos_t, 1e-14))
                    g_cos = g_q4 * (pair_valid_f / n_pairs_[None, :]) * (-(180.0 / math.pi) / sin_t)
                    g_ua = g_cos[..., None] * (ub - cos_t[..., None] * ua) / norms[:, :, a_i, :]
                    g_ub = g_cos[..., None] * (ua - cos_t[..., None] * ub) / norms[:, :, b_i, :]
                    g_s[:, :, a_i, :] += g_ua
                    g_s[:, :, b_i, :] += g_ub
                    g_d -= g_ua + g_ub
                cp.add.at(grad_X, (slice(None), self._ge._q4_s_pos, slice(None)), g_s)
                cp.add.at(grad_X, (slice(None), self._ge._q4_donor_pos, slice(None)), g_d)
        out_so = out_q3

        if self._q6_pmf is not None and self._q6_qweight > 0.0:
            from jacobik.stats.steric import evaluate_and_grad as _q6_eval_grad
            _q6_eval_grad(
                self._q6_pmf, Xc, self._ge._q6_a, self._ge._q6_b,
                scale=-self._q6_qweight / self._agg_norm(),
                out_val=out_q6, out_grad_X=grad_X,
            )
        elif self._ge.n_q6 > 0 and self._q6_mu.shape[0] > 0 and self._q6_qweight > 0.0:
            a6, b6 = self._ge._q6_a, self._ge._q6_b
            diff6 = Xc[:, a6] - Xc[:, b6]  # (B, M, 3)
            d6 = cp.sqrt((diff6 * diff6).sum(axis=-1))
            t6 = cp.maximum(self._q6_mu - d6, 0.0)
            var6 = cp.maximum(self._q6_sigma, _EPS) ** 2
            if agg_sum:
                scale6 = self._q6_qweight / Z
                out_q6 = -0.5 * scale6 * ((t6 * t6) / var6).sum(axis=-1)
                gc6 = scale6 * t6 / (var6 * cp.maximum(d6, 1e-12))
            else:
                tau = _Q6_LSE_TEMP
                var2 = 2.0 * var6
                x6 = (t6 * t6) * (tau / var2)
                m6 = cp.max(x6, axis=-1, keepdims=True)
                e6 = cp.exp(cp.maximum(x6 - m6, -80.0))
                s6 = cp.sum(e6, axis=-1, keepdims=True)
                out_q6 = self._q6_qweight * (-(m6.squeeze(-1) + cp.log(s6.squeeze(-1))) / tau + self._q6_lse_log_n_over_tau)
                gc6 = self._q6_qweight * (e6 / s6) * (2.0 * t6 / var2) / cp.maximum(d6, 1e-12)
            gp6 = diff6 * gc6[..., None]
            cp.add.at(grad_X, (slice(None), a6, slice(None)), gp6)
            cp.add.at(grad_X, (slice(None), b6, slice(None)), -gp6)

        out_q7 = cp.zeros(B, dtype=X.dtype)
        if self._q7 is not None:
            t7, a7, (free7, sig7, w7) = self._q7
            va = Xc[:, t7[:, 0]] - Xc[:, t7[:, 1]]
            vb = Xc[:, t7[:, 2]] - Xc[:, t7[:, 1]]
            na = cp.maximum(cp.linalg.norm(va, axis=-1, keepdims=True), 1e-8)
            nb = cp.maximum(cp.linalg.norm(vb, axis=-1, keepdims=True), 1e-8)
            u, v = va / na, vb / nb
            cos_a = cp.clip(cp.sum(u * v, axis=-1), -1.0 + 1e-6, 1.0 - 1e-6)
            diff_a = cp.arccos(cos_a) - a7[None]
            free_rad = math.radians(float(free7))
            sig_rad = math.radians(float(sig7))
            ex = cp.maximum(cp.abs(diff_a) - free_rad, 0.0) / sig_rad
            out_q7 = -w7 * 0.5 * (ex * ex).mean(axis=-1)
            n_t = max(int(t7.shape[0]), 1)
            g_ang = (-w7 / (n_t * sig_rad)) * ex * cp.sign(diff_a)
            sin_a = cp.sqrt(cp.maximum(1.0 - cos_a * cos_a, 1e-14))
            g_cos = g_ang * (-1.0 / sin_a)
            g_u = g_cos[..., None] * (v - cos_a[..., None] * u) / na
            g_v = g_cos[..., None] * (u - cos_a[..., None] * v) / nb
            cp.add.at(grad_X, (slice(None), t7[:, 0], slice(None)), g_u)
            cp.add.at(grad_X, (slice(None), t7[:, 2], slice(None)), g_v)
            cp.add.at(grad_X, (slice(None), t7[:, 1], slice(None)), -(g_u + g_v))

        scalar = out_q1 + out_q2 + out_q5 + out_so + out_q4 + out_q6 + out_q7
        comps = None
        if return_comps:
            comps = {
                "Q1": out_q1,
                "Q2": out_q2,
                "Q3": out_so,
                "Q4": out_q4,
                "Q5": out_q5,
                "Q6": out_q6,
            }
            if self._q7 is not None:
                comps["Q7"] = out_q7
        return scalar, grad_X, comps

    def score(
        self,
        X: cp.ndarray,
        q3_mode_override: Optional[cp.ndarray] = None,
        donor_pos: Optional[cp.ndarray] = None,
        row_groups: Optional[cp.ndarray] = None,
    ) -> Tuple[cp.ndarray, Dict[str, cp.ndarray]]:
        """
        Unified statistical score: the sum of the per-quantity weighted values.
        """
        breakdown = self._score_components(
            X, q3_mode_override=q3_mode_override, donor_pos=donor_pos, row_groups=row_groups
        )
        scalar = cp.zeros(X.shape[0], dtype=X.dtype)
        for v in breakdown.values():
            scalar = scalar + v
        return scalar, breakdown  # (B,), dict[str, (B,)]
