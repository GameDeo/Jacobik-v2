"""
fitter.py
=========
Step 5 -- Distribution fitter.

What is fitted (2026-10-01 refit spec, claude/refit-spec-additions-2026-09-30.md):

* Evidence weights everywhere: within a key, each complex has total weight 1 (1/m for its m rows); BIC uses the
  effective count, and a mixture component must carry >= N_COMP_MIN effective complexes (§2; replaces the old
  ``n // 30`` component cap).
* Q1 and Q2: partial pooling down the fallback ladder (§3, ``PooledFitter``): keys with >= N_INDEP complexes are
  fitted on their own data; every other key is MAP-adapted (relevance factor r) from its ladder parent, so a sparse
  key stays near its parent and a key with data follows its data. Entries are marked ``pooled`` and the lookup takes
  the most specific one. Q1 = Gaussian mixture; Q2 = Student-t mixture in cos space ("tmix_cos", §4) keyed by
  site-pair class (§5).
* Q3 / Q4: evidence-weighted GMMs (no pooling; superseded by S_orient at solve time).
* Q5: the torsion well model (claude/q5-investigation-2026-09-28.md §6): weighted VMM, one component per well
  (``merge_vmm_to_wells``), the minority off-well tail dropped; rows are symmetry-folded at extraction.
* Q6: contact-onset wall per pair key (``fit_contact_wall``): mu = the 0.1 % quantile of the non-bonded distances,
  sigma from the 1 % - 0.1 % quantile spread; polar H facing an acceptor has its own "Hp" keys.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import i0e as _i0e

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MIN: int = 30
K_MAX_GMM: int = 4
K_MAX_VMM: int = 6
VMM_MAX_ITER: int = 200
VMM_TOL: float = 1e-5

_DIST_TYPE: Dict[str, str] = {
    "Q1": "gmm",
    "Q2": "gmm_cos",  # D-M-D angle, bounded [0,180] deg, NOT circular -- see
    "Q3": "gmm_cos",  # fit_gmm_cosangle() docstring below for the rationale.
    "Q4": "gmm",
    "Q5": "vmm",  # torsion dihedral -- genuinely circular (0 == 360), VMM is correct here.
    "Q6": "halfgauss",
}

_K_MAX: Dict[str, int] = {
    "Q1": K_MAX_GMM,
    "Q2": K_MAX_VMM,  # component budget unrelated to model family; kept at 6
    "Q3": K_MAX_VMM,  # (Q2/Q3 legitimately need more components than Q1/Q4 --
    "Q4": K_MAX_GMM,  # e.g. Q2 for a homoleptic monodentate complex pools BOTH
    "Q5": K_MAX_VMM,  # cis-like and trans-like angles under one key -- see
    # ik/scorer.py Q2 section for how that pooling works at score time).
}

#: Fit hygiene (B2.3). A mixture component may not rest on fewer than N_COMP_MIN effective samples (weight * n): a K
#: whose fit has such a component is not selectable by BIC -- this is what removes single-sample spikes (a maximum-
#: likelihood GMM rewards a component collapsed onto one outlier). The variance floor is only a numerical minimum,
#: sized to the precision of the (xTB-optimised) geometries: 0.002 A, 0.2 deg, 0.002 in cosine space (A^2 / deg^2 /
#: cos^2 below); it is a clamp, not added (tight real modes, e.g. W-CO sd 0.006 A, keep their width).
#: Without these, ~1.5k Q1 components sat at sklearn's 1e-6 floor with weight ~1/n (single-sample spikes, e.g.
#: Q1 (Ru, C_alkyl, square_pyr) at 3.14 A), which the scorer then rewarded as a sharp, valid mode.
VAR_FLOOR: Dict[str, float] = {"Q1": 0.002**2, "Q4": 0.2**2, "Q2": 0.002**2, "Q3": 0.002**2}
N_COMP_MIN: int = 10

# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------


def fit_gmm(values: np.ndarray, k_max: int = K_MAX_GMM, var_floor: float = 1e-6,
            n_comp_min: int = N_COMP_MIN, weights: Optional[np.ndarray] = None) -> Optional[Dict]:
    """
    Evidence-weighted GMM with BIC-selected K (stats/mixture.fit_mixture). ``weights``: per-row evidence weights
    (default 1 each: every row an independent sample). A K whose fit has a component with fewer than ``n_comp_min``
    effective samples is not selectable; the variance floor is a clamp (a numerical minimum), not an addition.
    None if fewer than N_MIN rows or fewer than ``n_comp_min`` effective samples.
    """
    from jacobik.stats import mixture as MX

    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < N_MIN:
        return None
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    f = MX.fit_mixture(x, w, "gauss", k_max=k_max, var_floor=var_floor, n_comp_min=max(n_comp_min, 1))
    if f is None:
        return None
    return MX.to_entry(f, "gmm", n, f["n_eff"])


def fit_gmm_cosangle(values_deg: np.ndarray, k_max: int = K_MAX_GMM, var_floor: float = 1e-6,
                     n_comp_min: int = N_COMP_MIN, weights: Optional[np.ndarray] = None) -> Optional[Dict]:
    """
    GMM on cos(angle) for a bounded, non-circular angle (Q3 donor approach angle, legacy Q2), [0, 180] deg.
    Means / vars / weights are stored in cos space; "type": "gmm_cos" tells the scorer to evaluate them against
    cos(angle). Input in degrees.
    """
    result = fit_gmm(np.cos(np.deg2rad(values_deg)), k_max, var_floor=var_floor, n_comp_min=n_comp_min,
                     weights=weights)
    if result is not None:
        result["type"] = "gmm_cos"
    return result


# ---------------------------------------------------------------------------
# VMM (von Mises Mixture, EM)
# ---------------------------------------------------------------------------


def _kappa_from_rbar(R_bar: np.ndarray) -> np.ndarray:
    """
    Closed-form kappa estimate from mean resultant length R_bar in [0,1).
    Mardia & Jupp (2000) approximation, vectorised over an array of R_bar.
    """
    R = np.clip(R_bar, 0.0, 1.0 - 1e-6)
    return np.where(
        R < 0.53,
        R * (2.0 - R**2) / (1.0 - R**2),
        np.where(
            R < 0.85,
            -0.4 + 1.39 * R + 0.43 / (1.0 - R),
            1.0 / (R**3 - 4.0 * R**2 + 3.0 * R + 1e-8),
        ),
    )


# ---------------------------------------------------------------------------
# Evidence-weighted VMM, one component per well (Q5; 2026-09-28, adopted for the 2026-10 refit)
# ---------------------------------------------------------------------------

#: kappa ceiling from measurement precision only (xTB-optimised torsions are reproducible to ~0.2 deg), not a
#: modelling cap: 1 / (0.2 deg in rad)^2.
KAPPA_PRECISION_MAX: float = 1.0 / np.deg2rad(0.2) ** 2


def fit_vmm_weighted(values_deg: np.ndarray, w: np.ndarray, k_max: int = K_MAX_VMM,
                     n_comp_min: float = N_COMP_MIN, n_init: int = 3) -> Optional[Dict]:
    """VMM fitted to INDEPENDENT evidence. ``w`` are per-observation weights that sum to the number of independent
    units (here: each complex contributes total weight 1 per key, so m copies of a torsion in one complex -- the
    6-12 phenyl torsions of a PPh3-rich complex, the methyls of a tBu -- count once). Weighted EM; BIC with the
    effective count; a K whose fit has a component resting on < n_comp_min effective units is not selectable (the
    B2.3 GMM rule, applied to VMMs). Means in degrees; output also carries ``n_eff``."""
    x = np.deg2rad(np.asarray(values_deg, float)); x = (x + np.pi) % (2 * np.pi) - np.pi
    w = np.asarray(w, float); n_eff = float(w.sum()); n = len(x)
    if n < N_MIN:
        return None
    sinx, cosx = np.sin(x), np.cos(x)
    best = (np.inf, None, 1)
    k_upper = int(min(k_max, max(1, n_eff // max(n_comp_min, 1))))
    for k in range(1, k_upper + 1):
        best_k = (-np.inf, None)
        for _ in range(n_init):
            idx = np.random.choice(n, k, replace=(n < k), p=w / n_eff)
            means = x[idx] + np.random.uniform(-0.3, 0.3, k); kap = np.ones(k); wt = np.full(k, 1.0 / k); prev = -np.inf
            for _it in range(VMM_MAX_ITER):
                lnorm = np.log(2.0 * np.pi) + np.log(np.clip(_i0e(kap), 1e-300, None)) + kap
                lp = kap[None] * (cosx[:, None] * np.cos(means)[None] + sinx[:, None] * np.sin(means)[None]) - lnorm[None]
                lr = lp + np.log(wt + 1e-300)[None]
                mx = lr.max(1, keepdims=True); lse = mx.ravel() + np.log(np.exp(lr - mx).sum(1))
                ll = float((w * lse).sum())
                r = np.exp(lr - lse[:, None]) * w[:, None]
                rs = r.sum(0) + 1e-300; wt = rs / rs.sum()
                ss, cs = (r * sinx[:, None]).sum(0), (r * cosx[:, None]).sum(0)
                means = np.arctan2(ss, cs)
                kap = np.clip(_kappa_from_rbar(np.clip(np.sqrt(ss**2 + cs**2) / rs, 0.0, 1.0 - 1e-12)), 0.01, KAPPA_PRECISION_MAX)
                if abs(ll - prev) < VMM_TOL * max(1.0, n_eff / max(n, 1)):
                    break
                prev = ll
            if ll > best_k[0]:
                best_k = (ll, (means.copy(), kap.copy(), wt.copy()))
        if best_k[1] is None:
            continue
        if k > 1 and float((best_k[1][2] * n_eff).min()) < n_comp_min:
            continue
        bic = -2.0 * best_k[0] + (3 * k - 1) * math.log(max(n_eff, 2.0))
        if bic < best[0]:
            best = (bic, best_k[1], k)
    if best[1] is None:
        return None
    means, kap, wt = best[1]; o = np.argsort(means)
    return {"type": "vmm", "n": n, "n_eff": n_eff, "k": best[2], "means": np.rad2deg(means[o]).tolist(),
            "kappas": kap[o].tolist(), "weights": wt[o].tolist()}


def _A(k):
    """Mean resultant length of a von Mises with concentration k: I1(k)/I0(k)."""
    from scipy.special import i1e
    k = np.asarray(k, float)
    return i1e(k) / np.clip(_i0e(k), 1e-300, None)


KAPPA_WELL_MIN: float = 4.0


def merge_vmm_to_wells(fit: Optional[Dict], grid: int = 3600) -> Optional[Dict]:
    """One component per WELL (mode of the density). Components whose pairwise mixture has no dip between their
    means (they describe one well: a spike inside a shoulder, two overlapping halves) are merged into one von
    Mises with the SAME weight, circular mean and mean resultant length as the merged mass (moment matching: the
    well's spread is exactly the data's spread; the von Mises is the max-entropy density with that mean and
    resultant). Separate wells (a real dip between them, e.g. staggered +-60/180) are kept apart."""
    if fit is None or fit.get("k", 1) <= 1:
        return fit
    # only LOCALISED components (circular sd < ~30 deg, kappa >= 4) describe a well; broad components are the
    # off-well tail / a free rotor and are never merged into a well (that would smear the well's width)
    mu = list(np.deg2rad(fit["means"])); ka = list(fit["kappas"]); wt = list(fit["weights"])
    def dens(t, m, k, w):
        return w * np.exp(k * (np.cos(t - m) - 1.0)) / (2 * np.pi * np.clip(_i0e(k), 1e-300, None))
    changed = True
    while changed and len(mu) > 1:
        changed = False
        o = np.argsort(mu); mu = [mu[i] for i in o]; ka = [ka[i] for i in o]; wt = [wt[i] for i in o]
        n = len(mu)
        for a in range(n if n > 2 else 1):
            b = (a + 1) % n
            if ka[a] < KAPPA_WELL_MIN or ka[b] < KAPPA_WELL_MIN:
                continue
            d = (mu[b] - mu[a]) % (2 * np.pi)
            t = mu[a] + np.linspace(0.0, d, 400)
            f = dens(t, mu[a], ka[a], wt[a]) + dens(t, mu[b], ka[b], wt[b])
            # a dip = an interior local minimum below both ends' mixture density (the two form separate modes)
            interior = f[1:-1]
            dip = interior.min() < min(f[0], f[-1]) * (1 - 1e-6) and np.argmin(interior) not in (0, len(interior) - 1)
            if not dip:
                W = wt[a] + wt[b]
                C = wt[a] * _A(ka[a]) * np.exp(1j * mu[a]) + wt[b] * _A(ka[b]) * np.exp(1j * mu[b])
                R = min(abs(C) / W, 1 - 1e-12)
                m_new, k_new = float(np.angle(C)), float(np.clip(_kappa_from_rbar(np.array([R]))[0], 0.01, KAPPA_PRECISION_MAX))
                for i in sorted((a, b), reverse=True):
                    del mu[i]; del ka[i]; del wt[i]
                mu.append(m_new); ka.append(k_new); wt.append(W); changed = True
                break
    o = np.argsort(mu)
    out = dict(fit); out.update(k=len(mu), means=np.rad2deg(np.array(mu)[o]).tolist(), kappas=np.array(ka)[o].tolist(),
                                weights=np.array(wt)[o].tolist(), merged_from=fit["k"])
    return out


#: Q5 well model: an off-well (broad, kappa < KAPPA_WELL_MIN) tail is dropped from the scored density when it is a
#: minority (< Q5_TAIL_MAX of the weight) and real wells exist; a predominantly free rotor keeps its broad density.
Q5_TAIL_MAX: float = 0.5


def fit_vmm_wells(values_deg: np.ndarray, weights: Optional[np.ndarray] = None, k_max: int = K_MAX_VMM) -> Optional[Dict]:
    """The adopted Q5 model (09-28 "WS"): evidence-weighted VMM -> one component per well -> minority tail dropped
    (its share recorded as ``tail``)."""
    x = np.asarray(values_deg, dtype=float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    f = merge_vmm_to_wells(fit_vmm_weighted(x, w, k_max=k_max))
    if f is None:
        return None
    k = np.asarray(f["kappas"], float)
    wt = np.asarray(f["weights"], float)
    well = k >= KAPPA_WELL_MIN
    f["tail"] = 0.0
    if well.any() and (~well).any() and wt[~well].sum() < Q5_TAIL_MAX:
        f["tail"] = float(wt[~well].sum())
        f["means"] = np.asarray(f["means"], float)[well].tolist()
        f["kappas"] = k[well].tolist()
        f["weights"] = (wt[well] / wt[well].sum()).tolist()
        f["k"] = int(well.sum())
    f["model"] = "wells"
    return f


# ---------------------------------------------------------------------------
# Q6 contact wall (half-Gaussian)
# ---------------------------------------------------------------------------


#: Q6 contact-onset wall (2026-10-01): mu at this quantile of a pair key's non-bonded distances ...
Q6_ONSET_Q: float = 0.001
#: ... and sigma from the spread between it and this quantile, as a Gaussian left flank would have it:
#: sigma = (q(Q6_SPREAD_Q) - q(Q6_ONSET_Q)) / (z(Q6_SPREAD_Q) - z(Q6_ONSET_Q))
Q6_SPREAD_Q: float = 0.01
#: numerical minimum of the wall width (A)
Q6_SIGMA_MIN: float = 0.05
#: rows needed to resolve the Q6_ONSET_Q quantile (below 1 / Q6_ONSET_Q rows it is just the sample minimum, and the
#: 1 % - 0.1 % spread collapses to the floor: a hard wall at the smallest of a few rows; TACZUT Ti...F 2.60 A cost
#: 85 nats against such a wall from 134 rows). Sparser keys are not fitted: the lookup falls back to the pooled
#: (period group, X) wall or the covalent-radius fallback.
Q6_MIN_ROWS: int = int(round(1.0 / Q6_ONSET_Q))


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw = (cw - 0.5 * w[o]) / cw[-1]
    return float(np.interp(q, cw, x[o]))


def fit_contact_wall(values: np.ndarray, weights: Optional[np.ndarray] = None) -> Optional[Dict]:
    """Q6 one-sided wall -(mu - d)^2 / (2 sigma^2) for d < mu, placed at the ONSET of real contacts: mu = the
    Q6_ONSET_Q (0.1 %) evidence-weighted quantile of the pair key's non-bonded distances; sigma from the Q6_SPREAD_Q
    (1 %) - 0.1 % spread. The old wall (mu = 5th percentile) put ~5 % of real contacts inside the wall by
    construction: crystals lost median 7.7 nats (p10 26) to it, and H-bonds up to 46."""
    from scipy.stats import norm

    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < max(N_MIN, Q6_MIN_ROWS):
        return None
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    mu = _weighted_quantile(x, w, Q6_ONSET_Q)
    hi = _weighted_quantile(x, w, Q6_SPREAD_Q)
    sigma = max((hi - mu) / float(norm.ppf(Q6_SPREAD_Q) - norm.ppf(Q6_ONSET_Q)), Q6_SIGMA_MIN)
    return {"type": "halfgauss", "n": n, "n_eff": round(float(w.sum()), 3), "k": 1, "mu": mu, "sigma": sigma,
            "model": "onset"}


def fit_halfgauss(values: np.ndarray) -> Optional[Dict]:
    """Legacy Q6 wall (before 2026-10-01): mu = 5th percentile, sigma from the mirrored left tail (floor 0.1 A).
    Kept for reproducing old DBs; the refit uses fit_contact_wall."""
    n = len(values)
    if n < N_MIN:
        return None
    mu = float(np.percentile(values, 5))
    left = values[values <= mu]
    if len(left) < 3:
        sigma = float((np.percentile(values, 90) - np.percentile(values, 10)) / 4.0)
    else:
        sigma = float(np.std(np.concatenate([left, 2.0 * mu - left])))
    sigma = max(sigma, 0.1)
    return {"type": "halfgauss", "n": n, "k": 1, "mu": mu, "sigma": sigma}


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _fit_chunk(
    chunk: List[Tuple[str, str, np.ndarray, np.ndarray]],
    dist_type: Dict[str, str],
    k_max_by_quantity: Dict[str, int],
) -> List[Tuple[str, str, Optional[Dict]]]:
    """
    Fit a chunk of (quantity, key_str, values, weights) items in one worker call (amortises IPC overhead).
    """
    import zlib

    results = []
    for quantity, key_str, values, weights in chunk:
        # Deterministic per key (EM initialisation uses np.random): a re-run
        # reproduces the same database regardless of worker scheduling.
        np.random.seed(zlib.crc32(f"{quantity}:{key_str}".encode()) & 0x7FFFFFFF)
        quantity_dist_type = dist_type.get(quantity)
        k_max = k_max_by_quantity.get(quantity, K_MAX_GMM)
        try:
            if quantity_dist_type == "gmm":
                params = fit_gmm(values, k_max, var_floor=VAR_FLOOR.get(quantity, 1e-6), n_comp_min=N_COMP_MIN,
                                 weights=weights)
            elif quantity_dist_type == "gmm_cos":
                params = fit_gmm_cosangle(values, k_max, var_floor=VAR_FLOOR.get(quantity, 1e-6),
                                          n_comp_min=N_COMP_MIN, weights=weights)
            elif quantity_dist_type == "vmm":
                params = fit_vmm_wells(values, weights, k_max)
            elif quantity_dist_type == "halfgauss":
                params = fit_contact_wall(values, weights)
            elif quantity_dist_type == "halfgauss_p5":
                params = fit_halfgauss(values)
                if params is not None:
                    params["model"] = "p5"
            else:
                params = None
        except Exception as exc:
            log.debug("Fit failed %s %r: %s", quantity, key_str, exc)
            params = None
        results.append((quantity, key_str, params))
    return results


# ---------------------------------------------------------------------------
# Data loading  (pandas groupby -- C-level, much faster than Python loop)
# ---------------------------------------------------------------------------


def _evidence_weights(df: pd.DataFrame) -> np.ndarray:
    """1 / m per row, m = rows of the same structure under the same key (each independent structure counts once per
    key; LnQM siblings of one ligand set are one structure, mixture.structure_id)."""
    if "csd_code" not in df.columns:
        return np.ones(len(df))
    from jacobik.stats.mixture import structure_ids

    sid = pd.Series(structure_ids(df["csd_code"].to_numpy()), index=df.index)
    return 1.0 / df.groupby([df["key"], sid], sort=False, observed=True)["value"].transform("size").to_numpy()


def _load_observations(
    expanded_dir: Path,
    quantity: str,
    n_min: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load expanded_<Q>.parquet, group by key, keep keys with >= n_min rows. Returns key -> (values, evidence weights).
    """
    path = expanded_dir / f"expanded_{quantity}.parquet"
    if not path.exists():
        log.warning("%s: expanded file not found: %s", quantity, path)
        return {}

    cols = [c for c in ("key", "value", "csd_code") if c in pq_columns(path)]
    df = pd.read_parquet(str(path), columns=cols)
    df["w"] = _evidence_weights(df)
    grouped = df.groupby("key", sort=False)
    counts = grouped.size()
    valid = set(counts[counts >= n_min].index)

    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key_str, idx in grouped.indices.items():
        if key_str in valid:
            result[key_str] = (df["value"].to_numpy(np.float64)[idx], df["w"].to_numpy(np.float64)[idx])

    log.info(
        "%s: %d keys with n>=%d  (total keys seen: %d)", quantity, len(result), n_min, len(counts)
    )
    return result


def pq_columns(path: Path) -> List[str]:
    import pyarrow.parquet as _pq

    return list(_pq.read_schema(str(path)).names)


def _add_q6_pooled_metal_keys(obs: Dict[str, np.ndarray], n_min: int) -> Dict[str, np.ndarray]:
    """
    Add (period_group, X) pooled Q6 keys built from the per-metal (M, X) rows so
    a metal with few observations still resolves to a fitted wall (lookup order:
    exact (M, X) -> (period_group, X) -> covalent-radius hard fallback).
    """
    import ast

    from jacobik.stats.ladders import PERIOD_GROUP

    pooled: Dict[tuple, List[tuple]] = {}
    for key_str, vals in obs.items():
        try:
            a, b = ast.literal_eval(key_str)
        except Exception:
            continue
        for m, x in ((a, b), (b, a)):
            pg = PERIOD_GROUP.get(m)
            if pg is not None:
                pooled.setdefault(tuple(sorted((pg, x))), []).append(vals)
                break
    out = dict(obs)
    for k, arrs in pooled.items():
        v = np.concatenate([a[0] for a in arrs])
        w = np.concatenate([a[1] for a in arrs])
        if len(v) >= n_min:
            out[repr(k)] = (v, w)
    return out


# ---------------------------------------------------------------------------


def fit_all(
    expanded_dir: str | Path,
    out_path: str | Path,
    quantities: Optional[List[str]] = None,
    n_min: Optional[int] = None,
    n_workers: Optional[int] = None,
    chunk_size: int = 64,
    progress: bool = True,
    pooled: Optional[Dict[str, float]] = None,
    q1_variant: Optional[str] = None,
    q6_model: str = "onset",
) -> Dict[str, Dict]:
    """
    Fit distributions for all quantities and write the constraint database.


    Parameters
    ----------
    expanded_dir : directory with expanded_<Q>.parquet files
    out_path     : path for constraints.json
    quantities   : quantities to fit (default: Q1-Q6)
    n_min        : minimum observations per key
    n_workers    : parallel workers (default: cpu_count)
    chunk_size   : keys per worker call (amortises IPC overhead)
    progress     : print parent-side progress suitable for notebooks
    pooled       : {quantity: relevance factor r} for the partially pooled quantities (default {"Q1": R_DEFAULT,
                   "Q2": R_DEFAULT}; pass {} for the hard back-off fits of every key)
    q1_variant   : Q1 ladder variant whose candidate order defines the pooling parents (default ladders default)
    q6_model     : "onset" (fit_contact_wall: wall at the 0.1 % quantile) or "p5" (fit_halfgauss: the pre-2026-10
                   wall at the 5th percentile, stiffer against clashes but inside ~5 % of real contacts); the keys
                   (incl. the polar-H "Hp" keys) are the same, so the two are interchangeable in a DB

    Returns
    -------
    dict mapping quantity -> {fitted, skipped, elapsed_s}

    """
    from joblib import Parallel, delayed, parallel_config
    from pathlib import Path
    import json
    import multiprocessing
    import os
    import time

    if n_min is None:
        n_min = N_MIN
    expanded_dir = Path(expanded_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if quantities is None:
        quantities = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    database: Dict[str, Dict] = {}
    summary: Dict[str, Dict] = {}
    partial_path = out_path.with_name(f"{out_path.stem}.partial.json")

    def _write_json_atomic(path: Path) -> None:
        """Write a valid checkpoint without exposing a partial JSON file."""
        tmp_path = path.with_name(f"{path.name}.tmp")
        with open(tmp_path, "w") as fh:
            json.dump(database, fh, separators=(",", ":"))
        os.replace(tmp_path, path)

    def _report(message: str) -> None:
        log.info(message)
        if progress:
            print(message, flush=True)

    from fit import pooling as PL

    if q6_model not in ("onset", "p5"):
        raise ValueError("q6_model must be 'onset' or 'p5'")
    dist_type = dict(_DIST_TYPE, Q6="halfgauss" if q6_model == "onset" else "halfgauss_p5")
    if pooled is None:
        pooled = {"Q1": PL.R_DEFAULT, "Q2": PL.R_DEFAULT}

    for q in quantities:
        t0 = time.perf_counter()
        _report(f"=== Fitting {q} ===")

        if q in pooled:
            pf = PL.PooledFitter(q, PL.load_expanded(expanded_dir, q))
            n_ind = pf.fit_independent(n_workers=n_workers, progress=_report)
            ent = pf.entries(pooled[q], q1_variant if q == "Q1" else None)
            for k, e in ent.items():
                database[f"{q}:{k}"] = e
            n_ad = sum(1 for e in ent.values() if e.get("src") == "adapted")
            elapsed = time.perf_counter() - t0
            _report(f"{q}: pooled (r={pooled[q]:g}): {len(ent)} entries ({n_ind} fitted on own data, {n_ad} adapted)"
                    f"  {elapsed:.1f}s")
            summary[q] = {"fitted": len(ent), "skipped": len(pf.data) - len(ent), "elapsed_s": elapsed,
                          "independent": n_ind, "adapted": n_ad, "r": pooled[q]}
            _write_json_atomic(partial_path)
            continue

        obs = _load_observations(expanded_dir, q, n_min)
        if not obs:
            _report(f"{q}: no observations above n_min={n_min}")
            summary[q] = {"fitted": 0, "skipped": 0, "elapsed_s": 0.0}
            continue

        if q == "Q6":
            obs = _add_q6_pooled_metal_keys(obs, n_min)

        # Build work list and split into chunks; keys sorted by size and dealt round-robin, so the heavy keys are
        # spread over the chunks instead of one chunk finishing last while the other workers idle (2026-10-15 full
        # refit: Q5 204 s -> 668 s on its last chunk). Seeds are per key, so the grouping does not change any fit.
        items = sorted(((q, k, v[0], v[1]) for k, v in obs.items()), key=lambda it: -len(it[2]))
        n_ch = max(1, -(-len(items) // chunk_size))
        chunks = [items[i::n_ch] for i in range(n_ch)]

        fitted = 0
        skipped = 0

        n_chunks = len(chunks)
        n_keys = len(items)
        done_chunks = 0
        report_every = max(1, n_chunks // 20)  # progress every ~5%
        worker_count = min(n_workers, n_chunks)
        _report(f"{q}: {n_keys} keys → {n_chunks} chunks ({worker_count} loky workers)")

        # loky is joblib's resilient process executor.  Unlike the standard
        # Windows ProcessPoolExecutor path, large NumPy arrays are memmapped
        # rather than pushed through a bounded multiprocessing pipe.  Results
        # stream back one chunk at a time, keeping both memory and IPC bounded.
        with parallel_config(
            backend="loky",
            inner_max_num_threads=1,
            max_nbytes="1M",
            mmap_mode="r",
        ):
            results = Parallel(
                n_jobs=worker_count,
                return_as="generator_unordered",
                pre_dispatch="2 * n_jobs",
            )(
                delayed(_fit_chunk)(chunk, dist_type, _K_MAX)
                for chunk in chunks
            )
            for chunk_results in results:
                for qty, key_str, params in chunk_results:
                    if params is None:
                        skipped += 1
                    else:
                        database[f"{qty}:{key_str}"] = params
                        fitted += 1

                done_chunks += 1
                if done_chunks % report_every == 0 or done_chunks == n_chunks:
                    pct = 100.0 * done_chunks / n_chunks
                    elapsed_so_far = time.perf_counter() - t0
                    _report(
                        f"{q}: {done_chunks}/{n_chunks} chunks ({pct:.0f}%)  "
                        f"fitted={fitted}  skipped={skipped}  {elapsed_so_far:.1f}s"
                    )

        elapsed = time.perf_counter() - t0
        _report(f"{q}: fitted={fitted}  skipped={skipped}  {elapsed:.1f}s")
        summary[q] = {"fitted": fitted, "skipped": skipped, "elapsed_s": elapsed}
        _write_json_atomic(partial_path)
        _report(f"{q}: checkpointed {partial_path.name}")

    if "Q6" in quantities:
        _add_pooled_q6_metal_keys(database)

    _report(f"Writing {out_path}  ({len(database)} entries) ...")
    t_write = time.perf_counter()
    _write_json_atomic(out_path)
    if partial_path.exists():
        partial_path.unlink()
    _report(f"Write complete in {time.perf_counter() - t_write:.1f}s")

    return summary


def _add_pooled_q6_metal_keys(database: Dict[str, Dict]) -> None:
    """
    Metal-X Q6 walls for metals with too little data borrow from their period
    group: ('period4_3d', X) = n-weighted mean of the fitted (M, X) walls.
    The runtime lookup tries (M,X) -> (period_group,X) -> covalent fallback.
    """
    import ast

    from jacobik.stats.ladders import PERIOD_GROUP

    acc: Dict[Tuple[str, str], List[Tuple[int, float, float]]] = {}
    for jk, e in database.items():
        if not jk.startswith("Q6:") or e.get("type") != "halfgauss":
            continue
        a, b = ast.literal_eval(jk[3:])
        for m, x in ((a, b), (b, a)):
            pg = PERIOD_GROUP.get(m)
            if pg and x not in PERIOD_GROUP:
                acc.setdefault((pg, x), []).append((e["n"], e["mu"], e["sigma"]))
    for (pg, x), rows in acc.items():
        n = sum(r[0] for r in rows)
        if n < N_MIN:
            continue
        mu = sum(r[0] * r[1] for r in rows) / n
        sg = sum(r[0] * r[2] for r in rows) / n
        database[f"Q6:{repr((pg, x))}"] = {"type": "halfgauss", "n": n, "k": 1, "mu": mu, "sigma": sg}


# ---------------------------------------------------------------------------
