"""
src/fit/qo.py
=============
Quantity QO: S_orient joint donor orientation distributions.

Key hierarchy and model:
- Models joint orientation features of donor coordination:
  - Kappa donors: local spherical coordinates of substituents relative to M-D axis.
  - Haptic faces: ring tilt (g0), per-substituent bends (g1), and tangential slip (g2).
- Fit model: Multivariate Gaussian Mixture Model (mvgmm) with full covariance,
  BIC-selected component count K in 1..k_max, and covariance regularization floor.
- Ladder: Multi-level fallback over (k, ring, sub, metal, geom, anion) down to (k,).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import zlib
from jacobik.stats import orientation as O

log = logging.getLogger(__name__)

QUANTITY: str = "QO"
# Fit constants live in jacobik.stats.orientation (the production DB was fitted with them; copies here had drifted to
# n_min 15, hb_cut 2.4, cov_floor 1e-4 and reproduced none of the production QO entries, 2026-11-01).
N_MIN: int = O.N_MIN
HB_CUT: float = O.HB_CUT
K_MAX: int = O.K_MAX
COV_FLOOR: float = O.COV_FLOOR
W_MIN: float = O.W_MIN


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def donor_rows(mol) -> List[dict]:
    """
    Training rows for one parsed complex (crystal geometry): key parts, g0..g5, and hb_min (shortest
    intramolecular H...acceptor distance over the donor's H substituents, excluding the donor, its
    substituents and the metal; NaN when the donor has no H).
    """
    atoms = {a.idx: a for a in mol.atoms}
    M = np.asarray(atoms[mol.metal_idx].position, float)
    acc_all = [a for a in mol.atoms if a.symbol in O.ACCEPTORS]
    rows = []
    for s in O.site_specs(mol):
        d, xs = s["donor"], s["subs"]
        D = np.asarray(atoms[d].position, float)
        g = O.features(M, D, [atoms[x].position for x in xs])
        if g is None:
            continue
        hb = float("nan")
        Hs = [x for x in xs if atoms[x].symbol == "H"]
        if Hs:
            excl = set(xs) | {d, mol.metal_idx}
            acc = [a for a in acc_all if a.idx not in excl]
            if acc:
                A = np.array([a.position for a in acc], float)
                hb = float(min(np.linalg.norm(A - np.asarray(atoms[h].position, float), axis=1).min() for h in Hs))
        row = dict(csd_code=mol.csd_code, hb_min=hb, ring=s["ring"], sub=s["sub"], metal=s["metal"], geom=s["geom"],
                   anion=s["anion"], k=s["k"], sub_elems="".join(sorted(atoms[x].symbol for x in xs)))
        for i in range(6):
            row[f"g{i}"] = g[i] if i < len(g) else float("nan")
        rows.append(row)
    for s in O.haptic_site_specs(mol):
        F = np.array([atoms[a].position for a in s["face"]], float)
        g = O.haptic_features(M, F, s["closed"], [(np.asarray(atoms[a].position, float), np.asarray(atoms[x].position, float))
                                               for a, x in s["subs"]])
        if g is None:
            continue
        row = dict(csd_code=mol.csd_code, hb_min=float("nan"), ring=0, sub=s["sub"], metal=s["metal"], geom=s["geom"],
                   anion=s["anion"], k=s["k"], sub_elems="".join(sorted(atoms[x].symbol for _, x in s["subs"])))
        for i in range(6):
            row[f"g{i}"] = g[i] if i < len(g) else float("nan")
        # Raw face geometry relative to the metal, so any haptic feature can be recomputed from the parquet without
        # re-extracting: face atoms in face order (x,y,z each), closed flag, and substituents as (face position,
        # x, y, z) with face position the index of their face atom in the face order; raw_face_elems = face elements.
        fpos = {a: i for i, a in enumerate(s["face"])}
        row["raw_face"] = (F - M).ravel().tolist()
        row["raw_face_elems"] = [atoms[a].symbol for a in s["face"]]
        row["raw_closed"] = bool(s["closed"])
        row["raw_subs"] = [v for a, x in s["subs"]
                           for v in [float(fpos[a])] + (np.asarray(atoms[x].position, float) - M).tolist()]
        rows.append(row)
    return rows


def extract_rows(mol: Any) -> List[Dict[str, Any]]:
    """Extract joint donor-orientation feature rows from a TMComplex."""
    return donor_rows(mol)


# ---------------------------------------------------------------------------
# Ladder & Level Keys
# ---------------------------------------------------------------------------

def ladder_level_keys(
    sub: str,
    metal: str,
    geom: str,
    anion: int,
    k: int,
    ring: int,
) -> List[tuple]:
    """Return fallback hierarchy keys for a donor."""
    return O.level_keys(sub, metal, geom, anion, k, ring)


# ---------------------------------------------------------------------------
# Model Fitting
# ---------------------------------------------------------------------------

def fit_entry(
    G: np.ndarray,
    seed: int = 0,
    k_max: int = K_MAX,
    cov_floor: float = COV_FLOOR,
    w_min: float = W_MIN,
) -> Dict[str, Any]:
    """Fit a multivariate GMM on a feature matrix G with BIC-selected K."""
    return _fit_one(G, seed=seed, k_max=k_max, cov_floor=cov_floor, w_min=w_min)


def _fit_one(G: np.ndarray, seed: int, k_max: int, cov_floor: float, w_min: float) -> dict:
    from sklearn.mixture import GaussianMixture

    best = None
    for K in range(1, k_max + 1):
        if len(G) < 15 * K:
            break
        gm = GaussianMixture(K, covariance_type="full", reg_covar=cov_floor, random_state=seed, n_init=1,
                             max_iter=200).fit(G)
        bic = gm.bic(G)
        if best is None or bic < best[0]:
            best = (bic, gm)
    gm = best[1]
    keep = gm.weights_ >= w_min
    if not keep.any():
        keep = gm.weights_ == gm.weights_.max()
    w = gm.weights_[keep] / gm.weights_[keep].sum()
    return {"type": "mvgmm", "n": int(len(G)), "weights": w.tolist(), "means": gm.means_[keep].tolist(),
            "covs": gm.covariances_[keep].tolist()}


def training_groups(df, n_min: int = O.N_MIN, hb_cut: float = O.HB_CUT) -> Dict[tuple, np.ndarray]:
    """{ladder key: feature matrix} for every key with >= n_min training rows (H-bonded donors excluded)."""
    from jacobik.stats import ladders as L
    from jacobik.stats.mixture import structure_representative

    df = df[~(df["hb_min"] < hb_cut)]
    # LnQM: one ligand set re-optimised with up to 15 Ln. Levels that keep the metal see each sibling once; the
    # metal-free and period-group levels take the rows of one sibling per ligand set, so a ligand set counts once there
    # as a tmQM complex does (rows are unweighted in this fit).
    df = df.assign(_rep=structure_representative(df["csd_code"]) if "csd_code" in df.columns else True)
    groups: Dict[tuple, np.ndarray] = {}
    for k in sorted(int(x) for x in df["k"].unique()):
        if k >= O.HAPTIC_K0 and not O.HAPTIC_FACE_MIN <= O.face_size(k) <= O.HAPTIC_FACE_MAX:
            continue
        cols = O.feature_cols(k)
        sub = df[df.k == k].copy()
        if sub.empty:
            continue
        sub["ring"] = sub["ring"].astype(int)
        sub["anion"] = sub["anion"].astype(int)
        sub["bkt"] = sub["sub"].map(lambda s: L.bucket_family(s)[0])
        sub["fam"] = sub["sub"].map(lambda s: L.bucket_family(s)[1])
        sub["pg"] = sub["metal"].map(L.metal_period_group)
        levels = [["ring", "sub", "metal", "geom", "anion"], ["ring", "sub", "metal", "anion"], ["ring", "sub", "anion"],
                  ["ring", "sub", "metal", "geom"], ["ring", "sub", "metal"], ["ring", "sub", "pg"], ["ring", "sub"],
                  ["ring", "bkt"], ["ring", "fam"], ["ring"], []]
        for lv in levels:
            src = sub if "metal" in lv else sub[sub["_rep"]]
            if not lv:
                if len(src) >= n_min:
                    groups[(k,)] = src[cols].to_numpy(float)
                continue
            for key, part in src.groupby(lv, sort=False):
                key = key if isinstance(key, tuple) else (key,)
                if lv == ["ring", "sub", "pg"] and key[2] == L.METAL_ANY:
                    continue
                if len(part) >= n_min:
                    groups[(k,) + tuple(int(x) if isinstance(x, np.integer) else x for x in key)] = part[cols].to_numpy(float)
    return groups


def per_sub_rows(df):
    """Haptic face rows -> one row per substituent (class k + PER_SUB_OFFSET[H|X]) with g1 = that substituent's
    bend, computed from raw_face / raw_subs exactly as haptic_features does (tilt g0 and slip g2 are per face)."""
    import pandas as pd

    out = []
    for r in df[df["k"] >= O.HAPTIC_K0].to_dict("records"):
        F = np.asarray(r["raw_face"], float).reshape(-1, 3)
        S = np.asarray(r["raw_subs"], float).reshape(-1, 4)
        if not len(S):
            continue
        c = F.mean(0)
        uM = O._unit(-c)
        if uM is None:
            continue
        if len(F) == 2:
            b = O._unit(F[1] - F[0])
            n = O.eta2_plane_normal(F, [(F[int(s[0])], s[1:]) for s in S]) if O.ETA2_FACE_PLANE and b is not None else None
            if n is None:
                n = O._unit(uM - float(uM @ b) * b) if b is not None else None
        else:
            P = F - c
            idx = list(range(len(P)))
            nxt = idx[1:] + ([0] if r["raw_closed"] else [])
            n = O._unit(sum(np.cross(P[i], P[j]) for i, j in zip(idx, nxt)))
        if n is None:
            continue
        if float(n @ uM) < 0:
            n = -n
        for s in S:
            v = s[1:] - F[int(s[0])]
            L = float(np.linalg.norm(v))
            if L < 1e-8:
                continue
            q = dict(r)
            cls = "H" if L < O.PER_SUB_H_MAX else "X"
            q["k"] = int(r["k"]) + O.PER_SUB_OFFSET[cls]
            q["g1"] = float(v @ n) / L
            out.append(q)
    return pd.DataFrame(out, columns=df.columns)


def fit_rows(df):
    """The QO training table from observation rows: the rows themselves plus one row per haptic substituent
    (per_sub_rows). With eta2_face_plane, eta2 face rows are first recomputed from their raw geometry (haptic_features
    under the flag), so every row is fitted under the definition the scorer uses. The one entry point for every QO fit
    (fit.qo.fit_from_observations)."""
    import pandas as pd

    if O.ETA2_FACE_PLANE:
        e2 = df["k"] == O.HAPTIC_K0 + 2
        keep, g = [], []
        for r in df[e2].itertuples():
            F = np.asarray(r.raw_face, float).reshape(-1, 3)
            S = np.asarray(r.raw_subs, float).reshape(-1, 4)
            f = O.haptic_features(np.zeros(3), F, bool(r.raw_closed), [(F[int(x[0])], x[1:]) for x in S])
            keep.append(f is not None)
            if f is not None:
                g.append(f)
        face = df[e2][np.asarray(keep, bool)].copy()
        if len(face):
            face[["g0", "g1", "g2"]] = np.asarray(g, float)
        df = pd.concat([df[~e2], face]).sort_index()
    return pd.concat([df, per_sub_rows(df)], ignore_index=True)


def assemble_orientation(items, fitted, n_min: int = O.N_MIN, hb_cut: float = O.HB_CUT, k_max: int = O.K_MAX,
                         cov_floor: float = O.COV_FLOOR, w_min: float = O.W_MIN) -> Dict[str, dict]:
    """DB entries from ``items`` ([(ladder key, feature matrix)]) and the per-group fits (same order)."""
    meta = {"n_min": n_min, "hb_cut": hb_cut, "k_max": k_max, "cov_floor": cov_floor, "w_min": w_min}
    out = {}
    for (key, _), f in zip(items, fitted):
        f["meta"] = meta
        if key[0] >= O.HAPTIC_K0:
            f["feats"] = list(O.haptic_feats(key[0]))
            if O.per_sub_class(key[0]):
                f["per_sub"] = O.per_sub_class(key[0])
        out[f"{O.QUANTITY}:{key!r}"] = f
    return out


def group_seed(key) -> int:
    """Deterministic GMM seed of one ladder key (crc32 of its key string)."""
    return zlib.crc32(O.key_str(key).encode()) % (2**31)


def fit_orientation(df, n_min: int = O.N_MIN, hb_cut: float = O.HB_CUT, k_max: int = O.K_MAX, cov_floor: float = O.COV_FLOOR,
                    w_min: float = O.W_MIN, n_jobs: int = -1) -> Dict[str, dict]:
    """
    Fit every ladder key. Returns DB entries {"QO:<key repr>": {...}} ready to merge into constraints JSON
    (each entry also carries its fitting constants under "meta"). Deterministic: each key's GMM seed is the
    crc32 of its key string. (fit.build_db fits the same groups on its shared pool; this is the standalone path.)
    """
    from joblib import Parallel, delayed, parallel_config

    groups = training_groups(df, n_min=n_min, hb_cut=hb_cut)
    items = list(groups.items())
    tasks = (
        delayed(_fit_one)(G, group_seed(key), k_max, cov_floor, w_min)
        for key, G in items
    )
    if n_jobs == 1:
        fitted = Parallel(n_jobs=1, batch_size=8)(tasks)
    else:
        # Each fit is already NumPy / sklearn heavy.  Keep BLAS single-threaded inside every
        # process so n_jobs controls total CPU use instead of multiplying it, and let joblib
        # memmap larger arrays rather than repeatedly serialising them on Windows.
        with parallel_config(backend="loky", inner_max_num_threads=1, max_nbytes="1M", mmap_mode="r"):
            fitted = Parallel(n_jobs=n_jobs, batch_size=8, pre_dispatch="2 * n_jobs")(tasks)
    return assemble_orientation(items, fitted, n_min, hb_cut, k_max, cov_floor, w_min)


def heldout_loglik(df, entries_or_lookup, hb_cut: float = O.HB_CUT) -> dict:
    """Mean held-out log p / dim over rows (overall, non-H-bonded, H-bonded, by k); for validating a fit."""
    lookup = entries_or_lookup
    if isinstance(entries_or_lookup, dict):
        tab = {}
        import ast

        for jk, e in entries_or_lookup.items():
            tab[ast.literal_eval(jk.split(":", 1)[1])] = e
        lookup = tab.get
    vals = []
    for r in df.itertuples(index=False):
        k = int(r.k)
        f = None
        for key in O.level_keys(r.sub, r.metal, r.geom, int(r.anion), k, int(r.ring)):
            f = lookup(key)
            if f is not None:
                break
        if f is None:
            vals.append(np.nan)
            continue
        cols = O.feature_cols(k)
        if k >= O.HAPTIC_K0:
            fe = f.get("feats") if isinstance(f, dict) else getattr(f, "feats", None)
            cols = [f"g{i}" for i in (fe if fe is not None else range(O.HAPTIC_DIM))]
        g = np.array([getattr(r, c) for c in cols], float)
        vals.append(mixture_logpdf(g, f) / len(cols))
    v = np.asarray(vals)
    hb = (df["hb_min"] < hb_cut).to_numpy()
    out = {"all": float(np.nanmean(v)), "non_hbonded": float(np.nanmean(v[~hb])),
           "hbonded": float(np.nanmean(v[hb])) if hb.any() else float("nan"), "n": int(np.isfinite(v).sum())}
    for k in sorted(int(x) for x in df["k"].unique()):
        m = (df["k"] == k).to_numpy()
        if m.any():
            out[f"k{k}"] = float(np.nanmean(v[m]))
    return out


def mixture_logpdf(g: np.ndarray, f) -> float:
    """log density of one feature vector under a fitted mvgmm entry (dict or FittedDist)."""
    w = f["weights"] if isinstance(f, dict) else f.weights
    mu = f["means"] if isinstance(f, dict) else f.means
    cv = f["covs"] if isinstance(f, dict) else f.covs
    lps = []
    for wi, m, C in zip(w, mu, cv):
        C = np.asarray(C)
        diff = g - np.asarray(m)
        lps.append(np.log(wi) - 0.5 * (diff @ np.linalg.solve(C, diff) + np.linalg.slogdet(C)[1] + len(m) * np.log(2 * np.pi)))
    m = max(lps)
    return float(m + np.log(sum(np.exp(x - m) for x in lps)))


def fit_from_observations(
    df: pd.DataFrame,
    n_min: int = N_MIN,
    hb_cut: float = HB_CUT,
    k_max: int = K_MAX,
    cov_floor: float = COV_FLOOR,
    w_min: float = W_MIN,
    n_jobs: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit S_orient multivariate GMM distributions across all ladder keys.

    df: observation DataFrame with donor feature columns (from extract_rows / observations_QO); the training table is
    orientation.fit_rows(df) (per-substituent haptic rows; eta2 rows under the eta2_face_plane definition).
    """
    return fit_orientation(
        fit_rows(df),
        n_min=n_min,
        hb_cut=hb_cut,
        k_max=k_max,
        cov_floor=cov_floor,
        w_min=w_min,
        n_jobs=n_jobs,
    )


def fit_all(
    observations_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    n_min: int = N_MIN,
    hb_cut: float = HB_CUT,
    n_jobs: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Fit all QO entries from observations parquet file."""
    obs_p = Path(observations_path)
    if obs_p.is_dir():
        parquet_files = list(obs_p.glob("observations_QO_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No observations_QO_*.parquet found in {obs_p}")
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    else:
        df = pd.read_parquet(obs_p)

    results = fit_from_observations(df, n_min=n_min, hb_cut=hb_cut, n_jobs=n_jobs)
    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as fh:
            json.dump(results, fh, indent=1)
    return results
