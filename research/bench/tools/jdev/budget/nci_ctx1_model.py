"""Research-only strict-chemistry 3D density-ratio NCI model for HB candidates.

Fitting data and candidate runtime stay separate from deployed tables. The
runtime can be installed by a one-process monkeypatch in a solve driver.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from scipy.ndimage import gaussian_filter, map_coordinates, spline_filter
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_COHORT = Path("/tmp/nci_ctx1_native_partial/cohort")
DEFAULT_OUT = DEFAULT_COHORT / "nci_ctx1_surface_model_v1"
RT_KJ_MOL = 8.31446261815324e-3 * 298.15

# Sampled grid centers. Equal spacing makes the cell-volume factor cancel in
# the positive/background density ratio.
R0, DR, NR = 1.35, 0.05, 44
C0, DC, NC = -1.0, 0.05, 41
SIGMA_BINS = (3.0, 2.0, 3.0)
UNIFORM_PRIOR_MASS = 0.02
POSITIVE_SMOOTH_NATS = 0.20
SCORE_CAP_NATS = 5.0
AGG_TAU_NATS = 0.15
OUTER_TAPER_START, OUTER_TAPER_END = 2.50, 2.80
INNER_TAPER_START, INNER_TAPER_END = 1.30, 1.50
DHA_COS_LIMIT = math.cos(math.radians(150.0))  # strict positive requires cos(DHA) <= this
DHA_COS_FULL = -0.97
ACCEPTOR_COS_ZERO, ACCEPTOR_COS_FULL = 0.0, 0.25
MIN_POSITIVE_PARENTS = 12
MIN_POSITIVE_CONTACTS = 12
MIN_BACKGROUND_PARENTS = 12
MIN_BACKGROUND_OPPORTUNITIES = 200


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _key_label(key: Iterable[str]) -> str:
    return "|".join(str(x) for x in key)


def _level_ladder(raw: str | list) -> list[str]:
    ladder = json.loads(str(raw)) if isinstance(raw, (str, np.str_)) else raw
    return [_key_label(k) for k in ladder]


def _grid_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (R0 + DR * np.arange(NR), C0 + DC * np.arange(NC), C0 + DC * np.arange(NC))


def _weighted_hist(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    edges = (
        np.linspace(R0 - DR / 2, R0 + DR * (NR - 0.5), NR + 1),
        np.linspace(C0 - DC / 2, C0 + DC * (NC - 0.5), NC + 1),
        np.linspace(C0 - DC / 2, C0 + DC * (NC - 0.5), NC + 1),
    )
    h, _ = np.histogramdd(points, bins=edges, weights=weights)
    if h.sum() <= 0:
        raise ValueError("empty weighted 3D histogram")
    h /= h.sum()
    h = gaussian_filter(h, sigma=SIGMA_BINS, mode="reflect", truncate=4.0)
    h /= h.sum()
    # A small uniform prior prevents empty KDE tails from creating arbitrarily
    # deep log-ratio wells while keeping the observed regions data-dominated.
    h = (h + UNIFORM_PRIOR_MASS / h.size) / (1.0 + UNIFORM_PRIOR_MASS)
    return h


def _smooth_positive(x: np.ndarray, width: float) -> np.ndarray:
    """C2 positive part: zero below zero, identity above width."""
    out = np.zeros_like(x, dtype=np.float64)
    full = x >= width
    mid = (x > 0.0) & ~full
    out[full] = x[full]
    u = x[mid] / width
    out[mid] = width * (6 * u**3 - 8 * u**4 + 3 * u**5)
    return out


def _smoothstep(x: np.ndarray) -> np.ndarray:
    u = np.clip(x, 0.0, 1.0)
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


def _feature_gates(r: np.ndarray, cos_dha: np.ndarray, cos_acceptor: np.ndarray) -> np.ndarray:
    radial = _radial_window(r)
    linear = _smoothstep((-cos_dha - DHA_COS_LIMIT) / (-DHA_COS_FULL - DHA_COS_LIMIT))
    acceptor = _smoothstep((cos_acceptor - ACCEPTOR_COS_ZERO) /
                           (ACCEPTOR_COS_FULL - ACCEPTOR_COS_ZERO))
    return radial * linear * acceptor


def _radial_window(r: np.ndarray) -> np.ndarray:
    inner = _smoothstep((r - INNER_TAPER_START) / (INNER_TAPER_END - INNER_TAPER_START))
    outer = 1.0 - _smoothstep((r - OUTER_TAPER_START) / (OUTER_TAPER_END - OUTER_TAPER_START))
    return inner * outer


def _fit_surface(pos_points: np.ndarray, pos_weights: np.ndarray,
                 bg_points: np.ndarray, bg_weights: np.ndarray) -> tuple[np.ndarray, dict]:
    p = _weighted_hist(pos_points, pos_weights)
    q = _weighted_hist(bg_points, bg_weights)
    log_ratio = np.log(p / q)
    positive = _smooth_positive(log_ratio, POSITIVE_SMOOTH_NATS)
    positive = SCORE_CAP_NATS * np.tanh(positive / SCORE_CAP_NATS)
    rr, cd, ca = np.meshgrid(*_grid_axes(), indexing="ij")
    surface = positive * _feature_gates(rr, cd, ca)
    coeff = spline_filter(surface, order=3, mode="mirror").astype(np.float64)
    return coeff, {
        "native_log_ratio_min": float(np.min(log_ratio)),
        "native_log_ratio_max": float(np.max(log_ratio)),
        "surface_min": float(np.min(surface)),
        "surface_max": float(np.max(surface)),
        "surface_nonzero_cells": int(np.count_nonzero(surface > 1e-8)),
        "surface_mean": float(surface.mean()),
    }


def _rows_by_parent_donor(entries: list[dict], key_index: int, key: str,
                          *, positive: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    selected = []
    by_parent: dict[str, dict[tuple[int, int], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        if not bool(e["axis_defined"]):
            continue
        if e["ladder"][key_index] != key:
            continue
        by_parent[e["code"]][(int(e["donor"]), int(e["hydrogen"]))].append(e)
    parents = sorted(by_parent)
    for code in parents:
        for dh, group in by_parent[code].items():
            if positive:
                # Strict label selection already keeps at most one acceptor
                # per donor-H; retain the general normalization defensively.
                subw = 1.0 / len(group)
            else:
                # Every selected pose and every eligible acceptor for this
                # donor-H shares one unit of matched-background mass.
                subw = 1.0 / len(group)
            selected.extend((e, code, dh, subw) for e in group)
    if not selected:
        return np.empty((0, 3)), np.empty(0), {"parents": 0, "rows": 0, "donor_h": 0, "ess": 0.0}
    n_parent = len(parents)
    dh_counts = {code: len(groups) for code, groups in by_parent.items()}
    points = np.asarray([e["point"] for e, _, _, _ in selected], dtype=np.float64)
    weights = np.asarray([
        (1.0 / n_parent) / dh_counts[code] * subw
        for _e, code, _dh, subw in selected
    ], dtype=np.float64)
    weights /= weights.sum()
    meta = {"parents": n_parent, "rows": len(selected),
            "donor_h": int(sum(dh_counts.values())), "ess": float(1.0 / np.square(weights).sum())}
    return points, weights, meta


def _read_entries(cohort: Path, split: str, selection: dict,
                  native: list[dict]) -> tuple[list[dict], list[dict]]:
    positives = []
    for e in native:
        if e["split"] != split:
            continue
        positives.append({"code": e["code"], "donor": e["donor"], "hydrogen": e["hydrogen"],
                          "acceptor": e["acceptor"], "ladder": [_key_label(k) for k in e["keys"]],
                          "point": [e["r_ha"], e["cos_dha"], e["cos_acceptor"]],
                          "axis_defined": e["acceptor_axis_defined"]})
    backgrounds = []
    for row in selection["selected_rows"][split]:
        path = cohort / "strict_hb_opportunities_v1/opportunities" / split / f"{row['code']}_s0.npz"
        z = np.load(path, allow_pickle=False)
        for i in range(len(z["r_ha"])):
            backgrounds.append({
                "code": row["code"], "donor": int(z["donor"][i]),
                "hydrogen": int(z["hydrogen"][i]), "acceptor": int(z["acceptor"][i]),
                "ladder": _level_ladder(z["key_levels"][i]),
                "point": [float(z["r_ha"][i]), float(z["cos_dha"][i]), float(z["cos_acceptor"][i])],
                "axis_defined": bool(z["acceptor_axis_defined"][i]),
                "strict_selected": bool(z["strict_selected"][i]),
                "pool_rank": int(z["pool_rank"][i]),
            })
    return positives, backgrounds


def _candidate_key(ladder: list[str], supported: dict[str, int]) -> tuple[str, int] | None:
    for level, label in enumerate(ladder):
        sid = f"L{level}:{label}"
        if sid in supported:
            return sid, supported[sid]
    return None


def _interp_np(coeff: np.ndarray, points: np.ndarray) -> np.ndarray:
    coords = np.stack(((points[:, 0] - R0) / DR,
                       (points[:, 1] - C0) / DC,
                       (points[:, 2] - C0) / DC))
    coords[0] = np.clip(coords[0], 0.0, NR - 1.0)
    coords[1:] = np.clip(coords[1:], 0.0, NC - 1.0)
    return map_coordinates(coeff, coords, order=3, mode="mirror", prefilter=False)


def _group_scores(entries: list[dict], surfaces: dict[str, np.ndarray],
                  supported: dict[str, int], *, include_labels: bool = False) -> dict[tuple[str, int], float]:
    groups: dict[tuple[str, int, int], list[tuple[float, float]]] = defaultdict(list)
    for e in entries:
        if not bool(e["axis_defined"]):
            continue
        selected = _candidate_key(e["ladder"], supported)
        if selected is None:
            continue
        sid, _ = selected
        r = float(e["point"][0])
        w = float(_radial_window(np.asarray([r]))[0])
        if w <= 0:
            continue
        s = float(_interp_np(surfaces[sid], np.asarray([e["point"]]))[0])
        # Coefficients interpolate the gated positive surface. Numerical
        # spline undershoot is clipped at zero with a C2 transition at runtime.
        s = max(0.0, s)
        groups[(e["code"], int(e["pool_rank"]), int(e["hydrogen"]))].append((s, w))
    out: dict[tuple[str, int], float] = defaultdict(float)
    for (code, rank, _hydrogen), vals in groups.items():
        s = np.asarray([x[0] for x in vals]); w = np.asarray([x[1] for x in vals])
        val = AGG_TAU_NATS * (logsumexp(s / AGG_TAU_NATS + np.log(w)) - logsumexp(np.log(w)))
        out[(code, rank)] += float(val)
    return dict(out)


def fit_model(cohort: Path = DEFAULT_COHORT, out: Path = DEFAULT_OUT) -> dict:
    cohort, out = cohort.resolve(), out.resolve()
    selection_path = cohort / "selection.json"
    opportunity_dir = cohort / "strict_hb_opportunities_v1"
    native_path = opportunity_dir / "native_positive_contacts.json"
    selection = json.loads(selection_path.read_text())
    native_all = json.loads(native_path.read_text())
    native, background = _read_entries(cohort, "calibration", selection, native_all)

    support_rows = {}
    support_keys: dict[str, int] = {}
    all_key_pairs = set()
    for e in native + background:
        for level, label in enumerate(e["ladder"]):
            all_key_pairs.add((level, label))
    for level, label in sorted(all_key_pairs):
        pos_by_key = [e for e in native if e["axis_defined"] and e["ladder"][level] == label]
        bg_by_key = [e for e in background if e["axis_defined"] and e["ladder"][level] == label]
        p_parents = len({e["code"] for e in pos_by_key})
        b_parents = len({e["code"] for e in bg_by_key})
        support_rows[f"L{level}:{label}"] = {
            "level": level, "key": label, "native_parents": p_parents,
            "native_contacts": len(pos_by_key), "background_parents": b_parents,
            "background_opportunities": len(bg_by_key),
            "supported": (p_parents >= MIN_POSITIVE_PARENTS and len(pos_by_key) >= MIN_POSITIVE_CONTACTS
                          and b_parents >= MIN_BACKGROUND_PARENTS
                          and len(bg_by_key) >= MIN_BACKGROUND_OPPORTUNITIES),
        }
    for sid, row in support_rows.items():
        if row["supported"]:
            support_keys[sid] = len(support_keys)
    if not support_keys:
        raise RuntimeError("no chemical key has enough independent positive and Q6-background support")

    surfaces: dict[str, np.ndarray] = {}
    surface_meta = {}
    for sid in sorted(support_keys, key=lambda x: (int(x.split(":", 1)[0][1:]), x)):
        level = int(sid[1:sid.index(":")])
        label = sid.split(":", 1)[1]
        ppoints, pweights, pmeta = _rows_by_parent_donor(native, level, label, positive=True)
        qpoints, qweights, qmeta = _rows_by_parent_donor(background, level, label, positive=False)
        coeff, smeta = _fit_surface(ppoints, pweights, qpoints, qweights)
        surfaces[sid] = coeff
        surface_meta[sid] = {**support_rows[sid], "positive_weighting": pmeta,
                             "background_weighting": qmeta, **smeta}

    # Evaluate density-ratio enrichment at held-out strict positives and at
    # matched baseline poses, keeping all fits and scale calibration on train.
    eval_entries_by_split = {}
    eval_positive_by_split = {}
    for split in ("calibration", "validation"):
        eval_positive_by_split[split], eval_entries_by_split[split] = _read_entries(cohort, split, selection, native_all)
    positive_scores = {}
    for split, rows in eval_positive_by_split.items():
        vals = []
        for e in rows:
            selected = _candidate_key(e["ladder"], support_keys)
            if selected is None or not e["axis_defined"]:
                continue
            sid, _ = selected
            point = np.asarray([e["point"]], dtype=np.float64)
            val = max(0.0, float(_interp_np(surfaces[sid], point)[0]))
            vals.append(val)
        positive_scores[split] = vals

    group_scores = {split: _group_scores(entries, surfaces, support_keys)
                    for split, entries in eval_entries_by_split.items()}
    mace = {}
    pose_score = {}
    calibration_rows, validation_rows = [], []
    for split in ("calibration", "validation"):
        target_rows = selection["selected_rows"][split]
        records = []
        for row in target_rows:
            code = row["code"]
            z = np.load(cohort / "mace_polar1s_float64_top32" / split / f"{code}_s0.npz", allow_pickle=False)
            for rank, (de, solver_score) in enumerate(zip(z["dE_kJ_mol"], z["solver_score"])):
                records.append({"code": code, "rank": rank, "dE": float(de),
                                "solver_score": float(solver_score),
                                "nci_score": float(group_scores[split].get((code, rank), 0.0))})
        mace[split] = records
        pose_score[split] = group_scores[split]

    # One global energy scale is fit only on calibration parents, using
    # parent-centered MACE differences; a positive gamma means the learned
    # attraction is supported by lower MACE strain. It is diagnostic here and
    # the file retains gamma=1 until the calibration/validation result is reviewed.
    def scale_fit(records: list[dict]) -> dict:
        parents = sorted({r["code"] for r in records})
        num = den = 0.0
        xs, ys = [], []
        parent_slopes = []
        for code in parents:
            rr = [r for r in records if r["code"] == code]
            x = np.asarray([r["nci_score"] for r in rr], dtype=np.float64)
            y = np.asarray([r["dE"] for r in rr], dtype=np.float64)
            x -= x.mean(); y -= y.mean()
            w = 1.0 / max(len(rr), 1)
            local_den = float(np.dot(x, x))
            local_num = float(np.dot(x, y))
            num += w * local_num; den += w * local_den
            if local_den > 1e-8:
                parent_slopes.append(local_num / local_den)
            xs.extend(x.tolist()); ys.extend(y.tolist())
        beta = num / den if den > 0 else float("nan")
        robust_beta = float(np.median(parent_slopes)) if parent_slopes else float("nan")
        rng = np.random.default_rng(0)
        boot = (np.asarray([np.median(rng.choice(parent_slopes, len(parent_slopes), replace=True))
                            for _ in range(4000)]) if parent_slopes else np.asarray([]))
        from scipy.stats import spearmanr
        corr = spearmanr(xs, ys) if len(xs) > 1 else None
        return {"kJ_per_nci_nat_slope": beta,
                "gamma_from_RT": (-beta / RT_KJ_MOL) if beta < 0 else 0.0,
                "median_parent_kJ_per_nci_nat_slope": robust_beta,
                "median_parent_gamma_from_RT": (-robust_beta / RT_KJ_MOL) if robust_beta < 0 else 0.0,
                "median_parent_slope_bootstrap_95pct": ([float(x) for x in np.quantile(boot, [.025, .975])]
                                                          if len(boot) else [None, None]),
                "parents_with_nci_variation": len(parent_slopes),
                "equal_parent_within_parent_spearman": None if corr is None else float(corr.statistic),
                "parents": len(parents), "poses": len(records), "RT_kJ_mol": RT_KJ_MOL}

    scale = {split: scale_fit(records) for split, records in mace.items()}
    model = {
        "schema": "nci_ctx1_3d_density_ratio_v1",
        "fit_scope": "calibration parents only; strict native positives versus parent/donor-H-balanced u1+repulsive-Q6 opportunities",
        "geometry": {"variables": ["r_HA_A", "cos_DHA", "cos_acceptor"],
                     "grid": {"r_start": R0, "r_step": DR, "r_count": NR,
                              "cos_start": C0, "cos_step": DC, "cos_count": NC},
                     "kde_sigma_A_cosDHA_cosA": [DR * SIGMA_BINS[0], DC * SIGMA_BINS[1], DC * SIGMA_BINS[2]],
                     "support_radius_A": 3.5,
                     "positive_label": "r_HA<=2.50 A, D-H-A>=150 deg, acceptor cosine>=0 if graph axis is defined; one selected acceptor per donor-H",
                     "score_gates": {"inner_taper_A": [INNER_TAPER_START, INNER_TAPER_END],
                                     "outer_taper_A": [OUTER_TAPER_START, OUTER_TAPER_END],
                                     "DHA_cos_zero_above": DHA_COS_LIMIT,
                                     "DHA_cos_full_at_or_below": DHA_COS_FULL,
                                     "acceptor_cos_zero_at_or_below": ACCEPTOR_COS_ZERO,
                                     "acceptor_cos_full_at_or_above": ACCEPTOR_COS_FULL}},
        "weights": {"native": "equal parent mass, then equal donor-H mass", "background": "equal parent mass, then equal donor-H mass; within donor-H, equal mass per selected pose and typed acceptor opportunity", "aggregation": "per donor-H weighted log-mean-exp over active acceptors", "tau_nats": AGG_TAU_NATS},
        "regularization": {"uniform_prior_mass": UNIFORM_PRIOR_MASS,
                           "positive_part_C2_width_nats": POSITIVE_SMOOTH_NATS,
                           "smooth_cap_nats": SCORE_CAP_NATS,
                           "support_thresholds": {"native_parents": MIN_POSITIVE_PARENTS,
                                                  "native_contacts": MIN_POSITIVE_CONTACTS,
                                                  "background_parents": MIN_BACKGROUND_PARENTS,
                                                  "background_opportunities": MIN_BACKGROUND_OPPORTUNITIES}},
        "supported_keys": support_keys, "key_support": support_rows,
        "surface_metadata": surface_meta,
        "scale_diagnostic": scale,
        "gamma": max(0.0, float(scale["calibration"]["median_parent_gamma_from_RT"])),
        "gamma_note": "one global scale from the median calibration-parent fixed-effect MACE slope; validation is diagnostic only",
        "positive_native_score_nats": {s: {"n": len(v), "median": float(np.median(v)) if v else 0.0,
                                           "p10": float(np.quantile(v, .1)) if v else 0.0,
                                           "p90": float(np.quantile(v, .9)) if v else 0.0,
                                           "covered_fraction": len(v) / max(len(eval_positive_by_split[s]), 1)}
                                       for s, v in positive_scores.items()},
        "pose_score_summary": {},
        "selection_sha256": sha256(selection_path),
        "opportunity_summary_sha256": sha256(opportunity_dir / "summary.json"),
        "driver_sha256": sha256(Path(__file__)),
    }
    for split, records in mace.items():
        arr = np.asarray([r["nci_score"] for r in records])
        model["pose_score_summary"][split] = {"n": len(arr), "nonzero_fraction": float(np.mean(arr > 1e-8)) if len(arr) else 0.0,
                                               "median": float(np.median(arr)) if len(arr) else 0.0,
                                               "p90": float(np.quantile(arr, .9)) if len(arr) else 0.0,
                                               "mean": float(arr.mean()) if len(arr) else 0.0}

    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.json"
    model_path.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    labels = list(surfaces)
    np.savez_compressed(out / "surfaces.npz", surface_ids=np.asarray(labels),
                        coefficients=np.stack([surfaces[k] for k in labels]),
                        grid=np.asarray([R0, DR, NR, C0, DC, NC], dtype=np.float64))
    for split, records in mace.items():
        Path(out / f"poses_{split}.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"model": str(model_path), "supported_keys": support_keys,
                      "surface_metadata": surface_meta, "positive_native_score_nats": model["positive_native_score_nats"],
                      "scale_diagnostic": scale, "pose_score_summary": model["pose_score_summary"]}, indent=2), flush=True)
    return model


def _load_model(path: Path | str | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path or os.environ.get("JACOBIK_NCI_CTX1_MODEL", DEFAULT_OUT)).expanduser()
    model = json.loads((path / "model.json").read_text())
    z = np.load(path / "surfaces.npz", allow_pickle=False)
    surfaces = {str(k): v for k, v in zip(z["surface_ids"], z["coefficients"])}
    return model, surfaces


def _bspline_torch(coefficients, points):
    """Tensor-product cardinal cubic B-spline, C2 between all grid cells."""
    import torch

    u = torch.stack(((points[..., 0] - R0) / DR,
                     (points[..., 1] - C0) / DC,
                     (points[..., 2] - C0) / DC), dim=-1)
    hi = torch.tensor([NR - 1, NC - 1, NC - 1], dtype=u.dtype, device=u.device)
    u = torch.minimum(torch.maximum(u, torch.zeros_like(u)), hi)
    base = torch.floor(u.detach()).to(torch.long)
    t = u - base.to(u.dtype)
    weights = torch.stack((
        (1 - t) ** 3 / 6,
        (3 * t ** 3 - 6 * t ** 2 + 4) / 6,
        (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6,
        t ** 3 / 6,
    ), dim=-1)

    def mirror(idx, n):
        if n <= 1:
            return torch.zeros_like(idx)
        period = 2 * (n - 1)
        x = torch.remainder(idx, period)
        return torch.where(x >= n, period - x, x)

    ir = mirror(base[..., 0, None] + torch.arange(-1, 3, device=u.device), NR)
    ic = mirror(base[..., 1, None] + torch.arange(-1, 3, device=u.device), NC)
    ia = mirror(base[..., 2, None] + torch.arange(-1, 3, device=u.device), NC)
    grid = coefficients.to(dtype=points.dtype)
    vals = grid[ir[..., :, None, None], ic[..., None, :, None], ia[..., None, None, :]]
    w = weights[..., 0, :, None, None] * weights[..., 1, None, :, None] * weights[..., 2, None, None, :]
    return (vals * w).sum(dim=(-3, -2, -1))


def build_runtime(mol, tables=None, device="cpu"):
    """Build the candidate 3D HB runtime for one generated complex."""
    import torch
    from jacobik.topology.graph import donor_indices_of, get_ligand_neighbors
    from nci_hb_chemistry import candidates as typed_candidates, from_generated, from_training

    model, surfaces_np = _load_model()
    rdmols = getattr(mol, "_nci_ctx1_rdmols", None)
    chemistry = from_generated(mol, rdmols) if rdmols is not None else from_training(mol)
    cands = typed_candidates(mol, chemistry)
    atoms = list(mol.atoms)
    pos = {a.idx: i for i, a in enumerate(atoms)}
    supported = model["supported_keys"]
    surface_ids = list(surfaces_np)
    sid_idx = {s: i for i, s in enumerate(surface_ids)}
    nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    bound = set(donor_indices_of(mol))
    rows = []
    groups = {}
    for d, h, a, ladder in cands:
        label = [_key_label(k) for k in ladder]
        selected = _candidate_key(label, supported)
        if selected is None:
            continue
        sid, _ = selected
        neigh = list(nb.get(a, ()))
        if a in bound:
            neigh.append(mol.metal_idx)
        neigh = sorted({x for x in neigh if x in pos})
        if not neigh:
            continue
        gid = groups.setdefault((d, h), len(groups))
        rows.append((pos[d], pos[h], pos[a], [pos[x] for x in neigh], sid_idx[sid], gid))
    if not rows:
        return None
    max_nb = max(len(x[3]) for x in rows)
    nei = np.zeros((len(rows), max_nb), dtype=np.int64)
    nmask = np.zeros((len(rows), max_nb), dtype=np.float32)
    for i, row in enumerate(rows):
        nei[i, :len(row[3])] = row[3]
        nmask[i, :len(row[3])] = 1.0
    tensor = lambda x, dtype=torch.long: torch.as_tensor(x, dtype=dtype, device=device)
    coeff = np.stack([surfaces_np[s] for s in surface_ids])
    return {
        "__nci_ctx1__": True,
        "d": tensor([x[0] for x in rows]), "h": tensor([x[1] for x in rows]),
        "a": tensor([x[2] for x in rows]), "neighbors": tensor(nei),
        "neighbor_mask": tensor(nmask, torch.float32),
        "surface_id": tensor([x[4] for x in rows]), "group_id": tensor([x[5] for x in rows]),
        "coefficients": tensor(coeff, torch.float32), "n_groups": len(groups),
        "gamma": float(model.get("gamma", 1.0)), "tau": float(model["weights"]["tau_nats"]),
    }


def evaluate(rt, X):
    """Differentiable sum of parent-independent, per-donor-H attractive scores."""
    import torch
    import torch.nn.functional as F

    if not rt or not rt.get("__nci_ctx1__"):
        return X.new_zeros(X.shape[0])
    d, h, a = rt["d"], rt["h"], rt["a"]
    D, H, A = X[:, d], X[:, h], X[:, a]
    u, v = D - H, A - H
    r = torch.linalg.vector_norm(v, dim=-1).clamp(min=1e-8)
    cos_dha = ((u * v).sum(-1) /
               (torch.linalg.vector_norm(u, dim=-1) * r).clamp(min=1e-9)).clamp(-1.0, 1.0)
    N = X[:, rt["neighbors"]]
    dirs = F.normalize(N - A[:, :, None, :], dim=-1, eps=1e-9) * rt["neighbor_mask"][None, :, :, None]
    axis = F.normalize(-dirs.sum(dim=2), dim=-1, eps=1e-9)
    cos_acc = (F.normalize(H - A, dim=-1, eps=1e-9) * axis).sum(-1).clamp(-1.0, 1.0)
    points = torch.stack((r, cos_dha, cos_acc), dim=-1)
    scores = X.new_zeros(points.shape[:2])
    for sid in torch.unique(rt["surface_id"]).tolist():
        mask = rt["surface_id"] == sid
        val = _bspline_torch(rt["coefficients"][sid], points[:, mask])
        # Spline overshoot can only be a numerical interpolation artifact; a
        # C2 positive part prevents negative values from becoming repulsion.
        width = 0.02
        u = (val / width).clamp(0.0, 1.0)
        soft = width * (6 * u**3 - 8 * u**4 + 3 * u**5)
        val = torch.where(val <= 0.0, torch.zeros_like(val),
                          torch.where(val < width, soft, val))
        scores[:, mask] = val

    # Smooth radial window localizes the donor-H acceptor-choice aggregation.
    xlo = ((r - INNER_TAPER_START) / (INNER_TAPER_END - INNER_TAPER_START)).clamp(0.0, 1.0)
    inner = xlo ** 3 * (10.0 + xlo * (-15.0 + 6.0 * xlo))
    xhi = ((r - OUTER_TAPER_START) / (OUTER_TAPER_END - OUTER_TAPER_START)).clamp(0.0, 1.0)
    outer = 1.0 - xhi ** 3 * (10.0 + xhi * (-15.0 + 6.0 * xhi))
    weights = inner * outer
    total = X.new_zeros(X.shape[0])
    for gid in range(rt["n_groups"]):
        mask = rt["group_id"] == gid
        w = weights[:, mask]
        s = scores[:, mask]
        logw = torch.log(w.clamp(min=1e-20))
        numerator = torch.logsumexp(s / rt["tau"] + logw, dim=-1)
        denominator = torch.logsumexp(logw, dim=-1)
        active = w.max(dim=-1).values > 1e-10
        group_score = torch.where(active, rt["tau"] * (numerator - denominator), torch.zeros_like(numerator))
        total = total + group_score
    return float(rt["gamma"]) * total


def install_for_process(model_path: Path | str | None = None) -> None:
    """Replace only the NCI runtime hooks in the calling research process."""
    if model_path is not None:
        os.environ["JACOBIK_NCI_CTX1_MODEL"] = str(Path(model_path).resolve())
    from research.archive.code.interactions import nci
    nci.build_runtime = build_runtime
    nci.evaluate = evaluate


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fit", nargs="?", choices=("fit",), default="fit")
    p.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    fit_model(a.cohort, a.out)


if __name__ == "__main__":
    main()
