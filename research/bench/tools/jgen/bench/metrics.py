"""Metrics of the bench (research/archive/docs/roadmap/roadmap_generation_v1.md 1.4.1, roadmapv2 section 5.2).

Per unit (parent, seed) a method gives its native top-1 dE and floor dE (kJ/mol, MACE minus reference). Then

    e = top1_dE                          (signed)
    f = floor_dE                         (minimum over the valid rows)
    r = e - f                            (regret, >= 0)
    d_p^(x) = mean over seeds present in both arms of (x_cand - x_base),  x in {e, f, r}

Negative is favourable. Confidence intervals resample parents (seeds kept together), 4000 resamples, seed 0.
Nothing here decides a promotion: the simple mode prints status = descriptive.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Key = Tuple[str, int]
NOISE_BAND_KJ = 0.65          # per-parent 95th percentile of |d_p^(e)| from [D12] (cal and val)
UNIT_BAND_KJ = 10.0           # units above this are reported next to their parent
TAIL_THRESHOLDS = (100.0, 200.0, 500.0, 1000.0)
WITHIN_FLOOR_KJ = 4.184
N_BOOT = 4000
BOOT_SEED = 0


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def pair_units(cand: Dict[Key, Dict], base: Dict[Key, Dict]) -> Tuple[List[Key], Dict[Key, Dict]]:
    """Keys present in both arms with finite top-1 and floor in both. Returns (paired keys, the ledger entries of the
    keys that do not pair, by reason)."""
    paired, ledger = [], {}
    for k in sorted(set(base) | set(cand)):
        b, c = base.get(k), cand.get(k)
        if b is None:
            ledger[k] = "not_in_reference"
        elif c is None:
            ledger[k] = "missing_candidate"
        elif c.get("status") != "ok" or not (_finite(c.get("top1_dE")) and _finite(c.get("floor_dE"))):
            ledger[k] = "candidate_invalid:" + str(c.get("reason", c.get("status")))
        elif not (_finite(b.get("top1_dE")) and _finite(b.get("floor_dE"))):
            ledger[k] = "reference_invalid"
        else:
            paired.append(k)
    return paired, ledger


def parent_changes(cand: Dict[Key, Dict], base: Dict[Key, Dict], keys: Iterable[Key]) -> Dict[str, Dict[str, float]]:
    """d_p^(e), d_p^(f), d_p^(r) per parent over the paired keys."""
    by_parent: Dict[str, List[Key]] = {}
    for k in keys:
        by_parent.setdefault(k[0], []).append(k)
    out = {}
    for code, ks in sorted(by_parent.items()):
        de = [cand[k]["top1_dE"] - base[k]["top1_dE"] for k in ks]
        df = [cand[k]["floor_dE"] - base[k]["floor_dE"] for k in ks]
        dr = [(cand[k]["top1_dE"] - cand[k]["floor_dE"]) - (base[k]["top1_dE"] - base[k]["floor_dE"]) for k in ks]
        out[code] = dict(seeds=len(ks), e=float(np.mean(de)), f=float(np.mean(df)), r=float(np.mean(dr)))
    return out


def bootstrap_mean(values: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Tuple[float, float, float]:
    """Mean and 95% percentile interval of the parent-level mean, resampling parents."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(cand: Dict[Key, Dict], base: Dict[Key, Dict], keys_all: List[Key]) -> Dict:
    """All bench quantities for one arm on one panel. `keys_all` is the panel's full unit list (failures stay in)."""
    paired, ledger = pair_units(cand, base)
    pc = parent_changes(cand, base, paired)
    codes = sorted(pc)
    de = np.array([pc[c]["e"] for c in codes]) if codes else np.zeros(0)
    df = np.array([pc[c]["f"] for c in codes]) if codes else np.zeros(0)
    dr = np.array([pc[c]["r"] for c in codes]) if codes else np.zeros(0)
    unit_de = {k: cand[k]["top1_dE"] - base[k]["top1_dE"] for k in paired}
    unit_df = {k: cand[k]["floor_dE"] - base[k]["floor_dE"] for k in paired}
    valid_top1 = [k for k in keys_all if cand.get(k, {}).get("status") == "ok"
                  and _finite(cand.get(k, {}).get("top1_dE"))]
    within = [k for k in valid_top1 if cand[k]["top1_dE"] - cand[k]["floor_dE"] <= WITHIN_FLOOR_KJ]
    seed_range = {}
    for c in codes:
        vals = [cand[k]["top1_dE"] for k in paired if k[0] == c]
        seed_range[c] = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    mean_e, lo_e, hi_e = bootstrap_mean(de)
    mean_f, lo_f, hi_f = bootstrap_mean(df)
    mean_r, lo_r, hi_r = bootstrap_mean(dr)
    worst = max(unit_de.items(), key=lambda kv: kv[1], default=None)
    return dict(
        units_panel=len(keys_all), units_paired=len(paired), parents_paired=len(codes),
        valid_top1_share=(len(valid_top1) / len(keys_all)) if keys_all else float("nan"),
        top1_within_4184_share=(len(within) / len(valid_top1)) if valid_top1 else float("nan"),
        e=dict(mean=mean_e, ci95=[lo_e, hi_e]), f=dict(mean=mean_f, ci95=[lo_f, hi_f]),
        r=dict(mean=mean_r, ci95=[lo_r, hi_r]),
        parent_band_exceed=sorted([c for c in codes if abs(pc[c]["e"]) > NOISE_BAND_KJ]),
        parent_band_exceed_count=int(sum(abs(pc[c]["e"]) > NOISE_BAND_KJ for c in codes)),
        parent_mean_losses_gt10=sorted([c for c in codes if pc[c]["e"] > UNIT_BAND_KJ]),
        worst_single_seed_loss=dict(key=list(worst[0]), kj=round(worst[1], 3)) if worst else None,
        units_gt10=sorted([(list(k), round(v, 2)) for k, v in unit_de.items() if abs(v) > UNIT_BAND_KJ]),
        seed_range_candidate_max=max(seed_range.values(), default=0.0),
        seed_range_gt20=sorted([c for c, v in seed_range.items() if v > 20.0]),
        tail_counts={str(t): int(sum(1 for k in valid_top1 if cand[k]["top1_dE"] > t)) for t in TAIL_THRESHOLDS},
        parent_changes=pc,
        unit_changes={f"{k[0]}:{k[1]}": dict(e=round(unit_de[k], 3), f=round(unit_df[k], 3)) for k in paired},
        ledger=[dict(code=k[0], seed=k[1], reason=v) for k, v in sorted(ledger.items())],
    )
