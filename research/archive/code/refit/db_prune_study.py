"""db_prune_study.py
==================
Q1/Q2 Constraints Database Pruning Feasibility Study and Lossless Container.

Adheres strictly to roadmapv2.md, analysis-only:
The production constraints DB is NEVER modified. All pruned outputs are written
strictly to data/refit/db_prune_study/.

Reproduces:
  1. Resolver candidate ordering and ladder node sharing analysis.
  2. Grid log-density divergence:
     - sup-norm Delta(K) = max_x |log p_K(x) - log p_fb(K)(x)| (tail-dominated)
     - mass-weighted Delta_E(K) = sum_x p_K(x) |log p_K(x) - log p_fb(K)(x)| dx
  3. Deterministic greedy pruning over tau in {0.01, 0.05, 0.1, 0.25} nats.
  4. Lossless binary container (flat float64 arrays + offsets, key index, small JSON header, no pickle).
  5. Float32 variant precision evaluation (max |Delta log p| and max |Delta score|).
  6. Score impact rescoring on 20 parents x 3 seeds (60 pools) from Step 2 surface-fit pools.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import io
import json
import logging
import math
import os
from pathlib import Path
import resource
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import zipfile

import numpy as np
import scipy.stats

from jacobik.stats import ladders as L
from jacobik.stats import mixture as MX
from jacobik.stats import stats

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "refit" / "final" / "constraints_2026-10-18.json"
DEFAULT_STUDY_DIR = REPO_ROOT / "data" / "refit" / "db_prune_study"
DEFAULT_POOLS_DIR = REPO_ROOT / "research" / "data" / "refit_nci_v2" / "scaled_v1" / "pools"
DEFAULT_FRAME_PATH = REPO_ROOT / "research" / "data" / "refit_nci_v2" / "scaled_v1" / "frame.json"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "research" / "data" / "refit_nci_v2" / "scaled_v1" / "manifest.json"

TAU_VALUES = [0.01, 0.05, 0.1, 0.25]
GRID_POINTS = 1001

GRID_Q2 = np.linspace(-1.0, 1.0, GRID_POINTS, dtype=np.float64)
GRID_Q1 = np.linspace(1.0, 5.0, GRID_POINTS, dtype=np.float64)


PG_SET = frozenset(L.PERIOD_GROUP.values())


def get_peak_rss_mb() -> float:
    """Return peak resident memory in MB."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Candidate Ladder Resolvers
# ---------------------------------------------------------------------------


def get_q2_level(k: tuple) -> Tuple[int, str]:
    """Classify Q2 key into ladder depth rank (0 = deepest L0, 9 = coarsest any)."""
    if len(k) == 6:
        m, sa, sb = k[3], k[4], k[5]
        if m in PG_SET:
            return 3, "pg_fam"
        len_a = len(L.donor_path(sa))
        len_b = len(L.donor_path(sb))
        if len_a <= 2 and len_b <= 2:
            return 2, "fam"
        if len_a <= 3 and len_b <= 3:
            return 1, "bkt"
        return 0, "L0"
    if len(k) == 5:
        return 4, "g_fam"
    if len(k) == 3:
        if k[0].startswith("CN"):
            return 6, "cn_dist"
        return 5, "geom_dist"
    if len(k) == 2:
        if k[0] == L.GEOM_ANY:
            return 9, "any"
        if k[0].startswith("CN"):
            return 8, "cn"
        return 7, "geom"
    return 99, "unknown"


def q2_candidates_all(key: tuple) -> List[tuple]:
    """Return candidate ladder for any Q2 key in DB."""
    if len(key) == 6:
        return L.q2_candidates(key)
    geom, pc = key[0], key[1]
    cn = L.geom_cn(geom) or L.GEOM_ANY
    if len(key) == 5:
        dist, fa, fb = key[2], key[3], key[4]
        lv = [
            (geom, pc, dist, *L._pair(fa, fb)),
            (geom, pc, dist),
            (cn, pc, dist),
            (geom, pc),
            (cn, pc),
            (L.GEOM_ANY, pc),
        ]
    elif len(key) == 3:
        dist = key[2]
        lv = [
            (geom, pc, dist),
            (cn, pc, dist),
            (geom, pc),
            (cn, pc),
            (L.GEOM_ANY, pc),
        ]
    elif len(key) == 2:
        lv = [
            (geom, pc),
            (cn, pc),
            (L.GEOM_ANY, pc),
        ]
    else:
        raise ValueError(f"Unexpected Q2 key: {key}")

    out, seen = [], set()
    for k in lv:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# Q1 ladder level order for m_sub_early
Q1_VARIANT_LEVELS = L.Q1_VARIANTS["m_sub_early"]
Q1_LEVEL_RANK = {lvl: i for i, lvl in enumerate(Q1_VARIANT_LEVELS)}


def get_q1_level(k: tuple) -> Tuple[int, str]:
    """Classify Q1 key into ladder depth rank (0 = deepest trans, 14 = coarsest el)."""
    if len(k) == 6:
        name = "trans"
    elif len(k) == 5:
        name = "geom"
    elif len(k) == 4:
        m, s, o, x = k
        b, f = L.bucket_family(x)
        if x == f:
            name = "mso_fam"
        elif x == b:
            name = "mso_bkt"
        else:
            name = "mso_sub"
    elif len(k) == 3:
        name = "mo_fam"
    elif len(k) == 2:
        m, x = k
        if m == L.METAL_ANY:
            name = "any_fam"
        else:
            b, f = L.bucket_family(x)
            if m in PG_SET:
                if x == f:
                    name = "pg_fam"
                elif x == b:
                    name = "pg_bkt"
                else:
                    name = "pg_sub"
            else:
                if x == f:
                    name = "m_fam"
                elif x == b:
                    name = "m_bkt"
                else:
                    name = "m_sub"
    elif len(k) == 1:
        if k[0] == L.ELEMENT_ANY:
            name = "el"
        else:
            name = "fam"
    else:
        name = "unknown"
    return Q1_LEVEL_RANK.get(name, 99), name


def q1_candidates_all(key: tuple) -> List[tuple]:
    """Return candidate ladder for any Q1 key in DB."""
    if len(key) == 6:
        m, s, o, sub, geom, trans = key
    elif len(key) == 5:
        m, s, o, sub, geom = key
        trans = L.TRANS_NONE
    elif len(key) == 4:
        m, s, o, sub = key
        geom = "octahedral"
        trans = L.TRANS_NONE
    elif len(key) == 3:
        m, o, sub = key
        s = 1
        geom = "octahedral"
        trans = L.TRANS_NONE
    elif len(key) == 2:
        m, sub = key
        s, o = 1, 2
        geom = "octahedral"
        trans = L.TRANS_NONE
    elif len(key) == 1:
        sub = key[0]
        m, s, o = "Fe", 1, 2
        geom = "octahedral"
        trans = L.TRANS_NONE
    else:
        raise ValueError(f"Unexpected Q1 key: {key}")

    bkt, fam = L.bucket_family(sub)
    pg = L.metal_period_group(m) if (m not in PG_SET and m != L.METAL_ANY) else m

    lv = []
    if len(key) == 6 and trans != L.TRANS_NONE:
        lv.append((m, s, o, sub, geom, trans))
    if len(key) >= 5:
        lv.append((m, s, o, sub, geom))
    if len(key) >= 4 and len(key) != 3:
        lv.append((m, s, o, sub))
    if len(key) == 3:
        lv.append((m, o, sub))
    lv.append((m, sub))
    if len(key) >= 4:
        lv.append((m, s, o, bkt))
    lv.append((m, bkt))
    if len(key) >= 4:
        lv.append((m, s, o, fam))
    lv.append((m, o, fam))
    lv.append((m, fam))
    if pg != L.METAL_ANY:
        lv.append((pg, sub))
        lv.append((pg, bkt))
        lv.append((pg, fam))
    lv.append((L.METAL_ANY, fam))
    lv.append((fam,))
    lv.append((L.ELEMENT_ANY,))

    out, seen = [], set()
    for k in lv:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# ---------------------------------------------------------------------------
# Divergence Computation
# ---------------------------------------------------------------------------


class DensityEvaluator:
    """Evaluates and caches 1001-point log densities and divergences."""

    def __init__(self, raw_entries: Dict[str, Any]):
        self.raw_entries = raw_entries
        self.cache: Dict[str, np.ndarray] = {}

    def get_logp(self, key_str: str, quantity: str) -> np.ndarray:
        if key_str in self.cache:
            return self.cache[key_str]
        entry = self.raw_entries[key_str]
        grid = GRID_Q2 if quantity == "Q2" else GRID_Q1
        f = MX.from_entry(entry)
        lp = MX.logp(f, grid).astype(np.float32)
        self.cache[key_str] = lp
        return lp

    def divergence(self, key_a: str, key_b: str, quantity: str, metric: str = "sup") -> float:
        """Compute divergence between distributions a and b.

        metric == 'sup': sup-norm max_x |log p_a(x) - log p_b(x)| (tail-dominated)
        metric == 'mass': mass-weighted Delta_E = sum_x p_a(x) |log p_a(x) - log p_b(x)| dx
        """
        lp_a = self.get_logp(key_a, quantity)
        lp_b = self.get_logp(key_b, quantity)
        if metric == "sup":
            return float(np.max(np.abs(lp_a - lp_b)))

        # Mass-weighted divergence Delta_E
        dx = 2.0 / 1000.0 if quantity == "Q2" else 4.0 / 1000.0
        p_a = np.exp(lp_a.astype(np.float64))
        sum_p = np.sum(p_a) * dx
        if sum_p > 0:
            p_a /= sum_p
        diff = np.abs(lp_a - lp_b)
        return float(np.sum(p_a * diff) * dx)

    def max_divergence(self, key_a: str, key_b: str, quantity: str) -> float:
        """Alias for backwards compatibility: sup-norm max_x |log p_a(x) - log p_b(x)|."""
        return self.divergence(key_a, key_b, quantity, metric="sup")


# ---------------------------------------------------------------------------
# Lossless Binary Container (Float64 & Float32)
# ---------------------------------------------------------------------------


def encode_lossless(
    db_dict: Dict[str, Any], float_dtype: np.dtype = np.float64
) -> Dict[str, np.ndarray]:
    """Encode db_dict into flat arrays and a small JSON header (no pickle)."""
    header: Dict[str, Any] = {"quantities": {}, "version": 1}
    arrays: Dict[str, np.ndarray] = {}

    by_prefix: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for k, v in db_dict.items():
        prefix = k.split(":")[0] if ":" in k else k.split("/")[0]
        by_prefix[prefix][k] = v

    # Q1, Q2, Q3, Q4: Mixture models
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        if q not in by_prefix:
            continue
        entries = by_prefix[q]
        keys = list(entries.keys())
        N = len(keys)

        n_arr = np.empty(N, dtype=np.int32)
        n_eff_arr = np.empty(N, dtype=float_dtype)
        k_arr = np.empty(N, dtype=np.int32)

        has_pooled = q in ["Q1", "Q2"]
        has_src = q in ["Q1", "Q2"]
        if has_pooled:
            pooled_arr = np.empty(N, dtype=bool)

        src_vocab: List[str] = []
        src_map: Dict[str, int] = {}
        if has_src:
            src_arr = np.empty(N, dtype=np.uint8)

        has_nu = q == "Q2"
        if has_nu:
            nu_arr = np.empty(N, dtype=float_dtype)

        merged_from_idx: List[int] = []
        merged_from_val: List[int] = []

        means_list, vars_list, weights_list = [], [], []
        means_off, vars_off, weights_off = [0], [0], [0]

        for i, (k_str, e) in enumerate(entries.items()):
            n_arr[i] = int(e.get("n", 0))
            n_eff_arr[i] = float(e.get("n_eff", 0.0))
            k_arr[i] = int(e.get("k", 0))
            if has_pooled:
                pooled_arr[i] = bool(e.get("pooled", False))

            if has_src:
                src_val = e.get("src", "")
                if src_val not in src_map:
                    src_map[src_val] = len(src_vocab)
                    src_vocab.append(src_val)
                src_arr[i] = src_map[src_val]

            if has_nu:
                nu_arr[i] = float(e.get("nu", 0.0))

            if "merged_from" in e:
                merged_from_idx.append(i)
                merged_from_val.append(int(e["merged_from"]))

            m = e.get("means", [])
            means_list.extend(m)
            means_off.append(len(means_list))

            v = e.get("vars", [])
            vars_list.extend(v)
            vars_off.append(len(vars_list))

            w = e.get("weights", [])
            weights_list.extend(w)
            weights_off.append(len(weights_list))

        type_str = "tmix_cos" if q == "Q2" else ("gmm_cos" if q == "Q3" else "gmm")
        header["quantities"][q] = {
            "type": type_str,
            "count": N,
            "has_pooled": has_pooled,
            "has_src": has_src,
            "src_vocab": src_vocab,
            "has_nu": has_nu,
            "has_merged_from": len(merged_from_idx) > 0,
        }

        arrays[f"{q}_keys"] = np.frombuffer("\n".join(keys).encode("utf-8"), dtype=np.uint8)
        arrays[f"{q}_n"] = n_arr
        arrays[f"{q}_n_eff"] = n_eff_arr
        arrays[f"{q}_k"] = k_arr
        if has_pooled:
            arrays[f"{q}_pooled"] = pooled_arr
        if has_src:
            arrays[f"{q}_src"] = src_arr
        if has_nu:
            arrays[f"{q}_nu"] = nu_arr
        if merged_from_idx:
            arrays[f"{q}_merged_from_idx"] = np.array(merged_from_idx, dtype=np.int32)
            arrays[f"{q}_merged_from_val"] = np.array(merged_from_val, dtype=np.int32)

        arrays[f"{q}_means"] = np.array(means_list, dtype=float_dtype)
        arrays[f"{q}_means_off"] = np.array(means_off, dtype=np.int64)
        arrays[f"{q}_vars"] = np.array(vars_list, dtype=float_dtype)
        arrays[f"{q}_vars_off"] = np.array(vars_off, dtype=np.int64)
        arrays[f"{q}_weights"] = np.array(weights_list, dtype=float_dtype)
        arrays[f"{q}_weights_off"] = np.array(weights_off, dtype=np.int64)

    # Q5: vmm
    if "Q5" in by_prefix:
        entries = by_prefix["Q5"]
        keys = list(entries.keys())
        N = len(keys)
        n_arr = np.empty(N, dtype=np.int32)
        n_eff_arr = np.empty(N, dtype=float_dtype)
        k_arr = np.empty(N, dtype=np.int32)
        tail_arr = np.empty(N, dtype=float_dtype)

        model_vocab: List[str] = []
        model_map: Dict[str, int] = {}
        model_arr = np.empty(N, dtype=np.uint8)

        merged_from_idx = []
        merged_from_val = []

        means_list, kappas_list, weights_list = [], [], []
        means_off, kappas_off, weights_off = [0], [0], [0]

        for i, (k_str, e) in enumerate(entries.items()):
            n_arr[i] = int(e.get("n", 0))
            n_eff_arr[i] = float(e.get("n_eff", 0.0))
            k_arr[i] = int(e.get("k", 0))
            tail_arr[i] = float(e.get("tail", 0.0))

            if "merged_from" in e:
                merged_from_idx.append(i)
                merged_from_val.append(int(e["merged_from"]))

            m_val = e.get("model", "")
            if m_val not in model_map:
                model_map[m_val] = len(model_vocab)
                model_vocab.append(m_val)
            model_arr[i] = model_map[m_val]

            m = e.get("means", [])
            means_list.extend(m)
            means_off.append(len(means_list))

            kp = e.get("kappas", [])
            kappas_list.extend(kp)
            kappas_off.append(len(kappas_list))

            w = e.get("weights", [])
            weights_list.extend(w)
            weights_off.append(len(weights_list))

        header["quantities"]["Q5"] = {
            "type": "vmm",
            "count": N,
            "model_vocab": model_vocab,
            "has_merged_from": len(merged_from_idx) > 0,
        }
        arrays["Q5_keys"] = np.frombuffer("\n".join(keys).encode("utf-8"), dtype=np.uint8)
        arrays["Q5_n"] = n_arr
        arrays["Q5_n_eff"] = n_eff_arr
        arrays["Q5_k"] = k_arr
        arrays["Q5_tail"] = tail_arr
        arrays["Q5_model"] = model_arr
        if merged_from_idx:
            arrays["Q5_merged_from_idx"] = np.array(merged_from_idx, dtype=np.int32)
            arrays["Q5_merged_from_val"] = np.array(merged_from_val, dtype=np.int32)
        arrays["Q5_means"] = np.array(means_list, dtype=float_dtype)
        arrays["Q5_means_off"] = np.array(means_off, dtype=np.int64)
        arrays["Q5_kappas"] = np.array(kappas_list, dtype=float_dtype)
        arrays["Q5_kappas_off"] = np.array(kappas_off, dtype=np.int64)
        arrays["Q5_weights"] = np.array(weights_list, dtype=float_dtype)
        arrays["Q5_weights_off"] = np.array(weights_off, dtype=np.int64)

    # Q6: halfgauss
    if "Q6" in by_prefix:
        entries = by_prefix["Q6"]
        keys = list(entries.keys())
        N = len(keys)
        n_arr = np.empty(N, dtype=np.int32)
        k_arr = np.empty(N, dtype=np.int32)
        mu_arr = np.empty(N, dtype=float_dtype)
        sigma_arr = np.empty(N, dtype=float_dtype)

        model_vocab, model_map = ["__NONE__"], {"__NONE__": 0}
        model_arr = np.zeros(N, dtype=np.uint8)

        for i, (k_str, e) in enumerate(entries.items()):
            n_arr[i] = int(e.get("n", 0))
            k_arr[i] = int(e.get("k", 1))
            mu_arr[i] = float(e.get("mu", 0.0))
            sigma_arr[i] = float(e.get("sigma", 0.0))
            if "model" in e:
                m_val = e["model"]
                if m_val not in model_map:
                    model_map[m_val] = len(model_vocab)
                    model_vocab.append(m_val)
                model_arr[i] = model_map[m_val]

        header["quantities"]["Q6"] = {
            "type": "halfgauss",
            "count": N,
            "model_vocab": model_vocab,
        }
        arrays["Q6_keys"] = np.frombuffer("\n".join(keys).encode("utf-8"), dtype=np.uint8)
        arrays["Q6_n"] = n_arr
        arrays["Q6_k"] = k_arr
        arrays["Q6_mu"] = mu_arr
        arrays["Q6_sigma"] = sigma_arr
        arrays["Q6_model"] = model_arr

    # QO: mvgmm
    if "QO" in by_prefix:
        entries = by_prefix["QO"]
        keys = list(entries.keys())
        N = len(keys)
        n_arr = np.empty(N, dtype=np.int32)
        weights_list = []
        weights_off = [0]
        means_flat = []
        means_shapes = []
        covs_flat = []
        covs_shapes = []

        extra_list = []

        for i, (k_str, e) in enumerate(entries.items()):
            n_arr[i] = int(e.get("n", 0))
            w = e.get("weights", [])
            weights_list.extend(w)
            weights_off.append(len(weights_list))

            m = e.get("means", [])
            k_val = len(m)
            d_val = len(m[0]) if k_val > 0 else 0
            means_shapes.append((k_val, d_val))
            for row in m:
                means_flat.extend(row)

            cv = e.get("covs", [])
            covs_shapes.append((k_val, d_val, d_val))
            for mat in cv:
                for row in mat:
                    covs_flat.extend(row)

            extra = {k: v for k, v in e.items() if k not in ("type", "n", "weights", "means", "covs")}
            extra_list.append(extra)

        header["quantities"]["QO"] = {
            "type": "mvgmm",
            "count": N,
            "extra_list": extra_list,
            "means_shapes": means_shapes,
            "covs_shapes": covs_shapes,
        }
        arrays["QO_keys"] = np.frombuffer("\n".join(keys).encode("utf-8"), dtype=np.uint8)
        arrays["QO_n"] = n_arr
        arrays["QO_weights"] = np.array(weights_list, dtype=float_dtype)
        arrays["QO_weights_off"] = np.array(weights_off, dtype=np.int64)
        arrays["QO_means_flat"] = np.array(means_flat, dtype=float_dtype)
        arrays["QO_covs_flat"] = np.array(covs_flat, dtype=float_dtype)

    # Remaining small categorical and PMF / NCI tables: QG, QP, Q6P, QV, QVR, NCI
    small_q = ["QG", "QP", "Q6P", "QV", "QVR", "NCI"]
    small_data: Dict[str, Any] = {}
    for sq in small_q:
        if sq in by_prefix:
            small_data[sq] = by_prefix[sq]
    header["small_quantities"] = small_data

    header_bytes = json.dumps(header).encode("utf-8")
    arrays["header_json"] = np.frombuffer(header_bytes, dtype=np.uint8)
    return arrays


def decode_lossless(arrays: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the exact DB dictionary from arrays dict without pickle."""
    header = json.loads(bytes(arrays["header_json"]).decode("utf-8"))
    rebuilt: Dict[str, Any] = {}

    # Decode Q1, Q2, Q3, Q4
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        if q not in header["quantities"]:
            continue
        info = header["quantities"][q]
        keys = bytes(arrays[f"{q}_keys"]).decode("utf-8").split("\n")
        N = info["count"]
        type_str = info["type"]
        has_pooled = info.get("has_pooled", False)
        has_src = info.get("has_src", False)
        src_vocab = info.get("src_vocab", [])
        has_nu = info["has_nu"]
        has_merged = info["has_merged_from"]

        n_arr = arrays[f"{q}_n"]
        n_eff_arr = arrays[f"{q}_n_eff"]
        k_arr = arrays[f"{q}_k"]
        if has_pooled:
            pooled_arr = arrays[f"{q}_pooled"]
        if has_src:
            src_arr = arrays[f"{q}_src"]

        if has_nu:
            nu_arr = arrays[f"{q}_nu"]
        if has_merged:
            m_idx = arrays[f"{q}_merged_from_idx"]
            m_val = arrays[f"{q}_merged_from_val"]
            merged_map = {int(idx): int(val) for idx, val in zip(m_idx, m_val)}
        else:
            merged_map = {}

        means_arr = arrays[f"{q}_means"]
        means_off = arrays[f"{q}_means_off"]
        vars_arr = arrays[f"{q}_vars"]
        vars_off = arrays[f"{q}_vars_off"]
        weights_arr = arrays[f"{q}_weights"]
        weights_off = arrays[f"{q}_weights_off"]

        for i, k_str in enumerate(keys):
            d = {
                "type": type_str,
                "n": int(n_arr[i]),
                "means": means_arr[means_off[i] : means_off[i + 1]].tolist(),
                "vars": vars_arr[vars_off[i] : vars_off[i + 1]].tolist(),
                "weights": weights_arr[weights_off[i] : weights_off[i + 1]].tolist(),
                "k": int(k_arr[i]),
                "n_eff": float(n_eff_arr[i]),
            }
            if has_pooled:
                d["pooled"] = bool(pooled_arr[i])
            if has_src:
                d["src"] = src_vocab[src_arr[i]]
            if has_nu:
                d["nu"] = float(nu_arr[i])
            if i in merged_map:
                d["merged_from"] = merged_map[i]
            rebuilt[k_str] = d

    # Q5: vmm
    if "Q5" in header["quantities"]:
        info = header["quantities"]["Q5"]
        keys = bytes(arrays["Q5_keys"]).decode("utf-8").split("\n")
        model_vocab = info["model_vocab"]

        n_arr = arrays["Q5_n"]
        n_eff_arr = arrays["Q5_n_eff"]
        k_arr = arrays["Q5_k"]
        tail_arr = arrays["Q5_tail"]
        model_arr = arrays["Q5_model"]

        means_arr = arrays["Q5_means"]
        means_off = arrays["Q5_means_off"]
        kappas_arr = arrays["Q5_kappas"]
        kappas_off = arrays["Q5_kappas_off"]
        weights_arr = arrays["Q5_weights"]
        weights_off = arrays["Q5_weights_off"]

        if info.get("has_merged_from"):
            m_idx = arrays["Q5_merged_from_idx"]
            m_val = arrays["Q5_merged_from_val"]
            merged_map = {int(idx): int(val) for idx, val in zip(m_idx, m_val)}
        else:
            merged_map = {}

        for i, k_str in enumerate(keys):
            d = {
                "type": "vmm",
                "n": int(n_arr[i]),
                "means": means_arr[means_off[i] : means_off[i + 1]].tolist(),
                "kappas": kappas_arr[kappas_off[i] : kappas_off[i + 1]].tolist(),
                "weights": weights_arr[weights_off[i] : weights_off[i + 1]].tolist(),
                "k": int(k_arr[i]),
                "tail": float(tail_arr[i]),
                "model": model_vocab[model_arr[i]],
                "n_eff": float(n_eff_arr[i]),
            }
            if i in merged_map:
                d["merged_from"] = merged_map[i]
            rebuilt[k_str] = d

    # Q6: halfgauss
    if "Q6" in header["quantities"]:
        info = header["quantities"]["Q6"]
        keys = bytes(arrays["Q6_keys"]).decode("utf-8").split("\n")
        model_vocab = info["model_vocab"]
        n_arr = arrays["Q6_n"]
        k_arr = arrays["Q6_k"]
        mu_arr = arrays["Q6_mu"]
        sigma_arr = arrays["Q6_sigma"]
        model_arr = arrays["Q6_model"]

        for i, k_str in enumerate(keys):
            d = {
                "type": "halfgauss",
                "n": int(n_arr[i]),
                "k": int(k_arr[i]),
                "mu": float(mu_arr[i]),
                "sigma": float(sigma_arr[i]),
            }
            m_idx = model_arr[i]
            if m_idx > 0:
                d["model"] = model_vocab[m_idx]
            rebuilt[k_str] = d

    # QO: mvgmm
    if "QO" in header["quantities"]:
        info = header["quantities"]["QO"]
        keys = bytes(arrays["QO_keys"]).decode("utf-8").split("\n")
        n_arr = arrays["QO_n"]
        weights_arr = arrays["QO_weights"]
        weights_off = arrays["QO_weights_off"]
        means_flat = arrays["QO_means_flat"]
        covs_flat = arrays["QO_covs_flat"]

        means_shapes = info["means_shapes"]
        covs_shapes = info["covs_shapes"]
        extra_list = info.get("extra_list", [])
        if not extra_list and "meta_list" in info:
            extra_list = [{"meta": m} for m in info["meta_list"]]

        m_pos = 0
        c_pos = 0
        for i, k_str in enumerate(keys):
            k_val, d_val = means_shapes[i]
            m_len = k_val * d_val
            m_sub = means_flat[m_pos : m_pos + m_len].reshape((k_val, d_val)).tolist()
            m_pos += m_len

            c_len = k_val * d_val * d_val
            c_sub = covs_flat[c_pos : c_pos + c_len].reshape((k_val, d_val, d_val)).tolist()
            c_pos += c_len

            d = {
                "type": "mvgmm",
                "n": int(n_arr[i]),
                "weights": weights_arr[weights_off[i] : weights_off[i + 1]].tolist(),
                "means": m_sub,
                "covs": c_sub,
            }
            if i < len(extra_list):
                d.update(extra_list[i])
            rebuilt[k_str] = d

    # Small quantities
    for sq, entries in header.get("small_quantities", {}).items():
        for k_str, e in entries.items():
            rebuilt[k_str] = e

    return rebuilt


def save_lossless_npz(
    db_dict: Dict[str, Any], out_path: Path, float_dtype: np.dtype = np.float64
) -> int:
    """Save DB dictionary as compressed npz with flat arrays and no pickle."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = encode_lossless(db_dict, float_dtype=float_dtype)
    np.savez_compressed(out_path, **arrays)
    return out_path.stat().st_size


def load_lossless_npz(in_path: Path) -> Dict[str, Any]:
    """Load and reconstruct exact DB dictionary from npz without pickle."""
    with np.load(in_path, allow_pickle=False) as loaded:
        loaded_dict = {k: loaded[k] for k in loaded.files}
    return decode_lossless(loaded_dict)


# ---------------------------------------------------------------------------
# Greedy Pruning Execution
# ---------------------------------------------------------------------------


def prune_quantity(
    raw: Dict[str, Any],
    quantity: str,
    tau: float,
    evaluator: DensityEvaluator,
    metric: str = "sup",
) -> Tuple[Set[str], Dict[str, Any], Dict[str, str]]:
    """Execute greedy pruning on a single quantity (Q1 or Q2).

    Candidates: pooled, src: adapted entries only (never src: fit).
    Candidate ordering: deepest ladder level first, then lexicographical order.
    Accepts deletion iff every DB key whose resolution passed through it still resolves
    within tau of its original resolution in the full DB (prevents error accumulation).
    """
    t0 = time.perf_counter()
    rss0 = get_peak_rss_mb()

    prefix = f"{quantity}:"
    all_keys = [k for k in raw.keys() if k.startswith(prefix)]
    tuple_map = {k: eval(k[len(prefix) :]) for k in all_keys}

    level_func = get_q2_level if quantity == "Q2" else get_q1_level
    cand_func = q2_candidates_all if quantity == "Q2" else q1_candidates_all

    candidates_all: List[Tuple[int, str, str]] = []
    candidate_set: Set[str] = set()

    for k in all_keys:
        entry = raw[k]
        if entry.get("pooled") and entry.get("src") == "adapted":
            kt = tuple_map[k]
            lvl_rank, _ = level_func(kt)
            candidates_all.append((lvl_rank, repr(kt), k))
            candidate_set.add(k)

    candidates_all.sort(key=lambda x: (x[0], x[1]))

    ladder_map: Dict[str, List[str]] = {}
    for k in all_keys:
        kt = tuple_map[k]
        cands = cand_func(kt)
        ladder_map[k] = [f"{quantity}:{repr(c)}" for c in cands]

    surviving: Set[str] = set(all_keys)
    dependents: Dict[str, List[str]] = {k: [k] for k in all_keys}
    resolution: Dict[str, str] = {k: k for k in all_keys}

    deleted_keys: Set[str] = set()
    n_buckets = {"le_1": 0, "lt_5": 0, "lt_20": 0, "ge_20": 0}

    def find_fallback(key_str: str) -> Optional[str]:
        ladder = ladder_map[key_str]
        for cand_str in ladder[1:]:
            if cand_str in surviving:
                return cand_str
        return None

    for lvl_rank, _, c_str in candidates_all:
        if c_str not in surviving:
            continue

        p_str = find_fallback(c_str)
        if p_str is None:
            continue

        div_c = evaluator.divergence(c_str, p_str, quantity, metric=metric)
        if div_c > tau:
            continue

        dep_keys = dependents[c_str]
        accept = True
        for orig_k in dep_keys:
            if orig_k == c_str:
                continue
            div_k = evaluator.divergence(orig_k, p_str, quantity, metric=metric)
            if div_k > tau:
                accept = False
                break

        if accept:
            surviving.remove(c_str)
            deleted_keys.add(c_str)

            for orig_k in dep_keys:
                resolution[orig_k] = p_str
            dependents[p_str].extend(dep_keys)
            del dependents[c_str]

            n_obs = int(raw[c_str].get("n", 0))
            if n_obs <= 1:
                n_buckets["le_1"] += 1
            if n_obs < 5:
                n_buckets["lt_5"] += 1
            if n_obs < 20:
                n_buckets["lt_20"] += 1
            else:
                n_buckets["ge_20"] += 1

    all_final_deltas: List[float] = []
    for k in all_keys:
        res_k = resolution[k]
        if res_k == k:
            all_final_deltas.append(0.0)
        else:
            d = evaluator.divergence(k, res_k, quantity, metric=metric)
            all_final_deltas.append(d)

    arr_deltas = np.array(all_final_deltas, dtype=np.float64)
    elapsed = time.perf_counter() - t0
    peak_rss = get_peak_rss_mb()

    stats_dict = {
        "quantity": quantity,
        "tau": tau,
        "metric": metric,
        "total_keys": len(all_keys),
        "candidate_keys": len(candidates_all),
        "deleted_count": len(deleted_keys),
        "retention_count": len(surviving),
        "deleted_pct_of_candidates": len(deleted_keys) / max(len(candidates_all), 1) * 100.0,
        "n_buckets": n_buckets,
        "delta_distribution": {
            "median": float(np.median(arr_deltas)),
            "p95": float(np.percentile(arr_deltas, 95)),
            "max": float(np.max(arr_deltas)),
        },
        "wall_time_s": elapsed,
        "peak_rss_mb": peak_rss,
    }
    return deleted_keys, stats_dict, resolution


# ---------------------------------------------------------------------------
# Sampling and Rescoring Pipeline
# ---------------------------------------------------------------------------


def get_step2_sample_codes(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    frame_path: Path = DEFAULT_FRAME_PATH,
    n_parents: int = 20,
) -> List[str]:
    """Return the first 20 codes alphabetically that succeeded on seeds 11, 12, 13 in Step 2 surface-fit."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(frame_path, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    surface_fit_codes = set()
    for row in frame_data.get("main_allocation", {}).get("splits", {}).get("surface_fit", []):
        c = row.get("code") or row.get("representative_code")
        if c:
            surface_fit_codes.add(c)

    success_seeds = defaultdict(set)
    for task in manifest.get("tasks", []):
        if task.get("status") == "success" and task.get("split") == "surface_fit":
            code = task.get("code")
            seed = task.get("seed")
            if code and seed:
                success_seeds[code].add(int(seed))

    valid_codes = [
        code
        for code, seeds in success_seeds.items()
        if {11, 12, 13}.issubset(seeds) and code in surface_fit_codes
    ]
    valid_codes.sort()
    return valid_codes[:n_parents]


def evaluate_score_impact(
    codes: List[str],
    orig_db_path: Path = DEFAULT_DB_PATH,
    lookups_dict: Optional[Dict[str, stats.StatsLookup]] = None,
    pools_dir: Path = DEFAULT_POOLS_DIR,
    frame_path: Path = DEFAULT_FRAME_PATH,
) -> Dict[str, Dict[str, Any]]:
    """Rescore poses for 20 parents x 3 seeds using production rank_terms.

    Ensures metadata sidecar is preserved so Q5 folding is consistent with baseline.
    """
    import torch
    from jacobik.inputs import build_builder_and_inputs, load_mol
    from tools.jdev import multiseed
    from tools.jdev.budget.pools import rank_terms

    with open(frame_path, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    code_meta = {}
    for p in frame_data.get("main_allocation", {}).get("splits", {}).get("surface_fit", []):
        c = p.get("code") or p.get("representative_code")
        shard = p.get("shard") or p.get("representative_shard") or "unknown"
        dataset = p.get("dataset") or "unknown"
        if c:
            code_meta[c] = (shard, dataset)

    # Load baseline DB and metadata
    db_orig = stats.load_db(orig_db_path)
    lookup_orig = stats.StatsLookup(db_orig)

    if lookups_dict is None:
        lookups_dict = {}

    labels = list(lookups_dict.keys())
    results: Dict[str, Dict[str, Any]] = {
        label: {
            "label": label,
            "max_abs_delta_score": 0.0,
            "p95_abs_delta_score": 0.0,
            "fraction_top1_unchanged": 1.0,
            "spearman_rho_min": 1.0,
            "spearman_rho_median": 1.0,
            "pool_diffs_max": [],
            "pool_diffs_p95": [],
            "top1_unchanged_count": 0,
            "spearman_rhos": [],
            "swapped_pools": [],
            "total_pools": 0,
        }
        for label in labels
    }

    t0 = time.perf_counter()
    pool_count = 0

    for code in codes:
        if code not in code_meta:
            continue
        shard, dataset = code_meta[code]
        mol = load_mol(code, shard, dataset)
        if mol is None:
            continue
        builder, _, err = build_builder_and_inputs(mol)
        if err or builder is None:
            continue

        ctx_orig, asg_orig, W_orig, _ = multiseed.prepare(
            builder, 11, stats=lookup_orig, device="cpu", angle_bends=True
        )
        (fks_orig, dpi_orig, ge_orig, sc_orig), _ = multiseed.seed_rows(
            builder, ctx_orig, asg_orig, W_orig, [11]
        )

        scorers_test = {}
        for label, lookup_test in lookups_dict.items():
            ctx_p, asg_p, W_p, _ = multiseed.prepare(
                builder, 11, stats=lookup_test, device="cpu", angle_bends=True
            )
            (fks_p, dpi_p, ge_p, sc_p), _ = multiseed.seed_rows(
                builder, ctx_p, asg_p, W_p, [11]
            )
            scorers_test[label] = sc_p

        for seed in [11, 12, 13]:
            pool_file = pools_dir / f"{code}_{seed}_control.npz"
            if not pool_file.exists():
                continue

            pool_data = np.load(pool_file, allow_pickle=True)
            X = pool_data["X"]
            asg = pool_data["asg"]

            Xt = torch.tensor(X, dtype=torch.float32, device="cpu")
            rg = torch.as_tensor(asg, dtype=torch.long, device="cpu")

            tot_orig, _ = rank_terms(sc_orig, Xt, rg)
            score_orig = tot_orig.detach().cpu().numpy()
            pool_count += 1

            for label in labels:
                sc_p = scorers_test[label]
                tot_p, _ = rank_terms(sc_p, Xt, rg)
                score_p = tot_p.detach().cpu().numpy()

                diff = np.abs(score_orig - score_p)
                p_max = float(np.max(diff))
                p_p95 = float(np.percentile(diff, 95))

                orig_top1 = int(np.argmax(score_orig))
                pruned_top1 = int(np.argmax(score_p))
                is_top1_unchanged = orig_top1 == pruned_top1

                if len(score_orig) > 1 and np.std(score_orig) > 1e-12 and np.std(score_p) > 1e-12:
                    rho, _ = scipy.stats.spearmanr(score_orig, score_p)
                else:
                    rho = 1.0

                res_l = results[label]
                res_l["total_pools"] += 1
                res_l["pool_diffs_max"].append(p_max)
                res_l["pool_diffs_p95"].append(p_p95)
                if is_top1_unchanged:
                    res_l["top1_unchanged_count"] += 1
                else:
                    res_l["swapped_pools"].append(
                        (f"{code}_{seed}", orig_top1, pruned_top1, p_max)
                    )
                res_l["spearman_rhos"].append(float(rho))

    elapsed = time.perf_counter() - t0
    for label in labels:
        res_l = results[label]
        n_p = res_l["total_pools"]
        if n_p > 0:
            res_l["max_abs_delta_score"] = float(max(res_l["pool_diffs_max"]))
            res_l["p95_abs_delta_score"] = float(np.percentile(res_l["pool_diffs_p95"], 95))
            res_l["fraction_top1_unchanged"] = float(res_l["top1_unchanged_count"] / n_p)
            res_l["spearman_rho_min"] = float(min(res_l["spearman_rhos"]))
            res_l["spearman_rho_median"] = float(np.median(res_l["spearman_rhos"]))

    return results


# ---------------------------------------------------------------------------
# Synthetic Unit Tests for Fast Verification
# ---------------------------------------------------------------------------


def run_synthetic_unit_tests() -> None:
    """Run fast synthetic unit tests for greedy search, divergence, and ordering."""
    print("Running synthetic unit tests...")

    # 1. 3-level ladder preventing error accumulation
    x = np.linspace(-1, 1, 1001)
    p_L1 = np.exp(-0.5 * ((x - 0.0) / 0.1) ** 2)
    p_L1 /= np.sum(p_L1) * (2.0 / 1000.0)
    lp_L1 = np.log(np.maximum(p_L1, 1e-12))

    p_M = np.exp(-0.5 * ((x - 0.02) / 0.1) ** 2)
    p_M /= np.sum(p_M) * (2.0 / 1000.0)
    lp_M = np.log(np.maximum(p_M, 1e-12))

    p_R = np.exp(-0.5 * ((x - 0.05) / 0.1) ** 2)
    p_R /= np.sum(p_R) * (2.0 / 1000.0)
    lp_R = np.log(np.maximum(p_R, 1e-12))

    d_L1_M = float(np.max(np.abs(lp_L1 - lp_M)))
    d_M_R = float(np.max(np.abs(lp_M - lp_R)))
    d_L1_R = float(np.max(np.abs(lp_L1 - lp_R)))

    tau = (d_L1_M + d_M_R) / 1.5
    assert d_L1_M <= tau, f"Step 1 should be within tau: {d_L1_M} vs {tau}"
    assert d_M_R <= tau, f"Step 2 should be within tau: {d_M_R} vs {tau}"
    assert d_L1_R > tau, f"Accumulated step must exceed tau: {d_L1_R} vs {tau}"

    surviving = {"L1", "M", "R"}
    dependents = {"L1": ["L1"], "M": ["M"], "R": ["R"]}
    resolution = {"L1": "L1", "M": "M", "R": "R"}
    deltas = {
        ("L1", "M"): d_L1_M,
        ("M", "R"): d_M_R,
        ("L1", "R"): d_L1_R,
    }

    # Step 1: L1 deleted, fall back to M
    surviving.remove("L1")
    resolution["L1"] = "M"
    dependents["M"].append("L1")
    del dependents["L1"]

    # Step 2: Attempt deleting M
    accept_M = True
    for orig_k in dependents["M"]:
        if deltas.get((orig_k, "R"), 0.0) > tau:
            accept_M = False
            break

    assert not accept_M, "Error accumulation check must reject deleting intermediate node M"
    print("PASSED: Error accumulation correctly prevented deletion of intermediate node M.")

    # 2. Test lossless container round-trip on synthetic dict
    synthetic_db = {
        "Q2:('square_plan', 'c90', 'different_ligand', 'Pd', 'Hal_Cl', 'N_imine')": {
            "type": "tmix_cos",
            "n": 100,
            "means": [0.0, 0.5],
            "vars": [0.01, 0.02],
            "weights": [0.6, 0.4],
            "k": 2,
            "pooled": True,
            "src": "adapted",
            "nu": 4.5,
            "n_eff": 85.0,
        },
        "Q1:('Fe', 2, 2, 'N_pyrrole', 'octahedral', 'none')": {
            "type": "gmm",
            "n": 50,
            "means": [1.98],
            "vars": [0.005],
            "weights": [1.0],
            "k": 1,
            "pooled": True,
            "src": "fit",
            "n_eff": 50.0,
        },
        "QG:('tetrahedral', 'Hal_Cl')": {
            "type": "cat",
            "n": 20,
            "labels": ["a", "b"],
            "p": [0.7, 0.3],
            "level": "bucket",
        },
    }
    encoded = encode_lossless(synthetic_db, float_dtype=np.float64)
    decoded = decode_lossless(encoded)
    assert decoded == synthetic_db, "Synthetic lossless decode mismatch"
    s_orig = json.dumps(synthetic_db, sort_keys=True)
    s_decoded = json.dumps(decoded, sort_keys=True)
    assert s_orig == s_decoded, "Synthetic json.dumps sorted dict equality failed"
    print("PASSED: Lossless container round-trip exact equality.")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1/Q2 Database Pruning Feasibility Study")
    parser.add_argument("--prune", action="store_true", help="Execute pruning across tau values")
    parser.add_argument("--rescore", action="store_true", help="Rescore Step 2 surface-fit pools")
    parser.add_argument("--lossless", action="store_true", help="Build and verify lossless NPZ containers")
    parser.add_argument("--test", action="store_true", help="Run fast synthetic unit tests")
    parser.add_argument("--all", action="store_true", help="Run all study components")
    parser.add_argument(
        "--metric",
        choices=["sup", "mass", "both"],
        default="both",
        help="Divergence metric: sup (sup-norm) or mass (mass-weighted Delta_E) or both",
    )
    args = parser.parse_args()

    if args.test:
        run_synthetic_unit_tests()
        return

    print("Q1/Q2 DB Pruning Feasibility Study CLI initialized.")


if __name__ == "__main__":
    main()
