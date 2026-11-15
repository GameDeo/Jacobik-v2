"""Lossless float64 NPZ constraint DB container encoding, metadata I/O, and container inspection."""
from __future__ import annotations

import hashlib
import logging
import zipfile

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def _ref_arrays(entries: Dict[str, Any], float_dtype) -> Tuple[np.ndarray, np.ndarray]:
    """ref_ll / ref_sd per entry (NaN = the entry has none; the writers never emit a NaN)."""
    ll = np.full(len(entries), np.nan, dtype=float_dtype)
    sd = np.full(len(entries), np.nan, dtype=float_dtype)
    for i, e in enumerate(entries.values()):
        if "ref_ll" in e:
            ll[i], sd[i] = float(e["ref_ll"]), float(e["ref_sd"])
    return ll, sd


def _add_ref(d: Dict[str, Any], arrays: Dict[str, Any], q: str, i: int) -> None:
    ll = arrays[f"{q}_ref_ll"][i]
    if not np.isnan(ll):
        d["ref_ll"], d["ref_sd"] = float(ll), float(arrays[f"{q}_ref_sd"][i])


def encode_lossless(
    db_dict: Dict[str, Any],
    float_dtype: np.dtype = np.float64,
    header_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Encode db_dict into flat arrays and a small JSON header (no pickle)."""
    header: Dict[str, Any] = {"quantities": {}, "version": 1}
    if header_meta is not None:
        header["meta"] = header_meta
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
        if N == 0:
            continue

        n_arr = np.empty(N, dtype=np.int32)
        n_eff_arr = np.empty(N, dtype=float_dtype)
        k_arr = np.empty(N, dtype=np.int32)

        has_pooled = q in ["Q1", "Q2"] or any("pooled" in e for e in entries.values())
        has_src = q in ["Q1", "Q2"] or any("src" in e for e in entries.values())
        if has_pooled:
            pooled_arr = np.empty(N, dtype=bool)

        src_vocab: List[str] = []
        src_map: Dict[str, int] = {}
        if has_src:
            src_arr = np.empty(N, dtype=np.uint8)

        has_nu = q == "Q2" or any("nu" in e for e in entries.values())
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
            "has_ref": any("ref_ll" in e for e in entries.values()),
        }
        if header["quantities"][q]["has_ref"]:
            arrays[f"{q}_ref_ll"], arrays[f"{q}_ref_sd"] = _ref_arrays(entries, float_dtype)

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
        if N > 0:
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
                "has_ref": any("ref_ll" in e for e in entries.values()),
            }
            if header["quantities"]["Q5"]["has_ref"]:
                arrays["Q5_ref_ll"], arrays["Q5_ref_sd"] = _ref_arrays(entries, float_dtype)
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
        if N > 0:
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
        if N > 0:
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

    # Remaining small categorical, PMF, and NCI tables (QG, QP, Q6P, QV, QVR, NCI, etc.)
    large_quantities = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO"}
    small_data: Dict[str, Any] = {}
    for prefix, entries in by_prefix.items():
        if prefix not in large_quantities:
            small_data[prefix] = entries
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
        N = info["count"]
        if N == 0:
            continue
        keys = bytes(arrays[f"{q}_keys"]).decode("utf-8").split("\n")
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
            if info.get("has_ref"):
                _add_ref(d, arrays, q, i)
            rebuilt[k_str] = d

    # Q5: vmm
    if "Q5" in header["quantities"]:
        info = header["quantities"]["Q5"]
        N = info["count"]
        if N > 0:
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
                if info.get("has_ref"):
                    _add_ref(d, arrays, "Q5", i)
                rebuilt[k_str] = d

    # Q6: halfgauss
    if "Q6" in header["quantities"]:
        info = header["quantities"]["Q6"]
        N = info["count"]
        if N > 0:
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
        N = info["count"]
        if N > 0:
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
    db_dict: Dict[str, Any],
    out_path: Union[str, Path],
    float_dtype: np.dtype = np.float64,
    header_meta: Optional[Dict[str, Any]] = None,
) -> int:
    """Save DB dictionary as compressed npz with flat arrays and no pickle."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = encode_lossless(db_dict, float_dtype=float_dtype, header_meta=header_meta)
    np.savez_compressed(out_path, **arrays)
    return out_path.stat().st_size


def load_lossless_npz(npz_path: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load and reconstruct exact DB dictionary and metadata header from npz without pickle."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as loaded:
        loaded_dict = {k: loaded[k] for k in loaded.files}
    header = json.loads(bytes(loaded_dict["header_json"]).decode("utf-8"))
    rebuilt = decode_lossless(loaded_dict)
    header_meta = header.get("meta", {})
    return rebuilt, header_meta


# ---------------------------------------------------------------------------
# Container metadata, manifests & inspection
# ---------------------------------------------------------------------------
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

META_KEY = "_meta"
TABLES_KEY = "_tables"
_JSON_HEAD = '{"' + META_KEY + '":'


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def dumps_entries(db: Dict[str, Any]) -> str:
    return json.dumps(db, separators=(",", ":"), allow_nan=False)


def dumps_tables(tables: Dict[str, Any]) -> str:
    return json.dumps(tables, separators=(",", ":"), allow_nan=False, sort_keys=True)


def compose_container(meta: Dict[str, Any], tables_json: str, entries_json: str) -> str:
    """constraints.json text: ``_meta`` and ``_tables`` first, then the entries (already serialised, in key order)."""
    head = (_JSON_HEAD + json.dumps(meta, separators=(",", ":"), allow_nan=False, sort_keys=True)
            + ',"' + TABLES_KEY + '":' + tables_json)
    return head + ("," + entries_json[1:] if len(entries_json) > 2 else "}")


def dumps_container(db: Dict[str, Any], meta: Dict[str, Any], tables: Dict[str, Any]) -> str:
    return compose_container(meta, dumps_tables(tables), dumps_entries(db))


def _json_head(path: Path) -> Tuple[Optional[dict], Optional[dict]]:
    dec = json.JSONDecoder()
    size = 1 << 20
    with open(path, "rb") as fh:
        while True:
            fh.seek(0)
            raw = fh.read(size)
            text = raw.decode("utf-8", errors="ignore")
            if not text.startswith(_JSON_HEAD):
                return None, None
            try:
                meta, end = dec.raw_decode(text, len(_JSON_HEAD))
                marker = ',"' + TABLES_KEY + '":'
                if not text.startswith(marker, end):
                    return meta, {}
                tables, _ = dec.raw_decode(text, end + len(marker))
                return meta, tables
            except json.JSONDecodeError:
                if len(raw) < size:
                    raise
                size *= 4


def read_container_head(path: Path | str) -> Tuple[Optional[dict], Optional[dict]]:
    """(``_meta``, ``_tables``) of a self-describing container; (None, None) for a legacy DB. Cheap for both formats."""
    path = Path(path)
    if str(path).endswith(".npz"):
        import numpy as np

        with np.load(path, allow_pickle=False) as z:
            header = json.loads(bytes(z["header_json"]).decode("utf-8"))
        small = header.get("small_quantities", {})
        if META_KEY not in small:
            return None, None
        return small[META_KEY][META_KEY], small.get(TABLES_KEY, {}).get(TABLES_KEY, {})
    return _json_head(path)


def legacy_sidecar_paths(path: Path | str) -> Dict[str, Path]:
    p = Path(path)
    return {"meta": p.with_name(p.stem + ".meta.json"), "bends": p.with_suffix(".bends.json")}


def runtime_info(path: Path | str) -> Dict[str, Any]:
    """What the runtime will find with this DB: ``source`` ('embedded' | 'sidecar' | 'none'), ``meta`` keys, the
    names of the available ``tables``, and the container ``sha256``."""
    path = Path(path)
    meta, tables = read_container_head(path)
    if meta is not None:
        return {"source": "embedded", "meta": meta, "tables": sorted(tables or {}), "sha256": sha256_file(path)}
    side = legacy_sidecar_paths(path)
    tabs = [name for name in ("bends",) if side[name].exists()]
    if side["meta"].exists():
        return {"source": "sidecar", "meta": json.loads(side["meta"].read_text()), "tables": tabs,
                "sha256": sha256_file(path)}
    return {"source": "none", "meta": {}, "tables": tabs, "sha256": sha256_file(path)}


def missing_tables(tables_available, flags: Optional[Dict[str, Any]] = None) -> list:
    """[(flag, table)] for every active runtime flag whose required table the DB does not carry."""
    from jacobik.core import profile

    have = set(tables_available or ())
    out = []
    for flag, table in sorted(profile.FLAG_TABLES.items()):
        on = (flags or {}).get(flag) if flags is not None and flag in flags else profile.get(flag)
        if on and table not in have:
            out.append((flag, table))
    return out
