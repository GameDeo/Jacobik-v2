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


def _pack_vocab(vals: List[str]) -> Tuple[np.ndarray, List[str]]:
    vocab: List[str] = []
    vmap: Dict[str, int] = {}
    arr = np.empty(len(vals), dtype=np.uint8)
    for i, v in enumerate(vals):
        idx = vmap.get(v)
        if idx is None:
            idx = len(vocab)
            vmap[v] = idx
            vocab.append(v)
        arr[i] = idx
    return arr, vocab


def _pack_ragged(entries: Dict[str, Any], field: str, dtype) -> Tuple[np.ndarray, np.ndarray]:
    flat: List[Any] = []
    off = [0]
    for e in entries.values():
        flat.extend(e.get(field, ()))
        off.append(len(flat))
    return np.array(flat, dtype=dtype), np.array(off, dtype=np.int64)


def _pack_merged(entries: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    idx, val = [], []
    for i, e in enumerate(entries.values()):
        if "merged_from" in e:
            idx.append(i)
            val.append(int(e["merged_from"]))
    if not idx:
        return None, None
    return np.array(idx, dtype=np.int32), np.array(val, dtype=np.int32)


def _merged_map(arrays: Dict[str, Any], q: str, has_merged: bool) -> Dict[int, int]:
    if not has_merged:
        return {}
    return {int(i): int(v) for i, v in zip(arrays[f"{q}_merged_from_idx"], arrays[f"{q}_merged_from_val"])}


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
        by_prefix[k.split(":")[0] if ":" in k else k.split("/")[0]][k] = v

    for q in ("Q1", "Q2", "Q3", "Q4"):
        entries = by_prefix.get(q)
        if not entries:
            continue
        keys = list(entries.keys())
        vals = list(entries.values())
        has_pooled = q in ("Q1", "Q2") or any("pooled" in e for e in vals)
        has_src = q in ("Q1", "Q2") or any("src" in e for e in vals)
        has_nu = q == "Q2" or any("nu" in e for e in vals)
        has_ref = any("ref_ll" in e for e in vals)
        m_idx, m_val = _pack_merged(entries)
        src_arr, src_vocab = _pack_vocab([e.get("src", "") for e in vals]) if has_src else (None, [])

        header["quantities"][q] = {
            "type": "tmix_cos" if q == "Q2" else ("gmm_cos" if q == "Q3" else "gmm"),
            "count": len(keys), "has_pooled": has_pooled, "has_src": has_src, "src_vocab": src_vocab,
            "has_nu": has_nu, "has_merged_from": m_idx is not None, "has_ref": has_ref,
        }
        arrays[f"{q}_keys"] = np.frombuffer("\n".join(keys).encode("utf-8"), dtype=np.uint8)
        arrays[f"{q}_n"] = np.array([int(e.get("n", 0)) for e in vals], dtype=np.int32)
        arrays[f"{q}_n_eff"] = np.array([float(e.get("n_eff", 0.0)) for e in vals], dtype=float_dtype)
        arrays[f"{q}_k"] = np.array([int(e.get("k", 0)) for e in vals], dtype=np.int32)
        if has_pooled:
            arrays[f"{q}_pooled"] = np.array([bool(e.get("pooled", False)) for e in vals], dtype=bool)
        if has_src:
            arrays[f"{q}_src"] = src_arr
        if has_nu:
            arrays[f"{q}_nu"] = np.array([float(e.get("nu", 0.0)) for e in vals], dtype=float_dtype)
        if m_idx is not None:
            arrays[f"{q}_merged_from_idx"], arrays[f"{q}_merged_from_val"] = m_idx, m_val
        if has_ref:
            arrays[f"{q}_ref_ll"], arrays[f"{q}_ref_sd"] = _ref_arrays(entries, float_dtype)
        for field in ("means", "vars", "weights"):
            arrays[f"{q}_{field}"], arrays[f"{q}_{field}_off"] = _pack_ragged(entries, field, float_dtype)

    if entries := by_prefix.get("Q5"):
        keys, vals = list(entries.keys()), list(entries.values())
        m_idx, m_val = _pack_merged(entries)
        model_arr, model_vocab = _pack_vocab([e.get("model", "") for e in vals])
        has_ref = any("ref_ll" in e for e in vals)
        header["quantities"]["Q5"] = {
            "type": "vmm", "count": len(keys), "model_vocab": model_vocab,
            "has_merged_from": m_idx is not None, "has_ref": has_ref,
        }
        arrays["Q5_keys"] = np.frombuffer("\n".join(keys).encode("utf-8"), dtype=np.uint8)
        arrays["Q5_n"] = np.array([int(e.get("n", 0)) for e in vals], dtype=np.int32)
        arrays["Q5_n_eff"] = np.array([float(e.get("n_eff", 0.0)) for e in vals], dtype=float_dtype)
        arrays["Q5_k"] = np.array([int(e.get("k", 0)) for e in vals], dtype=np.int32)
        arrays["Q5_tail"] = np.array([float(e.get("tail", 0.0)) for e in vals], dtype=float_dtype)
        arrays["Q5_model"] = model_arr
        if m_idx is not None:
            arrays["Q5_merged_from_idx"], arrays["Q5_merged_from_val"] = m_idx, m_val
        if has_ref:
            arrays["Q5_ref_ll"], arrays["Q5_ref_sd"] = _ref_arrays(entries, float_dtype)
        for field in ("means", "kappas", "weights"):
            arrays[f"Q5_{field}"], arrays[f"Q5_{field}_off"] = _pack_ragged(entries, field, float_dtype)

    if entries := by_prefix.get("Q6"):
        keys, vals = list(entries.keys()), list(entries.values())
        model_vocab, model_map = ["__NONE__"], {"__NONE__": 0}
        model_arr = np.zeros(len(keys), dtype=np.uint8)
        for i, e in enumerate(vals):
            if "model" in e:
                m_val = e["model"]
                if m_val not in model_map:
                    model_map[m_val] = len(model_vocab)
                    model_vocab.append(m_val)
                model_arr[i] = model_map[m_val]
        header["quantities"]["Q6"] = {"type": "halfgauss", "count": len(keys), "model_vocab": model_vocab}
        arrays["Q6_keys"] = np.frombuffer("\n".join(keys).encode("utf-8"), dtype=np.uint8)
        arrays["Q6_n"] = np.array([int(e.get("n", 0)) for e in vals], dtype=np.int32)
        arrays["Q6_k"] = np.array([int(e.get("k", 1)) for e in vals], dtype=np.int32)
        arrays["Q6_mu"] = np.array([float(e.get("mu", 0.0)) for e in vals], dtype=float_dtype)
        arrays["Q6_sigma"] = np.array([float(e.get("sigma", 0.0)) for e in vals], dtype=float_dtype)
        arrays["Q6_model"] = model_arr

    if entries := by_prefix.get("QO"):
        keys, vals = list(entries.keys()), list(entries.values())
        means_flat, means_shapes, covs_flat, covs_shapes, extra_list = [], [], [], [], []
        for e in vals:
            m = e.get("means", [])
            k_val, d_val = len(m), (len(m[0]) if m else 0)
            means_shapes.append((k_val, d_val))
            for row in m:
                means_flat.extend(row)
            covs_shapes.append((k_val, d_val, d_val))
            for mat in e.get("covs", []):
                for row in mat:
                    covs_flat.extend(row)
            extra_list.append({k: v for k, v in e.items() if k not in ("type", "n", "weights", "means", "covs")})
        header["quantities"]["QO"] = {
            "type": "mvgmm", "count": len(keys), "extra_list": extra_list,
            "means_shapes": means_shapes, "covs_shapes": covs_shapes,
        }
        arrays["QO_keys"] = np.frombuffer("\n".join(keys).encode("utf-8"), dtype=np.uint8)
        arrays["QO_n"] = np.array([int(e.get("n", 0)) for e in vals], dtype=np.int32)
        arrays["QO_weights"], arrays["QO_weights_off"] = _pack_ragged(entries, "weights", float_dtype)
        arrays["QO_means_flat"] = np.array(means_flat, dtype=float_dtype)
        arrays["QO_covs_flat"] = np.array(covs_flat, dtype=float_dtype)

    large_quantities = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO"}
    header["small_quantities"] = {p: e for p, e in by_prefix.items() if p not in large_quantities}
    arrays["header_json"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
    return arrays


class QuantityRows:
    """Row decoder for one quantity of a container: ``rows(i)`` is the entry dict of key ``rows.keys[i]``.

    One decoder serves both reads: decode_lossless calls it for every row (columns converted to lists once), and
    stats.load_db calls it for the rows a solve looks up (columns kept as arrays). A plain class, so it pickles."""

    def __init__(self, arrays: Dict[str, Any], q: str, info: Dict[str, Any], as_lists: bool) -> None:
        self.q, self.info = q, info
        self.keys = bytes(arrays[f"{q}_keys"]).decode("utf-8").split("\n")
        self.n = np.asarray(arrays[f"{q}_n"])
        self.merged = _merged_map(arrays, q, bool(info.get("has_merged_from")))
        self.ref = (arrays[f"{q}_ref_ll"], arrays[f"{q}_ref_sd"]) if info.get("has_ref") else None
        if q in ("Q1", "Q2", "Q3", "Q4"):
            names = ["n", "n_eff", "k", "means", "means_off", "vars", "vars_off", "weights", "weights_off"]
            names += [f for f, flag in (("pooled", "has_pooled"), ("src", "has_src"), ("nu", "has_nu")) if info.get(flag)]
        elif q == "Q5":
            names = ["n", "n_eff", "k", "tail", "model", "means", "means_off", "kappas", "kappas_off", "weights",
                     "weights_off"]
        elif q == "Q6":
            names = ["n", "k", "mu", "sigma", "model"]
        else:  # QO
            names = ["n", "weights", "weights_off", "means_flat", "covs_flat"]
            shapes = np.asarray(info["means_shapes"], dtype=np.int64).reshape(-1, 2)
            kd = shapes[:, 0] * shapes[:, 1]
            self.shapes = shapes.tolist()
            self.m_off = np.concatenate([[0], np.cumsum(kd)]).tolist()
            self.c_off = np.concatenate([[0], np.cumsum(kd * shapes[:, 1])]).tolist()
            self.extra = info.get("extra_list") or [{"meta": m} for m in info.get("meta_list", [])]
            as_lists = False
        self.col = {f: (arrays[f"{q}_{f}"].tolist() if as_lists else arrays[f"{q}_{f}"]) for f in names}

    def __len__(self) -> int:
        return len(self.keys)

    def __call__(self, i: int) -> Dict[str, Any]:
        c, q, info = self.col, self.q, self.info

        def ragged(f):
            v = c[f][c[f + "_off"][i]: c[f + "_off"][i + 1]]
            return v if isinstance(v, list) else v.tolist()

        if q == "Q6":
            d = {"type": "halfgauss", "n": int(c["n"][i]), "k": int(c["k"][i]),
                 "mu": float(c["mu"][i]), "sigma": float(c["sigma"][i])}
            if c["model"][i] > 0:
                d["model"] = info["model_vocab"][c["model"][i]]
            return d
        if q == "QO":
            (k_val, d_val), m0, c0 = self.shapes[i], self.m_off[i], self.c_off[i]
            d = {"type": "mvgmm", "n": int(c["n"][i]), "weights": ragged("weights"),
                 "means": c["means_flat"][m0: self.m_off[i + 1]].reshape((k_val, d_val)).tolist(),
                 "covs": c["covs_flat"][c0: self.c_off[i + 1]].reshape((k_val, d_val, d_val)).tolist()}
            if i < len(self.extra):
                d.update(self.extra[i])
            return d
        if q == "Q5":
            d = {"type": "vmm", "n": int(c["n"][i]), "means": ragged("means"), "kappas": ragged("kappas"),
                 "weights": ragged("weights"), "k": int(c["k"][i]), "tail": float(c["tail"][i]),
                 "model": info["model_vocab"][c["model"][i]], "n_eff": float(c["n_eff"][i])}
        else:
            d = {"type": info["type"], "n": int(c["n"][i]), "means": ragged("means"), "vars": ragged("vars"),
                 "weights": ragged("weights"), "k": int(c["k"][i]), "n_eff": float(c["n_eff"][i])}
            if "pooled" in c:
                d["pooled"] = bool(c["pooled"][i])
            if "src" in c:
                d["src"] = info.get("src_vocab", [])[c["src"][i]]
            if "nu" in c:
                d["nu"] = float(c["nu"][i])
        if i in self.merged:
            d["merged_from"] = self.merged[i]
        if self.ref is not None and not np.isnan(ll := self.ref[0][i]):
            d["ref_ll"], d["ref_sd"] = float(ll), float(self.ref[1][i])
        return d


def quantity_rows(arrays: Dict[str, Any], as_lists: bool = False) -> Tuple[Dict[str, "QuantityRows"], Dict[str, Any]]:
    """({quantity: QuantityRows}, header) of a container's arrays, in the container's quantity order."""
    header = json.loads(bytes(arrays["header_json"]).decode("utf-8"))
    rows = {}
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO"):
        info = header["quantities"].get(q)
        if info and info["count"] > 0:
            rows[q] = QuantityRows(arrays, q, info, as_lists)
    return rows, header


def decode_lossless(arrays: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the exact DB dictionary from arrays dict without pickle."""
    rows, header = quantity_rows(arrays, as_lists=True)
    rebuilt: Dict[str, Any] = {}
    for r in rows.values():
        for i, k_str in enumerate(r.keys):
            rebuilt[k_str] = r(i)
    for entries in header.get("small_quantities", {}).values():
        rebuilt.update(entries)
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


def open_lossless_npz(npz_path: Union[str, Path]) -> Tuple[Dict[str, "QuantityRows"], Dict[str, Any], Dict[str, Any]]:
    """(row decoders, small entries, header meta) of an npz container, without decoding any row (stats.load_db)."""
    with np.load(Path(npz_path), allow_pickle=False) as loaded:
        rows, header = quantity_rows({k: loaded[k] for k in loaded.files})
    small: Dict[str, Any] = {}
    for entries in header.get("small_quantities", {}).values():
        small.update(entries)
    return rows, small, header.get("meta", {})


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
