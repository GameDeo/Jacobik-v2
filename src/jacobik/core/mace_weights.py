"""Pure-Python MACE-POLAR .model reader, downloader, and pre-folded weight cache.

Reads official unmodified MACE-POLAR-1-{S,M,L}.model ZIP archives directly via
``zipfile`` + ``pickle.Unpickler`` into NumPy arrays with ZERO ``torch``, ``mace``,
``e3nn``, ``graph_longrange``, or ``ase`` imports, and caches pre-folded weights
locally under ``~/.cache/jacobik/mace_cache/<sha256>.npz``.
"""
from __future__ import annotations

from collections import OrderedDict, namedtuple
import hashlib
import io
import logging
import os
from pathlib import Path
import pickle
import tempfile
import urllib.request
import zipfile
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: the models this engine implements (MACE-POLAR-1-S only), with the SHA-256 of the official release file: a download
#: is accepted only when it matches
_MODEL_FILES = {
    "polar-1-s": ("MACE-POLAR-1-S.model", "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510"),
}
#: builtins a MACE checkpoint's pickle may construct (plain containers and scalars); anything else (eval, exec,
#: getattr, __import__, ...) is refused, so a tampered .model cannot run code while it is read
_SAFE_BUILTINS = frozenset({"set", "frozenset", "list", "dict", "tuple", "int", "float", "complex", "bool", "str",
                            "bytes", "bytearray", "slice", "range"})
_MODEL_BASE_URL = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/"
_ASL_NOTICE = (
    "Downloading MACE-POLAR-1 model weights from ACEsuit/mace-foundations "
    "(Academic Software License / non-commercial; see ACEsuit/mace-foundations LICENSE)."
)

_E3NN_LIN_INST = namedtuple("Instruction", ["i_in", "i_out", "path_shape", "path_weight"])
_E3NN_TP_INST = namedtuple("Instruction", ["i_in1", "i_in2", "i_out", "connection_mode", "has_weight", "path_weight", "path_shape"])

_STORAGE_DTYPES = {
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "ByteStorage": np.uint8,
    "CharStorage": np.int8,
    "BoolStorage": np.bool_,
}


class _StubObj:
    """Generic stub for unpickled torch.nn.Module / mace / e3nn objects."""

    def __init__(self, cls_name: str = "") -> None:
        self._cls_name = cls_name

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    if storage is None:
        return np.empty(size, dtype=np.float32)
    if len(size) == 0:
        return storage[storage_offset : storage_offset + 1].reshape(()).copy()
    itemsize = storage.dtype.itemsize
    arr = np.lib.stride_tricks.as_strided(
        storage[storage_offset:],
        shape=tuple(size),
        strides=tuple(st * itemsize for st in stride),
    )
    return arr.copy()


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


def _make_stub_class(module: str, name: str):
    class _DynamicStub(_StubObj):
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __new__(cls, *args, **kwargs):
            obj = object.__new__(cls)
            _StubObj.__init__(obj, f"{module}.{name}")
            return obj

    _DynamicStub.__name__ = name
    _DynamicStub.__qualname__ = f"{module}.{name}"
    return _DynamicStub


class _MaceModelUnpickler(pickle.Unpickler):
    def __init__(self, file, zf: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(file)
        self._zf = zf
        self._prefix = prefix
        self._storages: Dict[str, np.ndarray] = {}
        self._cls_cache: Dict[Tuple[str, str], type] = {}

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            storage_type, key, _loc, numel = pid[1], str(pid[2]), pid[3], int(pid[4])
            if key not in self._storages:
                dt = _STORAGE_DTYPES.get(getattr(storage_type, "__name__", ""), np.float32)
                raw = self._zf.read(f"{self._prefix}data/{key}")
                self._storages[key] = np.frombuffer(raw, dtype=dt, count=numel).copy()
            return self._storages[key]
        return None

    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return type(name, (), {"__name__": name})
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "e3nn.o3._linear" and name == "Instruction":
            return _E3NN_LIN_INST
        if module == "e3nn.o3._tensor_product._instruction" and name == "Instruction":
            return _E3NN_TP_INST
        if module == "builtins":
            if name not in _SAFE_BUILTINS:
                raise pickle.UnpicklingError(f"refusing builtins.{name} in a MACE checkpoint")
            import builtins
            return getattr(builtins, name)
        key = (module, name)
        if key not in self._cls_cache:
            self._cls_cache[key] = _make_stub_class(module, name)
        return self._cls_cache[key]


def _load_model_stub(path: Path):
    """Unpickle a PyTorch .model ZIP archive into Python stubs + NumPy arrays."""
    with zipfile.ZipFile(path, "r") as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl_names:
            raise RuntimeError(f"Invalid MACE .model archive (no data.pkl): {path}")
        pkl_name = pkl_names[0]
        prefix = pkl_name[: -len("data.pkl")]
        with zf.open(pkl_name) as f:
            return _MaceModelUnpickler(io.BytesIO(f.read()), zf, prefix).load()


def resolve_model_path(model: str = "polar-1-s", model_path: Optional[str | Path] = None) -> Path:
    """Locate a local MACE-POLAR .model file or download it into ~/.cache/mace/."""
    if model_path is not None:
        p = Path(model_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"MACE checkpoint does not exist: {p}")
        return p
    env_path = os.environ.get("JACOBIK_MACE_MODEL_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"JACOBIK_MACE_MODEL_PATH does not exist: {p}")
        return p
    if model not in _MODEL_FILES:
        raise ValueError(f"Unknown MACE-POLAR model {model!r}; expected one of {sorted(_MODEL_FILES)}")
    fname, sha = _MODEL_FILES[model]
    cache_dir = Path(os.environ.get("MACE_CACHE_DIR", Path.home() / ".cache" / "mace"))
    local = cache_dir / fname
    if local.is_file():
        return local
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_MODEL_BASE_URL}{fname}"
    log.info("%s (%s -> %s)", _ASL_NOTICE, url, local)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=f".{fname}.", suffix=".part")   # unique per process (lanes)
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        got = _sha256(Path(tmp))
        if got != sha:
            raise RuntimeError(f"downloaded {fname} has SHA-256 {got}, expected {sha}; not used")
        os.replace(tmp, local)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return local


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _extract_e3nn_linear(lin) -> Dict[str, np.ndarray]:
    """Extract e3nn.o3.Linear block weights with path_weight folded."""
    w = np.asarray(lin._parameters["weight"], dtype=np.float64)
    if w.size == 0:
        return {}
    b_raw = lin._parameters.get("bias")
    b_arr = np.asarray(b_raw, dtype=np.float64).reshape(-1) if b_raw is not None else None
    blocks: Dict[str, np.ndarray] = {}
    offset = 0
    b_off = 0
    for inst in lin.instructions:
        if getattr(inst, "i_in", 0) == -1:
            if b_arr is not None and b_arr.size > 0:
                sz = int(inst.path_shape[0])
                blocks[f"bias_{inst.i_out}"] = b_arr[b_off : b_off + sz] * float(inst.path_weight)
                b_off += sz
            continue
        if not getattr(inst, "has_weight", True):
            continue
        if len(inst.path_shape) == 2:
            m_in, m_out = inst.path_shape
        else:
            m_in, m_out = 1, int(inst.path_shape[0])
        key = f"{inst.i_in}_{inst.i_out}" if hasattr(inst, "i_in") else f"{inst.i_in1}_{inst.i_in2}_{inst.i_out}"
        n = m_in * m_out
        blocks[key] = w[offset : offset + n].reshape(m_in, m_out) * float(inst.path_weight)
        offset += n
    return blocks


def _extract_from_model(path: Path) -> Dict[str, np.ndarray]:
    """Extract all PolarMACE weights and structural constants into float64 NumPy arrays."""
    m = _load_model_stub(path)
    mods = m._modules
    bufs = m._buffers
    d: Dict[str, np.ndarray] = {}

    inter_mods = mods["interactions"]._modules
    n_inter = len(inter_mods)
    d["num_interactions"] = np.array(n_inter, dtype=np.int64)
    d["r_max"] = np.array(float(bufs["r_max"]), dtype=np.float64)
    d["atomic_numbers"] = np.asarray(bufs["atomic_numbers"], dtype=np.int64)
    d["atomic_energies"] = np.asarray(mods["atomic_energies_fn"]._buffers["atomic_energies"], dtype=np.float64)

    ss = mods.get("scale_shift")
    d["energy_scale"] = np.array(float(np.asarray(ss._buffers["scale"]).item()) if ss is not None and "scale" in ss._buffers else 1.0, dtype=np.float64)
    d["energy_shift"] = np.array(float(np.asarray(ss._buffers["shift"]).item()) if ss is not None and "shift" in ss._buffers else 0.0, dtype=np.float64)

    rad = mods["radial_embedding"]._modules
    bfn = rad["bessel_fn"]
    d["bessel_weights"] = np.asarray(bfn._buffers["bessel_weights"], dtype=np.float64)
    d["bessel_pref"] = np.array(float(np.asarray(bfn._buffers["prefactor"]).item()), dtype=np.float64)
    d["cutoff_p"] = np.array(int(np.asarray(rad["cutoff_fn"]._buffers["p"]).item()), dtype=np.int64)

    ne_lin = mods["node_embedding"]._modules["linear"]
    d["node_emb_w"] = np.asarray(ne_lin._parameters["weight"], dtype=np.float64).reshape(len(d["atomic_numbers"]), -1)
    # Also store path_weight if present
    d["node_emb_pw"] = np.array(float(ne_lin.instructions[0].path_weight), dtype=np.float64)

    for k in range(n_inter):
        inter = inter_mods[str(k)]._modules
        p = f"inter{k}_"
        for bname, blk in _extract_e3nn_linear(inter["linear_up"]).items():
            d[f"{p}lin_up_{bname}"] = blk
        lin1 = inter.get("linear_1") or inter.get("linear")
        for bname, blk in _extract_e3nn_linear(lin1).items():
            d[f"{p}lin1_{bname}"] = blk
        if "linear_2" in inter:
            for bname, blk in _extract_e3nn_linear(inter["linear_2"]).items():
                d[f"{p}lin2_{bname}"] = blk
        for bname, blk in _extract_e3nn_linear(inter["skip_tp"]).items():
            d[f"{p}skip_{bname}"] = blk
        if "source_embedding" in inter:
            for bname, blk in _extract_e3nn_linear(inter["source_embedding"]).items():
                d[f"{p}src_emb_{bname}"] = blk
        if "target_embedding" in inter:
            for bname, blk in _extract_e3nn_linear(inter["target_embedding"]).items():
                d[f"{p}tgt_emb_{bname}"] = blk
        if "linear_res" in inter:
            for bname, blk in _extract_e3nn_linear(inter["linear_res"]).items():
                d[f"{p}lin_res_{bname}"] = blk
        im = inter_mods[str(k)]
        for ab in ("alpha", "beta"):
            val = im._parameters.get(ab, im._buffers.get(ab, getattr(im, ab, None)))
            if val is not None:
                d[f"{p}{ab}"] = np.array(float(np.asarray(val).item()), dtype=np.float64)
        eqnl = inter["equivariant_nonlin"]._modules
        d[f"{p}cst_silu"] = np.array(float(eqnl["act_scalars"]._modules["acts"]._modules["0"].cst), dtype=np.float64)
        d[f"{p}cst_sigmoid"] = np.array(float(eqnl["act_gates"]._modules["acts"]._modules["0"].cst), dtype=np.float64)
        d[f"{p}tp_path_weights"] = np.array(
            [float(inst.path_weight) for inst in inter["conv_tp"].instructions], dtype=np.float64
        )
        d[f"{p}avg_num_neighbors"] = np.array(float(inter_mods[str(k)].avg_num_neighbors), dtype=np.float64)

        mlp_net = inter["conv_tp_weights"]._modules["net"]._modules
        for idx_s, layer in mlp_net.items():
            if "weight" in layer._parameters:
                d[f"{p}mlp_{idx_s}_w"] = np.asarray(layer._parameters["weight"], dtype=np.float64)
                if layer._parameters.get("bias") is not None:
                    d[f"{p}mlp_{idx_s}_b"] = np.asarray(layer._parameters["bias"], dtype=np.float64)
        if "density_fn" in inter:
            den_net = inter["density_fn"]._modules["net"]._modules
            for idx_s, layer in den_net.items():
                if "weight" in layer._parameters:
                    d[f"{p}den_{idx_s}_w"] = np.asarray(layer._parameters["weight"], dtype=np.float64)
                    if layer._parameters.get("bias") is not None:
                        d[f"{p}den_{idx_s}_b"] = np.asarray(layer._parameters["bias"], dtype=np.float64)
        for sc_name in ("alpha_sc", "beta_sc"):
            if sc_name in inter_mods[str(k)]._buffers:
                d[f"{p}{sc_name}"] = np.asarray(inter_mods[str(k)]._buffers[sc_name], dtype=np.float64)
            elif sc_name in inter_mods[str(k)]._parameters:
                d[f"{p}{sc_name}"] = np.asarray(inter_mods[str(k)]._parameters[sc_name], dtype=np.float64)

    prod_mods = mods["products"]._modules
    for k in range(n_inter):
        pr = prod_mods[str(k)]._modules
        p = f"prod{k}_"
        for bname, blk in _extract_e3nn_linear(pr["linear"]).items():
            d[f"{p}lin_{bname}"] = blk
        c_mods = pr["symmetric_contractions"]._modules["contractions"]._modules
        for idx_s, c in c_mods.items():
            cp = f"{p}c{idx_s}_"
            d[f"{cp}weights_max"] = np.asarray(c._parameters["weights_max"], dtype=np.float64)
            for bname, key in (("U_matrix_1", "U1"), ("U_matrix_2", "U2"), ("U_matrix_3", "U3")):
                if bname in c._buffers and c._buffers[bname] is not None:
                    d[f"{cp}{key}"] = np.asarray(c._buffers[bname], dtype=np.float64)
            for idx_w in ("0", "1"):
                if "weights" in c._modules and idx_w in c._modules["weights"]._parameters:
                    d[f"{cp}weights_{idx_w}"] = np.asarray(c._modules["weights"]._parameters[idx_w], dtype=np.float64)
                elif f"weights.{idx_w}" in c._parameters:
                    d[f"{cp}weights_{idx_w}"] = np.asarray(c._parameters[f"weights.{idx_w}"], dtype=np.float64)

    ro_mods = mods["readouts"]._modules
    for k in range(len(ro_mods)):
        ro = ro_mods[str(k)]._modules
        p = f"ro{k}_"
        for sub_name, sub in ro.items():
            if hasattr(sub, "instructions") and sub.instructions and hasattr(sub.instructions[0], "i_in"):
                for bname, blk in _extract_e3nn_linear(sub).items():
                    d[f"{p}{sub_name}_{bname}"] = blk
            elif sub_name == "non_linearity" and "acts" in sub._modules:
                d[f"{p}cst"] = np.array(float(sub._modules["acts"]._modules["0"].cst), dtype=np.float64)
            else:
                for pname, pval in sub._parameters.items():
                    if pval is not None:
                        d[f"{p}{sub_name}_{pname}"] = np.asarray(pval, dtype=np.float64)

    # Electrostatics
    d["es_smearing_width"] = np.array(float(m.atomic_multipoles_smearing_width), dtype=np.float64)
    d["es_field_widths"] = np.array(m.field_feature_widths, dtype=np.float64)
    d["es_field_norms"] = np.array(m._field_feature_norms, dtype=np.float64)
    d["es_field_norm_factor"] = np.array(float(m.field_norm_factor), dtype=np.float64)

    rf = mods["electric_potential_descriptor"]._modules["realspace_features"]
    d["es_offset"] = np.array(float(rf.offset), dtype=np.float64)
    d["es_total_width_factors"] = np.asarray(rf._buffers["total_width_factors"], dtype=np.float64)
    d["es_l0_factors"] = np.asarray(rf._buffers["l0_factors"], dtype=np.float64)
    d["es_l1_factors"] = np.asarray(rf._buffers["l1_factors"], dtype=np.float64)
    si = rf._modules["self_interaction"]
    d["es_self_overlap_constants"] = np.asarray(si._buffers["overlap_constants"], dtype=np.float64)
    d["es_self_select_indices"] = np.asarray(si._buffers["select_indices"], dtype=np.int64)

    ce_re = mods["coulomb_energy"]._modules["realspace_energy"]
    ce_si = ce_re._modules["self_interaction"]
    d["ce_offset"] = np.array(float(ce_re.offset), dtype=np.float64)
    d["ce_self_overlap_constants"] = np.asarray(ce_si._buffers["overlap_constants"], dtype=np.float64)

    # Layer feature mixer & Fukui source map
    for idx_s, lin in mods["layer_feature_mixer"]._modules["linears"]._modules.items():
        for bname, blk in _extract_e3nn_linear(lin).items():
            d[f"mix_{idx_s}_{bname}"] = blk
    fk = mods["fukui_source_map"]._modules
    for bname, blk in _extract_e3nn_linear(fk["linear_1"]).items():
        d[f"fk_lin1_{bname}"] = blk
    for bname, blk in _extract_e3nn_linear(fk["linear_mid"]).items():
        d[f"fk_linm_{bname}"] = blk
    for bname, blk in _extract_e3nn_linear(fk["linear_2"]).items():
        d[f"fk_lin2_{bname}"] = blk
    d["fk_cst_silu"] = np.array(float(fk["non_linearity"]._modules["acts"]._modules["0"].cst), dtype=np.float64)

    lr_mods = mods["lr_source_maps"]._modules
    for k in range(len(lr_mods)):
        lin = lr_mods[str(k)]._modules["linear"]
        for bname, blk in _extract_e3nn_linear(lin).items():
            d[f"lr{k}_lin_{bname}"] = blk
        d[f"lr{k}_mask"] = np.asarray(lin._buffers["output_mask"], dtype=np.float64)

    fdc_mods = mods["field_dependent_charges_maps"]._modules
    for k in range(len(fdc_mods)):
        fdc = fdc_mods[str(k)]._modules
        p = f"fdc{k}_"
        for bname, blk in _extract_e3nn_linear(fdc["source_embedding"]).items():
            d[f"{p}src_emb_{bname}"] = blk
        pe = fdc["potential_embedding"]._modules
        for bname, blk in _extract_e3nn_linear(pe["potential_linear"]).items():
            d[f"{p}pe_pot_{bname}"] = blk
        for bname, blk in _extract_e3nn_linear(pe["node_feats_linear"]).items():
            d[f"{p}pe_nf_{bname}"] = blk
        for bname, blk in _extract_e3nn_linear(pe["charge_embedding"]).items():
            d[f"{p}pe_chg_{bname}"] = blk
        d[f"{p}dot_w"] = np.asarray(fdc["dot_products"]._parameters["weight"], dtype=np.float64)
        d[f"{p}dot_pw"] = np.array(float(fdc["dot_products"]._path_meta[0][8]), dtype=np.float64)
        if len(fdc["dot_products"]._path_meta) > 1:
            d[f"{p}dot_pw1"] = np.array(float(fdc["dot_products"]._path_meta[1][8]), dtype=np.float64)
        d[f"{p}tp_out_w"] = np.asarray(fdc["tp_out"]._parameters["weight"], dtype=np.float64)
        d[f"{p}tp_out_pw"] = np.array(float(fdc["tp_out"]._path_meta[0][8]), dtype=np.float64)
        if len(fdc["tp_out"]._path_meta) > 1:
            d[f"{p}tp_out_pw1"] = np.array(float(fdc["tp_out"]._path_meta[1][8]), dtype=np.float64)
        nl_net = fdc["nonlinearity"]._modules["net"]._modules
        for idx_s, layer in nl_net.items():
            if "weight" in layer._parameters:
                d[f"{p}nl_{idx_s}_w"] = np.asarray(layer._parameters["weight"], dtype=np.float64)
                if layer._parameters.get("bias") is not None:
                    d[f"{p}nl_{idx_s}_b"] = np.asarray(layer._parameters["bias"], dtype=np.float64)
        ro = fdc["readout"]._modules
        for bname, blk in _extract_e3nn_linear(ro["linear_1"]).items():
            d[f"{p}ro_lin1_{bname}"] = blk
        for bname, blk in _extract_e3nn_linear(ro["linear_mid"]).items():
            d[f"{p}ro_linm_{bname}"] = blk
        for bname, blk in _extract_e3nn_linear(ro["linear_2"]).items():
            d[f"{p}ro_lin2_{bname}"] = blk
        d[f"{p}ro_cst"] = np.array(
            float(ro["equivariant_nonlin"]._modules["act_scalars"]._modules["acts"]._modules["0"].cst),
            dtype=np.float64,
        )

    lee = mods["local_electron_energy"]._modules
    for bname, blk in _extract_e3nn_linear(lee["linear_up_q"]).items():
        d[f"lee_q_{bname}"] = blk
    for bname, blk in _extract_e3nn_linear(lee["linear_up_v"]).items():
        d[f"lee_v_{bname}"] = blk
    d["lee_dotq_w"] = np.asarray(lee["dot_products_q"]._parameters["weight"], dtype=np.float64)
    d["lee_dotq_pw"] = np.array(float(lee["dot_products_q"]._path_meta[0][8]), dtype=np.float64)
    if len(lee["dot_products_q"]._path_meta) > 1:
        d["lee_dotq_pw1"] = np.array(float(lee["dot_products_q"]._path_meta[1][8]), dtype=np.float64)
    d["lee_dotv_w"] = np.asarray(lee["dot_products_v"]._parameters["weight"], dtype=np.float64)
    d["lee_dotv_pw"] = np.array(float(lee["dot_products_v"]._path_meta[0][8]), dtype=np.float64)
    if len(lee["dot_products_v"]._path_meta) > 1:
        d["lee_dotv_pw1"] = np.array(float(lee["dot_products_v"]._path_meta[1][8]), dtype=np.float64)
    for idx_s, layer in lee["mlp"]._modules["net"]._modules.items():
        if "weight" in layer._parameters:
            d[f"lee_mlp_{idx_s}_w"] = np.asarray(layer._parameters["weight"], dtype=np.float64)
            if layer._parameters.get("bias") is not None:
                d[f"lee_mlp_{idx_s}_b"] = np.asarray(layer._parameters["bias"], dtype=np.float64)

    return d


def load_weights(model: str = "polar-1-s", model_path: Optional[str | Path] = None) -> Dict[str, np.ndarray]:
    """Load PolarMACE float64 weights from ~/.cache/jacobik/mace_cache/<sha256>.npz (zero torch)."""
    p = resolve_model_path(model, model_path)
    digest = _sha256(p)[:16]
    cache_dir = Path(os.environ.get("JACOBIK_MACE_CACHE_DIR", Path.home() / ".cache" / "jacobik" / "mace_cache"))
    cached = cache_dir / f"{digest}_v6.npz"
    if cached.is_file():
        with np.load(cached) as npz:
            return {k: npz[k] for k in npz.files}
    weights = _extract_from_model(p)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=f".{cached.stem}.", suffix=".npz")   # unique per process
    os.close(fd)
    try:
        np.savez(tmp, **weights)
        os.replace(tmp, cached)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return weights
