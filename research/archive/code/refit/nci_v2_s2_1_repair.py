"""S2.1 post-run repair tooling for the nci_v2 scaled_v1 control-arm pools.

Phase 1 (this revision): derive frame_n_heavy.json from the frozen frame.json by adding
an integer n_heavy to every main_allocation.splits entry, computed from the reference
structure the orchestrator itself loads (research.archive.code.refit.nci_v2_pilot.load_mol), and cross-check
n_heavy against the reference atoms stored in every successful pool.

The frozen data directory is never written to; all outputs go under <R>/repair_s2_1/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "research" / "refit", REPO_ROOT / "research/bench/tools/jdev/budget", REPO_ROOT / "research/bench"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

R_DIR = REPO_ROOT / "data/refit/nci_v2/scaled_v1"
REPAIR_DIR = R_DIR / "repair_s2_1"
FRAME_PATH = R_DIR / "frame.json"
MANIFEST_PATH = R_DIR / "manifest.json"
POOLS_DIR = R_DIR / "pools"
SPLITS = ("surface_fit", "calibration", "motion_validation", "reserve")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def count_heavy_from_reference(code: str, shard: str, dataset: str) -> int:
    from jacobik.inputs import load_mol

    mol = load_mol(code, shard, dataset)
    if mol is None:
        raise RuntimeError(f"load_mol returned None for {code} shard={shard} dataset={dataset}")
    return int(sum(1 for a in mol.atoms if a.symbol != "H"))


def derive_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of frame with n_heavy added to every split entry."""
    import copy

    out = copy.deepcopy(frame)
    cache: Dict[tuple, int] = {}
    for split in SPLITS:
        for entry in out["main_allocation"]["splits"][split]:
            if "n_heavy" in entry:
                raise RuntimeError(f"frame already contains n_heavy for {entry.get('representative_code')}")
            code = entry.get("code") or entry["representative_code"]
            shard = entry.get("shard") or entry["representative_shard"]
            dataset = entry.get("dataset") or "unknown"
            key = (code, shard, dataset)
            if key not in cache:
                cache[key] = count_heavy_from_reference(code, shard, dataset)
            entry["n_heavy"] = cache[key]
    return out


def cross_check_pools(derived: Dict[str, Any]) -> Dict[str, Any]:
    """Compare n_heavy with (ref_symbols != 'H').sum() for every successful pool."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    n_by_code: Dict[str, int] = {}
    split_by_code: Dict[str, str] = {}
    for split in SPLITS:
        for e in derived["main_allocation"]["splits"][split]:
            code = e.get("code") or e["representative_code"]
            n_by_code.setdefault(code, e["n_heavy"])
            split_by_code.setdefault(code, split)

    checked = 0
    mismatches: List[Dict[str, Any]] = []
    missing_code: List[str] = []
    per_code_pool_n: Dict[str, set] = defaultdict(set)
    for key, rec in manifest["tasks"].items():
        if rec.get("status") != "success":
            continue
        code, seed = rec["code"], rec["seed"]
        if code not in n_by_code:
            missing_code.append(key)
            continue
        with np.load(POOLS_DIR / f"{code}_{seed}_control.npz", allow_pickle=True) as z:
            ref_sym = z["ref_symbols"]
        n_pool = int(np.sum(ref_sym != "H"))
        per_code_pool_n[code].add(n_pool)
        checked += 1
        if n_pool != n_by_code[code]:
            mismatches.append({"task": key, "n_heavy_frame": n_by_code[code], "n_heavy_pool": n_pool})
    return {
        "successful_pools_checked": checked,
        "mismatches": mismatches,
        "successful_tasks_without_frame_entry": missing_code,
        "codes_with_inconsistent_pool_counts": sorted(c for c, s in per_code_pool_n.items() if len(s) != 1),
        "passed": (not mismatches) and (not missing_code),
    }


def cmd_derive_frame() -> None:
    REPAIR_DIR.mkdir(parents=True, exist_ok=True)
    src_sha = sha256_file(FRAME_PATH)
    frame = json.loads(FRAME_PATH.read_text(encoding="utf-8"))
    derived = derive_frame(frame)

    # Proof 1: stripping n_heavy gives the original canonical form.
    stripped = json.loads(json.dumps(derived))
    for split in SPLITS:
        for e in stripped["main_allocation"]["splits"][split]:
            e.pop("n_heavy", None)
    proof_equal = canonical(stripped) == canonical(frame)

    out_path = REPAIR_DIR / "frame_n_heavy.json"
    out_path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    derived_sha = sha256_file(out_path)

    # Proof 2: cross-check against the pools (every successful pool).
    pool_check = cross_check_pools(derived)

    counts = {s: len(derived["main_allocation"]["splits"][s]) for s in SPLITS}
    nh_hist = {
        s: dict(sorted(Counter(e["n_heavy"] for e in derived["main_allocation"]["splits"][s]).items()))
        for s in SPLITS
    }
    proof = {
        "source_frame": str(FRAME_PATH.relative_to(REPO_ROOT)),
        "source_frame_sha256": src_sha,
        "derived_frame": str(out_path.relative_to(REPO_ROOT)),
        "derived_frame_sha256": derived_sha,
        "entries_per_split": counts,
        "n_heavy_histogram_per_split": nh_hist,
        "canonical_equal_after_stripping_n_heavy": proof_equal,
        "pool_cross_check": pool_check,
    }
    (REPAIR_DIR / "frame_n_heavy_proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in proof.items() if k != "n_heavy_histogram_per_split"}, indent=2))
    if not proof_equal:
        raise SystemExit("STOP: canonical equality after stripping n_heavy FAILED")
    if not pool_check["passed"]:
        raise SystemExit("STOP: n_heavy cross-check against pools FAILED")


AB_DIR_NAME = "ab_gate"
AB_SIDES = ("c8538d7f", "c920d0a8", "repeat_c920d0a8")


def cmd_characterize() -> None:
    """Pairwise characterization of A/B and repeat runs (report only, no gate)."""
    ab = REPAIR_DIR / AB_DIR_NAME
    stored_m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["tasks"]
    frame = json.loads((REPAIR_DIR / "frame_n_heavy.json").read_text(encoding="utf-8"))
    n_heavy = {}
    split_of = {}
    for split in SPLITS:
        for e in frame["main_allocation"]["splits"][split]:
            code = e.get("code") or e["representative_code"]
            n_heavy[code] = e["n_heavy"]
            split_of[code] = split
    sides = {}
    for side in AB_SIDES:
        mp = ab / side / "manifest.json"
        if mp.exists():
            sides[side] = json.loads(mp.read_text(encoding="utf-8"))["tasks"]
    sides["stored"] = {k: v for k, v in stored_m.items()}

    def load_pool(side: str, key: str):
        base = (POOLS_DIR if side == "stored" else ab / side / "pools")
        return np.load(base / f"{key.replace(':', '_')}.npz", allow_pickle=True)

    pairs = [("c8538d7f", "c920d0a8"), ("c920d0a8", "repeat_c920d0a8"), ("c8538d7f", "repeat_c920d0a8"),
             ("stored", "c8538d7f"), ("stored", "c920d0a8")]
    keys = sorted(k for k in sides.get("c8538d7f", {}) if sides["c8538d7f"][k].get("status") == "success")
    rows = []
    for a_name, b_name in pairs:
        if a_name not in sides or b_name not in sides:
            continue
        for key in keys:
            if a_name == "stored" and sides["stored"].get(key, {}).get("status") != "success":
                continue
            if b_name not in ("stored",) and sides[b_name].get(key, {}).get("status") != "success":
                continue
            A = load_pool(a_name, key)
            B = load_pool(b_name, key)
            na, nb = len(A["X"]), len(B["X"])
            row = {
                "pair": f"{a_name} vs {b_name}", "key": key, "split": split_of.get(key.split(":")[0]),
                "n_heavy": n_heavy.get(key.split(":")[0]), "n_confs": [na, nb], "n_equal": na == nb,
                "top1_equal": bool(int(A["score"].argmax()) == int(B["score"].argmax())) if na and nb else None,
            }
            if na == nb and na:
                order_a = np.argsort(-A["score"], kind="stable")
                order_b = np.argsort(-B["score"], kind="stable")
                row["order_equal"] = bool(np.array_equal(order_a, order_b))
                row["max_dX_A"] = float(np.abs(A["X"] - B["X"]).max())
                row["max_dscore"] = float(np.abs(A["score"] - B["score"]).max())
            rows.append(row)
    out = {"pairs": [list(p) for p in pairs], "rows": rows}
    (ab / "characterization.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {ab / 'characterization.json'} with {len(rows)} rows")


# ---------------------------------------------------------------------------
# Phase 4 helpers: overlays (schema s2_1_overlay.1), R3 health metric, offline R2 mapping
# ---------------------------------------------------------------------------
OVERLAY_SCHEMA = "s2_1_overlay.1"
MAPPING_CONVENTION = "X[:, gen_to_ref_idx] is in reference atom order; symbols[gen_to_ref_idx] == ref_symbols"
BOND_FACTOR = 1.25
# Cordero et al. 2008 covalent radii (A), fixed for this repair. Elements outside this table are treated as metals.
HEALTH_RADII = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "As": 1.19, "Se": 1.20, "Br": 1.20, "I": 1.39,
}
EVAL_SPLITS = ("calibration", "motion_validation")
NOT_RETRIED = "not_retried_stopped_by_decision"
RETRY_CODES = ("MIYQIU", "OBABAX", "TATXIW", "FEKKAJ", "QEZVEY", "MIVXAR", "AMUKOJ", "AVUVET", "BAFQEH", "BICHON",
                "BIKQIY", "CAFLII", "COJSIG", "EPUPEL", "GOTKEH", "OMAYEJ", "PURHEP", "UZOREI", "VALJEB", "XUSNIJ", "ZEYQON")
STORED_MAPPING_KEYS = (
    "gen_to_ref_idx", "ref_to_gen_idx", "mapping_ambiguous", "num_automorphisms",
    "coord_intact", "coord_sigma_shift_A", "coord_haptic_slip_A", "coord_haptic_shift_A",
)


def _pdist(A: np.ndarray) -> np.ndarray:
    d = A[:, None, :] - A[None, :, :]
    return np.sqrt((d * d).sum(-1))


def health_metrics(ref_X: np.ndarray, ref_sym: Sequence[str], Xr: np.ndarray) -> Dict[str, np.ndarray]:
    """R3 per-conformer metrics. Xr [P,N,3] is in reference atom order; ref_X [N,3] is the stored reference."""
    ref_sym = [str(s) for s in ref_sym]
    N = len(ref_sym)
    lig = np.array([s in HEALTH_RADII for s in ref_sym])
    rad = np.array([HEALTH_RADII.get(s, np.nan) for s in ref_sym], dtype=float)
    heavy = np.array([(s != "H") and (s in HEALTH_RADII) for s in ref_sym])
    eye = np.eye(N, dtype=bool)
    cut = BOND_FACTOR * (rad[:, None] + rad[None, :])
    pair_ll = lig[:, None] & lig[None, :] & ~eye
    Dref = _pdist(np.asarray(ref_X, dtype=np.float64))
    bond_ref = pair_ll & (Dref < cut)
    far_nb = heavy[:, None] & heavy[None, :] & ~eye & ~bond_ref & (Dref > 2.6)
    P = len(Xr)
    out = {
        "max_bond_dev_A": np.zeros(P, dtype=np.float64),
        "min_heavy_nonbonded_A": np.full(P, np.nan, dtype=np.float64),
        "graph_match": np.zeros(P, dtype=bool),
        "max_nn_dist_A": np.zeros(P, dtype=np.float64),
    }
    for i, Xp in enumerate(np.asarray(Xr, dtype=np.float64)):
        D = _pdist(Xp)
        dev = np.abs(D - Dref)[bond_ref]
        out["max_bond_dev_A"][i] = float(dev.max()) if dev.size else 0.0
        mh = D[far_nb]
        out["min_heavy_nonbonded_A"][i] = float(mh.min()) if mh.size else np.nan
        out["graph_match"][i] = bool(np.array_equal(pair_ll & (D < cut), bond_ref))
        out["max_nn_dist_A"][i] = float((D + np.where(eye, np.inf, 0.0)).min(axis=1).max())
    return out


def offline_mapping(mol: Any, ref_X: np.ndarray, ref_sym: Sequence[str], pool_sym: Sequence[str],
                    X: np.ndarray) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Recompute the atom mapping and coordination integrity as worker_solve_group does.

    Differences from the in-process call: the generated shell comes from the builder (no solve),
    and X_gen is the first stored conformer. Returns (arrays, None) or (None, error)."""
    from jacobik.inputs import build_builder_and_inputs
    from jacobik.benchmark.coordination import coordination_integrity
    from jacobik.builder.atom_map import build_atom_mapping

    try:
        builder, _, err = build_builder_and_inputs(mol)
        if builder is None:
            return None, err or "build_builder_and_inputs returned None"
        gen_mol, _, _ = builder._build_shell(chelate_embed=True, fast_embed=False, debug=False)
        if [a.symbol for a in gen_mol.atoms] != [str(s) for s in pool_sym]:
            return None, "builder shell symbol order differs from pool symbols"
        am = build_atom_mapping(mol, gen_mol=gen_mol, X_ref=ref_X, X_gen=X[0])
        g = np.asarray(am.gen_to_ref_idx, dtype=np.int64)
        ci = coordination_integrity([str(s) for s in ref_sym], ref_X, X[:, g, :])
        return {
            "gen_to_ref_idx": g,
            "ref_to_gen_idx": np.asarray(am.ref_to_gen_idx, dtype=np.int64),
            "mapping_ambiguous": np.bool_(am.is_ambiguous),
            "num_automorphisms": np.int64(am.num_automorphisms),
            "coord_intact": np.asarray(ci["intact"], dtype=bool),
            "coord_sigma_shift_A": np.asarray(ci["sigma_shift_A"], dtype=np.float64),
            "coord_haptic_slip_A": np.asarray(ci["haptic_slip_A"], dtype=np.float64),
            "coord_haptic_shift_A": np.asarray(ci["haptic_shift_A"], dtype=np.float64),
        }, None
    except Exception as exc:  # same policy as the orchestrator: record, do not guess
        return None, f"{type(exc).__name__}: {exc}"


def frame_lookup() -> Dict[str, Dict[str, Any]]:
    frame = json.loads((REPAIR_DIR / "frame_n_heavy.json").read_text(encoding="utf-8"))
    lk: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        for e in frame["main_allocation"]["splits"][split]:
            code = e.get("code") or e["representative_code"]
            rec = {"shard": e["representative_shard"], "dataset": e["dataset"], "stratum": e["stratum"],
                   "n_heavy": int(e["n_heavy"])}
            if code in lk and (lk[code]["shard"], lk[code]["dataset"]) != (rec["shard"], rec["dataset"]):
                raise RuntimeError(f"inconsistent shard/dataset for {code}")
            lk.setdefault(code, rec)
    return lk


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _require_clean_code_tree() -> str:
    dirty = _git("status", "--porcelain", "--", "src", "refit", "research/archive/code/refit", "research/archive/code/interactions", "research/bench/tools", "tests", "tools")
    if dirty:
        raise RuntimeError("code roots are not clean; commit before producing overlays:\n" + dirty)
    return _git("rev-parse", "HEAD")


def _mace_evaluator() -> Tuple[Any, str]:
    import torch
    from jacobik.benchmark.energy import MACEPolarEvaluator

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = MACEPolarEvaluator(
        model="polar-1-s", device=dev, dtype="float64",
        model_path=str(REPO_ROOT / "research/bench/tools/jdev/MACE-POLAR-1-S.model"),
        mode="energy-only", threads=1,
    ).load()
    return ev, dev


def cmd_remap_check(n: int) -> None:
    """Validate the offline R2 mapping path against stored mappings on already-mapped pools."""
    from jacobik.inputs import load_mol

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    lk = frame_lookup()
    ok_keys = sorted(k for k, v in manifest["tasks"].items() if v.get("status") == "success")
    rng = np.random.default_rng(0)
    picks = [ok_keys[i] for i in sorted(rng.choice(len(ok_keys), size=min(n, len(ok_keys)), replace=False))]
    results = []
    for key in picks:
        code, seed = key.split(":")[0], int(key.split(":")[1])
        with np.load(POOLS_DIR / f"{code}_{seed}_control.npz", allow_pickle=True) as z:
            if "gen_to_ref_idx" not in z.files:
                continue
            X = np.asarray(z["X"], dtype=np.float64)
            pool_sym = [str(s) for s in z["symbols"]]
            ref_sym = [str(s) for s in z["ref_symbols"]]
            ref_X = np.asarray(z["ref_X"], dtype=np.float64)
            stored_g = np.asarray(z["gen_to_ref_idx"], dtype=np.int64)
            stored_ci = np.asarray(z["coord_intact"], dtype=bool) if "coord_intact" in z.files else None
        mol = load_mol(code, lk[code]["shard"], lk[code]["dataset"])
        res, err = offline_mapping(mol, ref_X, ref_sym, pool_sym, X) if mol is not None else (None, "no mol")
        row = {"key": key, "error": err}
        if res is not None:
            row["gen_to_ref_identical"] = bool(np.array_equal(res["gen_to_ref_idx"], stored_g))
            row["coord_intact_identical"] = bool(stored_ci is not None and np.array_equal(res["coord_intact"], stored_ci))
        results.append(row)
    print(json.dumps({"checked": len(results), "rows": results}, indent=2))


def cmd_overlay(codes: Optional[List[str]], skip_mace: bool) -> None:
    from jacobik.inputs import load_mol
    from research.archive.code.refit.nci_v2_pilot import evaluate_mace_energies

    repair_commit = _require_clean_code_tree()
    ov_dir = REPAIR_DIR / "overlay"
    ov_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tasks = {k: v for k, v in manifest["tasks"].items() if v.get("status") == "success"}
    if codes:
        tasks = {k: v for k, v in tasks.items() if v["code"] in set(codes)}
    lk = frame_lookup()
    ref_cache = json.loads((R_DIR / "ref_mace_cache.json").read_text(encoding="utf-8"))
    evaluator, dev = (None, None) if skip_mace else _mace_evaluator()

    pools_out: Dict[str, Any] = {}
    unmappable: Dict[str, str] = {}
    r1_list: List[Dict[str, Any]] = []
    r2_ok: List[str] = []
    health_summary: Dict[str, Any] = {"pools": 0, "graph_match_false_pools": [],
                                      "min_heavy_nonbonded_lt_1_5": [], "by_split": {}}
    mol_cache: Dict[str, Any] = {}

    for key in sorted(tasks):
        rec = tasks[key]
        code, seed, split = rec["code"], int(rec["seed"]), rec["split"]
        npz_path = POOLS_DIR / f"{code}_{seed}_control.npz"
        json_path = POOLS_DIR / f"{code}_{seed}_control.json"
        with np.load(npz_path, allow_pickle=True) as z:
            zf = set(z.files)
            X = np.asarray(z["X"], dtype=np.float64)
            pool_sym = [str(s) for s in z["symbols"]]
            ref_sym = [str(s) for s in z["ref_symbols"]]
            ref_X = np.asarray(z["ref_X"], dtype=np.float64)
            charge, mult = int(z["charge"]), int(z["mult"])
            model_hash = str(z["mace_model_sha256"])
            stored_map = {n: np.asarray(z[n]) for n in STORED_MAPPING_KEYS if n in zf}
            had_mapping_error = "atom_mapping_error" in zf
        npz_sha, json_sha = sha256_file(npz_path), sha256_file(json_path)
        repairs: List[str] = []
        arrays: Dict[str, Any] = {}
        g2r: Optional[np.ndarray] = None

        if "gen_to_ref_idx" in stored_map:
            g2r = stored_map["gen_to_ref_idx"].astype(np.int64)
        else:
            if code not in mol_cache:
                entry = lk[code]
                mol_cache[code] = load_mol(code, entry["shard"], entry["dataset"])
            mol = mol_cache[code]
            res, err = (None, "load_mol returned None") if mol is None else \
                offline_mapping(mol, ref_X, ref_sym, pool_sym, X)
            if res is None:
                unmappable[key] = err or "unknown"
            else:
                arrays.update(res)
                g2r = res["gen_to_ref_idx"]
                repairs.append("R2_mapping")
                r2_ok.append(key)
        if g2r is not None and not np.array_equal(np.asarray(pool_sym)[g2r], np.asarray(ref_sym)):
            raise RuntimeError(f"{key}: symbols[gen_to_ref_idx] != ref_symbols")

        if split in EVAL_SPLITS and had_mapping_error:
            # R1: the stored MACE energies used ref symbols with gen-order X. Recompute as HEAD does.
            if evaluator is None:
                raise RuntimeError(f"{key}: R1 requires MACE but --skip-mace was given")
            if g2r is not None:
                positions, syms = X[:, g2r, :], list(ref_sym)
            else:
                positions, syms = X, list(pool_sym)
            energies = np.asarray(evaluate_mace_energies(evaluator, syms, positions, charge, mult)[0], dtype=np.float64)
            cache_key = f"{code}:{model_hash}:{charge}:{mult}"
            if cache_key not in ref_cache:
                raise RuntimeError(f"no ref_mace_cache entry for {cache_key}")
            ref_kj = float(ref_cache[cache_key])
            top1 = int(rec["top1_idx"])
            floor = int(np.argmin(energies))
            arrays.update({
                "energies_kj": energies, "ref_energy_kj": np.float64(ref_kj),
                "top1_idx": np.int64(top1), "floor_idx": np.int64(floor),
                "top1_kj": np.float64(energies[top1]), "floor_kj": np.float64(energies[floor]),
                "top1_dE_kj": np.float64(energies[top1] - ref_kj),
                "floor_dE_kj": np.float64(energies[floor] - ref_kj),
            })
            repairs.append("R1_mace")
            r1_list.append({"key": key, "split": split, "stratum": rec["stratum"],
                            "mapped_for_mace": g2r is not None,
                            "old_floor_dE_kj": rec.get("floor_dE_kj"),
                            "new_floor_dE_kj": float(energies[floor] - ref_kj),
                            "new_top1_dE_kj": float(energies[top1] - ref_kj)})

        if g2r is not None:
            h = health_metrics(ref_X, ref_sym, X[:, g2r, :])
            arrays.update(h)
            arrays["health_radii_json"] = np.array(json.dumps(
                {"radii_A": HEALTH_RADII, "bond_factor": BOND_FACTOR}, sort_keys=True))
            repairs.append("R3_health")
            sb = health_summary["by_split"].setdefault(split, {"pools": 0, "pools_with_graph_mismatch": 0})
            sb["pools"] += 1
            health_summary["pools"] += 1
            n_false = int((~h["graph_match"]).sum())
            if n_false:
                sb["pools_with_graph_mismatch"] += 1
                health_summary["graph_match_false_pools"].append({"key": key, "n_false": n_false, "P": len(X)})
            finite = h["min_heavy_nonbonded_A"][np.isfinite(h["min_heavy_nonbonded_A"])]
            if finite.size and float(finite.min()) < 1.5:
                health_summary["min_heavy_nonbonded_lt_1_5"].append({"key": key, "min_A": float(finite.min())})

        ov_path = ov_dir / f"{code}_{seed}_control.npz"
        payload: Dict[str, Any] = {
            "schema_version": np.array(OVERLAY_SCHEMA),
            "source_pool_sha256": np.array(npz_sha),
            "source_summary_sha256": np.array(json_sha),
            "repair_commit": np.array(repair_commit),
            "repairs": np.array(json.dumps(repairs)),
        }
        payload.update(arrays)
        np.savez(ov_path, **payload)
        pools_out[key] = {
            "split": split, "stratum": rec["stratum"], "repairs": repairs,
            "overlay_path": str(ov_path.relative_to(R_DIR)), "overlay_sha256": sha256_file(ov_path),
            "source_pool_sha256": npz_sha, "source_summary_sha256": json_sha,
        }

    retries = _collect_retries(REPAIR_DIR / "retry_pools_v2")
    pre_fix = _collect_retries(REPAIR_DIR / "retry_pools")
    (REPAIR_DIR / "retry_outcomes_pre_fix.json").write_text(json.dumps(pre_fix, indent=2) + "\n", encoding="utf-8")
    slots: Dict[str, Any] = {}
    for failed, info in manifest.get("replaced_failures", {}).items():
        mine = {k: v for k, v in retries.items() if k.split(":")[0] == failed}
        restored = len(mine) == 3 and all(v["status"] == "success" for v in mine.values())
        attempted = any(v["status"] != NOT_RETRIED for v in mine.values())
        state = "restored" if restored else ("still_failed" if attempted else "not_retried")
        slots[failed] = {
            "stratum": info.get("stratum"), "state": state,
            "replacement_code": info.get("replacement_code"),
            "replacement_state": "unused_after_retry" if restored else "in_use",
        }

    repair_manifest = {
        "schema_version": OVERLAY_SCHEMA,
        "repair_commit": repair_commit,
        "mapping_convention": MAPPING_CONVENTION,
        "frame": {
            "frozen_path": str(FRAME_PATH.relative_to(REPO_ROOT)),
            "frozen_sha256": sha256_file(FRAME_PATH),
            "derived_path": str((REPAIR_DIR / "frame_n_heavy.json").relative_to(REPO_ROOT)),
            "derived_sha256": sha256_file(REPAIR_DIR / "frame_n_heavy.json"),
            "derivation": "added integer n_heavy only; canonical equality proven",
        },
        "pools": pools_out,
        "unmappable": unmappable,
        "retries": retries,
        "slots": slots,
        "unresolved": [f"{k}: no mapping after R2 ({v})" for k, v in sorted(unmappable.items())],
    }
    (REPAIR_DIR / "repair_manifest.json").write_text(json.dumps(repair_manifest, indent=2) + "\n", encoding="utf-8")
    summary = {"repair_commit": repair_commit, "mace_device": dev, "pools_overlaid": len(pools_out),
               "r1_pools": len(r1_list), "r2_remapped": len(r2_ok), "unmappable": unmappable,
               "r1": r1_list, "health": health_summary}
    (REPAIR_DIR / "overlay_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("r1", "health")}, indent=2))
    print("health:", {k: (len(v) if isinstance(v, list) else v) for k, v in health_summary.items()})


def _collect_retries(rdir: Path) -> Dict[str, Any]:
    """Per-task retry outcomes for one retry directory (manifest.json if present, else per-task summaries)."""
    out: Dict[str, Any] = {}
    mp = rdir / "manifest.json"
    if mp.exists():
        tasks = json.loads(mp.read_text(encoding="utf-8")).get("tasks", {})
        recs = {k: v for k, v in tasks.items() if k.split(":")[0] in RETRY_CODES}
    else:
        recs = {}
        for code in RETRY_CODES:
            for seed in (11, 12, 13):
                sp = rdir / "pools" / f"{code}_{seed}_control.json"
                key = f"{code}:{seed}:control"
                recs[key] = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {"status": NOT_RETRIED}
    for key in sorted({f"{c}:{s}:control" for c in RETRY_CODES for s in (11, 12, 13)}):
        rec = recs.get(key, {"status": NOT_RETRIED})
        code, seed = key.split(":")[0], key.split(":")[1]
        pool_path = rdir / "pools" / f"{code}_{seed}_control.npz"
        ok = rec.get("status") == "success" and pool_path.exists()
        prov = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {}
        out[key] = {
            "split": rec.get("split"), "stratum": rec.get("stratum"), "status": rec.get("status", NOT_RETRIED),
            "error": rec.get("error") or None,
            "max_rows": rec.get("max_rows"), "wall_s": rec.get("wall_s"), "peak_gpu_mb": rec.get("peak_gpu_mb"),
            "pool_path": str(pool_path.relative_to(R_DIR)) if ok else None,
            "pool_sha256": sha256_file(pool_path) if ok else None,
            "code_commit": prov.get("code_git_commit"),
            "retry_dir": str(rdir.relative_to(REPAIR_DIR)),
        }
    return out


def cmd_summarize() -> None:
    """Report tables: eval floor dE after R1, >200 kJ/mol complexes, R3 counts, top-1 impact, worst pools."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tasks = {k: v for k, v in manifest["tasks"].items() if v.get("status") == "success"}
    rm = json.loads((REPAIR_DIR / "repair_manifest.json").read_text(encoding="utf-8"))
    frame = json.loads((REPAIR_DIR / "frame_n_heavy.json").read_text(encoding="utf-8"))
    n_heavy = {}
    for split in SPLITS:
        for e in frame["main_allocation"]["splits"][split]:
            n_heavy.setdefault(e.get("code") or e["representative_code"], e["n_heavy"])

    per_pool = []
    for key, rec in tasks.items():
        ov_rec = rm["pools"].get(key)
        if ov_rec is None:
            continue
        ov = np.load(R_DIR / ov_rec["overlay_path"], allow_pickle=False)
        if "floor_dE_kj" in ov.files:
            floor_dE = float(ov["floor_dE_kj"])
            top1_dE = float(ov["top1_dE_kj"])
            src = "R1_overlay"
        else:
            floor_dE = rec.get("floor_dE_kj")
            top1_dE = rec.get("top1_dE_kj")
            src = "stored"
        row = {"key": key, "code": rec["code"], "seed": rec["seed"], "split": rec["split"],
               "stratum": rec["stratum"], "n_heavy": n_heavy.get(rec["code"]), "n_confs": rec["n_confs"],
               "floor_dE_kj": floor_dE, "top1_dE_kj": top1_dE, "dE_source": src,
               "top1_idx": rec["top1_idx"], "repairs": ov_rec["repairs"]}
        if "graph_match" in ov.files:
            gm = np.asarray(ov["graph_match"], dtype=bool)
            mb = np.asarray(ov["max_bond_dev_A"], dtype=float)
            mh = np.asarray(ov["min_heavy_nonbonded_A"], dtype=float)
            nn = np.asarray(ov["max_nn_dist_A"], dtype=float)
            t1 = int(rec["top1_idx"])
            row.update({
                "n_graph_false": int((~gm).sum()), "top1_graph_false": bool(not gm[t1]),
                "max_bond_dev_A": float(mb.max()),
                "min_heavy_nonbonded_A": float(np.nanmin(mh)) if np.isfinite(mh).any() else None,
                "max_nn_dist_A": float(nn.max()),
            })
        per_pool.append(row)

    eval_rows = [r for r in per_pool if r["split"] in EVAL_SPLITS]
    best: Dict[str, Dict[str, Any]] = {}
    for r in eval_rows:
        if r["floor_dE_kj"] is None:
            continue
        cur = best.get(r["code"])
        if cur is None or r["floor_dE_kj"] < cur["floor_dE_kj"]:
            best[r["code"]] = r
    vals = np.array([b["floor_dE_kj"] for b in best.values()], dtype=float)
    pct = {str(q): float(np.percentile(vals, q)) for q in (5, 25, 50, 75, 95)} if vals.size else {}
    over200 = sorted((b for b in best.values() if b["floor_dE_kj"] > 200), key=lambda b: -b["floor_dE_kj"])

    r3 = {"pools_with_any_graph_false": {}, "top1_graph_false": {}, "min_heavy_lt_1_5": {}, "pools": {}}
    for r in per_pool:
        if "n_graph_false" not in r:
            continue
        sp = r["split"]
        r3["pools"][sp] = r3["pools"].get(sp, 0) + 1
        if r["n_graph_false"] > 0:
            r3["pools_with_any_graph_false"][sp] = r3["pools_with_any_graph_false"].get(sp, 0) + 1
        if r["top1_graph_false"]:
            r3["top1_graph_false"][sp] = r3["top1_graph_false"].get(sp, 0) + 1
        if r["min_heavy_nonbonded_A"] is not None and r["min_heavy_nonbonded_A"] < 1.5:
            r3["min_heavy_lt_1_5"][sp] = r3["min_heavy_lt_1_5"].get(sp, 0) + 1
    worst = sorted((r for r in per_pool if "max_bond_dev_A" in r), key=lambda r: -r["max_bond_dev_A"])[:15]

    tables = {
        "eval_best_seed_floor_dE_percentiles_kj": pct,
        "eval_complexes": len(best),
        "eval_complexes_gt_200_kj": [{k: b[k] for k in ("code", "seed", "split", "stratum", "n_heavy", "floor_dE_kj", "dE_source", "top1_dE_kj")}
                                     | {"max_bond_dev_A": b.get("max_bond_dev_A"), "min_heavy_nonbonded_A": b.get("min_heavy_nonbonded_A"),
                                        "n_graph_false": b.get("n_graph_false"), "top1_graph_false": b.get("top1_graph_false")}
                                     for b in over200],
        "r3_counts": r3,
        "r3_worst_15_by_max_bond_dev": [{k: r.get(k) for k in ("key", "split", "stratum", "n_heavy", "n_confs", "max_bond_dev_A",
                                                             "min_heavy_nonbonded_A", "max_nn_dist_A", "n_graph_false", "top1_graph_false")}
                                        for r in worst],
    }
    (REPAIR_DIR / "report_tables.json").write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in tables.items() if k != "eval_complexes_gt_200_kj"}, indent=2))
    print("eval complexes > 200 kJ/mol:", len(tables["eval_complexes_gt_200_kj"]))


def cmd_post_hashes() -> None:
    """Re-hash every frozen file listed in pre_hashes.json and prove identity with the pre-run record."""
    pre = json.loads((REPAIR_DIR / "pre_hashes.json").read_text(encoding="utf-8"))
    extra = ["frame.json", "spec.json", "ref_mace_cache.json"]
    post: Dict[str, str] = {}
    for rel in list(pre.keys()) + extra:
        post[rel] = sha256_file(R_DIR / rel)
    pool_files = sorted(p for p in pre if p.startswith("pools/"))
    mismatches = [k for k in pre if post.get(k) != pre[k]]
    summary = {
        "pre_hashes_entries": len(pre),
        "pool_files_checked": len(pool_files),
        "extra_files": extra,
        "mismatches": mismatches,
        "identical": not mismatches,
    }
    (REPAIR_DIR / "post_hashes.json").write_text(
        json.dumps({"summary": summary, "post": post}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if mismatches:
        raise SystemExit(f"STOP: {len(mismatches)} frozen files changed")


def cmd_registry() -> None:
    """Append S2.1 record 21 (amending record 20) to the experiment index, validated by the registry module."""
    import research.archive.code.refit.nci_v2_registry as REG

    index_path = REPO_ROOT / "data/refit/experiments/index.json"
    records = json.loads(index_path.read_text(encoding="utf-8"))
    parent_id = "nci_v2_scaled_v1__step2_control__s11-13"
    new_id = "nci_v2_scaled_v1__s2_1_repair__amends_step2_control"
    if any(r.get("id") == new_id for r in records):
        raise SystemExit("record 21 already present")
    parent = next(r for r in records if r.get("id") == parent_id)
    mace_hash = parent["evaluation"]["model_hash"]
    db_hash = parent["generation"]["db_hash"]
    rel = lambda p: str(Path(p).relative_to(REPO_ROOT))
    artifacts = []
    for p, desc in [
        (REPAIR_DIR / "repair_manifest.json", "S2.1 overlay index: per-pool repairs, retries, slots, unmappable"),
        (REPAIR_DIR / "overlay_summary.json", "S2.1 overlay summary: R1 energies, R2 counts, R3 health"),
        (REPAIR_DIR / "report_tables.json", "S2.1 report tables: eval dE percentiles, >200 kJ/mol complexes, R3 counts"),
        (REPAIR_DIR / "frame_n_heavy_proof.json", "Derived frame canonical-equality proof and pool cross-check"),
        (REPAIR_DIR / "frame_n_heavy.json", "Frozen frame plus integer n_heavy (derived)"),
        (REPAIR_DIR / "post_hashes.json", "Post-repair hashes of frozen files, identical to pre_hashes.json"),
        (REPAIR_DIR / "ab_gate" / "characterization.json", "A/B and same-commit repeat characterization"),
        (REPAIR_DIR / "S2_1_REPORT.md", "S2.1 report"),
    ]:
        artifacts.append({"path": rel(p), "sha256": sha256_file(p), "availability": "available", "description": desc})
    record = {
        "id": new_id,
        "readable_name": "s2_1_post_run_repair",
        "mode": "audit",
        "status": "completed",
        "question": "Are Step 2 control pools valid after repair (MACE energies, atom mapping, health metrics), and is GPU solving reproducible run-to-run?",
        "parent_record_ids": [parent_id],
        "amends": [parent_id],
        "cohort_manifest": "data/refit/nci_v2/scaled_v1/manifest.json",
        "split": "scaled_v1_main",
        "selection_rule": "All 4,296 successful Step 2 pools overlaid (R1 eval MACE, R2 mapping, R3 health); 21 codes retried with the max_rows fix (stopped by decision)",
        "exclusions": [
            "Retries not run (stopped by decision): OBABAX, TATXIW, COJSIG, OMAYEJ, UZOREI, VALJEB",
            "Unmappable after R2 (no R3, no R2 overlay): IWOLEN, UQUNUQ, VENGUR",
        ],
        "seeds": [11, 12, 13],
        "generation": {
            "code_hash": "943bcc8",
            "db_hash": db_hash,
            "full_profile": {"profile": "u1", "nci": False, "deterministic_embed": True},
            "settings": {"cohort": "main", "arms": ["control"], "seeds": [11, 12, 13], "workers": 1,
                         "retry_fix_commit": "21d5446", "retry_mem_fraction_effective": 0.70},
            "environment": {"platform": "linux-64", "torch_dtype": "float64",
                            "cuda_device": "NVIDIA GeForce RTX 2060 SUPER"},
        },
        "ranking": {"code_hash": "943bcc8", "db_hash": db_hash,
                    "full_profile": {"profile": "u1", "nci": False, "deterministic_embed": True},
                    "source_pool_ids": None},
        "evaluation": {
            "model_hash": mace_hash,
            "evaluator_hash": parent["evaluation"]["evaluator_hash"],
            "dtype": "float64",
            "units": "kJ/mol",
            "state_reference_hashes": {},
        },
        "artifacts": artifacts,
        "metrics": {
            "definitions": {
                "r1_pools": "Eval pools whose stored MACE energies were invalid (recomputed)",
                "r2_remapped": "Pools with a mapping error rebuilt by the offline HEAD mapping path",
                "r3_graph_false": "Conformers whose covalent graph differs from the reference graph",
                "gate_ab": "confs_sha256 equality between two runs (redefined by auditor, see report)",
            },
            "aggregation_unit": "pool (code, seed)",
            "paired_comparator": "8538d7f vs 920d0a8 vs same-commit repeat on 6 complexes x 3 seeds",
            "values": {
                "pools_overlaid": 4296, "r1_pools": 36, "r1_complexes": 12, "r2_remapped": 273,
                "unmappable_pools": 9, "gate_ab_identical_pairs": 0,
                "characterization": "n_confs and conformer order identical for all 18 pairs; top-1 identical for all 18; same-commit repeat differs (RIQDEB:11 16 vs 15 conformers, FANTIZ:11 max dX 7.9 A)",
                "retries_v2_success": "14 of 15 attempted codes (BIKQIY OOM)",
            },
            "uncertainty": "not estimated",
        },
        "conclusion": "Step 2 control pools are repaired in overlays; frozen data unchanged. GPU solving is not run-to-run reproducible on this machine, so confs_sha256 certifies stored data only. 12 eval complexes had invalid MACE energies and are recomputed (repaired floor dE 27 to 317 kJ/mol). Step 2 OOMs were largely caused by an orchestrator retry bug (fixed in 21d5446).",
        "limitations": "Retries are fresh samples, not reproductions. Six of 21 retries not run. Stage of the IRAQUO stray H not attributed. 4f_zero_high coordination loss diagnosed from stored pools only. The registry --build step regenerates index.json from hard-coded records and will drop this record unless build_historical_records is updated.",
        "superseded_by": None,
    }
    errors = REG.validate_record(record, repo_root=REPO_ROOT)
    parent_amended = dict(parent)
    parent_amended["amended_by"] = [new_id]
    records = [parent_amended if r.get("id") == parent_id else r for r in records] + [record]
    errors += REG.validate_registry(records, repo_root=REPO_ROOT)
    if errors:
        raise SystemExit("registry validation failed:\n" + "\n".join(errors))
    index_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"wrote record 21 ({new_id}); index now has {len(records)} records; validation clean")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("derive-frame", help="Phase 1: write frame_n_heavy.json and its proof")
    sub.add_parser("characterize", help="A/B and repeat characterization (report only)")
    rc = sub.add_parser("remap-check", help="Validate offline R2 mapping against stored mappings")
    rc.add_argument("--n", type=int, default=30)
    sub.add_parser("summarize", help="Report tables from overlays and manifests")
    sub.add_parser("post-hashes", help="Re-hash frozen files and prove identity with pre_hashes.json")
    sub.add_parser("registry", help="Append S2.1 record 21 (amends record 20) and validate")
    ov = sub.add_parser("overlay", help="Phase 4: write overlays, repair_manifest.json, overlay_summary.json")
    ov.add_argument("--codes", type=str, default="")
    ov.add_argument("--skip-mace", action="store_true")
    args = ap.parse_args()
    if args.cmd == "derive-frame":
        cmd_derive_frame()
    elif args.cmd == "characterize":
        cmd_characterize()
    elif args.cmd == "remap-check":
        cmd_remap_check(args.n)
    elif args.cmd == "summarize":
        cmd_summarize()
    elif args.cmd == "post-hashes":
        cmd_post_hashes()
    elif args.cmd == "registry":
        cmd_registry()
    elif args.cmd == "overlay":
        codes = [c for c in args.codes.split(",") if c] or None
        cmd_overlay(codes, args.skip_mace)


if __name__ == "__main__":
    main()
