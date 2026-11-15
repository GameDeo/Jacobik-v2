"""Simple reference-baseline mode (research/archive/docs/roadmap/roadmap_generation_v1.md 1.4.3). Status: descriptive. Never a promotion claim.

Candidate units are JSON records written by tools/jgen/bench/run.py, one per (code, seed):
    {code, seed, status: "ok" | "failed", reason, top1_dE, floor_dE, n_rows, n_finite, ref_energy_kj,
     frame_sha256, constraints_db_sha256, code_git_commit, arm, fingerprint, ...}
Reference units are the stored Step 2 control pools (effective, poolio.py), or any stored pool directory of the same
layout (for example the S2b regeneration, raw), read by records_from_pool_dir.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diag"))  # poolio (read-only pool loader)

from poolio import Repairs, load_effective  # noqa: E402

from tools.jgen.bench import metrics  # noqa: E402
from tools.jgen.bench.panels import select, unit_keys  # noqa: E402

Key = Tuple[str, int]


def _record_from_dE(dE: np.ndarray, score: Optional[np.ndarray], meta: Dict) -> Dict:
    finite = np.isfinite(dE)
    rec = dict(meta)
    rec["n_rows"] = int(len(dE))
    rec["n_finite"] = int(finite.sum())
    if rec["n_finite"] == 0:
        rec.update(status="failed", reason="no_finite_energy", top1_dE=None, floor_dE=None)
        return rec
    top = int(np.argmax(score)) if score is not None and len(score) == len(dE) and np.isfinite(score).any() else 0
    if not finite[top]:
        rec.update(status="failed", reason="top1_not_finite", top1_dE=None, floor_dE=float(np.min(dE[finite])))
        return rec
    rec.update(status="ok", reason="", top1_dE=float(dE[top]), floor_dE=float(np.min(dE[finite])))
    return rec


def records_from_pool_dir(pool_dir: str, keys: List[Key], repairs: Optional[Repairs] = None,
                          effective: bool = False, root: Optional[str] = None) -> Dict[Key, Dict]:
    """Records of stored pools. effective=True applies the S2.1 overlays (needs repairs and the frozen root)."""
    out: Dict[Key, Dict] = {}
    for code, seed in keys:
        if effective:
            e = load_effective(pool_dir, repairs, code, seed)
            if e is None:
                out[(code, seed)] = dict(code=code, seed=seed, status="failed", reason="stored_error_or_absent",
                                         top1_dE=None, floor_dE=None)
                continue
            out[(code, seed)] = _record_from_dE(e["dE"], e["score"],
                                                dict(code=code, seed=seed, source=e["source"]))
            continue
        npz = os.path.join(pool_dir, "%s_%d_control.npz" % (code, seed))
        mj = os.path.join(pool_dir, "%s_%d_control.mace64.json" % (code, seed))
        if not (os.path.exists(npz) and os.path.exists(mj)):
            out[(code, seed)] = dict(code=code, seed=seed, status="failed", reason="absent",
                                     top1_dE=None, floor_dE=None)
            continue
        z = np.load(npz, allow_pickle=True)
        if "error" in z.files:
            out[(code, seed)] = dict(code=code, seed=seed, status="failed", reason="stored_error",
                                     top1_dE=None, floor_dE=None)
            continue
        m = json.load(open(mj))
        dE = np.asarray(m["energies_kj"], float) - float(m["ref_energy_kj"])
        out[(code, seed)] = _record_from_dE(dE, np.asarray(z["score"], float),
                                            dict(code=code, seed=seed, ref_energy_kj=float(m["ref_energy_kj"]),
                                                 source="raw"))
    return out


def read_candidate_dir(path: str) -> Dict[Key, Dict]:
    out: Dict[Key, Dict] = {}
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        rec = json.load(open(f))
        out[(rec["code"], int(rec["seed"]))] = rec
    return out


def pairing_identity(cand: Dict[Key, Dict], frozen: Dict[str, str]) -> Dict:
    """The pairing rule of 1.4.3: frame and constraints DB hashes must match the frozen values. The code commit is
    recorded; a candidate carries its own commit by design, so it is reported and never used to reject a pair."""
    bad, commits = [], {}
    for k, r in cand.items():
        for key in ("frame_sha256", "constraints_db_sha256"):
            if r.get(key) != frozen.get(key):
                bad.append(dict(code=k[0], seed=k[1], field=key, got=r.get(key), frozen=frozen.get(key)))
        commits[r.get("code_git_commit", "?")] = commits.get(r.get("code_git_commit", "?"), 0) + 1
    return dict(identity_mismatches=bad[:50], n_identity_mismatches=len(bad), code_commits=commits,
                frozen_frame_sha256=frozen.get("frame_sha256"),
                frozen_constraints_db_sha256=frozen.get("constraints_db_sha256"))


def run(panel: str, candidate: Dict[Key, Dict], reference_root: Optional[str], manifest_path: str,
        repair_path: str, control: Optional[Dict[Key, Dict]] = None) -> Dict:
    """Compare one arm with the reference on one panel. Missing or invalid pairs are in the ledger, never dropped.
    The reference is the stored Step 2 pools (reference_root) unless `control` is given: the records of a bench run of
    the control arm on the same machine (read_candidate_dir), which is the paired baseline for Colab arms."""
    manifest = json.load(open(manifest_path))
    sel = select(manifest, panel)
    keys = unit_keys(sel["units"])
    if control is not None:
        base = {k: control.get(k, dict(code=k[0], seed=k[1], status="failed", reason="absent_in_control",
                                       top1_dE=None, floor_dE=None)) for k in keys}
        repairs = None
    else:
        ref_pools = os.path.join(reference_root, "pools")
        repairs = Repairs(reference_root)
        base = records_from_pool_dir(ref_pools, keys, repairs, effective=True)
    frozen = json.load(open(manifest_path)).get("base_provenance", {})
    summary = metrics.summarize({k: candidate[k] for k in keys if k in candidate}, base, keys)
    # units in the candidate but not in the panel are not compared (reported)
    extra = sorted(set(candidate) - set(keys))
    summary.update(panel=panel, status="descriptive", not_in_cohort=sel["not_in_cohort"],
                   candidate_units_outside_panel=[list(k) for k in extra][:50],
                   identity=pairing_identity({k: candidate[k] for k in keys if k in candidate}, frozen),
                   candidate_missing=[list(k) for k in keys if k not in candidate][:200],
                   reference_status_counts=_count([base[k]["status"] for k in keys]),
                   candidate_status_counts=_count([candidate[k]["status"] for k in keys if k in candidate]))
    return summary


def _count(xs):
    out: Dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out
