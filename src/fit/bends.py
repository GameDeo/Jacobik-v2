"""
src/fit/bends.py
================
Backbone angle bend stiffness constants and haptic lean walls.

Physical model and parameters:
- Chelate backbone DOFs: for acyclic heavy-heavy bonds on shortest paths between kappa donors.
- Bend stiffness k (kJ/mol/rad^2): flat-bottom quadratic penalty (score term "Q7") at pivots:
  Q7 = -weight * mean_i 0.5 * (max(0, |dtheta_i| - free) / sigma)^2
- Constants (BEND_K):
  - in-plane sp2: 1300 kJ/mol/rad^2
  - in-plane sp3: 1800 kJ/mol/rad^2
  - out-of-plane sp2: 310 kJ/mol/rad^2
  - out-of-plane sp3: 1100 kJ/mol/rad^2
- Haptic lean walls (LEAN_FREE_DEG / LEAN_FREE_DEG_ALL):
  - eta2: 57 deg
  - eta3: 40 deg
  - eta4: 50 deg
  - eta5: 17 deg
  - eta6: 16 deg
  - eta7: 11 deg
  - eta8: 18 deg
  Derived from 2.5% / 97.5% quantiles of per-substituent bend features in QO reference data.

Fitting methodology:
- MACE-POLAR-1 single-point scans on free ligands after chelate-aware embed.
- Rotating the moving side through beta = +-4, +-8, +-12 deg.
- Harmonic fit through origin: dE = 0.5 * k * beta^2.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from jacobik.ik import angle_bends as AB
from jacobik.ik.angle_bends import bound_angle_fallbacks

log = logging.getLogger(__name__)

QUANTITY: str = "BENDS"
FREE_DEG: float = AB.FREE_DEG
RT_KJ: float = AB.RT_KJ
BEND_K: Dict[Tuple[str, str], float] = dict(AB.BEND_K)
LEAN_FREE_DEG: Dict[int, float] = dict(AB.LEAN_FREE_DEG)
LEAN_FREE_DEG_ALL: Dict[int, float] = dict(AB.LEAN_FREE_DEG_ALL)


def bound_angle_key(pivot_element: str, hybridisation: str, neighbour_elements: Sequence[str],
                    ring_size: Optional[int], metal_row: Optional[int]) -> Tuple[Any, ...]:
    """Canonical crystal-table key for a chelate-backbone angle pivot."""
    return (str(pivot_element), str(hybridisation), tuple(sorted(map(str, neighbour_elements))),
            None if ring_size is None else int(ring_size), None if metal_row is None else int(metal_row))


def build_bound_angle_table(observations: Sequence[Dict[str, Any]], min_n: int = 1) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    """Aggregate crystal pivot angles (degrees) into a sidecar table keyed by ``bound_angle_key`` fields.

    Each observation has ``key`` and ``angle_deg``; optional ``code`` is retained only in provenance counts.
    """
    grouped: Dict[Tuple[Any, ...], List[float]] = collections.defaultdict(list)
    parents: Dict[Tuple[Any, ...], set] = collections.defaultdict(set)
    for item in observations:
        key = tuple(item["key"])
        value = float(item["angle_deg"])
        if math.isfinite(value):
            # Materialize each fallback level with this same fitter so runtime lookup
            # never needs to infer or refit a generic row.
            for fallback in bound_angle_fallbacks(key):
                grouped[fallback].append(value)
                if item.get("code"):
                    parents[fallback].add(str(item["code"]))
    table = {}
    for key, values in grouped.items():
        support = len(parents[key]) if parents[key] else len(values)
        if support < min_n:
            continue
        table[key] = {"n": len(values), "mean_deg": statistics.fmean(values),
                      "sd_deg": statistics.stdev(values) if len(values) > 1 else 0.0,
                      "n_parents": len(parents[key])}
    return table


def build_bound_angle_table_from_frame(frame: Dict[str, Any], load_complex, min_n: int = 1):
    """Build the Q7 table from the fit frame's ``surface_fit`` split.

    ``load_complex(code, shard, dataset)`` supplies the crystal-frame complex. The
    return value is ``(table, report)``; per-parent extraction failures are recorded
    and the rest of the fit frame continues, matching the batch failure policy.
    """
    split_rows = frame.get("main_allocation", {}).get("splits", {}).get("surface_fit", [])
    parents = {}
    for row in split_rows:
        code = row.get("code") or row.get("representative_code")
        if not code:
            continue
        parents.setdefault(str(code), row)
    observations = []
    failures = []
    for code, row in sorted(parents.items()):
        shard = row.get("shard") or row.get("representative_shard")
        dataset = row.get("dataset")
        try:
            mol = load_complex(code, shard, dataset)
            observations.extend(extract_bound_angle_observations(mol, code))
        except Exception as exc:
            failures.append({"code": code, "error": f"{type(exc).__name__}: {exc}"[:400]})
    table = build_bound_angle_table(observations, min_n=min_n)
    report = {"frame_split": "surface_fit", "parents_requested": len(parents),
              "parents_failed": len(failures), "n_observations": len(observations),
              "n_entries": len(table), "failures": failures}
    return table, report


def write_bound_angle_sidecar(path: Union[str, Path], observations: Sequence[Dict[str, Any]],
                              provenance: Optional[Dict[str, Any]] = None, min_n: int = 1) -> Dict[str, Any]:
    """Write deterministic JSON sidecar from crystal observations using the canonical table builder above."""
    table = build_bound_angle_table(observations, min_n=min_n)
    payload = {"schema": "q7_bound_angles_v1", "provenance": provenance or {},
               "entries": [{"key": list(k), **v} for k, v in sorted(table.items(), key=lambda kv: repr(kv[0]))]}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def write_bends_sidecar(fit_frame_path: Union[str, Path], ref_dir: Union[str, Path],
                        out_dir: Union[str, Path], load_complex=None,
                        min_n: int = 1) -> Dict[str, Any]:
    """Extract and write the Q7 sidecar from the fit frame's surface_fit parents.

    ``load_complex`` is an optional ``(code, shard, dataset)`` loader seam for
    tests. Production uses the source-side raw geometry loader, keeping
    ``src/jacobik/fit`` independent of ``research/bench``.
    """
    frame_path, raw_root, output = Path(fit_frame_path), Path(ref_dir), Path(out_dir)
    frame_bytes = frame_path.read_bytes()
    frame = json.loads(frame_bytes)
    if load_complex is None:
        from fit.parser import load_raw_complex

        def load_complex(code, shard, dataset):
            mol = load_raw_complex(code, shard, dataset, raw_root)
            if mol is None:
                raise FileNotFoundError(f"no raw geometry found for {code} ({shard}, {dataset})")
            return mol

    table, report = build_bound_angle_table_from_frame(frame, load_complex, min_n=min_n)
    failures = report["parents_failed"]
    requested = report["parents_requested"]
    if failures / max(requested, 1) > 0.05:
        raise RuntimeError(
            f"Q7 bound-angle extraction failed for {failures}/{requested} surface_fit parents (>5%); "
            f"details: {report['failures'][:10]}"
        )

    output.mkdir(parents=True, exist_ok=True)
    sidecar = output / "constraints.bends.json"
    payload = {
        "schema": "q7_bound_angles_v1",
        "provenance": {
            "source": "fit-frame surface_fit crystal geometries",
            "fit_frame_path": str(frame_path.resolve()),
            "fit_frame_sha256": hashlib.sha256(frame_bytes).hexdigest(),
            "fit_frame_sha256_method": "raw_bytes",
            "extraction_report": report,
            "min_n": min_n,
        },
        "entries": [{"key": list(key), **value}
                    for key, value in sorted(table.items(), key=lambda item: repr(item[0]))],
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload["sidecar_path"] = str(sidecar)
    payload["sidecar_sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    return payload


def extract_bound_angle_observations(mol: Any, code: str) -> List[Dict[str, Any]]:
    """Extract crystal angles at Q7 bend pivots with the registered chemical key and fallback dimensions."""
    import networkx as nx

    atoms = {a.idx: a for a in mol.atoms}
    metal = atoms[mol.metal_idx]
    try:
        from rdkit import Chem
        pt = Chem.GetPeriodicTable()
        metal_row = int(pt.GetRow(pt.GetAtomicNumber(metal.symbol)))
    except Exception:
        metal_row = None
    ring_at: Dict[int, Optional[int]] = {}
    for lig in mol.ligands:
        graph = AB._lig_graph(mol, lig)
        ds = [d for d in lig.donor_atoms if d in graph]
        for pivot in lig.atom_indices:
            sizes=[]
            for i, a in enumerate(ds):
                for b in ds[i+1:]:
                    try:
                        path=nx.shortest_path(graph,a,b)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                    if pivot in path:
                        sizes.append(len(path)+2)  # include the metal closure in ring size
            if sizes:
                ring_at[pivot]=min(sizes)
    positions={i: np.asarray(a.position,dtype=float) for i,a in atoms.items()}
    out=[]
    for a,p,b in AB.angle_triplets(mol):
        u=positions[a]-positions[p]; v=positions[b]-positions[p]
        denom=float(np.linalg.norm(u)*np.linalg.norm(v))
        if denom <= 1e-12: continue
        angle=math.degrees(math.acos(float(np.clip(np.dot(u,v)/denom,-1.0,1.0))))
        key=bound_angle_key(atoms[p].symbol, AB.pivot_hybridisation(mol,p),
                            [atoms[a].symbol,atoms[b].symbol], ring_at.get(p), metal_row)
        out.append({"key": key, "angle_deg": angle, "code": code})
    return out


# ---------------------------------------------------------------------------
# Extraction & Classification
# ---------------------------------------------------------------------------

def pivot_hybridisation(mol: Any, p: int) -> str:
    """Classify pivot atom hybridisation ('sp2' or 'sp3')."""
    return AB.pivot_hybridisation(mol, p)


def stiffness(mol: Any, specs: List[Tuple]) -> List[float]:
    """Compute stiffness k (kJ/mol/rad^2) for each bend DOF."""
    return AB.stiffness(mol, specs)


def free_zone(specs: List[Tuple], free_deg: float = FREE_DEG) -> List[float]:
    """Compute free zone (deg) for each bend DOF."""
    return AB.free_zone(specs, free_deg=free_deg)


# ---------------------------------------------------------------------------
# Fitting Helpers (MACE / Reference Data)
# ---------------------------------------------------------------------------

def fit_bend_stiffness(
    dE_kJ_mol: np.ndarray | Sequence[float],
    beta_rad: np.ndarray | Sequence[float],
) -> float:
    """
    Fit harmonic bend stiffness k (kJ/mol/rad^2) through the origin:
    dE = 0.5 * k * beta^2  ==>  k = sum(dE * (beta^2 / 2)) / sum((beta^2 / 2)^2)
    """
    dE = np.asarray(dE_kJ_mol, dtype=float)
    beta = np.asarray(beta_rad, dtype=float)
    x = 0.5 * (beta ** 2)
    denom = float(np.sum(x ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(dE * x) / denom)


def fit_from_observations(
    scan_records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Summarize fitted bend stiffness values across scanned ligands by category.

    scan_records: list of dicts with keys 'pivot', 'hyb', 'axis', 'k'.
    Returns summary statistics: count, median, q25, q75.
    """
    grouped = collections.defaultdict(list)
    for r in scan_records:
        cat = f"{r['pivot']}|{r['hyb']}|{r['axis']}"
        grouped[cat].append(float(r["k"]))

    summary: Dict[str, Dict[str, Any]] = {}
    for cat, ks in grouped.items():
        arr = np.array(ks)
        summary[cat] = {
            "n": len(arr),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }
    return summary

