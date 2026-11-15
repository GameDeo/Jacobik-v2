"""NCI-CTX1 data census: strict native H-bonds and parent/scaffold split.

This is an isolated research pipeline. It does not edit the deployed NCI
tables, profile, scorer, or Q6. The broad geometric pass is only a cheap
prefilter; a positive label always passes the chemistry-aware strict gate in
``nci_hb_context``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).parent)]

from nci_hb_context import SUPPORT_MAX_HA, features, select_high_confidence

DATA = ROOT / "data/refit/clean/tmQM"
SCHEMA = "nci_ctx1_native_v1"
_PARSER = None
_OUT = None
_EXCLUDED: set[str] = set()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def scaffold(smiles: str) -> tuple[str, int]:
    """Murcko core, with full canonical graph for acyclic ligands."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("unreadable ligand SMILES")
    core = MurckoScaffold.GetScaffoldForMol(mol)
    chosen = core if core.GetNumAtoms() else mol
    return Chem.MolToSmiles(chosen, isomericSmiles=False), int(mol.GetNumHeavyAtoms())


def _init_worker(out: str, excluded: list[str]) -> None:
    global _PARSER, _OUT, _EXCLUDED
    import logging
    from fit.parser import TMQMParser
    from rdkit import RDLogger

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("jacobik.topology.mapping").setLevel(logging.ERROR)
    logging.getLogger("fit.parser").setLevel(logging.ERROR)
    RDLogger.DisableLog("rdApp.*")
    _OUT = Path(out)
    _EXCLUDED = set(excluded)
    _PARSER = TMQMParser(DATA)


def _broad_geometry_exists(mol) -> bool:
    """Loose geometry prefilter that cannot exclude any strict positive."""
    import numpy as np
    from scipy.spatial import cKDTree
    from jacobik.stats import keys as kb
    from research.archive.code.interactions.nci import hbond_candidates
    from jacobik.topology.graph import get_ligand_neighbors

    atoms = list(mol.atoms)
    xyz = {a.idx: np.asarray(a.position, dtype=np.float64) for a in atoms}
    nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    hydrogens = sorted(kb.q6_polar_h(mol, nb))
    acceptors = [a.idx for a in atoms if a.idx != mol.metal_idx and a.symbol in {"N", "O", "F", "S", "Cl", "Br", "I"}]
    if not hydrogens or not acceptors:
        return False
    tree = cKDTree(np.asarray([xyz[a] for a in acceptors]))
    if not any(tree.query_ball_point(xyz[h], SUPPORT_MAX_HA) for h in hydrogens):
        return False
    for d, h, a, _ in hbond_candidates(mol):
        v_dh = xyz[d] - xyz[h]
        v_ha = xyz[a] - xyz[h]
        r = float(np.linalg.norm(v_ha))
        if r > SUPPORT_MAX_HA or r < 1e-9:
            continue
        c = float(np.dot(v_dh, v_ha) / max(float(np.linalg.norm(v_dh)) * r, 1e-12))
        if c <= float(np.cos(np.radians(120.0))):
            return True
    return False


def _typed_record(mol) -> dict:
    from jacobik.builder.ligands import _build_ligand_groups

    groups = _build_ligand_groups(mol, mol.smiles)
    owner: dict[int, int] = {}
    scaffolds: dict[int, tuple[str, int]] = {}
    for j, group in enumerate(groups):
        smiles = group.get("canonical_smiles")
        if not smiles:
            raise ValueError("ligand group has no canonical SMILES")
        scaffolds[j] = scaffold(smiles)
        for atom0 in group["xyz_indices_0based"]:
            owner[mol.atoms[atom0].idx] = j

    all_features = features(mol)
    high = select_high_confidence(all_features)
    contacts = []
    for item in high:
        if item.donor not in owner or item.acceptor not in owner:
            raise ValueError("strict contact lacks mapped ligand owner")
        ligands = sorted({owner[item.donor], owner[item.acceptor]})
        partners = []
        for j in ligands:
            smi, heavy = scaffolds[j]
            partners.append({"ligand": j, "scaffold": smi, "heavy_atoms": heavy})
        contacts.append({
            "donor": item.donor,
            "hydrogen": item.hydrogen,
            "acceptor": item.acceptor,
            "keys": [list(k) for k in item.keys],
            "r_ha": item.r_ha,
            "dha_deg": item.dha_deg,
            "acceptor_cos": item.cos_acceptor,
            "acceptor_axis_defined": item.acceptor_axis_defined,
            "partners": partners,
        })
    # Retain the positive parent's geometry summary without persisting its
    # coordinates. Broad candidates are not positive labels.
    support = [x for x in all_features if x.r_ha <= SUPPORT_MAX_HA and x.dha_deg >= 120.0]
    return {"contacts": contacts, "n_chem_candidates": len(all_features),
            "n_support_candidates": len(support)}


def _scan_shard(shard: str) -> dict:
    import numpy as np
    from fit.parser import (
        _assemble_raw, _attach_qm_properties, _iter_bo_blocks, _iter_xyz_blocks,
    )
    from jacobik.topology.mapping import assemble

    started = time.perf_counter()
    rows = []
    counts = Counter()
    errors = []
    xyz_path = DATA / "xyz" / f"tmQM_{shard}.xyz"
    bo_path = DATA / "bo" / f"tmQM_{shard}.BO"
    for (comment, raw_atoms), (code, raw_bonds) in zip(_iter_xyz_blocks(xyz_path), _iter_bo_blocks(bo_path)):
        counts["input_parents"] += 1
        if counts["input_parents"] % 1000 == 0:
            print("PROGRESS", shard, counts["input_parents"],
                  "strict", counts["strict_positive_parents"], flush=True)
        if code in _EXCLUDED:
            counts["excluded_benchmark"] += 1
            continue
        try:
            csv_row = _PARSER.csv_index.get(code)
            mol = assemble(_assemble_raw(comment, raw_atoms, code, raw_bonds, csv_row))
            if csv_row is not None:
                _attach_qm_properties(mol, csv_row)
        except Exception as exc:
            counts["parse_errors"] += 1
            if len(errors) < 100:
                errors.append({"code": code, "stage": "parse", "error": f"{type(exc).__name__}: {exc}"})
            continue
        counts["parsed"] += 1
        if not mol.smiles:
            counts["missing_smiles"] += 1
            continue
        try:
            if not _broad_geometry_exists(mol):
                continue
            counts["broad_geometry_parents"] += 1
            result = _typed_record(mol)
            counts["chemistry_typed"] += 1
            counts["strict_positive_parents"] += bool(result["contacts"])
            counts["strict_positive_contacts"] += len(result["contacts"])
            if result["contacts"]:
                rows.append({
                    "code": code,
                    "shard": shard,
                    "dataset": shard.split("s", 1)[0],
                    "charge": int(mol.charge),
                    "mult": int(mol.spin_state),
                    "metal": mol.metal_symbol,
                    "n_atoms": len(mol.atoms),
                    **result,
                })
        except Exception as exc:
            counts["chemistry_unsupported"] += 1
            if len(errors) < 100:
                errors.append({"code": code, "stage": "chemistry_or_scaffold", "error": f"{type(exc).__name__}: {exc}"})
    return {"shard": shard, "counts": dict(counts), "positive_parents": rows,
            "errors": errors, "elapsed_s": time.perf_counter() - started}


def _split(rows: list[dict]) -> tuple[list[dict], dict]:
    """Split connected parent/organic-scaffold groups, keeping common tiny partners pooled.

    One- and two-heavy-atom species (water, halides, ammonia, etc.) are common
    chemical partners, not useful scaffold holdouts. Every larger participant
    scaffold remains disjoint between calibration and validation.
    """
    import networkx as nx

    graph = nx.Graph()
    row_by_code = {r["code"]: r for r in rows}
    for row in rows:
        parent = "parent:" + row["code"]
        for contact in row["contacts"]:
            for partner in contact["partners"]:
                if partner["heavy_atoms"] > 2:
                    graph.add_edge(parent, "scaffold:" + partner["scaffold"])
        if parent not in graph:
            graph.add_node(parent)
    components = list(nx.connected_components(graph))
    components.sort(key=lambda c: zlib.crc32(("nci-ctx1-split-v1:" + repr(sorted(c))).encode()))
    target = 0.60 * len(rows)
    calibration: set[str] = set()
    validation: set[str] = set()
    # Greedy deterministic component assignment minimizes the distance to a
    # 60:40 parent split; components are indivisible to prevent scaffold leak.
    for component in components:
        codes = {n.removeprefix("parent:") for n in component if n.startswith("parent:")}
        if len(calibration) < target and (len(validation) >= len(rows) - target or
                abs((len(calibration) + len(codes)) - target) <= abs(len(calibration) - target)):
            calibration.update(codes)
        else:
            validation.update(codes)
    # No component may cross the split. A scaffold can be absent from the graph
    # only for contacts made exclusively by tiny common partners.
    scaffold_split: dict[str, str] = {}
    for component in components:
        split = "calibration" if any(n.removeprefix("parent:") in calibration for n in component if n.startswith("parent:")) else "validation"
        for node in component:
            if node.startswith("scaffold:"):
                scaffold_split[node.removeprefix("scaffold:")] = split
    assert not (calibration & validation)
    for row in rows:
        row["split"] = "calibration" if row["code"] in calibration else "validation"
    return rows, {
        "calibration_parents": len(calibration),
        "validation_parents": len(validation),
        "components": len(components),
        "component_parent_sizes": sorted((sum(n.startswith("parent:") for n in c) for c in components), reverse=True),
        "large_scaffold_count": len(scaffold_split),
        "scaffold_split": scaffold_split,
    }


def scan(out: Path, workers: int = 4, shard_filter: list[str] | None = None) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from fit.pipeline.extract import benchmark_codes

    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(p.stem.removeprefix("tmQM_") for p in (DATA / "xyz").glob("tmQM_X[123]s*.xyz"))
    if shard_filter:
        unknown = sorted(set(shard_filter) - set(shards))
        if unknown:
            raise ValueError(f"unknown tmQM shards: {unknown}")
        shards = [s for s in shards if s in set(shard_filter)]
    excluded = sorted(benchmark_codes())
    sources = [DATA / "tmQM_y.csv"]
    for shard in shards:
        sources.extend([DATA / "xyz" / f"tmQM_{shard}.xyz", DATA / "bo" / f"tmQM_{shard}.BO"])
    identity = {
        "schema": SCHEMA,
        "shards": shards,
        "benchmark_codes_excluded": excluded,
        "strict_positive_rule": {
            "chemistry": "shared charge-aware hb_ctx1 donor/acceptor roles from authoritative SMILES",
            "r_HA_A_max": 2.50,
            "DHA_deg_min": 150.0,
            "acceptor_outward_hemisphere_cos_min_when_defined": 0.0,
            "one_acceptor_per_donor_H": "most linear, then shortest; deterministic",
        },
        "broad_prefilter_only": {"r_HA_A_max": SUPPORT_MAX_HA, "DHA_deg_min": 120.0},
        "scaffold_split": "connected components of parent and participant Murcko scaffolds with >2 heavy atoms; small common solvent/ion partners pooled across splits",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "chemistry_sha256": digest(Path(__file__).with_name("nci_hb_chemistry.py")),
        "geometry_sha256": digest(Path(__file__).with_name("nci_hb_context.py")),
        "driver_sha256": digest(Path(__file__)),
    }
    manifest = out / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()).get("identity") == identity, "input/code identity changed; choose a fresh output directory"
    else:
        write(manifest, {"identity": identity, "status": "running"})
    pending = [s for s in shards if not (out / f"{s}.json").exists()]
    done = [json.loads((out / f"{s}.json").read_text()) for s in shards if (out / f"{s}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(str(out), excluded)) as pool:
        for shard_result in pool.map(_scan_shard, pending):
            write(out / f"{shard_result['shard']}.json", shard_result)
            done.append(shard_result)
            print("SCANNED", shard_result["shard"], shard_result["counts"],
                  f"{shard_result['elapsed_s']:.1f}s", flush=True)
    counts = Counter()
    rows = []
    errors = []
    for result in done:
        counts.update(result["counts"])
        rows.extend(result["positive_parents"])
        errors.extend(result["errors"])
    rows, split_summary = _split(rows)
    fine = Counter()
    per_key_parents: dict[tuple[str, ...], set[str]] = defaultdict(set)
    per_key_split: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    fallback_parents = [defaultdict(set) for _ in range(4)]
    fallback_split = [defaultdict(Counter) for _ in range(4)]
    fine_fallback: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for row in rows:
        for contact in row["contacts"]:
            key0 = tuple(contact["keys"][0])
            fine[key0] += 1
            per_key_parents[key0].add(row["code"])
            per_key_split[key0][row["split"]] += 1
            fine_fallback[key0] = [tuple(k) for k in contact["keys"]]
            for level, key in enumerate(contact["keys"]):
                key = tuple(key)
                fallback_parents[level][key].add(row["code"])
                fallback_split[level][key][row["split"]] += 1
    summary = {
        "identity": identity,
        "status": "complete",
        "counts": dict(counts),
        "strict_positive_parent_rows": len(rows),
        "strict_positive_contacts": sum(len(r["contacts"]) for r in rows),
        "split": split_summary,
        "fine_key_support": [],
        "errors": errors,
    }
    for key, n in sorted(fine.items(), key=lambda kv: (-kv[1], repr(kv[0]))):
        fallback = []
        for level, fallback_key in enumerate(fine_fallback[key]):
            fallback.append({
                "key": list(fallback_key),
                "parents": len(fallback_parents[level][fallback_key]),
                "calibration_contacts": fallback_split[level][fallback_key]["calibration"],
                "validation_contacts": fallback_split[level][fallback_key]["validation"],
            })
        summary["fine_key_support"].append({
            "key": list(key),
            "contacts": n,
            "parents": len(per_key_parents[key]),
            "calibration_contacts": per_key_split[key]["calibration"],
            "validation_contacts": per_key_split[key]["validation"],
            "fallback": fallback,
        })
    write(out / "summary.json", summary)
    write(manifest, {"identity": identity, "status": "complete"})
    print(json.dumps({k: v for k, v in summary.items() if k not in ("identity", "errors", "fine_key_support")}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("scan",))
    parser.add_argument("--out", type=Path, default=Path("/tmp/nci_ctx1_native"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shards", nargs="+", help="optional subset for a smoke scan")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    scan(args.out.resolve(), args.workers, args.shards)


if __name__ == "__main__":
    main()
