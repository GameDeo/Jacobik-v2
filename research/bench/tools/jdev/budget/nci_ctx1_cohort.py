"""Parent-disjoint, chemistry-stratified cohort selection for NCI-CTX1.

Research-only companion to nci_ctx1.py. It reads the frozen shard records,
selects manageable calibration/validation parents, and writes its manifest
under /tmp by default. It does not edit the deployed scorer or parameter DB.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).parent)]

DEFAULT_SCAN = Path("/tmp/nci_ctx1_native_partial")
DEFAULT_OUT = DEFAULT_SCAN / "cohort"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parent_split(code: str, seed: int, calibration_fraction: float) -> str:
    digest = hashlib.sha256(f"nci-ctx1-parent-v{seed}:{code}".encode()).digest()
    return "calibration" if int.from_bytes(digest[:8], "big") / 2**64 < calibration_fraction else "validation"


def load_rows(scan: Path) -> tuple[list[dict], dict]:
    summary = json.loads((scan / "summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("scan summary is not complete")
    rows = []
    for shard in summary["identity"]["shards"]:
        obj = json.loads((scan / f"{shard}.json").read_text())
        rows.extend(obj["positive_parents"])
    return rows, summary


def eligible(row: dict, max_atoms: int, max_partner_heavy: int) -> bool:
    if row["n_atoms"] > max_atoms:
        return False
    return max(p["heavy_atoms"] for c in row["contacts"] for p in c["partners"]) <= max_partner_heavy


def _parent_keys(row: dict) -> set[tuple[str, ...]]:
    return {tuple(c["keys"][2]) for c in row["contacts"]}


def _tie(code: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"nci-ctx1-sample-v{seed}:{code}".encode()).digest()[:8], "big")


def greedy_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Maximize common level-2 key and dataset coverage without duplicate parents."""
    if len(rows) < n:
        raise ValueError(f"need {n} eligible parents, only have {len(rows)}")
    key_parents: dict[tuple[str, ...], set[str]] = defaultdict(set)
    dataset_parents: dict[str, set[str]] = defaultdict(set)
    row_keys = {r["code"]: _parent_keys(r) for r in rows}
    for r in rows:
        dataset_parents[r["dataset"]].add(r["code"])
        for key in row_keys[r["code"]]:
            key_parents[key].add(r["code"])

    # Only strata with real parent support enter the coverage objective. Rare
    # exact chemistries remain available through the recorded fallback ladder.
    common = [k for k, codes in key_parents.items() if len(codes) >= max(12, n // 4)]
    common.sort(key=lambda k: (-len(key_parents[k]), repr(k)))
    common = common[:16]
    max_support = max((len(key_parents[k]) for k in common), default=1)
    key_target = {
        k: min(max(4, n // 4), max(3, round((n // 4) * (len(key_parents[k]) / max_support) ** 0.5)))
        for k in common
    }
    dataset_target = {d: max(1, round(n * len(codes) / len(rows))) for d, codes in dataset_parents.items()}
    # Correct rounding so requested sample size remains fixed.
    while sum(dataset_target.values()) > n:
        d = max(dataset_target, key=lambda x: dataset_target[x] - n * len(dataset_parents[x]) / len(rows))
        dataset_target[d] -= 1
    while sum(dataset_target.values()) < n:
        d = max(dataset_parents, key=lambda x: n * len(dataset_parents[x]) / len(rows) - dataset_target[x])
        dataset_target[d] += 1

    selected: list[dict] = []
    remaining = {r["code"]: r for r in rows}
    have_key: Counter = Counter()
    have_dataset: Counter = Counter()
    while len(selected) < n:
        best_code, best_score = None, float("-inf")
        for code, row in remaining.items():
            key_gain = sum(
                max(0, key_target[k] - have_key[k]) / key_target[k]
                for k in row_keys[code] if k in key_target
            )
            ds = row["dataset"]
            ds_gain = max(0, dataset_target[ds] - have_dataset[ds]) / dataset_target[ds]
            score = key_gain + 0.2 * ds_gain
            # Stable, tiny tie-break; independent of dictionary/file order.
            score += (_tie(code, seed) / 2**64) * 1e-9
            if score > best_score:
                best_code, best_score = code, score
        row = remaining.pop(best_code)
        selected.append(row)
        have_dataset[row["dataset"]] += 1
        have_key.update(row_keys[best_code])
    return selected


def summarize(rows: list[dict], selected: dict[str, list[dict]], assignments: dict[str, str]) -> dict:
    by_code = {r["code"]: r for r in rows}
    key_support = {}
    for split in ("calibration", "validation"):
        parents: dict[tuple[str, ...], set[str]] = defaultdict(set)
        contacts: Counter = Counter()
        for r in rows:
            if assignments[r["code"]] != split:
                continue
            for c in r["contacts"]:
                k = tuple(c["keys"][2])
                parents[k].add(r["code"])
                contacts[k] += 1
        key_support[split] = {
            "parents_by_level2_key": {"|".join(k): len(v) for k, v in parents.items()},
            "contacts_by_level2_key": {"|".join(k): contacts[k] for k in parents},
        }

    train_motifs = set()
    for r in rows:
        if assignments[r["code"]] != "calibration":
            continue
        for c in r["contacts"]:
            scaffolds = tuple(sorted(p["scaffold"] for p in c["partners"] if p["heavy_atoms"] > 2))
            train_motifs.add((tuple(c["keys"][2]), scaffolds))
    val_contacts = val_novel = 0
    for r in rows:
        if assignments[r["code"]] != "validation":
            continue
        for c in r["contacts"]:
            scaffolds = tuple(sorted(p["scaffold"] for p in c["partners"] if p["heavy_atoms"] > 2))
            val_contacts += 1
            val_novel += (tuple(c["keys"][2]), scaffolds) not in train_motifs

    return {
        "parent_counts": dict(Counter(assignments.values())),
        "dataset_counts": {
            s: dict(Counter(by_code[c]["dataset"] for c, v in assignments.items() if v == s))
            for s in ("calibration", "validation")
        },
        "key_support": key_support,
        "validation_contact_scaffold_motif_novelty": {
            "novel_contacts": val_novel,
            "all_contacts": val_contacts,
            "fraction": val_novel / max(val_contacts, 1),
            "definition": "level-2 chemistry key plus sorted participant Murcko scaffolds (>2 heavy atoms), absent from calibration positives",
        },
        "selected": {
            s: {
                "n_parents": len(selected[s]),
                "parents_by_dataset": dict(Counter(r["dataset"] for r in selected[s])),
                "contacts_by_level2_key": dict(Counter(
                    "|".join(tuple(c["keys"][2])) for r in selected[s] for c in r["contacts"]
                )),
                "codes": [r["code"] for r in selected[s]],
            }
            for s in ("calibration", "validation")
        },
    }


def select(scan: Path, out: Path, n_per_split: int, max_atoms: int, max_partner_heavy: int, seed: int) -> dict:
    rows, scan_summary = load_rows(scan)
    eligible_rows = [r for r in rows if eligible(r, max_atoms, max_partner_heavy)]
    assignments = {r["code"]: parent_split(r["code"], seed, 0.60) for r in eligible_rows}
    selected = {
        split: greedy_sample([r for r in eligible_rows if assignments[r["code"]] == split], n_per_split, seed + i)
        for i, split in enumerate(("calibration", "validation"))
    }
    payload = {
        "schema": "nci_ctx1_parent_cohort_v1",
        "scope": "26 completed tmQM shards; 3 final X3 shards were stopped at one expensive resonance enumeration and are not counted as negatives",
        "scan_shards": scan_summary["identity"]["shards"],
        "scan_identity": scan_summary["identity"],
        "scan_counts": scan_summary["counts"],
        "cohort_rule": {
            "parent_split": f"SHA256 parent-code split, seed={seed}, 60% calibration target; all records from a parent stay in one split",
            "max_complex_atoms": max_atoms,
            "max_participant_heavy_atoms": max_partner_heavy,
            "sample_size_per_split": n_per_split,
            "sample_objective": "greedy max-coverage of common level-2 typed chemistry keys, with tmQM X1/X2/X3 balance and stable hash tie-break",
        },
        "source_sha256": sha256(scan / "summary.json"),
        "driver_sha256": sha256(Path(__file__)),
        "eligible_parent_count": len(eligible_rows),
        "eligible_parent_counts_by_split": dict(Counter(assignments.values())),
        "excluded_by_size": len(rows) - len(eligible_rows),
        "summary": summarize(eligible_rows, selected, assignments),
        "assignments": assignments,
        "selected_rows": selected,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "selection.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def _case_from_mol(mol):
    from jacobik import ComplexBuilder
    from tools.jdev.cases import Case
    from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol
    import numpy as np

    groups = _build_ligand_groups(mol, mol.smiles)
    inputs = ligand_inputs_from_mol(mol, groups)
    if inputs is None:
        raise ValueError(f"{mol.csd_code}: no complete ligand input mapping")
    def make_builder(m=mol, ligands=inputs):
        return ComplexBuilder(metal=m.metal_symbol, geometry=m.coord_geom, charge=m.charge,
                              spin=m.spin_state, ligands=ligands)
    atoms = list(mol.atoms)
    return Case(name=mol.csd_code, make_builder=make_builder, charge=mol.charge, mult=mol.spin_state,
                ref_symbols=[a.symbol for a in atoms], ref_X=np.asarray([a.position for a in atoms], dtype=np.float64),
                ref_metal_row=next(i for i, a in enumerate(atoms) if a.idx == mol.metal_idx),
                tmqm_mol=mol, ligand_groups=groups, ligand_inputs=inputs)


def _load_cases(scan: Path, selected_rows: dict[str, list[dict]]) -> dict[str, object]:
    import logging
    from fit.parser import TMQMParser, _assemble_raw, _attach_qm_properties, _iter_bo_blocks, _iter_xyz_blocks
    from jacobik.topology.mapping import assemble
    from rdkit import RDLogger

    logging.getLogger("jacobik.topology.mapping").setLevel(logging.ERROR)
    logging.getLogger("fit.parser").setLevel(logging.ERROR)
    RDLogger.DisableLog("rdApp.*")
    wanted: dict[str, set[str]] = defaultdict(set)
    for rows in selected_rows.values():
        for row in rows:
            wanted[row["shard"]].add(row["code"])
    parser = TMQMParser(scan_summary_data_dir(scan))
    out = {}
    for shard, codes in sorted(wanted.items()):
        xyz = parser.data_dir / "xyz" / f"tmQM_{shard}.xyz"
        bo = parser.data_dir / "bo" / f"tmQM_{shard}.BO"
        found = set()
        for (comment, raw_atoms), (code, raw_bonds) in zip(_iter_xyz_blocks(xyz), _iter_bo_blocks(bo)):
            if code not in codes:
                continue
            csv_row = parser.csv_index.get(code)
            mol = assemble(_assemble_raw(comment, raw_atoms, code, raw_bonds, csv_row))
            if csv_row is not None:
                _attach_qm_properties(mol, csv_row)
            out[code] = _case_from_mol(mol)
            found.add(code)
        missing = sorted(codes - found)
        if missing:
            raise KeyError(f"{shard}: selected parents not found: {missing}")
    return out


def scan_summary_data_dir(scan: Path) -> Path:
    # Current corpus source is fixed in the partial scan identity; recover its
    # repo path rather than relying on the caller's working directory.
    p = ROOT / "data/refit/clean/tmQM"
    if not (p / "tmQM_y.csv").is_file():
        raise FileNotFoundError(p / "tmQM_y.csv")
    return p


def _engine_to_tmqm(case, engine_complex, X_engine):
    """Reorder solved coordinates to the authoritative training graph's atom order."""
    from rdkit import Chem
    import numpy as np

    source = case.tmqm_mol
    raw_atoms = list(source.atoms)
    raw_pos = {a.idx: i for i, a in enumerate(raw_atoms)}
    raw_by_idx = {a.idx: a for a in raw_atoms}
    raw_neighbors: dict[int, list[int]] = defaultdict(list)
    for b in source.bonds:
        if b.order < 0.5:
            continue
        raw_neighbors[b.i].append(b.j)
        raw_neighbors[b.j].append(b.i)
    raw_to_engine: dict[int, int] = {source.metal_idx: engine_complex.metal_idx}
    ligands = sorted(engine_complex.ligands, key=lambda l: l.ligand_id)
    if len(ligands) != len(case.ligand_inputs) or len(ligands) != len(case.ligand_groups):
        raise ValueError(f"{case.name}: engine/input ligand count mismatch")
    for lig, li, group in zip(ligands, case.ligand_inputs, case.ligand_groups):
        rd = Chem.AddHs(Chem.MolFromSmiles(li.smiles))
        if rd.GetNumAtoms() != len(lig.atom_indices):
            raise ValueError(f"{case.name}: RDKit/engine atom count mismatch for ligand {lig.ligand_id}")
        rd_to_raw = group["rdkit_to_xyz"]
        for rdi, raw_idx in rd_to_raw.items():
            raw_idx = int(raw_idx)
            raw_to_engine[raw_idx] = lig.atom_indices[int(rdi)]
        for parent_rdi in rd_to_raw:
            parent_raw = int(rd_to_raw[parent_rdi])
            rd_h = sorted(n.GetIdx() for n in rd.GetAtomWithIdx(int(parent_rdi)).GetNeighbors()
                          if n.GetAtomicNum() == 1)
            raw_h = sorted(i for i in raw_neighbors.get(parent_raw, ()) if raw_by_idx[i].symbol == "H")
            if len(rd_h) != len(raw_h):
                raise ValueError(f"{case.name}: hydrogen mapping mismatch at raw atom {parent_raw}")
            for rdh, raw_idx in zip(rd_h, raw_h):
                raw_to_engine[raw_idx] = lig.atom_indices[rdh]
    if set(raw_to_engine) != set(raw_pos):
        missing = sorted(set(raw_pos) - set(raw_to_engine))
        extra = sorted(set(raw_to_engine) - set(raw_pos))
        raise ValueError(f"{case.name}: incomplete atom-order mapping (missing={missing[:5]}, extra={extra[:5]})")
    X_engine = np.asarray(X_engine, dtype=np.float64)
    X_raw = np.empty((len(raw_atoms), 3), dtype=np.float64)
    for raw_idx, engine_idx in raw_to_engine.items():
        raw = raw_by_idx[raw_idx]
        eng = engine_complex.atoms[engine_idx - 1]
        if raw.symbol != eng.symbol:
            raise ValueError(f"{case.name}: atom-order element mismatch at {raw_idx}: {raw.symbol}/{eng.symbol}")
        X_raw[raw_pos[raw_idx]] = X_engine[engine_idx - 1]
    return X_raw


def solve(scan: Path, out: Path, selection_path: Path, seed: int, limit: int, splits: list[str], device: str = "cpu") -> None:
    import numpy as np
    import torch
    from jacobik.core import profile
    from tools.jdev.start import starting_geometry
    from tools.jdev.budget.common import stats_from_db

    if profile.ACTIVE_NAME != "u1" or profile.get("q6_model") != "repulsive" or profile.get("nci"):
        raise RuntimeError(f"baseline must be u1 + repulsive Q6 + NCI off; got {profile.snapshot()}")
    torch.set_num_threads(1)
    selection = json.loads(selection_path.read_text())
    selected_all = selection["selected_rows"]
    todo = [(split, row) for split in splits for row in selected_all[split]]
    if limit:
        todo = todo[:limit]
    selected = {s: [r for split, r in todo if split == s] for s in set(s for s, _ in todo)}
    cases = _load_cases(scan, selected)
    db = ROOT / "data/refit/final/constraints_2026-10-18.json"
    stats = stats_from_db(db)
    profile_record = json.dumps(profile.snapshot(), sort_keys=True, default=str)
    for k, (split, row) in enumerate(todo, 1):
        code = row["code"]
        target = out / "pools" / "q6_u1rep" / split / f"{code}_s{seed}.npz"
        if target.exists():
            print(f"[{split}] {k}/{len(todo)} {code}: exists; skip", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            case = cases[code]
            with starting_geometry(case, "etkdg"):
                result = case.make_builder().solve(seed=seed, stats=stats, angle_bends=True, device=device)
            confs = result.conformers
            X_engine = np.asarray([[a.position for a in c.complex.atoms] for c in confs], dtype=np.float64)
            X_raw = np.stack([_engine_to_tmqm(case, c.complex, X) for c, X in zip(confs, X_engine)])
            sy = np.asarray([a.symbol for a in confs[0].complex.atoms])
            if len(sy) != len(case.ref_symbols):
                raise ValueError(f"{code}: solved/reference atom-count mismatch")
            np.savez_compressed(target, X=X_engine, X_raw=X_raw, symbols=sy,
                                score=np.asarray([c.score for c in confs], dtype=np.float64),
                                ref_X=case.ref_X, ref_symbols=np.asarray(case.ref_symbols),
                                charge=case.charge, mult=case.mult, seed=seed,
                                t_solve=float(result.elapsed_s) if hasattr(result, "elapsed_s") else float("nan"),
                                profile=profile_record, start="etkdg", device=device)
            print(f"[{split}] {k}/{len(todo)} {code}: {len(confs)} conformers", flush=True)
        except Exception as exc:
            np.savez_compressed(target, error=f"{type(exc).__name__}: {str(exc)[:500]}")
            print(f"[{split}] {k}/{len(todo)} {code}: ERROR {type(exc).__name__}: {str(exc)[:160]}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", nargs="?", choices=("select", "solve"), default="select")
    p.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--selection", type=Path, default=None)
    p.add_argument("--n-per-split", type=int, default=48)
    p.add_argument("--max-atoms", type=int, default=120)
    p.add_argument("--max-partner-heavy", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                   help="device for the differentiable IK solve (default: cpu)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--splits", nargs="+", choices=("calibration", "validation"), default=["calibration", "validation"])
    a = p.parse_args()
    scan_path, out_path = a.scan.resolve(), a.out.resolve()
    if a.stage == "select":
        payload = select(scan_path, out_path, a.n_per_split, a.max_atoms, a.max_partner_heavy, a.seed)
        s = payload["summary"]
        print(json.dumps({
            "eligible": payload["eligible_parent_count"],
            "eligible_by_split": payload["eligible_parent_counts_by_split"],
            "selected": {k: {"n": v["n_parents"], "datasets": v["parents_by_dataset"]} for k, v in s["selected"].items()},
            "novel_motif_fraction": s["validation_contact_scaffold_motif_novelty"]["fraction"],
            "selection": str(out_path / "selection.json"),
        }, indent=2), flush=True)
    else:
        from jacobik.core import profile
        if profile.ACTIVE_NAME != "u1" or profile.get("nci"):
            raise SystemExit("run solve with JACOBIK_PROFILE=u1 JACOBIK_PROFILE_OVERRIDES='{" + '"nci":false' + "}'")
        selection_path = (a.selection or (out_path / "selection.json")).resolve()
        solve(scan_path, out_path, selection_path, a.seed, a.limit, a.splits, a.device)


if __name__ == "__main__":
    main()
