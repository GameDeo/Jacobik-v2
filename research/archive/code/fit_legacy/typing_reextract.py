"""Re-extract Q1/Q2/S_orient data under T1 typing flag settings, without fitting."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Paths are set by main() (CLI) or by the caller through configure(); nothing is hard-coded.
DATA = None    # data root (contains ref/)
OUT = None     # output folder (<OUT>/<mode>/shards)
FRAME = None   # frozen fit frame json (calibration parents)
LOCKED = None  # locked_exclusions.json
EXTRA_CODES = ()  # parents added to the calibration representatives (the dev panels pass the tail codes)


def configure(data_root, out_dir, frame, locked, extra_codes=()):
    global DATA, OUT, FRAME, LOCKED, EXTRA_CODES
    DATA, OUT, FRAME, LOCKED = Path(data_root), Path(out_dir), Path(frame), Path(locked)
    EXTRA_CODES = tuple(extra_codes)


SPLITS = {"X1": "tmQM", "X2": "tmQM", "X3": "tmQM", "LNf": "lnQM"}
MODES = {"off": {}, "gate": {"hapto_bond_gate": True},
         "both": {"hapto_bond_gate": True, "hapto_label_canon": True}}
T1D_BASE_FLAGS = {"hapto_bond_gate": True, "hapto_label_canon": True, "hapto_chelate_exclusion": True}
ALL_MODES = {**MODES, "t1b": T1D_BASE_FLAGS, "t1c": T1D_BASE_FLAGS,
             "t1d_base": T1D_BASE_FLAGS}


def list_shards():
    from jacobik.fit.pipeline.extract import list_shards as _list

    return [(x.split, x.dataset, x.name) for x in _list(DATA / "ref")]


def target_codes(extra=()):
    """Calibration representatives of the frame plus ``extra`` (and the configured extra codes), minus locked sets."""
    frame = json.loads(FRAME.read_text())
    codes = {group["representative_code"]
             for group in frame["main_allocation"]["splits"]["calibration"]}
    codes.update(EXTRA_CODES)
    codes.update(extra)
    locked = json.loads(LOCKED.read_text())
    excluded = set(locked["reserve_all_codes"]) | set(locked["motion_validation_codes"])
    return codes - excluded


def _run_shard(args):
    split, dataset, shard, mode, parents = args
    if mode not in ALL_MODES:
        raise ValueError(mode)
    os.environ["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(ALL_MODES[mode], separators=(",", ":"))

    import logging
    import numpy as np
    import pandas as pd
    from jacobik.stats import key_builder as KB
    from jacobik.chem.elements import covalent_radius as _covalent_radius
    from jacobik.stats.orientation import ETA2_BOND_RATIO_MAX, HAPTIC_K0, fit_rows
    from jacobik.fit.parser import TMQMParser
    from jacobik.topology import donor_classifier as DC
    from jacobik.fit import q1, q2, qo

    logging.disable(logging.ERROR)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    declared = {}
    promotion_rows = []
    original_classify = DC.classify

    def capture_before_classification(mol, *a, **kw):
        if mol.csd_code in declared:
            return original_classify(mol, *a, **kw)
        before = {d for lig in mol.ligands for d in lig.donor_atoms}
        declared[mol.csd_code] = sorted(before)
        result = original_classify(mol, *a, **kw)
        all_haptic = {d for lig in mol.ligands for face in lig.haptic_groups for d in face.donor_indices}
        promoted = {d for lig in mol.ligands for d in lig.donor_atoms} - before - all_haptic
        if promoted:
            atom_map = {a.idx: a for a in mol.atoms}
            metal_pos = np.asarray(atom_map[mol.metal_idx].position, float)
            from jacobik.stats.orientation import ETA2_BOND_RATIO_MAX
            for donor in sorted(promoted):
                a = atom_map[donor]
                dist = float(np.linalg.norm(np.asarray(a.position, float) - metal_pos))
                ratio = dist / (_covalent_radius(mol.metal_symbol) + _covalent_radius(a.symbol))
                bonded_c = [b.j if b.i == donor else b.i for b in mol.bonds
                            if donor in (b.i, b.j) and atom_map[b.j if b.i == donor else b.i].symbol == "C"]
                pair = "C=" + a.symbol if bonded_c else "other"
                promotion_rows.append(dict(csd_code=mol.csd_code, atom_idx=donor, metal=mol.metal_symbol,
                    pair=pair, distance=dist, ratio=ratio, within_default_gate=ratio <= ETA2_BOND_RATIO_MAX))
        return result

    DC.classify = capture_before_classification
    parser = TMQMParser(DATA / "ref" / dataset)
    q1_rows, q2_rows, qo_rows, face_rows, parent_rows = [], [], [], [], []
    n_mol = n_error = 0
    start = time.perf_counter()
    for mol in parser.stream_split(shard):
        n_mol += 1
        try:
            a1, a2, ao = q1.extract_rows(mol), q2.extract_rows(mol), qo.extract_rows(mol)
            q1_rows.extend({k: v for k, v in r.items() if k != "key_tuple"} for r in a1)
            q2_rows.extend({k: v for k, v in r.items() if k != "key_tuple"} for r in a2)
            qo_rows.extend(ao)
            symbols = {a.idx: a.symbol for a in mol.atoms}
            atom_pos = {a.idx: np.asarray(a.position, float) for a in mol.atoms}
            metal_pos = atom_pos[mol.metal_idx]
            declared_ids = set(declared.get(mol.csd_code, ()))
            for row in ao:
                if int(row["k"]) < HAPTIC_K0 + 2:
                    continue
                face = np.asarray(row["raw_face"], float).reshape(-1, 3)
                elems = list(row["raw_face_elems"])
                ratios = [float(np.linalg.norm(x) / (_covalent_radius(mol.metal_symbol) + _covalent_radius(e)))
                          for x, e in zip(face, elems)]
                atom_ids = []
                for x, e in zip(face, elems):
                    matches = [i for i, p in atom_pos.items() if symbols[i] == e and
                               np.linalg.norm(p - (metal_pos + x)) < 1e-5]
                    atom_ids.append(matches[0] if len(matches) == 1 else None)
                pair = "=".join(sorted(elems)) if len(elems) == 2 else "other"
                face_rows.append(dict(csd_code=mol.csd_code, split=split, shard=shard,
                    k=int(row["k"]), eta=int(row["k"]) - HAPTIC_K0, elements="=".join(elems),
                    element_pair=pair, max_ratio=max(ratios), min_ratio=min(ratios),
                    removed_by_gate=bool(int(row["k"]) == HAPTIC_K0 + 2 and max(ratios) > ETA2_BOND_RATIO_MAX),
                    n_declared_donors=sum(i in declared_ids for i in atom_ids if i is not None),
                    face_atom_ids=json.dumps(atom_ids), raw_face=row["raw_face"], raw_face_elems=elems,
                    metal=mol.metal_symbol, value=float(np.linalg.norm(face.mean(0)))))
            site_labels = [KB._site_subclass(s, mol) for s in mol.coord_sites]
            parent_rows.append(dict(csd_code=mol.csd_code, split=split, shard=shard,
                donor_set=json.dumps(sorted({i for s in mol.coord_sites for i in s.donor_indices})),
                coordination_number=len(mol.coord_sites), site_labels=json.dumps(site_labels),
                geometry_label=mol.coord_geom, in_cal_tail=mol.csd_code in parents))
        except Exception as exc:  # keep the shard batch running; record each molecule-level failure
            n_error += 1
            print(f"{mode} {shard} {mol.csd_code}: {type(exc).__name__}: {exc}", flush=True)
    q1_df, q2_df, qo_df = pd.DataFrame(q1_rows), pd.DataFrame(q2_rows), pd.DataFrame(qo_rows)
    # This is the row table the QO fitter consumes, including per-substituent haptic bends.
    qo_fit = fit_rows(qo_df) if len(qo_df) else qo_df
    dest = OUT / mode / "shards"
    dest.mkdir(parents=True, exist_ok=True)
    q1_df.to_parquet(dest / f"{shard}_Q1.parquet", index=False)
    q2_df.to_parquet(dest / f"{shard}_Q2.parquet", index=False)
    qo_df.to_parquet(dest / f"{shard}_QO.parquet", index=False)
    qo_fit.to_parquet(dest / f"{shard}_S_orient.parquet", index=False)
    face_cols = ["csd_code", "split", "shard", "k", "eta", "elements", "element_pair", "max_ratio",
                 "min_ratio", "removed_by_gate", "n_declared_donors", "face_atom_ids", "raw_face",
                 "raw_face_elems", "metal", "value"]
    pd.DataFrame(face_rows, columns=face_cols).to_parquet(dest / f"{shard}_faces.parquet", index=False)
    pd.DataFrame(parent_rows).to_json(dest / f"{shard}_parents.jsonl", orient="records", lines=True)
    pd.DataFrame(promotion_rows, columns=["csd_code", "atom_idx", "metal", "pair", "distance", "ratio",
                                          "within_default_gate"]).to_parquet(
        dest / f"{shard}_promotions.parquet", index=False)
    elapsed = time.perf_counter() - start
    return dict(mode=mode, shard=shard, dataset=dataset, molecules=n_mol, errors=n_error + parser.n_failed,
                q1_rows=len(q1_df), q2_rows=len(q2_df), qo_rows=len(qo_df), s_orient_rows=len(qo_fit),
                faces=len(face_rows), parent_rows=len(parent_rows), elapsed_s=elapsed)


def run(mode, workers, selected_shard=None):
    if mode not in ALL_MODES:
        raise ValueError(f"mode must be one of {tuple(ALL_MODES)}")
    # Set before workers fork/spawn; the fit package lazy-loads quantity modules.
    os.environ["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(ALL_MODES[mode], separators=(",", ":"))
    parents = target_codes()
    shards = [x for x in list_shards() if selected_shard is None or x[2] == selected_shard]
    if selected_shard and not shards:
        raise SystemExit(f"unknown shard: {selected_shard}")
    args, skipped = [], []
    for sp, ds, sh in shards:
        shard_dir = OUT / mode / "shards"
        required = [shard_dir / f"{sh}_{kind}.parquet" for kind in ("Q1", "Q2", "QO", "S_orient", "faces")]
        required.extend([shard_dir / f"{sh}_parents.jsonl", shard_dir / f"{sh}_promotions.parquet"])
        if all(p.exists() for p in required):
            skipped.append((sp, ds, sh))
        else:
            args.append((sp, ds, sh, mode, parents))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if workers == 1:
        results = [_run_shard(x) for x in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_run_shard, args))
    summary = dict(mode=mode, workers=workers, n_shards=len(results), skipped_shards=len(skipped),
                   elapsed_s=time.perf_counter()-t0, shards=results)
    (OUT / mode / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "shards"}, indent=2))
    for r in results:
        print(json.dumps(r, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=["run"])
    ap.add_argument("--mode", required=True, choices=tuple(ALL_MODES))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--data-root", type=Path, required=True, help="data root containing ref/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--frame", type=Path, required=True, help="fit frame json")
    ap.add_argument("--locked", type=Path, required=True, help="locked_exclusions.json")
    ap.add_argument("--extra-codes", nargs="*", default=[], help="parents added to the calibration representatives")
    a = ap.parse_args()
    configure(a.data_root, a.out_dir, a.frame, a.locked, a.extra_codes)
    run(a.mode, a.workers, a.shard)


if __name__ == "__main__":
    main()
