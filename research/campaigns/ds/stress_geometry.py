"""Coherent geometry scenario for the stress driver: re-extract the affected parents from perturbed source coordinates.

The affected parents' xyz blocks are displaced by independent uniform-ball displacements of at most ``MAX_A`` (0.02 A, `confidence.perturb_coordinates`,
seeded from the trial id and the CSD code), written next to their unchanged bond-order blocks and CSV rows into a one-shard mini dataset, and run through the
production `extract_split` under the active build profile. The resulting observation rows replace the parents' rows in the retained table
(`confidence.perturb_observations`), so every quantity of the parent stays coherent. Observation ids are those the production extraction assigns
(data directory name, code, quantity, key, occurrence), so a perturbation that changes no key keeps its rows' ids.

    blocks = build_block_cache(ref_dir, codes)                 # one pass over the shards, before forking workers
    rows = geometry_replacements(G, trial, maximum_A=0.02)     # G holds raw, quantity, blocks
"""
from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

MAX_A = 0.02


def _seed(trial_id: str, code: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{trial_id}:{code}".encode()).digest()[:8], "big")


def build_block_cache(ref_dir, codes) -> dict:
    """{code: dict(dataset, comment, atoms=[(El, x, y, z)], bo=[lines], smiles)} for the wanted codes, from one pass over the shards."""
    from fit.parser import _iter_bo_blocks, _iter_xyz_blocks

    wanted, out = set(codes), {}
    for dataset in ("tmQM", "lnQM"):
        csv_path = Path(ref_dir) / dataset / "tmQM_y.csv"
        smiles = {}
        if csv_path.exists():
            for line in csv_path.read_text().splitlines()[1:]:
                code, _, smi = line.partition(";")
                if code in wanted:
                    smiles[code] = smi
        for xyz in sorted((Path(ref_dir) / dataset / "xyz").glob("tmQM_*.xyz")):
            bo = Path(ref_dir) / dataset / "bo" / (xyz.stem + ".BO")
            for (comment, atom_lines), (bo_code, bo_lines) in zip(_iter_xyz_blocks(xyz), _iter_bo_blocks(bo)):
                if bo_code in wanted:
                    atoms = [(f[0], *map(float, f[1:4])) for f in (ln.split() for ln in atom_lines)]
                    out[bo_code] = dict(dataset=dataset, comment=comment, atoms=atoms, bo=bo_lines, smiles=smiles.get(bo_code, ""))
    missing = wanted - set(out)
    if missing:
        raise KeyError(f"{len(missing)} affected parents not found in the source shards, e.g. {sorted(missing)[:3]}")
    return out


def perturbed_block(block: dict, trial_id: str, code: str, maximum_A: float = MAX_A):
    from fit.pipeline.confidence import perturb_coordinates

    xyz = np.array([a[1:] for a in block["atoms"]], float)
    new = perturb_coordinates(xyz, seed=_seed(trial_id, code), maximum_A=maximum_A) if maximum_A > 0 else xyz
    return [(a[0], *p) for a, p in zip(block["atoms"], new)]


def write_mini_dataset(root: Path, dataset: str, items: list, trial_id: str, maximum_A: float) -> Path:
    """<root>/<dataset>/{xyz,bo}/tmQM_T.* and tmQM_y.csv for the (code, block) items; returns the data directory."""
    d = root / dataset
    (d / "xyz").mkdir(parents=True)
    (d / "bo").mkdir()
    xyz, bo, csv = [], [], ["CSD_code;SMILES"]
    for code, block in items:
        atoms = perturbed_block(block, trial_id, code, maximum_A)
        xyz.append(f"{len(atoms)}\n{block['comment']}\n" + "\n".join(f"{e} {x:.10f} {y:.10f} {z:.10f}" for e, x, y, z in atoms) + "\n")
        bo.append(f"CSD_code = {code} | mini\n" + "\n".join(block["bo"]) + "\n")
        csv.append(f"{code};{block['smiles']}")
    (d / "xyz" / "tmQM_T.xyz").write_text("".join(xyz))
    (d / "bo" / "tmQM_T.BO").write_text("\n".join(bo) + "\n")
    (d / "tmQM_y.csv").write_text("\n".join(csv) + "\n")
    return d


def geometry_replacements(G: dict, trial: dict, maximum_A: float = MAX_A) -> pd.DataFrame:
    """Observation rows of the trial's affected parents re-extracted from perturbed coordinates."""
    from fit.extractor import extract_split
    from jacobik.stats.mixture import structure_id

    affected = set(trial["affected_parents"])
    codes = sorted({c for c in G["raw"]["csd_code"].astype(str).unique() if structure_id(c) in affected})
    q = G["quantity"]
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        by_dataset = {}
        for c in codes:
            by_dataset.setdefault(G["blocks"][c]["dataset"], []).append((c, G["blocks"][c]))
        for dataset, items in by_dataset.items():
            # the data directory must carry the dataset name: it enters the observation id
            d = write_mini_dataset(Path(tmp) / f"mini_{dataset}", dataset, items, trial["trial_id"], maximum_A)
            out = Path(tmp) / f"out_{dataset}"
            out.mkdir()
            extract_split(d, "T", out, progress=False, hypothesis=False, q6_hist=False, orientation=(q == "QO"))
            frames.append(pd.read_parquet(out / f"observations_{q}_T.parquet"))
    rows = pd.concat(frames, ignore_index=True)
    missing = affected - set(rows["csd_code"].astype(str).map(structure_id))
    if missing:
        raise ValueError(f"re-extraction produced no {q} rows for {len(missing)} affected parents")
    return rows[list(G["raw"].columns)]
