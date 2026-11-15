"""Shared CPU helpers for the input-consistency and failure diagnostics (d14 onward). Read-only.

Paths point at the shared data tree of the main checkout, because the worktree has no data/ directory.
Nothing here writes under data/. Only CSD_code, SMILES and geometry are read from tmQM; the QM property
fields (Electronic_E, Dispersion_E, Dipole_M, Metal_q, HL_Gap, HOMO/LUMO, Polarizability) are never touched.

Environment for every Python process that uses this module:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
"""
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]                     # worktree root (contains src/, refit/, research/)
DATA = Path(os.environ.get("JACOBIK_DATA", "/mnt/e/Dev/Projects/jacobik_2026-10-02_full/data"))
SCALED = DATA / "refit" / "frame_v1"
CLEAN = DATA / "ref"
CONSTRAINTS = DATA / "db" / "constraints_2026-10-18.json"

# Covalent radii (A) as in d03_bond_distortion.py; LIGHT is the set of ligand-capable elements.
COV = {"H": .31, "B": .84, "C": .76, "N": .71, "O": .66, "F": .57, "Si": 1.11, "P": 1.07, "S": 1.05,
       "Cl": 1.02, "Se": 1.20, "Br": 1.20, "I": 1.39, "As": 1.19, "Ge": 1.20, "Te": 1.38, "Sn": 1.39,
       "Sb": 1.39, "Bi": 1.48}
LIGHT = set(COV) - {"H"}

# Formal-charge windows (predeclared, reference-free). A SMILES formal charge outside the window is absurd.
# Light atoms: common windows (oxo O(-2), nitride N(-3) and the like are inside). Metals: the highest common
# oxidation state written as a formal charge in tmQM-style SMILES; the lowest is -2 (metal anions are rare).
LIGHT_QWIN = {"C": (-1, 1), "N": (-3, 2), "O": (-2, 1), "S": (-2, 2), "Se": (-2, 2), "Te": (-2, 2),
              "P": (-3, 3), "As": (-3, 3), "B": (-3, 1), "F": (-1, 1), "Cl": (-1, 1), "Br": (-1, 1), "I": (-1, 1),
              "Si": (-2, 2), "Ge": (-2, 2), "Sn": (-2, 2), "Sb": (-3, 5), "Bi": (-3, 5)}
MAX_OX = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Sc": 3, "Y": 3,
          "La": 3, "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5, "Cr": 6, "Mo": 6, "W": 6, "Mn": 7,
          "Re": 7, "Fe": 6, "Ru": 8, "Os": 8, "Co": 5, "Rh": 6, "Ir": 6, "Ni": 4, "Pd": 4, "Pt": 6, "Cu": 4,
          "Ag": 3, "Au": 5, "Zn": 2, "Cd": 2, "Hg": 2, "Al": 3, "Ga": 3, "In": 3, "Tl": 3, "Sn": 4, "Pb": 4,
          "Bi": 5, "Sb": 5}
MAX_OX_DEFAULT = 4          # lanthanides (other than the listed ones) and the rest of the table


def absurd_light(sym, q):
    lo, hi = LIGHT_QWIN.get(sym, (-2, 2))
    return q < lo or q > hi


def absurd_metal(sym, q):
    return q > MAX_OX.get(sym, MAX_OX_DEFAULT) or q < -2


def frame_index():
    """code -> {split, shard, dataset, stratum} from frame.json (representative code of each group)."""
    f = json.load(open(SCALED / "frame.json"))
    out = {}
    for split, groups in f["main_allocation"]["splits"].items():
        for g in groups:
            out[g["representative_code"]] = dict(split=split, shard=g["representative_shard"],
                                                 dataset=g["dataset"], stratum=g.get("stratum"))
    return out


def pool_parents():
    """Codes with a successful control pool in calibration or motion_validation (manifest order)."""
    m = json.load(open(SCALED / "manifest.json"))
    seen = {}
    for key, rec in m["tasks"].items():
        if rec.get("status") == "success" and rec.get("split") in ("calibration", "motion_validation"):
            seen.setdefault(rec["code"], rec["split"])
    return seen


def load_parent(code, idx=None):
    """Assembled TMComplex for one parent, or None. Uses the frame's shard and dataset."""
    from jacobik.core.env import load_mol
    if idx is None:
        idx = frame_index()
    rec = idx.get(code)
    if rec is None:
        return None
    return load_mol(code, rec["shard"], rec["dataset"], clean_root=CLEAN)


def load_ctx(mol, seed=11):
    """Builder, pre-beam solver context and assignments for one seed, CPU, as in tools/jdev/multiseed.py."""
    from jacobik.core.env import build_builder_and_inputs
    from tools.jdev import multiseed
    from jacobik.stats.stats import StatsLookup, load_db
    builder, _, err = build_builder_and_inputs(mol)
    if builder is None:
        return None, None, None, err
    stats = StatsLookup(load_db(str(CONSTRAINTS)))
    ctx, asg, W, _ = multiseed.prepare(builder, seed, stats=stats, device="cpu", angle_bends=True)
    return builder, ctx, (asg, W), None


def to_json(obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    return json.dumps(obj, indent=1, default=conv)
