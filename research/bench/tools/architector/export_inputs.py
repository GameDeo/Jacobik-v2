"""Export Architector inputs for the tmQM calibration/tail parents and the DOE aqua fixtures (root jacobik env).

Chemistry comes from the jgen bench's own loaders (tools.jgen.inputs.load_mol + jacobik.core.env
.build_builder_and_inputs), so Architector receives the metal, oxidation state, ligand SMILES, donor indices, total charge
and spin that jacobik receives. tmQM is read only through those loaders (CSD_code, SMILES, geometry).

Conventions (checked against the code, see README):
  jacobik spin_state = multiplicity (tmQM S + 1); Architector full_spin / metal_spin = unpaired electrons = mult - 1.
  Architector coreType names are mapped from jacobik coord_geom labels; "matched" needs an exact CN-consistent name.

Usage (root env, PYTHONPATH=src:.:research/bench, cwd = worktree):
    python -m tools.architector.export_inputs --run-dir <run dir> [--codes CODE,...] [--doe-dir <dir>]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
DOE_FIXTURES = REPO / "tests/regression/fixtures/doe_aqua"
#: Full DOE aqua set (owner re-download, DOI 10.25584/3019314): the same layout research/archive/code/refit/doe_aqua.py reads.
DOE_FULL_GLOB = str(REPO / "data" / "10.25584 - 3019314" / "*" / "opt-xyz")
TMQM_PANEL_SOURCE = REPO / "research/bench/benchmarks/panels/tmqm_cal_tail_120.json"
TAIL_CODES = ["DAMVAQ", "HOXJUZ", "IFUROQ", "JEGPUJ", "NEPNEF", "SOBHAS", "USAPOU", "XEYTUT", "YUCZUS", "ZELTAP"]
EXCLUSIONS = REPO / "research/bench/benchmarks/locked_exclusions.json"
MANIFEST = REPO / "data/refit/frame_v1/manifest.json"

#: jacobik coord_geom label -> Architector coreType (architector 0.0.10 geometry names).
JACOBIK_TO_ARCH = {
    "linear": "linear", "trigonal": "trigonal_planar", "t_shaped": "t_shaped", "square_plan": "square_planar",
    "tetrahedral": "tetrahedral", "see_saw": "seesaw", "square_pyr": "square_pyramidal",
    "trig_bipyr": "trigonal_bipyramidal", "octahedral": "octahedral", "trig_pris": "trigonal_prismatic",
    "pent_bipyr": "pentagonal_bipyramidal", "square_anti": "square_antiprismatic",
    "tricap_pris": "tricapped_trigonal_prismatic",
}
#: Architector cn_geo_dict (0.0.10): which coreTypes are valid for each coreCN. Used to refuse a CN-inconsistent match.
ARCH_CN_GEO = {
    1: ["single"], 2: ["bent_109", "bent_120", "linear"], 3: ["t_shaped", "trigonal_planar", "trigonal_pyramidal"],
    4: ["seesaw", "square_planar", "tetrahedral"],
    5: ["pentagonal_planar", "square_pyramidal", "trigonal_bipyramidal"],
    6: ["hexagonal_planar", "octahedral", "pentagonal_pyramidal", "trigonal_prismatic"],
    7: ["capped_octahedral", "capped_trigonal_prismatic", "hexagonal_pyramidal", "pentagonal_bipyramidal"],
    8: ["axial_bicapped_trigonal_prismatic", "bicapped_trigonal_prismatic", "dodecahedral", "hexagonal_bipyramidal",
        "square_antiprismatic", "square_prismatic"],
    9: ["capped_square_antiprismatic", "tri_tri_mer_capped", "tricapped_trigonal_prismatic"],
    10: ["axial_bicapped_hexagonal_planar"], 11: [], 12: ["penta_bi_capped_pyramidal"],
}
#: DOE geometry per CN, as jacobik's own DOE driver uses it (tests/regression/run_panel.py GEOM).
DOE_GEOM = {8: "square_anti", 9: "tricap_pris", 6: "octahedral"}
#: DOE spin table, as in research/archive/code/refit/doe_aqua.py: multiplicity per metal (Fe keyed by oxidation state).
DOE_SPIN = {"La": 1, "Ce": 2, "Pr": 3, "Nd": 4, "Pm": 5, "Sm": 6, "Eu": 7, "Gd": 8, "Tb": 7, "Dy": 6, "Ho": 5,
            "Er": 4, "Tm": 3, "Yb": 2, "Lu": 1, "Fe2": 5, "Fe3": 6}
SEED_DEFAULT = 11
ARCH_PARAMETERS = {  # the catalog's parameter set, with Architector's own defaults for conformer count (1) and symmetries (10)
    "test_alternate_metal_spin": False, "n_conformers": 1, "n_symmetries": 10, "relax": True, "save_init_geos": True,
    "return_timings": True, "assemble_method": "GFN2-xTB", "full_method": "GFN2-xTB", "crest_sampling": False,
    "xtb_solvent": "none", "force_generation": False,
}


def _load_exclusions():
    data = json.loads(EXCLUSIONS.read_text())
    return set(data["reserve_all_codes"]) | set(data["motion_validation_codes"]), data


def _cn_from_geo(label_arch, cn):
    return label_arch is not None and label_arch in ARCH_CN_GEO.get(cn, [])


def ligand_items(lig_in):
    """Architector ligand dicts from jacobik LigandInput objects; returns (items, haptic_reason or None)."""
    items = []
    for li in lig_in:
        sites = [list(map(int, s)) for s in li.donors]
        if any(len(s) != 1 for s in sites):
            return None, "haptic or multi-atom site (eta/pi group): Architector direct adapter not expressible"
        coord = [s[0] for s in sites]
        item = {"smiles": li.smiles, "coordList": coord}
        if len(coord) == 1:
            item["ligType"] = "mono"
        items.append(item)
    return items, None


def arch_spec(metal, ox, charge, mult, items, core):
    unpaired = int(mult) - 1
    params = dict(ARCH_PARAMETERS)
    params.update({"metal_ox": int(ox), "metal_spin": unpaired, "full_charge": int(charge), "full_spin": unpaired,
                   "seed": SEED_DEFAULT})
    return {"core": core, "ligands": items, "parameters": params}


def cores(metal, cn, jacobik_geom):
    """Both input modes. native = CN only; matched = jacobik's crystal geometry label as coreType (None if no name)."""
    native = {"metal": metal, "coreCN": int(cn)}
    arch_name = JACOBIK_TO_ARCH.get(jacobik_geom)
    if arch_name is not None and _cn_from_geo(arch_name, int(cn)):
        return native, {"metal": metal, "coreType": arch_name}, None
    return native, None, f"no Architector coreType for jacobik geometry {jacobik_geom!r} at CN {cn}"


def tmqm_parent(code, shard, frame_shard_path=None):
    from tools.jgen import inputs
    from jacobik.core.env import build_builder_and_inputs
    mol = inputs.load_mol(code, shard, "tmQM")
    if mol is None:
        return None, f"no geometry in tmQM shard {shard}"
    builder, lig_in, err = build_builder_and_inputs(mol)
    if builder is None:
        return None, f"jgen builder failed: {err}"
    return (mol, builder, lig_in), None


def tmqm_record(code, shard):
    loaded, err = tmqm_parent(code, shard)
    if loaded is None:
        return None, {"case": code, "panel": "", "status": "capability", "reason": err}
    mol, builder, lig_in = loaded
    items, why = ligand_items(lig_in)
    if why:
        return None, {"case": code, "status": "capability", "reason": why}
    cn = sum(len(it["coordList"]) for it in items)
    natural, matched, nomatch = cores(builder.metal, cn, builder.geometry)
    ref_symbols = [a.symbol for a in mol.atoms]
    ref_X = [list(map(float, a.position)) for a in mol.atoms]
    # jacobik TMComplex indices (metal_idx, coord_sites.donor_indices, ligand donor_atoms) are 1-based.
    metal0 = int(mol.metal_idx) - 1
    donors = [int(d) - 1 for site in mol.coord_sites for d in site.donor_indices]
    assert ref_symbols[metal0] == builder.metal, (ref_symbols[metal0], builder.metal)
    spec_common = dict(case=code, metal=builder.metal, oxidation_state=int(builder.oxidation_state),
                       charge=int(builder.charge), mult=int(builder.spin), unpaired=int(builder.spin) - 1,
                       jacobik_geometry=builder.geometry, coordination_number=cn,
                       charge_source="tmQM xyz comment q= (tools.jgen.inputs.load_mol)",
                       spin_source="tmQM xyz comment S= (+1 = multiplicity, parser.py)",
                       ox_source=str(builder.oxidation_source), ligands=items,
                       reference={"symbols": ref_symbols, "X": ref_X, "metal_index": metal0,
                                  "donor_indices": donors, "source": f"tmQM geometry shard {shard}"})
    modes = {"native": {"status": "direct", "spec": arch_spec(builder.metal, builder.oxidation_state, builder.charge,
                                                              builder.spin, items, natural)}}
    if matched is not None:
        modes["matched"] = {"status": "direct", "spec": arch_spec(builder.metal, builder.oxidation_state,
                                                                 builder.charge, builder.spin, items, matched)}
    else:
        modes["matched"] = {"status": "capability", "reason": nomatch}
    return dict(spec_common, modes=modes), None


def doe_record(path):
    name = Path(path).stem  # Ce3_9H2O, Fe2_6H2O
    m = re.match(r"([A-Za-z]+)(\d)_(\d+)H2O", name)
    metal, ox, cn = m.group(1), int(m.group(2)), int(m.group(3))
    lines = Path(path).read_text().splitlines()
    syms, X = [], []
    for ln in lines[2:]:
        p = ln.split()
        if len(p) >= 4:
            syms.append(p[0]); X.append([float(v) for v in p[1:4]])
    X = np.asarray(X, float)
    metal_idx = next(i for i, s in enumerate(syms) if s == metal)
    d = np.linalg.norm(X - X[metal_idx], axis=1)
    donors = [i for i, s in enumerate(syms) if s == "O" and d[i] < 3.2]
    donors = sorted(donors, key=lambda i: d[i])[:cn]
    mult = DOE_SPIN[metal] if metal not in ("Fe",) else DOE_SPIN[f"Fe{ox}"]
    items = [{"smiles": "O", "coordList": [0]} for _ in range(cn)]
    jgeom = DOE_GEOM.get(cn)
    natural, matched, nomatch = cores(metal, cn, jgeom)
    spec_common = dict(case=f"aqua_{name}", metal=metal, oxidation_state=ox, charge=ox, mult=int(mult),
                       unpaired=int(mult) - 1, jacobik_geometry=jgeom, coordination_number=cn,
                       charge_source="water ligands neutral; charge = oxidation state", spin_source="DOE_SPIN table",
                       ox_source="from fixture name", ligands=items,
                       reference={"symbols": syms, "X": X.tolist(), "metal_index": int(metal_idx),
                                  "donor_indices": [int(i) for i in donors], "source": f"DOE fixture {Path(path).name}"})
    modes = {"native": {"status": "direct", "spec": arch_spec(metal, ox, ox, mult, items, natural)}}
    if matched is not None:
        modes["matched"] = {"status": "direct", "spec": arch_spec(metal, ox, ox, mult, items, matched)}
    else:
        modes["matched"] = {"status": "capability", "reason": nomatch}
    return dict(spec_common, modes=modes)


def doe_paths():
    """The 8 local fixtures, or the full owner set when it is on disk (same layout as research/archive/code/refit/doe_aqua.py)."""
    full = sorted(glob.glob(os.path.join(DOE_FULL_GLOB, "Ln-xyz", "Ln-H2O_x", "*.xyz")))
    full += sorted(glob.glob(os.path.join(DOE_FULL_GLOB, "misc-metal-xyz", "metal-H2O_x", "*.xyz")))
    if full:
        return full, "full DOE aqua set"
    return sorted(str(p) for p in DOE_FIXTURES.glob("*.xyz")), "8 local fixtures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--codes", default="", help="comma list of tmQM codes; default: the 120-code calibration panel")
    ap.add_argument("--skip-doe", action="store_true")
    ap.add_argument("--skip-tmqm", action="store_true", help="Amendment 1: DOE only (no tmQM cal/tail)")
    a = ap.parse_args()
    run = Path(a.run_dir)
    locked, excl = _load_exclusions()
    panel_codes = json.loads(TMQM_PANEL_SOURCE.read_text())["codes"]
    if a.skip_tmqm:
        panel_codes = []
    if a.codes:
        panel_codes = [c for c in a.codes.split(",") if c]
    assert not (set(panel_codes) & locked), "locked/val/reserve code in the requested tmQM panel"
    records = []
    if panel_codes:
        from tools.jgen import inputs
        shards = inputs.frame_shards()
        for code in panel_codes:
            shard, ds = shards[code]
            rec, err = tmqm_record(code, shard)
            if rec is None:
                rec = err
            else:
                rec["panel"] = "tail" if code in TAIL_CODES else "cal"
            rec.setdefault("panel", "tail" if code in TAIL_CODES else "cal")
            records.append(rec)
            print(code, rec.get("status", "ok"), flush=True)
    if not a.skip_doe:
        paths, source = doe_paths()
        for p in paths:
            rec = doe_record(p)
            rec["panel"] = "doe_aqua"
            records.append(rec)
    out_root = run / "inputs"
    capability = []
    counts = {}
    for rec in records:
        panel = rec["panel"]
        if rec.get("status") == "capability":
            capability.append(rec)
            continue
        d = out_root / panel
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{rec['case']}.json").write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
        counts[panel] = counts.get(panel, 0) + 1
        for mode, body in rec["modes"].items():
            if body["status"] == "capability":
                capability.append({"case": rec["case"], "panel": panel, "mode": mode, "status": "capability",
                                   "reason": body["reason"]})
    (run / "capability_raw.json").write_text(json.dumps(capability, indent=1) + "\n")
    print("exported", counts, "capability records", len(capability))


if __name__ == "__main__":
    main()
