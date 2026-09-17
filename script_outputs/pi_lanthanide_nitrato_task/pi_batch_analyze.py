import sys, os, json, time
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, "/tmp/jscripts")

from pi_task_ligands import build_aqua, build_nitrato_bidentate
from jacobik.benchmark.geometry import kabsch

DATA = f"{ROOT}/data/10.25584 - 3019314/38435_fec372fd59c091ded705c5752c2ec652.tar"

def parse_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2+n]:
        parts = line.split()
        atoms.append((parts[0], np.array(list(map(float, parts[1:4])))))
    return atoms

def ref_inner_sphere_donors(path, cutoff=3.0):
    """Return metal pos + array of donor O positions that are truly
    inner-sphere (< cutoff A), matching how Jacobik defines coord_sites."""
    atoms = parse_xyz(path)
    metal_pos = atoms[0][1]
    o_atoms = [(i, s, p) for i, (s, p) in enumerate(atoms) if s == "O"]
    donors = np.array([p for i, s, p in o_atoms if np.linalg.norm(p - metal_pos) < cutoff])
    return metal_pos, donors

def rmsd_to_ref(donor_pos_from_metal, ref_donor_pos_from_metal):
    C = np.linalg.norm(donor_pos_from_metal[:, None, :] - ref_donor_pos_from_metal[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(C)
    rmsd, _ = kabsch(donor_pos_from_metal[ri], ref_donor_pos_from_metal[ci])
    return rmsd

def water_orientation_check(mol, metal_pos, angle_threshold=45.0):
    """For each water ligand, angle between (metal-O) and the H-O-H
    bisector should be ~180 deg (lone-pair region faces the metal).
    Flags anything off by more than angle_threshold from 180."""
    flags = []
    for lig in mol.ligands:
        if len(lig.atom_indices) != 3 or len(lig.donor_atoms) != 1:
            continue
        o_idx = lig.donor_atoms[0]
        h_idx = [i for i in lig.atom_indices if i != o_idx]
        if len(h_idx) != 2:
            continue
        o_pos = np.asarray(mol.get_atom(o_idx).position)
        h_pos = [np.asarray(mol.get_atom(i).position) for i in h_idx]
        oh1 = (h_pos[0] - o_pos); oh1 /= np.linalg.norm(oh1)
        oh2 = (h_pos[1] - o_pos); oh2 /= np.linalg.norm(oh2)
        bisector = oh1 + oh2
        bisector /= np.linalg.norm(bisector)
        m_o = (metal_pos - o_pos); m_o /= np.linalg.norm(m_o)
        cos_ang = np.clip(np.dot(m_o, bisector), -1, 1)
        ang = np.degrees(np.arccos(cos_ang))  # ~180 = good (bisector opposite metal dir)
        dev_from_180 = abs(180.0 - ang)
        flags.append(dict(o_idx=o_idx, angle_metal_vs_bisector=round(float(ang), 1),
                           deviation_from_ideal=round(float(dev_from_180), 1),
                           suspicious=bool(dev_from_180 > angle_threshold)))
    return flags

def nitrate_orientation_check(mol, metal_pos):
    flags = []
    for lig in mol.ligands:
        if len(lig.atom_indices) != 4 or len(lig.donor_atoms) not in (1, 2):
            continue
        syms = {i: mol.get_atom(i).symbol for i in lig.atom_indices}
        if list(syms.values()).count("N") != 1 or list(syms.values()).count("O") != 3:
            continue
        n_idx = [i for i, s in syms.items() if s == "N"][0]
        donor_o = lig.donor_atoms
        nondonor = [i for i in lig.atom_indices if i not in donor_o and i != n_idx]
        pts = np.array([np.asarray(mol.get_atom(i).position) for i in lig.atom_indices])
        centroid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centroid)
        planarity_dev = float(np.max(np.abs((pts - centroid) @ vt[2])))
        d_donor = [np.linalg.norm(np.asarray(mol.get_atom(i).position) - metal_pos) for i in donor_o]
        d_n = np.linalg.norm(np.asarray(mol.get_atom(n_idx).position) - metal_pos)
        d_nondonor = [np.linalg.norm(np.asarray(mol.get_atom(i).position) - metal_pos) for i in nondonor]
        clash = bool(d_n < min(d_donor)) or any(bool(d < min(d_donor)) for d in d_nondonor)
        flags.append(dict(
            donor_M_O=[round(float(x), 3) for x in d_donor],
            M_N=round(float(d_n), 3),
            nondonor_M_O=[round(float(x), 3) for x in d_nondonor],
            planarity_max_dev_A=round(planarity_dev, 5),
            suspicious=clash or planarity_dev > 0.15,
        ))
    return flags

SPIN = {"La": 1, "Nd": 4, "Dy": 6, "Fe": 6}
CASES = []
for m in ["La", "Nd", "Dy"]:
    CASES.append((f"{m}_aqua_y8", lambda m=m: build_aqua(m, 8, charge=3, spin=SPIN[m]), "aqua"))
    CASES.append((f"{m}_aqua_y9", lambda m=m: build_aqua(m, 9, charge=3, spin=SPIN[m]), "aqua"))
for m in ["La", "Nd", "Dy"]:
    CASES.append((f"{m}_nitrato_y7", lambda m=m: build_nitrato_bidentate(m, 7, charge=2, spin=SPIN[m]), "nitrato"))
CASES.append(("Fe_aqua_y6", lambda: build_aqua("Fe", 6, charge=3, spin=SPIN["Fe"]), "aqua"))
CASES.append(("Fe_nitrato_y4", lambda: build_nitrato_bidentate("Fe", 4, charge=2, spin=SPIN["Fe"]), "nitrato"))
CASES.append(("Fe_nitrato_y5", lambda: build_nitrato_bidentate("Fe", 5, charge=2, spin=SPIN["Fe"]), "nitrato"))

REF_FILES = {
    "La_aqua_y8": [f"{DATA}/opt-xyz/Ln-xyz/Ln-H2O_x/La3_8H2O.xyz"],
    "La_aqua_y9": [f"{DATA}/opt-xyz/Ln-xyz/Ln-H2O_x/La3_9H2O.xyz"],
    "Nd_aqua_y8": [f"{DATA}/opt-xyz/Ln-xyz/Ln-H2O_x/Nd3_8H2O.xyz"],
    "Nd_aqua_y9": [f"{DATA}/opt-xyz/Ln-xyz/Ln-H2O_x/Nd3_9H2O.xyz"],
    "Dy_aqua_y8": [f"{DATA}/opt-xyz/Ln-xyz/Ln-H2O_x/Dy3_8H2O.xyz"],
    "Dy_aqua_y9": [f"{DATA}/opt-xyz/Ln-xyz/Ln-H2O_x/Dy3_9H2O.xyz"],
    "La_nitrato_y7": [f"{DATA}/opt-xyz/Ln-xyz/Ln-NO3-H2O_x/La3_7H2O_NO3.xyz",
                       f"{DATA}/opt-xyz/Ln-xyz/Ln-NO3-H2O_x/La3_8H2O_NO3.xyz"],
    "Dy_nitrato_y7": [f"{DATA}/opt-xyz/Ln-xyz/Ln-NO3-H2O_x/Dy3_7H2O_NO3.xyz",
                       f"{DATA}/opt-xyz/Ln-xyz/Ln-NO3-H2O_x/Dy3_8H2O_NO3.xyz"],
    "Nd_nitrato_y7": [],  # no reference available in dataset
    "Fe_aqua_y6": [f"{DATA}/opt-xyz/misc-metal-xyz/metal-H2O_x/Fe3_6H2O.xyz"],
    "Fe_nitrato_y4": [f"{DATA}/opt-xyz/misc-metal-xyz/metal-NO3-H2O_x/Fe3_4H2O_1NO3.xyz"],
    "Fe_nitrato_y5": [f"{DATA}/opt-xyz/misc-metal-xyz/metal-NO3-H2O_x/Fe3_5H2O_1NO3.xyz"],
}

report = []
for name, build_fn, kind in CASES:
    builder = build_fn()
    result = builder.solve(seed=0)
    best = result.best
    mol = best.complex
    metal_pos = np.asarray(mol.get_atom(mol.metal_idx).position)
    donor_pos = np.array([np.asarray(s.position) for s in mol.coord_sites])
    dists = np.linalg.norm(donor_pos - metal_pos, axis=1)

    water_flags = water_orientation_check(mol, metal_pos)
    nitrate_flags = nitrate_orientation_check(mol, metal_pos) if kind == "nitrato" else []

    rmsds = []
    for ref_path in REF_FILES.get(name, []):
        if not os.path.exists(ref_path):
            rmsds.append(dict(ref=os.path.basename(ref_path), error="file not found"))
            continue
        ref_metal, ref_donors = ref_inner_sphere_donors(ref_path)
        p = donor_pos - metal_pos
        q = ref_donors - ref_metal
        if len(q) != len(p):
            rmsds.append(dict(ref=os.path.basename(ref_path), error=f"donor count mismatch: jacobik={len(p)} ref={len(q)}"))
            continue
        rmsd = rmsd_to_ref(p, q)
        rmsds.append(dict(ref=os.path.basename(ref_path), rmsd_A=round(float(rmsd), 4),
                           ref_bond_range=[round(float(np.linalg.norm(q,axis=1).min()),3),
                                           round(float(np.linalg.norm(q,axis=1).max()),3)]))

    row = dict(
        name=name, score=round(best.score, 4), cn=len(donor_pos),
        geometry=builder.geometry,
        jacobik_bond_range=[round(float(dists.min()), 3), round(float(dists.max()), 3)],
        rmsd_vs_ref=rmsds,
        water_orientation=water_flags,
        water_suspicious_count=sum(f["suspicious"] for f in water_flags),
        nitrate_orientation=nitrate_flags,
        nitrate_suspicious_count=sum(f["suspicious"] for f in nitrate_flags),
    )
    report.append(row)
    print(f"=== {name} === score={row['score']} CN={row['cn']} geom={row['geometry']} "
          f"bond_range={row['jacobik_bond_range']} water_suspicious={row['water_suspicious_count']}/{len(water_flags)} "
          f"nitrate_suspicious={row['nitrate_suspicious_count']}/{len(nitrate_flags)}")
    for r in rmsds:
        print(f"    vs {r.get('ref')}: {r}")

with open(os.path.join(OUT, "full_analysis.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved full analysis to {OUT}/full_analysis.json")
