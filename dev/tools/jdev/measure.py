"""Solve one case and measure it. MACE only MEASURES here (never feeds back into a solve).

Energy metrics (kJ/mol), all against the case's reference structure with identical charge/multiplicity:
  dE_sp     single-point E(solved) - E(reference). Strict: includes every small bond/angle strain.
  dE_basin  (audit --relax only) E(solved after local MACE relaxation) - E(reference after local relaxation):
            "is it the right structure/basin", the way Architector-style benchmarks compare structures.
  strain    (audit --relax only) E(solved) - E(solved relaxed): local strain; with the RMSD the atoms moved.
  thermal   (3N-6) kT/2 at 298 K: average potential energy of a room-temperature snapshot, for scale.
"""
from __future__ import annotations

import collections
import time

import numpy as np

KT_298 = 0.0083145 * 298.15  # kJ/mol
COV = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07, "S": 1.05,
       "Cl": 1.02, "Br": 1.20, "I": 1.39}


def solve(case, stats, start="etkdg", seed=11, embed_seed=None, **solve_kw):
    from .start import starting_geometry

    b = case.make_builder()
    kw = dict(stats=stats, seed=seed, **solve_kw)
    if embed_seed is not None:
        kw["embed_seed"] = embed_seed  # only exists in engines that support it
    with starting_geometry(case, start):
        t0 = time.perf_counter()
        res = b.solve(**kw)
        t = time.perf_counter() - t0
    best = max(res.conformers, key=lambda c: c.score)
    X = np.array([a.position for a in best.complex.atoms])
    sy = [a.symbol for a in best.complex.atoms]
    out = {"case": case.name, "start": start, "seed": seed, "embed_seed": embed_seed, "solve_s": t,
           "score": float(best.score), "n_conformers": len(res.conformers),
           "converged": int(sum(bool(getattr(c, "converged", False)) for c in res.conformers))}
    out.update(donor_metrics(case, best, X, sy))
    return out, best, sy, X


def donor_metrics(case, best, X, sy) -> dict:
    d = {}
    if case.tmqm_mol is not None:
        rows = {a.idx: i for i, a in enumerate(case.tmqm_mol.atoms)}
        per = []
        for lig, li, lg in zip(sorted(best.complex.ligands, key=lambda l: l.ligand_id), case.ligand_inputs,
                               case.ligand_groups):
            for g in li.donors:
                for dd in g:
                    if dd in lg["rdkit_to_xyz"]:
                        ref = np.linalg.norm(case.ref_X[rows[lg["rdkit_to_xyz"][dd]]] - case.ref_X[case.ref_metal_row])
                        got = np.linalg.norm(X[lig.atom_indices[dd] - 1] - X[0])
                        per.append((sy[lig.atom_indices[dd] - 1], float(ref), float(got)))
        d["donors"] = per
        d["n_donors"] = sum(len(li.donors) for li in case.ligand_inputs)
        d["n_donors_crystal"] = case.n_donors_crystal
        d["max_md_err"] = max(abs(g - r) for _, r, g in per) if per else None
    else:
        md = sorted(float(np.linalg.norm(X[i - 1] - X[0])) for lig in best.complex.ligands for i in lig.donor_atoms)
        d["n_donors"] = len(md)
        d["md"] = md
        if case.ref_X is not None:
            m = case.ref_metal_row
            ref = sorted(float(np.linalg.norm(case.ref_X[i] - case.ref_X[m])) for i, s in enumerate(case.ref_symbols)
                         if s in "NOS" and np.linalg.norm(case.ref_X[i] - case.ref_X[m]) < 3.2)
            d["n_donors_crystal"] = len(ref)
            if len(ref) == len(md):
                d["max_md_err"] = max(abs(a - b) for a, b in zip(ref, md))
    return d


def _h_profile(sy, X):
    prof = []
    for i, s in enumerate(sy):
        if s == "H" or s not in COV:
            continue
        prof.append((s, sum(1 for j, t in enumerate(sy)
                            if t == "H" and np.linalg.norm(X[i] - X[j]) < 1.2 * (COV[s] + 0.31))))
    return collections.Counter(prof)


def chem_checks(case, sy, X) -> dict:
    """Would the energy difference be comparing different molecules?"""
    if case.ref_symbols is None:
        return {}
    comp = collections.Counter(case.ref_symbols) == collections.Counter(sy)
    pr, ps = _h_profile(case.ref_symbols, case.ref_X), _h_profile(sy, X)
    return {"same_composition": comp, "same_H_per_heavy_atom": pr == ps,
            "H_profile_diff": {f"{k[0]}-{k[1]}H": v for k, v in ((pr - ps) + (ps - pr)).items()},
            "charge": case.charge, "multiplicity": case.mult}


def energies(case, ev, sy, X, relax_steps: int = 0) -> dict:
    """dE_sp always; dE_basin / strain / reference-strain only if relax_steps > 0 (slow for big complexes)."""
    from jacobik.core.energy import EV_TO_KJ_MOL as K

    if ev is None or case.ref_symbols is None:
        return {}
    e_ref = ev.energy(case.ref_symbols, case.ref_X, case.charge, case.mult)
    e_sol = ev.energy(sy, X, case.charge, case.mult)
    out = {"dE_sp": (e_sol - e_ref) * K, "thermal_scale": (3 * len(sy) - 6) * KT_298 / 2}
    if relax_steps > 0:
        from ase import Atoms
        from ase.optimize import FIRE

        calc = ev._get_calculator()

        def relax(s, P):
            a = Atoms(symbols=s, positions=P, pbc=False)
            a.info.update(charge=int(case.charge), spin=int(case.mult), external_field=[0.0, 0.0, 0.0])
            a.calc = calc
            FIRE(a, logfile=None).run(fmax=0.05, steps=relax_steps)
            return a.get_potential_energy(), a.get_positions()

        def rmsd(A, B):
            A, B = A - A.mean(0), B - B.mean(0)
            U, _, Vt = np.linalg.svd(A.T @ B)
            R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            return float(np.sqrt(((A @ R - B) ** 2).sum(1).mean()))

        er1, Pr = relax(case.ref_symbols, case.ref_X)
        es1, Ps = relax(sy, X)
        out.update(dE_basin=(es1 - er1) * K, strain=(e_sol - es1) * K, strain_rmsd=rmsd(X, Ps),
                   ref_strain=(e_ref - er1) * K, ref_strain_rmsd=rmsd(case.ref_X, Pr), relax_steps=relax_steps)
    return out
