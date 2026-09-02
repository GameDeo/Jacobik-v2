"""
benchmark.py
============
Symmetry-corrected full-complex RMSD benchmark of the Jacobik conformer
ensemble engine against tmQM reference geometries.

Version history
---------------
1.0.0 : Initial release.
        Implements BiSyRMSD: Kabsch alignment + inter-ligand permutation
        symmetry + intra-ligand graph automorphisms (handles phenyl flips,
        symmetric chelates, Cp rotations, etc.).
        Reports rank-1 RMSD (production metric) and best-of-N RMSD
        (capacity metric) per complex, plus aggregate statistics.
1.1.0 : Suppress all RDKit stderr warnings at module import time via
        rdBase.DisableLog.  Add tqdm progress bar to run_benchmark with
        live rank1_rmsd / errors postfix; graceful fallback if tqdm absent.

Algorithm
---------
For each tmQM complex:

  1.  Parse reference geometry via TMQMParser → TMComplex (CSD xyz coords).
  2.  Extract per-ligand SMILES by fragmenting the full complex SMILES on
      the metal; each fragment is canonicalised to a stable key.
  3.  Reconstruct LigandInput objects (SMILES + donor group indices on the
      H-implicit heavy-atom graph) from the parsed Ligand.donor_atoms and
      the SMILES→xyz VF2 mapping already computed by mapping.py.
  4.  Run ComplexBuilder.solve() → EnsembleResult (N conformers).
  5.  For each conformer, compute sym_rmsd(predicted, reference):
        a. Build the atom correspondence: predicted atom i ↔ reference atom i
           (same atom ordering guaranteed by engine construction).
        b. Enumerate inter-ligand permutations of groups with equal canonical
           SMILES (e.g. three CO ligands → 3! = 6 permutations).
        c. For each inter-ligand permutation, enumerate the Cartesian product
           of per-ligand graph automorphisms (RDKit self-substructure match)
           to handle intra-molecular symmetry (phenyl C₂, Cp C₅, acac C₂…).
        d. For each combined atom permutation: Kabsch-align predicted → ref,
           compute heavy-atom RMSD.  Take the minimum across all permutations.
  6.  Report:
        • rank1_rmsd  — RMSD of the top-scored conformer (production metric)
        • best_rmsd   — minimum RMSD across the ensemble (capacity metric)
        • converged   — whether the rank-1 conformer converged
        • n_sym       — number of symmetry permutations evaluated

Notes on index safety
---------------------
The engine builds each ligand by parsing its SMILES with RDKit, then maps
RDKit atom indices → global xyz indices internally.  The benchmark extracts
per-ligand SMILES from the full complex SMILES, parses with RDKit, and uses
GetSubstructMatches(mol, mol) to obtain automorphisms in RDKit index space.
These are then composed with the same VF2 mapping used by mapping.py to
produce automorphisms in xyz index space.  The two index spaces never mix.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: Ligands with more than this many heavy atoms are flagged as "flexible".
#: Full-complex RMSD is unreliable for these; use CS RMSD as primary metric.
FLEX_THRESHOLD: int = 15

# ---------------------------------------------------------------------------
# RDKit warning suppression
# ---------------------------------------------------------------------------
# RDKit writes valence errors, sanitisation warnings, and kekulisation
# failures to stderr via its own logging system (rdBase), bypassing Python's
# logging module entirely.  Silencing these here — before any rdkit.Chem call
# — keeps benchmark output clean regardless of what SMILES are encountered.
try:
    from rdkit import RDLogger as _RDLogger
    _RDLogger.logger().setLevel(_RDLogger.CRITICAL)
    from rdkit.rdBase import DisableLog as _DisableLog
    for _ch in ("rdApp.error", "rdApp.warning", "rdApp.info", "rdApp.debug"):
        _DisableLog(_ch)
except ImportError:
    pass  # RDKit not installed; helpers fall back gracefully

# ---------------------------------------------------------------------------
# Kabsch alignment
# ---------------------------------------------------------------------------


def kabsch_rmsd(
    P: np.ndarray,  # (N, 3) predicted (will be aligned to Q)
    Q: np.ndarray,  # (N, 3) reference
) -> Tuple[float, np.ndarray]:
    """
    Optimal rigid-body alignment of P onto Q via Kabsch SVD.

    Constrains to proper rotations (det R = +1) to avoid reflections,
    which is correct for chiral / non-centrosymmetric structures.

    Parameters
    ----------
    P : (N, 3) predicted coordinates
    Q : (N, 3) reference coordinates

    Returns
    -------
    rmsd : float — minimum RMSD after optimal rotation + translation
    P_aligned : (N, 3) — P after applying the optimal rotation + translation
    """
    assert P.shape == Q.shape, f"Shape mismatch: {P.shape} vs {Q.shape}"

    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    Pc = P - p_mean
    Qc = Q - q_mean

    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    sign_fix = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_fix @ U.T

    P_aligned = (Pc @ R.T) + q_mean
    rmsd = float(np.sqrt(((P_aligned - Q) ** 2).sum(axis=-1).mean()))
    return rmsd, P_aligned


def kabsch_rmsd_batch(
    Ps: np.ndarray,  # (B, N, 3) — B candidate predicted point clouds
    Q: np.ndarray,   # (N, 3)    — single reference
) -> np.ndarray:     # (B,)      — RMSD for each candidate
    """
    Batched Kabsch RMSD: align each of B point clouds in Ps onto Q.

    Used in sym_rmsd to evaluate all symmetry permutations in one
    vectorised pass instead of B separate SVD calls.

    Parameters
    ----------
    Ps : (B, N, 3) — B permuted versions of the predicted point cloud
    Q  : (N, 3)    — reference point cloud (same for all B)

    Returns
    -------
    rmsds : (B,) array of RMSD values, one per candidate
    """
    B, N, _ = Ps.shape
    p_means = Ps.mean(axis=1, keepdims=True)      # (B, 1, 3)
    q_mean  = Q.mean(axis=0)                       # (3,)
    Pc = Ps - p_means                              # (B, N, 3)
    Qc = Q - q_mean                                # (N, 3)

    # Cross-covariance matrices: H[b] = Pc[b].T @ Qc  →  (B, 3, 3)
    H = np.einsum("bni,nj->bij", Pc, Qc)

    # Batched SVD
    U, _, Vt = np.linalg.svd(H)                   # (B,3,3), (B,3), (B,3,3)

    # det sign for each item
    VtT = Vt.transpose(0, 2, 1)                   # (B, 3, 3)
    UT  = U.transpose(0, 2, 1)                    # (B, 3, 3)
    dets = np.linalg.det(VtT @ UT)                # (B,)
    sign_fixes = np.zeros((B, 3, 3), dtype=float)
    sign_fixes[:, 0, 0] = 1.0
    sign_fixes[:, 1, 1] = 1.0
    sign_fixes[:, 2, 2] = np.sign(dets)

    R = VtT @ sign_fixes @ UT                     # (B, 3, 3)

    # Rotate each Pc[b] by R[b]: (B, N, 3)
    Pc_aligned = np.einsum("bij,bnj->bni", R, Pc)
    P_aligned  = Pc_aligned + q_mean              # (B, N, 3)

    # RMSD per batch item — compare against Q (not Qc): P_aligned is in Q's frame
    diff  = P_aligned - Q[None, :, :]             # (B, N, 3)
    rmsds = np.sqrt((diff ** 2).sum(axis=-1).mean(axis=-1))  # (B,)
    return rmsds



# ---------------------------------------------------------------------------
# Coordination-sphere RMSD
# ---------------------------------------------------------------------------


def _graph_distance_atoms(mol, max_bonds: int) -> set:
    """
    Return the set of 1-based xyz atom indices within max_bonds bonds of the metal.

    This is the correct atom set for coordination-sphere RMSD:
    - distance 0: metal
    - distance 1: donors (direct M–D bonds)
    - distance 2: atoms bonded to donors (first ligand shell)
    - distance 3: atoms two bonds from donors (captures chelate ring atoms,
                  immediate backbone — these are directly constrained by IK)

    Atoms beyond this cutoff are part of flexible chains that the engine only
    weakly controls, and should not drive the Kabsch alignment.
    """
    # Build adjacency from mol.bonds, excluding H atoms
    atom_map = {a.idx: a for a in mol.atoms}
    adj: Dict[int, List[int]] = {a.idx: [] for a in mol.atoms}
    for b in mol.bonds:
        if atom_map.get(b.i) and atom_map.get(b.j):
            adj[b.i].append(b.j)
            adj[b.j].append(b.i)

    # BFS from metal
    visited = {mol.metal_idx: 0}
    queue = [mol.metal_idx]
    while queue:
        node = queue.pop(0)
        d = visited[node]
        if d >= max_bonds:
            continue
        for nb in adj[node]:
            if nb not in visited:
                visited[nb] = d + 1
                queue.append(nb)

    # Return heavy atoms only within cutoff
    return {idx for idx, d in visited.items()
            if atom_map.get(idx) and atom_map[idx].symbol != "H"}


def coord_sphere_rmsd(
    ref_mol,
    pred_conformer,
    ligand_groups: List[Dict],
    max_bonds: int = 2,
) -> Tuple[float, float]:
    """
    Graph-distance-limited coordination sphere RMSD.

    Only atoms within max_bonds bonds of the metal are used for both
    alignment and measurement. This correctly handles flexible ligands
    like IWAGOE: distal atoms (distance > 3) are excluded entirely,
    so the Kabsch rotation is driven by the chemically constrained
    inner coordination shell, not by floppy chain ends.

    Atom sets by distance from metal
    ---------------------------------
    d=0  metal (1 atom)
    d=1  donors — direct M–D bonds (CN atoms, e.g. 6 for octahedral)
    d=2  first ligand shell — atoms bonded to donors
    d=3  second ligand shell — immediate backbone, chelate ring atoms

    max_bonds=2 gives typically 13–25 atoms for a typical CN=6 complex:
    metal + 6 donors + their immediate bonded neighbours.  This avoids
    symmetric ring atoms (d=3) that corrupt alignment for ligands like
    VUCVUN's aziridine arms, while giving enough atoms for stable Kabsch.

    Parameters
    ----------
    ref_mol        : TMComplex — parsed reference geometry
    pred_conformer : ConformerResult — engine output
    ligand_groups  : from _build_ligand_groups(ref_mol, ...)
    max_bonds      : graph distance cutoff from metal (default 3)

    Returns
    -------
    (cs_rmsd, n_cs_atoms)
    cs_rmsd    : Å — RMSD on graph-local atoms, nan if < 3 atoms matched
    n_cs_atoms : number of atoms used
    """
    # Atoms within max_bonds of metal in the reference mol
    local_xyz = _graph_distance_atoms(ref_mol, max_bonds)

    # Build full matched arrays
    matched = _build_matched_arrays(ref_mol, pred_conformer, ligand_groups)
    if matched is None:
        return float("nan"), 0

    pred_m, ref_m, ref_xyz_list, pred_row_list = matched

    # Filter both pred and ref to graph-local atoms
    local_pred, local_ref = [], []
    for i, ref_xyz_1b in enumerate(ref_xyz_list):
        if ref_xyz_1b in local_xyz:
            local_pred.append(pred_m[i])
            local_ref.append(ref_m[i])

    n_cs = len(local_pred)
    if n_cs < 3:
        return float("nan"), n_cs

    # Kabsch on local atoms only — alignment is not corrupted by distal atoms
    P_local = np.array(local_pred, dtype=float)
    Q_local = np.array(local_ref,  dtype=float)
    cs_rmsd, _ = kabsch_rmsd(P_local, Q_local)
    return cs_rmsd, n_cs


def max_ligand_heavy_atoms(mol) -> int:
    """Return the number of heavy atoms in the largest ligand."""
    atom_map = {a.idx: a for a in mol.atoms}
    if not mol.ligands:
        return 0
    return max(
        sum(1 for idx in lig.atom_indices if atom_map.get(idx) and atom_map[idx].symbol != "H")
        for lig in mol.ligands
    )



def per_ligand_local_rmsd(
    ref_mol,
    pred_conformer,
    ligand_groups: List[Dict],
) -> List[Tuple[int, float, int]]:
    """
    Per-ligand shape RMSD: how similar is each predicted ligand conformation
    to the reference, in isolation from the complex?

    For each ligand:
      1. Extract the matched heavy-atom positions for that ligand from both
         the reference and the predicted conformer (using the
         isomorphism-refined correspondence from _build_matched_arrays).
      2. Kabsch-align the predicted ligand onto the reference ligand using
         ALL its heavy atoms — metal-free, complex-free.
      3. Try all intra-ligand graph automorphisms (symmetric arms, phenyl
         flips, etc.) and take the minimum RMSD — per-ligand BiSyRMSD.

    This answers: "ignoring where the ligand sits in the complex, does it
    have the same shape?" It is independent of CS RMSD and of the metal.

    Returns
    -------
    list of (ligand_id, shape_rmsd, n_atoms) — one per ligand.
    shape_rmsd is nan if fewer than 3 heavy atoms matched.
    """
    matched = _build_matched_arrays(ref_mol, pred_conformer, ligand_groups)
    if matched is None:
        return [(lg["ligand_id"], float("nan"), 0) for lg in ligand_groups]

    pred_m, ref_m, ref_xyz_list, _ = matched

    results = []
    for lg in ligand_groups:
        lig_xyz_0based = set(lg["xyz_indices_0based"])

        # Collect all heavy atoms belonging to this ligand
        lig_pred, lig_ref = [], []
        lig_row_indices   = []
        for row_i, ref_xyz_1b in enumerate(ref_xyz_list):
            if (ref_xyz_1b - 1) in lig_xyz_0based:
                lig_pred.append(pred_m[row_i])
                lig_ref.append(ref_m[row_i])
                lig_row_indices.append(row_i)

        n_atoms = len(lig_pred)
        if n_atoms < 3:
            results.append((lg["ligand_id"], float("nan"), n_atoms))
            continue

        P = np.array(lig_pred, dtype=float)  # predicted ligand heavy atoms
        Q = np.array(lig_ref,  dtype=float)  # reference ligand heavy atoms

        # Base RMSD with current isomorphism
        best_rmsd, _ = kabsch_rmsd(P, Q)

        # Try all intra-ligand automorphisms to find best shape match.
        # xyz_automorphisms are {ref_xyz_1b → ref_xyz_1b} permutations.
        # We build a row-permutation for P that applies each automorphism.
        xyz_to_row = {ref_xyz_1b: i
                      for i, ref_xyz_1b in enumerate(
                          [ref_xyz_list[r] for r in lig_row_indices]
                      )}

        for auto in lg.get("xyz_automorphisms", []):
            # Build permuted version of P under this automorphism
            try:
                P_perm = np.array([
                    P[xyz_to_row[auto.get(ref_xyz_list[lig_row_indices[i]],
                                          ref_xyz_list[lig_row_indices[i]])]]
                    for i in range(n_atoms)
                ], dtype=float)
            except KeyError:
                continue  # automorphism maps outside this ligand's atoms
            r, _ = kabsch_rmsd(P_perm, Q)
            if r < best_rmsd:
                best_rmsd = r

        results.append((lg["ligand_id"], best_rmsd, n_atoms))

    return results



# ---------------------------------------------------------------------------
# GFN2-xTB single-point energy gap
# ---------------------------------------------------------------------------

# Bohr per Angstrom (CODATA 2018)
_BOHR_PER_ANG: float = 1.0 / 0.529177210903

# Hartree to kJ/mol
_HARTREE_TO_KJ: float = 2625.4996394798

# Atomic number lookup for elements in tmQM
_ATOMIC_NUMBER: Dict[str, int] = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
    "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
    "Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,
    "Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,
    "Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,
    "Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,
    "Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,
    "Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,"Hf":72,"Ta":73,"W":74,
    "Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,
    "Bi":83,
}


def _xtb_single_point(
    atoms,
    charge: int,
    spin: int,
    e_temp: float = 300.0,
    label: str = "",         # e.g. "VUCVUN ref" — used in log messages
) -> Optional[float]:
    """
    Run a GFN2-xTB single-point and return the total energy in Hartree.
    Returns None on any failure; logs a warning with the reason.
    """
    def _warn(msg):
        """Print AND log so warnings are visible in Jupyter regardless of log config."""
        print(f"  ⚠ xTB: {msg}")
        log.warning("xTB: %s", msg)

    try:
        from xtb.interface import Calculator, Param
        from xtb.libxtb import VERBOSITY_MUTED
    except ImportError:
        _warn("package not importable — install with: conda install -c conda-forge xtb-python")
        return None

    # Pre-flight: check all elements are known to GFN2-xTB
    unknown = sorted({a.symbol for a in atoms if a.symbol not in _ATOMIC_NUMBER})
    if unknown:
        _warn(f"{label}: unknown element(s) {unknown} — GFN2-xTB has no parameters for these")
        return None

    uhf = max(0, spin - 1)
    print(f"    xTB {label}: {len(atoms)} atoms  charge={charge}  uhf={uhf}  T={e_temp:.0f} K")

    try:
        numbers   = np.array([_ATOMIC_NUMBER[a.symbol] for a in atoms], dtype=np.int32)
        positions = np.array([a.position for a in atoms], dtype=np.float64) * _BOHR_PER_ANG

        calc = Calculator(Param.GFN2xTB, numbers, positions,
                          charge=charge, uhf=uhf)
        calc.set_verbosity(VERBOSITY_MUTED)
        calc.set_electronic_temperature(e_temp)
        res    = calc.singlepoint()
        energy = float(res.get_energy())
        log.info("xTB %s: E=%.6f Ha", label, energy)  # absolute energy logged only, not printed
        return energy

    except Exception as exc:
        _warn(f"{label}: SCF failed [{type(exc).__name__}] {exc}")
        return None


def xtb_energy_gap(
    ref_mol,
    pred_conformer,
    charge: int,
    spin: int,
    ref_energy: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Compute GFN2-xTB single-point energies for the reference and predicted
    geometries and return their difference in kJ/mol.

    ΔE = E(pred) − E(ref).  Near-zero ΔE means the predicted geometry is in
    the same energy basin as the DFT reference.  Both calculations are raw
    single-points on the respective geometries (no optimisation, no alignment).

    Convergence failures are retried at 5000 K electronic temperature before
    giving up.  All failure reasons are logged at WARNING level.

    Returns
    -------
    (delta_e_kj, e_ref_hartree, e_pred_hartree) — all nan on failure.
    """
    _nan3 = (float("nan"), float("nan"), float("nan"))

    if ref_mol is None or pred_conformer is None:
        log.warning("xTB energy_gap: received None mol or conformer")
        return _nan3

    code  = getattr(ref_mol, "csd_code", "?")
    label_ref  = f"{code} ref"
    label_pred = f"{code} pred"

    # Always strip H from both ref and pred.
    #
    # Rationale: the reference H positions come from DFT geometry optimisation
    # (which respects the actual electronic structure, lone pairs, coordination).
    # The predicted H positions come from RDKit AddHs() (which uses simple
    # valence rules, ignoring coordination chemistry).  These two H-placement
    # schemes are systematically inconsistent — even one extra or missing H
    # atom contributes ~1000 kJ/mol to ΔE in GFN2-xTB, completely swamping
    # the geometric signal.
    #
    # Stripping H is standard practice when comparing energies across
    # structure generation methods with different H-placement strategies.
    # The engine does not control H placement, so H should not be scored.
    #
    # charge=0 for the heavy-atom-only system: the formal charge is carried
    # by the heavy atoms; removing H (which are neutral in most TM complexes)
    # does not change the charge state of the remaining heavy-atom skeleton
    # in a way that affects the *relative* ΔE.
    ref_atoms_xtb  = [a for a in ref_mol.atoms                if a.symbol != "H"]
    pred_atoms_xtb = [a for a in pred_conformer.complex.atoms if a.symbol != "H"]
    eff_charge     = charge  # formal charge stays on heavy atoms

    n_ref_h  = len(ref_mol.atoms)                - len(ref_atoms_xtb)
    n_pred_h = len(pred_conformer.complex.atoms) - len(pred_atoms_xtb)
    if n_ref_h != n_pred_h:
        print(f"  ⚠ xTB {code}: H count differs — ref has {n_ref_h} H, "
              f"pred has {n_pred_h} H (both stripped; this mismatch explains "
              f"high ΔE if you were comparing with H included)")
    else:
        print(f"  xTB {code}: stripping {n_ref_h} H atoms from both "
              f"(heavy-atom comparison only)")

    # ── Reference ──────────────────────────────────────────────────────────
    if ref_energy is not None and ref_energy == ref_energy:
        e_ref = ref_energy
    else:
        e_ref = _xtb_single_point(ref_atoms_xtb, eff_charge, spin,
                                   e_temp=300.0, label=label_ref)
        if e_ref is None:
            log.warning("xTB %s: ref at 300 K failed — retrying at 5000 K", code)
            e_ref = _xtb_single_point(ref_mol.atoms, charge, spin,
                                       e_temp=5000.0, label=label_ref + " 5kK")
        if e_ref is None:
            log.warning("xTB %s: ref geometry failed at both temperatures — ΔE = nan", code)
            return _nan3

    # ── Predicted ──────────────────────────────────────────────────────────
    e_pred = _xtb_single_point(pred_atoms_xtb, eff_charge, spin,
                                e_temp=300.0, label=label_pred)
    if e_pred is None:
        log.warning("xTB %s: pred at 300 K failed — retrying at 5000 K", code)
        e_pred = _xtb_single_point(pred_atoms_xtb, eff_charge, spin,
                                    e_temp=5000.0, label=label_pred + " 5kK")
    if e_pred is None:
        log.warning("xTB %s: pred geometry failed at both temperatures — ΔE = nan", code)
        return _nan3

    delta_kj  = (e_pred - e_ref) * _HARTREE_TO_KJ
    n_heavy   = sum(1 for a in ref_mol.atoms if a.symbol != "H")
    per_atom  = delta_kj / max(n_heavy, 1)
    # Interpretation guide printed alongside the result
    # Use |ΔE| for basin classification — sign is physically meaningless
    # since we're comparing DFT-optimised ref against xTB energy surface.
    # Negative ΔE = pred is lower on xTB surface than DFT ref (normal).
    # Positive ΔE = pred is higher (strained or wrong basin).
    abs_dkj = abs(delta_kj)
    if abs_dkj < 10:
        basin = "same basin ✓"
    elif abs_dkj < 50:
        basin = "nearby ✓"
    elif abs_dkj < 200:
        basin = "different basin ✗"
    else:
        basin = "severely strained ✗✗"
    sign_note = ("  (pred lower than DFT ref on xTB surface — normal)"
                 if delta_kj < -5 else "")
    print(f"  xTB {code}: ΔE = {delta_kj:+.1f} kJ/mol  "
          f"(|ΔE|/atom = {abs(per_atom):.2f} kJ/mol)  [{basin}]{sign_note}")
    log.info("xTB %s: ΔE=%+.2f kJ/mol  per_atom=%+.2f  ref=%.6f  pred=%.6f Ha",
             code, delta_kj, per_atom, e_ref, e_pred)
    return delta_kj, e_ref, e_pred


# ---------------------------------------------------------------------------
# Per-ligand SMILES extraction from a full complex SMILES
# ---------------------------------------------------------------------------


def _ligand_smiles_from_complex(
    complex_smiles: str,
    metal_symbol: str,
) -> List[str]:
    """
    Split a complex SMILES into per-ligand SMILES by fragmenting on the metal.

    The metal is replaced by a dummy attachment point, and each connected
    fragment (ligand) is extracted as a canonical SMILES.

    Parameters
    ----------
    complex_smiles : full complex SMILES (from tmQM_y.csv)
    metal_symbol   : e.g. "Fe", "Ru"

    Returns
    -------
    list of canonical SMILES strings, one per ligand fragment.
    Empty list if RDKit cannot parse the SMILES.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(complex_smiles)
        if mol is None:
            return []

        # Find the metal atom index in RDKit mol
        metal_idx_rdkit: Optional[int] = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == metal_symbol:
                metal_idx_rdkit = atom.GetIdx()
                break
        if metal_idx_rdkit is None:
            return []

        # Remove the metal atom — this disconnects the ligands
        edit = Chem.RWMol(mol)
        edit.RemoveAtom(metal_idx_rdkit)
        frag_mol = edit.GetMol()

        # GetMolFrags(asMols=True) reliably returns one RDKit Mol per fragment,
        # avoiding the empty-string issue with MolFragmentToSmiles on charged atoms.
        frag_mols = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
        frag_smiles = [Chem.MolToSmiles(f, canonical=True) for f in frag_mols]

        return frag_smiles

    except Exception as exc:
        log.warning("SMILES fragmentation failed: %s", exc)
        return []


def canonical_smiles(smi: str) -> str:
    """Return RDKit canonical SMILES, or the input if RDKit fails."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smi


# ---------------------------------------------------------------------------
# Graph automorphisms of a molecule (intra-ligand symmetry)
# ---------------------------------------------------------------------------


def _mol_automorphisms(smi: str) -> List[Tuple[int, ...]]:
    """
    Return all graph automorphisms of the molecule given by *smi*.

    Each automorphism is a tuple of RDKit heavy-atom indices: position i
    holds the atom that maps TO position i under this symmetry operation.
    Concretely, automorphisms[k][i] is the index of the atom in the k-th
    symmetric image that corresponds to atom i in the canonical ordering.

    This is obtained via RDKit self-substructure matching:
        mol.GetSubstructMatches(mol, useChirality=False)
    which returns exactly the automorphism group of the molecular graph
    (ignoring stereo, which is appropriate for 3-D RMSD symmetry correction).

    The identity permutation is always included.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [tuple(range(mol.GetNumAtoms()))] if mol else [()]

        # Remove Hs for the automorphism search (we work in heavy-atom space)
        mol = Chem.RemoveHs(mol)
        n = mol.GetNumAtoms()

        # uniquify=False is required — default (True) returns only one match
        # per unique set of matched atoms, collapsing all automorphisms to one.
        matches = mol.GetSubstructMatches(mol, useChirality=False, uniquify=False)
        if not matches:
            return [tuple(range(n))]
        return list(matches)

    except Exception as exc:
        log.debug("Automorphism search failed for %r: %s", smi, exc)
        n_fallback = len(smi)  # crude upper bound; identity only
        return [tuple(range(n_fallback))]


# ---------------------------------------------------------------------------
# Build the xyz-index automorphism from RDKit automorphism + VF2 mapping
# ---------------------------------------------------------------------------


def _rdkit_to_xyz_automorphisms(
    rdkit_autos: List[Tuple[int, ...]],
    rdkit_to_xyz: Dict[int, int],  # heavy-atom rdkit_idx -> 1-based xyz idx
    all_xyz_indices: List[int],    # all xyz atom indices (heavy+H) for this ligand
    xyz_atoms,                      # list of Atom objects (for H neighbour lookup)
    xyz_bonds,                      # list of Bond objects
) -> List[List[int]]:
    """
    Lift RDKit heavy-atom automorphisms to full-atom (heavy+H) xyz permutations.

    For each RDKit automorphism (permutation of heavy atoms), we:
      1. Map heavy-atom rdkit indices → xyz indices via rdkit_to_xyz.
      2. For each heavy atom that moves, reassign its H neighbours to match
         (H atoms are locally symmetric around any sp3/sp2 centre, so any
         assignment is valid for RMSD purposes — we use sorted order).
      3. Return a list of permutations, each mapping old xyz index → new xyz index,
         covering ALL atoms in this ligand (heavy + H).

    Parameters
    ----------
    rdkit_autos    : output of _mol_automorphisms (RDKit heavy-atom indices)
    rdkit_to_xyz   : dict mapping RDKit heavy-atom idx → 1-based xyz idx
    all_xyz_indices: all 1-based xyz indices belonging to this ligand (heavy+H)
    xyz_atoms      : Atom list for the full complex
    xyz_bonds      : Bond list for the full complex

    Returns
    -------
    List of permutations. Each permutation is a list where
    permutation[i] = j means "old xyz index i maps to new xyz index j".
    We represent it as a dict: {old_xyz_idx: new_xyz_idx}.
    """
    # Build xyz→H-neighbours lookup
    xyz_atom_map = {a.idx: a for a in xyz_atoms}
    h_neighbours: Dict[int, List[int]] = {}
    for bond in xyz_bonds:
        for (src, dst) in [(bond.i, bond.j), (bond.j, bond.i)]:
            if src in rdkit_to_xyz.values() and dst not in rdkit_to_xyz.values():
                # dst is an H (or non-heavy) neighbour of a ligand heavy atom
                if xyz_atom_map.get(dst) and xyz_atom_map[dst].symbol == "H":
                    h_neighbours.setdefault(src, []).append(dst)
    for k in h_neighbours:
        h_neighbours[k] = sorted(h_neighbours[k])

    # Inverse mapping: xyz_idx -> rdkit_heavy_idx
    xyz_to_rdkit = {v: k for k, v in rdkit_to_xyz.items()}

    result_perms: List[Dict[int, int]] = []

    for auto in rdkit_autos:
        # auto[i] = j means rdkit atom i maps to rdkit atom j under this symmetry
        perm: Dict[int, int] = {}

        # Map heavy atoms
        for rdkit_src, rdkit_dst in enumerate(auto):
            xyz_src = rdkit_to_xyz.get(rdkit_src)
            xyz_dst = rdkit_to_xyz.get(rdkit_dst)
            if xyz_src is None or xyz_dst is None:
                continue
            perm[xyz_src] = xyz_dst

            # Map H neighbours of this heavy atom
            h_src = h_neighbours.get(xyz_src, [])
            h_dst = h_neighbours.get(xyz_dst, [])
            # Match H atoms in sorted order — valid since H neighbours are
            # locally equivalent under any graph automorphism of the heavy atom.
            for hs, hd in zip(sorted(h_src), sorted(h_dst)):
                perm[hs] = hd

        # Any unmapped atoms in this ligand map to themselves
        for idx in all_xyz_indices:
            if idx not in perm:
                perm[idx] = idx

        result_perms.append(perm)

    return result_perms


# ---------------------------------------------------------------------------
# Symmetry-corrected RMSD (BiSyRMSD analogue)
# ---------------------------------------------------------------------------


@dataclass
class SymRMSDResult:
    rmsd: float
    n_perms_evaluated: int
    best_perm_idx: int


def sym_rmsd(
    pred_matched: np.ndarray,   # (N_heavy, 3) predicted heavy-atom coords, pre-matched to ref
    ref_matched: np.ndarray,    # (N_heavy, 3) reference heavy-atom coords
    ligand_groups: List[Dict],  # see _build_ligand_groups()
    heavy_atom_map: List[int],  # ref_heavy_idx[i] = 0-based ref array row for matched atom i
    pred_heavy_map: List[int],  # pred_heavy_idx[i] = 0-based pred array row for matched atom i
    max_perms: int = 2048,
) -> SymRMSDResult:
    """
    Symmetry-corrected heavy-atom RMSD between a predicted conformer and the
    reference (tmQM) geometry.

    Atom correspondence
    -------------------
    pred_matched[i] and ref_matched[i] are the same heavy atom — matched via
    the VF2 rdkit→xyz mapping built in _build_ligand_groups().  Caller
    constructs these arrays; sym_rmsd never indexes into a raw flat array.
    This eliminates the atom-ordering mismatch between the engine (metal +
    ETKDG-ordered ligand atoms) and tmQM (original xyz file order).

    Symmetry corrections applied
    ----------------------------
    1. Kabsch optimal rigid-body rotation (det R = +1).
    2. Inter-ligand permutations: all assignments of predicted ligands to
       reference ligand slots, within groups sharing the same canonical SMILES.
    3. Intra-ligand graph automorphisms: phenyl C₂ flips, Cp rotations, etc.
       via RDKit self-substructure matching.

    Parameters
    ----------
    pred_matched    : (N_heavy, 3) — predicted positions, row i = heavy atom i
    ref_matched     : (N_heavy, 3) — reference positions, row i = same heavy atom i
    ligand_groups   : output of _build_ligand_groups()
    heavy_atom_map  : list mapping matched-atom index → 0-based ref array index
    pred_heavy_map  : list mapping matched-atom index → 0-based pred array index
    max_perms       : cap on total permutations to evaluate

    Returns
    -------
    SymRMSDResult with the minimum RMSD found and diagnostics.
    """
    n_heavy = pred_matched.shape[0]
    assert ref_matched.shape == pred_matched.shape, (
        f"Shape mismatch: pred={pred_matched.shape} ref={ref_matched.shape}"
    )

    # ── Build the symmetry permutation space ──────────────────────────────
    #
    # Group ligands by canonical SMILES → each group can be permuted freely.
    # Within each group, each ligand also has a set of automorphisms.
    #
    # Total permutation space = Π_group (group_size!) × Π_ligand |Aut(L)|
    # We enumerate this lazily with itertools.product and stop at max_perms.

    # Organise ligands into equivalence classes by canonical SMILES
    from collections import defaultdict
    smi_groups: Dict[str, List[int]] = defaultdict(list)
    for i, lg in enumerate(ligand_groups):
        smi_groups[lg["canonical_smiles"]].append(i)

    # For each equivalence class, generate all permutations of ligand slots
    inter_lig_perms: List[List[Tuple[int, int]]] = []
    for smi, lig_indices in smi_groups.items():
        group_perms = list(itertools.permutations(lig_indices))
        inter_lig_perms.append([(orig, perm_i)
                                 for perm in group_perms
                                 for orig, perm_i in zip(lig_indices, perm)])
    # Combine inter-ligand permutations across groups via Cartesian product
    # Each element of inter_lig_product is a flat list of (src_lig, dst_lig) pairs
    inter_lig_product = list(itertools.product(
        *[list(itertools.permutations(indices)) for indices in smi_groups.values()]
    ))

    # For each ligand, list its intra-molecular xyz-index permutations
    intra_lig_perms: List[List[Dict[int, int]]] = [
        lg["xyz_automorphisms"] for lg in ligand_groups
    ]

    best_rmsd = float("inf")
    best_perm_idx = 0
    n_evaluated = 0

    ref_xyz_to_matched: Dict[int, int] = {
        ref_xyz: i for i, ref_xyz in enumerate(heavy_atom_map)
    }

    # Collect all permuted pred arrays into a batch, then evaluate with
    # a single vectorised Kabsch call rather than one per permutation.
    # We accumulate up to _BATCH_SIZE permutations then flush.
    _BATCH_SIZE = 256
    _batch: List[np.ndarray] = []
    _batch_start_idx = 0   # n_evaluated at the start of the current batch

    def _flush_batch() -> None:
        nonlocal best_rmsd, best_perm_idx
        if not _batch:
            return
        Ps = np.stack(_batch, axis=0)              # (B_local, N, 3)
        rmsds = kabsch_rmsd_batch(Ps, ref_matched) # (B_local,)
        local_best = int(np.argmin(rmsds))
        if rmsds[local_best] < best_rmsd:
            best_rmsd = float(rmsds[local_best])
            best_perm_idx = _batch_start_idx + local_best
        _batch.clear()

    for inter_perm_combo in inter_lig_product:
        lig_assignment: Dict[int, int] = {}
        flat_idx = 0
        for smi, lig_indices in smi_groups.items():
            perm = inter_perm_combo[flat_idx]
            for src, dst in zip(lig_indices, perm):
                lig_assignment[src] = dst
            flat_idx += 1

        intra_iters = [intra_lig_perms[lig_assignment[i]]
                       for i in range(len(ligand_groups))]

        for intra_combo in itertools.product(*intra_iters):
            if n_evaluated >= max_perms:
                break

            pred_perm = pred_matched.copy()

            for src_lig_i, (dst_lig_i, intra_auto) in enumerate(
                zip([lig_assignment[i] for i in range(len(ligand_groups))], intra_combo)
            ):
                src_lg = ligand_groups[src_lig_i]

                src_heavy_xyz = [
                    idx + 1 for idx in src_lg["xyz_indices_0based"]
                    if (idx + 1) in ref_xyz_to_matched
                ]
                for src_xyz_1based in src_heavy_xyz:
                    mapped_xyz_1based = intra_auto.get(src_xyz_1based, src_xyz_1based)
                    src_row = ref_xyz_to_matched.get(src_xyz_1based)
                    dst_row = ref_xyz_to_matched.get(mapped_xyz_1based)
                    if src_row is None or dst_row is None:
                        continue
                    pred_perm[src_row] = pred_matched[dst_row]

            if len(_batch) == 0:
                _batch_start_idx = n_evaluated
            _batch.append(pred_perm)
            n_evaluated += 1

            if len(_batch) >= _BATCH_SIZE:
                _flush_batch()

        if n_evaluated >= max_perms:
            log.debug("sym_rmsd: hit max_perms=%d cap", max_perms)
            break

    _flush_batch()  # flush any remaining

    return SymRMSDResult(
        rmsd=best_rmsd,
        n_perms_evaluated=n_evaluated,
        best_perm_idx=best_perm_idx,
    )


# ---------------------------------------------------------------------------
# Build matched heavy-atom position arrays for RMSD
# ---------------------------------------------------------------------------


def _build_matched_arrays(
    ref_mol,              # TMComplex — reference (tmQM) geometry
    pred_conformer,       # ConformerResult — engine output
    ligand_groups: List[Dict],
) -> Optional[Tuple[np.ndarray, np.ndarray, List[int], List[int]]]:
    """
    Build matched heavy-atom position arrays for RMSD computation.

    The engine atom ordering (metal + ETKDG ligand atoms, H included) differs
    completely from tmQM xyz ordering.  This function uses the rdkit_to_xyz
    mapping stored in each ligand_group — which maps RDKit heavy-atom index →
    1-based reference xyz index — to construct aligned rows:

        ref_matched[i]  = position of heavy atom i in the reference
        pred_matched[i] = position of the *same* heavy atom in the predicted
                          conformer, located via its RDKit index in the engine's
                          atom list

    Only heavy atoms with a valid rdkit_to_xyz entry are included.  H atoms
    are excluded entirely (they are not what the engine is responsible for).

    Isomorphism selection (symmetric ligands)
    -----------------------------------------
    For symmetric ligands (e.g. two identical aziridine arms), the VF2
    mapping in _build_ligand_groups selects the isomorphism with minimum
    bond-length cost in the *reference* geometry.  For perfectly symmetric
    molecules both arms have equal cost, so the choice is arbitrary.

    _build_matched_arrays refines this: for each ligand, if the ligand has
    multiple automorphisms, it tries all of them and picks the one that
    minimises ||ref_pos - pred_pos||² summed over matched atoms.  This is
    the isomorphism that actually aligns corresponding atoms correctly given
    the two real 3D structures being compared.

    Parameters
    ----------
    ref_mol         : parsed TMComplex (reference geometry)
    pred_conformer  : ConformerResult from ComplexBuilder.solve()
    ligand_groups   : output of _build_ligand_groups(ref_mol, ...)

    Returns
    -------
    (pred_matched, ref_matched, ref_xyz_1based_list, pred_row_list)
    or None if the predicted complex has no atoms.
    """
    ref_atom_map  = {a.idx: a for a in ref_mol.atoms}   # 1-based → Atom
    pred_atoms    = pred_conformer.complex.atoms          # list in engine order

    # Engine ordering: metal is atoms[0] (idx=1), then ligand atoms appended
    # in LigandInput order, each a full H-included RDKit mol.
    # pred_atoms[k].idx is 1-based; for ligand atoms it equals
    # (1 + cumulative atoms before this ligand) + rdkit_local_idx.
    # We reconstruct the rdkit_local_idx → pred_atom mapping per ligand.

    # Build: pred_1based_idx → Atom for quick lookup
    pred_atom_map = {a.idx: a for a in pred_atoms}

    # The engine places atoms as:
    #   idx=1           → metal
    #   idx=2..2+N0-1   → ligand 0 atoms (in rdmol_h atom order)
    #   idx=2+N0..      → ligand 1 atoms ...
    # ligand_groups are in sorted(ligand_id) order matching LigandInput order.
    # We use the engine's Ligand.atom_indices to get the per-ligand atom list.
    pred_complex  = pred_conformer.complex

    ref_rows:  List[np.ndarray] = []
    pred_rows: List[np.ndarray] = []
    ref_xyz_1based_list:  List[int] = []
    pred_row_list:        List[int] = []

    # Include the metal atom (always atom idx=1 in both ref and pred)
    ref_metal  = ref_atom_map.get(ref_mol.metal_idx)
    pred_metal = pred_atom_map.get(1)
    if ref_metal is not None and pred_metal is not None:
        ref_rows.append(ref_metal.position.copy())
        pred_rows.append(pred_metal.position.copy())
        ref_xyz_1based_list.append(ref_mol.metal_idx)
        pred_row_list.append(0)  # metal is always first in pred_atoms

    # For each ligand, map rdkit_heavy_idx → ref_xyz_1based (from ligand_groups)
    # and rdkit_heavy_idx → pred_atom position (from engine ligand atom list).
    #
    # NOTE on ligand_id alignment: the engine assigns ligand_id = lig_i (0-based,
    # matching LigandInput order). The parsed reference mol assigns ligand_id
    # starting at 1 (mapping.py enumerate(..., start=1)). We match by *sorted
    # list position*, not ligand_id value — consistent with how
    # ligand_inputs_from_mol builds the LigandInput list.
    pred_ligs_sorted = sorted(pred_complex.ligands, key=lambda l: l.ligand_id)

    for lig_i, lg in enumerate(ligand_groups):
        rdkit_to_xyz = lg.get("rdkit_to_xyz", {})  # rdkit_idx → ref_xyz_1based
        if not rdkit_to_xyz:
            continue

        if lig_i >= len(pred_ligs_sorted):
            continue
        pred_lig = pred_ligs_sorted[lig_i]

        pred_lig_atoms = pred_lig.atom_indices  # 1-based pred idx, in rdkit order

        # ── Isomorphism refinement using pred positions ───────────────────
        # _build_ligand_groups chose one rdkit_to_xyz isomorphism based on
        # reference bond lengths alone.  For symmetric ligands (equal-cost
        # arms), that choice is arbitrary.  Here we try all automorphisms
        # of the ligand and pick the one that minimises ||ref - pred||²,
        # using the actual 3D structures being compared.
        xyz_autos = lg.get("xyz_automorphisms", [])  # list of {xyz_1b→xyz_1b}
        xyz_to_rdkit = {v: k for k, v in rdkit_to_xyz.items()}

        # Build the base rows for the current (possibly wrong) isomorphism
        def _rows_for_rdkit_to_xyz(r2x):
            rr, pr, rxl, prl = [], [], [], []
            for rdkit_idx, ref_xyz_1based in sorted(r2x.items()):
                ref_atom = ref_atom_map.get(ref_xyz_1based)
                if ref_atom is None or ref_atom.symbol == "H":
                    continue
                if rdkit_idx >= len(pred_lig_atoms):
                    continue
                pred_1based = pred_lig_atoms[rdkit_idx]
                pred_atom   = pred_atom_map.get(pred_1based)
                if pred_atom is None or pred_atom.symbol == "H":
                    continue
                rr.append(ref_atom.position.copy())
                pr.append(pred_atom.position.copy())
                rxl.append(ref_xyz_1based)
                prl.append(pred_1based - 1)
            return rr, pr, rxl, prl

        best_cost = float("inf")
        best_rr, best_pr, best_rxl, best_prl = _rows_for_rdkit_to_xyz(rdkit_to_xyz)

        # Score the base isomorphism
        if best_rr:
            _r = np.array(best_rr); _p = np.array(best_pr)
            best_cost = float(((_r - _p) ** 2).sum())

        # Try each automorphism: apply it to rdkit_to_xyz and re-score
        for auto in xyz_autos:
            # auto: {xyz_1based → xyz_1based} — maps ref xyz to permuted ref xyz
            # Invert auto to get permuted_xyz → original_xyz
            # Then build new rdkit_to_xyz: rdkit_idx → auto[ref_xyz]
            alt_r2x = {}
            for rdkit_i, ref_xyz_1b in rdkit_to_xyz.items():
                alt_ref = auto.get(ref_xyz_1b, ref_xyz_1b)
                alt_r2x[rdkit_i] = alt_ref

            alt_rr, alt_pr, alt_rxl, alt_prl = _rows_for_rdkit_to_xyz(alt_r2x)
            if not alt_rr:
                continue
            _r = np.array(alt_rr); _p = np.array(alt_pr)
            cost = float(((_r - _p) ** 2).sum())
            if cost < best_cost:
                best_cost = cost
                best_rr, best_pr, best_rxl, best_prl = alt_rr, alt_pr, alt_rxl, alt_prl

        ref_rows.extend(best_rr)
        pred_rows.extend(best_pr)
        ref_xyz_1based_list.extend(best_rxl)
        pred_row_list.extend(best_prl)

    if not ref_rows:
        return None

    return (
        np.array(pred_rows, dtype=float),
        np.array(ref_rows,  dtype=float),
        ref_xyz_1based_list,
        pred_row_list,
    )


# ---------------------------------------------------------------------------
# Extract per-ligand information from a parsed TMComplex
# ---------------------------------------------------------------------------


def _build_ligand_groups(mol, complex_smiles: str) -> List[Dict]:
    """
    For each ligand in *mol*, build a descriptor dict containing:
      - canonical_smiles     : canonical SMILES for this ligand
      - xyz_indices_0based   : 0-based xyz indices of all atoms in this ligand
      - xyz_automorphisms    : list of {1based_xyz → 1based_xyz} dicts, one
                               per graph automorphism (intra-ligand symmetry)
      - donor_groups         : donor groups in RDKit heavy-atom index space
                               (for LigandInput construction)

    Index safety guarantee
    ----------------------
    Automorphisms are computed in RDKit heavy-atom index space, then lifted
    to 1-based xyz index space via the VF2 heavy-atom mapping (rdkit→xyz).
    The two index spaces are kept strictly separate and combined only here.
    """
    from rdkit import Chem
    import networkx as nx
    from collections import defaultdict

    atom_map = {a.idx: a for a in mol.atoms}  # 1-based → Atom

    # Build the xyz→symbol lookup for H detection
    def is_hydrogen(xyz_1based: int) -> bool:
        a = atom_map.get(xyz_1based)
        return a is not None and a.symbol == "H"

    # Fragment the complex SMILES → per-ligand canonical SMILES
    frag_smiles_list = _ligand_smiles_from_complex(complex_smiles, mol.metal_symbol)

    # Sort ligands by ligand_id to have a stable order
    sorted_ligands = sorted(mol.ligands, key=lambda l: l.ligand_id)

    # We need to match each Ligand object to one of the SMILES fragments.
    # Strategy: match by element composition (multiset of symbols).
    # This works as long as ligands have distinct element multisets;
    # for identical ligands we assign in order.

    def element_multiset(xyz_indices: List[int]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for idx in xyz_indices:
            sym = atom_map.get(idx, None)
            if sym:
                counts[sym.symbol] = counts.get(sym.symbol, 0) + 1
        return counts

    def smi_element_multiset(smi: str) -> Dict[str, int]:
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return {}
            m = Chem.AddHs(m)
            counts: Dict[str, int] = {}
            for atom in m.GetAtoms():
                s = atom.GetSymbol()
                counts[s] = counts.get(s, 0) + 1
            return counts
        except Exception:
            return {}

    # Match ligands → SMILES fragments by element multiset
    unmatched_smiles = list(enumerate(frag_smiles_list))
    lig_to_smi: Dict[int, str] = {}

    for lig in sorted_ligands:
        lig_elements = element_multiset(lig.atom_indices)
        best_match_idx = None
        for j, (orig_j, smi) in enumerate(unmatched_smiles):
            smi_elements = smi_element_multiset(smi)
            if smi_elements == lig_elements:
                best_match_idx = j
                break
        if best_match_idx is not None:
            _, matched_smi = unmatched_smiles.pop(best_match_idx)
            lig_to_smi[lig.ligand_id] = matched_smi
        else:
            # Fallback: no element-matched SMILES found; use empty string
            lig_to_smi[lig.ligand_id] = ""
            log.warning(
                "Could not match ligand %d to a SMILES fragment for %s",
                lig.ligand_id, mol.csd_code,
            )

    groups: List[Dict] = []
    for lig in sorted_ligands:
        smi = lig_to_smi.get(lig.ligand_id, "")
        can_smi = canonical_smiles(smi) if smi else ""

        # 0-based xyz indices for all atoms in this ligand
        xyz_0based = [idx - 1 for idx in lig.atom_indices]  # atom_indices are 1-based

        # Compute intra-ligand graph automorphisms
        if smi:
            rdkit_autos = _mol_automorphisms(smi)
        else:
            # No SMILES: identity only
            rdkit_autos = [tuple(range(len(lig.atom_indices)))]

        # Build rdkit_heavy_idx → xyz_1based mapping via element-order matching.
        # We use the same VF2 logic as mapping.py: parse the SMILES, remove H,
        # and match against the heavy atoms of this ligand by symbol.
        rdkit_to_xyz: Dict[int, int] = {}
        try:
            rdmol = Chem.MolFromSmiles(smi)
            if rdmol is not None:
                rdmol_h = Chem.RemoveHs(rdmol)
                # List of (rdkit_idx, symbol) for SMILES heavy atoms
                smi_heavy = [(a.GetIdx(), a.GetSymbol()) for a in rdmol_h.GetAtoms()]
                # List of (xyz_1based, symbol) for ligand heavy atoms
                lig_heavy = [
                    (idx, atom_map[idx].symbol)
                    for idx in lig.atom_indices
                    if not is_hydrogen(idx)
                ]
                # VF2 isomorphism via networkx for correct mapping
                G_smi = nx.Graph()
                for rdkit_i, sym in smi_heavy:
                    G_smi.add_node(rdkit_i, symbol=sym)
                for bond in rdmol_h.GetBonds():
                    G_smi.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

                G_xyz = nx.Graph()
                for xyz_idx, sym in lig_heavy:
                    G_xyz.add_node(xyz_idx, symbol=sym)
                for bond in mol.bonds:
                    bi, bj = bond.i, bond.j
                    if bi in {x for x, _ in lig_heavy} and bj in {x for x, _ in lig_heavy}:
                        G_xyz.add_edge(bi, bj)

                def node_match(n1, n2):
                    return n1["symbol"] == n2["symbol"]

                GM = nx.algorithms.isomorphism.GraphMatcher(G_smi, G_xyz, node_match=node_match)

                # Pick the geometrically consistent VF2 isomorphism.
                #
                # Problem: for symmetric ligands (e.g. two identical aziridine
                # arms), the first VF2 solution is arbitrary — it may map SMILES
                # atoms to the geometrically distant equivalent arm, making
                # rdkit_to_xyz a valid but spatially wrong correspondence.
                # For VUCVUN this caused 4.52 A self-RMSD and the long
                # cross-arm displacement vectors visible in the overlay.
                #
                # Solution: score each isomorphism by the sum of squared bond
                # lengths it implies in the reference geometry.  A correct
                # mapping preserves short SMILES bonds as short xyz distances;
                # a cross-arm swap inflates them.  We pick the minimum-cost
                # isomorphism.  Cap at 512 candidates for large automorphism
                # groups to avoid O(n!) enumeration on highly symmetric ligands.
                _MAX_ISO = 512
                best_iso = None
                best_iso_cost = float("inf")
                _n_iso_checked = 0
                for iso_candidate in GM.isomorphisms_iter():
                    if _n_iso_checked >= _MAX_ISO:
                        break
                    _n_iso_checked += 1
                    cost = 0.0
                    valid = True
                    for bond in rdmol_h.GetBonds():
                        a_rdkit = bond.GetBeginAtomIdx()
                        b_rdkit = bond.GetEndAtomIdx()
                        a_xyz = iso_candidate.get(a_rdkit)
                        b_xyz = iso_candidate.get(b_rdkit)
                        if a_xyz is None or b_xyz is None:
                            valid = False
                            break
                        pa = atom_map.get(a_xyz)
                        pb = atom_map.get(b_xyz)
                        if pa is None or pb is None:
                            valid = False
                            break
                        d = pa.position - pb.position
                        cost += float(d @ d)
                    if not valid:
                        continue
                    if cost < best_iso_cost:
                        best_iso_cost = cost
                        best_iso = iso_candidate

                if best_iso is not None:
                    rdkit_to_xyz = best_iso  # rdkit_idx -> xyz_1based
        except Exception as exc:
            log.debug("VF2 mapping failed for ligand %d: %s", lig.ligand_id, exc)

        # Lift automorphisms from RDKit idx space to xyz idx space
        if rdkit_to_xyz:
            xyz_autos = _rdkit_to_xyz_automorphisms(
                rdkit_autos,
                rdkit_to_xyz,
                lig.atom_indices,
                mol.atoms,
                mol.bonds,
            )
        else:
            # Identity only as fallback
            xyz_autos = [{idx: idx for idx in lig.atom_indices}]

        # Donor groups in RDKit heavy-atom index space (for LigandInput)
        # donor_atoms are 1-based xyz indices; map back to RDKit indices
        xyz_to_rdkit = {v: k for k, v in rdkit_to_xyz.items()}
        donor_groups_rdkit: List[List[int]] = []

        # Group donor atoms by haptic interactions first
        for site in mol.coord_sites:
            site_donors = set(site.donor_indices)
            lig_donor_set = set(lig.atom_indices)
            overlapping = sorted(site_donors & lig_donor_set)
            if not overlapping:
                continue
            rdkit_group = []
            for xyz_1based in overlapping:
                rdkit_i = xyz_to_rdkit.get(xyz_1based)
                if rdkit_i is not None:
                    rdkit_group.append(rdkit_i)
            if rdkit_group:
                donor_groups_rdkit.append(sorted(rdkit_group))

        # Deduplicate (same site might appear via multiple haptic atoms)
        seen = set()
        unique_donor_groups: List[List[int]] = []
        for grp in donor_groups_rdkit:
            key = tuple(sorted(grp))
            if key not in seen:
                seen.add(key)
                unique_donor_groups.append(grp)

        groups.append({
            "ligand_id": lig.ligand_id,
            "canonical_smiles": can_smi,
            "raw_smiles": smi,
            "xyz_indices_0based": xyz_0based,
            "xyz_automorphisms": xyz_autos,
            "donor_groups_rdkit": unique_donor_groups,
            "rdkit_to_xyz": rdkit_to_xyz,
        })

    return groups


# ---------------------------------------------------------------------------
# Build LigandInput list from a parsed TMComplex
# ---------------------------------------------------------------------------


def ligand_inputs_from_mol(mol, ligand_groups: List[Dict]):
    """
    Construct LigandInput objects for the engine from a parsed TMComplex
    and its pre-computed ligand_groups descriptor.

    Returns
    -------
    list of LigandInput objects (one per ligand, in sorted ligand_id order)
    """
    from jacobik.engine import LigandInput

    inputs = []
    for lg in ligand_groups:
        smi = lg["raw_smiles"]
        donor_groups = lg["donor_groups_rdkit"]
        if not smi or not donor_groups:
            log.warning(
                "Ligand %d: missing SMILES or donor groups — skipping complex",
                lg["ligand_id"],
            )
            return None
        inputs.append(LigandInput(smiles=smi, donors=donor_groups))
    return inputs


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRecord:
    """Result for one tmQM complex."""

    csd_code: str
    metal: str
    coord_geom: str
    cn: int                        # coordination number
    n_atoms: int                   # total heavy atoms
    n_ligands: int
    rank1_rmsd: float              # RMSD of top-scored conformer
    best_rmsd: float               # minimum RMSD across ensemble
    rank1_converged: bool
    n_conformers: int
    n_sym_perms: int               # symmetry permutations evaluated
    engine_time_s: float           # wall time for engine.solve()
    n_matched_heavy: int = 0       # heavy atoms successfully matched ref↔pred
    n_total_heavy: int = 0         # total heavy atoms in reference
    match_fraction: float = 0.0   # n_matched_heavy / n_total_heavy
    # ── Decomposed RMSD metrics ──────────────────────────────────────────────
    rank1_cs_rmsd: float = float("nan")   # coordination-sphere RMSD, rank-1
    best_cs_rmsd:  float = float("nan")   # coordination-sphere RMSD, best-of-N
    rank1_local_rmsd: float = float("nan") # mean per-ligand local RMSD, rank-1
    n_cs_atoms: int = 0                   # atoms used in CS alignment
    max_lig_heavy: int = 0                # heavy atoms in largest ligand
    is_flexible: bool = False             # True if max_lig_heavy > FLEX_THRESHOLD
    # ── xTB energy gap (optional) ────────────────────────────────────────────
    xtb_delta_kj: float = float("nan")      # E(rank1) - E(ref) in kJ/mol
    xtb_best_delta_kj: float = float("nan") # E(best-RMSD conformer) - E(ref)
    xtb_ref_hartree: float = float("nan")   # reference GFN2-xTB energy
    xtb_pred_hartree: float = float("nan")  # rank-1 GFN2-xTB energy
    solve_time_s: float = float("nan")      # engine solve() wall time (s)
    suspicious: bool = False       # True if failure likely not attributable to engine
    suspicious_reason: str = ""    # human-readable explanation
    error: str = ""                # non-empty if this complex was skipped


def _is_suspicious(r: "BenchmarkRecord") -> tuple:
    """
    Determine whether a valid (non-error) record's RMSD is likely not
    attributable to the engine — i.e. the pipeline itself had a problem.

    Returns (is_suspicious: bool, reason: str).

    Criteria
    --------
    match_fraction < 0.8  : VF2 atom mapping failed for >20% of heavy atoms;
                            the RMSD is computed on a partial/wrong correspondence.
    n_sym_perms == 1 and n_ligands > 1
                          : automorphism + inter-ligand permutation search found
                            only the identity; suggests the ligand_groups SMILES
                            or mapping was incomplete, so symmetry wasn't corrected.
    n_matched_heavy < 3   : fewer than 3 atoms matched — RMSD is meaningless.
    """
    if r.error:
        return False, ""   # error records are categorised separately
    reasons = []
    if r.n_total_heavy > 0 and r.match_fraction < 0.8:
        reasons.append(
            f"low atom match {r.n_matched_heavy}/{r.n_total_heavy} "
            f"({100*r.match_fraction:.0f}%)"
        )
    if r.n_matched_heavy < 3:
        reasons.append(f"only {r.n_matched_heavy} atoms matched")
    if r.n_sym_perms == 1 and r.n_ligands > 1:
        reasons.append("symmetry search found only identity (mapping likely incomplete)")
    # Flexible ligand: full RMSD not meaningful, but the complex itself is valid.
    # We don't flag it as suspicious (it's a genuine result), just note it.
    return bool(reasons), "; ".join(reasons)


@dataclass
class BenchmarkSummary:
    """Aggregate statistics across all benchmarked complexes."""

    records: List[BenchmarkRecord]

    @property
    def valid(self) -> List["BenchmarkRecord"]:
        """All records that completed without a hard pipeline error."""
        return [r for r in self.records if not r.error]

    @property
    def engine_valid(self) -> List["BenchmarkRecord"]:
        """
        Valid records that are NOT suspicious — i.e. where any RMSD failure
        can plausibly be attributed to the engine rather than the benchmark
        pipeline (atom mapping failures, incomplete symmetry correction, etc.).
        """
        return [r for r in self.valid if not r.suspicious]

    @property
    def suspicious(self) -> List["BenchmarkRecord"]:
        return [r for r in self.valid if r.suspicious]

    def rank1_rmsds(self, engine_valid_only: bool = False) -> np.ndarray:
        src = self.engine_valid if engine_valid_only else self.valid
        return np.array([r.rank1_rmsd for r in src])

    def best_rmsds(self, engine_valid_only: bool = False) -> np.ndarray:
        src = self.engine_valid if engine_valid_only else self.valid
        return np.array([r.best_rmsd for r in src])

    def _stat_block(self, r1: np.ndarray, bn: np.ndarray, label: str = "RMSD") -> str:
        """Format a two-column (rank-1 / best-of-N) stats block."""
        if len(r1) == 0:
            return "  (no data)\n"
        bn_safe = bn[:len(r1)]
        gap = r1 - bn_safe
        lines = [
            f"  {label+' — Metric':<32} {'Rank-1':>10} {'Best-of-N':>10}",
            f"  {'-'*54}",
            f"  {'Mean (Å)':<32} {r1.mean():>10.3f} {bn_safe.mean():>10.3f}",
            f"  {'Median (Å)':<32} {np.median(r1):>10.3f} {np.median(bn_safe):>10.3f}",
            f"  {'Std dev (Å)':<32} {r1.std():>10.3f} {bn_safe.std():>10.3f}",
            f"  {'< 0.5 Å (%)':<32} {100*(r1<0.5).mean():>10.1f} {100*(bn_safe<0.5).mean():>10.1f}",
            f"  {'< 1.0 Å (%)':<32} {100*(r1<1.0).mean():>10.1f} {100*(bn_safe<1.0).mean():>10.1f}",
            f"  {'< 2.0 Å (%)':<32} {100*(r1<2.0).mean():>10.1f} {100*(bn_safe<2.0).mean():>10.1f}",
            f"  {'90th pct (Å)':<32} {np.percentile(r1,90):>10.3f} {np.percentile(bn_safe,90):>10.3f}",
            f"  {'Scorer gap mean (Å)':<32} {gap.mean():>10.3f} {np.median(gap):>10.3f}",
        ]
        return "\n".join(lines) + "\n"

    def print_summary(self) -> None:
        n_total   = len(self.records)
        n_err     = sum(1 for r in self.records if r.error)
        n_valid   = len(self.valid)
        n_susp    = len(self.suspicious)
        n_eng     = len(self.engine_valid)
        n_flex    = sum(1 for r in self.engine_valid if r.is_flexible)
        n_rigid   = n_eng - n_flex
        n_conv    = sum(1 for r in self.engine_valid if r.rank1_converged)

        def _arr(attr, subset=None):
            rs = subset if subset is not None else self.engine_valid
            vals = [getattr(r, attr) for r in rs
                    if not r.error and getattr(r, attr) == getattr(r, attr)]
            return np.array(vals) if vals else np.array([])

        r1_cs    = _arr("rank1_cs_rmsd")
        bn_cs    = _arr("best_cs_rmsd")
        r1_loc   = _arr("rank1_local_rmsd")
        r1_full  = _arr("rank1_rmsd")
        bn_full  = _arr("best_rmsd")
        r1_full_flex  = _arr("rank1_rmsd",  [r for r in self.engine_valid if r.is_flexible])
        r1_full_rigid = _arr("rank1_rmsd",  [r for r in self.engine_valid if not r.is_flexible])
        r1_cs_flex    = _arr("rank1_cs_rmsd",[r for r in self.engine_valid if r.is_flexible])
        r1_cs_rigid   = _arr("rank1_cs_rmsd",[r for r in self.engine_valid if not r.is_flexible])

        W = 66
        print(f"\n{'='*W}")
        print(f"  Jacobik × tmQM Benchmark — Final Summary")
        print(f"  {n_total} complexes  |  split engine-valid: {n_eng} "
              f"({n_rigid} rigid / {n_flex} flexible)")
        print(f"{'='*W}")
        print(f"  Pipeline errors           : {n_err}")
        print(f"  Pipeline-suspicious       : {n_susp}  (mapping/sym failures)")
        print(f"  Rank-1 converged          : {n_conv}/{n_eng} "
              f"({100*n_conv/max(n_eng,1):.1f}%)")
        print(f"  Flexible ligands (>{FLEX_THRESHOLD} heavy atoms): {n_flex}")

        # ── PRIMARY: Coordination Sphere ─────────────────────────────────────
        print(f"\n{'─'*W}")
        print(f"  PRIMARY METRIC — Coordination Sphere RMSD")
        print(f"  (metal + donor atoms only; symmetry-corrected; engine-attributable)")
        print(f"  This is what the engine directly optimises.")
        print(f"{'─'*W}")
        if len(r1_cs):
            print(self._stat_block(r1_cs, bn_cs, "Coord Sphere"))
        if len(r1_cs_rigid) and len(r1_cs_flex):
            print(f"  By flexibility:  rigid CS={r1_cs_rigid.mean():.3f} Å  "
                  f"flexible CS={r1_cs_flex.mean():.3f} Å")

        # ── SECONDARY: Per-ligand local ───────────────────────────────────────
        print(f"\n{'─'*W}")
        print(f"  SECONDARY METRIC — Per-Ligand Local RMSD")
        print(f"  (backbone atoms, donor-anchored local frame; mean across ligands)")
        print(f"{'─'*W}")
        if len(r1_loc):
            bn_loc = r1_loc.copy()  # best-of-N local not tracked; show rank-1 only
            print(self._stat_block(r1_loc, bn_loc, "Local backbone"))

        # ── TERTIARY: Full complex ────────────────────────────────────────────
        print(f"\n{'─'*W}")
        print(f"  TERTIARY — Full Complex RMSD (symmetry-corrected)")
        print(f"  Note: for flexible ligands (>{FLEX_THRESHOLD} heavy atoms), Kabsch")
        print(f"  alignment is dominated by distal atoms; CS RMSD is more informative.")
        print(f"{'─'*W}")
        if len(r1_full):
            print(self._stat_block(r1_full, bn_full, "Full complex"))
        if len(r1_full_rigid) and len(r1_full_flex):
            print(f"  By flexibility:  rigid={r1_full_rigid.mean():.3f} Å  "
                  f"flexible={r1_full_flex.mean():.3f} Å  "
                  f"(gap={r1_full_flex.mean()-r1_cs_flex.mean():.3f} Å for flex)")

        if self.suspicious:
            print(f"\n── Pipeline-suspicious cases {'─'*(W-30)}")
            for r in self.suspicious:
                print(f"  {r.csd_code:<12} {r.metal:<4} {r.coord_geom:<14} "
                      f"CS={r.rank1_cs_rmsd:.3f} Å  full={r.rank1_rmsd:.3f} Å  "
                      f"{r.suspicious_reason}")
        print(f"\n{'='*W}\n")

        # ── xTB energy gap summary ────────────────────────────────────────────
        xtb_vals = [r.xtb_delta_kj for r in self.engine_valid
                    if r.xtb_delta_kj == r.xtb_delta_kj]
        if xtb_vals:
            xtb_arr = np.array(xtb_vals)
            print(f"\n── GFN2-xTB Energy Gap ΔE = E(pred) − E(ref) {'─'*(W-44)}")
            print(f"  n = {len(xtb_arr)}  computed")
            print(f"  Mean ΔE  : {xtb_arr.mean():+.1f} kJ/mol")
            print(f"  Median ΔE: {np.median(xtb_arr):+.1f} kJ/mol")
            print(f"  |ΔE| < 10 kJ/mol : {100*(np.abs(xtb_arr)<10).mean():.0f}%  "
                  f"(same basin — sign ignored, xTB ≠ DFT surface)")
            print(f"  |ΔE| < 50 kJ/mol : {100*(np.abs(xtb_arr)<50).mean():.0f}%")
            print(f"  90th pct ΔE      : {np.percentile(xtb_arr,90):+.1f} kJ/mol")

        # ── Poster talking points ─────────────────────────────────────────────
        print(f"\n── Poster / Reporting ({'─'*(W-22)})")
        if len(r1_cs):
            print(f"  Headline:  CS RMSD = {r1_cs.mean():.2f} ± {r1_cs.std():.2f} Å  "
                  f"({100*(r1_cs<1.0).mean():.0f}% under 1 Å)")
            if len(r1_cs_rigid):
                print(f"  Rigid:     CS RMSD = {r1_cs_rigid.mean():.2f} Å  "
                      f"({100*(r1_cs_rigid<1.0).mean():.0f}% under 1 Å)")
        if xtb_vals:
            print(f"  xTB:       {100*(np.abs(xtb_arr)<10).mean():.0f}% within 10 kJ/mol of DFT ref  "
                  f"(mean |ΔE| = {np.abs(xtb_arr).mean():.1f} kJ/mol)")
            print(f"  Framing:   'X% of predicted geometries are within 10 kJ/mol of the")
            print(f"              tmQM DFT reference at GFN2-xTB level — indicating correct")
            print(f"              basin placement without post-optimisation.'")
        print()


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(
    data_dir: str | Path,
    split: str,
    constraints_db: str | Path | None = None,
    limit: Optional[int] = None,
    beam_width: int = 16,
    n_ik_steps: int = 200,
    n_coord_samples: int = 8,
    seed: int = 42,
    device: str = "cpu",
    max_sym_perms: int = 2048,
    log_file: Optional[str | Path] = None,
    fast_embed: bool = True,
    n_conformers_per_complex: Optional[int] = None,
    use_xtb: bool = False,
) -> BenchmarkSummary:
    """
    Run the full benchmark on a tmQM split.

    Parameters
    ----------
    data_dir        : tmQM dataset root (xyz/ bo/ subdirs + tmQM_y.csv)
    split           : "X1", "X2", etc.
    constraints_db  : path to constraints.json; None = bundled default
    limit           : stop after this many complexes (None = full split)
    beam_width      : engine beam search width
    n_ik_steps      : IK gradient descent steps
    n_coord_samples : coordination sphere samples
    seed            : random seed for reproducibility
    device          : "cpu" or "cuda"
    max_sym_perms          : cap on symmetry permutations per RMSD call
    log_file               : optional path to write per-complex detailed log lines;
                             appended if file exists, created if not.
    fast_embed             : skip MMFF refinement in ETKDG (default True for
                             benchmarking — IK overwrites the geometry anyway).
    n_conformers_per_complex : pass as n_conformers to solve(); set to 1 to
                             assemble only the rank-1 conformer and skip all
                             deep-copy overhead for lower-ranked structures.
    use_xtb              : if True, compute GFN2-xTB single-point ΔE for each
                             rank-1 conformer (requires xtb package).  Adds
                             ~2–10 s per complex.  Results stored in
                             BenchmarkRecord.xtb_delta_kj.

    Returns
    -------
    BenchmarkSummary with per-complex BenchmarkRecord list
    """
    from jacobik.stats.parser import TMQMParser
    from jacobik.stats.stats import StatsLookup, load_db
    from jacobik.engine import ComplexBuilder

    data_dir = Path(data_dir)
    parser = TMQMParser(data_dir)
    records: List[BenchmarkRecord] = []

    # ── Pre-load stats DB once (avoid re-parsing per complex) ────────────────
    _db_path = constraints_db
    if _db_path is None:
        from importlib import resources as _res
        _db_path = _res.files("jacobik") / "constraints.json"
    print(f"Loading constraints DB: {_db_path} ...")
    _preloaded_stats = StatsLookup(load_db(_db_path))
    print("  ✓ Stats DB loaded.")

    # ── File logger (optional) ────────────────────────────────────────────────
    _file_log: Optional[object] = None
    if log_file is not None:
        import logging as _logging
        _fh = _logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        _fh.setLevel(_logging.DEBUG)
        _fh.setFormatter(_logging.Formatter("%(asctime)s %(message)s",
                                             datefmt="%H:%M:%S"))
        _file_logger = _logging.getLogger("benchmark.detail")
        _file_logger.setLevel(_logging.DEBUG)
        _file_logger.addHandler(_fh)
        _file_logger.propagate = False
        def _flog(msg: str) -> None:
            _file_logger.info(msg)
    else:
        def _flog(msg: str) -> None:
            pass

    # ── Running stats tracker ─────────────────────────────────────────────────
    # Tracks two parallel populations: all valid, and engine-attributable only.
    class _RunningStats:
        """Incrementally updated benchmark statistics, printed after each complex."""
        def __init__(self):
            self.r1:   List[float] = []
            self.bn:   List[float] = []
            self.r1e:  List[float] = []  # engine-valid only
            self.bne:  List[float] = []
            self.cs1:  List[float] = []  # CS RMSD rank-1
            self.csb:  List[float] = []  # CS RMSD best-of-N
            self.n_conv = 0
            self.n_eng  = 0
            self.n_susp = 0
            self.n_err  = 0

        def update(self, rec: BenchmarkRecord) -> None:
            if rec.error:
                self.n_err += 1
                return
            nan = rec.rank1_rmsd != rec.rank1_rmsd
            cs_nan = rec.rank1_cs_rmsd != rec.rank1_cs_rmsd
            if not nan:
                self.r1.append(rec.rank1_rmsd)
                self.bn.append(rec.best_rmsd)
            if not cs_nan:
                self.cs1.append(rec.rank1_cs_rmsd)
                self.csb.append(rec.best_cs_rmsd if rec.best_cs_rmsd == rec.best_cs_rmsd else rec.rank1_cs_rmsd)
            if rec.suspicious:
                self.n_susp += 1
            else:
                self.n_eng += 1
                if rec.rank1_converged:
                    self.n_conv += 1
                if not nan:
                    self.r1e.append(rec.rank1_rmsd)
                    self.bne.append(rec.best_rmsd)

        def _pct(self, arr: list, thresh: float) -> str:
            if not arr:
                return "  —"
            return f"{100*sum(v < thresh for v in arr)/len(arr):5.1f}%"

        def _mean(self, arr: list) -> str:
            return f"{sum(arr)/len(arr):.3f}" if arr else "  —"

        def summary_line(self) -> str:
            n = len(self.r1)
            ne = len(self.r1e)
            parts = [
                f"n={n}",
                f"CS={self._mean(self.cs1)} Å",
                f"full={self._mean(self.r1)} Å",
                f"<1Å(CS)={self._pct(self.cs1, 1.0)}",
                f"| eng n={ne}",
                f"CS={self._mean(self.cs1)} Å",
                f"| conv={self.n_conv}/{self.n_eng}",
                f"susp={self.n_susp}",
                f"err={self.n_err}",
            ]
            return "  ".join(parts)

    _stats = _RunningStats()

    # ── Pre-calibration self-check ────────────────────────────────────────
    # Run sym_rmsd(reference, reference) on the first parseable complex.
    # Result must be < 1e-6 Å — any larger value indicates a bug in the
    # atom-index mapping or Kabsch implementation before we waste time on
    # the full benchmark.
    print("Pre-calibration check: computing sym_rmsd(ref, ref) on first complex...")
    _calib_mol = None
    for _m in parser.stream_split(split, limit=50):
        if _m.smiles and _m.ligands and _m.coord_sites:
            _calib_mol = _m
            break
    if _calib_mol is None:
        raise RuntimeError(
            "Pre-calibration failed: could not find a parseable complex "
            f"with SMILES in the first 50 entries of split {split!r}."
        )
    try:
        _calib_groups = _build_ligand_groups(_calib_mol, _calib_mol.smiles)
    except Exception as _exc:
        raise RuntimeError(f"Pre-calibration: _build_ligand_groups failed: {_exc}") from _exc

    # Calibration test 1: Kabsch(P, P) must be ~0.
    # Build ref heavy-atom positions directly from ligand_groups (no fake
    # conformer — that path requires matching engine ligand_ids which differ
    # from parsed ligand_ids and would always produce false failures).
    _ref_atom_map = {a.idx: a for a in _calib_mol.atoms}
    _calib_ref_rows: List[np.ndarray] = []
    _calib_ref_xyz:  List[int] = []

    # Metal
    _calib_ref_rows.append(_ref_atom_map[_calib_mol.metal_idx].position.copy())
    _calib_ref_xyz.append(_calib_mol.metal_idx)
    # Ligand heavy atoms via rdkit_to_xyz mapping
    for _lg in _calib_groups:
        for _rdkit_idx, _ref_xyz_1b in sorted(_lg["rdkit_to_xyz"].items()):
            _a = _ref_atom_map.get(_ref_xyz_1b)
            if _a is None or _a.symbol == "H":
                continue
            _calib_ref_rows.append(_a.position.copy())
            _calib_ref_xyz.append(_ref_xyz_1b)

    if not _calib_ref_rows:
        raise RuntimeError(
            f"Pre-calibration: no heavy atoms matched for {_calib_mol.csd_code}. "
            "Check that rdkit_to_xyz mappings are non-empty in _build_ligand_groups."
        )

    _calib_ref_arr = np.array(_calib_ref_rows, dtype=float)
    # Self-RMSD: Kabsch(ref, ref) must be < 1e-6 Å
    _calib_self_rmsd, _ = kabsch_rmsd(_calib_ref_arr, _calib_ref_arr)
    _CALIB_TOL = 1e-6
    if _calib_self_rmsd > _CALIB_TOL:
        raise RuntimeError(
            f"Pre-calibration FAILED: kabsch_rmsd(ref, ref) = {_calib_self_rmsd:.2e} Å "
            f"on {_calib_mol.csd_code}. Bug in Kabsch implementation."
        )

    # Calibration test 2: sym_rmsd on identical matched arrays must also be ~0.
    _calib_sr = sym_rmsd(
        pred_matched=_calib_ref_arr,
        ref_matched=_calib_ref_arr,
        ligand_groups=_calib_groups,
        heavy_atom_map=_calib_ref_xyz,
        pred_heavy_map=list(range(len(_calib_ref_xyz))),
        max_perms=2048,
    )
    if _calib_sr.rmsd > _CALIB_TOL:
        raise RuntimeError(
            f"Pre-calibration FAILED: sym_rmsd(ref, ref) = {_calib_sr.rmsd:.2e} Å "
            f"on {_calib_mol.csd_code}. Bug in sym_rmsd permutation logic."
        )

    _n_matched = len(_calib_ref_xyz)
    _n_heavy   = sum(1 for a in _calib_mol.atoms if a.symbol != "H")
    print(
        f"  ✓ {_calib_mol.csd_code}  kabsch+sym_rmsd(ref,ref) = {_calib_sr.rmsd:.2e} Å  "
        f"matched {_n_matched}/{_n_heavy} heavy atoms  "
        f"({_calib_sr.n_perms_evaluated} sym perms)"
    )

    # ── Progress bar (tqdm optional) ──────────────────────────────────────
    # The parser streams lazily so total is unknown upfront; tqdm shows
    # rate + elapsed + a live postfix with rolling mean RMSD and error count.
    try:
        from tqdm import tqdm as _tqdm
        _use_tqdm = True
    except ImportError:
        _use_tqdm = False

    mol_stream = parser.stream_split(split, limit=limit)
    if _use_tqdm:
        pbar = _tqdm(
            mol_stream,
            desc=f"benchmark {split}",
            unit="mol",
            dynamic_ncols=True,
            smoothing=0.1,
        )
        mol_iter = pbar
    else:
        pbar = None
        mol_iter = mol_stream

    _n_errors = 0  # still used for error-path continues

    for mol in mol_iter:
        code = mol.csd_code
        log.info("Benchmarking %s ...", code)

        # ── Skip complexes without SMILES (cannot reconstruct ligand inputs) ──
        if not mol.smiles:
            records.append(BenchmarkRecord(
                csd_code=code, metal=mol.metal_symbol,
                coord_geom=mol.coord_geom or "unknown",
                cn=mol.coordination_number(), n_atoms=len(mol.atoms),
                n_ligands=len(mol.ligands), rank1_rmsd=float("nan"),
                best_rmsd=float("nan"), rank1_converged=False,
                n_conformers=0, n_sym_perms=0, engine_time_s=0.0,
                error="no_smiles",
            ))
            _n_errors += 1
            continue

        # ── Reference positions: all atoms including H ────────────────────
        ref_positions = np.array([a.position for a in mol.atoms])  # (N, 3)
        metal_idx_0 = mol.metal_idx - 1  # 0-based

        # ── Build per-ligand descriptors ───────────────────────────────────
        try:
            ligand_groups = _build_ligand_groups(mol, mol.smiles)
        except Exception as exc:
            log.warning("%s: _build_ligand_groups failed: %s", code, exc)
            records.append(BenchmarkRecord(
                csd_code=code, metal=mol.metal_symbol,
                coord_geom=mol.coord_geom or "unknown",
                cn=mol.coordination_number(), n_atoms=len(mol.atoms),
                n_ligands=len(mol.ligands), rank1_rmsd=float("nan"),
                best_rmsd=float("nan"), rank1_converged=False,
                n_conformers=0, n_sym_perms=0, engine_time_s=0.0,
                error=f"ligand_group_error: {exc}",
            ))
            _n_errors += 1
            continue

        # ── Build LigandInput list ─────────────────────────────────────────
        lig_inputs = ligand_inputs_from_mol(mol, ligand_groups)
        if lig_inputs is None:
            records.append(BenchmarkRecord(
                csd_code=code, metal=mol.metal_symbol,
                coord_geom=mol.coord_geom or "unknown",
                cn=mol.coordination_number(), n_atoms=len(mol.atoms),
                n_ligands=len(mol.ligands), rank1_rmsd=float("nan"),
                best_rmsd=float("nan"), rank1_converged=False,
                n_conformers=0, n_sym_perms=0, engine_time_s=0.0,
                error="missing_ligand_inputs",
            ))
            _n_errors += 1
            continue

        # ── Run engine ────────────────────────────────────────────────────
        geom = mol.coord_geom or "other"
        try:
            builder = ComplexBuilder(
                metal=mol.metal_symbol,
                geometry=geom,
                charge=mol.charge,
                spin=mol.spin_state,
                ligands=lig_inputs,
            )
        except ValueError as exc:
            records.append(BenchmarkRecord(
                csd_code=code, metal=mol.metal_symbol,
                coord_geom=geom, cn=mol.coordination_number(),
                n_atoms=len(mol.atoms), n_ligands=len(mol.ligands),
                rank1_rmsd=float("nan"), best_rmsd=float("nan"),
                rank1_converged=False, n_conformers=0, n_sym_perms=0,
                engine_time_s=0.0, error=f"builder_error: {exc}",
            ))
            _n_errors += 1
            continue

        t0 = time.perf_counter()
        try:
            result = builder.solve(
                stats=_preloaded_stats,
                beam_width=beam_width,
                n_ik_steps=n_ik_steps,
                n_coord_samples=n_coord_samples,
                seed=seed,
                device=device,
                fast_embed=fast_embed,
                n_conformers=n_conformers_per_complex,
            )
        except Exception as exc:
            records.append(BenchmarkRecord(
                csd_code=code, metal=mol.metal_symbol,
                coord_geom=geom, cn=mol.coordination_number(),
                n_atoms=len(mol.atoms), n_ligands=len(mol.ligands),
                rank1_rmsd=float("nan"), best_rmsd=float("nan"),
                rank1_converged=False, n_conformers=0, n_sym_perms=0,
                engine_time_s=time.perf_counter() - t0,
                error=f"solve_error: {exc}",
            ))
            _n_errors += 1
            continue
        engine_time = time.perf_counter() - t0

        if not result.conformers:
            records.append(BenchmarkRecord(
                csd_code=code, metal=mol.metal_symbol,
                coord_geom=geom, cn=mol.coordination_number(),
                n_atoms=len(mol.atoms), n_ligands=len(mol.ligands),
                rank1_rmsd=float("nan"), best_rmsd=float("nan"),
                rank1_converged=False, n_conformers=0, n_sym_perms=0,
                engine_time_s=engine_time, error="no_conformers",
            ))
            _n_errors += 1
            continue

        # ── Compute sym_rmsd + CS RMSD + local RMSD for each conformer ────
        rank1_rmsd     = float("nan")
        best_rmsd      = float("inf")
        rank1_cs_rmsd  = float("nan")
        best_cs_rmsd   = float("inf")
        rank1_local_rmsd = float("nan")
        n_cs_atoms     = 0
        total_n_sym    = 0
        n_matched_heavy = 0
        best_conformer  = None   # conformer with minimum full sym_rmsd

        for ci, conformer in enumerate(result.conformers):
            matched = _build_matched_arrays(mol, conformer, ligand_groups)
            if matched is None:
                log.warning("%s conformer %d: _build_matched_arrays returned None", code, ci)
                continue
            pred_matched, ref_matched, ref_xyz_list, pred_row_list = matched

            if ci == 0:
                n_matched_heavy = len(ref_xyz_list)

            # Full-complex sym_rmsd
            sr = sym_rmsd(
                pred_matched=pred_matched,
                ref_matched=ref_matched,
                ligand_groups=ligand_groups,
                heavy_atom_map=ref_xyz_list,
                pred_heavy_map=pred_row_list,
                max_perms=max_sym_perms,
            )
            total_n_sym = max(total_n_sym, sr.n_perms_evaluated)

            # Coordination-sphere RMSD (primary metric)
            cs_rmsd_val, n_cs = coord_sphere_rmsd(mol, conformer, ligand_groups)

            # Per-ligand local RMSD (rank-1 only — expensive for all conformers)
            if ci == 0:
                rank1_rmsd      = sr.rmsd
                rank1_cs_rmsd   = cs_rmsd_val
                n_cs_atoms      = n_cs
                rank1_converged = conformer.converged
                local_results   = per_ligand_local_rmsd(mol, conformer, ligand_groups)
                valid_locals    = [r for _, r, _ in local_results if r == r]
                rank1_local_rmsd = float(np.mean(valid_locals)) if valid_locals else float("nan")

            if sr.rmsd < best_rmsd:
                best_rmsd     = sr.rmsd
                best_conformer = conformer
            if cs_rmsd_val == cs_rmsd_val and cs_rmsd_val < best_cs_rmsd:
                best_cs_rmsd = cs_rmsd_val

        if best_cs_rmsd == float("inf"):
            best_cs_rmsd = float("nan")

        n_total_heavy   = sum(1 for a in mol.atoms if a.symbol != "H")
        match_fraction  = n_matched_heavy / max(n_total_heavy, 1)
        _max_lig_heavy  = max_ligand_heavy_atoms(mol)
        _is_flex        = _max_lig_heavy > FLEX_THRESHOLD
        rank1_score     = result.conformers[0].score if result.conformers else float("nan")
        rank1_gap       = (rank1_rmsd - best_rmsd) if (
            rank1_rmsd == rank1_rmsd and best_rmsd < float("inf")) else float("nan")
        best_rank       = (result.conformers.index(best_conformer) + 1
                           if best_conformer is not None else 1)

        # ── Build record ──────────────────────────────────────────────────
        rec = BenchmarkRecord(
            csd_code=code,
            metal=mol.metal_symbol,
            coord_geom=geom,
            cn=mol.coordination_number(),
            n_atoms=len(mol.atoms),
            n_ligands=len(mol.ligands),
            rank1_rmsd=rank1_rmsd,
            best_rmsd=best_rmsd,
            rank1_converged=rank1_converged,
            n_conformers=len(result.conformers),
            n_sym_perms=total_n_sym,
            engine_time_s=engine_time,
            n_matched_heavy=n_matched_heavy,
            n_total_heavy=n_total_heavy,
            match_fraction=match_fraction,
            rank1_cs_rmsd=rank1_cs_rmsd,
            best_cs_rmsd=best_cs_rmsd,
            rank1_local_rmsd=rank1_local_rmsd,
            n_cs_atoms=n_cs_atoms,
            max_lig_heavy=_max_lig_heavy,
            is_flexible=_is_flex,
            solve_time_s=engine_time,
        )
        rec.suspicious, rec.suspicious_reason = _is_suspicious(rec)

        # ── Optional xTB energy gap ───────────────────────────────────────
        if use_xtb and result.conformers:
            _uhf = max(0, mol.spin_state - 1)
            print(f"  xTB {code}: GFN2-xTB on rank-1 and best-RMSD "
                  f"(rank #{best_rank}) conformers  "
                  f"[charge={mol.charge} uhf={_uhf}, H stripped]")

            # rank-1
            _dkj, _eref, _epred = xtb_energy_gap(
                mol, result.conformers[0], mol.charge, mol.spin_state
            )
            rec.xtb_delta_kj    = _dkj
            rec.xtb_ref_hartree = _eref
            rec.xtb_pred_hartree = _epred

            # best-RMSD conformer (may be same as rank-1)
            if best_conformer is not None and best_conformer is not result.conformers[0]:
                _dkj_best, _, _ = xtb_energy_gap(
                    mol, best_conformer, mol.charge, mol.spin_state, ref_energy=_eref
                )
                rec.xtb_best_delta_kj = _dkj_best
            else:
                rec.xtb_best_delta_kj = _dkj  # same conformer

        records.append(rec)
        _stats.update(rec)

        # ── Per-complex stdout print ──────────────────────────────────────
        # Lead with coordination-sphere RMSD (what the engine controls).
        # Full RMSD is secondary and tagged [flex] for large ligands.
        _nan_cs   = rank1_cs_rmsd != rank1_cs_rmsd
        _nan_full = rank1_rmsd    != rank1_rmsd
        _nan_loc  = rank1_local_rmsd != rank1_local_rmsd
        _susp_tag = f"  ⚠ {rec.suspicious_reason}" if rec.suspicious else ""
        _flex_tag = " [flex]" if _is_flex else ""
        _conv_tag = "✓conv" if rank1_converged else "✗conv"
        _cs_str   = f"{rank1_cs_rmsd:.3f}"   if not _nan_cs   else "  nan"
        _full_str = f"{rank1_rmsd:.3f}"       if not _nan_full else "  nan"
        _loc_str  = f"{rank1_local_rmsd:.3f}" if not _nan_loc  else "  nan"
        _bcs_str  = f"{best_cs_rmsd:.3f}"     if best_cs_rmsd == best_cs_rmsd else "  nan"
        _xtb_nan     = rec.xtb_delta_kj      != rec.xtb_delta_kj
        _xtb_best_nan= rec.xtb_best_delta_kj != rec.xtb_best_delta_kj
        _xtb_r1_str  = f"{rec.xtb_delta_kj:+.1f}"      if not _xtb_nan      else "nan"
        _xtb_bn_str  = f"{rec.xtb_best_delta_kj:+.1f}" if not _xtb_best_nan else "nan"
        if use_xtb:
            if not _xtb_nan and not _xtb_best_nan and rec.xtb_best_delta_kj != rec.xtb_delta_kj:
                _xtb_tag = f"  ΔE(r1)={_xtb_r1_str} ΔE(best)={_xtb_bn_str} kJ/mol"
            elif not _xtb_nan:
                _xtb_tag = f"  ΔE={_xtb_r1_str} kJ/mol"
            else:
                _xtb_tag = "  ΔE=nan"
        else:
            _xtb_tag = ""
        print(
            f"  {code:<12} {mol.metal_symbol:<4} {geom:<14} "
            f"CS={_cs_str} Å  local={_loc_str} Å  full={_full_str} Å{_flex_tag}  "
            f"bestCS={_bcs_str} Å (#{best_rank})  "
            f"score={rank1_score:+.2f}  {_conv_tag}  "
            f"match={n_matched_heavy}/{n_total_heavy}  sym={total_n_sym:>4}  "
            f"solve={engine_time:.1f}s{_xtb_tag}{_susp_tag}"
        )

        # Running stats — CS RMSD front and centre
        print(f"  ↳ {_stats.summary_line()}")

        # File log — full detail
        _flog(
            f"{code} {mol.metal_symbol} {geom} "
            f"CS={_cs_str} local={_loc_str} full={_full_str} bestCS={_bcs_str} "
            f"xtb_dE={_xtb_str if not _xtb_nan else 'na'} kJ/mol "
            f"score={rank1_score:+.2f} match={n_matched_heavy}/{n_total_heavy} "
            f"sym={total_n_sym} {_conv_tag} flex={_is_flex} max_lig={_max_lig_heavy} "
            f"t={engine_time:.1f}s suspicious={rec.suspicious} "
            f"reason={rec.suspicious_reason!r} n_conf={len(result.conformers)}"
        )

        # tqdm postfix — CS RMSD as headline
        if pbar is not None:
            _cs_vals = _stats.cs1
            _csb_vals = _stats.csb
            pbar.set_postfix({
                "CS":   f"{sum(_cs_vals)/len(_cs_vals):.3f}" if _cs_vals else "—",
                "bCS":  f"{sum(_csb_vals)/len(_csb_vals):.3f}" if _csb_vals else "—",
                "full": f"{sum(_stats.r1)/len(_stats.r1):.3f}" if _stats.r1 else "—",
                "susp": _stats.n_susp,
                "err":  _n_errors,
            }, refresh=False)

    if pbar is not None:
        pbar.close()

    # ── Final summary to file log ─────────────────────────────────────────────
    _flog("=" * 60)
    _flog(f"FINAL  n={len(records)}  engine_valid={len(_stats.r1e)}"
          f"  r1_mean={sum(_stats.r1e)/max(len(_stats.r1e),1):.3f}"
          f"  susp={_stats.n_susp}  err={_stats.n_err}")
    _flog("=" * 60)

    return BenchmarkSummary(records=records)




def build_symmetry_corrected_arrays(
    ref_mol,
    pred_conformer,
    ligand_groups: List[Dict],
    max_perms: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Build matched arrays with the *best* symmetry permutation applied.

    Runs the full sym_rmsd permutation search and returns the pred/ref arrays
    corresponding to the minimum-RMSD permutation.  Used by overlay_view so
    displacement vectors reflect the best possible atom matching, not an
    arbitrary one that may swap symmetric arms.

    Returns the same tuple as _build_matched_arrays:
    (pred_matched, ref_matched, ref_xyz_list, pred_row_list)
    with pred_matched permuted to the best assignment.
    """
    matched = _build_matched_arrays(ref_mol, pred_conformer, ligand_groups)
    if matched is None:
        return None

    pred_m, ref_m, ref_xyz_list, pred_row_list = matched
    n = len(ref_xyz_list)
    if n == 0:
        return matched

    from collections import defaultdict
    smi_groups: Dict[str, List[int]] = defaultdict(list)
    for i, lg in enumerate(ligand_groups):
        smi_groups[lg["canonical_smiles"]].append(i)

    intra_lig_perms = [lg["xyz_automorphisms"] for lg in ligand_groups]
    inter_lig_product = list(itertools.product(
        *[list(itertools.permutations(indices)) for indices in smi_groups.values()]
    ))

    ref_xyz_to_matched = {ref_xyz: i for i, ref_xyz in enumerate(ref_xyz_list)}

    best_rmsd = float("inf")
    best_pred_m = pred_m.copy()
    n_evaluated = 0

    _BATCH_SIZE = 256
    _batch: List[np.ndarray] = []
    _batch_perms: List[np.ndarray] = []
    _batch_start = 0

    def _flush():
        nonlocal best_rmsd, best_pred_m
        if not _batch:
            return
        Ps = np.stack(_batch)
        rmsds = kabsch_rmsd_batch(Ps, ref_m)
        local_best = int(np.argmin(rmsds))
        if rmsds[local_best] < best_rmsd:
            best_rmsd = float(rmsds[local_best])
            best_pred_m = _batch_perms[local_best].copy()
        _batch.clear()
        _batch_perms.clear()

    for inter_combo in inter_lig_product:
        lig_assignment: Dict[int, int] = {}
        flat_idx = 0
        for smi, lig_indices in smi_groups.items():
            perm = inter_combo[flat_idx]
            for src, dst in zip(lig_indices, perm):
                lig_assignment[src] = dst
            flat_idx += 1

        intra_iters = [intra_lig_perms[lig_assignment[i]] for i in range(len(ligand_groups))]

        for intra_combo in itertools.product(*intra_iters):
            if n_evaluated >= max_perms:
                break

            pred_perm = pred_m.copy()
            for src_lig_i, (dst_lig_i, intra_auto) in enumerate(
                zip([lig_assignment[i] for i in range(len(ligand_groups))], intra_combo)
            ):
                src_lg = ligand_groups[src_lig_i]
                src_heavy_xyz = [
                    idx + 1 for idx in src_lg["xyz_indices_0based"]
                    if (idx + 1) in ref_xyz_to_matched
                ]
                for src_xyz_1b in src_heavy_xyz:
                    mapped = intra_auto.get(src_xyz_1b, src_xyz_1b)
                    src_row = ref_xyz_to_matched.get(src_xyz_1b)
                    dst_row = ref_xyz_to_matched.get(mapped)
                    if src_row is None or dst_row is None:
                        continue
                    pred_perm[src_row] = pred_m[dst_row]

            if len(_batch) == 0:
                _batch_start = n_evaluated
            _batch.append(pred_perm)
            _batch_perms.append(pred_perm)
            n_evaluated += 1

            if len(_batch) >= _BATCH_SIZE:
                _flush()

        if n_evaluated >= max_perms:
            break

    _flush()

    return best_pred_m, ref_m, ref_xyz_list, pred_row_list

# ---------------------------------------------------------------------------
# py3Dmol / overlay helpers (module-level so overlay_view can reference them)
# ---------------------------------------------------------------------------

# Standard CPK colours for common elements
_CPK: Dict[str, str] = {
    "H":  "#FFFFFF", "C":  "#404040", "N":  "#3050F8", "O":  "#FF0D0D",
    "S":  "#FFFF30", "P":  "#FF8000", "F":  "#90E050", "Cl": "#1FF01F",
    "Br": "#A62929", "I":  "#940094", "Fe": "#E06633", "Ru": "#248F8F",
    "Os": "#2194D6", "Co": "#F090A0", "Rh": "#0AD4D4", "Ir": "#175487",
    "Ni": "#50D050", "Pd": "#006985", "Pt": "#D0D0E0", "Cu": "#C88033",
    "Ag": "#C0C0C0", "Au": "#FFD123", "Zn": "#7D80B0", "Cd": "#FFD98F",
    "Mn": "#9C7AC7", "Re": "#267DAB", "W":  "#2194D6", "Mo": "#54B5B5",
    "Cr": "#8A99C7", "V":  "#A6A6AB", "Ti": "#BFC2C7",
}


def _cpk(symbol: str) -> str:
    """Return CPK hex colour for an element symbol, defaulting to grey."""
    return _CPK.get(symbol, "#AAAAAA")


def _xyz(pos) -> dict:
    """Convert a (3,) position array or sequence to a py3Dmol {x,y,z} dict."""
    return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}


def _lerp(a, b, t: float):
    """Linear interpolation between two positions at parameter t in [0,1]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a + t * (b - a)


# ---------------------------------------------------------------------------
# 3-D overlay visualisation
# ---------------------------------------------------------------------------


def overlay_view(
    ref_mol,
    pred_conformer,
    ligand_groups: List[Dict],
    width: int = 860,
    height: int = 520,
    ref_opacity: float = 1.0,
    pred_opacity: float = 0.72,
    show_displacement: bool = True,
    show_hydrogen: bool = False,
):
    """
    Render a py3Dmol overlay of the reference (tmQM) and predicted conformer,
    Kabsch-aligned on their matched heavy atoms.

    Colour convention
    -----------------
    Reference  : CPK colours, full opacity
    Predicted  : muted blue-grey tint, reduced opacity
    Metal      : bright gold (ref) vs steel-blue (pred)
    Displacement vectors : thin red sticks between matched atom pairs,
                           length proportional to RMSD contribution

    Parameters
    ----------
    ref_mol         : TMComplex — parsed reference geometry
    pred_conformer  : ConformerResult — engine output
    ligand_groups   : from _build_ligand_groups(ref_mol, ...)
    width, height   : py3Dmol view dimensions in pixels
    ref_opacity     : opacity of the reference structure (default 1.0)
    pred_opacity    : opacity of the predicted structure (default 0.72)
    show_displacement : draw red sticks between matched atom pairs
    show_hydrogen   : show H atoms in both structures

    Returns
    -------
    py3Dmol.view  — call .show() or display() in a notebook
    """
    import py3Dmol
    import copy

    # ── 1. Build symmetry-corrected matched arrays and Kabsch rotation ───────
    # Use build_symmetry_corrected_arrays so displacement vectors reflect the
    # best possible atom assignment (no cross-arm aziridine swaps, etc.).
    matched = build_symmetry_corrected_arrays(ref_mol, pred_conformer, ligand_groups)
    if matched is None:
        raise ValueError("build_symmetry_corrected_arrays returned None — cannot align")

    pred_matched, ref_matched, ref_xyz_list, pred_row_list = matched

    # Kabsch: find R, t that maps pred_matched → ref_matched
    p_mean = pred_matched.mean(axis=0)
    q_mean = ref_matched.mean(axis=0)
    Pc = pred_matched - p_mean
    Qc = ref_matched - q_mean
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    sign_fix = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_fix @ U.T          # (3, 3) optimal rotation
    t = q_mean - R @ p_mean            # (3,) translation

    def _transform(pos: np.ndarray) -> np.ndarray:
        """Apply Kabsch rotation+translation to a (3,) or (N,3) array."""
        if pos.ndim == 1:
            return R @ pos + t
        return (R @ pos.T).T + t

    # ── 2. Deep-copy predicted complex and apply the alignment transform ──────
    pred_aligned = copy.deepcopy(pred_conformer.complex)
    for atom in pred_aligned.atoms:
        atom.position = _transform(atom.position)
    # Also transform coord_sites if present
    for site in pred_aligned.coord_sites:
        site.position = _transform(site.position)

    # ── 3. Compute per-atom displacement from sym-corrected matched arrays ────
    # We use pred_matched / ref_matched directly (already Kabsch-aligned and
    # symmetry-permuted) rather than looking up pred_amap, which still has
    # atoms in original (pre-permutation) order and would give wrong vectors.
    #
    # Kabsch-align the sym-corrected pred_matched onto ref_matched
    p_mean_disp = pred_matched.mean(axis=0)
    q_mean_disp = ref_matched.mean(axis=0)
    Pc_disp = pred_matched - p_mean_disp
    Qc_disp = ref_matched  - q_mean_disp
    H_disp  = Pc_disp.T @ Qc_disp
    U_d, _, Vt_d = np.linalg.svd(H_disp)
    d_sign  = np.sign(np.linalg.det(Vt_d.T @ U_d.T))
    R_disp  = Vt_d.T @ np.diag([1., 1., d_sign]) @ U_d.T
    pred_aligned_rows = (R_disp @ Pc_disp.T).T + q_mean_disp  # (N_matched, 3)

    # displacement[i] = distance for matched-atom row i (in ref_xyz_list order)
    disp_per_row = np.linalg.norm(pred_aligned_rows - ref_matched, axis=1)
    max_disp = float(disp_per_row.max()) if len(disp_per_row) else 1.0
    # Map ref_xyz_1based → displacement Å
    displacement: Dict[int, float] = {
        ref_xyz_1b: float(disp_per_row[i])
        for i, ref_xyz_1b in enumerate(ref_xyz_list)
    }
    # Map ref_xyz_1based → aligned pred position (for drawing the vector end)
    pred_pos_corrected: Dict[int, np.ndarray] = {
        ref_xyz_1b: pred_aligned_rows[i]
        for i, ref_xyz_1b in enumerate(ref_xyz_list)
    }

    # ── 4. Build py3Dmol view ─────────────────────────────────────────────────
    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor("#0F1117")

    ref_atom_map         = {a.idx: a for a in ref_mol.atoms}
    ref_metal_pos        = ref_atom_map[ref_mol.metal_idx].position
    pred_atom_map_aligned = {a.idx: a for a in pred_aligned.atoms}   # Kabsch-aligned pred atoms
    pred_metal_pos_atom  = pred_atom_map_aligned.get(1)
    pred_metal_pos       = pred_metal_pos_atom.position if pred_metal_pos_atom else ref_metal_pos

    # ── Helper: CPK with metal override ──────────────────────────────────────
    def _ref_color(atom) -> str:
        if atom.idx == ref_mol.metal_idx:
            return "#FFD700"   # gold for reference metal
        return _cpk(atom.symbol)

    def _pred_color(atom) -> str:
        if atom.idx == 1:      # metal is always idx=1 in engine
            return "#4DA6FF"   # steel-blue for predicted metal
        return _cpk(atom.symbol)

    def _muted(hex_color: str, factor: float = 0.55) -> str:
        """Blend hex_color toward #8899AA (cool grey) to visually distinguish pred."""
        grey = np.array([0x88, 0x99, 0xAA], dtype=float)
        c    = np.array([int(hex_color[1:3], 16),
                         int(hex_color[3:5], 16),
                         int(hex_color[5:7], 16)], dtype=float)
        blended = c * (1 - factor) + grey * factor
        r, g, b = int(blended[0]), int(blended[1]), int(blended[2])
        return f"#{r:02X}{g:02X}{b:02X}"

    h_donor_set_ref  = set()
    h_donor_set_pred = set()
    donor_labels = getattr(ref_mol, "donor_labels", {})
    for idx, lbl in donor_labels.items():
        if lbl.subclass == "H_hydride":
            h_donor_set_ref.add(idx)

    # ── 4a. Reference atoms and bonds ─────────────────────────────────────────
    metal_bond_keys_ref = {
        (min(b.i, b.j), max(b.i, b.j))
        for b in ref_mol.bonds
        if b.i == ref_mol.metal_idx or b.j == ref_mol.metal_idx
    }
    for a in ref_mol.atoms:
        if not show_hydrogen and a.symbol == "H" and a.idx not in h_donor_set_ref:
            continue
        color = _ref_color(a)
        radius = 0.30 if a.idx == ref_mol.metal_idx else 0.22
        view.addSphere({
            "center": _xyz(a.position), "radius": radius,
            "color": color, "opacity": ref_opacity,
        })
    for b in ref_mol.bonds:
        key = (min(b.i, b.j), max(b.i, b.j))
        if key in metal_bond_keys_ref or b.order < 0.5:
            continue
        ai, aj = ref_atom_map[b.i], ref_atom_map[b.j]
        if not show_hydrogen and (
            (ai.symbol == "H" and b.i not in h_donor_set_ref) or
            (aj.symbol == "H" and b.j not in h_donor_set_ref)
        ):
            continue
        mid = _lerp(ai.position, aj.position, 0.5)
        for start, end, col in [
            (ai.position, mid, _ref_color(ai)),
            (mid, aj.position, _ref_color(aj)),
        ]:
            view.addCylinder({
                "start": _xyz(start), "end": _xyz(end),
                "radius": 0.06, "color": col, "opacity": ref_opacity,
            })

    # ── 4b. Predicted atoms and bonds (muted, aligned) ────────────────────────
    pred_atom_list   = pred_aligned.atoms
    pred_bond_list   = pred_aligned.bonds
    pred_metal_idx_p = 1   # always 1 in engine
    metal_bond_keys_pred = {
        (min(b.i, b.j), max(b.i, b.j))
        for b in pred_bond_list
        if b.i == pred_metal_idx_p or b.j == pred_metal_idx_p
    }
    pred_amap = {a.idx: a for a in pred_atom_list}
    for a in pred_atom_list:
        if not show_hydrogen and a.symbol == "H":
            continue
        color = _muted(_pred_color(a))
        radius = 0.28 if a.idx == pred_metal_idx_p else 0.20
        view.addSphere({
            "center": _xyz(a.position), "radius": radius,
            "color": color, "opacity": pred_opacity,
        })
    for b in pred_bond_list:
        key = (min(b.i, b.j), max(b.i, b.j))
        if key in metal_bond_keys_pred or b.order < 0.5:
            continue
        ai, aj = pred_amap.get(b.i), pred_amap.get(b.j)
        if ai is None or aj is None:
            continue
        if not show_hydrogen and (ai.symbol == "H" or aj.symbol == "H"):
            continue
        mid = _lerp(ai.position, aj.position, 0.5)
        for start, end, col in [
            (ai.position, mid, _muted(_pred_color(ai))),
            (mid, aj.position, _muted(_pred_color(aj))),
        ]:
            view.addCylinder({
                "start": _xyz(start), "end": _xyz(end),
                "radius": 0.055, "color": col, "opacity": pred_opacity,
            })

    # ── 4c. Reference coord-site annotations ──────────────────────────────────
    for site in ref_mol.coord_sites:
        pos = np.asarray(site.position, dtype=float)
        n_beads = 9
        for k in range(n_beads):
            t_f = (k + 0.5) / n_beads
            view.addSphere({
                "center": _xyz(_lerp(pos, ref_metal_pos, t_f)),
                "radius": 0.045, "color": "#FFD700", "opacity": 0.55,
            })
        view.addSphere({
            "center": _xyz(pos),
            "radius": 0.50 * (1.2 if site.is_haptic else 1.0),
            "color": "#FFD700",
            "opacity": 0.15,
        })
        if site.label:
            label_pos = pos + np.array([0.0, 0.75, 0.0])
            view.addLabel(site.label, {
                "position": _xyz(label_pos), "fontSize": 10,
                "fontColor": "#FFD700", "backgroundColor": "#0F1117",
                "backgroundOpacity": 0.6, "inFront": True, "showBackground": True,
            })

    # ── 4d. Displacement vectors (sym-corrected, from matched arrays) ─────────
    if show_displacement:
        for ref_xyz_1based in ref_xyz_list:
            ref_atom = ref_atom_map.get(ref_xyz_1based)
            if ref_atom is None:
                continue
            if not show_hydrogen and ref_atom.symbol == "H":
                continue
            disp = displacement.get(ref_xyz_1based, 0.0)
            if disp < 0.05:
                continue
            pred_end = pred_pos_corrected.get(ref_xyz_1based)
            if pred_end is None:
                continue
            # Green → yellow → red by displacement / max_disp
            frac = min(disp / max(max_disp, 0.1), 1.0)
            r_ = int(255 * min(frac * 2, 1.0))
            g_ = int(255 * max(1.0 - (frac - 0.5) * 2, 0.0)) if frac > 0.5 else 255
            disp_color = f"#{r_:02X}{g_:02X}00"
            view.addCylinder({
                "start": _xyz(ref_atom.position),
                "end":   _xyz(pred_end),
                "radius": 0.035, "color": disp_color, "opacity": 0.85,
            })

    # ── 4e. RMSD labels (full + CS) ───────────────────────────────────────────
    final_rmsd = float(np.sqrt((disp_per_row ** 2).mean())) if len(disp_per_row) else float("nan")

    # CS RMSD: subset at graph distance ≤ 2
    local_xyz = _graph_distance_atoms(ref_mol, max_bonds=2)
    cs_rows   = [i for i, r in enumerate(ref_xyz_list) if r in local_xyz]
    if len(cs_rows) >= 3:
        cs_rmsd = float(np.sqrt((disp_per_row[cs_rows] ** 2).mean()))
        cs_str  = f"  CS={cs_rmsd:.3f} Å"
    else:
        cs_str  = ""

    rank_str  = f"rank-{getattr(pred_conformer, 'rank', '?')}"
    label_pos = ref_metal_pos + np.array([0.0, 2.5, 0.0])
    view.addLabel(
        f"Full={final_rmsd:.3f} Å{cs_str}  |  {ref_mol.csd_code}  {rank_str}",
        {
            "position": _xyz(label_pos), "fontSize": 11,
            "fontColor": "#FFFFFF", "backgroundColor": "#0F1117",
            "backgroundOpacity": 0.75, "inFront": True, "showBackground": True,
        },
    )

    view.zoomTo()
    view.rotate(20, "y")
    view.rotate(10, "x")
    return view


def overlay_widget(
    ref_mol,
    result,              # EnsembleResult
    ligand_groups: List[Dict],
    width: int = 860,
    height: int = 520,
):
    """
    Interactive overlay viewer.

    Works in both classic Jupyter and VS Code notebooks.

    The 3-D view renders directly into the cell output (not inside an Output
    widget container) so py3Dmol mouse controls (rotate, zoom, pan) work
    correctly in VS Code.  The ipywidgets controls live above the view and
    each change re-renders into the *next* cell output via IPython.display.

    Controls
    --------
    Conformer selector, ref/pred opacity sliders, displacement toggle, H toggle.
    RMSD breakdown (Full / CS / Local, per-ligand) shown as text below controls.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display as _display, clear_output
    except ImportError:
        raise ImportError("ipywidgets required. Use overlay_view() for a plain py3Dmol view.")

    conformers = result.conformers
    if not conformers:
        raise ValueError("EnsembleResult has no conformers")

    # Pre-compute full RMSD labels for the conformer selector
    conf_rmsds: List[str] = []
    for cf in conformers:
        matched = _build_matched_arrays(ref_mol, cf, ligand_groups)
        if matched is None:
            conf_rmsds.append("?")
            continue
        pm, rm, _, _ = matched
        rv, _ = kabsch_rmsd(pm, rm)
        conf_rmsds.append(f"{rv:.3f}")

    conf_options = [
        f"#{i+1}  score={cf.score:.3f}  RMSD={conf_rmsds[i]} Å"
        + (" ✓" if cf.converged else "")
        for i, cf in enumerate(conformers)
    ]

    # ── Build controls ────────────────────────────────────────────────────
    conf_sel   = widgets.Select(
        options=conf_options, value=conf_options[0], description="Conformer",
        rows=min(len(conformers), 6), layout=widgets.Layout(width="380px"))
    ref_op_sl  = widgets.FloatSlider(
        value=1.0, min=0.0, max=1.0, step=0.05, description="Ref opacity",
        layout=widgets.Layout(width="300px"))
    pred_op_sl = widgets.FloatSlider(
        value=0.72, min=0.0, max=1.0, step=0.05, description="Pred opacity",
        layout=widgets.Layout(width="300px"))
    disp_toggle = widgets.Checkbox(value=True,  description="Displacement", indent=False)
    h_toggle    = widgets.Checkbox(value=False, description="Show H",       indent=False)
    render_btn  = widgets.Button(description="↻ Render", button_style="primary",
                                 layout=widgets.Layout(width="100px"))

    # RMSD text output (always visible, updates on render)
    stats_html = widgets.HTML(value="<i>Click ↻ Render or change a control</i>")

    # View output — py3Dmol renders here
    view_out = widgets.Output()

    def _rmsd_html(cf) -> str:
        matched = _build_matched_arrays(ref_mol, cf, ligand_groups)
        if matched is None:
            return "<b>No atom matching</b>"
        pm, rm, rxl, _ = matched
        _, P_al = kabsch_rmsd(pm, rm)
        full_r  = float(np.sqrt(((P_al - rm)**2).sum(-1).mean()))
        lxyz    = _graph_distance_atoms(ref_mol, 2)
        cs_rows = [i for i, r in enumerate(rxl) if r in lxyz]
        cs_r    = (float(np.sqrt((((P_al[cs_rows] - rm[cs_rows])**2).sum(-1).mean())))
                   if len(cs_rows) >= 3 else float("nan"))
        loc_res = per_ligand_local_rmsd(ref_mol, cf, ligand_groups)
        vloc    = [r for _, r, _ in loc_res if r == r]
        loc_r   = float(np.mean(vloc)) if vloc else float("nan")

        cs_s  = f"{cs_r:.3f}" if cs_r == cs_r else "nan"
        loc_s = f"{loc_r:.3f}" if loc_r == loc_r else "nan"
        rows  = (f"<tr><td><b>Score</b></td><td>{cf.score:.4f}</td></tr>"
                 f"<tr><td><b>Full RMSD</b></td><td>{full_r:.3f} Å "
                 f"<small>({len(rxl)} atoms)</small></td></tr>"
                 f"<tr><td><b>CS RMSD</b></td><td>{cs_s} Å "
                 f"<small>(d≤2, {len(cs_rows)} atoms)</small></td></tr>"
                 f"<tr><td><b>Local RMSD</b></td><td>{loc_s} Å "
                 f"<small>(backbone mean)</small></td></tr>")
        for lig_id, lr, nb in loc_res:
            lr_s = f"{lr:.3f} Å" if lr == lr else "nan"
            rows += (f"<tr><td style='padding-left:12px;color:#888'>"
                     f"lig {lig_id}</td><td>{lr_s} "
                     f"<small>({nb} backbone)</small></td></tr>")
        return (f"<table style='font-size:12px;border-collapse:collapse'>"
                f"{rows}</table>")

    def _render(*_):
        cf = conformers[conf_sel.index]
        stats_html.value = _rmsd_html(cf)
        # Render py3Dmol into the Output widget.
        # In VS Code: the Output widget renders inline and py3Dmol
        # mouse controls work because it owns the full output area.
        with view_out:
            clear_output(wait=True)
            v = overlay_view(
                ref_mol=ref_mol,
                pred_conformer=cf,
                ligand_groups=ligand_groups,
                width=width, height=height,
                ref_opacity=ref_op_sl.value,
                pred_opacity=pred_op_sl.value,
                show_displacement=disp_toggle.value,
                show_hydrogen=h_toggle.value,
            )
            _display(v)   # display() is more reliable than show() in VS Code

    render_btn.on_click(_render)
    # Also auto-render on conformer change (most useful control)
    conf_sel.observe(_render, names="value")
    # Sliders and toggles only render on explicit button click to avoid
    # excessive re-renders while dragging sliders (which would be laggy)

    _render()  # initial render

    controls = widgets.VBox([
        conf_sel,
        widgets.HBox([ref_op_sl, pred_op_sl]),
        widgets.HBox([disp_toggle, h_toggle, render_btn]),
        widgets.HTML("<b style='font-size:13px'>RMSD Breakdown</b>"),
        stats_html,
    ], layout=widgets.Layout(width="420px", padding="4px"))

    ui = widgets.HBox([controls, view_out])
    _display(ui)
    return ui

# ---------------------------------------------------------------------------
# Diagnostic function
# ---------------------------------------------------------------------------


def diagnose(
    csd_code: str,
    data_dir: str | Path,
    split: str,
    constraints_db: str | Path | None = None,
    beam_width: int = 16,
    n_ik_steps: int = 200,
    n_coord_samples: int = 8,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    """
    Detailed per-stage diagnosis for a single complex.

    Prints a structured report covering every failure mode:

      Stage 1  Reference parsing
               — atom count, coord sites, geometry, ligand SMILES match rate
      Stage 2  Ligand group construction
               — per-ligand SMILES, VF2 mapping success, donor groups,
                 automorphism count (reveals phenyl flips, Cp rotations, etc.)
      Stage 3  Atom matching (heavy-atom correspondence)
               — how many heavy atoms were matched vs total, per ligand;
                 mismatches here corrupt RMSD regardless of solver quality
      Stage 4  Engine solve
               — conformer scores, convergence flags
      Stage 5  Per-conformer RMSD breakdown
               — sym_rmsd for each conformer; score vs RMSD scatter to reveal
                 whether the scorer is anti-correlated with geometry quality
      Stage 6  Donor-position MAE
               — angle-invariant check: mean angular error on D-M-D angles
                 for the rank-1 conformer vs reference, after optimal alignment
    """
    from jacobik.stats.parser import TMQMParser
    from jacobik.engine import ComplexBuilder
    import copy

    data_dir = Path(data_dir)
    parser = TMQMParser(data_dir)

    # ── Load the target complex ───────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  DIAGNOSIS: {csd_code}")
    print(f"{'='*64}")

    mol = None
    for m in parser.stream_split(split):
        if m.csd_code == csd_code:
            mol = m
            break
    if mol is None:
        print(f"  ERROR: {csd_code} not found in split {split!r}")
        return

    # ── Stage 1: Reference ───────────────────────────────────────────────────
    print(f"\n── Stage 1  Reference geometry ─────────────────────────────────")
    print(f"  Metal        : {mol.metal_symbol}  (atom idx {mol.metal_idx})")
    print(f"  Coord geom   : {mol.coord_geom or 'unknown'}")
    print(f"  CN           : {mol.coordination_number()}")
    print(f"  Total atoms  : {len(mol.atoms)}  "
          f"(H: {sum(1 for a in mol.atoms if a.symbol=='H')}, "
          f"heavy: {sum(1 for a in mol.atoms if a.symbol!='H')})")
    print(f"  Ligands      : {len(mol.ligands)}")
    print(f"  SMILES       : {'present' if mol.smiles else 'MISSING — cannot benchmark'}")
    if not mol.smiles:
        return
    for site in mol.coord_sites:
        print(f"    site  {site.label:<8}  kind={site.kind:<6}  "
              f"pos=[{site.position[0]:+.2f} {site.position[1]:+.2f} {site.position[2]:+.2f}]")

    # ── Stage 2: Ligand groups ────────────────────────────────────────────────
    print(f"\n── Stage 2  Ligand group construction ──────────────────────────")
    try:
        ligand_groups = _build_ligand_groups(mol, mol.smiles)
    except Exception as exc:
        print(f"  ERROR: _build_ligand_groups failed: {exc}")
        return

    for lg in ligand_groups:
        n_auto  = len(lg["xyz_automorphisms"])
        n_donor = sum(len(g) for g in lg["donor_groups_rdkit"])
        n_map   = len(lg["rdkit_to_xyz"])
        n_heavy_ref = sum(1 for i in lg["xyz_indices_0based"]
                          if mol.atoms[i].symbol != "H")
        map_ok  = "✓" if n_map == n_heavy_ref else f"✗ ({n_map}/{n_heavy_ref} mapped)"
        print(f"  lig {lg['ligand_id']}  smi={lg['canonical_smiles'][:40]:<40}  "
              f"map={map_ok}  autos={n_auto:>4}  donors={n_donor}")

    # ── Stage 3: Atom matching ────────────────────────────────────────────────
    print(f"\n── Stage 3  Heavy-atom matching ────────────────────────────────")
    # Use a dummy conformer wrapping the reference to test _build_matched_arrays
    from dataclasses import dataclass as _dc2
    @_dc2
    class _Fake:
        complex: object
        converged: bool = True
        score: float = 0.0
    fake_conf = _Fake(complex=copy.deepcopy(mol))
    matched = _build_matched_arrays(mol, fake_conf, ligand_groups)
    if matched is None:
        print("  ERROR: _build_matched_arrays returned None (no atoms matched)")
        return
    pred_m, ref_m, ref_xyz_list, pred_rows = matched
    n_total_heavy = sum(1 for a in mol.atoms if a.symbol != "H")
    print(f"  Matched heavy atoms : {len(ref_xyz_list)} / {n_total_heavy}")
    if len(ref_xyz_list) < n_total_heavy:
        unmatched_xyz = set(a.idx for a in mol.atoms if a.symbol != "H") - set(ref_xyz_list)
        print(f"  Unmatched xyz idx   : {sorted(unmatched_xyz)[:20]}"
              f"{'...' if len(unmatched_xyz) > 20 else ''}")
    # Self-RMSD should be ~0
    self_rmsd, _ = kabsch_rmsd(pred_m, ref_m)
    status = "✓" if self_rmsd < 1e-6 else f"✗  BUG — {self_rmsd:.2e} Å"
    print(f"  Self-RMSD (ref,ref) : {self_rmsd:.2e} Å  {status}")

    # ── Stage 4: Engine solve ─────────────────────────────────────────────────
    print(f"\n── Stage 4  Engine solve ───────────────────────────────────────")
    lig_inputs = ligand_inputs_from_mol(mol, ligand_groups)
    if lig_inputs is None:
        print("  ERROR: ligand_inputs_from_mol returned None")
        return
    geom = mol.coord_geom or "other"
    try:
        builder = ComplexBuilder(
            metal=mol.metal_symbol, geometry=geom,
            charge=mol.charge, spin=mol.spin_state, ligands=lig_inputs,
        )
    except ValueError as exc:
        print(f"  ERROR: ComplexBuilder: {exc}")
        return

    import time
    t0 = time.perf_counter()
    try:
        result = builder.solve(
            constraints_db=constraints_db,
            beam_width=beam_width, n_ik_steps=n_ik_steps,
            n_coord_samples=n_coord_samples, seed=seed, device=device,
            fast_embed=False,  # diagnose uses full MMFF for accurate geometry
        )
    except Exception as exc:
        print(f"  ERROR: solve() raised: {exc}")
        return
    elapsed = time.perf_counter() - t0

    print(f"  Solved in {elapsed:.1f}s  →  {len(result.conformers)} conformers")
    print(f"  {'Rank':<5} {'Score':>10}  {'Converged':>9}  {'RMSD (Å)':>10}  {'Note'}")
    print(f"  {'-'*55}")

    # ── Stage 5: Per-conformer RMSD ──────────────────────────────────────────
    conformer_rmsds = []
    for ci, conformer in enumerate(result.conformers):
        matched_c = _build_matched_arrays(mol, conformer, ligand_groups)
        if matched_c is None:
            conformer_rmsds.append(float("nan"))
            print(f"  {ci+1:<5} {conformer.score:>10.4f}  "
                  f"{'yes' if conformer.converged else 'no':>9}  {'—':>10}  match failed")
            continue
        pred_mc, ref_mc, ref_xyz_c, pred_rows_c = matched_c
        sr = sym_rmsd(
            pred_matched=pred_mc, ref_matched=ref_mc,
            ligand_groups=ligand_groups,
            heavy_atom_map=ref_xyz_c, pred_heavy_map=pred_rows_c,
            max_perms=2048,
        )
        conformer_rmsds.append(sr.rmsd)
        note = ""
        if ci == 0:
            note = "← rank-1"
        if sr.rmsd == min(r for r in conformer_rmsds if r == r):
            note += " ← best"
        print(f"  {ci+1:<5} {conformer.score:>10.4f}  "
              f"{'yes' if conformer.converged else 'no':>9}  {sr.rmsd:>10.3f}  {note}")

    finite_rmsds = [r for r in conformer_rmsds if r == r]
    if finite_rmsds:
        print(f"\n  rank-1 RMSD : {conformer_rmsds[0]:.3f} Å")
        print(f"  best RMSD   : {min(finite_rmsds):.3f} Å")
        best_rank = conformer_rmsds.index(min(finite_rmsds)) + 1
        print(f"  best rank   : {best_rank}  "
              + ("(scorer selected best)" if best_rank == 1
                 else f"(scorer ranked best as #{best_rank} — gap = "
                      f"{conformer_rmsds[0] - min(finite_rmsds):.3f} Å)"))

    # ── Stage 6: Donor-position MAE ───────────────────────────────────────────
    print(f"\n── Stage 6  Donor D-M-D angle MAE (rank-1 vs reference) ────────")
    if result.conformers:
        rank1 = result.conformers[0]
        # Reference donor positions (metal-centred unit vectors)
        ref_metal_pos  = np.array(mol.atoms[mol.metal_idx - 1].position)
        pred_metal_pos = np.array(rank1.complex.atoms[0].position)  # metal always idx 0

        ref_donor_vecs, pred_donor_vecs = [], []
        for site in mol.coord_sites:
            ref_d = site.position - ref_metal_pos
            ref_d /= (np.linalg.norm(ref_d) + 1e-12)
            ref_donor_vecs.append(ref_d)

        # Match pred coord sites by nearest direction
        pred_sites = rank1.complex.coord_sites
        if pred_sites:
            for ref_d in ref_donor_vecs:
                best_cos, best_pred = -2.0, None
                for ps in pred_sites:
                    pd = ps.position - pred_metal_pos
                    pd /= (np.linalg.norm(pd) + 1e-12)
                    cos = float(np.dot(ref_d, pd))
                    if cos > best_cos:
                        best_cos, best_pred = cos, pd
                if best_pred is not None:
                    pred_donor_vecs.append(best_pred)

            if len(ref_donor_vecs) == len(pred_donor_vecs):
                angles_deg = []
                for rv, pv in zip(ref_donor_vecs, pred_donor_vecs):
                    cos = np.clip(np.dot(rv, pv), -1, 1)
                    angles_deg.append(np.degrees(np.arccos(cos)))
                mae = float(np.mean(angles_deg))
                print(f"  Mean angular error per donor: {mae:.2f}°")
                for i, ang in enumerate(angles_deg):
                    print(f"    site {i}: {ang:.2f}°")
            else:
                print(f"  Donor count mismatch: ref={len(ref_donor_vecs)} "
                      f"pred={len(pred_donor_vecs)}")
        else:
            print("  No coord sites on predicted conformer.")

    print(f"\n{'='*64}\n")


# ---------------------------------------------------------------------------
# Standalone single-complex analysis helper
# ---------------------------------------------------------------------------


def analyse_complex(
    csd_code: str,
    data_dir,
    split: str,
    constraints_db=None,
    beam_width: int = 16,
    n_ik_steps: int = 200,
    n_coord_samples: int = 8,
    seed: int = 42,
    device: str = "cpu",
    use_xtb: bool = False,
    max_bonds: int = 3,
) -> Dict:
    """
    Parse, solve, and compute all metrics for a single complex by CSD code.

    This is the entry point for standalone per-complex analysis — replaces
    the need to hold internal loop variables from run_benchmark.

    Returns a dict with keys:
      mol              : TMComplex (parsed reference)
      result           : EnsembleResult (engine output)
      ligand_groups    : list of ligand group dicts
      rank1_cs_rmsd    : float — graph-distance CS RMSD, rank-1 conformer
      rank1_full_rmsd  : float — full symmetry-corrected RMSD, rank-1
      rank1_local_rmsd : float — mean per-ligand local RMSD, rank-1
      xtb_delta_kj     : float — GFN2-xTB ΔE in kJ/mol (nan if use_xtb=False)
      records          : list with one BenchmarkRecord

    Example
    -------
    >>> result = analyse_complex("VUCVUN", DATA_DIR, "X1", use_xtb=True)
    >>> print(f"CS RMSD = {result['rank1_cs_rmsd']:.3f} Å")
    >>> print(f"ΔE = {result['xtb_delta_kj']:+.1f} kJ/mol")
    >>> overlay_widget(result['mol'], result['result'], result['ligand_groups'])
    """
    from jacobik.stats.parser import TMQMParser
    from jacobik.stats.stats import StatsLookup, load_db
    from jacobik.engine import ComplexBuilder
    from pathlib import Path as _Path

    data_dir = _Path(data_dir)
    parser   = TMQMParser(data_dir)

    mol = None
    for m in parser.stream_split(split):
        if m.csd_code == csd_code:
            mol = m
            break
    if mol is None:
        raise ValueError(f"{csd_code!r} not found in split {split!r}")

    if not mol.smiles:
        raise ValueError(f"{csd_code}: no SMILES — cannot run engine")

    ligand_groups = _build_ligand_groups(mol, mol.smiles)
    lig_inputs    = ligand_inputs_from_mol(mol, ligand_groups)
    if lig_inputs is None:
        raise ValueError(f"{csd_code}: could not build ligand inputs")

    # Load stats once
    db_path = constraints_db
    if db_path is None:
        from importlib import resources as _res
        db_path = _res.files("jacobik") / "constraints.json"
    stats = StatsLookup(load_db(db_path))

    geom    = mol.coord_geom or "other"
    builder = ComplexBuilder(
        metal=mol.metal_symbol, geometry=geom,
        charge=mol.charge, spin=mol.spin_state, ligands=lig_inputs,
    )
    result = builder.solve(
        stats=stats, beam_width=beam_width, n_ik_steps=n_ik_steps,
        n_coord_samples=n_coord_samples, seed=seed, device=device,
        fast_embed=False,  # full quality for interactive analysis
    )

    if not result.conformers:
        raise RuntimeError(f"{csd_code}: engine returned no conformers")

    rank1 = result.conformers[0]

    # CS RMSD
    cs_rmsd, n_cs = coord_sphere_rmsd(mol, rank1, ligand_groups, max_bonds=max_bonds)

    # Full sym RMSD
    matched = _build_matched_arrays(mol, rank1, ligand_groups)
    if matched is not None:
        pm, rm, rxl, prl = matched
        sr = sym_rmsd(pred_matched=pm, ref_matched=rm, ligand_groups=ligand_groups,
                      heavy_atom_map=rxl, pred_heavy_map=prl, max_perms=2048)
        full_rmsd = sr.rmsd
    else:
        full_rmsd = float("nan")

    # Local RMSD
    local_results   = per_ligand_local_rmsd(mol, rank1, ligand_groups)
    valid_locals    = [r for _, r, _ in local_results if r == r]
    local_rmsd      = float(np.mean(valid_locals)) if valid_locals else float("nan")

    # xTB
    dkj = float("nan")
    if use_xtb:
        dkj, _, _ = xtb_energy_gap(mol, rank1, mol.charge, mol.spin_state)

    return {
        "mol":              mol,
        "result":           result,
        "ligand_groups":    ligand_groups,
        "rank1_cs_rmsd":    cs_rmsd,
        "rank1_full_rmsd":  full_rmsd,
        "rank1_local_rmsd": local_rmsd,
        "xtb_delta_kj":     dkj,
        "n_cs_atoms":       n_cs,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Benchmark Jacobik engine against tmQM reference geometries."
    )
    ap.add_argument("data_dir", help="tmQM dataset root directory")
    ap.add_argument("--split", default="X1", help="tmQM split (default: X1)")
    ap.add_argument("--constraints-db", default=None, help="Path to constraints.json")
    ap.add_argument("--limit", type=int, default=None, help="Max complexes to benchmark")
    ap.add_argument("--beam-width", type=int, default=16)
    ap.add_argument("--n-ik-steps", type=int, default=200)
    ap.add_argument("--n-coord-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-sym-perms", type=int, default=2048)
    ap.add_argument("--output", default=None, help="Save records as JSON to this path")
    ap.add_argument("--log-file", default=None, help="Write per-complex detail log to this path")
    ap.add_argument("--use-xtb", action="store_true", default=False,
                    help="Compute GFN2-xTB single-point energy gap per complex (requires xtb package)")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    summary = run_benchmark(
        data_dir=args.data_dir,
        split=args.split,
        constraints_db=args.constraints_db,
        limit=args.limit,
        beam_width=args.beam_width,
        n_ik_steps=args.n_ik_steps,
        n_coord_samples=args.n_coord_samples,
        seed=args.seed,
        device=args.device,
        max_sym_perms=args.max_sym_perms,
        log_file=args.log_file,
        use_xtb=args.use_xtb,
    )

    summary.print_summary()

    if args.output:
        out = []
        for r in summary.records:
            out.append({
                "csd_code": r.csd_code,
                "metal": r.metal,
                "coord_geom": r.coord_geom,
                "cn": r.cn,
                "n_atoms": r.n_atoms,
                "n_ligands": r.n_ligands,
                "rank1_rmsd": r.rank1_rmsd,
                "best_rmsd": r.best_rmsd,
                "rank1_converged": r.rank1_converged,
                "n_conformers": r.n_conformers,
                "n_sym_perms": r.n_sym_perms,
                "engine_time_s": r.engine_time_s,
                "error": r.error,
            })
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Records saved to {args.output}")


if __name__ == "__main__":
    main()