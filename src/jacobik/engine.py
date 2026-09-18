"""
engine.py
=========
Public API and orchestration for the Jacobik conformer ensemble engine.

ComplexBuilder.solve() runs the full pipeline for one complex: build an
initial 3D shell from each ligand's SMILES, decompose ligands into
kinematic trees, beam-search torsion seeds, then jointly refine all
ligands' degrees of freedom with JointIKSolver (ik/ik_solver.py) against
the Scorer's statistical objective plus shape/symmetry constraints.
Returns an EnsembleResult of conformers ranked best-to-worst by score.

Notable non-default knobs on solve():
  enable_bend_dof         : gives branch-point donors (e.g. a tertiary
                             amine N) independent per-branch control via
                             extra "bend" joints (see ik/fk.py).
  use_assignment_search    : when the baseline Hungarian donor-to-site
                             assignment is ambiguous, tries alternative
                             assignments and keeps whichever solves best
                             (ik/assignment_search.py). Off by default;
                             unambiguous cases are unaffected either way.
  use_symmetry_enumeration : proactive alternative to use_assignment_search
                             (mutually exclusive with it): exhaustively
                             enumerates every symmetry-distinct donor-to-
                             site bijection up front (ik/site_enumeration.py,
                             see its module docstring for the Architector-
                             style motivation), instead of perturbing the
                             one baseline Hungarian assignment. Off by
                             default. See use_symmetry_enumeration's own
                             parameter docstring below for the tradeoffs.
  q3_mode_commit           : commits a topologically-symmetric donor group
                             to a single Q3 GMM mode (from its initial
                             pose) instead of tying members toward a mean
                             that can fall between two real modes.

See IKSolverConfig (ik/ik_solver.py) for the shape/symmetry-constraint
hyperparameters threaded through from here.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from jacobik.ik.site_enumeration import SiteEnumerationConfig

log = logging.getLogger(__name__)

#: Prominence floor for Q1 GMM modes: modes whose weight is below this
#: fraction of the maximum-weight mode are dropped as artefacts before
#: sampling.  The Q1 GMM for amine donors carries a spurious ~0.045 Å
#: "bond" mode (< 5% of max weight) that is a fitting artefact, not a
#: real distribution feature.  A 10% prominence floor drops it while
#: keeping all real modes.  (Finding F4 in REWRITE_PLAN.md.)
_Q1_PROMINENCE_FLOOR: float = 0.10


@dataclass
class AssignmentSearchConfig:
    """
    Tunable hyperparameters for ComplexBuilder.solve(use_assignment_search=True).

    k_max          : max candidate donor-to-site assignments (including the
        baseline) tried. Kept low by default -- 2026-09-17: no longer
        because "each candidate costs a full beam+IK solve" (that was true
        of the old sequential screen-then-resolve design; see
        ComplexBuilder._solve_for_assignments's "batched candidate
        screening" note), but because k_max also bounds how large the
        single batched solve's B dimension grows
        (B = k_max * beam_width * n_coord_samples) -- GPU memory, not
        wall-clock, is the real constraint now.
    margin_deg     : "near-tied" threshold (degrees) for flagging a donor as
        suspect; see ik/assignment_search.py's find_suspect_donors.
    max_ambiguous_group : cap on how many mutually-suspect donors are
        jointly permuted when generating candidates.

    2026-09-17: screen_beam_width/screen_n_ik_steps (the cheap-screening-
    pass knobs for the old sequential "screen every candidate cheaply,
    re-solve only the winner at full quality" design) are gone --
    every candidate assignment is now solved at the SAME (beam_width,
    n_ik_steps) quality in one batched call, since batching made the
    screen/resolve split's whole reason for existing (avoiding N separate
    full-quality solves) moot. See _solve_for_assignments.
    """

    k_max: int = 4
    margin_deg: float = 20.0
    max_ambiguous_group: int = 4


def geometry_sites(geometry: str) -> np.ndarray:
    """
    Return the (CN, 3) array of ideal unit-vector site positions for a
    coordination geometry.


    Parameters
    ----------
    geometry : str — one of the labels in IDEAL_GEOMETRIES
               (e.g. "octahedral", "square_plan", "tetrahedral", ...)

    Returns
    -------
    (CN, 3) float64 array, one row per site.

    Example
    -------
    >>> geometry_sites("octahedral")
    array([[ 1.,  0.,  0.],   # site 0 (+x)
           [-1.,  0.,  0.],   # site 1 (−x)
           [ 0.,  1.,  0.],   # site 2 (+y)
           [ 0., -1.,  0.],   # site 3 (−y)
           [ 0.,  0.,  1.],   # site 4 (+z)
           [ 0.,  0., -1.]])  # site 5 (−z)

    """
    from jacobik.ik.targets import IDEAL_GEOMETRIES

    if geometry not in IDEAL_GEOMETRIES:
        raise ValueError(f"Unknown geometry {geometry!r}. Valid: {sorted(IDEAL_GEOMETRIES)}")
    return IDEAL_GEOMETRIES[geometry].copy()


# ---------------------------------------------------------------------------
# LigandInput  (user-facing)
# ---------------------------------------------------------------------------


class LigandInput:
    """
    User-facing description of one ligand.

    Parameters
    ----------
    smiles : str
        SMILES string for the ligand.  May be hydrogen-explicit or
        hydrogen-implicit; the engine auto-detects which.
    donors : list[list[int]]
        Donor groups.  Each inner list is a set of atom indices (0-based
        on the SMILES heavy-atom graph for H-implicit SMILES, or 0-based
        on the full graph for H-explicit SMILES) that form one coordination
        point.  Single-atom kappa donor: ``[[9]]``.  Haptic Cp ring:
        ``[[0,1,2,3,4]]``.  Mixed bidentate: ``[[9], [1,2,3,4,5]]``.
    sites : list[int] | None
        Explicit 0-indexed coordination site assignments, one per donor
        group.  ``sites[i]`` is the site index (from ``geometry_sites()``)
        that ``donors[i]`` should point toward.  If None (default), the
        engine performs Hungarian matching to assign donors to sites
        automatically.

    Raises
    ------
    ValueError
        If ``sites`` is provided but has a different length than ``donors``.

    Examples
    --------
    # Monodentate PPh3 — let engine decide placement
    LigandInput("P(c1ccccc1)(c1ccccc1)c1ccccc1", donors=[[0]])

    # η5-Cp — auto-matched
    LigandInput("C1=CC=CC=1", donors=[[0,1,2,3,4]])

    # Bidentate with explicit sites (octahedral complex, trans positions)
    LigandInput("c1ccncc1", donors=[[0], [4]], sites=[0, 1])
    """

    def __init__(
        self,
        smiles: str,
        donors: List[List[int]],
        sites: Optional[List[int]] = None,
    ) -> None:
        if not donors:
            raise ValueError("donors must be a non-empty list of donor groups")
        if sites is not None and len(sites) != len(donors):
            raise ValueError(
                f"sites length ({len(sites)}) must equal donors length "
                f"({len(donors)}) when explicitly provided"
            )
        self.smiles = smiles
        self.donors = donors  # list[list[int]]
        self.sites = sites  # list[int] | None

    def __repr__(self) -> str:
        site_str = f", sites={self.sites}" if self.sites is not None else ""
        return f"LigandInput({self.smiles!r}, donors={self.donors}{site_str})"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ConformerResult:
    """One conformer from the ensemble."""

    complex: object  # TMComplex
    score: float
    converged: bool
    rank: int  # 1-based rank (1 = best)
    trial_idx: Optional[int] = None
    # 2026-09-17: which candidate donor-to-site assignment (index into
    # EnsembleResult.assignment_search["candidate_assignments"] when set)
    # this conformer came from. Meaningful once _solve_for_assignments
    # batches multiple candidates into one solve -- see that method and
    # the "batched candidate screening" changelog note on
    # ComplexBuilder.solve(). None for a single-assignment solve (the
    # common case: use_assignment_search=use_symmetry_enumeration=False,
    # or either one collapsing to exactly 1 surviving candidate).
    assignment_idx: Optional[int] = None

    def write_xyz(self, path: str | Path) -> None:
        """Write this conformer to an .xyz file."""
        mol = self.complex
        path = Path(path)
        lines = [str(len(mol.atoms)), f"{mol.csd_code}  score={self.score:.4f}"]
        for a in mol.atoms:
            x, y, z = a.position
            lines.append(f"{a.symbol:<4s} {x:12.6f} {y:12.6f} {z:12.6f}")
        path.write_text("\n".join(lines) + "\n")
        log.info("Wrote %s", path)

    def __repr__(self) -> str:
        conv = "converged" if self.converged else "not converged"
        return f"ConformerResult(rank={self.rank}, score={self.score:.4f}, {conv})"


@dataclass
class EnsembleResult:
    """
    Ranked ensemble of conformers returned by ComplexBuilder.solve().
    """

    conformers: List[ConformerResult]
    solve_trace: Optional[Dict[str, np.ndarray]] = None
    scorer: Optional[object] = None
    # 2.13.0: set only when solve(use_assignment_search=True) actually ran
    # more than one candidate donor-to-site assignment -- see that
    # changelog entry. dict with n_candidates, winning_candidate_idx,
    # winning_assignment, candidate_scores. None otherwise (incl. always
    # when use_assignment_search=False, the default).
    assignment_search: Optional[Dict[str, object]] = None

    @property
    def best(self) -> Optional[ConformerResult]:
        return self.conformers[0] if self.conformers else None

    @property
    def converged(self) -> List[ConformerResult]:
        return [c for c in self.conformers if c.converged]

    def summary(self) -> None:
        """Print a ranked table of conformers."""
        print(f"{'Rank':>4}  {'Score':>10}  {'Converged':>10}")
        print("-" * 32)
        for c in self.conformers:
            print(f"{c.rank:>4}  {c.score:>10.4f}  {'yes' if c.converged else 'no':>10}")
        n_conv = len(self.converged)
        print(f"\n{n_conv}/{len(self.conformers)} converged")


# ---------------------------------------------------------------------------
# CN table for validation
# ---------------------------------------------------------------------------

_GEOM_CN: Dict[str, int] = {
    "linear": 2,
    "trigonal": 3,
    "t_shaped": 3,
    "square_plan": 4,
    "tetrahedral": 4,
    "see_saw": 4,
    "square_pyr": 5,
    "trig_bipyr": 5,
    "octahedral": 6,
    "trig_pris": 6,
    "pent_bipyr": 7,
    "square_anti": 8,
    "tricap_pris": 9,
    "other": None,  # no fixed CN — skip validation
}


# ---------------------------------------------------------------------------
# Default Q1 bond length table (fallback when stats lookup fails)
# ---------------------------------------------------------------------------

# Mean M-D distances in Å, keyed by (metal_period, donor_family).
# Period 1 = Li-Ne, Period 4 = first-row TM (Fe, Co, Ni ...), etc.
_DEFAULT_BOND_LENGTH: float = 2.1

_BOND_LENGTH_TABLE: Dict[str, float] = {
    # donor_family defaults
    "N_donor": 2.05,
    "O_donor": 2.05,
    "P_donor": 2.25,
    "S_donor": 2.40,
    "C_donor": 2.00,
    "Hal_donor": 2.35,
    "H_donor": 1.65,
}


def _q1_sample_distances(
    stats,
    q1_key: tuple,
    donor_family: str,
    n_samples: int,
    rng: "np.random.Generator",
) -> "np.ndarray":
    """
    Draw ``n_samples`` M–D bond lengths from the Q1 GMM (spec §5.3).


    Parameters
    ----------
    stats        : StatsLookup
    q1_key       : L0 Q1 key tuple
    donor_family : donor family label (for table fallback)
    n_samples    : M — number of bond-length samples to draw
    rng          : numpy Generator for reproducible sampling

    Returns
    -------
    (n_samples,) float64 array of bond lengths in Å.

    """
    means = vars_ = weights = None
    try:
        dist = stats.dist("Q1", q1_key)
        if dist.means and dist.weights and dist.vars:
            means = np.asarray(dist.means, dtype=float)
            vars_ = np.asarray(dist.vars, dtype=float)
            weights = np.asarray(dist.weights, dtype=float)
    except Exception:
        pass

    if means is None:
        # Degenerate / missing Q1 — tight Gaussian around the table value.
        mu = _BOND_LENGTH_TABLE.get(donor_family, _DEFAULT_BOND_LENGTH)
        return rng.normal(mu, 0.05, size=n_samples)

    # Normalise weights defensively, then apply prominence floor to drop
    # spurious low-weight artefact modes (e.g. 0.045 Å amine "bond").
    # Modes below _Q1_PROMINENCE_FLOOR × max_weight are removed before
    # sampling; the remaining weights are re-normalised.
    w = weights / max(weights.sum(), 1e-12)
    max_w = w.max()
    keep = w >= _Q1_PROMINENCE_FLOOR * max_w
    if keep.sum() == 0:
        keep = np.ones(len(w), dtype=bool)  # degenerate: keep all
    means = means[keep]
    vars_ = vars_[keep]
    w = w[keep]
    w = w / w.sum()

    comp = rng.choice(len(means), size=n_samples, p=w)
    stds = np.sqrt(np.clip(vars_[comp], 1e-6, None))
    samples = rng.normal(means[comp], stds)
    # Guard against nonphysical (negative / near-zero) lengths from wide tails.
    return np.clip(samples, 0.5, None)


def _shallow_copy_mol_for_conformer(mol: "TMComplex") -> "TMComplex":
    """
    Make an independent-enough copy of `mol` for one output conformer.

    BUGFIX (2.4.0): the call site used `copy.deepcopy(mol)` once per output
    conformer (n_conformers = max(beam_width, n_coord_samples), so e.g. 32-64
    calls per solve). A TMComplex's topology metadata -- rigid_bodies,
    pivot_chains, decomposition, rotatable_bonds, donor_labels, bonds -- is
    IDENTICAL across every conformer of a single solve (it never changes
    after decompose_ligand()); only atom positions (and the coord-site /
    haptic centroids derived from them) differ per conformer. deepcopy
    recurses into all of it regardless, measured at 61,408 recursive copy
    calls across 32 top-level deepcopy() calls in one profiled EDTA solve.
    This does an explicit shallow copy instead: atoms are shallow-copied
    (their `.position` is about to be REASSIGNED to a fresh array, not
    mutated in place, so a shallow per-atom copy is sufficient); coord_sites
    and each ligand's haptic_groups are ALSO shallow-copied per-object,
    because _refresh_coord_geometry() mutates `.position` / `.centroid`
    in-place on those specific objects -- sharing them across conformers
    would have every conformer silently overwrite the same shared object.
    Everything else (rigid_bodies, pivot_chains, decomposition,
    rotatable_bonds, donor_labels, bonds, metal_idx, ...) is never mutated
    after construction and is safe to share by reference.
    """
    mol_copy = copy.copy(mol)
    mol_copy.atoms = [copy.copy(a) for a in mol.atoms]
    mol_copy.coord_sites = [copy.copy(s) for s in getattr(mol, "coord_sites", [])]

    new_ligands = []
    for lig in mol.ligands:
        lig_copy = copy.copy(lig)
        lig_copy.haptic_groups = [copy.copy(h) for h in getattr(lig, "haptic_groups", [])]
        new_ligands.append(lig_copy)
    mol_copy.ligands = new_ligands

    return mol_copy


def _refresh_coord_geometry(mol) -> None:
    """
    Recompute derived coordination geometry (coord_site positions and haptic
    centroids) from the current ``mol.atoms`` positions, in place.
    """
    pos_by_idx = {a.idx: a.position for a in mol.atoms}

    for site in getattr(mol, "coord_sites", []):
        idxs = [i for i in site.donor_indices if i in pos_by_idx]
        if not idxs:
            continue
        site.position = np.mean([pos_by_idx[i] for i in idxs], axis=0).copy()

    # Haptic centroids live on the ligands' haptic_groups (and, if present, on
    # mol.haptic_interactions).  Refresh both views from the same atom coords.
    for lig in getattr(mol, "ligands", []):
        for h in getattr(lig, "haptic_groups", []):
            idxs = [i for i in h.donor_indices if i in pos_by_idx]
            if idxs:
                h.centroid = np.mean([pos_by_idx[i] for i in idxs], axis=0).copy()

    for h in getattr(mol, "haptic_interactions", []):
        idxs = [i for i in h.donor_indices if i in pos_by_idx]
        if idxs:
            h.centroid = np.mean([pos_by_idx[i] for i in idxs], axis=0).copy()


# ---------------------------------------------------------------------------
# SMILES → 3-D conformer helper
# ---------------------------------------------------------------------------


def _smiles_to_conformer(
    smiles: str,
    fast_embed: bool = False,
) -> Tuple[object, np.ndarray, bool]:
    """
    Parse SMILES and generate a 3-D conformer via RDKit ETKDG.

    Parameters
    ----------
    smiles     : SMILES string
    fast_embed : if True, skip MMFF force-field refinement.  The conformer
                 is only used as a starting geometry for the IK solve, which
                 overwrites it completely, so MMFF quality is irrelevant in
                 the benchmark loop.  Saves ~0.2–0.5 s per ligand.

    Returns
    -------
    rdmol      : RDKit Mol with embedded 3-D coordinates
    positions  : (N, 3) float64 array of atom positions (all atoms incl. H)
    has_h      : True if SMILES contained explicit H atoms
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    has_h = "[H]" in smiles or "[H" in smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    # Add Hs for 3-D embedding (we'll remove them again for the index map
    # unless the original SMILES was H-explicit)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol_h, params)
    if result == -1:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())

    if not fast_embed:
        AllChem.MMFFOptimizeMolecule(mol_h)

    conf = mol_h.GetConformer()
    positions = conf.GetPositions().astype(np.float64)  # (N_all, 3) incl. H

    return mol_h, positions, has_h


def _donor_indices_to_full_graph(
    smiles: str,
    donor_groups: List[List[int]],
    has_h: bool,
    rdmol_h,  # RDKit Mol with Hs added
) -> List[List[int]]:
    """
    Map user-supplied donor indices (on the H-implicit graph if has_h=False,
    or on the full graph if has_h=True) to indices on the full H-included
    RDKit mol (rdmol_h).
    """
    if has_h:
        return donor_groups  # indices already on full graph

    # Build a mapping: original heavy-atom index → rdmol_h atom index
    # RDKit AddHs preserves original atom order; new H atoms are appended.

    heavy_to_full: Dict[int, int] = {}
    heavy_counter = 0
    for atom in rdmol_h.GetAtoms():
        if atom.GetAtomicNum() != 1:  # not H
            heavy_to_full[heavy_counter] = atom.GetIdx()
            heavy_counter += 1

    mapped = []
    for group in donor_groups:
        mapped_group = []
        for idx in group:
            if idx not in heavy_to_full:
                raise ValueError(
                    f"Donor index {idx} is out of range for the heavy-atom "
                    f"graph (heavy atoms: {heavy_counter})"
                )
            mapped_group.append(heavy_to_full[idx])
        mapped.append(mapped_group)
    return mapped


# ---------------------------------------------------------------------------
# TMComplex shell builder
# ---------------------------------------------------------------------------

_METAL_IDX_BASE = 1  # 1-based atom indices throughout


def _build_complex_shell(
    metal_symbol: str,
    charge: int,
    spin: int,
    geometry: str,
    ligand_inputs: List[LigandInput],
    fast_embed: bool = False,
) -> Tuple[object, List[object], List[np.ndarray], List[List[List[int]]]]:
    """
    Build a TMComplex shell from scratch:
      - Metal atom at origin (idx=1)
      - Ligand atoms from ETKDG conformers, appended after metal
      - Bonds within each ligand from RDKit; metal-donor bonds added later
        by classify() after donor positions are known.

    Returns
    -------
    mol            : TMComplex (no coord_sites yet; classify() fills them)
    rdmols         : list of RDKit Mol objects (one per ligand)
    lig_positions  : list of (N_lig, 3) arrays (one per ligand, in full-H coords)
    donor_idx_full : list of donor groups in full-H indices (one list per ligand)
    """

    from jacobik.core.types import Atom, Bond, HapticInteraction, Ligand, TMComplex

    atoms: List[Atom] = []
    bonds: List[Bond] = []
    ligands: List[Ligand] = []

    # --- Metal atom at origin ---
    metal_atom = Atom(
        idx=1,
        symbol=metal_symbol,
        position=np.zeros(3, dtype=float),
    )
    atoms.append(metal_atom)
    atom_cursor = 2  # next 1-based index

    rdmols = []
    lig_positions_out = []
    donor_idx_full_out = []

    for lig_i, lig_inp in enumerate(ligand_inputs):
        rdmol_h, positions, has_h = _smiles_to_conformer(lig_inp.smiles, fast_embed=fast_embed)

        # Map donor indices to full-H graph
        donor_full = _donor_indices_to_full_graph(lig_inp.smiles, lig_inp.donors, has_h, rdmol_h)

        # Place ligand so its donors are near the metal at reasonable
        # coordination distances.  The same centroid-to-2.5Å translation
        # is used for all denticity; for polydentate ligands an additional
        # push-outward step ensures no donor lands closer than min_clash_dist
        # to the metal, which would produce a degenerate direction vector for
        # the Hungarian assignment and a Kabsch init the solver cannot recover
        # from.  Bond lengths are never changed — only a rigid translation.
        all_donor_flat = [idx for grp in donor_full for idx in grp]
        n_donor_groups = len(donor_full)
        metal_pos = np.zeros(3)
        min_clash_dist = 1.2  # Å — below this the M→D direction is degenerate

        anchor = positions[all_donor_flat].mean(axis=0)
        angle = lig_i * 2 * math.pi / max(len(ligand_inputs), 1)
        target_dir = np.array([math.cos(angle), math.sin(angle), 0.3 * lig_i])
        target_dir /= np.linalg.norm(target_dir)
        offset = target_dir * 2.5 - anchor
        positions = positions + offset

        if n_donor_groups > 1:
            # Iteratively push the ligand outward until all donors clear
            # min_clash_dist.  Each iteration pushes along the direction from
            # the metal to the closest donor — not along target_dir — to
            # guarantee convergence regardless of ligand shape.  Bond lengths
            # are never changed; only a rigid translation per iteration.
            for _ in range(20):  # safety cap; converges in 1-3 iterations
                donor_vecs = positions[all_donor_flat] - metal_pos  # (n_d, 3)
                donor_dists = np.linalg.norm(donor_vecs, axis=1)
                min_idx = int(np.argmin(donor_dists))
                min_dist = donor_dists[min_idx]
                if min_dist >= min_clash_dist:
                    break
                # Push along the direction from metal to the offending donor
                push_dir = donor_vecs[min_idx]
                push_norm = np.linalg.norm(push_dir)
                if push_norm < 1e-8:
                    push_dir = target_dir  # degenerate: fall back to target
                else:
                    push_dir = push_dir / push_norm
                push = min_clash_dist - min_dist + 0.05
                positions = positions + push_dir * push

        # Build Atom list for this ligand
        lig_atom_indices = []
        for j, atom in enumerate(rdmol_h.GetAtoms()):
            idx = atom_cursor
            lig_atom_indices.append(idx)
            atoms.append(
                Atom(
                    idx=idx,
                    symbol=atom.GetSymbol(),
                    position=positions[j].copy(),
                )
            )
            atom_cursor += 1

        # Build Bond list (ligand-internal only; skip metal bonds for now)
        for bond in rdmol_h.GetBonds():
            i_local = bond.GetBeginAtomIdx()
            j_local = bond.GetEndAtomIdx()
            i_global = lig_atom_indices[i_local]
            j_global = lig_atom_indices[j_local]
            # Map RDKit bond type to approximate order
            bt = bond.GetBondTypeAsDouble()
            bonds.append(Bond(i=i_global, j=j_global, order=float(bt)))

        # Translate donor_full indices from rdmol-local (0-based) to global
        # 1-based atom indices
        donor_full_global = [[lig_atom_indices[idx] for idx in grp] for grp in donor_full]

        # Identify haptic groups (any donor group with >1 atom)
        haptic_groups = []
        kappa_donors = []
        for grp in donor_full_global:
            if len(grp) > 1:
                centroid = np.mean([atoms[i - 1].position for i in grp], axis=0)
                haptic_groups.append(
                    HapticInteraction(
                        metal_idx=1,
                        donor_indices=grp,
                        haptic_order=len(grp),
                        centroid=centroid,
                        donor_symbols=[atoms[i - 1].symbol for i in grp],
                        avg_bond_order=0.5,  # placeholder; classify() refines
                    )
                )
            else:
                kappa_donors.extend(grp)

        # Add metal-donor bonds (dative, low BO)
        for grp in donor_full_global:
            for donor_idx in grp:
                bonds.append(Bond(i=1, j=donor_idx, order=0.5))

        lig_obj = Ligand(
            ligand_id=lig_i,
            atom_indices=lig_atom_indices,
            donor_atoms=kappa_donors,
            haptic_groups=haptic_groups,
            donors_are_authoritative=True,
        )
        ligands.append(lig_obj)
        rdmols.append(rdmol_h)
        lig_positions_out.append(positions)
        donor_idx_full_out.append(donor_full_global)

    mol = TMComplex(
        csd_code=f"{metal_symbol}_generated",
        charge=charge,
        spin_state=spin,
        stoichiometry="",
        mnd=3,
        csd_years="",
        atoms=atoms,
        bonds=bonds,
        metal_idx=1,
        metal_symbol=metal_symbol,
        ligands=ligands,
        coord_geom=geometry,
    )

    return mol, rdmols, lig_positions_out, donor_idx_full_out


# ---------------------------------------------------------------------------
# _SolveContext
# ---------------------------------------------------------------------------


@dataclass
class _SolveContext:
    """
    Bundles everything about one ComplexBuilder.solve() call that is
    independent of *which* donor-to-site assignment candidate is being
    tried, so the assignment-candidate-generation and per-candidate solve
    helper methods below can take one `ctx` argument instead of a dozen
    loose parameters (refactor, 2026-09-17 -- solve() used to be a single
    ~1350-line method; see this module's docstring / that changelog entry
    for the rationale).

    Built once per solve() call, near the top of solve() itself.
    """

    mol: object  # TMComplex
    ideal: np.ndarray  # (n_sites, 3)
    all_donor_groups: List[Tuple[int, int, List[int]]]
    n_donors: int
    chelate_pairs_global: List[Tuple[int, int]]
    heavy_graph: object  # nx.Graph, or None if construction failed
    #: Donor-index equivalence classes from (SMILES, RDKit
    #: CanonicalRankAtoms) -- single-atom donor groups only, size >= 2.
    #: Computed once here and reused everywhere this concept is needed
    #: (previously recomputed via RDKit up to 3x per solve() call, once
    #: each in 3a's Fix A dedup, 3a', and _solve_for_assignments's own
    #: symmetric_groups -- see _symmetric_donor_groups()).
    symmetric_donor_groups: List[List[int]]
    stats: object  # StatsLookup
    device: torch.device
    np_rng: np.random.Generator
    lr: float
    # solve()-level knobs threaded straight through to the per-candidate
    # solve helpers, unchanged for every candidate in a given solve() call.
    quantity_weights: Optional[Dict[str, float]]
    q2_aggregation: str
    q2_power_p: float
    n_coord_samples: int
    enable_bend_dof: bool
    q3_mode_commit: bool
    ik_config: Optional[object]  # IKSolverConfig
    n_conformers: Optional[int]
    debug: bool


# ---------------------------------------------------------------------------
# ComplexBuilder
# ---------------------------------------------------------------------------


class ComplexBuilder:
    """
    Build a transition-metal complex from its components and solve for a
    conformer ensemble.

    Parameters
    ----------
    metal    : element symbol, e.g. "Fe", "Pt", "Ru"
    geometry : coordination geometry label, e.g. "octahedral"
               Use ``geometry_sites(geometry)`` to inspect site vectors.
    charge   : formal charge of the complex
    spin     : spin multiplicity (2S+1)
    ligands  : list of LigandInput objects.  Each ligand is explicitly
               defined — repeat identical LigandInput objects for symmetric
               complexes.  The engine validates that the total donor count
               (summed across all ligands) equals the CN of the geometry.

    Raises
    ------
    ValueError
        If the total donor count does not match the CN of the geometry.
    ValueError
        If any ligand specifies explicit ``sites=`` that conflict (duplicate
        site indices, or site index out of range for the geometry).

    Example
    -------
    >>> cp    = LigandInput("C1=CC=CC=1", donors=[[0,1,2,3,4]])
    >>> phsph = LigandInput("P(c1ccccc1)(c1ccccc1)c1ccccc1", donors=[[0]])
    >>> builder = ComplexBuilder(
    ...     metal="Fe", geometry="octahedral",
    ...     charge=0, spin=1,
    ...     ligands=[cp, cp, phsph],
    ... )
    >>> result = builder.solve(constraints_db="constraints.json")
    >>> result.summary()
    """

    def __init__(
        self,
        metal: str,
        geometry: str,
        charge: int,
        spin: int,
        ligands: List[LigandInput],
    ) -> None:
        from jacobik.ik.targets import IDEAL_GEOMETRIES

        if geometry not in IDEAL_GEOMETRIES and geometry != "other":
            raise ValueError(f"Unknown geometry {geometry!r}. Valid: {sorted(IDEAL_GEOMETRIES)}")

        # Count total donor groups
        total_donors = sum(len(lig.donors) for lig in ligands)
        expected_cn = _GEOM_CN.get(geometry)
        if expected_cn is not None and total_donors != expected_cn:
            raise ValueError(
                f"Geometry '{geometry}' requires CN={expected_cn}, but "
                f"total donor groups across all ligands = {total_donors}. "
                f"Each donor group (inner list in donors=) counts as one "
                f"coordination site."
            )

        # Validate explicit site assignments
        if expected_cn is not None:
            used_sites: set = set()
            for lig in ligands:
                if lig.sites is None:
                    continue
                for s in lig.sites:
                    if s < 0 or s >= expected_cn:
                        raise ValueError(
                            f"Site index {s} is out of range for "
                            f"geometry '{geometry}' (CN={expected_cn}, "
                            f"valid indices 0..{expected_cn - 1})"
                        )
                    if s in used_sites:
                        raise ValueError(
                            f"Site index {s} is assigned to more than one "
                            f"donor group across the ligand list"
                        )
                    used_sites.add(s)

            # If some ligands use explicit sites and others don't, that's
            # fine — partial explicit assignment is supported.

        self.metal = metal
        self.geometry = geometry
        self.charge = charge
        self.spin = spin
        self.ligands = ligands

        log.debug(
            "ComplexBuilder: %s %s CN=%s  %d ligands  %d total donors",
            metal,
            geometry,
            expected_cn,
            len(ligands),
            total_donors,
        )

    def solve(
        self,
        constraints_db: str | Path | None = None,
        beam_width: int = 16,
        n_ik_steps: int = 200,
        lr: float = 0.1,
        ik_config: Optional[IKSolverConfig] = None,
        q3_mode_commit: bool = True,
        enable_bend_dof: bool = True,
        n_coord_samples: int = 8,
        quantity_weights: Optional[Dict[str, float]] = None,
        q2_aggregation: str = "mean",
        q2_power_p: float = 3.0,
        device: str = "cpu",
        seed: Optional[int] = None,
        debug: bool = False,
        stats: Optional[object] = None,
        fast_embed: bool = False,
        n_conformers: Optional[int] = None,
        use_assignment_search: bool = False,
        assignment_search: Optional[AssignmentSearchConfig] = None,
        use_symmetry_enumeration: bool = False,
        site_enumeration_config: Optional[SiteEnumerationConfig] = None,
    ) -> EnsembleResult:
        """
        Solve for a conformer ensemble.

        Parameters
        ----------
        constraints_db   : path to constraints.json produced by fitter.py
        beam_width       : W — number of torsion seeds per ligand from beam search
        n_ik_steps       : T — gradient descent iterations per solve
        lr               : learning rate for Adam on the combined gradient
        ik_config        : IKSolverConfig (ik/ik_solver.py) with the shape/
                           symmetry-constraint hyperparameters (lambda_init,
                           shape_rho/tol, sym_weight/angle_weight, the
                           mu_shape_*/sym_*/mu_sym_max stall-safeguard
                           settings, lse_temp, adam_betas). None uses
                           IKSolverConfig's own defaults.
        q3_mode_commit   : bool, default True. When a symmetric donor
                           group's Q3 target is genuinely multi-modal,
                           commit the whole group to a single GMM component
                           (chosen from the initial pose) for the whole
                           solve, instead of letting sym_angle_weight tie
                           members toward a mean angle that can fall
                           between two modes. False restores the pre-mode-
                           commit behaviour -- useful for A/B comparison.
        enable_bend_dof  : bool, default True. Gives each branch-point
                           kappa donor (e.g. a tertiary amine N)
                           independent local control over each of its
                           outgoing branches via "bend" joints (see
                           ik/fk.py's ForwardKinematics), diversified per
                           beam trial the same way the base orientation is.
                           False restores the pre-bend-DOF joint count and
                           behaviour byte-for-byte.
        n_coord_samples  : M — coordination sphere samples (diversity of targets)
        quantity_weights : optional dict mapping "Q1".."Q6" → float weight.
                           Controls which quantities contribute and how much.
                           Default: all 1.0 (equal footing).  Common use:
                             {"Q2": 0.0}  — disable D–M–D angle scoring
                             {"Q2": 2.0}  — double-weight D–M–D angles
                           Weights are baked into the Scorer at construction
                           time (no per-step overhead).
        q2_aggregation   : "mean" (default) or "power_mean" (EXPERIMENTAL --
                           see Scorer.__init__ docstring). Addresses the
                           documented Q2 mean-dilution issue where a single
                           severely-wrong D-M-D angle is diluted across all
                           pairs; untested against a full benchmark suite as
                           of introduction.
        q2_power_p       : exponent for q2_aggregation="power_mean" (default 3.0).
        device           : "cpu" or "cuda"
        seed             : optional random seed for reproducibility
        debug            : if True, print detailed diagnostics at every stage.
        stats            : optional pre-built StatsLookup object.  When provided,
                           load_db() is skipped entirely — useful in benchmark
                           loops to avoid re-parsing constraints.json per complex.
                           Build once with:
                             from jacobik.stats.stats import StatsLookup, load_db
                             stats = StatsLookup(load_db(constraints_db))
        fast_embed       : if True, skip MMFF force-field refinement of the
                           ETKDG conformer.  Safe in benchmark contexts because
                           the IK solve overwrites the initial geometry completely.
                           Saves ~0.2–0.5 s per ligand.
        n_conformers     : maximum number of conformers to assemble from IK
                           results.  Default: max(beam_width, n_coord_samples).
                           Set to 1 to get only the rank-1 conformer and skip
                           all deep-copy overhead for lower-ranked structures.
        use_assignment_search : bool, default False. Try alternative
                           donor-to-site assignments (ik/assignment_search.py)
                           around the baseline Hungarian solve when it's
                           genuinely ambiguous. 2026-09-17: every surviving
                           candidate is now solved TOGETHER with the
                           baseline, in one batched call spanning
                           n_candidates x beam_width x n_coord_samples
                           trials (see _solve_for_assignments) -- not a
                           sequential re-solve-per-candidate loop -- and the
                           returned ensemble can contain conformers from
                           more than one candidate assignment, ranked
                           together by score. Unambiguous cases are
                           unaffected and pay no extra cost.
        assignment_search : AssignmentSearchConfig (k_max, margin_deg,
                           max_ambiguous_group) controlling the search when
                           use_assignment_search is True. None uses
                           AssignmentSearchConfig's own defaults.
        use_symmetry_enumeration : bool, default False. Proactive
                           alternative to use_assignment_search
                           (ik/site_enumeration.py): instead of perturbing
                           the one baseline Hungarian assignment, exhaustively
                           enumerates every symmetry-distinct donor-to-site
                           bijection consistent with ligand-graph
                           reachability, up front. Unlike
                           use_assignment_search, this can discover a
                           second valid coordination topology even when the
                           baseline embedding didn't look locally ambiguous
                           -- see ik/site_enumeration.py's module docstring
                           for why that's the actual point of the proactive
                           approach, and for two concrete, unresolved
                           limitations to be aware of before relying on
                           this in production: (1) donor-pair reachability
                           reuses assignment_search.py's REACHABILITY_
                           BREAKPOINTS completely unchanged, which can make
                           the physically correct topology UNREACHABLE for
                           some real ligands (a documented, regression-
                           tested EDTA case) rather than just under-
                           explored -- if that happens, enumeration returns
                           zero candidates and this falls back to the
                           baseline assignment automatically (a warning is
                           logged, nothing raises); (2) the combinatorial
                           blowup risk for CN8/CN9 ligands is real (see
                           SiteEnumerationConfig's own docstring for
                           real-CN8/CN9-case raw-search measurements), so
                           this is OFF BY DEFAULT and should be benchmarked
                           against your actual ligand set, not enabled
                           blindly. 2026-09-17: every surviving candidate
                           bijection is now solved TOGETHER in one batched
                           call (n_candidates x beam_width x n_coord_samples
                           trials, see _solve_for_assignments) rather than
                           a sequential per-candidate re-solve, so the
                           per-candidate wall-clock cost that used to scale
                           with n_candidates mostly doesn't anymore (GPU
                           memory for the larger batch is the real
                           constraint now); the returned ensemble is no
                           longer collapsed to one "winning" candidate --
                           it can and typically will contain conformers
                           from more than one distinct donor-to-site
                           assignment, ranked together by score (each
                           conformer's ConformerResult.assignment_idx says
                           which). Requires n_donors == the geometry's
                           n_sites (a fully-saturated complex); silently
                           falls back to the baseline assignment otherwise
                           (a partially-explicit or under-coordinated
                           complex has no well-defined "bijection" for this
                           enumerator to search over). Mutually exclusive
                           with use_assignment_search -- if both are True,
                           use_symmetry_enumeration takes precedence and a
                           warning is logged; running both together
                           (enumerate proactively, then still perturb
                           locally on top) is a documented possible
                           follow-on in ik/site_enumeration.py's module
                           docstring, not implemented here.
        site_enumeration_config : SiteEnumerationConfig (max_candidates,
                           time_budget_s) controlling both the raw
                           enumeration AND the dedup pass when
                           use_symmetry_enumeration is True (see
                           ik/site_enumeration.py's `dedup_combined`
                           docstring for why the dedup pass needs its own
                           budget too, not just the raw search). None uses
                           SiteEnumerationConfig's own defaults
                           (max_candidates=5000, time_budget_s=10.0 each,
                           i.e. up to ~20s combined worst case).

        Returns
        -------
        EnsembleResult with conformers sorted best→worst by score.
        """
        _device = torch.device(device)
        np_rng = np.random.default_rng(seed)  # for Q1 GMM target sampling (§5.3)

        # BUGFIX (2.5.0): seed was previously used to seed np.random's global
        # state (below) plus a LOCAL torch.Generator() that was created,
        # seeded, and then never passed to anything -- dead code. Every
        # unqualified torch.randn/torch.rand call anywhere in the pipeline
        # (beam search, kabsch_init's diversify_orientation) draws from
        # PyTorch's GLOBAL RNG, which was never actually seeded by solve()
        # at all. Reproducibility for a given seed therefore depended on
        # whatever torch random state happened to exist from code that ran
        # BEFORE solve() was called -- different in every notebook, script,
        # or call order, matching reported "same seed, different result"
        # confusion. torch.manual_seed(seed) seeds the actual global RNG
        # that those calls read from.
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ------------------------------------------------------------------
        # 1-3. Build the complex, classify donors, and compute the baseline
        #      (Hungarian or explicit) donor-to-site assignment. See each
        #      helper's own docstring for its numbered step(s); this used
        #      to all be inline here before the 2026-09-17 refactor split
        #      solve() (previously ~1350 lines on its own) into the
        #      methods below it in this class.
        # ------------------------------------------------------------------
        stats = self._load_stats(constraints_db, stats)
        mol, rdmols, lig_positions, donor_idx_full = self._build_shell(fast_embed, debug)
        ideal, all_donor_groups, n_donors = self._collect_donor_groups(mol, donor_idx_full)
        chelate_pairs_global, heavy_graph = self._detect_chelate_pairs(
            mol, all_donor_groups, n_donors
        )
        assignment, donor_dirs, subclasses = self._compute_baseline_assignment(
            mol, ideal, all_donor_groups, n_donors, chelate_pairs_global
        )
        log.info("Site assignment: %s", assignment.tolist())

        symmetric_donor_groups = self._symmetric_donor_groups(mol, rdmols, all_donor_groups)

        ctx = _SolveContext(
            mol=mol,
            ideal=ideal,
            all_donor_groups=all_donor_groups,
            n_donors=n_donors,
            chelate_pairs_global=chelate_pairs_global,
            heavy_graph=heavy_graph,
            symmetric_donor_groups=symmetric_donor_groups,
            stats=stats,
            device=_device,
            np_rng=np_rng,
            lr=lr,
            quantity_weights=quantity_weights,
            q2_aggregation=q2_aggregation,
            q2_power_p=q2_power_p,
            n_coord_samples=n_coord_samples,
            enable_bend_dof=enable_bend_dof,
            q3_mode_commit=q3_mode_commit,
            ik_config=ik_config,
            n_conformers=n_conformers,
            debug=debug,
        )

        # ------------------------------------------------------------------
        # 3a/3a'. OPTIONAL alternative donor-to-site assignment candidates
        #     around the baseline assignment above -- see
        #     use_assignment_search / use_symmetry_enumeration's own
        #     parameter docstrings above for the full rationale, and
        #     _run_assignment_search / _run_symmetry_enumeration for the
        #     implementations. Mutually exclusive; symmetry enumeration
        #     takes precedence if both are requested (a warning is
        #     logged). Fully explicit (every site caller-pinned)
        #     assignments skip both -- nothing to search when nothing is
        #     free.
        # ------------------------------------------------------------------
        _assignment_candidates: List[np.ndarray] = [assignment]
        _is_fully_explicit_assignment = all(lig_inp.sites is not None for lig_inp in self.ligands)

        if use_assignment_search and use_symmetry_enumeration:
            log.warning(
                "Both use_assignment_search and use_symmetry_enumeration were "
                "requested; running only use_symmetry_enumeration (proactive "
                "enumeration takes precedence -- see ik/site_enumeration.py's "
                "module docstring for why composing them is not yet implemented)."
            )

        if (
            use_assignment_search
            and not use_symmetry_enumeration
            and not _is_fully_explicit_assignment
        ):
            _assignment_candidates = self._run_assignment_search(
                ctx, assignment, donor_dirs, subclasses, assignment_search, debug
            )
        elif use_symmetry_enumeration and not _is_fully_explicit_assignment:
            _assignment_candidates = self._run_symmetry_enumeration(
                ctx, assignment, site_enumeration_config, debug
            )

        return self._solve_for_assignments(ctx, _assignment_candidates, beam_width, n_ik_steps)

    def _load_stats(self, constraints_db, stats):
        """Step 1: load the constraints DB (or use the caller's pre-built
        StatsLookup, skipping load_db() entirely)."""
        from jacobik.stats.stats import StatsLookup, load_db

        if stats is None:
            if constraints_db is None:
                from importlib import resources

                constraints_db = resources.files("jacobik") / "constraints.json"
            log.info("Loading constraints DB: %s", constraints_db)
            db = load_db(constraints_db)
            stats = StatsLookup(db)
        else:
            log.info("Using pre-built StatsLookup (skipping load_db)")
        return stats

    def _build_shell(self, fast_embed, debug):
        """Step 2: build the initial TMComplex shell from each ligand's
        SMILES, then run donor classification."""
        from jacobik.topology.donor_classifier import classify

        log.info("Building complex shell (%s %s)...", self.metal, self.geometry)
        mol, rdmols, lig_positions, donor_idx_full = _build_complex_shell(
            metal_symbol=self.metal,
            charge=self.charge,
            spin=self.spin,
            geometry=self.geometry,
            ligand_inputs=self.ligands,
            fast_embed=fast_embed,
        )

        classify(mol, smiles_path=True)
        log.info("Classification done: %d coord sites", len(mol.coord_sites))

        if debug:
            print("\n[DEBUG] ── Step 2: Complex shell built ──────────────────────")
            print(
                f"  metal={self.metal}  geometry={self.geometry}  "
                f"charge={self.charge}  spin={self.spin}"
            )
            print(f"  Total atoms : {len(mol.atoms)}")
            print(f"  Coord sites : {len(mol.coord_sites)}")
            for si, site in enumerate(mol.coord_sites):
                hap = "haptic" if site.is_haptic else "kappa"
                print(f"    site[{si}] {hap}  donor_indices={site.donor_indices}")

        if not mol.coord_sites:
            raise RuntimeError(
                "Donor classification produced no coordination sites. "
                "Check that donor indices point to actual donor atoms."
            )
        return mol, rdmols, lig_positions, donor_idx_full

    def _collect_donor_groups(self, mol, donor_idx_full):
        """Step 3 (part 1): resolve the geometry's ideal site directions
        and collect all donor groups, in order, across ligands."""
        from jacobik.ik.targets import IDEAL_GEOMETRIES

        ideal = IDEAL_GEOMETRIES.get(self.geometry)
        if ideal is None:
            # "other" geometry — use identity directions
            cn = len(mol.coord_sites)
            ideal = np.eye(cn, 3, dtype=float)

        # Collect all donor groups in order across ligands
        all_donor_groups: List[Tuple[int, int, List[int]]] = []
        # (ligand_idx, group_idx_within_ligand, global_atom_indices)
        for li, (lig_inp, lig_obj) in enumerate(zip(self.ligands, mol.ligands)):
            for gi, grp in enumerate(donor_idx_full[li]):
                all_donor_groups.append((li, gi, grp))

        n_donors = len(all_donor_groups)
        return ideal, all_donor_groups, n_donors

    def _detect_chelate_pairs(self, mol, all_donor_groups, n_donors):
        """
        Step 2b: chelate pairs (short ligand-graph-distance donor pairs,
        same ligand) — passed to hungarian_assign so it can repair
        assignments that put a rigid short chelate on two sites too far
        apart to be kinematically reachable. See BUGFIX note in
        ik/targets.py (_repair_chelate_assignment) for the full rationale:
        a linear per-donor Hungarian cost cannot express this pairwise
        constraint on its own.

        Also returns the heavy-atom ligand graph (or None if it could not
        be built), reused by _build_hop_matrix below.
        """
        import networkx as nx

        from jacobik.topology.graph import build_heavy_ligand_graph

        _CHELATE_SHORT_MAX = 4  # ligand-graph bond hops; matches ik/scorer.py
        chelate_pairs_global: List[Tuple[int, int]] = []
        heavy_graph = None
        try:
            heavy_graph = build_heavy_ligand_graph(mol.atoms, mol.bonds, mol.metal_idx)
            for _di in range(n_donors):
                _li, _gi, _grp = all_donor_groups[_di]
                _a0 = _grp[0]
                if not heavy_graph.has_node(_a0):
                    continue
                for _dj in range(_di + 1, n_donors):
                    _lj, _gj, _grpj = all_donor_groups[_dj]
                    if _lj != _li:
                        continue  # chelate pairs only within the same ligand
                    _b0 = _grpj[0]
                    if not heavy_graph.has_node(_b0):
                        continue
                    try:
                        _hops = nx.shortest_path_length(heavy_graph, _a0, _b0)
                    except nx.NetworkXNoPath:
                        continue
                    if _hops <= _CHELATE_SHORT_MAX:
                        chelate_pairs_global.append((_di, _dj))
        except Exception as exc:
            log.warning("Chelate-pair detection failed (%s); skipping repair pass.", exc)
            chelate_pairs_global = []
        return chelate_pairs_global, heavy_graph

    def _build_hop_matrix(self, n_donors, all_donor_groups, heavy_graph):
        """
        Ligand-graph hop-distance matrix between donor atoms (sentinel -1
        where no path exists), or None if the heavy-atom ligand graph
        itself could not be built -- callers then treat all donor pairs
        as reachability-unconstrained. Shared by _run_assignment_search
        and _run_symmetry_enumeration (2026-09-17 refactor: previously
        identical ~25-line construction duplicated verbatim in both, per
        each site's own "keep the two in sync" comment).
        """
        import networkx as nx

        if heavy_graph is None:
            log.warning(
                "Hop-matrix build skipped: heavy-atom ligand graph was not "
                "available; treating all donor pairs as reachability-"
                "unconstrained."
            )
            return None

        hop_matrix_full: Optional[np.ndarray] = np.full((n_donors, n_donors), -1, dtype=int)
        try:
            for _hi in range(n_donors):
                _ai = all_donor_groups[_hi][2][0]
                if not heavy_graph.has_node(_ai):
                    continue
                for _hj in range(_hi + 1, n_donors):
                    _aj = all_donor_groups[_hj][2][0]
                    if not heavy_graph.has_node(_aj):
                        continue
                    try:
                        _hops = nx.shortest_path_length(heavy_graph, _ai, _aj)
                    except nx.NetworkXNoPath:
                        continue
                    hop_matrix_full[_hi, _hj] = _hops
                    hop_matrix_full[_hj, _hi] = _hops
        except Exception as exc:
            log.warning(
                "Full hop-matrix build failed (%s); treating all donor "
                "pairs as reachability-unconstrained.",
                exc,
            )
            hop_matrix_full = None
        return hop_matrix_full

    def _compute_baseline_assignment(
        self, mol, ideal, all_donor_groups, n_donors, chelate_pairs_global
    ):
        """Step 3 (part 2): the baseline donor-to-site assignment --
        fully explicit (caller-pinned every site), partially explicit
        (Hungarian on the rest), or fully Hungarian."""
        from jacobik.ik.targets import hungarian_assign

        # donor_dirs/subclasses are cheap and assignment-scheme-independent
        # (2026-09-17 refactor: previously computed only inside the
        # Hungarian branch below; hoisted so both branches -- and 3a/3a'
        # downstream, which only ever run in the non-fully-explicit case
        # anyway -- can rely on them being present without an
        # Optional/None special case).
        atom_map = {a.idx: a for a in mol.atoms}
        metal_pos = mol.atoms[0].position

        donor_dirs = np.zeros((n_donors, 3), dtype=float)
        for di, (li, gi, grp) in enumerate(all_donor_groups):
            centroid = np.mean([atom_map[i].position for i in grp], axis=0)
            vec = centroid - metal_pos
            norm = np.linalg.norm(vec)
            donor_dirs[di] = vec / norm if norm > 1e-8 else ideal[di % len(ideal)]

        subclasses = []
        for li, gi, grp in all_donor_groups:
            lbl = mol.donor_labels.get(grp[0])
            subclasses.append(lbl.subclass if lbl else "element_any")

        if all(lig_inp.sites is not None for lig_inp in self.ligands):
            # Fully explicit
            assignment = np.empty(n_donors, dtype=int)
            idx = 0
            for li, lig_inp in enumerate(self.ligands):
                for gi, site in enumerate(lig_inp.sites):
                    assignment[idx] = site
                    idx += 1
        else:
            # Hungarian (or partial explicit — treat unassigned as Hungarian)
            explicit_map: Dict[int, int] = {}
            idx = 0
            for li, lig_inp in enumerate(self.ligands):
                for gi in range(len(lig_inp.donors)):
                    if lig_inp.sites is not None:
                        explicit_map[idx] = lig_inp.sites[gi]
                    idx += 1

            if explicit_map:
                # Partial explicit: fix explicit ones, Hungarian on rest
                free_donors = [i for i in range(n_donors) if i not in explicit_map]
                used_sites = set(explicit_map.values())
                free_sites = [s for s in range(len(ideal)) if s not in used_sites]

                assignment = np.empty(n_donors, dtype=int)
                for di, si in explicit_map.items():
                    assignment[di] = si

                if free_donors and free_sites:
                    free_dirs = donor_dirs[free_donors]
                    free_ideal = ideal[free_sites]
                    free_subs = [subclasses[i] for i in free_donors]
                    _free_local = {g: l for l, g in enumerate(free_donors)}
                    _free_chelate_pairs = [
                        (_free_local[a], _free_local[b])
                        for a, b in chelate_pairs_global
                        if a in _free_local and b in _free_local
                    ]
                    free_assign = hungarian_assign(
                        free_dirs,
                        free_ideal,
                        donor_subclasses=free_subs,
                        geometry_label=self.geometry,
                        chelate_pairs=_free_chelate_pairs,
                    )
                    for di, local_si in zip(free_donors, free_assign):
                        assignment[di] = free_sites[local_si]
            else:
                assignment = hungarian_assign(
                    donor_dirs,
                    ideal,
                    donor_subclasses=subclasses,
                    geometry_label=self.geometry,
                    chelate_pairs=chelate_pairs_global,
                )

        return assignment, donor_dirs, subclasses

    def _symmetric_donor_groups(self, mol, rdmols, all_donor_groups):
        """
        Topologically-equivalent donor groups (2.3.0): RDKit canonical
        ranking (breakTies=False) detects donor atoms related by a graph
        automorphism of their own ligand's molecular graph. Two donors
        are equivalent iff they share a SMILES string AND have the same
        canonical rank within that SMILES's graph -- true both for
        donors within one symmetric polydentate ligand (e.g. EDTA's two
        amine N) and for donors across separate instances of a repeated
        ligand (e.g. all 4 Cl- in PtCl4, or corresponding N atoms across
        Co(en)3's three en ligands), since CanonicalRankAtoms is a pure
        function of molecular-graph topology -- identical SMILES always
        gives identical ranks. Multi-atom (haptic) donor groups are
        skipped -- no single canonical rank to key on; excluding a donor
        from every group only means it's treated as its own singleton
        class elsewhere, always safe, just possibly under-deduplicated
        for that donor.

        Assignment-independent (depends only on ligand topology), so
        this is computed exactly ONCE per solve() call and reused by
        _run_assignment_search's Fix A dedup, _run_symmetry_enumeration's
        donor_equivalence_groups, and _solve_for_assignments's own
        symmetric_groups/q3_symmetric_groups restraint groups
        (2026-09-17 refactor: previously identical RDKit ranking code was
        duplicated in all three of those places -- up to 3x per solve()
        call -- per each site's own "keep these in sync" comment).
        """
        try:
            from rdkit import Chem

            key_to_donors: Dict[Tuple[str, int], List[int]] = {}
            rank_cache: Dict[int, list] = {}
            for di, (li, gi, grp) in enumerate(all_donor_groups):
                if len(grp) != 1:
                    continue
                if li not in rank_cache:
                    rank_cache[li] = list(Chem.CanonicalRankAtoms(rdmols[li], breakTies=False))
                offset = mol.ligands[li].atom_indices[0]
                local_idx = grp[0] - offset
                if local_idx < 0 or local_idx >= len(rank_cache[li]):
                    continue
                rank = rank_cache[li][local_idx]
                smiles = self.ligands[li].smiles
                key_to_donors.setdefault((smiles, rank), []).append(di)
            return [v for v in key_to_donors.values() if len(v) >= 2]
        except Exception as exc:
            log.warning(
                "Symmetric-donor-group detection failed (%s); proceeding "
                "with no donor equivalence groups (may under-dedup / skip "
                "the similarity restraint, never incorrectly over-dedup).",
                exc,
            )
            return []

    def _run_assignment_search(
        self, ctx, assignment, donor_dirs, subclasses, assignment_search, debug
    ):
        """
        3a. Assignment search (2.13.0) -- OPTIONAL alternative donor-to-
        site assignment candidates around the baseline Hungarian solve,
        via ik/assignment_search.py::generate_candidate_assignments.
        Only called when the baseline assignment is not fully explicit
        (nothing to search when every site is caller-pinned). See
        solve()'s use_assignment_search parameter docstring for the full
        rationale: generate_candidate_assignments only proposes extra
        candidates when the baseline is genuinely ambiguous
        (find_suspect_donors flags >=2 donors); the common/clean case
        gets back exactly [baseline] and pays zero extra solve cost, by
        construction. Every candidate that does survive gets solved
        together, in the same batched call -- see _solve_for_assignments.
        """
        from jacobik.ik.assignment_search import generate_candidate_assignments

        as_cfg = assignment_search or AssignmentSearchConfig()

        hop_matrix_full = self._build_hop_matrix(
            ctx.n_donors, ctx.all_donor_groups, ctx.heavy_graph
        )

        try:
            _assignment_candidates = list(
                generate_candidate_assignments(
                    donor_dirs,
                    ctx.ideal,
                    donor_subclasses=subclasses,
                    geometry_label=self.geometry,
                    chelate_pairs=ctx.chelate_pairs_global,
                    hop_matrix=hop_matrix_full,
                    margin_deg=as_cfg.margin_deg,
                    max_ambiguous_group=as_cfg.max_ambiguous_group,
                    k_max=as_cfg.k_max,
                )
            )
        except Exception as exc:
            log.warning(
                "Assignment search failed (%s); falling back to the baseline assignment only.",
                exc,
            )
            _assignment_candidates = [assignment]

        # 2.14.0 Fix A: drop candidates that are pure relabelings of an
        # already-kept candidate among chemically-identical donors (see
        # the 2.14.0 changelog entry -- this is what made Fe(H2O)6 solve
        # 4 physically-indistinguishable "candidates" at full cost each).
        # ctx.symmetric_donor_groups is empty either when there genuinely
        # are no symmetric donor groups, or when detecting them failed
        # (see _symmetric_donor_groups) -- both cases correctly skip
        # dedup and keep every generated candidate, matching this
        # method's pre-refactor behaviour in both cases.
        if len(_assignment_candidates) > 1 and ctx.symmetric_donor_groups:

            def _is_trivial_relabeling(cand, base) -> bool:
                grouped: set = set()
                for g in ctx.symmetric_donor_groups:
                    grouped.update(g)
                    if sorted(int(cand[d]) for d in g) != sorted(int(base[d]) for d in g):
                        return False
                return all(
                    int(cand[d]) == int(base[d]) for d in range(len(cand)) if d not in grouped
                )

            _deduped = [_assignment_candidates[0]]
            for _cand_raw in _assignment_candidates[1:]:
                if any(_is_trivial_relabeling(_cand_raw, _kept) for _kept in _deduped):
                    continue
                _deduped.append(_cand_raw)
            if len(_deduped) < len(_assignment_candidates):
                log.info(
                    "Assignment search: dropped %d candidate(s) that were "
                    "pure relabelings among chemically-identical donors "
                    "(no physical difference); keeping %d distinct "
                    "candidate(s).",
                    len(_assignment_candidates) - len(_deduped),
                    len(_deduped),
                )
            _assignment_candidates = _deduped

        log.info(
            "Assignment search: %d candidate assignment(s) (1 == unambiguous, baseline kept as-is)",
            len(_assignment_candidates),
        )
        if debug:
            print(f"\n[DEBUG] Assignment search: {len(_assignment_candidates)} candidate(s)")
            for _ci_dbg, _cand_dbg in enumerate(_assignment_candidates):
                _tag = "(baseline)" if _ci_dbg == 0 else f"(alt {_ci_dbg})"
                print(f"  {_tag} {_cand_dbg.tolist()}")

        return _assignment_candidates

    def _run_symmetry_enumeration(self, ctx, assignment, site_enumeration_config, debug):
        """
        3a'. Symmetry enumeration (phase 3, 2026-09-17) -- OPTIONAL
        PROACTIVE alternative to _run_assignment_search, via
        ik/site_enumeration.py::enumerate_unique_assignments. Feeds the
        same candidate-list mechanism _run_assignment_search does, so
        everything downstream (dedup already baked into
        enumerate_unique_assignments's dedup_combined, then the batched
        _solve_for_assignments) is reused completely unchanged -- see
        ik/site_enumeration.py's module docstring, "Proposed
        architecture" piece 4. Only called when use_symmetry_enumeration
        is True and the baseline assignment is not fully explicit; see
        solve()'s use_symmetry_enumeration parameter docstring for the
        full rationale and its two known limitations. Internally falls
        back to [assignment] (with an explanatory log message) when the
        geometry has no fixed IDEAL_GEOMETRIES entry, when n_donors !=
        the geometry's n_sites, or when enumeration itself finds zero
        reachable candidates.
        """
        from jacobik.ik.targets import IDEAL_GEOMETRIES

        if self.geometry not in IDEAL_GEOMETRIES:
            log.info(
                "Symmetry enumeration requested but geometry %r has no "
                "fixed IDEAL_GEOMETRIES entry (e.g. 'other') -- "
                "site_symmetry_group has nothing to compute against; "
                "falling back to the baseline assignment only.",
                self.geometry,
            )
            return [assignment]
        if ctx.n_donors != len(ctx.ideal):
            log.info(
                "Symmetry enumeration requested but n_donors (%d) != "
                "geometry %r's n_sites (%d) -- enumerate_unique_assignments "
                "only handles fully-saturated complexes; falling back to "
                "the baseline assignment only.",
                ctx.n_donors,
                self.geometry,
                len(ctx.ideal),
            )
            return [assignment]

        from jacobik.ik.site_enumeration import enumerate_unique_assignments

        se_cfg = site_enumeration_config or SiteEnumerationConfig()
        hop_matrix_full = self._build_hop_matrix(
            ctx.n_donors, ctx.all_donor_groups, ctx.heavy_graph
        )

        try:
            se_result = enumerate_unique_assignments(
                ctx.n_donors,
                self.geometry,
                hop_matrix_full,
                donor_equivalence_groups=ctx.symmetric_donor_groups,
                config=se_cfg,
            )
        except Exception as exc:
            log.warning(
                "Symmetry enumeration failed (%s); falling back to the baseline assignment only.",
                exc,
            )
            return [assignment]

        if not se_result.assignments:
            # n_raw == 0: the reachability model made every complete
            # bijection unreachable (see use_symmetry_enumeration's
            # parameter docstring, limitation 1, and
            # ik/site_enumeration.py's "Findings from standalone
            # validation against EDTA" -- this is a known, documented
            # limitation, not a bug in this call site) -- fall back
            # rather than proceeding with zero candidates.
            log.warning(
                "Symmetry enumeration returned zero candidates (the "
                "reachability model made every complete assignment "
                "unreachable -- see ik/site_enumeration.py's module "
                "docstring, 'Findings from standalone validation against "
                "EDTA', finding 1); falling back to the baseline "
                "assignment only."
            )
            return [assignment]

        _assignment_candidates = list(se_result.assignments)
        if se_result.truncated or se_result.dedup_truncated:
            log.warning(
                "Symmetry enumeration hit its safety budget (raw search "
                "truncated=%s, dedup truncated=%s) -- the %d candidate(s) "
                "found are all genuinely valid, but the true symmetry-"
                "unique set may be incomplete. See site_enumeration_config "
                "to raise max_candidates / time_budget_s.",
                se_result.truncated,
                se_result.dedup_truncated,
                len(_assignment_candidates),
            )
        log.info(
            "Symmetry enumeration: %d candidate assignment(s) (n_raw=%d before dedup)",
            len(_assignment_candidates),
            se_result.n_raw,
        )
        if debug:
            print(f"\n[DEBUG] Symmetry enumeration: {len(_assignment_candidates)} candidate(s)")
            for _ci_dbg, _cand_dbg in enumerate(_assignment_candidates):
                _tag = "(baseline)" if _ci_dbg == 0 else f"(alt {_ci_dbg})"
                print(f"  {_tag} {_cand_dbg.tolist()}")

        return _assignment_candidates

    def _solve_for_assignments(self, ctx, assignments, beam_width, n_ik_steps):
        """
        Runs steps 3b-6 for one OR MANY candidate donor-to-site
        `assignments`, batched into a SINGLE JointIKSolver.solve() call
        spanning B_total = len(assignments) * beam_width * n_coord_samples
        trials, and returns one EnsembleResult whose conformers can come
        from any of the candidate assignments, ranked together by score
        (see ConformerResult.assignment_idx).

        BATCHED CANDIDATE SCREENING (2026-09-17, replaces 2.13.0/2.14.0's
        sequential design): _solve_for_assignment used to be called once
        per candidate from a Python `for` loop in _resolve_candidates,
        which first cheaply screened every candidate (small beam_width/
        n_ik_steps) and then re-solved only the highest-scoring one at
        full quality -- N sequential beam-search-plus-IK rounds for N
        candidates, discarding N-1 of them after only a cheap look. That
        was never actually required by anything in the beam-search/IK
        machinery: a donor-to-site assignment only changes two things --
        which coordination-sphere target each donor aims at (built here,
        below, from the CHEAP per-candidate stats-lookup loop this method
        keeps) and the `ideal_directions` polyhedron target
        JointIKSolver aligns against. Everything else -- the ligands'
        ForwardKinematics graphs, the shared Scorer/GeometryExtractor, the
        donor symmetry/Q3 groups -- is assignment-independent, and
        JointIKSolver.solve() was ALREADY a fully batch-item-independent
        computation across its beam_width x n_coord_samples trial
        dimension (no step anywhere aggregates across batch items -- see
        ik_solver.py's per-item Procrustes rotation, shape-deviation, and
        symmetry-tie math). So "which candidate assignment" is just
        another trial axis to fold into that same batch, not a reason for
        an outer sequential loop: this method builds ONE combined batch
        (candidates are the slowest-varying axis, beam_width x
        n_coord_samples nested inside each candidate's slice, exactly as
        a single-assignment solve already nests coord-samples x beam
        trials -- see _beam_search_all_candidates), solves it in one
        JointIKSolver.solve() call at FULL (beam_width, n_ik_steps)
        quality for every candidate, and lets the existing best-to-worst
        ranking in _assemble_conformers naturally surface the best
        conformers regardless of which candidate produced them. There is
        no cheap-screening tier anymore (AssignmentSearchConfig's old
        screen_beam_width/screen_n_ik_steps are gone) because the whole
        point of screening -- avoiding N separate full-quality solves --
        is moot once N candidates cost one batched solve instead of N
        sequential ones. A single-assignment call (the common case) is
        just this same code path with len(assignments) == 1.
        """
        symmetric_groups, q3_symmetric_groups = self._symmetric_and_q3_groups(ctx)
        (
            fks,
            all_theta_inits,
            all_donor_pos_idxs,
            ge_joint,
            scorer_joint,
            X_full,
            ideal_directions_batched,
            candidate_idx_per_b,
        ) = self._beam_search_all_candidates(ctx, assignments, beam_width)
        beam_width_total = beam_width * len(assignments)
        thetas_final, converged, scores, solve_trace = self._run_joint_ik(
            ctx,
            fks,
            scorer_joint,
            ge_joint,
            X_full,
            all_theta_inits,
            all_donor_pos_idxs,
            ideal_directions_batched,
            symmetric_groups,
            q3_symmetric_groups,
            beam_width_total,
            n_ik_steps,
        )
        conformers = self._assemble_conformers(
            ctx,
            ctx.mol,
            fks,
            X_full,
            thetas_final,
            scores,
            converged,
            beam_width_total,
            candidate_idx_per_b,
        )
        result = EnsembleResult(conformers=conformers, solve_trace=solve_trace, scorer=scorer_joint)
        if len(assignments) > 1:
            result.assignment_search = dict(
                n_candidates=len(assignments),
                candidate_assignments=[a.tolist() for a in assignments],
            )
        return result

    def _ideal_directions(self, ctx, assignment):
        """
        3b. Ideal polyhedron directions in global donor order (2.3.0) --
        feeds JointIKSolver's per-step Augmented-Lagrangian shape
        constraint (ik_solver.py 2.3.0). Replaces the old fixed
        absolute-Cartesian-target anchor, whose target orientation went
        stale as soon as the ligand's free rigid pose or torsions moved
        away from the initial Kabsch fit -- see that module's changelog
        for the full rationale.
        """
        ideal_directions_full = np.stack(
            [ctx.ideal[assignment[di]] for di in range(ctx.n_donors)], axis=0
        )  # (n_donors, 3) unit vectors, fixed for the whole solve
        return torch.tensor(ideal_directions_full, dtype=torch.float32, device=ctx.device)

    def _symmetric_and_q3_groups(self, ctx):
        """
        3c. symmetric_groups/q3_symmetric_groups for the M-D bond-length
        and Q3 approach-angle similarity restraints. symmetric_groups is
        just ctx.symmetric_donor_groups (computed once per solve() call,
        assignment-independent -- see _symmetric_donor_groups).
        q3_symmetric_groups further restricts to donors that share a
        single ligand INSTANCE.

        BUGFIX (2.12.0): q3_symmetric_groups is used ONLY for the Q3
        approach-angle tie/mode-commit (sym_angle_weight, q3_mode_commit),
        NOT for the M-D bond-length tie (sym_weight), which keeps using
        the unrestricted symmetric_groups. Root cause this addresses:
        symmetric_groups groups donors purely by (SMILES, canonical
        rank), so N separate instances of the same repeated monodentate
        ligand (e.g. 6 independent water molecules on [M(H2O)6]) land in
        ONE group exactly like a single ligand's own internally-
        symmetric donor arms (e.g. EDTA's 4 carboxylate O, all within one
        ligand instance). For EDTA that grouping is correct: the four
        arms are mechanically coupled through one shared backbone, so a
        genuine "which mode do they jointly commit to" question exists,
        which is exactly what q3_mode_commit (2.5.0) was built to
        answer. For 6 independent water ligands there is no such
        coupling -- nothing chemically requires separate, freely-
        rotating monodentate donors to share a single Q3 approach-angle
        mode, and the real fitted Q3:('O_water', metal, geometry)
        distribution is itself genuinely multi-modal (pooled over many
        different real crystal environments). Confirmed empirically:
        committing all 6 waters of [Fe(H2O)6]3+ to one shared mode via
        the initial-pose argmax reliably locked onto a LOW-weight
        minority mode (~4% of the fitted mixture) purely because the
        (arbitrary, embedding-derived) initial pose happened to sit
        closest to it -- producing a chemically nonsensical, wildly
        asymmetric M-O-H geometry (one H at ~68 deg from the metal, the
        other at ~172 deg, instead of the expected symmetric ~120/120)
        in every trial, despite a HIGHER solve score than the correct,
        symmetric geometry (q3_mode_commit=False on the same case gives
        clean ~126/126 deg M-O-H angles). Fix: restrict
        q3_mode_commit's (and the matching Q3 sym_angle_weight tie's)
        group membership to donors that share a single ligand INSTANCE,
        leaving cross-instance donors to each independently score
        against the full (correctly weighted) Q3 mixture instead of
        being hard-committed to one arbitrarily-chosen mode. EDTA-style
        single-ligand multidentate cases are completely unaffected (all
        their donors already share one instance).

        (2026-09-17 refactor: RDKit ranking itself, and its own failure
        handling, now live once in _symmetric_donor_groups -- this
        method no longer needs its own try/except around that.)
        """
        symmetric_groups = ctx.symmetric_donor_groups
        q3_symmetric_groups = [
            g for g in symmetric_groups if len({ctx.all_donor_groups[d][0] for d in g}) == 1
        ]
        if ctx.debug and symmetric_groups:
            print(f"\n[DEBUG]   Symmetric donor groups (2.3.0): {symmetric_groups}")
        if ctx.debug and symmetric_groups != q3_symmetric_groups:
            print(
                f"\n[DEBUG]   Q3-eligible symmetric groups, within-ligand-"
                f"instance only (2.12.0): {q3_symmetric_groups}"
            )
        return symmetric_groups, q3_symmetric_groups

    def _beam_search_all_candidates(self, ctx, assignments, beam_width):
        """
        Steps 4 + 5a-5g, batched across every candidate donor-to-site
        `assignments` at once (2026-09-17 -- replaces the old
        _beam_search_all_ligands, called once per candidate from a
        sequential loop; see _solve_for_assignments' "BATCHED CANDIDATE
        SCREENING" docstring for the full rationale). A donor-to-site
        assignment only changes two things: which coordination-sphere
        target each donor aims at, and the ideal_directions polyhedron
        JointIKSolver aligns against -- both built here PER CANDIDATE
        (cheap: n_candidates x n_donors numpy/stats-lookup work, not
        GPU-bound) and then tiled beam_width x n_coord_samples trials
        inside each candidate's own slice of one combined batch, exactly
        as a single-assignment solve already tiles n_coord_samples x
        beam_width trials -- candidates are just the new, outermost axis.
        FK, the shared GeometryExtractor/Scorer, and the W conformational
        beam-search torsion seeds are assignment-independent and are
        therefore each built ONCE and shared across every candidate's
        slice (previously rebuilt once per candidate, purely as a side
        effect of this method itself being called once per candidate --
        not by design; sharing them removes that redundant work too, on
        top of the IK solve batching itself). Torsion seeds being shared
        across candidates means every candidate now explores the
        IDENTICAL set of W conformational hypotheses (apples to apples),
        rather than an arbitrary RNG-order-dependent set each; per-trial
        orientation diversity is unaffected -- kabsch_init below still
        injects an independent random rotation per batch row.

        Returns everything _run_joint_ik needs to run ONE shared
        JointIKSolver call over the full combined batch: the per-ligand
        ForwardKinematics objects, their theta seeds/donor-position
        indices, the shared GeometryExtractor/Scorer built on the full
        complex, the flattened reference coordinates, the per-batch-item
        ideal_directions tensor (see ik_solver.py's 2026-09-17 batched-
        Procrustes change), and candidate_idx_per_b -- a (B_total,) int
        array saying which candidate (index into `assignments`) each
        batch row belongs to, for _assemble_conformers's assignment_idx
        bookkeeping.
        """
        from jacobik.ik.beam import BeamSearch
        from jacobik.ik.fk import ForwardKinematics
        from jacobik.ik.geometry import GeometryExtractor
        from jacobik.ik.ik_solver import kabsch_init
        from jacobik.ik.scorer import Scorer
        from jacobik.topology.decompose import decompose_ligand

        mol = ctx.mol
        debug = ctx.debug
        n_candidates = len(assignments)

        # ------------------------------------------------------------------
        # 4. Build coord-sphere targets + ideal_directions PER CANDIDATE
        #    (cheap CPU/numpy stats-lookup work -- see this method's
        #    docstring for why this loop being O(n_candidates) is fine).
        # ------------------------------------------------------------------
        targets_global_per_cand: List[np.ndarray] = []  # each: (n_donors, M, 3)
        ideal_directions_per_cand: List[torch.Tensor] = []  # each: (n_donors, 3)

        for assignment in assignments:
            targets_global = np.zeros((ctx.n_donors, ctx.n_coord_samples, 3), dtype=float)

            for di, (li, gi, grp) in enumerate(ctx.all_donor_groups):
                lbl = mol.donor_labels.get(grp[0])
                subclass = lbl.subclass if lbl else "element_any"
                family = lbl.family if lbl else "element_any"
                site_vec = ctx.ideal[assignment[di]]  # (3,) unit vector

                # Trans partner
                trans_p = "none"
                for dj, (lj, gj, grpj) in enumerate(ctx.all_donor_groups):
                    if dj == di:
                        continue
                    dot = float(np.dot(site_vec, ctx.ideal[assignment[dj]]))
                    if dot < -0.7:
                        lblj = mol.donor_labels.get(grpj[0])
                        trans_p = lblj.subclass if lblj else "element_any"
                        break

                q1_key = (
                    self.metal,
                    self.spin,
                    self.charge,
                    subclass,
                    self.geometry,
                    trans_p,
                )
                # See exp.md for detailed rationale
                sampled_dists = _q1_sample_distances(
                    ctx.stats,
                    q1_key,
                    family,
                    ctx.n_coord_samples,
                    ctx.np_rng,
                )  # (M,)
                targets_global[di] = site_vec[None, :] * sampled_dists[:, None]

                # Per-donor debug printout only for the common single-
                # candidate case -- n_candidates x n_donors of this would
                # be unreadable for a real multi-candidate run.
                if debug and n_candidates == 1:
                    print(f"\n[DEBUG] ── Donor {di} (lig={li} grp={gi}) ──────────────")
                    print(f"  subclass={subclass!r}  family={family!r}  trans_partner={trans_p!r}")
                    print(f"  Q1 key   : {q1_key}")
                    print(
                        f"  sampled M-D lengths: mean={float(np.mean(sampled_dists)):.4f} "
                        f"min={float(np.min(sampled_dists)):.4f} "
                        f"max={float(np.max(sampled_dists)):.4f} Å "
                        f"(M={ctx.n_coord_samples})"
                    )
                    print(f"  site_vec : {site_vec}")
                    print(
                        f"  targets[0]: {targets_global[di][0]}  "
                        f"(norm={np.linalg.norm(targets_global[di][0]):.4f} Å)"
                    )

            targets_global_per_cand.append(targets_global)
            ideal_directions_per_cand.append(self._ideal_directions(ctx, assignment))

        if debug and n_candidates > 1:
            print(
                f"\n[DEBUG] Built coordination-sphere targets for "
                f"{n_candidates} candidate assignments."
            )

        # See exp.md for detailed rationale
        for lig_obj in mol.ligands:
            decompose_ligand(mol, lig_obj)

        # Build the shared GE and scorer once on the full complex --
        # assignment-independent, so shared across every candidate.
        # quantity_weights are baked into Scorer at construction time —
        # no per-step overhead during the solve loop.
        ge_joint = GeometryExtractor(mol, ctx.device)
        scorer_joint = Scorer(
            mol,
            ge_joint,
            ctx.stats,
            device=ctx.device,
            quantity_weights=ctx.quantity_weights,
            q2_aggregation=ctx.q2_aggregation,
            q2_power_p=ctx.q2_power_p,
        )

        X_full = torch.tensor(
            np.stack([a.position for a in mol.atoms]),
            dtype=torch.float32,
            device=ctx.device,
        )  # (N_total, 3)

        W = beam_width
        M = ctx.n_coord_samples
        B_per_cand = M * W
        B = B_per_cand * n_candidates

        # (B,): which candidate each batch row belongs to -- candidates
        # are the slowest-varying axis, exactly like M is slower than W
        # within one candidate's own B_per_cand-sized slice below.
        candidate_idx_per_b = np.repeat(np.arange(n_candidates), B_per_cand)

        ideal_directions_batched = torch.cat(
            [d.unsqueeze(0).expand(B_per_cand, -1, -1) for d in ideal_directions_per_cand],
            dim=0,
        )  # (B, n_donors, 3)

        fks: List = []
        all_theta_inits: List[torch.Tensor] = []
        all_donor_pos_idxs: List[torch.Tensor] = []
        all_lig_di: List[List[int]] = []
        _fk_n_joints: List[int] = []  # cache: n_joints per ligand, built incrementally

        for li, (lig_inp, lig_obj) in enumerate(zip(self.ligands, mol.ligands)):
            log.info(
                "Ligand %d/%d: FK + beam search (%d candidate assignment(s))...",
                li + 1,
                len(self.ligands),
                n_candidates,
            )

            # 5a. FK for this ligand (decomposition already done above, before
            #     ge_joint/scorer_joint were constructed)
            fk = ForwardKinematics(mol, lig_obj, ctx.device, enable_bend_dof=ctx.enable_bend_dof)
            fks.append(fk)
            # 2.11.0: cache n_joints_base (not the possibly bend-inflated
            # n_joints) -- this feeds the Q5 offset/slicing below, and Q5
            # has no data for bend joints (see fk.py 2.4.0 / beam.py 2.1.0).
            _fk_n_joints.append(fk.n_joints_base)

            if debug:
                print(f"\n[DEBUG] ── Ligand {li} ─────────────────────────────────")
                print(f"  SMILES       : {lig_inp.smiles}")
                print(f"  J (joints)   : {fk.n_joints}")
                print(f"  n_donors(GE) : {ge_joint.n_donors}")
                _X_smoke = X_full.unsqueeze(0)
                try:
                    _s_smoke, _bd_smoke = scorer_joint.score(_X_smoke)
                    print(
                        f"  scorer smoke : score={_s_smoke[0].item():.4f}  "
                        f"nan={torch.isnan(_s_smoke).any().item()}"
                    )
                except Exception as _e:
                    print(f"  scorer smoke : EXCEPTION → {_e}")

            # 5b. Donor targets for this ligand, across ALL candidates
            lig_di = [di for di, (l, g, grp) in enumerate(ctx.all_donor_groups) if l == li]
            n_d_lig = len(lig_di)
            all_lig_di.append(lig_di)

            if n_d_lig == 0:
                log.warning("Ligand %d has no donor groups — using zero theta", li)
                J_full = fk.n_joints  # full count incl. any bend joints
                all_theta_inits.append(torch.zeros(B, 6 + J_full, device=ctx.device))
                all_donor_pos_idxs.append(torch.zeros(1, dtype=torch.long, device=ctx.device))
                continue

            # (n_candidates, M, n_d_lig, 3)
            donor_targets_np = np.stack(
                [
                    np.stack([targets_global[di] for di in lig_di], axis=1)
                    for targets_global in targets_global_per_cand
                ],
                axis=0,
            )

            if debug:
                print(f"  n_d_lig={n_d_lig}  lig_di={lig_di}")
                print(
                    "  donor_targets_np shape (n_candidates, M, n_d_lig, 3): "
                    f"{donor_targets_np.shape}"
                )
                print(
                    "  target norms (candidate 0, sample 0): "
                    + "  ".join(
                        f"{np.linalg.norm(donor_targets_np[0, 0, d]):.3f} Å" for d in range(n_d_lig)
                    )
                )

            # See exp.md for detailed rationale
            # 2.11.0: J is pinned to n_joints_base for Q5 sizing/offsets
            # (see fk.py 2.4.0 / beam.py 2.1.0 changelogs) -- bend joints
            # have no Q5 data and are handled separately below via J_full.
            J = fk.n_joints_base
            J_full = fk.n_joints
            log.info(
                "  Ligand %d: J=%d joints (%d bend), beam_width=%d",
                li,
                J_full,
                J_full - J,
                beam_width,
            )
            ge_li = GeometryExtractor(mol, ctx.device)
            scorer_li = Scorer(
                mol,
                ge_li,
                ctx.stats,
                device=ctx.device,
                quantity_weights=ctx.quantity_weights,
                q2_aggregation=ctx.q2_aggregation,
                q2_power_p=ctx.q2_power_p,
            )
            # ge_li sees the same complex but its _q5_keys covers all ligands;
            # BeamSearch needs only this ligand's joints in rows 0..J-1.
            # Compute the joint offset: sum of n_joints for ligands 0..li-1.
            _joint_offset = sum(_fk_n_joints[:li])  # O(1): cached from FK build loop
            _q5_m = scorer_li._q5_means[_joint_offset : _joint_offset + J]  # (J, K)
            _q5_k = scorer_li._q5_kappas[_joint_offset : _joint_offset + J]
            _q5_w = scorer_li._q5_weights[_joint_offset : _joint_offset + J]
            beam = BeamSearch(fk, beam_width=beam_width, device=ctx.device)
            # Computed ONCE, shared across every candidate's slice of the
            # batch (see this method's docstring) -- not redrawn per
            # candidate.
            beam_torsions, _ = beam.search(
                X_ref=X_full,
                base_theta=torch.zeros(1, 6, device=ctx.device),
                q5_means=_q5_m,
                q5_kappas=_q5_k,
                q5_weights=_q5_w,
                q6_mu=scorer_li._q6_mu,
                q6_sigma=scorer_li._q6_sigma,
                q6_pairs=getattr(ge_li, "_q6_idx_pairs", None),
            )  # (W, J)

            if debug:
                print(f"\n[DEBUG]   Beam search → shape={tuple(beam_torsions.shape)}")
                if J > 0:
                    print(
                        f"  torsion range : [{beam_torsions.min().item():.3f}, "
                        f"{beam_torsions.max().item():.3f}] rad"
                    )

            # 5e. Tile n_candidates × M × W → B parallel seeds
            donor_targets_torch = (
                torch.tensor(
                    donor_targets_np,
                    dtype=torch.float32,
                    device=ctx.device,
                )
                .unsqueeze(2)  # (n_candidates, M, 1, n_d_lig, 3)
                .expand(n_candidates, M, W, n_d_lig, 3)
                .reshape(B, n_d_lig, 3)
            )

            beam_torsions_B = (
                beam_torsions.unsqueeze(0)
                .unsqueeze(0)  # (1, 1, W, J)
                .expand(n_candidates, M, W, J)
                .reshape(B, J)
            )

            # Append the bend joints (fk.py), which beam search never
            # touches (no Q5 statistics for them). Seed each at its own
            # current-conformer dihedral (zero delta -- exact no-op at
            # t=0), then randomise per beam trial so different trials
            # explore different local pyramidal configurations at
            # branch-point donors, mirroring how the base orientation
            # is diversified below. Left entirely to gradient descent
            # from there.
            n_bend = J_full - J
            if n_bend > 0:
                bend_seed = fk.bend_seed_angles(X_full)  # (n_bend,)
                bend_seed_B = bend_seed.unsqueeze(0).expand(B, n_bend).clone()
                noise = (torch.rand(B, n_bend, device=ctx.device) - 0.5) * 2.0 * math.pi
                bend_seed_B = bend_seed_B + noise
                beam_torsions_B = torch.cat([beam_torsions_B, bend_seed_B], dim=-1)  # (B, J_full)

            if debug:
                print(f"  bend joints  : {n_bend}")

            # 5f. Donor position indices (indices into GE's global donor list)
            # We pass the group indices (lig_di) directly instead of atom indices,
            # so the solver can correctly fetch centroids for haptic groups.
            donor_pos_idx_t = torch.tensor(lig_di, dtype=torch.long, device=ctx.device)

            if debug:
                print(f"  donor_pos_idx  : {lig_di}")
                _actual_all = ge_joint.donor_positions(X_full.unsqueeze(0))[0].cpu().numpy()
                _actual = _actual_all[lig_di]
                for _d, (_act, _tgt) in enumerate(zip(_actual, donor_targets_np[0, 0])):
                    print(
                        f"    donor[{_d}] current={np.array2string(_act, precision=3)}  "
                        f"target (candidate 0)={np.array2string(_tgt, precision=3)}  "
                        f"dist={np.linalg.norm(_act - _tgt):.3f} Å"
                    )

            # 5g. Kabsch init (per-ligand, ALL candidates in one call --
            # kabsch_init already loops independently over its batch
            # dimension, so passing B = n_candidates*M*W here needs no
            # change to it whatsoever).
            base_theta = kabsch_init(
                ge=ge_joint,
                X_ref=X_full,
                donor_targets=donor_targets_torch,
                donor_pos_idx=donor_pos_idx_t,
                diversify_orientation=True,
                diversity_scale=1.0,
            )  # (B, 6)
            theta_init = torch.cat([base_theta, beam_torsions_B], dim=-1)  # (B, 6+J)

            if debug:
                print(
                    f"\n[DEBUG]   Kabsch init → shape={tuple(theta_init.shape)}"
                    f"  nan={torch.isnan(theta_init).any().item()}"
                )
                print(
                    f"  translation norm: min={base_theta[:, :3].norm(dim=-1).min():.3f}"
                    f"  max={base_theta[:, :3].norm(dim=-1).max():.3f} Å"
                )

            all_theta_inits.append(theta_init)
            all_donor_pos_idxs.append(donor_pos_idx_t)

        return (
            fks,
            all_theta_inits,
            all_donor_pos_idxs,
            ge_joint,
            scorer_joint,
            X_full,
            ideal_directions_batched,
            candidate_idx_per_b,
        )

    def _run_joint_ik(
        self,
        ctx,
        fks,
        scorer_joint,
        ge_joint,
        X_full,
        all_theta_inits,
        all_donor_pos_idxs,
        ideal_directions_full_t,
        symmetric_groups,
        q3_symmetric_groups,
        beam_width,
        n_ik_steps,
    ):
        """
        5h. Joint IK solve — all ligands (and, since 2026-09-17, every
        candidate donor-to-site assignment being explored) optimised
        simultaneously against the shared Scorer/GeometryExtractor built
        by _beam_search_all_candidates. `beam_width` here is the caller's
        TOTAL beam width across all candidates (see
        _solve_for_assignments), so B = beam_width * n_coord_samples
        already comes out as the full combined batch size -- no other
        change was needed here for multi-candidate batching to work.
        """
        from jacobik.ik.ik_solver import IKSolverConfig, JointIKSolver

        debug = ctx.debug
        B = beam_width * ctx.n_coord_samples
        log.info("Running joint IK: %d ligands × B=%d × T=%d steps", len(fks), B, n_ik_steps)

        # n_ik_steps/lr are the two solve()-level knobs everyone actually
        # tunes per-call; everything else comes from ik_config.
        _resolved_ik_config = replace(
            ctx.ik_config or IKSolverConfig(), n_steps=n_ik_steps, lr=ctx.lr
        )
        joint_solver = JointIKSolver(
            fks=fks,
            scorer=scorer_joint,
            ge=ge_joint,
            config=_resolved_ik_config,
            device=ctx.device,
        )
        thetas_final, converged, scores, solve_trace = joint_solver.solve(
            theta_inits=all_theta_inits,
            X_ref=X_full,
            donor_pos_idxs=all_donor_pos_idxs,
            ideal_directions=ideal_directions_full_t,
            debug=debug,
            symmetric_groups=symmetric_groups,
            q3_symmetric_groups=q3_symmetric_groups,
            q3_mode_commit=ctx.q3_mode_commit,
        )

        if debug:
            n_conv = int(converged.sum().item())
            print(
                f"\n[DEBUG] Joint IK done: {n_conv}/{B} converged"
                f"  score min={scores.min().item():.4f}"
                f"  max={scores.max().item():.4f}"
            )

            # See exp.md for detailed rationale
            _scores_sorted_idx = np.argsort(-scores.cpu().numpy())
            print(f"\n[DEBUG] ── All {B} raw IK trials (sorted by score) ──────────────")
            print(f"  {'trial':>6}  {'score':>12}  {'converged':>9}")
            for b in _scores_sorted_idx.tolist():
                print(
                    f"  {b:>6}  {scores[b].item():>12.4f}  "
                    f"{'yes' if converged[b].item() else 'no':>9}"
                )

        return thetas_final, converged, scores, solve_trace

    def _assemble_conformers(
        self,
        ctx,
        mol,
        fks,
        X_full,
        thetas_final,
        scores,
        converged,
        beam_width,
        candidate_idx_per_b=None,
    ):
        """
        6. Assemble full 3D complexes (deep-copied per kept conformer)
        from the joint IK solve's final thetas, sorted best-to-worst by
        score. `candidate_idx_per_b` (2026-09-17, optional, (B,) int
        array) tags each kept conformer with which candidate donor-to-
        site assignment it came from, via ConformerResult.assignment_idx
        -- see _solve_for_assignments. None (the pre-2026-09-17 default,
        still what every single-assignment solve effectively gets) means
        every conformer shares the one assignment the caller used, so
        assignment_idx is left None throughout -- nothing downstream that
        doesn't care about multi-candidate batching needs to change.
        """
        debug = ctx.debug
        _n_conformers = (
            ctx.n_conformers
            if ctx.n_conformers is not None
            else max(beam_width, ctx.n_coord_samples)
        )
        conformers: List[ConformerResult] = []

        score_np = scores.cpu().numpy()
        sorted_b = np.argsort(-score_np)

        for ci, b in enumerate(sorted_b[:_n_conformers]):
            mol_copy = _shallow_copy_mol_for_conformer(mol)

            X_assembled = X_full.unsqueeze(0)  # (1, N, 3)
            for li, fk_li in enumerate(fks):
                theta_b = thetas_final[li][b : b + 1]  # (1, 6+Ji)
                with torch.no_grad():
                    X_assembled = fk_li.apply(theta_b, X_assembled)  # (1, N, 3)

            X_np = X_assembled[0].cpu().numpy()
            for atom_pos_idx, atom in enumerate(mol_copy.atoms):
                atom.position = X_np[atom_pos_idx].copy()

            # See exp.md for detailed rationale
            _refresh_coord_geometry(mol_copy)

            conformers.append(
                ConformerResult(
                    complex=mol_copy,
                    score=float(score_np[b]),
                    converged=bool(converged[b].item()),
                    rank=ci + 1,
                    trial_idx=int(b),
                    assignment_idx=(
                        int(candidate_idx_per_b[b]) if candidate_idx_per_b is not None else None
                    ),
                )
            )

        # Sort and re-rank
        conformers.sort(key=lambda c: c.score, reverse=True)
        for i, c in enumerate(conformers):
            c.rank = i + 1

        n_conv = sum(1 for c in conformers if c.converged)
        log.info(
            "Engine done: %d conformers, %d converged, best score=%.4f",
            len(conformers),
            n_conv,
            conformers[0].score if conformers else float("nan"),
        )

        if debug and len(conformers) > 1:
            # See exp.md for detailed rationale
            print(
                "\n[DEBUG] ── Pairwise RMSD between kept conformers "
                "(heavy atoms, Kabsch-aligned) ──"
            )
            heavy_idx = [
                i
                for i, a in enumerate(conformers[0].complex.atoms)
                if a.symbol != "H" and a.idx != conformers[0].complex.metal_idx
            ]
            coords = [
                np.array([c.complex.atoms[i].position for i in heavy_idx]) for c in conformers
            ]
            n_c = len(conformers)
            header = "        " + "".join(f"{i + 1:>8}" for i in range(n_c))
            print(header)
            for i in range(n_c):
                row = f"  rank{i + 1:>3} "
                for j in range(n_c):
                    if j < i:
                        row += " " * 8
                        continue
                    if i == j:
                        row += f"{0.0:>8.3f}"
                        continue
                    Pc = coords[i] - coords[i].mean(axis=0)
                    Qc = coords[j] - coords[j].mean(axis=0)
                    Hm = Pc.T @ Qc
                    Uu, _, Vt = np.linalg.svd(Hm)
                    dsign = np.sign(np.linalg.det(Vt.T @ Uu.T))
                    sign_fix = np.eye(3)
                    sign_fix[-1, -1] = dsign
                    Rm = Vt.T @ sign_fix @ Uu.T
                    Pc_aligned = Pc @ Rm.T
                    rmsd = float(np.sqrt(((Pc_aligned - Qc) ** 2).sum(axis=-1).mean()))
                    row += f"{rmsd:>8.3f}"
                print(row)
            print(
                "  (columns/rows are final rank; 0.000 on the diagonal; "
                "values near 0 off-diagonal mean two kept conformers are "
                "essentially the same geometry)"
            )

        return conformers

    # 2026-09-17: _resolve_candidates (the old single-vs-multi-candidate
    # dispatch, and the 2.14.0 "screen every candidate cheaply, re-solve
    # only the winner" logic it dispatched to) is gone -- both cases are
    # now the SAME call, _solve_for_assignments(ctx, assignment_candidates,
    # beam_width, n_ik_steps), since batching made "single vs. multiple
    # candidates" no longer a meaningfully different code path. See that
    # method's docstring for the full rationale and solve()'s own call
    # site below.
