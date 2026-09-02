"""
engine.py
=========
Public API and orchestration for the Jacobik conformer ensemble engine.

Version history
---------------
1.0.0 : initial release
1.1.0 : _DEFAULT_QUANTITY_WEIGHTS introduced with Q3=3.0, Q4=3.0 to
         compensate for gradient dilution in broad amine distributions
         (superseded by 1.2.0).
1.2.0 : _DEFAULT_QUANTITY_WEIGHTS reverted to all-1.0.  Per-DOF lr
         adaptation is now handled by Adam on c_grad (ik_solver.py 1.1.0),
         making the weight boost unnecessary and removing a magic constant.
         New solve() params threaded through to JointIKSolver: chem_lr,
         adam_betas, lambda_decay_tol, anchor_lse_temp.
1.3.0 : Fixed polydentate ligand initial placement in _build_complex_shell.
         The old logic placed the donor centroid at 2.5 Å from the metal for
         ALL ligands.  For monodentate this is correct.  For polydentate
         (EDTA: 6 donors spanning ±5 Å in extended ETKDG conformation) it
         placed individual donors anywhere from 0 to 10 Å away, with at least
         one nitrogen at ~0.66 Å (essentially on the metal).  This poisoned
         the Hungarian assignment (garbage directions near 0°) and the Kabsch
         init, making the IK solve impossible to converge regardless of solver
         quality.  Fix: for n_donor_groups ≥ 2, scale the conformer so the
         donor radius of gyration equals the target M-D bond length (~2.0 Å)
         and center the donor centroid at the metal origin.  All donors then
         start at approximately the correct coordination distance.
2.1.0 : Speed improvements aligned with scorer.py 2.1.0 and ik_solver.py 2.1.0.
         q2_weight replaced by general quantity_weights dict passed to Scorer
         at construction time (no per-step overhead).  JointIKSolver no longer
         takes q2_weight or quantity_weights — all weight logic lives in Scorer.
2.2.0 : Benchmark-oriented speed improvements.
         solve() gains three new optional parameters:
           stats        — accept a pre-built StatsLookup, bypassing load_db()
                          (eliminates repeated JSON parse across benchmark loop).
           fast_embed   — skip MMFF optimisation in _smiles_to_conformer()
                          (saves ~0.2–0.5 s per ligand; conformer is overwritten
                          by IK anyway so MMFF quality is irrelevant).
           n_conformers — cap the number of conformers assembled from IK results
                          (default: max(beam_width, n_coord_samples) as before).
         Joint-offset O(L²) FK construction replaced by a single n_joints cache
         built during the per-ligand FK construction pass.
2.3.0 : solve() gains anchor_floor_frac, diversify_orientation, and
         diversity_scale, threaded through to kabsch_init() (ik_solver.py
         2.2.0) and JointIKSolver (ik_solver.py 2.2.0). Root cause: for
         polydentate ligands with 3+ non-collinear donors fit jointly, the
         Kabsch base-orientation fit is a well-posed unique solution, so
         every trial in the batch received an IDENTICAL starting rigid-body
         placement regardless of beam_width/n_coord_samples -- confirmed on
         EDTA (6 donors): 128 raw IK trials collapsed to 3 distinct final
         geometries (pairwise RMSD ~0 within each cluster). Setting
         diversify_orientation=True gives each trial an independent random
         starting orientation, so batch size actually buys distinct basins
         instead of near-copies of one. Measured on EDTA: best score
         -20.82 -> -12.47 at beam_width=16, improving further to -9.99 at
         beam_width=128 (monotonic with more trials, unlike before this fix
         where more trials/steps did not help). anchor_floor_frac was also
         added and tested in isolation (kept a small permanent pull toward
         the assigned geometry instead of fully decaying to 0) but had
         negligible effect on EDTA on its own -- default 0.0 (old
         behaviour) until it's validated to help in combination with the
         orientation fix on a broader test set.
2.4.0 : Replaced copy.deepcopy(mol) at the per-output-conformer assembly
         step with _shallow_copy_mol_for_conformer() (see that function's
         docstring for the full rationale/numbers). deepcopy recursed into
         static topology metadata (rigid_bodies, pivot_chains,
         decomposition, rotatable_bonds, donor_labels, bonds) that's
         identical across every conformer of one solve -- measured 61,408
         recursive copy calls across 32 top-level deepcopy() calls in one
         profiled EDTA solve. Only atoms, coord_sites, and each ligand's
         haptic_groups actually need independent copies (their fields get
         mutated in-place by _refresh_coord_geometry); everything else is
         shared by reference.
2.5.0 : BUGFIX -- solve(seed=...) never actually seeded PyTorch's global
         RNG. It seeded np.random's global state and a LOCAL
         torch.Generator() that was created, seeded, and then never passed
         to anything anywhere in the codebase (confirmed via grep: zero
         `generator=` usages in engine.py, ik_solver.py, beam.py). Every
         unqualified torch.randn/torch.rand call in the pipeline (beam
         search, kabsch_init's diversify_orientation added in 2.3.0) reads
         from PyTorch's actual global RNG, so results for a given seed
         depended entirely on whatever torch random state happened to
         exist from code that ran BEFORE solve() was called -- different
         across notebooks, scripts, or call order, even for the identical
         seed. Fixed by calling torch.manual_seed(seed) directly (the
         orphaned local Generator was removed, not repurposed, since
         nothing reads from it). Verified: same seed now gives bit-
         identical results whether solve() is the first torch call in the
         process or is preceded by hundreds of unrelated torch.randn/rand
         calls with a different manual_seed set beforehand.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

#: Prominence floor for Q1 GMM modes: modes whose weight is below this
#: fraction of the maximum-weight mode are dropped as artefacts before
#: sampling.  The Q1 GMM for amine donors carries a spurious ~0.045 Å
#: "bond" mode (< 5% of max weight) that is a fitting artefact, not a
#: real distribution feature.  A 10% prominence floor drops it while
#: keeping all real modes.  (Finding F4 in REWRITE_PLAN.md.)
_Q1_PROMINENCE_FLOOR: float = 0.10


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
        lambda_init: float = 10.0,
        anchor_floor_frac: float = 0.0,
        diversify_orientation: bool = False,
        diversity_scale: float = 1.0,
        n_coord_samples: int = 8,
        quantity_weights: Optional[Dict[str, float]] = None,
        lse_temp: float = 2.0,
        adam_betas: tuple = (0.9, 0.999),
        device: str = "cpu",
        seed: Optional[int] = None,
        debug: bool = False,
        stats: Optional[object] = None,
        fast_embed: bool = False,
        n_conformers: Optional[int] = None,
    ) -> EnsembleResult:
        """
        Solve for a conformer ensemble.

        Parameters
        ----------
        constraints_db   : path to constraints.json produced by fitter.py
        beam_width       : W — number of torsion seeds per ligand from beam search
        n_ik_steps       : T — gradient descent iterations per solve
        lr               : learning rate for Adam on the combined gradient
        lambda_init      : anchor weight at t=0; decays toward 0 as score
                           improves (init-only anchor semantics).
        n_coord_samples  : M — coordination sphere samples (diversity of targets)
        quantity_weights : optional dict mapping "Q1".."Q6" → float weight.
                           Controls which quantities contribute and how much.
                           Default: all 1.0 (equal footing).  Common use:
                             {"Q2": 0.0}  — disable D–M–D angle scoring
                             {"Q2": 2.0}  — double-weight D–M–D angles
                           Weights are baked into the Scorer at construction
                           time (no per-step overhead).
        lse_temp         : temperature for log-sum-exp soft-max over per-donor
                           squared errors in the anchor loss. Default 2.0.
        adam_betas        : (β₁, β₂) for Adam. Default (0.9, 0.999).
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

        Returns
        -------
        EnsembleResult with conformers sorted best→worst by score.
        """
        from jacobik.ik.beam import BeamSearch
        from jacobik.ik.fk import ForwardKinematics
        from jacobik.ik.geometry import GeometryExtractor
        from jacobik.ik.ik_solver import kabsch_init
        from jacobik.ik.scorer import Scorer
        from jacobik.ik.targets import IDEAL_GEOMETRIES, hungarian_assign
        from jacobik.stats.stats import StatsLookup, load_db
        from jacobik.topology.decompose import decompose_ligand
        from jacobik.topology.donor_classifier import classify

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
        # 1. Load stats DB (or use pre-built StatsLookup from caller)
        # ------------------------------------------------------------------
        if stats is None:
            if constraints_db is None:
                from importlib import resources
                constraints_db = resources.files("jacobik") / "constraints.json"
            log.info("Loading constraints DB: %s", constraints_db)
            db = load_db(constraints_db)
            stats = StatsLookup(db)
        else:
            log.info("Using pre-built StatsLookup (skipping load_db)")

        # ------------------------------------------------------------------
        # 2. Build TMComplex shell + run donor classification
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 3. Build site assignment (Hungarian or explicit)
        # ------------------------------------------------------------------
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

        # Determine assignment
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

            # For donors without explicit sites, run Hungarian on remainder
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
                    free_assign = hungarian_assign(
                        free_dirs,
                        free_ideal,
                        donor_subclasses=free_subs,
                        geometry_label=self.geometry,
                    )
                    for di, local_si in zip(free_donors, free_assign):
                        assignment[di] = free_sites[local_si]
            else:
                assignment = hungarian_assign(
                    donor_dirs,
                    ideal,
                    donor_subclasses=subclasses,
                    geometry_label=self.geometry,
                )

        log.info("Site assignment: %s", assignment.tolist())

        # ------------------------------------------------------------------
        # 4. Build coord-sphere targets (M samples per donor)
        # ------------------------------------------------------------------
        # targets_global : (n_donors, M, 3)  metal-centred
        targets_global = np.zeros((n_donors, n_coord_samples, 3), dtype=float)

        for di, (li, gi, grp) in enumerate(all_donor_groups):
            lbl = mol.donor_labels.get(grp[0])
            subclass = lbl.subclass if lbl else "element_any"
            family = lbl.family if lbl else "element_any"
            site_vec = ideal[assignment[di]]  # (3,) unit vector

            # Trans partner
            trans_p = "none"
            for dj, (lj, gj, grpj) in enumerate(all_donor_groups):
                if dj == di:
                    continue
                dot = float(np.dot(site_vec, ideal[assignment[dj]]))
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
                stats,
                q1_key,
                family,
                n_coord_samples,
                np_rng,
            )  # (M,)
            targets_global[di] = site_vec[None, :] * sampled_dists[:, None]

            if debug:
                print(f"\n[DEBUG] ── Donor {di} (lig={li} grp={gi}) ──────────────")
                print(f"  subclass={subclass!r}  family={family!r}  trans_partner={trans_p!r}")
                print(f"  Q1 key   : {q1_key}")
                print(
                    f"  sampled M-D lengths: mean={float(np.mean(sampled_dists)):.4f} "
                    f"min={float(np.min(sampled_dists)):.4f} "
                    f"max={float(np.max(sampled_dists)):.4f} Å "
                    f"(M={n_coord_samples})"
                )
                print(f"  site_vec : {site_vec}")
                print(
                    f"  targets[0]: {targets_global[di][0]}  "
                    f"(norm={np.linalg.norm(targets_global[di][0]):.4f} Å)"
                )

        # See exp.md for detailed rationale

        from jacobik.ik.ik_solver import JointIKSolver

        # See exp.md for detailed rationale
        for lig_obj in mol.ligands:
            decompose_ligand(mol, lig_obj)

        # Build the shared GE and scorer once on the full complex.
        # quantity_weights are baked into Scorer at construction time —
        # no per-step overhead during the solve loop.
        ge_joint = GeometryExtractor(mol, _device)
        scorer_joint = Scorer(
            mol, ge_joint, stats, device=_device, quantity_weights=quantity_weights
        )

        X_full = torch.tensor(
            np.stack([a.position for a in mol.atoms]),
            dtype=torch.float32,
            device=_device,
        )  # (N_total, 3)

        W = beam_width
        M = n_coord_samples
        B = M * W

        fks: List = []
        all_theta_inits: List[torch.Tensor] = []
        all_donor_targets: List[torch.Tensor] = []
        all_donor_pos_idxs: List[torch.Tensor] = []
        all_lig_di: List[List[int]] = []
        _fk_n_joints: List[int] = []  # cache: n_joints per ligand, built incrementally

        for li, (lig_inp, lig_obj) in enumerate(zip(self.ligands, mol.ligands)):
            log.info("Ligand %d/%d: FK + beam search...", li + 1, len(self.ligands))

            # 5a. FK for this ligand (decomposition already done above, before
            #     ge_joint/scorer_joint were constructed)
            fk = ForwardKinematics(mol, lig_obj, _device)
            fks.append(fk)
            _fk_n_joints.append(fk.n_joints)  # cache for O(1) joint-offset lookup

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

            # 5b. Donor targets for this ligand
            lig_di = [di for di, (l, g, grp) in enumerate(all_donor_groups) if l == li]
            n_d_lig = len(lig_di)
            all_lig_di.append(lig_di)

            if n_d_lig == 0:
                log.warning("Ligand %d has no donor groups — using zero theta", li)
                J = fk.n_joints
                all_theta_inits.append(torch.zeros(B, 6 + J, device=_device))
                all_donor_targets.append(torch.zeros(B, 1, 3, device=_device))
                all_donor_pos_idxs.append(torch.zeros(1, dtype=torch.long, device=_device))
                continue

            donor_targets_np = np.stack(
                [targets_global[di] for di in lig_di], axis=1
            )  # (M, n_d_lig, 3)

            if debug:
                print(f"  n_d_lig={n_d_lig}  lig_di={lig_di}")
                print(f"  donor_targets_np shape : {donor_targets_np.shape}")
                print(
                    "  target norms (sample 0): "
                    + "  ".join(
                        f"{np.linalg.norm(donor_targets_np[0, d]):.3f} Å" for d in range(n_d_lig)
                    )
                )

            # See exp.md for detailed rationale
            J = fk.n_joints
            log.info("  Ligand %d: J=%d joints, beam_width=%d", li, J, beam_width)
            ge_li = GeometryExtractor(mol, _device)
            scorer_li = Scorer(mol, ge_li, stats, device=_device, quantity_weights=quantity_weights)
            # ge_li sees the same complex but its _q5_keys covers all ligands;
            # BeamSearch needs only this ligand's joints in rows 0..J-1.
            # Compute the joint offset: sum of n_joints for ligands 0..li-1.
            _joint_offset = sum(_fk_n_joints[:li])  # O(1): cached from FK build loop
            _q5_m = scorer_li._q5_means[_joint_offset : _joint_offset + J]  # (J, K)
            _q5_k = scorer_li._q5_kappas[_joint_offset : _joint_offset + J]
            _q5_w = scorer_li._q5_weights[_joint_offset : _joint_offset + J]
            beam = BeamSearch(fk, beam_width=beam_width, device=_device)
            beam_torsions, _ = beam.search(
                X_ref=X_full,
                base_theta=torch.zeros(1, 6, device=_device),
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

            # 5e. Tile M × W → B parallel seeds
            donor_targets_torch = (
                torch.tensor(
                    donor_targets_np,
                    dtype=torch.float32,
                    device=_device,
                )
                .unsqueeze(1)
                .expand(M, W, n_d_lig, 3)
                .reshape(B, n_d_lig, 3)
            )

            beam_torsions_B = beam_torsions.unsqueeze(0).expand(M, W, J).reshape(B, J)

            # 5f. Donor position indices (indices into GE's global donor list)
            # We pass the group indices (lig_di) directly instead of atom indices,
            # so the solver can correctly fetch centroids for haptic groups.
            donor_pos_idx_t = torch.tensor(lig_di, dtype=torch.long, device=_device)

            if debug:
                print(f"  donor_pos_idx  : {lig_di}")
                _actual_all = ge_joint.donor_positions(X_full.unsqueeze(0))[0].cpu().numpy()
                _actual = _actual_all[lig_di]
                for _d, (_act, _tgt) in enumerate(zip(_actual, donor_targets_np[0])):
                    print(
                        f"    donor[{_d}] current={np.array2string(_act, precision=3)}  "
                        f"target={np.array2string(_tgt, precision=3)}  "
                        f"dist={np.linalg.norm(_act - _tgt):.3f} Å"
                    )

            # 5g. Kabsch init (per-ligand)
            base_theta = kabsch_init(
                ge=ge_joint,
                X_ref=X_full,
                donor_targets=donor_targets_torch,
                donor_pos_idx=donor_pos_idx_t,
                diversify_orientation=diversify_orientation,
                diversity_scale=diversity_scale,
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
            all_donor_targets.append(donor_targets_torch)
            all_donor_pos_idxs.append(donor_pos_idx_t)

        # ------------------------------------------------------------------
        # 5h. Joint IK solve — all ligands optimised simultaneously
        # ------------------------------------------------------------------
        log.info("Running joint IK: %d ligands × B=%d × T=%d steps", len(fks), B, n_ik_steps)

        joint_solver = JointIKSolver(
            fks=fks,
            scorer=scorer_joint,
            ge=ge_joint,
            n_steps=n_ik_steps,
            lr=lr,
            lambda_init=lambda_init,
            lse_temp=lse_temp,
            adam_betas=adam_betas,
            anchor_floor_frac=anchor_floor_frac,
            device=_device,
        )
        thetas_final, converged, scores, solve_trace = joint_solver.solve(
            theta_inits=all_theta_inits,
            X_ref=X_full,
            donor_targets=all_donor_targets,
            donor_pos_idxs=all_donor_pos_idxs,
            debug=debug,
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

        # ------------------------------------------------------------------
        # 6. Assemble full complexes from joint results
        # ------------------------------------------------------------------
        n_conformers = n_conformers if n_conformers is not None else max(beam_width, n_coord_samples)
        conformers: List[ConformerResult] = []

        score_np = scores.cpu().numpy()
        sorted_b = np.argsort(-score_np)

        for ci, b in enumerate(sorted_b[:n_conformers]):
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

        return EnsembleResult(conformers=conformers, solve_trace=solve_trace, scorer=scorer_joint)