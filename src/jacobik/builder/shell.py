"""Shell geometry helpers of the engine (split out of engine.py; engine re-exports every name): the coordination-number
table, the Q1 distance sampler for the shell, the per-conformer shallow copy of a complex, the coordination-geometry
refresh and the donor-site ordering."""
from __future__ import annotations

import copy
import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from jacobik.builder.result import LigandInput

if TYPE_CHECKING:
    from jacobik.core.types import Atom, Bond, HapticInteraction, Ligand, TMComplex

log = logging.getLogger("jacobik.builder.engine")

#: Q1 GMM modes weighing less than this fraction of the largest mode are
#: dropped before sampling (spurious ~0.045 A "bond" mode of amine donors).
_Q1_PROMINENCE_FLOOR: float = 0.10



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

# Mean M-D distances in Å by donor family.
_DEFAULT_BOND_LENGTH: float = 2.1

_BOND_LENGTH_TABLE: Dict[str, float] = {
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
    Draw ``n_samples`` M–D bond lengths from the Q1 GMM.

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

    # Normalise weights, drop low-prominence modes, re-normalise.
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
    Copy `mol` for one output conformer. Atoms, coord sites and haptic groups
    are copied per object (their positions/centroids are reassigned or updated
    in place per conformer); immutable topology metadata is shared by reference.
    """
    from jacobik.core.types import Atom

    mol_copy = copy.copy(mol)
    mol_copy.atoms = [Atom(a.idx, a.symbol, a.position, a.total_valence) for a in mol.atoms]
    mol_copy.coord_sites = [copy.copy(s) for s in getattr(mol, "coord_sites", [])]

    new_ligands = []
    for lig in mol.ligands:
        lig_copy = copy.copy(lig)
        lig_copy.haptic_groups = [copy.copy(h) for h in getattr(lig, "haptic_groups", [])]
        new_ligands.append(lig_copy)
    mol_copy.ligands = new_ligands

    return mol_copy


def _refresh_coord_geometry(mol) -> None:
    """Recompute derived coordination geometry (coord_site positions and haptic centroids) from ``mol.atoms``."""
    pos_by_idx = {a.idx: a.position for a in mol.atoms}

    def _centroid(idxs):
        pts = [pos_by_idx[i] for i in idxs if i in pos_by_idx]
        if not pts:
            return None
        return pts[0].copy() if len(pts) == 1 else np.mean(pts, axis=0)

    for site in getattr(mol, "coord_sites", []):
        if (c := _centroid(site.donor_indices)) is not None:
            site.position = c
    for lig in getattr(mol, "ligands", []):
        for h in getattr(lig, "haptic_groups", []):
            if (c := _centroid(h.donor_indices)) is not None:
                h.centroid = c
    for h in getattr(mol, "haptic_interactions", []):
        if (c := _centroid(h.donor_indices)) is not None:
            h.centroid = c


# ---------------------------------------------------------------------------
# SMILES → 3-D conformer helper
# ---------------------------------------------------------------------------


def _order_sites_like_donor_groups(mol, donor_idx_full) -> None:
    """Reorder mol.coord_sites into the solver's donor order (ligand by ligand, donor group by group).

    The classifier lists haptic sites first, but targets, ideal directions, assignments and Q1 tables are indexed by
    the donor groups in ligand order, while the GeometryExtractor reads donor positions in coord_sites order. When a
    haptic ligand was not the first ligand the two orders differed and every per-donor quantity was paired with
    another donor's target (e.g. Cp*Fe(dppe)X with the Cp* listed last: one P ended 3 A from Fe)."""
    order = [frozenset(g) for groups in donor_idx_full for g in groups]
    key = lambda s: frozenset(d for d in s.donor_indices if d != mol.metal_idx)
    by_key = {key(s): s for s in mol.coord_sites}
    if len(by_key) != len(mol.coord_sites) or set(by_key) != set(order):
        log.warning("coord sites do not match the donor groups one to one; site order left as classified")
        return
    mol.coord_sites = [by_key[k] for k in order]


def _e():
    from jacobik.builder import engine

    return engine


def _smiles_to_conformer(
    smiles: str,
    fast_embed: bool = False,
    embed_seed: Optional[int] = None,
    donors: Optional[List[List[int]]] = None,
    chelate_embed: bool = False,
    embed: Optional[str] = None,
    ez: Optional[list] = None,
    chiral: Optional[list] = None,
    strict_stereo: bool = False,
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
    embed_seed : RDKit ETKDG random seed (default 42, matching the prior
                 hardcoded behaviour). For most ligands this genuinely does
                 not matter -- IK's torsion optimisation reaches the target
                 regardless of the starting fold. It matters a great deal
                 when donor atoms sit on two or more RIGID sub-structures
                 that are connected only by bonds outside the ligand's own
                 rotatable set (or by very few of them) -- e.g. a bis-chelate
                 cage whose two donor-bearing halves are linked through a
                 rigid fused-ring backbone. There, the *relative* orientation
                 of those rigid pieces is fixed by this one embedding and
                 cannot be corrected by torsion optimisation at all, so a
                 single unlucky seed can make otherwise-achievable
                 coordination geometrically impossible for every donor-to-
                 site assignment (see ``ComplexBuilder.solve_multi_embed``).

    Returns
    -------
    rdmol      : RDKit Mol with embedded 3-D coordinates
    positions  : (N, 3) float64 array of atom positions (all atoms incl. H)
    has_h      : True if SMILES contained explicit H atoms

    With ``chelate_embed`` and the ligand's ``donors``, ligands with two or more kappa donors (or formally anionic
    donors) are embedded by ``chelate_embed.embed`` (donor-pair distance bounds + temporary H caps on anions);
    everything else, and any failure there, takes the plain path below.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    has_h = "[H]" in smiles or "[H" in smiles
    method = embed or ("etkdg" if fast_embed else _e().DEFAULT_EMBED)
    if method not in _e().EMBED_METHODS:
        raise ValueError(f"embed must be one of {_e().EMBED_METHODS}, got {method!r}")
    fast_embed = method == "etkdg"
    use_gfn2 = method == "gfn2"

    from jacobik.builder import embed as _cem

    sur = _cem.bound_surrogate(smiles, donors or []) if (_cem.BOUND_SURROGATE and donors and not has_h) else None
    if sur is not None:  # embed the BOUND ligand (cap on the donor), place it on the ligand as written
        sur_mol, rewritten = sur
        sur_h = Chem.AddHs(sur_mol)
        p0 = AllChem.ETKDGv3()
        p0.randomSeed = 42 if embed_seed is None else int(embed_seed)
        _e()._embed_etkdg_with_retries(sur_h, p0.randomSeed)
        if not fast_embed:
            AllChem.MMFFOptimizeMolecule(sur_h)
        spos = sur_h.GetConformer().GetPositions().astype(np.float64)
        if use_gfn2:
            spos = _e()._embed_refine(sur_h, spos, donors, experiment_flags=False)
        orig_h = Chem.AddHs(Chem.MolFromSmiles(smiles))
        positions = _cem.map_surrogate_positions(orig_h, sur_h, spos)
        conf = Chem.Conformer(orig_h.GetNumAtoms())
        for i, xyz in enumerate(positions):
            conf.SetAtomPosition(i, xyz.tolist())
        orig_h.AddConformer(conf, assignId=True)
        _e().log.debug("bound-surrogate embed for %s: capped donors %s", smiles, rewritten)
        return orig_h, positions, has_h

    if chelate_embed and donors and not has_h:
        from jacobik.builder import embed as _ce

        try:
            r = _ce.embed(smiles, donors, seed=42 if embed_seed is None else int(embed_seed), mmff=not fast_embed,
                          ez_extra=ez, chiral_extra=chiral, strict_stereo=strict_stereo)
        except Exception as exc:  # never let the embed variant break a solve
            _e().log.warning("chelate-aware embed failed for %s (%s); using the plain embed", smiles, exc)
            r = None
        if r is not None:
            rdmol_h, positions, info = r
            _e().log.debug("chelate-aware embed %s: %s", smiles, info)
            if use_gfn2:
                positions = _e()._embed_refine(rdmol_h, positions, donors)
            return rdmol_h, positions, has_h

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    # Add Hs for 3-D embedding (we'll remove them again for the index map
    # unless the original SMILES was H-explicit)
    mol_h = Chem.AddHs(mol)
    if ez:
        from jacobik.builder.embed import set_ez

        set_ez(mol_h, ez, strict=strict_stereo)
    if chiral:
        from jacobik.builder.embed import set_chiral

        set_chiral(mol_h, chiral)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42 if embed_seed is None else int(embed_seed)
    _e()._embed_etkdg_with_retries(mol_h, params.randomSeed)

    if not fast_embed:
        AllChem.MMFFOptimizeMolecule(mol_h)

    conf = mol_h.GetConformer()
    positions = conf.GetPositions().astype(np.float64)  # (N_all, 3) incl. H
    if use_gfn2:
        positions = _e()._embed_refine(mol_h, positions, donors)

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


def _build_complex_shell(
    metal_symbol: str,
    charge: int,
    spin: int,
    geometry: str,
    ligand_inputs: List[LigandInput],
    fast_embed: bool = False,
    embed_seed: Optional[int] = None,
    chelate_embed: bool = False,
    embed: Optional[str] = None,
) -> Tuple[object, List[object], List[List[List[int]]]]:
    """
    Build a TMComplex shell (1-based atom indices, metal at the origin as
    atom 1) from ligand ETKDG conformers. Metal-donor bonds are added here as
    placeholders; classify() fills in coord sites afterwards.

    ``embed_seed`` is forwarded to every ligand's ``_smiles_to_conformer``
    call (same seed for every ligand in one call; call this function again
    with a different seed to get an independently re-embedded shell -- see
    ``_smiles_to_conformer`` for why this matters for rigid multidentate
    ligands).

    Returns
    -------
    mol            : TMComplex (no coord_sites yet)
    rdmols         : RDKit Mol per ligand (with Hs)
    donor_idx_full : donor groups per ligand, as global 1-based atom indices
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
    donor_idx_full_out = []

    for lig_i, lig_inp in enumerate(ligand_inputs):
        rdmol_h, positions, has_h = _e()._smiles_to_conformer(
            lig_inp.smiles, fast_embed=fast_embed, embed_seed=embed_seed,
            donors=lig_inp.donors, chelate_embed=chelate_embed, embed=embed,
            ez=getattr(lig_inp, "ez_fix", None),
            chiral=getattr(lig_inp, "chiral_fix", None),
            strict_stereo=bool(getattr(lig_inp, "_strict_stereo", False)),
        )

        donor_full = _e()._donor_indices_to_full_graph(lig_inp.smiles, lig_inp.donors, has_h, rdmol_h)

        # Rigidly translate the ligand so its donor centroid sits 2.5 A from the
        # metal, then (polydentate) push outward until no donor is closer than
        # min_clash_dist (bond lengths unchanged).
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
            for _ in range(20):  # safety cap; converges in 1-3 iterations
                donor_vecs = positions[all_donor_flat] - metal_pos  # (n_d, 3)
                donor_dists = np.linalg.norm(donor_vecs, axis=1)
                min_idx = int(np.argmin(donor_dists))
                min_dist = donor_dists[min_idx]
                if min_dist >= min_clash_dist:
                    break
                push_dir = donor_vecs[min_idx]
                push_norm = np.linalg.norm(push_dir)
                if push_norm < 1e-8:
                    push_dir = target_dir  # degenerate: fall back to target
                else:
                    push_dir = push_dir / push_norm
                push = min_clash_dist - min_dist + 0.05
                positions = positions + push_dir * push

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

        # Ligand-internal bonds
        for bond in rdmol_h.GetBonds():
            i_local = bond.GetBeginAtomIdx()
            j_local = bond.GetEndAtomIdx()
            i_global = lig_atom_indices[i_local]
            j_global = lig_atom_indices[j_local]
            bt = bond.GetBondTypeAsDouble()
            bonds.append(Bond(i=i_global, j=j_global, order=float(bt)))

        # rdmol-local (0-based) -> global (1-based) atom indices
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

        # Metal-donor bonds (dative, low order)
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

    return mol, rdmols, donor_idx_full_out
