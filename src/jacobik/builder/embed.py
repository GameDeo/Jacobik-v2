"""
chelate_embed.py
================
Chelate-aware free-ligand embedding (``solve(chelate_embed=True)``, the default).

Why: the solver only moves torsions, so a ligand's bond lengths, bond angles, ring puckers and the E/Z configuration of
its double bonds are frozen at its free-ligand embed. A plain ETKDG embed knows nothing about the metal, so it can put
the donors of one ligand where no metal could bind them all (the classic case: a C=C between two donor branches
embedded trans). No donor-to-site assignment can repair that.

What: during the ligand's own embed only, add distance bounds between every pair of kappa donors of the SAME ligand
that any chelate must satisfy (no metal, no reference data):
  * every pair:                     d(Di, Dj) >= D_MIN              (two donors cannot overlap at one metal)
                                    d(Di, Dj) <= r_i + r_j          (both within their M-D length of one point)
  * pairs closing a 4/5/6 ring:     d(Di, Dj) <= law-of-cosines(r_i, r_j, BITE_MAX_DEG)   (small chelates span cis sites)
r = typical M-D length by donor element (R_MD); ring size = shortest ligand-graph path between the donors + 2.
ETKDG treats bounds as soft, so if the first embed violates them, N_TRY candidates are embedded and the one with the
smallest total violation is kept (first on ties; deterministic for a given seed). The bounds are kept as flat-bottom
restraints in the free-ligand MMFF step, which could otherwise relax them away.

Also: anionic lone-pair donors written as formal anions ([C-]=O, [N-], [O-], [S-]) are neutralised and capped with a
temporary H for the embed + MMFF only (a surrogate for the metal bond). Without the cap MMFF types a [C-]=O carbon as
isocyanide-like and straightens it to ~180 deg. Group-13 "ate" centres ([B-], [Al-], ...) are not capped: their charge
is a fourth bond, not a metal-facing lone pair.

Ligands with fewer than two kappa donors and no anionic atom use the plain embed unchanged.
"""
from __future__ import annotations

import logging
import math
import os as _os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: Typical metal-donor distance by donor element (A); only used to bound donor-donor distances.
R_MD = {"C": 2.00, "N": 2.05, "O": 2.05, "P": 2.25, "S": 2.40, "Se": 2.50, "As": 2.35,
        "F": 2.00, "Cl": 2.35, "Br": 2.50, "I": 2.70, "Si": 2.35, "B": 2.20}
R_MD_DEFAULT = 2.2
BITE_MAX_DEG = 100.0      # upper bite angle for donors closing a 4-, 5- or 6-membered chelate ring
SMALL_RING_MAX = 6
D_MIN = 2.4               # A, minimum donor-donor distance
N_TRY = 8                 # candidates embedded when the first one violates the bounds
RESTRAINT_K = 100.0       # MMFF flat-bottom restraint force constant (kcal/mol/A^2)
NO_CAP = frozenset({"B", "Al", "Ga", "In", "Tl"})
# Production resolved configuration (hardcoded values)
CARBENE_YLIDE = True
CHELATE_EZ = True
CHELATE_EZ_RING_MAX = 7


def chelate_ring_stereo(base, donor_groups: Sequence[Sequence[int]], ring_max: int = CHELATE_EZ_RING_MAX):
    """[(begin, end, stereo_atom_begin, stereo_atom_end, cis)] for every non-ring, non-aromatic double bond of ``base``
    (heavy-atom indices) that lies on the shortest path between two kappa donors of the ligand, when that path plus the
    metal forms a ring of at most ``ring_max`` atoms, and whose stereo the SMILES leaves unspecified."""
    from rdkit import Chem

    dm = Chem.GetDistanceMatrix(base)
    ds = [g[0] for g in donor_groups if len(g) == 1 and g[0] < base.GetNumAtoms()]
    out, seen = [], set()
    for x in range(len(ds)):
        for y in range(x + 1, len(ds)):
            i, j = ds[x], ds[y]
            d = dm[i, j]
            if not np.isfinite(d) or d > 1e6 or d < 2 or int(d) + 2 > ring_max:
                continue
            p = list(Chem.GetShortestPath(base, i, j))
            for k in range(len(p) - 1):
                b, c = p[k], p[k + 1]
                bd = base.GetBondBetweenAtoms(b, c)
                if (bd is None or bd.GetBondType() != Chem.BondType.DOUBLE or bd.GetIsAromatic() or bd.IsInRing()
                        or bd.GetStereo() != Chem.BondStereo.STEREONONE or bd.GetIdx() in seen):
                    continue
                nb = [a.GetIdx() for a in base.GetAtomWithIdx(b).GetNeighbors() if a.GetIdx() != c]
                nc = [a.GetIdx() for a in base.GetAtomWithIdx(c).GetNeighbors() if a.GetIdx() != b]
                if not nb or not nc:
                    continue   # a terminal end (e.g. =CH2, =O): no configuration to fix
                rb = p[k - 1] if k > 0 else None             # ring neighbour of b (None: the metal)
                rc = p[k + 2] if k + 2 < len(p) else None
                a = rb if rb is not None else nb[0]
                e = rc if rc is not None else nc[0]
                seen.add(bd.GetIdx())
                if bd.GetBeginAtomIdx() == b:
                    out.append((b, c, a, e, (rb is not None) == (rc is not None)))
                else:
                    out.append((c, b, e, a, (rb is not None) == (rc is not None)))
    return out


# Production resolved configuration (hardcoded values)
EZ_ENUMERATE = True
EZ_MAX_BONDS = 3


def undecided_ez(base, donor_groups: Sequence[Sequence[int]]) -> List[Tuple[int, int, int, int]]:
    """[(begin, end, stereo_atom_begin, stereo_atom_end)] of the non-ring double bonds of ``base`` (heavy-atom indices)
    with an unspecified configuration that chelate_ring_stereo does not fix."""
    from rdkit import Chem

    fixed = {base.GetBondBetweenAtoms(b, c).GetIdx() for b, c, *_ in chelate_ring_stereo(base, donor_groups)}
    out = []
    for si in Chem.FindPotentialStereo(base):
        if (si.type != Chem.StereoType.Bond_Double or si.specified != Chem.StereoSpecified.Unspecified
                or si.centeredOn in fixed):
            continue
        bd = base.GetBondWithIdx(si.centeredOn)
        if bd.IsInRing() or bd.GetIsAromatic():
            continue
        b, c = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        nb = [x.GetIdx() for x in bd.GetBeginAtom().GetNeighbors() if x.GetIdx() != c]
        nc = [x.GetIdx() for x in bd.GetEndAtom().GetNeighbors() if x.GetIdx() != b]
        if nb and nc:
            out.append((b, c, min(nb), min(nc)))
    return out


def ez_variants(smiles: str, donor_groups: Sequence[Sequence[int]]) -> List[List[Tuple[int, int, int, int, bool]]]:
    """Every E/Z combination of the undecided bonds of one ligand: [[(b, c, a, e, cis), ...], ...]; [[]] if none (or
    more than EZ_MAX_BONDS: the embed then decides, as without enumeration)."""
    import itertools

    from rdkit import Chem

    base = Chem.MolFromSmiles(smiles)
    und = undecided_ez(base, donor_groups) if base is not None else []
    if not und or len(und) > EZ_MAX_BONDS:
        return [[]]
    return [[(b, c, a, e, cis) for (b, c, a, e), cis in zip(und, combo)]
            for combo in itertools.product((False, True), repeat=len(und))]


# Production resolved configuration (hardcoded True)
BOUND_SURROGATE = True
SURROGATE_DONORS = frozenset({"B", "Al", "Ga"})


def bound_surrogate(smiles: str, donor_groups: Sequence[Sequence[int]]):
    """(capped Mol with the cap as an explicit H on each rewritten donor, [donor indices]) or None when no donor of the
    ligand is a group-13 atom double-bonded to a positively charged neighbour."""
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    rw = Chem.RWMol(m)
    done = []
    for g in donor_groups:
        if len(g) != 1:
            continue
        a = rw.GetAtomWithIdx(int(g[0]))
        if a.GetSymbol() not in SURROGATE_DONORS:
            continue
        for bd in a.GetBonds():
            o = bd.GetOtherAtom(a)
            if bd.GetBondType() == Chem.BondType.DOUBLE and o.GetFormalCharge() > 0:
                bd.SetBondType(Chem.BondType.SINGLE)
                o.SetFormalCharge(o.GetFormalCharge() - 1)
                o.SetNoImplicit(False)
                a.SetNumExplicitHs(a.GetNumExplicitHs() + 1)
                a.SetNoImplicit(True)
                done.append(a.GetIdx())
                break
    if not done:
        return None
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out, done


def map_surrogate_positions(orig_h, sur_h, sur_pos):
    """Positions of the surrogate's atoms placed on the original AddHs mol: heavy atoms share indices; each heavy
    atom's H are matched in index order (the cap H on a rewritten donor is the one left over and dropped)."""
    import numpy as np

    n_heavy = sum(1 for a in orig_h.GetAtoms() if a.GetAtomicNum() > 1)
    out = np.zeros((orig_h.GetNumAtoms(), 3))
    out[:n_heavy] = sur_pos[:n_heavy]
    for a in orig_h.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        ho = sorted(n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() == 1)
        hs = sorted(n.GetIdx() for n in sur_h.GetAtomWithIdx(a.GetIdx()).GetNeighbors() if n.GetAtomicNum() == 1)
        for i, j in zip(ho, hs):
            out[i] = sur_pos[j]
    return out


def set_ez(mh, fixes, *, strict: bool = False) -> int:
    """Set bond stereo (b, c, a, e, cis) on mol ``mh``; returns how many applied."""
    from rdkit import Chem

    if not fixes:
        return 0

    if not strict:
        n = 0
        for b, c, a, e, cis in fixes:
            bd = mh.GetBondBetweenAtoms(int(b), int(c))
            if bd is None or bd.GetBondType() != Chem.BondType.DOUBLE:
                log.debug("set_ez: skipping non-double or missing bond (%s, %s)", b, c)
                continue
            bd.SetStereoAtoms(int(a), int(e))
            bd.SetStereo(Chem.BondStereo.STEREOCIS if cis else Chem.BondStereo.STEREOTRANS)
            n += 1
        return n

    n_applied = 0
    for fix in fixes:
        if len(fix) == 5:
            b, c, a, e, is_z = fix
        else:
            raise ValueError(f"set_ez fix tuple must have length 5: {fix}")

        b = int(b)
        c = int(c)
        bd = mh.GetBondBetweenAtoms(b, c)
        if bd is None:
            raise ValueError(f"set_ez: no bond between atom {b} and atom {c}")

        if bd.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError(f"set_ez: bond ({b}, {c}) has type {bd.GetBondType()}, expected DOUBLE")

        # Resolve reference neighbors
        a = int(a)
        if a < 0 or a == b:
            b_nbrs = [n.GetIdx() for n in mh.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != c]
            if not b_nbrs:
                raise ValueError(f"set_ez: atom {b} has no reference neighbor != {c}")
            a = b_nbrs[0]

        e = int(e)
        if e < 0 or e == c:
            c_nbrs = [n.GetIdx() for n in mh.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() != b]
            if not c_nbrs:
                raise ValueError(f"set_ez: atom {c} has no reference neighbor != {b}")
            e = c_nbrs[0]

        # Verify connectivity
        if mh.GetBondBetweenAtoms(b, a) is None:
            raise ValueError(f"set_ez: atom a={a} is not connected to begin atom b={b}")
        if mh.GetBondBetweenAtoms(c, e) is None:
            raise ValueError(f"set_ez: atom e={e} is not connected to end atom c={c}")

        if bd.GetBeginAtomIdx() == b:
            bd.SetStereoAtoms(a, e)
        elif bd.GetBeginAtomIdx() == c:
            bd.SetStereoAtoms(e, a)
        else:
            raise ValueError(f"set_ez: bond between {b} and {c} has begin={bd.GetBeginAtomIdx()}, end={bd.GetEndAtomIdx()}")

        bd.SetStereo(Chem.BondStereo.STEREOZ if is_z else Chem.BondStereo.STEREOE)
        n_applied += 1

    if n_applied != len(fixes):
        raise ValueError(f"set_ez applied {n_applied} of {len(fixes)} fixes")
    return n_applied


def set_chiral(mh, fixes) -> int:
    """Set atom chiral tag (atom_idx, tag) on mol mh; returns how many applied."""
    from rdkit import Chem

    if not fixes:
        return 0

    n_applied = 0
    for fix in fixes:
        if len(fix) != 2:
            raise ValueError(f"set_chiral fix tuple must have length 2: {fix}")
        atom_idx, tag = fix
        atom_idx = int(atom_idx)
        if atom_idx < 0 or atom_idx >= mh.GetNumAtoms():
            raise ValueError(f"set_chiral: atom index {atom_idx} out of range (num atoms: {mh.GetNumAtoms()})")
        at = mh.GetAtomWithIdx(atom_idx)
        if at.GetDegree() < 3:
            raise ValueError(f"set_chiral: atom {atom_idx} has degree {at.GetDegree()} < 3")
        at.SetChiralTag(tag)
        n_applied += 1

    if n_applied != len(fixes):
        raise ValueError(f"set_chiral applied {n_applied} of {len(fixes)} fixes")
    return n_applied


def donor_pair_bounds(base, donor_groups: Sequence[Sequence[int]]) -> List[Tuple[int, int, float, float]]:
    """[(i, j, lo, hi)] for pairs of single-atom (kappa) donors of one ligand (heavy-atom indices of ``base``)."""
    from rdkit import Chem

    dm = Chem.GetDistanceMatrix(base)
    ds = [g[0] for g in donor_groups if len(g) == 1 and g[0] < base.GetNumAtoms()]
    out = []
    for a in range(len(ds)):
        for b in range(a + 1, len(ds)):
            i, j = ds[a], ds[b]
            path = dm[i, j]
            if not np.isfinite(path) or path > 1e6 or path < 2:
                continue  # unreachable, or directly bonded (one eta2 site given as two kappa donors)
            ri = R_MD.get(base.GetAtomWithIdx(i).GetSymbol(), R_MD_DEFAULT)
            rj = R_MD.get(base.GetAtomWithIdx(j).GetSymbol(), R_MD_DEFAULT)
            hi = ri + rj
            if int(path) + 2 <= SMALL_RING_MAX:
                hi = math.sqrt(ri * ri + rj * rj - 2 * ri * rj * math.cos(math.radians(BITE_MAX_DEG)))
            out.append((i, j, D_MIN, hi))
    return out


def _cap_keeps_aromaticity(base, i: int) -> bool:
    """True if neutralising aromatic atom i and adding one H keeps every atom of its rings aromatic
    (true for an aryl anion [c-] in benzene, false for cyclopentadienide [cH-])."""
    from rdkit import Chem

    rw = Chem.RWMol(base)
    a = rw.GetAtomWithIdx(i)
    a.SetFormalCharge(0)
    a.SetNoImplicit(True)
    h = rw.AddAtom(Chem.Atom(1))
    rw.AddBond(i, h, Chem.BondType.SINGLE)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return False
    ri = base.GetRingInfo()
    ring_atoms = {j for r in ri.AtomRings() if i in r and all(base.GetAtomWithIdx(q).GetIsAromatic() for q in r) for j in r}
    return all(m.GetAtomWithIdx(j).GetIsAromatic() for j in ring_atoms)


def cap_anions(base, donor_groups: Sequence[Sequence[int]] = ()):
    """Valence-completing caps for the embed + MMFF only: (capped mol, cap atom indices). Caps come after the heavy
    atoms of ``base``; the caller drops them (and their H) afterwards. Pure valence geometry, no metal, no data.

    * A kappa donor atom written with missing valence (a formal anion and/or radical electrons: [O-], [N-], [CH3-], [C-]#[O+],
      carbene [C], alkylidene [CH], alkylidyne [C-3]) stands for a metal bond of order
      n = radicals + max(0, -charge), capped at 3. It is neutralised and bonded to a cap with that order: H (n = 1,
      tetrahedral / trigonal as its other bonds dictate), =CH2 (n = 2, a trigonal carbene / alkylidene carbon), #CH
      (n = 3, a linear alkylidyne). Without it ETKDG/MMFF embed the free fragment: [CH] comes out linear (the metal
      then has to bind at ~90 deg), [CH3-] can come out flat, [C-]=O straightens to ~180 deg.
      Group-13 "ate" centres ([B-], [Al-], ...) are skipped (their charge is a fourth bond) and so are bare ions with
      no neighbour at all ([Cl-], [S-2]), whose geometry does not matter.
    * Two directly bonded kappa donors of one ligand, or the two atoms of an eta2 site (alkene, alkyne), get one
      shared CH2 cap, and their bond order drops by one: the metallacyclopropene / -propane surrogate bends the
      substituents back as in the coordinated ligand.
    """
    from rdkit import Chem

    rw = Chem.RWMol(base)
    caps = []
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    # eta2 sites are not capped.
    nmax = 1
    haptic = {i for g in donor_groups if len(g) > nmax for i in g}
    kappa_donors = {i for g in donor_groups if len(g) <= nmax for i in g}
    for a in base.GetAtoms():
        # Only metal-bound (kappa donor) atoms: a charge or radical elsewhere is not a metal bond (e.g. the free O- of a
        # kappa1 carboxylate, a genuine radical ligand).
        if a.GetIdx() not in kappa_donors or a.GetSymbol() in NO_CAP or a.GetIdx() in haptic:
            continue
        if a.GetIsAromatic() and not _cap_keeps_aromaticity(base, a.GetIdx()):
            # e.g. Cp- / Cp*- written [cH-] / [c-]: its charge belongs to the pi system, and an H cap would make that
            # carbon sp3 and tilt its H or methyl out of the ring plane.
            continue
        n = a.GetNumRadicalElectrons() + max(0, -a.GetFormalCharge())
        if n <= 0 or (a.GetDegree() == 0 and a.GetTotalNumHs() == 0):
            continue
        n = min(n, 3)
        ylide = False
        if CARBENE_YLIDE and n == 2 and a.GetNumRadicalElectrons() == 2 and a.GetFormalCharge() == 0:
            # Carbene next to a lone-pair heteroatom (NHC, CAAC, Fischer): embed its ylidic / iminium form, X+=C with a
            # single-bonded cap, so the C-X bond gets its partial double-bond length (C-N ~1.31 A, not 1.41 A).
            het = next((x for x in a.GetNeighbors() if x.GetSymbol() in ("N", "O", "S") and x.GetFormalCharge() == 0
                        and not x.GetIsAromatic() and all(bd.GetBondType() == Chem.BondType.SINGLE for bd in x.GetBonds())), None)
            if het is not None:
                rw.GetBondBetweenAtoms(a.GetIdx(), het.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
                hx = rw.GetAtomWithIdx(het.GetIdx())
                hx.SetFormalCharge(1)
                hx.SetNoImplicit(True)
                n, ylide = 1, True
        at = rw.GetAtomWithIdx(a.GetIdx())
        at.SetFormalCharge(min(0, a.GetFormalCharge() + n) if a.GetFormalCharge() < 0 else a.GetFormalCharge())
        at.SetNumRadicalElectrons(max(0, a.GetNumRadicalElectrons() - max(0, n - max(0, -a.GetFormalCharge()))))
        at.SetNoImplicit(True)
        if ylide:
            at.SetNumRadicalElectrons(0)
        c = rw.AddAtom(Chem.Atom(1 if n == 1 else 6))
        rw.AddBond(a.GetIdx(), c, order[n])
        caps.append(c)
    kappa = [i for g in donor_groups if len(g) <= nmax for i in g if i < base.GetNumAtoms()]
    for x in range(len(kappa)):
        for y in range(x + 1, len(kappa)):
            b = rw.GetBondBetweenAtoms(kappa[x], kappa[y])
            if b is None or b.GetIsAromatic() or b.GetBondType() not in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
                continue
            if base.GetBondBetweenAtoms(kappa[x], kappa[y]).IsInRing():
                continue  # ring alkene (COD, NBD): the ring holds its substituents; a bridge would strain the ring
            b.SetBondType(Chem.BondType.SINGLE if b.GetBondType() == Chem.BondType.DOUBLE else Chem.BondType.DOUBLE)
            c = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(kappa[x], c, Chem.BondType.SINGLE)
            rw.AddBond(kappa[y], c, Chem.BondType.SINGLE)
            caps.append(c)
    if not caps:
        return Chem.Mol(base), []
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception as exc:
        log.debug("valence capping failed to sanitise (%s); embedding uncapped", exc)
        return Chem.Mol(base), []
    return m, caps


def _violation(X, pairs) -> float:
    v = 0.0
    for i, j, lo, hi in pairs:
        d = float(np.linalg.norm(X[i] - X[j]))
        v += max(0.0, d - hi) + max(0.0, lo - d)
    return v


#: process cache of embed() results (deterministic in its arguments; every configuration class of a request re-embedded
#: the same unchanged ligands: FUTSOD 51 calls, 10 s). Values are copied in and out.
_EMBED_CACHE: dict = {}


def embed(smiles: str, donor_groups: Sequence[Sequence[int]], seed: int = 42, mmff: bool = True, ez_extra=None, chiral_extra=None, *, strict_stereo: bool = False):
    """Cached _embed (identical result for identical arguments; the caller gets its own copy)."""
    from rdkit import Chem

    key = (smiles, tuple(tuple(int(a) for a in g) for g in donor_groups), int(seed), bool(mmff), repr(ez_extra),
           repr(chiral_extra), bool(strict_stereo))
    if key not in _EMBED_CACHE:
        r = _embed(smiles, donor_groups, seed, mmff, ez_extra, chiral_extra, strict_stereo=strict_stereo)
        _EMBED_CACHE[key] = None if r is None else (Chem.Mol(r[0]), np.array(r[1], copy=True), dict(r[2]))
    hit = _EMBED_CACHE[key]
    if hit is None:
        return None
    return Chem.Mol(hit[0]), hit[1].copy(), dict(hit[2])


def _embed(smiles: str, donor_groups: Sequence[Sequence[int]], seed: int = 42, mmff: bool = True, ez_extra=None, chiral_extra=None, *, strict_stereo: bool = False):
    """
    Chelate-aware embed of one ligand. Returns (rdmol_h, positions (N, 3), info) in the engine's atom order,
    AddHs(MolFromSmiles(smiles)), or None when the ligand needs no special treatment (the caller then uses the
    plain embed). ``donor_groups`` use the engine's indices (heavy-atom indices = full-graph indices here).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom
    from rdkit.DistanceGeometry import DoTriangleSmoothing

    base = Chem.MolFromSmiles(smiles)
    if base is None:
        return None
    pairs = donor_pair_bounds(base, donor_groups)
    capped, caps = cap_anions(base, donor_groups)
    if not pairs and not caps and not ez_extra and not chiral_extra:
        return None
    mh = Chem.RWMol(Chem.AddHs(capped))
    info = {"pairs": len(pairs), "caps": len(caps), "bounded": False, "violation": None, "n_embedded": 0, "ez": 0, "chiral": 0}
    if CHELATE_EZ:
        for b, c, a, e, cis in chelate_ring_stereo(base, donor_groups):
            bd = mh.GetBondBetweenAtoms(b, c)
            if bd is None or bd.GetBondType() != Chem.BondType.DOUBLE:
                continue  # the valence caps changed this bond (directly bonded donor pair)
            bd.SetStereoAtoms(a, e)
            bd.SetStereo(Chem.BondStereo.STEREOCIS if cis else Chem.BondStereo.STEREOTRANS)
            info["ez"] += 1
    info["ez"] += set_ez(mh, ez_extra, strict=strict_stereo)  # E/Z enumeration variant (EZ_ENUMERATE)
    info["chiral"] += set_chiral(mh, chiral_extra)  # Chiral centre variants
    ok = False
    if pairs:
        bm = rdDistGeom.GetMoleculeBoundsMatrix(mh)
        for i, j, lo, hi in pairs:
            a, b = max(i, j), min(i, j)   # upper bound in bm[min, max], lower bound in bm[max, min]
            bm[b, a] = min(bm[b, a], hi)
            bm[a, b] = max(bm[a, b], lo)
            if bm[a, b] > bm[b, a]:
                bm[a, b] = bm[b, a]
        if DoTriangleSmoothing(bm):
            def params():
                p = AllChem.ETKDGv3()
                p.randomSeed = int(seed)
                p.SetBoundsMat(bm)
                p.useRandomCoords = True
                return p

            # First candidate alone; it is identical to candidate 0 of a multi-conformer embed with the same seed,
            # so the early exit never changes which conformer is kept.
            cid = AllChem.EmbedMolecule(mh, params())
            if cid != -1:
                v0 = _violation(mh.GetConformer().GetPositions(), pairs)
                info["n_embedded"] = 1
                ok = True
                if v0 > 0.0:
                    trial = Chem.Mol(mh)
                    trial.RemoveAllConformers()
                    cids = list(AllChem.EmbedMultipleConfs(trial, N_TRY, params()))
                    if cids:
                        v = [_violation(trial.GetConformer(c).GetPositions(), pairs) for c in cids]
                        best = cids[int(np.argmin(v))]
                        keep = Chem.Conformer(trial.GetConformer(best))
                        mh.RemoveAllConformers()
                        mh.AddConformer(keep, assignId=True)
                        v0 = min(v)
                        info["n_embedded"] = 1 + len(cids)
                info["violation"] = round(float(v0), 4)
                info["bounded"] = True
    if not ok:
        p = AllChem.ETKDGv3()
        p.randomSeed = int(seed)
        cid = AllChem.EmbedMolecule(mh, p)
        if cid == -1:
            failures = []
            for retry_seed in range(int(seed) + 1, int(seed) + 6):
                mh.RemoveAllConformers()
                retry = AllChem.ETKDGv3()
                retry.randomSeed = retry_seed
                retry.useRandomCoords = True
                cid = AllChem.EmbedMolecule(mh, retry)
                if cid >= 0:
                    log.warning("ETKDGv3 failed; deterministic random-coordinate retry succeeded (seed=%d)", retry_seed)
                    break
                failures.append(f"seed {retry_seed}: {cid}")
            if cid < 0:
                log.warning("ETKDGv3 failed; deterministic retries exhausted (%s)", ", ".join(failures))
                return None
    used = pairs if ok else []
    if mmff and AllChem.MMFFHasAllMoleculeParams(mh):
        ff = AllChem.MMFFGetMoleculeForceField(mh, AllChem.MMFFGetMoleculeProperties(mh))
        for i, j, lo, hi in used:
            ff.MMFFAddDistanceConstraint(i, j, False, lo, hi, RESTRAINT_K)
        ff.Minimize(maxIts=2000)
    elif mmff:
        AllChem.MMFFOptimizeMolecule(mh)  # same as the plain path: a no-op without parameters (e.g. boron)

    # Back to AddHs(MolFromSmiles(smiles)) atom order, caps dropped. Heavy atoms keep their indices; each
    # reference H takes the position of an H on the same parent.
    ref_h = Chem.AddHs(base)
    P = mh.GetConformer().GetPositions()
    n_heavy = base.GetNumAtoms()
    out = np.zeros((ref_h.GetNumAtoms(), 3))
    out[:n_heavy] = P[:n_heavy]
    cap_set = set(caps)
    pool = {}
    for a in mh.GetAtoms():
        if a.GetAtomicNum() == 1 and a.GetIdx() not in cap_set and a.GetIdx() >= n_heavy:
            par = a.GetNeighbors()[0].GetIdx()
            if par < n_heavy:  # H on a cap atom is dropped with the cap
                pool.setdefault(par, []).append(a.GetIdx())
    for a in ref_h.GetAtoms():
        if a.GetIdx() >= n_heavy:
            out[a.GetIdx()] = P[pool[a.GetNeighbors()[0].GetIdx()].pop(0)]
    conf = Chem.Conformer(ref_h.GetNumAtoms())
    for i, xyz in enumerate(out):
        conf.SetAtomPosition(i, xyz.tolist())
    ref_h.AddConformer(conf, assignId=True)
    if strict_stereo:
        if ez_extra:
            set_ez(ref_h, ez_extra, strict=True)
        if chiral_extra:
            ref_chiral = [(i, t) for i, t in chiral_extra if ref_h.GetAtomWithIdx(int(i)).GetDegree() >= 3]
            if ref_chiral:
                set_chiral(ref_h, ref_chiral)
    return ref_h, out.astype(np.float64), info


def _e():
    from jacobik.builder import engine

    return engine


def _bond_refinement_violations(rdmol_h, before: np.ndarray, after: np.ndarray) -> List[dict]:
    """Describe ligand-bond stretches that exceed the one-sided F-4 guard."""
    from rdkit import Chem

    periodic = Chem.GetPeriodicTable()
    issues = []
    for bond in rdmol_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        d0 = float(np.linalg.norm(before[i] - before[j]))
        d1 = float(np.linalg.norm(after[i] - after[j]))
        rsum = float(periodic.GetRcovalent(rdmol_h.GetAtomWithIdx(i).GetAtomicNum()) +
                     periodic.GetRcovalent(rdmol_h.GetAtomWithIdx(j).GetAtomicNum()))
        limit = 1.25 * rsum
        if d1 - d0 > 0.25 or (d1 > limit and d0 <= limit):
            issues.append({"atoms": [i, j], "before_A": d0, "after_A": d1,
                           "delta_A": d1 - d0, "covalent_limit_A": limit})
    return issues


#: (graph, stereo, seed) -> embedded positions or the failure message of _embed_etkdg_with_retries. ETKDG with a fixed
#: seed is deterministic, so a hit gives the same conformer; the same ligand is embedded again for every shell build
#: (MEBHEK: a ring C#S typing artefact makes each embed fail three times first, 27 s per call, 2026-11-20).
_ETKDG_CACHE: dict = {}
_ETKDG_CACHE_MAX = 512


def _etkdg_key(mol_h, seed: int):
    atoms = tuple((a.GetAtomicNum(), a.GetFormalCharge(), int(a.GetChiralTag()), a.GetNumExplicitHs(),
                   a.GetNumRadicalElectrons(), a.GetIsotope()) for a in mol_h.GetAtoms())
    bonds = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondType()), int(b.GetStereo()),
                   tuple(b.GetStereoAtoms())) for b in mol_h.GetBonds())
    return atoms, bonds, int(seed)


def _embed_etkdg_with_retries(mol_h, seed: int) -> None:
    """Embed with the historical deterministic attempt, then bounded random-coordinate retries (memoised)."""
    from rdkit import Chem

    try:
        key = _etkdg_key(mol_h, seed)
    except Exception:  # noqa: BLE001 - an unkeyable molecule just embeds
        key = None
    hit = _ETKDG_CACHE.get(key) if key is not None else None
    if hit is not None:
        if isinstance(hit, str):
            raise _e()._LigandEmbedFailure(hit)
        mol_h.RemoveAllConformers()
        conf = Chem.Conformer(mol_h.GetNumAtoms())
        for i, xyz in enumerate(hit):
            conf.SetAtomPosition(i, xyz.tolist())
        mol_h.AddConformer(conf, assignId=True)
        return
    try:
        _embed_etkdg_uncached(mol_h, seed)
    except _e()._LigandEmbedFailure as exc:
        if key is not None:
            _etkdg_store(key, str(exc))
        raise
    if key is not None and mol_h.GetNumConformers():
        _etkdg_store(key, np.array(mol_h.GetConformer().GetPositions(), dtype=np.float64, copy=True))


def _etkdg_store(key, value) -> None:
    if len(_ETKDG_CACHE) >= _ETKDG_CACHE_MAX:
        _ETKDG_CACHE.pop(next(iter(_ETKDG_CACHE)))
    _ETKDG_CACHE[key] = value


def _embed_etkdg_uncached(mol_h, seed: int) -> None:
    from rdkit.Chem import AllChem

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol_h, params) >= 0:
        return
    failures = []
    for retry_seed in range(int(seed) + 1, int(seed) + 6):
        mol_h.RemoveAllConformers()
        retry = AllChem.ETKDGv3()
        retry.randomSeed = retry_seed
        retry.useRandomCoords = True
        code = AllChem.EmbedMolecule(mol_h, retry)
        if code >= 0:
            _e().log.warning("ETKDG random-coordinate retry succeeded (seed=%d)", retry_seed)
            return
        failures.append(f"seed {retry_seed}: {code}")
    raise _e()._LigandEmbedFailure(
        f"ETKDG failed for seed {seed}; random-coordinate retries exhausted ({', '.join(failures)})"
    )


def pop_embed_bond_check_events() -> List[dict]:
    """Return and clear diagnostics for the active embed_bond_check arm."""
    events = list(_e()._EMBED_BOND_CHECK_EVENTS)
    _e()._EMBED_BOND_CHECK_EVENTS.clear()
    return events


def _embed_refine(rdmol_h, positions: np.ndarray, donors=None, *, experiment_flags=True) -> np.ndarray:
    """The embed refinement: GFN2 (production) or, with profile field embed_refine, the batched MACE refinement
    (builder.mace_embed)."""
    mode = _e().EMBED_REFINE
    if mode:
        from jacobik.builder import mace_embed

        return mace_embed.refine(rdmol_h, positions, donors, mode=mode, experiment_flags=experiment_flags)
    return _e()._gfn2_refine(rdmol_h, positions, donors, experiment_flags=experiment_flags)


def _li_cap_positions(rdmol_h, positions, donors, use_li_cap, use_haptic_li_cap, experiment_flags) -> list:
    """Temporary Li counter-ion positions of the embed refinement (embed_li_cap, embed_haptic_li_cap, li_cap_spread):
    1.9 A from each locally anionic kappa donor along its lone pair, and above each eligible anionic haptic face."""
    kappa_donors = sorted({int(g[0]) for g in (donors or []) if len(g) == 1})
    def local_group_charge(atom_idx):
        atom = rdmol_h.GetAtomWithIdx(atom_idx)
        return int(atom.GetFormalCharge()) + sum(int(n.GetFormalCharge()) for n in atom.GetNeighbors())

    # Formal charge on a single donor is insufficient: zwitterionic donors such as
    # [C-]#[O+] in CO are locally neutral and must not receive a Li counter-ion.
    li_donors = sorted(a for a in kappa_donors
                       if use_li_cap and local_group_charge(a) < 0)
    haptic_li_groups = []
    seen_haptic_groups = set()
    if use_haptic_li_cap:
        try:
            for group in (donors or []):
                group = sorted({int(a) for a in group})
                group_key = tuple(group)
                if (len(group) < 2 or group_key in seen_haptic_groups
                        or sum(int(rdmol_h.GetAtomWithIdx(a).GetFormalCharge()) for a in group) >= 0):
                    continue
                local_atoms = set(group) | {n.GetIdx() for a in group
                                            for n in rdmol_h.GetAtomWithIdx(a).GetNeighbors()}
                if sum(rdmol_h.GetAtomWithIdx(a).GetFormalCharge() for a in local_atoms) >= 0:
                    continue  # locally neutral zwitterion, as for the kappa cap
                charged = [a for a in group if rdmol_h.GetAtomWithIdx(a).GetFormalCharge() < 0]
                if charged and all(rdmol_h.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in charged):
                    # Localized heteroatom charge uses the already established
                    # kappa lone-pair placement, rather than a carbon pi cap.
                    li_donors.extend(a for a in charged if a not in li_donors and local_group_charge(a) < 0)
                    seen_haptic_groups.add(group_key)
                    continue
                seen_haptic_groups.add(group_key)
                haptic_li_groups.append(group)
        except Exception as exc:
            _e().log.warning("haptic Li-cap group inspection failed (%s); refining without haptic caps", exc)
            haptic_li_groups = []
    li_positions = []
    li_donors.sort()
    for atom_idx in li_donors:
        atom = rdmol_h.GetAtomWithIdx(atom_idx)
        neighbor_vecs = []
        for neighbor in atom.GetNeighbors():
            # Hydrogens do not define the open-valence direction for a donor.
            if neighbor.GetAtomicNum() == 1:
                continue
            vec = np.asarray(positions[neighbor.GetIdx()] - positions[atom_idx], dtype=np.float64)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-10:
                neighbor_vecs.append(vec / norm)
        if neighbor_vecs:
            direction = -np.sum(neighbor_vecs, axis=0)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-10:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction /= norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        li_positions.append(np.asarray(positions[atom_idx], dtype=np.float64) + 1.9 * direction)
    if li_positions and experiment_flags and _e().LI_CAP_SPREAD:
        li_positions = _spread_li_caps(np.asarray(positions, dtype=np.float64), li_donors, li_positions)
    heavy = [a.GetIdx() for a in rdmol_h.GetAtoms() if a.GetAtomicNum() > 1]
    haptic_li_positions = []
    try:
        for group in haptic_li_groups:
            centroid = np.asarray(positions[group], dtype=np.float64).mean(axis=0)
            outside = [i for i in heavy if i not in set(group)]
            if len(group) >= 3:
                # For a planar haptic group the open coordination face is normal to its
                # plane; a centroid-minus-ligand-centroid vector can lie in the plane.
                centered = np.asarray(positions[group], dtype=np.float64) - centroid
                _, _, vt = np.linalg.svd(centered, full_matrices=True)
                direction = vt[-1]
                if outside:
                    outward = centroid - np.asarray(positions[outside], dtype=np.float64).mean(axis=0)
                    if float(np.dot(direction, outward)) < 0:
                        direction = -direction
                else:
                    # A free ligand has no metal-side label. Choose a stable side in
                    # its embedded frame; either face is symmetry-equivalent here.
                    nz = np.flatnonzero(np.abs(direction) > 1e-10)
                    if len(nz) and direction[nz[0]] < 0:
                        direction = -direction
            else:
                direction = (centroid - np.asarray(positions[outside], dtype=np.float64).mean(axis=0)
                             if outside else np.array([1.0, 0.0, 0.0]))
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-10:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / norm
            haptic_li_positions.append(centroid + 1.9 * direction)
    except Exception as exc:
        _e().log.warning("haptic Li-cap placement failed (%s); refining without haptic caps", exc)
        haptic_li_positions = []
    li_positions.extend(haptic_li_positions)
    return li_positions


def _gfn2_refine(rdmol_h, positions: np.ndarray, donors=None, *, experiment_flags=True) -> np.ndarray:
    """embed="gfn2" refinement (the default, DEFAULT_EMBED): GFN2-xTB optimisation (--opt crude) of the free ligand from its ETKDG/MMFF
    embed, once per ligand (cached). Only the frozen internals IK starts from change. The tmQM references are GFN2
    geometries, and GFN2 has parameters for every element (no silent no-op as with MMFF on boron). The donor set's
    own geometry (every distance among donor atoms, incl. within a haptic face) is held at the embed's values: that is
    what binding imposes (chelate bite, macrocycle cavity, bound diene face); GFN2 relaxes everything else. Any failure
    keeps the MMFF geometry."""
    import subprocess
    import tempfile

    from rdkit import Chem

    dn = sorted({int(a) for g in (donors or []) for a in g})
    use_bond_check = bool(experiment_flags and _e().EMBED_BOND_CHECK)
    use_li_cap = bool(experiment_flags and _e().EMBED_LI_CAP)
    use_haptic_li_cap = bool(experiment_flags and _e().EMBED_HAPTIC_LI_CAP)
    # the start geometry is part of the key (2026-10-16): keyed on SMILES alone, every embed_seed after the first got
    # the first seed's GFN2 geometry back, so solve_multi_embed returned one embed n times under the default embed
    key = (Chem.MolToSmiles(rdmol_h), positions.shape[0], tuple(dn), use_bond_check, use_li_cap, use_haptic_li_cap,
           (_e().EMBED_GFN2_OPT if experiment_flags else None), bool(experiment_flags and _e().LI_CAP_SPREAD),
           hash(np.round(np.asarray(positions, dtype=np.float64), 4).tobytes()))
    if key in _e()._GFN2_CACHE:
        if use_bond_check:
            detail = _e()._GFN2_BOND_CHECK_CACHE.get(key, {"triggered": False, "bonds": []})
            _e()._EMBED_BOND_CHECK_EVENTS.append(dict(smiles=Chem.MolToSmiles(rdmol_h), **detail))
        return _e()._GFN2_CACHE[key].copy()
    sy = [a.GetSymbol() for a in rdmol_h.GetAtoms()]
    n_ligand_atoms = len(sy)
    q_ligand = Chem.GetFormalCharge(rdmol_h)
    li_positions = _li_cap_positions(rdmol_h, positions, donors, use_li_cap, use_haptic_li_cap, experiment_flags)
    if li_positions:
        sy.extend(["Li"] * len(li_positions))
    q = q_ligand + len(li_positions)
    uhf = (sum(a.GetAtomicNum() for a in rdmol_h.GetAtoms()) + 3 * len(li_positions) - q) % 2
    out = positions
    fallback_used = "none"
    try:
        with tempfile.TemporaryDirectory() as td:
            xyz_in = np.vstack([positions, li_positions]) if li_positions else np.asarray(positions)

            def refine(start, constrained_bonds=()):
                with open(f"{td}/l.xyz", "w") as f:
                    f.write(f"{len(sy)}\n\n" + "".join(
                        f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(sy, start)))
                Path(td, "xtbopt.xyz").unlink(missing_ok=True)
                level = (_e().EMBED_GFN2_OPT or "crude") if experiment_flags else "crude"
                args = [_e().XTB_BIN, "l.xyz", "--gfn", "2", "--opt", level, "--chrg", str(q), "--uhf", str(uhf)]
                if len(dn) >= 2 or constrained_bonds:
                    with open(f"{td}/c.inp", "w") as f:
                        f.write("$constrain\n force constant=1.0\n")
                        for ii, a in enumerate(dn):
                            for b in dn[ii + 1 :]:
                                f.write(f" distance: {a + 1}, {b + 1}, auto\n")
                        for bond in constrained_bonds:
                            a, b = bond["atoms"]
                            f.write(f" distance: {a + 1}, {b + 1}, {bond['before_A']:.8f}\n")
                        f.write("$end\n")
                    args += ["--input", "c.inp"]
                # output unused (geometry read from xtbopt.xyz); not captured: on Windows the text decode of xtb output
                # raised UnicodeDecodeError in the reader threads (cp1252), printing tracebacks
                env = dict(_os.environ, OMP_NUM_THREADS="1") if _e().DETERMINISTIC_EMBED else None
                subprocess.run(args, cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600, env=env)
                lines = open(f"{td}/xtbopt.xyz").read().split("\n")[2 : 2 + len(sy)]
                refined = np.array([[float(v) for v in ln.split()[1:4]] for ln in lines])
                if refined.shape != xyz_in.shape or not np.isfinite(refined).all():
                    raise ValueError("GFN2 returned invalid coordinate array")
                # Li contributes to the optimization but is removed before fitting the original ligand frame.
                ligand = refined[:n_ligand_atoms]
                ca, cb = positions.mean(0), ligand.mean(0)
                U, _, Vt = np.linalg.svd((ligand - cb).T @ (positions - ca))
                R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
                aligned = (ligand - cb) @ R + ca
                return aligned

            out = refine(xyz_in)
            if use_bond_check:
                issues = _e()._bond_refinement_violations(rdmol_h, positions, out)
                if issues:
                    try:
                        retried = refine(xyz_in, issues)
                        remaining = _e()._bond_refinement_violations(rdmol_h, positions, retried)
                        if remaining:
                            out = np.asarray(positions, dtype=np.float64).copy()
                            fallback_used = "pre_refinement_after_constrained_violation"
                        else:
                            out = retried
                            fallback_used = "bond_constrained_gfn2"
                    except Exception as retry_exc:
                        out = np.asarray(positions, dtype=np.float64).copy()
                        fallback_used = f"pre_refinement_after_constrained_failure:{type(retry_exc).__name__}"
                detail = {"triggered": bool(issues), "bonds": issues, "fallback_used": fallback_used}
                _e()._GFN2_BOND_CHECK_CACHE[key] = detail
                _e()._EMBED_BOND_CHECK_EVENTS.append(dict(smiles=Chem.MolToSmiles(rdmol_h), **detail))
                if issues:
                    _e().log.warning("embed_bond_check constrained GFN2 refinement for %s: %s (fallback=%s)",
                                key[0], issues, fallback_used)
    except Exception as exc:  # never let the refinement break a solve
        _e().log.warning("GFN2 embed refinement failed for %s (%s); keeping the MMFF embed", key[0], exc)
        if use_bond_check:
            detail = {"triggered": False, "bonds": [], "fallback_used": f"pre_refinement_after_initial_failure:{type(exc).__name__}"}
            _e()._GFN2_BOND_CHECK_CACHE[key] = detail
            _e()._EMBED_BOND_CHECK_EVENTS.append(dict(smiles=Chem.MolToSmiles(rdmol_h), **detail))
    _e()._GFN2_CACHE[key] = out
    return out.copy()


# ---------------------------------------------------------------------------
# Metal-templated ring conformers (macrocycle pool, profile field macro_template)
# ---------------------------------------------------------------------------

#: donor angle bound for the virtual metal: every donor-substituent direction makes >= this angle with the M-D bond,
#: so the donor's lone pair (or face) points at the metal
TEMPLATE_ANGLE_MIN_DEG = 100.0
#: minimum distance of any non-donor heavy atom / H (two or more bonds from every donor) from the virtual metal (A)
TEMPLATE_CLEAR_HEAVY, TEMPLATE_CLEAR_H = 2.8, 2.3


def _template_fit(P: np.ndarray, donors: Sequence[Sequence[int]], radii: Sequence[float], iters: int = 8):
    """Least-squares metal point p for donor groups (centroids) at distances ``radii``; returns (p, rms residual)."""
    D = np.stack([P[list(g)].mean(0) for g in donors])
    r = np.asarray(radii, float)
    p = D.mean(0)
    for _ in range(iters):
        v = p - D
        d = np.linalg.norm(v, axis=1).clip(1e-6)
        res = d - r
        J = v / d[:, None]
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        p = p + step
    res = np.linalg.norm(p - D, axis=1) - r
    return p, float(np.sqrt((res ** 2).mean()))


def templated_conformers(rd, donors: Sequence[Sequence[int]], radii: Sequence[float], n: int, seed: int) -> List[np.ndarray]:
    """Free-ligand conformers that could wrap one metal: ETKDG with a virtual, unbonded metal atom in the bounds
    matrix (each donor at its M-D distance, every donor substituent at >= TEMPLATE_ANGLE_MIN_DEG from the M-D bond,
    every other atom clear of the metal), then MMFF of the ligand alone with its donor-donor distances restrained to
    the embedded values. A macrocycle embedded without the metal rarely takes the folds it has around one (all
    ring donors on one face, lone pairs converging), and the solve, which moves torsions with the ring closed,
    cannot reach them from a free fold. No reference data: the radii are the Q1 means of the donors' keys.

    ``rd``: the ligand (explicit H) whose atom order the result uses; ``donors``: local atom index groups (one atom
    for kappa, several for a haptic face). Returns up to n (N, 3) arrays, sorted by template residual; [] on failure.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom
    from rdkit.DistanceGeometry import DoTriangleSmoothing

    try:
        n_at = rd.GetNumAtoms()
        mh = Chem.RWMol(rd)
        mh.RemoveAllConformers()
        m_idx = mh.AddAtom(Chem.Atom(0))
        mol = mh.GetMol()
        mol.UpdatePropertyCache(strict=False)
        bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)

        def setb(i, j, lo, hi):
            a, b = max(i, j), min(i, j)   # upper bound in bm[min, max], lower in bm[max, min]
            bm[b, a] = hi
            bm[a, b] = min(lo, hi)

        def blen(i, j):
            a, b = max(i, j), min(i, j)
            return 0.5 * (bm[b, a] + bm[a, b])

        cos_min = math.cos(math.radians(TEMPLATE_ANGLE_MIN_DEG))
        near = set()
        for g, r in zip(donors, radii):
            g = [int(a) for a in g]
            near.update(g)
            if len(g) == 1:
                d = g[0]
                setb(m_idx, d, r - 0.05, r + 0.05)
                for nb in mol.GetAtomWithIdx(d).GetNeighbors():
                    x = nb.GetIdx()
                    if x == m_idx:
                        continue
                    b = blen(d, x)
                    setb(m_idx, x, math.sqrt(r * r + b * b - 2 * r * b * cos_min), r + b)
                    near.add(x)
            else:
                for a in g:
                    setb(m_idx, a, r - 0.1, r + 0.6)
                    near.update(nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors())
        for k in range(n_at):
            if k in near:
                continue
            lo = TEMPLATE_CLEAR_H if mol.GetAtomWithIdx(k).GetAtomicNum() == 1 else TEMPLATE_CLEAR_HEAVY
            a, b = max(k, m_idx), min(k, m_idx)
            bm[a, b] = max(bm[a, b], lo)
        if not DoTriangleSmoothing(bm):
            log.info("templated ring conformers: bounds not triangle-smoothable")
            return []
        ps = AllChem.ETKDGv3()
        ps.randomSeed = int(seed)
        ps.SetBoundsMat(bm)
        ps.useRandomCoords = True
        ps.numThreads = 0
        cids = list(AllChem.EmbedMultipleConfs(mol, int(n), ps))
        if not cids:
            return []
        lig = Chem.Mol(rd)
        lig.RemoveAllConformers()
        for c in cids:
            P = mol.GetConformer(c).GetPositions()[:n_at]
            conf = Chem.Conformer(n_at)
            for ai in range(n_at):
                conf.SetAtomPosition(ai, P[ai].tolist())
            lig.AddConformer(conf, assignId=True)
        flat = [int(g[0]) for g in donors if len(g) == 1]
        if AllChem.MMFFHasAllMoleculeParams(lig):
            props = AllChem.MMFFGetMoleculeProperties(lig)
            for conf in lig.GetConformers():
                ff = AllChem.MMFFGetMoleculeForceField(lig, props, confId=conf.GetId())
                P = conf.GetPositions()
                for ia in range(len(flat)):
                    for ib in range(ia + 1, len(flat)):
                        dd = float(np.linalg.norm(P[flat[ia]] - P[flat[ib]]))
                        ff.MMFFAddDistanceConstraint(flat[ia], flat[ib], False, dd - 0.1, dd + 0.1, RESTRAINT_K)
                ff.Minimize(maxIts=500)
        out = []
        for conf in lig.GetConformers():
            P = np.asarray(conf.GetPositions(), float)
            p, rms = _template_fit(P, donors, radii)
            clear = np.linalg.norm(P - p, axis=1)
            mask = np.ones(n_at, bool)
            mask[list(near)] = False
            clash = float(np.clip(TEMPLATE_CLEAR_H - clear[mask], 0, None).sum()) if mask.any() else 0.0
            out.append((rms + clash, P))
        out.sort(key=lambda t: t[0])
        return [P for _, P in out]
    except Exception as exc:  # the generator never fails: the caller keeps its free-ligand pool
        log.warning("templated ring conformers failed (%s: %s)", type(exc).__name__, exc)
        return []


#: li_cap_spread: minimum Li-Li and Li-(other ligand atom) distances of the GFN2 counter-ion caps, A
LI_CAP_MIN_LI, LI_CAP_MIN_ATOM = 2.8, 1.7


def _fibonacci_dirs(n: int = 96) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)


def _spread_li_caps(positions: np.ndarray, donors, caps) -> list:
    """Profile field li_cap_spread: keep every Li cap (the charge bookkeeping is unchanged) but never two in one
    place. Each cap sits 1.9 A from its donor along the lone-pair direction; in a macrocycle every donor's direction
    points into the cavity, so the caps coincide (UYOVUA: two Li 0.55 A apart, GFN2 SCF diverged and the embed fell
    back to MMFF). Greedy, in donor order: a cap keeps its direction when it is >= LI_CAP_MIN_LI from the caps placed
    so far and >= LI_CAP_MIN_ATOM from every other ligand atom; otherwise it takes, among directions in its donor's
    open hemisphere (<= 90 deg from the lone-pair direction), the one with the largest clearance."""
    dirs = _fibonacci_dirs()
    placed = []
    for d, cap in zip(donors, caps):
        p0 = positions[d]
        lp = np.asarray(cap) - p0
        lp = lp / max(np.linalg.norm(lp), 1e-12)
        others = np.delete(positions, d, axis=0)

        def clearance(c):
            da = np.linalg.norm(others - c, axis=1).min() / LI_CAP_MIN_ATOM if len(others) else np.inf
            dl = min((np.linalg.norm(c - q) for q in placed), default=np.inf) / LI_CAP_MIN_LI
            return min(da, dl)

        best = np.asarray(cap, dtype=np.float64)
        if clearance(best) < 1.0:
            cands = [p0 + 1.9 * u for u in dirs if float(u @ lp) >= 0.0]
            cl = [clearance(c) for c in cands]
            if cands and max(cl) > clearance(best):
                best = cands[int(np.argmax(cl))]
        placed.append(best)
    return placed
