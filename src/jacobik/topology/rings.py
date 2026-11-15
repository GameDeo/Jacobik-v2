"""Ring chairification and saturated five/six-ring conformer-state enumeration (MMFF constrained minimisation)."""
from __future__ import annotations

import itertools
import logging
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from jacobik.core.geometry import dihedral_deg

log = logging.getLogger(__name__)

CHAIR_DEG = 55.0
CHAIR_MIN_DEG = 35.0       # |torsion| above this, alternating in sign, is a chair
TOL_DEG = 1.0
FORCE_CONSTANT = 100.0
CHAIR_MAX_ITERS = 2000
CHAIR_RAMP = (0.25, 0.5, 0.75, 1.0)
MAX_JOINT_RINGS = 3        # rings sharing atoms are enumerated jointly up to this many (2^k pattern combinations)
VOL_MIN = 0.3              # |signed volume| of a reference centre below which it is planar (not a stereocentre)

_SP3_HEAVY = {6, 7, 8, 14, 15, 16}   # C N O Si P S


def _torsion(X, a, b, c, d) -> float:
    return dihedral_deg(X[a], X[b], X[c], X[d])


def ring_torsions(X, ring) -> List[float]:
    n = len(ring)
    return [_torsion(X, *[ring[(k + j) % n] for j in range(4)]) for k in range(n)]


def is_chair(X, ring) -> bool:
    t = np.asarray(ring_torsions(X, ring))
    return bool(np.all(np.abs(t) > CHAIR_MIN_DEG) and np.all(np.sign(t) * np.sign(np.roll(t, 1)) < 0))


def saturated_six_rings(mol_h) -> List[Tuple[int, ...]]:
    """Ordered atom tuples of the saturated, non-aromatic six-rings (sp3 atoms, single ring bonds)."""
    return saturated_rings(mol_h, 6)


def saturated_rings(mol_h, size: int) -> List[Tuple[int, ...]]:
    """Ordered atom tuples of the saturated, non-aromatic rings of `size` atoms (sp3 atoms, single ring bonds)."""
    from rdkit import Chem
    ri = mol_h.GetRingInfo()
    try:
        ri.AtomRings()
    except RuntimeError:       # ring info not initialised on this molecule
        Chem.GetSymmSSSR(mol_h)
        ri = mol_h.GetRingInfo()
    out = []
    for ring in ri.AtomRings():
        if len(ring) != size:
            continue
        atoms = [mol_h.GetAtomWithIdx(i) for i in ring]
        if any(a.GetIsAromatic() or a.GetAtomicNum() not in _SP3_HEAVY for a in atoms):
            continue
        if any(a.GetHybridization() != Chem.HybridizationType.SP3 for a in atoms):
            continue
        ok = True
        for k in range(size):
            b = mol_h.GetBondBetweenAtoms(ring[k], ring[(k + 1) % size])
            if b is None or b.GetBondType() != Chem.BondType.SINGLE or b.GetIsAromatic():
                ok = False
                break
        if ok:
            out.append(tuple(int(i) for i in ring))
    return out


def _groups(rings: List[Tuple[int, ...]]) -> List[List[Tuple[int, ...]]]:
    """Connected components of the rings that share at least one atom."""
    left = list(rings)
    groups = []
    while left:
        g = [left.pop(0)]
        grew = True
        while grew:
            grew = False
            for r in list(left):
                if any(set(r) & set(q) for q in g):
                    g.append(r)
                    left.remove(r)
                    grew = True
        groups.append(g)
    return groups


def _centres(mol_h, X) -> Dict[int, Tuple[int, int, int]]:
    """Stereocentres: atoms with >= 3 neighbours that are pairwise constitutionally inequivalent (distinct canonical
    ranks) and whose three lowest-index neighbours span a volume in X (not planar).  Equivalent neighbours (CH2 hydrogens,
    the two ring arms of a cyclohexyl C1) are excluded: swapping them is not a change of configuration."""
    from rdkit import Chem
    rank = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    out = {}
    for a in mol_h.GetAtoms():
        i = a.GetIdx()
        nb = sorted(n.GetIdx() for n in a.GetNeighbors())
        if len(nb) < 3 or a.GetAtomicNum() == 1:
            continue
        if len({rank[j] for j in nb}) != len(nb):
            continue
        t = tuple(nb[:3])
        if abs(float(np.linalg.det(np.stack([X[j] - X[i] for j in t])))) >= VOL_MIN:
            out[i] = t
    return out


def _signs(X, centres) -> Dict[int, int]:
    return {i: int(np.sign(np.linalg.det(np.stack([X[j] - X[i] for j in t])))) for i, t in centres.items()}


def _bond_stereo(mol_h, X) -> Dict[Tuple[int, int], str]:
    from rdkit import Chem
    m = Chem.Mol(mol_h)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(X):
        conf.SetAtomPosition(i, [float(v) for v in p])
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m, confId=0, replaceExistingTags=True)
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetStereo().name for b in m.GetBonds()
            if b.GetBondType() == Chem.BondType.DOUBLE and b.GetStereo().name not in ("STEREONONE", "STEREOANY")}


def _with_positions(mol_h, X):
    from rdkit import Chem
    m = Chem.Mol(mol_h)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(X):
        conf.SetAtomPosition(i, [float(v) for v in p])
    m.AddConformer(conf, assignId=True)
    return m


def _minimise(m, props, X, constraints=(), fixed=(), max_iters=None):
    """MMFF minimisation from X with optional torsion constraints and fixed atoms; returns the new positions."""
    from rdkit.Chem import rdForceFieldHelpers as FF
    mm = _with_positions(m, X)
    ff = FF.MMFFGetMoleculeForceField(mm, props, confId=0)
    if ff is None:
        raise ValueError("MMFF force field could not be set up")
    for (a, b, c, d, t) in constraints:
        ff.MMFFAddTorsionConstraint(int(a), int(b), int(c), int(d), False, t - TOL_DEG, t + TOL_DEG, FORCE_CONSTANT)
    for i in fixed:
        ff.AddFixedPoint(int(i))
    ff.Initialize()
    ff.Minimize(maxIts=CHAIR_MAX_ITERS if max_iters is None else max_iters)
    return np.asarray(ff.Positions(), float).reshape(-1, 3)


def _energy(m, props, X) -> float:
    from rdkit.Chem import rdForceFieldHelpers as FF
    mm = _with_positions(m, X)              # the force field does not own the molecule: keep it alive
    ff = FF.MMFFGetMoleculeForceField(mm, props, confId=0)
    return float(ff.CalcEnergy(list(np.asarray(X, float).ravel())))


def _pattern(ring, sign) -> list:
    n = len(ring)
    return [(ring[k], ring[(k + 1) % n], ring[(k + 2) % n], ring[(k + 3) % n], sign * CHAIR_DEG * (-1) ** k) for k in range(n)]


def chairify(mol_h, positions) -> Tuple[np.ndarray, dict]:
    """Bring every saturated six-ring of the embedded ligand to a chair.  Returns (positions, info); the input positions
    are returned unchanged on any failure or when there is nothing to do.  info: n_rings, n_chair_before, n_chair_after,
    n_changed_groups, n_rejected (stereo), skipped (reason), wall_s."""
    import time
    from rdkit.Chem import AllChem
    t0 = time.perf_counter()
    X0 = np.asarray(positions, float).copy()
    info = dict(n_rings=0, n_chair_before=0, n_chair_after=0, n_changed_groups=0, n_rejected=0, skipped=None, wall_s=0.0)
    try:
        rings = saturated_six_rings(mol_h)
        info["n_rings"] = len(rings)
        info["n_chair_before"] = sum(is_chair(X0, r) for r in rings)
        info["n_chair_after"] = info["n_chair_before"]
        if not rings or info["n_chair_before"] == len(rings):
            return X0, info
        props = AllChem.MMFFGetMoleculeProperties(mol_h)
        if props is None:
            info["skipped"] = "no MMFF parameters"
            return X0, info
        centres = _centres(mol_h, X0)
        sign0 = _signs(X0, centres)
        bonds0 = _bond_stereo(mol_h, X0)
        X = X0.copy()

        def stereo_kept(Y) -> bool:
            s = _signs(Y, centres)
            return s == sign0 and _bond_stereo(mol_h, Y) == bonds0

        for group in _groups(rings):
            if all(is_chair(X, r) for r in group):
                continue
            units = [group] if len(group) <= MAX_JOINT_RINGS else [[r] for r in group]
            for unit in units:
                best = (sum(is_chair(X, r) for r in unit), -_energy(mol_h, props, X), X)   # (chairs, -energy, positions)
                for signs in itertools.product((1, -1), repeat=len(unit)):
                    target = [c for r, s in zip(unit, signs) for c in _pattern(r, s)]
                    try:
                        Y = X
                        for frac in CHAIR_RAMP:      # ramp the torsion targets: a continuous path cannot invert a centre
                            cons = []
                            for (a, b, c, d, t) in target:
                                cur = _torsion(Y, a, b, c, d)
                                dt = (t - cur + 180.0) % 360.0 - 180.0
                                cons.append((a, b, c, d, cur + frac * dt))
                            Y = _minimise(mol_h, props, Y, cons)
                        Y = _minimise(mol_h, props, Y)
                    except Exception as exc:
                        log.debug("chair candidate failed: %s", exc)
                        continue
                    if not np.isfinite(Y).all() or not stereo_kept(Y):
                        info["n_rejected"] += 1
                        continue
                    cand = (sum(is_chair(Y, r) for r in unit), -_energy(mol_h, props, Y), Y)
                    if cand[:2] > best[:2]:
                        best = cand
                if best[2] is not X:
                    info["n_changed_groups"] += 1
                    X = best[2]
        info["n_chair_after"] = sum(is_chair(X, r) for r in rings)
        if info["n_chair_after"] < info["n_chair_before"] or not stereo_kept(X) or not np.isfinite(X).all():
            info["skipped"] = "result worse or stereo changed; reverted"
            return X0, info
        return X, info
    except Exception as exc:  # never break the embed
        info["skipped"] = f"{type(exc).__name__}: {exc}"
        log.warning("embed_ring_chair failed (%s); keeping the original embed", exc)
        return X0, info
    finally:
        info["wall_s"] = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Ring-pucker state library & allocation (P2R)
# ---------------------------------------------------------------------------
import itertools
from typing import Dict, List, Optional, Tuple

from jacobik.core import profile as _profile
from jacobik.core.env import RT_KJ


MAX_STATES = 8                # ligand states, embed included
MAX_PER_RING = 4              # ring states entering the combinations, embed state included
NEAR_BONDS = 2
SIG_TOL_DEG = 20.0            # two ring states are the same when every ring torsion agrees within this
RAMP = (0.5, 1.0)
MAX_ITERS = 500
KCAL_KJ = 4.184

FIVE = _profile.get("ring_states_five")
_PSEUDO_AMP = 40.0

_TB = (31.0, 31.0, -66.0, 31.0, 31.0, -66.0)
_CHAIR = (55.0, -55.0, 55.0, -55.0, 55.0, -55.0)
_CACHE: Dict[tuple, Optional[dict]] = {}


def _patterns(n: int = 6) -> List[Tuple[str, Tuple[float, ...]]]:
    if n == 5:
        return [(f"P{p}", tuple(_PSEUDO_AMP * float(np.cos(np.radians(p + 144.0 * j))) for j in range(5)))
                for p in range(0, 360, 18)]
    out = [("chair+", _CHAIR), ("chair-", tuple(-t for t in _CHAIR))]
    for k in range(3):
        rot = tuple(_TB[(j + k) % 6] for j in range(6))
        out += [(f"tb{k}+", rot), (f"tb{k}-", tuple(-t for t in rot))]
    return out


def _rings(rd_h) -> List[Tuple[int, ...]]:
    return saturated_six_rings(rd_h) + (saturated_rings(rd_h, 5) if FIVE else [])


def near_rings(rd_h, donors_local) -> List[Tuple[int, ...]]:
    from rdkit import Chem
    rings = _rings(rd_h)
    if not rings or not donors_local:
        return []
    dm = Chem.GetDistanceMatrix(rd_h)
    return [r for r in rings if min(dm[a][d] for a in r for d in donors_local) <= NEAR_BONDS]


def _sig(X, ring) -> np.ndarray:
    return np.asarray(ring_torsions(X, ring))


def _same(s1, s2) -> bool:
    return bool(np.all(np.abs((s1 - s2 + 180.0) % 360.0 - 180.0) <= SIG_TOL_DEG))


def _kind(X, ring) -> str:
    if len(ring) == 5:          # envelope: one ring torsion near zero; otherwise a twist
        t = np.abs(_sig(X, ring))
        return "env" if t.min() < 0.25 * t.max() else "tw5"
    return "chair" if is_chair(X, ring) else "twist"


def library(rd_h, X_lig, donors_local) -> Optional[dict]:
    """Ring-state library of one ligand: dict(coords (K, n, 3) [state 0 = X_lig], dE_kj (K,), share (K,), labels,
    info) or None (no near ring, MMFF unavailable, or any failure: the rows keep the embed). Never raises."""
    from rdkit import Chem
    X0 = np.asarray(X_lig, float)
    try:
        key = (Chem.MolToSmiles(rd_h), tuple(sorted(donors_local)), X0.round(3).tobytes())
    except Exception:
        key = None
    if key is not None and key in _CACHE:
        return _CACHE[key]
    t0 = time.perf_counter()
    out = None
    try:
        out = _build(rd_h, X0, list(donors_local))
        if out is not None:
            out["info"]["wall_s"] = time.perf_counter() - t0
            log.warning("ring_states: %d states (%s), dE kJ/mol %s, %.1f s", len(out["dE_kj"]), out["labels"],
                        np.round(out["dE_kj"], 1).tolist(), out["info"]["wall_s"])
    except Exception as exc:  # the generator never fails: the embed stands
        log.warning("ring_states failed (%s: %s); embed rings kept", type(exc).__name__, exc)
        out = None
    if key is not None:
        _CACHE[key] = out
    return out


def _build(rd_h, X0, donors) -> Optional[dict]:
    from rdkit.Chem import AllChem
    from rdkit import Chem
    rings = near_rings(rd_h, donors)
    if not rings:
        return None
    props = AllChem.MMFFGetMoleculeProperties(rd_h)
    if props is None:
        return None
    centres = _centres(rd_h, X0)
    sign0 = _signs(X0, centres)
    bonds0 = _bond_stereo(rd_h, X0)

    hs = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() == 1] for a in rd_h.GetAtoms()}

    def free_of(rs):
        f = set()
        for r in rs:
            for a in r:
                f.add(a); f.update(hs[a])
        return f - set(donors)            # a donor in the ring (HOHZAG's carbanion) stays at the embed

    def realise(start, targets, free):
        """targets: [(ring, pattern torsions)]; ramp, release, minimise with only `free` atoms moving -> (Y, E kJ)."""
        fixed = [i for i in range(len(start)) if i not in free]
        Y = start
        for frac in RAMP:
            cons = []
            for ring, pat in targets:
                n = len(ring)
                for k in range(n):
                    a, b, c, d = (ring[(k + j) % n] for j in range(4))
                    cur = _torsion(Y, a, b, c, d)
                    dt = (pat[k] - cur + 180.0) % 360.0 - 180.0
                    cons.append((a, b, c, d, cur + frac * dt))
            Y = _minimise(rd_h, props, Y, cons, fixed, max_iters=MAX_ITERS)
        Y = _minimise(rd_h, props, Y, (), fixed, max_iters=MAX_ITERS)
        if not np.isfinite(Y).all() or _signs(Y, centres) != sign0 or _bond_stereo(rd_h, Y) != bonds0:
            return None
        return Y, _energy(rd_h, props, Y) * KCAL_KJ

    all_free = free_of(rings)
    base = _minimise(rd_h, props, X0, (), [i for i in range(len(X0)) if i not in all_free], max_iters=MAX_ITERS)
    E_base = _energy(rd_h, props, base) * KCAL_KJ
    # remote rings (plan: "remote rings start in their lowest pucker"): each brought to its lowest state, with only
    # that ring free; the result is the start of every non-embed state (state 0 keeps the embed as it is)
    remote = [r for r in _rings(rd_h) if r not in rings]
    n_remote_changed = 0
    for ring in remote:
        best = None
        for name, pat in _patterns(len(ring)):
            try:
                r = realise(base, [(ring, pat)], free_of([ring]))
            except Exception:
                r = None
            if r is not None and (best is None or r[1] < best[1]):
                best = r
        E_cur = _energy(rd_h, props, base) * KCAL_KJ
        if best is not None and best[1] < E_cur - 1e-6:
            base = best[0]
            n_remote_changed += 1
    if n_remote_changed:
        base = _minimise(rd_h, props, base, (), [i for i in range(len(X0)) if i not in all_free], max_iters=MAX_ITERS)
    E_embed, E_base = E_base, _energy(rd_h, props, base) * KCAL_KJ   # state 0 (embed) / start of the others
    # per ring: distinct states (state 0 = the embed's), realised with the other rings where they are
    per_ring = []
    for ring in rings:
        states = [(_sig(base, ring), E_base, None, _kind(base, ring))]
        for name, pat in _patterns(len(ring)):
            try:
                r = realise(base, [(ring, pat)], free_of([ring]))
            except Exception as exc:
                log.debug("ring state %s failed: %s", name, exc)
                r = None
            if r is None:
                continue
            Y, E = r
            s = _sig(Y, ring)
            # same state = same ring torsions only: equal free-ligand energies do not make two puckers equivalent in
            # the complex (HOHZAG's cyclohexyl: metal axial vs equatorial, equal MMFF energy without the metal)
            hit = next((j for j, st in enumerate(states) if _same(st[0], s)), None)
            if hit is None:
                states.append((s, E, s, _kind(Y, ring)))
            elif E < states[hit][1] - 1e-6 and hit > 0:
                states[hit] = (s, E, s, _kind(Y, ring))
        states = [states[0]] + sorted(states[1:], key=lambda t: t[1])[:MAX_PER_RING - 1]
        per_ring.append(states)
    # combinations by summed ring energy; the all-embed combination is state 0
    combos = []
    for pick in itertools.product(*[range(len(s)) for s in per_ring]):
        if not any(pick) and not n_remote_changed:      # the embed itself (state 0)
            continue
        combos.append((sum(per_ring[r][k][1] - E_base for r, k in enumerate(pick)), pick))
    combos.sort()
    capped = len(combos) > MAX_STATES - 1
    coords, energies, labels, sigs = [X0], [E_embed], ["embed:" + "/".join(s[0][3] for s in per_ring)], [None]
    for _, pick in combos:
        if len(coords) >= MAX_STATES:
            break
        targets = [(rings[r], tuple(per_ring[r][k][2])) for r, k in enumerate(pick) if k > 0]
        try:
            res = (base, E_base) if not targets else realise(base, targets, free_of([t[0] for t in targets]))
        except Exception as exc:
            log.debug("ring combination %s failed: %s", pick, exc)
            res = None
        if res is None:
            continue
        Y, E = res
        sg = [_sig(Y, r) for r in rings]
        lab = "/".join(_kind(Y, r) for r in rings)
        if any(s is not None and all(_same(a, b) for a, b in zip(sg, s)) for s in sigs[1:]) or \
                (targets and all(_same(a, _sig(base, r)) for a, r in zip(sg, rings))):
            continue
        coords.append(Y)                    # every atom outside the changed rings is at the embed
        energies.append(E)
        labels.append(lab)
        sigs.append(sg)
    if len(coords) == 1:
        return None
    E = np.asarray(energies)
    dE = E - E.min()
    w = np.exp(-dE / RT_KJ)
    w = w / w.sum()
    share = 0.5 / len(w) + 0.5 * w           # every state at least 1/(2K) of the rows; the other half by population
    if capped:
        log.warning("ring_states: %d ring-state combinations, %d realised (cap MAX_STATES=%d)", len(combos),
                    len(coords) - 1, MAX_STATES)
    return dict(coords=np.stack(coords), dE_kj=dE, share=share, labels=labels,
                info=dict(n_near_rings=len(rings), n_remote_rings=len(remote), n_remote_changed=n_remote_changed, n_ring_states=[len(s) for s in per_ring], n_combos=len(combos),
                          capped=capped))


def allocate(share: np.ndarray, B: int, rng) -> np.ndarray:
    """(B,) state index per row: counts by largest remainder of share * B, every state at least one row when B allows,
    in an order permuted by `rng` (a seed's generator, as the macrocycle folds)."""
    K = len(share)
    raw = np.asarray(share, float) * B
    n = np.floor(raw).astype(int)
    if B >= K:
        n = np.maximum(n, 1)
    while n.sum() > B:
        n[np.argmax(n)] -= 1
    rem = raw - np.floor(raw)
    for k in np.argsort(-rem):
        if n.sum() >= B:
            break
        n[k] += 1
    idx = np.repeat(np.arange(K), n)[:B]
    return idx[rng.permutation(len(idx))]
