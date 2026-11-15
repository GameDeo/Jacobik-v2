"""Input-consistency audit: does the SMILES agree with the reference geometry? (evaluation only, CPU)

NOT a generator rule. The generator never rejects an input (research/archive/docs/roadmap/roadmap_generation_v1.md section 2.0). This script
labels each parent so that scoreboards can be split by label. The reference geometry is used here, so the label
is an evaluation label. Section 2.0 of the roadmap defines the reference-free production status that D14b
compares against it.

Method
  1. SMILES side (RDKit, metal atoms removed, sanitised): bond orders, formal charges, valences, rings.
  2. Geometry side: the reference coordinates of the parent (mol.atoms position). Heavy non-metal atoms are
     bonded when d < 1.20 * (r_i + r_j) (covalent radii of ctxio.COV).
  3. Graph isomorphism (networkx) between the two heavy non-metal graphs with element labels. A failed match
     gives label 'ambiguous' (reason 'unmatched_graph'); no geometry check is possible then.
  4. Checks on the matched graph (thresholds below).

Thresholds (frozen for the reported run; the revision history is stated here, not hidden)
  SPREAD_AROM   = 0.03 A   max-min of ring C-C bond lengths: at or below = aromatic geometry
  SPREAD_LOCAL  = 0.10 A   at or above = localised geometry; between = intermediate (partial alternation)
  BOND_SOFT     = 0.08 A   distance outside the order's length band above this is a soft mismatch
  BOND_HARD     = 0.15 A   above this (and not charge-adjacent) it is a hard mismatch
  Formal-charge windows: ctxio.LIGHT_QWIN (per element) and ctxio.MAX_OX (highest common oxidation state of a
  metal, written as a formal charge; lower limit -2). Oxo O(-2), nitride N(-3) are inside their windows.
  Revision history. Run 1 used point ideal lengths (C-C single 1.53 A, C-N single 1.47 A, ...). Its soft flags were
  dominated by conjugated and heteroaromatic single bonds, so bands (BANDS) replaced the point ideals. The
  charge-adjacent cap and the SMILES-aromatic skip were added while prototyping on 10 named parents, before the
  first full run.
  The first full run (232 parents) is reproduced by --point-ideal: it already had the SMILES-aromatic skip and the
  charge-adjacent cap. Both runs are reported; the label is the band run's label.

Flags
  hard  quinoid_written_aromatic  ring of 5/6 C/N atoms, geometry aromatic (spread <= SPREAD_AROM), SMILES has an
                                  exocyclic double bond on a ring atom and fewer than floor(n/2) ring-internal doubles
  hard  bond_contradicts_order    bond outside its length band by more than BOND_HARD, not charge-adjacent
  hard  hypervalent_S             S/Se/Te with valence 4, neutral, and no O neighbour
  hard  absurd_light_charge       formal charge outside the element window (ctxio.LIGHT_QWIN) on a ligand atom
  hard  absurd_metal_charge       formal charge above the metal's highest common oxidation state (ctxio.MAX_OX)
  hard  charge_imbalance_N        neutral N with >= 2 O- neighbours (nitro written as N([O-])[O-])
  soft  partial_alternation       quinoid pattern on an intermediate-geometry ring
  soft  bond_soft_mismatch        distance outside the band in (BOND_SOFT, BOND_HARD]
  soft  donor_typing              S donor typed by SMILES (thiocarbonyl / thiolate / thioether) disagrees with C-S length
  soft  sanitize_failed           RDKit sanitisation of the metal-free ligand graph failed

Label: inconsistent if any hard flag; ambiguous if any soft flag or an unmatched graph; otherwise consistent.

    python tools/jgen/diag/d14_input_consistency.py --split all --out-json <path>
    python tools/jgen/diag/d14_input_consistency.py --codes HOXJUZ,NEPNEF --verbose
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxio  # noqa: E402
from ctxio import absurd_light, absurd_metal  # noqa: E402

SPREAD_AROM, SPREAD_LOCAL = 0.03, 0.10
BOND_SOFT, BOND_HARD = 0.08, 0.15
ADJ = 1.20

# Length bands (A) per element pair and SMILES bond order (1, 1.5 aromatic, 2). A bond is consistent with its
# order when its length lies inside the band; the mismatch is the distance outside the band. Bands cover
# conjugated single bonds (C-C about 1.44-1.48 A), heteroaromatic single bonds (C-N about 1.33-1.37 A) and
# charge-separated groups (see the charged-atom cap). The bands are textbook ranges, not fitted to the parents.
# The first run used point ideals (C-C single 1.53, C-N single 1.47, ...); its results are kept as a sensitivity
# run (flag --point-ideal), because the change came after seeing that conjugated singles dominated the soft flags.
BANDS = {
    ("C", "C"): {1.0: (1.40, 1.60), 1.5: (1.33, 1.42), 2.0: (1.28, 1.40)},
    ("C", "N"): {1.0: (1.33, 1.52), 1.5: (1.31, 1.38), 2.0: (1.25, 1.36)},
    ("C", "O"): {1.0: (1.30, 1.48), 2.0: (1.19, 1.28)},
    ("C", "S"): {1.0: (1.70, 1.85), 2.0: (1.60, 1.70)},
    ("N", "N"): {1.0: (1.30, 1.50), 2.0: (1.20, 1.32)},
    ("N", "O"): {1.0: (1.30, 1.50), 2.0: (1.15, 1.25)},
    ("N", "S"): {1.0: (1.60, 1.78), 2.0: (1.50, 1.60)},
    ("O", "S"): {1.0: (1.50, 1.70), 2.0: (1.40, 1.50)},
    ("S", "S"): {1.0: (1.95, 2.15)},
}
# Sensitivity run only (--point-ideal): the v1 point ideals, each as a zero-width band.
POINT_IDEAL = {
    ("C", "C"): {1.0: 1.53, 1.5: 1.40, 2.0: 1.34},
    ("C", "N"): {1.0: 1.47, 1.5: 1.34, 2.0: 1.29},
    ("C", "O"): {1.0: 1.43, 2.0: 1.22},
    ("C", "S"): {1.0: 1.77, 2.0: 1.67},
    ("N", "N"): {1.0: 1.45, 2.0: 1.25},
    ("N", "O"): {1.0: 1.42, 2.0: 1.21},
    ("N", "S"): {1.0: 1.70, 2.0: 1.55},
    ("O", "S"): {1.0: 1.60, 2.0: 1.45},
    ("S", "S"): {1.0: 2.05},
}
BANDS_POINT = {k: {o: (v, v) for o, v in d.items()} for k, d in POINT_IDEAL.items()}


def pair_key(a, b):
    return tuple(sorted((a, b)))


def classify(mol, point_ideal=False):
    """Return a dict with the label, flags and per-ring / per-bond evidence for one parent."""
    import networkx as nx
    from networkx.algorithms import isomorphism as iso
    from rdkit import Chem
    from rdkit.Chem import rdchem

    out = dict(code=mol.csd_code, label=None, hard=[], soft=[], reasons=[], rings=[], n_heavy=0)
    smi = mol.smiles
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        out.update(label="ambiguous", soft=["unparseable_smiles"])
        return out
    m.UpdatePropertyCache(strict=False)
    metal_charges = [(a.GetSymbol(), a.GetFormalCharge()) for a in m.GetAtoms()
                     if a.GetSymbol() not in ctxio.LIGHT and a.GetSymbol() != "H"]
    if any(absurd_metal(sym_m, q) for sym_m, q in metal_charges):
        out["hard"].append("absurd_metal_charge")
        out["metal_charges"] = [list(x) for x in metal_charges]
    if any(a.GetSymbol() == "H" for a in m.GetAtoms()):
        out["explicit_H_atoms_in_smiles"] = True
    # Implicit H counts are taken before metal atoms are removed.
    nH = {a.GetIdx(): a.GetNumImplicitHs() + a.GetNumExplicitHs() for a in m.GetAtoms()}
    rw = Chem.RWMol(m)
    for a in sorted(rw.GetAtoms(), key=lambda x: -x.GetIdx()):
        if a.GetSymbol() not in ctxio.LIGHT:
            rw.RemoveAtom(a.GetIdx())
    lig = rw.GetMol()
    smi_index = {}
    kept = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() in ctxio.LIGHT]
    for new, old in enumerate(kept):
        smi_index[old] = new
    try:
        Chem.SanitizeMol(lig)
    except Exception as exc:  # RDKit raises on valence or kekulisation failures
        out["soft"].append("sanitize_failed")
        out["reasons"].append("sanitize: %s" % str(exc).splitlines()[0][:120])
    # Aromaticity as perceived from the SMILES, before Kekulisation clears the aromatic flags.
    smiles_aromatic_atoms = {a.GetIdx() for a in lig.GetAtoms() if a.GetIsAromatic()}
    smiles_aromatic_bonds = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                             for b in lig.GetBonds() if b.GetIsAromatic()}
    charged_atoms = {a.GetIdx() for a in lig.GetAtoms() if a.GetFormalCharge() != 0}
    try:
        Chem.Kekulize(lig, clearAromaticFlags=True)
    except Exception as exc:
        out["soft"].append("kekulize_failed")
        out["reasons"].append("kekulize: %s" % str(exc).splitlines()[0][:120])

    # Geometry: heavy light atoms of the reference complex.
    heavy = [a for a in mol.atoms if a.symbol in ctxio.LIGHT]
    pos = np.array([a.position for a in heavy], float)
    sym = [a.symbol for a in heavy]
    n_s, n_g = lig.GetNumAtoms(), len(heavy)
    out["n_heavy"] = n_g
    Gg = nx.Graph()
    for i in range(n_g):
        Gg.add_node(i, sym=sym[i])
    for i in range(n_g):
        for j in range(i + 1, n_g):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < ADJ * (ctxio.COV[sym[i]] + ctxio.COV[sym[j]]):
                Gg.add_edge(i, j)
    Gs = nx.Graph()
    for a in lig.GetAtoms():
        Gs.add_node(a.GetIdx(), sym=a.GetSymbol())
    for b in lig.GetBonds():
        Gs.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), order=b.GetBondTypeAsDouble())
    if n_s != n_g:
        out.update(label="ambiguous", soft=out["soft"] + ["heavy_count_mismatch"],
                   reasons=out["reasons"] + ["smiles %d vs geometry %d heavy atoms" % (n_s, n_g)])
        return out
    gm = iso.GraphMatcher(Gg, Gs, node_match=lambda x, y: x["sym"] == y["sym"])
    if not gm.is_isomorphic():
        out.update(label="ambiguous", soft=out["soft"] + ["unmatched_graph"],
                   reasons=out["reasons"] + ["SMILES and geometry heavy graphs not isomorphic"])
        return out
    g2s = gm.mapping                                  # geometry index -> SMILES index
    s2g = {v: k for k, v in g2s.items()}

    def dist(gi, gj):
        return float(np.linalg.norm(pos[gi] - pos[gj]))

    # Bond-order checks on bonds that are not inside an aromatic or intermediate ring.
    ring_info = lig.GetRingInfo()
    ring_atoms_sets = [set(r) for r in ring_info.AtomRings()]
    ring_status = {}
    for r in ring_atoms_sets:
        rl = sorted(r)
        n = len(rl)
        if n not in (5, 6):
            continue
        if not all(lig.GetAtomWithIdx(s).GetSymbol() in ("C", "N") for s in rl):
            continue
        cc = []
        for b in lig.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if u in r and v in r and lig.GetAtomWithIdx(u).GetSymbol() == "C" and lig.GetAtomWithIdx(v).GetSymbol() == "C":
                cc.append(dist(s2g[u], s2g[v]))
        if len(cc) < 3:
            continue
        spread = max(cc) - min(cc)
        ring_double = 0
        exo_double = 0
        for b in lig.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if b.GetBondType() != rdchem.BondType.DOUBLE:
                continue
            if u in r and v in r:
                ring_double += 1
            elif u in r or v in r:
                exo_double += 1
        quinoid_pattern = exo_double >= 1 and ring_double < n // 2
        if spread <= SPREAD_AROM:
            geom = "aromatic"
        elif spread < SPREAD_LOCAL:
            geom = "intermediate"
        else:
            geom = "localised"
        key = tuple(rl)
        ring_status[key] = geom
        rec = dict(atoms=[int(s2g[s]) + 1 for s in rl], size=n, spread=round(spread, 3), geom=geom,
                   ring_double=ring_double, exo_double=exo_double, quinoid_pattern=quinoid_pattern)
        out["rings"].append(rec)
        if quinoid_pattern and geom == "aromatic":
            out["hard"].append("quinoid_written_aromatic")
        elif quinoid_pattern and geom == "intermediate":
            out["soft"].append("partial_alternation")

    # Bond-order check. Bonds that the SMILES itself perceives as aromatic are skipped (their order is not
    # written). Bonds touching a formally charged atom are capped at soft: charge-separated resonance forms
    # (nitro, N-oxide, carbanion-bearing groups) make single-bond ideals unreliable there.
    worst = []
    for b in lig.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (min(u, v), max(u, v)) in smiles_aromatic_bonds:
            continue
        if b.GetBondType() == rdchem.BondType.DATIVE:
            continue
        su, sv = lig.GetAtomWithIdx(u).GetSymbol(), lig.GetAtomWithIdx(v).GetSymbol()
        order = b.GetBondTypeAsDouble()
        table = BANDS_POINT.get(pair_key(su, sv)) if point_ideal else BANDS.get(pair_key(su, sv))
        if table is None or order not in table:
            continue
        d = dist(s2g[u], s2g[v])
        lo, hi = table[order]
        diff = max(lo - d, d - hi, 0.0)                 # distance outside the band (0 when inside)
        capped = u in charged_atoms or v in charged_atoms
        if diff > BOND_HARD and not capped:
            out["hard"].append("bond_contradicts_order")
            worst.append((round(diff, 3), int(s2g[u]) + 1, su, int(s2g[v]) + 1, sv, order, round(d, 3), "hard"))
        elif diff > BOND_SOFT:
            out["soft"].append("bond_charged_soft" if capped and diff > BOND_HARD else "bond_soft_mismatch")
            worst.append((round(diff, 3), int(s2g[u]) + 1, su, int(s2g[v]) + 1, sv, order, round(d, 3),
                          "charged" if capped else "soft"))

    # Valence, charge and donor checks on the SMILES graph.
    for a in lig.GetAtoms():
        s = a.GetSymbol()
        q = a.GetFormalCharge()
        if absurd_light(a.GetSymbol(), q):
            out["hard"].append("absurd_light_charge")
        if s in ("S", "Se", "Te"):
            val = a.GetTotalValence()
            has_o = any(n.GetSymbol() == "O" for n in a.GetNeighbors())
            if val == 4 and q == 0 and not has_o:
                out["hard"].append("hypervalent_S")
        if s == "N" and q == 0:
            n_om = sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "O" and n.GetFormalCharge() == -1)
            if n_om >= 2:
                out["hard"].append("charge_imbalance_N")
    # Donor typing: sulfur donors bonded to a metal in the reference geometry.
    metal_pos = [np.asarray(a.position, float) for a in mol.atoms if a.symbol not in ctxio.LIGHT and a.symbol != "H"]
    donor_types = []
    for gi, sy in enumerate(sym):
        if sy != "S" or not metal_pos:
            continue
        near_metal = any(np.linalg.norm(pos[gi] - mp) <= 1.15 * (ctxio.COV["S"] + 1.4) for mp in metal_pos)
        if not near_metal:
            continue
        a = lig.GetAtomWithIdx(g2s[gi])
        cs = [n for n in a.GetNeighbors() if n.GetSymbol() == "C"]
        if not cs:
            donor_types.append(dict(atom=gi + 1, smiles_type="none", c_s=None, geom="n/a"))
            continue
        c = cs[0]
        bo = lig.GetBondBetweenAtoms(a.GetIdx(), c.GetIdx()).GetBondTypeAsDouble()
        dcs = dist(gi, s2g[c.GetIdx()])
        if bo == 2.0:
            stype = "thiocarbonyl"
        elif a.GetFormalCharge() == -1:
            stype = "thiolate"
        else:
            stype = "thioether"
        geom_type = "double-like" if dcs <= 1.71 else ("single-like" if dcs >= 1.74 else "ambiguous")
        donor_types.append(dict(atom=gi + 1, smiles_type=stype, c_s=round(dcs, 3), geom=geom_type))
        if (stype == "thiocarbonyl" and geom_type == "single-like") or \
           (stype in ("thiolate", "thioether") and geom_type == "double-like"):
            out["soft"].append("donor_typing")
    out["donor_types"] = donor_types
    out["worst_bonds"] = sorted(worst, reverse=True)[:6]

    hard_flags = sorted(set(out["hard"]))
    soft_flags = sorted(set(out["soft"]))
    out["hard"], out["soft"] = hard_flags, soft_flags
    if hard_flags:
        out["label"] = "inconsistent"
    elif soft_flags:
        out["label"] = "ambiguous"
    else:
        out["label"] = "consistent"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=None, help="comma-separated; default: all parents of --split")
    ap.add_argument("--split", default="pools", help="pools (232 pool parents), all (pools + failure cases), or a frame split name")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--point-ideal", action="store_true", help="sensitivity run with v1 point ideals")
    a = ap.parse_args()
    idx = ctxio.frame_index()
    if a.codes:
        codes = a.codes.split(",")
    elif a.split == "pools":
        codes = list(ctxio.pool_parents().keys())
    else:
        codes = [c for c, r in idx.items() if r["split"] == a.split]
    results = []
    for code in codes:
        mol = ctxio.load_parent(code, idx)
        if mol is None:
            results.append(dict(code=code, label="ambiguous", hard=[], soft=["not_loadable"], reasons=["no mol"]))
            continue
        res = classify(mol, point_ideal=a.point_ideal)
        res["split"] = idx.get(code, {}).get("split")
        res["stratum"] = idx.get(code, {}).get("stratum")
        results.append(res)
        if a.verbose:
            print(res["code"], res["label"], res["hard"], res["soft"], res["rings"], res.get("worst_bonds"))
    cnt = Counter(r["label"] for r in results)
    flag_cnt = Counter(f for r in results for f in set(r.get("hard", []) + r.get("soft", [])))
    summary = dict(n=len(results), labels=dict(cnt), flags=dict(flag_cnt))
    print(ctxio.to_json(dict(summary=summary)))
    if a.out_json:
        with open(a.out_json, "w") as fh:
            fh.write(ctxio.to_json(dict(summary=summary, parents=results)))


if __name__ == "__main__":
    main()
