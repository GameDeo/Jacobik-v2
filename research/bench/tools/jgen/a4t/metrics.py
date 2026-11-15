"""Frozen A4t geometry and decision-list metrics."""
from __future__ import annotations

import itertools
import math
from itertools import islice
from functools import lru_cache
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from tools.runkit import geom

CLASSES = ("F", "T", "G", "G*", "S", "Z", "R", "C")
KNOWN_TYPING = {"MIDQEY", "NUHPEL", "UMUBAF", "YUCZUS", "NUWFAM"}


def _kabsch_rmsd(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    u, _, vt = np.linalg.svd(Ac.T @ Bc)
    fix = np.eye(3); fix[-1, -1] = np.linalg.det(u @ vt)
    R = u @ fix @ vt
    return float(np.sqrt(np.mean(np.sum((Ac @ R - Bc) ** 2, axis=1))))


def _hungarian_perm(A, B, groups, seed_perm=None):
    """Iterate assignment and Kabsch to a fixed point, from one seed alignment."""
    p = np.arange(len(A)) if seed_perm is None else np.asarray(seed_perm, int).copy()
    best = (float("inf"), p.copy())
    for _ in range(20):
        d = A - A.mean(0); e = B[p] - B[p].mean(0)
        u, _, vt = np.linalg.svd(d.T @ e); fix = np.eye(3); fix[-1, -1] = np.linalg.det(u @ vt)
        R = u @ fix @ vt
        moved = (B - B.mean(0)) @ R.T
        new = np.arange(len(A))
        for idx in groups:
            cost = ((A[idx, None, :] - moved[None, idx, :]) ** 2).sum(-1)
            rr, cc = linear_sum_assignment(cost)
            new[idx[rr]] = idx[cc]
        score = _kabsch_rmsd(A, B[new])
        if score < best[0]: best = (score, new.copy())
        if np.array_equal(new, p): break
        p = new
    return best


@lru_cache(maxsize=64)
def _shell_setup(symbols, metal, donors):
    labels=["M"]+[str(symbols[i]) for i in donors]
    groups=[np.asarray([i for i,el in enumerate(labels) if el==e],int) for e in dict.fromkeys(labels)]
    nperm=math.prod(math.factorial(len(g)) for g in groups);limit=min(nperm,40320)
    perms=[]
    nmade=0
    def combinations(k,chosen):
        nonlocal nmade
        if nmade>=limit:return
        if k==len(groups):
            p=np.arange(len(labels))
            for g,perm in zip(groups,chosen):p[g]=perm
            perms.append(p);nmade+=1;return
        # Recurse through permutations lazily. itertools.product would first
        # materialize each group's full factorial, which explodes for e.g. 12 C donors.
        for perm in itertools.permutations(groups[k].tolist()):
            if nmade>=limit:break
            yield from combinations(k+1,chosen+[perm])
    for _ in combinations(0,[]):pass
    return groups,np.asarray(perms,int),nperm>40320


def shell_rmsd(row_X, ref_X, symbols, metal, donors, return_capped=False):
    """Metal+donor RMSD, minimised over within-element donor permutations and Kabsch."""
    ix = [int(metal), *map(int, donors)]
    A, B = np.asarray(row_X, float)[ix], np.asarray(ref_X, float)[ix]
    groups,perms,capped=_shell_setup(tuple(map(str,symbols)),int(metal),tuple(map(int,donors)))
    def batch_min(perms):
        Bp=B[np.asarray(perms,int)]
        Ac=A-A.mean(0); Bp=Bp-Bp.mean(1,keepdims=True)
        H=np.einsum('ni,knj->kij',Ac,Bp,optimize=True)
        u,s,vt=np.linalg.svd(H)
        det=np.linalg.det(u@vt);s[:,-1]*=np.sign(det)
        ss=(Ac**2).sum()+((B-B.mean(0))**2).sum()-2*s.sum(1)
        besti=int(np.argmin(ss))
        return float(np.sqrt(max(0.0,float(ss[besti]))/len(A))),np.asarray(perms[besti],int)
    if not capped:
        best,_=batch_min(perms)
    else:
        _,best_seed=batch_min(perms)
        best,_=_hungarian_perm(A,B,groups,best_seed);capped=True
    return (best, capped) if return_capped else best


def shell_radial_lower_bound(X1, X2, symbols, metal, donors):
    """Permutation/rotation invariant lower bound for shell RMSD (runkit.geom radial bound)."""
    ix=[int(metal),*map(int,donors)]
    A=np.asarray(X1,float)[ix];B=np.asarray(X2,float)[ix]
    classes=["__METAL__"]+[str(symbols[i]) for i in donors]
    A=A-A.mean(0);B=B-B.mean(0)
    ra=np.linalg.norm(A,axis=1);rb=np.linalg.norm(B,axis=1)
    ss=0.0
    for el in dict.fromkeys(classes):
        ia=np.asarray([i for i,x in enumerate(classes) if x==el],int)
        diff=np.sort(ra[ia])-np.sort(rb[ia])
        ss+=float(diff@diff)
    return float(np.sqrt(max(0.0,ss)/len(ix)))


def shape_measure(directions, ideal):
    """Continuous-shape measure (100 x normalized squared residual) over permutations and rotations."""
    D = np.asarray(directions, float); I = np.asarray(ideal, float)
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    I = I / np.maximum(np.linalg.norm(I, axis=1, keepdims=True), 1e-12)
    n = len(D); best = float("inf")
    Dc = D - D.mean(0)
    perms=np.asarray(list(islice(itertools.permutations(range(n)),40320)),int)
    if math.factorial(n) <=40320:
        P=I[perms];Pc=P-P.mean(1,keepdims=True);H=np.einsum('kni,nj->kij',Pc,Dc,optimize=True)
        u,s,vt=np.linalg.svd(H);s[:,-1]*=np.sign(np.linalg.det(u@vt));res=(Pc**2).sum((1,2))+(Dc**2).sum()-2*s.sum(1)
        best=float(100*max(0.0,float(res.min()))/max(float((Dc**2).sum()),1e-12))
    else:
        _,p0=_best_perm_kabsch(I,Dc,perms)
        for start in (p0,np.arange(n)[::-1]):
            p=np.asarray(start,int).copy()
            for _ in range(40):
                P=I[p]-I[p].mean(0);u,_,vt=np.linalg.svd(P.T@Dc);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vt);R=u@fix@vt
                cost=((Dc[:,None,:]-(I-I.mean(0))@R[None,:,:])**2).sum(-1)
                rr,cc=linear_sum_assignment(cost);new=np.empty(n,int);new[rr]=cc
                if np.array_equal(new,p):break
                p=new
            P=I[p]-I[p].mean(0);u,_,vt=np.linalg.svd(P.T@Dc);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vt);R=u@fix@vt
            best=min(best,100*float(((P@R-Dc)**2).sum())/max(float((Dc**2).sum()),1e-12))
    return best


def _best_perm_kabsch(I,Dc,perms):
    P=I[perms];Pc=P-P.mean(1,keepdims=True);H=np.einsum('kni,nj->kij',Pc,Dc,optimize=True)
    u,s,vt=np.linalg.svd(H);s[:,-1]*=np.sign(np.linalg.det(u@vt));res=(Pc**2).sum((1,2))+(Dc**2).sum()-2*s.sum(1)
    i=int(np.argmin(res));return float(res[i]),perms[i]


def csm_by_geometry(ref_X, metal, donors):
    from jacobik.ik.targets import IDEAL_GEOMETRIES
    D = np.asarray(ref_X)[np.asarray(donors, int)] - np.asarray(ref_X)[int(metal)]
    cn = len(D)
    return {name: shape_measure(D, shape) for name, shape in IDEAL_GEOMETRIES.items() if len(shape) == cn}


def decision(rows, top_index, donor_mismatch, geometry_mismatch, any_shell, stereo_mismatch, ring_share,
             code=None, hit_radius=.3, eq_threshold=5.0, geometry_evaluable=True):
    """Apply the frozen A4t ordered decision list; `rows` are all rows in a pool group."""
    if any(bool(r.get("hit")) or (not bool(r.get("hit")) and _finite(r.get("offset")) and float(r["offset"]) <= eq_threshold) for r in rows):
        return "F"
    if donor_mismatch or code in KNOWN_TYPING: return "T"
    if not geometry_evaluable: return "G*"
    if geometry_mismatch: return "G"
    if not any_shell: return "S"
    if stereo_mismatch: return "Z"
    if _finite(ring_share) and float(ring_share) > .5: return "R"
    return "C"


def _finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False


def stereo_graph(mol, symbols, X):
    """Assign ligand and coordination-created stereo from an XYZ geometry on the input molecular graph."""
    from rdkit import Chem
    rw = Chem.RWMol()
    for s in symbols: rw.AddAtom(Chem.Atom(str(s)))
    metal = int(mol.metal_idx) - 1
    for b in mol.bonds:
        i, j = int(b.i) - 1, int(b.j) - 1
        order = float(getattr(b, "order", 1.0))
        # tmQM's inferred bond list can include weak metal contacts (even to H); the
        # builder graph has ligand covalent bonds plus authoritative donor links.
        if i == j or i == metal or j == metal or order < .5 or rw.GetBondBetweenAtoms(i, j): continue
        bt = Chem.BondType.DOUBLE if order >= 1.75 else (Chem.BondType.TRIPLE if order >= 2.75 else Chem.BondType.SINGLE)
        rw.AddBond(i, j, bt)
    donors = sorted({int(i)-1 for lig in mol.ligands for i in lig.donor_atoms})
    for j in donors:
        if str(symbols[j]) == "H": continue
        if rw.GetBondBetweenAtoms(metal, j) is None: rw.AddBond(metal, j, Chem.BondType.SINGLE)
    m = rw.GetMol(); m.UpdatePropertyCache(strict=False); conf = Chem.Conformer(len(symbols))
    for i, xyz in enumerate(np.asarray(X, float)): conf.SetAtomPosition(i, xyz.tolist())
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    atoms = {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}
    bonds = {tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))): b.GetStereo().name
             for b in m.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE and b.GetStereo() != Chem.BondStereo.STEREONONE}
    rings = set()
    for ring in Chem.GetSymmSSSR(m):
        if len(ring) >= 5: rings.update(map(int, ring))
    return atoms, bonds, rings


def stereo_diff(ref_labels, row_labels, row_to_ref=None, metal=None, allow_mirror=True,
                require_reference_centres=False, ignore_undetermined=True):
    """Compare row stereo after atom mapping; optionally ignore metal and accept global inversion."""
    ra,rb,_=ref_labels;xa,xb,_=row_labels
    mapping={} if row_to_ref is None else row_to_ref
    mapped_atoms={mapping.get(int(k),int(k)):v for k,v in xa.items()}
    mapped_bonds={tuple(sorted((mapping.get(int(i),int(i)),mapping.get(int(j),int(j))))):v
                  for (i,j),v in xb.items()}
    ref_atoms={int(k):v for k,v in ra.items() if metal is None or int(k)!=int(metal)}
    row_atoms={int(k):v for k,v in mapped_atoms.items() if metal is None or int(k)!=int(metal)}
    if require_reference_centres and any(k not in row_atoms for k in ref_atoms):return True
    compared=sorted(k for k in ref_atoms if k in row_atoms)
    ref_bonds={tuple(sorted(map(int,k))):v for k,v in rb.items()}
    # RDKit's STEREOANY means the geometry did not determine a specific
    # configuration. Compare a bond only when both structures have a definite
    # stereo label for the same mapped bond.
    if ignore_undetermined:
        for key in set(ref_bonds)|set(mapped_bonds):
            ref_label=ref_bonds.get(key);row_label=mapped_bonds.get(key)
            if ref_label in (None,"STEREOANY") or row_label in (None,"STEREOANY"):
                continue
            if ref_label!=row_label:return True
    elif any(mapped_bonds.get(k)!=v for k,v in ref_bonds.items()):
        return True
    ref_values=[ref_atoms[k] for k in compared]
    row_values=[row_atoms[k] for k in compared]
    if ref_values==row_values:return False
    inverse={"R":"S","S":"R","r":"s","s":"r"}
    return not (allow_mirror and all(inverse.get(a)==b for a,b in zip(ref_values,row_values)))


def stereo_diff_causes(ref_labels, row_labels, row_to_ref, metal, donors):
    """Describe legacy differences, mapping/mirror changes, and Amendment 3 causes."""
    ra,rb,_=ref_labels;xa,xb,_=row_labels
    donor_set=set(map(int,donors))
    categories=set()
    old_center_diff=[]
    if ra.get(int(metal))!=xa.get(int(metal)) and (int(metal) in ra or int(metal) in xa):
        categories.add("metal")
    for k,v in ra.items():
        if xa.get(k)!=v:
            categories.add("metal" if int(k)==int(metal) else ("donor" if int(k) in donor_set else "ligand centre"))
            if int(k)!=int(metal):old_center_diff.append(int(k))
    old_bond_diff=[k for k,v in rb.items() if xb.get(k)!=v]
    if old_bond_diff:categories.add("double bond")
    mapped_atoms={row_to_ref.get(int(k),int(k)):v for k,v in xa.items()}
    ref_centres={int(k):v for k,v in ra.items() if int(k)!=int(metal)}
    row_centres={int(k):v for k,v in mapped_atoms.items() if int(k)!=int(metal)}
    common=sorted(set(ref_centres)&set(row_centres))
    refv=[ref_centres[k] for k in common];rowv=[row_centres[k] for k in common]
    inverse={"R":"S","S":"R","r":"s","s":"r"}
    centres_equal=refv==rowv
    centres_inverted=bool(refv) and all(inverse.get(a)==b for a,b in zip(refv,rowv))
    centre_match=centres_equal or centres_inverted
    mapped_bonds={tuple(sorted((row_to_ref.get(int(i),int(i)),row_to_ref.get(int(j),int(j))))):v
                  for (i,j),v in xb.items()}
    ref_bonds={tuple(sorted(map(int,k))):v for k,v in rb.items()}
    comparable_bonds=[]
    for key in set(ref_bonds)|set(mapped_bonds):
        ref_label=ref_bonds.get(key);row_label=mapped_bonds.get(key)
        if ref_label in (None,"STEREOANY") or row_label in (None,"STEREOANY"):
            continue
        comparable_bonds.append((ref_label,row_label))
    bonds_match=all(ref_label==row_label for ref_label,row_label in comparable_bonds)
    causes=[]
    if "metal" in categories:causes.append("metal exclusion")
    if old_center_diff and centre_match and centres_equal:causes.append("RMSD permutation")
    elif old_center_diff and centre_match and centres_inverted:causes.append("mirror tolerance")
    if old_bond_diff and bonds_match:
        if any(rb.get(k)=="STEREOANY" or xb.get(k)=="STEREOANY" or xb.get(k) is None for k in old_bond_diff):
            causes.append("STEREOANY/absent label ignored")
        else:
            causes.append("RMSD permutation (double-bond labels remapped)")
    remaining=[]
    if not centre_match:
        differing_centres=[k for k in common if ref_centres[k]!=row_centres[k]]
        remaining.append("donor" if any(k in donor_set for k in differing_centres) else "ligand centre")
    if not bonds_match:remaining.append("double bond")
    return sorted(categories),causes,sorted(set(remaining))


def ring_share(row_X, ref_X, symbols, P, rings, return_atom_map=False):
    heavy = np.asarray([i for i, s in enumerate(symbols) if s != "H"], int)
    A, B = np.asarray(row_X)[heavy], np.asarray(ref_X)[heavy]
    # Find the best automorphism, then measure deviation carried by >=5-membered rings.
    perms = [np.asarray(p, int) for p in P]
    P=np.asarray(perms,int);Bp=B[P];Ac=A-A.mean(0);Bp=Bp-Bp.mean(1,keepdims=True)
    H=np.einsum('ni,knj->kij',Ac,Bp,optimize=True);u,s,vt=np.linalg.svd(H);det=np.linalg.det(u@vt);s[:,-1]*=np.sign(det)
    ss=(Ac**2).sum()+((B-B.mean(0))**2).sum()-2*s.sum(1);best_i=int(np.argmin(ss));best_ss=max(0.0,float(ss[best_i]))
    p=P[best_i]
    atom_map={int(heavy[i]):int(heavy[j]) for i,j in enumerate(p)}
    if best_ss <= 1e-16:return (0.0,atom_map) if return_atom_map else 0.0
    bb=B[p];u,_,vt=np.linalg.svd(Ac.T@(bb-bb.mean(0)));fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vt);R=u@fix@vt
    delta=Ac@R-(bb-bb.mean(0))
    ring_heavy_idx={j for j, atom in enumerate(heavy) if int(atom) in rings}
    value=float((delta[sorted(ring_heavy_idx)]**2).sum()/best_ss) if ring_heavy_idx else 0.0
    return (value,atom_map) if return_atom_map else value

