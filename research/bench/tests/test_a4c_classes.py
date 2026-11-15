"""A4c class descriptors on synthetic complexes: one test per descriptor kind (directive_p2r_metric_2026-11-12)."""
import numpy as np
import pytest

from tools.jgen.a4c.classes import Reference
from tools.jgen.a4c import constructs

E = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
D = {"N": 2.0, "Cl": 2.3, "P": 2.3}


def mono(placement, metal="Co"):
    """Metal at the origin with monodentate atoms {site: element}; returns symbols, X (site order sorted by element)."""
    items = sorted(placement.items(), key=lambda kv: (kv[1], kv[0]))
    syms = [metal] + [el for _, el in items]
    X = np.array([[0, 0, 0]] + [np.array(E[s], float) * D[el] for s, el in items])
    return syms, X


def rot(X, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return X @ q.T + np.array([0.3, -1.2, 2.0])


def same(ref_syms, ref_X, syms, X):
    assert syms == ref_syms
    return Reference(ref_syms, ref_X).compare(X)


def test_octahedral_cis_trans():
    trans = mono({"+z": "Cl", "-z": "Cl", "+x": "N", "-x": "N", "+y": "N", "-y": "N"})
    trans2 = mono({"+x": "Cl", "-x": "Cl", "+z": "N", "-z": "N", "+y": "N", "-y": "N"})
    cis = mono({"+z": "Cl", "+x": "Cl", "-z": "N", "-x": "N", "+y": "N", "-y": "N"})
    r = same(*trans, trans2[0], rot(trans2[1]))
    assert r["same_class"] and r["core_rmsd_A"] < 1e-6 and r["hit"]
    assert not same(*trans, *cis)["same_class"]
    assert not same(*cis, *trans)["same_class"]


def test_fac_mer():
    fac = mono({"+x": "Cl", "+y": "Cl", "+z": "Cl", "-x": "N", "-y": "N", "-z": "N"})
    mer = mono({"+x": "Cl", "-x": "Cl", "+y": "Cl", "-y": "N", "+z": "N", "-z": "N"})
    assert same(*fac, fac[0], rot(fac[1], 3))["same_class"]
    assert not same(*fac, *mer)["same_class"] and not same(*mer, *fac)["same_class"]


def test_square_planar_cis_trans():
    trans = mono({"+x": "Cl", "-x": "Cl", "+y": "N", "-y": "N"}, "Pt")
    cis = mono({"+x": "Cl", "+y": "Cl", "-x": "N", "-y": "N"}, "Pt")
    assert same(*trans, trans[0], rot(trans[1], 1))["same_class"]
    assert not same(*trans, *cis)["same_class"]


def tris_chelate(edges):
    """M(N-C-C-N)3 with planar five-membered rings on the given cis edges."""
    syms, X = ["Co"], [np.zeros(3)]
    t = np.radians(71.0)
    for a, b in edges:
        s1, s2 = np.array(E[a], float), np.array(E[b], float)
        n1, n2 = 2.0 * s1, 2.0 * s2
        c1 = n1 + 1.47 * (np.cos(t) * s1 + np.sin(t) * s2)
        c2 = n2 + 1.47 * (np.cos(t) * s2 + np.sin(t) * s1)
        syms += ["N", "C", "C", "N"]; X += [n1, c1, c2, n2]
    return syms, np.array(X)


DELTA = [("+x", "+y"), ("-x", "+z"), ("-y", "-z")]


def test_delta_lambda_is_an_enantiomer_pair():
    d = tris_chelate(DELTA)
    lam = (d[0], d[1] * np.array([-1.0, 1, 1]))
    ref = Reference(*d)
    assert len(ref.sites) == 6 and len(ref.core) == 13
    r = ref.compare(rot(lam[1], 5))
    assert not r["same_class"] and r["enantiomer"] and r["reason"] == "enantiomer"   # enantiomers are distinct classes
    assert not ref.compare(rot(lam[1], 5), energy_offset=0.0)["hit"]
    assert ref.compare(rot(d[1], 6))["mirror"] is False


def tet(centre, back, length):
    """Three tetrahedral substituent directions around `centre`, opposite the bond to `back`."""
    a = centre - back; a /= np.linalg.norm(a)
    p = np.cross(a, [0.3, 0.7, 0.1]); p /= np.linalg.norm(p); q = np.cross(a, p)
    out = []
    for k in range(3):
        ang = 2 * np.pi * k / 3
        d = np.cos(np.radians(70.5)) * a + np.sin(np.radians(70.5)) * (np.cos(ang) * p + np.sin(ang) * q)
        out.append(centre + length * d)
    return out


def chiral_arm(direction, swap=False):
    """M-N-C*(F)(Cl)(Br) on one site; swap=True inverts C*."""
    s = np.array(E[direction], float)
    n = 2.0 * s; c = n + 1.47 * s
    f, cl, br = tet(c, n, 1.0)
    if swap:
        f, cl = cl, f
    return ["N", "C", "F", "Cl", "Br"], [n, c, c + 1.35 * (f - c), c + 1.77 * (cl - c), c + 1.94 * (br - c)]


def two_centres(swap_first=False):
    syms, X = ["Co"], [np.zeros(3)]
    for d, sw in (("+x", swap_first), ("-x", False)):
        s, x = chiral_arm(d, sw); syms += s; X += x
    return syms, np.array(X)


def test_centre_inversion_is_a_different_class():
    ref = Reference(*two_centres())
    assert sum(k == "chi" for k, _ in ref.desc) == 2
    assert ref.compare(rot(two_centres()[1], 2))["same_class"]
    inv = two_centres(swap_first=True)
    assert not ref.compare(inv[1])["same_class"]


def test_constructed_inversion_flips_one_centre():
    syms, X = two_centres()
    H = X[2] + 0.0
    ref = Reference(syms, X)
    c = [g[0][0] for k, g in ref.desc if k == "chi"][0]
    Y = X.copy(); hc = ref.heavy[c]
    # reflect the centre through the plane of its three heavy neighbours: inverts it (no H here: F is the 4th)
    nb = [ref.heavy[x] for x in ref._neighbours(c)]
    n = np.cross(Y[nb[1]] - Y[nb[0]], Y[nb[2]] - Y[nb[0]]); n /= np.linalg.norm(n)
    Y[hc] = Y[hc] - 2 * ((Y[hc] - Y[nb[0]]) @ n) * n
    assert not ref.compare(Y)["same_class"]
    assert np.allclose(H, X[2])


def vinyl(z):
    """M-N=... monodentate N-C1(H)=C2(H)-Br, Br cis (Z) or trans (E) to N."""
    n = np.array([2.0, 0, 0]); c1 = n + np.array([1.40, 0, 0])
    u = np.array([np.cos(np.radians(60)), np.sin(np.radians(60)), 0])
    c2 = c1 + 1.34 * u
    side = np.array([np.cos(np.radians(120 if z else 0)), np.sin(np.radians(120 if z else 0)), 0])
    other = np.array([np.cos(np.radians(0 if z else 120)), np.sin(np.radians(0 if z else 120)), 0])
    br = c2 + 1.89 * side; h2 = c2 + 1.08 * other
    h1 = c1 + 1.08 * np.array([np.cos(np.radians(-60)), np.sin(np.radians(-60)), 0])
    return ["Co", "N", "C", "C", "Br", "H", "H"], np.array([[0, 0, 0], n, c1, c2, br, h1, h2])


def test_double_bond_E_Z():
    ref = Reference(*vinyl(True))
    assert [k for k, _ in ref.desc].count("ez") == 1
    assert ref.compare(rot(vinyl(True)[1], 4))["same_class"]
    assert not ref.compare(vinyl(False)[1])["same_class"]


def bound_n(swap):
    """Binding-created N centre: M-N(C)(F)(Cl) plus a fixed C* arm on -x."""
    syms, X = ["Co"], [np.zeros(3)]
    n = np.array([2.0, 0, 0]); a, b, c = tet(n, np.zeros(3), 1.0)
    if swap:
        a, b = b, a
    syms += ["N", "C", "F", "Cl"]; X += [n, n + 1.47 * (a - n), n + 1.36 * (b - n), n + 1.75 * (c - n)]
    s, x = chiral_arm("-x"); syms += s; X += x
    return syms, np.array(X)


def test_binding_created_centre():
    ref = Reference(*bound_n(False))
    chi = [g[0][0] for k, g in ref.desc if k == "chi"]
    assert ref.hpos[1] in chi                              # the bound N is a centre only because of the M-N bond
    assert not ref.compare(bound_n(True)[1])["same_class"]


def butyl():
    syms = ["Co", "N", "C", "C", "C", "C", "Cl", "Cl"]
    X = np.array([[0, 0, 0], [2.0, 0, 0], [3.47, 0, 0], [4.0, 1.4, 0], [5.5, 1.5, 0.3], [6.1, 2.9, 0.5],
                  [-2.3, 0, 0], [0, 2.3, 0]])
    return syms, X


def test_peripheral_rotor_keeps_class_and_core():
    syms, X = butyl()
    ref = Reference(syms, X)
    rotors = constructs.peripheral_rotors(ref)
    assert rotors and all(b not in {int(ref.heavy[i]) for i in ref.core} for a, b, side in rotors)
    Y, applied = constructs.rotor_variant(ref, X, n_rotors=1)
    assert applied and np.abs(Y - X).max() > 0.5
    r = ref.compare(Y)
    assert r["same_class"] and r["core_rmsd_A"] < 1e-6 and r["hit_by"] == "rmsd"


def test_energy_path_needs_the_same_class():
    trans = mono({"+z": "Cl", "-z": "Cl", "+x": "N", "-x": "N", "+y": "N", "-y": "N"})
    cis = mono({"+z": "Cl", "+x": "Cl", "-z": "N", "-x": "N", "+y": "N", "-y": "N"})
    ref = Reference(*trans)
    assert not ref.compare(cis[1], energy_offset=-20.0)["hit"]     # lower energy, other class: never a hit
    far = trans[1] * 1.0; far[1:] *= 1.2                            # same class, core moved > 0.3 A
    r = ref.compare(far, energy_offset=3.0)
    assert r["same_class"] and r["core_rmsd_A"] > 0.3 and r["hit_by"] == "energy"
    assert not ref.compare(far, energy_offset=8.0)["hit"]


def test_topology_change_is_not_the_same_class():
    syms, X = butyl()
    ref = Reference(syms, X)
    Y = X.copy(); Y[5] = Y[2] + np.array([0.0, 0.0, 1.5])            # C4 now bonded to C1
    assert not ref.compare(Y)["topology_same"] and not ref.compare(Y)["same_class"]


def test_contact_change_needs_a_real_shift():
    from tools.runkit import geom
    syms, X = mono({"+z": "Cl", "-z": "Cl", "+x": "N", "-x": "N", "+y": "N", "-y": "N"})
    cut = geom.rcov("Co") + geom.rcov("Cl") + 0.6
    X = X.copy(); X[1] = np.array([0, 0, 1.0]) * (cut - 0.01)               # one Cl just inside the contact cutoff
    ref = Reference(syms, X)
    Y = X.copy(); Y[1] = np.array([0, 0, 1.0]) * (cut + 0.02)               # noise across the cutoff: same class
    assert ref.compare(Y)["same_class"]
    Y[1] = np.array([0, 0, 1.0]) * (cut + 0.5)                              # dissociated by 0.5 A: contact change
    assert not ref.compare(Y)["donors_same"] and not ref.compare(Y)["same_class"]


def test_bound_thioether_is_not_a_centre_but_bound_phosphine_is():
    for el, expect in (("S", False), ("P", True)):
        syms, X = ["Co"], [np.zeros(3)]
        d = np.array([2.3, 0, 0]); a, b, c = tet(d, np.zeros(3), 1.0)
        syms += [el, "C", "C"]; X += [d, d + 1.82 * (a - d), d + 1.82 * (b - d)]
        if el == "P":
            syms += ["F"]; X += [d + 1.6 * (c - d)]
        s, x = chiral_arm("-x"); syms += s; X += x
        syms[2] = "C"; ref = Reference(syms, np.array(X))
        centres = {g[0][0] for k, g in ref.desc if k == "chi"}
        assert (ref.hpos[1] in centres) is expect
