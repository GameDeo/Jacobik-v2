"""Plan U1 / U2 (2026-10-16 d): thermal width floors and the u1 profile (one objective for solve and ranking)."""
import math

import numpy as np
import torch


def test_cos_floor_var_is_second_order_exact_at_linear_angles():
    """Var(cos t) for t ~ N(t0, s^2): sin^2 t0 s^2 at 90 deg, s^4 / 2 at 0 / 180 deg (where d cos / dt = 0), and the
    formula matches a Monte Carlo estimate at an intermediate angle."""
    from jacobik.ik.scorer import _cos_floor_var

    s = math.radians(4.5)
    assert abs(float(_cos_floor_var(torch.tensor(0.0), s)) - s ** 2) < 1e-9
    assert abs(float(_cos_floor_var(torch.tensor(-1.0), s)) - 0.5 * s ** 4) < 1e-12
    t0 = math.radians(150.0)
    t = np.random.default_rng(0).normal(t0, s, 400_000)
    assert abs(np.var(np.cos(t)) / float(_cos_floor_var(np.array(math.cos(t0)), s)) - 1.0) < 0.02


def test_thermal_floor_acts_only_below_the_thermal_width(monkeypatch):
    """solve_floor='thermal': a component narrower than sigma_T is widened to it, a broader one is untouched; Q2 floors
    depend on the component's mean (cos); legacy floor modes are unchanged."""
    import jacobik.ik.scorer as S

    monkeypatch.setitem(S.THERMAL_SD, "Q1", 0.06)
    monkeypatch.setitem(S.THERMAL_SD, "Q2", math.radians(4.5))
    v = torch.tensor([1e-5, 0.0036, 0.01])
    out = S._floored("Q1", v, torch.tensor([2.0, 2.1, 2.2]), "thermal")
    assert torch.allclose(out, torch.tensor([0.0036, 0.0036, 0.01]))
    mu = torch.tensor([0.0, -0.999])
    q2 = S._floored("Q2", torch.tensor([1e-6, 1e-6]), mu, "thermal")
    assert q2[0] > 100 * q2[1] and torch.allclose(q2, S._cos_floor_var(mu, math.radians(4.5)).float())
    assert math.isclose(float(S._floored("Q1", torch.tensor([1e-5]), None, "scorer")), S._GMM_VAR_MIN["Q1"],
                        rel_tol=1e-6)


def test_loewner_floor_is_minimal_and_keeps_broad_covariances():
    """S_orient: C -> the smallest C' >= diag(sd^2) (Loewner order). Broad covariances are returned unchanged; a needle
    direction is widened to the floor and only that direction changes."""
    from jacobik.ik.scorer import _loewner_floor

    sd = np.array([0.05, 0.05])
    broad = np.array([[0.01, 0.002], [0.002, 0.02]])
    assert np.allclose(_loewner_floor(broad, sd), broad)
    needle = np.array([[0.01, 0.0], [0.0, 1e-6]])
    out = _loewner_floor(needle, sd)
    assert np.allclose(out, np.diag([0.01, 0.0025]))
    Di = np.diag(1 / sd)
    assert np.linalg.eigvalsh(Di @ _loewner_floor(np.array([[0.01, 0.0099], [0.0099, 0.01]]), sd) @ Di).min() > 1 - 1e-9


def test_m13_tables_are_core_only():
    """Metal 1-3 class (plan U4): the fitted table is 0 at and beyond the 0.1 % onset (the data region belongs to the
    M-D-X angle statistics) and rises steeply below it; keys below N_MIN pairs get no table (pooled key instead)."""
    from jacobik.stats import q6pmf as Q

    rng = np.random.default_rng(0)
    d = rng.normal(2.9, 0.15, 20_000)
    h = np.histogram(d, bins=Q.EDGES)[0].astype(float)
    tabs = Q.fit_tables({"H_W|m13|": h, "H_Mx|m13|": h * 3, "Pt_Se|m13|": h[:] * (100 / h.sum()), "H_W|m13h|": h})
    assert f"Q6P:{('H', 'W', 'm13h', '')!r}" in tabs
    e = tabs[f"Q6P:{('H', 'W', 'm13', '')!r}"]
    grid = np.round(np.arange(Q.R_LO, Q.R_HI + 1e-9, Q.DR_TAB), 4)
    w = np.array(e["w"])
    assert np.all(w[grid >= e["r0"]] == 0.0)
    assert 2.3 < e["r0"] < 2.6
    assert w[np.searchsorted(grid, e["r0"] - 0.5)] > 5.0          # > 5 kT half an angstrom inside the onset
    assert np.all(np.diff(w[grid < e["r0"]]) <= 0)                  # monotone core
    assert f"Q6P:{('Pt', 'Se', 'm13', '')!r}" not in tabs and f"Q6P:{('H', 'Mx', 'm13', '')!r}" in tabs


def test_m13_walls_match_the_q6p5_wall_fit():
    """The metal 1-3 walls (from 0.02 A histograms) reproduce fitter.fit_halfgauss (the q6p5 wall form) on the raw
    distances within the histogram resolution."""
    from jacobik.stats import q6pmf as Q
    from jacobik.stats.fitter import fit_halfgauss

    d = np.random.default_rng(1).gamma(30.0, 0.1, 50_000)  # skewed, ~3 A
    ref = fit_halfgauss(d)
    w = Q.fit_m13_walls({"H_Zr|m13h|": np.histogram(d, bins=Q.EDGES)[0].astype(float)})
    e = w[f"Q6P:{('H', 'Zr', 'm13h', 'wall')!r}"]
    assert abs(e["mu"] - ref["mu"]) < 0.02 and abs(e["sigma"] - ref["sigma"]) < 0.02


def test_bound_surrogate_caps_group13_onium_donors_only():
    """KIQYEQ's boryl written as B=[O+] is embedded as H-Bpin (B with three bonds, O neutral) and placed back on the
    ligand as written (cap dropped, same atoms); CO ([C-]#[O+]) and ordinary donors are not rewritten."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from jacobik.chelate_embed import bound_surrogate, map_surrogate_positions

    smi = "CC1(C)OB=[O+]C1(C)C"
    sur, done = bound_surrogate(smi, [[4]])
    assert done == [4]
    b = sur.GetAtomWithIdx(4)
    assert b.GetTotalNumHs() == 1 and all(n.GetFormalCharge() == 0 for n in b.GetNeighbors())
    sh = Chem.AddHs(sur)
    AllChem.EmbedMolecule(sh, randomSeed=1)
    orig = Chem.AddHs(Chem.MolFromSmiles(smi))
    X = map_surrogate_positions(orig, sh, sh.GetConformer().GetPositions())
    assert X.shape == (orig.GetNumAtoms(), 3) and orig.GetNumAtoms() == sh.GetNumAtoms() - 1
    assert bound_surrogate("[C-]#[O+]", [[0]]) is None and bound_surrogate("c1ccncc1", [[3]]) is None


def test_ez_enumeration_keeps_given_stereo_and_chelate_rule():
    """D6 (owner rule): only configurations the SMILES leaves open are enumerated; a bond given as E or Z in the SMILES
    and a bond the chelate-ring rule decides are never enumerated."""
    from rdkit import Chem

    from jacobik.chelate_embed import chelate_ring_stereo, ez_variants, undecided_ez

    # CBTYCO-like exocyclic imine, off the donor-donor path: undecided -> two combinations
    smi = "COC(N=c1[nH]c2ccccc2s1)C1=CC=C=C(C)[N-]1"
    assert len(undecided_ez(Chem.MolFromSmiles(smi), [[3], [19]])) == 1
    assert len(ez_variants(smi, [[3], [19]])) == 2
    # the same bond written with its configuration: nothing to enumerate
    m = Chem.MolFromSmiles("C/C=N/c1ccccc1")
    assert undecided_ez(m, [[2]]) == [] and ez_variants("C/C=N/c1ccccc1", [[2]]) == [[]]
    # hydrazone in a chelate ring (VUYWIX): decided by the chelate rule, not enumerated
    smi = "O=C(NN=Cc1ccccn1)c1ccccn1"
    m = Chem.MolFromSmiles(smi)
    assert chelate_ring_stereo(m, [[10], [3]]) and undecided_ez(m, [[10], [3]]) == []


def test_u1_profile_sets_one_objective():
    """The u1 profile (default): summed aggregation in both, thermal floors in both, purely repulsive Q6, and NCI on."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import jacobik.profile as P

    src = str(Path(P.__file__).resolve().parents[1])
    code = ("import json, jacobik.profile as P, jacobik.ik.scorer as S, jacobik.ik.ik_solver as I, jacobik.stats.nci as N; "
            "print(json.dumps([S.SCORE_AGG, S.RANK_AGG, S.GMM_FLOOR_MODE, I._RANK_FLOOR, S.Q1_TAIL_SIGMA, "
            "I._BEND_PER_Z, __import__('jacobik.ik.angle_bends', fromlist=['x']).HAPTIC_LEAN_ALL, "
            "__import__('jacobik.chelate_embed', fromlist=['x']).EZ_ENUMERATE, "
            "__import__('jacobik.chelate_embed', fromlist=['x']).BOUND_SURROGATE, S.Q6_MODEL, "
            "N.NCI_ON, list(N.ACTIVE_TYPES), P.snapshot()['differs_from_profile']]))")
    env = {k: v for k, v in os.environ.items() if k not in P.RETIRED_ENV and k != "JACOBIK_PROFILE_OVERRIDES"}
    env.update(PYTHONPATH=src + os.pathsep + env.get("PYTHONPATH", ""), JACOBIK_PROFILE="u1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [
        "sum", "sum", "thermal", False, 0.0, True, True, True, True, "repulsive", True, ["hb"], {}
    ]


def test_q6_pure_repulsive_core_has_no_attractive_wells():
    """Purely repulsive Q6 (2026-10-18): non-polar and polar-H pairs cost 0 at and beyond their onset r0 whatever the
    table says there, and only the repulsive core below r0."""
    import torch

    from jacobik.stats import q6pmf as Q

    grid = torch.arange(Q.R_LO, Q.R_HI + 1e-9, Q.DR_TAB, dtype=torch.float64)
    r0 = 2.0
    data = torch.where(grid >= r0, 0.3 * torch.sin(4 * grid) - 1.0, torch.zeros_like(grid))
    core = torch.where(grid < r0, data[int(round((r0 - Q.R_LO) / Q.DR_TAB))] + (torch.exp(4.0 * (r0 - grid)) - 1.0), 0.0)
    T = (data + core).float()[None].repeat(2, 1)
    t = lambda v, dt=torch.long: torch.tensor(v, dtype=dt)
    rt = dict(T=T, tot=t([0, 1]), has=t([True, True], torch.bool), r_lo=Q.R_LO, dr=Q.DR_TAB,
              G=T.shape[1], mu=t([3.0, 3.0], torch.float32), sigma=t([0.2, 0.2], torch.float32),
              r0=t([r0, r0], torch.float32))
    a, b = t([0, 2]), t([1, 3])

    def pair_w(dist):
        X = torch.zeros(1, 4, 3)
        X[0, 1, 0] = dist
        X[0, 3, 0] = dist
        X[0, 2, 1] = 50.0
        X[0, 3, 1] = 50.0
        w = [Q.evaluate(dict(rt, has=t([i == 0, i == 1], torch.bool), mu=t([0.0, 0.0], torch.float32)), X, a, b)
             for i in (0, 1)]
        return float(w[0]), float(w[1])

    for d in (2.0, 2.4, 3.5, 4.5):
        assert pair_w(d) == (0.0, 0.0)                     # purely repulsive: 0 at and beyond the onset for all pairs
    assert pair_w(1.8)[0] > 1.0 and pair_w(1.8)[1] > 1.0   # repulsive core below r0


def test_q6_repulsive_form():
    """q6_pmf_wells='repulsive' (2026-10-17 c, plan 3 N1): the physical exchange repulsion (shifted force) on its pairs:
    0 at and beyond the contact distance r_m with zero force there, positive and decreasing below, linear below r_lin;
    pairs without the flag keep the contact form (core below the onset only)."""
    import math

    import torch

    from jacobik.stats import q6pmf as Q

    A, b, rm, rl = Q.rep_pair_params("C", "H")
    assert abs(rm - (Q.ALVAREZ_RADII["C"] + Q.ALVAREZ_RADII["H"])) < 1e-12
    assert Q.rep_pair_params("Hp", "C") == Q.rep_pair_params("H", "C") and Q.rep_pair_params("C", "Xx") is None
    d = torch.tensor([0.5, rl - 0.05, rl + 0.05, 2.0, 2.6, rm - 1e-4, rm, rm + 0.3], dtype=torch.float64, requires_grad=True)
    u = Q.rep_energy(d, A, b, rm, rl)
    (g,) = torch.autograd.grad(u.sum(), d)
    u, g = u.detach(), g.detach()
    assert u[-2] == 0.0 and u[-1] == 0.0 and g[-1] == 0.0
    assert 0.0 < float(u[-3]) < 1e-6 and abs(float(g[-3])) < 1e-3          # zero value and force at r_m
    assert all(float(u[k]) > float(u[k + 1]) for k in range(5))              # monotone repulsion below r_m
    assert abs(float(u[3]) - A * (math.exp(-b * 2.0) - math.exp(-b * rm) + b * math.exp(-b * rm) * (2.0 - rm)) / Q.RT_KJ) < 1e-9
    if rl > 0:                                                               # linear continuation, continuous
        assert abs(float(Q.rep_energy(torch.tensor(rl, dtype=torch.float64), A, b, rm, rl)) - Q.W_LINEAR_ABOVE) < 1e-6
        assert abs(float(g[0]) - float(g[1])) < 1e-6 * abs(float(g[0]))


def test_bvs_runtime_ladder_and_term():
    """Bond-valence term (plan 3 S2, 2026-10-17 c): R0 per site down the (metal, el) -> (period-group, el) -> (el)
    ladder, level (metal, OS) first, the metal-only level without an OS; no term with a haptic site, without an R0, or
    where the level has no coupling (c = 0); the value is -c / 2 (V / ref - mu)^2."""
    import math
    from types import SimpleNamespace

    import torch

    import jacobik.ik.scorer as S
    from jacobik.stats.key_builder import OS_UNKNOWN

    sites = [SimpleNamespace(kind="sigma") for _ in range(3)]
    mol = SimpleNamespace(coord_sites=sites, metal_symbol="Ti")
    keys = [("Ti", 1, 4, "O_alkoxide", "tetrahedral", "none")] * 2 + [("Ti", 1, 4, "Hal_Cl", "tetrahedral", "none")]
    qvr = {("el", "Ti", "O"): dict(R0=1.80), ("*", "Cl"): dict(R0=2.10)}
    qv = {("mo", "Ti", 4): dict(c=100.0, mu=1.0, v_ref=None), ("m", "Ti"): dict(c=50.0, mu=1.0, v_ref=3.5)}
    sites_i, r0, c, mu, ref = S._bvs_runtime(mol, keys, qv, qvr, "cpu")
    assert r0.tolist() == [1.8, 1.8, 2.1] or torch.allclose(r0, torch.tensor([1.8, 1.8, 2.1]))
    assert (c, mu, ref) == (100.0, 1.0, 4.0)
    keys_u = [k[:2] + (OS_UNKNOWN,) + k[3:] for k in keys]
    assert S._bvs_runtime(mol, keys_u, qv, qvr, "cpu")[2:] == (50.0, 1.0, 3.5)          # metal-only level
    assert S._bvs_runtime(mol, keys, {("mo", "Ti", 4): dict(c=0.0, mu=1.0)}, qvr, "cpu") is None
    assert S._bvs_runtime(mol, keys, qv, {("el", "Ti", "O"): dict(R0=1.8)}, "cpu") is None  # Cl without R0
    hap = SimpleNamespace(coord_sites=sites[:2] + [SimpleNamespace(kind="haptic")], metal_symbol="Ti")
    assert S._bvs_runtime(hap, keys, qv, qvr, "cpu") is None
    d = torch.tensor([1.9, 1.9, 2.2])
    x = float(torch.exp((r0 - d) / S.BVS_B).sum()) / ref
    assert abs(x - (2 * math.exp(-0.1 / 0.37) + math.exp(-0.1 / 0.37)) / 4.0) < 1e-6


def test_nci_features_reference_and_taper():
    """NCI term (plan 3 N2, 2026-10-17 c): features are the stated geometry (H-bond distance / D-H...A angle; ring
    centroid distance / inter-plane angle; H...centroid distance / angle to the normal); a histogram of RANDOM placement
    (counts = bin volumes) fits to s = 0 (the reference is right); s is 0 beyond the taper and the r range."""
    import math

    import numpy as np
    import torch

    from jacobik.stats import nci as N

    # H-bond: D at origin, H at (1, 0, 0), A at 1 + 2 cos-free placement -> r 2.0, D-H...A angle 180 / 90
    X = torch.tensor([[[0, 0, 0], [1, 0, 0], [3, 0, 0], [1, 2, 0]]], dtype=torch.float64)
    idx = dict(d=torch.tensor([0, 0]), h=torch.tensor([1, 1]), a=torch.tensor([2, 3]))
    r, a = N.features("hb", X, idx)
    assert torch.allclose(r, torch.tensor([[2.0, 2.0]], dtype=torch.float64)) and abs(float(a[0, 0]) - 180) < 0.1  # acos clamp (finite gradients)
    assert abs(float(a[0, 1]) - 90) < 1e-6
    # two parallel hexagons 3.5 A apart (pi-pi), one tilted by 40 deg; an H 2.5 A above the first ring's centre
    hexa = [[math.cos(k * math.pi / 3) * 1.4, math.sin(k * math.pi / 3) * 1.4, 0.0] for k in range(6)]
    up = [[x, y, z + 3.5] for x, y, z in hexa]
    t = math.radians(40)
    tilt = [[x, y * math.cos(t), y * math.sin(t) + 7.0] for x, y, z in hexa]
    X = torch.tensor([hexa + up + tilt + [[0.0, 0.0, -2.5]]], dtype=torch.float64)
    rings = torch.tensor([list(range(6)), list(range(6, 12)), list(range(12, 18))])
    r, a = N.features("pp", X, dict(rings=rings, ri=torch.tensor([0, 0]), rj=torch.tensor([1, 2])))
    assert abs(float(r[0, 0]) - 3.5) < 1e-9 and float(a[0, 0]) < 0.1 and abs(float(a[0, 1]) - 40) < 1e-3
    r, a = N.features("chp", X, dict(rings=rings, ring=torch.tensor([0]), h=torch.tensor([18])))
    assert abs(float(r[0, 0]) - 2.5) < 1e-9 and float(a[0, 0]) < 0.1
    # random placement -> s = 0 everywhere (up to the pseudo-count, which only pulls toward 0)
    for typ in N.TYPES:
        e = N.fit_table(typ, N.bin_volumes(typ) * 1e6)
        assert np.abs(np.array(e["s"])).max() < 1e-3
    # an enriched bin -> s > 0 there, 0 beyond the taper end
    h = N.bin_volumes("hb") * 1e5
    h[10, 30] *= 20  # r ~1.9 A, angle ~152 deg
    e = N.fit_table("hb", h)
    s = np.array(e["s"])
    assert s[10, 30] > 1.0
    rc = e["r_lo"] + e["dr"] * (np.arange(s.shape[0]) + 0.5)
    assert np.all(s[rc >= N.SPEC["hb"]["win"][1]] == 0.0)
    T = torch.tensor(s[None], dtype=torch.float32)
    val = N._bilinear(T, torch.tensor([0]), torch.tensor([[3.7]]), torch.tensor([[150.0]]), N._grid("hb"))
    assert float(val) == 0.0
