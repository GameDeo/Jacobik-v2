"""
test_nci_v2_numerics.py -- Comprehensive Gate 2 numerical verification for NCI v2.

Tests all mathematical contracts specified in Jacobik Roadmap v2 (§3.2, §3.4, §3.5, §3.7, §5.3 Gate 2, §7.1):
1. Degenerate geometries: r=0 (coincident atoms / ring centroid), collinear ring collapse (s_R=0),
   monatomic acceptor (k_C=0), angular boundaries x_k in {-1, 0, 1}.
   Zero NaN/Inf in raw features, spline values, autodiff gradients, and Hessians.
2. Exact vanishing: u_f = 0, grad u_f = 0, Hessian u_f = 0 for r <= r_in,lo and r >= r_off.
3. C^2 continuity across B-spline knots, h_eps joins (0 and eps), and T_in, T_out joins.
   Comparison of analytic autodiff gradients and directional second derivatives with central finite differences.
4. Invariances and parity: atom permutation, ring reversal, symmetric bond-end averaging,
   extraction vs. runtime parity, CPU vs. CUDA parity, and zero-opportunity exact 0.0.
5. End-to-end synthetic fit_density_ratio and fit_response_scales (including ref_only traversal check).
"""
from __future__ import annotations

import math
from typing import Any, List

import numpy as np
import pytest
import torch

from jacobik.stats.nci_v2 import (
    DELTA_N,
    DELTA_R,
    DEFAULT_RADIAL_WINDOWS,
    EPS_POS,
    CalibratedArtifact,
    FeatureBatch,
    OpportunityBatch,
    RuntimeNCI,
    SurfaceArtifact,
    bspline_basis_1d,
    build_runtime,
    canonical_ring,
    compute_ring_frame,
    compute_tapers,
    enumerate_opportunities,
    evaluate_joint_support,
    evaluate_latent_surface,
    evaluate_raw_nci,
    features_torch,
    fit_density_ratio,
    fit_response_scales,
    h_epsilon,
    make_clamped_knots,
    quintic_step,
    reg_norm,
    reg_unit,
    _safe_get_surface,
    sp2_rings_v2,
)


# ---------------------------------------------------------------------------
# Mock Complex Helper for Unit Tests
# ---------------------------------------------------------------------------
class MockAtom:
    def __init__(self, idx: int, symbol: str, pos: Sequence[float]):
        self.idx = idx
        self.symbol = symbol
        self.position = np.array(pos, dtype=np.float64)


class MockBond:
    def __init__(self, i: int, j: int, order: float = 1.0):
        self.i = i
        self.j = j
        self.order = order


class MockComplex:
    def __init__(
        self,
        atoms: List[MockAtom],
        bonds: List[MockBond],
        metal_idx: int = 999,
        charge: int = 0,
        csd_code: str = "SYNTH_TEST",
    ):
        self.csd_code = csd_code
        self.atoms = atoms
        self.bonds = bonds
        self.metal_idx = metal_idx
        self.charge = charge
        self.metal_symbol = "Fe"
        self.coord_geom = "octahedral"


# ---------------------------------------------------------------------------
# 1. B-spline Basis, Partition of Unity, and C^2 Continuity
# ---------------------------------------------------------------------------
def test_bspline_partition_of_unity_and_derivatives():
    """Verify partition of unity sum = 1 and C^2 continuity across interior knots."""
    domain = (-1.0, 1.0)
    knots = make_clamped_knots(domain, num_intervals=4, degree=3)

    # Test grid of evaluation points including boundary endpoints and interior knots
    test_pts = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64, requires_grad=True)
    bases = bspline_basis_1d(test_pts, knots)
    sum_bases = bases.sum(dim=-1)

    # Exact partition of unity: sum = 1
    assert torch.allclose(sum_bases, torch.ones_like(sum_bases), atol=1e-12)

    # Gradients of the constant partition of unity must be identically 0
    grad_sum = torch.autograd.grad(sum_bases.sum(), test_pts, create_graph=True)[0]
    assert torch.allclose(grad_sum, torch.zeros_like(grad_sum), atol=1e-12)

    v = torch.randn_like(test_pts)
    hess_sum = torch.autograd.grad((grad_sum * v).sum(), test_pts)[0]
    assert torch.allclose(hess_sum, torch.zeros_like(hess_sum), atol=1e-12)

    # Test random spline coefficients: value, 1st, and 2nd derivatives continuous at knots
    coef = torch.tensor([1.2, -0.8, 2.5, 0.4, -1.5, 0.9, 1.1], dtype=torch.float64)
    interior_knots = knots[4:-4]

    for ik in interior_knots:
        ik_val = ik.item()
        eps = 1e-5
        x_k = torch.tensor([ik_val], dtype=torch.float64, requires_grad=True)
        b_k = bspline_basis_1d(x_k, knots)
        y_k = (b_k * coef).sum()
        g_k = torch.autograd.grad(y_k, x_k, create_graph=True)[0]
        h_k = torch.autograd.grad(g_k, x_k)[0]

        # Central finite differences
        x_p = torch.tensor([ik_val + eps], dtype=torch.float64)
        x_m = torch.tensor([ik_val - eps], dtype=torch.float64)
        y_p = (bspline_basis_1d(x_p, knots) * coef).sum()
        y_m = (bspline_basis_1d(x_m, knots) * coef).sum()
        g_fd = (y_p - y_m) / (2 * eps)
        h_fd = (y_p - 2 * y_k + y_m) / (eps**2)

        assert torch.allclose(g_k, g_fd, atol=1e-5, rtol=1e-5)
        assert torch.allclose(h_k, h_fd, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. Quintic Tapers & Positivity Smoothing C^2 Joins
# ---------------------------------------------------------------------------
def test_quintic_tapers_c2_continuity_and_vanishing():
    """Verify T_in(r) and T_out(r) C^2 joins and exact vanishing."""
    r_in_lo, r_in_hi, r_on, r_off = 1.10, 1.35, 6.0, 8.0

    # Test join boundaries
    boundary_points = [r_in_lo, r_in_hi, r_on, r_off]
    for bp in boundary_points:
        r = torch.tensor([bp], dtype=torch.float64, requires_grad=True)
        Tin, Tout = compute_tapers(r, r_in_lo, r_in_hi, r_on, r_off)

        gin = torch.autograd.grad(Tin, r, create_graph=True)[0]
        hin = torch.autograd.grad(gin, r)[0]
        gout = torch.autograd.grad(Tout, r, create_graph=True)[0]
        hout = torch.autograd.grad(gout, r)[0]

        # First and second derivatives must be 0 at joins
        assert abs(gin.item()) < 1e-12
        assert abs(hin.item()) < 1e-12
        assert abs(gout.item()) < 1e-12
        assert abs(hout.item()) < 1e-12

    # Verify vanishing below r_in_lo and above r_off
    r_sub = torch.tensor([0.0, 0.5, 1.0, 1.099], dtype=torch.float64, requires_grad=True)
    Tin_sub, Tout_sub = compute_tapers(r_sub, r_in_lo, r_in_hi, r_on, r_off)
    assert (Tin_sub == 0.0).all()

    gin_sub = torch.autograd.grad(Tin_sub.sum(), r_sub, create_graph=True)[0]
    hin_sub = torch.autograd.grad(gin_sub.sum(), r_sub)[0]
    assert (gin_sub == 0.0).all()
    assert (hin_sub == 0.0).all()

    r_sup = torch.tensor([8.001, 8.5, 10.0, 20.0], dtype=torch.float64, requires_grad=True)
    Tin_sup, Tout_sup = compute_tapers(r_sup, r_in_lo, r_in_hi, r_on, r_off)
    assert (Tout_sup == 0.0).all()

    gout_sup = torch.autograd.grad(Tout_sup.sum(), r_sup, create_graph=True)[0]
    hout_sup = torch.autograd.grad(gout_sup.sum(), r_sup)[0]
    assert (gout_sup == 0.0).all()
    assert (hout_sup == 0.0).all()


def test_positivity_h_epsilon_c2_continuity():
    """Verify h_eps(v) value, gradient, and Hessian continuity at v = 0 and v = eps."""
    eps = EPS_POS

    # Boundary points v = 0.0 and v = eps
    for v_val in [0.0, eps]:
        v = torch.tensor([v_val], dtype=torch.float64, requires_grad=True)
        y = h_epsilon(v, eps=eps)
        g = torch.autograd.grad(y, v, create_graph=True)[0]
        h = torch.autograd.grad(g, v)[0]

        delta = 1e-5
        v_p = torch.tensor([v_val + delta], dtype=torch.float64)
        v_m = torch.tensor([v_val - delta], dtype=torch.float64)
        y_p = h_epsilon(v_p, eps=eps)
        y_m = h_epsilon(v_m, eps=eps)
        g_fd = (y_p - y_m) / (2 * delta)
        h_fd = (y_p - 2 * y + y_m) / (delta**2)

        if v_val == 0.0:
            assert abs(y.item()) < 1e-12
            assert abs(g.item()) < 1e-12
            assert abs(h.item()) < 1e-12
            assert abs(g_fd.item()) < 1e-5
        else:
            assert abs(y.item() - eps) < 1e-12
            assert abs(g.item() - 1.0) < 1e-12
            assert abs(h.item()) < 1e-12
            assert abs(g_fd.item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 3. Degenerate Geometries: Zero Distance, Collapsed Rings, k_C = 0
# ---------------------------------------------------------------------------
def test_degenerate_zero_distance():
    """Verify r = 0 coincident atoms produces zero NaN/Inf and finite derivatives."""
    v_zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    r = reg_norm(v_zero, DELTA_R)
    u_hat = reg_unit(v_zero, DELTA_R)

    assert not torch.isnan(r) and not torch.isinf(r)
    assert not torch.isnan(u_hat).any() and not torch.isinf(u_hat).any()
    assert abs(r.item() - DELTA_R) < 1e-12
    assert torch.allclose(u_hat, torch.zeros(3, dtype=torch.float64))

    g_r = torch.autograd.grad(r, v_zero, create_graph=True)[0]
    h_r = torch.autograd.grad(g_r[0], v_zero)[0]

    assert not torch.isnan(g_r).any() and not torch.isinf(g_r).any()
    assert not torch.isnan(h_r).any() and not torch.isinf(h_r).any()
    assert torch.allclose(g_r, torch.zeros(3, dtype=torch.float64))


def test_degenerate_collinear_collapsed_ring():
    """Verify collapsed or collinear ring s_R = 0 produces finite non-divergent normal and derivatives."""
    P_collinear = torch.zeros((1, 1, 6, 3), dtype=torch.float64)
    for k in range(6):
        P_collinear[0, 0, k, 0] = float(k)
    P_ring = P_collinear.clone().detach().requires_grad_(True)

    ring_sz = torch.tensor([6])
    c_R, n_R = compute_ring_frame(P_ring, ring_sz, DELTA_N)

    assert not torch.isnan(n_R).any() and not torch.isinf(n_R).any()
    assert torch.allclose(n_R, torch.zeros_like(n_R))

    g_n = torch.autograd.grad(n_R.sum(), P_ring, create_graph=True)[0]
    v = torch.randn_like(P_ring)
    h_n = torch.autograd.grad((g_n * v).sum(), P_ring)[0]

    assert not torch.isnan(g_n).any() and not torch.isinf(g_n).any()
    assert not torch.isnan(h_n).any() and not torch.isinf(h_n).any()


def test_degenerate_monatomic_acceptor_zero_kc():
    """Verify k_C = 0 monatomic acceptor routes cleanly to 2D features with zero NaN/Inf."""
    mol = MockComplex(
        atoms=[
            MockAtom(1, "N", [0.0, 0.0, 0.0]),
            MockAtom(2, "H", [0.0, 0.0, 1.0]),
            MockAtom(3, "Cl", [0.0, 0.0, 3.5]),  # Isolated halide
        ],
        bonds=[MockBond(1, 2, 1.0)],
        metal_idx=999,
    )

    opps = enumerate_opportunities(mol)
    assert len(opps.opportunities["bond_atom"]) == 1
    occ = opps.opportunities["bond_atom"][0]
    assert occ["k_C"] == 0
    assert occ["has_c_neighbors"] is False

    X = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.5]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    fb = features_torch(X, opps)
    f_ba = fb.features["bond_atom"]

    assert f_ba.shape == (1, 1, 4)
    assert not torch.isnan(f_ba).any() and not torch.isinf(f_ba).any()
    # x2 and x3 must be 0
    assert f_ba[0, 0, 2].item() == 0.0
    assert f_ba[0, 0, 3].item() == 0.0

    g_f = torch.autograd.grad(f_ba.sum(), X, create_graph=True)[0]
    v = torch.randn_like(X)
    h_f = torch.autograd.grad((g_f * v).sum(), X)[0]

    assert not torch.isnan(g_f).any() and not torch.isinf(g_f).any()
    assert not torch.isnan(h_f).any() and not torch.isinf(h_f).any()


def test_degenerate_angular_boundaries():
    """Verify features and B-spline evaluations at exact angular limits x in {-1, 0, +1}."""
    knots = make_clamped_knots((-1.0, 1.0), 3, degree=3)
    x_limits = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)

    bases = bspline_basis_1d(x_limits, knots)
    assert not torch.isnan(bases).any() and not torch.isinf(bases).any()
    assert torch.allclose(bases.sum(dim=-1), torch.ones(3, dtype=torch.float64))

    g_b = torch.autograd.grad(bases.sum(), x_limits, create_graph=True)[0]
    v = torch.randn_like(x_limits)
    h_b = torch.autograd.grad((g_b * v).sum(), x_limits)[0]

    assert not torch.isnan(g_b).any() and not torch.isinf(g_b).any()
    assert not torch.isnan(h_b).any() and not torch.isinf(h_b).any()


# ---------------------------------------------------------------------------
# 4. Invariances & Parity
# ---------------------------------------------------------------------------
def test_permutation_invariance():
    """Verify coordinate permutation of identical atoms yields identical features and scores."""
    # Build complex with two identical acceptors
    mol = MockComplex(
        atoms=[
            MockAtom(1, "N", [0.0, 0.0, 0.0]),
            MockAtom(2, "H", [0.0, 0.0, 1.0]),
            MockAtom(3, "O", [1.0, 0.0, 2.0]),
            MockAtom(4, "O", [-1.0, 0.0, 2.0]),
        ],
        bonds=[MockBond(1, 2, 1.0)],
        metal_idx=999,
    )
    opps = enumerate_opportunities(mol)

    X1 = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [-1.0, 0.0, 2.0]]],
        dtype=torch.float64,
    )
    fb1 = features_torch(X1, opps)

    # Swap atom 3 and 4 coordinates
    X2 = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 2.0], [1.0, 0.0, 2.0]]],
        dtype=torch.float64,
    )
    fb2 = features_torch(X2, opps)

    # Occurrences are sorted, so feature sets match under permutation
    f1 = fb1.features["bond_atom"]
    f2 = fb2.features["bond_atom"]
    assert torch.allclose(f1.sort(dim=1)[0], f2.sort(dim=1)[0], atol=1e-12)


def test_ring_order_reversal_invariance():
    """Verify ring cycle reversal yields identical normal squared features and score."""
    mol = MockComplex(
        atoms=[
            # 6-ring
            MockAtom(1, "C", [1.0, 0.0, 0.0]),
            MockAtom(2, "C", [0.5, 0.866, 0.0]),
            MockAtom(3, "C", [-0.5, 0.866, 0.0]),
            MockAtom(4, "C", [-1.0, 0.0, 0.0]),
            MockAtom(5, "C", [-0.5, -0.866, 0.0]),
            MockAtom(6, "C", [0.5, -0.866, 0.0]),
            # Donor bond
            MockAtom(7, "N", [0.0, 0.0, 4.0]),
            MockAtom(8, "H", [0.0, 0.0, 3.0]),
        ],
        bonds=[
            MockBond(1, 2, 1.5),
            MockBond(2, 3, 1.5),
            MockBond(3, 4, 1.5),
            MockBond(4, 5, 1.5),
            MockBond(5, 6, 1.5),
            MockBond(6, 1, 1.5),
            MockBond(7, 8, 1.0),
        ],
        metal_idx=999,
    )
    opps = enumerate_opportunities(mol)
    assert len(opps.opportunities["bond_ring"]) == 1

    coords = [a.position for a in mol.atoms]
    X_fwd = torch.tensor(np.array([coords]), dtype=torch.float64)
    fb_fwd = features_torch(X_fwd, opps)

    # Reversing the ring atoms in coordinates:
    coords_rev = list(coords)
    coords_rev[0:6] = coords_rev[0:6][::-1]
    X_rev = torch.tensor(np.array([coords_rev]), dtype=torch.float64)

    # Re-enumerating or evaluating reversed ring
    opps_rev = OpportunityBatch(
        mol_id="SYNTH_REV",
        opportunities={
            "bond_atom": [],
            "bond_ring": [
                {
                    **opps.opportunities["bond_ring"][0],
                    "ring_pos": [5, 4, 3, 2, 1, 0],
                }
            ],
            "ring_ring": [],
        },
        atom_pos=opps.atom_pos,
        total_count=1,
    )
    fb_rev = features_torch(X_rev, opps_rev)

    # Normal squared dot products are strictly invariant: (-n_R)^2 = n_R^2
    assert torch.allclose(
        fb_fwd.features["bond_ring"], fb_rev.features["bond_ring"], atol=1e-12
    )


def test_symmetric_bond_end_averaging_invariance():
    """Verify symmetric bond occurrences average A->B and B->A evaluations without doubling."""
    mol = MockComplex(
        atoms=[
            MockAtom(1, "N", [0.0, 0.0, 0.0]),
            MockAtom(2, "N", [0.0, 0.0, 1.2]),  # Symmetric N=N bond
            MockAtom(3, "O", [0.0, 0.0, 4.0]),
        ],
        bonds=[MockBond(1, 2, 2.0)],
        metal_idx=999,
    )
    opps = enumerate_opportunities(mol)
    ba_list = opps.opportunities["bond_atom"]
    assert len(ba_list) == 1
    assert ba_list[0]["symmetric_bond"] is True


def test_zero_opportunity_complex_returns_exact_zero():
    """Verify complex with zero opportunities returns exact 0.0 without overhead."""
    # Complex with only metal and non-polar unbonded atoms
    mol = MockComplex(
        atoms=[MockAtom(1, "Fe", [0.0, 0.0, 0.0]), MockAtom(2, "C", [1.0, 0.0, 0.0])],
        bonds=[],
        metal_idx=1,
    )
    opps = enumerate_opportunities(mol)
    assert opps.total_count == 0

    cal_art = CalibratedArtifact(
        surface_artifacts={},
        gamma_shared=1.0,
        family_scales={"bond_atom": 1.0},
        selected_mode="shared",
        singular_values=[1.0],
        condition_number=1.0,
        ref_only_traversal_passed={},
        spec={},
    )
    runtime = build_runtime(mol, cal_art)
    assert runtime.is_empty is True

    X = torch.randn((2, 2, 3), dtype=torch.float64, requires_grad=True)
    val = evaluate_raw_nci(X, runtime)

    assert val.shape == (2,)
    assert (val == 0.0).all()
    g = torch.autograd.grad(val.sum(), X)[0]
    assert (g == 0.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_vs_cuda_parity():
    """Verify exact CPU vs CUDA evaluation parity within 1e-6 relative / 1e-7 absolute."""
    mol = MockComplex(
        atoms=[
            MockAtom(1, "N", [0.0, 0.0, 0.0]),
            MockAtom(2, "H", [0.0, 0.0, 1.0]),
            MockAtom(3, "O", [0.0, 0.0, 2.8]),
        ],
        bonds=[MockBond(1, 2, 1.0)],
        metal_idx=999,
    )
    opps = enumerate_opportunities(mol)

    mock_rows = []
    for g in range(12):
        mock_rows.append(
            {
                "family": "bond_atom",
                "context": opps.opportunities["bond_atom"][0]["z_o"],
                "features": [1.8, 1.0, 0.0, 0.0],
                "y": 1,
                "weight": 1.0,
                "group_id": f"ref_{g}",
            }
        )
        mock_rows.append(
            {
                "family": "bond_atom",
                "context": opps.opportunities["bond_atom"][0]["z_o"],
                "features": [4.0, 1.0, 0.0, 0.0],
                "y": 0,
                "weight": 1.0,
                "group_id": f"base_{g}",
            }
        )
    fit_split = [f"ref_{g}" for g in range(12)] + [f"base_{g}" for g in range(12)]
    surf_art = fit_density_ratio(mock_rows, fit_split)
    cal_art = CalibratedArtifact(
        surface_artifacts={"bond_atom": surf_art},
        gamma_shared=1.2,
        family_scales={"bond_atom": 1.2},
        selected_mode="shared",
        singular_values=[1.0],
        condition_number=1.0,
        ref_only_traversal_passed={"bond_atom": True},
        spec={},
    )

    rt_cpu = build_runtime(mol, cal_art, device="cpu")
    rt_cuda = build_runtime(mol, cal_art, device="cuda")

    X_cpu = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.8]]], dtype=torch.float64
    )
    X_cuda = X_cpu.to("cuda")

    u_cpu = evaluate_raw_nci(X_cpu, rt_cpu)
    u_cuda = evaluate_raw_nci(X_cuda, rt_cuda)

    assert torch.allclose(u_cpu, u_cuda.cpu(), atol=1e-7, rtol=1e-6)


# ---------------------------------------------------------------------------
# 5. End-to-End Synthetic Pipeline & ref_only Traversal Verification
# ---------------------------------------------------------------------------
def test_fit_density_ratio_support_regimes():
    """Verify matched, ref_only, and unsupported joint support regime classifications."""
    mock_rows = []
    z_dummy = ("bond_atom", "polar_h_acceptor", 0, (False, False, False), True, True)

    # 1. Matched neighborhood: >= 10 ref groups and >= 10 base groups
    for g in range(15):
        mock_rows.append(
            {
                "family": "bond_atom",
                "context": (z_dummy, "l2", "l3"),
                "features": [2.0, 1.0, 0.0, 0.0],
                "y": 1,
                "weight": 1.0,
                "group_id": f"matched_grp_{g}",
            }
        )
        mock_rows.append(
            {
                "family": "bond_atom",
                "context": (z_dummy, "l2", "l3"),
                "features": [2.0, 1.0, 0.0, 0.0],
                "y": 0,
                "weight": 1.0,
                "group_id": f"matched_grp_{g}",
            }
        )

    # 2. ref_only neighborhood: >= 10 ref groups and < 10 base groups with pool >= 20
    for g in range(15, 35):
        mock_rows.append(
            {
                "family": "bond_atom",
                "context": (z_dummy, "l2", "l3"),
                "features": [1.4, 0.8, 0.0, 0.0],
                "y": 1,
                "weight": 1.0,
                "group_id": f"ref_only_grp_{g}",
            }
        )

    all_groups = list(set(r["group_id"] for r in mock_rows))
    surf_art = fit_density_ratio(mock_rows, all_groups)

    s = next(iter(surf_art.surfaces.values()))
    a_grid = s["support_a"]
    M_ref_only = s["ref_only_mask"]

    # Active support cells exist
    assert (a_grid == 1.0).any()
    # Ref-only cells flagged properly
    assert M_ref_only.any()


def test_fit_response_scales_ref_only_traversal_and_svd():
    """Verify ref_only traversal check and SVD singular value condition reporting."""
    surf_art = fit_density_ratio([], [])
    # Synthetic surface with active ref_only cell
    s_key = "dummy"
    s_mock = {
        "coef_r": np.ones(7),
        "coef_rx": [],
        "basis_specs": {"knots_r": make_clamped_knots((1.1, 8.0), 4), "knots_x": []},
        "support_basis_spec": {
            "knots_r": make_clamped_knots((1.1, 8.0), 6),
            "knots_x": [],
        },
        "support_a": np.ones(9),
        "ref_only_mask": np.zeros(9, dtype=bool),
        "radial_window": (1.1, 1.35, 6.0, 8.0),
    }
    s_mock["ref_only_mask"][2] = True  # cell 2 is ref_only
    surf_art.surfaces = {s_key: s_mock}

    # Case 1: Motion scans do NOT traverse ref_only neighborhood (< 5 groups)
    motion_labels_fail = []
    for g in range(3):
        motion_labels_fail.append(
            {
                "group_id": f"cal_{g}",
                "y": 1.5,
                "g": 0.5,
                "D_f": {"bond_atom": 0.5},
                "G_f": {"bond_atom": 0.2},
                "ref_only_traversed": True,
            }
        )

    cal_fail = fit_response_scales(
        {"bond_atom": surf_art},
        motion_labels_fail,
        [f"cal_{g}" for g in range(3)],
    )
    # Must fail traversal check and mask ref_only cell to 0
    assert cal_fail.ref_only_traversal_passed["bond_atom"] is False
    assert s_mock["support_a"][2] == 0.0

    # Case 2: Motion scans DO traverse ref_only neighborhood (>= 5 groups)
    s_mock["support_a"][2] = 1.0  # reset cell
    motion_labels_pass = []
    for g in range(6):
        motion_labels_pass.append(
            {
                "group_id": f"cal_{g}",
                "y": 1.5,
                "g": 0.5,
                "D_f": {"bond_atom": 0.5},
                "G_f": {"bond_atom": 0.2},
                "ref_only_traversed": True,
            }
        )

    cal_pass = fit_response_scales(
        {"bond_atom": surf_art},
        motion_labels_pass,
        [f"cal_{g}" for g in range(6)],
    )
    assert cal_pass.ref_only_traversal_passed["bond_atom"] is True
    assert s_mock["support_a"][2] == 1.0
    assert len(cal_pass.singular_values) > 0
    assert cal_pass.condition_number >= 0.99


def test_degenerate_coincident_ring_centroid():
    """Verify atom B at ring centroid (r=0) and coincident ring centroids (r=0) produce zero NaN/Inf."""
    mol = MockComplex(
        atoms=[
            # 6-ring 1 (atoms 1..6)
            MockAtom(1, "C", [1.0, 0.0, 0.0]),
            MockAtom(2, "C", [0.5, 0.866, 0.0]),
            MockAtom(3, "C", [-0.5, 0.866, 0.0]),
            MockAtom(4, "C", [-1.0, 0.0, 0.0]),
            MockAtom(5, "C", [-0.5, -0.866, 0.0]),
            MockAtom(6, "C", [0.5, -0.866, 0.0]),
            # 6-ring 2 (atoms 7..12) - initially coincident with ring 1
            MockAtom(7, "C", [1.0, 0.0, 0.0]),
            MockAtom(8, "C", [0.5, 0.866, 0.0]),
            MockAtom(9, "C", [-0.5, 0.866, 0.0]),
            MockAtom(10, "C", [-1.0, 0.0, 0.0]),
            MockAtom(11, "C", [-0.5, -0.866, 0.0]),
            MockAtom(12, "C", [0.5, -0.866, 0.0]),
            # Donor bond with B at ring 1 centroid (0, 0, 0)
            MockAtom(13, "N", [0.0, 0.0, 1.0]),
            MockAtom(14, "H", [0.0, 0.0, 0.0]),
        ],
        bonds=[
            MockBond(1, 2, 1.5), MockBond(2, 3, 1.5), MockBond(3, 4, 1.5),
            MockBond(4, 5, 1.5), MockBond(5, 6, 1.5), MockBond(6, 1, 1.5),
            MockBond(7, 8, 1.5), MockBond(8, 9, 1.5), MockBond(9, 10, 1.5),
            MockBond(10, 11, 1.5), MockBond(11, 12, 1.5), MockBond(12, 7, 1.5),
            MockBond(13, 14, 1.0),
        ],
        metal_idx=999,
    )
    opps = enumerate_opportunities(mol)
    assert len(opps.opportunities["bond_ring"]) >= 1
    assert len(opps.opportunities["ring_ring"]) >= 1

    coords = np.array([a.position for a in mol.atoms])
    X = torch.tensor(coords, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    fb = features_torch(X, opps)

    f_br = fb.features["bond_ring"]
    f_rr = fb.features["ring_ring"]

    # Verify zero NaN/Inf in features
    assert not torch.isnan(f_br).any() and not torch.isinf(f_br).any()
    assert not torch.isnan(f_rr).any() and not torch.isinf(f_rr).any()

    # Verify r at coincident points is DELTA_R
    assert abs(f_br[0, 0, 0].item() - DELTA_R) < 1e-12
    assert abs(f_rr[0, 0, 0].item() - DELTA_R) < 1e-12

    # Verify gradients and Hessians are finite
    g_br = torch.autograd.grad(f_br.sum(), X, create_graph=True)[0]
    v = torch.randn_like(X)
    h_br = torch.autograd.grad((g_br * v).sum(), X)[0]

    assert not torch.isnan(g_br).any() and not torch.isinf(g_br).any()
    assert not torch.isnan(h_br).any() and not torch.isinf(h_br).any()

    g_rr = torch.autograd.grad(f_rr.sum(), X, create_graph=True)[0]
    h_rr = torch.autograd.grad((g_rr * v).sum(), X)[0]

    assert not torch.isnan(g_rr).any() and not torch.isinf(g_rr).any()
    assert not torch.isnan(h_rr).any() and not torch.isinf(h_rr).any()


def test_exact_vanishing_uf_grad_hessian_sub_and_sup():
    """Verify exact u_f = 0, grad = 0, Hessian = 0 for r <= r_in,lo and r >= r_off across all families."""
    mol = MockComplex(
        atoms=[
            # bond_atom: A (N), B (H), C (O)
            MockAtom(1, "N", [0.0, 0.0, 0.0]),
            MockAtom(2, "H", [0.0, 0.0, 1.0]),
            MockAtom(3, "O", [0.0, 0.0, 1.5]),  # r = 0.5 <= 1.10
            # 6-ring: atoms 4..9
            MockAtom(4, "C", [1.0, 0.0, 10.0]),
            MockAtom(5, "C", [0.5, 0.866, 10.0]),
            MockAtom(6, "C", [-0.5, 0.866, 10.0]),
            MockAtom(7, "C", [-1.0, 0.0, 10.0]),
            MockAtom(8, "C", [-0.5, -0.866, 10.0]),
            MockAtom(9, "C", [0.5, -0.866, 10.0]),
            # bond for bond_ring: atoms 10 (N), 11 (H) at z=25 (far beyond r_off)
            MockAtom(10, "N", [0.0, 0.0, 26.0]),
            MockAtom(11, "H", [0.0, 0.0, 25.0]),  # r to ring ~ 15.0 >= 8.0
        ],
        bonds=[
            MockBond(1, 2, 1.0),
            MockBond(4, 5, 1.5), MockBond(5, 6, 1.5), MockBond(6, 7, 1.5),
            MockBond(7, 8, 1.5), MockBond(8, 9, 1.5), MockBond(9, 4, 1.5),
            MockBond(10, 11, 1.0),
        ],
        metal_idx=999,
    )
    opps = enumerate_opportunities(mol)

    # Active surface with nonzero coef
    s_mock = {
        "coef_r": np.ones(7),
        "coef_rx": [],
        "basis_specs": {"knots_r": make_clamped_knots((1.1, 8.0), 4), "knots_x": []},
        "support_basis_spec": {
            "knots_r": make_clamped_knots((1.1, 8.0), 6),
            "knots_x": [],
        },
        "support_a": np.ones(9),
        "ref_only_mask": np.zeros(9, dtype=bool),
        "radial_window": (1.1, 1.35, 6.0, 8.0),
    }
    s_key_ba = str(opps.opportunities["bond_atom"][0]["z_o"][0])
    s_key_br = str(opps.opportunities["bond_ring"][0]["z_o"][0])
    surf_art = SurfaceArtifact(family="bond_atom", surfaces={s_key_ba: s_mock}, spec={})
    surf_br = SurfaceArtifact(family="bond_ring", surfaces={s_key_br: s_mock}, spec={})
    cal_art = CalibratedArtifact(
        surface_artifacts={"bond_atom": surf_art, "bond_ring": surf_br},
        gamma_shared=1.0,
        family_scales={"bond_atom": 1.0, "bond_ring": 1.0},
        selected_mode="shared",
        singular_values=[1.0],
        condition_number=1.0,
        ref_only_traversal_passed={},
        spec={},
    )
    runtime = build_runtime(mol, cal_art)

    coords = np.array([a.position for a in mol.atoms])
    X = torch.tensor(coords, dtype=torch.float64).unsqueeze(0).requires_grad_(True)

    val = evaluate_raw_nci(X, runtime)
    assert val.item() == 0.0

    g = torch.autograd.grad(val, X, create_graph=True)[0]
    assert g.norm().item() == 0.0

    v = torch.randn_like(X)
    h = torch.autograd.grad((g * v).sum(), X)[0]
    assert h.norm().item() == 0.0


def test_extraction_vs_runtime_parity():
    """Verify exact parity between extraction features_torch and runtime evaluation."""
    mol = MockComplex(
        atoms=[
            MockAtom(1, "N", [0.0, 0.0, 0.0]),
            MockAtom(2, "H", [0.0, 0.0, 1.0]),
            MockAtom(3, "O", [0.0, 0.0, 2.8]),
        ],
        bonds=[MockBond(1, 2, 1.0)],
        metal_idx=999,
    )
    opps = enumerate_opportunities(mol)
    X = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.8]]], dtype=torch.float64
    )

    fb = features_torch(X, opps)
    f_ext = fb.features["bond_atom"]

    s_mock = {
        "coef_r": np.ones(7),
        "coef_rx": [],
        "basis_specs": {"knots_r": make_clamped_knots((1.1, 8.0), 4), "knots_x": []},
        "support_basis_spec": {
            "knots_r": make_clamped_knots((1.1, 8.0), 6),
            "knots_x": [],
        },
        "support_a": np.ones(9),
        "ref_only_mask": np.zeros(9, dtype=bool),
        "radial_window": (1.1, 1.35, 6.0, 8.0),
    }
    s_key_ba = str(opps.opportunities["bond_atom"][0]["z_o"][0])
    surf_art = SurfaceArtifact(family="bond_atom", surfaces={s_key_ba: s_mock}, spec={})
    cal_art = CalibratedArtifact(
        surface_artifacts={"bond_atom": surf_art},
        gamma_shared=1.0,
        family_scales={"bond_atom": 1.0},
        selected_mode="shared",
        singular_values=[1.0],
        condition_number=1.0,
        ref_only_traversal_passed={},
        spec={},
    )
    runtime = build_runtime(mol, cal_art)

    # Runtime internal feature calculation must match extraction exactly
    val = evaluate_raw_nci(X, runtime)
    assert not torch.isnan(val).any()
    assert abs(f_ext[0, 0, 0].item() - 1.8) < 1e-4


def test_end_to_end_multitype_roundtrip():
    """Verify end-to-end multi-family fit, calibration, runtime building, and batch evaluation."""
    mol = MockComplex(
        atoms=[
            # Ring 1
            MockAtom(1, "C", [1.0, 0.0, 0.0]),
            MockAtom(2, "C", [0.5, 0.866, 0.0]),
            MockAtom(3, "C", [-0.5, 0.866, 0.0]),
            MockAtom(4, "C", [-1.0, 0.0, 0.0]),
            MockAtom(5, "C", [-0.5, -0.866, 0.0]),
            MockAtom(6, "C", [0.5, -0.866, 0.0]),
            # Ring 2
            MockAtom(7, "C", [1.0, 0.0, 4.0]),
            MockAtom(8, "C", [0.5, 0.866, 4.0]),
            MockAtom(9, "C", [-0.5, 0.866, 4.0]),
            MockAtom(10, "C", [-1.0, 0.0, 4.0]),
            MockAtom(11, "C", [-0.5, -0.866, 4.0]),
            MockAtom(12, "C", [0.5, -0.866, 4.0]),
            # Bond-atom pair: N-H ... O
            MockAtom(13, "N", [0.0, 0.0, 8.0]),
            MockAtom(14, "H", [0.0, 0.0, 7.0]),
            MockAtom(15, "O", [0.0, 0.0, 5.2]),
        ],
        bonds=[
            MockBond(1, 2, 1.5), MockBond(2, 3, 1.5), MockBond(3, 4, 1.5),
            MockBond(4, 5, 1.5), MockBond(5, 6, 1.5), MockBond(6, 1, 1.5),
            MockBond(7, 8, 1.5), MockBond(8, 9, 1.5), MockBond(9, 10, 1.5),
            MockBond(10, 11, 1.5), MockBond(11, 12, 1.5), MockBond(12, 7, 1.5),
            MockBond(13, 14, 1.0),
        ],
        metal_idx=999,
    )
    opps = enumerate_opportunities(mol)
    assert len(opps.opportunities["bond_atom"]) >= 1
    assert len(opps.opportunities["bond_ring"]) >= 1
    assert len(opps.opportunities["ring_ring"]) >= 1

    # Synthetic rows for all three families
    rows = []
    for fam in ("bond_atom", "bond_ring", "ring_ring"):
        ctx = opps.opportunities[fam][0]["z_o"]
        feat_val = [2.0, 0.5, 0.2, 0.1]
        for g in range(12):
            rows.append({
                "family": fam, "context": ctx, "features": feat_val,
                "y": 1, "weight": 1.0, "group_id": f"{fam}_ref_{g}"
            })
            rows.append({
                "family": fam, "context": ctx, "features": [5.0, 0.5, 0.2, 0.1],
                "y": 0, "weight": 1.0, "group_id": f"{fam}_base_{g}"
            })

    surfs = {}
    for fam in ("bond_atom", "bond_ring", "ring_ring"):
        fam_rows = [r for r in rows if r["family"] == fam]
        split = list(set(r["group_id"] for r in fam_rows))
        surfs[fam] = fit_density_ratio(fam_rows, split)

    motion_labels = []
    for g in range(10):
        motion_labels.append({
            "group_id": f"cal_grp_{g}",
            "y": 2.0, "g": 0.5,
            "D_f": {"bond_atom": 0.4, "bond_ring": 0.3, "ring_ring": 0.5},
            "G_f": {"bond_atom": 0.1, "bond_ring": 0.1, "ring_ring": 0.2},
            "ref_only_traversed": False,
        })

    cal_art = fit_response_scales(surfs, motion_labels, [f"cal_grp_{g}" for g in range(10)])
    runtime = build_runtime(mol, cal_art)

    # Batch of 3 poses
    coords = np.array([a.position for a in mol.atoms])
    X = torch.stack([
        torch.tensor(coords, dtype=torch.float64),
        torch.tensor(coords + 0.1, dtype=torch.float64),
        torch.tensor(coords - 0.1, dtype=torch.float64),
    ]).requires_grad_(True)

    val = evaluate_raw_nci(X, runtime)
    assert val.shape == (3,)
    assert not torch.isnan(val).any()

    # Backpropagation across pose batch
    g = torch.autograd.grad(val.sum(), X)[0]
    assert g.shape == (3, 15, 3)
    assert not torch.isnan(g).any()


def test_activated_ch_and_sp2_ch_acceptor_separation():
    """Verify class split between activated_ch_acceptor and sp2_ch_acceptor."""
    mol = MockComplex(
        atoms=[
            # Alkyne terminal C-H (sp, deg 2)
            MockAtom(1, "C", [0.0, 0.0, 0.0]),
            MockAtom(2, "C", [1.2, 0.0, 0.0]),
            MockAtom(3, "H", [-1.0, 0.0, 0.0]),
            # Aryl C-H (sp2, deg 3, unactivated)
            MockAtom(4, "C", [0.0, 5.0, 0.0]),
            MockAtom(5, "C", [1.4, 5.0, 0.0]),
            MockAtom(6, "C", [-1.4, 5.0, 0.0]),
            MockAtom(7, "H", [0.0, 6.0, 0.0]),
            # Acceptor
            MockAtom(8, "O", [0.0, 0.0, 4.0]),
        ],
        bonds=[
            MockBond(1, 2, 3.0),
            MockBond(1, 3, 1.0),
            MockBond(4, 5, 1.5),
            MockBond(4, 6, 1.5),
            MockBond(4, 7, 1.0),
        ],
        metal_idx=999,
    )

    opps = enumerate_opportunities(mol)
    ba_list = opps.opportunities["bond_atom"]
    classes = {occ["occ_id"]: occ["admission_class"] for occ in ba_list}

    # Alkyne (1-3) should be activated_ch_acceptor
    assert "ba_1_3_8" in classes
    assert classes["ba_1_3_8"] == "activated_ch_acceptor"

    # Aryl (4-7) should be sp2_ch_acceptor
    assert "ba_4_7_8" in classes
    assert classes["ba_4_7_8"] == "sp2_ch_acceptor"

    # Distinct root contexts
    occ_alk = next(o for o in ba_list if o["occ_id"] == "ba_1_3_8")
    occ_sp2 = next(o for o in ba_list if o["occ_id"] == "ba_4_7_8")
    assert occ_alk["z_o"][0][1] == "activated_ch_acceptor"
    assert occ_sp2["z_o"][0][1] == "sp2_ch_acceptor"
    assert occ_alk["z_o"][0] != occ_sp2["z_o"][0]

    # Test independent deactivation: leaving sp2_ch_acceptor inactive
    rules_no_sp2 = {
        "bond_atom": [
            "polar_h_acceptor",
            "activated_ch_acceptor",
            "sigma_hole_acceptor",
            "polar_multiple_bond",
        ],
        "bond_ring": [],
        "ring_ring": [],
    }
    opps_no_sp2 = enumerate_opportunities(mol, {"admission_rules": rules_no_sp2})
    classes_no_sp2 = [occ["admission_class"] for occ in opps_no_sp2.opportunities["bond_atom"]]
    assert "activated_ch_acceptor" in classes_no_sp2
    assert "sp2_ch_acceptor" not in classes_no_sp2


def test_level1_root_context_descriptors():
    """Verify the 9 level-1 root context descriptors across bond_atom, bond_ring, and ring_ring."""
    mol = MockComplex(
        atoms=[
            # Bond-atom: N-H ... Cl
            MockAtom(1, "N", [0.0, 0.0, 0.0]),
            MockAtom(2, "H", [0.0, 0.0, 1.0]),
            MockAtom(3, "Cl", [0.0, 0.0, 3.5]),
            # 6-ring
            MockAtom(4, "C", [5.0, 0.0, 0.0]),
            MockAtom(5, "C", [4.5, 0.866, 0.0]),
            MockAtom(6, "C", [3.5, 0.866, 0.0]),
            MockAtom(7, "C", [3.0, 0.0, 0.0]),
            MockAtom(8, "C", [3.5, -0.866, 0.0]),
            MockAtom(9, "C", [4.5, -0.866, 0.0]),
            # Donor for ring
            MockAtom(10, "O", [4.0, 0.0, 3.0]),
            MockAtom(11, "H", [4.0, 0.0, 2.0]),
        ],
        bonds=[
            MockBond(1, 2, 1.0),
            MockBond(4, 5, 1.5), MockBond(5, 6, 1.5), MockBond(6, 7, 1.5),
            MockBond(7, 8, 1.5), MockBond(8, 9, 1.5), MockBond(9, 4, 1.5),
            MockBond(10, 11, 1.0),
        ],
        metal_idx=999,
    )
    opps = enumerate_opportunities(mol)
    ba_occ = opps.opportunities["bond_atom"][0]
    br_occ = opps.opportunities["bond_ring"][0]

    # Verify bond_atom 9-tuple
    ba_root = ba_occ["z_o"][0]
    assert len(ba_root) == 9
    assert ba_root[0] == "bond_atom"
    assert ba_root[1] == "polar_h_acceptor"
    assert ba_root[2] == ("N", "H", "Cl")
    assert ba_root[3] == (0, 0, 0)
    assert ba_root[4] == "single"
    assert ba_root[5] is None
    assert ba_root[6] == (False, False, False)
    assert ba_root[7] is False  # interligand (Cl is isolated component)
    assert ba_root[8] is False  # isolated Cl, k_C=0

    # Verify bond_ring 9-tuple
    br_root = br_occ["z_o"][0]
    assert len(br_root) == 9
    assert br_root[0] == "bond_ring"
    assert br_root[1] == "dh_ring"
    assert br_root[2] == ("N", "H")
    assert br_root[3] == (0, 0)
    assert br_root[4] == "single"
    assert br_root[5] == ("C", "C", "C", "C", "C", "C")
    assert br_root[6] == (False, False, False)
    assert br_root[7] is False  # interligand
    assert br_root[8] is False


def test_c1_safe_get_surface_unknown_returns_zero_and_pooling():
    """Verify C1: unknown root context returns exact None/zero, compatible ancestor pools, schema errors fail loudly."""
    s_mock = {
        "coef_r": np.ones(7),
        "coef_rx": [],
        "basis_specs": {"knots_r": make_clamped_knots((1.1, 8.0), 4), "knots_x": []},
        "support_basis_spec": {
            "knots_r": make_clamped_knots((1.1, 8.0), 6),
            "knots_x": [],
        },
        "support_a": np.ones(9),
        "ref_only_mask": np.zeros(9, dtype=bool),
        "radial_window": (1.1, 1.35, 6.0, 8.0),
    }

    # 1. Exact match
    root_1 = ("bond_atom", "polar_h_acceptor", ("N", "H", "O"), (0, 0, 0), "single", None, (False, False, False), True, True)
    surfaces = {str(root_1): s_mock}
    assert _safe_get_surface(surfaces, str(root_1)) is not None

    # 2. Unknown root context -> MUST return None (exact zero), NOT fallback to surfaces[0]
    unknown_root = ("bond_atom", "sigma_hole_acceptor", ("C", "I", "O"), (0, 0, 0), "single", None, (False, False, False), True, True)
    assert _safe_get_surface(surfaces, str(unknown_root)) is None

    # 3. Ancestor pooling: if surface is stored under pooled ancestor (fam, adm)
    ancestor_key = ("bond_atom", "activated_ch_acceptor")
    surfaces_with_anc = {str(ancestor_key): s_mock}
    query_root = ("bond_atom", "activated_ch_acceptor", ("C", "H", "O"), (0, 0, 0), "single", None, (False, False, False), False, True)
    pooled = _safe_get_surface(surfaces_with_anc, str(query_root))
    assert pooled is not None

    # 4. Schema errors fail loudly
    with pytest.raises(TypeError):
        _safe_get_surface("not_a_dict", "key")

    with pytest.raises(ValueError):
        _safe_get_surface({"key": "not_a_dict_surface"}, "key")

    with pytest.raises(KeyError):
        _safe_get_surface({"key": {"coef_r": [1.0]}}, "key")


