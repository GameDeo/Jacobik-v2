"""
nci_v2.py -- Transferable, globally C^2, data-derived NCI model (Jacobik Roadmap v2).

Exact implementation of Roadmap v2 (§2.2, §3.1, §3.2, §3.4, §3.5, §3.6, §3.7, §3.9, §5.3 Gate 2, §7.1).
Supports bond-atom, bond-ring, and ring-ring families with C^2 regularized coordinates,
tensor-product cubic B-spline support and latent surfaces, and offline energy/force calibration.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy.optimize
import torch

# ---------------------------------------------------------------------------
# Global constants and default numerical parameters (Roadmap v2 §3.2, §7.1)
# ---------------------------------------------------------------------------
DELTA_R = 1e-4  # Regularized displacement norm parameter in Angstrom
DELTA_N = 1e-6  # Regularized ring normal area parameter in Angstrom^2
EPS_POS = 0.20  # Positivity smoothing parameter in nats (Roadmap v2 §3.5)
RT_KJ_MOL = 8.31446261815324e-3 * 298.15  # 2.478956 kJ/mol (Roadmap v2 §2.2)

FAMILIES = ("bond_atom", "bond_ring", "ring_ring")
ACCEPTORS = frozenset({"N", "O", "F", "S", "Cl", "Br", "I"})
RING_ELEMENTS = frozenset({"C", "N", "O", "S", "B", "P"})

DEFAULT_RADIAL_WINDOWS = {
    # (r_in_lo, r_in_hi, r_on, r_off) in Angstrom (Roadmap v2 §3.5, §7.1)
    "bond_atom_H": (1.10, 1.35, 6.0, 8.0),
    "bond_atom_heavy": (1.80, 2.20, 6.0, 8.0),
    "bond_ring": (1.60, 2.00, 6.0, 8.0),
    "ring_ring": (2.40, 2.90, 8.0, 10.0),
}

DEFAULT_SCHEMA = {
    "version": "2.0.0-pilot.2",
    "delta_r": DELTA_R,
    "delta_n": DELTA_N,
    "eps_pos": EPS_POS,
    "admission_rules": {
        "min_ligand_graph_separation_edges": 5,
        "bond_atom": {
            "polar_h_acceptor": {
                "donor_A_elements": ["N", "O", "F", "S"],
                "hydrogen_B_element": "H",
                "acceptor_C_elements": ["N", "O", "F", "S", "Cl", "Br", "I"],
                "allow_anionic_carbon_acceptor": False,
            },
            "activated_ch_acceptor": {
                "donor_A_elements": ["C"],
                "hydrogen_B_element": "H",
                "acceptor_C_elements": ["N", "O", "F", "S", "Cl", "Br", "I"],
                "activation_criteria": [
                    "sp_hybridized_terminal_alkyne_deg_le_2",
                    "directly_bound_to_metal",
                    "bonded_to_heteroatom_in_N_O_F_S_P_Cl_Br_I",
                ],
                "heteroatom_elements": ["N", "O", "F", "S", "P", "Cl", "Br", "I"],
            },
            "sp2_ch_acceptor": {
                "donor_A_elements": ["C"],
                "hydrogen_B_element": "H",
                "acceptor_C_elements": ["N", "O", "F", "S", "Cl", "Br", "I"],
                "donor_max_degree": 3,
                "activation_criteria": ["unactivated_sp2_deg_3_not_activated_ch"],
            },
            "sigma_hole_acceptor": {
                "donor_A_elements": ["C", "N", "O", "S", "P"],
                "polar_atom_B_elements": ["Cl", "Br", "I", "S", "Se"],
                "acceptor_C_elements": ["N", "O", "F", "S", "Cl", "Br", "I"],
            },
            "polar_multiple_bond": {
                "endpoint_A_elements": ["C", "N", "S", "P"],
                "endpoint_B_elements": ["O", "N", "S"],
                "min_bond_order": 1.4,
                "acceptor_C_elements": ["N", "O", "F", "S", "Cl", "Br", "I"],
            },
        },
        "bond_ring": {
            "dh_ring": {
                "donor_A_elements": ["N", "O", "F", "S"],
                "hydrogen_B_element": "H",
            },
            "ch_ring": {
                "donor_A_elements": ["C"],
                "hydrogen_B_element": "H",
            },
            "lp_or_halogen_ring": {
                "atom_B_elements": ["F", "Cl", "Br", "I"],
            },
        },
        "ring_ring": {
            "sp2_5_6_ring_pair": {},
            "ring_ring": {},
        },
    },
    "radial_windows": DEFAULT_RADIAL_WINDOWS,
}


# ---------------------------------------------------------------------------
# Dataclasses for logical interfaces (Roadmap v2 §3.9)
# ---------------------------------------------------------------------------
@dataclass
class OpportunityBatch:
    """Stores enumerated graph occurrences, discrete contexts, indices, and symmetries."""

    mol_id: str
    opportunities: Dict[str, List[Dict[str, Any]]]
    atom_pos: Dict[int, int]
    total_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureBatch:
    """Continuous features, masks, occurrence IDs, and units across families."""

    features: Dict[str, torch.Tensor]  # family -> (B, O_f, D_f)
    occurrence_ids: Dict[str, List[str]]
    contexts: Dict[str, List[Tuple[Any, ...]]]
    masks: Dict[str, torch.Tensor]
    units: Dict[str, List[str]]


@dataclass
class SurfaceArtifact:
    """Statistical density-ratio surfaces, B-spline coefficients, support grids, and metadata."""

    family: str
    surfaces: Dict[str, Dict[str, Any]]
    spec: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibratedArtifact:
    """Calibrated energy/force scales, surface artifacts, diagnostic SVD and ref_only stats."""

    surface_artifacts: Dict[str, SurfaceArtifact]
    gamma_shared: float
    family_scales: Dict[str, float]
    selected_mode: str
    singular_values: List[float]
    condition_number: float
    ref_only_traversal_passed: Dict[str, bool]
    spec: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeNCI:
    """Pre-indexed execution container for fast autodiff evaluation on coordinates X."""

    mol_id: str
    device: torch.device
    is_empty: bool
    family_runtimes: Dict[str, Dict[str, Any]]
    gamma_shared: float
    family_scales: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Globally C^2 Regularized Vector Norms & Ring Frames (Roadmap v2 §3.2)
# ---------------------------------------------------------------------------
def reg_norm(v: torch.Tensor, delta_r: float = DELTA_R) -> torch.Tensor:
    """Regularized displacement norm ||v||_delta = sqrt(||v||^2 + delta_r^2). Globally C^inf."""
    return torch.sqrt((v * v).sum(dim=-1, keepdim=True) + (delta_r**2))


def reg_unit(v: torch.Tensor, delta_r: float = DELTA_R) -> torch.Tensor:
    """Regularized unit displacement vector u_hat_delta(v) = v / ||v||_delta. At v=0, u_hat=0."""
    return v / reg_norm(v, delta_r=delta_r)


def compute_ring_frame(
    P_ring: torch.Tensor, ring_size: torch.Tensor, delta_n: float = DELTA_N
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Centroid c_R and regularized normal n_R = s_R / sqrt(||s_R||^2 + delta_n^2).

    Parameters:
      P_ring: (B, O, 6, 3) padded ring vertex positions
      ring_size: (O,) tensor of 5 or 6
    Returns:
      c_R: (B, O, 3)
      n_R: (B, O, 3) regularized unit normal
    """
    B, O, _, _ = P_ring.shape
    device = P_ring.device
    dtype = P_ring.dtype

    ar = torch.arange(6, device=device).view(1, 1, 6)
    valid_mask = ar < ring_size.view(1, O, 1)  # (1, O, 6)
    valid_mask_exp = valid_mask.unsqueeze(-1)  # (1, O, 6, 1)

    P_valid = torch.where(valid_mask_exp, P_ring, torch.zeros_like(P_ring))
    sz = ring_size.view(1, O, 1).to(dtype)
    c_R = P_valid.sum(dim=2) / sz  # (B, O, 3)

    q = P_ring - c_R.unsqueeze(2)  # (B, O, 6, 3)

    idx_next_6 = torch.tensor([1, 2, 3, 4, 5, 0], device=device)
    idx_next_5 = torch.tensor([1, 2, 3, 4, 0, 0], device=device)
    is_6 = (ring_size == 6).view(1, O, 1)
    idx_next = torch.where(is_6, idx_next_6.view(1, 1, 6), idx_next_5.view(1, 1, 6))

    idx_next_exp = idx_next.expand(B, O, 6).unsqueeze(-1).expand(B, O, 6, 3)
    q_next = torch.gather(q, dim=2, index=idx_next_exp)

    cross = torch.linalg.cross(q, q_next, dim=-1)  # (B, O, 6, 3)
    cross_masked = torch.where(valid_mask_exp, cross, torch.zeros_like(cross))
    s_R = cross_masked.sum(dim=2)  # (B, O, 3)

    norm_s = torch.sqrt((s_R * s_R).sum(dim=-1, keepdim=True) + (delta_n**2))
    n_R = s_R / norm_s
    return c_R, n_R


def canonical_ring(cycle: Sequence[int]) -> List[int]:
    """Unique cycle representation invariant to cyclic rotation and reversal."""
    m = len(cycle)
    if m == 0:
        return []
    c = list(cycle)
    fwd = [c[i:] + c[:i] for i in range(m)]
    rev = c[::-1]
    rev_rots = [rev[i:] + rev[:i] for i in range(m)]
    return min(fwd + rev_rots)


def sp2_rings_v2(mol, nb=None) -> List[List[int]]:
    """Unique 5- and 6-membered sp2 rings in ligand graph (metal excluded)."""
    if nb is None:
        from jacobik.topology.graph import get_ligand_neighbors

        nb = get_ligand_neighbors(mol.atoms, mol.bonds, getattr(mol, "metal_idx", -1))

    metal_idx = getattr(mol, "metal_idx", -1)
    sym = {a.idx: a.symbol for a in mol.atoms}
    ok = {
        a.idx
        for a in mol.atoms
        if a.idx != metal_idx and sym.get(a.idx) in RING_ELEMENTS and len(nb.get(a.idx, ())) <= 3
    }
    seen_sets = set()
    out = []

    def walk(start, cur, path):
        for v in nb.get(cur, ()):
            if v == start and len(path) in (5, 6):
                key = frozenset(path)
                if key not in seen_sets:
                    seen_sets.add(key)
                    out.append(canonical_ring(path))
            elif v in ok and v > start and v not in path and len(path) < 6:
                walk(start, v, path + [v])

    for s in sorted(ok):
        walk(s, s, [s])
    out.sort()
    return out


# ---------------------------------------------------------------------------
# C^2 Quintic Tapers & Positivity Smoothing (Roadmap v2 §3.5)
# ---------------------------------------------------------------------------
def quintic_step(t: torch.Tensor) -> torch.Tensor:
    """C^2 quintic step function S_5(t) = 10t^3 - 15t^4 + 6t^5 on [0, 1].

    Equal to 0 for t <= 0, and 1 for t >= 1. Value, 1st, and 2nd derivatives match at joins.
    """
    cond_lo = t <= 0.0
    cond_hi = t >= 1.0
    p = 10.0 * (t**3) - 15.0 * (t**4) + 6.0 * (t**5)
    return torch.where(cond_lo, torch.zeros_like(t), torch.where(cond_hi, torch.ones_like(t), p))


def compute_tapers(
    r: torch.Tensor, r_in_lo: float, r_in_hi: float, r_on: float, r_off: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inner taper T_in(r) and outer cutoff T_out(r) matching Roadmap v2 §3.5."""
    t_in = (r - r_in_lo) / (r_in_hi - r_in_lo)
    t_out = (r - r_on) / (r_off - r_on)
    T_in = quintic_step(t_in)
    T_out = 1.0 - quintic_step(t_out)
    return T_in, T_out


def h_epsilon(v: torch.Tensor, eps: float = EPS_POS) -> torch.Tensor:
    """C^2 smooth positivity function matching Roadmap v2 §3.5.

    h_eps(v) = 0 for v <= 0; eps*(6t^3 - 8t^4 + 3t^5) for 0 < v < eps; v for v >= eps.
    """
    cond1 = v <= 0.0
    cond2 = v >= eps
    t = v / eps
    p = eps * (6.0 * (t**3) - 8.0 * (t**4) + 3.0 * (t**5))
    return torch.where(cond1, torch.zeros_like(v), torch.where(cond2, v, p))


# ---------------------------------------------------------------------------
# C^2 Cubic B-Spline Basis & Tensor-Product Support (Roadmap v2 §3.4, §3.5)
# ---------------------------------------------------------------------------
def make_clamped_knots(
    domain: Tuple[float, float], num_intervals: int, degree: int = 3
) -> torch.Tensor:
    """Generate clamped knot vector with degree-multiplicity boundary knots."""
    lo, hi = domain
    interior = torch.linspace(lo, hi, num_intervals + 1)[1:-1]
    knots = torch.cat(
        [
            torch.full((degree + 1,), lo, dtype=torch.float64),
            interior,
            torch.full((degree + 1,), hi, dtype=torch.float64),
        ]
    )
    return knots


def bspline_basis_1d(
    x: torch.Tensor, knots: torch.Tensor, degree: int = 3
) -> torch.Tensor:
    """Pure-PyTorch Cox-de Boor C^2 cubic B-spline evaluator. Exact autograd derivatives."""
    # Ensure x stays inside knot range cleanly
    x_clamped = torch.clamp(x, knots[0].item(), knots[-1].item())
    K = len(knots)

    # Find last non-empty interval for boundary inclusion
    last_non_empty = -1
    for i in range(K - 1):
        if knots[i] < knots[i + 1]:
            last_non_empty = i

    d0 = []
    for i in range(K - 1):
        if knots[i] == knots[i + 1]:
            d0.append(torch.zeros_like(x_clamped))
        elif i == last_non_empty:
            cond = (x_clamped >= knots[i]) & (x_clamped <= knots[i + 1])
            d0.append(cond.to(x.dtype))
        else:
            cond = (x_clamped >= knots[i]) & (x_clamped < knots[i + 1])
            d0.append(cond.to(x.dtype))
    bases = torch.stack(d0, dim=-1)

    for p in range(1, degree + 1):
        new_bases = []
        for i in range(K - p - 1):
            w1_denom = knots[i + p] - knots[i]
            w1 = (
                (x_clamped - knots[i]) / w1_denom
                if w1_denom > 0
                else torch.zeros_like(x_clamped)
            )
            term1 = w1 * bases[..., i]

            w2_denom = knots[i + p + 1] - knots[i + 1]
            w2 = (
                (knots[i + p + 1] - x_clamped) / w2_denom
                if w2_denom > 0
                else torch.zeros_like(x_clamped)
            )
            term2 = w2 * bases[..., i + 1]
            new_bases.append(term1 + term2)
        bases = torch.stack(new_bases, dim=-1)
    return bases


def build_second_diff_matrix(n: int) -> np.ndarray:
    """Constructs (n-2, n) second-difference operator matrix."""
    if n <= 2:
        return np.zeros((1, n), dtype=np.float64)
    D = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def evaluate_latent_surface(
    features: torch.Tensor,
    basis_specs: Dict[str, Any],
    coef_r: torch.Tensor,
    coef_rx: List[torch.Tensor],
) -> torch.Tensor:
    """Evaluates additive latent surface F(x, z) = b_r(r) + sum_k b_{rk}(r, x_k).

    Parameters:
      features: (M, D) tensor
      basis_specs: dict containing 'knots_r' and 'knots_x'
      coef_r: (K_r,)
      coef_rx: list of (K_r, K_x_k) coefficient matrices
    Returns:
      F: (M,) tensor
    """
    knots_r = basis_specs["knots_r"]
    basis_r = bspline_basis_1d(features[:, 0], knots_r)  # (M, K_r)
    val = (basis_r * coef_r).sum(dim=-1)  # (M,)

    D = features.shape[1]
    knots_x_list = basis_specs["knots_x"]
    for k in range(D - 1):
        if k < len(coef_rx):
            kx = knots_x_list[k]
            basis_xk = bspline_basis_1d(features[:, k + 1], kx)  # (M, K_x)
            # 2D tensor product: (basis_r @ C * basis_xk).sum(-1)
            C = coef_rx[k]
            val = val + ((basis_r @ C) * basis_xk).sum(dim=-1)
    return val


def evaluate_joint_support(
    features: torch.Tensor,
    support_basis_spec: Dict[str, Any],
    a_grid: torch.Tensor,
) -> torch.Tensor:
    """Evaluates tensor-product joint support function m(x) = sum_j a_j B_j(x) in [0, 1].

    Parameters:
      features: (M, D) tensor
      support_basis_spec: dict with 'knots_r' and 'knots_x'
      a_grid: tensor of shape (K_r, K_x1, ...) matching D
    Returns:
      m: (M,) tensor in [0, 1]
    """
    D = features.shape[1]
    knots_r = support_basis_spec["knots_r"]
    b_r = bspline_basis_1d(features[:, 0], knots_r)
    knots_x = support_basis_spec.get("knots_x", [])

    if len(knots_x) == 0:
        return (b_r * a_grid).sum(dim=-1)
    elif len(knots_x) == 1 or D == 2:
        kx1 = knots_x[0]
        b_x1 = bspline_basis_1d(features[:, 1], kx1)
        return torch.einsum("m i, m j, i j -> m", b_r, b_x1, a_grid)
    elif len(knots_x) >= 3 and D >= 4:
        kx1 = knots_x[0]
        kx2 = knots_x[1]
        kx3 = knots_x[2]
        b_x1 = bspline_basis_1d(features[:, 1], kx1)
        b_x2 = bspline_basis_1d(features[:, 2], kx2)
        b_x3 = bspline_basis_1d(features[:, 3], kx3)
        return torch.einsum(
            "m i, m j, m k, m l, i j k l -> m", b_r, b_x1, b_x2, b_x3, a_grid
        )
    else:
        return (b_r * a_grid).sum(dim=-1)


def _get_atom_charge_class(atom_or_fc: Any) -> int:
    """Formal charge class: 0 if neutral, +1 if positive, -1 if negative."""
    if hasattr(atom_or_fc, "formal_charge"):
        fc = getattr(atom_or_fc, "formal_charge", 0)
    elif hasattr(atom_or_fc, "charge"):
        fc = getattr(atom_or_fc, "charge", 0)
    elif isinstance(atom_or_fc, (int, float)):
        fc = atom_or_fc
    else:
        fc = 0
    if fc > 0:
        return 1
    elif fc < 0:
        return -1
    return 0


def _get_bond_order_class(bo: float) -> str:
    """Bond order class: 'single' (<1.4), 'double' (1.4-2.4), 'triple' (>2.4)."""
    if bo < 1.4:
        return "single"
    elif bo <= 2.4:
        return "double"
    else:
        return "triple"


# ---------------------------------------------------------------------------
# Declarative Admission & Opportunity Enumeration (Roadmap v2 §3.2, §3.9)
# ---------------------------------------------------------------------------
def enumerate_opportunities(
    mol: Any, schema: Optional[Dict[str, Any]] = None
) -> OpportunityBatch:
    """Enumerates candidate NCI occurrences according to Roadmap v2 §3.2 admission rules."""
    from research.archive.code.interactions.q6pmf import ligand_separation
    from jacobik.topology.graph import (
        donor_indices_of,
        get_ligand_neighbors,
        wl_classes,
    )

    if schema is not None:
        if "admission_rules" in schema:
            sc_rules = schema["admission_rules"]
        elif any(f in schema for f in ("bond_atom", "bond_ring", "ring_ring")):
            sc_rules = schema
        else:
            sc_rules = DEFAULT_SCHEMA["admission_rules"]
    else:
        sc_rules = DEFAULT_SCHEMA["admission_rules"]

    def _is_rule_active(fam_rules: Any, class_name: str) -> bool:
        if isinstance(fam_rules, dict):
            return class_name in fam_rules
        elif isinstance(fam_rules, (list, tuple, set)):
            return class_name in fam_rules
        return False

    def _get_rule_cfg(fam_rules: Any, class_name: str) -> Dict[str, Any]:
        if isinstance(fam_rules, dict) and isinstance(fam_rules.get(class_name), dict):
            return fam_rules[class_name]
        return {}

    min_sep = int(sc_rules.get("min_ligand_graph_separation_edges", 5))

    metal_idx = getattr(mol, "metal_idx", -1)
    atoms = mol.atoms
    bonds = mol.bonds
    sym = {a.idx: a.symbol for a in atoms}
    atom_pos = {a.idx: i for i, a in enumerate(atoms)}

    nb = get_ligand_neighbors(atoms, bonds, metal_idx)
    sep = ligand_separation(mol, max_depth=min_sep - 1)

    try:
        bound = set(donor_indices_of(mol))
    except Exception:
        bound = set()

    cls = wl_classes([a.idx for a in atoms], nb, sym)

    # Compute connected ligand graph components
    comp_map = {}
    visited = set()
    cid = 0
    for a in atoms:
        idx = a.idx
        if idx == metal_idx or idx in visited:
            continue
        q = [idx]
        visited.add(idx)
        for u in q:
            comp_map[u] = cid
            for v in nb.get(u, ()):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        cid += 1

    rings = sp2_rings_v2(mol, nb)

    # Double/multiple bond lookup
    bond_orders = {}
    for b in bonds:
        pair = (min(b.i, b.j), max(b.i, b.j))
        bond_orders[pair] = max(bond_orders.get(pair, 0.0), getattr(b, "order", 1.0))

    opportunities: Dict[str, List[Dict[str, Any]]] = {
        "bond_atom": [],
        "bond_ring": [],
        "ring_ring": [],
    }

    # Declarative rule configs for bond_atom
    ba_rules = sc_rules.get("bond_atom", {})
    pha_active = _is_rule_active(ba_rules, "polar_h_acceptor")
    pha_cfg = _get_rule_cfg(ba_rules, "polar_h_acceptor")
    pha_donors = set(pha_cfg.get("donor_A_elements", ["N", "O", "F", "S"]))
    pha_h = pha_cfg.get("hydrogen_B_element", "H")
    pha_acceptors = set(pha_cfg.get("acceptor_C_elements", list(ACCEPTORS)))
    pha_allow_anionic_c = bool(pha_cfg.get("allow_anionic_carbon_acceptor", False))

    ach_active = _is_rule_active(ba_rules, "activated_ch_acceptor")
    ach_cfg = _get_rule_cfg(ba_rules, "activated_ch_acceptor")
    ach_donors = set(ach_cfg.get("donor_A_elements", [ach_cfg.get("donor_A_element", "C")]))
    ach_h = ach_cfg.get("hydrogen_B_element", "H")
    ach_acceptors = set(ach_cfg.get("acceptor_C_elements", list(ACCEPTORS)))
    ach_hetero = set(ach_cfg.get("heteroatom_elements", list(ACCEPTORS | {"P"})))

    sp2_active = _is_rule_active(ba_rules, "sp2_ch_acceptor")
    sp2_cfg = _get_rule_cfg(ba_rules, "sp2_ch_acceptor")
    sp2_donors = set(sp2_cfg.get("donor_A_elements", [sp2_cfg.get("donor_A_element", "C")]))
    sp2_h = sp2_cfg.get("hydrogen_B_element", "H")
    sp2_acceptors = set(sp2_cfg.get("acceptor_C_elements", list(ACCEPTORS)))
    sp2_max_deg = int(sp2_cfg.get("donor_max_degree", 3))

    sha_active = _is_rule_active(ba_rules, "sigma_hole_acceptor")
    sha_cfg = _get_rule_cfg(ba_rules, "sigma_hole_acceptor")
    sha_donors = set(sha_cfg.get("donor_A_elements", ["C", "N", "O", "S", "P"]))
    sha_b = set(sha_cfg.get("polar_atom_B_elements", ["Cl", "Br", "I", "S", "Se"]))
    sha_acceptors = set(sha_cfg.get("acceptor_C_elements", list(ACCEPTORS)))

    pmb_active = _is_rule_active(ba_rules, "polar_multiple_bond")
    pmb_cfg = _get_rule_cfg(ba_rules, "polar_multiple_bond")
    pmb_donors = set(pmb_cfg.get("endpoint_A_elements", ["C", "N", "S", "P"]))
    pmb_b = set(pmb_cfg.get("endpoint_B_elements", ["O", "N", "S"]))
    pmb_min_bo = float(pmb_cfg.get("min_bond_order", 1.4))
    pmb_acceptors = set(pmb_cfg.get("acceptor_C_elements", list(ACCEPTORS)))

    # 1. bond_atom admission
    eligible_bonds = []
    seen_bonds = set()
    for b in bonds:
        if b.i == metal_idx or b.j == metal_idx:
            continue
        bp = (min(b.i, b.j), max(b.i, b.j))
        if bp not in seen_bonds:
            seen_bonds.add(bp)
            eligible_bonds.append(b)

    for b in eligible_bonds:
        u, v = b.i, b.j
        is_sym = cls.get(u) == cls.get(v)
        directions = [(u, v)] if is_sym else [(u, v), (v, u)]

        for A, B in directions:
            sym_A, sym_B = sym.get(A, ""), sym.get(B, "")
            bo = bond_orders.get((min(A, B), max(A, B)), 1.0)

            # Potential partner C
            for C_atom in atoms:
                C = C_atom.idx
                if C == metal_idx or C == A or C == B:
                    continue
                # >4 ligand-graph edges separation
                if C in sep.get(A, {}) or C in sep.get(B, {}):
                    continue

                sym_C = C_atom.symbol
                adm_class = None

                # Rule 1: polar_h_acceptor
                if pha_active and sym_A in pha_donors and sym_B == pha_h:
                    if sym_C in pha_acceptors or (
                        pha_allow_anionic_c
                        and sym_C == "C"
                        and _get_atom_charge_class(C_atom) < 0
                    ):
                        adm_class = "polar_h_acceptor"

                # Rule 2: activated_ch_acceptor or sp2_ch_acceptor
                if (
                    adm_class is None
                    and sym_A in ach_donors
                    and sym_B == ach_h
                    and sym_C in ach_acceptors
                ):
                    deg_A = len(nb.get(A, ()))
                    is_sp = deg_A <= 2
                    is_bound = A in bound
                    has_hetero = any(sym.get(x) in ach_hetero for x in nb.get(A, ()))
                    is_activated = is_sp or is_bound or has_hetero
                    if is_activated and ach_active:
                        adm_class = "activated_ch_acceptor"
                    elif (not is_activated) and deg_A <= sp2_max_deg and sp2_active:
                        adm_class = "sp2_ch_acceptor"

                # Rule 3: sigma_hole_acceptor
                if (
                    adm_class is None
                    and sha_active
                    and sym_A in sha_donors
                    and sym_B in sha_b
                    and sym_C in sha_acceptors
                ):
                    adm_class = "sigma_hole_acceptor"

                # Rule 4: polar_multiple_bond
                if (
                    adm_class is None
                    and pmb_active
                    and sym_A in pmb_donors
                    and sym_B in pmb_b
                    and bo >= pmb_min_bo
                    and sym_C in pmb_acceptors
                ):
                    adm_class = "polar_multiple_bond"

                if adm_class is not None:
                    c_neighbors = [
                        x for x in nb.get(C, ()) if x != metal_idx and x != C
                    ]
                    k_C = len(c_neighbors)
                    has_c_nb = k_C > 0
                    is_intra = comp_map.get(A) == comp_map.get(C)
                    binding_role = (A in bound, B in bound, C in bound)

                    endpoint_elements = (sym_A, sym_B, sym_C)
                    endpoint_charge_classes = (
                        _get_atom_charge_class(atoms[atom_pos[A]]),
                        _get_atom_charge_class(atoms[atom_pos[B]]),
                        _get_atom_charge_class(C_atom),
                    )
                    bo_class = _get_bond_order_class(bo)
                    ring_composition = None

                    root_key = (
                        "bond_atom",
                        adm_class,
                        endpoint_elements,
                        endpoint_charge_classes,
                        bo_class,
                        ring_composition,
                        binding_role,
                        is_intra,
                        has_c_nb,
                    )
                    level2_key = (root_key, k_C)
                    level3_key = (
                        level2_key,
                        getattr(mol, "metal_symbol", "None"),
                        getattr(mol, "coord_geom", "None"),
                    )

                    occ = {
                        "occ_id": f"ba_{A}_{B}_{C}",
                        "family": "bond_atom",
                        "admission_class": adm_class,
                        "idx_A": A,
                        "idx_B": B,
                        "idx_C": C,
                        "pos_A": atom_pos[A],
                        "pos_B": atom_pos[B],
                        "pos_C": atom_pos[C],
                        "neighbor_indices": c_neighbors,
                        "neighbor_pos": [atom_pos[x] for x in c_neighbors],
                        "k_C": k_C,
                        "has_c_neighbors": has_c_nb,
                        "symmetric_bond": is_sym,
                        "z_o": (root_key, level2_key, level3_key),
                        "B_is_H": (sym_B == "H"),
                    }
                    opportunities["bond_atom"].append(occ)

    # 2. bond_ring admission
    br_rules = sc_rules.get("bond_ring", {})
    dhr_active = _is_rule_active(br_rules, "dh_ring")
    chr_active = _is_rule_active(br_rules, "ch_ring")
    lpr_active = _is_rule_active(br_rules, "lp_or_halogen_ring")
    dhr_donors = set(_get_rule_cfg(br_rules, "dh_ring").get("donor_A_elements", ["N", "O", "F", "S"]))
    dhr_h = _get_rule_cfg(br_rules, "dh_ring").get("hydrogen_B_element", "H")
    chr_donors = set(_get_rule_cfg(br_rules, "ch_ring").get("donor_A_elements", ["C"]))
    chr_h = _get_rule_cfg(br_rules, "ch_ring").get("hydrogen_B_element", "H")
    lpr_b = set(_get_rule_cfg(br_rules, "lp_or_halogen_ring").get("atom_B_elements", ["F", "Cl", "Br", "I"]))

    for b in eligible_bonds:
        u, v = b.i, b.j
        is_sym = cls.get(u) == cls.get(v)
        directions = [(u, v)] if is_sym else [(u, v), (v, u)]

        for A, B in directions:
            sym_A, sym_B = sym.get(A, ""), sym.get(B, "")
            bo = bond_orders.get((min(A, B), max(A, B)), 1.0)

            for R_idx, R in enumerate(rings):
                R_set = set(R)
                if A in R_set or B in R_set:
                    continue
                # All constituent atoms > 4 edges
                if any(v in sep.get(A, {}) or v in sep.get(B, {}) for v in R):
                    continue

                adm_class = None
                if sym_B == dhr_h and sym_A in dhr_donors and dhr_active:
                    adm_class = "dh_ring"
                elif sym_B == chr_h and sym_A in chr_donors and chr_active:
                    adm_class = "ch_ring"
                elif sym_B in lpr_b and lpr_active:
                    adm_class = "lp_or_halogen_ring"

                if adm_class is not None:
                    is_intra = comp_map.get(A) == comp_map.get(R[0])
                    ring_bound = any(x in bound for x in R)
                    binding_role = (A in bound, B in bound, ring_bound)

                    endpoint_elements = (sym_A, sym_B)
                    endpoint_charge_classes = (
                        _get_atom_charge_class(atoms[atom_pos[A]]),
                        _get_atom_charge_class(atoms[atom_pos[B]]),
                    )
                    bo_class = _get_bond_order_class(bo)
                    ring_composition = tuple(sorted(sym[x] for x in R))

                    root_key = (
                        "bond_ring",
                        adm_class,
                        endpoint_elements,
                        endpoint_charge_classes,
                        bo_class,
                        ring_composition,
                        binding_role,
                        is_intra,
                        False,
                    )
                    level2_key = (root_key, len(R))
                    level3_key = (
                        level2_key,
                        getattr(mol, "metal_symbol", "None"),
                        getattr(mol, "coord_geom", "None"),
                    )

                    occ = {
                        "occ_id": f"br_{A}_{B}_R{R_idx}",
                        "family": "bond_ring",
                        "admission_class": adm_class,
                        "idx_A": A,
                        "idx_B": B,
                        "ring_indices": R,
                        "pos_A": atom_pos[A],
                        "pos_B": atom_pos[B],
                        "ring_pos": [atom_pos[x] for x in R],
                        "ring_size": len(R),
                        "symmetric_bond": is_sym,
                        "z_o": (root_key, level2_key, level3_key),
                    }
                    opportunities["bond_ring"].append(occ)

    # 3. ring_ring admission
    rr_rules = sc_rules.get("ring_ring", {})
    rr_active = _is_rule_active(rr_rules, "sp2_5_6_ring_pair") or _is_rule_active(rr_rules, "ring_ring")

    if rr_active:
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                R, S = rings[i], rings[j]
                if set(R) & set(S):
                    continue
                if any(v in sep.get(u, {}) for u in R for v in S):
                    continue

                is_intra = comp_map.get(R[0]) == comp_map.get(S[0])
                r_bound = any(x in bound for x in R)
                s_bound = any(x in bound for x in S)
                r_comp = tuple(sorted(sym[x] for x in R))
                s_comp = tuple(sorted(sym[x] for x in S))

                if (r_comp, r_bound) <= (s_comp, s_bound):
                    ring_composition = (r_comp, s_comp)
                    binding_role = (r_bound, s_bound)
                else:
                    ring_composition = (s_comp, r_comp)
                    binding_role = (s_bound, r_bound)

                root_key = (
                    "ring_ring",
                    "sp2_5_6_ring_pair",
                    None,
                    None,
                    None,
                    ring_composition,
                    binding_role,
                    is_intra,
                    False,
                )
                level2_key = (root_key, len(R), len(S))
                level3_key = (
                    level2_key,
                    getattr(mol, "metal_symbol", "None"),
                    getattr(mol, "coord_geom", "None"),
                )

                occ = {
                    "occ_id": f"rr_R{i}_R{j}",
                    "family": "ring_ring",
                    "admission_class": "sp2_5_6_ring_pair",
                    "ring_indices_R": R,
                    "ring_indices_S": S,
                    "ring_pos_R": [atom_pos[x] for x in R],
                    "ring_pos_S": [atom_pos[x] for x in S],
                    "ring_size_R": len(R),
                    "ring_size_S": len(S),
                    "z_o": (root_key, level2_key, level3_key),
                }
                opportunities["ring_ring"].append(occ)

    tot = sum(len(v) for v in opportunities.values())
    return OpportunityBatch(
        mol_id=getattr(mol, "csd_code", "mol"),
        opportunities=opportunities,
        atom_pos=atom_pos,
        total_count=tot,
        metadata={"num_rings": len(rings)},
    )


# ---------------------------------------------------------------------------
# Vectorized Continuous Feature Evaluation (Roadmap v2 §3.2, §3.9)
# ---------------------------------------------------------------------------
def features_torch(
    X: torch.Tensor,
    opportunities: OpportunityBatch,
    schema: Optional[Dict[str, Any]] = None,
) -> FeatureBatch:
    """Computes continuous regularized features x_o(X) in pure PyTorch."""
    if X.ndim == 2:
        X = X.unsqueeze(0)
    B, N, _ = X.shape
    device = X.device
    dtype = X.dtype

    features_out: Dict[str, torch.Tensor] = {}
    occ_ids_out: Dict[str, List[str]] = {}
    contexts_out: Dict[str, List[Tuple[Any, ...]]] = {}
    masks_out: Dict[str, torch.Tensor] = {}
    units_out: Dict[str, List[str]] = {
        "bond_atom": [
            "Angstrom",
            "dimensionless",
            "dimensionless",
            "dimensionless",
        ],
        "bond_ring": [
            "Angstrom",
            "dimensionless",
            "dimensionless",
            "dimensionless",
        ],
        "ring_ring": [
            "Angstrom",
            "dimensionless",
            "dimensionless",
            "dimensionless",
        ],
    }

    # 1. bond_atom features
    ba_cands = opportunities.opportunities.get("bond_atom", [])
    occ_ids_out["bond_atom"] = [c["occ_id"] for c in ba_cands]
    contexts_out["bond_atom"] = [c["z_o"] for c in ba_cands]

    if not ba_cands:
        features_out["bond_atom"] = torch.zeros((B, 0, 4), device=device, dtype=dtype)
        masks_out["bond_atom"] = torch.zeros((0,), device=device, dtype=torch.bool)
    else:
        O = len(ba_cands)
        pos_A = torch.tensor([c["pos_A"] for c in ba_cands], device=device)
        pos_B = torch.tensor([c["pos_B"] for c in ba_cands], device=device)
        pos_C = torch.tensor([c["pos_C"] for c in ba_cands], device=device)

        P_A = X[:, pos_A, :]  # (B, O, 3)
        P_B = X[:, pos_B, :]  # (B, O, 3)
        P_C = X[:, pos_C, :]  # (B, O, 3)

        v_CB = P_C - P_B
        v_BA = P_B - P_A

        r = reg_norm(v_CB, DELTA_R)  # (B, O, 1)
        u_CB = reg_unit(v_CB, DELTA_R)
        u_BA = reg_unit(v_BA, DELTA_R)
        x_1 = (u_BA * u_CB).sum(dim=-1, keepdim=True)

        k_max = max([c["k_C"] for c in ba_cands] + [1])
        nb_tensor = torch.zeros((O, k_max), dtype=torch.long, device=device)
        mask_nb = torch.zeros((O, k_max), dtype=torch.bool, device=device)
        k_C_t = torch.tensor([c["k_C"] for c in ba_cands], device=device, dtype=dtype)

        for i, c in enumerate(ba_cands):
            for j, p in enumerate(c["neighbor_pos"]):
                nb_tensor[i, j] = p
                mask_nb[i, j] = True

        P_D = X[:, nb_tensor, :]  # (B, O, k_max, 3)
        P_C_exp = P_C.unsqueeze(2)
        P_B_exp = P_B.unsqueeze(2)

        v_DC = P_D - P_C_exp
        v_BC = P_B_exp - P_C_exp
        u_DC = reg_unit(v_DC, DELTA_R)
        u_BC = reg_unit(v_BC, DELTA_R)

        c_D = (u_DC * u_BC).sum(dim=-1)  # (B, O, k_max)
        mask_exp = mask_nb.unsqueeze(0)
        c_D_masked = torch.where(mask_exp, c_D, torch.zeros_like(c_D))

        k_clamped = torch.clamp(k_C_t.unsqueeze(0), min=1.0)
        mu_1 = c_D_masked.sum(dim=-1) / k_clamped
        mu_2 = (c_D_masked**2).sum(dim=-1) / k_clamped

        has_c = (k_C_t > 0).unsqueeze(0)
        mu_1 = torch.where(has_c, mu_1, torch.zeros_like(mu_1)).unsqueeze(-1)
        mu_2 = torch.where(has_c, mu_2, torch.zeros_like(mu_2)).unsqueeze(-1)

        features_out["bond_atom"] = torch.cat([r, x_1, mu_1, mu_2], dim=-1)
        masks_out["bond_atom"] = (k_C_t > 0)

    # 2. bond_ring features
    br_cands = opportunities.opportunities.get("bond_ring", [])
    occ_ids_out["bond_ring"] = [c["occ_id"] for c in br_cands]
    contexts_out["bond_ring"] = [c["z_o"] for c in br_cands]

    if not br_cands:
        features_out["bond_ring"] = torch.zeros((B, 0, 4), device=device, dtype=dtype)
        masks_out["bond_ring"] = torch.zeros((0,), device=device, dtype=torch.bool)
    else:
        O = len(br_cands)
        pos_A = torch.tensor([c["pos_A"] for c in br_cands], device=device)
        pos_B = torch.tensor([c["pos_B"] for c in br_cands], device=device)
        sz_t = torch.tensor([c["ring_size"] for c in br_cands], device=device)

        P_A = X[:, pos_A, :]
        P_B = X[:, pos_B, :]

        ring_mat = torch.zeros((O, 6), dtype=torch.long, device=device)
        for i, c in enumerate(br_cands):
            for j, p in enumerate(c["ring_pos"]):
                ring_mat[i, j] = p

        P_ring = X[:, ring_mat, :]  # (B, O, 6, 3)
        c_R, n_R = compute_ring_frame(P_ring, sz_t, DELTA_N)

        v_r = c_R - P_B
        v_BA = P_B - P_A
        v_Bc = P_B - c_R

        r = reg_norm(v_r, DELTA_R)
        u_r = reg_unit(v_r, DELTA_R)
        u_BA = reg_unit(v_BA, DELTA_R)
        u_Bc = reg_unit(v_Bc, DELTA_R)

        x_1 = (u_BA * u_r).sum(dim=-1, keepdim=True)
        x_2 = ((n_R * u_Bc).sum(dim=-1, keepdim=True)) ** 2
        x_3 = ((n_R * u_BA).sum(dim=-1, keepdim=True)) ** 2

        features_out["bond_ring"] = torch.cat([r, x_1, x_2, x_3], dim=-1)
        masks_out["bond_ring"] = torch.ones((O,), device=device, dtype=torch.bool)

    # 3. ring_ring features
    rr_cands = opportunities.opportunities.get("ring_ring", [])
    occ_ids_out["ring_ring"] = [c["occ_id"] for c in rr_cands]
    contexts_out["ring_ring"] = [c["z_o"] for c in rr_cands]

    if not rr_cands:
        features_out["ring_ring"] = torch.zeros((B, 0, 4), device=device, dtype=dtype)
        masks_out["ring_ring"] = torch.zeros((0,), device=device, dtype=torch.bool)
    else:
        O = len(rr_cands)
        sz_R = torch.tensor([c["ring_size_R"] for c in rr_cands], device=device)
        sz_S = torch.tensor([c["ring_size_S"] for c in rr_cands], device=device)

        ring_mat_R = torch.zeros((O, 6), dtype=torch.long, device=device)
        ring_mat_S = torch.zeros((O, 6), dtype=torch.long, device=device)
        for i, c in enumerate(rr_cands):
            for j, p in enumerate(c["ring_pos_R"]):
                ring_mat_R[i, j] = p
            for j, p in enumerate(c["ring_pos_S"]):
                ring_mat_S[i, j] = p

        P_ring_R = X[:, ring_mat_R, :]
        P_ring_S = X[:, ring_mat_S, :]
        c_R, n_R = compute_ring_frame(P_ring_R, sz_R, DELTA_N)
        c_S, n_S = compute_ring_frame(P_ring_S, sz_S, DELTA_N)

        v_SR = c_S - c_R
        r = reg_norm(v_SR, DELTA_R)
        d = reg_unit(v_SR, DELTA_R)

        x_1 = ((n_R * n_S).sum(dim=-1, keepdim=True)) ** 2
        dot_R = (n_R * d).sum(dim=-1, keepdim=True)
        dot_S = (n_S * d).sum(dim=-1, keepdim=True)
        x_2 = (dot_R**2) + (dot_S**2)
        x_3 = (dot_R**2) * (dot_S**2)

        features_out["ring_ring"] = torch.cat([r, x_1, x_2, x_3], dim=-1)
        masks_out["ring_ring"] = torch.ones((O,), device=device, dtype=torch.bool)

    return FeatureBatch(
        features=features_out,
        occurrence_ids=occ_ids_out,
        contexts=contexts_out,
        masks=masks_out,
        units=units_out,
    )


# ---------------------------------------------------------------------------
# Density-Ratio & Support Fitting (Roadmap v2 §3.4, §3.5)
# ---------------------------------------------------------------------------
def fit_density_ratio(
    matched_rows: List[Dict[str, Any]],
    fit_split: Union[List[str], Set[str]],
    spec: Optional[Dict[str, Any]] = None,
) -> SurfaceArtifact:
    """Fits regularized log-density ratio F(x, z) and joint support function m(x)."""
    sp = spec or {}
    lambda_r = sp.get("lambda_r", 0.1)
    lambda_0 = sp.get("lambda_0", 0.01)
    lambda_z = sp.get("lambda_z", 0.1)
    tau_grp = sp.get("tau_grp", 1e-3)
    min_ref_groups = sp.get("min_ref_groups", 10)
    min_base_groups = sp.get("min_base_groups", 10)
    min_pool_groups = sp.get("min_pool_groups", 20)

    fit_groups = set(fit_split)
    valid_rows = [r for r in matched_rows if r.get("group_id") in fit_groups]

    family = matched_rows[0]["family"] if matched_rows else "bond_atom"

    # Context grouping: root_key
    by_context: Dict[Any, List[Dict[str, Any]]] = {}
    for r in valid_rows:
        ctx = r.get("context")
        root_key = ctx[0] if isinstance(ctx, (tuple, list)) else ctx
        by_context.setdefault(root_key, []).append(r)

    surfaces_out: Dict[str, Dict[str, Any]] = {}

    for root_key, rows in by_context.items():
        ctx_str = str(root_key)
        X_all = np.array([r["features"] for r in rows], dtype=np.float64)
        y_all = np.array([r["y"] for r in rows], dtype=np.float64)
        w_all = np.array([r.get("weight", 1.0) for r in rows], dtype=np.float64)
        groups = [r.get("group_id") for r in rows]

        # Determine feature dimensions and domains
        D = X_all.shape[1]
        is_H = True
        if family == "bond_atom" and isinstance(root_key, (tuple, list)) and len(root_key) >= 3:
            ep_elem = root_key[2]
            if isinstance(ep_elem, (tuple, list)) and len(ep_elem) >= 2:
                is_H = (ep_elem[1] == "H")

        r_win = DEFAULT_RADIAL_WINDOWS.get(
            f"{family}_{'H' if is_H else 'heavy'}",
            DEFAULT_RADIAL_WINDOWS.get(family, (1.1, 1.35, 6.0, 8.0)),
        )
        r_domain = (r_win[0], r_win[3])

        # Support grid specs
        knots_r_sup = make_clamped_knots(r_domain, 6, degree=3)
        knots_x_sup_list = []
        for k in range(D - 1):
            x_min = 0.0 if (family in ("bond_ring", "ring_ring") and k >= 1) else -1.0
            x_max = 2.0 if (family == "ring_ring" and k == 1) else 1.0
            knots_x_sup_list.append(make_clamped_knots((x_min, x_max), 3, degree=3))

        # Fit grid specs
        knots_r_fit = make_clamped_knots(r_domain, 4, degree=3)
        knots_x_fit_list = []
        for k in range(D - 1):
            x_min = 0.0 if (family in ("bond_ring", "ring_ring") and k >= 1) else -1.0
            x_max = 2.0 if (family == "ring_ring" and k == 1) else 1.0
            knots_x_fit_list.append(make_clamped_knots((x_min, x_max), 3, degree=3))

        # Evaluate support basis for counting
        X_torch = torch.from_numpy(X_all)
        b_r_sup = bspline_basis_1d(X_torch[:, 0], knots_r_sup).numpy()
        b_x_sup = [
            bspline_basis_1d(X_torch[:, k + 1], knots_x_sup_list[k]).numpy()
            for k in range(D - 1)
        ]

        Kr_s = b_r_sup.shape[1]
        Kx_s = [b.shape[1] for b in b_x_sup]

        # Calculate group support masses
        unique_groups = sorted(list(set(groups)))
        grid_shape = tuple([Kr_s] + Kx_s)
        v_ref = np.zeros((len(unique_groups),) + grid_shape, dtype=np.float64)
        v_base = np.zeros((len(unique_groups),) + grid_shape, dtype=np.float64)

        for g_idx, g in enumerate(unique_groups):
            g_mask_ref = np.array([grp == g and y == 1 for grp, y in zip(groups, y_all)])
            g_mask_base = np.array([grp == g and y == 0 for grp, y in zip(groups, y_all)])

            if np.any(g_mask_ref):
                w_g = w_all[g_mask_ref]
                br_g = b_r_sup[g_mask_ref]
                if D == 2:
                    prod = np.einsum("i, i j, i k -> j k", w_g, br_g, b_x_sup[0][g_mask_ref])
                elif D == 4:
                    prod = np.einsum(
                        "i, i j, i k, i l, i m -> j k l m",
                        w_g,
                        br_g,
                        b_x_sup[0][g_mask_ref],
                        b_x_sup[1][g_mask_ref],
                        b_x_sup[2][g_mask_ref],
                    )
                else:
                    prod = (w_g[:, None] * br_g).sum(axis=0)
                v_ref[g_idx] = prod

            if np.any(g_mask_base):
                w_g = w_all[g_mask_base]
                br_g = b_r_sup[g_mask_base]
                if D == 2:
                    prod = np.einsum("i, i j, i k -> j k", w_g, br_g, b_x_sup[0][g_mask_base])
                elif D == 4:
                    prod = np.einsum(
                        "i, i j, i k, i l, i m -> j k l m",
                        w_g,
                        br_g,
                        b_x_sup[0][g_mask_base],
                        b_x_sup[1][g_mask_base],
                        b_x_sup[2][g_mask_base],
                    )
                else:
                    prod = (w_g[:, None] * br_g).sum(axis=0)
                v_base[g_idx] = prod

        # Group support statistics
        w_bar_grp = np.mean([w_all[np.array(groups) == g].sum() for g in unique_groups])
        tau = tau_grp * w_bar_grp

        N_ref = (v_ref > tau).sum(axis=0)
        sum_v_ref = v_ref.sum(axis=0)
        sum_v2_ref = (v_ref**2).sum(axis=0)
        ESS_ref = np.where(sum_v2_ref > 1e-12, (sum_v_ref**2) / (sum_v2_ref + 1e-12), 0.0)

        N_base = (v_base > tau).sum(axis=0)
        sum_v_base = v_base.sum(axis=0)
        sum_v2_base = (v_base**2).sum(axis=0)
        ESS_base = np.where(sum_v2_base > 1e-12, (sum_v_base**2) / (sum_v2_base + 1e-12), 0.0)

        N_pool = ((v_ref + v_base) > tau).sum(axis=0)
        N_root_pool = len(unique_groups)

        # Regimes and support coefficients a_j
        a_grid = np.zeros(grid_shape, dtype=np.float64)
        M_ref_only = np.zeros(grid_shape, dtype=bool)

        matched_mask = (
            (N_ref >= min_ref_groups)
            & (ESS_ref >= min_ref_groups)
            & (N_base >= min_base_groups)
            & (ESS_base >= min_base_groups)
        )
        ref_only_mask = (
            (N_ref >= min_ref_groups)
            & (ESS_ref >= min_ref_groups)
            & ((N_base < min_base_groups) | (ESS_base < min_base_groups))
            & (N_root_pool >= min_pool_groups)
        )

        a_grid[matched_mask] = 1.0
        a_grid[ref_only_mask] = 1.0
        M_ref_only[ref_only_mask] = True

        # Fit latent spline F_beta
        b_r_fit = bspline_basis_1d(X_torch[:, 0], knots_r_fit).numpy()
        b_x_fit = [
            bspline_basis_1d(X_torch[:, k + 1], knots_x_fit_list[k]).numpy()
            for k in range(D - 1)
        ]

        Phi_blocks = [b_r_fit]
        for bx in b_x_fit:
            # 2D interaction basis
            Phi_2d = b_r_fit[:, :, None] * bx[:, None, :]  # (M, Kr, Kx)
            Phi_blocks.append(Phi_2d.reshape(len(X_all), -1))

        Phi = np.concatenate(Phi_blocks, axis=1)  # (M, P)
        P = Phi.shape[1]

        # Normalized class weights
        ref_mask = y_all == 1
        base_mask = y_all == 0
        w_norm = np.copy(w_all)
        if ref_mask.sum() > 0:
            w_norm[ref_mask] /= (2.0 * w_norm[ref_mask].sum())
        if base_mask.sum() > 0:
            w_norm[base_mask] /= (2.0 * w_norm[base_mask].sum())

        # Roughness matrix
        Kr_f = b_r_fit.shape[1]
        D2_r = build_second_diff_matrix(Kr_f)
        K_rough = np.zeros((P, P), dtype=np.float64)
        K_rough[:Kr_f, :Kr_f] = D2_r.T @ D2_r

        cur_idx = Kr_f
        for bx in b_x_fit:
            Kx_f = bx.shape[1]
            D2_x = build_second_diff_matrix(Kx_f)
            # 2D Kronecker roughness
            K_rx = (
                np.kron(np.eye(Kx_f), D2_r.T @ D2_r)
                + np.kron(D2_x.T @ D2_x, np.eye(Kr_f))
            )
            K_rough[cur_idx : cur_idx + Kr_f * Kx_f, cur_idx : cur_idx + Kr_f * Kx_f] = K_rx
            cur_idx += Kr_f * Kx_f

        # Logistic loss with L-BFGS-B
        def loss_and_grad(beta):
            logits = Phi @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            # weighted loss
            log_1_exp = np.logaddexp(0.0, logits)
            loss_data = np.sum(w_norm * (log_1_exp - y_all * logits))
            reg_rough = lambda_r * (beta.T @ K_rough @ beta)
            reg_ridge = lambda_0 * np.sum(beta**2)
            total_loss = loss_data + reg_rough + reg_ridge

            grad_data = Phi.T @ (w_norm * (p - y_all))
            grad_rough = 2.0 * lambda_r * (K_rough @ beta)
            grad_ridge = 2.0 * lambda_0 * beta
            return total_loss, grad_data + grad_rough + grad_ridge

        beta0 = np.zeros(P, dtype=np.float64)
        res = scipy.optimize.minimize(
            loss_and_grad, beta0, jac=True, method="L-BFGS-B"
        )
        beta_opt = res.x

        coef_r = beta_opt[:Kr_f]
        coef_rx = []
        cur_idx = Kr_f
        for bx in b_x_fit:
            Kx_f = bx.shape[1]
            coef_rx.append(beta_opt[cur_idx : cur_idx + Kr_f * Kx_f].reshape(Kr_f, Kx_f))
            cur_idx += Kr_f * Kx_f

        surfaces_out[ctx_str] = {
            "root_key": root_key,
            "coef_r": coef_r,
            "coef_rx": coef_rx,
            "basis_specs": {
                "knots_r": knots_r_fit,
                "knots_x": knots_x_fit_list,
            },
            "support_basis_spec": {
                "knots_r": knots_r_sup,
                "knots_x": knots_x_sup_list,
            },
            "support_a": a_grid,
            "ref_only_mask": M_ref_only,
            "radial_window": r_win,
            "counts_ref": N_ref,
            "ess_ref": ESS_ref,
            "counts_base": N_base,
            "ess_base": ESS_base,
        }

    return SurfaceArtifact(family=family, surfaces=surfaces_out, spec=sp)


# ---------------------------------------------------------------------------
# Offline Response Scale Calibration (Roadmap v2 §3.7)
# ---------------------------------------------------------------------------
def fit_response_scales(
    surfaces: Union[Dict[str, SurfaceArtifact], SurfaceArtifact],
    motion_labels: List[Dict[str, Any]],
    calibration_split: Union[List[str], Set[str]],
    spec: Optional[Dict[str, Any]] = None,
) -> CalibratedArtifact:
    """Calibrates shared and family NCI energy/force scales gamma >= 0 with SVD reporting."""
    sp = spec or {}
    lambda_F = sp.get("lambda_F", 1.0)
    lambda_gamma = sp.get("lambda_gamma", 1.0)
    cal_groups = set(calibration_split)

    surface_dict = (
        surfaces
        if isinstance(surfaces, dict)
        else {surfaces.family: surfaces}
    )

    valid_labels = [m for m in motion_labels if m.get("group_id") in cal_groups]
    if not valid_labels:
        return CalibratedArtifact(
            surface_artifacts=surface_dict,
            gamma_shared=1.0,
            family_scales={f: 1.0 for f in FAMILIES},
            selected_mode="shared",
            singular_values=[1.0],
            condition_number=1.0,
            ref_only_traversal_passed={f: True for f in FAMILIES},
            spec=sp,
        )

    # Mandatory ref_only traversal check
    ref_only_passed: Dict[str, bool] = {}
    for fam, art in surface_dict.items():
        has_ref_only = any(
            np.any(s["ref_only_mask"]) for s in art.surfaces.values()
        )
        if has_ref_only:
            # Check traversals in calibration motions
            traversing_groups = set()
            for m in valid_labels:
                occ_ref_only = m.get("ref_only_traversed", False)
                if occ_ref_only and abs(m.get("D_f", {}).get(fam, 0.0)) > 0.1:
                    traversing_groups.add(m.get("group_id"))
            if len(traversing_groups) < 5:
                ref_only_passed[fam] = False
                # Mask out ref_only cells
                for s in art.surfaces.values():
                    s["support_a"][s["ref_only_mask"]] = 0.0
            else:
                ref_only_passed[fam] = True
        else:
            ref_only_passed[fam] = True

    # Assemble design matrices
    active_fams = [f for f in FAMILIES if f in surface_dict]
    if not active_fams:
        active_fams = ["bond_atom"]

    n_pts = len(valid_labels)
    D_mat = np.zeros((n_pts, len(active_fams)), dtype=np.float64)
    y_vec = np.zeros(n_pts, dtype=np.float64)
    G_mat = np.zeros((n_pts, len(active_fams)), dtype=np.float64)
    g_vec = np.zeros(n_pts, dtype=np.float64)
    w_vec = np.array([m.get("weight", 1.0) for m in valid_labels], dtype=np.float64)

    for i, m in enumerate(valid_labels):
        y_vec[i] = m.get("y", 0.0)
        g_vec[i] = m.get("g", 0.0)
        for j, fam in enumerate(active_fams):
            D_mat[i, j] = m.get("D_f", {}).get(fam, 0.0)
            G_mat[i, j] = m.get("G_f", {}).get(fam, 0.0)

    # Robust target scales floored at 1.0 kJ/mol
    s_E = max(1.0, float(np.std(y_vec)))
    s_F = max(1.0, float(np.std(g_vec)))

    D_std = D_mat / s_E
    y_std = y_vec / s_E
    G_std = G_mat / s_F
    g_std = g_vec / s_F

    # SVD analysis
    full_design = np.vstack([D_std, np.sqrt(lambda_F) * G_std])
    s_vals = np.linalg.svd(full_design, compute_uv=False)
    cond_num = float(s_vals[0] / (s_vals[-1] + 1e-12)) if len(s_vals) > 0 else 1.0

    # 1. Fit shared scale gamma_0 >= 0
    D_tot = D_std.sum(axis=1)
    G_tot = G_std.sum(axis=1)
    w_sum = w_vec.sum() + 1e-12
    w_n = w_vec / w_sum

    num = (w_n * (D_tot * y_std + lambda_F * G_tot * g_std)).sum()
    denom = (w_n * (D_tot**2 + lambda_F * (G_tot**2))).sum()
    gamma_0 = max(0.0, float(num / (denom + 1e-12)))

    # 2. Fit family scales gamma_f >= 0
    def family_obj(gammas):
        res_E = (D_std @ gammas) - y_std
        res_F = (G_std @ gammas) - g_std
        loss = np.sum(w_n * (res_E**2 + lambda_F * (res_F**2)))
        prior = lambda_gamma * np.sum((gammas - gamma_0) ** 2)
        return loss + prior

    init_gammas = np.full(len(active_fams), gamma_0)
    res_opt = scipy.optimize.minimize(
        family_obj,
        init_gammas,
        bounds=[(0.0, None)] * len(active_fams),
        method="L-BFGS-B",
    )
    family_scales_arr = res_opt.x

    family_scales_dict = {
        fam: float(family_scales_arr[i]) for i, fam in enumerate(active_fams)
    }

    return CalibratedArtifact(
        surface_artifacts=surface_dict,
        gamma_shared=gamma_0,
        family_scales=family_scales_dict,
        selected_mode="shared",  # Default baseline mode
        singular_values=[float(x) for x in s_vals],
        condition_number=cond_num,
        ref_only_traversal_passed=ref_only_passed,
        spec=sp,
        metadata={"s_E": s_E, "s_F": s_F},
    )


# ---------------------------------------------------------------------------
# Pre-Indexed Runtime Construction & Autodiff Evaluator (Roadmap v2 §3.9)
# ---------------------------------------------------------------------------
def build_runtime(
    mol: Any,
    calibrated_artifact: CalibratedArtifact,
    device: Union[str, torch.device] = "cpu",
) -> RuntimeNCI:
    """Constructs pre-indexed runtime tensor container on target device."""
    dev = torch.device(device)
    opps = enumerate_opportunities(mol)

    if opps.total_count == 0:
        return RuntimeNCI(
            mol_id=opps.mol_id,
            device=dev,
            is_empty=True,
            family_runtimes={},
            gamma_shared=calibrated_artifact.gamma_shared,
            family_scales=calibrated_artifact.family_scales,
        )

    fam_runtimes: Dict[str, Dict[str, Any]] = {}

    for fam, art in calibrated_artifact.surface_artifacts.items():
        cands = opps.opportunities.get(fam, [])
        if not cands:
            continue

        O = len(cands)
        # Convert pre-computed surface tables to device
        surfaces_tensor = {}
        for k_str, s in art.surfaces.items():
            surfaces_tensor[k_str] = {
                "coef_r": torch.as_tensor(s["coef_r"], device=dev, dtype=torch.float64),
                "coef_rx": [
                    torch.as_tensor(crx, device=dev, dtype=torch.float64)
                    for crx in s["coef_rx"]
                ],
                "basis_specs": {
                    "knots_r": torch.as_tensor(s["basis_specs"]["knots_r"], device=dev, dtype=torch.float64),
                    "knots_x": [
                        torch.as_tensor(kx, device=dev, dtype=torch.float64)
                        for kx in s["basis_specs"]["knots_x"]
                    ],
                },
                "support_basis_spec": {
                    "knots_r": torch.as_tensor(s["support_basis_spec"]["knots_r"], device=dev, dtype=torch.float64),
                    "knots_x": [
                        torch.as_tensor(kx, device=dev, dtype=torch.float64)
                        for kx in s["support_basis_spec"]["knots_x"]
                    ],
                },
                "support_a": torch.as_tensor(s["support_a"], device=dev, dtype=torch.float64),
                "radial_window": s["radial_window"],
            }

        if fam == "bond_atom":
            pos_A = torch.tensor([c["pos_A"] for c in cands], dtype=torch.long, device=dev)
            pos_B = torch.tensor([c["pos_B"] for c in cands], dtype=torch.long, device=dev)
            pos_C = torch.tensor([c["pos_C"] for c in cands], dtype=torch.long, device=dev)
            symm = torch.tensor([c["symmetric_bond"] for c in cands], dtype=torch.bool, device=dev)

            k_max = max([c["k_C"] for c in cands] + [1])
            nb_mat = torch.zeros((O, k_max), dtype=torch.long, device=dev)
            mask_nb = torch.zeros((O, k_max), dtype=torch.bool, device=dev)
            k_C = torch.tensor([c["k_C"] for c in cands], dtype=torch.float64, device=dev)

            for i, c in enumerate(cands):
                for j, p in enumerate(c["neighbor_pos"]):
                    nb_mat[i, j] = p
                    mask_nb[i, j] = True

            r_wins = torch.tensor(
                [
                    DEFAULT_RADIAL_WINDOWS["bond_atom_H"]
                    if c["B_is_H"]
                    else DEFAULT_RADIAL_WINDOWS["bond_atom_heavy"]
                    for c in cands
                ],
                dtype=torch.float64,
                device=dev,
            )

            fam_runtimes[fam] = {
                "pos_A": pos_A,
                "pos_B": pos_B,
                "pos_C": pos_C,
                "nb_mat": nb_mat,
                "mask_nb": mask_nb,
                "k_C": k_C,
                "symmetric": symm,
                "radial_windows": r_wins,
                "surfaces": surfaces_tensor,
                "surface_keys": [str(c["z_o"][0]) for c in cands],
            }

        elif fam == "bond_ring":
            pos_A = torch.tensor([c["pos_A"] for c in cands], dtype=torch.long, device=dev)
            pos_B = torch.tensor([c["pos_B"] for c in cands], dtype=torch.long, device=dev)
            sz_t = torch.tensor([c["ring_size"] for c in cands], dtype=torch.long, device=dev)
            symm = torch.tensor([c["symmetric_bond"] for c in cands], dtype=torch.bool, device=dev)

            ring_mat = torch.zeros((O, 6), dtype=torch.long, device=dev)
            for i, c in enumerate(cands):
                for j, p in enumerate(c["ring_pos"]):
                    ring_mat[i, j] = p

            r_wins = torch.tensor(
                [DEFAULT_RADIAL_WINDOWS["bond_ring"] for _ in range(O)],
                dtype=torch.float64,
                device=dev,
            )

            fam_runtimes[fam] = {
                "pos_A": pos_A,
                "pos_B": pos_B,
                "ring_mat": ring_mat,
                "ring_size": sz_t,
                "symmetric": symm,
                "radial_windows": r_wins,
                "surfaces": surfaces_tensor,
                "surface_keys": [str(c["z_o"][0]) for c in cands],
            }

        elif fam == "ring_ring":
            sz_R = torch.tensor([c["ring_size_R"] for c in cands], dtype=torch.long, device=dev)
            sz_S = torch.tensor([c["ring_size_S"] for c in cands], dtype=torch.long, device=dev)

            ring_mat_R = torch.zeros((O, 6), dtype=torch.long, device=dev)
            ring_mat_S = torch.zeros((O, 6), dtype=torch.long, device=dev)
            for i, c in enumerate(cands):
                for j, p in enumerate(c["ring_pos_R"]):
                    ring_mat_R[i, j] = p
                for j, p in enumerate(c["ring_pos_S"]):
                    ring_mat_S[i, j] = p

            r_wins = torch.tensor(
                [DEFAULT_RADIAL_WINDOWS["ring_ring"] for _ in range(O)],
                dtype=torch.float64,
                device=dev,
            )

            fam_runtimes[fam] = {
                "ring_mat_R": ring_mat_R,
                "ring_size_R": sz_R,
                "ring_mat_S": ring_mat_S,
                "ring_size_S": sz_S,
                "radial_windows": r_wins,
                "surfaces": surfaces_tensor,
                "surface_keys": [str(c["z_o"][0]) for c in cands],
            }

    return RuntimeNCI(
        mol_id=opps.mol_id,
        device=dev,
        is_empty=(len(fam_runtimes) == 0),
        family_runtimes=fam_runtimes,
        gamma_shared=calibrated_artifact.gamma_shared,
        family_scales=calibrated_artifact.family_scales,
    )


REQUIRED_SURFACE_KEYS = ("coef_r", "basis_specs", "support_basis_spec", "support_a")


def _validate_surface_schema(s: Any, key: Any) -> Dict[str, Any]:
    """Validates surface dictionary schema, failing loudly on schema errors."""
    if not isinstance(s, dict):
        raise ValueError(
            f"Invalid surface for key {key!r}: expected dict, got {type(s).__name__}"
        )
    missing = [k for k in REQUIRED_SURFACE_KEYS if k not in s]
    if missing:
        raise KeyError(
            f"Invalid surface schema for key {key!r}: missing required keys: {missing}"
        )
    return s


def _safe_get_surface(surfaces: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    """
    Safely retrieves the surface for a context key.

    Implements Jacobik Roadmap v2 §3.9 / Item C1:
    - Returns EXACT ZERO (None) for an unknown root context.
    - Pools to a compatible ancestor for a known root context.
    - Fails loudly (TypeError, ValueError, KeyError) on invalid surface dicts or schema errors.
    - Never falls back to an arbitrary first surface.
    """
    import ast

    if not isinstance(surfaces, dict):
        raise TypeError(
            f"Invalid surfaces container: expected dict, got {type(surfaces).__name__}"
        )
    if not surfaces:
        return None

    # 1. Exact string / object match
    if key in surfaces:
        return _validate_surface_schema(surfaces[key], key)

    # 2. Try parsing key from string into tuple
    k_obj = None
    if isinstance(key, str) and (key.startswith("(") or key.startswith("[")):
        try:
            k_obj = ast.literal_eval(key)
        except Exception:
            k_obj = None
    elif isinstance(key, (tuple, list)):
        k_obj = tuple(key)

    if k_obj is not None:
        if k_obj in surfaces:
            return _validate_surface_schema(surfaces[k_obj], key)
        str_k = str(k_obj)
        if str_k in surfaces:
            return _validate_surface_schema(surfaces[str_k], key)

    # 3. Compatible ancestor pooling
    root_candidate = None
    if isinstance(k_obj, tuple):
        if len(k_obj) == 3 and isinstance(k_obj[0], tuple):
            # z_o = (root_key, level2_key, level3_key)
            root_candidate = k_obj[0]
        elif len(k_obj) in (6, 9):
            root_candidate = k_obj

    if root_candidate is not None:
        # Check direct root_candidate match
        if root_candidate in surfaces:
            return _validate_surface_schema(surfaces[root_candidate], key)
        str_rc = str(root_candidate)
        if str_rc in surfaces:
            return _validate_surface_schema(surfaces[str_rc], key)

        if len(root_candidate) == 9:
            (fam, adm, ep_elem, ep_q, bo_c, ring_c, m_roles, is_intra, has_c) = root_candidate
            candidate_ancestors = [
                # Relax has_c_neighbors
                (fam, adm, ep_elem, ep_q, bo_c, ring_c, m_roles, is_intra),
                # Relax is_intra
                (fam, adm, ep_elem, ep_q, bo_c, ring_c, m_roles),
                # Relax bo_c & ring_c
                (fam, adm, ep_elem, ep_q, m_roles),
                # Relax ep_elem (preserve charge class and metal role)
                (fam, adm, ep_q, m_roles),
                # Preserve metal role
                (fam, adm, m_roles),
                # General class ancestor
                (fam, adm),
            ]
            for anc in candidate_ancestors:
                if anc in surfaces:
                    return _validate_surface_schema(surfaces[anc], key)
                if str(anc) in surfaces:
                    return _validate_surface_schema(surfaces[str(anc)], key)

            # Check if any surface in surfaces shares the same (family, admission_class, metal_binding_roles)
            for s_k, s_v in surfaces.items():
                try:
                    s_parsed = ast.literal_eval(s_k) if isinstance(s_k, str) and s_k.startswith("(") else s_k
                except Exception:
                    s_parsed = s_k
                if isinstance(s_parsed, tuple) and len(s_parsed) >= 2:
                    if s_parsed[0] == fam and s_parsed[1] == adm:
                        if len(s_parsed) >= 7 and s_parsed[6] == m_roles:
                            return _validate_surface_schema(s_v, key)
                        elif len(s_parsed) == 2 or (len(s_parsed) == 3 and s_parsed[2] == m_roles):
                            return _validate_surface_schema(s_v, key)

    # 4. Unknown root context: return None (exact zero reward)
    return None


def evaluate_raw_nci(X: torch.Tensor, runtime: RuntimeNCI) -> torch.Tensor:
    """Evaluates total raw NCI reward per pose A_theta(X) of shape (B,) in raw nats."""
    if runtime.is_empty:
        if X.ndim == 2:
            return (0.0 * X).sum(dim=(-1, -2)).unsqueeze(0)
        return (0.0 * X).sum(dim=(-1, -2))

    if X.ndim == 2:
        X = X.unsqueeze(0)
    B, N, _ = X.shape
    device = X.device
    dtype = X.dtype

    total_reward = torch.zeros(B, device=device, dtype=dtype)

    for fam, rt in runtime.family_runtimes.items():
        scale = runtime.family_scales.get(fam, runtime.gamma_shared)

        if fam == "bond_atom":
            pos_A, pos_B, pos_C = rt["pos_A"], rt["pos_B"], rt["pos_C"]
            symm = rt["symmetric"]
            k_C = rt["k_C"]
            nb_mat = rt["nb_mat"]
            mask_nb = rt["mask_nb"]

            def eval_ba(p_a, p_b, p_c):
                P_A = X[:, p_a, :]
                P_B = X[:, p_b, :]
                P_C = X[:, p_c, :]
                v_CB = P_C - P_B
                v_BA = P_B - P_A
                r = reg_norm(v_CB, DELTA_R)
                u_CB = reg_unit(v_CB, DELTA_R)
                u_BA = reg_unit(v_BA, DELTA_R)
                x_1 = (u_BA * u_CB).sum(dim=-1, keepdim=True)

                P_D = X[:, nb_mat, :]
                P_C_exp = P_C.unsqueeze(2)
                P_B_exp = P_B.unsqueeze(2)
                v_DC = P_D - P_C_exp
                v_BC = P_B_exp - P_C_exp
                u_DC = reg_unit(v_DC, DELTA_R)
                u_BC = reg_unit(v_BC, DELTA_R)
                c_D = (u_DC * u_BC).sum(dim=-1)
                c_D_masked = torch.where(mask_nb.unsqueeze(0), c_D, torch.zeros_like(c_D))

                k_clamped = torch.clamp(k_C.unsqueeze(0), min=1.0)
                mu_1 = c_D_masked.sum(dim=-1) / k_clamped
                mu_2 = (c_D_masked**2).sum(dim=-1) / k_clamped
                has_c = (k_C > 0).unsqueeze(0)
                mu_1 = torch.where(has_c, mu_1, torch.zeros_like(mu_1)).unsqueeze(-1)
                mu_2 = torch.where(has_c, mu_2, torch.zeros_like(mu_2)).unsqueeze(-1)
                return torch.cat([r, x_1, mu_1, mu_2], dim=-1)

            feat = eval_ba(pos_A, pos_B, pos_C)  # (B, O, 4)
            O = feat.shape[1]
            r_wins = rt["radial_windows"]

            # Surface evaluation
            u_occ = torch.zeros((B, O), device=device, dtype=dtype)
            for i in range(O):
                s_key = rt["surface_keys"][i]
                s = _safe_get_surface(rt.get("surfaces", {}), s_key)
                if s is None:
                    continue

                rw = r_wins[i]
                r_val = feat[:, i, 0]
                T_in, T_out = compute_tapers(r_val, rw[0].item(), rw[1].item(), rw[2].item(), rw[3].item())

                f_i = feat[:, i : i + 1, :].reshape(B, 4)
                m_sup = evaluate_joint_support(f_i, s["support_basis_spec"], s["support_a"])
                F_val = evaluate_latent_surface(f_i, s["basis_specs"], s["coef_r"], s["coef_rx"])
                h_val = h_epsilon(F_val, EPS_POS)
                u_occ[:, i] = T_in * T_out * m_sup * h_val

            # Symmetric bond averaging
            if torch.any(symm):
                feat_rev = eval_ba(pos_B, pos_A, pos_C)
                u_occ_rev = torch.zeros((B, O), device=device, dtype=dtype)
                for i in range(O):
                    if not symm[i]:
                        continue
                    s_key = rt["surface_keys"][i]
                    s = _safe_get_surface(rt.get("surfaces", {}), s_key)
                    if s is None:
                        continue
                    rw = r_wins[i]
                    r_val = feat_rev[:, i, 0]
                    T_in, T_out = compute_tapers(r_val, rw[0].item(), rw[1].item(), rw[2].item(), rw[3].item())

                    f_i = feat_rev[:, i : i + 1, :].reshape(B, 4)
                    m_sup = evaluate_joint_support(f_i, s["support_basis_spec"], s["support_a"])
                    F_val = evaluate_latent_surface(f_i, s["basis_specs"], s["coef_r"], s["coef_rx"])
                    h_val = h_epsilon(F_val, EPS_POS)
                    u_occ_rev[:, i] = T_in * T_out * m_sup * h_val

                u_occ = torch.where(symm.unsqueeze(0), 0.5 * (u_occ + u_occ_rev), u_occ)

            total_reward = total_reward + scale * u_occ.sum(dim=-1)

        elif fam == "bond_ring":
            pos_A, pos_B = rt["pos_A"], rt["pos_B"]
            ring_mat, sz_t = rt["ring_mat"], rt["ring_size"]
            symm = rt["symmetric"]

            def eval_br(p_a, p_b):
                P_A = X[:, p_a, :]
                P_B = X[:, p_b, :]
                P_ring = X[:, ring_mat, :]
                c_R, n_R = compute_ring_frame(P_ring, sz_t, DELTA_N)

                v_r = c_R - P_B
                v_BA = P_B - P_A
                v_Bc = P_B - c_R
                r = reg_norm(v_r, DELTA_R)
                u_r = reg_unit(v_r, DELTA_R)
                u_BA = reg_unit(v_BA, DELTA_R)
                u_Bc = reg_unit(v_Bc, DELTA_R)

                x_1 = (u_BA * u_r).sum(dim=-1, keepdim=True)
                x_2 = ((n_R * u_Bc).sum(dim=-1, keepdim=True)) ** 2
                x_3 = ((n_R * u_BA).sum(dim=-1, keepdim=True)) ** 2
                return torch.cat([r, x_1, x_2, x_3], dim=-1)

            feat = eval_br(pos_A, pos_B)
            O = feat.shape[1]
            r_wins = rt["radial_windows"]

            u_occ = torch.zeros((B, O), device=device, dtype=dtype)
            for i in range(O):
                s_key = rt["surface_keys"][i]
                s = _safe_get_surface(rt.get("surfaces", {}), s_key)
                if s is None:
                    continue
                rw = r_wins[i]
                r_val = feat[:, i, 0]
                T_in, T_out = compute_tapers(r_val, rw[0].item(), rw[1].item(), rw[2].item(), rw[3].item())

                f_i = feat[:, i : i + 1, :].reshape(B, 4)
                m_sup = evaluate_joint_support(f_i, s["support_basis_spec"], s["support_a"])
                F_val = evaluate_latent_surface(f_i, s["basis_specs"], s["coef_r"], s["coef_rx"])
                h_val = h_epsilon(F_val, EPS_POS)
                u_occ[:, i] = T_in * T_out * m_sup * h_val

            if torch.any(symm):
                feat_rev = eval_br(pos_B, pos_A)
                u_occ_rev = torch.zeros((B, O), device=device, dtype=dtype)
                for i in range(O):
                    if not symm[i]:
                        continue
                    s_key = rt["surface_keys"][i]
                    s = _safe_get_surface(rt.get("surfaces", {}), s_key)
                    if s is None:
                        continue
                    rw = r_wins[i]
                    r_val = feat_rev[:, i, 0]
                    T_in, T_out = compute_tapers(r_val, rw[0].item(), rw[1].item(), rw[2].item(), rw[3].item())

                    f_i = feat_rev[:, i : i + 1, :].reshape(B, 4)
                    m_sup = evaluate_joint_support(f_i, s["support_basis_spec"], s["support_a"])
                    F_val = evaluate_latent_surface(f_i, s["basis_specs"], s["coef_r"], s["coef_rx"])
                    h_val = h_epsilon(F_val, EPS_POS)
                    u_occ_rev[:, i] = T_in * T_out * m_sup * h_val

                u_occ = torch.where(symm.unsqueeze(0), 0.5 * (u_occ + u_occ_rev), u_occ)

            total_reward = total_reward + scale * u_occ.sum(dim=-1)

        elif fam == "ring_ring":
            ring_mat_R, sz_R = rt["ring_mat_R"], rt["ring_size_R"]
            ring_mat_S, sz_S = rt["ring_mat_S"], rt["ring_size_S"]

            P_ring_R = X[:, ring_mat_R, :]
            P_ring_S = X[:, ring_mat_S, :]
            c_R, n_R = compute_ring_frame(P_ring_R, sz_R, DELTA_N)
            c_S, n_S = compute_ring_frame(P_ring_S, sz_S, DELTA_N)

            v_SR = c_S - c_R
            r = reg_norm(v_SR, DELTA_R)
            d = reg_unit(v_SR, DELTA_R)

            x_1 = ((n_R * n_S).sum(dim=-1, keepdim=True)) ** 2
            dot_R = (n_R * d).sum(dim=-1, keepdim=True)
            dot_S = (n_S * d).sum(dim=-1, keepdim=True)
            x_2 = (dot_R**2) + (dot_S**2)
            x_3 = (dot_R**2) * (dot_S**2)
            feat = torch.cat([r, x_1, x_2, x_3], dim=-1)

            O = feat.shape[1]
            r_wins = rt["radial_windows"]

            u_occ = torch.zeros((B, O), device=device, dtype=dtype)
            for i in range(O):
                s_key = rt["surface_keys"][i]
                s = _safe_get_surface(rt.get("surfaces", {}), s_key)
                if s is None:
                    continue
                rw = r_wins[i]
                r_val = feat[:, i, 0]
                T_in, T_out = compute_tapers(r_val, rw[0].item(), rw[1].item(), rw[2].item(), rw[3].item())

                f_i = feat[:, i : i + 1, :].reshape(B, 4)
                m_sup = evaluate_joint_support(f_i, s["support_basis_spec"], s["support_a"])
                F_val = evaluate_latent_surface(f_i, s["basis_specs"], s["coef_r"], s["coef_rx"])
                h_val = h_epsilon(F_val, EPS_POS)
                u_occ[:, i] = T_in * T_out * m_sup * h_val

            total_reward = total_reward + scale * u_occ.sum(dim=-1)

    return total_reward
