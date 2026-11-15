"""End-to-end C2-continuous NCI surface fitting and evaluation.

Enforces C2 smoothness across:
1. Radial interior knots (quartic-initial + C2 cubic spline with f'(r0)=f''(r0)=0
   for C2 constant plateau below r0 = r_lo + 0.5*dr).
2. Angular domain and folding:
   - 'hb' (0..180 deg): exact DCT-II / Chebyshev basis T_m(cos(theta)) in cos(theta) in [-1, 1],
     eliminating acos() endpoint clamping kinks at 0 and 180 deg.
   - 'pp' and 'chp' (0..90 deg, ring-normal sign invariant): exact DCT-II / Chebyshev basis
     T_m(2*cos^2(theta) - 1) in cos(2*theta) in [-1, 1], eliminating both the abs() fold
     kink at 90 deg and the acos() singularity at 0 deg.
3. Soft saturation: S_CLIP * tanh(f / S_CLIP) instead of hard np.clip.
4. C2 quintic radial taper: 1 - 10*t^3 + 15*t^4 - 6*t^5 on [r_on, r_off], matching
   value, first derivative, and second derivative identically to 0 at r >= r_off.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.fft import dct
from scipy.optimize import minimize

_SRC = str(Path(__file__).resolve().parents[5] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from research.archive.code.interactions import nci


# Median full-complex compliance SDs from n2_thermal_audit_2026-10-18 (research regularizer reference)
THERMAL_COMPLIANCE_SD = {
    "hb": (0.1906741756954919, math.radians(14.566764014704948)),
    "pp": (0.26223202412665503, math.radians(10.205406667282261)),
    "chp": (0.35082490524557286, math.radians(8.94368205953425)),
}


def quintic_taper_np(r: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """C2 quintic taper: 1 for r <= lo, 0 for r >= hi, zero 1st and 2nd derivatives at lo and hi."""
    t = np.clip((r - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - t * t * t * (10.0 - t * (15.0 - 6.0 * t))


def quintic_taper_torch(r, lo: float, hi: float):
    """Torch C2 quintic taper: 1 for r <= lo, 0 for r >= hi."""
    import torch

    t = ((r - lo) / (hi - lo)).clamp(0.0, 1.0)
    return 1.0 - t * t * t * (10.0 - t * (15.0 - 6.0 * t))


def _untaper_empirical_table(t: str, s_table: np.ndarray) -> np.ndarray:
    """Recover untapered, unclipped log-enrichment on bin centers inside the active window,
    or fit directly from raw counts when available."""
    r_lo, dr, nr, _, _, _ = nci._grid(t)
    rc = r_lo + dr * (np.arange(nr) + 0.5)
    win_lo, win_hi = nci.SPEC[t]["win"]
    old_taper = nci._taper(rc, win_lo, win_hi)[:, None]
    out = np.zeros_like(s_table, dtype=np.float64)
    mask = old_taper > 1e-3
    out[mask[:, 0]] = s_table[mask[:, 0]] / old_taper[mask[:, 0]]
    # Invert hard clip gently into pre-tanh space bounded away from infinity
    clipped = np.clip(out / nci.S_CLIP, -0.995, 0.995)
    return nci.S_CLIP * np.arctanh(clipped)


def fit_empirical_surface(
    t: str,
    h: Optional[np.ndarray] = None,
    s_existing: Optional[np.ndarray] = None,
    lam_rough: float = 0.0,
    lam_sharp: float = 0.0,
    gamma_compliance: float = 1.0,
    ess_ratio: float = 1.0,
) -> np.ndarray:
    """Fit pre-taper, pre-tanh log-enrichment grid F (nr, na) on bin centers.

    When lam_rough == 0 and lam_sharp == 0:
      - If h is provided, computes the exact smoothed count/volume ratio log((hs+5)/(E+5))
        before clipping and tapering.
      - If s_existing is provided, recovers the pre-taper values from the existing table.
    When lam_rough > 0 or lam_sharp > 0:
      - Minimizes the explicit empirical fitting objective:
        J[F] = sum w_ij (F_ij - y_ij)^2 + lam_rough * ||D2 F||^2
               + lam_sharp * sum ||(-Hess(F)_ij - gamma * C_reg^-1)_+||^2
        where w_ij = ess_ratio / (1/(hs_ij + 5) + 1/(E_ij + 5)).
    """
    from scipy.signal import convolve2d

    r_lo, dr, nr, a_lo, da, na = nci._grid(t)
    rc = r_lo + dr * (np.arange(nr) + 0.5)
    if h is not None:
        vol = nci.bin_volumes(t)
        k = np.ones((3, 3)) / 9.0
        sm = lambda x: convolve2d(x, k, mode="same", boundary="symm")
        hs, vs = sm(h), sm(vol)
        shell = (rc >= nci.SPEC[t]["shell"][0]) & (rc < nci.SPEC[t]["shell"][1])
        rho = hs[shell].sum() / max(vs[shell].sum(), 1e-12)
        E = rho * vs
        y = np.log((hs + nci.PSEUDO) / (E + nci.PSEUDO))
        w = float(ess_ratio) / (1.0 / (hs + nci.PSEUDO) + 1.0 / (E + nci.PSEUDO))
        w = w / np.mean(w)
    elif s_existing is not None:
        y = _untaper_empirical_table(t, np.asarray(s_existing, dtype=np.float64))
        w = np.ones_like(y)
    else:
        raise ValueError("Provide either raw histogram h or s_existing")

    if lam_rough <= 0.0 and lam_sharp <= 0.0:
        return y

    sd_r, sd_a_rad = THERMAL_COMPLIANCE_SD[t]
    da_rad = math.radians(da)
    # Upper stiffness ceiling in (nat / A^2) and (nat / rad^2) under local harmonic regularizer
    ceil_r = gamma_compliance / (sd_r ** 2)
    ceil_a = gamma_compliance / (sd_a_rad ** 2)

    def objective_and_grad(flat: np.ndarray) -> Tuple[float, np.ndarray]:
        F = flat.reshape(nr, na)
        diff = F - y
        loss = float(np.sum(w * diff * diff))
        grad = 2.0 * w * diff

        # Second finite differences in r (interior) and angle (reflective/Neumann at 0 and na-1)
        d2r = F[2:, :] - 2.0 * F[1:-1, :] + F[:-2, :]
        F_pad = np.pad(F, ((0, 0), (1, 1)), mode="symmetric")
        d2a = F_pad[:, 2:] - 2.0 * F + F_pad[:, :-2]

        if lam_rough > 0.0:
            loss += lam_rough * float(np.sum(d2r * d2r) + np.sum(d2a * d2a))
            gr = 2.0 * lam_rough * d2r
            grad[2:, :] += gr
            grad[1:-1, :] -= 2.0 * gr
            grad[:-2, :] += gr

            ga = 2.0 * lam_rough * d2a
            grad -= 2.0 * ga
            grad[:, 1:] += ga[:, :-1]
            grad[:, 0] += ga[:, 0]  # symmetric left boundary
            grad[:, :-1] += ga[:, 1:]
            grad[:, -1] += ga[:, -1]  # symmetric right boundary

        if lam_sharp > 0.0:
            # Local attractive curvature is -d2F/dr^2 and -d2F/da^2
            curv_r = -d2r / (dr * dr)
            excess_r = np.maximum(0.0, curv_r - ceil_r)
            if np.any(excess_r > 0.0):
                # Normalize by ceil_r^2 so radial and angular penalties are dimensionless
                scale_r = lam_sharp / (ceil_r * ceil_r)
                loss += scale_r * float(np.sum(excess_r * excess_r))
                gr_s = -2.0 * scale_r * excess_r / (dr * dr)
                grad[2:, :] += gr_s
                grad[1:-1, :] -= 2.0 * gr_s
                grad[:-2, :] += gr_s

            curv_a = -d2a / (da_rad * da_rad)
            excess_a = np.maximum(0.0, curv_a - ceil_a)
            if np.any(excess_a > 0.0):
                scale_a = lam_sharp / (ceil_a * ceil_a)
                loss += scale_a * float(np.sum(excess_a * excess_a))
                ga_s = -2.0 * scale_a * excess_a / (da_rad * da_rad)
                grad -= 2.0 * ga_s
                grad[:, 1:] += ga_s[:, :-1]
                grad[:, 0] += ga_s[:, 0]
                grad[:, :-1] += ga_s[:, 1:]
                grad[:, -1] += ga_s[:, -1]

        return loss, grad.ravel()

    res = minimize(
        objective_and_grad,
        y.ravel().copy(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8},
    )
    return res.x.reshape(nr, na)


def _solve_radial_c2_slopes(Y: np.ndarray, dr: float) -> np.ndarray:
    """Solve for radial slopes M (K, nr, na_modes) at radial bin centers r_0..r_{nr-1}
    such that:
      - Interval [r_0, r_1] is a quartic with p(r_0)=Y_0, p'(r_0)=0, p''(r_0)=0,
        p(r_1)=Y_1, p'(r_1)=M_1, matching p''(r_1) with the cubic spline on [r_1, r_{nr-1}].
      - Intervals [r_i, r_{i+1}] for i >= 1 are C2 cubic Hermite segments.
      - Outer knot r_{nr-1} has natural boundary condition p''(r_{nr-1}) = 0.
    """
    *lead, nr, nm = Y.shape
    Y2 = Y.reshape(-1, nr, nm)
    B = Y2.shape[0]
    M = np.zeros_like(Y2, dtype=np.float64)

    # Tridiagonal system on indices 1 .. nr-1 (size N = nr - 1)
    N = nr - 1
    diag = np.full(N, 4.0, dtype=np.float64)
    diag[0] = 5.0
    diag[-1] = 2.0
    upper = np.ones(N - 1, dtype=np.float64)
    lower = np.ones(N - 1, dtype=np.float64)

    rhs = np.zeros((B, N, nm), dtype=np.float64)
    rhs[:, 0, :] = (6.0 * (Y2[:, 1, :] - Y2[:, 0, :]) + 3.0 * (Y2[:, 2, :] - Y2[:, 1, :])) / dr
    rhs[:, 1:-1, :] = 3.0 * (Y2[:, 3:, :] - Y2[:, 1:-2, :]) / dr
    rhs[:, -1, :] = 3.0 * (Y2[:, -1, :] - Y2[:, -2, :]) / dr

    # Thomas algorithm
    c_star = np.zeros(N - 1, dtype=np.float64)
    d_star = np.zeros((B, N, nm), dtype=np.float64)
    c_star[0] = upper[0] / diag[0]
    d_star[:, 0, :] = rhs[:, 0, :] / diag[0]
    for i in range(1, N):
        denom = diag[i] - lower[i - 1] * c_star[i - 1]
        if i < N - 1:
            c_star[i] = upper[i] / denom
        d_star[:, i, :] = (rhs[:, i, :] - lower[i - 1] * d_star[:, i - 1, :]) / denom

    x = np.zeros((B, N, nm), dtype=np.float64)
    x[:, -1, :] = d_star[:, -1, :]
    for i in range(N - 2, -1, -1):
        x[:, i, :] = d_star[:, i, :] - c_star[i] * x[:, i + 1, :]

    M[:, 1:, :] = x
    return M.reshape(*lead, nr, nm)


def build_c2_surface_params(t: str, F_grid: np.ndarray) -> dict:
    """Convert pre-taper bin-center values F_grid (K, nr, na) or (nr, na) into C2 radial-Chebyshev
    spline parameters.

    For angular bin centers theta_j = (j + 0.5) * pi / na (or pi/2 / na for pp/chp),
    the DCT-II along the angular axis expands F(r_i, theta) in the exact cosine series:
      F(r_i, theta) = sum_{m=0}^{na-1} C[i, m] * cos(m * psi)
    where:
      - For 'hb': psi = pi - theta, so cos(psi) = -cos(theta) in [-1, 1]
        (or directly psi = theta -> cos(psi) = +cos(theta), ordering j=0 at theta=0 -> cos(theta)=1).
        Wait: in SPEC, angle j=0 is theta = 2.5 deg (cos(theta) ~ +1) and j=na-1 is theta = 177.5 deg
        (cos(theta) ~ -1). Since DCT-II basis k_m(j) = cos(pi * m * (j + 0.5) / na) has
        psi_j = pi * (j + 0.5) / na = theta_j (in radians) for 'hb' (where a_hi = 180 deg = pi)
        and psi_j = 2 * theta_j for 'pp'/'chp' (where a_hi = 90 deg = pi/2):
          cos(psi) = cos(theta)           for 'hb'
          cos(psi) = cos(2*theta) = 2*cos^2(theta) - 1   for 'pp' and 'chp'!
       Thus T_m(cos(psi)) equals cos(m * psi_j) at every single bin center j!
    """
    single = F_grid.ndim == 2
    F = F_grid[None, :, :] if single else np.asarray(F_grid, dtype=np.float64)
    r_lo, dr, nr, a_lo, da, na = nci._grid(t)
    assert F.shape[1:] == (nr, na), (F.shape, nr, na)

    # DCT-II along axis=-1: inverse is F[i, j] = C[i, 0] + sum_{m=1}^{na-1} C[i, m] * cos(m * psi_j)
    raw_dct = dct(F, type=2, axis=-1, norm=None)
    C = raw_dct / float(na)
    C[:, :, 0] *= 0.5

    # Solve radial C2 slopes M for each Chebyshev mode m
    M = _solve_radial_c2_slopes(C, dr)
    return dict(
        type=t,
        C=C[0] if single else C,
        M=M[0] if single else M,
        grid=(r_lo, dr, nr, a_lo, da, na),
        win=nci.SPEC[t]["win"],
        s_clip=nci.S_CLIP,
    )


def features_cos_torch(rt_type: str, X, idx: dict):
    """Return (r, z) of shape (B, M) where z = cos(psi) in [-1, 1] is C^infty in Cartesian X:
      - For 'hb': z = cos(theta) = (u . v) / (||u|| ||v||)
      - For 'pp': z = 2 * (n_i . n_j)^2 - 1
      - For 'chp': z = 2 * ((v . n) / r)^2 - 1
    No acos() or abs() is used anywhere.
    """
    import torch

    if rt_type == "hb":
        H, D, A = X[:, idx["h"]], X[:, idx["d"]], X[:, idx["a"]]
        u, v = D - H, A - H
        r = v.norm(dim=-1)
        un = u.norm(dim=-1)
        z = (u * v).sum(-1) / (un * r).clamp(min=1e-12)
        return r, z.clamp(-1.0, 1.0)
    c, n = nci._ring_frame(X, idx["rings"])
    if rt_type == "pp":
        ci, cj, ni, nj = c[:, idx["ri"]], c[:, idx["rj"]], n[:, idx["ri"]], n[:, idx["rj"]]
        r = (ci - cj).norm(dim=-1)
        dot = (ni * nj).sum(-1)
        z = 2.0 * (dot * dot) - 1.0
        return r, z.clamp(-1.0, 1.0)
    Hh = X[:, idx["h"]]
    v = Hh - c[:, idx["ring"]]
    r = v.norm(dim=-1)
    dot = (v * n[:, idx["ring"]]).sum(-1) / r.clamp(min=1e-12)
    z = 2.0 * (dot * dot) - 1.0
    return r, z.clamp(-1.0, 1.0)


def angle_deg_to_z_torch(rt_type: str, angle_deg):
    """Convert angle in degrees to z = cos(psi) for direct grid/table testing."""
    import torch

    rad = torch.deg2rad(angle_deg)
    if rt_type == "hb":
        return torch.cos(rad)
    c = torch.cos(rad)
    return 2.0 * (c * c) - 1.0


def evaluate_c2_from_rz_torch(C, M, row, r, z, grid, win, s_clip: float = nci.S_CLIP):
    """Evaluate C2 surface s(r, z) at (r, z) of shape (B, M_cand).
    C, M have shape (K, nr, na); row has shape (M_cand,).
    """
    import torch

    r_lo, dr, nr, _, _, na = grid
    win_lo, win_hi = win

    # Chebyshev polynomial basis T_0(z) .. T_{na-1}(z): shape (B, M_cand, na)
    T_list = [torch.ones_like(z), z]
    for _ in range(2, na):
        T_list.append(2.0 * z * T_list[-1] - T_list[-2])
    Cheb = torch.stack(T_list[:na], dim=-1)  # (B, M_cand, na)

    # Radial coordinate u = (r - r_0) / dr where r_0 = r_lo + 0.5 * dr
    u = (r - r_lo) / dr - 0.5
    u_clamped = u.clamp(0.0, float(nr - 1))
    i0 = torch.floor(u_clamped.detach()).long().clamp(0, nr - 2)
    t = (u_clamped - i0.to(u.dtype)).clamp(0.0, 1.0)  # local [0, 1] on segment [i0, i0+1]

    rw = row[None, :].expand_as(i0)  # (B, M_cand)
    y0 = C[rw, i0]        # (B, M_cand, na)
    y1 = C[rw, i0 + 1]    # (B, M_cand, na)
    m0 = M[rw, i0]        # (B, M_cand, na)
    m1 = M[rw, i0 + 1]    # (B, M_cand, na)

    # Standard cubic Hermite on [i0, i0+1] for i0 >= 1
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    f_cubic = h00[..., None] * y0 + h10[..., None] * (dr * m0) + h01[..., None] * y1 + h11[..., None] * (dr * m1)

    # Quartic on first interval i0 == 0 with p(0)=y0, p'(0)=0, p''(0)=0, p(1)=y1, p'(1)=dr*m1
    t4 = t2 * t2
    f_quartic = y0 + (y1 - y0) * (4.0 * t3 - 3.0 * t4)[..., None] - (dr * m1) * (t3 - t4)[..., None]

    f_modes = torch.where((i0 == 0)[..., None], f_quartic, f_cubic)
    # Constant C2 plateau for r <= r_0 (u <= 0) since f'(r_0) = f''(r_0) = 0
    f_modes = torch.where((u <= 0.0)[..., None], C[rw, 0], f_modes)

    # Contract over Chebyshev modes
    f_val = (f_modes * Cheb).sum(dim=-1)
    # Smooth C^infty saturation instead of hard clip
    f_sat = s_clip * torch.tanh(f_val / s_clip)
    # C2 quintic radial taper (identically 0 with 0 first and second derivatives for r >= win_hi)
    taper = quintic_taper_torch(r, win_lo, win_hi)
    return f_sat * taper


def evaluate_c2_runtime(rt_c2: dict, X):
    """Evaluate total C2 NCI score (B,) in nats for a batch of Cartesian coordinates X (B, N, 3)."""
    tot = X.new_zeros(X.shape[0])
    for t, e in rt_c2.items():
        r, z = features_cos_torch(t, X, e["idx"])
        s = evaluate_c2_from_rz_torch(e["C"], e["M"], e["row"], r, z, e["grid"], e["win"], e["s_clip"])
        tot = tot + s.sum(-1)
    return tot
