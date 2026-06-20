"""
torsion_landscape_diag.py
==========================
Confirms (or refutes) the JACOBIK_SESSION_HANDOFF.md section-5 hypothesis:
that the donor-atom anchor over-constrains the inter-ring torsion DOF in a
bidentate chelate, so the anchor's cost minimum and Q5's cost minimum sit at
different torsion angles and the anchor wins because it dominates the
combined residual.
"""

import sys

import numpy as np
import torch

from jacobik.engine import ComplexBuilder, LigandInput
from jacobik.ik.kernels import vmm_log_prob

bipy_smiles = "c1ccnc(c1)-c1ccccn1"
bipy = LigandInput(bipy_smiles, donors=[[3], [11]])

builder = ComplexBuilder(
    metal="Fe",
    geometry="octahedral",
    charge=2,
    spin=1,
    ligands=[bipy, bipy, bipy],
)

# See exp.md for detailed rationale
import jacobik.ik.ik_solver as ik_solver_mod

_captured = {}
_orig_solve = ik_solver_mod.JointIKSolver.solve


def _capturing_solve(self, theta_inits, X_ref, donor_targets, donor_pos_idxs, debug=False):
    thetas_final, converged, final_score = _orig_solve(
        self, theta_inits, X_ref, donor_targets, donor_pos_idxs, debug=debug
    )
    _captured["solver"] = self
    _captured["thetas_final"] = thetas_final
    _captured["converged"] = converged
    _captured["final_score"] = final_score
    _captured["donor_targets"] = donor_targets
    _captured["donor_pos_idxs"] = donor_pos_idxs
    _captured["X_ref"] = X_ref
    return thetas_final, converged, final_score


ik_solver_mod.JointIKSolver.solve = _capturing_solve

result = builder.solve(
    constraints_db="constraints_synthetic.json",
    beam_width=8,
    n_ik_steps=1000,
    n_coord_samples=4,
    device="cpu",
    seed=0,
    debug=False,
)
ik_solver_mod.JointIKSolver.solve = _orig_solve  # restore

print(f"\n{result.summary() if False else ''}")
n_conv = sum(1 for c in result.conformers if c.converged)
print(
    f"Engine result: {len(result.conformers)} conformers, {n_conv} converged, "
    f"best score={result.conformers[0].score:.4f}"
)

solver = _captured["solver"]
thetas_final = _captured["thetas_final"]  # list[L] of (B, 6+Ji)
donor_targets = _captured["donor_targets"]  # list[L] of (B, n_d_i, 3)
donor_pos_idxs = _captured["donor_pos_idxs"]  # list[L] of (n_d_i,)
X_ref = _captured["X_ref"]  # (N, 3)
final_score = _captured["final_score"]  # (B,)
fks = solver.fks
scorer = solver.scorer

print(f"\n[CHECK] scorer._ge.n_q5 = {scorer._ge.n_q5}")
print(f"[CHECK] scorer._q5_means.shape = {scorer._q5_means.shape}")
print(f"[CHECK] scorer._ge._q5_keys = {scorer._ge._q5_keys}")

B = thetas_final[0].shape[0]
L = len(fks)

# ---------------------------------------------------------------------------
# Pick the best-scoring batch element, then find its worst-converged ligand.
# ---------------------------------------------------------------------------
b_best = int(torch.argmax(final_score).item())
print(f"\nUsing batch element b={b_best} (score={final_score[b_best].item():.4f})")

with torch.no_grad():
    X_b = X_ref.unsqueeze(0).clone()  # (1, N, 3)
    for li, fk_li in enumerate(fks):
        theta_li_b = thetas_final[li][b_best : b_best + 1]
        X_b = fk_li.apply(theta_li_b, X_b)

    per_lig_err = []
    for li, (didx, dt) in enumerate(zip(donor_pos_idxs, donor_targets)):
        dp = X_b[0, didx, :]  # (n_d_i, 3)
        tgt = dt[b_best]  # (n_d_i, 3)
        err = (dp - tgt).norm(dim=-1).mean().item()
        per_lig_err.append(err)
    print("Per-ligand mean donor-target error (Å):", [f"{e:.3f}" for e in per_lig_err])

    # Also rank by how far each ligand's Q5 torsion sits from its nearest
    # fitted mode — this is the axis the handoff actually cares about
    # (a ligand can have small anchor error yet a badly twisted ring, or
    # vice versa; these are not the same failure mode).
    q5_modes_deg = (scorer._q5_means[0] * 180.0 / np.pi).tolist()  # assume shared modes
    cur_phi_deg = torch.rad2deg(scorer._ge.q5_torsions(X_b)[0]).tolist()
    torsion_dev = []
    for phi in cur_phi_deg:
        devs = [min(abs(phi - m), 360 - abs(phi - m)) for m in q5_modes_deg]
        torsion_dev.append(min(devs))
    print(f"Q5 torsions (deg): {[f'{p:.1f}' for p in cur_phi_deg]}")
    print(f"Deviation from nearest Q5 mode (deg): {[f'{d:.1f}' for d in torsion_dev]}")

li_worst = int(np.argmax(torsion_dev))
print(
    f"Worst ligand (by Q5 torsion deviation): li={li_worst}  "
    f"dev={torsion_dev[li_worst]:.1f} deg  anchor_err={per_lig_err[li_worst]:.3f} Å"
)

fk_worst = fks[li_worst]
J_worst = fk_worst.n_joints
print(f"Worst ligand has J={J_worst} torsion joint(s)")

if J_worst == 0:
    print(
        "Worst ligand has no rotatable bond — torsion sweep not applicable. "
        "Re-examine which DOF is actually fighting the anchor."
    )
    sys.exit(0)

# See exp.md for detailed rationale
N_SWEEP = 73  # every 5 degrees over [-180, 180]
taus = np.linspace(-np.pi, np.pi, N_SWEEP)

theta_base = thetas_final[li_worst][b_best : b_best + 1].clone()  # (1, 6+J)

didx_worst = donor_pos_idxs[li_worst]
dt_worst = donor_targets[li_worst][b_best : b_best + 1]  # (1, n_d, 3)

anchor_costs = np.zeros(N_SWEEP)
q5_costs = np.zeros(N_SWEEP)
q3_costs = np.zeros(N_SWEEP)  # this ligand's donors' Q3 approach-angle cost

# Map this ligand's donor pos-indices to their Q3 (kappa-site) index, so we
# can isolate just this ligand's approach-angle term from the scorer's full
# per-kappa Q3 tensors (which span ALL ligands' donors).
kappa_pos_list = scorer._ge._kappa_pos_indices  # list[int], pos idx per kappa site
didx_worst_list = didx_worst.tolist()
q3_idx_for_lig = [kappa_pos_list.index(p) for p in didx_worst_list if p in kappa_pos_list]
print(f"Q3 kappa indices for worst ligand's donors: {q3_idx_for_lig}")

# Other ligands' contribution to X is fixed (their thetas don't change),
# so we only need to recompute fk_worst.apply for the swept ligand each step
# and reuse the frozen positions of the rest of the molecule.
with torch.no_grad():
    X_frozen = X_ref.unsqueeze(0).clone()  # (1, N, 3) — start from reference
    for li, fk_li in enumerate(fks):
        if li == li_worst:
            continue
        theta_li_b = thetas_final[li][b_best : b_best + 1]
        X_frozen = fk_li.apply(theta_li_b, X_frozen)

    for si, tau in enumerate(taus):
        theta_sweep = theta_base.clone()
        theta_sweep[:, 6] = float(tau)
        X_swept = fk_worst.apply(theta_sweep, X_frozen.clone())

        # anchor cost for this ligand only
        dp = X_swept[0, didx_worst, :]  # (n_d, 3)
        diff = dp - dt_worst[0]  # (n_d, 3)
        anchor_costs[si] = (diff**2).sum(-1).mean().item()

        # See exp.md for detailed rationale
        if scorer._ge.n_q5 > 0 and scorer._q5_means.shape[0] > 0:
            q5_t = scorer._ge.q5_torsions(X_swept)  # (1, n_q5) radians
            lp_total = 0.0
            for i in range(scorer._ge.n_q5):
                lp = vmm_log_prob(
                    q5_t[:, i],
                    scorer._q5_means[i],
                    scorer._q5_kappas[i],
                    scorer._q5_weights[i],
                )
                lp_total += lp.item()
            q5_costs[si] = -lp_total
        else:
            q5_costs[si] = float("nan")

        # Q3 approach-angle cost for just this ligand's donors
        if q3_idx_for_lig and scorer._q3_means.shape[0] > 0:
            q3_a = scorer._ge.q3_approach_angles(X_swept)  # (1, n_q3_total) radians
            lp_total = 0.0
            for qi in q3_idx_for_lig:
                lp = vmm_log_prob(
                    q3_a[:, qi],
                    scorer._q3_means[qi],
                    scorer._q3_kappas[qi],
                    scorer._q3_weights[qi],
                )
                lp_total += lp.item()
            q3_costs[si] = -lp_total
        else:
            q3_costs[si] = float("nan")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
deg = np.degrees(taus)
i_anchor_min = int(np.argmin(anchor_costs))
i_q5_min = int(np.argmin(q5_costs))
i_q3_min = int(np.argmin(q3_costs)) if not np.isnan(q3_costs).all() else -1

print("\n--- Torsion landscape sweep (worst ligand, best batch element) ---")
print(f"{'tau(deg)':>10} {'anchor_cost':>14} {'q5_cost':>14} {'q3_cost':>14}")
for si in range(0, N_SWEEP, 6):
    print(f"{deg[si]:>10.1f} {anchor_costs[si]:>14.5f} {q5_costs[si]:>14.5f} {q3_costs[si]:>14.5f}")

print(
    f"\nAnchor-cost minimum at tau = {deg[i_anchor_min]:.1f} deg "
    f"(cost={anchor_costs[i_anchor_min]:.5f})"
)
print(f"Q5-cost minimum     at tau = {deg[i_q5_min]:.1f} deg (cost={q5_costs[i_q5_min]:.5f})")
if i_q3_min >= 0:
    print(f"Q3-cost minimum     at tau = {deg[i_q3_min]:.1f} deg (cost={q3_costs[i_q3_min]:.5f})")

offset = abs(deg[i_anchor_min] - deg[i_q5_min])
offset = min(offset, 360 - offset)
print(f"\nMinima offset (anchor vs Q5): {offset:.1f} deg")

# See exp.md for detailed rationale
n_terms = scorer.n_score_terms
lambda_final = solver._lambda(solver.n_steps - 1, solver.n_steps)  # floor value
combined_late = (q5_costs + q3_costs) / n_terms + lambda_final * (anchor_costs / L)
i_combined_min = int(np.argmin(combined_late))
print(
    f"Combined late-stage cost (lambda={lambda_final:.3f}) minimum at "
    f"tau = {deg[i_combined_min]:.1f} deg"
)
print("(Compare to the solver's actual converged torsion for this ligand, printed below.)")

# Sensitivity check: what if the anchor weight were 0 (pure statistics)?
# If the combined minimum shifts toward the Q5/Q3 basin, the anchor (not a
# saddle or bad seed) is mechanically what's pinning the torsion in place.
combined_zero_lambda = (q5_costs + q3_costs) / n_terms
i_zero_min = int(np.argmin(combined_zero_lambda))
print(
    f"Combined cost with lambda=0 (pure statistics, no anchor) minimum at "
    f"tau = {deg[i_zero_min]:.1f} deg  "
    f"(vs {deg[i_combined_min]:.1f} deg with the anchor included)"
)
shift = abs(deg[i_zero_min] - deg[i_combined_min])
shift = min(shift, 360 - shift)
if shift > 20:
    print(
        f"=> Removing the anchor shifts the optimum by {shift:.1f} deg: "
        f"the anchor is mechanically responsible for pinning this torsion "
        f"away from the statistically-preferred basin. HYPOTHESIS CONFIRMED."
    )
else:
    print(
        f"=> Removing the anchor only shifts the optimum by {shift:.1f} deg: "
        f"the anchor is not the dominant cause here."
    )

with torch.no_grad():
    cur_phi = scorer._ge.q5_torsions(X_b)[0]  # reuse X_b from b_best assembly
    # find which of the n_q5 entries belongs to li_worst by re-deriving from
    # the FK joint atom indices would require more plumbing; report all three
    # ligands' torsions since n_q5==3==L here for single-joint ligands.
    print(
        f"Actual solved Q5 torsions (all ligands, deg): "
        f"{[f'{float(torch.rad2deg(t)):.1f}' for t in cur_phi]}"
    )

np.save("sweep_taus_deg.npy", deg)
np.save("sweep_anchor_costs.npy", anchor_costs)
np.save("sweep_q5_costs.npy", q5_costs)
print("\nSaved sweep_taus_deg.npy, sweep_anchor_costs.npy, sweep_q5_costs.npy")
