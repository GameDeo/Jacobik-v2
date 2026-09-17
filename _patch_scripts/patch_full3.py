#!/usr/bin/env python3
import sys

PATH = "src/jacobik/ik/ik_solver.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

steps_done = []


def do_replace(label, old, new, count=1):
    global src
    n = src.count(old)
    if n == 0:
        print(f"FATAL: pattern for step [{label}] not found", file=sys.stderr)
        sys.exit(1)
    if n != count:
        print(f"FATAL: pattern for step [{label}] found {n} times, expected {count}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, count)
    steps_done.append(label)


# ---------------------------------------------------------------------------
# Step 5: solve() signature
# ---------------------------------------------------------------------------
OLD_SIG = '''    def solve(
        self,
        theta_inits: list,  # [Tensor(B, 6+Ji), ...]
        X_ref: torch.Tensor,  # (N, 3)
        donor_targets: list,  # [Tensor(B, n_d_i, 3), ...]
        donor_pos_idxs: list,  # [LongTensor(n_d_i), ...]
        debug: bool = False,
    ):'''
NEW_SIG = '''    def solve(
        self,
        theta_inits: list,  # [Tensor(B, 6+Ji), ...]
        X_ref: torch.Tensor,  # (N, 3)
        donor_targets: list,  # [Tensor(B, n_d_i, 3), ...]
        donor_pos_idxs: list,  # [LongTensor(n_d_i), ...]
        debug: bool = False,
        ideal_directions: Optional[torch.Tensor] = None,  # (n_donors_total, 3)
        symmetric_groups: Optional[List[List[int]]] = None,
    ):'''
do_replace("5-solve-signature", OLD_SIG, NEW_SIG)

# ---------------------------------------------------------------------------
# Step 6: solve() docstring param additions
# ---------------------------------------------------------------------------
OLD_SDOC = '''        donor_targets  : list of (B, n_d_i, 3) target tensors, one per ligand.
        donor_pos_idxs : list of (n_d_i,) LongTensors of pos-indices into X_ref.
        debug          : print per-step diagnostics.

        Returns
        -------'''
NEW_SDOC = '''        donor_targets  : list of (B, n_d_i, 3) target tensors, one per ligand.
                         Still used to seed kabsch_init() before solve() is
                         called; inside solve() itself it now only feeds the
                         OLD anchor fallback path (ideal_directions=None).
        donor_pos_idxs : list of (n_d_i,) LongTensors of pos-indices into X_ref.
        debug          : print per-step diagnostics.
        ideal_directions : (n_donors_total, 3) fixed unit vectors, one per
                         donor in ge.donor_positions()'s global donor order
                         (= ideal[assignment[di]] from engine.py's Hungarian
                         assignment). When given, activates the 2.3.0
                         Augmented-Lagrangian shape constraint in place of
                         the old fixed-target anchor (see class docstring).
        symmetric_groups : list of donor-index groups (each a list of
                         indices into the same global donor order as
                         ideal_directions) that are topologically equivalent
                         (RDKit canonical-rank match, engine.py) and should
                         be pulled toward a common M-D distance. Optional;
                         only meaningful alongside ideal_directions, though
                         harmless without it.

        Returns
        -------'''
do_replace("6-solve-docstring", OLD_SDOC, NEW_SDOC)

# ---------------------------------------------------------------------------
# Step 7: history lists + mu_shape init, right before the T-step loop
# ---------------------------------------------------------------------------
OLD_HIST = '''        history_residuals = []
        history_scores = []
        history_anchors = []
        history_grad_norms = []

        if debug:'''
NEW_HIST = '''        history_residuals = []
        history_scores = []
        history_anchors = []
        history_grad_norms = []
        history_mu_shape = []
        history_sym_penalty = []

        # Augmented-Lagrangian dual variable for the shape constraint (only
        # meaningful when ideal_directions is not None -- see 2.3.0
        # changelog above). Initialised at lambda_init for continuity with
        # the old anchor's initial strength; unlike the old fixed lambda(t)
        # schedule it then self-adjusts via dual ascent every step instead
        # of decaying on a fixed curve.
        mu_shape = torch.full((B,), self.lambda_init, device=device)
        ideal_directions_dev = (
            ideal_directions.to(device) if ideal_directions is not None else None
        )

        if debug:'''
do_replace("7-history-init", OLD_HIST, NEW_HIST)

# ---------------------------------------------------------------------------
# Step 8: the anchor-loss computation block -> shape/ALM + symmetry branch
# ---------------------------------------------------------------------------
OLD_ANCHOR = '''            # --- Anchor loss with LSE soft-max over per-donor errors ---
            # Focuses gradient on the worst-placed donor (lse_temp=2.0,
            # same temperature as Q6 LSE in scorer.py — one device, one constant).
            anchor_loss = X_ref.new_zeros(B)
            all_donor_pos_grad = self.ge.donor_positions(X_current)  # (B, n_donors, 3)
            all_donor_pos_det = all_donor_pos_grad.detach()

            for li, (didx, dt) in enumerate(zip(donor_pos_idxs, donor_targets)):
                donor_pos_li = all_donor_pos_grad[:, didx, :]  # (B, n_d_i, 3)
                diff_li = donor_pos_li - dt  # (B, n_d_i, 3)
                sq_li = (diff_li**2).sum(-1)  # (B, n_d_i)

                if self.lse_temp > 0.0 and sq_li.shape[1] > 1:
                    tau = self.lse_temp
                    lse_li = torch.logsumexp(tau * sq_li, dim=-1) / tau  # (B,)
                    anchor_loss = anchor_loss + lse_li / sq_li.shape[1]
                else:
                    anchor_loss = anchor_loss + sq_li.mean(-1)

            anchor_loss = anchor_loss / L'''
NEW_ANCHOR = '''            # --- Shape / anchor term (2.3.0) ---
            # Two mechanisms, selected once per call by whether the caller
            # supplied ideal_directions (see class docstring / changelog):
            #  * NEW (ideal_directions given): Augmented-Lagrangian shape
            #    constraint via a per-step-recomputed optimal rotation
            #    (Procrustes/Kabsch, rotation-only) between the fixed ideal
            #    polyhedron directions and the current donor unit vectors,
            #    plus a small symmetry-tying restraint on topologically
            #    equivalent donors (symmetric_groups).
            #  * OLD (ideal_directions is None, backward-compat fallback):
            #    fixed absolute-Cartesian-target anchor with the original
            #    lambda(t) decay schedule, unchanged from pre-2.3.0.
            all_donor_pos_grad = self.ge.donor_positions(X_current)  # (B, n_donors, 3)
            all_donor_pos_det = all_donor_pos_grad.detach()
            sym_penalty = X_ref.new_zeros(B)

            if ideal_directions_dev is not None:
                donor_unit_vecs = F.normalize(all_donor_pos_grad, dim=-1)  # (B, n_donors, 3)

                # Optimal rotation recomputed every step from the CURRENT
                # donor arrangement, under no_grad and detached: by the
                # envelope theorem d(loss)/dR = 0 at R's own optimum, so
                # detaching it costs nothing to first order while avoiding
                # SVD-backward instability on near-degenerate inputs.
                with torch.no_grad():
                    R_opt = _optimal_rotation_nocentroid(
                        ideal_directions_dev, donor_unit_vecs.detach()
                    )  # (B, 3, 3): R_opt @ ideal_i ~= donor_unit_i

                aligned_ideal = torch.einsum(
                    "bij,nj->bni", R_opt, ideal_directions_dev
                )  # (B, n_donors, 3)
                per_donor_sq = ((aligned_ideal - donor_unit_vecs) ** 2).sum(-1)  # (B, n_donors)

                n_sh = per_donor_sq.shape[-1]
                if self.lse_temp > 0.0 and n_sh > 1:
                    tau = self.lse_temp
                    shape_dev = torch.logsumexp(tau * per_donor_sq, dim=-1) / tau  # (B,)
                else:
                    shape_dev = per_donor_sq.mean(dim=-1)

                violation = (shape_dev - self.shape_tol).clamp(min=0.0)  # (B,)
                constraint_loss = (
                    mu_shape.detach() * violation + 0.5 * self.shape_rho * violation**2
                )
                anchor_loss = shape_dev  # trace/back-compat alias (see solve_trace)

                if symmetric_groups:
                    all_donor_norms = all_donor_pos_grad.norm(dim=-1)  # (B, n_donors)
                    n_groups = 0
                    for group in symmetric_groups:
                        if len(group) < 2:
                            continue
                        idx_t = torch.as_tensor(group, dtype=torch.long, device=device)
                        d_g = all_donor_norms[:, idx_t]  # (B, len(group))
                        dev_g = d_g - d_g.mean(dim=-1, keepdim=True)
                        sym_penalty = sym_penalty + (dev_g**2).mean(dim=-1)
                        n_groups += 1
                    if n_groups > 0:
                        sym_penalty = sym_penalty / n_groups

                constraint_loss = constraint_loss + self.sym_weight * sym_penalty
            else:
                # --- OLD anchor: fixed absolute-Cartesian target, LSE soft-max ---
                anchor_loss = X_ref.new_zeros(B)
                for li, (didx, dt) in enumerate(zip(donor_pos_idxs, donor_targets)):
                    donor_pos_li = all_donor_pos_grad[:, didx, :]  # (B, n_d_i, 3)
                    diff_li = donor_pos_li - dt  # (B, n_d_i, 3)
                    sq_li = (diff_li**2).sum(-1)  # (B, n_d_i)

                    if self.lse_temp > 0.0 and sq_li.shape[1] > 1:
                        tau = self.lse_temp
                        lse_li = torch.logsumexp(tau * sq_li, dim=-1) / tau  # (B,)
                        anchor_loss = anchor_loss + lse_li / sq_li.shape[1]
                    else:
                        anchor_loss = anchor_loss + sq_li.mean(-1)

                anchor_loss = anchor_loss / L
                # lambda_eff (score-driven decay schedule) is computed above,
                # before the forward pass -- used unchanged on this path only.
                constraint_loss = lambda_eff * anchor_loss'''
do_replace("8-anchor-block", OLD_ANCHOR, NEW_ANCHOR)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
