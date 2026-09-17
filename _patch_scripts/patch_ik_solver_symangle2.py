#!/usr/bin/env python3
"""
ik_solver.py 2.4.0 part 2: the solve()-body changes for the Q3
approach-angle symmetry tie (see patch_ik_solver_symangle.py for the full
rationale -- that script must run first, this one touches solve() itself).
"""
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
# J1: history list init -- add history_sym_angle_penalty
# ---------------------------------------------------------------------------
do_replace(
    "J1-history-init",
    "        history_mu_shape = []\n        history_sym_penalty = []",
    "        history_mu_shape = []\n        history_sym_penalty = []\n        history_sym_angle_penalty = []",
)

# ---------------------------------------------------------------------------
# J2: precompute the static site-index -> Q3-array-index map, once per
# solve() call (Q3 eligibility doesn't depend on X). Placed right next to
# the other per-solve-call static setup (ideal_directions_dev).
# ---------------------------------------------------------------------------
OLD_SETUP = """        mu_shape = torch.full((B,), self.lambda_init, device=device)
        ideal_directions_dev = (
            ideal_directions.to(device) if ideal_directions is not None else None
        )"""
NEW_SETUP = """        mu_shape = torch.full((B,), self.lambda_init, device=device)
        ideal_directions_dev = (
            ideal_directions.to(device) if ideal_directions is not None else None
        )

        # Site-index -> Q3-array-index map (2.4.0): ge.q3_approach_angles()
        # only returns Q3-eligible sites' angles, in site-index order (same
        # convention ik/scorer.py's Q3 construction loop relies on), so this
        # recovers which q3-array slot (if any) each symmetric_groups donor
        # index corresponds to. Static (depends only on ge._q3_eligible), so
        # built once here rather than every step of the T-loop below.
        q3_site_to_idx: Dict[int, int] = {}
        if symmetric_groups:
            _q3_running = 0
            for _s_i, _elig in enumerate(self.ge._q3_eligible):
                if _elig:
                    q3_site_to_idx[_s_i] = _q3_running
                    _q3_running += 1"""
do_replace("J2-q3-site-map-precompute", OLD_SETUP, NEW_SETUP)

# ---------------------------------------------------------------------------
# J3: initialise sym_angle_penalty alongside sym_penalty (top-level, used
# in both branches / for trace bookkeeping regardless of branch taken).
# ---------------------------------------------------------------------------
do_replace(
    "J3-sym-angle-penalty-init",
    "            all_donor_pos_det = all_donor_pos_grad.detach()\n            sym_penalty = X_ref.new_zeros(B)",
    "            all_donor_pos_det = all_donor_pos_grad.detach()\n            sym_penalty = X_ref.new_zeros(B)\n            sym_angle_penalty = X_ref.new_zeros(B)",
)

# ---------------------------------------------------------------------------
# J4: the actual angle-tying computation, right after the existing
# distance-tying block, plus adding it into constraint_loss.
# ---------------------------------------------------------------------------
OLD_SYMBLOCK = """                if symmetric_groups:
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

                constraint_loss = constraint_loss + self.sym_weight * sym_penalty"""
NEW_SYMBLOCK = """                if symmetric_groups:
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

                    # --- Q3 approach-angle symmetry tie (2.4.0) ---------------
                    # See 2.4.0 changelog: the distance tie above leaves each
                    # donor's own Q3 approach-angle outcome as an independent
                    # draw, with nothing preventing two topologically-
                    # equivalent donors (e.g. EDTA's two amine N) from landing
                    # near different modes of a (now genuinely differentiated,
                    # post Q2/Q3-refit) Q3 distribution. Mirrors the distance
                    # tie's pattern exactly, just on a different observable.
                    if q3_site_to_idx:
                        q3_angles_now = None  # lazy: skip the extra fwd pass if no group qualifies
                        n_angle_groups = 0
                        for group in symmetric_groups:
                            q3_idxs = [q3_site_to_idx[g] for g in group if g in q3_site_to_idx]
                            if len(q3_idxs) < 2:
                                continue
                            if q3_angles_now is None:
                                q3_angles_now = self.ge.q3_approach_angles(X_current)  # (B, n_q3) rad
                            idx_t2 = torch.as_tensor(q3_idxs, dtype=torch.long, device=device)
                            a_g = q3_angles_now[:, idx_t2]  # (B, len(q3_idxs))
                            dev_a = a_g - a_g.mean(dim=-1, keepdim=True)
                            sym_angle_penalty = sym_angle_penalty + (dev_a**2).mean(dim=-1)
                            n_angle_groups += 1
                        if n_angle_groups > 0:
                            sym_angle_penalty = sym_angle_penalty / n_angle_groups

                constraint_loss = (
                    constraint_loss
                    + self.sym_weight * sym_penalty
                    + self.sym_angle_weight * sym_angle_penalty
                )"""
do_replace("J4-angle-tie-block", OLD_SYMBLOCK, NEW_SYMBLOCK)

# ---------------------------------------------------------------------------
# J5: trace bookkeeping -- append sym_angle_penalty history
# ---------------------------------------------------------------------------
do_replace(
    "J5-trace-append",
    "                history_mu_shape.append(mu_shape.detach().clone())\n                history_sym_penalty.append(sym_penalty.detach())",
    "                history_mu_shape.append(mu_shape.detach().clone())\n                history_sym_penalty.append(sym_penalty.detach())\n                history_sym_angle_penalty.append(sym_angle_penalty.detach())",
)

# ---------------------------------------------------------------------------
# J6: solve_trace dict -- add sym_angle_penalty key
# ---------------------------------------------------------------------------
do_replace(
    "J6-solve-trace-dict",
    '                "sym_penalty": torch.stack(history_sym_penalty, dim=0).cpu().numpy(),  # (T, B)\n            }',
    '                "sym_penalty": torch.stack(history_sym_penalty, dim=0).cpu().numpy(),  # (T, B)\n'
    '                "sym_angle_penalty": torch.stack(history_sym_angle_penalty, dim=0).cpu().numpy(),  # (T, B)\n'
    "            }",
)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
