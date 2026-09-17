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
# Step 9: combined_loss uses constraint_loss instead of lambda_eff * anchor_loss
# ---------------------------------------------------------------------------
do_replace(
    "9-combined-loss",
    '            combined_loss = (-stat_score + lambda_eff * anchor_loss).sum()  # scalar\n',
    '            combined_loss = (-stat_score + constraint_loss).sum()  # scalar\n',
)

# ---------------------------------------------------------------------------
# Step 10: r (logging residual) uses constraint_loss too
# ---------------------------------------------------------------------------
do_replace(
    "10-r-logging",
    '            r = -stat_score + lambda_eff * anchor_loss  # (B,) for logging\n',
    '            r = -stat_score + constraint_loss  # (B,) for logging\n',
)

# ---------------------------------------------------------------------------
# Step 11: trace bookkeeping -- add mu_shape / sym_penalty history + dual update
# ---------------------------------------------------------------------------
OLD_TRACE = '''                history_residuals.append(r.detach())
                history_scores.append(stat_score.detach())
                history_anchors.append(anchor_loss.detach())
                history_grad_norms.append(grad_norm_b)

            if t in _debug_steps and debug:'''
NEW_TRACE = '''                history_residuals.append(r.detach())
                history_scores.append(stat_score.detach())
                history_anchors.append(anchor_loss.detach())
                history_grad_norms.append(grad_norm_b)
                history_mu_shape.append(mu_shape.detach().clone())
                history_sym_penalty.append(sym_penalty.detach())

                # Dual ascent: mu_shape self-adjusts to the violation just
                # observed this step. No-op on the old-anchor fallback path
                # (violation is only defined when ideal_directions is given).
                if ideal_directions_dev is not None:
                    mu_shape = (mu_shape + self.shape_rho * violation.detach()).clamp(min=0.0)

            if t in _debug_steps and debug:'''
do_replace("11-trace-and-dual-update", OLD_TRACE, NEW_TRACE)

# ---------------------------------------------------------------------------
# Step 12: solve_trace dict -- add mu_shape / sym_penalty keys
# ---------------------------------------------------------------------------
OLD_STRACE = '''            solve_trace = {
                "residual": torch.stack(history_residuals, dim=0).cpu().numpy(),  # (T, B)
                "score": torch.stack(history_scores, dim=0).cpu().numpy(),  # (T, B)
                "anchor_loss": torch.stack(history_anchors, dim=0).cpu().numpy(),  # (T, B)
                "grad_norm": torch.stack(history_grad_norms, dim=0).cpu().numpy(),  # (T, B)
            }'''
NEW_STRACE = '''            solve_trace = {
                "residual": torch.stack(history_residuals, dim=0).cpu().numpy(),  # (T, B)
                "score": torch.stack(history_scores, dim=0).cpu().numpy(),  # (T, B)
                # anchor_loss: OLD path -> fixed-target squared error (as before);
                # NEW path (ideal_directions given) -> shape_dev, the raw
                # (pre-multiplier) shape-deviation diagnostic. See 2.3.0 changelog.
                "anchor_loss": torch.stack(history_anchors, dim=0).cpu().numpy(),  # (T, B)
                "grad_norm": torch.stack(history_grad_norms, dim=0).cpu().numpy(),  # (T, B)
                "mu_shape": torch.stack(history_mu_shape, dim=0).cpu().numpy(),  # (T, B)
                "sym_penalty": torch.stack(history_sym_penalty, dim=0).cpu().numpy(),  # (T, B)
            }'''
do_replace("12-solve-trace-dict", OLD_STRACE, NEW_STRACE)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
