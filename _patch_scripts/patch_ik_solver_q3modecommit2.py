#!/usr/bin/env python3
"""
ik_solver.py 2.5.0 part 2: solve()-body changes for mode-committed Q3 init.
(patch_ik_solver_q3modecommit.py adds the changelog entry; run that first.)
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
# M1: solve() signature -- add q3_mode_commit
# ---------------------------------------------------------------------------
do_replace(
    "M1-solve-signature",
    "        ideal_directions: Optional[torch.Tensor] = None,  # (n_donors_total, 3)\n"
    "        symmetric_groups: Optional[List[List[int]]] = None,\n"
    "    ):",
    "        ideal_directions: Optional[torch.Tensor] = None,  # (n_donors_total, 3)\n"
    "        symmetric_groups: Optional[List[List[int]]] = None,\n"
    "        q3_mode_commit: bool = True,\n"
    "    ):",
)

# ---------------------------------------------------------------------------
# M2: solve() docstring -- document q3_mode_commit
# ---------------------------------------------------------------------------
do_replace(
    "M2-solve-docstring",
    "        symmetric_groups : list of donor-index groups (each a list of\n"
    "                         indices into the same global donor order as\n"
    "                         ideal_directions) that are topologically equivalent\n"
    "                         (RDKit canonical-rank match, engine.py) and should\n"
    "                         be pulled toward a common M-D distance. Optional;\n"
    "                         only meaningful alongside ideal_directions, though\n"
    "                         harmless without it.\n",
    "        symmetric_groups : list of donor-index groups (each a list of\n"
    "                         indices into the same global donor order as\n"
    "                         ideal_directions) that are topologically equivalent\n"
    "                         (RDKit canonical-rank match, engine.py) and should\n"
    "                         be pulled toward a common M-D distance. Optional;\n"
    "                         only meaningful alongside ideal_directions, though\n"
    "                         harmless without it.\n"
    "        q3_mode_commit  : bool, default True (2.5.0). When True and a\n"
    "                         symmetric group has >=2 Q3-eligible members, commit\n"
    "                         that group to a single GMM component (chosen from\n"
    "                         the initial pose) for the whole solve, instead of\n"
    "                         letting each member independently draw from the\n"
    "                         full (possibly multi-modal) Q3 mixture. See class\n"
    "                         2.5.0 changelog. False restores pre-2.5.0 behaviour\n"
    "                         byte-identically -- useful for A/B comparison.\n",
)

# ---------------------------------------------------------------------------
# M3: after the existing q3_site_to_idx precompute, add the initial-pose
# forward pass and the mode-commit computation itself.
# ---------------------------------------------------------------------------
OLD_BLOCK = """        q3_site_to_idx: Dict[int, int] = {}
        if symmetric_groups:
            _q3_running = 0
            for _s_i, _elig in enumerate(self.ge._q3_eligible):
                if _elig:
                    q3_site_to_idx[_s_i] = _q3_running
                    _q3_running += 1

        if debug:"""
NEW_BLOCK = """        q3_site_to_idx: Dict[int, int] = {}
        if symmetric_groups:
            _q3_running = 0
            for _s_i, _elig in enumerate(self.ge._q3_eligible):
                if _elig:
                    q3_site_to_idx[_s_i] = _q3_running
                    _q3_running += 1

        # Mode-committed Q3 initialization (2.5.0) -- see class changelog.
        # q3_mode_override: (B, n_q3) int64, -1 = no override (default,
        # scorer.py 2.8.0 no-ops on this). Computed ONCE here from the
        # initial pose (theta_inits, before any gradient steps) and held
        # fixed for the whole trajectory -- deliberately not recomputed
        # every step, since the point is a single up-front commitment per
        # trial, not a moving target.
        q3_mode_override: Optional[torch.Tensor] = None
        if q3_mode_commit and symmetric_groups and q3_site_to_idx:
            q3_means_static = getattr(self.scorer, "_q3_means", None)
            q3_eff_vars_static = getattr(self.scorer, "_q3_eff_vars", None)
            q3_log_w_static = getattr(self.scorer, "_q3_log_w", None)
            if (
                q3_means_static is not None
                and q3_means_static.shape[0] > 0
                and q3_eff_vars_static is not None
                and q3_log_w_static is not None
            ):
                with torch.no_grad():
                    X_init = X_ref.unsqueeze(0).expand(B, -1, -1).clone()
                    for _fk_li, _theta_li in zip(self.fks, thetas):
                        X_init = _fk_li.apply(_theta_li, X_init)
                    q3_cos_init = torch.cos(self.ge.q3_approach_angles(X_init))  # (B, n_q3)
                    n_q3_total = q3_means_static.shape[0]
                    override = torch.full(
                        (B, n_q3_total), -1, dtype=torch.long, device=device
                    )
                    any_group_qualifies = False
                    for group in symmetric_groups:
                        q3_idxs = [
                            q3_site_to_idx[g] for g in group if g in q3_site_to_idx
                        ]
                        if len(q3_idxs) < 2:
                            continue
                        any_group_qualifies = True
                        idx_t = torch.as_tensor(q3_idxs, dtype=torch.long, device=device)
                        obs_g = q3_cos_init[:, idx_t]  # (B, G)
                        m_g = q3_means_static[idx_t]  # (G, K)
                        v_g = q3_eff_vars_static[idx_t]  # (G, K)
                        lw_g = q3_log_w_static[idx_t]  # (G, K)
                        # Un-normalised log-density is enough here (only the
                        # argmax over k matters, and the -0.5*log(2*pi) term
                        # is a k-independent constant that drops out).
                        diff_g = obs_g.unsqueeze(-1) - m_g.unsqueeze(0)  # (B, G, K)
                        log_comp_g = (
                            -0.5 * torch.log(v_g).unsqueeze(0)
                            - 0.5 * diff_g**2 / v_g.unsqueeze(0)
                        )
                        log_p_g = log_comp_g + lw_g.unsqueeze(0)  # (B, G, K)
                        group_log_p_per_k = log_p_g.sum(dim=1)  # (B, K) -- total if ALL members forced to k
                        best_k = group_log_p_per_k.argmax(dim=-1)  # (B,)
                        override[:, idx_t] = best_k.unsqueeze(-1).expand(-1, len(q3_idxs))
                    if any_group_qualifies:
                        q3_mode_override = override
                    if debug and any_group_qualifies:
                        print(
                            f"\\n[DEBUG]   Q3 mode-commit (2.5.0): committed "
                            f"{int((override[0] >= 0).sum().item())} of {n_q3_total} "
                            f"Q3 terms for trial 0 -- modes {override[0][override[0] >= 0].tolist()}"
                        )

        if debug:"""
do_replace("M3-mode-commit-precompute", OLD_BLOCK, NEW_BLOCK)

# ---------------------------------------------------------------------------
# M4: thread q3_mode_override into the in-loop scorer.score() call
# ---------------------------------------------------------------------------
do_replace(
    "M4-loop-score-call",
    "            # --- Unified score: one call, one number, honest objective ---\n"
    "            stat_score, _ = self.scorer.score(X_current)",
    "            # --- Unified score: one call, one number, honest objective ---\n"
    "            stat_score, _ = self.scorer.score(X_current, q3_mode_override=q3_mode_override)",
)

# ---------------------------------------------------------------------------
# M5: thread q3_mode_override into the final scorer.score() call
# ---------------------------------------------------------------------------
do_replace(
    "M5-final-score-call",
    "            final_score, _ = self.scorer.score(X_final)  # (B,)",
    "            final_score, _ = self.scorer.score(X_final, q3_mode_override=q3_mode_override)  # (B,)",
)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
