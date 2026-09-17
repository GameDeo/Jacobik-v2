#!/usr/bin/env python3
"""
Patch src/jacobik/ik/scorer.py: Q2/Q3 move from VMM to GMM-on-cos(angle),
matching the stats/fitter.py refit (Q2/Q3 are bounded [0,180] deg via
arccos, never wrap to 360, so they are not circular quantities -- VMM
assumed periodicity that doesn't exist. See fitter.py's fit_gmm_cosangle()
docstring for the full rationale).

Version bump: 2.6.0 -> 2.7.0.
"""
import sys

PATH = "src/jacobik/ik/scorer.py"

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
# S0: version-history docstring
# ---------------------------------------------------------------------------
OLD_HIST_TAIL = '''         due to gradient concentrating entirely on whichever term is
         currently worst each step. Reverted to plain mean; see the
         KNOWN ISSUE note on the _vmm/_gmm helpers in _score_components.
"""'''
NEW_HIST_TAIL = '''         due to gradient concentrating entirely on whichever term is
         currently worst each step. Reverted to plain mean; see the
         KNOWN ISSUE note on the _vmm/_gmm helpers in _score_components.
2.7.0 : Q2 (D-M-D angle) and Q3 (donor approach angle) move from VMM to
         GMM-on-cos(angle). Both are computed via arccos(), bounded
         [0,180] deg, and never wrap to 360 -- they are not circular
         quantities, so fitting/scoring them with a von Mises mixture
         assumed a periodicity that doesn't exist. Concretely: a VMM
         component whose mean sits near a real physical boundary (e.g.
         CO's M-C-O approach angle piling up near 180 deg) is forced to
         "spend" spread on the non-existent far side of that boundary,
         systematically understating true peakedness there. GMM-on-
         cos(angle) respects the bound (cos maps [0,180] deg monotonically
         onto [-1,1]) and lets a component sit sharply peaked right at a
         boundary. See stats/fitter.py's fit_gmm_cosangle() docstring for
         the full rationale (that is where the actual model-family switch
         happens; this version's changes are the score-time counterpart).
         _q2_kappas/_q3_kappas/_q2_log_norm/_q3_log_norm are replaced by
         _q2_vars/_q3_vars (raw) and _q2_eff_vars/_q3_eff_vars (floored,
         precomputed) -- same naming convention as Q1/Q4. Q2/Q3 means are
         now dimensionless cos-space values in [-1,1], not radians -- the
         degree->radian conversion at construction time is gone for both.
         New _GMM_VAR_MIN entries for Q2/Q3 (cos-space units). The local
         _gmm() helper in _score_components gains the same aggregation/
         power_p option _vmm() already had, so q2_aggregation="power_mean"
         keeps working unchanged. q2_score() (post-IK ranking) rewritten
         to the same GMM-cos form. Backward compatible for every other
         quantity: Q1/Q4/Q6 untouched; Q5 (torsion dihedral) stays VMM,
         since it is genuinely circular (0 deg == 360 deg).
"""'''
do_replace("S0-version-history", OLD_HIST_TAIL, NEW_HIST_TAIL)

# ---------------------------------------------------------------------------
# S1: _GMM_VAR_MIN -- add Q2/Q3 entries (cos-space units)
# ---------------------------------------------------------------------------
OLD_VARMIN = '''_GMM_VAR_MIN: Dict[str, float] = {
    "Q1": 0.04,  # Å²   -> sigma floor ~0.2 Å
    "Q4": 4.00,  # deg² -> sigma floor ~2.0 deg
}'''
NEW_VARMIN = '''_GMM_VAR_MIN: Dict[str, float] = {
    "Q1": 0.04,  # Å²   -> sigma floor ~0.2 Å
    "Q4": 4.00,  # deg² -> sigma floor ~2.0 deg
    # Q2/Q3 (2.7.0): cos-space, dimensionless. sigma_cos=0.05 -> var=0.0025.
    # Same guard as Q1/Q4: without a floor, a near-delta-function component
    # fit from duplicate/near-duplicate tmQM structures could crush any
    # donor a fraction of a degree off its exact angle. 0.05 in cos-space
    # is roughly a 3 deg sigma near 90 deg and tighter still near a 0/180
    # deg boundary (where d(cos)/d(angle) is largest), which is generous
    # enough not to fight legitimately sharp boundary-peaked components
    # (e.g. CO's near-180 deg approach) while still bounding worst case.
    "Q2": 0.0025,
    "Q3": 0.0025,
}'''
do_replace("S1-gmm-var-min", OLD_VARMIN, NEW_VARMIN)

# ---------------------------------------------------------------------------
# S2: _vmm_fields -- "Q3/Q5" comment is stale now that Q3 no longer routes
# through this function (only Q5 does).
# ---------------------------------------------------------------------------
OLD_COMMENT = '''            # type='gmm' misfiled for a VMM slot.
            # The stats pipeline stores Q3/Q5 GMM vars in degrees² (matching
            # the degree-valued means), but vmm_log_prob works in radians.'''
NEW_COMMENT = '''            # type='gmm' misfiled for a VMM slot.
            # (2.7.0: only Q5 routes through _vmm_fields now -- Q2/Q3 moved
            # to _gmm_fields / GMM-on-cos(angle), see fitter.py's
            # fit_gmm_cosangle() docstring.) The stats pipeline stores Q5
            # GMM vars in degrees² (matching the degree-valued means), but
            # vmm_log_prob works in radians.'''
do_replace("S2-vmm-fields-comment", OLD_COMMENT, NEW_COMMENT)

# ---------------------------------------------------------------------------
# S3: Q3 construction block -- _vmm_fields -> _gmm_fields, drop deg->rad
# conversion, rename kappas->vars, new fallback.
# ---------------------------------------------------------------------------
OLD_Q3_BLOCK = '''        q3_means_list: List[List[float]] = []
        q3_kappas_list: List[List[float]] = []
        q3_weights_list: List[List[float]] = []

        for s_i, site in enumerate(mol.coord_sites):
            if not ge._q3_eligible[s_i]:
                continue

            d_idx = site.donor_indices[0]
            label = mol.donor_labels.get(d_idx)
            subclass = label.subclass if label is not None else "Unknown_donor"
            key = (subclass, metal_sym, coord_geom)
            try:
                dist = stats.dist("Q3", key)
                fields = _vmm_fields(dist, key, "Q3")
            except Exception as exc:
                log.warning("Q3 stats.dist failed for key %s: %s", key, exc)
                fields = None
            if fields is not None:
                q3_means_list.append([m * _DEG2RAD for m in fields[0]])
                q3_kappas_list.append(fields[1])
                q3_weights_list.append(fields[2])
            else:
                q3_means_list.append([math.pi])  # 180° — linear default
                q3_kappas_list.append([1.0])
                q3_weights_list.append([1.0])

        self._q3_means, _ = _pad_and_stack(q3_means_list, device)
        self._q3_kappas, _ = _pad_and_stack(q3_kappas_list, device)
        self._q3_weights, _ = _pad_and_stack(q3_weights_list, device)
        self._n_q3 = len(q3_means_list)'''
NEW_Q3_BLOCK = '''        # Q3 (2.7.0): GMM-on-cos(angle), not VMM -- Q3 is bounded [0,180] deg
        # via arccos and never wraps to 360, see fitter.py's
        # fit_gmm_cosangle() docstring. Means/vars are dimensionless
        # cos-space values; no degree->radian conversion needed here.
        q3_means_list: List[List[float]] = []
        q3_vars_list: List[List[float]] = []
        q3_weights_list: List[List[float]] = []

        for s_i, site in enumerate(mol.coord_sites):
            if not ge._q3_eligible[s_i]:
                continue

            d_idx = site.donor_indices[0]
            label = mol.donor_labels.get(d_idx)
            subclass = label.subclass if label is not None else "Unknown_donor"
            key = (subclass, metal_sym, coord_geom)
            try:
                dist = stats.dist("Q3", key)
                fields = _gmm_fields(dist, key, "Q3")
            except Exception as exc:
                log.warning("Q3 stats.dist failed for key %s: %s", key, exc)
                fields = None
            if fields is not None:
                q3_means_list.append(list(fields[0]))
                q3_vars_list.append(list(fields[1]))
                q3_weights_list.append(list(fields[2]))
            else:
                # Fallback: cos(180°) = -1, linear default (mirrors the old
                # VMM fallback of mean=pi/kappa=1.0 -- moderately tight).
                q3_means_list.append([-1.0])
                q3_vars_list.append([0.1])
                q3_weights_list.append([1.0])

        self._q3_means, _ = _pad_and_stack(q3_means_list, device)
        self._q3_vars, _ = _pad_and_stack(q3_vars_list, device)
        self._q3_weights, _ = _pad_and_stack(q3_weights_list, device)
        self._n_q3 = len(q3_means_list)'''
do_replace("S3-q3-construction", OLD_Q3_BLOCK, NEW_Q3_BLOCK)

# ---------------------------------------------------------------------------
# S4: Q2 construction block -- same treatment.
# ---------------------------------------------------------------------------
OLD_Q2_HEADER = '''        # ==================================================================
        # Q2 — D–M–D angle VMM per donor pair (for joint scoring only)
        # ge.q2_dmd_angles(X) → (B, n_pairs) in RADIANS (pairs in i<j order)
        # ==================================================================

        q2_means_list: List[List[float]] = []
        q2_kappas_list: List[List[float]] = []
        q2_weights_list: List[List[float]] = []'''
NEW_Q2_HEADER = '''        # ==================================================================
        # Q2 — D–M–D angle GMM-on-cos(angle) per donor pair (2.7.0; was VMM
        # -- Q2 is bounded [0,180] deg via arccos and never wraps to 360,
        # see fitter.py's fit_gmm_cosangle() docstring) (for joint scoring
        # only)
        # ge.q2_dmd_angles(X) → (B, n_pairs) in RADIANS (pairs in i<j order)
        # ==================================================================

        q2_means_list: List[List[float]] = []
        q2_vars_list: List[List[float]] = []
        q2_weights_list: List[List[float]] = []'''
do_replace("S4a-q2-header", OLD_Q2_HEADER, NEW_Q2_HEADER)

OLD_Q2_FIELDS = '''                q2_key = (coord_geom, dist_class, metal_sym, sub_A, sub_B)
                try:
                    dist_obj = stats.dist("Q2", q2_key)
                    fields = _vmm_fields(dist_obj, q2_key, "Q2")
                except Exception as exc:
                    log.warning("Q2 stats.dist failed for key %s: %s", q2_key, exc)
                    fields = None
                if fields is not None:
                    q2_means_list.append([m * _DEG2RAD for m in fields[0]])
                    q2_kappas_list.append(fields[1])
                    q2_weights_list.append(fields[2])
                else:
                    # Broad VMM centred at 90° (κ=0.5 ≈ weakly concentrated)
                    q2_means_list.append([math.pi / 2])
                    q2_kappas_list.append([0.5])
                    q2_weights_list.append([1.0])

                self._q2_pair_meta.append((s_i, s_j, dist_class))

        self._q2_means, _ = _pad_and_stack(q2_means_list, device)
        self._q2_kappas, _ = _pad_and_stack(q2_kappas_list, device)
        self._q2_weights, _ = _pad_and_stack(q2_weights_list, device)'''
NEW_Q2_FIELDS = '''                q2_key = (coord_geom, dist_class, metal_sym, sub_A, sub_B)
                try:
                    dist_obj = stats.dist("Q2", q2_key)
                    fields = _gmm_fields(dist_obj, q2_key, "Q2")
                except Exception as exc:
                    log.warning("Q2 stats.dist failed for key %s: %s", q2_key, exc)
                    fields = None
                if fields is not None:
                    q2_means_list.append(list(fields[0]))
                    q2_vars_list.append(list(fields[1]))
                    q2_weights_list.append(list(fields[2]))
                else:
                    # Broad GMM-cos centred at cos(90°)=0 (weakly concentrated)
                    q2_means_list.append([0.0])
                    q2_vars_list.append([0.5])
                    q2_weights_list.append([1.0])

                self._q2_pair_meta.append((s_i, s_j, dist_class))

        self._q2_means, _ = _pad_and_stack(q2_means_list, device)
        self._q2_vars, _ = _pad_and_stack(q2_vars_list, device)
        self._q2_weights, _ = _pad_and_stack(q2_weights_list, device)'''
do_replace("S4b-q2-fields", OLD_Q2_FIELDS, NEW_Q2_FIELDS)

# ---------------------------------------------------------------------------
# S5: precompute section -- Q2/Q3 switch from _vmm_precomp to _gmm_precomp
# ---------------------------------------------------------------------------
OLD_PRECOMP = '''        # Q2 (VMM) — weight already controls whether Q2 contributes at all
        self._q2_log_norm, self._q2_log_w = _vmm_precomp(
            self._q2_kappas, self._q2_weights, qw["Q2"]
        )

        # Q3 (VMM)
        self._q3_log_norm, self._q3_log_w = _vmm_precomp(
            self._q3_kappas, self._q3_weights, qw["Q3"]
        )'''
NEW_PRECOMP = '''        # Q2 (GMM-cos, 2.7.0) — weight already controls whether Q2 contributes
        self._q2_eff_vars, self._q2_log_w = _gmm_precomp(
            self._q2_vars, self._q2_weights, qw["Q2"], _GMM_VAR_MIN["Q2"]
        )

        # Q3 (GMM-cos, 2.7.0)
        self._q3_eff_vars, self._q3_log_w = _gmm_precomp(
            self._q3_vars, self._q3_weights, qw["Q3"], _GMM_VAR_MIN["Q3"]
        )'''
do_replace("S5-precompute", OLD_PRECOMP, NEW_PRECOMP)

# ---------------------------------------------------------------------------
# S6: generalise the local _gmm() helper in _score_components to accept
# aggregation/power_p, mirroring _vmm()'s existing pattern.
# ---------------------------------------------------------------------------
OLD_GMM_HELPER = '''        def _gmm(obs, means, eff_vars, log_w, n_terms):
            diff = obs.unsqueeze(-1) - means.unsqueeze(0)  # (B, N, K)
            log_comp = (
                -0.5 * _LOG_2PI
                - 0.5 * torch.log(eff_vars).unsqueeze(0)
                - 0.5 * diff**2 / eff_vars.unsqueeze(0)
            )  # (B, N, K)
            log_p = log_comp + log_w.unsqueeze(0)  # (B, N, K)
            return torch.logsumexp(log_p, dim=-1).sum(dim=-1) / max(n_terms, 1)'''
NEW_GMM_HELPER = '''        def _gmm(obs, means, eff_vars, log_w, n_terms, aggregation="mean", power_p=3.0):
            diff = obs.unsqueeze(-1) - means.unsqueeze(0)  # (B, N, K)
            log_comp = (
                -0.5 * _LOG_2PI
                - 0.5 * torch.log(eff_vars).unsqueeze(0)
                - 0.5 * diff**2 / eff_vars.unsqueeze(0)
            )  # (B, N, K)
            log_p = log_comp + log_w.unsqueeze(0)  # (B, N, K)
            per_term = torch.logsumexp(log_p, dim=-1)  # (B, N)
            if aggregation == "mean":
                return per_term.sum(dim=-1) / max(n_terms, 1)
            # "power_mean": mirrors _vmm's power-mean aggregation above.
            # (2.7.0: added so Q2's q2_aggregation="power_mean" option keeps
            # working now that Q2 is scored via _gmm instead of _vmm.)
            badness = F.softplus(-per_term)
            pmean_badness = (badness.pow(power_p).mean(dim=-1) + 1e-12).pow(1.0 / power_p)
            return -pmean_badness'''
do_replace("S6-gmm-helper-generalise", OLD_GMM_HELPER, NEW_GMM_HELPER)

# ---------------------------------------------------------------------------
# S7: Q2 evaluation call site -- cos(angle), dispatch through _gmm.
# ---------------------------------------------------------------------------
OLD_Q2_CALL = '''        if self._n_q2 > 0 and self._q2_means.shape[0] > 0 and self._qweights["Q2"] > 0.0:
            q2_angles = self._ge.q2_dmd_angles(X)  # (B, n_pairs)
            comp["Q2"] = _vmm(
                q2_angles,
                self._q2_means,
                self._q2_kappas,
                self._q2_log_norm,
                self._q2_log_w,
                self._n_q2,
                aggregation=self._q2_aggregation,
                power_p=self._q2_power_p,
            )'''
NEW_Q2_CALL = '''        if self._n_q2 > 0 and self._q2_means.shape[0] > 0 and self._qweights["Q2"] > 0.0:
            q2_angles = self._ge.q2_dmd_angles(X)  # (B, n_pairs) radians
            q2_cos = torch.cos(q2_angles)  # GMM-cos, 2.7.0
            comp["Q2"] = _gmm(
                q2_cos,
                self._q2_means,
                self._q2_eff_vars,
                self._q2_log_w,
                self._n_q2,
                aggregation=self._q2_aggregation,
                power_p=self._q2_power_p,
            )'''
do_replace("S7-q2-call", OLD_Q2_CALL, NEW_Q2_CALL)

# ---------------------------------------------------------------------------
# S8: Q3 evaluation call site -- cos(angle), dispatch through _gmm.
# ---------------------------------------------------------------------------
OLD_Q3_CALL = '''        if self._n_q3 > 0 and self._q3_means.shape[0] > 0:
            q3_a = self._ge.q3_approach_angles(X)  # (B, n_q3)
            comp["Q3"] = _vmm(
                q3_a, self._q3_means, self._q3_kappas, self._q3_log_norm, self._q3_log_w, self._n_q3
            )'''
NEW_Q3_CALL = '''        if self._n_q3 > 0 and self._q3_means.shape[0] > 0:
            q3_a = self._ge.q3_approach_angles(X)  # (B, n_q3) radians
            q3_cos = torch.cos(q3_a)  # GMM-cos, 2.7.0
            comp["Q3"] = _gmm(
                q3_cos, self._q3_means, self._q3_eff_vars, self._q3_log_w, self._n_q3
            )'''
do_replace("S8-q3-call", OLD_Q3_CALL, NEW_Q3_CALL)

# ---------------------------------------------------------------------------
# S9: q2_score() (post-IK joint ranking) -- rewrite VMM form to GMM-cos form.
# ---------------------------------------------------------------------------
OLD_Q2SCORE = '''    def q2_score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Q2 (D–M–D angle) log-probability for post-IK joint ranking.
        Uses precomputed _q2_log_norm and _q2_log_w (no i0e/log per call).

        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        q2 : (B,)  — summed Q2 log-prob; higher = more plausible
        """
        B = X.shape[0]
        if self._q2_means.shape[0] == 0:
            return X.new_zeros(B)

        q2_angles = self._ge.q2_dmd_angles(X)  # (B, n_pairs)
        phi_e = q2_angles.unsqueeze(-1)  # (B, n_pairs, 1)
        cos_diff = torch.cos(phi_e - self._q2_means.unsqueeze(0))  # (B, n_pairs, K)
        log_comp = self._q2_kappas.unsqueeze(0) * cos_diff - self._q2_log_norm.unsqueeze(0)
        # Use raw log_w (not quantity-weighted) for post-IK ranking
        log_w = torch.log(self._q2_weights.clamp(min=_EPS))
        log_p = log_comp + log_w.unsqueeze(0)  # (B, n_pairs, K)
        return torch.logsumexp(log_p, dim=-1).sum(dim=-1)  # (B,)'''
NEW_Q2SCORE = '''    def q2_score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Q2 (D–M–D angle) log-probability for post-IK joint ranking.
        Uses precomputed _q2_eff_vars and _q2_log_w (no per-call clamping).

        2.7.0: Q2 moved from VMM to GMM-on-cos(angle) -- see fitter.py's
        fit_gmm_cosangle() docstring. Evaluated here against cos(angle)
        directly; means/vars are already in cos-space (dimensionless).

        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        q2 : (B,)  — summed Q2 log-prob; higher = more plausible
        """
        B = X.shape[0]
        if self._q2_means.shape[0] == 0:
            return X.new_zeros(B)

        q2_angles = self._ge.q2_dmd_angles(X)  # (B, n_pairs) radians
        q2_cos = torch.cos(q2_angles).unsqueeze(-1)  # (B, n_pairs, 1)
        diff = q2_cos - self._q2_means.unsqueeze(0)  # (B, n_pairs, K)
        log_comp = (
            -0.5 * _LOG_2PI
            - 0.5 * torch.log(self._q2_eff_vars).unsqueeze(0)
            - 0.5 * diff**2 / self._q2_eff_vars.unsqueeze(0)
        )  # (B, n_pairs, K)
        # Use raw log_w (not quantity-weighted) for post-IK ranking
        log_w = torch.log(self._q2_weights.clamp(min=_EPS))
        log_p = log_comp + log_w.unsqueeze(0)  # (B, n_pairs, K)
        return torch.logsumexp(log_p, dim=-1).sum(dim=-1)  # (B,)'''
do_replace("S9-q2-score-method", OLD_Q2SCORE, NEW_Q2SCORE)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
