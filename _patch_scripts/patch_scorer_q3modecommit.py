#!/usr/bin/env python3
"""
scorer.py 2.8.0: Q3 mode-commit support.

See ik_solver.py 2.5.0 changelog for the full rationale. Summary: now that
Q3 is genuinely live (post 2.7.0 refit + the _is_empty_dist bugfix), some
donor subclasses (e.g. O_carboxylate on Co/octahedral) have a real,
multi-modal fitted GMM (5 well-separated cos-space modes, ~87-130 deg).
Symmetric donor groups (e.g. EDTA's 4 carboxylate O) independently pulled
toward whichever mode each is nearest to, then tied toward their MEAN
angle by sym_angle_weight (2.4.0), can land in the low-density valley
BETWEEN modes -- worse than either mode alone, and prone to oscillation
(never plateauing) rather than convergence. This adds an optional
q3_mode_override: (B, n_q3) int64 tensor, -1 = no override (score against
the full mixture as always), else hard-select exactly one GMM component
for that (trial, term) -- collapses the multi-modal mixture to a single
Gaussian so tied donors have one consistent, well-behaved target instead
of fighting over which mode to occupy. Computed by ik_solver.py once per
solve() call from the initial pose (mode-committed initialization) and
held fixed for the whole trajectory.
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
# T0: version-history docstring
# ---------------------------------------------------------------------------
OLD_TAIL = '''         quantity: Q1/Q4/Q6 untouched; Q5 (torsion dihedral) stays VMM,
         since it is genuinely circular (0 deg == 360 deg).
"""'''
NEW_TAIL = '''         quantity: Q1/Q4/Q6 untouched; Q5 (torsion dihedral) stays VMM,
         since it is genuinely circular (0 deg == 360 deg).
2.8.0 : _score_components/score/score_breakdown gain an optional
         q3_mode_override: (B, n_q3) int64 tensor, -1 = no override
         (default; behaviour unchanged), else a hard-selected GMM
         component index to score that (trial, Q3-term) against
         EXCLUSIVELY, in place of the full mixture. Implemented by
         building a batched (B, N, K) log_w (all components -1e30 except
         the selected one at log-weight 0) and passing it to the local
         _gmm() helper, which now accepts either the usual static (N, K)
         log_w (broadcast across the batch as before) or an already-
         batched (B, N, K) one. Computed and supplied by ik_solver.py
         2.5.0's mode-committed initialization -- see that module's
         changelog for the full rationale and why this is needed (a
         multi-modal Q3 GMM plus sym_angle_weight's mean-tie can land
         symmetric donors in the low-density valley between two modes,
         worse than either mode alone). No effect when the caller passes
         q3_mode_override=None (every existing call site outside
         ik_solver.py's new logic).
"""'''
do_replace("T0-version-history", OLD_TAIL, NEW_TAIL)

# ---------------------------------------------------------------------------
# T1: module constant for the override's masked-out log-weight
# ---------------------------------------------------------------------------
do_replace(
    "T1-neg-constant",
    "_LOG_2PI: float = math.log(2.0 * math.pi)",
    "_LOG_2PI: float = math.log(2.0 * math.pi)\n"
    "#: 2.8.0 -- effectively -inf log-weight for GMM components masked out by\n"
    "#: q3_mode_override.  A large finite value (not -inf) avoids -inf - -inf\n"
    "#: = nan if a masked row's logsumexp ever sees only masked entries.\n"
    "_Q3_OVERRIDE_NEG: float = -1.0e30",
)

# ---------------------------------------------------------------------------
# T2: _gmm() helper -- accept a batched (B, N, K) log_w
# ---------------------------------------------------------------------------
OLD_GMM = '''        def _gmm(obs, means, eff_vars, log_w, n_terms, aggregation="mean", power_p=3.0):
            diff = obs.unsqueeze(-1) - means.unsqueeze(0)  # (B, N, K)
            log_comp = (
                -0.5 * _LOG_2PI
                - 0.5 * torch.log(eff_vars).unsqueeze(0)
                - 0.5 * diff**2 / eff_vars.unsqueeze(0)
            )  # (B, N, K)
            log_p = log_comp + log_w.unsqueeze(0)  # (B, N, K)
            per_term = torch.logsumexp(log_p, dim=-1)  # (B, N)'''
NEW_GMM = '''        def _gmm(obs, means, eff_vars, log_w, n_terms, aggregation="mean", power_p=3.0):
            diff = obs.unsqueeze(-1) - means.unsqueeze(0)  # (B, N, K)
            log_comp = (
                -0.5 * _LOG_2PI
                - 0.5 * torch.log(eff_vars).unsqueeze(0)
                - 0.5 * diff**2 / eff_vars.unsqueeze(0)
            )  # (B, N, K)
            # log_w is normally (N, K) -- static, broadcast across the batch.
            # 2.8.0: Q3's mode-commit override instead passes an already-
            # batched (B, N, K) log_w (see the Q3 call site below), so only
            # unsqueeze when it's still the static 2D form.
            log_w_b = log_w.unsqueeze(0) if log_w.dim() == 2 else log_w  # (B, N, K)
            log_p = log_comp + log_w_b  # (B, N, K)
            per_term = torch.logsumexp(log_p, dim=-1)  # (B, N)'''
do_replace("T2-gmm-helper-batched-log-w", OLD_GMM, NEW_GMM)

# ---------------------------------------------------------------------------
# T3: Q3 call site -- build and apply the override
# ---------------------------------------------------------------------------
OLD_Q3 = '''        if self._n_q3 > 0 and self._q3_means.shape[0] > 0:
            q3_a = self._ge.q3_approach_angles(X)  # (B, n_q3) radians
            q3_cos = torch.cos(q3_a)  # GMM-cos, 2.7.0
            comp["Q3"] = _gmm(
                q3_cos, self._q3_means, self._q3_eff_vars, self._q3_log_w, self._n_q3
            )'''
NEW_Q3 = '''        if self._n_q3 > 0 and self._q3_means.shape[0] > 0:
            q3_a = self._ge.q3_approach_angles(X)  # (B, n_q3) radians
            q3_cos = torch.cos(q3_a)  # GMM-cos, 2.7.0
            q3_log_w_eff = self._q3_log_w
            # 2.8.0: mode-commit override (see ik_solver.py 2.5.0 changelog).
            # q3_mode_override is (B, n_q3) int64, -1 = no override. Where
            # overridden, collapse the mixture to the single named component
            # by setting every other component's log-weight to
            # _Q3_OVERRIDE_NEG (effectively zero mixture mass) and the named
            # one to 0 (log(1)) -- logsumexp then reduces exactly to that
            # component's own log-density, regardless of its original
            # fitted mixture weight (we are committing to it as THE target
            # for this donor/trial, not blending in leftover uncertainty
            # about whether it's the right mode).
            if q3_mode_override is not None:
                Bq = q3_cos.shape[0]
                Nq, Kq = self._q3_log_w.shape
                mask = q3_mode_override >= 0  # (B, N)
                override_log_w = q3_cos.new_full((Bq, Nq, Kq), _Q3_OVERRIDE_NEG)
                sel = q3_mode_override.clamp(min=0).unsqueeze(-1)  # (B, N, 1) -- safe idx everywhere
                override_log_w.scatter_(-1, sel, torch.zeros_like(sel, dtype=override_log_w.dtype))
                q3_log_w_eff = torch.where(
                    mask.unsqueeze(-1),
                    override_log_w,
                    self._q3_log_w.unsqueeze(0).expand(Bq, -1, -1),
                )
            comp["Q3"] = _gmm(
                q3_cos, self._q3_means, self._q3_eff_vars, q3_log_w_eff, self._n_q3
            )'''
do_replace("T3-q3-call-site-override", OLD_Q3, NEW_Q3)

# ---------------------------------------------------------------------------
# T4: _score_components signature
# ---------------------------------------------------------------------------
do_replace(
    "T4-score-components-signature",
    '''    def _score_components(
        self,
        X: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:''',
    '''    def _score_components(
        self,
        X: torch.Tensor,
        q3_mode_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:''',
)

# ---------------------------------------------------------------------------
# T5: score() signature + docstring + call-through
# ---------------------------------------------------------------------------
do_replace(
    "T5-score-signature",
    '''    def score(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:''',
    '''    def score(
        self,
        X: torch.Tensor,
        q3_mode_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:''',
)
do_replace(
    "T5b-score-docstring",
    '''        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        scalar    : (B,)  — sum of per-quantity weighted values; higher = better
        breakdown : dict with keys "Q1".."Q6", each a (B,) contribution
        """
        breakdown = self._score_components(X)''',
    '''        Parameters
        ----------
        X : (B, N, 3)
        q3_mode_override : optional (B, n_q3) int64 tensor -- see 2.8.0
                           changelog. None (default) = unchanged behaviour.

        Returns
        -------
        scalar    : (B,)  — sum of per-quantity weighted values; higher = better
        breakdown : dict with keys "Q1".."Q6", each a (B,) contribution
        """
        breakdown = self._score_components(X, q3_mode_override=q3_mode_override)''',
)

# ---------------------------------------------------------------------------
# T6: score_breakdown() signature + call-through
# ---------------------------------------------------------------------------
do_replace(
    "T6-score-breakdown",
    '''    def score_breakdown(
        self,
        X: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible alias: return per-quantity breakdown only.
        Equivalent to score(X)[1].

        Parameters
        ----------
        X : (B, N, 3)

        Returns
        -------
        dict with keys "Q1".."Q6", each a (B,) tensor.
        """
        _, breakdown = self.score(X)
        return breakdown''',
    '''    def score_breakdown(
        self,
        X: torch.Tensor,
        q3_mode_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible alias: return per-quantity breakdown only.
        Equivalent to score(X)[1].

        Parameters
        ----------
        X : (B, N, 3)
        q3_mode_override : see score(). None (default) = unchanged behaviour.

        Returns
        -------
        dict with keys "Q1".."Q6", each a (B,) tensor.
        """
        _, breakdown = self.score(X, q3_mode_override=q3_mode_override)
        return breakdown''',
)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
