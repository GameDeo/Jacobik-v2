"""Capture and reproduce the exact final score from a production IK solve.

Captures are intentionally process-local: they retain the live FK/scorer objects
and are not a serialization format.  A capture is made for each call to
``ComplexBuilder._run_joint_ik`` (including E/Z variants and assignments).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _cast_container_tensors(value, dtype, device):
    """Cast tensor leaves in scorer-owned containers without walking external object graphs."""
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
    if isinstance(value, dict):
        return {k: _cast_container_tensors(v, dtype, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_cast_container_tensors(v, dtype, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_cast_container_tensors(v, dtype, device) for v in value)
    return value


def _cast_scorer_tensors(scorer, dtype, device):
    for name, value in vars(scorer).items():
        if torch.is_tensor(value) or isinstance(value, (dict, list, tuple)):
            setattr(scorer, name, _cast_container_tensors(value, dtype, device))


@dataclass
class SolveCapture:
    fks: list
    scorer: Any
    ge: Any
    X_full: torch.Tensor
    theta_inits: list
    row_groups: torch.Tensor | None
    bend_prior: Any
    score_kwargs: dict
    scores: torch.Tensor
    thetas_final: list
    candidate_idx_per_b: np.ndarray
    variant_index: int
    scorer_cast_dtype: Any = None

    def _rows(self, theta_rows):
        dtype = torch.as_tensor(theta_rows[0]).dtype if len(theta_rows) else self.X_full.dtype
        return [torch.as_tensor(t, dtype=dtype, device=self.X_full.device).reshape(1, -1)
                for t in theta_rows]

    def score(self, theta_rows, trial_row: int) -> torch.Tensor:
        """Evaluate one row using the captured scorer, bend prior and seam penalty."""
        from jacobik.ik import angle_bends as AB
        from jacobik.ik.ik_solver import _BEND_PER_Z, _CLOSURE_W

        theta = self._rows(theta_rows)
        Xref = self.X_full.to(dtype=theta[0].dtype)
        if Xref.ndim == 3:
            Xref = Xref[trial_row:trial_row + 1]
        X = Xref
        for fk, th in zip(self.fks, theta):
            X = fk.apply(th, X)
        # The production score includes the seam residual returned by the fixed
        # closure projection. Recompute the same residual smoothly from this
        # candidate geometry so the local minimiser has a useful gradient.
        seam = X.new_zeros((1,))
        for li, fk in enumerate(self.fks):
            if getattr(fk, "_cl", None) is not None:
                d0 = fk.seam_d0(Xref)
                residual, _ = fk.closure_residual_jacobian(X, d0)
                seam = seam + (residual * residual).sum(dim=-1)
        kwargs = dict(self.score_kwargs)
        for key, value in tuple(kwargs.items()):
            if torch.is_tensor(value) and value.ndim and value.shape[0] == len(self.scores):
                kwargs[key] = value[trial_row:trial_row + 1]
            if torch.is_tensor(value) and value.is_floating_point():
                kwargs[key] = kwargs[key].to(dtype=theta[0].dtype)
        rg = self.row_groups
        if rg is not None:
            kwargs["row_groups"] = rg[trial_row:trial_row + 1]
        # Captured q3 override is preserved; row_groups is restricted to this row.
        if self.scorer_cast_dtype != theta[0].dtype:
            _cast_scorer_tensors(self.scorer, theta[0].dtype, X.device)
            self.scorer_cast_dtype = theta[0].dtype
        stat, _ = self.scorer.score(X, **kwargs)
        total = stat
        cfg = self.bend_prior
        if cfg:
            bend = X.new_zeros((1,))
            for fk, th in zip(self.fks, theta):
                nb = getattr(fk, "n_angle_bends", 0)
                if nb:
                    start = 6 + fk.n_joints
                    free = getattr(fk, "_bend_free", None)
                    free = torch.clamp(free, min=float(cfg["free_deg"])) if free is not None else cfg["free_deg"]
                    bend = bend + AB.energy_prior(th[:, start:start + nb], fk._bend_k, free, cfg["scale"])
            if _BEND_PER_Z and hasattr(self.scorer, "_agg_norm"):
                bend = bend / float(self.scorer._agg_norm())
            total = total + bend
        return (total - _CLOSURE_W * seam).reshape(())

    def verify_conformer(self, conf, *, atol: float = 1e-4) -> dict:
        if conf.solver_state is None or "thetas" not in conf.solver_state:
            raise ValueError("conformer has no captured solver_state/thetas")
        row = int(conf.trial_idx)
        calc = float(self.score(conf.solver_state["thetas"], row).detach().cpu())
        stored = float(conf.score)
        return {"key": conformer_key(conf), "score_calc": calc, "score_stored": stored,
                "abs_diff": abs(calc - stored), "pass": abs(calc - stored) < atol}

def conformer_key(conf) -> tuple[int | None, int, int]:
    """Stable row identity across embed variants, donor assignments and IK trials."""
    if conf.trial_idx is None or conf.assignment_idx is None:
        raise ValueError("conformer is missing trial_idx or assignment_idx")
    return (conf.embed_idx, int(conf.assignment_idx), int(conf.trial_idx))


class CaptureSession:
    """Patch one builder so every production joint IK solve retains its inputs."""
    def __init__(self, builder):
        self.builder = builder
        self.captures: list[SolveCapture] = []
        self._original = None
        self._had_instance_method = False
        self._original_instance_method = None

    def __enter__(self):
        self._original = self.builder._run_joint_ik
        self._had_instance_method = "_run_joint_ik" in vars(self.builder)
        self._original_instance_method = vars(self.builder).get("_run_joint_ik")

        def wrapped(ctx, fks, scorer_joint, ge_joint, X_full, theta_inits,
                    donor_pos_idxs, ideal_directions_full_t, symmetric_groups,
                    q3_symmetric_groups, beam_width, n_ik_steps, candidate_idx_per_b):
            # Observe actual score call arguments while the solver executes.
            calls = []
            original_score = scorer_joint.score
            had_score_attribute = "score" in vars(scorer_joint)
            original_score_attribute = vars(scorer_joint).get("score")
            def observed_score(*args, **kwargs):
                calls.append({k: (v.detach().clone() if torch.is_tensor(v) else v)
                              for k, v in kwargs.items()})
                return original_score(*args, **kwargs)
            scorer_joint.score = observed_score
            try:
                result = self._original(ctx, fks, scorer_joint, ge_joint, X_full,
                                        theta_inits, donor_pos_idxs, ideal_directions_full_t,
                                        symmetric_groups, q3_symmetric_groups, beam_width,
                                        n_ik_steps, candidate_idx_per_b)
            finally:
                if had_score_attribute:
                    scorer_joint.score = original_score_attribute
                else:
                    delattr(scorer_joint, "score")
            thetas, _, scores, _ = result
            final_kwargs = calls[-1] if calls else {}
            final_kwargs.pop("row_groups", None)  # stored explicitly below
            self.captures.append(SolveCapture(
                fks=list(fks), scorer=scorer_joint, ge=ge_joint,
                X_full=X_full.detach().clone(), theta_inits=[t.detach().clone() for t in theta_inits],
                row_groups=torch.as_tensor(np.asarray(candidate_idx_per_b), dtype=torch.long,
                                           device=scores.device),
                bend_prior=getattr(ctx, "bend_prior", None), score_kwargs=final_kwargs,
                scores=scores.detach().clone(), thetas_final=[t.detach().clone() for t in thetas],
                candidate_idx_per_b=np.asarray(candidate_idx_per_b).copy(),
                variant_index=len(self.captures)))
            return result

        self.builder._run_joint_ik = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._had_instance_method:
            self.builder._run_joint_ik = self._original_instance_method
        else:
            delattr(self.builder, "_run_joint_ik")
        return False

    def for_conformer(self, conf) -> SolveCapture:
        """Resolve a conformer against all captured solves; ambiguity is an error."""
        key = conformer_key(conf)
        matches = [c for c in self.captures
                   if 0 <= key[2] < len(c.scores)
                   and int(c.candidate_idx_per_b[key[2]]) == key[1]]
        variant = 0 if conf.embed_idx is None else int(conf.embed_idx)
        matches = [c for c in matches if c.variant_index == variant]
        if len(matches) != 1:
            raise LookupError(f"capture lookup for {key} found {len(matches)} solves")
        return matches[0]

    def attach_conformers(self, conformers) -> None:
        """Attach theta rows from the live solve, including cases with no bend prior."""
        for conf in conformers:
            cap = self.for_conformer(conf)
            row = int(conf.trial_idx)
            conf.solver_state = {"thetas": [t[row].detach().double().cpu().numpy().copy()
                                              for t in cap.thetas_final]}

