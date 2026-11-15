"""Fixed-shape end-of-solve optimisation of the delivered score."""
from __future__ import annotations

import torch


def optimise(theta0, score_fn, *, method, steps=50, delta0=None, diagnostics=None):
    """Retain the best score including the baseline and the final update.

    ``method`` is ``'irprop'`` (iRprop+, with backtracking) or ``'irprop_minus'`` (iRprop-, no backtracking). Both inherit
    the absolute Adam moment steps through ``delta0`` (or .01 when unavailable); diagnostics are optional and never
    affect updates.
    """
    if method not in ('irprop', 'irprop_minus'):
        raise ValueError(f'Unknown polish method: {method}')
    th = [t.detach().clone().requires_grad_(True) for t in theta0]
    best = [t.detach().clone() for t in th]
    best_score = score_fn(th).detach()
    state = IRpropState(th, best_score, delta0, record=diagnostics is not None, backtracking=method == 'irprop')
    for it in range(steps):
        score = score_fn(th)
        grad = torch.autograd.grad(score.sum(), th, allow_unused=True)
        force = [g if g is not None else torch.zeros_like(t) for g, t in zip(grad, th)]
        with torch.no_grad():
            delta = state.update(force, score.detach(), it)
            for t, d in zip(th, delta):
                t.add_(d)
        new_score = score_fn(th).detach()
        with torch.no_grad():
            improve = torch.isfinite(new_score) & (new_score > best_score)
            best_score = torch.where(improve, new_score, best_score)
            best = [torch.where(improve[:, None], t.detach(), b) for t, b in zip(th, best)]
    if diagnostics is not None:
        diagnostics.update(state.diagnostics())
    return best, best_score


class IRpropState:
    """iRprop+/iRprop− for independent maximisation rows with fixed-shape state."""
    def __init__(self, theta, score, delta0=None, *, record=False, backtracking=True):
        initial = [torch.full_like(t, .01) for t in theta] if delta0 is None else delta0
        self.delta = [d.detach().abs().clone().clamp(1e-6, .1) for d in initial]
        self.prev_grad = [torch.zeros_like(t) for t in theta]
        self.prev_update = [torch.zeros_like(t) for t in theta]
        self.prev_score = score.detach().clone()
        self.converged = torch.full_like(score, -1, dtype=torch.long)
        self.history = []
        self.record = record
        self.backtracking = backtracking

    def update(self, grad, score, iteration):
        updates = []
        active = self.converged < 0
        for j, g in enumerate(grad):
            product = g * self.prev_grad[j]
            flip = product < 0
            d = torch.where(product > 0, (self.delta[j] * 1.2).clamp(max=.1),
                            torch.where(flip, (self.delta[j] * .5).clamp(min=1e-6), self.delta[j]))
            # A row-level loss in score triggers only its flipped DOFs.
            reverse = (torch.where((score < self.prev_score)[:, None],
                       -self.prev_update[j], torch.zeros_like(g))
                       if self.backtracking else torch.zeros_like(g))
            u = torch.where(flip, reverse, g.sign() * d)
            u = torch.where(active[:, None], u, torch.zeros_like(u))
            self.delta[j] = torch.where(active[:, None], d, self.delta[j])
            self.prev_grad[j] = torch.where(flip, torch.zeros_like(g), g)
            self.prev_update[j] = u
            updates.append(u)
        max_update = torch.cat(updates, -1).abs().amax(-1)
        grad_norm = torch.cat(grad, -1).norm(dim=-1)
        done = (max_update < 1e-5) | (grad_norm < 1e-8)
        self.converged = torch.where(active & done, torch.full_like(self.converged, iteration + 1), self.converged)
        self.prev_score = score.detach().clone()
        if self.record:
            self.history.append(torch.stack([torch.cat(self.delta, -1).median(-1).values,
                                (torch.cat(self.delta, -1) >= .1).float().mean(-1),
                                (torch.cat(self.delta, -1) <= 1e-6).float().mean(-1)], -1).detach())
        return updates

    def diagnostics(self):
        return {'convergence_iteration': self.converged.cpu().tolist(),
                'delta_history_median_max_fraction_min_fraction': torch.stack(self.history).cpu().tolist()}
