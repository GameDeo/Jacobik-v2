"""Fixed-shape end-of-solve optimisation of the delivered score (CuPy + Triton)."""
from __future__ import annotations

import triton
import triton.language as tl

from jacobik.core import gpu

cp = gpu.cp

@triton.jit
def _polish_irprop_lig_kernel(
    g_ptr, pg_ptr, delta_ptr, pu_ptr, u_ptr, score_ptr, prev_score_ptr, conv_ptr,
    B, D, BACKTRACKING: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * D
    mask = offs < total
    row = offs // D

    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    pg = tl.load(pg_ptr + offs, mask=mask, other=0.0)
    delta = tl.load(delta_ptr + offs, mask=mask, other=0.01)
    pu = tl.load(pu_ptr + offs, mask=mask, other=0.0)
    active = tl.load(conv_ptr + row, mask=mask, other=0) < 0

    product = g * pg
    pos = product > 0.0
    flip = product < 0.0
    d = tl.where(
        pos,
        tl.minimum(delta * 1.2, 0.1),
        tl.where(flip, tl.maximum(delta * 0.5, 1e-6), delta),
    )
    if BACKTRACKING:
        sc = tl.load(score_ptr + row, mask=mask, other=0.0)
        psc = tl.load(prev_score_ptr + row, mask=mask, other=0.0)
        reverse = tl.where(sc < psc, -pu, 0.0)
    else:
        reverse = 0.0

    g_sign = tl.where(g > 0.0, 1.0, tl.where(g < 0.0, -1.0, 0.0))
    u = tl.where(flip, reverse, g_sign * d)
    u = tl.where(active, u, 0.0)
    d_new = tl.where(active, d, delta)
    pg_new = tl.where(flip, 0.0, g)

    tl.store(delta_ptr + offs, d_new, mask=mask)
    tl.store(pg_ptr + offs, pg_new, mask=mask)
    tl.store(pu_ptr + offs, u, mask=mask)
    tl.store(u_ptr + offs, u, mask=mask)


def optimise(theta0, score_fn, *, method, steps=50, delta0=None, diagnostics=None):
    """Retain the best score including the baseline and the final update.

    ``method`` is ``'irprop'`` (iRprop+, with backtracking) or ``'irprop_minus'`` (iRprop-, no backtracking). Both inherit
    the absolute Adam moment steps through ``delta0`` (or .01 when unavailable); diagnostics are optional and never
    affect updates. ``score_fn(th)`` may return ``(score, grads)`` or ``score`` (central finite differences).
    """
    if method not in ('irprop', 'irprop_minus'):
        raise ValueError(f'Unknown polish method: {method}')
    th = [cp.ascontiguousarray(t.copy()) for t in theta0]
    best = [t.copy() for t in th]

    def _eval(thetas_list):
        out = score_fn(thetas_list)
        if isinstance(out, tuple):
            return out[0], out[1]
        sc = out
        eps = 1e-4
        grads = []
        for idx, arr in enumerate(thetas_list):
            g = cp.zeros_like(arr)
            B, D = arr.shape
            for d in range(D):
                orig = arr[:, d].copy()
                arr[:, d] = orig + eps
                fp = score_fn(thetas_list)
                if isinstance(fp, tuple):
                    fp = fp[0]
                arr[:, d] = orig - eps
                fm = score_fn(thetas_list)
                if isinstance(fm, tuple):
                    fm = fm[0]
                arr[:, d] = orig
                g[:, d] = (fp - fm) / (2.0 * eps)
            grads.append(g)
        return sc, grads

    best_score, _ = _eval(th)
    best_score = best_score.copy()
    state = IRpropState(th, best_score, delta0, record=diagnostics is not None, backtracking=method == 'irprop')
    for it in range(steps):
        score, force = _eval(th)
        delta = state.update(force, score, it)
        for t, d in zip(th, delta):
            t += d
        new_score, _ = _eval(th)
        improve = cp.isfinite(new_score) & (new_score > best_score)
        best_score = cp.where(improve, new_score, best_score)
        best = [cp.where(improve[:, None], t.copy(), b) for t, b in zip(th, best)]
    if diagnostics is not None:
        diagnostics.update(state.diagnostics())
    return best, best_score


class IRpropState:
    """iRprop+/iRprop− for independent maximisation rows with fixed-shape state."""
    def __init__(self, theta, score, delta0=None, *, record=False, backtracking=True):
        initial = [cp.full_like(t, .01) for t in theta] if delta0 is None else delta0
        self.delta = [cp.clip(cp.abs(d.copy()), 1e-6, .1) for d in initial]
        self.prev_grad = [cp.zeros_like(t) for t in theta]
        self.prev_update = [cp.zeros_like(t) for t in theta]
        self._u_bufs = [cp.empty_like(t) for t in theta]
        self.prev_score = score.copy()
        self.converged = cp.full(score.shape, -1, dtype=cp.int64)
        self.history = []
        self.record = record
        self.backtracking = backtracking

    def update(self, grad, score, iteration):
        if not isinstance(score, cp.ndarray):
            raise RuntimeError("IRpropState requires CUDA CuPy arrays.")
        updates = []
        active = self.converged < 0
        B = score.shape[0]
        for j, g in enumerate(grad):
            D = g.shape[1]
            u_out = self._u_bufs[j]
            if u_out.shape != g.shape:
                u_out = self._u_bufs[j] = cp.empty_like(g)
            gpu.launch(
                _polish_irprop_lig_kernel,
                (gpu.cdiv(B * D, 256),),
                cp.ascontiguousarray(g),
                self.prev_grad[j],
                self.delta[j],
                self.prev_update[j],
                u_out,
                score,
                self.prev_score,
                self.converged,
                B,
                D,
                BACKTRACKING=bool(self.backtracking),
                BLOCK_SIZE=256,
            )
            updates.append(u_out)
        cat_u = cp.concatenate(updates, axis=-1)
        cat_g = cp.concatenate(grad, axis=-1)
        max_update = cp.max(cp.abs(cat_u), axis=-1)
        grad_norm = cp.sqrt(cp.sum(cat_g * cat_g, axis=-1))
        done = (max_update < 1e-5) | (grad_norm < 1e-8)
        self.converged = cp.where(active & done, cp.full_like(self.converged, iteration + 1), self.converged)
        self.prev_score = score.copy()
        if self.record:
            cat_d = cp.concatenate(self.delta, axis=-1)
            self.history.append(cp.stack([
                cp.median(cat_d, axis=-1),
                cp.mean((cat_d >= .1).astype(cp.float32), axis=-1),
                cp.mean((cat_d <= 1e-6).astype(cp.float32), axis=-1),
            ], axis=-1))
        return updates

    def diagnostics(self):
        return {'convergence_iteration': gpu.to_cpu(self.converged).tolist(),
                'delta_history_median_max_fraction_min_fraction': gpu.to_cpu(cp.stack(self.history)).tolist()}
