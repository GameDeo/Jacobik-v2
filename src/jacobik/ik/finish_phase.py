"""Device-resident two-phase optimizer state (the IA-k finish phase; production profile field finish_phase).

All policy choices are supplied in the profile dictionary. Widths are in FK
coordinates: translations in Angstrom, every other coordinate in radians.
"""
import numpy as np
import triton
import triton.language as tl

from jacobik.core import gpu

cp = gpu.cp

@triton.jit
def _finish_observe_state_kernel(
    t_ptr, loss_ptr, ring_ptr, active_ptr, switched_ptr, count_ptr, prev_loss_ptr,
    B, window: tl.constexpr, atol, start, latest, fixed_switch,
    HAS_FIXED_SWITCH: tl.constexpr, WHOLE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row < B

    t = tl.load(t_ptr)
    slot = t % window
    loss = tl.load(loss_ptr + row, mask=mask, other=0.0)
    tl.store(ring_ptr + slot * B + row, loss, mask=mask)

    r_min = loss
    r_max = loss
    for w in range(window):
        val = tl.load(ring_ptr + w * B + row, mask=mask, other=0.0)
        r_min = tl.minimum(r_min, val)
        r_max = tl.maximum(r_max, val)

    flat = (r_max - r_min) < atol
    allowed = t >= start
    if HAS_FIXED_SWITCH:
        trigger = allowed & (t >= fixed_switch)
    else:
        trigger = allowed & (flat | (t >= latest))

    act_old = tl.load(active_ptr + row, mask=mask, other=False)
    sw = (trigger & (~act_old)) | (WHOLE & (t == 0))
    act_new = act_old | trigger

    cnt_old = tl.load(count_ptr + row, mask=mask, other=0)
    cnt_new = tl.where(sw, 0, cnt_old)

    pl_old = tl.load(prev_loss_ptr + row, mask=mask, other=0.0)
    pl_new = tl.where(sw, float("inf"), pl_old)

    tl.store(switched_ptr + row, sw, mask=mask)
    tl.store(active_ptr + row, act_new, mask=mask)
    tl.store(count_ptr + row, cnt_new, mask=mask)
    tl.store(prev_loss_ptr + row, pl_new, mask=mask)

@triton.jit
def _finish_observe_lig_kernel(switched_ptr, sd_ptr, m_ptr, v_ptr, pg_ptr, delta_ptr, ps_ptr, B, D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * D
    mask = offs < total
    row = offs // D
    sw = tl.load(switched_ptr + row, mask=mask, other=False)
    msw = mask & sw
    sd = tl.load(sd_ptr + offs, mask=msw, other=0.0)
    tl.store(m_ptr + offs, 0.0, mask=msw)
    tl.store(v_ptr + offs, 0.0, mask=msw)
    tl.store(pg_ptr + offs, 0.0, mask=msw)
    tl.store(ps_ptr + offs, 0.0, mask=msw)
    tl.store(delta_ptr + offs, 0.14 * sd, mask=msw)

@triton.jit
def _finish_update_lig_kernel(
    g_ptr, sd_ptr, m_ptr, v_ptr, pg_ptr, delta_ptr, ps_ptr, step_ptr,
    active_ptr, done_ptr, count_ptr, loss_ptr, prev_loss_ptr, scale_ptr, bc1_ptr, bc2_ptr,
    B, D, search_lr, search_sigma,
    SEARCH_MODE: tl.constexpr, FINISH_MODE: tl.constexpr,
    HAS_SCALE: tl.constexpr, NEED_ADAM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * D
    mask = offs < total
    row = offs // D
    col = offs % D

    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    sd = tl.load(sd_ptr + offs, mask=mask, other=1.0)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    pg = tl.load(pg_ptr + offs, mask=mask, other=0.0)
    delta = tl.load(delta_ptr + offs, mask=mask, other=0.0)
    ps = tl.load(ps_ptr + offs, mask=mask, other=0.0)

    active = tl.load(active_ptr + row, mask=mask, other=False)
    done = tl.load(done_ptr + row, mask=mask, other=False)

    m_new = m * 0.9 + g * 0.1
    v_new = v * 0.999 + (g * g) * 0.001

    tl.store(m_ptr + offs, m_new, mask=mask)
    tl.store(v_ptr + offs, v_new, mask=mask)

    adam = 0.0
    if NEED_ADAM:
        cnt = tl.load(count_ptr + row, mask=mask, other=1)
        bc1 = tl.load(bc1_ptr + cnt, mask=mask, other=1.0)
        bc2 = tl.load(bc2_ptr + cnt, mask=mask, other=1.0)
        m_hat = m_new / bc1
        v_hat = tl.sqrt(v_new / bc2) + 1e-8
        lr_eff = tl.where(active, 0.14 * sd, search_lr)
        adam = (m_hat / v_hat) * lr_eff

    product = g * pg
    pos = product > 0.0
    neg = product < 0.0
    delta_new = tl.where(pos, delta * 1.2, tl.where(neg, delta * 0.5, delta))
    bound = tl.where(active, sd, search_sigma)
    min_bound = 0.001 * bound
    delta_new = tl.minimum(tl.maximum(delta_new, min_bound), bound)
    tl.store(delta_ptr + offs, delta_new, mask=mask)

    effective = tl.where(neg, 0.0, g)
    eff_sign = tl.where(effective > 0.0, 1.0, tl.where(effective < 0.0, -1.0, 0.0))
    rp = eff_sign * delta_new

    loss = tl.load(loss_ptr + row, mask=mask, other=0.0)
    prev_loss = tl.load(prev_loss_ptr + row, mask=mask, other=0.0)
    rollback = neg & (loss > prev_loss)
    rp_plus = tl.where(rollback, -ps, rp)

    if SEARCH_MODE == 0:
        search_step = adam
    elif SEARCH_MODE == 1:
        search_step = rp_plus
    else:
        search_step = rp

    if FINISH_MODE == 0:
        finish_step = adam
    elif FINISH_MODE == 1:
        finish_step = rp_plus
    else:
        finish_step = rp

    step = tl.where(active, finish_step, search_step)
    if HAS_SCALE:
        sc = tl.load(scale_ptr + col, mask=mask, other=1.0)
        step = tl.where(active, step, step * sc)
    step = tl.where(done, 0.0, step)

    tl.store(pg_ptr + offs, effective, mask=mask)
    tl.store(ps_ptr + offs, step, mask=mask)
    tl.store(step_ptr + offs, step, mask=mask)


class FinishPhase:
    def __init__(self, theta, fks, steps, bend_end, config, window, atol, search_lr=.1):
        required = {'release', 'sigma'}
        if not required <= config.keys():
            raise ValueError('finish_phase requires explicit release and sigma')
        self.sigma_source = config.get('sigma_source', 'thermal')
        if self.sigma_source not in ('thermal', 'curvature'):
            raise ValueError('unknown sigma source')
        self.release = set(config['release'])
        if not self.release <= {'shape', 'symmetry', 'projection'}:
            raise ValueError('unknown finish aid')
        self.search = config.get('search', 'adam')
        self.optimizer = config.get('optimizer', 'adam')
        if self.search not in ('adam', 'irprop+', 'irprop-') or self.optimizer not in ('adam', 'irprop+', 'irprop-'):
            raise ValueError('unknown phase optimizer')
        self.start = int(bend_end)
        self.whole = bool(config.get('score_only', False))
        self.fixed_switch = config.get('switch')
        if 'cap' in config:
            self.latest = steps - int(config['cap'])
            if not self.start <= self.latest < steps:
                raise ValueError('finish cap must leave bend release intact')
        elif self.whole or self.fixed_switch is not None:
            # Whole-run and fixed-switch modes never use the adaptive deadline.
            self.latest = steps - 1
            # A short requested solve can end before its fixed switch. It runs
            # the unchanged search updates and never activates the finish.
            if not self.whole and not self.start <= int(self.fixed_switch):
                raise ValueError('fixed switch must leave bend release intact')
        else:
            raise ValueError('adaptive finish_phase requires explicit cap')
        self.search_sigma = float(search_lr) / .14
        self.window, self.atol = window, atol
        B = len(theta[0])
        dt = theta[0].dtype
        self.active = cp.full((B,), self.whole, dtype=cp.bool_)
        self.done = cp.zeros((B,), dtype=cp.bool_)
        self.switched = cp.zeros((B,), dtype=cp.bool_)
        self.count = cp.zeros((B,), dtype=cp.int64)
        self.ring = cp.full((window, B), float('inf'), dtype=dt)
        self.previous_loss = cp.full((B,), float('inf'), dtype=dt)
        self.stop = cp.zeros((1,), dtype=cp.bool_)
        self.sigma, self.m, self.v, self.previous_g, self.delta, self.previous_step = [], [], [], [], [], []
        self.record = bool(config.get("diagnostics", False))
        self.sign_flips = [cp.zeros((20, *th.shape), dtype=cp.bool_) for th in theta] if self.record else None
        self.invalid_curvature = [cp.zeros(th.shape, dtype=cp.bool_) for th in theta]
        self.probes = []
        if self.sigma_source == 'curvature':
            raise RuntimeError("FinishPhase curvature sigma_source is not supported in zero-torch GPU build.")
        widths = config['sigma']
        w_tr = float(widths['translation'])
        w_rot = float(widths['rotation'])
        w_tor = float(widths['torsion'])
        w_ring = float(widths['ring'])
        w_bend = float(widths['bend'])
        if min(w_tr, w_rot, w_tor, w_ring, w_bend) <= 0.0:
            raise ValueError('thermal widths must be positive')
        for th, fk in zip(theta, fks):
            D = th.shape[1]
            row = np.empty(D, dtype=np.float32 if dt == cp.float32 else np.float64)
            row[:3] = w_tr
            row[3:6] = w_rot
            nm = getattr(fk, 'n_macro_joints', 0)
            end = 6 + fk.n_joints
            row[6:end-nm] = w_tor
            row[end-nm:end] = w_ring
            row[end:] = w_bend
            sd = cp.broadcast_to(cp.asarray(row)[None, :], th.shape).copy()
            self.sigma.append(sd)
            for state in (self.m, self.v, self.previous_g, self.previous_step):
                state.append(cp.zeros_like(th))
            self.delta.append(.14 * sd if self.whole else cp.full_like(sd, float(search_lr)))
        self._step_bufs = [cp.empty_like(th) for th in theta]
        _mode_map = {'adam': 0, 'irprop+': 1, 'irprop-': 2}
        self._sm = _mode_map[self.search]
        self._fm = _mode_map[self.optimizer]
        self._need_adam = (self._sm == 0 or self._fm == 0)
        if self._need_adam:
            age_t = cp.arange(steps + 4, dtype=dt)
            self._bc1_tbl = cp.maximum(1.0 - 0.9 ** age_t, 1e-12)
            self._bc2_tbl = cp.maximum(1.0 - 0.999 ** age_t, 1e-12)
        else:
            self._bc1_tbl = self._bc2_tbl = None

    def observe(self, t, loss):
        if not isinstance(loss, cp.ndarray):
            raise RuntimeError("FinishPhase requires CUDA CuPy arrays.")
        B = loss.shape[0]
        gpu.launch(
            _finish_observe_state_kernel,
            (gpu.cdiv(B, 128),),
            t, loss, self.ring, self.active, self.switched, self.count, self.previous_loss,
            B, int(self.window), float(self.atol), int(self.start), int(self.latest),
            int(self.fixed_switch or 0),
            HAS_FIXED_SWITCH=self.fixed_switch is not None,
            WHOLE=self.whole,
            BLOCK_SIZE=128,
        )
        for sd, m, v, pg, delta, ps in zip(self.sigma, self.m, self.v, self.previous_g, self.delta, self.previous_step):
            D = sd.shape[1]
            gpu.launch(
                _finish_observe_lig_kernel,
                (gpu.cdiv(B * D, 256),),
                self.switched, sd, m, v, pg, delta, ps, B, D, BLOCK_SIZE=256,
            )

    def aid(self, name, value):
        return cp.where(self.active, 0.0, value) if name in self.release or self.whole else value

    def update(self, t, grads, score_grads, loss, scales, search_lr):
        if not isinstance(loss, cp.ndarray):
            raise RuntimeError("FinishPhase requires CUDA CuPy arrays.")
        if score_grads is not None:
            gain = 0.5 * sum(cp.sum((g * sd) ** 2, axis=-1) for g, sd in zip(score_grads, self.sigma))
            self.done |= self.active & (gain < 0.1)
        self.count += 1
        B = loss.shape[0]
        out = []
        for i, (g, sd) in enumerate(zip(grads, self.sigma)):
            if self.sign_flips is not None:
                self.sign_flips[i][int(t.item()) % 20] = (g * self.previous_g[i]) < 0
            D = g.shape[1]
            st_out = self._step_bufs[i]
            if st_out.shape != g.shape:
                st_out = self._step_bufs[i] = cp.empty_like(g)
            has_sc = scales is not None and scales[i] is not None
            sc_ptr = scales[i] if has_sc else sd
            bc1_p = self._bc1_tbl if self._need_adam else sd
            bc2_p = self._bc2_tbl if self._need_adam else sd
            gpu.launch(
                _finish_update_lig_kernel,
                (gpu.cdiv(B * D, 256),),
                cp.ascontiguousarray(g), sd, self.m[i], self.v[i], self.previous_g[i], self.delta[i], self.previous_step[i], st_out,
                self.active, self.done, self.count, loss, self.previous_loss,
                sc_ptr, bc1_p, bc2_p,
                B, D,
                float(search_lr), float(self.search_sigma),
                SEARCH_MODE=self._sm, FINISH_MODE=self._fm,
                HAS_SCALE=has_sc, NEED_ADAM=self._need_adam,
                BLOCK_SIZE=256,
            )
            out.append(st_out)
        self.previous_loss[...] = loss
        self.stop[...] = cp.all(self.done)
        return out

    def select(self, keep):
        for name in ('active', 'done', 'switched', 'count', 'previous_loss'):
            setattr(self, name, getattr(self, name)[keep])
        self.ring = self.ring[:, keep]
        self.probes = [[v[keep] for v in probe] for probe in self.probes]
        if self.sign_flips is not None:
            self.sign_flips = [v[:, keep] for v in self.sign_flips]
        for name in ('sigma', 'm', 'v', 'previous_g', 'delta', 'previous_step', 'invalid_curvature', '_step_bufs'):
            setattr(self, name, [v[keep] for v in getattr(self, name)])
