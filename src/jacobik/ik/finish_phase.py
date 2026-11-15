"""Device-resident two-phase optimizer state (experimental, default disabled).

All policy choices are supplied in the profile dictionary. Widths are in FK
coordinates: translations in Angstrom, every other coordinate in radians.
"""
import torch


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
        self.active = torch.full((len(theta[0]),), self.whole, device=theta[0].device, dtype=torch.bool)
        self.done = torch.zeros_like(self.active)
        self.switched = torch.zeros_like(self.active)
        self.count = torch.zeros_like(self.active, dtype=torch.long)
        self.ring = theta[0].new_full((window, len(theta[0])), float('inf'))
        self.previous_loss = theta[0].new_full((len(theta[0]),), float('inf'))
        self.stop = torch.zeros(1, device=theta[0].device, dtype=torch.bool)
        self.sigma, self.m, self.v, self.previous_g, self.delta, self.previous_step = [], [], [], [], [], []
        self.record = bool(config.get("diagnostics", False))
        self.sign_flips = [torch.zeros((20, *th.shape), dtype=torch.bool, device=th.device) for th in theta] if self.record else None
        self.invalid_curvature = [torch.zeros_like(th, dtype=torch.bool) for th in theta]
        self.probes = []
        generator = torch.Generator(device=theta[0].device).manual_seed(11)
        for _ in range(4):
            self.probes.append([torch.randint(0, 2, th.shape, device=th.device, generator=generator).to(th.dtype)*2-1 for th in theta])
        widths = config['sigma']
        for th, fk in zip(theta, fks):
            sd = torch.empty_like(th)
            sd[:, :3] = float(widths['translation'])
            sd[:, 3:6] = float(widths['rotation'])
            nm = getattr(fk, 'n_macro_joints', 0)
            end = 6 + fk.n_joints
            sd[:, 6:end-nm] = float(widths['torsion'])
            sd[:, end-nm:end] = float(widths['ring'])
            sd[:, end:] = float(widths['bend'])
            if not bool((sd > 0).all()):
                raise ValueError('thermal widths must be positive')
            self.sigma.append(sd)
            for state in (self.m, self.v, self.previous_g, self.previous_step):
                state.append(torch.zeros_like(th))
            self.delta.append(.14 * sd if self.whole else torch.full_like(sd, float(search_lr)))

    @torch.no_grad()
    def observe(self, t, loss):
        self.ring.index_copy_(0, t % self.window, loss.detach()[None])
        flat = self.ring.max(0).values - self.ring.min(0).values < self.atol
        allowed = t >= self.start
        trigger = allowed & (flat | (t >= self.latest))
        if self.fixed_switch is not None:
            trigger = allowed & (t >= int(self.fixed_switch))
        self.switched.copy_((trigger & ~self.active) | (self.whole & (t == 0)))
        self.active.logical_or_(trigger)
        self.count.copy_(torch.where(self.switched, 0, self.count))
        for sd, m, v, pg, delta, ps in zip(self.sigma, self.m, self.v, self.previous_g, self.delta, self.previous_step):
            mask = self.switched[:, None]
            for state in (m, v, pg, ps):
                state.masked_fill_(mask, 0)
            delta.copy_(torch.where(mask, .14 * sd, delta))
        self.previous_loss.copy_(torch.where(self.switched, float('inf'), self.previous_loss))

    def curvature(self, gradients, theta, normalization):
        """Four fixed Rademacher probes; update only switching rows on device.

        Nonpositive/nonfinite estimates retain the explicit class width. The
        diagnostic must measure estimator cost and invalid-estimate frequency.
        """
        diagonal = [torch.zeros_like(th) for th in theta]
        for probe in self.probes:
            dot = sum((g * z).sum() for g, z in zip(gradients, probe))
            hv = torch.autograd.grad(dot, theta, retain_graph=True, allow_unused=True)
            for i, (h, z) in enumerate(zip(hv, probe)):
                if h is not None:
                    diagonal[i] = diagonal[i] + h.detach() * z * (normalization / len(self.probes))
        with torch.no_grad():
            for sd, h, delta, invalid in zip(self.sigma, diagonal, self.delta, self.invalid_curvature):
                valid = (h > 0) & torch.isfinite(h)
                invalid.copy_(torch.where(self.switched[:, None], ~valid, invalid))
                estimate = h.clamp_min(torch.finfo(h.dtype).tiny).rsqrt()
                sd.copy_(torch.where(self.switched[:, None] & valid, estimate, sd))
                delta.copy_(torch.where(self.switched[:, None], .14 * sd, delta))

    def aid(self, name, value):
        return torch.where(self.active, 0., value) if name in self.release or self.whole else value

    @torch.no_grad()
    def update(self, t, grads, score_grads, loss, scales, search_lr):
        gain = .5 * sum((g * sd).square().sum(-1) for g, sd in zip(score_grads, self.sigma))
        self.done.logical_or_(self.active & (gain < .1))
        self.count.add_(1)
        out = []
        for i, (g, sd) in enumerate(zip(grads, self.sigma)):
            m, v, pg, delta, ps = self.m[i], self.v[i], self.previous_g[i], self.delta[i], self.previous_step[i]
            m.mul_(.9).add_(g, alpha=.1)
            v.mul_(.999).addcmul_(g, g, value=.001)
            # Neither phase reads the Adam proposal in a pure iRprop run.
            # Keep moment/count state identical, but skip its unused proposal.
            adam = None
            if self.search == 'adam' or self.optimizer == 'adam':
                age = self.count[:, None].to(g.dtype)
                adam = m / (1 - .9 ** age) / ((v / (1 - .999 ** age)).sqrt() + 1e-8)
                adam = adam * torch.where(self.active[:, None], .14 * sd, torch.full_like(sd, search_lr))
            product = g * pg
            if self.sign_flips is not None:
                self.sign_flips[i].index_copy_(0, t % 20, (product < 0)[None])
            delta.copy_(torch.where(product > 0, delta * 1.2, torch.where(product < 0, delta * .5, delta)))
            bound = torch.where(self.active[:, None], sd, torch.full_like(sd, self.search_sigma))
            delta.copy_(torch.minimum(torch.maximum(delta, .001 * bound), bound))
            effective = torch.where(product < 0, 0., g)
            rp = effective.sign() * delta
            rollback = (product < 0) & (loss > self.previous_loss)[:, None]
            rp_plus = torch.where(rollback, -ps, rp)
            search_step = adam if self.search == 'adam' else (rp_plus if self.search == 'irprop+' else rp)
            finish_step = adam if self.optimizer == 'adam' else (rp_plus if self.optimizer == 'irprop+' else rp)
            step = torch.where(self.active[:, None], finish_step, search_step)
            if scales is not None:
                step = torch.where(self.active[:, None], step, step * scales[i])
            step = torch.where(self.done[:, None], 0., step)
            pg.copy_(effective)
            ps.copy_(step)
            out.append(step)
        self.previous_loss.copy_(loss)
        self.stop.copy_(self.done.all())
        return out

    def select(self, keep):
        for name in ('active', 'done', 'switched', 'count', 'previous_loss'):
            setattr(self, name, getattr(self, name)[keep])
        self.ring = self.ring[:, keep]
        self.probes = [[v[keep] for v in probe] for probe in self.probes]
        if self.sign_flips is not None:
            self.sign_flips = [v[:, keep] for v in self.sign_flips]
        for name in ('sigma', 'm', 'v', 'previous_g', 'delta', 'previous_step', 'invalid_curvature'):
            setattr(self, name, [v[keep] for v in getattr(self, name)])
