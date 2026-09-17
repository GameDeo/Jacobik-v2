#!/usr/bin/env python3
"""
Patch src/jacobik/vis/ensemble_view.py: Q2 histogram-overlay panel switches
from VMM to GMM-on-cos(angle), matching the stats/fitter.py + ik/scorer.py
refit. scorer._q2_kappas no longer exists post-refit (replaced by
scorer._q2_vars); this file was reading it directly and would otherwise
break at runtime with an AttributeError.

torsion_landscape.py's separate scorer._q3_kappas reference is deliberately
left untouched -- already broken/stale from an earlier, unrelated signature
mismatch (out of scope here).
"""
import sys

PATH = "src/jacobik/vis/ensemble_view.py"

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
# EV1: Q2 data collection -- stop degrees-converting means, read _q2_vars
# instead of the no-longer-existing _q2_kappas.
# ---------------------------------------------------------------------------
OLD_COLLECT = '''    # ── Collect Q2 data ──────────────────────────────────────────────────
    q2_entries = []  # (label, actual_angle_deg, means_deg, kappas, weights)
    n_pairs = scorer._q2_means.shape[0]
    sites = mol.coord_sites
    for p, (si, sj, dist_cls) in enumerate(scorer._q2_pair_meta[:n_pairs]):
        pa = np.asarray(sites[si].position, dtype=float)
        pb = np.asarray(sites[sj].position, dtype=float)
        va, vb = pa - metal_pos, pb - metal_pos
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-8 or nb < 1e-8:
            continue
        cos_t = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_t)))
        la = sites[si].label or f"D{si}"
        lb = sites[sj].label or f"D{sj}"
        means_deg = np.degrees(scorer._q2_means[p].cpu().numpy())
        kappas = scorer._q2_kappas[p].cpu().numpy()
        weights_arr = scorer._q2_weights[p].cpu().numpy()
        q2_entries.append((f"{la}–M–{lb}", angle_deg, means_deg, kappas, weights_arr))'''
NEW_COLLECT = '''    # ── Collect Q2 data ──────────────────────────────────────────────────
    # 2.7.0: Q2 moved from VMM to GMM-on-cos(angle) -- see stats/fitter.py's
    # fit_gmm_cosangle() docstring (Q2 is bounded [0,180] deg via arccos,
    # never wraps to 360, so it is not a circular quantity). means/vars are
    # already dimensionless cos-space values -- no np.degrees() conversion.
    q2_entries = []  # (label, actual_angle_deg, means_cos, vars_cos, weights)
    n_pairs = scorer._q2_means.shape[0]
    sites = mol.coord_sites
    for p, (si, sj, dist_cls) in enumerate(scorer._q2_pair_meta[:n_pairs]):
        pa = np.asarray(sites[si].position, dtype=float)
        pb = np.asarray(sites[sj].position, dtype=float)
        va, vb = pa - metal_pos, pb - metal_pos
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-8 or nb < 1e-8:
            continue
        cos_t = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_t)))
        la = sites[si].label or f"D{si}"
        lb = sites[sj].label or f"D{sj}"
        means_cos = scorer._q2_means[p].cpu().numpy()
        vars_cos = scorer._q2_vars[p].cpu().numpy()
        weights_arr = scorer._q2_weights[p].cpu().numpy()
        q2_entries.append((f"{la}–M–{lb}", angle_deg, means_cos, vars_cos, weights_arr))'''
do_replace("EV1-q2-collect", OLD_COLLECT, NEW_COLLECT)

# ---------------------------------------------------------------------------
# EV2: Q2 plotting -- VMM density formula -> GMM-in-cos-space equivalent,
# same display grid/normalisation/legend/axis code otherwise unchanged.
# ---------------------------------------------------------------------------
OLD_PLOT = '''        # ── Q2 subplots ──────────────────────────────────────────────────
        offset = len(q1_entries)
        for j, (lbl, actual_deg, means_deg, kappas, weights_arr) in enumerate(q2_entries):
            ax = axes_flat[offset + j]
            ax.set_facecolor("#0F0F1A")

            xs_deg = np.linspace(0, 180, 360)
            xs_rad = np.radians(xs_deg)
            density = np.zeros_like(xs_rad)
            active = weights_arr > 1e-6
            for k in range(len(means_deg)):
                if not active[k]:
                    continue
                kappa = float(kappas[k])
                mu_rad = math.radians(means_deg[k])
                # See exp.md for detailed rationale
                lp = kappa * np.cos(xs_rad - mu_rad)
                density += weights_arr[k] * np.exp(lp - lp.max())  # relative
            if density.max() > 0:
                density /= density.max()  # normalise to 1 for display'''
NEW_PLOT = '''        # ── Q2 subplots ──────────────────────────────────────────────────
        # 2.7.0: GMM-on-cos(angle) density, not VMM -- evaluated at
        # cos(theta) for each display-grid angle theta. See stats/
        # fitter.py's fit_gmm_cosangle() docstring for the rationale.
        offset = len(q1_entries)
        _Q2_VAR_FLOOR = 0.0025  # mirrors ik/scorer.py's _GMM_VAR_MIN["Q2"]
        for j, (lbl, actual_deg, means_cos, vars_cos, weights_arr) in enumerate(q2_entries):
            ax = axes_flat[offset + j]
            ax.set_facecolor("#0F0F1A")

            xs_deg = np.linspace(0, 180, 360)
            xs_rad = np.radians(xs_deg)
            xs_cos = np.cos(xs_rad)
            density = np.zeros_like(xs_rad)
            active = weights_arr > 1e-6
            for k in range(len(means_cos)):
                if not active[k]:
                    continue
                var_k = max(float(vars_cos[k]), _Q2_VAR_FLOOR)
                mu_k = float(means_cos[k])
                # Relative density (const -0.5*log(2*pi) term omitted, same
                # as the VMM formula this replaces omitted its log-norm
                # constant -- both immediately renormalised to max=1 below).
                lp = -0.5 * math.log(var_k) - 0.5 * (xs_cos - mu_k) ** 2 / var_k
                density += weights_arr[k] * np.exp(lp - lp.max())  # relative
            if density.max() > 0:
                density /= density.max()  # normalise to 1 for display'''
do_replace("EV2-q2-plot", OLD_PLOT, NEW_PLOT)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
