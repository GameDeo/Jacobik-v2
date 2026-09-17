#!/usr/bin/env python3
"""
Patch src/jacobik/vis/visualizer.py: Q2/Q3 diagnostic panels switch from
VMM to GMM-on-cos(angle), matching the stats/fitter.py + ik/scorer.py
refit (Q2/Q3 are bounded [0,180] deg via arccos, never wrap to 360 -- not
circular quantities; see fitter.py's fit_gmm_cosangle() docstring).
"""
import sys

PATH = "src/jacobik/vis/visualizer.py"

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
# V1: import -- no change needed, _gmm_fields already imported alongside
# _vmm_fields. Add the Q2/Q3 cos-space var floor constant right after the
# import block's neighbourhood (module-level, next to eval_gmm_log_prob).
# ---------------------------------------------------------------------------
OLD_HEADER = '''# ---------------------------------------------------------------------------
# GMM/VMM evaluation functions for individual term analysis
# ---------------------------------------------------------------------------
def eval_gmm_log_prob(
    x: float, means: List[float], vars_: List[float], weights: List[float]
) -> float:
    _LOG_2PI = math.log(2 * math.pi)
    log_probs = []
    for m, v, w in zip(means, vars_, weights):
        v = max(v, 0.04)  # variance floor
        lp = -0.5 * _LOG_2PI - 0.5 * math.log(v) - 0.5 * (x - m) ** 2 / v + math.log(max(w, 1e-12))
        log_probs.append(lp)
    from scipy.special import logsumexp

    return float(logsumexp(log_probs))'''
NEW_HEADER = '''# ---------------------------------------------------------------------------
# GMM/VMM evaluation functions for individual term analysis
# ---------------------------------------------------------------------------

#: Q2/Q3 (2.7.0) cos-space GMM variance floor -- mirrors
#: ik/scorer.py's _GMM_VAR_MIN["Q2"]/["Q3"]. 0.04 (this module's old
#: hardcoded default, still used for Q1/Q4's Å²/deg² floors) is the wrong
#: scale for a dimensionless cos-space quantity in [-1,1] -- see
#: stats/fitter.py's fit_gmm_cosangle() docstring for why Q2/Q3 are fit in
#: cos-space at all.
_Q2Q3_GMM_COS_VAR_FLOOR = 0.0025


def eval_gmm_log_prob(
    x: float,
    means: List[float],
    vars_: List[float],
    weights: List[float],
    var_floor: float = 0.04,
) -> float:
    _LOG_2PI = math.log(2 * math.pi)
    log_probs = []
    for m, v, w in zip(means, vars_, weights):
        v = max(v, var_floor)  # variance floor
        lp = -0.5 * _LOG_2PI - 0.5 * math.log(v) - 0.5 * (x - m) ** 2 / v + math.log(max(w, 1e-12))
        log_probs.append(lp)
    from scipy.special import logsumexp

    return float(logsumexp(log_probs))'''
do_replace("V1-eval-gmm-log-prob", OLD_HEADER, NEW_HEADER)

OLD_MAXPROB = '''def get_gmm_max_log_prob(means: List[float], vars_: List[float], weights: List[float]) -> float:
    # Evaluate GMM over a dense 1D grid to find the peak value
    stds = [math.sqrt(max(v, 0.04)) for v in vars_]
    grid_min = min(means) - 3.0 * max(stds)
    grid_max = max(means) + 3.0 * max(stds)
    xs = np.linspace(grid_min, grid_max, 500)
    lps = [eval_gmm_log_prob(x, means, vars_, weights) for x in xs]
    return float(max(lps))'''
NEW_MAXPROB = '''def get_gmm_max_log_prob(
    means: List[float], vars_: List[float], weights: List[float], var_floor: float = 0.04
) -> float:
    # Evaluate GMM over a dense 1D grid to find the peak value
    stds = [math.sqrt(max(v, var_floor)) for v in vars_]
    grid_min = min(means) - 3.0 * max(stds)
    grid_max = max(means) + 3.0 * max(stds)
    xs = np.linspace(grid_min, grid_max, 500)
    lps = [eval_gmm_log_prob(x, means, vars_, weights, var_floor=var_floor) for x in xs]
    return float(max(lps))'''
do_replace("V2-get-gmm-max-log-prob", OLD_MAXPROB, NEW_MAXPROB)

# ---------------------------------------------------------------------------
# V3: Q2 term-collection block -- _vmm_fields -> _gmm_fields, cos-space.
# ---------------------------------------------------------------------------
OLD_Q2 = '''                    q2_key = (coord_geom, dist_class, metal_sym, sub_A, sub_B)
                    fd, resolved_key, _ = scr.stats.dist_trace("Q2", q2_key)

                    fields = _vmm_fields(fd, q2_key, "Q2")
                    if fields is not None:
                        means, kappas, weights = fields
                        means_rad = [math.radians(m) for m in means]
                        lp = eval_vmm_log_prob(q2_angles[p], means_rad, kappas, weights)
                        max_lp = get_vmm_max_log_prob(means_rad, kappas, weights)
                        residual = max(0.0, max_lp - lp)
                    else:
                        residual = 0.0
                        means, kappas, weights = None, None, None

                    terms.append(
                        {
                            "label": f"Q2: D{s_i}–M–D{s_j}",
                            "value": f"{val_deg:.1f}°",
                            "val_num": q2_angles[p],
                            "val_deg": val_deg,
                            "resolved_key": str(resolved_key or "Fallback"),
                            "residual": residual,
                            "unit": "deg",
                            "dist": {
                                "type": "VMM",
                                "means": means_rad,
                                "kappas": kappas,
                                "weights": weights,
                            }
                            if means is not None
                            else None,
                        }
                    )'''
NEW_Q2 = '''                    q2_key = (coord_geom, dist_class, metal_sym, sub_A, sub_B)
                    fd, resolved_key, _ = scr.stats.dist_trace("Q2", q2_key)

                    # 2.7.0: Q2 moved from VMM to GMM-on-cos(angle) -- see
                    # stats/fitter.py's fit_gmm_cosangle() docstring. means/
                    # vars are already dimensionless cos-space values, no
                    # degree->radian conversion needed.
                    fields = _gmm_fields(fd, q2_key, "Q2")
                    if fields is not None:
                        means, vars_, weights = fields
                        val_cos = math.cos(q2_angles[p])
                        lp = eval_gmm_log_prob(
                            val_cos, means, vars_, weights, var_floor=_Q2Q3_GMM_COS_VAR_FLOOR
                        )
                        max_lp = get_gmm_max_log_prob(
                            means, vars_, weights, var_floor=_Q2Q3_GMM_COS_VAR_FLOOR
                        )
                        residual = max(0.0, max_lp - lp)
                    else:
                        residual = 0.0
                        means, vars_, weights = None, None, None

                    terms.append(
                        {
                            "label": f"Q2: D{s_i}–M–D{s_j}",
                            "value": f"{val_deg:.1f}°",
                            "val_num": q2_angles[p],
                            "val_deg": val_deg,
                            "resolved_key": str(resolved_key or "Fallback"),
                            "residual": residual,
                            "unit": "deg",
                            "dist": {
                                "type": "GMM_cos",
                                "means": means,
                                "vars": vars_,
                                "weights": weights,
                            }
                            if means is not None
                            else None,
                        }
                    )'''
do_replace("V3-q2-block", OLD_Q2, NEW_Q2)

# ---------------------------------------------------------------------------
# V4: Q3 term-collection block -- same treatment.
# ---------------------------------------------------------------------------
OLD_Q3 = '''                    key = (subclass, metal_sym, coord_geom)
                    fd, resolved_key, _ = scr.stats.dist_trace("Q3", key)

                    fields = _vmm_fields(fd, key, "Q3")
                    if fields is not None:
                        means, kappas, weights = fields
                        means_rad = [math.radians(m) for m in means]
                        lp = eval_vmm_log_prob(q3_a[q3_idx], means_rad, kappas, weights)
                        max_lp = get_vmm_max_log_prob(means_rad, kappas, weights)
                        residual = max(0.0, max_lp - lp)
                    else:
                        residual = 0.0
                        means, kappas, weights = None, None, None

                    terms.append(
                        {
                            "label": f"Q3: D{s_i} approach",
                            "value": f"{val_deg:.1f}°",
                            "val_num": q3_a[q3_idx],
                            "val_deg": val_deg,
                            "resolved_key": str(resolved_key or "Fallback"),
                            "residual": residual,
                            "unit": "deg",
                            "dist": {
                                "type": "VMM",
                                "means": means_rad,
                                "kappas": kappas,
                                "weights": weights,
                            }
                            if means is not None
                            else None,
                        }
                    )
                    q3_idx += 1'''
NEW_Q3 = '''                    key = (subclass, metal_sym, coord_geom)
                    fd, resolved_key, _ = scr.stats.dist_trace("Q3", key)

                    # 2.7.0: Q3 moved from VMM to GMM-on-cos(angle) -- see
                    # stats/fitter.py's fit_gmm_cosangle() docstring.
                    fields = _gmm_fields(fd, key, "Q3")
                    if fields is not None:
                        means, vars_, weights = fields
                        val_cos = math.cos(q3_a[q3_idx])
                        lp = eval_gmm_log_prob(
                            val_cos, means, vars_, weights, var_floor=_Q2Q3_GMM_COS_VAR_FLOOR
                        )
                        max_lp = get_gmm_max_log_prob(
                            means, vars_, weights, var_floor=_Q2Q3_GMM_COS_VAR_FLOOR
                        )
                        residual = max(0.0, max_lp - lp)
                    else:
                        residual = 0.0
                        means, vars_, weights = None, None, None

                    terms.append(
                        {
                            "label": f"Q3: D{s_i} approach",
                            "value": f"{val_deg:.1f}°",
                            "val_num": q3_a[q3_idx],
                            "val_deg": val_deg,
                            "resolved_key": str(resolved_key or "Fallback"),
                            "residual": residual,
                            "unit": "deg",
                            "dist": {
                                "type": "GMM_cos",
                                "means": means,
                                "vars": vars_,
                                "weights": weights,
                            }
                            if means is not None
                            else None,
                        }
                    )
                    q3_idx += 1'''
do_replace("V4-q3-block", OLD_Q3, NEW_Q3)

# ---------------------------------------------------------------------------
# V5: plotting dispatch -- add a GMM_cos branch alongside the existing
# GMM/VMM branches, so Q2/Q3 diagnostic panels still render (rather than
# silently falling through to a blank axis).
# ---------------------------------------------------------------------------
OLD_DISPATCH = '''                        elif dist["type"] == "VMM":
                            val_rad = t["val_num"]
                            val_deg = t["val_deg"]
                            means_rad = dist["means"]
                            kappas = dist["kappas"]
                            weights = dist["weights"]
                            xs_rad = np.linspace(-math.pi, math.pi, 200)
                            ys = np.array(
                                [
                                    math.exp(eval_vmm_log_prob(x, means_rad, kappas, weights))
                                    for x in xs_rad
                                ]
                            )
                            xs_deg = np.degrees(xs_rad)

                            v_plot = (val_rad + math.pi) % (2 * math.pi) - math.pi

                            ax.plot(xs_deg, ys, color="#00E5FF")
                            ax.axvline(np.degrees(v_plot), color="#FFFFFF", linestyle="--")
                            ax.set_title(
                                f"{t['label']} Res: {t['residual']:.2f} | Key: {t['resolved_key']}",
                                color="#AAD4FF",
                                fontsize=9,
                            )
                    else:'''
NEW_DISPATCH = '''                        elif dist["type"] == "VMM":
                            val_rad = t["val_num"]
                            val_deg = t["val_deg"]
                            means_rad = dist["means"]
                            kappas = dist["kappas"]
                            weights = dist["weights"]
                            xs_rad = np.linspace(-math.pi, math.pi, 200)
                            ys = np.array(
                                [
                                    math.exp(eval_vmm_log_prob(x, means_rad, kappas, weights))
                                    for x in xs_rad
                                ]
                            )
                            xs_deg = np.degrees(xs_rad)

                            v_plot = (val_rad + math.pi) % (2 * math.pi) - math.pi

                            ax.plot(xs_deg, ys, color="#00E5FF")
                            ax.axvline(np.degrees(v_plot), color="#FFFFFF", linestyle="--")
                            ax.set_title(
                                f"{t['label']} Res: {t['residual']:.2f} | Key: {t['resolved_key']}",
                                color="#AAD4FF",
                                fontsize=9,
                            )
                        elif dist["type"] == "GMM_cos":
                            # 2.7.0: Q2/Q3, fit in cos-space (bounded
                            # [0,180] deg via arccos, not circular -- see
                            # fitter.py's fit_gmm_cosangle() docstring).
                            # Plotted here in degrees for readability, with
                            # the Jacobian |d(cos theta)/d theta| = sin(theta)
                            # applied so the curve is a proper density over
                            # degrees rather than a raw cos-space density
                            # stretched onto a degree axis.
                            val_deg = t["val_deg"]
                            means = dist["means"]
                            vars_ = dist["vars"]
                            weights = dist["weights"]
                            xs_deg = np.linspace(0.5, 179.5, 200)
                            xs_rad = np.radians(xs_deg)
                            xs_cos = np.cos(xs_rad)
                            log_p_cos = np.array(
                                [
                                    eval_gmm_log_prob(
                                        x, means, vars_, weights, var_floor=_Q2Q3_GMM_COS_VAR_FLOOR
                                    )
                                    for x in xs_cos
                                ]
                            )
                            ys = np.exp(log_p_cos) * np.sin(xs_rad)
                            ax.plot(xs_deg, ys, color="#00E5FF")
                            ax.axvline(val_deg, color="#FFFFFF", linestyle="--")
                            ax.set_title(
                                f"{t['label']} Res: {t['residual']:.2f} | Key: {t['resolved_key']}",
                                color="#AAD4FF",
                                fontsize=9,
                            )
                    else:'''
do_replace("V5-plot-dispatch", OLD_DISPATCH, NEW_DISPATCH)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
