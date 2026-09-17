"""
analyze_edta_trace.py
======================
Offline, numpy/matplotlib-only analysis of a solve_trace dump produced by
diagnose_edta_trace.py. Deliberately has NO torch/rdkit dependency so it
can run in any environment that has the raw .npz + .json.

For each trial in the beam, classifies its trajectory as one of:

  converged           -- plateaued at some point (matches engine's own
                          plateau criterion: score range over some
                          8-step window < 0.02), recomputed independently
                          here as a sanity check against kept_converged.
  still-improving      -- score is still trending up in the final window;
                          truncated by T, not actually stuck. Points at
                          "needs more steps" / slower LR schedule.
  oscillating          -- score bounces (many sign changes in its step-to-
                          step delta) in the final window without settling.
                          Points at "needs a smarter optimizer/schedule"
                          (e.g. LR too high, or two restraints fighting).
  regressed-from-peak  -- score reached a good value well before T then
                          fell and stayed lower. Points at instability /
                          a restraint whose pressure grows over the run
                          (check mu_shape) overpowering the chemistry term.
  plateaued-bad        -- flat (small range) but NOT caught by the 8-step
                          window check anywhere, or caught only briefly
                          then knocked loose -- i.e. stuck at a bad score,
                          contradicts "still moving" framing. Points at
                          initialization / basin problem, not an
                          optimizer-schedule problem.
  ambiguous            -- doesn't cleanly match the above.

Also reports the augmented-Lagrangian dual variable mu_shape's trend
(saturating vs still climbing at T) as separate evidence about whether the
shape constraint itself is being satisfied.

Usage
-----
    python analyze_edta_trace.py
    python analyze_edta_trace.py --dump edta_trace_dump.npz --meta edta_trace_meta.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def recompute_plateau(score_col: np.ndarray, window: int = 8, atol: float = 0.02) -> tuple[bool, list[int]]:
    """Reproduce JointIKSolver's own ever_plateaued check exactly: checked
    only every `window` steps (t % window == window-1), range over that
    trailing window < atol. Returns (ever_plateaued, [steps where it fired])."""
    T = len(score_col)
    fired = []
    for t in range(window - 1, T, 1):
        if t % window == window - 1:
            w = score_col[t - window + 1 : t + 1]
            if (w.max() - w.min()) < atol:
                fired.append(t)
    return (len(fired) > 0), fired


def linfit_slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=np.float64)
    if len(y) < 2 or np.allclose(y, y[0]):
        return 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def classify_trial(score_col: np.ndarray, plateau_atol: float = 0.02, tail_frac: float = 0.15):
    """
    Classification is driven primarily by a MACRO-smoothed view of the score
    trajectory (smoothing window ~ T/12), not the raw per-step values or a
    light denoise. Rationale (see handoff §3): "a trajectory oscillating
    between two attractors would never register as converged even near a
    good structure" -- that kind of oscillation is a property of the macro
    trend (the trajectory's centre of mass swinging between two basins), not
    single-step jitter. A fine-grained sign-flip count on raw deltas can't
    tell a genuinely flat-but-noisy trace from a slowly-oscillating one
    (both can have ~50% raw sign flips); the macro curve can, because a
    flat trace stays macro-flat while a between-attractors trace shows the
    macro curve itself swinging with a range well above plateau_atol.
    """
    T = len(score_col)
    tail_n = max(10, int(T * tail_frac))
    macro_window = max(15, T // 12)

    sm_macro = moving_average(score_col, macro_window)
    tail_macro = sm_macro[-tail_n:]
    macro_range = float(tail_macro.max() - tail_macro.min())
    macro_slope = linfit_slope(tail_macro)
    macro_deltas = np.diff(tail_macro)
    macro_nonzero = macro_deltas[macro_deltas != 0]
    macro_sign_changes = int(np.sum(np.diff(np.sign(macro_nonzero)) != 0)) if len(macro_nonzero) > 1 else 0
    osc_frac_macro = macro_sign_changes / max(1, len(macro_nonzero) - 1)

    # Fine-grained (near-raw) jitter, reported as a supporting diagnostic
    # only -- NOT the primary classifier (see docstring above).
    fine = moving_average(score_col, max(1, macro_window // 5))
    tail_fine = fine[-tail_n:]
    fine_deltas = np.diff(tail_fine)
    fine_nonzero = fine_deltas[fine_deltas != 0]
    fine_sign_changes = int(np.sum(np.diff(np.sign(fine_nonzero)) != 0)) if len(fine_nonzero) > 1 else 0
    osc_frac_fine = fine_sign_changes / max(1, len(fine_nonzero) - 1)

    best_step = int(np.argmax(sm_macro))
    best_score = float(sm_macro[best_step])
    final_score = float(sm_macro[-1])

    ever_plat, plat_steps = recompute_plateau(score_col, atol=plateau_atol)

    # Decision tree, in priority order.
    label = "ambiguous"
    if ever_plat:
        label = "converged"
    elif macro_slope > 0.25 * plateau_atol:
        label = "still-improving"
    elif macro_range > 2 * plateau_atol and osc_frac_macro >= 0.3:
        label = "oscillating"
    elif (best_step < T - tail_n) and (best_score - final_score) > 2 * plateau_atol:
        label = "regressed-from-peak"
    elif macro_range < 2 * plateau_atol:
        label = "plateaued-bad"

    return {
        "label": label,
        "best_step": best_step,
        "best_score": best_score,
        "final_score": final_score,
        "macro_range": macro_range,
        "macro_slope": macro_slope,
        "osc_frac_macro": osc_frac_macro,
        "osc_frac_fine": osc_frac_fine,
        "ever_plateaued_recomputed": ever_plat,
        "plateau_first_step": plat_steps[0] if plat_steps else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default="edta_trace_dump.npz")
    ap.add_argument("--meta", default="edta_trace_meta.json")
    ap.add_argument("--plateau-atol", type=float, default=0.02)
    ap.add_argument("--plot-top", type=int, default=4,
                     help="number of representative trials to plot")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    data = np.load(args.dump)
    with open(args.meta) as f:
        meta = json.load(f)

    score = data["score"]          # (T, B)
    grad_norm = data["grad_norm"]  # (T, B)
    anchor_loss = data["anchor_loss"]
    mu_shape = data["mu_shape"]
    sym_penalty = data["sym_penalty"]
    sym_angle_penalty = data["sym_angle_penalty"]
    T, B = score.shape
    print(f"loaded trace: T={T} steps, B={B} trials")
    print(f"run params: {meta.get('steps')=} {meta.get('beam_width')=} "
          f"{meta.get('seed')=} {meta.get('lr')=}")
    print(f"kept conformers converged (engine): {meta.get('n_converged')}/{meta.get('n_kept')}")

    kept_by_trial = {c["trial_idx"]: c for c in meta.get("kept_conformers", [])}

    rows = []
    for b in range(B):
        c = classify_trial(score[:, b], plateau_atol=args.plateau_atol)
        c["trial_idx"] = b
        c["kept"] = b in kept_by_trial
        c["engine_converged"] = kept_by_trial.get(b, {}).get("converged")
        c["mu_shape_final"] = float(mu_shape[-1, b])
        c["mu_shape_tail_slope"] = linfit_slope(mu_shape[-max(10, T // 10):, b])
        c["anchor_final"] = float(anchor_loss[-1, b])
        c["grad_norm_final"] = float(grad_norm[-1, b])
        rows.append(c)

    label_counts: dict[str, int] = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    print("\n=== classification counts across all B trials ===")
    for lbl, n in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {lbl:>20s}: {n:3d}/{B}")

    mismatch = [r for r in rows if r["kept"] and (r["engine_converged"] != r["ever_plateaued_recomputed"])]
    if mismatch:
        print(f"\nWARNING: {len(mismatch)} trials where recomputed plateau flag "
              f"disagrees with engine's own converged flag -- check window "
              f"alignment / atol match before trusting labels.")
    else:
        print("\n(sanity check OK: recomputed plateau flag matches engine's "
              "converged flag on all kept trials)")

    print("\n=== per-trial detail (sorted by final score, desc) ===")
    header = (f"{'trial':>5} {'kept':>4} {'label':>18} {'best_score':>10} "
              f"{'final_score':>11} {'best_step':>9} {'macro_slope':>11} "
              f"{'macro_rng':>9} {'osc_fine':>8} {'mu_final':>9} {'mu_slope':>9} {'anchor_final':>12}")
    print(header)
    for r in sorted(rows, key=lambda r: -r["final_score"]):
        print(f"{r['trial_idx']:>5} {'Y' if r['kept'] else 'n':>4} {r['label']:>18} "
              f"{r['best_score']:>10.4f} {r['final_score']:>11.4f} {r['best_step']:>9} "
              f"{r['macro_slope']:>11.5f} {r['macro_range']:>9.4f} {r['osc_frac_fine']:>8.2f} "
              f"{r['mu_shape_final']:>9.3f} {r['mu_shape_tail_slope']:>9.5f} {r['anchor_final']:>12.5f}")

    # mu_shape trend summary (constraint-pressure diagnostic)
    still_climbing = [r for r in rows if not r["ever_plateaued_recomputed"] and r["mu_shape_tail_slope"] > 1e-3]
    print(f"\n=== AL dual variable (mu_shape) trend, non-converged trials ===")
    print(f"  {len(still_climbing)}/{sum(1 for r in rows if not r['ever_plateaued_recomputed'])} "
          f"non-converged trials have mu_shape STILL RISING in the tail window "
          f"(constraint pressure never resolves within T steps).")

    # Plots: worst non-converged, median non-converged, best non-converged, and
    # one converged trial if any exist, for contrast.
    non_conv = [r for r in rows if not r["ever_plateaued_recomputed"]]
    conv = [r for r in rows if r["ever_plateaued_recomputed"]]
    to_plot = []
    if non_conv:
        by_final = sorted(non_conv, key=lambda r: r["final_score"])
        to_plot.append(("worst_nonconverged", by_final[0]))
        to_plot.append(("median_nonconverged", by_final[len(by_final) // 2]))
        to_plot.append(("best_nonconverged", by_final[-1]))
    if conv:
        to_plot.append(("converged_baseline", max(conv, key=lambda r: r["final_score"])))
    to_plot = to_plot[: args.plot_top]

    if to_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(to_plot), 4, figsize=(18, 4 * len(to_plot)), squeeze=False)
        steps = np.arange(T)
        for row_i, (tag, r) in enumerate(to_plot):
            b = r["trial_idx"]
            ax = axes[row_i]
            ax[0].plot(steps, score[:, b]); ax[0].set_title(f"[{tag}] trial {b} ({r['label']}) score")
            ax[1].plot(steps, grad_norm[:, b]); ax[1].set_title("grad_norm")
            ax[1].set_yscale("log")
            ax[2].plot(steps, anchor_loss[:, b]); ax[2].set_title("anchor_loss (shape_dev)")
            ax[3].plot(steps, mu_shape[:, b]); ax[3].set_title("mu_shape (AL dual var)")
            for a in ax:
                a.set_xlabel("step")
        fig.tight_layout()
        out_path = f"{args.outdir}/edta_trace_diagnosis.png"
        fig.savefig(out_path, dpi=110)
        print(f"\nSaved diagnostic plot: {out_path}")

    print("\n=== verdict inputs (for the handoff's three hypotheses) ===")
    print(f"  still-improving count : {label_counts.get('still-improving', 0)}  "
          f"(favors: needs more steps / slower schedule)")
    print(f"  oscillating count     : {label_counts.get('oscillating', 0)}  "
          f"(favors: needs a smarter optimizer/schedule)")
    print(f"  regressed-from-peak   : {label_counts.get('regressed-from-peak', 0)}  "
          f"(favors: restraint pressure destabilizing an initially-good pose)")
    print(f"  plateaued-bad count   : {label_counts.get('plateaued-bad', 0)}  "
          f"(favors: needs better initialization / stuck in a bad basin)")


if __name__ == "__main__":
    main()
