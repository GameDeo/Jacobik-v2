"""
visualizer.py
=============
Unified interactive visualizer for the Jacobik conformer ensemble engine.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from jacobik.vis.render_3d import (
    CPK,
    _cpk,
    _render_complex_with_opacity,
    show_ensemble_structs,
    show_geometry_mappings,
    show_overlay,
)
from jacobik.vis.stats_plots import (
    _Q2Q3_GMM_COS_VAR_FLOOR,
    eval_gmm_log_prob,
    eval_vmm_log_prob,
    get_gmm_max_log_prob,
    get_vmm_max_log_prob,
    show_ensemble_stats,
)
import ipywidgets as widgets
from IPython.display import display

# ---------------------------------------------------------------------------
# Visualizer class definition
# ---------------------------------------------------------------------------
class Visualizer:
    """
    Unified public-facing visualizer for Jacobik conformer ensembles.
    """

    def __init__(
        self,
        result: Optional[Any] = None,
        scorer: Optional[Any] = None,
        show_hydrogen: bool = False,
    ) -> None:
        self.result = result
        self.scorer = scorer
        self.show_hydrogen = show_hydrogen

        # Auto-extract scorer and solve_trace from EnsembleResult if possible
        if result is not None:
            if scorer is None and hasattr(result, "scorer"):
                self.scorer = result.scorer

    def _get_result(self, result: Optional[Any]) -> Any:
        r = result if result is not None else self.result
        if r is None:
            raise ValueError("No EnsembleResult provided. Pass it to the method or constructor.")
        return r

    def _get_scorer(self, scorer: Optional[Any]) -> Any:
        s = scorer if scorer is not None else self.scorer
        if s is None:
            raise ValueError("No Scorer provided. Pass it to the method or constructor.")
        return s

    # -----------------------------------------------------------------------
    # 1. view_ensemble_structs()
    # -----------------------------------------------------------------------
    def view_ensemble_structs(
        self, result: Optional[Any] = None, width: int = 620, height: int = 460
    ) -> None:
        """Interactive 3D structure viewer to step through conformers."""
        show_ensemble_structs(self._get_result(result), show_hydrogen=self.show_hydrogen, width=width, height=height)

    def view_ensemble_stats(
        self, result: Optional[Any] = None, scorer: Optional[Any] = None
    ) -> None:
        """Interactive tool to view FittedDist satisfaction details."""
        show_ensemble_stats(self._get_result(result), self._get_scorer(scorer))

    # 3. view_ranking()
    # -----------------------------------------------------------------------
    def view_ranking(self, result: Optional[Any] = None) -> None:
        """
        Plots a bar chart of the ensemble scores, ranked highest to lowest.
        """
        res = self._get_result(result)
        all_confs = res.conformers
        if not all_confs:
            print("EnsembleResult is empty.")
            return

        ranks = [c.rank for c in all_confs]
        scores = [c.score for c in all_confs]
        conv_list = [c.converged for c in all_confs]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0F0F1A")
        ax.set_facecolor("#0F0F1A")

        # Color code bars by convergence status
        colors = ["#00FF88" if conv else "#FF6B6B" for conv in conv_list]

        bars = ax.bar(ranks, scores, color=colors, edgecolor="#333355", width=0.6)
        ax.set_xlabel("Conformer Rank", color="#888888", fontsize=9)
        ax.set_ylabel("Score (Log Probability)", color="#888888", fontsize=9)
        ax.set_title(
            "Ensemble Conformer Rankings (Green = Converged, Red = Failed)",
            color="#AAD4FF",
            fontsize=11,
            pad=12,
        )
        ax.set_xticks(ranks)
        ax.tick_params(colors="#666666", labelsize=8)
        ax.grid(axis="y", color="#22223B", linestyle="--")

        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        # Text labels on top of bars
        for bar, score in zip(bars, scores):
            yval = bar.get_height()
            offset = 0.5 if yval >= 0 else -1.5
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + offset,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                color="#C8D8E8",
                fontsize=7,
            )

        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------

    def view_overlay(
        self, result: Optional[Any] = None, width: int = 620, height: int = 460
    ) -> None:
        """Interactive viewer showing all conformers overlayed ghostly transparent."""
        show_overlay(self._get_result(result), show_hydrogen=self.show_hydrogen, width=width, height=height)

    # 5. view_solve()
    # -----------------------------------------------------------------------
    def view_solve(self, result: Optional[Any] = None) -> None:
        """
        Interactive solver trajectory dashboard showing residuals, scores,
        donor errors, and gradient norms as a function of steps.
        """
        res = self._get_result(result)
        trace = getattr(res, "solve_trace", None)
        if trace is None or not trace:
            print("Solve trace is not available on this result.")
            return

        all_confs = res.conformers
        if not all_confs:
            print("EnsembleResult is empty.")
            return

        # trace contains arrays of shape (T, B) where B is the number of trials
        T, B = trace["residual"].shape
        state = {"idx": 0}

        # Widgets
        btn_prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px"))
        btn_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px"))
        hdr_html = widgets.HTML(value="")
        plot_out = widgets.Output()

        def _render():
            idx = state["idx"]
            active_conf = all_confs[idx]
            trial_idx = active_conf.trial_idx

            hdr_html.value = (
                f"<b style='color:#AAD4FF'>Showing Solve Trajectory for Rank {active_conf.rank}</b>"
                f"&nbsp;&nbsp;<span style='color:#888'>(original trial element index = {trial_idx})</span>"
            )
            btn_prev.disabled = idx == 0
            btn_next.disabled = idx == len(all_confs) - 1

            with plot_out:
                plot_out.clear_output(wait=True)
                fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor="#0F0F1A")
                steps = np.arange(T)

                # Helper to plot background trials and highlight the active one
                def plot_metric(ax, data_key, label_y, title):
                    ax.set_facecolor("#0F0F1A")
                    data = trace[data_key]  # (T, B)
                    # Plot all other trials in background
                    for b in range(B):
                        if b != trial_idx:
                            ax.plot(steps, data[:, b], color="#22223B", alpha=0.4, linewidth=0.8)
                    # Highlight active trial in cyan
                    ax.plot(
                        steps,
                        data[:, trial_idx],
                        color="#00E5FF",
                        linewidth=1.8,
                        label=f"Rank {active_conf.rank}",
                    )
                    ax.set_title(title, color="#AAD4FF", fontsize=9, pad=5)
                    ax.set_xlabel("Iteration step", color="#666666", fontsize=7)
                    ax.set_ylabel(label_y, color="#666666", fontsize=7)
                    ax.tick_params(colors="#666666", labelsize=7)
                    ax.grid(color="#1A1A2E", linestyle=":")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#333355")
                    ax.legend(
                        fontsize=7, facecolor="#1A1A2E", labelcolor="#C8D8E8", edgecolor="#333355"
                    )

                plot_metric(
                    axes[0, 0], "residual", "Residual Loss", "Joint Residual (-score + λ*anchor)"
                )
                plot_metric(
                    axes[0, 1],
                    "score",
                    "Log Prob (Score)",
                    "Chemical Statistics Score (higher = better)",
                )
                plot_metric(
                    axes[1, 0],
                    "worst_error",
                    "Max Donor Error (Å)",
                    "Worst-Ligand Target Position Placement Error",
                )
                plot_metric(
                    axes[1, 1], "grad_norm", "Gradient Norm", "Parameter Gradient Magnitude"
                )

                fig.suptitle(
                    f"Optimization Trajectory Dashboard (CSD: {active_conf.complex.csd_code})",
                    color="#AAD4FF",
                    fontsize=11,
                    y=0.98,
                )
                plt.tight_layout()
                plt.show()

        def on_prev(_):
            state["idx"] = max(0, state["idx"] - 1)
            _render()

        def on_next(_):
            state["idx"] = min(len(all_confs) - 1, state["idx"] + 1)
            _render()

        btn_prev.on_click(on_prev)
        btn_next.on_click(on_next)

        nav_bar = widgets.HBox(
            [btn_prev, btn_next, hdr_html], layout=widgets.Layout(align_items="center", gap="8px")
        )
        ui = widgets.VBox(
            [nav_bar, plot_out],
            layout=widgets.Layout(
                border="1px solid #333", padding="10px", background_color="#0F0F1A"
            ),
        )
        display(ui)
        _render()

    # -----------------------------------------------------------------------

    def geometry_mappings(self, width: int = 600, height: int = 460) -> None:
        """Interactive tool to step through and view index order mappings of ideal coordination geometries."""
        show_geometry_mappings(width=width, height=height)
