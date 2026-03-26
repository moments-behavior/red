#!/usr/bin/env python3
"""Generate publication-quality figure PDFs.

Recreates Fig 1a (agent comparison) and Fig 3a (PN examples)
from the executive summary using the shared arena plotting library.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plot_utils import apply_style, save_figure
from arena import (plot_agent_comparison, plot_pn_trial, ArenaConfig,
                   COLORS, FPS)

DATA_DIR = Path("/Users/johnsonr/datasets/rat_results_031826/claude_notes/modelingMarch25")
HERE = Path(__file__).parent

ANIMAL_ORDER = ["captain", "emilie", "heisenberg", "mario", "remy"]


def make_fig1a():
    """Fig 1a: 5 agent-comparison arena maps (one per rat)."""
    traj = pd.read_csv(DATA_DIR / "example_trajectories.csv")
    results = pd.read_csv(DATA_DIR / "trial_results.csv")
    dur_lookup = dict(zip(results.trial_id, results.real_duration))

    # One trial per animal
    trials_by_animal = {}
    for _, row in traj.drop_duplicates("trial_id").iterrows():
        trials_by_animal[row.animal] = row.trial_id

    fig, axes = plt.subplots(2, 3, figsize=(30, 20), facecolor="white",
                             gridspec_kw={"hspace": 0.08})
    axes = axes.flatten()

    for idx, animal in enumerate(ANIMAL_ORDER):
        ax = axes[idx]
        tid = trials_by_animal[animal]
        df = traj[traj.trial_id == tid]
        real_dur = int(dur_lookup[tid])
        dur_s = real_dur / FPS

        plot_agent_comparison(
            ax, df, real_dur,
            title=f"{animal.capitalize()} (trial {tid}, {dur_s:.2f}s)",
        )

    # Hide unused subplot
    axes[5].set_visible(False)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig


def make_fig3a():
    """Fig 3a: 6 best-R2 PN trials with heading arrows and LOS lines."""
    features = pd.read_csv(DATA_DIR / "out_phase_features.csv")
    pn = pd.read_csv(DATA_DIR / "pn_results_per_trial.csv")
    top = pn.nlargest(6, "R2")

    apply_style()
    fig = plt.figure(figsize=(11, 9))

    for i, (_, row) in enumerate(top.iterrows()):
        tid = int(row.trial_id)
        tf = features[features.trial_id == tid].sort_values("rel_frame")
        if len(tf) < 10:
            continue

        ax = fig.add_subplot(3, 2, i + 1)
        plot_pn_trial(
            ax, tf, row.N, int(row.tau), row.R2,
            title=f"Trial {tid} ({row.animal}, day {int(row.session_day)})\n"
                  f"N={row.N:.2f}, \u03c4={int(row.tau)}f, R\u00b2={row.R2:.3f}",
        )

    fig.suptitle("Fig. 3a: PN Example Trials (highest R\u00b2)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


if __name__ == "__main__":
    print(f"Output: {HERE}\n")

    # Combined PDF
    pdf_path = HERE / "fig_test.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        print("Fig 1a: Agent comparison (5 rats)...")
        fig1 = make_fig1a()
        pdf.savefig(fig1, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig1)

        print("Fig 3a: PN example trials (6 best R2)...")
        fig3 = make_fig3a()
        pdf.savefig(fig3, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig3)
    print(f"  -> {pdf_path}")

    # Individual PDFs
    fig1 = make_fig1a()
    save_figure(fig1, HERE / "fig1a_arena_trajectories", formats=("pdf",),
                data_files=[DATA_DIR / "example_trajectories.csv",
                            DATA_DIR / "trial_results.csv"],
                script=__file__)
    plt.close(fig1)
    print(f"  -> fig1a_arena_trajectories.pdf")

    fig3 = make_fig3a()
    save_figure(fig3, HERE / "fig3a_pn_example_trials", formats=("pdf",),
                data_files=[DATA_DIR / "out_phase_features.csv",
                            DATA_DIR / "pn_results_per_trial.csv"],
                script=__file__)
    plt.close(fig3)
    print(f"  -> fig3a_pn_example_trials.pdf")

    print("\nDone.")
