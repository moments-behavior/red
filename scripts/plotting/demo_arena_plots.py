#!/usr/bin/env python3
"""Demo: regenerate arena map plots from the executive summary using the shared library.

Reads example_trajectories.csv and produces clean arena maps matching Figure 1a style
(white background, no axes) instead of the old default matplotlib style.
"""

import sys
from pathlib import Path

# Add parent to path so we can import the plotting module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from plotting import plot_trial_arena, plot_trial_grid, save_figure, ArenaConfig
from plotting.plot_utils import apply_style, AGENT_COLORS, new_figure
from plotting.arena import plot_arena, plot_trajectory
import matplotlib.pyplot as plt

DATA_DIR = Path("/Users/johnsonr/datasets/rat_results_031826/claude_notes/modelingMarch25")
OUT_DIR = DATA_DIR / "figures_v2"


def load_example_trajectories():
    """Load the example trajectory CSV and group by trial."""
    df = pd.read_csv(DATA_DIR / "example_trajectories.csv")
    trials = []
    for trial_id, grp in df.groupby("trial_id"):
        g = grp.sort_values("rel_frame")
        trials.append({
            "trial_id": int(trial_id),
            "animal": g["animal"].iloc[0],
            "data": {
                "ball_x": g["ball_x"].values,
                "ball_y": g["ball_y"].values,
                "rat_x": g["rat_x"].values,
                "rat_y": g["rat_y"].values,
                "chase_x": g["chase_x"].values,
                "chase_y": g["chase_y"].values,
                "track_x": g["track_x"].values,
                "track_y": g["track_y"].values,
                "oracle_x": g["oracle_x"].values,
                "oracle_y": g["oracle_y"].values,
            },
            "title": f"{g['animal'].iloc[0]} (trial {trial_id})",
        })
    return trials


def demo_fig1a():
    """Regenerate Fig 1a: one example trial per rat (5-panel grid)."""
    trials = load_example_trajectories()

    # Pick one trial per animal (first occurrence)
    seen = set()
    selected = []
    for t in trials:
        if t["animal"] not in seen:
            seen.add(t["animal"])
            selected.append(t)

    fig, axes = plot_trial_grid(selected, ncols=3, show_agents=True)
    fig.suptitle("", fontsize=1)  # clear any auto-title

    out = OUT_DIR / "fig1a_arena_trajectories"
    save_figure(fig, out, formats=("png", "pdf", "svg"),
                data_files=[DATA_DIR / "example_trajectories.csv"],
                script=__file__,
                provenance={"figure": "Fig 1a", "description": "Arena trajectories, one per rat"})
    print(f"Saved: {out}.png/pdf/svg")
    plt.close(fig)


def demo_single_trial():
    """Demo: single clean arena map for one trial."""
    trials = load_example_trajectories()
    t = trials[0]

    fig, ax = plot_trial_arena(
        t["data"],
        title=f"Trial {t['trial_id']} ({t['animal']})",
        show_agents=True,
        show_legend=True,
        show_markers=True,
    )

    out = OUT_DIR / f"demo_trial_{t['trial_id']}"
    save_figure(fig, out, formats=("png",),
                data_files=[DATA_DIR / "example_trajectories.csv"],
                script=__file__)
    print(f"Saved: {out}.png")
    plt.close(fig)


def demo_minimal():
    """Demo: minimal arena map — just rat and ball, no agents, no legend."""
    trials = load_example_trajectories()
    t = trials[2]  # pick a different trial

    fig, ax = plot_trial_arena(
        t["data"],
        show_agents=False,
        show_legend=False,
        show_markers=True,
    )

    out = OUT_DIR / f"demo_minimal_{t['trial_id']}"
    save_figure(fig, out, formats=("png",), script=__file__)
    print(f"Saved: {out}.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_DIR}/\n")

    demo_single_trial()
    demo_minimal()
    demo_fig1a()

    print("\nDone. Compare with the old figures in the executive summary PDF.")
