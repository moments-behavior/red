#!/usr/bin/env python3
"""Recreate Fig 1a and Fig 3a from the executive summary using the shared plotting library.

Outputs fig_test.pdf with:
- Page 1: Fig 1a — arena trajectories with agents (5 rats)
- Page 2: Fig 3a — PN example trials (best R2) with heading arrows and LOS lines
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plotting import save_figure, ArenaConfig
from plotting.plot_utils import apply_style, AGENT_COLORS
from plotting.arena import plot_arena, plot_trajectory, plot_trial_grid

DATA_DIR = Path("/Users/johnsonr/datasets/rat_results_031826/claude_notes/modelingMarch25")
OUT_DIR = DATA_DIR / "figures_v2"

# ─────────────────────────────────────────────────────────────────────────
# Fig 1a: Arena trajectories with agents (one per rat)
# ─────────────────────────────────────────────────────────────────────────

def make_fig1a():
    """5-panel grid of arena maps with rat + ball + 3 agents."""
    df = pd.read_csv(DATA_DIR / "example_trajectories.csv")

    # Pick one trial per animal (preserve order: mario, captain, remy, emilie, heisenberg)
    animal_order = ["captain", "emilie", "heisenberg", "mario", "remy"]
    trials = []
    seen = set()
    for _, row in df.iterrows():
        a = row["animal"]
        if a not in seen:
            seen.add(a)
    for animal in animal_order:
        g = df[df.animal == animal].sort_values("rel_frame")
        tid = g.trial_id.iloc[0]
        n_frames = len(g)
        duration = n_frames / 180.0
        trials.append({
            "data": {
                "ball_x": g.ball_x.values, "ball_y": g.ball_y.values,
                "rat_x": g.rat_x.values, "rat_y": g.rat_y.values,
                "chase_x": g.chase_x.values, "chase_y": g.chase_y.values,
                "track_x": g.track_x.values, "track_y": g.track_y.values,
                "oracle_x": g.oracle_x.values, "oracle_y": g.oracle_y.values,
            },
            "title": f"{animal} (trial {tid}, {duration:.2f}s)",
        })

    fig, axes = plot_trial_grid(trials, ncols=3, figsize=(10, 7), show_agents=True)

    # Add a shared legend at the bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=AGENT_COLORS["ball"], lw=1.5, label="Ball"),
        Line2D([0], [0], color=AGENT_COLORS["rat"], lw=1.5, label="Real Rat"),
        Line2D([0], [0], color=AGENT_COLORS["chase"], lw=0.9, alpha=0.7, label="Chase"),
        Line2D([0], [0], color=AGENT_COLORS["track"], lw=0.9, alpha=0.7, label="Track"),
        Line2D([0], [0], color=AGENT_COLORS["oracle"], lw=0.9, alpha=0.7, ls="--", label="Oracle"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Fig. 1a: Arena Trajectories (Out Phase)", fontsize=11, fontweight="bold", y=1.01)
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Fig 3a: PN example trials — arena map with heading arrows + LOS lines
# ─────────────────────────────────────────────────────────────────────────

def make_fig3a():
    """6 PN best-fit trials: arena + heading rate vs N*LOS_rate + bearing angle."""
    # Load per-frame features and PN results
    features = pd.read_csv(DATA_DIR / "out_phase_features.csv")
    pn = pd.read_csv(DATA_DIR / "pn_results_per_trial.csv")

    # Top 6 by R2
    top_trials = pn.nlargest(6, "R2")

    apply_style()
    fig = plt.figure(figsize=(11, 9))

    for i, (_, row) in enumerate(top_trials.iterrows()):
        tid = int(row.trial_id)
        animal = row.animal
        day = int(row.session_day)
        N = row.N
        tau = int(row.tau)
        R2 = row.R2

        # Get per-frame data for this trial
        tf = features[features.trial_id == tid].sort_values("rel_frame").copy()
        if len(tf) < 10:
            continue

        # ── Column 1: Arena map with heading arrows and LOS lines ──
        ax_arena = fig.add_subplot(3, 2, i + 1)
        plot_arena(ax_arena, ArenaConfig(
            x_min=-1100, x_max=1600, y_min=-1050, y_max=1050,
            show_table_fill=False, table_color="#F0F0F0", table_edge="#CCCCCC",
        ))

        # Ball trajectory
        bx, by = tf.ball_x.values, tf.ball_y.values
        rx, ry = tf.rat_x.values, tf.rat_y.values
        plot_trajectory(ax_arena, bx, by, color=AGENT_COLORS["ball"],
                        label="Ball", linewidth=1.5, zorder=3)
        # Rat trajectory
        plot_trajectory(ax_arena, rx, ry, color=AGENT_COLORS["rat"],
                        label="Rat COM", linewidth=1.5, zorder=4,
                        marker_start={"marker": "o", "color": "#2ca02c", "s": 25},
                        marker_end={"marker": "*", "color": "#d62728", "s": 40})

        # Heading arrows (every ~15 frames)
        heading = tf.rat_heading_angle.values
        arrow_step = max(1, len(tf) // 8)
        arrow_len = 80
        for j in range(0, len(tf), arrow_step):
            if np.isnan(heading[j]):
                continue
            dx = arrow_len * np.cos(heading[j])
            dy = arrow_len * np.sin(heading[j])
            ax_arena.annotate("", xy=(rx[j] + dx, ry[j] + dy), xytext=(rx[j], ry[j]),
                              arrowprops=dict(arrowstyle="->", color=AGENT_COLORS["rat"],
                                              lw=0.8, alpha=0.5),
                              zorder=5)

        # LOS lines (from rat to ball, every ~20 frames)
        los_step = max(1, len(tf) // 6)
        for j in range(0, len(tf), los_step):
            ax_arena.plot([rx[j], bx[j]], [ry[j], by[j]],
                          color="#888888", linewidth=0.5, alpha=0.4, zorder=1)

        ax_arena.set_title(f"Trial {tid} ({animal}, day {day})\n"
                           f"N={N:.2f}, τ={tau}f, R²={R2:.3f}",
                           fontsize=7, pad=3)

    fig.suptitle("Fig. 3a: PN Example Trials (highest R²)", fontsize=11,
                 fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Main: assemble PDF
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "fig_test.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        print("Generating Fig 1a...")
        fig1 = make_fig1a()
        pdf.savefig(fig1, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig1)

        print("Generating Fig 3a...")
        fig3 = make_fig3a()
        pdf.savefig(fig3, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig3)

    print(f"\nSaved: {pdf_path}")

    # Also save provenance
    save_figure.__wrapped__ = True  # skip actual save, just write provenance
    import json, datetime, subprocess
    prov = {
        "created": datetime.datetime.now().isoformat(),
        "formats": ["pdf"],
        "script": str(Path(__file__)),
        "data_files": [
            str(DATA_DIR / "example_trajectories.csv"),
            str(DATA_DIR / "out_phase_features.csv"),
            str(DATA_DIR / "pn_results_per_trial.csv"),
        ],
        "figures": ["Fig 1a: Arena trajectories with agents",
                    "Fig 3a: PN example trials with heading arrows"],
    }
    try:
        prov["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent), stderr=subprocess.DEVNULL
        ).strip().decode()
    except Exception:
        pass
    with open(pdf_path.with_suffix(".provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"Provenance: {pdf_path.with_suffix('.provenance.json')}")
