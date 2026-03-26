"""Arena map plotting for the rat interception study.

Provides clean, publication-quality bird's-eye views of the arena
with rat and ball trajectories, agent simulations, and annotations.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .plot_utils import (
    apply_style, clean_axes, AGENT_COLORS, save_figure,
)


@dataclass
class ArenaConfig:
    """Arena geometry (all units in mm)."""
    x_min: float = -1000.0
    x_max: float = 1500.0
    y_min: float = -1000.0
    y_max: float = 1000.0
    # Table boundary (the actual play surface)
    table_x_min: float = -914.0
    table_x_max: float = 914.0
    table_y_min: float = -914.0
    table_y_max: float = 914.0
    # Ramp entry point (ball release)
    ramp_x: float = -500.0
    ramp_y: float = 920.0
    # Bridge/exit zone
    bridge_x_min: float = 880.0
    bridge_x_max: float = 980.0
    bridge_y_min: float = -73.0
    bridge_y_max: float = 73.0
    # Visual style
    table_color: str = "#E8E8E8"     # light gray fill
    table_edge: str = "#999999"      # gray border
    table_linewidth: float = 1.0
    show_table_fill: bool = True


def plot_arena(ax, config: Optional[ArenaConfig] = None):
    """Draw the arena boundary on an axes with clean styling.

    Args:
        ax: matplotlib Axes
        config: ArenaConfig (uses defaults if None)
    """
    if config is None:
        config = ArenaConfig()

    # Table surface
    table = patches.Rectangle(
        (config.table_x_min, config.table_y_min),
        config.table_x_max - config.table_x_min,
        config.table_y_max - config.table_y_min,
        linewidth=config.table_linewidth,
        edgecolor=config.table_edge,
        facecolor=config.table_color if config.show_table_fill else "none",
        zorder=0,
    )
    ax.add_patch(table)

    # Set view limits with padding
    ax.set_xlim(config.x_min, config.x_max)
    ax.set_ylim(config.y_min, config.y_max)

    # Clean axes — no ticks, no labels, white background
    clean_axes(ax)
    ax.set_facecolor("white")


def plot_trajectory(ax, x, y, color, label=None, linewidth=1.2, alpha=1.0,
                    linestyle="-", zorder=2, marker_start=None, marker_end=None):
    """Plot a single trajectory on the arena.

    Args:
        ax: matplotlib Axes
        x, y: coordinate arrays
        color: line color
        label: legend label
        linewidth, alpha, linestyle: line style
        marker_start: dict with marker kwargs for start point (e.g., {'marker': 'o', 'color': 'green'})
        marker_end: dict with marker kwargs for end point
    """
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha,
            linestyle=linestyle, label=label, zorder=zorder)

    if marker_start and len(x) > 0:
        defaults = {"marker": "o", "s": 25, "zorder": zorder + 1, "edgecolors": "none"}
        defaults.update(marker_start)
        c = defaults.pop("color", color)
        ax.scatter([x[0]], [y[0]], color=c, **defaults)

    if marker_end and len(x) > 0:
        defaults = {"marker": "*", "s": 40, "zorder": zorder + 1, "edgecolors": "none"}
        defaults.update(marker_end)
        c = defaults.pop("color", color)
        ax.scatter([x[-1]], [y[-1]], color=c, **defaults)


def plot_trial_arena(trial_data, title=None, config=None, figsize=None,
                     show_agents=True, show_legend=True, show_markers=True,
                     agent_list=("chase", "track", "oracle")):
    """Create a complete arena map figure for one trial.

    Args:
        trial_data: dict with keys 'ball_x', 'ball_y', 'rat_x', 'rat_y',
                    and optionally 'chase_x', 'chase_y', 'track_x', 'track_y',
                    'oracle_x', 'oracle_y'. Values are arrays.
        title: optional title string (e.g., "Trial 1069 (remy, day 1)")
        config: ArenaConfig
        figsize: (width, height) in inches. Default: (3.4, 2.8)
        show_agents: whether to plot synthetic agent trajectories
        show_legend: whether to show legend
        show_markers: whether to show start/end markers
        agent_list: which agents to plot

    Returns:
        (fig, ax)
    """
    apply_style()
    if figsize is None:
        figsize = (3.4, 2.8)
    fig, ax = plt.subplots(figsize=figsize)

    plot_arena(ax, config)

    start_kw = {"marker": "o", "color": "#2ca02c", "s": 30} if show_markers else None
    end_kw = {"marker": "*", "color": "#d62728", "s": 50} if show_markers else None

    # Ball trajectory
    plot_trajectory(ax, trial_data["ball_x"], trial_data["ball_y"],
                    color=AGENT_COLORS["ball"], label="Ball", linewidth=1.5,
                    zorder=3)

    # Rat trajectory
    plot_trajectory(ax, trial_data["rat_x"], trial_data["rat_y"],
                    color=AGENT_COLORS["rat"], label="Rat", linewidth=1.5,
                    marker_start=start_kw, marker_end=end_kw, zorder=4)

    # Synthetic agents
    if show_agents:
        agent_styles = {
            "chase":  {"linestyle": "-",  "linewidth": 0.9, "alpha": 0.7},
            "track":  {"linestyle": "-",  "linewidth": 0.9, "alpha": 0.7},
            "oracle": {"linestyle": "--", "linewidth": 0.9, "alpha": 0.7},
        }
        agent_labels = {"chase": "Chase", "track": "Track", "oracle": "Oracle"}
        for agent in agent_list:
            xk, yk = f"{agent}_x", f"{agent}_y"
            if xk in trial_data and yk in trial_data:
                style = agent_styles.get(agent, {})
                plot_trajectory(ax, trial_data[xk], trial_data[yk],
                                color=AGENT_COLORS.get(agent, "#888888"),
                                label=agent_labels.get(agent, agent),
                                **style)

    if show_legend:
        ax.legend(loc="upper left", fontsize=6, framealpha=0.8,
                  edgecolor="none", facecolor="white")

    if title:
        ax.set_title(title, fontsize=8, pad=4)

    return fig, ax


def plot_trial_grid(trials, ncols=3, figsize=None, config=None, **kwargs):
    """Plot multiple trials in a grid layout.

    Args:
        trials: list of dicts, each with trial_data + optional 'title' key
        ncols: columns in grid
        figsize: override figure size
        config: ArenaConfig
        **kwargs: passed to plot_trial_arena internals

    Returns:
        (fig, axes)
    """
    apply_style()
    ntrials = len(trials)
    nrows = (ntrials + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 3.0, nrows * 2.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, trial in enumerate(trials):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        plot_arena(ax, config)

        data = trial.get("data", trial)
        title = trial.get("title", None)

        start_kw = {"marker": "o", "color": "#2ca02c", "s": 20}
        end_kw = {"marker": "*", "color": "#d62728", "s": 35}

        plot_trajectory(ax, data["ball_x"], data["ball_y"],
                        color=AGENT_COLORS["ball"], label="Ball", linewidth=1.2, zorder=3)
        plot_trajectory(ax, data["rat_x"], data["rat_y"],
                        color=AGENT_COLORS["rat"], label="Rat", linewidth=1.2,
                        marker_start=start_kw, marker_end=end_kw, zorder=4)

        show_agents = kwargs.get("show_agents", True)
        if show_agents:
            for agent in ("chase", "track", "oracle"):
                xk, yk = f"{agent}_x", f"{agent}_y"
                if xk in data:
                    ls = "--" if agent == "oracle" else "-"
                    plot_trajectory(ax, data[xk], data[yk],
                                    color=AGENT_COLORS.get(agent, "#888"),
                                    linewidth=0.8, alpha=0.7, linestyle=ls)

        if title:
            ax.set_title(title, fontsize=7, pad=2)

    # Hide unused axes
    for i in range(ntrials, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    return fig, axes
