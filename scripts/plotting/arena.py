"""Arena map plotting for the rat interception study.

Provides clean, publication-quality bird's-eye views of the arena
with rat and ball trajectories, agent simulations, and annotations.

Ported from fig1_detailed_arena.py (executive summary figures).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .plot_utils import apply_style, clean_axes

# ── Constants ──
INTERCEPT_RADIUS = 157.7  # mm, COM-to-ball interception threshold
FPS = 180.0

# Arena outline vertices (mm) — includes bridge/exit corridor
ARENA_VERTS = np.array([
    [-914, 914], [914, 914], [914, 73], [1485, 73], [1648, 121],
    [1648, -121], [1485, -73], [914, -73], [914, -914], [-914, -914],
])

# Colors (matching the executive summary figures)
COLORS = {
    "rat":    "#2563eb",   # blue
    "ball":   "#f59e0b",   # orange/amber
    "chase":  "#dc2626",   # red
    "track":  "#16a34a",   # green
    "oracle": "#7c3aed",   # purple
}


@dataclass
class ArenaConfig:
    """Arena display configuration."""
    x_min: float = -1000.0
    x_max: float = 1750.0
    y_min: float = -1100.0
    y_max: float = 1050.0
    arena_linewidth: float = 2.0
    arena_color: str = "#444444"


def path_length(x, y):
    """Compute cumulative path length from coordinate arrays."""
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def find_intercept_frame(agent_x, agent_y, ball_x, ball_y,
                         radius=INTERCEPT_RADIUS):
    """Find first frame where agent is within radius of ball."""
    dist = np.sqrt((agent_x - ball_x)**2 + (agent_y - ball_y)**2)
    hits = np.where(dist < radius)[0]
    return hits[0] if len(hits) > 0 else None


def draw_arena_boundary(ax, config=None):
    """Draw the arena outline polygon including bridge/exit corridor."""
    if config is None:
        config = ArenaConfig()
    arena_poly = Polygon(ARENA_VERTS, closed=True, fill=False,
                         edgecolor=config.arena_color,
                         linewidth=config.arena_linewidth, linestyle="-")
    ax.add_patch(arena_poly)
    ax.set_xlim(config.x_min, config.x_max)
    ax.set_ylim(config.y_min, config.y_max)


def draw_entity(ax, x, y, ball_x, ball_y, end_frame, color, label,
                marker="o", linestyle="-", linewidth=2, zorder=3):
    """Draw an entity trajectory truncated at end_frame with endpoint markers.

    Returns (label, color, dur_frames, dur_sec, path_len, speed) for annotation.
    """
    sl = slice(0, end_frame + 1)
    ax.plot(x[sl], y[sl], color=color, linewidth=linewidth,
            linestyle=linestyle, label=label, zorder=zorder, alpha=0.9)

    # Endpoint marker
    ex, ey = x[end_frame], y[end_frame]
    if marker == "*":
        ax.plot(ex, ey, marker=marker, color=color, markersize=18,
                zorder=zorder + 2, markeredgecolor="black", markeredgewidth=0.8)
    else:
        ax.plot(ex, ey, marker="o", color=color, markersize=10,
                zorder=zorder + 2, markeredgecolor="black", markeredgewidth=0.8)

    # Ball position at this entity's intercept time
    bx, by = ball_x[end_frame], ball_y[end_frame]
    ax.plot(bx, by, marker="o", color=color, markersize=6,
            zorder=zorder + 1, markeredgecolor="black", markeredgewidth=0.5,
            alpha=0.7)

    # Connecting line from entity endpoint to ball position
    ax.plot([ex, bx], [ey, by], color=color, linewidth=0.8,
            linestyle=":", alpha=0.6, zorder=zorder)

    dur_f = end_frame
    dur_s = dur_f / FPS
    pl = path_length(x[:end_frame + 1], y[:end_frame + 1])
    spd = pl / dur_f if dur_f > 0 else 0
    return (label, color, dur_f, dur_s, pl, spd)


def plot_agent_comparison(ax, df, real_duration, title=None, show_legend=True,
                          show_annotations=True, config=None):
    """Plot a single trial: rat + ball + 3 agents, each truncated at its own intercept.

    Args:
        ax: matplotlib Axes
        df: DataFrame with rel_frame, rat_x/y, ball_x/y, chase_x/y, track_x/y, oracle_x/y
        real_duration: int, rat intercept frame count
        title: optional title string
        show_legend: show legend
        show_annotations: show text annotations below arena
        config: ArenaConfig

    Returns:
        list of (label, color, dur_f, dur_s, path_len, speed) tuples
    """
    if config is None:
        config = ArenaConfig()

    df = df.sort_values("rel_frame").reset_index(drop=True)

    rat_x, rat_y = df.rat_x.values, df.rat_y.values
    ball_x, ball_y = df.ball_x.values, df.ball_y.values
    chase_x, chase_y = df.chase_x.values, df.chase_y.values
    track_x, track_y = df.track_x.values, df.track_y.values
    oracle_x, oracle_y = df.oracle_x.values, df.oracle_y.values

    # Find intercept frames
    rat_end = min(int(real_duration), len(df) - 1)
    chase_end = find_intercept_frame(chase_x, chase_y, ball_x, ball_y) or len(df) - 1
    track_end = find_intercept_frame(track_x, track_y, ball_x, ball_y) or len(df) - 1
    oracle_end = find_intercept_frame(oracle_x, oracle_y, ball_x, ball_y) or len(df) - 1

    # Ball shown up to the latest termination
    ball_end = min(max(rat_end, chase_end, track_end, oracle_end), len(df) - 1)

    # Draw arena boundary
    draw_arena_boundary(ax, config)

    # Ball trajectory
    ax.plot(ball_x[:ball_end + 1], ball_y[:ball_end + 1],
            color=COLORS["ball"], linewidth=3, alpha=0.7, label="Ball", zorder=1)

    # Draw entities
    annotations = []
    annotations.append(draw_entity(
        ax, rat_x, rat_y, ball_x, ball_y, rat_end,
        COLORS["rat"], "Rat", marker="*", linewidth=3, zorder=5))
    annotations.append(draw_entity(
        ax, chase_x, chase_y, ball_x, ball_y, chase_end,
        COLORS["chase"], "Chase", zorder=4))
    annotations.append(draw_entity(
        ax, track_x, track_y, ball_x, ball_y, track_end,
        COLORS["track"], "Track", zorder=4))
    annotations.append(draw_entity(
        ax, oracle_x, oracle_y, ball_x, ball_y, oracle_end,
        COLORS["oracle"], "Oracle", linestyle="--", zorder=4))

    # Clean axes
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=20, fontweight="bold")
    if show_legend:
        ax.legend(loc="upper left", fontsize=14, framealpha=0.9)

    # Text annotations below arena in 2x2 grid
    if show_annotations:
        y_top = -960
        y_bot = y_top - 90
        x_left = -950
        x_right = 400
        positions = [(x_left, y_top), (x_right, y_top),
                     (x_left, y_bot), (x_right, y_bot)]
        for j, (lbl, clr, dur_f, dur_s, pl, spd) in enumerate(annotations):
            txt = f"{lbl}: {dur_f}f ({dur_s:.2f}s), {pl:.0f}mm, {spd:.1f}mm/f"
            ax.text(positions[j][0], positions[j][1], txt, fontsize=11,
                    color=clr, fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=clr, alpha=0.9, linewidth=1.2))

    return annotations


def plot_pn_trial(ax, df, N, tau, R2, title=None, config=None):
    """Plot a single PN trial with heading arrows and LOS lines.

    Args:
        ax: matplotlib Axes
        df: DataFrame with rel_frame, rat_x/y, ball_x/y, rat_heading_angle
        N, tau, R2: PN fit parameters
        title: optional title
        config: ArenaConfig (uses lighter defaults for PN plots)
    """
    if config is None:
        config = ArenaConfig(arena_color="#CCCCCC", arena_linewidth=1.5)

    df = df.sort_values("rel_frame").reset_index(drop=True)
    rx, ry = df.rat_x.values, df.rat_y.values
    bx, by = df.ball_x.values, df.ball_y.values

    draw_arena_boundary(ax, config)

    # Ball trajectory
    ax.plot(bx, by, color=COLORS["ball"], linewidth=1.5, zorder=3)

    # Rat trajectory
    ax.plot(rx, ry, color=COLORS["rat"], linewidth=1.5, zorder=4)

    # Start/end markers
    ax.plot(rx[0], ry[0], marker="o", color="#2ca02c", markersize=8,
            zorder=6, markeredgecolor="black", markeredgewidth=0.5)
    ax.plot(rx[-1], ry[-1], marker="*", color="#d62728", markersize=12,
            zorder=6, markeredgecolor="black", markeredgewidth=0.5)

    # Heading arrows
    if "rat_heading_angle" in df.columns:
        heading = df.rat_heading_angle.values
        arrow_step = max(1, len(df) // 8)
        for j in range(0, len(df), arrow_step):
            if np.isnan(heading[j]):
                continue
            dx = 80 * np.cos(heading[j])
            dy = 80 * np.sin(heading[j])
            ax.annotate("", xy=(rx[j] + dx, ry[j] + dy),
                        xytext=(rx[j], ry[j]),
                        arrowprops=dict(arrowstyle="->", color=COLORS["rat"],
                                        lw=0.8, alpha=0.5), zorder=5)

    # LOS lines
    los_step = max(1, len(df) // 6)
    for j in range(0, len(df), los_step):
        ax.plot([rx[j], bx[j]], [ry[j], by[j]],
                color="#888888", linewidth=0.5, alpha=0.4, zorder=1)

    # Clean axes
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=7, pad=3)


# ── Convenience: multi-panel grid ──

def plot_trial_grid(trials, ncols=3, figsize=None, show_agents=True, **kwargs):
    """Plot multiple agent-comparison trials in a grid. Legacy compatibility wrapper."""
    # This is kept for backward compat but the new preferred API
    # is plot_agent_comparison() called per-axes.
    pass
