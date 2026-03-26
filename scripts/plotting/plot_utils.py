"""Shared plotting utilities for RED publication figures.

Provides consistent styling, figure creation, and export with provenance tracking.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import subprocess
import datetime

# ── Figure dimensions (inches) ──
SINGLE_COL = 3.4    # single journal column
DOUBLE_COL = 7.0    # double column / full width
FULL_PAGE = 9.0     # full page width

STYLE_PATH = Path(__file__).parent / "paper.mplstyle"

# ── Color palettes ──
OUTCOME_COLORS = {
    "Complete":    "#4878A8",
    "Catch/NoRet": "#E8A838",
    "Approach":    "#68A870",
    "NoApproach":  "#C86858",
    "NoRelease":   "#9878B8",
    "NoTrack":     "#A0A0A0",
}

PHASE_COLORS = {
    "Out":   "#4878A8",
    "Catch": "#E8A838",
    "In":    "#68A870",
}

AGENT_COLORS = {
    "ball":    "#E8A838",   # orange/gold
    "rat":     "#4878A8",   # steel blue
    "chase":   "#C86858",   # dusty red
    "track":   "#68A870",   # sage green
    "oracle":  "#9878B8",   # muted purple
}

ANIMAL_COLORS = {
    "captain":     "#4878A8",
    "emilie":      "#E8A838",
    "heisenberg":  "#68A870",
    "mario":       "#C86858",
    "remy":        "#9878B8",
}


def apply_style():
    """Apply the RED paper matplotlib style."""
    plt.style.use(str(STYLE_PATH))


def new_figure(width=SINGLE_COL, height=None, aspect=0.75, nrows=1, ncols=1,
               **subplot_kw):
    """Create a figure at publication size with paper style applied.

    Returns (fig, ax) for single subplot, (fig, axes) for grids.
    """
    apply_style()
    if height is None:
        height = width * aspect
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **subplot_kw)
    return fig, axes


def clean_axes(ax):
    """Remove all axes, spines, ticks for clean map/trajectory plots."""
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")


def pub_spines(ax, keep=("bottom", "left")):
    """Keep only specified spines, remove others."""
    for spine in ax.spines:
        ax.spines[spine].set_visible(spine in keep)


def save_figure(fig, path, formats=("png", "pdf"), dpi=300,
                provenance=None, data_files=None, script=None):
    """Save figure in multiple formats with optional provenance sidecar.

    Args:
        fig: matplotlib Figure
        path: base path (without extension)
        formats: tuple of format strings
        dpi: resolution for raster formats
        provenance: dict of extra provenance metadata
        data_files: list of data file paths used to generate this figure
        script: path to the generating script
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), format=fmt, dpi=dpi,
                    facecolor="white", bbox_inches="tight", pad_inches=0.02)

    # Write provenance sidecar
    prov = {
        "created": datetime.datetime.now().isoformat(),
        "formats": list(formats),
        "dpi": dpi,
    }
    if script:
        prov["script"] = str(script)
    if data_files:
        prov["data_files"] = [str(f) for f in data_files]
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent), stderr=subprocess.DEVNULL
        ).strip().decode()
        prov["git_commit"] = git_hash
    except Exception:
        pass
    if provenance:
        prov.update(provenance)

    with open(path.with_suffix(".provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
