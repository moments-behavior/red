"""RED publication plotting library.

Shared utilities for generating publication-quality figures
for the Johnson Lab rat interception study.
"""

from .plot_utils import (
    new_figure, save_figure, clean_axes, pub_spines,
    SINGLE_COL, DOUBLE_COL, FULL_PAGE,
    OUTCOME_COLORS, PHASE_COLORS, AGENT_COLORS, ANIMAL_COLORS,
)
from .arena import plot_arena, plot_trial_arena, plot_trial_grid, ArenaConfig
