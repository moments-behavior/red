#pragma once
#include "implot.h"
#include "skeleton.h"
#include "user_settings.h"

// Keypoint colormaps.
//
// node_colors are the per-keypoint colors shown in every camera view and in
// the Keypoints table. Historically they were a fixed HSV rainbow. We now let
// the user pick from a set of colormaps (the built-in ImPlot maps, which are
// the same matplotlib/MATLAB maps: Viridis, Plasma, Jet, Spectral, ...). The
// choice is a single global (g_keypoint_colormap) so every skeleton-load path
// recolors consistently without threading the value through setup_project.
//
// Convention: colormap < 0 means the legacy HSV rainbow (the historical
// default, kept so existing projects look unchanged until the user opts in).
// colormap >= 0 is an ImPlotColormap_ index sampled continuously across nodes.
static constexpr int KEYPOINT_COLORMAP_RAINBOW = -1;

// Color for keypoint `index` of `num_nodes`, under `colormap`.
inline ImVec4 keypoint_node_color(int index, int num_nodes, int colormap) {
    if (num_nodes <= 0)
        return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    if (colormap < 0) {
        // Legacy rainbow: evenly spaced hues around the wheel.
        return (ImVec4)ImColor::HSV(index / (float)num_nodes, 1.0f, 1.0f);
    }
    // Spread nodes across the full [0,1] range of the colormap. Guard the
    // single-node case (avoid divide-by-zero) by sampling the midpoint.
    float t = (num_nodes == 1) ? 0.5f : index / (float)(num_nodes - 1);
    return ImPlot::SampleColormap(t, colormap);
}

// Overwrite skeleton->node_colors from the given colormap. Safe to call any
// time after the ImPlot context exists (created at startup).
inline void apply_keypoint_colormap(SkeletonContext &skeleton, int colormap) {
    for (int i = 0; i < (int)skeleton.node_colors.size(); i++)
        skeleton.node_colors[i] =
            keypoint_node_color(i, skeleton.num_nodes, colormap);
}

// The user-chosen highlight color for the active (selected) keypoint. Falls
// back to white if unset/malformed. Alpha is applied by the caller.
inline ImVec4 active_keypoint_color(const UserSettings &s) {
    if (s.active_keypoint_color.size() >= 3)
        return ImVec4(s.active_keypoint_color[0], s.active_keypoint_color[1],
                      s.active_keypoint_color[2], 1.0f);
    return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
}
