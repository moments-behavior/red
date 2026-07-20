#pragma once
// bout_filter_preview.h — live reprojection of the Bout Filter's floor-Z /
// Y-wall / X-wall threshold planes into a camera view, so the user can see
// where "floor" or "wall" sits relative to the tracked animal while dragging
// the Bout Filter sliders. Mirrors gui_plot_prediction_overlay's structure
// (prediction_overlay.h): reprojects 3D points per camera and draws via the
// current ImPlot draw list. Read-only, UI-only — never touches
// boutfilter::compute().
//
// floor_z_threshold / y_wall_min / y_wall_max / x_wall_margin are mm, in the
// same arena frame as boutfilter::Inputs (raw .rpred store value / profile
// scale). CameraParams calibration operates in raw, unscaled store units, so
// every mm-frame corner is multiplied by the active profile's scale before
// reprojection.

#include "implot.h"
#include "render.h"
#include "camera.h"
#include "gui/bout_filter_window.h"   // BoutFilterState, BoutFilterProfile
#include "gui/gui_keypoints.h"        // reproject_3d_to_cam

#include <Eigen/Core>
#include <array>
#include <vector>

namespace {

// Draws one mm-frame quad's 4 edges reprojected into camera view_idx; skips
// an edge (not the whole quad) if either endpoint fails to reproject —
// matches how gui_plot_prediction_overlay/gui_plot_keypoints degrade
// skeleton edges near the FOV boundary (only draw when both endpoints valid).
inline void bout_filter_draw_quad_mm(
    const std::array<Eigen::Vector3d, 4> &corners_mm, double scale,
    const CameraParams &cp, int W, int H, ImU32 color, float thickness) {
    std::array<bool, 4> ok{};
    std::array<double, 4> sx{}, sy{};
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3d raw = corners_mm[i] * scale;
        ok[i] = reproject_3d_to_cam(raw, cp, W, H, sx[i], sy[i]);
    }
    ImDrawList *dl = ImPlot::GetPlotDrawList();
    for (int e = 0; e < 4; ++e) {
        int a = e, b = (e + 1) % 4;
        if (!ok[a] || !ok[b]) continue;
        dl->AddLine(ImPlot::PlotToPixels(sx[a], sy[a]),
                    ImPlot::PlotToPixels(sx[b], sy[b]), color, thickness);
    }
}

}  // namespace

inline void bout_filter_draw_wall_preview(
    const BoutFilterState &st, int view_idx,
    const std::vector<CameraParams> &camera_params, RenderScene *scene) {
    if (!st.inputs_valid) return;
    if (st.profile_idx < 0 || st.profile_idx >= (int)st.profiles.size()) return;
    if (!(st.show_floor_preview || st.show_ywall_preview || st.show_xwall_preview)) return;
    if (view_idx >= (int)scene->num_cams || view_idx >= (int)camera_params.size()) return;

    const BoutFilterProfile &prof = st.profiles[st.profile_idx];
    const double scale = prof.scale != 0 ? (double)prof.scale : 1.0;
    const double ax = (double)prof.arena_x_mm;
    const double ay = (double)prof.arena_y_mm;
    const double h  = (double)st.preview_height_mm;
    const int W = (int)scene->image_width[view_idx];
    const int H = (int)scene->image_height[view_idx];
    const CameraParams &cp = camera_params[view_idx];
    const float thickness = 1.5f;

    const ImU32 COL_FLOOR = IM_COL32(255, 170, 40, 200);  // amber
    const ImU32 COL_YWALL = IM_COL32(60, 140, 255, 200);  // blue
    const ImU32 COL_XWALL = IM_COL32(230, 90, 230, 200);  // magenta

    if (st.show_floor_preview) {
        double z = (double)st.params.floor_z_threshold;
        std::array<Eigen::Vector3d, 4> q = {
            Eigen::Vector3d(0, 0, z), Eigen::Vector3d(ax, 0, z),
            Eigen::Vector3d(ax, ay, z), Eigen::Vector3d(0, ay, z)};
        bout_filter_draw_quad_mm(q, scale, cp, W, H, COL_FLOOR, thickness);
    }
    if (st.show_ywall_preview) {
        for (double y : {(double)st.params.y_wall_min, (double)st.params.y_wall_max}) {
            std::array<Eigen::Vector3d, 4> q = {
                Eigen::Vector3d(0, y, 0), Eigen::Vector3d(ax, y, 0),
                Eigen::Vector3d(ax, y, h), Eigen::Vector3d(0, y, h)};
            bout_filter_draw_quad_mm(q, scale, cp, W, H, COL_YWALL, thickness);
        }
    }
    if (st.show_xwall_preview) {
        double m = (double)st.params.x_wall_margin;
        for (double x : {m, ax - m}) {
            std::array<Eigen::Vector3d, 4> q = {
                Eigen::Vector3d(x, 0, 0), Eigen::Vector3d(x, ay, 0),
                Eigen::Vector3d(x, ay, h), Eigen::Vector3d(x, 0, h)};
            bout_filter_draw_quad_mm(q, scale, cp, W, H, COL_XWALL, thickness);
        }
    }
}
