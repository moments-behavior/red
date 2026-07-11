#pragma once
// prediction_overlay.h — draw JARVIS predictions (from the separate, mmap'd
// prediction store) as a READ-ONLY overlay on a camera view, without them ever
// entering the AnnotationMap / Labeling Tool.
//
// Predictions are stored as 3D only (x,y,z,conf per keypoint); this reprojects
// each point to 2D exactly as reprojection() in gui_keypoints.h does (telecentric
// vs pinhole, y flipped by image height), then draws skeleton edges + points via
// the ImPlot draw list. Points are colored by confidence (red = low, green =
// high) so predictions read as visually distinct from the interactive,
// node-colored manual labels drawn by gui_plot_keypoints().

#include "implot.h"
#include "render.h"
#include "skeleton.h"
#include "camera.h"
#include "red_math.h"
#include "gui/gui_keypoints.h"  // is_in_camera_fov

#include <algorithm>
#include <cmath>
#include <vector>

// pose: pointer to 4*num_nodes float32 (x,y,z,conf per keypoint), as returned by
// predstore::PredictionReader::frame(). Draws into the current ImPlot plot.
inline void gui_plot_prediction_overlay(
    const float *pose, int view_idx,
    SkeletonContext *skeleton,
    const std::vector<CameraParams> &camera_params,
    RenderScene *scene,
    float conf_threshold = 0.0f,
    float alpha = 0.9f)
{
    if (!pose || skeleton == nullptr || camera_params.empty()) return;
    if (view_idx >= (int)scene->num_cams) return;
    if (view_idx >= (int)camera_params.size()) return;

    const bool telecentric = camera_params[view_idx].telecentric;
    const double W = (double)scene->image_width[view_idx];
    const double H = (double)scene->image_height[view_idx];
    const int nn = (int)skeleton->num_nodes;

    struct P2 { double x = 0, y = 0; float conf = 0; bool valid = false; };
    std::vector<P2> pts(nn);

    for (int node = 0; node < nn; ++node) {
        float x3 = pose[node * 4 + 0], y3 = pose[node * 4 + 1];
        float z3 = pose[node * 4 + 2], conf = pose[node * 4 + 3];
        if (std::isnan(x3) || std::isnan(y3) || std::isnan(z3)) continue;
        if (conf < conf_threshold) continue;

        Eigen::Vector3d p3d(x3, y3, z3);
        double x = 0, y = 0;
        bool ok = false;
        if (telecentric) {
            auto rp = red_math::projectPointTelecentric(
                p3d, camera_params[view_idx].projection_mat,
                camera_params[view_idx].k, camera_params[view_idx].dist_coeffs,
                camera_params[view_idx].dist_center);
            x = rp(0);
            y = H - rp(1);
            ok = (x > 0 && x < W && y > 0 && y < H);
        } else if (is_in_camera_fov(p3d, camera_params[view_idx].r,
                                    camera_params[view_idx].tvec,
                                    camera_params[view_idx].k,
                                    (int)W, (int)H)) {
            auto rp = red_math::projectPointR(
                p3d, camera_params[view_idx].r, camera_params[view_idx].tvec,
                camera_params[view_idx].k, camera_params[view_idx].dist_coeffs);
            x = rp(0);
            y = H - rp(1);
            ok = (x > 0 && x < W && y > 0 && y < H);
        }
        if (ok) pts[node] = {x, y, conf, true};
    }

    ImDrawList *dl = ImPlot::GetPlotDrawList();
    const int a_edge = (int)(std::clamp(alpha, 0.f, 1.f) * 160.f);
    const int a_pt = (int)(std::clamp(alpha, 0.f, 1.f) * 255.f);

    // Edges first, so points render on top.
    for (u32 e = 0; e < skeleton->num_edges; ++e) {
        auto [a, b] = skeleton->edges[e];
        if (a < (u32)nn && b < (u32)nn && pts[a].valid && pts[b].valid) {
            ImVec2 pa = ImPlot::PlotToPixels(pts[a].x, pts[a].y);
            ImVec2 pb = ImPlot::PlotToPixels(pts[b].x, pts[b].y);
            dl->AddLine(pa, pb, IM_COL32(190, 190, 190, a_edge), 1.5f);
        }
    }
    // Confidence-colored points (red low -> green high).
    for (int node = 0; node < nn; ++node) {
        if (!pts[node].valid) continue;
        ImVec2 pp = ImPlot::PlotToPixels(pts[node].x, pts[node].y);
        float c = std::clamp(pts[node].conf, 0.f, 1.f);
        ImU32 col = IM_COL32((int)((1.f - c) * 255.f), (int)(c * 255.f), 40, a_pt);
        dl->AddCircleFilled(pp, 4.0f, col);
        dl->AddCircle(pp, 4.0f, IM_COL32(15, 15, 15, a_pt), 0, 1.0f);
    }
}
