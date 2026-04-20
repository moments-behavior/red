#pragma once

// bout_pose_overlay.h — Draw 3D prediction skeleton on camera views
//
// Projects 3D keypoints from the PredictionReader to 2D using the camera
// calibration (perspective or telecentric DLT), then draws skeleton edges
// and keypoint circles on the ImPlot subplot.

#include "../traj_reader.h"
#include "../app_context.h"
#include "../red_math.h"
#include "../skeleton.h"
#include "imgui.h"
#include "implot.h"

// Draw prediction overlay for one camera view.
// pred_data: pointer to elements_per_frame floats for current frame
//            (layout: kp0_x, kp0_y, kp0_z, kp0_conf, kp1_x, ...)
//            coordinates in mm (calibration frame)
// cam_idx: which camera
inline void DrawPredictionOverlay(
    const float *pred_data,
    int cam_idx,
    const AppContext &ctx,
    float conf_threshold = 0.3f,
    float alpha = 0.8f)
{
    if (!pred_data || !ctx.scene || cam_idx < 0 ||
        cam_idx >= (int)ctx.pm.camera_params.size())
        return;

    const auto &cam = ctx.pm.camera_params[cam_idx];
    const auto &skeleton = ctx.skeleton;
    int n_kp = skeleton.num_nodes;
    if (n_kp <= 0 || n_kp > 50) return;

    int img_h = ctx.scene->image_height[cam_idx];

    // Project each 3D keypoint to 2D pixel coordinates
    std::vector<float> px(n_kp), py(n_kp);
    std::vector<bool> visible(n_kp, false);

    for (int k = 0; k < n_kp; ++k) {
        float x_mm = pred_data[k * 4 + 0];
        float y_mm = pred_data[k * 4 + 1];
        float z_mm = pred_data[k * 4 + 2];
        float conf = pred_data[k * 4 + 3];

        if (conf < conf_threshold || (x_mm == 0 && y_mm == 0 && z_mm == 0))
            continue;

        Eigen::Vector3d pt(x_mm, y_mm, z_mm);
        Eigen::Vector2d uv;

        if (cam.telecentric) {
            uv = red_math::projectPointTelecentric(pt, cam.projection_mat, cam.k, cam.dist_coeffs);
        } else {
            uv = red_math::projectPointR(pt, cam.r, cam.tvec, cam.k, cam.dist_coeffs);
        }

        if (std::isnan(uv.x()) || std::isnan(uv.y())) continue;

        // ImPlot Y: 0 at bottom (RED convention)
        px[k] = (float)uv.x();
        py[k] = (float)(img_h - uv.y());
        visible[k] = true;
    }

    ImDrawList *dl = ImPlot::GetPlotDrawList();
    ImU32 edge_color = ImGui::GetColorU32(ImVec4(0.2f, 0.8f, 1.0f, alpha * 0.6f));

    // Draw skeleton edges
    for (int e = 0; e < skeleton.num_edges; ++e) {
        int a = skeleton.edges[e].x;
        int b = skeleton.edges[e].y;
        if (a < 0 || b < 0 || a >= n_kp || b >= n_kp) continue;
        if (!visible[a] || !visible[b]) continue;

        ImVec2 pa = ImPlot::PlotToPixels(ImPlotPoint(px[a], py[a]));
        ImVec2 pb = ImPlot::PlotToPixels(ImPlotPoint(px[b], py[b]));
        dl->AddLine(pa, pb, edge_color, 1.5f);
    }

    // Draw keypoints (colored by confidence)
    for (int k = 0; k < n_kp; ++k) {
        if (!visible[k]) continue;
        float conf = pred_data[k * 4 + 3];

        float t = std::clamp((conf - 0.3f) / 0.7f, 0.0f, 1.0f);
        ImVec4 col(1.0f - t * 0.6f, 0.3f + t * 0.7f, 0.2f, alpha);
        ImU32 kp_color = ImGui::GetColorU32(col);

        ImVec2 p = ImPlot::PlotToPixels(ImPlotPoint(px[k], py[k]));
        dl->AddCircleFilled(p, 3.0f, kp_color);
    }
}

