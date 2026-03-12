#pragma once
// tele_viewer_window.h — Interactive 3D visualization of telecentric DLT calibration.
// Optical axis lines, image plane billboards, 3D landmarks with error coloring.
// Uses ImPlot3D for immediate-mode 3D rendering inside an ImGui window.

#include "imgui.h"
#include "implot3d.h"
#include "telecentric_dlt.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct TeleViewerState {
    bool show = false;
    float plane_scale = 5.0f;     // image plane billboard half-size
    float axis_length = 10.0f;    // optical axis line half-length
    bool show_landmarks = true;
    bool show_cameras = true;
    bool show_labels = true;
    bool color_by_error = true;
    int hovered_camera = -1;
    int selected_camera = -1;     // -1 = all

    // Data (set externally before drawing)
    const TelecentricDLT::DLTResult *dlt_result = nullptr;
    std::vector<Eigen::Vector3d> landmarks_3d;  // known 3D points
};

inline void DrawTeleViewerWindow(TeleViewerState &state) {
    if (!state.show || !state.dlt_result || !state.dlt_result->success) return;

    const auto &res = *state.dlt_result;
    int nc = (int)res.cameras.size();

    ImGui::SetNextWindowSize(ImVec2(800, 650), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Telecentric 3D Viewer", &state.show)) {
        ImGui::End(); return;
    }

    // ── Controls ──
    ImVec4 label_col(0.5f, 0.7f, 1.0f, 1.0f);
    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("##plane", &state.plane_scale, 1.0f, 20.0f, "%.1f");
    ImGui::SameLine(); ImGui::TextColored(label_col, "Plane");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("##axis", &state.axis_length, 1.0f, 50.0f, "%.1f");
    ImGui::SameLine(); ImGui::TextColored(label_col, "Axis");

    ImGui::PushStyleColor(ImGuiCol_Text, label_col);
    ImGui::SameLine(); ImGui::Checkbox("Landmarks", &state.show_landmarks);
    ImGui::SameLine(); ImGui::Checkbox("Cameras", &state.show_cameras);
    ImGui::SameLine(); ImGui::Checkbox("Labels", &state.show_labels);
    ImGui::SameLine(); ImGui::Checkbox("Color/Err", &state.color_by_error);
    ImGui::PopStyleColor();

    // Camera selector
    {
        const char *preview = (state.selected_camera < 0) ? "All Cameras" :
            (state.selected_camera < nc ? res.cameras[state.selected_camera].serial.c_str() : "?");
        ImGui::SetNextItemWidth(180);
        if (ImGui::BeginCombo("Camera", preview)) {
            if (ImGui::Selectable("All Cameras", state.selected_camera < 0))
                state.selected_camera = -1;
            for (int c = 0; c < nc; c++) {
                char label[128];
                snprintf(label, sizeof(label), "%s (%.3f px, %d pts)",
                    res.cameras[c].serial.c_str(), res.cameras[c].final_rmse(), res.cameras[c].num_points);
                if (ImGui::Selectable(label, state.selected_camera == c))
                    state.selected_camera = c;
            }
            ImGui::EndCombo();
        }
    }

    // Summary line
    {
        double rmse_display = res.mean_rmse;
        if (state.selected_camera >= 0 && state.selected_camera < nc) {
            const auto &cam = res.cameras[state.selected_camera];
            rmse_display = cam.final_rmse();
            ImGui::Text("%s: RMSE=%.3f px | sx=%.2f sy=%.2f | %d pts | %s",
                cam.serial.c_str(), rmse_display, cam.sx, cam.sy,
                cam.num_points, TelecentricDLT::method_name(res.method));
        } else {
            ImGui::Text("Mean RMSE: %.3f px | %d cameras | %d landmarks | %s",
                rmse_display, nc, (int)state.landmarks_3d.size(),
                TelecentricDLT::method_name(res.method));
        }
    }

    // ── Compute centroid of 3D landmarks ──
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    if (!state.landmarks_3d.empty()) {
        for (const auto &pt : state.landmarks_3d) centroid += pt;
        centroid /= (double)state.landmarks_3d.size();
    }

    // Compute scene extent for world axes
    float scene_extent = 0;
    for (const auto &pt : state.landmarks_3d)
        scene_extent = std::max(scene_extent, (float)(pt - centroid).norm());
    scene_extent = std::max(scene_extent, state.axis_length * 2.0f);
    float world_axis_len = std::max(5.0f, scene_extent * 0.5f);

    // Find RMSE range for error coloring
    double rmse_min = 1e9, rmse_max = 0;
    for (int c = 0; c < nc; c++) {
        double rmse = res.cameras[c].final_rmse();
        rmse_min = std::min(rmse_min, rmse);
        rmse_max = std::max(rmse_max, rmse);
    }
    if (rmse_max <= rmse_min) rmse_max = rmse_min + 1.0;

    auto avail = ImGui::GetContentRegionAvail();
    ImPlot3DFlags plot_flags = ImPlot3DFlags_Equal | ImPlot3DFlags_NoClip | ImPlot3DFlags_CanvasOnly;

    ImPlot3D::PushStyleColor(ImPlot3DCol_PlotBg, ImVec4(0, 0, 0, 0));
    ImPlot3D::PushStyleColor(ImPlot3DCol_FrameBg, ImVec4(0, 0, 0, 0));
    ImPlot3D::PushStyleColor(ImPlot3DCol_PlotBorder, ImVec4(0, 0, 0, 0));

    if (ImPlot3D::BeginPlot("##tele3d", avail, plot_flags)) {
        ImPlot3DAxisFlags ax_flags = ImPlot3DAxisFlags_NoDecorations | ImPlot3DAxisFlags_NoTickMarks;
        ImPlot3D::SetupAxis(ImAxis3D_X, nullptr, ax_flags);
        ImPlot3D::SetupAxis(ImAxis3D_Y, nullptr, ax_flags);
        ImPlot3D::SetupAxis(ImAxis3D_Z, nullptr, ax_flags);

        state.hovered_camera = -1;
        bool single_cam = (state.selected_camera >= 0);

        // ── Camera optical axes and image planes ──
        if (state.show_cameras) {
            for (int c = 0; c < nc; c++) {
                const auto &cam = res.cameras[c];
                bool is_selected = (c == state.selected_camera);
                bool is_dimmed = single_cam && !is_selected;

                // Recover rotation columns
                Eigen::Vector3d r1 = cam.R.col(0);
                Eigen::Vector3d r2 = cam.R.col(1);
                Eigen::Vector3d r3 = cam.R.col(2); // optical axis direction

                // Anchor: pseudo-inverse solution A^+ * (p0 - t) where p0 = [0;0]
                // A^+ = A^T * (A * A^T)^{-1}
                Eigen::Matrix2d AAt = cam.A * cam.A.transpose();
                if (std::abs(AAt.determinant()) < 1e-20) continue; // skip degenerate camera
                Eigen::Matrix<double, 3, 2> Aplus = cam.A.transpose() * AAt.inverse();
                Eigen::Vector3d X0 = Aplus * (-cam.t);

                // Color by camera index (HSV) or by error
                ImU32 col;
                float lw = 2.0f;
                if (is_dimmed) {
                    col = IM_COL32(100, 100, 100, 60);
                    lw = 1.0f;
                } else if (state.color_by_error) {
                    double rmse = cam.final_rmse();
                    float t = (float)((rmse - rmse_min) / (rmse_max - rmse_min));
                    t = std::clamp(t, 0.0f, 1.0f);
                    // green (low error) -> red (high error)
                    int r_ch = (int)(t * 255);
                    int g_ch = (int)((1.0f - t * 0.7f) * 255);
                    col = IM_COL32(r_ch, g_ch, 25, 255);
                } else {
                    // HSV color wheel based on camera index
                    float hue = (float)c / (float)nc;
                    ImVec4 hsv_col;
                    ImGui::ColorConvertHSVtoRGB(hue, 0.8f, 0.9f, hsv_col.x, hsv_col.y, hsv_col.z);
                    col = IM_COL32((int)(hsv_col.x*255), (int)(hsv_col.y*255), (int)(hsv_col.z*255), 255);
                }
                if (is_selected) {
                    col = IM_COL32(255, 220, 50, 255); // gold
                    lw = 3.0f;
                }

                // Hover detection
                if (!is_dimmed) {
                    ImVec2 scr = ImPlot3D::PlotToPixels(X0.x(), X0.y(), X0.z());
                    ImVec2 mouse = ImGui::GetMousePos();
                    float dx = scr.x - mouse.x, dy = scr.y - mouse.y;
                    if (dx*dx + dy*dy < 400.0f) {
                        state.hovered_camera = c;
                        if (!is_selected) { col = IM_COL32(255, 255, 80, 255); lw = 3.0f; }
                    }
                }

                // Optical axis line: from X0 - r3*axis_length to X0 + r3*axis_length
                // (MATLAB sets P1 = origin, P2 = X0 + L*d; we draw a full segment through X0)
                Eigen::Vector3d line_start = Eigen::Vector3d::Zero();
                Eigen::Vector3d line_end = X0 + r3 * (double)state.axis_length;
                {
                    float lx[2] = {(float)line_start.x(), (float)line_end.x()};
                    float ly[2] = {(float)line_start.y(), (float)line_end.y()};
                    float lz[2] = {(float)line_start.z(), (float)line_end.z()};
                    ImPlot3D::PlotLine(("##axis_" + std::to_string(c)).c_str(), lx, ly, lz, 2,
                        {ImPlot3DProp_LineColor, col, ImPlot3DProp_LineWeight, lw});
                }

                // Image plane billboard: rectangle in (r1, r2) plane centered at X0
                float s = state.plane_scale;
                Eigen::Vector3d corners[4] = {
                    X0 - r1*s - r2*s,
                    X0 + r1*s - r2*s,
                    X0 + r1*s + r2*s,
                    X0 - r1*s + r2*s,
                };

                // Draw plane outline (closed loop)
                {
                    float rxs[5], rys[5], rzs[5];
                    for (int i = 0; i < 4; i++) {
                        rxs[i] = (float)corners[i].x();
                        rys[i] = (float)corners[i].y();
                        rzs[i] = (float)corners[i].z();
                    }
                    rxs[4] = rxs[0]; rys[4] = rys[0]; rzs[4] = rzs[0];
                    ImPlot3D::PlotLine(("##plane_" + std::to_string(c)).c_str(), rxs, rys, rzs, 5,
                        {ImPlot3DProp_LineColor, col, ImPlot3DProp_LineWeight, lw * 0.75f});
                }

                // Draw diagonals to show orientation (cross in the plane)
                if (is_selected || !single_cam) {
                    float dx1[2] = {(float)corners[0].x(), (float)corners[2].x()};
                    float dy1[2] = {(float)corners[0].y(), (float)corners[2].y()};
                    float dz1[2] = {(float)corners[0].z(), (float)corners[2].z()};
                    ImU32 diag_col = is_dimmed ? IM_COL32(80,80,80,30) : IM_COL32(
                        (col & 0xFF), ((col >> 8) & 0xFF), ((col >> 16) & 0xFF), 60);
                    ImPlot3D::PlotLine(("##diag1_" + std::to_string(c)).c_str(), dx1, dy1, dz1, 2,
                        {ImPlot3DProp_LineColor, diag_col, ImPlot3DProp_LineWeight, 1.0});
                    float dx2[2] = {(float)corners[1].x(), (float)corners[3].x()};
                    float dy2[2] = {(float)corners[1].y(), (float)corners[3].y()};
                    float dz2[2] = {(float)corners[1].z(), (float)corners[3].z()};
                    ImPlot3D::PlotLine(("##diag2_" + std::to_string(c)).c_str(), dx2, dy2, dz2, 2,
                        {ImPlot3DProp_LineColor, diag_col, ImPlot3DProp_LineWeight, 1.0});
                }

                // Label
                if (state.show_labels && !is_dimmed)
                    ImPlot3D::PlotText(cam.serial.c_str(), X0.x(), X0.y(), X0.z());
            }
        }

        // ── 3D landmarks ──
        if (state.show_landmarks && !state.landmarks_3d.empty()) {
            int np = (int)state.landmarks_3d.size();

            // Check if we have cross-validation per-point errors for coloring
            bool have_cv_errors = false;
            std::vector<double> point_max_error(np, 0.0);
            if (state.color_by_error && !res.cv_results.empty()) {
                // For each landmark, find the max LOO error across all cameras
                for (int m = 0; m < nc; m++) {
                    if (m >= (int)res.cv_results.size()) continue;
                    const auto &cv = res.cv_results[m];
                    for (int i = 0; i < (int)cv.point_indices.size(); i++) {
                        int pi = cv.point_indices[i];
                        if (pi >= 0 && pi < np && i < (int)cv.point_errors.size()) {
                            point_max_error[pi] = std::max(point_max_error[pi], cv.point_errors[i]);
                            have_cv_errors = true;
                        }
                    }
                }
            }

            if (have_cv_errors) {
                // Draw each point individually with error coloring
                // Find error range
                double err_min = 1e9, err_max = 0;
                for (int i = 0; i < np; i++) {
                    if (point_max_error[i] > 0) {
                        err_min = std::min(err_min, point_max_error[i]);
                        err_max = std::max(err_max, point_max_error[i]);
                    }
                }
                if (err_max <= err_min) err_max = err_min + 1.0;

                // Batch points into low/medium/high error groups for fewer draw calls
                std::vector<float> lx, ly, lz; // low error (green)
                std::vector<float> mx, my, mz; // medium (yellow)
                std::vector<float> hx, hy, hz; // high error (red)

                for (int i = 0; i < np; i++) {
                    float t = (float)((point_max_error[i] - err_min) / (err_max - err_min));
                    t = std::clamp(t, 0.0f, 1.0f);
                    float px = (float)state.landmarks_3d[i].x();
                    float py = (float)state.landmarks_3d[i].y();
                    float pz = (float)state.landmarks_3d[i].z();
                    if (t < 0.33f) { lx.push_back(px); ly.push_back(py); lz.push_back(pz); }
                    else if (t < 0.66f) { mx.push_back(px); my.push_back(py); mz.push_back(pz); }
                    else { hx.push_back(px); hy.push_back(py); hz.push_back(pz); }
                }

                if (!lx.empty())
                    ImPlot3D::PlotScatter("Low err", lx.data(), ly.data(), lz.data(), (int)lx.size(),
                        {ImPlot3DProp_MarkerSize, 3.0, ImPlot3DProp_MarkerFillColor, (ImU32)IM_COL32(50, 220, 50, 200)});
                if (!mx.empty())
                    ImPlot3D::PlotScatter("Med err", mx.data(), my.data(), mz.data(), (int)mx.size(),
                        {ImPlot3DProp_MarkerSize, 3.0, ImPlot3DProp_MarkerFillColor, (ImU32)IM_COL32(220, 220, 50, 200)});
                if (!hx.empty())
                    ImPlot3D::PlotScatter("High err", hx.data(), hy.data(), hz.data(), (int)hx.size(),
                        {ImPlot3DProp_MarkerSize, 4.0, ImPlot3DProp_MarkerFillColor, (ImU32)IM_COL32(255, 50, 50, 220)});
            } else {
                // Uniform color
                std::vector<float> px(np), py(np), pz(np);
                for (int i = 0; i < np; i++) {
                    px[i] = (float)state.landmarks_3d[i].x();
                    py[i] = (float)state.landmarks_3d[i].y();
                    pz[i] = (float)state.landmarks_3d[i].z();
                }
                ImPlot3D::PlotScatter("Landmarks", px.data(), py.data(), pz.data(), np,
                    {ImPlot3DProp_MarkerSize, 3.0, ImPlot3DProp_MarkerFillColor, (ImU32)IM_COL32(80, 130, 255, 180)});
            }
        }

        // ── World axes ──
        {
            float ax[2] = {0, world_axis_len}, ay[2] = {0, 0}, az[2] = {0, 0};
            ImPlot3D::PlotLine("X", ax, ay, az, 2,
                {ImPlot3DProp_LineColor, (ImU32)IM_COL32(255, 60, 60, 255), ImPlot3DProp_LineWeight, 2.5});
            float bx[2] = {0, 0}, by[2] = {0, world_axis_len}, bz[2] = {0, 0};
            ImPlot3D::PlotLine("Y", bx, by, bz, 2,
                {ImPlot3DProp_LineColor, (ImU32)IM_COL32(60, 255, 60, 255), ImPlot3DProp_LineWeight, 2.5});
            float cx[2] = {0, 0}, cy[2] = {0, 0}, cz[2] = {0, world_axis_len};
            ImPlot3D::PlotLine("Z", cx, cy, cz, 2,
                {ImPlot3DProp_LineColor, (ImU32)IM_COL32(80, 80, 255, 255), ImPlot3DProp_LineWeight, 2.5});
        }

        ImPlot3D::EndPlot();
    }

    ImPlot3D::PopStyleColor(3);

    // ── Hover tooltip ──
    if (state.hovered_camera >= 0 && state.hovered_camera < nc) {
        int c = state.hovered_camera;
        const auto &cam = res.cameras[c];
        double rmse = cam.final_rmse();

        ImGui::BeginTooltip();
        ImGui::TextUnformatted(("Camera: " + cam.serial).c_str());
        ImGui::Separator();
        ImGui::Text("sx=%.3f  sy=%.3f  skew=%.4f", cam.sx, cam.sy, cam.skew);
        ImGui::Text("RMSE: %.3f px (init: %.3f)", rmse, cam.rmse_init);
        ImGui::Text("Points: %d", cam.num_points);
        if (cam.k1 != 0 || cam.k2 != 0)
            ImGui::Text("Distortion: k1=%.6f  k2=%.6f", cam.k1, cam.k2);
        if (c < (int)res.cv_results.size() && res.cv_results[c].loo_rmse > 0) {
            ImGui::Separator();
            ImGui::Text("LOO CV RMSE: %.3f px", res.cv_results[c].loo_rmse);
            int n_outliers = (int)res.cv_results[c].outlier_indices.size();
            if (n_outliers > 0)
                ImGui::Text("Outliers: %d", n_outliers);
        }
        ImGui::Text("(Click to select)");
        ImGui::EndTooltip();

        // Click to select/deselect
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            state.selected_camera = (state.selected_camera == c) ? -1 : c;
        }
    }

    ImGui::End();
}
