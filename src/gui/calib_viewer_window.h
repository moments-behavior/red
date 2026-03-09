#pragma once
// calib_viewer_window.h — Interactive 3D visualization of calibration results.
// Shows camera frustums, 3D point cloud, and per-camera error color coding.
// Uses ImPlot3D for immediate-mode 3D rendering inside an ImGui window.

#include "imgui.h"
#include "implot3d.h"
#include "calibration_pipeline.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <string>
#include <vector>

struct CalibViewerState {
    bool show = false;
    const CalibrationPipeline::CalibrationResult *result = nullptr;
    float frustum_scale = 0.15f;
    bool show_points = true;
    bool show_frustums = true;
    bool show_labels = true;
    bool color_by_error = true;
    int selected_camera = -1;
};

// Compute camera frustum vertices in world coordinates
struct FrustumGeometry {
    Eigen::Vector3d center;
    Eigen::Vector3d corners[4]; // image plane corners at frustum_depth
};

inline FrustumGeometry compute_frustum(
    const CalibrationPipeline::CameraPose &pose,
    int image_width, int image_height, float depth) {
    FrustumGeometry f;
    Eigen::Matrix3d Rt = pose.R.transpose();
    f.center = -Rt * pose.t;

    double w = image_width, h = image_height;
    Eigen::Vector2d img_corners[4] = {{0, 0}, {w, 0}, {w, h}, {0, h}};

    Eigen::Matrix3d Kinv = pose.K.inverse();
    for (int i = 0; i < 4; i++) {
        Eigen::Vector3d ray = Kinv * Eigen::Vector3d(
            img_corners[i].x(), img_corners[i].y(), 1.0);
        ray *= depth;
        f.corners[i] = f.center + Rt * ray;
    }
    return f;
}

inline void DrawCalibViewerWindow(CalibViewerState &state) {
    if (!state.show || !state.result || !state.result->success) return;

    ImGui::SetNextWindowSize(ImVec2(650, 550), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Calibration 3D Viewer", &state.show)) {
        ImGui::End();
        return;
    }

    const auto &res = *state.result;
    int nc = (int)res.cameras.size();

    // Controls
    ImGui::SliderFloat("Frustum Size", &state.frustum_scale, 0.01f, 1.0f);
    ImGui::SameLine();
    ImGui::Checkbox("Points", &state.show_points);
    ImGui::SameLine();
    ImGui::Checkbox("Labels", &state.show_labels);
    ImGui::SameLine();
    ImGui::Checkbox("Color by Error", &state.color_by_error);

    // Summary line
    ImGui::Text("Mean: %.3f px | Cameras: %d | Points: %d",
        res.mean_reproj_error, nc, (int)res.points_3d.size());

    auto avail = ImGui::GetContentRegionAvail();
    if (ImPlot3D::BeginPlot("##calib3d", avail)) {
        ImPlot3D::SetupAxes("X", "Y", "Z");

        // Draw camera frustums
        if (state.show_frustums) {
            for (int c = 0; c < nc; c++) {
                auto f = compute_frustum(res.cameras[c],
                    res.image_width, res.image_height, state.frustum_scale);

                // Color by reprojection error: green=0px, red=2+px
                ImVec4 color(0.3f, 0.8f, 0.3f, 1.0f); // default green
                if (state.color_by_error && c < (int)res.per_camera_metrics.size()) {
                    float err = (float)res.per_camera_metrics[c].mean_reproj;
                    float t = std::min(err / 2.0f, 1.0f);
                    color = ImVec4(t, 1.0f - t, 0.0f, 1.0f);
                }

                // 8 line segments: center→corner × 4
                float xs[8], ys[8], zs[8];
                for (int i = 0; i < 4; i++) {
                    xs[i * 2] = (float)f.center.x();
                    ys[i * 2] = (float)f.center.y();
                    zs[i * 2] = (float)f.center.z();
                    xs[i * 2 + 1] = (float)f.corners[i].x();
                    ys[i * 2 + 1] = (float)f.corners[i].y();
                    zs[i * 2 + 1] = (float)f.corners[i].z();
                }

                std::string label = "##cam_" + std::to_string(c);
                ImPlot3D::SetNextLineStyle(color, 2.0f);
                ImPlot3D::PlotLine(label.c_str(), xs, ys, zs, 8,
                    ImPlot3DLineFlags_Segments);

                // Rectangle around image plane
                float rxs[5], rys[5], rzs[5];
                for (int i = 0; i < 4; i++) {
                    rxs[i] = (float)f.corners[i].x();
                    rys[i] = (float)f.corners[i].y();
                    rzs[i] = (float)f.corners[i].z();
                }
                rxs[4] = rxs[0]; rys[4] = rys[0]; rzs[4] = rzs[0];
                std::string rlabel = "##rect_" + std::to_string(c);
                ImPlot3D::SetNextLineStyle(color, 1.5f);
                ImPlot3D::PlotLine(rlabel.c_str(), rxs, rys, rzs, 5);

                // Camera label
                if (state.show_labels && c < (int)res.cam_names.size()) {
                    ImPlot3D::PlotText(res.cam_names[c].c_str(),
                        f.center.x(), f.center.y(), f.center.z());
                }
            }
        }

        // Draw 3D point cloud
        if (state.show_points && !res.points_3d.empty()) {
            std::vector<float> px, py, pz;
            px.reserve(res.points_3d.size());
            py.reserve(res.points_3d.size());
            pz.reserve(res.points_3d.size());
            for (const auto &[id, pt] : res.points_3d) {
                px.push_back((float)pt.x());
                py.push_back((float)pt.y());
                pz.push_back((float)pt.z());
            }
            ImPlot3D::SetNextMarkerStyle(ImPlot3DMarker_Circle, 2.0f,
                ImVec4(0.4f, 0.6f, 1.0f, 0.8f));
            ImPlot3D::PlotScatter("Landmarks",
                px.data(), py.data(), pz.data(), (int)px.size());
        }

        // World origin axes
        {
            float ax[2] = {0, 100}, ay[2] = {0, 0}, az[2] = {0, 0};
            ImPlot3D::SetNextLineStyle(ImVec4(1, 0, 0, 1), 2.0f);
            ImPlot3D::PlotLine("X", ax, ay, az, 2);
            float bx[2] = {0, 0}, by[2] = {0, 100}, bz[2] = {0, 0};
            ImPlot3D::SetNextLineStyle(ImVec4(0, 1, 0, 1), 2.0f);
            ImPlot3D::PlotLine("Y", bx, by, bz, 2);
            float cx[2] = {0, 0}, cy[2] = {0, 0}, cz[2] = {0, 100};
            ImPlot3D::SetNextLineStyle(ImVec4(0, 0, 1, 1), 2.0f);
            ImPlot3D::PlotLine("Z", cx, cy, cz, 2);
        }

        ImPlot3D::EndPlot();
    }

    ImGui::End();
}
