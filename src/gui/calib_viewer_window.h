#pragma once
// calib_viewer_window.h — Interactive 3D visualization of calibration results.
// Camera frustums, 3D point cloud, per-camera error coloring,
// hover tooltips with calibration metadata. Uses ImPlot3D.

#include "imgui.h"
#include "implot3d.h"
#include "calibration_pipeline.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct CalibViewerState {
    bool show = false;
    const CalibrationPipeline::CalibrationResult *result = nullptr;
    float frustum_scale = 100.0f;
    bool show_points = true;
    bool show_frustums = true;
    bool show_labels = true;
    bool show_grid = true;
    bool color_by_error = true;
    int hovered_camera = -1;
};

struct FrustumGeometry {
    Eigen::Vector3d center;
    Eigen::Vector3d corners[4];
};

inline FrustumGeometry compute_frustum(
    const CalibrationPipeline::CameraPose &pose,
    int image_width, int image_height, float depth) {
    FrustumGeometry f;
    Eigen::Matrix3d Rt = pose.R.transpose();
    f.center = -Rt * pose.t;
    double w = image_width, h = image_height;
    Eigen::Vector2d img_corners[4] = {{0,0},{w,0},{w,h},{0,h}};
    Eigen::Matrix3d Kinv = pose.K.inverse();
    for (int i = 0; i < 4; i++) {
        Eigen::Vector3d ray = Kinv * Eigen::Vector3d(img_corners[i].x(), img_corners[i].y(), 1.0);
        ray *= depth;
        f.corners[i] = f.center + Rt * ray;
    }
    return f;
}

// Helper to convert ImVec4 color to ImU32 for ImPlot3D
inline ImU32 Vec4ToU32(const ImVec4 &c) {
    return IM_COL32((int)(c.x*255), (int)(c.y*255), (int)(c.z*255), (int)(c.w*255));
}

inline void DrawCalibViewerWindow(CalibViewerState &state) {
    if (!state.show || !state.result || !state.result->success) return;

    ImGui::SetNextWindowSize(ImVec2(800, 650), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Calibration 3D Viewer", &state.show)) {
        ImGui::End(); return;
    }

    const auto &res = *state.result;
    int nc = (int)res.cameras.size();

    // Controls
    ImGui::SliderFloat("Frustum (mm)", &state.frustum_scale, 10.0f, 500.0f);
    ImGui::SameLine();
    ImGui::Checkbox("Points", &state.show_points);
    ImGui::SameLine();
    ImGui::Checkbox("Labels", &state.show_labels);
    ImGui::SameLine();
    ImGui::Checkbox("Grid", &state.show_grid);

    ImGui::Text("Mean: %.3f px | Cameras: %d | Points: %d",
        res.mean_reproj_error, nc, (int)res.points_3d.size());

    // Precompute frustums
    std::vector<FrustumGeometry> frustums(nc);
    for (int c = 0; c < nc; c++)
        frustums[c] = compute_frustum(res.cameras[c], res.image_width, res.image_height, state.frustum_scale);

    // Auto-scale axes
    float scene_extent = 0;
    for (int c = 0; c < nc; c++)
        scene_extent = std::max(scene_extent, (float)frustums[c].center.norm());
    float axis_len = std::max(100.0f, scene_extent * 0.3f);

    auto avail = ImGui::GetContentRegionAvail();
    if (ImPlot3D::BeginPlot("##calib3d", avail, ImPlot3DFlags_Equal)) {
        ImPlot3D::SetupAxes("X (mm)", "Y (mm)", "Z (mm)");

        state.hovered_camera = -1;

        // ── Camera frustums ──
        if (state.show_frustums) {
            for (int c = 0; c < nc; c++) {
                const auto &f = frustums[c];

                // Color
                ImU32 col = IM_COL32(150, 200, 255, 255);
                if (state.color_by_error && c < (int)res.per_camera_metrics.size()) {
                    float err = (float)res.per_camera_metrics[c].mean_reproj;
                    float t = std::min(err / 1.5f, 1.0f);
                    col = IM_COL32((int)(t*255), (int)((1.0f-t*0.7f)*255), 25, 255);
                }
                float lw = 2.0f;

                // Hover detection
                ImVec2 cam_scr = ImPlot3D::PlotToPixels(f.center.x(), f.center.y(), f.center.z());
                ImVec2 mouse = ImGui::GetMousePos();
                float dx = cam_scr.x - mouse.x, dy = cam_scr.y - mouse.y;
                if (dx*dx + dy*dy < 400.0f) {
                    state.hovered_camera = c;
                    col = IM_COL32(255, 255, 80, 255);
                    lw = 3.0f;
                }

                // Frustum edges (4 lines from center to corners)
                float xs[8], ys[8], zs[8];
                for (int i = 0; i < 4; i++) {
                    xs[i*2]=(float)f.center.x(); ys[i*2]=(float)f.center.y(); zs[i*2]=(float)f.center.z();
                    xs[i*2+1]=(float)f.corners[i].x(); ys[i*2+1]=(float)f.corners[i].y(); zs[i*2+1]=(float)f.corners[i].z();
                }
                ImPlot3D::PlotLine(("##cam_"+std::to_string(c)).c_str(), xs, ys, zs, 8,
                    {ImPlot3DProp_LineColor, col, ImPlot3DProp_LineWeight, lw, ImPlot3DProp_Flags, (double)ImPlot3DLineFlags_Segments});

                // Image plane rectangle
                float rxs[5], rys[5], rzs[5];
                for (int i = 0; i < 4; i++) {
                    rxs[i]=(float)f.corners[i].x(); rys[i]=(float)f.corners[i].y(); rzs[i]=(float)f.corners[i].z();
                }
                rxs[4]=rxs[0]; rys[4]=rys[0]; rzs[4]=rzs[0];
                ImPlot3D::PlotLine(("##rect_"+std::to_string(c)).c_str(), rxs, rys, rzs, 5,
                    {ImPlot3DProp_LineColor, col, ImPlot3DProp_LineWeight, lw*0.75f});

                // Label
                if (state.show_labels && c < (int)res.cam_names.size())
                    ImPlot3D::PlotText(res.cam_names[c].c_str(), f.center.x(), f.center.y(), f.center.z());
            }
        }

        // ── 3D point cloud ──
        if (state.show_points && !res.points_3d.empty()) {
            std::vector<float> px, py, pz;
            px.reserve(res.points_3d.size()); py.reserve(res.points_3d.size()); pz.reserve(res.points_3d.size());
            for (const auto &[id, pt] : res.points_3d) {
                px.push_back((float)pt.x()); py.push_back((float)pt.y()); pz.push_back((float)pt.z());
            }
            ImPlot3D::PlotScatter("Landmarks", px.data(), py.data(), pz.data(), (int)px.size(),
                {ImPlot3DProp_MarkerSize, 1.5, ImPlot3DProp_MarkerFillColor, (ImU32)IM_COL32(80,130,255,160)});
        }

        // ── World axes ──
        {
            float ax[2]={0,axis_len}, ay[2]={0,0}, az[2]={0,0};
            ImPlot3D::PlotLine("X", ax, ay, az, 2, {ImPlot3DProp_LineColor, (ImU32)IM_COL32(255,60,60,255), ImPlot3DProp_LineWeight, 2.5});
            float bx[2]={0,0}, by[2]={0,axis_len}, bz[2]={0,0};
            ImPlot3D::PlotLine("Y", bx, by, bz, 2, {ImPlot3DProp_LineColor, (ImU32)IM_COL32(60,255,60,255), ImPlot3DProp_LineWeight, 2.5});
            float cx2[2]={0,0}, cy2[2]={0,0}, cz2[2]={0,axis_len};
            ImPlot3D::PlotLine("Z", cx2, cy2, cz2, 2, {ImPlot3DProp_LineColor, (ImU32)IM_COL32(80,80,255,255), ImPlot3DProp_LineWeight, 2.5});
        }

        // ── Ground grid ──
        if (state.show_grid) {
            float gs = std::max(50.0f, std::round(scene_extent/5.0f/50.0f)*50.0f);
            float gr = gs * 5;
            ImU32 grid_col = IM_COL32(128,128,128,50);
            for (float v = -gr; v <= gr; v += gs) {
                float gx[2]={-gr,gr}, gy[2]={v,v}, gz[2]={0,0};
                ImPlot3D::PlotLine("##g", gx, gy, gz, 2, {ImPlot3DProp_LineColor, grid_col, ImPlot3DProp_LineWeight, 1.0});
                float gx2[2]={v,v}, gy2[2]={-gr,gr}, gz2[2]={0,0};
                ImPlot3D::PlotLine("##g", gx2, gy2, gz2, 2, {ImPlot3DProp_LineColor, grid_col, ImPlot3DProp_LineWeight, 1.0});
            }
        }

        ImPlot3D::EndPlot();
    }

    // ── Hover tooltip ──
    if (state.hovered_camera >= 0 && state.hovered_camera < nc) {
        int c = state.hovered_camera;
        const auto &cam = res.cameras[c];
        const std::string &name = (c < (int)res.cam_names.size()) ? res.cam_names[c] : "?";
        Eigen::Vector3d center = -cam.R.transpose() * cam.t;

        ImGui::BeginTooltip();
        ImGui::TextUnformatted(("Camera: " + name).c_str());
        ImGui::Separator();
        ImGui::Text("Position: (%.1f, %.1f, %.1f) mm", center.x(), center.y(), center.z());
        ImGui::Text("Focal: fx=%.1f  fy=%.1f", cam.K(0,0), cam.K(1,1));
        ImGui::Text("Principal: (%.1f, %.1f)", cam.K(0,2), cam.K(1,2));
        ImGui::Text("Dist: k1=%.4f k2=%.4f p1=%.4f p2=%.4f k3=%.4f",
            cam.dist(0), cam.dist(1), cam.dist(2), cam.dist(3), cam.dist(4));
        if (c < (int)res.per_camera_metrics.size()) {
            const auto &m = res.per_camera_metrics[c];
            ImGui::Separator();
            ImGui::Text("Detections: %d frames | Observations: %d", m.detection_count, m.observation_count);
            ImGui::Text("Reproj: mean=%.3f  median=%.3f  max=%.3f px", m.mean_reproj, m.median_reproj, m.max_reproj);
            ImGui::Text("Intrinsic reproj: %.3f px", m.intrinsic_reproj);
        }
        ImGui::EndTooltip();
    }

    ImGui::End();
}
