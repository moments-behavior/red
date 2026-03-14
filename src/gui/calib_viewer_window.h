#pragma once
// calib_viewer_window.h — Interactive 3D visualization of calibration results.
// Camera frustums, 3D point cloud, per-camera inspection with hover tooltips.
// Uses ImPlot3D for immediate-mode 3D rendering inside an ImGui window.

#include "imgui.h"
#include "implot3d.h"
#include "calibration_pipeline.h"
#include "red_math.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct CalibViewerState {
    bool show = false;
    CalibrationPipeline::CalibrationResult *result = nullptr;
    float frustum_scale = 100.0f;
    bool show_points = true;
    bool show_frustums = true;
    bool show_labels = true;
    bool show_reg_graph = false;
    bool show_board_poses = false;
    bool show_axes_box = false; // show 3D box, axis labels, grid
    bool color_by_error = true;
    bool flip_z = false;       // display-only Z reflection
    int hovered_camera = -1;
    int hovered_edge = -1;
    int selected_camera = -1; // -1 = show all cameras
    // Cached per-camera point indices (built on selection change)
    std::vector<int> selected_cam_point_ids;
    int cached_selection = -2; // force rebuild on first frame
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

// Find which 3D points this camera actually observed during calibration.
// Uses the landmark map from the database (exact associations from BA),
// falling back to projection-based visibility if no database available.
inline std::vector<int> find_camera_observed_points(
    const CalibrationPipeline::CalibrationResult &res, int cam_idx) {
    std::vector<int> ids;
    if (cam_idx < 0 || cam_idx >= (int)res.cam_names.size()) return ids;
    const auto &cam_name = res.cam_names[cam_idx];

    // Use actual landmark associations from the database
    auto lm_it = res.db.landmarks.find(cam_name);
    if (lm_it != res.db.landmarks.end()) {
        for (const auto &[pid, px] : lm_it->second) {
            if (res.points_3d.count(pid))
                ids.push_back(pid);
        }
        return ids;
    }

    // Fallback: projection-based (less accurate)
    const auto &cam = res.cameras[cam_idx];
    Eigen::Vector3d rvec = red_math::rotationMatrixToVector(cam.R);
    for (const auto &[pid, pt3d] : res.points_3d) {
        Eigen::Vector2d proj = red_math::projectPoint(pt3d, rvec, cam.t, cam.K, cam.dist);
        if (proj.x() >= 0 && proj.x() < res.image_width &&
            proj.y() >= 0 && proj.y() < res.image_height)
            ids.push_back(pid);
    }
    return ids;
}

inline void DrawCalibViewerWindow(CalibViewerState &state) {
    if (!state.show || !state.result || !state.result->success) return;

    ImGui::SetNextWindowSize(ImVec2(800, 650), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Calibration 3D Viewer", &state.show)) {
        ImGui::End(); return;
    }

    const auto &res = *state.result;
    int nc = (int)res.cameras.size();

    // ── Controls ──
    ImVec4 label_col(0.5f, 0.7f, 1.0f, 1.0f); // light blue, matches Labeling Tool
    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("##frustum", &state.frustum_scale, 10.0f, 500.0f, "%.0f mm");
    ImGui::SameLine(); ImGui::TextColored(label_col, "Frustum");
    ImGui::PushStyleColor(ImGuiCol_Text, label_col);
    ImGui::SameLine(); ImGui::Checkbox("Points", &state.show_points);
    ImGui::SameLine(); ImGui::Checkbox("Labels", &state.show_labels);
    ImGui::SameLine(); ImGui::Checkbox("Reg. Graph", &state.show_reg_graph);
    ImGui::SameLine(); ImGui::Checkbox("Boards", &state.show_board_poses);
    ImGui::SameLine(); ImGui::Checkbox("Axes", &state.show_axes_box);
    ImGui::PopStyleColor();

    // Flip Z: reflects all poses and points across Z=0.
    // Saves R*F (det=-1) to YAML. The reload path uses projectPointR()
    // which handles improper rotations directly (no Rodrigues roundtrip).
    ImGui::SameLine();
    if (ImGui::Button("Flip Z")) {
        auto &r = *state.result;
        Eigen::Matrix3d F = Eigen::Vector3d(1, 1, -1).asDiagonal();
        for (auto &cam : r.cameras)
            cam.R = cam.R * F;
        for (auto &[id, pt] : r.points_3d)
            pt.z() = -pt.z();

        if (!r.output_folder.empty()) {
            std::string werr;
            CalibrationPipeline::write_calibration(
                r.cameras, r.cam_names, r.output_folder,
                r.image_width, r.image_height, &werr);

            // Also update ba_points.json
            namespace fs = std::filesystem;
            std::string pts_path = r.output_folder +
                "/summary_data/bundle_adjustment/ba_points.json";
            if (fs::exists(pts_path) && !r.points_3d.empty()) {
                try {
                    nlohmann::json pts_j;
                    for (const auto &[id, pt] : r.points_3d)
                        pts_j[std::to_string(id)] = {pt.x(), pt.y(), pt.z()};
                    std::ofstream pf(pts_path);
                    pf << pts_j.dump(2);
                } catch (...) {}
            }
            printf("[Flip Z] Saved %d cameras + 3D points to %s\n",
                   (int)r.cameras.size(), r.output_folder.c_str());
        }
        state.flip_z = !state.flip_z;  // toggle display flag to match
        state.cached_selection = -2;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Reflect cameras and points across Z=0.\n"
                          "Saves to YAML files. Click twice to undo.");

    // Camera selector
    {
        const char *preview = (state.selected_camera < 0) ? "All Cameras" :
            (state.selected_camera < nc ? res.cam_names[state.selected_camera].c_str() : "?");
        ImGui::SetNextItemWidth(180);
        if (ImGui::BeginCombo("Camera", preview)) {
            if (ImGui::Selectable("All Cameras", state.selected_camera < 0))
                state.selected_camera = -1;
            for (int c = 0; c < nc; c++) {
                char label[128];
                if (c < (int)res.per_camera_metrics.size())
                    snprintf(label, sizeof(label), "%s (%.3f px, %d dets)",
                        res.cam_names[c].c_str(),
                        res.per_camera_metrics[c].mean_reproj,
                        res.per_camera_metrics[c].detection_count);
                else
                    snprintf(label, sizeof(label), "%s", res.cam_names[c].c_str());
                if (ImGui::Selectable(label, state.selected_camera == c))
                    state.selected_camera = c;
            }
            ImGui::EndCombo();
        }
    }

    // Rebuild cached point list when selection changes
    if (state.cached_selection != state.selected_camera) {
        state.cached_selection = state.selected_camera;
        if (state.selected_camera >= 0)
            state.selected_cam_point_ids = find_camera_observed_points(res, state.selected_camera);
        else
            state.selected_cam_point_ids.clear();
    }

    // Summary
    if (state.selected_camera < 0) {
        ImGui::Text("Mean: %.3f px | Cameras: %d | Points: %d",
            res.mean_reproj_error, nc, (int)res.points_3d.size());
    } else if (state.selected_camera < (int)res.per_camera_metrics.size()) {
        const auto &m = res.per_camera_metrics[state.selected_camera];
        ImGui::Text("%s: reproj mean=%.3f median=%.3f px | %d dets | %d obs | %d visible 3D pts",
            m.name.c_str(), m.mean_reproj, m.median_reproj,
            m.detection_count, m.observation_count,
            (int)state.selected_cam_point_ids.size());
    }

    // Precompute frustums
    std::vector<FrustumGeometry> frustums(nc);
    for (int c = 0; c < nc; c++)
        frustums[c] = compute_frustum(res.cameras[c], res.image_width, res.image_height, state.frustum_scale);

    // Apply display-only Z flip if enabled
    if (state.flip_z) {
        for (auto &f : frustums) {
            f.center.z() = -f.center.z();
            for (auto &corner : f.corners)
                corner.z() = -corner.z();
        }
    }

    float scene_extent = 0;
    for (int c = 0; c < nc; c++)
        scene_extent = std::max(scene_extent, (float)frustums[c].center.norm());
    float axis_len = std::max(100.0f, scene_extent * 0.3f);

    auto avail = ImGui::GetContentRegionAvail();
    ImPlot3DFlags plot_flags = ImPlot3DFlags_Equal | ImPlot3DFlags_NoClip;
    if (!state.show_axes_box)
        plot_flags |= ImPlot3DFlags_CanvasOnly;

    // Hide all plot chrome when axes are off
    if (!state.show_axes_box) {
        ImPlot3D::PushStyleColor(ImPlot3DCol_PlotBg, ImVec4(0, 0, 0, 0));
        ImPlot3D::PushStyleColor(ImPlot3DCol_FrameBg, ImVec4(0, 0, 0, 0));
        ImPlot3D::PushStyleColor(ImPlot3DCol_PlotBorder, ImVec4(0, 0, 0, 0));
    }

    if (ImPlot3D::BeginPlot("##calib3d", avail, plot_flags)) {
        ImPlot3DAxisFlags ax_flags = state.show_axes_box ? 0 :
            (ImPlot3DAxisFlags_NoDecorations | ImPlot3DAxisFlags_NoTickMarks);
        ImPlot3D::SetupAxis(ImAxis3D_X, state.show_axes_box ? "X (mm)" : nullptr, ax_flags);
        ImPlot3D::SetupAxis(ImAxis3D_Y, state.show_axes_box ? "Y (mm)" : nullptr, ax_flags);
        ImPlot3D::SetupAxis(ImAxis3D_Z, state.show_axes_box ? "Z (mm)" : nullptr, ax_flags);

        state.hovered_camera = -1;
        bool single_cam = (state.selected_camera >= 0);

        // ── Camera frustums ──
        for (int c = 0; c < nc; c++) {
            const auto &f = frustums[c];
            bool is_selected = (c == state.selected_camera);
            bool is_dimmed = single_cam && !is_selected;

            // Color
            ImU32 col;
            float lw = 2.0f;
            if (is_dimmed) {
                col = IM_COL32(100, 100, 100, 60); // gray ghost
                lw = 1.0f;
            } else if (state.color_by_error && c < (int)res.per_camera_metrics.size()) {
                float err = (float)res.per_camera_metrics[c].mean_reproj;
                float t = std::min(err / 1.5f, 1.0f);
                col = IM_COL32((int)(t*255), (int)((1.0f-t*0.7f)*255), 25, 255);
            } else {
                col = IM_COL32(150, 200, 255, 255);
            }
            if (is_selected) {
                col = IM_COL32(255, 220, 50, 255); // gold for selected
                lw = 3.0f;
            }

            // Hover detection (only for non-dimmed cameras)
            if (!is_dimmed) {
                ImVec2 scr = ImPlot3D::PlotToPixels(f.center.x(), f.center.y(), f.center.z());
                ImVec2 mouse = ImGui::GetMousePos();
                float dx = scr.x-mouse.x, dy = scr.y-mouse.y;
                if (dx*dx+dy*dy < 400.0f) {
                    state.hovered_camera = c;
                    if (!is_selected) { col = IM_COL32(255,255,80,255); lw = 3.0f; }
                }
            }

            // Frustum edges
            float xs[8], ys[8], zs[8];
            for (int i = 0; i < 4; i++) {
                xs[i*2]=(float)f.center.x(); ys[i*2]=(float)f.center.y(); zs[i*2]=(float)f.center.z();
                xs[i*2+1]=(float)f.corners[i].x(); ys[i*2+1]=(float)f.corners[i].y(); zs[i*2+1]=(float)f.corners[i].z();
            }
            ImPlot3D::PlotLine(("##cam_"+std::to_string(c)).c_str(), xs, ys, zs, 8,
                {ImPlot3DProp_LineColor, col, ImPlot3DProp_LineWeight, lw,
                 ImPlot3DProp_Flags, (double)ImPlot3DLineFlags_Segments});

            // Image plane
            float rxs[5], rys[5], rzs[5];
            for (int i = 0; i < 4; i++) {
                rxs[i]=(float)f.corners[i].x(); rys[i]=(float)f.corners[i].y(); rzs[i]=(float)f.corners[i].z();
            }
            rxs[4]=rxs[0]; rys[4]=rys[0]; rzs[4]=rzs[0];
            ImPlot3D::PlotLine(("##rect_"+std::to_string(c)).c_str(), rxs, rys, rzs, 5,
                {ImPlot3DProp_LineColor, col, ImPlot3DProp_LineWeight, lw*0.75f});

            // Optical axis (selected camera only)
            if (is_selected) {
                Eigen::Matrix3d Rt = res.cameras[c].R.transpose();
                Eigen::Vector3d look_dir = Rt * Eigen::Vector3d(0, 0, 1); // camera Z in world
                Eigen::Vector3d axis_end = f.center + look_dir * state.frustum_scale * 1.5;
                float lx[2]={(float)f.center.x(),(float)axis_end.x()};
                float ly[2]={(float)f.center.y(),(float)axis_end.y()};
                float lz[2]={(float)f.center.z(),(float)axis_end.z()};
                ImPlot3D::PlotLine("##axis", lx, ly, lz, 2,
                    {ImPlot3DProp_LineColor, (ImU32)IM_COL32(255,255,0,180), ImPlot3DProp_LineWeight, 1.5});
            }

            // Label
            if (state.show_labels && !is_dimmed && c < (int)res.cam_names.size())
                ImPlot3D::PlotText(res.cam_names[c].c_str(), f.center.x(), f.center.y(), f.center.z());
        }

        // ── 3D points ──
        float zs = state.flip_z ? -1.0f : 1.0f; // Z sign for display
        if (state.show_points) {
            if (single_cam && !state.selected_cam_point_ids.empty()) {
                // Show only points visible to the selected camera
                std::vector<float> px, py, pz;
                for (int pid : state.selected_cam_point_ids) {
                    auto it = res.points_3d.find(pid);
                    if (it != res.points_3d.end()) {
                        px.push_back((float)it->second.x());
                        py.push_back((float)it->second.y());
                        pz.push_back((float)it->second.z() * zs);
                    }
                }
                if (!px.empty()) {
                    ImPlot3D::PlotScatter(
                        ("Observed by " + res.cam_names[state.selected_camera]).c_str(),
                        px.data(), py.data(), pz.data(), (int)px.size(),
                        {ImPlot3DProp_MarkerSize, 2.0, ImPlot3DProp_MarkerFillColor, (ImU32)IM_COL32(255,200,50,200)});
                }
            } else if (!single_cam && !res.points_3d.empty()) {
                // Show all points
                std::vector<float> px, py, pz;
                px.reserve(res.points_3d.size()); py.reserve(res.points_3d.size()); pz.reserve(res.points_3d.size());
                for (const auto &[id, pt] : res.points_3d) {
                    px.push_back((float)pt.x()); py.push_back((float)pt.y()); pz.push_back((float)pt.z() * zs);
                }
                ImPlot3D::PlotScatter("Landmarks", px.data(), py.data(), pz.data(), (int)px.size(),
                    {ImPlot3DProp_MarkerSize, 1.5, ImPlot3DProp_MarkerFillColor, (ImU32)IM_COL32(80,130,255,160)});
            }
        }

        // ── Registration graph ──
        state.hovered_edge = -1;
        if (state.show_reg_graph && !res.db.registration_order.empty()) {
            const auto &steps = res.db.registration_order;
            // Build camera name → index map
            std::map<std::string, int> name_to_idx;
            for (int c = 0; c < nc; c++) name_to_idx[res.cam_names[c]] = c;

            for (int e = 0; e < (int)steps.size(); e++) {
                const auto &step = steps[e];
                auto it_child = name_to_idx.find(step.camera_name);
                auto it_parent = name_to_idx.find(step.parent_camera);
                if (it_child == name_to_idx.end()) continue;
                if (step.parent_camera == "origin") continue; // skip the root camera
                if (it_parent == name_to_idx.end()) continue;

                const auto &fc = frustums[it_child->second].center;
                const auto &fp = frustums[it_parent->second].center;

                // Color by shared frame count
                float t = std::min((float)step.num_shared_frames / 100.0f, 1.0f);
                ImU32 edge_col = IM_COL32((int)((1-t)*255), (int)(t*255), 50, 180);
                float edge_w = 2.0f;

                // Hover detection on edge midpoint
                Eigen::Vector3d mid = (fc + fp) * 0.5;
                ImVec2 mid_scr = ImPlot3D::PlotToPixels(mid.x(), mid.y(), mid.z());
                ImVec2 mouse = ImGui::GetMousePos();
                float dx = mid_scr.x-mouse.x, dy = mid_scr.y-mouse.y;
                if (dx*dx+dy*dy < 400.0f) {
                    state.hovered_edge = e;
                    edge_col = IM_COL32(255, 255, 100, 255);
                    edge_w = 3.5f;
                }

                float ex[2]={(float)fc.x(),(float)fp.x()};
                float ey[2]={(float)fc.y(),(float)fp.y()};
                float ez[2]={(float)fc.z(),(float)fp.z()};
                ImPlot3D::PlotLine(("##edge_"+std::to_string(e)).c_str(), ex, ey, ez, 2,
                    {ImPlot3DProp_LineColor, edge_col, ImPlot3DProp_LineWeight, edge_w});
            }
        }

        // ── Board poses (selected camera only) ──
        if (state.show_board_poses && state.selected_camera >= 0 &&
            state.selected_camera < nc) {
            const auto &cam_name = res.cam_names[state.selected_camera];
            auto bp_it = res.db.board_poses.find(cam_name);
            if (bp_it != res.db.board_poses.end()) {
                const auto &cs = res.db.board_poses.at(cam_name);
                // Board geometry: rectangle from (0,0,0) to (board_w, board_h, 0)
                // We'll use a small square for each frame's board position
                for (const auto &[fi, bp] : cs) {
                    // Board corners in board frame: 4 corners of the ChArUco board
                    // Approximate: use 240mm x 240mm board (5x5 @ 60mm squares = 4x4 corners over 240mm)
                    float bw = 240.0f, bh = 240.0f; // approximate board size
                    Eigen::Vector3d bc[4] = {
                        {0, 0, 0}, {bw, 0, 0}, {bw, bh, 0}, {0, bh, 0}
                    };
                    // Transform to world using PnP pose: X_cam = R*X_board + t
                    // Then to world using camera pose: X_world = R_cam^T * (X_cam - t_cam)
                    // But the PnP pose IS the camera-to-board transform, so board corners
                    // in camera frame are: R_pnp * corner + t_pnp
                    // World frame: R_cam_world^T * (R_pnp * corner + t_pnp - t_cam)
                    // Actually, the PnP pose gives camera-in-board-frame coordinates.
                    // Board corner in camera frame: R_pnp * [x,y,0] + t_pnp
                    // Board corner in world: Rt_cam * (R_pnp * [x,y,0] + t_pnp - t_cam_world)
                    // Simpler: just transform board corners through PnP pose then camera inverse
                    const auto &cam = res.cameras[state.selected_camera];
                    Eigen::Matrix3d Rt_cam = cam.R.transpose();
                    Eigen::Vector3d cam_center = -Rt_cam * cam.t;

                    float rxs[5], rys[5], rzs[5];
                    for (int i = 0; i < 4; i++) {
                        Eigen::Vector3d pt_cam = bp.R * bc[i] + bp.t;
                        Eigen::Vector3d pt_world = Rt_cam * (pt_cam - cam.t);
                        rxs[i] = (float)pt_world.x();
                        rys[i] = (float)pt_world.y();
                        rzs[i] = (float)pt_world.z();
                    }
                    rxs[4]=rxs[0]; rys[4]=rys[0]; rzs[4]=rzs[0];

                    // Color by PnP reproj error
                    float t = std::min((float)bp.reproj / 2.0f, 1.0f);
                    ImU32 board_col = IM_COL32((int)(t*255), (int)((1-t)*200), 100, 150);
                    ImPlot3D::PlotLine(("##board_"+std::to_string(fi)).c_str(), rxs, rys, rzs, 5,
                        {ImPlot3DProp_LineColor, board_col, ImPlot3DProp_LineWeight, 1.5});
                }
            }
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

        ImPlot3D::EndPlot();
    }
    if (!state.show_axes_box)
        ImPlot3D::PopStyleColor(3);

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
            ImGui::Text("Detections: %d frames | Obs: %d", m.detection_count, m.observation_count);
            ImGui::Text("Reproj: mean=%.3f  median=%.3f  max=%.3f px", m.mean_reproj, m.median_reproj, m.max_reproj);
            ImGui::Text("Intrinsic reproj: %.3f px", m.intrinsic_reproj);
        }
        ImGui::Text("(Click to select)");
        ImGui::EndTooltip();

        // Click to select
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            state.selected_camera = (state.selected_camera == c) ? -1 : c;
        }
    }

    // ── Registration edge tooltip ──
    if (state.hovered_edge >= 0 && state.hovered_edge < (int)res.db.registration_order.size()) {
        const auto &step = res.db.registration_order[state.hovered_edge];
        ImGui::BeginTooltip();
        ImGui::Text("Registration step %d/%d", state.hovered_edge + 1,
            (int)res.db.registration_order.size());
        ImGui::Separator();
        ImGui::Text("%s -> %s", step.parent_camera.c_str(), step.camera_name.c_str());
        ImGui::Text("Method: %s", step.method.c_str());
        ImGui::Text("Shared frames: %d", step.num_shared_frames);
        if (step.num_3d_points > 0)
            ImGui::Text("3D points visible: %d", step.num_3d_points);
        ImGui::EndTooltip();
    }

    ImGui::End();
}
