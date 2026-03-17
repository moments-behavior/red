#pragma once
// calib_kp_manual_section.h — Manual Keypoint Calibration Refinement section.
//
// The user labels N keypoints (e.g., 4 arena corners) in each camera view,
// then evaluates calibration quality and optionally runs BA refinement.
// Pattern follows calib_tele_section.h / calib_laser_section.h.

#include "calib_tool_state.h"
#include "app_context.h"
#include "feature_refinement.h"
#include "annotation_csv.h"
#include "calibration_pipeline.h"
#include "red_math.h"
#include "imgui.h"
#include <filesystem>
#include <map>
#include <set>

// ─────────────────────────────────────────────────────────────────────────────
// Collect labeled 2D keypoints from AnnotationMap into a landmarks map
// compatible with FeatureRefinement.
// ─────────────────────────────────────────────────────────────────────────────
inline std::map<std::string, std::map<int, Eigen::Vector2d>>
collect_manual_landmarks(
    const AnnotationMap &annotations,
    int frame_num,
    int num_keypoints,
    int num_cameras,
    const std::vector<std::string> &camera_serials) {

    std::map<std::string, std::map<int, Eigen::Vector2d>> landmarks;

    auto it = annotations.find((u32)frame_num);
    if (it == annotations.end()) return landmarks;

    const auto &fa = it->second;
    for (int c = 0; c < num_cameras && c < (int)fa.cameras.size(); c++) {
        const auto &cam = fa.cameras[c];
        std::string serial = (c < (int)camera_serials.size())
            ? camera_serials[c] : std::to_string(c);

        for (int k = 0; k < num_keypoints && k < (int)cam.keypoints.size(); k++) {
            const auto &kp = cam.keypoints[k];
            if (kp.labeled && kp.x < UNLABELED * 0.9 && kp.y < UNLABELED * 0.9) {
                landmarks[serial][k] = Eigen::Vector2d(kp.x, kp.y);
            }
        }
    }
    return landmarks;
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluate calibration quality: triangulate manual landmarks + compute reproj
// Uses KPEvalState defined in calib_tool_state.h
// ─────────────────────────────────────────────────────────────────────────────
using KPEvalResult = CalibrationToolState::KPEvalState;

inline KPEvalResult evaluate_manual_calibration(
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    const std::string &calib_folder,
    const std::vector<std::string> &camera_names) {

    KPEvalResult result;
    namespace fs = std::filesystem;
    int nc = (int)camera_names.size();

    // Load calibration
    std::vector<CalibrationPipeline::CameraPose> poses(nc);
    for (int c = 0; c < nc; c++) {
        std::string yaml_path = calib_folder + "/Cam" + camera_names[c] + ".yaml";
        if (!fs::exists(yaml_path)) continue;
        try {
            auto yaml = opencv_yaml::read(yaml_path);
            poses[c].K = yaml.getMatrix("camera_matrix").block<3, 3>(0, 0);
            Eigen::MatrixXd dist_mat = yaml.getMatrix("distortion_coefficients");
            for (int j = 0; j < 5; j++) poses[c].dist(j) = dist_mat(j, 0);
            poses[c].R = yaml.getMatrix("rc_ext").block<3, 3>(0, 0);
            Eigen::MatrixXd t_mat = yaml.getMatrix("tc_ext");
            poses[c].t = Eigen::Vector3d(t_mat(0, 0), t_mat(1, 0), t_mat(2, 0));
        } catch (...) {}
    }

    // Count labels
    std::set<int> all_kp_ids;
    int total_labeled = 0;
    std::set<std::string> cams_with_labels;
    for (const auto &[cam, pts] : landmarks) {
        total_labeled += (int)pts.size();
        if (!pts.empty()) cams_with_labels.insert(cam);
        for (const auto &[id, _] : pts) all_kp_ids.insert(id);
    }
    result.total_keypoints = total_labeled;
    result.total_cameras = (int)cams_with_labels.size();

    if (all_kp_ids.empty()) {
        result.error = "No labeled keypoints found";
        return result;
    }

    // Triangulate each keypoint seen by 2+ cameras
    CalibrationTool::CalibConfig calib_config;
    calib_config.cam_ordered = camera_names;

    std::map<int, Eigen::Vector3d> points_3d;
    CalibrationPipeline::triangulate_landmarks_multiview(
        calib_config, landmarks, poses, points_3d, 50.0); // generous threshold

    result.triangulated = (int)points_3d.size();

    // Compute reprojection error per camera
    result.per_camera.resize(nc);
    using CamEval = CalibrationToolState::KPCamEval;
    double err_sum = 0;
    int err_count = 0;

    for (int c = 0; c < nc; c++) {
        result.per_camera[c].serial = camera_names[c];
        auto it = landmarks.find(camera_names[c]);
        if (it == landmarks.end()) continue;

        double cam_err_sum = 0;
        int cam_count = 0;
        result.per_camera[c].labeled = (int)it->second.size();

        for (const auto &[pid, px] : it->second) {
            auto pit = points_3d.find(pid);
            if (pit == points_3d.end()) continue;
            auto pr = red_math::projectPointR(pit->second, poses[c].R, poses[c].t,
                                               poses[c].K, poses[c].dist);
            double e = (pr - px).norm();
            cam_err_sum += e;
            cam_count++;
            err_sum += e;
            err_count++;
            if (e > result.max_reproj) result.max_reproj = e;
        }

        result.per_camera[c].mean_reproj = cam_count > 0 ? cam_err_sum / cam_count : 0.0;
    }

    result.mean_reproj = err_count > 0 ? err_sum / err_count : 0.0;
    result.success = true;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// UI Section
// ─────────────────────────────────────────────────────────────────────────────
inline void DrawCalibKPManualSection(
    CalibrationToolState &state, AppContext &ctx,
    const CalibrationToolCallbacks &cb) {

    auto &pm = ctx.pm;
    auto &ps = ctx.ps;
    auto *scene = ctx.scene;
    auto *dc_context = ctx.dc_context;
    auto &imgs_names = ctx.imgs_names;
#ifdef __APPLE__
    auto &mac_last_uploaded_frame = ctx.mac_last_uploaded_frame;
#endif

    // Poll async refinement result
    if (state.kp_running && state.kp_future.valid()) {
        auto fs = state.kp_future.wait_for(std::chrono::milliseconds(0));
        if (fs == std::future_status::ready) {
            state.kp_feat_result = state.kp_future.get();
            state.kp_running = false;
            state.kp_refine_done = true;
            if (state.kp_feat_result.success) {
                state.kp_status = "Refinement done: " +
                    std::to_string(state.kp_feat_result.mean_reproj_after).substr(0, 5) + " px";
            } else {
                state.kp_status = "Refinement failed: " + state.kp_feat_result.error;
            }
        }
    }

    if (!ImGui::CollapsingHeader("Manual Keypoint Refinement"))
        return;

    ImGui::Indent();

    // ---- Video Folder ----
    ImGui::Text("Video Folder:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 160.0f);
    ImGui::InputText("##kp_vid_path", &state.kp_video_folder);
    ImGui::SameLine();
    if (ImGui::Button("Browse##kp_vid")) {
        IGFD::FileDialogConfig cfg;
        cfg.countSelectionMax = 1;
        if (!state.kp_video_folder.empty())
            cfg.path = state.kp_video_folder;
        cfg.flags = ImGuiFileDialogFlags_Modal;
        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseKPVideoFolder", "Select Video Folder", nullptr, cfg);
    }
    if (ImGuiFileDialog::Instance()->Display(
            "ChooseKPVideoFolder", ImGuiWindowFlags_NoCollapse,
            ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk())
            state.kp_video_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::SameLine();
    bool can_load = !state.kp_video_folder.empty() &&
                    !state.project.camera_names.empty();
    ImGui::BeginDisabled(!can_load);
    if (ImGui::Button("Load Videos##kp")) {
        cb.deferred->enqueue([&state, &pm, &ps, &cb, &imgs_names,
                              scene, dc_context
#ifdef __APPLE__
                              , &mac_last_uploaded_frame
#endif
        ]() {
            if (ps.video_loaded) cb.unload_media();
            imgs_names.clear();
#ifdef __APPLE__
            for (size_t ci = 0; ci < mac_last_uploaded_frame.size(); ci++)
                mac_last_uploaded_frame[ci] = -1;
#endif
            pm.media_folder = state.kp_video_folder;
            pm.camera_names.clear();
            for (const auto &cn : state.project.camera_names)
                pm.camera_names.push_back("Cam" + cn);
            cb.load_videos();
            cb.print_metadata();
            state.kp_videos_loaded = true;
            state.kp_status = "Videos loaded (" +
                std::to_string(state.project.camera_names.size()) + " cameras)";
        });
    }
    ImGui::EndDisabled();

    if (state.kp_videos_loaded) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Videos loaded");
    }

    ImGui::Spacing();

    // ---- Setup ----
    ImGui::SliderInt("Num Keypoints##kp", &state.kp_num_points, 1, 20);

    if (ImGui::Button("Setup Keypoints##kp")) {
        setup_landmark_skeleton(ctx.skeleton, state.kp_num_points, pm,
                                 state.project.project_path);

        // Ensure AnnotationMap has entries for frame 0
        auto &fa = ctx.annotations[0];
        if ((int)fa.cameras.size() < scene->num_cams)
            fa.cameras.resize(scene->num_cams);
        for (int c = 0; c < scene->num_cams; c++) {
            if ((int)fa.cameras[c].keypoints.size() < state.kp_num_points)
                fa.cameras[c].keypoints.resize(state.kp_num_points);
        }
        if ((int)fa.kp3d.size() < state.kp_num_points)
            fa.kp3d.resize(state.kp_num_points);

        state.kp_skeleton_ready = true;
        state.kp_status = "Skeleton created. Label " +
            std::to_string(state.kp_num_points) +
            " keypoints in each camera view.";
    }

    if (state.kp_skeleton_ready) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Ready to label");
    }

    // ---- Label count summary ----
    if (state.kp_skeleton_ready && scene->num_cams > 0) {
        int total_labeled = 0;
        int cams_with_labels = 0;
        auto it = ctx.annotations.find(0);
        if (it != ctx.annotations.end()) {
            const auto &fa = it->second;
            for (int c = 0; c < scene->num_cams && c < (int)fa.cameras.size(); c++) {
                int cam_labeled = 0;
                for (int k = 0; k < state.kp_num_points && k < (int)fa.cameras[c].keypoints.size(); k++) {
                    if (fa.cameras[c].keypoints[k].labeled) cam_labeled++;
                }
                if (cam_labeled > 0) cams_with_labels++;
                total_labeled += cam_labeled;
            }
        }
        ImGui::Text("Labels: %d points across %d cameras", total_labeled, cams_with_labels);
    }

    // ---- Save labels ----
    if (state.kp_skeleton_ready) {
        if (ImGui::Button("Save Labels##kp")) {
            std::string save_err;
            std::string saved = AnnotationCSV::save_all(
                pm.keypoints_root_folder, ctx.skeleton.name,
                ctx.annotations, scene->num_cams, ctx.skeleton.num_nodes,
                pm.camera_names, &save_err);
            if (saved.empty())
                state.kp_status = "Save failed: " + save_err;
            else
                state.kp_status = "Labels saved to " + saved;
        }
    }

    ImGui::Separator();

    // ---- Evaluate ----
    bool can_eval = state.kp_skeleton_ready &&
                    !state.project.calibration_folder.empty();

    ImGui::BeginDisabled(!can_eval);
    if (ImGui::Button("Evaluate Calibration##kp")) {
        // Collect landmarks from annotations
        auto landmarks = collect_manual_landmarks(
            ctx.annotations, 0, state.kp_num_points,
            scene->num_cams, state.project.camera_names);

        state.kp_eval = evaluate_manual_calibration(
            landmarks, state.project.calibration_folder,
            state.project.camera_names);

        if (state.kp_eval.success) {
            state.kp_status = "Reproj error: " +
                std::to_string(state.kp_eval.mean_reproj).substr(0, 5) +
                " px (" + std::to_string(state.kp_eval.triangulated) + " triangulated)";
        } else {
            state.kp_status = "Evaluation failed: " + state.kp_eval.error;
        }
    }
    ImGui::EndDisabled();

    // Show evaluation results
    if (state.kp_eval.success) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
            "Mean reproj: %.3f px  |  Max: %.3f px  |  Triangulated: %d/%d",
            state.kp_eval.mean_reproj, state.kp_eval.max_reproj,
            state.kp_eval.triangulated, (int)state.kp_eval.per_camera.size());

        // Per-camera table
        if (ImGui::TreeNode("Per-camera##kp_eval")) {
            if (ImGui::BeginTable("kp_eval_tab", 3,
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                    ImGuiTableFlags_SizingFixedFit)) {
                ImGui::TableSetupColumn("Camera", 0, 100.0f);
                ImGui::TableSetupColumn("Labels", 0, 60.0f);
                ImGui::TableSetupColumn("Reproj (px)", 0, 90.0f);
                ImGui::TableHeadersRow();
                for (const auto &ce : state.kp_eval.per_camera) {
                    if (ce.labeled == 0) continue;
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("Cam%s", ce.serial.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%d", ce.labeled);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.3f", ce.mean_reproj);
                }
                ImGui::EndTable();
            }
            ImGui::TreePop();
        }
    }

    ImGui::Separator();

    // ---- BA Refinement ----
    if (ImGui::CollapsingHeader("BA Refinement##kp")) {
        ImGui::Indent();
        ImGui::SliderFloat("Rotation Prior##kp", &state.kp_rot_prior, 1.0f, 200.0f, "%.1f");
        ImGui::SliderFloat("Translation Prior##kp", &state.kp_trans_prior, 10.0f, 2000.0f, "%.1f");
        ImGui::Checkbox("Lock Intrinsics##kp", &state.kp_lock_intrinsics);
        ImGui::Checkbox("Lock Distortion##kp", &state.kp_lock_distortion);

        bool can_refine = state.kp_skeleton_ready &&
                          !state.project.calibration_folder.empty() &&
                          !state.kp_running;

        ImGui::BeginDisabled(!can_refine);
        if (ImGui::Button("Run Refinement##kp")) {
            // Collect landmarks
            auto landmarks = collect_manual_landmarks(
                ctx.annotations, 0, state.kp_num_points,
                scene->num_cams, state.project.camera_names);

            state.kp_landmarks = landmarks; // save for async use

            FeatureRefinement::FeatureConfig feat_cfg;
            feat_cfg.calibration_folder = state.project.calibration_folder;
            feat_cfg.output_folder = state.project.project_path + "/manual_refined";
            feat_cfg.camera_names = state.project.camera_names;
            feat_cfg.prior_rot_weight = state.kp_rot_prior;
            feat_cfg.prior_trans_weight = state.kp_trans_prior;
            feat_cfg.lock_intrinsics = state.kp_lock_intrinsics;
            feat_cfg.lock_distortion = state.kp_lock_distortion;
            feat_cfg.ba_outlier_th1 = 30.0; // generous for manual labels
            feat_cfg.ba_outlier_th2 = 10.0;
            feat_cfg.ba_max_rounds = 5;
            feat_cfg.ba_convergence_eps = 0.001;
            feat_cfg.holdout_fraction = 0.2;
            feat_cfg.holdout_seed = 42;

            state.kp_running = true;
            state.kp_refine_done = false;
            state.kp_status = "Running BA refinement...";

            state.kp_future = std::async(
                std::launch::async,
                [feat_cfg, lm = state.kp_landmarks,
                 status_ptr = &state.kp_status]() {
                    return FeatureRefinement::run_feature_refinement_direct(
                        feat_cfg, lm, status_ptr);
                });
        }
        ImGui::EndDisabled();

        if (state.kp_running) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Running...");
        }
        ImGui::Unindent();
    }

    // ---- Refinement results ----
    if (state.kp_refine_done && state.kp_feat_result.success) {
        if (ImGui::CollapsingHeader("Refinement Results##kp", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent();

            ImGui::Text("Before: %.3f px  ->  After: %.3f px (train)",
                state.kp_feat_result.mean_reproj_before,
                state.kp_feat_result.train_reproj);

            if (state.kp_feat_result.holdout_observations > 0) {
                ImVec4 color = state.kp_feat_result.holdout_ratio > 1.5
                    ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
                    : ImVec4(0.3f, 1.0f, 0.3f, 1.0f);
                ImGui::TextColored(color,
                    "Holdout: %.3f px (%.1fx train) %s",
                    state.kp_feat_result.holdout_reproj,
                    state.kp_feat_result.holdout_ratio,
                    state.kp_feat_result.holdout_ratio > 1.5 ? "OVERFITTING" : "OK");
            }

            ImGui::Text("Tracks: %d | Surviving: %d | Outliers: %d | Rounds: %d",
                state.kp_feat_result.total_tracks,
                state.kp_feat_result.valid_3d_points,
                state.kp_feat_result.ba_outliers_removed,
                state.kp_feat_result.ba_rounds_completed);

            // Per-camera changes
            if (!state.kp_feat_result.camera_changes.empty()) {
                if (ImGui::TreeNode("Per-camera changes##kp_ref")) {
                    if (ImGui::BeginTable("kp_ref_tab", 3,
                            ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                            ImGuiTableFlags_SizingFixedFit)) {
                        ImGui::TableSetupColumn("Camera", 0, 100.0f);
                        ImGui::TableSetupColumn("Rot (deg)", 0, 80.0f);
                        ImGui::TableSetupColumn("Trans (mm)", 0, 80.0f);
                        ImGui::TableHeadersRow();
                        for (const auto &cc : state.kp_feat_result.camera_changes) {
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("Cam%s", cc.name.c_str());
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%.4f", cc.d_rot_deg);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("%.4f", cc.d_trans_mm);
                        }
                        ImGui::EndTable();
                    }
                    ImGui::TreePop();
                }
            }

            // 3D viewer button
            if (ImGui::Button("Open 3D Viewer##kp")) {
                // Load refined result for viewer
                state.loaded_result = SuperPointRefinement::load_calib_result_from_folder(
                    state.kp_feat_result.output_folder, state.project.camera_names);
                state.loaded_result.mean_reproj_error = state.kp_feat_result.mean_reproj_after;
                state.calib_viewer.result = &state.loaded_result;
                state.calib_viewer.show = true;
                state.calib_viewer.cached_selection = -2;
            }

            ImGui::Unindent();
        }
    }

    // ---- Status ----
    if (!state.kp_status.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", state.kp_status.c_str());
    }

    ImGui::Unindent();
    ImGui::Spacing();
}
