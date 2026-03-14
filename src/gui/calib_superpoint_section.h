#pragma once
#include "superpoint_refinement.h"
#include "../calibration_tool.h"

// Forward declarations from the calibration tool framework
struct CalibrationToolState;
struct CalibrationToolCallbacks;
struct AppContext;

// Draw the SuperPoint Refinement section inside the Calibration Tool window.
inline void DrawCalibSuperPointSection(
    CalibrationToolState &state, AppContext &ctx,
    const CalibrationToolCallbacks &cb) {
    const auto &user_settings = ctx.user_settings;
    const auto &red_data_dir = ctx.red_data_dir;

    // Poll async result every frame
    if (state.sp_running && state.sp_future.valid()) {
        auto fs = state.sp_future.wait_for(std::chrono::milliseconds(0));
        if (fs == std::future_status::ready) {
            state.sp_result = state.sp_future.get();
            state.sp_running = false;
            state.sp_done = true;
            if (state.sp_result.success) {
                state.sp_status =
                    "Complete! Reproj: " +
                    std::to_string(state.sp_result.mean_reproj_after)
                        .substr(0, 5) +
                    " px";
            } else {
                state.sp_status = "Error: " + state.sp_result.error;
            }
        }
    }

    if (ImGui::CollapsingHeader("SuperPoint Refinement")) {
        ImGui::Indent();

        // --- Video Folder ---
        ImGui::Text("Video Folder:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80.0f);
        ImGui::InputText("##sp_vid_path", &state.sp_video_folder);
        ImGui::SameLine();
        if (ImGui::Button("Browse##sp_vid")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            if (!state.sp_video_folder.empty())
                cfg.path = state.sp_video_folder;
            else if (!user_settings.default_media_root_path.empty())
                cfg.path = user_settings.default_media_root_path;
            else
                cfg.path = red_data_dir;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseSPVideoFolder",
                "Select Video Folder", nullptr, cfg);
        }

        // Handle browse dialog result
        if (ImGuiFileDialog::Instance()->Display(
                "ChooseSPVideoFolder", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                state.sp_video_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            ImGuiFileDialog::Instance()->Close();
        }

        // --- Reference Camera combo ---
        if (!state.config.cam_ordered.empty()) {
            ImGui::Text("Reference Camera:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::BeginCombo("##sp_ref_cam",
                                  state.sp_ref_camera.empty()
                                      ? "Select..."
                                      : state.sp_ref_camera.c_str())) {
                for (const auto &cam : state.config.cam_ordered) {
                    bool selected = (cam == state.sp_ref_camera);
                    if (ImGui::Selectable(cam.c_str(), selected))
                        state.sp_ref_camera = cam;
                    if (selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

        // --- Frame Selection ---
        if (ImGui::CollapsingHeader("Frame Selection##sp")) {
            ImGui::Indent();
            ImGui::SliderInt("Frame Sets##sp", &state.sp_num_sets, 10, 100);
            ImGui::SliderFloat("Scan Interval (sec)##sp",
                               &state.sp_scan_interval, 0.5f, 10.0f, "%.1f");
            ImGui::SliderFloat("Min Separation (sec)##sp",
                               &state.sp_min_separation, 1.0f, 30.0f, "%.1f");
            ImGui::Unindent();
        }

        // --- Feature Matching ---
        if (ImGui::CollapsingHeader("Feature Matching##sp")) {
            ImGui::Indent();
            ImGui::SetNextItemWidth(200.0f);
            ImGui::InputText("Python##sp", &state.sp_python_path);
            ImGui::SliderInt("Workers##sp", &state.sp_workers, 1, 16);
            ImGui::SliderInt("Max Keypoints##sp", &state.sp_max_keypoints,
                             1024, 8192);
            ImGui::SliderFloat("Reproj Threshold (px)##sp",
                               &state.sp_reproj_thresh, 1.0f, 50.0f, "%.1f");
            ImGui::Unindent();
        }

        // --- Bundle Adjustment ---
        if (ImGui::CollapsingHeader("Bundle Adjustment##sp")) {
            ImGui::Indent();
            ImGui::SliderFloat("Rotation Prior##sp", &state.sp_rot_prior,
                               0.1f, 100.0f, "%.1f");
            ImGui::SliderFloat("Translation Prior##sp", &state.sp_trans_prior,
                               1.0f, 1000.0f, "%.1f");
            ImGui::Checkbox("Lock Intrinsics##sp", &state.sp_lock_intrinsics);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Fix focal length and principal point.\n"
                    "Recommended unless you have very high-quality matches.");
            ImGui::Checkbox("Lock Distortion##sp", &state.sp_lock_distortion);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Fix distortion coefficients during BA.");
            ImGui::SliderFloat("Outlier Th 1 (px)##sp",
                               &state.sp_outlier_th1, 1.0f, 50.0f, "%.1f");
            ImGui::SliderFloat("Outlier Th 2 (px)##sp",
                               &state.sp_outlier_th2, 0.5f, 20.0f, "%.1f");
            ImGui::Unindent();
        }

        ImGui::Separator();

        // --- Run button ---
        bool can_run = !state.sp_running &&
                       !state.sp_video_folder.empty() &&
                       !state.sp_ref_camera.empty() &&
                       !state.config.cam_ordered.empty();

        ImGui::BeginDisabled(!can_run);
        if (ImGui::Button("Run SuperPoint Refinement")) {
            SuperPointRefinement::SPConfig sp_cfg;
            sp_cfg.video_folder = state.sp_video_folder;
            sp_cfg.calibration_folder = state.project.calibration_folder;
            sp_cfg.camera_names = state.config.cam_ordered;
            sp_cfg.ref_camera = state.sp_ref_camera;
            sp_cfg.num_frame_sets = state.sp_num_sets;
            sp_cfg.scan_interval_sec = state.sp_scan_interval;
            sp_cfg.min_separation_sec = state.sp_min_separation;
            sp_cfg.python_path = state.sp_python_path;
            sp_cfg.workers = state.sp_workers;
            sp_cfg.max_keypoints = state.sp_max_keypoints;
            sp_cfg.reproj_thresh = state.sp_reproj_thresh;
            sp_cfg.prior_rot_weight = state.sp_rot_prior;
            sp_cfg.prior_trans_weight = state.sp_trans_prior;
            sp_cfg.lock_intrinsics = state.sp_lock_intrinsics;
            sp_cfg.lock_distortion = state.sp_lock_distortion;
            sp_cfg.ba_outlier_th1 = state.sp_outlier_th1;
            sp_cfg.ba_outlier_th2 = state.sp_outlier_th2;

            state.sp_running = true;
            state.sp_done = false;
            state.sp_status = "Starting SuperPoint refinement pipeline...";
            state.sp_progress =
                std::make_shared<SuperPointRefinement::SPProgress>();

            state.sp_future = std::async(
                std::launch::async,
                [sp_cfg, status_ptr = &state.sp_status,
                 progress = state.sp_progress]() {
                    return SuperPointRefinement::run_superpoint_refinement(
                        sp_cfg, status_ptr, progress);
                });
        }
        ImGui::EndDisabled();

        if (state.sp_running) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Running...");
        }

        // --- Progress (visible during/after run) ---
        if (state.sp_running && state.sp_progress) {
            if (ImGui::CollapsingHeader("Progress##sp",
                                        ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                auto &prog = *state.sp_progress;
                int step = prog.current_step.load(std::memory_order_relaxed);

                // Step 1: Frame Selection
                if (step >= 1) {
                    int scanned = prog.frames_scanned.load(
                        std::memory_order_relaxed);
                    int total_scan = prog.total_scan_frames.load(
                        std::memory_order_relaxed);
                    float f1 = total_scan > 0
                                   ? (float)scanned / total_scan
                                   : 0.0f;
                    char overlay1[64];
                    snprintf(overlay1, sizeof(overlay1), "%d/%d", scanned,
                             total_scan);
                    ImGui::Text("Step 1/5: Frame Selection");
                    ImGui::ProgressBar(f1, ImVec2(-1, 0), overlay1);
                }

                // Step 2: Frame Extraction
                if (step >= 2) {
                    int extracted = prog.frames_extracted.load(
                        std::memory_order_relaxed);
                    int total_extract = prog.total_extract_frames.load(
                        std::memory_order_relaxed);
                    float f2 = total_extract > 0
                                   ? (float)extracted / total_extract
                                   : 0.0f;
                    char overlay2[64];
                    snprintf(overlay2, sizeof(overlay2), "%d/%d", extracted,
                             total_extract);
                    ImGui::Text("Step 2/5: Frame Extraction");
                    ImGui::ProgressBar(f2, ImVec2(-1, 0), overlay2);
                }

                // Step 3: Feature Matching
                if (step >= 3) {
                    int matched = prog.sets_matched.load(
                        std::memory_order_relaxed);
                    int total_sets = prog.total_sets.load(
                        std::memory_order_relaxed);
                    float f3 = total_sets > 0
                                   ? (float)matched / total_sets
                                   : 0.0f;
                    char overlay3[64];
                    snprintf(overlay3, sizeof(overlay3), "%d/%d sets", matched,
                             total_sets);
                    ImGui::Text("Step 3/5: Feature Matching");
                    ImGui::ProgressBar(f3, ImVec2(-1, 0), overlay3);
                }

                // Step 4: Bundle Adjustment
                if (step >= 4) {
                    ImGui::Text("Step 4/5: Bundle Adjustment");
                    if (step == 4)
                        ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(),
                                           ImVec2(-1, 0), "Running...");
                    else
                        ImGui::ProgressBar(1.0f, ImVec2(-1, 0), "Done");
                }

                // Step 5: Complete
                if (step >= 5) {
                    ImGui::Text("Step 5/5: Complete!");
                    ImGui::ProgressBar(1.0f, ImVec2(-1, 0), "Done");
                }

                ImGui::Unindent();
            }
        }

        // --- Results (visible after completion) ---
        if (state.sp_done && state.sp_result.success) {
            if (ImGui::CollapsingHeader("Results##sp",
                                        ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
                                   "Reprojection error: %.3f px",
                                   state.sp_result.mean_reproj_after);
                ImGui::Text("Tracks: %d | Surviving: %d | Outliers: %d",
                            state.sp_result.total_tracks,
                            state.sp_result.valid_3d_points,
                            state.sp_result.ba_outliers_removed);

                // Per-camera changes table
                if (!state.sp_result.camera_changes.empty()) {
                    if (ImGui::TreeNode("Per-camera changes##sp")) {
                        if (ImGui::BeginTable(
                                "sp_changes", 3,
                                ImGuiTableFlags_RowBg |
                                    ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_Borders |
                                    ImGuiTableFlags_SizingFixedFit)) {
                            ImGui::TableSetupColumn("Camera", 0, 100.0f);
                            ImGui::TableSetupColumn("Rot (deg)", 0, 80.0f);
                            ImGui::TableSetupColumn("Trans (mm)", 0, 80.0f);
                            ImGui::TableHeadersRow();

                            for (const auto &cc :
                                 state.sp_result.camera_changes) {
                                ImGui::TableNextRow();
                                ImGui::TableSetColumnIndex(0);
                                ImGui::Text("Cam%s", cc.name.c_str());
                                ImGui::TableSetColumnIndex(1);
                                ImGui::Text("%.3f", cc.d_rot_deg);
                                ImGui::TableSetColumnIndex(2);
                                ImGui::Text("%.3f", cc.d_trans_mm);
                            }
                            ImGui::EndTable();
                        }
                        ImGui::TreePop();
                    }
                }

                // 3D viewer buttons
                if (ImGui::Button("Open 3D Viewer##sp")) {
                    state.calib_viewer.result =
                        &state.sp_result.calib_result;
                    state.calib_viewer.show = true;
                    state.calib_viewer.cached_selection = -2;
                }
                ImGui::SameLine();
                if (ImGui::Button("Compare: Show Initial##sp")) {
                    state.calib_viewer.result =
                        &state.sp_result.init_calib_result;
                    state.calib_viewer.show = true;
                    state.calib_viewer.cached_selection = -2;
                }

                ImGui::Unindent();
            }
        }

        // --- Status ---
        if (!state.sp_status.empty()) {
            ImGui::Separator();
            if (state.sp_status.find("Error") != std::string::npos) {
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s",
                                   state.sp_status.c_str());
            } else if (state.sp_done && state.sp_result.success) {
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%s",
                                   state.sp_status.c_str());
            } else if (state.sp_running) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "%s",
                                   state.sp_status.c_str());
            } else {
                ImGui::TextWrapped("%s", state.sp_status.c_str());
            }
        }

        ImGui::Unindent();
    } // end CollapsingHeader("SuperPoint Refinement")
    ImGui::Spacing();
}
