#pragma once
#include "imgui.h"
#include "imgui_internal.h"
#include "app_context.h"
#include "calib_tool_state.h"
#include "calib_create_dialog.h"
#include "calib_tele_section.h"
#include "calib_aruco_section.h"
#include "calib_pointsource_section.h"
#include "calib_superpoint_section.h"
#include "calib_kp_manual_section.h"
#include "calib_viewer_window.h"
#include "tele_viewer_window.h"
#include <ImGuiFileDialog.h>
#include <string>

inline void DrawCalibrationToolWindow(
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

    // Always process file dialogs (even when window is hidden)

    // Calibration: Browse for root directory (creation dialog)
    if (ImGuiFileDialog::Instance()->Display("ChooseCalibRootDir", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            state.project.project_root_path =
                ImGuiFileDialog::Instance()->GetCurrentPath();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // Calibration: Browse for config.json (creation dialog)
    if (ImGuiFileDialog::Instance()->Display("ChooseCalibConfigCreate", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string selected =
                ImGuiFileDialog::Instance()->GetFilePathName();
            if (!selected.empty())
                state.project.config_file = selected;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // Calibration: Load existing .redproj (unified -- handles both aruco and laser projects)
    if (ImGuiFileDialog::Instance()->Display(
            "LoadCalibProject", ImGuiWindowFlags_NoCollapse,
            ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string selected =
                ImGuiFileDialog::Instance()->GetFilePathName();
            if (!selected.empty()) {
                // Clean up any existing project state
                close_project(ctx);

                CalibrationTool::CalibProject loaded;
                std::string err;
                if (CalibrationTool::load_project(
                        &loaded, selected, &err)) {
                    state.project = loaded;

                    // Switch layout ini to project folder
                    cb.switch_ini(state.project.project_path);

                    // Load aruco config if present
                    if (state.project.has_aruco()) {
                        if (!state.project.config_file.empty()) {
                            // Config file provided — parse it
                            state.config_path = state.project.config_file;
                            if (CalibrationTool::parse_config(
                                    state.config_path, state.config, err)) {
                                state.config_loaded = true;
                                state.init_camera_enabled();
                                state.images_loaded = false;
                                state.aruco_done = false;
                            } else {
                                state.config_loaded = false;
                                state.status = "Error parsing config: " + err;
                            }
                        } else if (!state.project.camera_names.empty()) {
                            // Config-free project — synthesize CalibConfig
                            // from persisted project fields (no media scan needed)
                            state.config.cam_ordered = state.project.camera_names;
                            state.config.charuco_setup = state.project.charuco_setup;
                            state.config.img_path = state.project.aruco_media_folder;
                            state.config_loaded = true;
                            state.init_camera_enabled();
                            state.images_loaded = false;
                            state.aruco_done = false;
                            state.status =
                                "Project loaded (config-free): " +
                                std::to_string(state.config.cam_ordered.size()) +
                                " cameras";
                        }
                    }

                    // Set up laser config if laser inputs present.
                    // For pure laser projects (no aruco), auto-load videos.
                    // For aruco+laser projects, defer to the "Load PointSource Videos" button.
                    if (state.project.has_laser_input() &&
                        !state.project.has_aruco()) {
                        state.pointsource_config.media_folder = state.project.media_folder;
                        state.pointsource_config.calibration_folder =
                            state.project.calibration_folder;
                        state.pointsource_config.camera_names = state.project.camera_names;
                        state.pointsource_config.output_folder =
                            state.project.project_path + "/pointsource_calibration";
                        state.pointsource_ready = true;
                        state.pointsource_focus_window = true;

                        // Load videos into 2x2 grid
                        if (!ps.video_loaded) {
                            pm.media_folder = state.project.media_folder;
                            pm.camera_names.clear();
                            for (const auto &cn : state.project.camera_names)
                                pm.camera_names.push_back("Cam" + cn);
                            cb.load_videos();
                            cb.print_metadata();
                        }
                        state.pointsource_total_frames = dc_context->estimated_num_frames;
                    }

                    state.project_loaded = true;
                    state.dock_pending = true;
                    state.show_create_dialog = false;
                    state.show = true;

                    // Auto-load telecentric videos on project open (direct, like laser)
                    // Label import is deferred by a few frames to avoid dock crash.
                    if (state.project.is_telecentric() &&
                        !state.project.media_folder.empty() &&
                        !state.project.camera_names.empty()) {
                        if (ps.video_loaded)
                            cb.unload_media();
                        pm.media_folder = state.project.media_folder;
                        pm.camera_names.clear();
                        for (const auto &cn : state.project.camera_names)
                            pm.camera_names.push_back("Cam" + cn);
                        cb.load_videos();
                        cb.print_metadata();
                        state.tele_videos_loaded = true;
                        // Schedule label import after dock layout stabilizes
                        state.tele_deferred_label_frames = 3;
                    }

                    // Restore DLT results from persisted metadata
                    if (state.project.is_telecentric() &&
                        state.project.dlt_method >= 0 &&
                        !state.project.tele_output_folder.empty()) {
                        state.tele_dlt_done = true;
                        auto &r = state.tele_dlt_result;
                        r.success = true;
                        r.method = static_cast<TelecentricDLT::Method>(
                            state.project.dlt_method);
                        r.mean_rmse = state.project.dlt_mean_rmse;
                        r.output_folder = state.project.tele_output_folder;
                        r.cameras.resize(state.project.camera_names.size());
                        for (int i = 0; i < (int)state.project.camera_names.size(); i++) {
                            r.cameras[i].serial = state.project.camera_names[i];
                            if (i < (int)state.project.dlt_per_camera_rmse.size()) {
                                r.cameras[i].rmse_ba = state.project.dlt_per_camera_rmse[i];
                                r.cameras[i].rmse_init = state.project.dlt_per_camera_rmse[i];
                            }
                        }
                        state.tele_dlt_status =
                            std::string(TelecentricDLT::method_name(r.method)) +
                            " loaded. Mean RMSE: " +
                            std::to_string(r.mean_rmse).substr(0, 6) + " px";
                    }

                    // Status message
                    if (state.project.is_telecentric()) {
                        state.status =
                            "Loaded " +
                            std::to_string(state.project.camera_names.size()) +
                            " cameras. Click 'Start Labeling' to annotate.";
                    } else if (state.project.has_aruco() && state.config_loaded) {
                        state.status =
                            "Project loaded: " +
                            std::to_string(state.config.cam_ordered.size()) +
                            " cameras";
                    } else if (state.project.has_laser_input()) {
                        state.status =
                            "Project loaded: " +
                            std::to_string(state.project.camera_names.size()) +
                            " cameras (pointsource refinement)";
                    }
                } else {
                    state.status = "Error loading project: " + err;
                }
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (!state.show)
        return;

    // ---- Calibration Tool: Unified Creation Dialog + Tool Window ----

    // Phase 1: Show creation dialog when no project is loaded
    if (state.show_create_dialog && !state.project_loaded) {
        DrawCalibCreateDialog(state, ctx, cb);

    } else if (state.project_loaded) {
        // Phase 2: Unified Calibration Tool window (project loaded)
        // Clear stale dock-pending flag (no longer auto-docking)
        state.dock_pending = false;
        ImGui::SetNextWindowSize(ImVec2(580, 600), ImGuiCond_FirstUseEver);
        if (state.pointsource_focus_window) {
            ImGui::SetNextWindowFocus();
            state.pointsource_focus_window = false;
        }
        if (ImGui::Begin("Calibration Tool", &state.show)) {

            // Deferred label import: wait a few frames for dock layout
            if (state.tele_deferred_label_frames > 0) {
                state.tele_deferred_label_frames--;
                if (state.tele_deferred_label_frames == 0 &&
                    state.tele_videos_loaded) {
                    int n_lm = CalibrationTool::count_landmarks_3d(
                        state.project.landmarks_3d_file);
                    if (n_lm > 0) {
                        setup_landmark_skeleton(ctx.skeleton, n_lm, pm,
                                                 state.project.project_path);
                        std::string labels_dir =
                            state.project.effective_labels_folder();
                        int imported = TelecentricDLT::import_dlt_labels(
                            ctx.annotations, 0, n_lm,
                            (int)scene->num_cams, pm.camera_names,
                            labels_dir);
                        if (imported > 0)
                            state.status = "Loaded " +
                                std::to_string(imported) +
                                " labels. Labeling active.";
                    }
                }
            }

            // ---- Section 1: Project Info ----
            if (ImGui::CollapsingHeader("Project", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();
                ImGui::Text("Project: %s",
                            state.project.project_name.c_str());
                ImGui::Text("Path:    %s",
                            state.project.project_path.c_str());
                ImGui::Unindent();
            }
            ImGui::Spacing();

            // ---- Section: Telecentric DLT Calibration (if telecentric) ----
            if (state.project.is_telecentric()) {
                DrawCalibTeleSection(state, ctx, cb);
            }
            ImGui::Spacing();

            // ---- Section 2: Aruco Calibration (if has_aruco) ----
            if (state.project.has_aruco() && state.config_loaded) {
                DrawCalibArucoSection(state, ctx, cb);
            }

            // ---- Section 3: PointSource Refinement ----
            bool aruco_succeeded =
                (state.aruco_done && state.aruco_result.success);
            bool show_pointsource_section =
                state.project.has_laser_input() ||
                state.project.has_pointsource_videos() ||  // No Init: videos alone sufficient
                aruco_succeeded;

            // Auto-populate pointsource calibration_folder from aruco output
            if (state.project.calibration_folder.empty()) {
                // Check unified output first, then legacy
                std::string aruco_out = state.project.aruco_output_folder;
                if (aruco_out.empty())
                    aruco_out = state.project.video_output_folder.empty()
                        ? state.project.image_output_folder
                        : state.project.video_output_folder;
                if (!aruco_out.empty() &&
                    CalibrationPipeline::has_calibration_database(aruco_out)) {
                    state.project.calibration_folder = aruco_out;
                    state.project.camera_names =
                        CalibrationTool::derive_camera_names_from_yaml(
                            state.project.calibration_folder);
                    state.project.pointsource_output_folder =
                        state.project.project_path + "/pointsource_calibration";
                    show_pointsource_section = true;
                    std::string proj_file =
                        state.project.project_path + "/" +
                        state.project.project_name + ".redproj";
                    std::string save_err;
                    CalibrationTool::save_project(
                        state.project, proj_file, &save_err);
                }
            }

            if (show_pointsource_section) {
                DrawCalibPointSourceSection(state, ctx, cb);
            }

            // SuperPoint refinement section (available after any calibration succeeds)
            DrawCalibSuperPointSection(state, ctx, cb);

            // Manual keypoint calibration refinement
            DrawCalibKPManualSection(state, ctx, cb);

            // Status text (general)
            if (!state.status.empty()) {
                ImGui::Separator();
                if (state.status.find("Error") != std::string::npos) {
                    ImGui::TextColored(
                        ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s",
                        state.status.c_str());
                } else {
                    ImGui::Text("%s", state.status.c_str());
                }
            }
        }
        ImGui::End();
    }

    // Draw 3D viewers as separate windows (if active)
    DrawCalibViewerWindow(state.calib_viewer);
    DrawTeleViewerWindow(state.tele_viewer);

    // Reset state when calibration tool window is closed
    if (!state.show) {
        state.calib_viewer.show = false;
        state.tele_viewer.show = false;
        state.project_loaded = false;
        state.show_create_dialog = true;
        state.config_loaded = false;
        state.images_loaded = false;
        // Unified aruco pipeline
        state.aruco_running_flag = false;
        state.aruco_done = false;
        state.aruco_media_loaded = false;
        // PointSource refinement
        state.pointsource_running = false;
        state.pointsource_done = false;
        state.pointsource_ready = false;
        state.pointsource_status.clear();
        state.pointsource_show_detection = false;
        // SuperPoint refinement
        state.sp_running = false;
        state.sp_done = false;
        state.sp_status.clear();
        // Manual keypoint refinement
        state.kp_skeleton_ready = false;
        state.kp_running = false;
        state.kp_refine_done = false;
        state.kp_videos_loaded = false;
        state.kp_status.clear();
        // Telecentric
        state.tele_videos_loaded = false;
        state.tele_dlt_running = false;
        state.tele_dlt_done = false;
        state.tele_dlt_status.clear();
        state.tele_run_history.clear();
        state.tele_deferred_label_frames = 0;
        // General
        state.status.clear();
        state.project.camera_names.clear();
        if (state.pointsource_viz.worker.joinable())
            state.pointsource_viz.worker.join();
        state.pointsource_viz.ready.clear();
        state.pointsource_viz.pending.clear();
#ifdef __APPLE__
        for (int ci = 0; ci < scene->num_cams; ci++)
            mac_last_uploaded_frame[ci] = -1;
#endif
    }
}
