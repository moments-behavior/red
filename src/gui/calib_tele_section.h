#pragma once
#include "calib_tool_state.h"
#include "app_context.h"
#include "telecentric_dlt.h"
#include "reprojection_diagnostics.h"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <filesystem>
#include <string>

// Draw the Telecentric DLT Calibration section inside the Calibration Tool window.
// Called only when state.project.is_telecentric().
inline void DrawCalibTeleSection(CalibrationToolState &state, AppContext &ctx,
                                  const CalibrationToolCallbacks &cb) {
    auto &pm = ctx.pm;
    auto &ps = ctx.ps;
    auto *scene = ctx.scene;
    auto *dc_context = ctx.dc_context;
    auto &imgs_names = ctx.imgs_names;
#ifdef __APPLE__
    auto &mac_last_uploaded_frame = ctx.mac_last_uploaded_frame;
#endif

    // Helper: enqueue a deferred close-media action
    auto close_media_deferred = [&](bool &loaded_flag, std::string &status_out,
                                    const char *msg) {
        cb.deferred->enqueue([&loaded_flag, &status_out, &cb, &imgs_names, msg
#ifdef __APPLE__
            , &mac_last_uploaded_frame
#endif
        ]() {
            cb.unload_media();
            imgs_names.clear();
#ifdef __APPLE__
            for (size_t ci = 0; ci < mac_last_uploaded_frame.size(); ci++)
                mac_last_uploaded_frame[ci] = -1;
#endif
            loaded_flag = false;
            status_out = msg;
        });
    };

    if (ImGui::CollapsingHeader("Telecentric DLT Calibration", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                // Camera info
                {
                    std::string cam_list;
                    for (int i = 0; i < (int)state.project.camera_names.size(); i++) {
                        if (i > 0) cam_list += ", ";
                        cam_list += "Cam" + state.project.camera_names[i];
                    }
                    ImGui::Text("Cameras:    %d (%s)",
                                (int)state.project.camera_names.size(),
                                cam_list.c_str());
                }
                ImGui::Text("Videos:     %s",
                            state.project.media_folder.c_str());
                ImGui::Text("Labels:     %s",
                            state.project.landmark_labels_folder.c_str());
                ImGui::Text("3D File:    %s",
                            state.project.landmarks_3d_file.c_str());

                // Count 3D landmarks (cached)
                static std::string cached_3d_path;
                static int cached_3d_count = 0;
                if (cached_3d_path != state.project.landmarks_3d_file) {
                    cached_3d_path = state.project.landmarks_3d_file;
                    cached_3d_count = CalibrationTool::count_landmarks_3d(cached_3d_path);
                }
                ImGui::Text("Landmarks:  %d points", cached_3d_count);

                ImGui::Spacing();

                // Per-camera label coverage (cached)
                static std::string cached_label_folder;
                static std::vector<std::string> cached_label_cams;
                static std::map<std::string, int> cached_label_counts;
                if (cached_label_folder != state.project.landmark_labels_folder ||
                    cached_label_cams != state.project.camera_names) {
                    cached_label_folder = state.project.landmark_labels_folder;
                    cached_label_cams = state.project.camera_names;
                    cached_label_counts =
                        CalibrationTool::validate_telecentric_labels(
                            cached_label_folder, cached_label_cams);
                }
                ImGui::Text("Label coverage:");
                ImGui::Indent();
                for (const auto &serial : state.project.camera_names) {
                    auto it = cached_label_counts.find(serial);
                    if (it != cached_label_counts.end()) {
                        int count = it->second;
                        bool full = (cached_3d_count > 0 && count >= cached_3d_count);
                        ImVec4 col = full ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f)
                                         : ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
                        ImGui::TextColored(col, "Cam%s: %d/%d %s",
                            serial.c_str(), count, cached_3d_count,
                            full ? "\xe2\x9c\x93" : "");
                    } else {
                        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                            "Cam%s: no labels \xe2\x9c\x97", serial.c_str());
                    }
                }
                ImGui::Unindent();

                ImGui::Spacing();
                ImGui::Separator();

                // Load Videos button
                if (!state.tele_videos_loaded) {
                    ImGui::BeginDisabled(
                        state.project.camera_names.empty() ||
                        state.project.media_folder.empty());
                    if (ImGui::Button("Load Videos##tele_vid")) {
                        state.status = "Loading telecentric videos...";
                        cb.deferred->enqueue([&state, &pm, &ps, &cb,
                                              &imgs_names, &ctx, dc_context, scene
#ifdef __APPLE__
                                              , &mac_last_uploaded_frame
#endif
                        ]() {
                            try {
                                if (ps.video_loaded)
                                    cb.unload_media();
                                imgs_names.clear();
#ifdef __APPLE__
                                for (size_t ci = 0;
                                     ci < mac_last_uploaded_frame.size(); ci++)
                                    mac_last_uploaded_frame[ci] = -1;
#endif
                                pm.media_folder =
                                    state.project.media_folder;
                                pm.camera_names.clear();
                                for (const auto &cn :
                                     state.project.camera_names)
                                    pm.camera_names.push_back("Cam" + cn);
                                cb.load_videos();
                                cb.print_metadata();
                                state.tele_videos_loaded = true;

                                // Auto-setup skeleton + import labels
                                int n_lm = CalibrationTool::count_landmarks_3d(
                                    state.project.landmarks_3d_file);
                                if (n_lm > 0) {
                                    setup_landmark_skeleton(ctx.skeleton, n_lm,
                                                             pm, state.project.project_path);

                                    std::string labels_dir =
                                        state.project.effective_labels_folder();
                                    int imported = TelecentricDLT::import_dlt_labels(
                                        ctx.annotations, 0, n_lm,
                                        (int)scene->num_cams, pm.camera_names,
                                        labels_dir);

                                    state.status =
                                        "Loaded " + std::to_string(
                                            state.project.camera_names.size()) +
                                        " videos, " + std::to_string(imported) +
                                        " labels. Labeling active.";
                                } else {
                                    state.status =
                                        "Loaded " + std::to_string(
                                            state.project.camera_names.size()) +
                                        " telecentric videos";
                                }
                            } catch (const std::exception &e) {
                                state.status =
                                    std::string("Error loading videos: ") +
                                    e.what();
                            }
                        });
                    }
                    ImGui::EndDisabled();
                } else {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                       "Videos loaded");
                    ImGui::SameLine();
                    if (ImGui::Button("Close Videos##tele_close")) {
                        close_media_deferred(state.tele_videos_loaded, state.status, "Videos closed");
                    }
                }

                // ---- Start Labeling button ----
                if (state.tele_videos_loaded && cached_3d_count > 0) {
                    ImGui::Spacing();
                    auto &skeleton = ctx.skeleton;
                    bool labeling_active = skeleton.has_skeleton &&
                                           skeleton.num_nodes == cached_3d_count &&
                                           pm.plot_keypoints_flag;
                    if (!labeling_active) {
                        if (ImGui::Button("Start Labeling##tele_label")) {
                            setup_landmark_skeleton(skeleton, cached_3d_count,
                                                     pm, state.project.project_path);
                            state.status = "Labeling enabled (" +
                                std::to_string(cached_3d_count) +
                                " landmarks). Use Labeling Tool to annotate.";
                        }
                        ImGui::SameLine();
                        ImGui::TextDisabled(
                            "Set up %d-point skeleton for labeling",
                            cached_3d_count);
                    } else {
                        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                            "Labeling active (%d landmarks)", cached_3d_count);

                        // Export Labels for DLT
                        ImGui::SameLine();
                        if (ImGui::Button("Export Labels for DLT##tele")) {
                            std::string export_folder =
                                state.project.project_path + "/red_data";
                            int frame_num = ps.to_display_frame_number;
                            // Use pm.camera_names (has "Cam" prefix)
                            auto export_result =
                                TelecentricDLT::export_labels_for_dlt(
                                    ctx.annotations, frame_num,
                                    skeleton.num_nodes,
                                    (int)scene->num_cams,
                                    pm.camera_names,
                                    export_folder,
                                    skeleton.name);

                            if (export_result.success) {
                                state.project.landmark_labels_folder =
                                    export_folder;
                                char msg[256];
                                snprintf(msg, sizeof(msg),
                                    "Exported %d labels across %d cameras "
                                    "to %s",
                                    export_result.total_labeled,
                                    export_result.num_cameras,
                                    export_folder.c_str());
                                ctx.toasts.pushSuccess(msg);
                                state.status = msg;
                            } else {
                                ctx.toasts.pushError(
                                    "Export failed: " +
                                    export_result.error);
                            }
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Export labels from current frame as "
                                "DLT-format Cam*.csv files");
                        }
                    }
                }

                ImGui::Spacing();
                ImGui::Separator();

                // ---- DLT Options ----
                ImGui::Text("Calibration Method:");
                const char *method_items[] = {
                    "Linear DLT (affine only)",
                    "DLT + k1 (radial distortion)",
                    "DLT + k1,k2 (radial distortion)"
                };
                ImGui::Combo("##tele_method", &state.tele_method,
                             method_items, 3);

                ImGui::Spacing();
                ImGui::Text("Options:");
                ImGui::Checkbox("Flip Y coordinates##tele", &state.tele_flip_y);
                ImGui::SameLine();
                ImGui::TextDisabled("(MATLAB convention)");
                ImGui::Checkbox("Square pixels##tele", &state.tele_square_pixels);
                ImGui::Checkbox("Zero skew##tele", &state.tele_zero_skew);
                if (state.tele_method == 0) {
                    ImGui::Checkbox("Bundle adjustment##tele", &state.tele_do_ba);
                } else {
                    ImGui::TextDisabled("Bundle adjustment: always on (estimates distortion)");
                }

                ImGui::Spacing();

                // ---- Poll async DLT future ----
                if (state.tele_dlt_running && state.tele_dlt_future.valid()) {
                    auto fut_status = state.tele_dlt_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fut_status == std::future_status::ready) {
                        state.tele_dlt_result = state.tele_dlt_future.get();
                        state.tele_dlt_running = false;
                        state.tele_dlt_done = true;
                        if (state.tele_dlt_result.success) {
                            char buf[256];
                            snprintf(buf, sizeof(buf),
                                     "%s complete. Mean RMSE: %.4f px",
                                     TelecentricDLT::method_name(
                                         state.tele_dlt_result.method),
                                     state.tele_dlt_result.mean_rmse);
                            state.tele_dlt_status = buf;
                            state.project.tele_output_folder =
                                state.tele_dlt_result.output_folder;
                            // Add to run history for comparison
                            state.tele_run_history.push_back(
                                state.tele_dlt_result);
                            // Persist result metadata in project
                            state.project.dlt_method =
                                static_cast<int>(state.tele_dlt_result.method);
                            state.project.dlt_mean_rmse =
                                state.tele_dlt_result.mean_rmse;
                            state.project.dlt_per_camera_rmse.clear();
                            for (const auto &cam : state.tele_dlt_result.cameras) {
                                state.project.dlt_per_camera_rmse.push_back(cam.final_rmse());
                            }
                            // Auto-save project
                            std::string proj_file =
                                state.project.project_path + "/" +
                                state.project.project_name + ".redproj";
                            std::string save_err;
                            CalibrationTool::save_project(
                                state.project, proj_file, &save_err);
                        } else {
                            state.tele_dlt_status =
                                "Error: " + state.tele_dlt_result.error;
                        }
                    }
                }

                // ---- Run DLT Calibration button ----
                {
                    bool can_run = !state.tele_dlt_running &&
                                   !state.project.camera_names.empty() &&
                                   state.project.has_telecentric_input();
                    ImGui::BeginDisabled(!can_run);
                    if (ImGui::Button("Run DLT Calibration")) {
                        state.tele_dlt_running = true;
                        state.tele_dlt_done = false;
                        state.tele_dlt_status = "Starting DLT calibration...";

                        TelecentricDLT::DLTConfig dlt_cfg;
                        dlt_cfg.camera_names = state.project.camera_names;
                        dlt_cfg.landmark_labels_folder =
                            state.project.landmark_labels_folder;
                        dlt_cfg.landmarks_3d_file =
                            state.project.landmarks_3d_file;
                        const char *method_suffix[] = {
                            "dlt_linear", "dlt_k1", "dlt_k1k2"};
                        dlt_cfg.output_folder =
                            state.project.project_path + "/" +
                            method_suffix[state.tele_method];
                        dlt_cfg.flip_y = state.tele_flip_y;
                        dlt_cfg.square_pixels = state.tele_square_pixels;
                        dlt_cfg.zero_skew = state.tele_zero_skew;
                        dlt_cfg.do_ba = state.tele_do_ba;
                        dlt_cfg.method = static_cast<TelecentricDLT::Method>(
                            state.tele_method);

                        // Get image dimensions from loaded video
                        if (scene->num_cams > 0) {
                            dlt_cfg.image_width = scene->image_width[0];
                            dlt_cfg.image_height = scene->image_height[0];
                        }

                        state.tele_dlt_future = std::async(
                            std::launch::async,
                            [dlt_cfg,
                             status_ptr = &state.tele_dlt_status]() {
                                return TelecentricDLT::run_dlt_calibration(
                                    dlt_cfg, status_ptr);
                            });
                    }
                    ImGui::EndDisabled();

                    if (state.tele_dlt_running) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                           "Running...");
                    }
                }

                // ---- DLT Status ----
                if (!state.tele_dlt_status.empty()) {
                    ImGui::TextWrapped("Status: %s",
                                       state.tele_dlt_status.c_str());
                }

                // ---- DLT Results (latest run) ----
                if (state.tele_dlt_done && state.tele_dlt_result.success) {
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Text("Latest: %s",
                        TelecentricDLT::method_name(
                            state.tele_dlt_result.method));

                    bool has_dist = (state.tele_dlt_result.method !=
                                     TelecentricDLT::Method::LinearDLT);
                    int ncols = has_dist ? 6 : 4;
                    if (ImGui::BeginTable("##dlt_results", ncols,
                            ImGuiTableFlags_Borders |
                            ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Camera");
                        ImGui::TableSetupColumn("Points");
                        ImGui::TableSetupColumn("RMSE (init)");
                        ImGui::TableSetupColumn("RMSE (final)");
                        if (has_dist) {
                            ImGui::TableSetupColumn("k1");
                            ImGui::TableSetupColumn("k2");
                        }
                        ImGui::TableHeadersRow();

                        for (const auto &cam :
                             state.tele_dlt_result.cameras) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("Cam%s", cam.serial.c_str());
                            ImGui::TableNextColumn();
                            ImGui::Text("%d", cam.num_points);
                            ImGui::TableNextColumn();
                            ImGui::Text("%.4f", cam.rmse_init);
                            ImGui::TableNextColumn();
                            if (cam.rmse_ba > 0)
                                ImGui::Text("%.4f", cam.rmse_ba);
                            else
                                ImGui::TextDisabled("--");
                            if (has_dist) {
                                ImGui::TableNextColumn();
                                ImGui::Text("%.2e", cam.k1);
                                ImGui::TableNextColumn();
                                ImGui::Text("%.2e", cam.k2);
                            }
                        }
                        ImGui::EndTable();
                    }

                    ImGui::Text("Mean RMSE: %.4f px",
                                state.tele_dlt_result.mean_rmse);
                    ImGui::Text("Output: %s",
                                state.tele_dlt_result.output_folder.c_str());

                    // Cross-validation results
                    if (!state.tele_dlt_result.cv_results.empty()) {
                        ImGui::Spacing();
                        if (ImGui::TreeNode("Cross-Validation (LOO)")) {
                            if (ImGui::BeginTable("##cv_results", 4,
                                    ImGuiTableFlags_Borders |
                                    ImGuiTableFlags_RowBg)) {
                                ImGui::TableSetupColumn("Camera");
                                ImGui::TableSetupColumn("LOO RMSE");
                                ImGui::TableSetupColumn("Outliers");
                                ImGui::TableSetupColumn("Flagged Points");
                                ImGui::TableHeadersRow();

                                for (int m = 0; m < (int)state.tele_dlt_result.cv_results.size(); m++) {
                                    const auto &cv = state.tele_dlt_result.cv_results[m];
                                    ImGui::TableNextRow();
                                    ImGui::TableNextColumn();
                                    if (m < (int)state.tele_dlt_result.cameras.size())
                                        ImGui::Text("Cam%s", state.tele_dlt_result.cameras[m].serial.c_str());
                                    ImGui::TableNextColumn();
                                    ImGui::Text("%.4f", cv.loo_rmse);
                                    ImGui::TableNextColumn();
                                    if (!cv.outlier_indices.empty())
                                        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                                            "%d", (int)cv.outlier_indices.size());
                                    else
                                        ImGui::Text("0");
                                    ImGui::TableNextColumn();
                                    if (!cv.outlier_indices.empty()) {
                                        std::string pts;
                                        for (int idx : cv.outlier_indices) {
                                            if (!pts.empty()) pts += ", ";
                                            pts += std::to_string(idx);
                                        }
                                        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                                            "%s", pts.c_str());
                                    } else {
                                        ImGui::TextDisabled("--");
                                    }
                                }
                                ImGui::EndTable();
                            }
                            ImGui::TreePop();
                        }
                    }
                }

                // ---- Action Buttons ----
                if (state.tele_dlt_done && state.tele_dlt_result.success) {
                    ImGui::Spacing();

                    // 3D Viewer button
                    if (ImGui::Button("3D Viewer##tele")) {
                        state.tele_viewer.show = true;
                        state.tele_viewer.dlt_result = &state.tele_dlt_result;
                        // Load 3D landmarks if not already loaded
                        if (state.tele_viewer.landmarks_3d.empty()) {
                            state.tele_viewer.landmarks_3d =
                                TelecentricDLT::parse_3d_landmarks(
                                    state.project.landmarks_3d_file);
                        }
                    }

                }

                // ---- Run Comparison Table ----
                if (state.tele_run_history.size() > 1) {
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Text("Run Comparison:");
                    if (ImGui::BeginTable("##dlt_comparison",
                            3, ImGuiTableFlags_Borders |
                            ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Method");
                        ImGui::TableSetupColumn("Mean RMSE");
                        ImGui::TableSetupColumn("Output");
                        ImGui::TableHeadersRow();

                        for (const auto &run :
                             state.tele_run_history) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%s",
                                TelecentricDLT::method_name(run.method));
                            ImGui::TableNextColumn();
                            ImGui::Text("%.4f px", run.mean_rmse);
                            ImGui::TableNextColumn();
                            ImGui::TextWrapped("%s",
                                run.output_folder.c_str());
                        }
                        ImGui::EndTable();
                    }
                    if (ImGui::Button("Clear History##tele")) {
                        state.tele_run_history.clear();
                    }
                }

                ImGui::Spacing();

                // General status
                if (!state.status.empty()) {
                    ImGui::TextWrapped("Status: %s", state.status.c_str());
                }

                ImGui::Unindent();
            }

    // ─── Reprojection Diagnostics ─────────────────────────────────────────
    // Triangulate-and-reproject diagnostic. Uses saved DLT coefficients +
    // exported 2D labels. Pure: does not modify AnnotationMap or any fit.
    if (ImGui::CollapsingHeader("Reprojection Diagnostics")) {
        ImGui::Indent();
        ImGui::TextWrapped(
            "For each landmark, reproject its 3D position into every camera "
            "that labeled it, and report ||reproj - observed||. Read-only -- "
            "labels are not modified. Two modes:");
        ImGui::BulletText("Triangulate: 3D is computed from multi-view observations "
                          "(tests camera agreement).");
        ImGui::BulletText("Known 3D: 3D is read from landmarks_3d_file "
                          "(matches DLT fit RMSE convention).");

        ImGui::Spacing();

        // Mode selection
        const char *mode_items[] = {
            "Triangulate -> reproject",
            "Known 3D -> reproject"
        };
        ImGui::Combo("Mode##reproj_diag", &state.reproj_diag_mode,
                     mode_items, 2);

        // Pick DLT folder: prefer project's tele_output_folder (last-run dir);
        // otherwise match the currently selected method suffix.
        const char *method_suffix[] = {"dlt_linear", "dlt_k1", "dlt_k1k2"};
        std::string dlt_folder = state.project.tele_output_folder;
        if (dlt_folder.empty()) {
            dlt_folder = state.project.project_path + "/" +
                         method_suffix[state.tele_method];
        }
        std::string labels_folder =
            state.project.effective_labels_folder();
        std::string known_3d_file = (state.reproj_diag_mode == 1)
            ? state.project.landmarks_3d_file : std::string{};

        ImGui::Text("DLT folder:    %s", dlt_folder.c_str());
        ImGui::Text("Labels folder: %s", labels_folder.c_str());
        if (state.reproj_diag_mode == 1) {
            ImGui::Text("3D landmarks:  %s", known_3d_file.c_str());
        }

        int image_height = (ctx.scene && ctx.scene->num_cams > 0)
            ? ctx.scene->image_height[0] : 0;
        ImGui::Text("Image height:  %d px (from loaded videos)",
                    image_height);

        bool can_run = !state.project.camera_names.empty() &&
                       !dlt_folder.empty() &&
                       std::filesystem::is_directory(dlt_folder) &&
                       !labels_folder.empty() &&
                       std::filesystem::is_directory(labels_folder) &&
                       image_height > 0;
        if (state.reproj_diag_mode == 1) {
            can_run = can_run && !known_3d_file.empty() &&
                      std::filesystem::is_regular_file(known_3d_file);
        }

        ImGui::BeginDisabled(!can_run);
        if (ImGui::Button("Compute Reprojection Diagnostics")) {
            state.reproj_diag = ReprojectionDiagnostics::compute_from_disk(
                state.project.camera_names,
                dlt_folder,
                labels_folder,
                image_height,
                state.tele_flip_y,
                known_3d_file);
            state.reproj_diag_done = state.reproj_diag.success;
            if (state.reproj_diag.success) {
                char buf[256];
                snprintf(buf, sizeof(buf),
                    "%s: %d/%d points, %d residuals, overall RMSE %.4f px",
                    ReprojectionDiagnostics::mode_label(state.reproj_diag.mode),
                    state.reproj_diag.n_points_triangulated,
                    state.reproj_diag.n_points_total,
                    (int)state.reproj_diag.residuals.size(),
                    state.reproj_diag.overall_rmse);
                state.reproj_diag_status = buf;
            } else {
                state.reproj_diag_status =
                    "Error: " + state.reproj_diag.error;
            }
        }
        ImGui::EndDisabled();
        if (!can_run) {
            ImGui::SameLine();
            if (state.reproj_diag_mode == 1 &&
                (known_3d_file.empty() ||
                 !std::filesystem::is_regular_file(known_3d_file))) {
                ImGui::TextDisabled("(3D landmarks file missing)");
            } else {
                ImGui::TextDisabled("(load videos and run DLT first)");
            }
        }

        if (!state.reproj_diag_status.empty()) {
            ImGui::Spacing();
            ImGui::TextWrapped("%s", state.reproj_diag_status.c_str());
        }

        if (state.reproj_diag_done && state.reproj_diag.success) {
            const auto &rd = state.reproj_diag;

            ImGui::Spacing();
            ImGui::Separator();

            // ── Overall stats strip ──
            ImGui::Text("Mode: %s",
                ReprojectionDiagnostics::mode_label(rd.mode));
            ImGui::Text("Overall (%d residuals across %d points, %d skipped):",
                (int)rd.residuals.size(),
                rd.n_points_triangulated,
                rd.n_points_skipped);
            ImGui::Indent();
            ImGui::Text("Mean +- s.d.  %.4f +- %.4f px",
                        rd.overall_mean, rd.overall_std);
            ImGui::Text("Median        %.4f px", rd.overall_median);
            ImGui::SameLine(260);
            ImGui::Text("P95    %.4f px", rd.overall_p95);
            ImGui::Text("Max           %.4f px", rd.overall_max);
            ImGui::SameLine(260);
            ImGui::Text("RMSE   %.4f px", rd.overall_rmse);
            ImGui::Unindent();

            // ── Histogram ──
            if (!rd.residuals.empty()) {
                double hist_max = std::max(rd.overall_p95 * 1.25,
                                           rd.overall_max);
                hist_max = std::max(hist_max, 1.0);
                const int num_bins = 50;
                double bin_width = hist_max / num_bins;
                std::vector<double> bins(num_bins, 0);
                std::vector<double> centers(num_bins);
                for (int i = 0; i < num_bins; i++)
                    centers[i] = (i + 0.5) * bin_width;
                for (const auto &r : rd.residuals) {
                    int b = std::clamp((int)(r.error_px / bin_width),
                                       0, num_bins - 1);
                    bins[b]++;
                }
                if (ImPlot::BeginPlot("Reprojection Error Distribution",
                                      ImVec2(-1, 200))) {
                    ImPlot::SetupAxes("Error (px)", "Count");
                    ImPlot::PlotBars("residuals",
                                     centers.data(), bins.data(),
                                     num_bins, bin_width);
                    ImPlot::EndPlot();
                }
            }

            // ── Per-camera table ──
            if (ImGui::BeginTable("##reproj_diag_per_cam", 8,
                    ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Camera");
                ImGui::TableSetupColumn("N");
                ImGui::TableSetupColumn("Mean");
                ImGui::TableSetupColumn("s.d.");
                ImGui::TableSetupColumn("Median");
                ImGui::TableSetupColumn("P95");
                ImGui::TableSetupColumn("Max");
                ImGui::TableSetupColumn("RMSE");
                ImGui::TableHeadersRow();
                for (const auto &pc : rd.per_camera) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", pc.name.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", pc.n_obs);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pc.mean);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pc.std);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pc.median);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pc.p95);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pc.max);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pc.rmse);
                }
                ImGui::EndTable();
            }

            // ── Export CSV ──
            ImGui::Spacing();
            const char *csv_name = (rd.mode ==
                ReprojectionDiagnostics::Mode::Known3D)
                ? "/reprojection_errors_known3d.csv"
                : "/reprojection_errors_triangulated.csv";
            if (ImGui::Button("Export CSV##reproj_diag")) {
                std::string out_path = dlt_folder + csv_name;
                if (ReprojectionDiagnostics::save_csv(rd, out_path)) {
                    state.reproj_diag_status =
                        "Wrote " + std::to_string(rd.residuals.size()) +
                        " residuals to " + out_path;
                } else {
                    state.reproj_diag_status =
                        "Failed to write " + out_path;
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("-> %s%s", dlt_folder.c_str(), csv_name);
        }

        ImGui::Unindent();
    }
}
