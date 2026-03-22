#pragma once
#include "calib_tool_state.h"
#include "app_context.h"
#include "laser_calibration.h"
#include "imgui.h"
#include <ImGuiFileDialog.h>
#include <algorithm>
#include <string>

// Draw the Laser Refinement section inside the Calibration Tool window.
// Called only when show_laser_section is true.
inline void DrawCalibLaserSection(CalibrationToolState &state, AppContext &ctx,
                                   const CalibrationToolCallbacks &cb) {
    auto &pm = ctx.pm;
    auto &ps = ctx.ps;
    auto *scene = ctx.scene;
    auto *dc_context = ctx.dc_context;
    const auto &user_settings = ctx.user_settings;
    const auto &red_data_dir = ctx.red_data_dir;
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

                if (ImGui::CollapsingHeader("Laser Refinement", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                // Poll async result
                if (state.laser_running && state.laser_future.valid()) {
                    auto fut_status = state.laser_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fut_status == std::future_status::ready) {
                        state.laser_result = state.laser_future.get();
                        state.laser_running = false;
                        state.laser_done = true;
                        if (state.laser_result.success) {
                            state.laser_status =
                                "Complete! Reproj: " +
                                std::to_string(
                                    state.laser_result.mean_reproj_before)
                                    .substr(0, 5) +
                                " -> " +
                                std::to_string(
                                    state.laser_result.mean_reproj_after)
                                    .substr(0, 5) +
                                " px. Output: " +
                                state.laser_result.output_folder;
                        } else {
                            state.laser_status =
                                "Error: " + state.laser_result.error;
                        }
                    }
                }

                ImGui::Text("Calibration: %s",
                            state.project.calibration_folder.c_str());

                // Laser Video Folder -- text field + Browse + Load button
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 200.0f);
                ImGui::InputText("##laser_vid_path",
                                 &state.project.media_folder);
                ImGui::SameLine();
                if (ImGui::Button("Browse##laser_vid_tool")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    if (!state.project.media_folder.empty())
                        cfg.path = state.project.media_folder;
                    else if (!user_settings.default_media_root_path.empty())
                        cfg.path = user_settings.default_media_root_path;
                    else
                        cfg.path = red_data_dir;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseLaserVideoTool",
                        "Select Laser Video Folder", nullptr, cfg);
                }
                ImGui::SameLine();
                {
                    // Validate cameras only when paths change (avoid per-frame filesystem I/O)
                    static std::string last_media, last_calib;
                    bool paths_changed = (state.project.media_folder != last_media ||
                                          state.project.calibration_folder != last_calib);
                    if (paths_changed &&
                        !state.project.media_folder.empty() &&
                        !state.project.calibration_folder.empty()) {
                        last_media = state.project.media_folder;
                        last_calib = state.project.calibration_folder;
                        state.project.camera_names = LaserCalibration::validate_cameras(
                            state.project.media_folder,
                            state.project.calibration_folder);
                    }
                    bool has_valid_cameras = !state.project.camera_names.empty();

                    ImGui::BeginDisabled(
                        state.project.media_folder.empty() ||
                        !has_valid_cameras || state.laser_running);
                    if (!state.laser_ready) {
                    if (ImGui::Button("Load Laser Videos")) {
                        // Save updated media_folder to .redproj
                        state.project.laser_output_folder =
                            state.project.project_path + "/laser_calibration";
                        std::string proj_file =
                            state.project.project_path + "/" +
                            state.project.project_name + ".redproj";
                        std::string save_err;
                        CalibrationTool::save_project(
                            state.project, proj_file, &save_err);

                        // Set up laser_config
                        state.laser_config.media_folder = state.project.media_folder;
                        state.laser_config.calibration_folder =
                            state.project.calibration_folder;
                        state.laser_config.camera_names = state.project.camera_names;
                        state.laser_config.output_folder =
                            state.project.laser_output_folder;

                        // Defer the actual unload+load to next frame start
                        // (freeing Metal textures mid-frame crashes ImGui rendering)
                        state.laser_status = "Loading laser videos...";
                        cb.deferred->enqueue([&state, &pm, &ps, &cb, &imgs_names,
                                              dc_context
#ifdef __APPLE__
                                              , &mac_last_uploaded_frame
#endif
                        ]() {
                            try {
                                if (ps.video_loaded)
                                    cb.unload_media();
                                imgs_names.clear();
#ifdef __APPLE__
                                for (size_t ci = 0; ci < mac_last_uploaded_frame.size(); ci++)
                                    mac_last_uploaded_frame[ci] = -1;
#endif
                                pm.media_folder = state.project.media_folder;
                                for (const auto &cn : state.project.camera_names)
                                    pm.camera_names.push_back("Cam" + cn);
                                cb.load_videos();
                                cb.print_metadata();
                                state.laser_total_frames = dc_context->estimated_num_frames;
                                state.laser_ready = true;
                                state.laser_status =
                                    "Loaded " +
                                    std::to_string(state.project.camera_names.size()) +
                                    " laser videos";
                            } catch (const std::exception &e) {
                                state.laser_status =
                                    std::string("Error loading videos: ") + e.what();
                            }
                        });
                    }
                    } else {
                        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                           "Videos loaded");
                        ImGui::SameLine();
                        if (ImGui::Button("Close Videos##laser_close")) {
                            close_media_deferred(state.laser_ready, state.laser_status, "Videos closed");
                        }
                    }
                    ImGui::EndDisabled();
                }

                // Handle video folder browse dialog
                if (ImGuiFileDialog::Instance()->Display(
                        "ChooseLaserVideoTool", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                    if (ImGuiFileDialog::Instance()->IsOk()) {
                        state.project.media_folder =
                            ImGuiFileDialog::Instance()->GetCurrentPath();
                    }
                    ImGuiFileDialog::Instance()->Close();
                }

                // Show matched cameras
                if (!state.project.camera_names.empty()) {
                    char cam_header[64];
                    snprintf(cam_header, sizeof(cam_header), "Cameras (%d matched)",
                             (int)state.project.camera_names.size());
                    if (ImGui::CollapsingHeader(cam_header, ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();
                        if (ImGui::BeginTable("##cam_grid", 4)) {
                            for (int i = 0; i < (int)state.project.camera_names.size(); i++) {
                                ImGui::TableNextColumn();
                                ImGui::Text("Cam%s", state.project.camera_names[i].c_str());
                            }
                            ImGui::EndTable();
                        }
                        ImGui::Unindent();
                    }
                } else if (!state.project.media_folder.empty() &&
                           !state.project.calibration_folder.empty()) {
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                        "No matching videos found -- update the video folder path");
                }

                if (state.laser_ready) {
                    // Detection parameters
                    if (ImGui::CollapsingHeader("Detection Parameters")) {
                    ImGui::Indent();
                    ImGui::SliderInt("Green Threshold",
                                     &state.laser_config.green_threshold, 20, 255);
                    ImGui::SliderInt("Green Dominance",
                                     &state.laser_config.green_dominance, 5, 100);
                    ImGui::SliderInt("Min Blob Pixels",
                                     &state.laser_config.min_blob_pixels, 1, 100);
                    ImGui::SliderInt("Max Blob Pixels",
                                     &state.laser_config.max_blob_pixels, 50, 5000);

                    int slider_max = state.laser_total_frames > 0 ? state.laser_total_frames : 100000;
                    ImGui::SliderInt("Start Frame",
                                     &state.laser_config.start_frame, 0, slider_max);
                    ImGui::SliderInt("Stop Frame (0=all)",
                                     &state.laser_config.stop_frame, 0, slider_max);
                    ImGui::SliderInt("Every Nth Frame",
                                     &state.laser_config.frame_step, 1, 100);
                    {
                        int eff_stop = state.laser_config.stop_frame > 0
                            ? state.laser_config.stop_frame
                            : (state.laser_total_frames > 0 ? state.laser_total_frames : 0);
                        if (eff_stop > state.laser_config.start_frame && state.laser_config.frame_step > 0) {
                            int est = (eff_stop - state.laser_config.start_frame) / state.laser_config.frame_step;
                            ImGui::Text("~%d frames per camera", est);
                        } else if (state.laser_config.stop_frame == 0 && state.laser_total_frames == 0) {
                            ImGui::Text("~all frames per camera");
                        }
                    }
                    ImGui::Unindent();
                    } // end Detection Parameters

                    // Filtering parameters
                    if (ImGui::CollapsingHeader("Filtering")) {
                    ImGui::Indent();
                    int max_min_cams =
                        std::max(2, (int)state.laser_config.camera_names.size());
                    ImGui::SliderInt("Min Cameras",
                                     &state.laser_config.min_cameras, 2,
                                     max_min_cams);
                    float reproj_th = (float)state.laser_config.reproj_threshold;
                    if (ImGui::SliderFloat("Reproj Threshold (px)", &reproj_th,
                                           1.0f, 50.0f))
                        state.laser_config.reproj_threshold = reproj_th;
                    ImGui::Unindent();
                    } // end Filtering

                    // BA parameters
                    if (ImGui::CollapsingHeader("Bundle Adjustment")) {
                    ImGui::Indent();
                    float ba_th1 = (float)state.laser_config.ba_outlier_th1;
                    float ba_th2 = (float)state.laser_config.ba_outlier_th2;
                    if (ImGui::SliderFloat("BA Outlier Pass 1 (px)", &ba_th1,
                                           1.0f, 50.0f))
                        state.laser_config.ba_outlier_th1 = ba_th1;
                    if (ImGui::SliderFloat("BA Outlier Pass 2 (px)", &ba_th2,
                                           0.5f, 20.0f))
                        state.laser_config.ba_outlier_th2 = ba_th2;
                    ImGui::SliderInt("BA Max Iterations",
                                     &state.laser_config.ba_max_iter, 10, 200);
                    // Optimization mode dropdown
                    {
                        const char *opt_labels[] = {
                            "Extrinsics only",
                            "Extrinsics + focal length",
                            "Extrinsics + all intrinsics",
                            "Full (all parameters free)"};
                        int opt_idx = static_cast<int>(state.laser_config.opt_mode);
                        if (ImGui::Combo("Optimization Mode", &opt_idx, opt_labels, 4))
                            state.laser_config.opt_mode =
                                static_cast<LaserCalibration::LaserOptMode>(opt_idx);
                        // Sync legacy field
                        state.laser_config.lock_intrinsics =
                            (state.laser_config.opt_mode == LaserCalibration::LaserOptMode::ExtrinsicsOnly);
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Extrinsics only: Fix all intrinsics, optimize camera poses.\n"
                            "  Recommended when aruco intrinsics are trusted.\n\n"
                            "Extrinsics + focal length: Also refine fx, fy.\n"
                            "  Use when focal length may have drifted.\n\n"
                            "Extrinsics + all intrinsics: Refine fx, fy, cx, cy, k1, k2.\n"
                            "  Locks p1, p2, k3. Use when initial calibration is poor.\n\n"
                            "Full: All parameters free including p1, p2, k3.\n"
                            "  Only recommended with high-quality laser data (many points,\n"
                            "  good spatial coverage, high camera redundancy).");
                    ImGui::Unindent();
                    } // end Bundle Adjustment

                    ImGui::Separator();

                    // Run button
                    bool can_run_laser = !state.laser_running &&
                                       !state.laser_config.camera_names.empty();
                    ImGui::BeginDisabled(!can_run_laser);
                    if (ImGui::Button("Run Laser Refinement")) {
                        state.laser_running = true;
                        state.laser_done = false;
                        state.laser_status =
                            "Starting laser calibration pipeline...";
                        state.laser_future = std::async(
                            std::launch::async,
                            [config = state.laser_config,
                             status_ptr = &state.laser_status,
                             prog = state.laser_progress]() {
                                return LaserCalibration::
                                    run_laser_refinement(config, status_ptr,
                                                         prog.get());
                            });
                    }
                    ImGui::EndDisabled();

                    if (state.laser_running) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                           "Running...");
                    }

                    // Progress sub-section (visible during/after detection)
                    if (state.laser_running && !state.laser_progress->cameras.empty()) {
                        if (ImGui::CollapsingHeader("Progress", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();
                            int eff_stop = state.laser_config.stop_frame > 0
                                ? state.laser_config.stop_frame
                                : (state.laser_total_frames > 0 ? state.laser_total_frames : 0);
                            int eff_start = state.laser_config.start_frame;
                            int eff_step = std::max(1, state.laser_config.frame_step);
                            int prog_total = (eff_stop > eff_start)
                                ? (eff_stop - eff_start + eff_step - 1) / eff_step
                                : 0;
                            int total_done = 0;
                            for (int ci = 0; ci < (int)state.laser_progress->cameras.size(); ci++)
                                if (state.laser_progress->cameras[ci]->done.load(std::memory_order_relaxed))
                                    total_done++;
                            ImGui::Text("Detection: %d / %d cameras complete",
                                        total_done, (int)state.laser_progress->cameras.size());

                            if (ImGui::BeginTable(
                                    "laser_det_progress", 4,
                                    ImGuiTableFlags_RowBg |
                                        ImGuiTableFlags_BordersInnerV)) {
                                ImGui::TableSetupColumn("Camera", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                                ImGui::TableSetupColumn("Progress", ImGuiTableColumnFlags_WidthStretch);
                                ImGui::TableSetupColumn("Spots", ImGuiTableColumnFlags_WidthFixed, 60.0f);
                                ImGui::TableSetupColumn("Rate", ImGuiTableColumnFlags_WidthFixed, 50.0f);
                                ImGui::TableHeadersRow();

                                for (int ci = 0; ci < (int)state.laser_progress->cameras.size(); ci++) {
                                    auto &cp = state.laser_progress->cameras[ci];
                                    int fr = cp->frames_processed.load(std::memory_order_relaxed);
                                    int sp = cp->spots_detected.load(std::memory_order_relaxed);
                                    bool dn = cp->done.load(std::memory_order_relaxed);
                                    float frac = prog_total > 0 ? (float)fr / prog_total : 0.0f;
                                    if (frac > 1.0f) frac = 1.0f;

                                    ImGui::TableNextRow();
                                    ImGui::TableSetColumnIndex(0);
                                    if (ci < (int)state.laser_config.camera_names.size())
                                        ImGui::Text("Cam%s", state.laser_config.camera_names[ci].c_str());
                                    ImGui::TableSetColumnIndex(1);
                                    char overlay[64];
                                    if (prog_total > 0)
                                        snprintf(overlay, sizeof(overlay), "%d / %d", fr, prog_total);
                                    else
                                        snprintf(overlay, sizeof(overlay), "%d", fr);
                                    if (dn)
                                        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
                                    ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0), overlay);
                                    if (dn)
                                        ImGui::PopStyleColor();
                                    ImGui::TableSetColumnIndex(2);
                                    ImGui::Text("%d", sp);
                                    ImGui::TableSetColumnIndex(3);
                                    ImGui::Text("%.0f%%", fr > 0 ? 100.0 * sp / fr : 0.0);
                                }
                                ImGui::EndTable();
                            }
                        ImGui::Unindent();
                        } // end Progress
                    }

                    // Results
                    if (state.laser_done && state.laser_result.success) {
                        if (ImGui::CollapsingHeader("Results", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();
                        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                           "Reprojection error: %.3f -> %.3f px",
                                           state.laser_result.mean_reproj_before,
                                           state.laser_result.mean_reproj_after);
                        double avg_obs = state.laser_result.valid_3d_points > 0
                            ? (double)state.laser_result.total_observations / state.laser_result.valid_3d_points
                            : 0.0;
                        ImGui::Text("3D points: %d | Observations: %d (avg %.1f cameras/point)",
                                    state.laser_result.valid_3d_points,
                                    state.laser_result.total_observations,
                                    avg_obs);
                        if (state.laser_result.ba_outliers_removed > 0)
                            ImGui::Text("BA outliers removed: %d", state.laser_result.ba_outliers_removed);
                        ImGui::Text("Output: %s",
                                    state.laser_result.output_folder.c_str());

                        // Per-camera changes table
                        if (!state.laser_result.camera_changes.empty()) {
                            if (ImGui::TreeNode("Per-camera changes")) {
                                if (ImGui::BeginTable(
                                        "laser_cam_changes", 9,
                                        ImGuiTableFlags_RowBg |
                                            ImGuiTableFlags_BordersInnerV |
                                            ImGuiTableFlags_Resizable |
                                            ImGuiTableFlags_SizingFixedFit)) {
                                    ImGui::TableSetupColumn("Camera", 0, 80.0f);
                                    ImGui::TableSetupColumn("Spots", 0, 45.0f);
                                    ImGui::TableSetupColumn("Obs", 0, 40.0f);
                                    ImGui::TableSetupColumn("dfx", 0, 55.0f);
                                    ImGui::TableSetupColumn("dfy", 0, 55.0f);
                                    ImGui::TableSetupColumn("dcx", 0, 55.0f);
                                    ImGui::TableSetupColumn("dcy", 0, 55.0f);
                                    ImGui::TableSetupColumn("|dt|", 0, 55.0f);
                                    ImGui::TableSetupColumn("drot", 0, 60.0f);
                                    ImGui::TableHeadersRow();

                                    for (const auto &cc : state.laser_result.camera_changes) {
                                        ImGui::TableNextRow();
                                        ImGui::TableSetColumnIndex(0);
                                        ImGui::Text("Cam%s", cc.name.c_str());
                                        ImGui::TableSetColumnIndex(1);
                                        ImGui::Text("%d", cc.detections);
                                        ImGui::TableSetColumnIndex(2);
                                        ImGui::Text("%d", cc.observations);
                                        ImGui::TableSetColumnIndex(3);
                                        ImGui::Text("%+.2f", cc.dfx);
                                        ImGui::TableSetColumnIndex(4);
                                        ImGui::Text("%+.2f", cc.dfy);
                                        ImGui::TableSetColumnIndex(5);
                                        ImGui::Text("%+.2f", cc.dcx);
                                        ImGui::TableSetColumnIndex(6);
                                        ImGui::Text("%+.2f", cc.dcy);
                                        ImGui::TableSetColumnIndex(7);
                                        ImGui::Text("%.3f", cc.dt_norm);
                                        ImGui::TableSetColumnIndex(8);
                                        ImGui::Text("%.4f%s", cc.drot_deg, "\xC2\xB0");
                                    }
                                    ImGui::EndTable();
                                }
                                ImGui::TreePop();
                            }
                        }
                        ImGui::Unindent();
                        } // end CollapsingHeader("Results")
                    }

                    // Laser status
                    if (!state.laser_status.empty()) {
                        ImGui::Separator();
                        if (state.laser_status.find("Error") !=
                            std::string::npos) {
                            ImGui::TextColored(
                                ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s",
                                state.laser_status.c_str());
                        } else {
                            ImGui::TextWrapped("%s", state.laser_status.c_str());
                        }
                    }

                    // Detection Processing visualization
                    if (ImGui::CollapsingHeader("Detection Processing")) {
                    ImGui::Indent();
                    bool prev_detection = state.laser_show_detection;
                    ImGui::Checkbox("Enable", &state.laser_show_detection);
                    if (prev_detection && !state.laser_show_detection) {
#ifdef __APPLE__
                        for (int ci = 0; ci < scene->num_cams; ci++)
                            mac_last_uploaded_frame[ci] = -1;
#endif
                        state.laser_viz.ready.clear();
                    }
                    if (state.laser_show_detection && ps.video_loaded) {
                        if (state.laser_viz.computing.load(std::memory_order_relaxed))
                            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                               "  Processing...");
                        int detecting_count = 0;
                        int total_cams = std::min((int)state.laser_viz.ready.size(), (int)scene->num_cams);
                        for (int ci = 0; ci < total_cams; ci++) {
                            auto &cr = state.laser_viz.ready[ci];
                            const char *blob_str =
                                cr.num_blobs == 0  ? "0 blobs" :
                                cr.num_blobs == 1  ? "1 blob (OK)" :
                                cr.num_blobs == -1 ? "invalid" :
                                "multiple blobs";
                            ImVec4 col = cr.num_blobs == 1
                                ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f)
                                : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
                            ImGui::TextColored(col, "  %s: %s, %d px",
                                ci < (int)pm.camera_names.size()
                                    ? pm.camera_names[ci].c_str() : "?",
                                blob_str, cr.total_mask_pixels);
                            if (cr.num_blobs == 1) detecting_count++;
                        }
                        if (total_cams > 0) {
                            ImVec4 summary_col = (detecting_count == total_cams)
                                ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f)
                                : ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
                            ImGui::TextColored(summary_col,
                                "  %d/%d cameras detecting", detecting_count, total_cams);
                        }
                        ImGui::TextWrapped(
                            "Green=valid blob, Yellow=too small, "
                            "Red=too large, Gray=filtered by erode/dilate");
                    }
                    ImGui::Unindent();
                    } // end Detection Processing
                } else {
                    // Laser inputs not yet complete -- show hint
                    if (state.project.media_folder.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                            "Set Video Folder to enable laser refinement");
                    } else if (state.project.camera_names.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                            "Set Video Folder and click Load Laser Videos");
                    }
                }
                ImGui::Unindent();
                } // end CollapsingHeader("Laser Refinement")
                ImGui::Spacing();
}
