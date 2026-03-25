#pragma once
#include "calib_tool_state.h"
#include "app_context.h"
#include "pointsource_calibration.h"
#include "imgui.h"
#include <ImGuiFileDialog.h>
#include <algorithm>
#include <string>

// Draw the PointSource Refinement section inside the Calibration Tool window.
// Called only when show_pointsource_section is true.
inline void DrawCalibPointSourceSection(CalibrationToolState &state, AppContext &ctx,
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

                if (ImGui::CollapsingHeader("PointSource Refinement", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                // Poll async result
                if (state.pointsource_running && state.pointsource_future.valid()) {
                    auto fut_status = state.pointsource_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fut_status == std::future_status::ready) {
                        state.pointsource_result = state.pointsource_future.get();
                        state.pointsource_running = false;
                        state.pointsource_done = true;
                        if (state.pointsource_result.success) {
                            state.pointsource_status =
                                "Complete! Reproj: " +
                                std::to_string(
                                    state.pointsource_result.mean_reproj_before)
                                    .substr(0, 5) +
                                " -> " +
                                std::to_string(
                                    state.pointsource_result.mean_reproj_after)
                                    .substr(0, 5) +
                                " px. Output: " +
                                state.pointsource_result.output_folder;
                        } else {
                            state.pointsource_status =
                                "Error: " + state.pointsource_result.error;
                        }
                    }
                }

                ImGui::Text("Calibration: %s",
                            state.project.calibration_folder.c_str());

                // PointSource Video Folder -- text field + Browse + Load button
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 290.0f);
                ImGui::InputText("##ps_vid_path",
                                 &state.project.media_folder);
                ImGui::SameLine();
                if (ImGui::Button("Browse##ps_vid_tool")) {
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
                        "ChoosePointSourceVideoTool",
                        "Select PointSource Video Folder", nullptr, cfg);
                }
                ImGui::SameLine();
                {
                    // Validate cameras only when paths change (avoid per-frame filesystem I/O)
                    static std::string last_media, last_calib;
                    bool paths_changed = (state.project.media_folder != last_media ||
                                          state.project.calibration_folder != last_calib);
                    if (paths_changed && !state.project.media_folder.empty()) {
                        last_media = state.project.media_folder;
                        last_calib = state.project.calibration_folder;
                        if (!state.project.calibration_folder.empty()) {
                            state.project.camera_names = PointSourceCalibration::validate_cameras(
                                state.project.media_folder,
                                state.project.calibration_folder);
                        } else {
                            // No Init: derive camera names from video filenames alone
                            state.project.camera_names = PointSourceCalibration::camera_names_from_videos(
                                state.project.media_folder);
                        }
                    }
                    // Sync to pointsource_config if videos are loaded but config.camera_names is stale
                    if (state.pointsource_ready &&
                        state.pointsource_config.camera_names.empty() &&
                        !state.project.camera_names.empty()) {
                        state.pointsource_config.camera_names = state.project.camera_names;
                    }
                    bool has_valid_cameras = !state.project.camera_names.empty();

                    ImGui::BeginDisabled(
                        state.project.media_folder.empty() ||
                        !has_valid_cameras || state.pointsource_running);
                    if (!state.pointsource_ready) {
                    if (ImGui::Button("Load PointSource Videos")) {
                        // Save updated media_folder to .redproj
                        state.project.pointsource_output_folder =
                            state.project.project_path + "/pointsource_calibration";
                        std::string proj_file =
                            state.project.project_path + "/" +
                            state.project.project_name + ".redproj";
                        std::string save_err;
                        CalibrationTool::save_project(
                            state.project, proj_file, &save_err);

                        // Set up pointsource_config
                        state.pointsource_config.media_folder = state.project.media_folder;
                        state.pointsource_config.calibration_folder =
                            state.project.calibration_folder;
                        state.pointsource_config.camera_names = state.project.camera_names;
                        state.pointsource_config.output_folder =
                            state.project.pointsource_output_folder;

                        // Auto-enable No Init for PointSourceFromScratch
                        if (state.project.subtype == CalibrationTool::CalibSubtype::PointSourceFromScratch &&
                            state.project.calibration_folder.empty() &&
                            !state.project.global_reg_media_folder.empty()) {
                            state.pointsource_config.no_init = true;
                            state.pointsource_config.loose_init = true;
                        }

                        // Defer the actual unload+load to next frame start
                        // (freeing Metal textures mid-frame crashes ImGui rendering)
                        state.pointsource_status = "Loading pointsource videos...";
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
                                // Query frame count from first PS camera (dc_context->estimated_num_frames
                                // is unreliable with mixed-fps cameras — last decoder thread wins)
                                if (!state.project.camera_names.empty()) {
                                    std::string first_vid = state.project.media_folder + "/Cam" +
                                        state.project.camera_names[0] + ".mp4";
                                    state.pointsource_total_frames =
                                        PointSourceCalibration::get_video_frame_count(first_vid);
                                }
                                if (state.pointsource_total_frames <= 0)
                                    state.pointsource_total_frames = dc_context->estimated_num_frames;
                                state.pointsource_ready = true;
                                state.pointsource_status =
                                    "Loaded " +
                                    std::to_string(state.project.camera_names.size()) +
                                    " pointsource videos";
                            } catch (const std::exception &e) {
                                state.pointsource_status =
                                    std::string("Error loading videos: ") + e.what();
                            }
                        });
                    }
                    } else {
                        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                           "Videos loaded");
                        ImGui::SameLine();
                        if (ImGui::Button("Close Videos##ps_close")) {
                            close_media_deferred(state.pointsource_ready, state.pointsource_status, "Videos closed");
                        }
                    }
                    ImGui::EndDisabled();
                }

                // Handle video folder browse dialog
                if (ImGuiFileDialog::Instance()->Display(
                        "ChoosePointSourceVideoTool", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                    if (ImGuiFileDialog::Instance()->IsOk()) {
                        state.project.media_folder =
                            ImGuiFileDialog::Instance()->GetCurrentPath();
                    }
                    ImGuiFileDialog::Instance()->Close();
                }
                // Handle global reg folder browse dialog
                if (ImGuiFileDialog::Instance()->Display(
                        "ChoosePSGlobalRegFolder", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                    if (ImGuiFileDialog::Instance()->IsOk()) {
                        state.project.global_reg_media_folder =
                            ImGuiFileDialog::Instance()->GetCurrentPath();
                        auto info = CalibrationTool::detect_aruco_media(
                            state.project.global_reg_media_folder);
                        state.project.global_reg_media_type = info.type;
                        state.pointsource_global_reg_info = info;
                    }
                    ImGuiFileDialog::Instance()->Close();
                }

                // Show matched cameras
                if (!state.project.camera_names.empty()) {
                    int ncams = (int)state.project.camera_names.size();

                    // Sync enable vector and frame count cache with camera list
                    if ((int)state.pointsource_camera_enabled.size() != ncams)
                        state.pointsource_camera_enabled.assign(ncams, true);
                    if ((int)state.pointsource_camera_frames.size() != ncams) {
                        state.pointsource_camera_frames.resize(ncams);
                        for (int i = 0; i < ncams; i++) {
                            std::string vid = state.project.media_folder + "/Cam" +
                                state.project.camera_names[i] + ".mp4";
                            state.pointsource_camera_frames[i] =
                                PointSourceCalibration::get_video_frame_count(vid);
                        }
                        // Set total frames from median of enabled cameras
                        std::vector<int> counts;
                        for (int i = 0; i < ncams; i++)
                            if (state.pointsource_camera_frames[i] > 0)
                                counts.push_back(state.pointsource_camera_frames[i]);
                        if (!counts.empty()) {
                            std::sort(counts.begin(), counts.end());
                            state.pointsource_total_frames = counts[counts.size() / 2];
                        }
                    }

                    // Detect frame count outliers (> 10% from median)
                    int median_frames = state.pointsource_total_frames;

                    int n_enabled = 0;
                    for (int i = 0; i < ncams; i++)
                        if (state.pointsource_camera_enabled[i]) n_enabled++;

                    char cam_header[64];
                    snprintf(cam_header, sizeof(cam_header), "Cameras (%d/%d enabled)",
                             n_enabled, ncams);
                    if (ImGui::CollapsingHeader(cam_header, ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();
                        // Enable All / Disable All buttons
                        if (ImGui::SmallButton("All##ps_cam")) {
                            for (int i = 0; i < ncams; i++)
                                state.pointsource_camera_enabled[i] = true;
                        }
                        ImGui::SameLine();
                        if (ImGui::SmallButton("None##ps_cam")) {
                            for (int i = 0; i < ncams; i++)
                                state.pointsource_camera_enabled[i] = false;
                        }
                        if (ImGui::BeginTable("##cam_grid", 3,
                                ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                ImGuiTableFlags_SizingFixedFit)) {
                            ImGui::TableSetupColumn("Camera", 0, 150.0f);
                            ImGui::TableSetupColumn("Frames", 0, 70.0f);
                            ImGui::TableSetupColumn("Status", 0, 80.0f);
                            ImGui::TableHeadersRow();
                            for (int i = 0; i < ncams; i++) {
                                ImGui::TableNextRow();
                                ImGui::TableSetColumnIndex(0);
                                char label[32];
                                snprintf(label, sizeof(label), "Cam%s", state.project.camera_names[i].c_str());
                                bool enabled = state.pointsource_camera_enabled[i];
                                if (ImGui::Checkbox(label, &enabled))
                                    state.pointsource_camera_enabled[i] = enabled;
                                ImGui::TableSetColumnIndex(1);
                                ImGui::Text("%d", state.pointsource_camera_frames[i]);
                                ImGui::TableSetColumnIndex(2);
                                int fc = state.pointsource_camera_frames[i];
                                if (fc == 0) {
                                    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "missing");
                                } else if (median_frames > 0 &&
                                           std::abs(fc - median_frames) > median_frames / 10) {
                                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "mismatch");
                                } else {
                                    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "ok");
                                }
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

                // ── Load Previous Results button ──
                // Shown when output folder exists on disk but results aren't loaded yet
                if (!state.pointsource_done &&
                    !state.project.pointsource_output_folder.empty()) {
                    std::string latest_folder =
                        PointSourceCalibration::find_latest_timestamped_subfolder(
                            state.project.pointsource_output_folder);
                    if (!latest_folder.empty()) {
                        namespace fs = std::filesystem;
                        if (fs::exists(latest_folder + "/summary_data/summary.json")) {
                            ImGui::Spacing();
                            // Extract just the timestamp for display
                            std::string ts_name = fs::path(latest_folder).filename().string();
                            if (ImGui::Button("Load Previous Results")) {
                                std::string load_err;
                                if (PointSourceCalibration::load_result_from_summary(
                                        latest_folder, state.pointsource_result, &load_err)) {
                                    state.pointsource_done = true;
                                    state.pointsource_status =
                                        "Loaded results from " + ts_name +
                                        ". Reproj: " +
                                        std::to_string(state.pointsource_result.mean_reproj_before).substr(0, 5) +
                                        " -> " +
                                        std::to_string(state.pointsource_result.mean_reproj_after).substr(0, 5) + " px";
                                    // Populate config camera_names from result if empty
                                    if (state.pointsource_config.camera_names.empty()) {
                                        for (const auto &cc : state.pointsource_result.camera_changes)
                                            state.pointsource_config.camera_names.push_back(cc.name);
                                    }
                                    // Ensure config fields are set for 3D viewer
                                    if (state.pointsource_config.output_folder.empty())
                                        state.pointsource_config.output_folder =
                                            state.project.pointsource_output_folder;
                                    state.pointsource_config.charuco_setup =
                                        state.project.charuco_setup;
                                } else {
                                    state.pointsource_status = "Error: " + load_err;
                                }
                            }
                            ImGui::SameLine();
                            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                                "(%s)", ts_name.c_str());
                        }
                    }
                }

                if (state.pointsource_ready) {
                    // Detection parameters
                    if (ImGui::CollapsingHeader("Detection Parameters")) {
                    ImGui::Indent();
                    ImGui::SliderInt("Green Threshold",
                                     &state.pointsource_config.green_threshold, 20, 255);
                    ImGui::SliderInt("Green Dominance",
                                     &state.pointsource_config.green_dominance, 5, 100);
                    ImGui::SliderInt("Min Blob Pixels",
                                     &state.pointsource_config.min_blob_pixels, 1, 100);
                    ImGui::SliderInt("Max Blob Pixels",
                                     &state.pointsource_config.max_blob_pixels, 50, 5000);

                    int slider_max = state.pointsource_total_frames > 0 ? state.pointsource_total_frames : 100000;
                    ImGui::SliderInt("Start Frame",
                                     &state.pointsource_config.start_frame, 0, slider_max);
                    ImGui::SliderInt("Stop Frame (0=all)",
                                     &state.pointsource_config.stop_frame, 0, slider_max);
                    ImGui::SliderInt("Every Nth Frame",
                                     &state.pointsource_config.frame_step, 1, 100);
                    {
                        int eff_stop = state.pointsource_config.stop_frame > 0
                            ? state.pointsource_config.stop_frame
                            : (state.pointsource_total_frames > 0 ? state.pointsource_total_frames : 0);
                        if (eff_stop > state.pointsource_config.start_frame && state.pointsource_config.frame_step > 0) {
                            int est = (eff_stop - state.pointsource_config.start_frame) / state.pointsource_config.frame_step;
                            ImGui::Text("~%d frames per camera", est);
                        } else if (state.pointsource_config.stop_frame == 0 && state.pointsource_total_frames == 0) {
                            ImGui::Text("~all frames per camera");
                        }
                    }
                    ImGui::Spacing();
                    ImGui::Checkbox("Smart Blob", &state.pointsource_config.smart_blob);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "When multiple green blobs are detected in a frame,\n"
                            "pick the largest valid blob instead of rejecting the frame.\n\n"
                            "Use when cameras have persistent green artifacts\n"
                            "(LEDs, reflections) that interfere with detection.");
                    ImGui::Unindent();
                    } // end Detection Parameters

                    // Filtering parameters
                    if (ImGui::CollapsingHeader("Filtering")) {
                    ImGui::Indent();
                    int max_min_cams =
                        std::max(2, (int)state.pointsource_config.camera_names.size());
                    ImGui::SliderInt("Min Cameras",
                                     &state.pointsource_config.min_cameras, 2,
                                     max_min_cams);
                    float reproj_th = (float)state.pointsource_config.reproj_threshold;
                    if (ImGui::SliderFloat("Reproj Threshold (px)", &reproj_th,
                                           1.0f, 50.0f))
                        state.pointsource_config.reproj_threshold = reproj_th;

                    // Loose Init: auto-detect and PnP re-init poorly-calibrated cameras
                    if (state.pointsource_config.no_init) {
                        bool forced = true;
                        ImGui::BeginDisabled();
                        ImGui::Checkbox("Loose Init", &forced);
                        ImGui::EndDisabled();
                    } else {
                        ImGui::Checkbox("Loose Init", &state.pointsource_config.loose_init);
                    }
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip(
                            "Auto-detect cameras with poor initial calibration\n"
                            "and re-initialize their extrinsics via PnP from\n"
                            "well-calibrated cameras' 3D points.\n\n"
                            "Use when some cameras have old/inaccurate YAMLs.");
                    ImGui::Unindent();
                    } // end Filtering

                    // BA parameters
                    if (ImGui::CollapsingHeader("Bundle Adjustment")) {
                    ImGui::Indent();
                    float ba_th1 = (float)state.pointsource_config.ba_outlier_th1;
                    float ba_th2 = (float)state.pointsource_config.ba_outlier_th2;
                    if (ImGui::SliderFloat("BA Outlier Pass 1 (px)", &ba_th1,
                                           1.0f, 50.0f))
                        state.pointsource_config.ba_outlier_th1 = ba_th1;
                    if (ImGui::SliderFloat("BA Outlier Pass 2 (px)", &ba_th2,
                                           0.5f, 20.0f))
                        state.pointsource_config.ba_outlier_th2 = ba_th2;
                    ImGui::SliderInt("BA Max Iterations",
                                     &state.pointsource_config.ba_max_iter, 10, 200);
                    // Optimization mode dropdown
                    {
                        const char *opt_labels[] = {
                            "Extrinsics only",
                            "Extrinsics + focal length",
                            "Extrinsics + all intrinsics",
                            "Full (all parameters free)"};
                        int opt_idx = static_cast<int>(state.pointsource_config.opt_mode);
                        if (ImGui::Combo("Optimization Mode", &opt_idx, opt_labels, 4))
                            state.pointsource_config.opt_mode =
                                static_cast<PointSourceCalibration::PointSourceOptMode>(opt_idx);
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
                            "  Only recommended with high-quality pointsource data (many points,\n"
                            "  good spatial coverage, high camera redundancy).");
                    ImGui::Unindent();
                    } // end Bundle Adjustment

                    // Global Registration (optional — Procrustes alignment to world frame)
                    if (ImGui::CollapsingHeader("Global Registration (optional)")) {
                    ImGui::Indent();
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                        "Align output to world frame using a ChArUco board");

                    // Global Reg. Media folder picker
                    ImGui::Text("Global Reg. Media:");
                    ImGui::SetNextItemWidth(-80.0f);
                    if (ImGui::InputText("##ps_globalreg",
                                         &state.project.global_reg_media_folder)) {
                        auto info = CalibrationTool::detect_aruco_media(
                            state.project.global_reg_media_folder);
                        state.project.global_reg_media_type = info.type;
                        state.pointsource_global_reg_info = info;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##ps_globalreg")) {
                        IGFD::FileDialogConfig cfg;
                        cfg.countSelectionMax = 1;
                        if (!state.project.global_reg_media_folder.empty())
                            cfg.path = state.project.global_reg_media_folder;
                        cfg.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChoosePSGlobalRegFolder",
                            "Select Global Registration Media Folder", nullptr, cfg);
                    }
                    // Show auto-detection result
                    if (!state.pointsource_global_reg_info.description.empty()) {
                        if (state.pointsource_global_reg_info.type.empty()) {
                            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                                "No images or videos found in folder");
                        } else {
                            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                                "%s", state.pointsource_global_reg_info.description.c_str());
                        }
                    }

                    // Board Setup (when global reg media is set and no config file)
                    if (!state.project.global_reg_media_folder.empty() &&
                        state.project.config_file.empty()) {
                        ImGui::Spacing();
                        ImGui::Text("Board Setup:");
                        auto &cs = state.project.charuco_setup;

                        ImGui::SetNextItemWidth(120);
                        ImGui::InputInt("Squares X##ps_board", &cs.w);
                        cs.w = std::max(3, std::min(cs.w, 20));

                        ImGui::SetNextItemWidth(120);
                        ImGui::InputInt("Squares Y##ps_board", &cs.h);
                        cs.h = std::max(3, std::min(cs.h, 20));

                        ImGui::SetNextItemWidth(120);
                        ImGui::InputFloat("Square Size (mm)##ps_board",
                                          &cs.square_side_length, 1.0f, 10.0f, "%.1f");
                        cs.square_side_length = std::max(1.0f, cs.square_side_length);

                        ImGui::SetNextItemWidth(120);
                        ImGui::InputFloat("Marker Size (mm)##ps_board",
                                          &cs.marker_side_length, 1.0f, 10.0f, "%.1f");
                        cs.marker_side_length = std::max(1.0f, cs.marker_side_length);

                        ImGui::SetNextItemWidth(200);
                        {
                            const char *combo_labels[] = {
                                "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250",
                                "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250",
                                "DICT_6X6_50", "DICT_6X6_250"};
                            int combo_items[] = {0, 1, 2, 4, 5, 6, 8, 10};
                            int combo_count = 8;
                            int sel = 0;
                            for (int i = 0; i < combo_count; i++)
                                if (combo_items[i] == cs.dictionary) { sel = i; break; }
                            if (ImGui::Combo("Dictionary##ps_board", &sel, combo_labels, combo_count))
                                cs.dictionary = combo_items[sel];
                        }

                        // Show auto-generated gt_pts summary
                        {
                            int n_pts = (cs.w - 1) * (cs.h - 1);
                            float half_x = (cs.w - 2) * cs.square_side_length / 2.0f;
                            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                                "%d pts, center-origin, z=0 (%.0f to %.0f mm)",
                                n_pts, -half_x, half_x);
                        }
                    }
                    // No Init checkbox (requires global reg media + board params)
                    ImGui::Spacing();
                    bool can_no_init = !state.project.global_reg_media_folder.empty() &&
                                       state.project.charuco_setup.w >= 3;
                    ImGui::BeginDisabled(!can_no_init);
                    if (ImGui::Checkbox("No Init", &state.pointsource_config.no_init)) {
                        // No Init implies Loose Init
                        if (state.pointsource_config.no_init)
                            state.pointsource_config.loose_init = true;
                    }
                    ImGui::EndDisabled();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip(
                            "Bootstrap all camera parameters from ChArUco board\n"
                            "images alone. Does not require calibration YAML files.\n\n"
                            "Uses default intrinsics (f=image_width) and PnP from\n"
                            "board corners. Requires Global Reg. Media and Board Setup.");

                    ImGui::Unindent();
                    } // end Global Registration

                    ImGui::Separator();

                    // Run button
                    bool can_run_laser = !state.pointsource_running &&
                                       !state.pointsource_config.camera_names.empty();
                    ImGui::BeginDisabled(!can_run_laser);
                    if (ImGui::Button("Run PointSource Refinement")) {
                        state.pointsource_running = true;
                        state.pointsource_done = false;
                        state.pointsource_status =
                            "Starting pointsource calibration pipeline...";

                        // Filter camera names by enabled checkboxes
                        {
                            std::vector<std::string> enabled_cams;
                            for (int i = 0; i < (int)state.project.camera_names.size(); i++) {
                                if (i < (int)state.pointsource_camera_enabled.size() &&
                                    state.pointsource_camera_enabled[i])
                                    enabled_cams.push_back(state.project.camera_names[i]);
                            }
                            state.pointsource_config.camera_names = std::move(enabled_cams);
                        }

                        // Populate global registration config from project
                        state.pointsource_config.do_global_reg =
                            !state.project.global_reg_media_folder.empty();
                        state.pointsource_config.global_reg_media_folder =
                            state.project.global_reg_media_folder;
                        state.pointsource_config.global_reg_media_type =
                            state.project.global_reg_media_type;
                        state.pointsource_config.charuco_setup =
                            state.project.charuco_setup;
                        if (state.pointsource_config.do_global_reg) {
                            auto gt = CalibrationTool::generate_charuco_gt_pts(
                                state.project.charuco_setup);
                            state.pointsource_config.world_coordinate_imgs = {"0"};
                            state.pointsource_config.gt_pts["0"] = std::move(gt);
                        }

                        state.pointsource_future = std::async(
                            std::launch::async,
                            [config = state.pointsource_config,
                             status_ptr = &state.pointsource_status,
                             prog = state.pointsource_progress]() {
                                return PointSourceCalibration::
                                    run_pointsource_refinement(config, status_ptr,
                                                         prog.get());
                            });
                    }
                    ImGui::EndDisabled();

                    if (state.pointsource_running) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                           "Running...");
                    }

                    // Progress sub-section (visible during/after detection)
                    if (state.pointsource_running && !state.pointsource_progress->cameras.empty()) {
                        if (ImGui::CollapsingHeader("Progress", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();
                            int eff_stop = state.pointsource_config.stop_frame > 0
                                ? state.pointsource_config.stop_frame
                                : (state.pointsource_total_frames > 0 ? state.pointsource_total_frames : 0);
                            int eff_start = state.pointsource_config.start_frame;
                            int eff_step = std::max(1, state.pointsource_config.frame_step);
                            int prog_total = (eff_stop > eff_start)
                                ? (eff_stop - eff_start + eff_step - 1) / eff_step
                                : 0;
                            int total_done = 0;
                            for (int ci = 0; ci < (int)state.pointsource_progress->cameras.size(); ci++)
                                if (state.pointsource_progress->cameras[ci]->done.load(std::memory_order_relaxed))
                                    total_done++;
                            ImGui::Text("Detection: %d / %d cameras complete",
                                        total_done, (int)state.pointsource_progress->cameras.size());

                            if (ImGui::BeginTable(
                                    "ps_det_progress", 4,
                                    ImGuiTableFlags_RowBg |
                                        ImGuiTableFlags_BordersInnerV)) {
                                ImGui::TableSetupColumn("Camera", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                                ImGui::TableSetupColumn("Progress", ImGuiTableColumnFlags_WidthStretch);
                                ImGui::TableSetupColumn("Spots", ImGuiTableColumnFlags_WidthFixed, 60.0f);
                                ImGui::TableSetupColumn("Rate", ImGuiTableColumnFlags_WidthFixed, 50.0f);
                                ImGui::TableHeadersRow();

                                for (int ci = 0; ci < (int)state.pointsource_progress->cameras.size(); ci++) {
                                    auto &cp = state.pointsource_progress->cameras[ci];
                                    int fr = cp->frames_processed.load(std::memory_order_relaxed);
                                    int sp = cp->spots_detected.load(std::memory_order_relaxed);
                                    bool dn = cp->done.load(std::memory_order_relaxed);
                                    float frac = prog_total > 0 ? (float)fr / prog_total : 0.0f;
                                    if (frac > 1.0f) frac = 1.0f;

                                    ImGui::TableNextRow();
                                    ImGui::TableSetColumnIndex(0);
                                    if (ci < (int)state.pointsource_config.camera_names.size())
                                        ImGui::Text("Cam%s", state.pointsource_config.camera_names[ci].c_str());
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
                    if (state.pointsource_done && state.pointsource_result.success) {
                        if (ImGui::CollapsingHeader("Results", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();
                        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                           "Reprojection error: %.3f -> %.3f px",
                                           state.pointsource_result.mean_reproj_before,
                                           state.pointsource_result.mean_reproj_after);
                        double avg_obs = state.pointsource_result.valid_3d_points > 0
                            ? (double)state.pointsource_result.total_observations / state.pointsource_result.valid_3d_points
                            : 0.0;
                        ImGui::Text("3D points: %d | Observations: %d (avg %.1f cameras/point)",
                                    state.pointsource_result.valid_3d_points,
                                    state.pointsource_result.total_observations,
                                    avg_obs);
                        if (state.pointsource_result.ba_outliers_removed > 0)
                            ImGui::Text("BA outliers removed: %d", state.pointsource_result.ba_outliers_removed);
                        ImGui::Text("Output: %s",
                                    state.pointsource_result.output_folder.c_str());

                        // Global registration status
                        if (!state.pointsource_result.global_reg_status.empty()) {
                            if (state.pointsource_result.global_reg_status.find("failed") != std::string::npos) {
                                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                                    "Global Reg: %s", state.pointsource_result.global_reg_status.c_str());
                            } else {
                                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                                    "Global Reg: %s", state.pointsource_result.global_reg_status.c_str());
                            }
                        }

                        // Per-camera changes table
                        if (!state.pointsource_result.camera_changes.empty()) {
                            if (ImGui::TreeNode("Per-camera changes")) {
                                if (ImGui::BeginTable(
                                        "ps_cam_changes", 9,
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

                                    for (const auto &cc : state.pointsource_result.camera_changes) {
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
                        // ── 3D Viewer button ──
                        ImGui::Separator();
                        if (ImGui::Button("3D Viewer##ps_viewer")) {
                            // Build a CalibrationResult from PointSource output files
                            const auto &ps_res = state.pointsource_result;
                            const auto &ps_cfg = state.pointsource_config;
                            namespace fs = std::filesystem;
                            std::string out_dir = ps_res.output_folder;
                            std::string summary_dir = out_dir + "/summary_data";

                            // Load cameras from YAML files
                            state.loaded_result = CalibrationPipeline::CalibrationResult();
                            auto &lr = state.loaded_result;
                            lr.output_folder = out_dir;
                            lr.success = true;
                            lr.mean_reproj_error = ps_res.mean_reproj_after;

                            for (const auto &cam_name : ps_cfg.camera_names) {
                                std::string yaml_path = out_dir + "/Cam" + cam_name + ".yaml";
                                if (!fs::exists(yaml_path)) continue;
                                try {
                                    auto yf = opencv_yaml::read(yaml_path);
                                    CalibrationPipeline::CameraPose pose;
                                    pose.K = yf.getMatrix("camera_matrix");
                                    Eigen::MatrixXd dist_mat = yf.getMatrix("distortion_coefficients");
                                    for (int j = 0; j < 5 && j < (int)dist_mat.rows(); j++)
                                        pose.dist(j) = dist_mat(j, 0);
                                    pose.R = yf.getMatrix("rc_ext");
                                    Eigen::MatrixXd t_mat = yf.getMatrix("tc_ext");
                                    pose.t = Eigen::Vector3d(t_mat(0,0), t_mat(1,0), t_mat(2,0));
                                    int iw = yf.getInt("image_width"), ih = yf.getInt("image_height");
                                    if (lr.cameras.empty()) { lr.image_width = iw; lr.image_height = ih; }
                                    lr.cameras.push_back(pose);
                                    lr.cam_names.push_back(cam_name);
                                } catch (const std::exception &e) {
                                    fprintf(stderr, "[PS Viewer] Skipping %s: %s\n",
                                            cam_name.c_str(), e.what());
                                }
                            }

                            // Load ba_points.json (array of [x,y,z])
                            std::string pts_path = summary_dir + "/ba_points.json";
                            if (fs::exists(pts_path)) {
                                try {
                                    std::ifstream pf(pts_path);
                                    nlohmann::json pts_j;
                                    pf >> pts_j;
                                    for (int i = 0; i < (int)pts_j.size(); i++) {
                                        lr.points_3d[i] = Eigen::Vector3d(
                                            pts_j[i][0].get<double>(),
                                            pts_j[i][1].get<double>(),
                                            pts_j[i][2].get<double>());
                                    }
                                } catch (const std::exception &e) {
                                    fprintf(stderr, "[PS Viewer] Failed to load ba_points.json: %s\n", e.what());
                                }
                            }

                            // Load observations.json and build landmarks map
                            std::string obs_path = summary_dir + "/observations.json";
                            if (fs::exists(obs_path)) {
                                try {
                                    std::ifstream of(obs_path);
                                    nlohmann::json obs_j;
                                    of >> obs_j;
                                    // obs_j is array of arrays: obs_j[point_idx] = [{cam, px:[x,y]}, ...]
                                    for (int pi = 0; pi < (int)obs_j.size(); pi++) {
                                        for (const auto &o : obs_j[pi]) {
                                            int ci = o["cam"].get<int>();
                                            double px = o["px"][0].get<double>();
                                            double py = o["px"][1].get<double>();
                                            if (ci >= 0 && ci < (int)lr.cam_names.size()) {
                                                lr.db.landmarks[lr.cam_names[ci]][pi] =
                                                    Eigen::Vector2d(px, py);
                                            }
                                        }
                                    }
                                } catch (const std::exception &e) {
                                    fprintf(stderr, "[PS Viewer] Failed to load observations.json: %s\n", e.what());
                                }
                            }

                            // Build per-camera metrics from CameraChange data
                            int nc = (int)lr.cam_names.size();
                            lr.per_camera_metrics.resize(nc);
                            for (int c = 0; c < nc; c++) {
                                auto &m = lr.per_camera_metrics[c];
                                m.name = lr.cam_names[c];
                                // Find matching CameraChange
                                for (const auto &cc : ps_res.camera_changes) {
                                    if (cc.name == lr.cam_names[c]) {
                                        m.detection_count = cc.detections;
                                        m.observation_count = cc.observations;
                                        break;
                                    }
                                }
                                // Compute reprojection stats from landmarks + 3D points
                                auto lm_it = lr.db.landmarks.find(lr.cam_names[c]);
                                if (lm_it != lr.db.landmarks.end() && !lr.cameras.empty()) {
                                    Eigen::Vector3d rvec = red_math::rotationMatrixToVector(lr.cameras[c].R);
                                    std::vector<double> errs;
                                    for (const auto &[pid, px] : lm_it->second) {
                                        auto pt_it = lr.points_3d.find(pid);
                                        if (pt_it == lr.points_3d.end()) continue;
                                        Eigen::Vector2d proj = red_math::projectPoint(
                                            pt_it->second, rvec, lr.cameras[c].t,
                                            lr.cameras[c].K, lr.cameras[c].dist);
                                        double err = (proj - px).norm();
                                        errs.push_back(err);
                                    }
                                    if (!errs.empty()) {
                                        std::sort(errs.begin(), errs.end());
                                        double sum = 0;
                                        for (double e : errs) sum += e;
                                        m.mean_reproj = sum / errs.size();
                                        m.median_reproj = errs[errs.size() / 2];
                                        m.max_reproj = errs.back();
                                    }
                                }
                            }

                            // Set board dimensions for the board quad visualization
                            state.calib_viewer.board_width_mm =
                                ps_cfg.charuco_setup.square_side_length * (ps_cfg.charuco_setup.w - 1);
                            state.calib_viewer.board_height_mm =
                                ps_cfg.charuco_setup.square_side_length * (ps_cfg.charuco_setup.h - 1);

                            // Open viewer
                            state.calib_viewer.result = &state.loaded_result;
                            state.calib_viewer.cached_points_version = -1; // force cache rebuild
                            state.calib_viewer.cached_selection = -2;
                            state.calib_viewer.show = true;
                            fprintf(stderr, "[PS Viewer] Loaded %d cameras, %d 3D points, %d landmark cameras\n",
                                    nc, (int)lr.points_3d.size(), (int)lr.db.landmarks.size());
                        }

                        ImGui::Unindent();
                        } // end CollapsingHeader("Results")
                    }

                    // PointSource status
                    if (!state.pointsource_status.empty()) {
                        ImGui::Separator();
                        if (state.pointsource_status.find("Error") !=
                            std::string::npos) {
                            ImGui::TextColored(
                                ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s",
                                state.pointsource_status.c_str());
                        } else {
                            ImGui::TextWrapped("%s", state.pointsource_status.c_str());
                        }
                    }

                    // Detection Processing visualization
                    if (ImGui::CollapsingHeader("Detection Processing")) {
                    ImGui::Indent();
                    bool prev_detection = state.pointsource_show_detection;
                    ImGui::Checkbox("Enable", &state.pointsource_show_detection);
                    if (prev_detection && !state.pointsource_show_detection) {
#ifdef __APPLE__
                        for (int ci = 0; ci < scene->num_cams; ci++)
                            mac_last_uploaded_frame[ci] = -1;
#endif
                        state.pointsource_viz.ready.clear();
                    }
                    if (state.pointsource_show_detection && ps.video_loaded) {
                        if (state.pointsource_viz.computing.load(std::memory_order_relaxed))
                            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                               "  Processing...");
                        int detecting_count = 0;
                        int total_cams = std::min((int)state.pointsource_viz.ready.size(), (int)scene->num_cams);
                        for (int ci = 0; ci < total_cams; ci++) {
                            auto &cr = state.pointsource_viz.ready[ci];
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
                    // PointSource inputs not yet complete -- show hint
                    if (state.project.media_folder.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                            "Set Video Folder to enable pointsource refinement");
                    } else if (state.project.camera_names.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                            "Set Video Folder and click Load PointSource Videos");
                    }
                }

                // Results section — shown when results are loaded, even without videos
                // (e.g., after "Load Previous Results")
                if (!state.pointsource_ready &&
                    state.pointsource_done && state.pointsource_result.success) {
                    if (ImGui::CollapsingHeader("Results", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Indent();
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                       "Reprojection error: %.3f -> %.3f px",
                                       state.pointsource_result.mean_reproj_before,
                                       state.pointsource_result.mean_reproj_after);
                    double avg_obs = state.pointsource_result.valid_3d_points > 0
                        ? (double)state.pointsource_result.total_observations / state.pointsource_result.valid_3d_points
                        : 0.0;
                    ImGui::Text("3D points: %d | Observations: %d (avg %.1f cameras/point)",
                                state.pointsource_result.valid_3d_points,
                                state.pointsource_result.total_observations,
                                avg_obs);
                    if (!state.pointsource_result.output_folder.empty())
                        ImGui::Text("Output: %s",
                                    state.pointsource_result.output_folder.c_str());

                    // 3D Viewer button (same as inside pointsource_ready block)
                    if (!state.pointsource_result.output_folder.empty()) {
                        ImGui::Spacing();
                        if (ImGui::Button("3D Viewer##ps_viewer_loaded")) {
                            namespace fs = std::filesystem;
                            auto &lr = state.loaded_result;
                            lr = CalibrationPipeline::CalibrationResult{};
                            std::string out_folder = state.pointsource_result.output_folder;
                            std::string summary_dir = out_folder + "/summary_data";

                            // Load camera poses from YAML files
                            for (const auto &cn : state.pointsource_config.camera_names) {
                                std::string yaml_path = out_folder + "/Cam" + cn + ".yaml";
                                if (!fs::exists(yaml_path)) continue;
                                try {
                                    auto yaml = opencv_yaml::read(yaml_path);
                                    CalibrationPipeline::CameraPose cp;
                                    cp.K = yaml.getMatrix("camera_matrix").block<3, 3>(0, 0);
                                    Eigen::MatrixXd dm = yaml.getMatrix("distortion_coefficients");
                                    for (int j = 0; j < 5; j++) cp.dist(j) = dm(j, 0);
                                    cp.R = yaml.getMatrix("rc_ext").block<3, 3>(0, 0);
                                    Eigen::MatrixXd tm = yaml.getMatrix("tc_ext");
                                    cp.t = Eigen::Vector3d(tm(0, 0), tm(1, 0), tm(2, 0));
                                    lr.cameras.push_back(cp);
                                    lr.cam_names.push_back(cn);
                                    if (lr.image_width == 0) {
                                        lr.image_width = yaml.getInt("image_width");
                                        lr.image_height = yaml.getInt("image_height");
                                    }
                                } catch (...) {}
                            }

                            // Load ba_points.json
                            std::string pts_path = summary_dir + "/ba_points.json";
                            if (fs::exists(pts_path)) {
                                try {
                                    std::ifstream pf(pts_path);
                                    nlohmann::json pts_j;
                                    pf >> pts_j;
                                    for (int i = 0; i < (int)pts_j.size(); i++)
                                        lr.points_3d[i] = Eigen::Vector3d(
                                            pts_j[i][0].get<double>(),
                                            pts_j[i][1].get<double>(),
                                            pts_j[i][2].get<double>());
                                } catch (...) {}
                            }

                            // Load observations.json for per-camera landmarks
                            std::string obs_path = summary_dir + "/observations.json";
                            if (fs::exists(obs_path)) {
                                try {
                                    std::ifstream of2(obs_path);
                                    nlohmann::json obs_j;
                                    of2 >> obs_j;
                                    for (int pi = 0; pi < (int)obs_j.size(); pi++) {
                                        for (const auto &obs : obs_j[pi]) {
                                            int ci = obs["cam"].get<int>();
                                            if (ci < (int)lr.cam_names.size()) {
                                                auto px = obs["px"];
                                                lr.db.landmarks[lr.cam_names[ci]][pi] =
                                                    Eigen::Vector2d(px[0].get<double>(), px[1].get<double>());
                                            }
                                        }
                                    }
                                } catch (...) {}
                            }

                            state.calib_viewer.result = &state.loaded_result;
                            state.calib_viewer.show = true;
                            state.calib_viewer.cached_points_version = -1;
                            state.calib_viewer.board_width_mm =
                                (state.project.charuco_setup.w - 1) * state.project.charuco_setup.square_side_length;
                            state.calib_viewer.board_height_mm =
                                (state.project.charuco_setup.h - 1) * state.project.charuco_setup.square_side_length;
                        }
                    }
                    ImGui::Unindent();
                    } // end Results (loaded from disk)
                }

                // Status
                if (!state.pointsource_status.empty()) {
                    ImGui::Separator();
                    if (state.pointsource_status.find("Error") != std::string::npos) {
                        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s",
                                           state.pointsource_status.c_str());
                    } else {
                        ImGui::TextWrapped("%s", state.pointsource_status.c_str());
                    }
                }

                ImGui::Unindent();
                } // end CollapsingHeader("PointSource Refinement")
                ImGui::Spacing();
}
