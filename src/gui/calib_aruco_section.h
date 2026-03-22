#pragma once
#include "calib_tool_state.h"
#include "app_context.h"
#include "calibration_pipeline.h"
#include "aruco_metal.h"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <numeric>
#include <string>

// Draw the Aruco Calibration section inside the Calibration Tool window.
// Called only when state.project.has_aruco() && state.config_loaded.
inline void DrawCalibArucoSection(CalibrationToolState &state, AppContext &ctx,
                                   const CalibrationToolCallbacks &cb) {
    auto &pm = ctx.pm;
    auto &ps = ctx.ps;
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

    if (ImGui::CollapsingHeader("Aruco Calibration", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent();

            // Camera enable/disable checkboxes
            {
                int n_cams = (int)state.config.cam_ordered.size();
                int n_enabled = 0;
                for (int i = 0; i < n_cams; i++)
                    if (i < (int)state.camera_enabled.size() && state.camera_enabled[i])
                        n_enabled++;
                ImGui::Text("Cameras:      %d / %d enabled", n_enabled, n_cams);

                if (n_cams > 0 && ImGui::TreeNode("Camera Selection")) {
                    // Select All / None buttons
                    if (ImGui::SmallButton("All")) {
                        for (size_t i = 0; i < state.camera_enabled.size(); i++)
                            state.camera_enabled[i] = true;
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("None")) {
                        for (size_t i = 0; i < state.camera_enabled.size(); i++)
                            state.camera_enabled[i] = false;
                    }

                    // 2-column checkbox grid
                    int n_rows = (n_cams + 1) / 2;
                    if (ImGui::BeginTable("##calib_cam_grid", 2)) {
                        for (int row = 0; row < n_rows; row++) {
                            ImGui::TableNextRow();
                            for (int col = 0; col < 2; col++) {
                                int idx = row + col * n_rows;
                                ImGui::TableSetColumnIndex(col);
                                if (idx < n_cams && idx < (int)state.camera_enabled.size()) {
                                    bool enabled = state.camera_enabled[idx];
                                    if (ImGui::Checkbox(
                                        ("##calib_cam_" + std::to_string(idx)).c_str(),
                                        &enabled))
                                        state.camera_enabled[idx] = enabled;
                                    ImGui::SameLine(0.0f, 2.0f);
                                    ImGui::TextUnformatted(
                                        state.config.cam_ordered[idx].c_str());
                                }
                            }
                        }
                        ImGui::EndTable();
                    }
                    ImGui::TreePop();
                }
            }

            ImGui::Text("Board:        %d x %d  (%.1f mm squares)",
                        state.config.charuco_setup.w,
                        state.config.charuco_setup.h,
                        state.config.charuco_setup.square_side_length);

            // ── Unified ArUco Calibration ──
            {
                std::string media_folder = state.project.effective_aruco_media();
                bool is_video = state.project.aruco_is_video();
                bool is_image = state.project.aruco_is_image();

                // Auto-detect media info (cached)
                static std::string cached_media_folder;
                static CalibrationTool::ArucoMediaInfo cached_media_info;
                if (cached_media_folder != media_folder) {
                    cached_media_folder = media_folder;
                    cached_media_info = CalibrationTool::detect_aruco_media(media_folder);
                    if (is_video) {
                        auto vids = CalibrationTool::discover_aruco_videos(
                            media_folder, state.config.cam_ordered);
                        state.aruco_video_count = (int)vids.size();
                        if (!vids.empty())
                            state.aruco_total_frames =
                                CalibrationPipeline::get_video_frame_count(
                                    vids.begin()->second);
                    }
                }

                ImGui::Text("Media: %s", media_folder.c_str());
                if (!cached_media_info.description.empty())
                    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                        "%s", cached_media_info.description.c_str());

                // Load Media button
                if (is_video) {
                    if (!state.aruco_media_loaded) {
                        ImGui::BeginDisabled(
                            state.aruco_video_count == 0 ||
                            state.aruco_running());
                        if (ImGui::Button("Load Videos##aruco_load")) {
                            state.status = "Loading aruco videos...";
                            cb.deferred->enqueue([&state, &pm, &ps, &cb,
                                                  &imgs_names, dc_context, media_folder
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
                                    pm.media_folder = media_folder;
                                    pm.camera_names.clear();
                                    for (const auto &cn :
                                         state.config.cam_ordered)
                                        pm.camera_names.push_back("Cam" + cn);
                                    cb.load_videos();
                                    cb.print_metadata();
                                    state.aruco_total_frames =
                                        dc_context->estimated_num_frames;
                                    state.aruco_media_loaded = true;
                                    state.status =
                                        "Loaded " +
                                        std::to_string(
                                            state.config.cam_ordered.size()) +
                                        " aruco videos";
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
                        if (ImGui::Button("Close##aruco_close")) {
                            close_media_deferred(state.aruco_media_loaded,
                                                 state.status, "Media closed");
                        }
                    }
                } else if (is_image) {
                    if (!state.aruco_media_loaded) {
                        if (ImGui::Button("Load Images##aruco_load")) {
                            auto files =
                                CalibrationTool::discover_images(state.config);
                            if (files.empty()) {
                                state.status =
                                    "Error: No matching images found in " +
                                    media_folder;
                            } else {
                                pm.media_folder = media_folder;
                                pm.camera_names.clear();
                                imgs_names.clear();
                                cb.load_images(files);
                                state.aruco_media_loaded = true;
                                state.status =
                                    "Loaded " + std::to_string(files.size()) +
                                    " images across " +
                                    std::to_string(
                                        state.config.cam_ordered.size()) +
                                    " cameras";
                            }
                        }
                    } else {
                        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                           "Images loaded");
                        ImGui::SameLine();
                        if (ImGui::Button("Close##aruco_close")) {
                            close_media_deferred(state.aruco_media_loaded,
                                                 state.status, "Media closed");
                        }
                    }
                }

                ImGui::Separator();

                // Video-only: frame range sliders
                if (is_video) {
                    int slider_max = state.aruco_total_frames > 0
                        ? state.aruco_total_frames : 100000;
                    ImGui::SliderInt("Start Frame##aruco",
                                     &state.aruco_start_frame, 0, slider_max);
                    ImGui::SliderInt("Stop Frame (0=all)##aruco",
                                     &state.aruco_stop_frame, 0, slider_max);
                    ImGui::SliderInt("Every Nth Frame##aruco",
                                     &state.aruco_frame_step, 1, 100);
                    {
                        int eff_stop = state.aruco_stop_frame > 0
                            ? state.aruco_stop_frame
                            : (state.aruco_total_frames > 0
                                ? state.aruco_total_frames : 0);
                        if (eff_stop > state.aruco_start_frame &&
                            state.aruco_frame_step > 0) {
                            int est = (eff_stop - state.aruco_start_frame) /
                                      state.aruco_frame_step;
                            ImGui::Text("~%d frames per camera (%d cameras = %d total)",
                                        est, state.aruco_video_count,
                                        est * state.aruco_video_count);
                        }
                    }
                    ImGui::Separator();
                }

                // Global registration section
                {
                    bool has_config_gt = !state.config.world_coordinate_imgs.empty() &&
                                         !state.config.gt_pts.empty();
                    bool has_global_reg_media = !state.project.global_reg_media_folder.empty();
                    bool config_free = state.project.config_file.empty();

                    if (has_config_gt && !config_free) {
                        // Config-loaded: show read-only
                        ImGui::Text("Global Registration Frame: %s",
                            state.config.world_coordinate_imgs[0].c_str());
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                            "(%d ground truth points)",
                            (int)state.config.gt_pts.begin()->second.size());
                    } else if (config_free) {
                        // Config-free: interactive global registration
                        if (has_global_reg_media) {
                            int n_pts = (state.project.charuco_setup.w - 1) *
                                        (state.project.charuco_setup.h - 1);
                            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                                "Global Reg: separate media (%d pts, center-origin, z=0)",
                                n_pts);
                        } else if (is_video) {
                            ImGui::Text("Global Reg Frame:");
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(80);
                            ImGui::InputInt("##global_reg_frame", &state.global_reg_frame);
                            state.global_reg_frame = std::max(0, state.global_reg_frame);
                            int n_pts = (state.project.charuco_setup.w - 1) *
                                        (state.project.charuco_setup.h - 1);
                            ImGui::SameLine();
                            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                                "(%d pts, center-origin, z=0)", n_pts);
                        } else {
                            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                                "Global Reg: provide Global Reg. Media or config.json");
                        }
                    }
                }

                ImGui::Separator();

                // Poll unified calibration future
                if (state.aruco_running_flag && state.aruco_future.valid()) {
                    auto fs = state.aruco_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fs == std::future_status::ready) {
                        state.aruco_result = state.aruco_future.get();
                        state.aruco_running_flag = false;
                        state.aruco_done = true;
                        if (state.aruco_result.success) {
                            state.project.aruco_output_folder =
                                state.aruco_result.output_folder;
                            // Persist calibration run parameters
                            state.project.last_aruco_start_frame = state.aruco_start_frame;
                            state.project.last_aruco_stop_frame = state.aruco_stop_frame;
                            state.project.last_aruco_frame_step = is_video ? state.aruco_frame_step : 0;
                            state.project.last_aruco_total_video_frames = state.aruco_total_frames;
                            state.project.last_aruco_cameras_used = (int)state.aruco_result.cam_names.size();
                            state.project.last_aruco_mean_reproj = state.aruco_result.mean_reproj_error;
                            std::string pf =
                                state.project.project_path + "/" +
                                state.project.project_name + ".redproj";
                            std::string se;
                            CalibrationTool::save_project(
                                state.project, pf, &se);
                            std::string msg =
                                "Calibration complete! Reproj: " +
                                std::to_string(
                                    state.aruco_result.mean_reproj_error)
                                    .substr(0, 5) + " px";
                            if (!state.aruco_result.warning.empty()) {
                                msg += " | " + state.aruco_result.warning;
                                for (size_t i = 0; i < state.config.cam_ordered.size(); i++) {
                                    bool found = false;
                                    for (const auto &name : state.aruco_result.cam_names)
                                        if (name == state.config.cam_ordered[i]) { found = true; break; }
                                    if (!found && i < state.camera_enabled.size())
                                        state.camera_enabled[i] = false;
                                }
                                ctx.toasts.push(state.aruco_result.warning,
                                                Toast::Warning, 8.0f);
                            }
                            state.status = msg;
                            // Show global registration status
                            if (!state.aruco_result.global_reg_status.empty()) {
                                bool gr_ok = state.aruco_result.global_reg_status.find("Aligning") != std::string::npos ||
                                             state.aruco_result.global_reg_status.find("Using") != std::string::npos;
                                ctx.toasts.push("Global Reg: " + state.aruco_result.global_reg_status,
                                                gr_ok ? Toast::Info : Toast::Warning, 10.0f);
                            }
                            // Auto-open 3D viewer
                            state.calib_viewer.result = &state.aruco_result;
                            state.calib_viewer.show = true;
                            state.calib_viewer.selected_camera = -1;
                            state.calib_viewer.cached_selection = -2;
                        } else {
                            state.status =
                                "Error: " + state.aruco_result.error;
                        }
                    }
                }

                // Calibrate button (unified — runs experimental pipeline)
                bool can_run = state.config_loaded &&
                    !media_folder.empty() &&
                    !state.aruco_running();
                ImGui::BeginDisabled(!can_run);
                if (ImGui::Button("Calibrate##aruco_unified")) {
                    state.aruco_running_flag = true;
                    state.aruco_done = false;
                    state.status = "Starting calibration...";
                    std::string base =
                        state.project.project_path +
                        "/aruco_calibration";

                    auto enabled = state.enabled_cameras();
                    auto filtered_config = state.config;
                    filtered_config.cam_ordered = enabled;
                    // Pass global registration media info to pipeline
                    filtered_config.global_reg_media_folder =
                        state.project.global_reg_media_folder;
                    filtered_config.global_reg_media_type =
                        state.project.global_reg_media_type;

                    // Config-free: auto-generate gt_pts from board geometry
                    if (state.project.config_file.empty() &&
                        filtered_config.gt_pts.empty()) {
                        auto gt = CalibrationTool::generate_charuco_gt_pts(
                            state.project.charuco_setup);
                        if (!state.project.global_reg_media_folder.empty()) {
                            // Separate global reg media — gt_pts keyed as "0"
                            filtered_config.world_coordinate_imgs = {"0"};
                            std::vector<std::vector<float>> pts_vec;
                            for (const auto &p : gt) pts_vec.push_back(p);
                            filtered_config.gt_pts["0"] = std::move(pts_vec);
                        } else if (is_video && state.global_reg_frame >= 0) {
                            // Use specific frame from calibration video
                            std::string fr = std::to_string(state.global_reg_frame);
                            filtered_config.world_coordinate_imgs = {fr};
                            std::vector<std::vector<float>> pts_vec;
                            for (const auto &p : gt) pts_vec.push_back(p);
                            filtered_config.gt_pts[fr] = std::move(pts_vec);
                        }
                    }

                    if (is_video) {
                        CalibrationPipeline::VideoFrameRange vfr;
                        vfr.video_folder = media_folder;
                        vfr.cam_ordered = enabled;
                        vfr.start_frame = state.aruco_start_frame;
                        vfr.stop_frame = state.aruco_stop_frame;
                        vfr.frame_step = state.aruco_frame_step;
                        state.aruco_future = std::async(
                            std::launch::async,
                            [config = filtered_config, base,
                             status_ptr = &state.status, vfr]() {
#ifdef __APPLE__
                                auto am = aruco_metal_create();
                                aruco_detect::GpuThresholdFunc gfn =
                                    am ? aruco_metal_threshold_batch : nullptr;
                                auto r = CalibrationPipeline::run_experimental_pipeline(
                                    config, base, status_ptr,
                                    &vfr, gfn, am);
                                aruco_metal_destroy(am);
                                return r;
#else
                                return CalibrationPipeline::run_experimental_pipeline(
                                    config, base, status_ptr, &vfr);
#endif
                            });
                    } else {
                        state.aruco_future = std::async(
                            std::launch::async,
                            [config = filtered_config, base,
                             status_ptr = &state.status]() {
#ifdef __APPLE__
                                auto am = aruco_metal_create();
                                aruco_detect::GpuThresholdFunc gfn =
                                    am ? aruco_metal_threshold_batch : nullptr;
                                auto r = CalibrationPipeline::run_experimental_pipeline(
                                    config, base, status_ptr,
                                    nullptr, gfn, am);
                                aruco_metal_destroy(am);
                                return r;
#else
                                return CalibrationPipeline::run_experimental_pipeline(
                                    config, base, status_ptr);
#endif
                            });
                    }
                }
                ImGui::EndDisabled();
                if (state.aruco_running_flag) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                       "Running...");
                }

                // Results display
                if (state.aruco_done && state.aruco_result.success) {
                    ImGui::TextColored(
                        ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                        "Mean reproj error: %.3f px",
                        state.aruco_result.mean_reproj_error);
                    ImGui::Text("Output: %s",
                        state.project.aruco_output_folder.c_str());
                    ImGui::SameLine();
                    if (ImGui::Button("3D Viewer##aruco_viewer")) {
                        state.calib_viewer.result = &state.aruco_result;
                        state.calib_viewer.show = true;
                        state.calib_viewer.selected_camera = -1;
                        state.calib_viewer.cached_selection = -2;
                    }
                }

                // Load previous calibration from disk
                {
                    std::string cal_folder = state.project.project_path + "/aruco_calibration";
                    bool has_db = CalibrationPipeline::has_calibration_database(cal_folder);
                    if (!has_db && state.aruco_done && state.aruco_result.success)
                        has_db = true;
                    // Also check legacy folders
                    if (!has_db) {
                        std::string legacy_img = state.project.project_path + "/aruco_image_experimental";
                        std::string legacy_vid = state.project.project_path + "/aruco_video_experimental";
                        if (CalibrationPipeline::has_calibration_database(legacy_img))
                            { cal_folder = legacy_img; has_db = true; }
                        else if (CalibrationPipeline::has_calibration_database(legacy_vid))
                            { cal_folder = legacy_vid; has_db = true; }
                    }
                    ImGui::BeginDisabled(!has_db);
                    if (ImGui::Button("Load Previous##aruco_load_prev")) {
                        if (state.aruco_done && state.aruco_result.success) {
                            state.calib_viewer.result = &state.aruco_result;
                        } else {
                            state.loaded_result = CalibrationPipeline::load_calibration_from_folder(
                                cal_folder, state.config.cam_ordered);
                            if (state.loaded_result.success) {
                                state.aruco_result = state.loaded_result;
                                state.aruco_done = true;
                                state.calib_viewer.result = &state.aruco_result;
                                std::string msg = "Loaded calibration (" +
                                    std::to_string((int)state.loaded_result.cameras.size()) +
                                    " cameras): " +
                                    std::to_string(state.aruco_result.mean_reproj_error).substr(0,5) + " px";
                                if (!state.loaded_result.warning.empty()) {
                                    msg += " | " + state.loaded_result.warning;
                                    for (size_t i = 0; i < state.config.cam_ordered.size(); i++) {
                                        bool found = false;
                                        for (const auto &name : state.loaded_result.cam_names)
                                            if (name == state.config.cam_ordered[i]) { found = true; break; }
                                        if (!found && i < state.camera_enabled.size())
                                            state.camera_enabled[i] = false;
                                    }
                                    ctx.toasts.push(state.loaded_result.warning,
                                                    Toast::Warning, 8.0f);
                                }
                                state.status = msg;
                            } else {
                                state.status = "Error: " + state.loaded_result.error;
                            }
                        }
                        state.calib_viewer.show = true;
                        state.calib_viewer.selected_camera = -1;
                        state.calib_viewer.cached_selection = -2;
                    }
                    ImGui::EndDisabled();
                    if (!has_db) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("(no previous calibration)");
                    }
                }
            } // end unified ArUco Calibration

            // ---- Quality Dashboard (visible after calibration completes) ----
            {
                const CalibrationPipeline::CalibrationResult *exp_result = nullptr;
                if (state.aruco_done && state.aruco_result.success)
                    exp_result = &state.aruco_result;

                if (exp_result && !exp_result->per_camera_metrics.empty()) {
                    if (ImGui::CollapsingHeader("Quality Dashboard",
                            ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();

                        const auto &metrics = exp_result->per_camera_metrics;
                        int nc = (int)metrics.size();

                        // Summary text
                        ImGui::Text("BA rounds: %d  |  Outliers removed: %d  |  "
                                    "Mean reproj: %.3f px",
                                    exp_result->ba_rounds,
                                    exp_result->outliers_removed,
                                    exp_result->mean_reproj_error);

                        // Global multi-view consistency (the metric that matters for 3D tracking)
                        if (exp_result->global_consistency.computed) {
                            const auto &gc = exp_result->global_consistency;
                            ImVec4 gc_color = gc.mean_reproj < 3.0
                                ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)     // green
                                : gc.mean_reproj < 8.0
                                    ? ImVec4(1.0f, 0.8f, 0.0f, 1.0f) // yellow
                                    : ImVec4(1.0f, 0.3f, 0.3f, 1.0f); // red
                            ImGui::TextColored(gc_color,
                                "Multi-view consistency: %.2f px (median %.2f, 95th pct %.2f)  |  "
                                "%d landmarks triangulated",
                                gc.mean_reproj, gc.median_reproj, gc.pct95_reproj,
                                gc.landmarks_triangulated);
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip(
                                    "Triangulates each board corner from ALL cameras that see it,\n"
                                    "then measures reprojection error. This tests global extrinsic\n"
                                    "consistency — the accuracy that matters for 3D tracking.\n\n"
                                    "The per-board reproj (%.3f px) measures local fit quality.\n"
                                    "Multi-view reproj (%.2f px) measures global consistency.\n\n"
                                    "< 3 px = excellent | 3-8 px = acceptable | > 8 px = poor",
                                    exp_result->mean_reproj_error, gc.mean_reproj);
                        }
                        ImGui::Spacing();

                        // Prepare data arrays for ImPlot
                        std::vector<double> mean_errs(nc), median_errs(nc);
                        std::vector<double> det_counts(nc);
                        std::vector<double> tick_positions(nc);
                        std::vector<const char *> labels(nc);
                        std::vector<std::string> label_strs(nc);
                        for (int i = 0; i < nc; i++) {
                            mean_errs[i] = metrics[i].mean_reproj;
                            median_errs[i] = metrics[i].median_reproj;
                            det_counts[i] = (double)metrics[i].detection_count;
                            tick_positions[i] = (double)i;
                            label_strs[i] = metrics[i].name;
                            labels[i] = label_strs[i].c_str();
                        }

                        // Bar chart: Per-camera reprojection error (mean + median)
                        if (ImPlot::BeginPlot("Per-Camera Reprojection Error",
                                ImVec2(-1, 240))) {
                            ImPlot::SetupAxes("", "Error (px)");
                            ImPlot::SetupAxisTicks(ImAxis_X1, tick_positions.data(), nc, nullptr);
                            ImPlot::PlotBars("Mean", mean_errs.data(), nc, 0.3, -0.15);
                            ImPlot::PlotBars("Median", median_errs.data(), nc, 0.3, 0.15);
                            // Vertical camera labels
                            for (int i = 0; i < nc; i++)
                                ImPlot::PlotText(labels[i], tick_positions[i], 0,
                                    ImVec2(0, 10), ImPlotTextFlags_Vertical);
                            ImPlot::EndPlot();
                        }

                        // Bar chart: Detection count per camera
                        if (ImPlot::BeginPlot("Detections Per Camera",
                                ImVec2(-1, 200))) {
                            ImPlot::SetupAxes("", "Frames");
                            ImPlot::SetupAxisTicks(ImAxis_X1, tick_positions.data(), nc, nullptr);
                            ImPlot::PlotBars("Detections", det_counts.data(),
                                              nc, 0.5);
                            for (int i = 0; i < nc; i++)
                                ImPlot::PlotText(labels[i], tick_positions[i], 0,
                                    ImVec2(0, 10), ImPlotTextFlags_Vertical);
                            ImPlot::EndPlot();
                        }

                        // Histogram: reprojection error distribution
                        if (!exp_result->all_reproj_errors.empty()) {
                            // Pre-bin into 50 bins from 0 to max
                            const auto &all_errs = exp_result->all_reproj_errors;
                            double max_err = *std::max_element(
                                all_errs.begin(), all_errs.end());
                            max_err = std::max(max_err, 1.0);
                            int num_bins = 50;
                            double bin_width = max_err / num_bins;
                            std::vector<double> bins(num_bins, 0);
                            std::vector<double> bin_centers(num_bins);
                            for (int i = 0; i < num_bins; i++)
                                bin_centers[i] = (i + 0.5) * bin_width;
                            for (double e : all_errs) {
                                int b = std::min((int)(e / bin_width),
                                                  num_bins - 1);
                                bins[b]++;
                            }

                            if (ImPlot::BeginPlot("Error Distribution",
                                    ImVec2(-1, 160))) {
                                ImPlot::SetupAxes("Error (px)", "Count");
                                ImPlot::PlotBars("Observations", bin_centers.data(),
                                                  bins.data(), num_bins, bin_width);
                                ImPlot::EndPlot();
                            }
                        }

                        // Per-camera detail table (sortable)
                        if (ImGui::TreeNode("Per-Camera Details")) {
                            if (ImGui::BeginTable("exp_cam_details", 7,
                                    ImGuiTableFlags_RowBg |
                                    ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_Sortable |
                                    ImGuiTableFlags_SortTristate)) {
                                ImGui::TableSetupColumn("Camera", ImGuiTableColumnFlags_DefaultSort, 80.0f);
                                ImGui::TableSetupColumn("Dets",   ImGuiTableColumnFlags_PreferSortDescending, 45.0f);
                                ImGui::TableSetupColumn("Obs",    ImGuiTableColumnFlags_PreferSortDescending, 50.0f);
                                ImGui::TableSetupColumn("Mean",   ImGuiTableColumnFlags_PreferSortAscending, 55.0f);
                                ImGui::TableSetupColumn("Median", ImGuiTableColumnFlags_PreferSortAscending, 55.0f);
                                ImGui::TableSetupColumn("Std",    ImGuiTableColumnFlags_PreferSortAscending, 55.0f);
                                ImGui::TableSetupColumn("Max",    ImGuiTableColumnFlags_PreferSortAscending, 55.0f);
                                ImGui::TableHeadersRow();

                                // Build sorted index array
                                std::vector<int> sorted_idx(nc);
                                std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
                                if (ImGuiTableSortSpecs *sort_specs = ImGui::TableGetSortSpecs()) {
                                    if (sort_specs->SpecsCount > 0) {
                                        int col = sort_specs->Specs[0].ColumnIndex;
                                        bool asc = (sort_specs->Specs[0].SortDirection == ImGuiSortDirection_Ascending);
                                        std::sort(sorted_idx.begin(), sorted_idx.end(),
                                            [&](int a, int b) {
                                                double va = 0, vb = 0;
                                                switch (col) {
                                                    case 0: return asc ? (metrics[a].name < metrics[b].name)
                                                                       : (metrics[a].name > metrics[b].name);
                                                    case 1: va = metrics[a].detection_count; vb = metrics[b].detection_count; break;
                                                    case 2: va = metrics[a].observation_count; vb = metrics[b].observation_count; break;
                                                    case 3: va = metrics[a].mean_reproj; vb = metrics[b].mean_reproj; break;
                                                    case 4: va = metrics[a].median_reproj; vb = metrics[b].median_reproj; break;
                                                    case 5: va = metrics[a].std_reproj; vb = metrics[b].std_reproj; break;
                                                    case 6: va = metrics[a].max_reproj; vb = metrics[b].max_reproj; break;
                                                }
                                                return asc ? (va < vb) : (va > vb);
                                            });
                                    }
                                }

                                for (int i : sorted_idx) {
                                    const auto &m = metrics[i];
                                    ImGui::TableNextRow();
                                    ImGui::TableSetColumnIndex(0);
                                    ImGui::Text("%s", m.name.c_str());
                                    ImGui::TableSetColumnIndex(1);
                                    ImGui::Text("%d", m.detection_count);
                                    ImGui::TableSetColumnIndex(2);
                                    ImGui::Text("%d", m.observation_count);
                                    ImGui::TableSetColumnIndex(3);
                                    ImGui::Text("%.3f", m.mean_reproj);
                                    ImGui::TableSetColumnIndex(4);
                                    ImGui::Text("%.3f", m.median_reproj);
                                    ImGui::TableSetColumnIndex(5);
                                    ImGui::Text("%.3f", m.std_reproj);
                                    ImGui::TableSetColumnIndex(6);
                                    ImGui::Text("%.3f", m.max_reproj);
                                }
                                ImGui::EndTable();
                            }
                            ImGui::TreePop();
                        }

                        ImGui::Unindent();
                    } // end Quality Dashboard
                }
            }

            ImGui::Unindent();
            } // end CollapsingHeader("Aruco Calibration")
            ImGui::Spacing();
}
