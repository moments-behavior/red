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
            ImGui::Text("Cameras:      %d",
                        (int)state.config.cam_ordered.size());
            ImGui::Text("Board:        %d x %d  (%.1f mm squares)",
                        state.config.charuco_setup.w,
                        state.config.charuco_setup.h,
                        state.config.charuco_setup.square_side_length);

            // ---- Sub-section: Image Calibration ----
            bool has_images = !state.config.img_path.empty();
            if (has_images) {
            if (ImGui::CollapsingHeader("Image Calibration",
                    ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();
                ImGui::Text("Image Path: %s",
                            state.config.img_path.c_str());

                // Count images (cached)
                static std::string cached_img_path;
                static int cached_img_count = 0;
                static int cached_per_cam = 0;
                if (cached_img_path != state.config.img_path) {
                    cached_img_path = state.config.img_path;
                    auto files =
                        CalibrationTool::discover_images(state.config);
                    cached_img_count = (int)files.size();
                    if (!state.config.cam_ordered.empty()) {
                        cached_per_cam =
                            CalibrationTool::count_images_per_camera(
                                files, state.config.cam_ordered[0]);
                    }
                }
                ImGui::Text("Images: %d total (%d per camera)",
                            cached_img_count, cached_per_cam);

                // Load Images button
                if (!state.images_loaded) {
                    if (ImGui::Button("Load Images")) {
                        auto files =
                            CalibrationTool::discover_images(state.config);
                        if (files.empty()) {
                            state.status =
                                "Error: No matching images found in " +
                                state.config.img_path;
                        } else {
                            pm.media_folder = state.config.img_path;
                            pm.camera_names.clear();
                            imgs_names.clear();
                            cb.load_images(files);
                            state.images_loaded = true;
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
                    if (ImGui::Button("Close Images##aruco_close_img")) {
                        close_media_deferred(state.images_loaded, state.status, "Images closed");
                    }
                }

                ImGui::Separator();

                // Poll image pipeline future
                if (state.img_running && state.img_future.valid()) {
                    auto fs = state.img_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fs == std::future_status::ready) {
                        state.img_result = state.img_future.get();
                        state.img_running = false;
                        state.img_done = true;
                        if (state.img_result.success) {
                            state.project.image_output_folder =
                                state.img_result.output_folder;
                            std::string pf =
                                state.project.project_path + "/" +
                                state.project.project_name + ".redproj";
                            std::string se;
                            CalibrationTool::save_project(
                                state.project, pf, &se);
                            state.status =
                                "Image calibration complete! Reproj: " +
                                std::to_string(
                                    state.img_result.mean_reproj_error)
                                    .substr(0, 5) + " px";
                        } else {
                            state.status =
                                "Error: " + state.img_result.error;
                        }
                    }
                }

                // Calibrate (images) button
                bool img_can_run = state.config_loaded &&
                    !state.config.img_path.empty() &&
                    !state.aruco_running();
                ImGui::BeginDisabled(!img_can_run);
                if (ImGui::Button("Calibrate##img")) {
                    state.img_running = true;
                    state.img_done = false;
                    state.status = "Starting image calibration...";
                    std::string base =
                        state.project.project_path +
                        "/aruco_image_calibration";
                    state.img_future = std::async(
                        std::launch::async,
                        [config = state.config, base,
                         status_ptr = &state.status]() {
#ifdef __APPLE__
                            auto am = aruco_metal_create();
                            aruco_detect::GpuThresholdFunc gfn =
                                am ? aruco_metal_threshold_batch : nullptr;
                            auto r = CalibrationPipeline::run_full_pipeline(
                                config, base, status_ptr,
                                nullptr, gfn, am);
                            aruco_metal_destroy(am);
                            return r;
#else
                            return CalibrationPipeline::run_full_pipeline(
                                config, base, status_ptr);
#endif
                        });
                }
                ImGui::EndDisabled();
                if (state.img_running) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                       "Running...");
                }
                if (state.img_done && state.img_result.success) {
                    ImGui::TextColored(
                        ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                        "Mean reproj error: %.3f px",
                        state.img_result.mean_reproj_error);
                    ImGui::Text("Output: %s",
                        state.project.image_output_folder.c_str());
                }

                // Show 3D button for image experimental results (loads from disk if needed)
                {
                    std::string exp_folder = state.project.project_path + "/aruco_image_experimental";
                    bool has_db = CalibrationPipeline::has_calibration_database(exp_folder);
                    bool already_showing = (state.calib_viewer.show &&
                        state.calib_viewer.result == &state.exp_img_result);
                    // Also check if experimental result is in memory
                    if (!has_db && state.exp_img_done && state.exp_img_result.success)
                        has_db = true;
                    ImGui::BeginDisabled(!has_db);
                    if (ImGui::Button("Load Calibration##img_exp_load")) {
                        if (state.exp_img_done && state.exp_img_result.success) {
                            state.calib_viewer.result = &state.exp_img_result;
                        } else {
                            state.loaded_result = CalibrationPipeline::load_calibration_from_folder(
                                exp_folder, state.config.cam_ordered);
                            if (state.loaded_result.success) {
                                state.exp_img_result = state.loaded_result;
                                state.exp_img_done = true; // enables Quality Dashboard
                                state.calib_viewer.result = &state.exp_img_result;
                                state.status = "Loaded calibration: " +
                                    std::to_string(state.exp_img_result.mean_reproj_error).substr(0,5) + " px";
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
                        ImGui::TextDisabled("(run Experimental first)");
                    }
                }

                ImGui::Spacing();

                // Poll experimental image pipeline future
                if (state.exp_img_running && state.exp_img_future.valid()) {
                    auto fs = state.exp_img_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fs == std::future_status::ready) {
                        state.exp_img_result = state.exp_img_future.get();
                        state.exp_img_running = false;
                        state.exp_img_done = true;
                        if (state.exp_img_result.success) {
                            state.project.image_experimental_folder =
                                state.exp_img_result.output_folder;
                            std::string pf =
                                state.project.project_path + "/" +
                                state.project.project_name + ".redproj";
                            std::string se;
                            CalibrationTool::save_project(
                                state.project, pf, &se);
                            state.status =
                                "Experimental image calibration complete! Reproj: " +
                                std::to_string(
                                    state.exp_img_result.mean_reproj_error)
                                    .substr(0, 5) + " px";
                            // Auto-open 3D viewer
                            state.calib_viewer.result = &state.exp_img_result;
                            state.calib_viewer.show = true;
                            state.calib_viewer.selected_camera = -1;
                            state.calib_viewer.cached_selection = -2;
                        } else {
                            state.status =
                                "Error: " + state.exp_img_result.error;
                        }
                    }
                }

                // Experimental (images) button
                ImGui::BeginDisabled(!img_can_run);
                if (ImGui::Button("Experimental##exp_img")) {
                    state.exp_img_running = true;
                    state.exp_img_done = false;
                    state.status = "Starting experimental image calibration...";
                    std::string base =
                        state.project.project_path +
                        "/aruco_image_experimental";
                    state.exp_img_future = std::async(
                        std::launch::async,
                        [config = state.config, base,
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
                ImGui::EndDisabled();
                if (state.exp_img_running) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                       "Running...");
                }
                if (state.exp_img_done && state.exp_img_result.success) {
                    ImGui::TextColored(
                        ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                        "Experimental reproj error: %.3f px",
                        state.exp_img_result.mean_reproj_error);
                    ImGui::Text("Output: %s",
                        state.project.image_experimental_folder.c_str());
                    ImGui::SameLine();
                    if (ImGui::Button("Load Calibration##exp_img")) {
                        state.calib_viewer.result = &state.exp_img_result;
                        state.calib_viewer.show = true;
                    }
                }

                ImGui::Unindent();
            } // end Image Calibration header
            } // end has_images

            // ---- Sub-section: Video Calibration ----
            bool has_videos = !state.project.aruco_video_folder.empty();
            if (has_videos) {
            if (ImGui::CollapsingHeader("Video Calibration",
                    ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                // Cache video discovery
                static std::string cached_vid_folder;
                if (cached_vid_folder != state.project.aruco_video_folder) {
                    cached_vid_folder = state.project.aruco_video_folder;
                    auto vids = CalibrationTool::discover_aruco_videos(
                        state.project.aruco_video_folder,
                        state.config.cam_ordered);
                    state.aruco_video_count = (int)vids.size();
                    if (!vids.empty()) {
                        state.aruco_total_frames =
                            CalibrationPipeline::get_video_frame_count(
                                vids.begin()->second);
                    }
                }

                ImGui::Text("Video Path: %s",
                            state.project.aruco_video_folder.c_str());
                ImGui::Text("Videos: %d cameras, ~%d frames each",
                            state.aruco_video_count,
                            state.aruco_total_frames);

                // Load Videos button (display in camera viewports)
                if (!state.aruco_videos_loaded) {
                    ImGui::BeginDisabled(
                        state.aruco_video_count == 0 ||
                        state.aruco_running());
                    if (ImGui::Button("Load Videos##aruco_vid")) {
                        // Defer unload+load to next frame start
                        // (freeing Metal textures mid-frame crashes ImGui)
                        state.status = "Loading aruco videos...";
                        cb.deferred->enqueue([&state, &pm, &ps, &cb,
                                              &imgs_names, dc_context
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
                                    state.project.aruco_video_folder;
                                pm.camera_names.clear();
                                for (const auto &cn :
                                     state.config.cam_ordered)
                                    pm.camera_names.push_back("Cam" + cn);
                                cb.load_videos();
                                cb.print_metadata();
                                state.aruco_total_frames =
                                    dc_context->estimated_num_frames;
                                state.aruco_videos_loaded = true;
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
                    if (ImGui::Button("Close Videos##aruco_close")) {
                        close_media_deferred(state.aruco_videos_loaded, state.status, "Videos closed");
                    }
                }

                ImGui::Separator();

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

                // Poll video pipeline future
                if (state.vid_running && state.vid_future.valid()) {
                    auto fs = state.vid_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fs == std::future_status::ready) {
                        state.vid_result = state.vid_future.get();
                        state.vid_running = false;
                        state.vid_done = true;
                        if (state.vid_result.success) {
                            state.project.video_output_folder =
                                state.vid_result.output_folder;
                            std::string pf =
                                state.project.project_path + "/" +
                                state.project.project_name + ".redproj";
                            std::string se;
                            CalibrationTool::save_project(
                                state.project, pf, &se);
                            state.status =
                                "Video calibration complete! Reproj: " +
                                std::to_string(
                                    state.vid_result.mean_reproj_error)
                                    .substr(0, 5) + " px";
                        } else {
                            state.status =
                                "Error: " + state.vid_result.error;
                        }
                    }
                }

                // Calibrate (video) button
                bool vid_can_run = state.config_loaded &&
                    state.aruco_video_count > 0 &&
                    !state.aruco_running();
                ImGui::BeginDisabled(!vid_can_run);
                if (ImGui::Button("Calibrate##vid")) {
                    state.vid_running = true;
                    state.vid_done = false;
                    state.status = "Starting video calibration...";
                    std::string base =
                        state.project.project_path +
                        "/aruco_video_calibration";

                    CalibrationPipeline::VideoFrameRange vfr;
                    vfr.video_folder = state.project.aruco_video_folder;
                    vfr.cam_ordered = state.config.cam_ordered;
                    vfr.start_frame = state.aruco_start_frame;
                    vfr.stop_frame = state.aruco_stop_frame;
                    vfr.frame_step = state.aruco_frame_step;

                    state.vid_future = std::async(
                        std::launch::async,
                        [config = state.config, base,
                         status_ptr = &state.status, vfr]() {
#ifdef __APPLE__
                            auto am = aruco_metal_create();
                            aruco_detect::GpuThresholdFunc gfn =
                                am ? aruco_metal_threshold_batch : nullptr;
                            auto r = CalibrationPipeline::run_full_pipeline(
                                config, base, status_ptr,
                                &vfr, gfn, am);
                            aruco_metal_destroy(am);
                            return r;
#else
                            return CalibrationPipeline::run_full_pipeline(
                                config, base, status_ptr, &vfr);
#endif
                        });
                }
                ImGui::EndDisabled();
                if (state.vid_running) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                       "Running...");
                }
                if (state.vid_done && state.vid_result.success) {
                    ImGui::TextColored(
                        ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                        "Mean reproj error: %.3f px",
                        state.vid_result.mean_reproj_error);
                    ImGui::Text("Output: %s",
                        state.project.video_output_folder.c_str());
                }

                ImGui::Spacing();

                // Poll experimental video pipeline future
                if (state.exp_vid_running && state.exp_vid_future.valid()) {
                    auto fs = state.exp_vid_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fs == std::future_status::ready) {
                        state.exp_vid_result = state.exp_vid_future.get();
                        state.exp_vid_running = false;
                        state.exp_vid_done = true;
                        if (state.exp_vid_result.success) {
                            state.project.video_experimental_folder =
                                state.exp_vid_result.output_folder;
                            std::string pf =
                                state.project.project_path + "/" +
                                state.project.project_name + ".redproj";
                            std::string se;
                            CalibrationTool::save_project(
                                state.project, pf, &se);
                            state.status =
                                "Experimental video calibration complete! Reproj: " +
                                std::to_string(
                                    state.exp_vid_result.mean_reproj_error)
                                    .substr(0, 5) + " px";
                            // Auto-open 3D viewer
                            state.calib_viewer.result = &state.exp_vid_result;
                            state.calib_viewer.show = true;
                            state.calib_viewer.selected_camera = -1;
                            state.calib_viewer.cached_selection = -2;
                        } else {
                            state.status =
                                "Error: " + state.exp_vid_result.error;
                        }
                    }
                }

                // Experimental (video) button
                ImGui::BeginDisabled(!vid_can_run);
                if (ImGui::Button("Experimental##exp_vid")) {
                    state.exp_vid_running = true;
                    state.exp_vid_done = false;
                    state.status = "Starting experimental video calibration...";
                    std::string base =
                        state.project.project_path +
                        "/aruco_video_experimental";

                    CalibrationPipeline::VideoFrameRange exp_vfr;
                    exp_vfr.video_folder = state.project.aruco_video_folder;
                    exp_vfr.cam_ordered = state.config.cam_ordered;
                    exp_vfr.start_frame = state.aruco_start_frame;
                    exp_vfr.stop_frame = state.aruco_stop_frame;
                    exp_vfr.frame_step = state.aruco_frame_step;

                    state.exp_vid_future = std::async(
                        std::launch::async,
                        [config = state.config, base,
                         status_ptr = &state.status, exp_vfr]() {
#ifdef __APPLE__
                            auto am = aruco_metal_create();
                            aruco_detect::GpuThresholdFunc gfn =
                                am ? aruco_metal_threshold_batch : nullptr;
                            auto r = CalibrationPipeline::run_experimental_pipeline(
                                config, base, status_ptr,
                                &exp_vfr, gfn, am);
                            aruco_metal_destroy(am);
                            return r;
#else
                            return CalibrationPipeline::run_experimental_pipeline(
                                config, base, status_ptr, &exp_vfr);
#endif
                        });
                }
                ImGui::EndDisabled();
                if (state.exp_vid_running) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                       "Running...");
                }
                if (state.exp_vid_done && state.exp_vid_result.success) {
                    ImGui::TextColored(
                        ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                        "Experimental reproj error: %.3f px",
                        state.exp_vid_result.mean_reproj_error);
                    ImGui::Text("Output: %s",
                        state.project.video_experimental_folder.c_str());
                    ImGui::SameLine();
                    if (ImGui::Button("Load Calibration##exp_vid")) {
                        state.calib_viewer.result = &state.exp_vid_result;
                        state.calib_viewer.show = true;
                        state.calib_viewer.selected_camera = -1;
                        state.calib_viewer.cached_selection = -2;
                    }
                }

                ImGui::Unindent();
            } // end Video Calibration header
            } // end has_videos

            // ---- Quality Dashboard (visible after any experimental pipeline completes) ----
            {
                const CalibrationPipeline::CalibrationResult *exp_result = nullptr;
                if (state.exp_vid_done && state.exp_vid_result.success)
                    exp_result = &state.exp_vid_result;
                else if (state.exp_img_done && state.exp_img_result.success)
                    exp_result = &state.exp_img_result;

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
