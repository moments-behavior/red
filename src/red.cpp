#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"
#include "IconsForkAwesome.h"
#include "camera.h"
#include "filesystem"
#include "global.h"
#include "gui.h"
#include "gui/help_window.h"
#include "gui/jarvis_export_window.h"
#include "gui/export_window.h"
#include "gui/bbox_tool.h"
#include "gui/obb_tool.h"
#include "gui/sam_tool.h"
#include "jarvis_inference.h"
#ifdef __APPLE__
#include "jarvis_coreml.h"
#endif
#include "gui/jarvis_predict_window.h"
#include "gui/annotation_dialog.h"
#include "gui/calibration_tool_window.h"
#include "gui/labeling_tool_window.h"
#include "gui/project_window.h"
#include "gui/settings_window.h"
#include "gui/main_menu_dialogs.h"
#include "gui/main_menu_bar.h"
#include "gui/panel_registry.h"
#include "gui/transport_bar.h"
#include "gui/popup_stack.h"
#include "gui/toast.h"
#include "imgui_impl_glfw.h"
#ifdef __APPLE__
#include "metal_context.h"
#include <CoreFoundation/CoreFoundation.h>  // CFRelease for CVPixelBuffer
#include <mach-o/dyld.h>                   // _NSGetExecutablePath
#else
#include "imgui_impl_opengl3.h"
#endif
#include "implot.h"
#include "implot_internal.h"
#include "project.h"
#include "render.h"
#include "skeleton.h"
#include "utils.h"
#include "calibration_tool.h"
#include "calibration_pipeline.h"
#include "app_context.h"
#include "deferred_queue.h"
#include "user_settings.h"
#include "jarvis_export.h"
#include "laser_calibration.h"
#include "aruco_metal.h"
#include "laser_metal.h"
#include <ImGuiFileDialog.h>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <misc/cpp/imgui_stdlib.h> // for InputText(std::string&)
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#ifndef __APPLE__
#include "kernel.cuh"
#endif
#include "keypoints_table.h"

static void print_video_metadata(const std::vector<FFmpegDemuxer *> &demuxers,
                                 const std::vector<std::string> &camera_names,
                                 int seek_interval) {
    if (demuxers.empty()) return;
    int n = (int)demuxers.size();
    // Find max camera name length for column width
    size_t max_name = 6; // minimum "Camera"
    for (int i = 0; i < n; i++) {
        size_t len = (i < (int)camera_names.size()) ? camera_names[i].size() : 0;
        if (len > max_name) max_name = len;
    }
    int name_w = (int)max_name + 2;

    std::cout << "\nVideo metadata (" << n << " camera" << (n != 1 ? "s" : "") << "):\n";
    std::cout << std::left << std::setw(name_w) << "Camera"
              << std::right << std::setw(12) << "Frame Rate"
              << std::setw(12) << "Length"
              << std::setw(16) << "Seek Interval" << "\n";
    for (int i = 0; i < n; i++) {
        std::string name = (i < (int)camera_names.size()) ? camera_names[i] : "?";
        std::cout << std::left << std::setw(name_w) << name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << demuxers[i]->GetFramerate()
                  << std::setw(12) << demuxers[i]->GetDuration()
                  << std::setw(16) << seek_interval << "\n";
    }
    std::cout << std::endl;
}

static void print_project_summary(const ProjectManager &pm,
                                  const std::string &skeleton_name,
                                  const std::string &label_folder) {
    namespace fs = std::filesystem;
    std::cout << "\n=== Project Loaded ===" << std::endl;
    std::cout << "Name:        " << pm.project_name << std::endl;
    std::cout << "Skeleton:    " << skeleton_name << std::endl;
    std::cout << "Cameras:     " << pm.camera_names.size() << " (";
    for (size_t i = 0; i < pm.camera_names.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << pm.camera_names[i];
    }
    std::cout << ")" << std::endl;
    std::cout << "Media:       " << pm.media_folder << std::endl;
    std::cout << "Calibration: " << pm.calibration_folder << std::endl;

    if (!label_folder.empty()) {
        std::string folder_name = fs::path(label_folder).filename().string();
        // Count labeled frames from keypoints3d.csv
        int labeled_frames = 0;
        std::vector<int> labeled_frame_ids;
        std::string kp3d_path = label_folder + "/keypoints3d.csv";
        {
            std::ifstream f(kp3d_path);
            if (f.is_open()) {
                std::string line;
                bool header = true;
                while (std::getline(f, line)) {
                    if (line.empty()) continue;
                    if (header) { header = false; continue; }
                    labeled_frames++;
                    // Parse frame_id (first field before comma)
                    auto comma = line.find(',');
                    if (comma != std::string::npos) {
                        try { labeled_frame_ids.push_back(std::stoi(line.substr(0, comma))); }
                        catch (...) {}
                    }
                }
            }
        }
        std::cout << "Labels:      " << folder_name;
        if (labeled_frames > 0)
            std::cout << " (" << labeled_frames << " labeled frames)";
        std::cout << std::endl;

        // Count complete/incomplete 2D annotations per camera x frame
        if (!labeled_frame_ids.empty() && !pm.camera_names.empty()) {
            int n_complete = 0, n_incomplete = 0;
            for (const auto &cam : pm.camera_names) {
                std::string csv_path = label_folder + "/" + cam + ".csv";
                // Parse 2D CSV: check each labeled frame for 1e7 sentinels
                std::map<int, bool> frame_complete; // frame_id -> all valid?
                std::ifstream cf(csv_path);
                if (cf.is_open()) {
                    std::string line;
                    bool header = true;
                    while (std::getline(cf, line)) {
                        if (line.empty()) continue;
                        if (header) { header = false; continue; }
                        std::stringstream ss(line);
                        std::string token;
                        if (!std::getline(ss, token, ',')) continue;
                        int fid = 0;
                        try { fid = std::stoi(token); } catch (...) { continue; }
                        bool valid = true;
                        while (std::getline(ss, token, ',')) {
                            try {
                                if (std::stod(token) == 1e7) { valid = false; break; }
                            } catch (...) { valid = false; break; }
                        }
                        frame_complete[fid] = valid;
                    }
                }
                for (int fid : labeled_frame_ids) {
                    auto it = frame_complete.find(fid);
                    if (it != frame_complete.end() && it->second)
                        n_complete++;
                    else
                        n_incomplete++;
                }
            }
            std::cout << "Annotations: " << n_complete << " complete, "
                      << n_incomplete << " incomplete ("
                      << pm.camera_names.size() << " cameras x "
                      << labeled_frame_ids.size() << " frames)" << std::endl;
        }
    } else {
        std::cout << "Labels:      (none)" << std::endl;
    }
    std::cout << std::endl;
}

// discover_mp4_cameras moved to gui/annotation_dialog.h

int main(int argc, char **argv) {
    gx_context *window = (gx_context *)malloc(sizeof(gx_context));
    *window =
        (gx_context){.swap_interval = 1, // use vsync
                     .width = 1920,
                     .height = 1080,
                     .render_target_title = (char *)malloc(100), // window title
                     .glsl_version = (char *)malloc(100)};
    // Resolve the real path of the executable.
    // std::filesystem::canonical(argv[0]) fails when the binary is invoked
    // via PATH (argv[0] is just "red" with no directory component).
    // On macOS use _NSGetExecutablePath which always returns the full path.
#ifdef __APPLE__
    {
        char exe_buf[PATH_MAX];
        uint32_t exe_buf_size = sizeof(exe_buf);
        if (_NSGetExecutablePath(exe_buf, &exe_buf_size) == 0)
            window->exe_dir = std::filesystem::canonical(exe_buf).parent_path().string();
        else
            window->exe_dir = std::filesystem::canonical(argv[0]).parent_path().string();
    }
#else
    window->exe_dir =
        std::filesystem::canonical(argv[0]).parent_path().string();
#endif

    render_initialize_target(window);
    RenderScene *scene = (RenderScene *)malloc(sizeof(RenderScene));
    std::string red_data_dir;
    std::string media_root_dir;
    prepare_application_folders(red_data_dir, media_root_dir);
    UserSettings user_settings = load_user_settings();
    std::string skeleton_dir = red_data_dir + "/skeleton";
    std::vector<std::thread> decoder_threads;
    std::vector<FFmpegDemuxer *> demuxers;

    DecoderContext *dc_context =
        (DecoderContext *)malloc(sizeof(DecoderContext));
    *dc_context = (DecoderContext){.decoding_flag = false,
                                   .stop_flag = false,
                                   .total_num_frame = int(INT_MAX),
                                   .estimated_num_frames = 0,
                                   .gpu_index = 0,
                                   .seek_interval = 250,  // overwritten by auto-detect in media_loader
                                   .video_fps = 60.0f};

    // gui states, todo: bundle this later
    LabelingToolState labeling_state;
    bool save_requested = false;
    int current_frame_num = 0;
    int previous_frame_num = -1;
    std::vector<std::string> imgs_names;

    // for labeling
    SkeletonContext skeleton;
    bool keypoints_find = false;
    std::map<std::string, SkeletonPrimitive> skeleton_map = skeleton_get_all();

    // Annotation model
    AnnotationMap annotations;

    // others
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
    ImGuiIO &io = ImGui::GetIO();
    static std::string project_ini_path;

    ImPlotStyle &style = ImPlot::GetStyle();
    ImVec4 *colors = style.Colors;


    // JARVIS Export Tool state
    JarvisExportState jarvis_export_state;
    // JARVIS Import Tool state
    JarvisImportState jarvis_import_state;
    JarvisPredictState jarvis_predict_state;
    jarvis_export_state.margin = user_settings.jarvis_margin;
    jarvis_export_state.train_ratio = user_settings.jarvis_train_ratio;
    jarvis_export_state.seed = user_settings.jarvis_seed;
    jarvis_export_state.jpeg_quality = user_settings.jarvis_jpeg_quality;

    // Calibration Tool state (extracted to gui/calibration_tool_window.h)
    CalibrationToolState calib_state;
    calib_state.project.project_root_path =
        user_settings.default_project_root_path.empty()
            ? red_data_dir
            : user_settings.default_project_root_path;
    if (!user_settings.default_media_root_path.empty())
        calib_state.project.config_file = user_settings.default_media_root_path;

    // Annotation dialog state
    AnnotationDialogState annot_state;

    // Settings window state
    SettingsState settings_state;
    TransportBarState transport_state;

    // New annotation tools state
    ExportWindowState export_state;
    export_state.margin = user_settings.jarvis_margin;
    export_state.train_ratio = user_settings.jarvis_train_ratio;
    export_state.seed = user_settings.jarvis_seed;
    export_state.jpeg_quality = user_settings.jarvis_jpeg_quality;
    BBoxToolState bbox_state;
    OBBToolState obb_state;
    SamToolState sam_tool_state;
    SamState sam_state;
    JarvisState jarvis_state;
#ifdef __APPLE__
    JarvisCoreMLState jarvis_coreml_state;
#endif

    // Default SAM model paths: look relative to exe (../models/mobilesam/)
    // and in the source tree. User can override in SAM Assist panel.
    {
        std::string exe = window->exe_dir;
        std::vector<std::string> search = {
            exe + "/../models/mobilesam",   // build tree (release/)
            exe + "/models/mobilesam",      // installed
            exe + "/../share/red/models/mobilesam", // Homebrew
        };
        for (const auto &dir : search) {
            std::string enc = dir + "/mobile_sam_encoder.onnx";
            std::string dec = dir + "/mobile_sam_decoder.onnx";
            if (std::filesystem::exists(enc) && std::filesystem::exists(dec)) {
                sam_tool_state.encoder_path = std::filesystem::canonical(enc).string();
                sam_tool_state.decoder_path = std::filesystem::canonical(dec).string();
                break;
            }
        }
    }

    annot_state.video_folder = user_settings.default_media_root_path.empty()
                                   ? media_root_dir
                                   : user_settings.default_media_root_path;

    colors[ImPlotCol_Crosshairs] = ImVec4(0.3f, 0.10f, 0.64f, 1.00f);

    int label_buffer_size = user_settings.default_buffer_size;
    bool show_help_window = false;
    std::vector<bool> is_view_focused;
    bool input_is_imgs = false;
    PopupStack popups;
    ToastQueue toasts;
    DeferredQueue deferred;

    std::unordered_map<std::string, bool> window_was_decoding;
    std::unordered_map<std::string, bool> window_is_visible;  // actual ImGui visibility (prev frame)
    PlaybackState ps;
    ps.set_playback_speed = user_settings.default_playback_speed;
    ps.realtime_playback = user_settings.default_realtime_playback;
    DisplayState display;
    display.brightness = user_settings.default_brightness;
    display.contrast = user_settings.default_contrast;
    display.pivot_midgray = user_settings.default_pivot_midgray;

    // variables for project management
    ProjectManager pm = ProjectManager();
    pm.project_root_path = user_settings.default_project_root_path.empty()
                               ? red_data_dir
                               : user_settings.default_project_root_path;
    pm.media_folder = user_settings.default_media_root_path.empty()
                          ? media_root_dir
                          : user_settings.default_media_root_path;

    bool main_loop_running = false;

#ifdef __APPLE__
    // Per-camera last-uploaded frame number for Metal (skip redundant uploads)
    std::vector<int> mac_last_uploaded_frame(MAX_VIEWS, -1);
#endif

    // Build AppContext — a reference bundle for all shared state
    AppContext ctx{
        pm, ps, scene, dc_context,
        skeleton, skeleton_map,
        annotations,
        popups, toasts, deferred,
        user_settings, red_data_dir, skeleton_dir,
        imgs_names, demuxers, decoder_threads,
        is_view_focused, window_was_decoding,
        input_is_imgs, label_buffer_size,
        display, window, save_requested, project_ini_path, main_loop_running
#ifdef __APPLE__
        , mac_last_uploaded_frame
#endif
    };

    // Callbacks for static console-output functions in this file
    auto print_metadata = [&]() {
        print_video_metadata(demuxers, pm.camera_names, dc_context->seek_interval);
    };
    auto print_summary = [&](const std::string &label_folder) {
        print_project_summary(pm, pm.skeleton_name, label_folder);
    };

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::filesystem::path path = argv[i];
            if (std::filesystem::is_directory(path))
                path = path.parent_path() / (path.filename().string() + ".redproj");
            ProjectManager loaded;
            std::string err;
            if (!load_project_manager_json(&loaded, path, &err)) {
                popups.pushError(err);
            } else {
                pm = loaded;
                if (setup_project(pm, skeleton, skeleton_map, &err)) {
                    on_project_loaded(ctx, print_metadata, print_summary);
                } else
                    popups.pushError(err);
            }
        }
    }

    // Calibration Tool callbacks
    CalibrationToolCallbacks calib_cb;
    calib_cb.load_images = [&ctx](std::map<std::string, std::string> &files) {
        load_images(files, ctx.ps, ctx.pm, ctx.imgs_names, ctx.scene,
                    ctx.dc_context, ctx.label_buffer_size,
                    ctx.decoder_threads, ctx.is_view_focused,
                    ctx.window_was_decoding);
        ctx.input_is_imgs = true;
    };
    calib_cb.load_videos = [&ctx]() {
        std::map<std::string, std::string> empty_selected_files;
        load_videos(empty_selected_files, ctx.ps, ctx.pm,
                    ctx.window_was_decoding, ctx.demuxers, ctx.dc_context,
                    ctx.scene, ctx.label_buffer_size, ctx.decoder_threads,
                    ctx.is_view_focused);
        ctx.input_is_imgs = false;
    };
    calib_cb.unload_media = [&ctx]() {
        unload_media(ctx.ps, ctx.pm, ctx.demuxers, ctx.dc_context,
                     ctx.scene, ctx.decoder_threads,
                     ctx.is_view_focused, ctx.window_was_decoding);
    };
    calib_cb.copy_default_layout = [&ctx](const std::string &proj_path) {
        copy_default_layout_to_project(ctx, proj_path);
    };
    calib_cb.switch_ini = [&ctx](const std::string &proj_path) {
        ctx.project_ini_path = proj_path + "/imgui_layout.ini";
        ImGuiIO &io = ImGui::GetIO();
        io.IniFilename = ctx.project_ini_path.c_str();
        if (std::filesystem::exists(ctx.project_ini_path))
            ImGui::LoadIniSettingsFromDisk(ctx.project_ini_path.c_str());
    };
    calib_cb.print_metadata = print_metadata;
    calib_cb.deferred = &deferred;

    // Annotation create callback (shared by annotation dialog panel)
    AnnotationCreateCallback annot_create_cb =
        [&](ProjectManager &pm_ref, std::string &err) -> bool {
        if (!ensure_dir_exists(pm_ref.project_path, &err))
            return false;
        if (!setup_project(pm_ref, skeleton, skeleton_map, &err))
            return false;
        std::filesystem::path redproj_path =
            std::filesystem::path(pm_ref.project_path) / (pm_ref.project_name + ".redproj");
        if (!save_project_manager_json(pm_ref, redproj_path, &err))
            return false;
        on_project_loaded(ctx, print_metadata, print_summary);
        return true;
    };

    // Panel registry — replaces manual draw calls
    PanelRegistry panels;
    panels.add({"Create Project",
                [&]() { DrawProjectWindow(ctx); }, nullptr});
    panels.add({"Annotation Dialog",
                [&]() { DrawAnnotationDialog(annot_state, ctx, annot_create_cb); },
                nullptr});
    panels.add({"Keypoints",
                [&]() { DrawKeypointsWindow(ctx, current_frame_num); },
                [&]() { return pm.plot_keypoints_flag; }});
    panels.add({"Labeling Tool",
                [&]() {
                    DrawLabelingToolWindow(labeling_state, ctx,
                                           current_frame_num, keypoints_find);
                    if (keypoints_find &&
                        ImGui::IsKeyPressed(ImGuiKey_T, false) &&
                        !ImGui::GetIO().WantTextInput) {
                        if (!pm.camera_params.empty()) {
                            reprojection(annotations.at(current_frame_num),
                                         &skeleton, pm.camera_params, scene);
                        } else {
                            toasts.push("No calibration loaded",
                                        Toast::Warning, 3.0f);
                        }
                    }
                },
                [&]() { return pm.plot_keypoints_flag; }});
    panels.add({"Help", [&]() { DrawHelpWindow(show_help_window); }, nullptr});
    panels.add({"JARVIS Export",
                [&]() { DrawJarvisExportWindow(jarvis_export_state, ctx); },
                nullptr});
    panels.add({"JARVIS Import",
                [&]() { DrawJarvisImportWindow(jarvis_import_state, ctx); },
                nullptr});
    panels.add({"Calibration Tool",
                [&]() { DrawCalibrationToolWindow(calib_state, ctx, calib_cb); },
                nullptr});
    panels.add({"Settings",
                [&]() { DrawSettingsWindow(settings_state, ctx); },
                nullptr});
    panels.add({"Export Tool",
                [&]() { DrawExportWindow(export_state, ctx, annotations); },
                nullptr});
    panels.add({"Bbox Tool",
                [&]() { DrawBBoxToolWindow(bbox_state, ctx); },
                nullptr});
    panels.add({"OBB Tool",
                [&]() { DrawOBBToolWindow(obb_state, ctx); },
                nullptr});
    panels.add({"SAM Assist",
                [&]() { DrawSamToolWindow(sam_tool_state, sam_state, ctx); },
                nullptr});
    panels.add({"JARVIS Predict",
                [&]() { DrawJarvisPredictWindow(jarvis_predict_state, jarvis_state,
#ifdef __APPLE__
                                                 jarvis_coreml_state,
#endif
                                                 ctx); },
                nullptr});

    main_loop_running = true;
    while (!glfwWindowShouldClose(window->render_target)) {
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        // When minimized, block until the user restores the window.
        // Avoids spinning the render loop (Metal nextDrawable returns nil,
        // CPU/GPU burn for nothing).
        if (glfwGetWindowAttrib(window->render_target, GLFW_ICONIFIED)) {
            glfwWaitEvents();
            continue;
        }

#ifdef __APPLE__
        // Acquire drawable and open command buffer; calls ImGui_ImplMetal_NewFrame
        if (!metal_begin_frame()) {
            // No drawable available (window minimized) — skip this frame
            continue;
        }

        // Invalidate cached uploads when display params change (forces re-upload + shader)
        {
            static float prev_contrast = display.contrast;
            static int   prev_brightness = display.brightness;
            static bool  prev_pivot = display.pivot_midgray;
            if (display.contrast != prev_contrast ||
                display.brightness != prev_brightness ||
                display.pivot_midgray != prev_pivot) {
                std::fill(mac_last_uploaded_frame.begin(),
                          mac_last_uploaded_frame.end(), -1);
                prev_contrast   = display.contrast;
                prev_brightness = display.brightness;
                prev_pivot      = display.pivot_midgray;
            }
        }
#endif

        // Start the Dear ImGui frame
#ifdef __APPLE__
        // ImGui_ImplMetal_NewFrame was called inside metal_begin_frame()
        (void)0;
#else
        ImGui_ImplOpenGL3_NewFrame();
#endif
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Flush deferred callbacks (runs before any rendering to avoid
        // freeing Metal textures that ImGui draw commands still reference).
        deferred.flush();

        // App-level main menu bar (always visible)
        DrawMainMenuBar(ctx, calib_state, annot_state, settings_state,
                        jarvis_export_state, jarvis_import_state,
                        export_state, bbox_state, obb_state, sam_tool_state,
                        jarvis_predict_state, show_help_window);

        // --- Update playback time ---
        auto now = std::chrono::steady_clock::now();

        if (ps.play_video) {
            ps.accumulated_play_time +=
                std::chrono::duration<double>(now - ps.last_play_time_start)
                    .count() *
                ps.set_playback_speed;
            ps.last_play_time_start = now;
        }
        double playback_time_now = ps.accumulated_play_time;

        // Instantaneous speed computation (logic, not UI)
        if (ps.video_loaded) {
            auto now_wall = std::chrono::steady_clock::now();
            double wall_seconds =
                std::chrono::duration<double>(now_wall - ps.last_wall_time_playspeed).count();
            int frame_delta = current_frame_num - ps.last_frame_num_playspeed;
            if (wall_seconds > 0.5 && ps.play_video) {
                ps.inst_speed = frame_delta / (dc_context->video_fps * wall_seconds);
                ps.last_frame_num_playspeed = current_frame_num;
                ps.last_wall_time_playspeed = now_wall;
            }
        }

        // Transport bar — horizontal controls below menu bar (only when video loaded)
        DrawTransportBar(transport_state, ctx, current_frame_num);

        ImGui::DockSpaceOverViewport(0x00000001);

        // Draw all registered panels
        panels.drawAll();

        // Handle main menu file dialogs
        HandleMainMenuDialogs(ctx, calib_state, annot_state,
                              media_root_dir, print_metadata, print_summary);

        static int select_corr_head = 0;
        if (ps.video_loaded && (!ps.play_video)) {
            int visible_idx = 0;
            if (!ps.pause_seeked) {
                for (int i = 0; i < scene->num_cams; i++) {
                    if (window_was_decoding[pm.camera_names[i]]) {
                        visible_idx = i;
                        break;
                    }
                }
            }

            // Frame buffer keyboard navigation — global so it works
            // even when the "Frames in the buffer" tab is hidden.
            bool selection_changed = false;

            // Clamp just in case
            if (ps.pause_selected < 0)
                ps.pause_selected = 0;
            if (ps.pause_selected >= (int)scene->size_of_buffer)
                ps.pause_selected = scene->size_of_buffer - 1;

            if (ImGui::IsKeyPressed(ImGuiKey_Comma, true) &&
                !io.WantTextInput) {
                if (ps.pause_selected > 0) {
                    ps.pause_selected--;
                    selection_changed = true;
                }
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Period, true) &&
                !io.WantTextInput) {
                if (ps.pause_selected < (int)scene->size_of_buffer - 1) {
                    ps.pause_selected++;
                    selection_changed = true;
                }
            }

            select_corr_head =
                (ps.pause_selected + ps.read_head) % scene->size_of_buffer;
            current_frame_num =
                scene->display_buffer[visible_idx][select_corr_head]
                    .frame_number;
        }

        // Frame Buffer window — always visible when video is loaded,
        // grayed out when playing to avoid disruptive tab appearing/disappearing.
        if (ps.video_loaded) {
            ImGui::SetNextWindowSize(ImVec2(500, 90), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Frame Buffer")) {
                if (ps.play_video) {
                    ImGui::BeginDisabled();
                    ImGui::TextDisabled("Playing...");
                    ImGui::EndDisabled();
                } else {
                    int visible_idx = 0;
                    if (!ps.pause_seeked) {
                        for (int i = 0; i < scene->num_cams; i++) {
                            if (window_was_decoding[pm.camera_names[i]]) {
                                visible_idx = i;
                                break;
                            }
                        }
                    }

                    // Horizontal scrollable row of frames with vertical text
                    float scale = 1.15f;
                    float font_size = ImGui::GetFontSize() * scale;
                    float item_w = font_size + 2.0f;
                    float item_h = ImGui::GetContentRegionAvail().y;
                    if (item_h < 40.0f) item_h = 40.0f;

                    ImGui::SetWindowFontScale(scale);
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1.0f, 0.0f));
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2.0f, 2.0f));
                    ImGui::BeginChild("##hscroll", ImVec2(0, 0), false,
                                      ImGuiWindowFlags_HorizontalScrollbar |
                                      ImGuiWindowFlags_NoScrollWithMouse |
                                      ImGuiWindowFlags_NoScrollbar);
                    ImDrawList *dl = ImGui::GetWindowDrawList();

                    for (u32 i = 0; i < scene->size_of_buffer; i++) {
                        int buf_idx =
                            (i + ps.read_head) % scene->size_of_buffer;
                        int frame_num =
                            scene->display_buffer[visible_idx][buf_idx].frame_number;

                        char label[32];
                        if (input_is_imgs)
                            snprintf(label, sizeof(label), "%d:%s",
                                     frame_num, imgs_names[i].c_str());
                        else
                            snprintf(label, sizeof(label), "%d", frame_num);

                        bool is_selected = (ps.pause_selected == (int)i);

                        if (i > 0) ImGui::SameLine();

                        ImGui::PushID((int)i);
                        ImVec2 pos = ImGui::GetCursorScreenPos();
                        if (ImGui::Selectable("##fbuf", is_selected, 0,
                                              ImVec2(item_w, item_h))) {
                            if (!is_selected) {
                                ps.pause_selected = (int)i;
                            }
                        }

                        // Draw vertical text over the selectable
                        // Color code: green = fully labeled + triangulated,
                        // teal = partially labeled, default = unlabeled
                        const char *text = label;
                        float cx = pos.x + item_w * 0.5f;
                        ImU32 text_col;
                        auto ann_it = annotations.find((u32)frame_num);
                        if (ann_it != annotations.end() &&
                            frame_has_any_keypoints(ann_it->second)) {
                            bool complete = frame_is_complete(ann_it->second);
                            if (complete && skeleton.has_skeleton && scene->num_cams > 1) {
                                for (int k = 0; k < skeleton.num_nodes; ++k)
                                    if (!ann_it->second.kp3d[k].triangulated)
                                        complete = false;
                            }
                            text_col = complete
                                ? IM_COL32(51, 204, 77, 255)   // green
                                : IM_COL32(51, 179, 179, 255); // teal
                        } else {
                            text_col = is_selected
                                ? ImGui::GetColorU32(ImGuiCol_Text)
                                : ImGui::GetColorU32(ImGuiCol_TextDisabled);
                        }
                        // Draw rotated text (90 deg CCW) — read bottom-to-top like a book spine
                        float str_w = ImGui::CalcTextSize(text).x;
                        ImVec2 text_pos(pos.x + (item_w - font_size) * 0.5f,
                                       pos.y + (item_h + str_w) * 0.5f);
                        ImPlot::AddTextVertical(dl, text_pos, text_col, text);
                        ImGui::PopID();
                    }

                    // Mouse wheel → horizontal scroll
                    if (ImGui::IsWindowHovered()) {
                        float wheel = ImGui::GetIO().MouseWheel;
                        if (wheel != 0.0f)
                            ImGui::SetScrollX(ImGui::GetScrollX() - wheel * item_w * 3.0f);
                    }
                    ImGui::EndChild();
                    ImGui::PopStyleVar(2);  // WindowPadding, ItemSpacing
                    ImGui::SetWindowFontScale(1.0f);
                }
            }
            ImGui::End();
        }

        // Render a video frame
        if (ps.video_loaded) {
#ifdef __APPLE__
            // --- Laser detection viz: dispatch once before camera loop ---
            // This must run before per-camera iteration so that hidden cameras
            // (not in a visible ImGui window) still get processed.
            if (calib_state.laser_show_detection && calib_state.laser_ready) {
                auto &lv = calib_state.laser_viz;
                auto &lc = calib_state.laser_config;
                int mac_head_dispatch = ps.play_video ? ps.read_head : select_corr_head;
                int fn0 = scene->display_buffer[0][mac_head_dispatch].frame_number;

                // Collect results from background thread
                if (!lv.computing.load(std::memory_order_acquire) &&
                    !lv.pending.empty()) {
                    if (lv.worker.joinable())
                        lv.worker.join();
                    lv.ready = std::move(lv.pending);
                    lv.pending.clear();
                    // Mark all as needing GPU upload
                    for (auto &cr : lv.ready)
                        cr.uploaded = false;
                }

                // Check if we need new work
                bool params_changed =
                    lc.green_threshold != lv.last_green_th ||
                    lc.green_dominance != lv.last_green_dom ||
                    lc.min_blob_pixels != lv.last_min_blob ||
                    lc.max_blob_pixels != lv.last_max_blob;
                bool frame_changed = lv.ready.empty() ||
                    fn0 != lv.ready[0].frame_num;
                bool need_dispatch = (frame_changed || params_changed) &&
                    !lv.computing.load(std::memory_order_relaxed);

                if (need_dispatch) {
                    if (lv.worker.joinable())
                        lv.worker.join();

                    // Lazy-init Metal context for GPU viz
                    if (!lv.metal_ctx)
                        lv.metal_ctx = laser_metal_create();

                    // Retain CVPixelBuffers for background thread
                    struct CamInput {
                        CVPixelBufferRef pixel_buffer;
                        int width, height, frame_num;
                        bool needs_rgba;  // visible cameras need RGBA for texture upload
                    };
                    auto inputs = std::make_shared<std::vector<CamInput>>(scene->num_cams);
                    for (int ci = 0; ci < scene->num_cams; ci++) {
                        auto &inp = (*inputs)[ci];
                        inp.width = scene->image_width[ci];
                        inp.height = scene->image_height[ci];
                        inp.frame_num = scene->display_buffer[ci][mac_head_dispatch].frame_number;
                        const std::string &cam_name = pm.camera_names[ci];
                        inp.needs_rgba = window_is_visible.count(cam_name) &&
                                         window_is_visible.at(cam_name);
                        CVPixelBufferRef cpb = scene->display_buffer[ci][mac_head_dispatch].pixel_buffer;
                        if (cpb) {
                            CVPixelBufferRetain(cpb);
                            inp.pixel_buffer = cpb;
                        } else {
                            inp.pixel_buffer = nullptr;
                        }
                    }

                    int green_th = lc.green_threshold;
                    int green_dom = lc.green_dominance;
                    int min_blob = lc.min_blob_pixels;
                    int max_blob = lc.max_blob_pixels;
                    int ncams = scene->num_cams;

                    lv.computing.store(true, std::memory_order_release);
                    lv.last_green_th = green_th;
                    lv.last_green_dom = green_dom;
                    lv.last_min_blob = min_blob;
                    lv.last_max_blob = max_blob;

                    auto metal_ctx = lv.metal_ctx;
                    lv.worker = std::thread(
                        [inputs, ncams, green_th, green_dom,
                         min_blob, max_blob, metal_ctx, &lv]() {
                            std::vector<LaserVizState::CamResult> results(ncams);

                            // Phase 1: ALL cameras in parallel via fast detect (for stats)
                            {
                                std::vector<std::thread> threads;
                                for (int ci = 0; ci < ncams; ci++) {
                                    auto &inp = (*inputs)[ci];
                                    if (!inp.pixel_buffer) continue;
                                    threads.emplace_back([&inp, &results, ci,
                                        metal_ctx, green_th, green_dom, min_blob, max_blob]() {
                                        auto &res = results[ci];
                                        res.frame_num = inp.frame_num;
                                        auto spot = laser_metal_detect(
                                            metal_ctx, inp.pixel_buffer,
                                            green_th, green_dom, min_blob, max_blob);
                                        if (spot.found) {
                                            res.num_blobs = 1;
                                        } else if (spot.pixel_count > 0) {
                                            res.num_blobs = -1; // ambiguous
                                        }
                                        res.total_mask_pixels = spot.pixel_count;
                                    });
                                }
                                for (auto &t : threads) t.join();
                            }

                            // Phase 2: visible cameras get RGBA overlay (sequential, shared ctx)
                            for (int ci = 0; ci < ncams; ci++) {
                                auto &inp = (*inputs)[ci];
                                if (!inp.pixel_buffer || !inp.needs_rgba) {
                                    if (inp.pixel_buffer) CVPixelBufferRelease(inp.pixel_buffer);
                                    continue;
                                }
                                auto &res = results[ci];
                                res.rgba.resize(inp.width * inp.height * 4);
                                laser_metal_detect_viz(
                                    metal_ctx, inp.pixel_buffer,
                                    green_th, green_dom, min_blob, max_blob,
                                    res.rgba.data());
                                // Stats already populated by Phase 1 — don't overwrite
                                CVPixelBufferRelease(inp.pixel_buffer);
                            }

                            lv.pending = std::move(results);
                            lv.computing.store(false, std::memory_order_release);
                        });
                }
            }
#endif // __APPLE__

            for (int j = 0; j < scene->num_cams; j++) {
                const std::string &win_name = pm.camera_names[j];

                // Dock cameras into the 2x2 grid from default layout.
                // Dock IDs 0x05..0x08 map to TL, BL, TR, BR quadrants.
                {
                    static const ImGuiID quad_ids[4] = {
                        0x00000005, 0x00000006, 0x00000007, 0x00000008};
                    ImGui::SetNextWindowDockID(quad_ids[j % 4],
                                               ImGuiCond_FirstUseEver);
                }
                ImGui::SetNextWindowSize(ImVec2(500, 400),
                                         ImGuiCond_FirstUseEver);
                bool is_visible = ImGui::Begin(win_name.c_str());
                window_is_visible[win_name] = is_visible;

                if (!window_was_decoding[win_name] && is_visible &&
                    ps.play_video) {
                    // seek if visibility has changed
                    seek_all_cameras(scene, current_frame_num,
                                     dc_context->video_fps, ps, true);
                }

                if (!window_was_decoding[win_name] && is_visible &&
                    !ps.play_video && !ps.pause_seeked) {
                    // seek if visibility has changed
                    seek_all_cameras(scene, current_frame_num,
                                     dc_context->video_fps, ps, true);
                    ps.pause_selected = 0; // stay on the seeked frame
                    for (auto &[key, value] : window_need_decoding) {
                        value.store(true);
                    }
                }

                if (ps.play_video) {
                    window_need_decoding[win_name].store(
                        is_visible || (calib_state.laser_show_detection && calib_state.laser_ready));
                };

                if (is_visible) {
#ifdef __APPLE__
                    // macOS: upload frame to Metal texture for display.
                    // Phase 2/3: if a CVPixelBuffer is available, use GPU
                    //            NV12→RGBA compute (metal_upload_pixelbuf).
                    // Phase 1 fallback: CPU RGBA frame via metal_upload_texture.
                    if (ps.play_video) {
                        current_frame_num = ps.to_display_frame_number;
                    }
                    {
                        int mac_head =
                            ps.play_video ? ps.read_head : select_corr_head;
                        int fn = scene->display_buffer[j][mac_head].frame_number;
                        uint32_t w = scene->image_width[j];
                        uint32_t h = scene->image_height[j];
                        bool did_upload = false;

                        if (calib_state.laser_show_detection && calib_state.laser_ready) {
                            // Upload ready result for this camera if available
                            auto &lv = calib_state.laser_viz;
                            if (j < (int)lv.ready.size() &&
                                !lv.ready[j].rgba.empty() &&
                                !lv.ready[j].uploaded) {
                                metal_upload_texture(j,
                                    lv.ready[j].rgba.data(), w, h);
                                lv.ready[j].uploaded = true;
                                mac_last_uploaded_frame[j] = -1; // force re-upload when viz off
                                did_upload = true;
                            } else if (fn != mac_last_uploaded_frame[j] && lv.ready.empty()) {
                                // No results yet — show normal frame
                                CVPixelBufferRef pb =
                                    scene->display_buffer[j][mac_head].pixel_buffer;
                                if (pb)
                                    metal_upload_pixelbuf(j, pb, w, h);
                                else
                                    metal_upload_texture(j,
                                        scene->display_buffer[j][mac_head].frame, w, h);
                                mac_last_uploaded_frame[j] = fn;
                                did_upload = true;
                            }
                        } else if (fn != mac_last_uploaded_frame[j]) {
                            CVPixelBufferRef pb =
                                scene->display_buffer[j][mac_head].pixel_buffer;
                            if (pb) {
                                metal_upload_pixelbuf(j, pb, w, h);
                            } else {
                                metal_upload_texture(j,
                                    scene->display_buffer[j][mac_head].frame,
                                    w, h);
                            }
                            mac_last_uploaded_frame[j] = fn;
                            did_upload = true;
                        }
                        if (did_upload)
                            metal_apply_contrast_brightness(j, display.contrast,
                                (float)display.brightness, display.pivot_midgray);
                    }
#else
                    if (ps.play_video) {
                        // if the current frame is ready, upload for
                        // display, otherwise wait for the frame to get
                        // ready while
                        // (scene->display_buffer[j][ps.read_head]
                        //            .frame_number !=
                        //        ps.to_display_frame_number) {
                        //     std::cout
                        //         << win_name << " , read head: " <<
                        //         ps.read_head
                        //         << ", frame_number: "
                        //         << scene->display_buffer[j][ps.read_head]
                        //                .frame_number
                        //         << ", to_display_frame_number: "
                        //         << ps.to_display_frame_number <<
                        //         std::endl;
                        //     std::this_thread::sleep_for(
                        //         std::chrono::milliseconds(1));
                        // }

                        current_frame_num = ps.to_display_frame_number;
                        if (scene->use_cpu_buffer) {
                            // upload_texture(&scene->image_texture[j],
                            // scene->display_buffer[j][read_head].frame,
                            // scene->image_width[j],
                            // scene->image_height[j]);
                            // // 2x slower than pbo copy frame to cuda
                            // buffer
                            ck(cudaMemcpy(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->display_buffer[j][ps.read_head].frame,
                                scene->image_width[j] * scene->image_height[j] *
                                    4,
                                cudaMemcpyHostToDevice));
                        } else {
                            ck(cudaMemcpy(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->display_buffer[j][ps.read_head].frame,
                                scene->image_width[j] * scene->image_height[j] *
                                    4,
                                cudaMemcpyDeviceToDevice));
                        }
                    } else {
                        if (scene->use_cpu_buffer) {
                            // upload_texture(&scene->image_texture[j],
                            // scene->display_buffer[j][select_corr_head].frame,
                            // scene->image_width[j],
                            // scene->image_height[j]);
                            ck(cudaMemcpy(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->display_buffer[j][select_corr_head]
                                    .frame,
                                scene->image_width[j] * scene->image_height[j] *
                                    4,
                                cudaMemcpyHostToDevice));

                            // // brighten in-place on GPU using npp
                            // NppiSize roi = {
                            //     static_cast<int>(scene->image_width[j]),
                            //     static_cast<int>(scene->image_height[j])};
                            // Npp8u *d_img =
                            //     (Npp8u *)scene->pbo_cuda[j].cuda_buffer;
                            // int step = scene->image_width[j] *
                            //            4; // RGBA = 4 bytes per pixel

                            // Npp8u addC[3] = {(Npp8u)brightness,
                            //                  (Npp8u)brightness,
                            //                  (Npp8u)brightness};

                            // nppiAddC_8u_AC4IRSfs(addC, d_img, step, roi,
                            //                      0); // in-place RGBA

                            apply_contrast_brightness_rgba(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->image_width[j], scene->image_height[j],
                                display.contrast,
                                (float)display.brightness,
                                display.pivot_midgray,
                                0);

                        } else {
                            ck(cudaMemcpy(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->display_buffer[j][select_corr_head]
                                    .frame,
                                scene->image_width[j] * scene->image_height[j] *
                                    4,
                                cudaMemcpyDeviceToDevice));
                        }
                    }
                    bind_pbo(&scene->pbo_cuda[j].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(scene->image_width[j],
                                                scene->image_height[j]);
                    unbind_pbo();
                    unbind_texture();
#endif // __APPLE__

                    std::string scene_name = "scene view" + std::to_string(j);
                    ImGui::BeginChild(
                        scene_name.c_str(),
                        ImVec2(0, 0));
                    ImVec2 avail_size = ImGui::GetContentRegionAvail();

                    // ImGui::Image((void*)(intptr_t)image_texture[j],
                    // avail_size);
                    //
                    if (pm.plot_keypoints_flag) {
                        keypoints_find = (annotations.find(current_frame_num) !=
                                          annotations.end());
                    }

                    // When SAM mask cycling is active THIS FRAME
                    // (Shift held + pending mask), temporarily set
                    // ImPlot's ZoomMod to require Ctrl so that
                    // Shift+scroll does NOT zoom.  Default ZoomMod =
                    // ImGuiMod_None means scroll always zooms, stealing
                    // the event from SAM.  We only override when Shift
                    // is actually held to preserve normal scroll-to-zoom.
                    bool sam_override_zoom = false;
                    int saved_zoom_mod = 0;
                    if (sam_tool_state.enabled &&
                        sam_tool_state.has_pending_mask &&
                        !sam_tool_state.multi_mask.masks.empty() &&
                        ImGui::GetIO().KeyShift) {
                        auto &imap = ImPlot::GetInputMap();
                        saved_zoom_mod = imap.ZoomMod;
                        imap.ZoomMod = ImGuiMod_Ctrl;
                        sam_override_zoom = true;
                    }

                    if (ImPlot::BeginPlot("##no_plot_name", avail_size,
                                          ImPlotFlags_Equal |
                                              ImPlotFlags_Crosshairs |
                                              ImPlotFlags_NoMenus)) {
                        ImPlot::SetupAxisLimits(
                            ImAxis_X1, 0, scene->image_width[j],
                            ImPlotCond_Once);
                        ImPlot::SetupAxisLimits(
                            ImAxis_Y1, 0, scene->image_height[j],
                            ImPlotCond_Once);
                        ImPlot::PlotImage(
                            "##no_image_name",
#ifdef __APPLE__
                            (ImTextureID)scene->image_descriptor[j],
#else
                            (ImTextureID)(intptr_t)scene->image_texture[j],
#endif
                            ImVec2(0, 0),
                            ImVec2(scene->image_width[j],
                                   scene->image_height[j]));

                        if (pm.plot_keypoints_flag) {
                            // labeling (keypoints)
                            // OBB tool uses G key (not W), so no keypoint conflict
                            if (ImPlot::IsPlotHovered()) {
                                is_view_focused[j] = true;
                                if (ImGui::IsKeyPressed(ImGuiKey_B, false) &&
                                    !io.WantTextInput) {
                                    // create frame annotation
                                    if (!keypoints_find) {
                                        get_or_create_frame(annotations,
                                            current_frame_num,
                                            skeleton.num_nodes,
                                            scene->num_cams);
                                    }
                                }

                                if (keypoints_find && skeleton.has_skeleton) {
                                    u32 *kp = &annotations.at(current_frame_num)
                                                   .cameras[j].active_id;
                                    if (ImGui::IsKeyPressed(ImGuiKey_W,
                                                            false) &&
                                        !io.WantTextInput) {
                                        // labeling sequentially each view
                                        ImPlotPoint mouse =
                                            ImPlot::GetPlotMousePos();
                                        auto &fa = annotations.at(current_frame_num);
                                        auto &kp2d = fa.cameras[j].keypoints[*kp];
                                        kp2d.x = mouse.x;
                                        kp2d.y = mouse.y;
                                        kp2d.labeled = true;
                                        if (*kp < (skeleton.num_nodes - 1)) {
                                            (*kp)++;
                                        }
                                    }

                                    if (ImGui::IsKeyPressed(ImGuiKey_A, true) &&
                                        !io.WantTextInput) {
                                        if (*kp <= 0) {
                                            *kp = 0;
                                        } else
                                            (*kp)--;
                                    }

                                    if (ImGui::IsKeyPressed(ImGuiKey_D, true) &&
                                        !io.WantTextInput) {
                                        if (*kp >= skeleton.num_nodes - 1) {
                                            *kp = skeleton.num_nodes - 1;
                                        } else
                                            (*kp)++;
                                    }

                                    if (ImGui::IsKeyPressed(ImGuiKey_E,
                                                            false) &&
                                        !io.WantTextInput) {
                                        *kp = skeleton.num_nodes - 1;
                                    }

                                    if (ImGui::IsKeyPressed(ImGuiKey_Q,
                                                            false) &&
                                        !io.WantTextInput) {
                                        *kp = 0;
                                    }

                                    // delete all keypoints on a frame
                                    // (skip if SAM has active prompts — Backspace is SAM undo)
                                    if (ImGui::IsKeyPressed(ImGuiKey_Backspace,
                                                            false) &&
                                        !io.WantTextInput &&
                                        !(sam_tool_state.enabled &&
                                          (!sam_tool_state.fg_points.empty() ||
                                           !sam_tool_state.bg_points.empty()))) {
                                        annotations.erase(current_frame_num);
                                        keypoints_find = false;
                                    }
                                }
                            } else {
                                is_view_focused[j] = false;
                            }

                            if (keypoints_find && skeleton.has_skeleton &&
                                display.show_keypoints) {
                                gui_plot_keypoints(
                                    annotations.at(current_frame_num),
                                    &skeleton, j, scene->num_cams);
                            }
                        }

                        // --- Annotation tool overlays + input (bbox, OBB, SAM) ---
                        {
                            int iw = (int)scene->image_width[j];
                            int ih = (int)scene->image_height[j];
                            u32 frame = (u32)current_frame_num;
                            int nn = skeleton.num_nodes;
                            int nc = (int)scene->num_cams;

                            // Bbox tool
                            if (bbox_state.enabled) {
                                bbox_handle_input(bbox_state, annotations,
                                                  frame, j, nn, nc, iw, ih);
                            }
                            if (display.show_bboxes) {
                                bbox_draw_overlays(bbox_state, annotations,
                                                   frame, j, iw, ih);
                            }

                            // OBB tool
                            if (obb_state.enabled) {
                                obb_handle_input(obb_state, bbox_state,
                                                 annotations, frame, j,
                                                 nn, nc, iw, ih);
                            }
                            if (display.show_bboxes) {
                                obb_draw_overlays(obb_state, bbox_state,
                                                  annotations, frame, j, iw, ih);
                            }

                            // Accepted mask overlays (stored in AnnotationMap)
                            if (display.show_masks) {
                                draw_accepted_masks(annotations, frame, j, iw, ih);
                            }

                            // SAM assist
                            if (sam_tool_state.enabled) {
                                const uint8_t *sam_rgb = nullptr;
#ifdef __APPLE__
                                // Extract RGB from CVPixelBuffer on click
                                // (lazy — only when SAM needs to run)
                                static std::vector<uint8_t> sam_rgb_buf;
                                static int sam_rgb_frame = -1;
                                static int sam_rgb_cam = -1;
                                bool need_rgb = ImPlot::IsPlotHovered() &&
                                    (ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                                     ImGui::IsMouseClicked(ImGuiMouseButton_Right));
                                if (need_rgb || (sam_rgb_frame == (int)frame && sam_rgb_cam == j)) {
                                    if (sam_rgb_frame != (int)frame || sam_rgb_cam != j) {
                                        int mh = ps.play_video ? ps.read_head : select_corr_head;
                                        CVPixelBufferRef pb = scene->display_buffer[j][mh].pixel_buffer;
                                        if (pb) {
                                            sam_rgb_buf.resize(iw * ih * 3);
                                            CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                                            const uint8_t *bgra = (const uint8_t *)CVPixelBufferGetBaseAddress(pb);
                                            int stride = (int)CVPixelBufferGetBytesPerRow(pb);
                                            for (int y = 0; y < ih; y++) {
                                                const uint8_t *src = bgra + y * stride;
                                                uint8_t *dst = sam_rgb_buf.data() + y * iw * 3;
                                                for (int x = 0; x < iw; x++) {
                                                    dst[x*3+0] = src[x*4+2]; // R from BGRA
                                                    dst[x*3+1] = src[x*4+1]; // G
                                                    dst[x*3+2] = src[x*4+0]; // B
                                                }
                                            }
                                            CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                                            sam_rgb_frame = (int)frame;
                                            sam_rgb_cam = j;
                                        }
                                    }
                                    if (sam_rgb_frame == (int)frame && sam_rgb_cam == j)
                                        sam_rgb = sam_rgb_buf.data();
                                }
#endif
                                sam_handle_input(sam_tool_state, sam_state,
                                                 annotations, frame, j,
                                                 nn, nc, iw, ih, sam_rgb);
                            }
                            if (display.show_masks)
                                sam_draw_overlay(sam_tool_state, j, iw, ih);
                        }

                        // Plot context menu: press 1 key while hovering
                        // (right-click reserved for SAM background points)
                        if (ImPlot::IsPlotHovered() &&
                            ImGui::IsKeyPressed(ImGuiKey_2, false) &&
                            !io.WantTextInput) {
                            ImGui::OpenPopup("##plot_settings");
                        }
                        if (ImGui::BeginPopup("##plot_settings")) {
                            ImGui::SeparatorText("Plot Settings");
                            if (ImGui::MenuItem("Fit X Axis"))
                                ImPlot::SetupAxisLimits(ImAxis_X1, 0, scene->image_width[j]);
                            if (ImGui::MenuItem("Fit Y Axis"))
                                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, scene->image_height[j]);
                            if (ImGui::MenuItem("Fit Both")) {
                                ImPlot::SetupAxisLimits(ImAxis_X1, 0, scene->image_width[j]);
                                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, scene->image_height[j]);
                            }
                            ImGui::SeparatorText("Visibility");
                            ImGui::Checkbox("Keypoints", &display.show_keypoints);
                            ImGui::Checkbox("Masks / Contours", &display.show_masks);
                            ImGui::Checkbox("Bounding Boxes", &display.show_bboxes);
                            ImGui::EndPopup();
                        }

                        ImPlot::EndPlot();
                    }

                    // Restore ImPlot zoom modifier if we overrode it
                    if (sam_override_zoom) {
                        ImPlot::GetInputMap().ZoomMod = saved_zoom_mod;
                    }

                    ImGui::EndChild();
                }
                ImGui::End();
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Space, false) &&
                !io.WantTextInput) {
                ps.play_video = !ps.play_video;
                if (ps.play_video) {
                    ps.pause_seeked = false;
                    ps.last_play_time_start = std::chrono::steady_clock::now();
                } else {
                    ps.pause_selected = 0;
                }
            }


            // Hotkey 6: Run JARVIS prediction on current frame (all cameras)
            bool jarvis_predict_trigger =
                (ImGui::IsKeyPressed(ImGuiKey_6, false) && !io.WantTextInput) ||
                jarvis_predict_state.predict_requested;
            jarvis_predict_state.predict_requested = false;

            bool jarvis_any_loaded = jarvis_state.loaded;
#ifdef __APPLE__
            jarvis_any_loaded = jarvis_any_loaded || jarvis_coreml_state.loaded;
#endif
            if (jarvis_predict_trigger && !ps.play_video &&
                jarvis_any_loaded && scene->num_cams > 0) {
#ifdef __APPLE__
                int mh = ps.play_video ? ps.read_head : select_corr_head;
                std::vector<int> widths(scene->num_cams), heights(scene->num_cams);
                for (int c = 0; c < (int)scene->num_cams; ++c) {
                    widths[c] = (int)scene->image_width[c];
                    heights[c] = (int)scene->image_height[c];
                }

                // Prefer CoreML (GPU/ANE) over ONNX Runtime (CPU)
                if (jarvis_coreml_state.loaded) {
                    std::vector<CVPixelBufferRef> pbs(scene->num_cams, nullptr);
                    for (int c = 0; c < (int)scene->num_cams; ++c)
                        pbs[c] = scene->display_buffer[c][mh].pixel_buffer;

                    jarvis_coreml_predict_frame(jarvis_coreml_state, annotations,
                        (u32)current_frame_num, pbs, widths, heights,
                        skeleton, (int)scene->num_cams,
                        jarvis_predict_state.confidence_threshold);
                    // Triangulate (reprojection is in gui_keypoints.h, only in this TU)
                    reprojection(annotations.at(current_frame_num),
                                 &skeleton, pm.camera_params, scene);
                    printf("[JARVIS CoreML] %s\n", jarvis_coreml_state.status.c_str());
                } else if (jarvis_state.loaded) {
                    // Fallback: ONNX Runtime (CPU) with BGRA→RGB conversion
                    std::vector<const uint8_t *> rgb_bufs(scene->num_cams, nullptr);
                    std::vector<std::vector<uint8_t>> rgb_storage(scene->num_cams);
                    for (int c = 0; c < (int)scene->num_cams; ++c) {
                        int w = widths[c], h = heights[c];
                        CVPixelBufferRef pb = scene->display_buffer[c][mh].pixel_buffer;
                        if (!pb) continue;
                        rgb_storage[c].resize(w * h * 3);
                        CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                        const uint8_t *bgra = (const uint8_t *)CVPixelBufferGetBaseAddress(pb);
                        int stride = (int)CVPixelBufferGetBytesPerRow(pb);
                        for (int y = 0; y < h; y++) {
                            const uint8_t *src = bgra + y * stride;
                            uint8_t *dst = rgb_storage[c].data() + y * w * 3;
                            for (int x = 0; x < w; x++) {
                                dst[x*3+0] = src[x*4+2];
                                dst[x*3+1] = src[x*4+1];
                                dst[x*3+2] = src[x*4+0];
                            }
                        }
                        CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                        rgb_bufs[c] = rgb_storage[c].data();
                    }
                    jarvis_predict_frame(jarvis_state, annotations,
                        (u32)current_frame_num, rgb_bufs, widths, heights,
                        skeleton, pm.camera_params, scene,
                        jarvis_predict_state.confidence_threshold);
                    printf("[JARVIS ONNX] %s\n", jarvis_state.status.c_str());
                }
#endif
            }

            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false) &&
                !io.WantTextInput) {
                if (ImGui::GetIO().KeyShift) {
                    int clamped_frame = std::max(
                        0, current_frame_num - 10 * dc_context->seek_interval);
                    seek_all_cameras(scene, clamped_frame,
                                     dc_context->video_fps, ps, false);
                } else {
                    int clamped_frame = std::max(
                        0, current_frame_num - dc_context->seek_interval);
                    seek_all_cameras(scene, clamped_frame,
                                     dc_context->video_fps, ps, false);
                }
            }

            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false) &&
                !io.WantTextInput) {
                if (ImGui::GetIO().KeyShift) {
                    int clamped_frame = std::min(
                        dc_context->total_num_frame,
                        current_frame_num + 10 * dc_context->seek_interval);
                    seek_all_cameras(scene, clamped_frame,
                                     dc_context->video_fps, ps, false);
                } else {
                    int clamped_frame =
                        std::min(dc_context->total_num_frame,
                                 current_frame_num + dc_context->seek_interval);
                    seek_all_cameras(scene, clamped_frame,
                                     dc_context->video_fps, ps, false);
                }
            }

            for (const auto &[name, flag] : window_need_decoding) {
                window_was_decoding[name] = flag.load();
            }
        }

        // H-key help toggle
        if (ImGui::IsKeyPressed(ImGuiKey_H, false) && !io.WantTextInput) {
            show_help_window = !show_help_window;
        }

        drawToasts(toasts);
        drawPopups(popups);

        // Rendering
        ImGui::Render();
#ifdef __APPLE__
        metal_end_frame();  // creates render encoder, renders ImGui, presents
#else
        int display_w, display_h;
        glfwGetFramebufferSize(window->render_target, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w,
                     clear_color.y * clear_color.w,
                     clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow *backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window->render_target);
#endif

        // Window title
        glfwSetWindowTitle(window->render_target, "Red");

        if (ps.just_seeked) {
            ps.just_seeked = false;
        } else {
            if (dc_context->decoding_flag && ps.play_video &&
                scene->num_cams > 0 && scene->display_buffer) {
                int frame_to_show = ps.to_display_frame_number;
                // Cap to slowest decoded camera (applied in both modes)
                int min_decoded_frame = INT_MAX;
                for (const auto &[cam_name, visible] : window_need_decoding) {
                    if (visible.load()) {
                        int decoded = latest_decoded_frame[cam_name].load();
                        min_decoded_frame =
                            std::min(min_decoded_frame, decoded);
                    }
                }

                // CHOOSE MODE
                if (ps.realtime_playback) {
                    // --- Real-time frame selection: advance by wall clock ---
                    frame_to_show = static_cast<int>(
                        std::ceil(playback_time_now * dc_context->video_fps));
                } else {
                    // --- Tick-based mode: advance one frame per render tick,
                    //     but never past what the decoder has filled ---
                    frame_to_show = ps.to_display_frame_number + 1;
                }
                frame_to_show = std::min(frame_to_show, min_decoded_frame);
                frame_to_show =
                    std::min(frame_to_show, dc_context->total_num_frame - 1);
                int frame_delta = frame_to_show - ps.to_display_frame_number;
                if (frame_delta > 0) {
                    ps.to_display_frame_number = frame_to_show;
                    for (int offset = 0; offset < frame_delta; ++offset) {
                        int index =
                            (ps.read_head + offset) % scene->size_of_buffer;
                        for (int j = 0; j < scene->num_cams; j++) {
#ifdef __APPLE__
                            // Release CVPixelBuffer before relinquishing slot
                            if (scene->display_buffer[j][index].pixel_buffer) {
                                CFRelease(scene->display_buffer[j][index].pixel_buffer);
                                scene->display_buffer[j][index].pixel_buffer = nullptr;
                            }
#endif
                            scene->display_buffer[j][index].available_to_write =
                                true;
                        }
                    }
                    ps.read_head =
                        (ps.read_head + frame_delta) % scene->size_of_buffer;
                    if (!ps.slider_text_editing)
                        ps.slider_frame_number = ps.to_display_frame_number;
                }
            }
        }
    }
    // Cleanup
#ifdef __APPLE__
    metal_cleanup();  // waits for GPU, shuts down ImGui Metal backend
#else
    ImGui_ImplOpenGL3_Shutdown();
#endif
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window->render_target);
    glfwTerminate();

    dc_context->stop_flag = true;
    // wait for threads to join
    for (auto &t : decoder_threads)
        t.join();

    return 0;
}
