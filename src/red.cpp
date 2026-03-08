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
#include "gui/annotation_dialog.h"
#include "gui/calibration_tool_window.h"
#include "gui/labeling_tool_window.h"
#include "gui/project_window.h"
#include "gui/settings_window.h"
#include "gui/navigator_dialogs.h"
#include "gui/panel_registry.h"
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
    bool cpu_buffer_toggle = true;
    int current_frame_num = 0;
    int previous_frame_num = -1;
    std::vector<std::string> imgs_names;

    // for labeling
    SkeletonContext skeleton;
    std::map<u32, KeyPoints *> keypoints_map;
    bool keypoints_find = false;
    std::map<std::string, SkeletonPrimitive> skeleton_map = skeleton_get_all();

    // others
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
    ImGuiIO &io = ImGui::GetIO();
    static std::string project_ini_path;

    ImPlotStyle &style = ImPlot::GetStyle();
    ImVec4 *colors = style.Colors;


    // JARVIS Export Tool state
    JarvisExportState jarvis_export_state;
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
        skeleton, skeleton_map, keypoints_map,
        popups, toasts, deferred,
        user_settings, red_data_dir, skeleton_dir,
        imgs_names, demuxers, decoder_threads,
        is_view_focused, window_was_decoding,
        input_is_imgs, label_buffer_size,
        display, window, project_ini_path, main_loop_running
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
                if (setup_project(pm, skeleton, skeleton_map, &err))
                    on_project_loaded(ctx, print_metadata, print_summary);
                else
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
                        reprojection(keypoints_map.at(current_frame_num),
                                     &skeleton, pm.camera_params, scene);
                    }
                },
                [&]() { return pm.plot_keypoints_flag; }});
    panels.add({"Help", [&]() { DrawHelpWindow(show_help_window); }, nullptr});
    panels.add({"JARVIS Export",
                [&]() { DrawJarvisExportWindow(jarvis_export_state, ctx); },
                nullptr});
    panels.add({"Calibration Tool",
                [&]() { DrawCalibrationToolWindow(calib_state, ctx, calib_cb); },
                nullptr});
    panels.add({"Settings",
                [&]() { DrawSettingsWindow(settings_state, ctx); },
                nullptr});

    main_loop_running = true;
    while (!glfwWindowShouldClose(window->render_target)) {
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

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

        if (ImGui::Begin("Navigator", NULL, ImGuiWindowFlags_MenuBar)) {
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    ImGui::BeginDisabled(ps.video_loaded);
                    if (ImGui::MenuItem("Open Video(s)")) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 0;
                        config.path = pm.media_folder;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseMedia", "Choose Media", ".mp4", config);
                    }
                    if (ImGui::MenuItem("Open Images")) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 0;
                        config.path = pm.media_folder;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseImages", "Choose Images",
                            ".jpg,.tiff,.jpeg,.png", config);
                    }
                    ImGui::EndDisabled();
                    ImGui::BeginDisabled(!ps.video_loaded);
                    if (ImGui::MenuItem("Create Project")) {
                        pm.show_project_window = true;
                    }
                    ImGui::EndDisabled();
                    ImGui::BeginDisabled(ps.video_loaded);
                    if (ImGui::MenuItem("Load Project")) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = pm.project_root_path;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseProject", "Choose Project File", ".redproj",
                            config);
                    }
                    ImGui::EndDisabled();
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Annotate")) {
                    ImGui::BeginDisabled(ps.video_loaded);
                    if (ImGui::MenuItem("Create Annotation Project")) {
                        annot_state.show = true;
                        annot_state.discovered_cameras.clear();
                        annot_state.camera_selected.clear();
                        annot_state.status.clear();
                    }
                    if (ImGui::MenuItem("Load Annotation Project")) {
                        IGFD::FileDialogConfig cfg;
                        cfg.countSelectionMax = 1;
                        cfg.path = pm.project_root_path;
                        cfg.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "LoadAnnotProject", "Load Annotation Project",
                            "Red Project{.redproj}", cfg);
                    }
                    ImGui::EndDisabled();
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Calibrate")) {
                    if (ImGui::MenuItem("Create Calibration Project")) {
                        calib_state.show = true;
                        calib_state.show_create_dialog = true;
                    }
                    if (ImGui::MenuItem("Load Calibration Project")) {
                        IGFD::FileDialogConfig cfg;
                        cfg.countSelectionMax = 1;
                        cfg.path =
                            user_settings.default_project_root_path.empty()
                                ? red_data_dir
                                : user_settings.default_project_root_path;
                        cfg.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "LoadCalibProject", "Load Calibration Project",
                            "Red Project{.redproj}", cfg);
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Tools")) {
                    if (ImGui::MenuItem("JARVIS Export Tool")) {
                        jarvis_export_state.show = true;
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::MenuItem("Settings...")) {
                    settings_state.show = true;
                }

                ImGui::EndMenuBar();
            }
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                        1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);

            if (!ps.video_loaded) {
#ifndef __APPLE__
                {
                    const char *items[] = {"CPU Buffer", "GPU Buffer"};
                    static int item_current = 0;
                    ImGui::Combo("Buffer Type", &item_current, items,
                                 IM_ARRAYSIZE(items));
                    scene->use_cpu_buffer = (item_current == 0);
                }
#endif
            }

            if (ps.video_loaded) {
                ImGui::InputInt("Seek Step", &dc_context->seek_interval, 10,
                                100);
                static int seek_accurate_frame_num = 0;
                ImGui::InputInt("Seek Accurate", &seek_accurate_frame_num, 1,
                                100);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    seek_all_cameras(scene, seek_accurate_frame_num,
                                     dc_context->video_fps, ps, true);
                }

                auto now_wall = std::chrono::steady_clock::now();
                double wall_seconds =
                    std::chrono::duration<double>(now_wall -
                                                  ps.last_wall_time_playspeed)
                        .count();
                int frame_delta =
                    current_frame_num - ps.last_frame_num_playspeed;
                if (wall_seconds > 0.5 && ps.play_video) {
                    ps.inst_speed =
                        frame_delta / (dc_context->video_fps *
                                       wall_seconds); // Real-time normalized
                    ps.last_frame_num_playspeed = current_frame_num;
                    ps.last_wall_time_playspeed = now_wall;
                }

                // === Playback section ===
                ImGui::SeparatorText("Playback");
                ImGui::Checkbox("Realtime", &ps.realtime_playback);
                // Two-column table: label | control
                ImGui::BeginDisabled(!ps.realtime_playback);
                if (ImGui::BeginTable("##playback_tbl", 2,
                                      ImGuiTableFlags_SizingStretchProp |
                                          ImGuiTableFlags_BordersInnerV)) {
                    ImGui::TableSetupColumn(
                        "Label", ImGuiTableColumnFlags_WidthFixed, 170.0f);
                    ImGui::TableSetupColumn("Control",
                                            ImGuiTableColumnFlags_WidthStretch);

                    // Row: FPS (read-only)
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Video FPS");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.1f", dc_context->video_fps);

                    // Row: Playback speed slider
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Set Playback Speed");
                    ImGui::SameLine();
                    HelpMarker("Log2 scale: 1/16x to 1x.");

                    ImGui::TableSetColumnIndex(1);
                    // Format label: show as fraction, e.g. "1/8x"
                    char speed_label[16];
                    int denom = (int)roundf(1.0f / ps.set_playback_speed);
                    if (denom <= 1)
                        snprintf(speed_label, sizeof(speed_label), "1x");
                    else
                        snprintf(speed_label, sizeof(speed_label), "1/%dx", denom);
                    ImGui::SliderFloat("##set_playback_speed", &ps.set_playback_speed,
                                       1.0f / 16.0f, 1.0f, speed_label,
                                       ImGuiSliderFlags_Logarithmic);

                    // Row: Current speed readout
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Current Speed");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.2fx", ps.inst_speed);

                    ImGui::EndTable();
                }
                ImGui::EndDisabled();

                // Tip (wrapped, subtle)
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
                ImGui::Spacing();
                ImGui::TextDisabled(
                    "Tip: If playback is slower than real-time (< 1.0x), "
                    "collapse camera views to improve speed.");
                ImGui::PopTextWrapPos();

                // === Display section ===
                ImGui::SeparatorText("Display Controls");
#ifdef __APPLE__
                // Metal compute shader applies contrast/brightness live during playback
                (void)0;
#else
                ImGui::BeginDisabled(ps.play_video); // Disable if playing video
#endif

                if (ImGui::BeginTable("##display_tbl", 2,
                                      ImGuiTableFlags_SizingStretchProp |
                                          ImGuiTableFlags_BordersInnerV)) {
                    ImGui::TableSetupColumn(
                        "Label", ImGuiTableColumnFlags_WidthFixed, 170.0f);
                    ImGui::TableSetupColumn("Control",
                                            ImGuiTableColumnFlags_WidthStretch);

                    // Contrast
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Contrast (alpha)");
                    ImGui::SameLine();
                    HelpMarker("1.00 = neutral. Increase to boost separation.");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SliderFloat("##contrast", &display.contrast, 0.0f, 3.0f,
                                       "%.2f");

                    // Brightness
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Brightness (beta)");
                    ImGui::SameLine();
                    HelpMarker("Shift pixel values. 0 = neutral.");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SliderInt("##brightness", &display.brightness, -150, 150);

                    // Reset row
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Display Preset");
                    ImGui::TableSetColumnIndex(1);
                    if (ImGui::Button("Reset##display")) {
                        display.contrast = 1.0f;
                        display.brightness = 0;
                        display.pivot_midgray = true;
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(restores neutral)");

                    ImGui::EndTable();
                }
#ifndef __APPLE__
                ImGui::EndDisabled();
#endif
            }
        }
        ImGui::End();

        // Draw all registered panels
        panels.drawAll();

        // Handle Navigator file dialogs
        HandleNavigatorDialogs(ctx, calib_state, annot_state,
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

            ImGui::SetNextWindowSize(ImVec2(500, 440), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Frames in the buffer")) {

                for (u32 i = 0; i < scene->size_of_buffer; i++) {
                    int seletable_frame_id =
                        (i + ps.read_head) % scene->size_of_buffer;

                    char label[64];
                    if (input_is_imgs) {
                        snprintf(label, sizeof(label), "%d: %s",
                                 scene
                                     ->display_buffer[visible_idx]
                                                     [seletable_frame_id]
                                     .frame_number,
                                 imgs_names[i].c_str());
                    } else {
                        snprintf(label, sizeof(label), "Frame %d",
                                 scene
                                     ->display_buffer[visible_idx]
                                                     [seletable_frame_id]
                                     .frame_number);
                    }

                    bool is_selected = (ps.pause_selected == (int)i);

                    if (ImGui::Selectable(label, is_selected)) {
                        if (!is_selected) { // clicked a new one
                            ps.pause_selected = (int)i;
                            selection_changed = true;
                        }
                    }

                    // Only recenter when selection *changed* this frame
                    if (selection_changed && ps.pause_selected == (int)i) {
                        // 0.5f = roughly center; tweak if you like
                        ImGui::SetScrollHereY(0.5f);
                    }
                }
            }
            ImGui::End();

            select_corr_head =
                (ps.pause_selected + ps.read_head) % scene->size_of_buffer;
            current_frame_num =
                scene->display_buffer[visible_idx][select_corr_head]
                    .frame_number;

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

                    ImGui::BeginGroup();
                    std::string scene_name = "scene view" + std::to_string(j);
                    ImGui::BeginChild(
                        scene_name.c_str(),
                        ImVec2(0, -ImGui::GetFrameHeightWithSpacing()));
                    ImVec2 avail_size = ImGui::GetContentRegionAvail();

                    // ImGui::Image((void*)(intptr_t)image_texture[j],
                    // avail_size);
                    //
                    if (pm.plot_keypoints_flag) {
                        if (keypoints_map.find(current_frame_num) ==
                            keypoints_map.end()) {
                            keypoints_find = false;
                        } else {
                            keypoints_find = true;
                        }
                    }

                    if (ImPlot::BeginPlot("##no_plot_name", avail_size,
                                          ImPlotFlags_Equal |
                                              ImPlotFlags_Crosshairs)) {
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
                            // plot arena for testing camera parameters
                            // gui_plot_perimeter(&camera_params[j],
                            // scene->image_height[j]); if (scene->num_cams
                            // > 1)
                            // {
                            //     gui_plot_world_coordinates(&camera_params[j],
                            //     j, scene->image_height[j]);
                            // }

                            // labeling
                            if (ImPlot::IsPlotHovered()) {
                                is_view_focused[j] = true;
                                if (ImGui::IsKeyPressed(ImGuiKey_B, false) &&
                                    !io.WantTextInput) {
                                    // create keypoints
                                    if (!keypoints_find) {
                                        // not found
                                        KeyPoints *keypoints =
                                            (KeyPoints *)malloc(
                                                sizeof(KeyPoints));
                                        allocate_keypoints(keypoints, scene,
                                                           &skeleton);
                                        keypoints_map[current_frame_num] =
                                            keypoints;
                                    }
                                }

                                if (keypoints_find && skeleton.has_skeleton) {
                                    u32 *kp = &(keypoints_map[current_frame_num]
                                                    ->active_id[j]);
                                    if (ImGui::IsKeyPressed(ImGuiKey_W,
                                                            false) &&
                                        !io.WantTextInput) {
                                        // labeling sequentially each view
                                        ImPlotPoint mouse =
                                            ImPlot::GetPlotMousePos();
                                        keypoints_map[current_frame_num]
                                            ->kp2d[j][*kp]
                                            .position = {mouse.x, mouse.y};
                                        keypoints_map[current_frame_num]
                                            ->kp2d[j][*kp]
                                            .is_labeled = true;
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

                                    // delete all keypoint on a frame
                                    if (ImGui::IsKeyPressed(ImGuiKey_Backspace,
                                                            false) &&
                                        !io.WantTextInput) {
                                        free_keypoints(
                                            keypoints_map[current_frame_num],
                                            scene);
                                        keypoints_map.erase(current_frame_num);
                                        keypoints_find = false;
                                    }
                                }
                            } else {
                                is_view_focused[j] = false;
                            }

                            if (keypoints_find && skeleton.has_skeleton) {
                                gui_plot_keypoints(
                                    keypoints_map.at(current_frame_num),
                                    &skeleton, j, scene->num_cams);
                            }
                        }
                        ImPlot::EndPlot();
                    }

                    ImGui::EndChild();

                    float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
                    if (ImGui::Button(ICON_FK_FAST_BACKWARD)) {
                        int clamped_frame =
                            std::max(0, current_frame_num -
                                            10 * dc_context->seek_interval);
                        seek_all_cameras(scene, clamped_frame,
                                         dc_context->video_fps, ps, false);
                    }
                    ImGui::SameLine(0.0f, spacing);
                    if (ImGui::Button(ICON_FK_STEP_BACKWARD)) {
                        int clamped_frame = std::max(
                            0, current_frame_num - dc_context->seek_interval);
                        seek_all_cameras(scene, clamped_frame,
                                         dc_context->video_fps, ps, false);
                    }
                    ImGui::SameLine(0.0f, spacing);

                    if (ps.to_display_frame_number ==
                        (dc_context->total_num_frame - 1)) {

                        ImVec4 repeat_normal =
                            ImVec4(0.85f, 0.75f, 0.20f, 1.0f);
                        ImVec4 repeat_hover = ImVec4(0.90f, 0.80f, 0.25f, 1.0f);
                        ImVec4 repeat_active =
                            ImVec4(0.80f, 0.70f, 0.18f, 1.0f);
                        ImGui::PushStyleColor(ImGuiCol_Button, repeat_normal);
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                              repeat_hover);
                        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                              repeat_active);

                        if (ImGui::Button(ICON_FK_REPEAT)) {
                            // seek to zero
                            seek_all_cameras(scene, 0, dc_context->video_fps,
                                             ps, false);
                        }
                        ImGui::PopStyleColor(3);
                    } else {
                        ImVec4 normal, hover, active;
                        if (ps.play_video) {
                            normal = ImVec4(0.8f, 0.3f, 0.3f, 1.0f);
                            hover = ImVec4(0.9f, 0.4f, 0.4f, 1.0f);
                            active = ImVec4(0.7f, 0.2f, 0.2f, 1.0f);
                        } else {
                            // green
                            normal = ImVec4(0.2f, 0.6f, 0.2f, 1.0f);
                            hover = ImVec4(0.4f, 0.9f, 0.4f, 1.0f);
                            active = ImVec4(0.3f, 0.75f, 0.3f, 1.0f);
                        }
                        ImGui::PushStyleColor(ImGuiCol_Button, normal);
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
                        ImGui::PushStyleColor(ImGuiCol_ButtonActive, active);
                        if (ImGui::Button(ps.play_video ? ICON_FK_PAUSE
                                                        : ICON_FK_PLAY)) {
                            ps.play_video = !ps.play_video;
                            if (ps.play_video) {
                                ps.pause_seeked = false;
                                ps.last_play_time_start =
                                    std::chrono::steady_clock::now();
                            } else {
                                ps.pause_selected = 0;
                            }
                        }
                        ImGui::PopStyleColor(3);
                    }

                    ImGui::SameLine(0.0f, spacing);
                    if (ImGui::Button(ICON_FK_STEP_FORWARD)) {
                        int clamped_frame = std::min(
                            dc_context->total_num_frame,
                            current_frame_num + dc_context->seek_interval);
                        seek_all_cameras(scene, clamped_frame,
                                         dc_context->video_fps, ps, false);
                    }
                    ImGui::SameLine(0.0f, spacing);
                    if (ImGui::Button(ICON_FK_FAST_FORWARD)) {
                        int clamped_frame = std::min(
                            dc_context->total_num_frame,
                            current_frame_num + 10 * dc_context->seek_interval);
                        seek_all_cameras(scene, clamped_frame,
                                         dc_context->video_fps, ps, false);
                    }
                    ImGui::SameLine();
                    ps.slider_just_changed = ImGui::SliderInt(
                        "##frame count", &ps.slider_frame_number, 0,
                        dc_context->estimated_num_frames);
                    ImGui::SameLine();
                    float current_time_sec =
                        ps.slider_frame_number / dc_context->video_fps;
                    float total_time_sec = dc_context->estimated_num_frames /
                                           dc_context->video_fps;

                    std::string current_str = format_time(current_time_sec);
                    std::string total_str = format_time(total_time_sec);
                    ImGui::Text("%s / %s", current_str.c_str(),
                                total_str.c_str());

                    if (ps.slider_just_changed) {
                        // std::cout << "main, seeking: " <<
                        // ps.slider_frame_number
                        //           << std::endl;
                        seek_all_cameras(scene, ps.slider_frame_number,
                                         dc_context->video_fps, ps, false);
                    }

                    ImGui::EndGroup();
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
