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
#ifdef RED_HAS_MUJOCO
#include "mujoco_context.h"
#include "gui/body_model_window.h"
#endif
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
#include "gui/welcome_window.h"
#include "gui/transport_bar.h"
#include "gui/frame_buffer_window.h"
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
#include "pointsource_calibration.h"
#include "aruco_metal.h"
#include "pointsource_metal.h"
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

#ifdef __APPLE__
// Extract RGB pixel data from a BGRA CVPixelBuffer.
inline void extract_rgb_from_cvpixelbuf(CVPixelBufferRef pb, std::vector<uint8_t> &rgb, int w, int h) {
    CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(pb);
    size_t stride = CVPixelBufferGetBytesPerRow(pb);
    rgb.resize(w * h * 3);
    for (int y = 0; y < h; y++) {
        const uint8_t *src = base + y * stride;
        uint8_t *dst = rgb.data() + y * w * 3;
        for (int x = 0; x < w; x++) {
            dst[x * 3 + 0] = src[x * 4 + 2]; // R
            dst[x * 3 + 1] = src[x * 4 + 1]; // G
            dst[x * 3 + 2] = src[x * 4 + 0]; // B
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
}
#endif

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

int main(int argc, char **argv) {
    // Use new (not malloc) because gx_context contains a std::string member,
    // which cannot be default-initialized via malloc()'s raw memory.
    gx_context *window = new gx_context{};
    window->swap_interval = 1;  // use vsync
    window->width = 1920;
    window->height = 1080;
    window->render_target_title = (char *)malloc(100);
    window->glsl_version = (char *)malloc(100);
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
    // calloc instead of malloc: RenderScene has plain bool/pointer fields that
    // must start zeroed. Previously `use_cpu_buffer` was read before being
    // assigned, silently using whatever garbage the heap provided.
    RenderScene *scene = (RenderScene *)calloc(1, sizeof(RenderScene));
    std::string red_data_dir;
    std::string media_root_dir;
    prepare_application_folders(red_data_dir, media_root_dir);
    UserSettings user_settings = load_user_settings();
    std::string skeleton_dir = red_data_dir + "/skeleton";
    std::vector<std::thread> decoder_threads;
    std::vector<FFmpegDemuxer *> demuxers;

    DecoderContext *dc_context = new DecoderContext{};
    dc_context->total_num_frame = int(INT_MAX);
    dc_context->seek_interval = 250;  // overwritten by auto-detect in media_loader
    dc_context->video_fps = 60.0f;

    // gui states — bundled into WindowStates (gui/window_states.h)
    WindowStates win;
    bool save_requested = false;
    int current_frame_num = 0;
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

    // Initialize window states from user settings
    win.jarvis_export.margin = user_settings.jarvis_margin;
    win.jarvis_export.train_ratio = user_settings.jarvis_train_ratio;
    win.jarvis_export.seed = user_settings.jarvis_seed;
    win.jarvis_export.jpeg_quality = user_settings.jarvis_jpeg_quality;

    win.calibration.project.project_root_path =
        user_settings.default_project_root_path.empty()
            ? red_data_dir
            : user_settings.default_project_root_path;
    if (!user_settings.default_media_root_path.empty())
        win.calibration.project.config_file = user_settings.default_media_root_path;

    win.export_win.margin = user_settings.jarvis_margin;
    win.export_win.train_ratio = user_settings.jarvis_train_ratio;
    win.export_win.seed = user_settings.jarvis_seed;
    win.export_win.jpeg_quality = user_settings.jarvis_jpeg_quality;

    // Inference engine states (not window states — kept separate)
    SamState sam_state;
    JarvisState jarvis_state;
#ifdef RED_HAS_MUJOCO
    MujocoContext mujoco_ctx;
#endif
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
                win.sam_tool.encoder_path = std::filesystem::canonical(enc).string();
                win.sam_tool.decoder_path = std::filesystem::canonical(dec).string();
                break;
            }
        }
    }

    // Default SuperPoint model path: look relative to exe (../models/superpoint/)
    {
        std::string exe = window->exe_dir;
        std::vector<std::string> search = {
            exe + "/../models/superpoint/superpoint.mlpackage",   // build tree
            exe + "/models/superpoint/superpoint.mlpackage",      // installed
            exe + "/../share/red/models/superpoint/superpoint.mlpackage", // Homebrew
        };
        for (const auto &path : search) {
            if (std::filesystem::exists(path)) {
                win.calibration.sp_model_path = std::filesystem::canonical(path).string();
                break;
            }
        }
    }

    win.annotation.video_folder = user_settings.default_media_root_path.empty()
                                     ? media_root_dir
                                     : user_settings.default_media_root_path;

    colors[ImPlotCol_Crosshairs] = ImVec4(0.3f, 0.10f, 0.64f, 1.00f);

    int label_buffer_size = user_settings.default_buffer_size;
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
        input_is_imgs, label_buffer_size, current_frame_num,
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
        // Validate new project BEFORE closing old project — if setup fails
        // we want to keep the old project intact and show the error.
        ProjectManager new_pm = pm_ref;
        if (!ensure_dir_exists(new_pm.project_path, &err))
            return false;
        if (!setup_project(new_pm, skeleton, skeleton_map, &err))
            return false;
        std::filesystem::path redproj_path =
            std::filesystem::path(new_pm.project_path) / (new_pm.project_name + ".redproj");
        if (!save_project_manager_json(new_pm, redproj_path, &err))
            return false;

        // Validation passed — now safe to close old project
        close_project(ctx);
        win.reset();
        // Nuke inference engines (different project may use different models)
        sam_state = SamState{};
        jarvis_state = JarvisState{};
#ifdef RED_HAS_MUJOCO
        mujoco_ctx.unload();
#endif
#ifdef __APPLE__
        jarvis_coreml_state = JarvisCoreMLState{};
#endif
        pm_ref = new_pm;
        // Re-initialize skeleton after close_project() cleared it.
        // (close_project resets ctx.skeleton; we must re-run setup_project
        // so the skeleton is populated for the new project.)
        std::string setup_err;
        if (!setup_project(pm_ref, skeleton, skeleton_map, &setup_err)) {
            err = setup_err;
            return false;
        }
        on_project_loaded(ctx, print_metadata, print_summary);
        return true;
    };

    // Panel registry — replaces manual draw calls
    PanelRegistry panels;
    panels.add({"Create Project",
                [&]() { DrawProjectWindow(ctx); }, nullptr});
    panels.add({"Annotation Dialog",
                [&]() { DrawAnnotationDialog(win.annotation, ctx, annot_create_cb); },
                nullptr});
    panels.add({"Keypoints",
                [&]() { DrawKeypointsWindow(ctx); },
                [&]() { return pm.plot_keypoints_flag; }});
    panels.add({"Labeling Tool",
                [&]() {
                    DrawLabelingToolWindow(win.labeling, ctx);
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
    panels.add({"Help", [&]() { DrawHelpWindow(win.show_help); }, nullptr});
    panels.add({"JARVIS Export",
                [&]() { DrawJarvisExportWindow(win.jarvis_export, ctx); },
                nullptr});
    panels.add({"JARVIS Import",
                [&]() { DrawJarvisImportWindow(win.jarvis_import, ctx); },
                nullptr});
    panels.add({"Calibration Tool",
                [&]() { DrawCalibrationToolWindow(win.calibration, ctx, calib_cb); },
                nullptr});
    panels.add({"Settings",
                [&]() { DrawSettingsWindow(win.settings, ctx); },
                nullptr});
    panels.add({"Export Tool",
                [&]() { DrawExportWindow(win.export_win, ctx, annotations); },
                nullptr});
    panels.add({"Bbox Tool",
                [&]() { DrawBBoxToolWindow(win.bbox, ctx); },
                nullptr});
    panels.add({"OBB Tool",
                [&]() { DrawOBBToolWindow(win.obb, ctx); },
                nullptr});
    panels.add({"SAM Assist",
                [&]() { DrawSamToolWindow(win.sam_tool, sam_state, ctx); },
                nullptr});
#ifdef RED_HAS_MUJOCO
    panels.add({"Body Model",
                [&]() { DrawBodyModelWindow(win.body_model, mujoco_ctx, ctx); },
                nullptr});
#endif
    panels.add({"Welcome",
                [&]() { DrawWelcomeWindow(ctx, win); },
                [&]() { return pm.project_path.empty() && !ps.video_loaded &&
                                !win.calibration.show && !win.annotation.show &&
                                !ImGuiFileDialog::Instance()->IsOpened(); }});
    panels.add({"JARVIS Predict",
                [&]() { DrawJarvisPredictWindow(win.jarvis_predict, jarvis_state,
#ifdef __APPLE__
                                                 jarvis_coreml_state,
#endif
                                                 ctx); },
                nullptr});

    // Helper: find the first visible camera index (for frame-buffer display).
    auto find_visible_cam = [&]() -> int {
        if (ps.pause_seeked) return 0;
        for (int i = 0; i < scene->num_cams; i++)
            if (window_was_decoding[pm.camera_names[i]]) return i;
        return 0;
    };

    // Helper: seek by a signed multiplier of the seek interval.
    auto seek_relative = [&](int multiplier) {
        int target = std::clamp(current_frame_num + multiplier * dc_context->seek_interval,
                                0, dc_context->total_num_frame);
        seek_all_cameras(scene, target, dc_context->video_fps, ps, false);
    };

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
        DrawMainMenuBar(ctx, win);

        // --- Update playback time ---
        auto now = std::chrono::steady_clock::now();

        if (ps.play_video) {
            ps.accumulated_play_time +=
                std::chrono::duration<double>(now - ps.last_play_time_start)
                    .count() *
                ps.set_playback_speed;
            ps.last_play_time_start = now;
        }

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
        DrawTransportBar(win.transport, ctx);

        ImGui::DockSpaceOverViewport(0x00000001);

        // Draw all registered panels
        panels.drawAll();

        // Handle main menu file dialogs
        HandleMainMenuDialogs(ctx, win, media_root_dir,
                              print_metadata, print_summary,
                              [&]() {
                                  sam_state = SamState{};
                                  jarvis_state = JarvisState{};
#ifdef RED_HAS_MUJOCO
                                  mujoco_ctx.unload();
#endif
#ifdef __APPLE__
                                  jarvis_coreml_state = JarvisCoreMLState{};
#endif
                              });

        static int select_corr_head = 0;
        if (ps.video_loaded && (!ps.play_video)) {
            int visible_idx = find_visible_cam();

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

        DrawFrameBufferWindow(ctx, select_corr_head);

        // Render a video frame
        if (ps.video_loaded) {
#ifdef __APPLE__
            // --- Laser detection viz: dispatch once before camera loop ---
            // This must run before per-camera iteration so that hidden cameras
            // (not in a visible ImGui window) still get processed.
            if (win.calibration.pointsource_show_detection && win.calibration.pointsource_ready) {
                auto &lv = win.calibration.pointsource_viz;
                auto &lc = win.calibration.pointsource_config;
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
                        lv.metal_ctx = pointsource_metal_create();

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
                    bool smart_blob = lc.smart_blob;
                    int ncams = scene->num_cams;

                    lv.computing.store(true, std::memory_order_release);
                    lv.last_green_th = green_th;
                    lv.last_green_dom = green_dom;
                    lv.last_min_blob = min_blob;
                    lv.last_max_blob = max_blob;

                    auto metal_ctx = lv.metal_ctx;
                    lv.worker = std::thread(
                        [inputs, ncams, green_th, green_dom,
                         min_blob, max_blob, smart_blob, metal_ctx, &lv]() {
                            std::vector<PointSourceVizState::CamResult> results(ncams);

                            // Phase 1: ALL cameras in parallel via fast detect (for stats)
                            {
                                std::vector<std::thread> threads;
                                for (int ci = 0; ci < ncams; ci++) {
                                    auto &inp = (*inputs)[ci];
                                    if (!inp.pixel_buffer) continue;
                                    threads.emplace_back([&inp, &results, ci,
                                        metal_ctx, green_th, green_dom, min_blob, max_blob, smart_blob]() {
                                        auto &res = results[ci];
                                        res.frame_num = inp.frame_num;
                                        auto spot = pointsource_metal_detect(
                                            metal_ctx, inp.pixel_buffer,
                                            green_th, green_dom, min_blob, max_blob, smart_blob);
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
                                pointsource_metal_detect_viz(
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
                    // Recompute buffer index after seek reset read_head/pause_selected
                    select_corr_head = (ps.pause_selected + ps.read_head) % scene->size_of_buffer;
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
                    // Recompute buffer index after seek reset read_head/pause_selected
                    select_corr_head = (ps.pause_selected + ps.read_head) % scene->size_of_buffer;
                }

                if (ps.play_video) {
                    window_need_decoding[win_name].store(
                        is_visible || (win.calibration.pointsource_show_detection && win.calibration.pointsource_ready));
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

                        if (win.calibration.pointsource_show_detection && win.calibration.pointsource_ready) {
                            // Upload ready result for this camera if available
                            auto &lv = win.calibration.pointsource_viz;
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
                        current_frame_num = ps.to_display_frame_number;
                        if (scene->use_cpu_buffer) {
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
                            ck(cudaMemcpy(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->display_buffer[j][select_corr_head]
                                    .frame,
                                scene->image_width[j] * scene->image_height[j] *
                                    4,
                                cudaMemcpyHostToDevice));

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
                    if (win.sam_tool.enabled &&
                        win.sam_tool.has_pending_mask &&
                        !win.sam_tool.multi_mask.masks.empty() &&
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
                                        !(win.sam_tool.enabled &&
                                          (!win.sam_tool.fg_points.empty() ||
                                           !win.sam_tool.bg_points.empty()))) {
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
                            if (win.bbox.enabled) {
                                bbox_handle_input(win.bbox, annotations,
                                                  frame, j, nn, nc, iw, ih);
                            }
                            if (display.show_bboxes) {
                                bbox_draw_overlays(win.bbox, annotations,
                                                   frame, j, iw, ih);
                            }

                            // OBB tool
                            if (win.obb.enabled) {
                                obb_handle_input(win.obb, win.bbox,
                                                 annotations, frame, j,
                                                 nn, nc, iw, ih);
                            }
                            if (display.show_bboxes) {
                                obb_draw_overlays(win.obb, win.bbox,
                                                  annotations, frame, j, iw, ih);
                            }

                            // Accepted mask overlays (stored in AnnotationMap)
                            if (display.show_masks) {
                                draw_accepted_masks(annotations, frame, j, iw, ih);
                            }

                            // SAM assist
                            if (win.sam_tool.enabled) {
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
                                            extract_rgb_from_cvpixelbuf(pb, sam_rgb_buf, iw, ih);
                                            sam_rgb_frame = (int)frame;
                                            sam_rgb_cam = j;
                                        }
                                    }
                                    if (sam_rgb_frame == (int)frame && sam_rgb_cam == j)
                                        sam_rgb = sam_rgb_buf.data();
                                }
#endif
                                sam_handle_input(win.sam_tool, sam_state,
                                                 annotations, frame, j,
                                                 nn, nc, iw, ih, sam_rgb);
                            }
                            if (display.show_masks)
                                sam_draw_overlay(win.sam_tool, j, iw, ih);
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
                !io.WantTextInput && !win.jarvis_predict.batch_running) {
                ps.play_video = !ps.play_video;
                if (ps.play_video) {
                    ps.pause_seeked = false;
                    ps.last_play_time_start = std::chrono::steady_clock::now();
                    ps.accumulated_play_time = ps.to_display_frame_number / dc_context->video_fps;
                } else {
                    ps.pause_selected = 0;
                }
            }


            // Hotkey 6: Run JARVIS prediction on current frame
            bool jarvis_predict_trigger =
                (ImGui::IsKeyPressed(ImGuiKey_6, false) && !io.WantTextInput) ||
                win.jarvis_predict.predict_requested;
            win.jarvis_predict.predict_requested = false;

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

                // "All" mode: ensure every camera has the current frame
                if (win.jarvis_predict.predict_from_all) {
                    // Check if any camera is missing a valid pixel buffer
                    bool needs_seek = false;
                    for (int c = 0; c < (int)scene->num_cams; ++c) {
                        auto &slot = scene->display_buffer[c][mh];
                        if (!slot.pixel_buffer ||
                            slot.frame_number.load() != current_frame_num) {
                            needs_seek = true;
                            break;
                        }
                    }
                    if (needs_seek) {
                        seek_all_cameras(scene, current_frame_num,
                                         dc_context->video_fps, ps, true);
                        ps.pause_selected = 0;
                        for (auto &[key, value] : window_need_decoding)
                            value.store(true);
                        mh = (ps.pause_selected + ps.read_head) % scene->size_of_buffer;
                        // Wait for all cameras to fill slot (up to ~2s)
                        for (int wait = 0; wait < 2000; ++wait) {
                            bool ready = true;
                            for (int c = 0; c < (int)scene->num_cams; ++c) {
                                auto &slot = scene->display_buffer[c][0];
                                if (slot.available_to_write.load() || !slot.pixel_buffer) {
                                    ready = false;
                                    break;
                                }
                            }
                            if (ready) break;
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                        mh = 0; // seek resets to slot 0
                        select_corr_head = mh;
                    }
                }

                // Build pixel buffer array, filtering by mode
                auto cam_included = [&](int c) -> bool {
                    if (win.jarvis_predict.predict_from_all) return true;
                    // "Shown" mode: only cameras visible AND with fresh frames
                    if (c < (int)pm.camera_names.size() &&
                        window_is_visible.count(pm.camera_names[c]) &&
                        window_is_visible[pm.camera_names[c]]) {
                        auto &slot = scene->display_buffer[c][mh];
                        return slot.pixel_buffer &&
                               slot.frame_number.load() == current_frame_num;
                    }
                    return false;
                };

                // Prefer CoreML (GPU/ANE) over ONNX Runtime (CPU)
                if (jarvis_coreml_state.loaded) {
                    std::vector<CVPixelBufferRef> pbs(scene->num_cams, nullptr);
                    int cams_used = 0;
                    for (int c = 0; c < (int)scene->num_cams; ++c) {
                        if (cam_included(c)) {
                            pbs[c] = scene->display_buffer[c][mh].pixel_buffer;
                            if (pbs[c]) cams_used++;
                        }
                    }

                    jarvis_coreml_predict_frame(jarvis_coreml_state, annotations,
                        (u32)current_frame_num, pbs, widths, heights,
                        skeleton, (int)scene->num_cams,
                        win.jarvis_predict.confidence_threshold);
                    // Triangulate (reprojection is in gui_keypoints.h, only in this TU)
                    reprojection(annotations.at(current_frame_num),
                                 &skeleton, pm.camera_params, scene);
                    printf("[JARVIS CoreML] %s (%d/%d cameras)\n",
                           jarvis_coreml_state.status.c_str(),
                           cams_used, (int)scene->num_cams);
                } else if (jarvis_state.loaded) {
                    // Fallback: ONNX Runtime (CPU) with BGRA→RGB conversion
                    std::vector<const uint8_t *> rgb_bufs(scene->num_cams, nullptr);
                    std::vector<std::vector<uint8_t>> rgb_storage(scene->num_cams);
                    for (int c = 0; c < (int)scene->num_cams; ++c) {
                        if (!cam_included(c)) continue;
                        CVPixelBufferRef pb = scene->display_buffer[c][mh].pixel_buffer;
                        if (!pb) continue;
                        extract_rgb_from_cvpixelbuf(pb, rgb_storage[c], widths[c], heights[c]);
                        rgb_bufs[c] = rgb_storage[c].data();
                    }
                    jarvis_predict_frame(jarvis_state, annotations,
                        (u32)current_frame_num, rgb_bufs, widths, heights,
                        skeleton, pm.camera_params, scene,
                        win.jarvis_predict.confidence_threshold);
                    printf("[JARVIS ONNX] %s\n", jarvis_state.status.c_str());
                }
#else
                // Linux / non-Apple path: pull RGBA frames (CPU or cudaMalloc'd
                // GPU) from the display buffer, convert to RGB, run JARVIS
                // ONNX Runtime inference (CUDA EP if available), triangulate.
                int mh = ps.play_video ? ps.read_head : select_corr_head;
                std::vector<int> widths(scene->num_cams), heights(scene->num_cams);
                for (int c = 0; c < (int)scene->num_cams; ++c) {
                    widths[c] = (int)scene->image_width[c];
                    heights[c] = (int)scene->image_height[c];
                }

                auto cam_included_lx = [&](int c) -> bool {
                    if (c >= (int)pm.camera_names.size()) return false;
                    if (win.jarvis_predict.predict_from_all) return true;
                    // "Shown" mode: only cameras whose window is the active
                    // docked tab (ImGui::Begin returned true this frame) AND
                    // whose decoder buffer actually holds current_frame_num.
                    auto it = window_is_visible.find(pm.camera_names[c]);
                    if (it == window_is_visible.end() || !it->second)
                        return false;
                    auto &slot = scene->display_buffer[c][mh];
                    return slot.frame != nullptr &&
                           slot.frame_number.load() == current_frame_num;
                };

                std::vector<const uint8_t *> rgb_bufs(scene->num_cams, nullptr);
                std::vector<std::vector<uint8_t>> rgb_storage(scene->num_cams);
                std::vector<std::vector<uint8_t>> rgba_scratch(scene->num_cams);
                int cams_used = 0;
                std::string included_names, skipped_names;
                for (int c = 0; c < (int)scene->num_cams; ++c) {
                    const std::string &nm = pm.camera_names[c];
                    if (!cam_included_lx(c)) {
                        if (!skipped_names.empty()) skipped_names += ",";
                        skipped_names += nm;
                        continue;
                    }
                    auto &slot = scene->display_buffer[c][mh];
                    if (!slot.frame) {
                        if (!skipped_names.empty()) skipped_names += ",";
                        skipped_names += nm + "(nullframe)";
                        continue;
                    }

                    size_t npix = (size_t)widths[c] * heights[c];
                    const uint8_t *rgba = nullptr;
                    if (scene->use_cpu_buffer) {
                        rgba = (const uint8_t *)slot.frame;
                    } else {
                        rgba_scratch[c].resize(npix * 4);
                        cudaMemcpy(rgba_scratch[c].data(), slot.frame,
                                   npix * 4, cudaMemcpyDeviceToHost);
                        rgba = rgba_scratch[c].data();
                    }

                    rgb_storage[c].resize(npix * 3);
                    uint8_t *dst = rgb_storage[c].data();
                    for (size_t i = 0; i < npix; ++i) {
                        dst[i*3+0] = rgba[i*4+0];
                        dst[i*3+1] = rgba[i*4+1];
                        dst[i*3+2] = rgba[i*4+2];
                    }
                    rgb_bufs[c] = dst;
                    cams_used++;
                    if (!included_names.empty()) included_names += ",";
                    included_names += nm;
                }

                // Clear stale Predicted keypoints on cameras we intentionally
                // skipped this run so they don't bias triangulation or display
                // as "fresh" results.
                if (!win.jarvis_predict.predict_from_all &&
                    annotations.count(current_frame_num)) {
                    auto &fa = annotations.at(current_frame_num);
                    for (int c = 0; c < (int)scene->num_cams && c < (int)fa.cameras.size(); ++c) {
                        if (rgb_bufs[c] != nullptr) continue;
                        for (auto &kp : fa.cameras[c].keypoints) {
                            if (kp.source == LabelSource::Predicted) {
                                kp.labeled = false;
                                kp.confidence = 0.0f;
                            }
                        }
                    }
                }

                printf("[JARVIS ONNX] predicting on %d/%d cams: [%s]  (skipped: [%s])\n",
                       cams_used, (int)scene->num_cams,
                       included_names.c_str(), skipped_names.c_str());

                jarvis_predict_frame(jarvis_state, annotations,
                    (u32)current_frame_num, rgb_bufs, widths, heights,
                    skeleton, pm.camera_params, scene,
                    win.jarvis_predict.confidence_threshold);
                if (!pm.camera_params.empty() &&
                    annotations.count(current_frame_num)) {
                    reprojection(annotations.at(current_frame_num),
                                 &skeleton, pm.camera_params, scene);
                }
                printf("[JARVIS ONNX] %s\n", jarvis_state.status.c_str());
#endif
            }

            // --- Batch prediction (non-blocking state machine) ---
            // Processes one frame per render iteration so the UI stays
            // responsive and camera viewports update live.
            {
                auto &bp = win.jarvis_predict;
                using Phase = JarvisPredictState::BatchPhase;
                int buf_size = (int)scene->size_of_buffer;

                // --- Initialize ---
                if (bp.batch_requested && !bp.batch_running) {
                    bp.batch_requested = false;
                    bp.batch_running = true;
                    bp.batch_current = bp.batch_start;
                    bp.batch_completed = 0;
                    bp.batch_skipped = 0;
                    bp.batch_total = (bp.batch_end - bp.batch_start) / bp.batch_step + 1;
                    bp.batch_status = "Running...";
                    bp.batch_predict_ms = 0;
                    bp.batch_t0 = std::chrono::steady_clock::now();
                    bp.batch_phase = Phase::SEEK;
                    bp.batch_chunk_start = bp.batch_start;
                    ps.play_video = false;
                    printf("[Batch] Starting: frames %d-%d step %d (%d frames)\n",
                           bp.batch_start, bp.batch_end, bp.batch_step, bp.batch_total);
                }

                // --- Per-frame state machine tick ---
                if (bp.batch_running && jarvis_any_loaded && scene->num_cams > 0) {
                    switch (bp.batch_phase) {

                    case Phase::SEEK: {
                        // Seek to current chunk start. Blocking (~3-5s) but
                        // only happens once per 64-frame buffer fill.
                        seek_all_cameras(scene, bp.batch_chunk_start,
                                         dc_context->video_fps, ps, false);
                        current_frame_num = bp.batch_chunk_start;
                        // Force all decoders to fill the buffer
                        for (auto &[key, value] : window_need_decoding)
                            value.store(true);

                        int frames_needed = std::min(buf_size,
                            bp.batch_end - bp.batch_chunk_start + 1);
                        bp.batch_chunk_last_slot = frames_needed - 1;
                        bp.batch_wait_frames = 0;
                        bp.batch_phase = Phase::WAIT_BUFFER;
                        printf("[Batch] Seeking to frame %d (need %d buffer slots)...\n",
                               bp.batch_chunk_start, frames_needed);
                        break;
                    }

                    case Phase::WAIT_BUFFER: {
                        // Check if all cameras have filled the last needed slot.
                        // Returns to the render loop each iteration (~16ms) to
                        // keep UI responsive while decoders fill in background.
                        bool ready = true;
                        for (int c = 0; c < (int)scene->num_cams; c++) {
                            auto &slot = scene->display_buffer[c][bp.batch_chunk_last_slot];
                            if (slot.available_to_write
#ifdef __APPLE__
                                || !slot.pixel_buffer
#endif
                               ) {
                                ready = false;
                                break;
                            }
                        }
                        if (ready) {
                            printf("[Batch] Buffer filled (waited %d frames)\n",
                                   bp.batch_wait_frames);
                            // Stop decoder threads to free CPU cores for CoreML
                            for (auto &[key, value] : window_need_decoding)
                                value.store(false);
                            bp.batch_phase = Phase::PREDICT;
                        } else {
                            bp.batch_wait_frames++;
                            if (bp.batch_wait_frames > 900) { // ~15s at 60fps
                                bp.batch_status = "Error: buffer fill timeout";
                                bp.batch_running = false;
                                printf("[Batch] Timeout waiting for buffer fill\n");
                            }
                        }
                        break;
                    }

                    case Phase::PREDICT: {
                        // Process all target frames in this chunk in a tight
                        // loop (no Metal render between them). This avoids
                        // IOSurface lock contention from Metal viewport blits.
#ifdef __APPLE__
                        int nc_pred = (int)scene->num_cams;
                        std::vector<int> w_b(nc_pred), h_b(nc_pred);
                        std::vector<CVPixelBufferRef> pbs(nc_pred, nullptr);
                        for (int c = 0; c < nc_pred; ++c) {
                            w_b[c] = (int)scene->image_width[c];
                            h_b[c] = (int)scene->image_height[c];
                        }
#endif
                        while (bp.batch_current <= bp.batch_end && bp.batch_running) {
                            int slot = bp.batch_current - bp.batch_chunk_start;
                            if (slot >= buf_size) {
                                // Need a new buffer chunk
                                bp.batch_chunk_start = bp.batch_current;
                                bp.batch_phase = Phase::SEEK;
                                // Re-enable decoders for next chunk fill
                                for (auto &[key, value] : window_need_decoding)
                                    value.store(true);
                                break;
                            }

                            u32 frame = (u32)bp.batch_current;

                            // Skip frames with manual labels
                            bool has_manual = false;
                            if (annotations.count(frame)) {
                                const auto &fa = annotations.at(frame);
                                for (const auto &cam : fa.cameras) {
                                    for (const auto &kp : cam.keypoints)
                                        if (kp.labeled && kp.source == LabelSource::Manual) {
                                            has_manual = true; break;
                                        }
                                    if (has_manual) break;
                                }
                            }

                            if (has_manual) {
                                bp.batch_skipped++;
                            } else {
#ifdef __APPLE__
                                auto tp0 = std::chrono::steady_clock::now();
                                for (int c = 0; c < nc_pred; ++c)
                                    pbs[c] = scene->display_buffer[c][slot].pixel_buffer;
                                jarvis_coreml_predict_frame(jarvis_coreml_state,
                                    annotations, frame, pbs, w_b, h_b,
                                    skeleton, (int)scene->num_cams,
                                    bp.confidence_threshold);
                                if (!pm.camera_params.empty())
                                    reprojection(annotations.at(frame),
                                                 &skeleton, pm.camera_params, scene);
                                auto tp1 = std::chrono::steady_clock::now();
                                bp.batch_predict_ms += std::chrono::duration<float, std::milli>(tp1 - tp0).count();
                                bp.batch_completed++;
                                printf("[Batch] Frame %u (slot %d): %.0f ms  [%d/%d]\n",
                                       frame, slot, jarvis_coreml_state.last_total_ms,
                                       bp.batch_completed, bp.batch_total);
#else
                                bp.batch_completed++;
                                printf("[Batch] Frame %u (slot %d)  [%d/%d]\n",
                                       frame, slot,
                                       bp.batch_completed, bp.batch_total);
#endif
                            }

                            bp.batch_current += bp.batch_step;
                        }

                        // Show last predicted frame in viewports
                        int last_slot = std::min(bp.batch_current - bp.batch_step - bp.batch_chunk_start,
                                                  buf_size - 1);
                        ps.pause_selected = last_slot;
                        ps.pause_seeked = true;
                        current_frame_num = bp.batch_current - bp.batch_step;

                        if (bp.batch_current > bp.batch_end)
                            bp.batch_phase = Phase::FINISHING;
                        break;
                    }

                    case Phase::FINISHING: {
                        auto t1 = std::chrono::steady_clock::now();
                        float total_ms = std::chrono::duration<float, std::milli>(t1 - bp.batch_t0).count();
                        bp.batch_running = false;
                        bp.batch_phase = Phase::IDLE;
                        bp.batch_status = "Complete: " +
                            std::to_string(bp.batch_completed) + " frames in " +
                            std::to_string((int)(total_ms / 1000.0f)) + "s (" +
                            std::to_string((int)(total_ms / std::max(1, bp.batch_completed))) +
                            " ms/frame)";
                        if (bp.batch_skipped > 0)
                            bp.batch_status += " (" + std::to_string(bp.batch_skipped) +
                                " skipped)";
                        printf("[Batch] %s\n", bp.batch_status.c_str());
                        break;
                    }

                    case Phase::IDLE:
                        break;
                    }
                }
            }

            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false) &&
                !io.WantTextInput && !win.jarvis_predict.batch_running) {
                seek_relative(ImGui::GetIO().KeyShift ? -10 : -1);
            }

            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false) &&
                !io.WantTextInput && !win.jarvis_predict.batch_running) {
                seek_relative(ImGui::GetIO().KeyShift ? 10 : 1);
            }

            for (const auto &[name, flag] : window_need_decoding) {
                window_was_decoding[name] = flag.load();
            }
        }

        // H-key help toggle
        if (ImGui::IsKeyPressed(ImGuiKey_H, false) && !io.WantTextInput) {
            win.show_help = !win.show_help;
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
                        std::ceil(ps.accumulated_play_time * dc_context->video_fps));
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
