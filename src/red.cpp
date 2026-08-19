#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"
#include "mac_modifier_fix.h"
#include "IconsForkAwesome.h"
#include "camera.h"
#include "filesystem"
#include "global.h"
#include "gui.h"
#include "gui/help_window.h"
#include "gui/shortcuts.h"
#include "gui/jarvis_export_window.h"
#include "gui/jarvis_import_window.h"
#include "gui/pose_stats_window.h"
#include "gui/prediction_overlay.h"
#include "prediction_store.h"
#include "gui/export_window.h"
#include "gui/bbox_tool.h"
#include "gui/obb_tool.h"
#include "gui/midline_tool.h"
#ifdef __APPLE__
#include <CoreGraphics/CoreGraphics.h>
#endif
#include "gui/switch_skeleton_window.h"
#include "gui/annotation_dialog.h"
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
#ifdef _WIN32
#include <windows.h>                       // GetModuleFileName
#endif
#if defined(__linux__)
#include <limits.h>                        // PATH_MAX
#include <unistd.h>                        // readlink
#endif
#include "implot.h"
#include "implot_internal.h"
#include "project.h"
#include "render.h"
#include "skeleton.h"
#include "utils.h"
#include "app_context.h"
#include "deferred_queue.h"
#include "user_settings.h"
#include "jarvis_export.h"
#include <ImGuiFileDialog.h>
#include <algorithm>
#include <climits>                        // INT_MAX (only pulled in via a __linux__-guarded <limits.h> otherwise)
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
    // Print build timestamp so the user can verify they're running the latest
    // rebuild (debugging stale-binary issues during integration work).
    printf("red built %s %s\n", __DATE__, __TIME__);
    gx_context *window = (gx_context *)malloc(sizeof(gx_context));
    memset(window, 0, sizeof(gx_context));
    window->swap_interval = 1; // use vsync
    window->width = 1920;
    window->height = 1080;
    window->render_target_title = (char *)malloc(100); // window title
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
#elif defined(_WIN32)
    {
        char exe_buf[MAX_PATH];
        GetModuleFileNameA(NULL, exe_buf, MAX_PATH);
        window->exe_dir = std::filesystem::canonical(exe_buf).parent_path().string();
    }
#else
    {
        // On Linux read the real binary path from /proc/self/exe, so red works
        // when launched via PATH (e.g. a `red` symlink): argv[0] is then just
        // "red" and canonical(argv[0]) would fail. Fall back to argv[0] only if
        // /proc isn't available.
        char exe_buf[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
        if (n > 0) {
            exe_buf[n] = '\0';
            window->exe_dir =
                std::filesystem::path(exe_buf).parent_path().string();
        } else {
            window->exe_dir =
                std::filesystem::canonical(argv[0]).parent_path().string();
        }
    }
#endif

    render_initialize_target(window);
    RenderScene *scene = (RenderScene *)malloc(sizeof(RenderScene));
    // scene is malloc'd uninitialized; explicitly set the bool fields we
    // read before buffer allocation so behavior is deterministic. Default
    // CPU Buffer — see UserSettings::use_cpu_buffer for rationale.
    scene->use_cpu_buffer = true;
    std::string red_data_dir;
    std::string media_root_dir;
    prepare_application_folders(red_data_dir, media_root_dir);
    UserSettings user_settings = load_user_settings();
    // Honor the persisted buffer mode (default GPU Buffer on first launch).
    // Frame buffers are allocated later in media_loader::render_allocate_scene_memory
    // based on this flag, so it must be set before any project is loaded.
    scene->use_cpu_buffer = user_settings.use_cpu_buffer;
    // Seed the keypoint colormap global from persisted settings before any
    // project loads (setup_project reads it to color node_colors).
    g_keypoint_colormap = user_settings.keypoint_colormap;
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

    // Predictions live in a separate, memory-mapped store rather than in
    // `annotations`, so importing a whole video never floods the Labeling Tool.
    // Frames are promoted into `annotations` one at a time via Pose Stats.
    predstore::PredictionReader prediction_store;
    std::string active_store_path;

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

    win.export_win.margin = user_settings.jarvis_margin;
    win.export_win.train_ratio = user_settings.jarvis_train_ratio;
    win.export_win.seed = user_settings.jarvis_seed;
    win.export_win.jpeg_quality = user_settings.jarvis_jpeg_quality;


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
    DeferredQueue preframe;  // flushed before ImGui::NewFrame() (ini/dock reloads)

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
        popups, toasts, deferred, preframe,
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
        prediction_store.close();
        active_store_path.clear();
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
                    if (keypoints_find && keys::pressed(keys::Sc::Triangulate)) {
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
    panels.add({"Help", [&]() {
                    help::Context hctx;
                    hctx.project_open = !pm.project_path.empty();
                    hctx.is_3d        = hctx.project_open && !project_is_2d(pm);
                    hctx.bbox_on      = win.bbox.enabled;
                    hctx.obb_on       = win.obb.enabled;
                    hctx.midline_on   = win.midline.enabled;
                    DrawHelpWindow(win.show_help, hctx);
                }, nullptr});
    panels.add({"JARVIS Export",
                [&]() { DrawJarvisExportWindow(win.jarvis_export, ctx); },
                nullptr});
    panels.add({"Import JARVIS Predictions",
                [&]() { DrawJarvisImportWindow(win.jarvis_import, ctx); },
                nullptr});
    panels.add({"Settings",
                [&]() { DrawSettingsWindow(win.settings, ctx); },
                nullptr});
    panels.add({"Export Tool",
                [&]() { DrawExportWindow(win.export_win, ctx, annotations); },
                nullptr});
    panels.add({"Group JARVIS Export",
                [&]() { DrawGroupExportWindow(win.group_export, ctx); },
                nullptr});
    panels.add({"Bbox Tool",
                [&]() { DrawBBoxToolWindow(win.bbox, ctx); },
                nullptr});
    panels.add({"OBB Tool",
                [&]() { DrawOBBToolWindow(win.obb, ctx); },
                nullptr});
    panels.add({"Midline Tool",
                [&]() { DrawMidlineToolWindow(win.midline, ctx); },
                [&]() { return pm.plot_keypoints_flag; }});
    panels.add({"Triangulation Diagnostics",
                [&]() { DrawTriangulationDiagnosticsWindow(
                            win.triangulation_diag, ctx); },
                nullptr});
    panels.add({"Welcome",
                [&]() { DrawWelcomeWindow(ctx, win); },
                [&]() { return pm.project_path.empty() && !ps.video_loaded &&
                                !win.annotation.show &&
                                !ImGuiFileDialog::Instance()->IsOpened(); }});
    panels.add({"Pose Stats",
                [&]() { DrawPoseStatsWindow(win.pose_stats, prediction_store,
                                            active_store_path, skeleton,
                                            current_frame_num); },
                nullptr});
    panels.add({"Frame Drops",
                [&]() { DrawFrameDropsWindow(win.frame_drops, ctx); },
                nullptr});
    panels.add({"Switch Skeleton",
                [&]() { DrawSwitchSkeletonWindow(win.switch_skeleton, ctx); },
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

        // Run work that must happen OUTSIDE the ImGui frame (project-switch
        // ini/dock reloads enqueued by switch_ini_to_path). Reloading dock
        // settings mid-frame crashes: nodes are destroyed while windows are
        // already submitted, and rebuilt nodes can bind to memory-compacted
        // host windows.
        preframe.flush();

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

        // Override the backend's cached modifier state with the real hardware
        // state (macOS only; no-op elsewhere). Must sit between the backend's
        // NewFrame and ImGui::NewFrame so these events are queued last and win.
        red_sync_mac_modifiers();

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

        // Dock cameras into the central node as tabs -- but only when the
        // project has no camera layout of its own.
        //
        // SetNextWindowDockID(..., FirstUseEver) cannot do this alone:
        // switch_ini_to_path() reloads the .ini in a preframe callback, and
        // LoadIniSettingsFromDisk undocks every live window (DockId = 0). It
        // then restores only windows that HAVE an .ini entry, which the
        // sidebar panels do and camera windows do not -- so cameras were
        // undocked on every load and ended up floating. This pass runs after
        // that reload, so it has the last word.
        //
        // Skipped as soon as the project's .ini has any entry for a camera
        // window: that project already has a remembered arrangement, and
        // silently re-tabbing it would destroy the user's layout AND get
        // written back to the .ini on exit. Only projects that have never
        // placed their cameras get the default tab layout. Keyed on the camera
        // list so it evaluates once per project rather than every frame.
        if (ps.video_loaded && scene->num_cams > 0) {
            const int ncam = std::min((int)scene->num_cams,
                                      (int)pm.camera_names.size());
            std::string cam_signature;
            for (int j = 0; j < ncam; ++j)
                cam_signature += pm.camera_names[j] + "|";
            static std::string docked_signature;
            if (cam_signature != docked_signature) {
                docked_signature = cam_signature;
                bool has_saved_layout = false;
                for (int j = 0; j < ncam && !has_saved_layout; ++j) {
                    ImGuiID wid = ImHashStr(pm.camera_names[j].c_str());
                    has_saved_layout =
                        ImGui::FindWindowSettingsByID(wid) != nullptr;
                }
                const ImGuiID central = 0x00000005;
                if (!has_saved_layout && ImGui::DockBuilderGetNode(central)) {
                    for (int j = 0; j < ncam; ++j)
                        ImGui::DockBuilderDockWindow(
                            pm.camera_names[j].c_str(), central);
                    ImGui::DockBuilderFinish(0x00000001);
                }
            }
        }

        // Draw all registered panels
        panels.drawAll();

        // A fresh import asks for its store to be activated. Done here (after
        // the panels have run) so the reader is swapped outside any draw call
        // that might already hold a pointer into the old mmap.
        if (!win.jarvis_import.store_to_load.empty()) {
            std::string p = win.jarvis_import.store_to_load;
            win.jarvis_import.store_to_load.clear();
            prediction_store.close();
            if (prediction_store.open(p)) {
                if ((int)prediction_store.num_keypoints() != skeleton.num_nodes) {
                    ctx.toasts.pushError(
                        "Store has " +
                        std::to_string(prediction_store.num_keypoints()) +
                        " keypoints, skeleton has " +
                        std::to_string(skeleton.num_nodes) + " — not activated.");
                    prediction_store.close();
                    active_store_path.clear();
                } else {
                    active_store_path = p;
                    win.pose_stats.show = true;
                }
            } else {
                ctx.toasts.pushError("Could not open store: " + p);
                active_store_path.clear();
            }
        }

        // A skeleton switch re-indexes keypoints, so a store written at the old
        // node count can never display again — release it instead of leaving a
        // live mmap that every consumer silently skips.
        if (prediction_store.is_open() &&
            (int)prediction_store.num_keypoints() != skeleton.num_nodes) {
            prediction_store.close();
            active_store_path.clear();
            ctx.toasts.push("Prediction store closed (skeleton changed)");
        }

        // Pose Stats "Fix this frame": promote one predicted frame into the
        // editable Labeling Tool. Reprojects the stored 3D to each camera as
        // Predicted-source keypoints, keeps the 3D, and flags the frame
        // Needs-Improvement so it lands in that section of the labeling strip.
        if (win.pose_stats.promote_requested) {
            win.pose_stats.promote_requested = false;
            int pf = win.pose_stats.promote_frame;
            const float *pose = prediction_store.frame((uint32_t)pf);
            if (pose &&
                (int)prediction_store.num_keypoints() == skeleton.num_nodes &&
                scene->num_cams > 0) {
                FrameAnnotation &fa = get_or_create_frame(
                    annotations, (u32)pf, skeleton.num_nodes, scene->num_cams);
                fa.needs_improvement = true;
                int placed_3d = 0;
                for (int k = 0; k < skeleton.num_nodes; ++k) {
                    float x = pose[k * 4 + 0], y = pose[k * 4 + 1];
                    float z = pose[k * 4 + 2], c = pose[k * 4 + 3];
                    if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue;
                    fa.kp3d[k].x = x; fa.kp3d[k].y = y; fa.kp3d[k].z = z;
                    fa.kp3d[k].set_imported(c);  // predicted, awaiting review
                    placed_3d++;
                    Eigen::Vector3d p3d(x, y, z);
                    for (int cam = 0; cam < scene->num_cams &&
                                      cam < (int)pm.camera_params.size(); ++cam) {
                        double px, py;
                        if (reproject_3d_to_cam(p3d, pm.camera_params[cam],
                                                (int)scene->image_width[cam],
                                                (int)scene->image_height[cam],
                                                px, py)) {
                            auto &kp2d = fa.cameras[cam].keypoints[k];
                            kp2d.x = px; kp2d.y = py; kp2d.labeled = true;
                            kp2d.confidence = c;
                            kp2d.source = LabelSource::Predicted;
                        }
                    }
                }
                seek_all_cameras(scene, pf, dc_context->video_fps, ps, true);
                current_frame_num = pf;
                ps.pause_selected = 0; ps.pause_seeked = true;
                for (auto &[key, value] : window_need_decoding)
                    value.store(true);
                pm.plot_keypoints_flag = true;  // reveal the Labeling Tool
                ctx.toasts.pushSuccess("Frame " + std::to_string(pf) +
                    " -> Needs Improvement (" +
                    std::to_string(placed_3d) + " keypoints)");
            } else {
                ctx.toasts.pushError("Could not promote frame " +
                    std::to_string(pf) + " (no matching prediction).");
            }
        }


        // Frame Drops: double-click-to-seek request (already mapped to
        // playback coordinates by the window).
        if (win.frame_drops.seek_requested) {
            win.frame_drops.seek_requested = false;
            int tgt = win.frame_drops.seek_frame;
            seek_all_cameras(scene, tgt, dc_context->video_fps, ps, true);
            current_frame_num = tgt;
            ps.pause_selected = 0;
            ps.pause_seeked = true;
            for (auto &[key, value] : window_need_decoding)
                value.store(true);
        }





        // Handle main menu file dialogs
        HandleMainMenuDialogs(ctx, win, media_root_dir,
                              print_metadata, print_summary,
                              [&]() {});

        // Recent Projects click: the path is already known, so load it straight
        // through the shared loader instead of routing back through the file
        // dialog. Handled here (not in the panel) because close_project() +
        // win.reset() must not run while the Welcome window is mid-draw.
        if (!win.load_project_request.empty()) {
            std::filesystem::path req = win.load_project_request;
            win.load_project_request.clear();
            load_project_from_path(ctx, win, req, print_metadata, print_summary,
                                   [&]() {});
        }

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

            if (keys::pressed(keys::Sc::BufferPrev)) {
                if (ps.pause_selected > 0) {
                    ps.pause_selected--;
                    selection_changed = true;
                }
            }

            if (keys::pressed(keys::Sc::BufferNext)) {
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

            for (int j = 0; j < scene->num_cams; j++) {
                const std::string &win_name = pm.camera_names[j];

                // Dock every camera into the central node from the default
                // layout, so they arrive as tabs in one pane rather than
                // spread across a 2x2 grid. FirstUseEver, so a project whose
                // saved .ini already places them keeps its own arrangement.
                ImGui::SetNextWindowDockID(0x00000005, ImGuiCond_FirstUseEver);
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
                    window_need_decoding[win_name].store(is_visible);
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

                        if (fn != mac_last_uploaded_frame[j]) {
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
                    {
                        // CUDA-GL interop sync: only re-map/unmap the PBO
                        // around CUDA writes WHEN the contrast/brightness
                        // kernel actually runs. The kernel's mid-stream
                        // writes are what GL was racing with — visible as
                        // horizontal stripes. When sliders are at identity
                        // we skip the kernel AND the map/unmap, restoring
                        // the per-cam CUDA↔GL overlap that keeps playback
                        // fast. Pure cudaMemcpy paths (viz upload, frame
                        // copy at defaults) work without explicit sync
                        // because they don't change the data structure
                        // significantly between GL reads.
                        const bool render_contrast_identity =
                            display.contrast == 1.0f &&
                            display.brightness == 0;
                        const bool needs_pbo_sync = !render_contrast_identity;
                        if (needs_pbo_sync) {
                            // Unmap-then-remap establishes a CUDA↔GL barrier
                            // at the start of CUDA writes (any prior GL op
                            // on this PBO finishes before CUDA writes begin).
                            unmap_cuda_resource(&scene->pbo_cuda[j].cuda_resource);
                            map_cuda_resource(&scene->pbo_cuda[j].cuda_resource);
                            cuda_pointer_from_resource(
                                &scene->pbo_cuda[j].cuda_buffer,
                                &scene->pbo_cuda[j].cuda_pbo_storage_buffer_size,
                                &scene->pbo_cuda[j].cuda_resource);
                        }

                        {
                            // Shared by play + paused branches below. The
                            // contrast/brightness kernel is a no-op at
                            // identity (alpha=1, beta=0); skip when defaults
                            // to save kernel + sync cost on every render.
                            const bool contrast_identity =
                                display.contrast == 1.0f &&
                                display.brightness == 0;
                            if (ps.play_video) {
                                current_frame_num = ps.to_display_frame_number;
                                if (scene->use_cpu_buffer) {
                                    ck(cudaMemcpy(
                                        scene->pbo_cuda[j].cuda_buffer,
                                        scene->display_buffer[j][ps.read_head].frame,
                                        scene->image_width[j] * scene->image_height[j] *
                                            4,
                                        cudaMemcpyHostToDevice));
                                    if (!contrast_identity) {
                                        apply_contrast_brightness_rgba(
                                            scene->pbo_cuda[j].cuda_buffer,
                                            scene->image_width[j], scene->image_height[j],
                                            display.contrast,
                                            (float)display.brightness,
                                            display.pivot_midgray,
                                            0);
                                    }
                                } else {
                                    ck(cudaMemcpy(
                                        scene->pbo_cuda[j].cuda_buffer,
                                        scene->display_buffer[j][ps.read_head].frame,
                                        scene->image_width[j] * scene->image_height[j] *
                                            4,
                                        cudaMemcpyDeviceToDevice));
                                    if (!contrast_identity) {
                                        apply_contrast_brightness_rgba(
                                            scene->pbo_cuda[j].cuda_buffer,
                                            scene->image_width[j], scene->image_height[j],
                                            display.contrast,
                                            (float)display.brightness,
                                            display.pivot_midgray,
                                            0);
                                    }
                                }
                            } else {
                                // contrast_identity already declared above
                                // for the play branch — reused here.
                                if (scene->use_cpu_buffer) {
                                    ck(cudaMemcpy(
                                        scene->pbo_cuda[j].cuda_buffer,
                                        scene->display_buffer[j][select_corr_head]
                                            .frame,
                                        scene->image_width[j] * scene->image_height[j] *
                                            4,
                                        cudaMemcpyHostToDevice));
                                    if (!contrast_identity) {
                                        apply_contrast_brightness_rgba(
                                            scene->pbo_cuda[j].cuda_buffer,
                                            scene->image_width[j], scene->image_height[j],
                                            display.contrast,
                                            (float)display.brightness,
                                            display.pivot_midgray,
                                            0);
                                    }
                                } else {
                                    ck(cudaMemcpy(
                                        scene->pbo_cuda[j].cuda_buffer,
                                        scene->display_buffer[j][select_corr_head]
                                            .frame,
                                        scene->image_width[j] * scene->image_height[j] *
                                            4,
                                        cudaMemcpyDeviceToDevice));
                                    if (!contrast_identity) {
                                        apply_contrast_brightness_rgba(
                                            scene->pbo_cuda[j].cuda_buffer,
                                            scene->image_width[j], scene->image_height[j],
                                            display.contrast,
                                            (float)display.brightness,
                                            display.pivot_midgray,
                                            0);
                                    }
                                }
                            }
                        }
                        if (needs_pbo_sync) {
                            // Unmap before GL reads — sync point between
                            // the kernel and GL. PBO will be remapped at
                            // the start of the next render that needs it.
                            unmap_cuda_resource(&scene->pbo_cuda[j].cuda_resource);
                            map_cuda_resource(&scene->pbo_cuda[j].cuda_resource);
                            cuda_pointer_from_resource(
                                &scene->pbo_cuda[j].cuda_buffer,
                                &scene->pbo_cuda[j].cuda_pbo_storage_buffer_size,
                                &scene->pbo_cuda[j].cuda_resource);
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

                        // Desync fix: the displayed slot is a duplicate
                        // standing in for a frame this camera dropped.
                        if (dc_context->sync_fix_active.load()) {
                            int disp_head = ps.play_video ? ps.read_head
                                                          : select_corr_head;
                            if (scene->display_buffer[j][disp_head]
                                    .dropped.load())
                                DrawDroppedFrameBadge(
                                    (float)scene->image_width[j],
                                    (float)scene->image_height[j]);
                        }

                        if (pm.plot_keypoints_flag) {
                            // labeling (keypoints)
                            // OBB tool uses G key (not W), so no keypoint conflict
                            if (ImPlot::IsPlotHovered()) {
                                // Focus is sticky: the last view the cursor
                                // entered stays focused until another view is
                                // entered. It must NOT clear on hover-exit --
                                // keypoints are ImPlot::DragPoint items, and
                                // hovering one makes IsPlotHovered() false, so
                                // clearing here would un-float this camera's row
                                // in the Keypoints table exactly while the user
                                // is working on a keypoint in it.
                                for (int v = 0; v < (int)is_view_focused.size(); ++v)
                                    is_view_focused[v] = (v == j);
                                if (keys::pressed(keys::Sc::CreateFrame)) {
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
                                    if (keys::pressed(keys::Sc::PlaceKeypoint)) {
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

                                    if (keys::pressed(keys::Sc::ActivePrev)) {
                                        if (*kp <= 0) {
                                            *kp = 0;
                                        } else
                                            (*kp)--;
                                    }

                                    if (keys::pressed(keys::Sc::ActiveNext)) {
                                        if (*kp >= skeleton.num_nodes - 1) {
                                            *kp = skeleton.num_nodes - 1;
                                        } else
                                            (*kp)++;
                                    }

                                    if (keys::pressed(keys::Sc::ActiveLast)) {
                                        *kp = skeleton.num_nodes - 1;
                                    }

                                    if (keys::pressed(keys::Sc::ActiveFirst)) {
                                        *kp = 0;
                                    }

                                    // delete all keypoints on a frame
                                    if (keys::pressed(keys::Sc::DeleteAllKp)) {
                                        annotations.erase(current_frame_num);
                                        keypoints_find = false;
                                    }
                                }
                            }

                            // Hold P while hovering a view to hide its label
                            // overlay (manual keypoints + prediction overlay) and
                            // peek at the raw image underneath. Per-view: affects
                            // only the hovered image; release to restore.
                            bool peek_raw = ImPlot::IsPlotHovered() &&
                                            keys::held(keys::Sc::PeekRaw);

                            if (keypoints_find && skeleton.has_skeleton &&
                                display.show_keypoints && !peek_raw) {
                                gui_plot_keypoints(
                                    annotations.at(current_frame_num),
                                    &skeleton, j, scene->num_cams,
                                    active_keypoint_color(user_settings));
                            }

                            // Read-only prediction overlay. Skipped once the
                            // frame has its own annotation entry (promoted or
                            // hand-labeled) so it doesn't linger over data the
                            // user now owns.
                            if (!keypoints_find && skeleton.has_skeleton &&
                                display.show_keypoints && !peek_raw &&
                                prediction_store.is_open() &&
                                (int)prediction_store.num_keypoints() ==
                                    skeleton.num_nodes) {
                                const float *pose = prediction_store.frame(
                                    (uint32_t)current_frame_num);
                                if (pose)
                                    gui_plot_prediction_overlay(
                                        pose, j, &skeleton, pm.camera_params,
                                        scene);
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

                            // Midline tool (line drawing in the line camera)
                            if (win.midline.enabled) {
                                midline_handle_input(win.midline, annotations,
                                                     frame, j, nn, nc, iw, ih);
                            }
                            if (display.show_keypoints) {
                                midline_draw_overlay(win.midline, annotations,
                                                     frame, j);
                            }

                        }

                        // Plot context menu: press 1 key while hovering
                        if (ImPlot::IsPlotHovered() &&
                            keys::pressed(keys::Sc::PlotMenu)) {
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
                            ImGui::Checkbox("Bounding Boxes", &display.show_bboxes);
                            ImGui::EndPopup();
                        }

                        ImPlot::EndPlot();
                    }

                    ImGui::EndChild();
                }
                ImGui::End();
            }

            if (keys::pressed(keys::Sc::PlayPause)) {
                ps.play_video = !ps.play_video;
                if (ps.play_video) {
                    ps.pause_seeked = false;
                    ps.accumulated_play_time =
                        ps.to_display_frame_number / dc_context->video_fps;
                    ps.last_play_time_start = std::chrono::steady_clock::now();
                    ps.accumulated_play_time = ps.to_display_frame_number / dc_context->video_fps;
                } else {
                    ps.pause_selected = 0;
                }
            }


            if (keys::pressed(keys::Sc::SeekBack)) {
                seek_relative(ImGui::GetIO().KeyShift ? -10 : -1);
            }

            if (keys::pressed(keys::Sc::SeekFwd)) {
                seek_relative(ImGui::GetIO().KeyShift ? 10 : 1);
            }

            for (const auto &[name, flag] : window_need_decoding) {
                window_was_decoding[name] = flag.load();
            }
        }

        // H-key help toggle
        if (keys::pressed(keys::Sc::ToggleHelp)) {
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
