#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"
#include "IconsForkAwesome.h"
#include "camera.h"
#include "filesystem"
#include "global.h"
#include "gui.h"
#include "gui/help_window.h"
#include "gui/shortcuts.h"
#include "gui/jarvis_export_window.h"
#include "gui/export_window.h"
#include "gui/bbox_tool.h"
#include "gui/obb_tool.h"
#include "gui/midline_tool.h"
#include "gui/sam_tool.h"
#ifdef RED_HAS_MUJOCO
#include "mujoco_context.h"
#include "gui/body_model_window.h"
#endif
#include "jarvis_inference.h"
#ifdef __APPLE__
#include "jarvis_coreml.h"
#include <CoreGraphics/CoreGraphics.h>
#elif defined(_WIN32)
#include "jarvis_tensorrt.h"
#endif
#if defined(__linux__) || defined(_WIN32)
#include "jarvis_hybridnet.h"
#endif
#include "gui/jarvis_predict_window.h"
#include "jarvis_predict_export.h"
#include "jarvis_predict_import.h"
#include "prediction_store.h"
#include "prediction_merge.h"
#include "gui/switch_skeleton_window.h"
#include "gui/prediction_overlay.h"
#include "gui/bout_filter_preview.h"
#include "gui/annotation_dialog.h"
#include "gui/body_parts_window.h"
#include "gui/calibration_tool_window.h"
#include "gui/crop_designer.h"
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
#include "calibration_tool.h"
#include "calibration_pipeline.h"
#include "app_context.h"
#include "deferred_queue.h"
#include "user_settings.h"
#include "jarvis_export.h"
#include "pointsource_calibration.h"
#ifdef __APPLE__
#include "aruco_metal.h"
#include "pointsource_metal.h"
#elif defined(_WIN32) || defined(__linux__)
#include "aruco_cuda.h"
#if defined(USE_CUDA_POINTSOURCE)
#include "pointsource_cuda.h"
#endif
#endif
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

    // Separate prediction store (Batch Predict → Store mode). Kept out of
    // `annotations` so predicting large sections never floods the Labeling
    // Tool. `prediction_writer` streams to a .rpred file during a batch;
    // `prediction_store` mmaps the finished file for the read-only overlay
    // and (later) Pose Stats.
    predstore::PredictionWriter prediction_writer;
    predstore::PredictionReader prediction_store;

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
#elif defined(_WIN32)
    JarvisTensorRTState jarvis_trt_state;
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
    JarvisHybridNetState jarvis_hn_state;
#endif
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
                    // Reopen the JARVIS Predict panel if it was open last time.
                    win.jarvis_predict.show = pm.show_jarvis_predict;
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
        // Use the shared robust ini-switch (with dock-node fixup) so calibration
        // projects get a correct layout on a mid-session load -- same machinery
        // the annotation path uses. The bare LoadIniSettingsFromDisk used here
        // previously left the dock tree scrambled (labeling windows orphaned).
        switch_ini_to_path(ctx, proj_path);
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
#elif defined(_WIN32)
        jarvis_trt_state = JarvisTensorRTState{};
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
        jarvis_hybridnet_unload(jarvis_hn_state);
#endif
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
    panels.add({"Body Parts",
                [&]() { DrawBodyPartsWindow(ctx); },
                [&]() { return pm.plot_keypoints_flag; }});
    panels.add({"Help", [&]() {
                    help::Context hctx;
                    hctx.project_open = !pm.project_path.empty();
                    hctx.is_3d        = hctx.project_open && !project_is_2d(pm);
                    hctx.bbox_on      = win.bbox.enabled;
                    hctx.obb_on       = win.obb.enabled;
                    hctx.sam_on       = win.sam_tool.enabled;
                    hctx.midline_on   = win.midline.enabled;
                    DrawHelpWindow(win.show_help, hctx);
                }, nullptr});
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
    panels.add({"SAM Assist",
                [&]() { DrawSamToolWindow(win.sam_tool, sam_state, ctx); },
                nullptr});
    panels.add({"Triangulation Diagnostics",
                [&]() { DrawTriangulationDiagnosticsWindow(
                            win.triangulation_diag, ctx); },
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
#elif defined(_WIN32)
                                                 jarvis_trt_state,
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                                                 jarvis_hn_state,
#endif
#endif
                                                 ctx); },
                nullptr});
    panels.add({"Pose Stats",
                [&]() { DrawPoseStatsWindow(win.pose_stats, prediction_store,
                                            win.jarvis_predict.active_store_path,
                                            skeleton, current_frame_num,
                                            win.bouts, &win.bout_filter); },
                nullptr});
    panels.add({"Frame Drops",
                [&]() { DrawFrameDropsWindow(win.frame_drops, ctx); },
                nullptr});
    panels.add({"Pump Events",
                [&]() { DrawPumpEventsWindow(win.pump_events, ctx); },
                nullptr});
    panels.add({"Bouts",
                [&]() { DrawBoutsWindow(win.bouts, prediction_store,
                                        win.jarvis_predict.active_store_path,
                                        skeleton); },
                nullptr});
    panels.add({"Bout Filter",
                [&]() { DrawBoutFilterWindow(win.bout_filter, prediction_store,
                                             win.jarvis_predict.active_store_path,
                                             skeleton, ctx); },
                nullptr});
    panels.add({"Switch Skeleton",
                [&]() { DrawSwitchSkeletonWindow(win.switch_skeleton, ctx,
                                                 prediction_store, win.jarvis_predict,
                                                 win.bout_filter); },
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

        // "Label Landmarks" request: dock the Keypoints + Labeling Tool windows
        // into a fresh left sidebar and focus them. Done here (right after the
        // dockspace node exists, before panels submit) via DockBuilder so the
        // placement is deterministic -- it does not depend on the .ini layout,
        // which can be corrupted by a mid-session reload on a fresh project.
        if (win.calibration.request_dock_labeling) {
            win.calibration.request_dock_labeling = false;
            ImGuiID root = 0x00000001;
            ImGuiID left = 0;
            ImGui::DockBuilderSplitNode(root, ImGuiDir_Left, 0.20f, &left, &root);
            ImGui::DockBuilderDockWindow("Keypoints", left);
            ImGui::DockBuilderDockWindow("Labeling Tool", left);
            ImGui::DockBuilderFinish(0x00000001);
            ImGui::SetWindowFocus("Labeling Tool");
        }

        // pumpctl writes its dispense log into orange's recording folder, so
        // pick it up whenever the loaded media changes. Cheap no-op otherwise.
        pump_events_auto_load(win.pump_events, ctx);

        // Draw all registered panels
        panels.drawAll();

        // Pose Stats: double-click-to-seek request.
        if (win.pose_stats.seek_requested) {
            win.pose_stats.seek_requested = false;
            int tgt = win.pose_stats.seek_frame;
            seek_all_cameras(scene, tgt, dc_context->video_fps, ps, true);
            current_frame_num = tgt;
            ps.pause_selected = 0;
            ps.pause_seeked = true;
            for (auto &[key, value] : window_need_decoding)
                value.store(true);
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

        // Pump Events: seek to a dispense (already in playback coordinates —
        // the panel maps mp4 index -> canonical slot when the sync fix is on).
        if (win.pump_events.seek_requested) {
            win.pump_events.seek_requested = false;
            int tgt = win.pump_events.seek_frame;
            seek_all_cameras(scene, tgt, dc_context->video_fps, ps, true);
            current_frame_num = tgt;
            ps.pause_selected = 0;
            ps.pause_seeked = true;
            for (auto &[key, value] : window_need_decoding)
                value.store(true);
        }

        // Bouts: seek to a bout, or export bouts to CSV next to the store.
        if (win.bouts.seek_requested) {
            win.bouts.seek_requested = false;
            int tgt = win.bouts.seek_frame;
            seek_all_cameras(scene, tgt, dc_context->video_fps, ps, true);
            current_frame_num = tgt;
            ps.pause_selected = 0;
            ps.pause_seeked = true;
            for (auto &[key, value] : window_need_decoding)
                value.store(true);
        }
        if (win.bouts.export_requested) {
            win.bouts.export_requested = false;
            uint32_t fps = prediction_store.is_open() ? prediction_store.fps() : 0;
            win.bouts.export_status = bouts_export_csv(
                win.bouts, win.jarvis_predict.active_store_path, fps);
            printf("[Bouts] %s\n", win.bouts.export_status.c_str());
        }

        // Bout Filter: seek to a bout, or export the filtered bout list.
        if (win.bout_filter.seek_requested) {
            win.bout_filter.seek_requested = false;
            int tgt = win.bout_filter.seek_frame;
            // Preload ~50 frames before the bout start so the user can scrub
            // backward (e.g. while nudging the start via "Adjust...") without
            // waiting on a fresh decode. The buffer fills forward from the
            // seek target, so seek early and offset display via pause_selected
            // to land on the actual bout start frame.
            const int kBoutPreloadFrames = 50;
            int preload_from = std::max(0, tgt - kBoutPreloadFrames);
            seek_all_cameras(scene, preload_from, dc_context->video_fps, ps, true);
            current_frame_num = tgt;
            ps.pause_selected = tgt - preload_from;
            ps.pause_seeked = true;
            for (auto &[key, value] : window_need_decoding)
                value.store(true);
            // Arm the bout boundary: playback either wraps back to the bout
            // start or pauses at its end when it reaches loop_end, depending
            // on the "Loop bout" checkbox (checked live at the wrap site).
            // seek_all_cameras above cleared loop_bout; re-arm it here with
            // this bout's range. Any later manual seek clears it.
            // loop_start stays pinned to the real bout start, not preload_from.
            ps.loop_bout  = true;
            ps.loop_start = tgt;
            ps.loop_end   = win.bout_filter.seek_frame_end;
        }
        if (win.bout_filter.export_requested) {
            win.bout_filter.export_requested = false;
            win.bout_filter.export_status = bout_filter_export_csv(
                win.bout_filter, win.jarvis_predict.active_store_path);
            printf("[Bout Filter] %s\n", win.bout_filter.export_status.c_str());
        }
        // Persist the manual-edit overlay whenever it changed (kept off the
        // draw/hot path; writes only when edits actually mutate).
        if (win.bout_filter.edits_save_requested) {
            win.bout_filter.edits_save_requested = false;
            std::string err = bout_filter_save_edits(
                win.bout_filter, win.jarvis_predict.active_store_path);
            if (!err.empty()) printf("[Bout Filter] %s\n", err.c_str());
        }

        // Pose Stats: promote a predicted frame into the Labeling Tool so the
        // user can manually correct it. Reprojects the store's 3D pose to each
        // camera as editable (Predicted-source) keypoints, keeps the 3D, and
        // flags the frame Needs-Improvement so it lands in that section and is
        // protected from a later Batch Predict overwrite.
        if (win.pose_stats.promote_requested) {
            win.pose_stats.promote_requested = false;
            int pf = win.pose_stats.promote_frame;
            const float *pose = prediction_store.frame((uint32_t)pf);
            if (pose && (int)prediction_store.num_keypoints() == skeleton.num_nodes &&
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
                    " → Needs Improvement (" + std::to_string(placed_3d) +
                    " keypoints)");
            } else {
                ctx.toasts.pushError("Could not promote frame " +
                    std::to_string(pf) + " (no matching prediction).");
            }
        }

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
#elif defined(_WIN32)
                                  jarvis_trt_state = JarvisTensorRTState{};
#endif
                              });

        // Jump to the next/previous pump dispense. Outside the paused-only
        // block below so it also works during playback; the seek itself is
        // performed by the seek_requested handler on the next iteration.
        if (ps.video_loaded && !win.pump_events.events.empty() &&
            !io.WantTextInput) {
            if (ImGui::IsKeyPressed(ImGuiKey_RightBracket, false) &&
                !pump_events_jump(win.pump_events, current_frame_num, true))
                ctx.toasts.push("No later pump dispense");
            if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket, false) &&
                !pump_events_jump(win.pump_events, current_frame_num, false))
                ctx.toasts.push("No earlier pump dispense");
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
#elif (defined(_WIN32) || defined(__linux__)) && defined(USE_CUDA_POINTSOURCE)
            // --- Windows/Linux: Laser detection viz dispatch (CUDA) ---
            if (win.calibration.pointsource_show_detection && win.calibration.pointsource_ready) {
                auto &lv = win.calibration.pointsource_viz;
                auto &lc = win.calibration.pointsource_config;
                int win_head = ps.play_video ? ps.read_head : select_corr_head;
                int fn0 = scene->display_buffer[0][win_head].frame_number;

                // Collect results from background thread
                if (!lv.computing.load(std::memory_order_acquire) &&
                    !lv.pending.empty()) {
                    if (lv.worker.joinable())
                        lv.worker.join();
                    lv.ready = std::move(lv.pending);
                    lv.pending.clear();
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

                    // Lazy-init CUDA context for GPU viz
                    if (!lv.cuda_ctx)
                        lv.cuda_ctx = pointsource_cuda_create();

                    // Snapshot RGBA frame data for background thread.
                    // Display buffer is RGBA (from Nv12ToColor32<RGBA32>).
                    // The CUDA kernels work with RGBA: G is at offset 1 in both
                    // RGBA and BGRA, and the R/B threshold checks are symmetric.
                    struct CamInput {
                        std::vector<uint8_t> rgba_cpu;  // CPU copy of frame
                        int width, height, frame_num;
                        bool needs_rgba;  // visible cameras need viz overlay
                    };
                    int ncams = scene->num_cams;
                    auto inputs = std::make_shared<std::vector<CamInput>>(ncams);
                    for (int ci = 0; ci < ncams; ci++) {
                        auto &inp = (*inputs)[ci];
                        int w = scene->image_width[ci];
                        int h = scene->image_height[ci];
                        inp.width = w;
                        inp.height = h;
                        inp.frame_num = scene->display_buffer[ci][win_head].frame_number;
                        const std::string &cam_name = pm.camera_names[ci];
                        inp.needs_rgba = window_is_visible.count(cam_name) &&
                                         window_is_visible.at(cam_name);

                        // Copy frame to CPU for background processing
                        size_t frame_bytes = (size_t)w * h * 4;
                        inp.rgba_cpu.resize(frame_bytes);
                        unsigned char *src_frame = scene->display_buffer[ci][win_head].frame;
                        if (src_frame) {
                            if (scene->use_cpu_buffer) {
                                memcpy(inp.rgba_cpu.data(), src_frame, frame_bytes);
                            } else {
                                cudaMemcpy(inp.rgba_cpu.data(), src_frame, frame_bytes,
                                           cudaMemcpyDeviceToHost);
                            }
                        }
                    }

                    int green_th = lc.green_threshold;
                    int green_dom = lc.green_dominance;
                    int min_blob = lc.min_blob_pixels;
                    int max_blob = lc.max_blob_pixels;
                    bool smart_blob = lc.smart_blob;

                    lv.computing.store(true, std::memory_order_release);
                    lv.last_green_th = green_th;
                    lv.last_green_dom = green_dom;
                    lv.last_min_blob = min_blob;
                    lv.last_max_blob = max_blob;

                    auto cuda_ctx = lv.cuda_ctx;
                    lv.worker = std::thread(
                        [inputs, ncams, green_th, green_dom,
                         min_blob, max_blob, smart_blob, cuda_ctx, &lv]() {
                            std::vector<PointSourceVizState::CamResult> results(ncams);

                            // Phase 1: detect spots on all cameras (stats)
                            for (int ci = 0; ci < ncams; ci++) {
                                auto &inp = (*inputs)[ci];
                                if (inp.rgba_cpu.empty()) continue;
                                auto &res = results[ci];
                                res.frame_num = inp.frame_num;
                                int stride = inp.width * 4;
                                auto spot = pointsource_cuda_detect(
                                    cuda_ctx, inp.rgba_cpu.data(),
                                    inp.width, inp.height, stride,
                                    green_th, green_dom, min_blob, max_blob, smart_blob);
                                if (spot.found) {
                                    res.num_blobs = 1;
                                } else if (spot.pixel_count > 0) {
                                    res.num_blobs = -1; // ambiguous
                                }
                                res.total_mask_pixels = spot.pixel_count;
                            }

                            // Phase 2: visible cameras get RGBA overlay
                            for (int ci = 0; ci < ncams; ci++) {
                                auto &inp = (*inputs)[ci];
                                if (inp.rgba_cpu.empty() || !inp.needs_rgba) continue;
                                auto &res = results[ci];
                                res.rgba.resize(inp.width * inp.height * 4);
                                int stride = inp.width * 4;
                                pointsource_cuda_detect_viz(
                                    cuda_ctx, inp.rgba_cpu.data(),
                                    inp.width, inp.height, stride,
                                    green_th, green_dom, min_blob, max_blob,
                                    res.rgba.data());
                            }

                            lv.pending = std::move(results);
                            lv.computing.store(false, std::memory_order_release);
                        });
                }
            }
#endif // __APPLE__ / _WIN32

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

                        bool viz_uploaded = false;
#ifdef USE_CUDA_POINTSOURCE
                        // Check for PointSource viz overlay data
                        if (win.calibration.pointsource_show_detection &&
                            win.calibration.pointsource_ready) {
                            auto &lv = win.calibration.pointsource_viz;
                            if (j < (int)lv.ready.size() &&
                                !lv.ready[j].rgba.empty() &&
                                !lv.ready[j].uploaded) {
                                // Upload viz overlay (CPU RGBA) to PBO
                                ck(cudaMemcpy(
                                    scene->pbo_cuda[j].cuda_buffer,
                                    lv.ready[j].rgba.data(),
                                    scene->image_width[j] * scene->image_height[j] * 4,
                                    cudaMemcpyHostToDevice));
                                lv.ready[j].uploaded = true;
                                viz_uploaded = true;
                            }
                        }
#endif
                        if (!viz_uploaded) {
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
                                is_view_focused[j] = true;
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
                                    // (skip if SAM has active prompts — Backspace is SAM undo)
                                    if (keys::pressed(keys::Sc::DeleteAllKp) &&
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

                            // Read-only overlay of JARVIS predictions from the
                            // separate store. Skipped once the frame has its own
                            // annotation entry (promoted for correction, or
                            // hand-labeled) so the automatic prediction doesn't
                            // linger once a frame has manual/human-owned data.
                            if (!keypoints_find && skeleton.has_skeleton &&
                                display.show_keypoints && !peek_raw &&
                                win.jarvis_predict.show_prediction_overlay &&
                                prediction_store.is_open() &&
                                (int)prediction_store.num_keypoints() ==
                                    skeleton.num_nodes) {
                                const float *pose = prediction_store.frame(
                                    (uint32_t)current_frame_num);
                                if (pose)
                                    gui_plot_prediction_overlay(
                                        pose, j, &skeleton, pm.camera_params, scene,
                                        win.jarvis_predict.confidence_threshold);
                            }
                        }

                        // Bout Filter threshold-plane live preview —
                        // independent of keypoint display / skeleton presence.
                        if (win.bout_filter.inputs_valid &&
                            (win.bout_filter.show_floor_preview ||
                             win.bout_filter.show_ywall_preview ||
                             win.bout_filter.show_xwall_preview)) {
                            bout_filter_draw_wall_preview(
                                win.bout_filter, j, pm.camera_params, scene);
                        }

                        // --- Annotation tool overlays + input (bbox, OBB, SAM) ---
                        {
                            int iw = (int)scene->image_width[j];
                            int ih = (int)scene->image_height[j];
                            u32 frame = (u32)current_frame_num;
                            int nn = skeleton.num_nodes;
                            int nc = (int)scene->num_cams;

                            // Crop designer (cropped-sensor calibration wizard)
                            crop_designer_draw(win.calibration, annotations,
                                               j, iw, ih);

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

            if (keys::pressed(keys::Sc::PlayPause) &&
                !win.jarvis_predict.batch_running) {
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


            // Hotkey 6: Run JARVIS prediction on current frame
            bool jarvis_predict_trigger =
                keys::pressed(keys::Sc::PredictCurrent) ||
                win.jarvis_predict.predict_requested;
            win.jarvis_predict.predict_requested = false;

            bool jarvis_any_loaded = jarvis_state.loaded;
#ifdef __APPLE__
            jarvis_any_loaded = jarvis_any_loaded || jarvis_coreml_state.loaded;
#elif defined(_WIN32)
            jarvis_any_loaded = jarvis_any_loaded || jarvis_trt_state.loaded;
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
            jarvis_any_loaded = jarvis_any_loaded || jarvis_hn_state.loaded;
#endif
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
                        skeleton, (int)scene->num_cams, pm.camera_params,
                        win.jarvis_predict.confidence_threshold);
                    // HybridNet writes 3D directly; only 2D path needs triangulation
                    // (reprojection is in gui_keypoints.h, only in this TU).
                    if (!jarvis_coreml_state.hybridnet)
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
#elif defined(_WIN32) || defined(__linux__)
                // Windows: TensorRT (GPU) preferred, ONNX (GPU/CPU) fallback
                // Linux: ONNX Runtime only (TensorRT isolated to orange toolchain)
                int mh = ps.play_video ? ps.read_head : select_corr_head;
                std::vector<int> widths(scene->num_cams), heights(scene->num_cams);
                for (int c = 0; c < (int)scene->num_cams; ++c) {
                    widths[c] = (int)scene->image_width[c];
                    heights[c] = (int)scene->image_height[c];
                }

                // HybridNet needs all N cams (model has a fixed input shape);
                // the 2D + DLT path can work on whatever's visible.
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                const bool hn_active = jarvis_hn_state.loaded;
#else
                const bool hn_active = false;
#endif

#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                // Force a global seek + decoder-wake when HN predict is about
                // to fire and some cam doesn't yet have the current frame in
                // its ring buffer. Mirrors the macOS Predict-from-All path.
                // Without this, hidden camera tabs keep their decoder threads
                // paused (window_need_decoding=false) and HN predict bails
                // with "decoder hasn't caught up yet" — even with All selected.
                if (hn_active) {
                    auto has_target_frame = [&](int c) -> bool {
                        for (int s = 0; s < (int)scene->size_of_buffer; ++s) {
                            auto &cand = scene->display_buffer[c][s];
                            if (!cand.available_to_write.load() &&
                                cand.frame_number.load() == (int)current_frame_num)
                                return true;
                        }
                        return false;
                    };
                    bool any_missing = false;
                    for (int c = 0; c < (int)scene->num_cams; ++c) {
                        if (!has_target_frame(c)) { any_missing = true; break; }
                    }
                    if (any_missing) {
                        fprintf(stderr,
                            "[HN dispatch] forcing decode-all to reach frame %d "
                            "(some cams hidden / decoders paused)\n",
                            (int)current_frame_num);
                        seek_all_cameras(scene, current_frame_num,
                                         dc_context->video_fps, ps, true);
                        ps.pause_selected = 0;
                        for (auto &[key, value] : window_need_decoding)
                            value.store(true);
                        // Wait up to ~2 s for every cam to land the target
                        // frame in some slot of its ring buffer.
                        bool all_ready = false;
                        for (int wait = 0; wait < 2000; ++wait) {
                            all_ready = true;
                            for (int c = 0; c < (int)scene->num_cams; ++c) {
                                if (!has_target_frame(c)) { all_ready = false; break; }
                            }
                            if (all_ready) break;
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                        if (!all_ready) {
                            fprintf(stderr,
                                "[HN dispatch] decode-all wait timed out (>2s) — "
                                "some cams still missing frame %d; predict may skip\n",
                                (int)current_frame_num);
                        }
                        // seek_all_cameras resets ring buffers to slot 0.
                        mh = 0;
                        select_corr_head = mh;
                    }
                }
#endif

                auto cam_included = [&](int c) -> bool {
                    if (hn_active || win.jarvis_predict.predict_from_all) return true;
                    if (c < (int)pm.camera_names.size() &&
                        window_is_visible.count(pm.camera_names[c]) &&
                        window_is_visible[pm.camera_names[c]]) {
                        auto &slot = scene->display_buffer[c][mh];
                        return !slot.available_to_write &&
                               slot.frame_number.load() == current_frame_num;
                    }
                    return false;
                };

                // Extract RGBA→RGB from GPU frame buffers
                std::vector<const uint8_t *> rgb_bufs(scene->num_cams, nullptr);
                std::vector<std::vector<uint8_t>> rgb_storage(scene->num_cams);
                // Per-cam slot index: when HN is active, search each cam's
                // ring buffer for the slot whose frame_number matches the
                // target. The display's shared `mh` index works for cams
                // that happen to be synced, but at a seek target only a
                // subset of decoders may have caught up to current_frame_num,
                // and other cams' slot[mh] still holds stale data from when
                // the ring last wrapped through that index. Without this, HN
                // tries to fuse multi-view detections from different time
                // points and produces nonsense 3D.
                std::vector<int> mh_per_cam(scene->num_cams, mh);
                bool hn_can_proceed = true;
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                if (hn_active) {
                    const char *hn_verbose_env = std::getenv("RED_HN_VERBOSE");
                    const bool hn_verbose = (hn_verbose_env && hn_verbose_env[0] == '1');
                    for (int c = 0; c < (int)scene->num_cams; ++c) {
                        int found = -1;
                        for (int s = 0; s < (int)scene->size_of_buffer; ++s) {
                            auto &cand = scene->display_buffer[c][s];
                            if (!cand.available_to_write.load() &&
                                cand.frame_number.load() == (int)current_frame_num) {
                                found = s; break;
                            }
                        }
                        if (found < 0) {
                            // Abort reason — always logged so the user sees why
                            // predict was skipped (decoders out of sync).
                            fprintf(stderr,
                                "[HN dispatch] cam %d: no slot has frame %d "
                                "(decoder hasn't caught up yet); skipping predict\n",
                                c, (int)current_frame_num);
                            hn_can_proceed = false;
                            break;
                        }
                        mh_per_cam[c] = found;
                    }
                    if (hn_can_proceed && hn_verbose) {
                        fprintf(stderr, "[HN dispatch] target frame=%d, mh_per_cam={",
                                (int)current_frame_num);
                        for (int c = 0; c < (int)scene->num_cams; ++c)
                            fprintf(stderr, "%s%d", c ? "," : "", mh_per_cam[c]);
                        fprintf(stderr, "}\n");
                    }
                }
#endif
                // Device-resident RGBA32 pointers for the HN+TRT device path.
                // Always collected (zero cost — just pointer copies). The host
                // RGB extraction below is skipped when we'll use the device path,
                // saving ~30 ms of cudaMemcpy + alpha-strip per predict.
                std::vector<const uint8_t *> rgba_device_bufs(scene->num_cams, nullptr);
                for (int c = 0; c < (int)scene->num_cams; ++c) {
                    if (!cam_included(c)) continue;
                    auto &slot = scene->display_buffer[c][mh_per_cam[c]];
                    if (slot.frame) rgba_device_bufs[c] = slot.frame;
                }
#ifdef RED_HAS_TENSORRT_HN
                // Device kernel path is only valid when (a) HN is loaded and
                // (b) the frame buffers are actually device memory.
                // scene->use_cpu_buffer is a runtime-toggleable flag that can
                // be out of sync with how the buffers were allocated, so we
                // probe the actual cam 0 frame and let cudaPointerAttributes
                // be the authority.
                bool hn_device_path = hn_active;
                if (hn_device_path) {
                    // Pick the first valid frame pointer to probe.
                    const uint8_t *probe = nullptr;
                    for (int c = 0; c < (int)scene->num_cams; ++c) {
                        if (rgba_device_bufs[c]) { probe = rgba_device_bufs[c]; break; }
                    }
                    if (probe) {
                        cudaPointerAttributes a{};
                        cudaError_t err = cudaPointerGetAttributes(&a, probe);
                        if (err != cudaSuccess || a.type != cudaMemoryTypeDevice) {
                            hn_device_path = false;
                        }
                    } else {
                        hn_device_path = false;
                    }
                }
#else
                const bool hn_device_path = false;
#endif
                if (!hn_device_path) {
                    for (int c = 0; c < (int)scene->num_cams; ++c) {
                        if (!cam_included(c)) continue;
                        auto &slot = scene->display_buffer[c][mh_per_cam[c]];
                        if (!slot.frame) continue;
                        int w = widths[c], h = heights[c];
                        // slot.frame is RGBA32 in GPU memory — copy to CPU, strip α.
                        // cudaMemcpyDefault lets the runtime resolve the source device
                        // (NVDEC decoders can land buffers on different devices on
                        // multi-GPU boxes). Only needed for non-TRT paths now.
                        std::vector<uint8_t> rgba(w * h * 4);
                        cudaMemcpy(rgba.data(), slot.frame, w * h * 4, cudaMemcpyDefault);
                        rgb_storage[c].resize(w * h * 3);
                        for (int i = 0; i < w * h; ++i) {
                            rgb_storage[c][i * 3 + 0] = rgba[i * 4 + 0]; // R
                            rgb_storage[c][i * 3 + 1] = rgba[i * 4 + 1]; // G
                            rgb_storage[c][i * 3 + 2] = rgba[i * 4 + 2]; // B
                        }
                        rgb_bufs[c] = rgb_storage[c].data();
                    }
                }

#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                if (jarvis_hn_state.loaded) {
                    // HybridNet handles the full 3-stage pipeline internally
                    // (CenterDetect → DLT triangulate → effTrack → Hybrid3D)
                    // and writes both 3D and per-cam 2D back-projections to
                    // AnnotationMap. No separate reprojection() call needed.
                    // Outer try/catch in addition to predict_frame's own:
                    // an exception escaping from a TRT internal thread or
                    // destructor unwinding becomes a logged failure here,
                    // not a process abort.
                    bool ok = false;
                    if (!hn_can_proceed) {
                        fprintf(stderr,
                            "[JARVIS HybridNet] skipped: not all cams have frame %d "
                            "loaded yet (try \"Predict from All\" mode to force "
                            "decode, or wait/scrub until the 3D viewer shows all "
                            "cams synced)\n", (int)current_frame_num);
                    } else try {
                        if (hn_device_path) {
                            // GPU-buffer mode + TRT: kernels do Stages 1+3 on GPU.
                            ok = jarvis_hybridnet_predict_frame_device(
                                jarvis_hn_state, rgba_device_bufs,
                                widths, heights, pm.camera_params,
                                annotations, skeleton, (u32)current_frame_num);
                        } else {
                            // CPU-buffer mode: host RGB extracted above.
                            ok = jarvis_hybridnet_predict_frame(
                                jarvis_hn_state, rgb_bufs, widths, heights,
                                pm.camera_params, annotations, skeleton,
                                (u32)current_frame_num);
                        }
                    } catch (const std::exception &e) {
                        fprintf(stderr, "[JARVIS HybridNet] uncaught exception "
                                "in predict_frame: %s\n", e.what());
                    } catch (...) {
                        fprintf(stderr, "[JARVIS HybridNet] uncaught non-std "
                                "exception in predict_frame\n");
                    }
                    printf("[JARVIS HybridNet] %s  total=%.0fms  cams=%d/%d\n",
                           ok ? "ok" : "failed",
                           jarvis_hn_state.last_center_ms +
                           jarvis_hn_state.last_efftrack_ms +
                           jarvis_hn_state.last_hybrid3d_ms,
                           jarvis_hn_state.last_center_cams_used,
                           (int)scene->num_cams);
                } else
#endif
#ifdef _WIN32
                if (jarvis_trt_state.loaded) {
                    jarvis_tensorrt_predict_frame(jarvis_trt_state, annotations,
                        (u32)current_frame_num, rgb_bufs, widths, heights,
                        skeleton, (int)scene->num_cams,
                        win.jarvis_predict.confidence_threshold);
                    if (!pm.camera_params.empty())
                        reprojection(annotations.at(current_frame_num),
                                     &skeleton, pm.camera_params, scene);
                    printf("[JARVIS TensorRT] %s\n", jarvis_trt_state.status.c_str());
                } else
#endif
                if (jarvis_state.loaded) {
                    jarvis_predict_frame(jarvis_state, annotations,
                        (u32)current_frame_num, rgb_bufs, widths, heights,
                        skeleton, pm.camera_params, scene,
                        win.jarvis_predict.confidence_threshold);
                    printf("[JARVIS ONNX] %s\n", jarvis_state.status.c_str());
                }
#endif

                // Optional: mirror this single predicted frame into a
                // JARVIS-CLI-compatible Predictions_3D_<ts>/ folder (data3D.csv
                // + info.yaml), in addition to the in-red labeled frame. A
                // one-frame export: frame_start = current frame, count = 1.
                if (win.jarvis_predict.export_predictions3D &&
                    !pm.project_path.empty()) {
                    std::string dir = jarvis_make_predictions_dir(pm.project_path);
                    JarvisExportResult er = jarvis_export_predictions3D(
                        dir, annotations, skeleton, pm,
                        (int)current_frame_num, 1);
                    win.jarvis_predict.export_status = er.message;
                    printf("[JARVIS export] %s\n", er.message.c_str());
                }
            }

            // --- Prediction store: load-on-request + auto-open newest ---
            {
                auto &jp = win.jarvis_predict;
                // On project change, drop the active store and auto-open the
                // newest saved store for the new project (if any).
                static std::string last_store_project = "\x01";  // != any real path
                if (last_store_project != pm.project_path) {
                    last_store_project = pm.project_path;
                    jp.store_list_dirty = true;
                    jp.active_store_path.clear();
                    jp.store_status.clear();
                    prediction_store.close();
                    if (!pm.project_path.empty()) {
                        std::filesystem::path dir =
                            std::filesystem::path(pm.project_path) /
                            "predictions" / "red_store";
                        std::error_code ec;
                        std::string newest;
                        if (std::filesystem::is_directory(dir, ec)) {
                            for (auto &e :
                                 std::filesystem::directory_iterator(dir, ec)) {
                                if (e.path().extension() != ".rpred") continue;
                                if (e.path().string() <= newest) continue;
                                // Skip files with an unreadable/invalid header
                                // (e.g. a half-written cluster import) so we
                                // auto-open the newest VALID store.
                                if (predstore::read_store_header(
                                        e.path().string()).ok)
                                    newest = e.path().string();
                            }
                        }
                        if (!newest.empty()) jp.load_store_request = newest;
                    }
                }

                // Import a JARVIS-CLI 3D prediction folder (data3D.csv +
                // info.yaml) into this project's store, then load it through the
                // normal guarded path below. Never import mid-batch (the writer
                // would collide with the batch's own store).
                if (!jp.import_request.empty() && !jp.batch_running) {
                    std::string src = jp.import_request;
                    jp.import_request.clear();
                    if (pm.project_path.empty()) {
                        jp.import_status = "Open a project before importing.";
                    } else {
                        std::string store_dir =
                            (std::filesystem::path(pm.project_path) /
                             "predictions" / "red_store").string();
                        std::string model_name;
                        if (pm.active_jarvis_model >= 0 &&
                            pm.active_jarvis_model < (int)pm.jarvis_models.size())
                            model_name =
                                pm.jarvis_models[pm.active_jarvis_model].name;
                        int fps_hint = (int)std::lround(
                            dc_context->video_fps > 0 ? dc_context->video_fps : 0);
                        int total_hint =
                            (dc_context->total_num_frame > 0 &&
                             dc_context->total_num_frame != INT_MAX)
                                ? dc_context->total_num_frame
                                : 0;
                        JarvisImportResult ir = jarvis_import_predictions3D(
                            src, store_dir, pm.media_folder, model_name,
                            pm.skeleton_name, fps_hint, total_hint);
                        jp.import_status = ir.message;
                        printf("[Import] %s\n", ir.message.c_str());
                        if (ir.ok) {
                            jp.store_list_dirty = true;
                            if (ir.num_keypoints != skeleton.num_nodes)
                                jp.import_status +=
                                    "  (⚠ " + std::to_string(ir.num_keypoints) +
                                    " keypoints vs skeleton's " +
                                    std::to_string(skeleton.num_nodes) + ")";
                            else
                                jp.load_store_request = ir.store_path;  // auto-open
                        }
                    }
                }

                // The 3D-label importer (Import 3D Predictions) writes a .rpred
                // alongside the editable labels; activate it through the same
                // guarded path so Bouts / Pose Stats pick it up.
                if (!win.jarvis_import.store_to_load.empty()) {
                    jp.load_store_request = win.jarvis_import.store_to_load;
                    jp.store_list_dirty = true;   // refresh Saved Predictions list
                    win.jarvis_import.store_to_load.clear();
                }

                // Consume a load request (picker click or auto-open). Never
                // load while a batch is writing its own store.
                if (!jp.load_store_request.empty() && !jp.batch_running) {
                    std::string p = jp.load_store_request;
                    jp.load_store_request.clear();
                    prediction_store.close();
                    if (prediction_store.open(p)) {
                        int nkp = (int)prediction_store.num_keypoints();
                        if (nkp != skeleton.num_nodes) {
                            // A store whose keypoint count doesn't match the
                            // active skeleton must NOT stay active: Pose Stats /
                            // Bouts would read past each frame's block. Close it
                            // and leave no active store.
                            prediction_store.close();
                            jp.active_store_path.clear();
                            jp.store_status = "⚠ store has " +
                                std::to_string(nkp) + " keypoints, skeleton has " +
                                std::to_string(skeleton.num_nodes) +
                                " — not loaded";
                        } else {
                            jp.active_store_path = p;
                            jp.store_status = "Loaded " +
                                std::to_string(prediction_store.stored_frames()) +
                                " frames from " +
                                std::filesystem::path(p).filename().string();
                        }
                        printf("[Store] %s\n", jp.store_status.c_str());
                    } else {
                        jp.active_store_path.clear();
                        jp.store_status = "Failed to open store: " +
                            std::filesystem::path(p).filename().string();
                    }
                }
            }

            // --- Batch prediction (non-blocking state machine) ---
            // Processes one frame per render iteration so the UI stays
            // responsive and camera viewports update live.
            {
                auto &bp = win.jarvis_predict;
                using Phase = JarvisPredictState::BatchPhase;
                using PredDest = JarvisPredictState::PredDest;
                int buf_size = (int)scene->size_of_buffer;

                // Extract frame's 3D+conf from `annotations` into a
                // 4*num_nodes float buffer (NaN where a keypoint has no valid
                // 3D). Returns true if at least one keypoint is valid.
                auto extract_pred_row = [&](u32 frame,
                                            std::vector<float> &buf) -> bool {
                    const int nn = skeleton.num_nodes;
                    buf.assign((size_t)nn * 4,
                               std::numeric_limits<float>::quiet_NaN());
                    auto it = annotations.find(frame);
                    if (it == annotations.end()) return false;
                    const auto &kp3d = it->second.kp3d;
                    bool any = false;
                    for (int jn = 0; jn < nn && jn < (int)kp3d.size(); ++jn) {
                        const auto &p = kp3d[jn];
                        if (p.source != Kp3DSource::None && p.x != UNLABELED) {
                            buf[jn * 4 + 0] = (float)p.x;
                            buf[jn * 4 + 1] = (float)p.y;
                            buf[jn * 4 + 2] = (float)p.z;
                            buf[jn * 4 + 3] = p.confidence;
                            any = true;
                        }
                    }
                    return any;
                };

                // In Store mode: pull the just-predicted frame's 3D into the
                // store writer, then erase it from `annotations` so it never
                // reaches the Labeling Tool. No-op in LabelingTool mode.
                auto stash_prediction = [&](u32 frame) {
                    if (bp.batch_prediction_dest != PredDest::Store) return;
                    // Store the prediction if the writer is healthy, then ALWAYS
                    // erase the frame from `annotations` — in Store mode a
                    // prediction must never leak into the Labeling Tool, even if
                    // the writer failed to open or a write errored mid-batch.
                    if (prediction_writer.is_open()) {
                        std::vector<float> buf;
                        if (extract_pred_row(frame, buf))
                            prediction_writer.add_frame(frame, buf.data());
                    }
                    annotations.erase(frame);
                };
#ifdef __APPLE__
                // Keep the live HybridNet center-camera count in sync with the
                // persisted user setting (predict clamps to [2, num_cameras]).
                jarvis_coreml_state.hn_center_cams = user_settings.jarvis_center_cams;
#endif

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
                    bp.batch_seek_ms = 0;
                    bp.batch_decode_ms = 0;
                    bp.batch_chunks = 0;
                    bp.batch_cancel_requested = false;
                    bp.batch_cancelled = false;
                    bp.batch_t0 = std::chrono::steady_clock::now();

                    // Store mode: open a fresh .rpred store to stream this
                    // batch's predictions into (kept out of the Labeling Tool).
                    bp.store_status.clear();
                    prediction_store.close();  // release any prior mmap
                    if (bp.batch_prediction_dest == PredDest::Store &&
                        !pm.project_path.empty() && skeleton.num_nodes > 0) {
                        std::error_code sec;
                        std::filesystem::path store_dir =
                            std::filesystem::path(pm.project_path) /
                            "predictions" / "red_store";
                        std::filesystem::create_directories(store_dir, sec);
                        std::time_t t = std::time(nullptr);
                        std::tm tmb{};
#ifdef _WIN32
                        localtime_s(&tmb, &t);
#else
                        localtime_r(&t, &tmb);
#endif
                        char ts[32];
                        std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", &tmb);
                        // A model-set pass points the writer at a private temp
                        // store it merges + deletes later; otherwise a fresh
                        // timestamped store per batch.
                        bp.store_path = !bp.store_path_override.empty()
                            ? bp.store_path_override
                            : (store_dir / (std::string("pred_") + ts + ".rpred"))
                                  .string();
                        if (!prediction_writer.open(bp.store_path,
                                                    skeleton.num_nodes)) {
                            // Can't write the store — fall back to loading
                            // predictions as labeled frames rather than silently
                            // dropping them (stash erases in Store mode).
                            bp.batch_prediction_dest = PredDest::LabelingTool;
                            bp.store_status =
                                "Could not open prediction store — predictions "
                                "will load as labeled frames instead";
                            bp.store_path.clear();
                        }
                    }
#ifdef __APPLE__
                    bp.batch_phase = bp.batch_streaming ? Phase::STREAM_SEEK : Phase::SEEK;
#else
                    bp.batch_phase = Phase::SEEK;   // streaming path is macOS-only
#endif
                    bp.batch_chunk_start = bp.batch_start;
                    ps.play_video = false;
                    printf("[Batch] Starting: frames %d-%d step %d (%d frames)\n",
                           bp.batch_start, bp.batch_end, bp.batch_step, bp.batch_total);
                }

                // --- Per-frame state machine tick ---
                if (bp.batch_running && jarvis_any_loaded && scene->num_cams > 0) {
                    // Cancel is routed here so cleanup (decoder stop) runs via
                    // FINISHING rather than being skipped by a bare running=false.
                    if (bp.batch_cancel_requested) {
                        bp.batch_cancel_requested = false;
                        bp.batch_cancelled = true;
                        bp.batch_phase = Phase::FINISHING;
                    }
                    switch (bp.batch_phase) {

                    case Phase::SEEK: {
                        // Seek to current chunk start. Blocking (~3-5s) but
                        // only happens once per 64-frame buffer fill. MUST be an
                        // ACCURATE seek: a keyframe-aligned (fast) seek lands the
                        // ring buffer on the nearest preceding keyframe, so slot 0
                        // would be an EARLIER frame than batch_chunk_start and every
                        // prediction would be stored against the wrong frame.
                        bp.chunk_seek_t0 = std::chrono::steady_clock::now();
                        seek_all_cameras(scene, bp.batch_chunk_start,
                                         dc_context->video_fps, ps, true);
                        bp.batch_seek_ms += std::chrono::duration<float, std::milli>(
                            std::chrono::steady_clock::now() - bp.chunk_seek_t0).count();
                        current_frame_num = bp.batch_chunk_start;
                        // Force all decoders to fill the buffer
                        for (auto &[key, value] : window_need_decoding)
                            value.store(true);

                        int frames_needed = std::min(buf_size,
                            bp.batch_end - bp.batch_chunk_start + 1);
                        bp.batch_chunk_last_slot = frames_needed - 1;
                        bp.batch_wait_frames = 0;
                        bp.chunk_wait_t0 = std::chrono::steady_clock::now();
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
#ifdef __APPLE__
                            if (slot.available_to_write || !slot.pixel_buffer) {
#else
                            if (slot.available_to_write) {
#endif
                                ready = false;
                                break;
                            }
                        }
                        if (ready) {
                            float decode_ms = std::chrono::duration<float, std::milli>(
                                std::chrono::steady_clock::now() - bp.chunk_wait_t0).count();
                            bp.batch_decode_ms += decode_ms;
                            bp.batch_chunks++;
                            printf("[Batch] Buffer filled (waited %d frames, %.0f ms decode)\n",
                                   bp.batch_wait_frames, decode_ms);
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
#if defined(__APPLE__) || defined(_WIN32) || defined(__linux__)
                        int nc_pred = (int)scene->num_cams;
                        std::vector<int> w_b(nc_pred), h_b(nc_pred);
                        for (int c = 0; c < nc_pred; ++c) {
                            w_b[c] = (int)scene->image_width[c];
                            h_b[c] = (int)scene->image_height[c];
                        }
#ifdef __APPLE__
                        std::vector<CVPixelBufferRef> pbs(nc_pred, nullptr);
#endif
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

                            // Guard against buffer/frame misalignment. The decoder
                            // stamps each slot with its true frame_number; if the
                            // expected slot doesn't hold `frame` (e.g. a seek landed
                            // the ring on a different frame), find the correct slot,
                            // and if `frame` isn't in this chunk at all, re-seek a
                            // fresh chunk starting here. Mirrors the macOS streaming
                            // path's frame_number match — without it, predictions get
                            // silently stored against the wrong (offset) frame.
                            if (scene->num_cams > 0 &&
                                scene->display_buffer[0][slot].frame_number.load() != (int)frame) {
                                int found = -1;
                                for (int s = 0; s < buf_size; ++s)
                                    if (scene->display_buffer[0][s].frame_number.load() == (int)frame) {
                                        found = s; break;
                                    }
                                if (found < 0) {
                                    bp.batch_chunk_start = bp.batch_current;
                                    bp.batch_phase = Phase::SEEK;
                                    for (auto &[key, value] : window_need_decoding)
                                        value.store(true);
                                    break;
                                }
                                slot = found;
                            }

                            // Skip frames with manual labels
                            bool has_manual = false;
                            if (annotations.count(frame)) {
                                const auto &fa = annotations.at(frame);
                                // Protect manual labels AND promoted "needs
                                // improvement" frames from being overwritten.
                                if (fa.needs_improvement) has_manual = true;
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
                                    skeleton, (int)scene->num_cams, pm.camera_params,
                                    bp.confidence_threshold);
                                if (!jarvis_coreml_state.hybridnet &&
                                    !pm.camera_params.empty())
                                    reprojection(annotations.at(frame),
                                                 &skeleton, pm.camera_params, scene);
                                auto tp1 = std::chrono::steady_clock::now();
                                bp.batch_predict_ms += std::chrono::duration<float, std::milli>(tp1 - tp0).count();
                                bp.batch_completed++;
                                printf("[Batch] Frame %u (slot %d): %.0f ms  [%d/%d]\n",
                                       frame, slot, jarvis_coreml_state.last_total_ms,
                                       bp.batch_completed, bp.batch_total);
#elif defined(_WIN32) || defined(__linux__)
                                {
#ifdef _WIN32
                                    const bool trt_loaded = jarvis_trt_state.loaded;
#else
                                    const bool trt_loaded = false;
#endif
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                                    const bool hn_loaded = jarvis_hn_state.loaded;
#else
                                    const bool hn_loaded = false;
#endif
                                if (trt_loaded || jarvis_state.loaded || hn_loaded) {
                                    auto tp0 = std::chrono::steady_clock::now();
                                    // Extract RGB from RGBA GPU frame buffers
                                    std::vector<const uint8_t *> rgb_bufs(nc_pred, nullptr);
                                    std::vector<std::vector<uint8_t>> rgb_storage(nc_pred);
                                    for (int c = 0; c < nc_pred; ++c) {
                                        auto &s = scene->display_buffer[c][slot];
                                        if (!s.frame) continue;
                                        int w = w_b[c], h = h_b[c];
                                        std::vector<uint8_t> rgba(w * h * 4);
                                        // cudaMemcpyDefault so this works whether
                                        // s.frame is device memory (GPU Buffer) or
                                        // host memory (CPU Buffer). Previously
                                        // hardcoded D2H would crash under CPU Buffer.
                                        cudaMemcpy(rgba.data(), s.frame, w * h * 4, cudaMemcpyDefault);
                                        rgb_storage[c].resize(w * h * 3);
                                        for (int i = 0; i < w * h; ++i) {
                                            rgb_storage[c][i*3+0] = rgba[i*4+0];
                                            rgb_storage[c][i*3+1] = rgba[i*4+1];
                                            rgb_storage[c][i*3+2] = rgba[i*4+2];
                                        }
                                        rgb_bufs[c] = rgb_storage[c].data();
                                    }
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                                    if (hn_loaded) {
                                        // HybridNet handles 3D + per-cam 2D internally;
                                        // skip the separate reprojection() call.
                                        try {
                                            jarvis_hybridnet_predict_frame(jarvis_hn_state,
                                                rgb_bufs, w_b, h_b, pm.camera_params,
                                                annotations, skeleton, frame);
                                        } catch (const std::exception &e) {
                                            fprintf(stderr, "[Batch HybridNet] frame %u: %s\n",
                                                    frame, e.what());
                                        } catch (...) {
                                            fprintf(stderr, "[Batch HybridNet] frame %u: non-std exception\n", frame);
                                        }
                                    } else
#endif
#ifdef _WIN32
                                    if (trt_loaded) {
                                        jarvis_tensorrt_predict_frame(jarvis_trt_state,
                                            annotations, frame, rgb_bufs, w_b, h_b,
                                            skeleton, nc_pred, bp.confidence_threshold);
                                        if (!pm.camera_params.empty())
                                            reprojection(annotations.at(frame),
                                                         &skeleton, pm.camera_params, scene);
                                    } else
#endif
                                    {
                                        jarvis_predict_frame(jarvis_state, annotations,
                                            frame, rgb_bufs, w_b, h_b,
                                            skeleton, pm.camera_params, scene,
                                            bp.confidence_threshold);
                                        if (!pm.camera_params.empty())
                                            reprojection(annotations.at(frame),
                                                         &skeleton, pm.camera_params, scene);
                                    }
                                    auto tp1 = std::chrono::steady_clock::now();
                                    bp.batch_predict_ms += std::chrono::duration<float, std::milli>(tp1 - tp0).count();
                                    bp.batch_completed++;
#ifdef _WIN32
                                    float last_ms = trt_loaded ?
                                        jarvis_trt_state.last_total_ms : jarvis_state.last_total_ms;
#else
                                    float last_ms = jarvis_state.last_total_ms;
#endif
                                    printf("[Batch] Frame %u (slot %d): %.0f ms  [%d/%d]\n",
                                           frame, slot, last_ms,
                                           bp.batch_completed, bp.batch_total);
                                }
                                }
#else
                                bp.batch_completed++;
                                printf("[Batch] Frame %u (slot %d)  [%d/%d]\n",
                                       frame, slot,
                                       bp.batch_completed, bp.batch_total);
#endif
                                // Store mode: move this frame's 3D into the
                                // prediction store, out of the Labeling Tool.
                                stash_prediction((u32)frame);
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

                    case Phase::STREAM_SEEK: {
                        // Streaming: seek ONCE to the batch start, then keep the
                        // decoders filling the ring continuously ahead of the
                        // predict cursor (no per-chunk re-seek, decode overlaps
                        // predict). seek_all_cameras puts batch_start in slot 0.
                        bp.chunk_seek_t0 = std::chrono::steady_clock::now();
                        seek_all_cameras(scene, bp.batch_start,
                                         dc_context->video_fps, ps, false);
                        bp.batch_seek_ms += std::chrono::duration<float, std::milli>(
                            std::chrono::steady_clock::now() - bp.chunk_seek_t0).count();
                        current_frame_num = bp.batch_start;
                        bp.stream_read_head = 0;
                        bp.batch_current = bp.batch_start;
                        bp.batch_chunks = 1;              // one seek for the whole batch
                        bp.batch_wait_frames = 0;
                        ps.pause_seeked = true;           // suppress visibility re-seek
                        for (auto &[key, value] : window_need_decoding)
                            value.store(true);            // decoders run for the whole batch
                        bp.batch_phase = Phase::STREAM_RUN;
                        printf("[Batch] Streaming from frame %d (single seek)...\n",
                               bp.batch_start);
                        break;
                    }

                    case Phase::STREAM_RUN: {
#ifdef __APPLE__
                        int nc_pred = (int)scene->num_cams;
                        std::vector<int> w_b(nc_pred), h_b(nc_pred);
                        for (int c = 0; c < nc_pred; ++c) {
                            w_b[c] = (int)scene->image_width[c];
                            h_b[c] = (int)scene->image_height[c];
                        }
                        std::vector<CVPixelBufferRef> pbs(nc_pred, nullptr);

                        // Consume a bounded number of frames per tick, then yield
                        // to the render loop (progress/cancel). Decoders keep
                        // filling in the background — they stay well ahead since
                        // decode (~26ms/frame) is far faster than predict.
                        const int kStreamYield = 20;
                        int processed = 0;
                        while (bp.batch_current <= bp.batch_end && bp.batch_running &&
                               processed < kStreamYield) {
                            int frame = bp.batch_current;
                            int slot = bp.stream_read_head;

                            // Stop cleanly at end of video (total_num_frame is
                            // INT_MAX until a decoder hits EOF, then the real
                            // count) — avoids the ~15s not-ready timeout below.
                            if (frame >= dc_context->total_num_frame) {
                                printf("[Batch] Reached end of video (%d frames) "
                                       "at frame %d\n", dc_context->total_num_frame, frame);
                                bp.batch_phase = Phase::FINISHING;
                                break;
                            }

                            // Wait until every camera has THIS frame in the slot.
                            // available_to_write is published last by the decoder,
                            // so false ⇒ pixel_buffer + frame_number are valid;
                            // match frame_number as the source of truth.
                            bool ready = true;
                            for (int c = 0; c < nc_pred; ++c) {
                                auto &sl = scene->display_buffer[c][slot];
                                if (sl.available_to_write.load() ||
                                    sl.frame_number.load() != frame ||
                                    !sl.pixel_buffer) { ready = false; break; }
                            }
                            if (!ready) {
                                // Decoder hasn't caught up (start-up, or EOF).
                                bp.batch_wait_frames++;
                                if (bp.batch_wait_frames > 900) {   // ~15s → EOF/stall
                                    printf("[Batch] Streaming stopped at frame %d "
                                           "(decoder idle / end of video)\n", frame);
                                    bp.batch_phase = Phase::FINISHING;
                                }
                                break;   // yield; decoders keep filling
                            }
                            bp.batch_wait_frames = 0;

                            // Predict step-matched frames; consume/release every
                            // slot regardless so the ring keeps streaming.
                            if (((frame - bp.batch_start) % bp.batch_step) == 0) {
                                bool has_manual = false;
                                if (annotations.count((u32)frame)) {
                                    const auto &fa = annotations.at((u32)frame);
                                    if (fa.needs_improvement) has_manual = true;
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
                                    auto tp0 = std::chrono::steady_clock::now();
                                    for (int c = 0; c < nc_pred; ++c)
                                        pbs[c] = scene->display_buffer[c][slot].pixel_buffer;
                                    jarvis_coreml_predict_frame(jarvis_coreml_state,
                                        annotations, (u32)frame, pbs, w_b, h_b,
                                        skeleton, (int)scene->num_cams, pm.camera_params,
                                        bp.confidence_threshold);
                                    if (!jarvis_coreml_state.hybridnet &&
                                        !pm.camera_params.empty())
                                        reprojection(annotations.at((u32)frame),
                                                     &skeleton, pm.camera_params, scene);
                                    bp.batch_predict_ms += std::chrono::duration<float, std::milli>(
                                        std::chrono::steady_clock::now() - tp0).count();
                                    bp.batch_completed++;
                                    printf("[Batch] Frame %d (slot %d): %.0f ms  [%d/%d]\n",
                                           frame, slot, jarvis_coreml_state.last_total_ms,
                                           bp.batch_completed, bp.batch_total);
                                    // Store mode: move 3D into the store, out of
                                    // the Labeling Tool.
                                    stash_prediction((u32)frame);
                                }
                            }

                            // Release the slot back to the decoder (mandatory —
                            // otherwise the ring fills and the decoder stalls).
                            for (int c = 0; c < nc_pred; ++c) {
                                auto &sl = scene->display_buffer[c][slot];
                                if (sl.pixel_buffer) {
                                    CFRelease(sl.pixel_buffer);
                                    sl.pixel_buffer = nullptr;
                                }
                                sl.available_to_write = true;
                            }
                            bp.stream_read_head = (bp.stream_read_head + 1) % buf_size;
                            bp.batch_current += 1;
                            current_frame_num = frame;
                            processed++;
                        }
                        if (bp.batch_current > bp.batch_end)
                            bp.batch_phase = Phase::FINISHING;
#else
                        // Streaming is a macOS (CVPixelBuffer) optimization; other
                        // platforms are never routed here (init keeps them chunked).
                        bp.batch_phase = Phase::FINISHING;
#endif
                        break;
                    }

                    case Phase::FINISHING: {
                        auto t1 = std::chrono::steady_clock::now();
                        float total_ms = std::chrono::duration<float, std::milli>(t1 - bp.batch_t0).count();
                        bp.batch_running = false;
                        bp.batch_phase = Phase::IDLE;
#ifdef __APPLE__
                        // macOS: decoders idle post-batch; the render loop re-enables
                        // the visible camera on the next interaction (streaming path).
                        for (auto &[key, value] : window_need_decoding)
                            value.store(false);
#else
                        // Linux/Windows (chunked path): RE-ENABLE decoding so the app
                        // returns to the normal interactive state. If left disabled,
                        // a subsequent seek (e.g. clicking a Keypoint Labels square)
                        // calls seek_all_cameras(), which spins the UI thread waiting
                        // on the decoders' seek-ack that never comes — a hard hang.
                        // Also lets spacebar playback resume. Decoders fill the ring
                        // then park (1ms sleep) when paused, so this is cheap.
                        for (auto &[key, value] : window_need_decoding)
                            value.store(true);
                        ps.pause_seeked = false;  // allow the render loop to resync
#endif
                        bp.batch_status = (bp.batch_cancelled ? "Cancelled: " : "Complete: ") +
                            std::to_string(bp.batch_completed) + " frames in " +
                            std::to_string((int)(total_ms / 1000.0f)) + "s (" +
                            std::to_string((int)(total_ms / std::max(1, bp.batch_completed))) +
                            " ms/frame)";
                        if (bp.batch_skipped > 0)
                            bp.batch_status += " (" + std::to_string(bp.batch_skipped) +
                                " skipped)";
                        bp.batch_cancelled = false;
                        printf("[Batch] %s\n", bp.batch_status.c_str());

                        // Store mode: finalize the .rpred store and mmap it for
                        // the read-only overlay + Pose Stats.
                        if (prediction_writer.is_open()) {
                            uint32_t total_v =
                                (dc_context->total_num_frame > 0 &&
                                 dc_context->total_num_frame != INT_MAX)
                                    ? (uint32_t)dc_context->total_num_frame
                                    : (uint32_t)(bp.batch_end + 1);
                            uint32_t stored = prediction_writer.stored_frames();
                            uint32_t fps = (uint32_t)std::lround(
                                dc_context->video_fps > 0 ? dc_context->video_fps : 0);
                            if (prediction_writer.finalize(total_v, fps) &&
                                prediction_store.open(bp.store_path)) {
                                bp.store_status = "Prediction store: " +
                                    std::to_string(stored) + " frames → " +
                                    std::filesystem::path(bp.store_path)
                                        .filename().string();
                                printf("[Store] %s\n", bp.store_status.c_str());
                                bp.active_store_path = bp.store_path;
                                bp.store_list_dirty = true;  // show in picker

                                // Provenance sidecar (<store>.json): which video
                                // / model / frame range these predictions are for.
                                std::string model_name;
                                if (pm.active_jarvis_model >= 0 &&
                                    pm.active_jarvis_model < (int)pm.jarvis_models.size())
                                    model_name = pm.jarvis_models[pm.active_jarvis_model].name;
                                nlohmann::json meta = {
                                    {"store_file", std::filesystem::path(bp.store_path)
                                                       .filename().string()},
                                    {"media_folder", pm.media_folder},
                                    {"model_name", model_name},
                                    {"skeleton", pm.skeleton_name},
                                    {"num_keypoints", (int)skeleton.num_nodes},
                                    {"frame_start", bp.batch_start},
                                    {"frame_end", bp.batch_end},
                                    {"n_stored", (int)stored},
                                    {"fps", (int)fps},
                                };
                                std::filesystem::path side = bp.store_path;
                                side.replace_extension(".json");
                                std::ofstream sf(side);
                                if (sf) sf << meta.dump(2) << "\n";
                            } else {
                                bp.store_status = "Failed to finalize prediction store";
                            }
                        }

                        // Optional: write the batch's whole [start, end] range to
                        // a JARVIS-CLI-compatible Predictions_3D_<ts>/ folder.
                        // Frames skipped by Step (or never predicted, e.g. after a
                        // cancel) are NaN rows, so row index == frame - start, as
                        // the CLI expects. In Store mode the frames were moved out
                        // of `annotations` into the store, so read them from there.
                        if (bp.export_predictions3D && !pm.project_path.empty()) {
                            int fstart = bp.batch_start;
                            int nframes = bp.batch_end - bp.batch_start + 1;
                            std::string dir =
                                jarvis_make_predictions_dir(pm.project_path);
                            JarvisExportResult er =
                                (bp.batch_prediction_dest == PredDest::Store &&
                                 prediction_store.is_open())
                                    ? jarvis_export_predictions3D_from_reader(
                                          dir, prediction_store, skeleton, pm,
                                          fstart, nframes)
                                    : jarvis_export_predictions3D(
                                          dir, annotations, skeleton, pm,
                                          fstart, nframes);
                            bp.export_status = er.message;
                            printf("[JARVIS export] %s\n", er.message.c_str());
                        }
                        // I/O breakdown: predict is the useful work; seek+decode
                        // is per-chunk overhead the 370ms/frame number hides.
                        float io_ms = bp.batch_seek_ms + bp.batch_decode_ms;
                        printf("[Batch] Wall %.1fs = predict %.1fs + seek %.1fs + "
                               "decode %.1fs (+%.1fs other) | %d chunks, I/O overhead %.0f%%\n",
                               total_ms / 1000.f, bp.batch_predict_ms / 1000.f,
                               bp.batch_seek_ms / 1000.f, bp.batch_decode_ms / 1000.f,
                               (total_ms - bp.batch_predict_ms - io_ms) / 1000.f,
                               bp.batch_chunks, 100.f * io_ms / std::max(1.f, total_ms));
                        break;
                    }

                    case Phase::IDLE:
                        break;
                    }
                }
            }

            // --- Multi-model "set" batch controller ---
            // Runs the active model set as a sequence of single-model batches
            // (each into a private temp store), then merges those stores into
            // one combined store and switches the project to a concatenated
            // skeleton so the merged result displays. Sits on top of the batch
            // state machine above, which it drives one pass at a time.
            {
                auto &jp = win.jarvis_predict;
                auto &bp = win.jarvis_predict;   // same struct; alias for clarity
                using SetPhase = JarvisPredictState::SetPhase;
                using PredDest = JarvisPredictState::PredDest;

                auto any_loaded = [&]() -> bool {
                    bool a = jarvis_state.loaded;
#ifdef __APPLE__
                    a = a || jarvis_coreml_state.loaded;
#elif defined(_WIN32)
                    a = a || jarvis_trt_state.loaded;
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                    a = a || jarvis_hn_state.loaded;
#endif
#endif
                    return a;
                };
                auto load_model = [&](int model_index) -> bool {
                    if (model_index < 0 ||
                        model_index >= (int)pm.jarvis_models.size())
                        return false;
                    std::string base = pm.project_path + "/" +
                        pm.jarvis_models[model_index].relative_path;
                    jarvis_load_from_dir(base, jarvis_state
#ifdef __APPLE__
                        , jarvis_coreml_state
#elif defined(_WIN32)
                        , jarvis_trt_state
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                        , jarvis_hn_state
#endif
#endif
                    );
                    return any_loaded();
                };
                auto store_dir = [&]() {
                    return std::filesystem::path(pm.project_path) /
                           "predictions" / "red_store";
                };
                auto set_abort = [&](const std::string &msg) {
                    jp.set_status = msg;
                    printf("[Set] %s\n", msg.c_str());
                    // Best-effort cleanup of any temp stores written so far.
                    std::error_code ec;
                    for (auto &p : jp.set_passes) {
                        if (p.store_path.empty()) continue;
                        std::filesystem::remove(p.store_path, ec);
                        std::filesystem::path side = p.store_path;
                        side.replace_extension(".json");
                        std::filesystem::remove(side, ec);
                    }
                    jp.export_predictions3D = jp.set_saved_export;
                    jp.store_path_override.clear();
                    jp.set_passes.clear();
                    jp.set_running = false;
                    jp.set_phase = SetPhase::IDLE;
                };

                // --- Start a set run: validate + build the pass list ---
                if (jp.set_requested && !jp.set_running && !bp.batch_running) {
                    jp.set_requested = false;
                    if (pm.active_jarvis_model_set < 0 ||
                        pm.active_jarvis_model_set >=
                            (int)pm.jarvis_model_sets.size()) {
                        jp.set_status = "No model set selected.";
                    } else if (pm.jarvis_model_sets[pm.active_jarvis_model_set]
                                   .members.empty()) {
                        jp.set_status = "Selected model set has no members.";
                    } else if (pm.project_path.empty() || !ps.video_loaded) {
                        jp.set_status = "Open a project and load video first.";
                    } else if (project_has_any_manual_labels(annotations)) {
                        jp.set_status =
                            "Project has manual labels — a model set switches "
                            "skeletons between passes, which would re-index them. "
                            "Clear/export manual labels first.";
                    } else {
                        const auto &set =
                            pm.jarvis_model_sets[pm.active_jarvis_model_set];
                        jp.set_passes.clear();
                        std::string err;
                        bool ok = true;
                        for (const auto &m : set.members) {
                            if (m.model_index < 0 ||
                                m.model_index >= (int)pm.jarvis_models.size()) {
                                err = "member references an unknown model";
                                ok = false; break;
                            }
                            SkeletonContext sk;
                            if (!resolve_member_skeleton(m, skeleton_map, sk, &err)) {
                                ok = false; break;
                            }
                            JarvisPredictState::SetPass pass;
                            pass.model_index = m.model_index;
                            pass.skel_json = m.skeleton_from_json;
                            pass.skel_file = m.skeleton_file;
                            pass.skel_name = m.skeleton_name;
                            pass.num_kp = sk.num_nodes;
                            jp.set_passes.push_back(std::move(pass));
                        }
                        if (!ok) {
                            jp.set_status = "Cannot start set: " + err;
                            jp.set_passes.clear();
                        } else {
                            jp.set_name = set.name;
                            jp.set_pass_idx = 0;
                            jp.set_batch_start = bp.batch_start;
                            jp.set_batch_end = bp.batch_end;
                            jp.set_batch_step = bp.batch_step;
                            jp.set_saved_export = bp.export_predictions3D;
                            jp.set_cancel_requested = false;
                            jp.set_running = true;
                            jp.set_phase = SetPhase::START_PASS;
                            jp.set_status = "Running set '" + set.name + "' (" +
                                std::to_string(jp.set_passes.size()) + " models)...";
                            printf("[Set] %s\n", jp.set_status.c_str());
                        }
                    }
                }

                if (jp.set_running) {
                    switch (jp.set_phase) {
                    case SetPhase::START_PASS: {
                        int idx = jp.set_pass_idx;
                        auto &pass = jp.set_passes[idx];
                        // Switch to this model's skeleton (clears annotations +
                        // closes any open store — safe, no manual labels).
                        std::string err;
                        if (!switch_project_skeleton(
                                ctx, prediction_store, jp, win.bout_filter,
                                pass.skel_json, pass.skel_file, pass.skel_name,
                                &err)) {
                            set_abort("Set failed switching skeleton: " + err);
                            break;
                        }
                        if (!load_model(pass.model_index)) {
                            set_abort("Set failed loading model '" +
                                pm.jarvis_models[pass.model_index].name + "'.");
                            break;
                        }
                        // Configure the underlying batch for this pass.
                        std::error_code ec;
                        std::filesystem::create_directories(store_dir(), ec);
                        std::string tmp = (store_dir() /
                            ("pred_set_tmp_" + std::to_string(idx) + ".rpred"))
                            .string();
                        pass.store_path = tmp;
                        bp.store_path_override = tmp;
                        bp.batch_prediction_dest = PredDest::Store;
                        bp.export_predictions3D = false;  // export the merged store, not each pass
                        bp.batch_start = jp.set_batch_start;
                        bp.batch_end = jp.set_batch_end;
                        bp.batch_step = jp.set_batch_step;
                        bp.batch_requested = true;
                        jp.set_pass_seen_running = false;
                        jp.set_phase = SetPhase::WAIT_PASS;
                        printf("[Set] Pass %d/%d: model '%s' (%d kp)\n",
                               idx + 1, (int)jp.set_passes.size(),
                               pm.jarvis_models[pass.model_index].name.c_str(),
                               pass.num_kp);
                        break;
                    }
                    case SetPhase::WAIT_PASS: {
                        if (bp.batch_running) {
                            jp.set_pass_seen_running = true;
                            break;
                        }
                        if (!jp.set_pass_seen_running) break;  // not started yet
                        // Pass finished. store_path_override already recorded in
                        // pass.store_path during START_PASS.
                        bp.store_path_override.clear();
                        if (jp.set_cancel_requested) {
                            set_abort("Set cancelled.");
                            break;
                        }
                        jp.set_pass_idx++;
                        jp.set_phase = (jp.set_pass_idx < (int)jp.set_passes.size())
                            ? SetPhase::START_PASS : SetPhase::MERGE;
                        break;
                    }
                    case SetPhase::MERGE: {
                        // Build the concatenated skeleton from the members.
                        std::vector<SkeletonContext> parts;
                        std::string err;
                        bool ok = true;
                        for (auto &pass : jp.set_passes) {
                            ProjectManager::JarvisModelSetMember m;
                            m.model_index = pass.model_index;
                            m.skeleton_from_json = pass.skel_json;
                            m.skeleton_file = pass.skel_file;
                            m.skeleton_name = pass.skel_name;
                            SkeletonContext sk;
                            if (!resolve_member_skeleton(m, skeleton_map, sk, &err)) {
                                ok = false; break;
                            }
                            parts.push_back(std::move(sk));
                        }
                        if (!ok) { set_abort("Set merge failed: " + err); break; }

                        SkeletonContext combined;
                        std::string comb_name = jp.set_name.empty()
                            ? "combined" : (jp.set_name + "_combined");
                        build_combined_skeleton(parts, comb_name, combined);

                        std::error_code ec;
                        std::filesystem::path skel_dir =
                            std::filesystem::path(pm.project_path) / "skeletons";
                        std::filesystem::create_directories(skel_dir, ec);
                        std::string skel_path =
                            (skel_dir / (comb_name + ".json")).string();
                        if (!write_skeleton_json(combined, skel_path, &err)) {
                            set_abort("Set merge failed: " + err); break;
                        }
                        // Switch the project to the combined skeleton (closes the
                        // per-pass store, clears annotations, persists .redproj).
                        if (!switch_project_skeleton(
                                ctx, prediction_store, jp, win.bout_filter,
                                /*json*/ true, skel_path, comb_name, &err)) {
                            set_abort("Set merge failed switching to combined "
                                      "skeleton: " + err);
                            break;
                        }

                        // Merge the per-pass temp stores into a timestamped store.
                        std::vector<std::string> inputs;
                        for (auto &p : jp.set_passes)
                            if (!p.store_path.empty()) inputs.push_back(p.store_path);
                        std::time_t t = std::time(nullptr);
                        std::tm tmb{};
#ifdef _WIN32
                        localtime_s(&tmb, &t);
#else
                        localtime_r(&t, &tmb);
#endif
                        char ts[32];
                        std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", &tmb);
                        std::string final_path =
                            (store_dir() / (std::string("pred_") + ts + ".rpred"))
                            .string();
                        if (!predstore::merge_concat(inputs, final_path, &err)) {
                            set_abort("Set merge failed: " + err); break;
                        }
                        // Delete the temp per-pass stores + sidecars.
                        for (auto &p : jp.set_passes) {
                            if (p.store_path.empty()) continue;
                            std::filesystem::remove(p.store_path, ec);
                            std::filesystem::path side = p.store_path;
                            side.replace_extension(".json");
                            std::filesystem::remove(side, ec);
                        }
                        // Open the combined store as active.
                        if (!prediction_store.open(final_path)) {
                            set_abort("Set merge wrote store but could not open "
                                      "it: " + final_path);
                            break;
                        }
                        jp.active_store_path = final_path;
                        jp.store_path = final_path;
                        uint32_t stored = prediction_store.stored_frames();
                        // Provenance sidecar (model_name = set name).
                        nlohmann::json meta = {
                            {"store_file",
                             std::filesystem::path(final_path).filename().string()},
                            {"media_folder", pm.media_folder},
                            {"model_name", jp.set_name},
                            {"skeleton", comb_name},
                            {"num_keypoints", (int)combined.num_nodes},
                            {"frame_start", jp.set_batch_start},
                            {"frame_end", jp.set_batch_end},
                            {"n_stored", (int)stored},
                            {"fps", (int)prediction_store.fps()},
                            {"model_set", true},
                        };
                        std::filesystem::path side = final_path;
                        side.replace_extension(".json");
                        std::ofstream sf(side);
                        if (sf) sf << meta.dump(2) << "\n";

                        jp.export_predictions3D = jp.set_saved_export;
                        jp.store_path_override.clear();
                        jp.store_list_dirty = true;
                        jp.set_status = "Model set '" + jp.set_name +
                            "' complete: " + std::to_string(stored) +
                            " frames, " + std::to_string(combined.num_nodes) +
                            " keypoints → " +
                            std::filesystem::path(final_path).filename().string();
                        printf("[Set] %s\n", jp.set_status.c_str());
                        jp.set_phase = SetPhase::DONE;
                        break;
                    }
                    case SetPhase::DONE: {
                        jp.set_passes.clear();
                        jp.set_running = false;
                        jp.set_phase = SetPhase::IDLE;
                        break;
                    }
                    case SetPhase::IDLE:
                        break;
                    }
                }
            }

            if (keys::pressed(keys::Sc::SeekBack) &&
                !win.jarvis_predict.batch_running) {
                seek_relative(ImGui::GetIO().KeyShift ? -10 : -1);
            }

            if (keys::pressed(keys::Sc::SeekFwd) &&
                !win.jarvis_predict.batch_running) {
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
                bool at_bout_end = ps.loop_bout && frame_to_show > ps.loop_end;
                if (at_bout_end && win.bout_filter.loop_bout_enabled) {
                    // Reached the end of the looped bout: wrap to its start.
                    // seek_all_cameras clears loop_bout, so re-arm afterward
                    // (loop_start/loop_end are untouched by the seek).
                    seek_all_cameras(scene, ps.loop_start,
                                     dc_context->video_fps, ps, true);
                    ps.loop_bout = true;
                    current_frame_num = ps.loop_start;
                    for (auto &[key, value] : window_need_decoding)
                        value.store(true);
                } else {
                if (at_bout_end) {
                    // Looping is disabled: stop at the bout's end frame
                    // instead of playing past it. Clear loop_bout so a
                    // subsequent manual "play" isn't clamped here again.
                    frame_to_show = ps.loop_end;
                    ps.play_video = false;
                    ps.loop_bout = false;
                }
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
