#include "IconsForkAwesome.h"
#include "camera.h"
#include "filesystem"
#include "global.h"
#include "gui.h"
#include "gui/help_window.h"
#include "gui/skeleton_creator_window.h"
#include "gui/yolo_export_window.h"
#include "gui/jarvis_export_window.h"
#include "gui/annotation_dialog.h"
#include "gui/calibration_tool_window.h"
#include "gui/popup_stack.h"
#include "gui/toast.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#ifdef __APPLE__
#include "metal_context.h"
#include <CoreFoundation/CoreFoundation.h>  // CFRelease for CVPixelBuffer
#include <mach-o/dyld.h>                   // _NSGetExecutablePath
#else
#include "imgui_impl_opengl3.h"
#endif
#include "implot.h"
#include "live_table.h"
#include "project.h"
#include "render.h"
#include "reprojection_tool.h"
#include "skeleton.h"
#include "utils.h"
#include "calibration_tool.h"
#include "calibration_pipeline.h"
#include "deferred_queue.h"
#include "user_settings.h"
#include "jarvis_export.h"
#include "laser_calibration.h"
#include "aruco_metal.h"
#include "laser_metal.h"
#include "yolo_export.h"
#include "yolo_torch.h"
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
#include "reprojection_tool.h"

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
    std::string yolo_model_dir = red_data_dir + "/yolo_model";
    std::vector<std::thread> decoder_threads;
    std::vector<FFmpegDemuxer *> demuxers;

    DecoderContext *dc_context =
        (DecoderContext *)malloc(sizeof(DecoderContext));
    *dc_context = (DecoderContext){.decoding_flag = false,
                                   .stop_flag = false,
                                   .total_num_frame = int(INT_MAX),
                                   .estimated_num_frames = 0,
                                   .gpu_index = 0,
                                   .seek_interval = 250,
                                   .video_fps = 60.0f};

    // gui states, todo: bundle this later
    std::time_t last_saved = static_cast<std::time_t>(-1);
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

    // Skeleton Creator state
    SkeletonCreatorState skeleton_creator_state;
    std::string skeleton_file_path = ""; // Track currently loaded skeleton file

    // YOLO Export Tool state
    YoloExportState yolo_export_state;
    yolo_export_state.label_dir = media_root_dir + "/labeled_data";
    yolo_export_state.video_dir = media_root_dir;
    yolo_export_state.output_dir = media_root_dir + "/export";

    // JARVIS Export Tool state
    JarvisExportState jarvis_export_state;

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
    annot_state.video_folder = user_settings.default_media_root_path.empty()
                                   ? media_root_dir
                                   : user_settings.default_media_root_path;

    // Bounding box class management
    std::vector<std::string> bbox_class_names = {"Class_1"};
    std::vector<ImVec4> bbox_class_colors = {ImVec4(0.3f, 1.0f, 1.0f, 1.0f)};
    int current_bbox_class = 0;
    int current_bbox_id =
        0; // Track the currently selected bbox ID within the class
    bool show_bbox_ids = false; // Toggle for displaying bbox IDs on frame
    std::string new_class_name = "";

    // Helper function to create a new bbox class
    auto create_new_bbox_class = [&]() {
        std::string new_class_name =
            "Class_" + std::to_string(bbox_class_names.size() + 1);
        bbox_class_names.push_back(new_class_name);
        // Generate a unique color for the new class (HSV with different hues)
        float hue = (bbox_class_colors.size() *
                     0.618034f); // Golden ratio for nice color distribution
        while (hue > 1.0f)
            hue -= 1.0f;
        ImVec4 new_color = (ImVec4)ImColor::HSV(hue, 0.8f, 1.0f);
        bbox_class_colors.push_back(new_color);
        current_bbox_class = bbox_class_names.size() - 1;
    };

    // Helper function to cleanup YOLO drag boxes
    auto cleanup_yolo_drag_boxes = [&]() {
        for (int cam_id = 0; cam_id < MAX_VIEWS; cam_id++) {
            for (auto &drag_box : yolo_drag_boxes[cam_id]) {
                if (drag_box.rect) {
                    delete drag_box.rect;
                    drag_box.rect = nullptr;
                }
            }
            yolo_drag_boxes[cam_id].clear();
            yolo_active_bbox_idx[cam_id] = -1;
        }
    };

    colors[ImPlotCol_Crosshairs] = ImVec4(0.3f, 0.10f, 0.64f, 1.00f);

    bool yolo_detection = false;
    int label_buffer_size = 64;
    bool show_help_window = false;
    std::vector<bool> is_view_focused;
    bool input_is_imgs = false;
    PopupStack popups;
    ToastQueue toasts;
    DeferredQueue deferred;

    int hovered_bbox_cam = -1;
    int hovered_bbox_idx = -1;
    float hovered_bbox_confidence = 0.0f;
    int hovered_bbox_class = -1;
    int hovered_bbox_id = -1; // Track the ID of the hovered bbox

    // Hovered OBB tracking variables
    int hovered_obb_cam = -1;
    int hovered_obb_idx = -1;
    float hovered_obb_confidence = 0.0f;
    int hovered_obb_class = -1;
    int hovered_obb_id = -1; // Track the ID of the hovered OBB

    bool auto_yolo_labeling = false;
    std::set<int>
        yolo_processed_frames; // Track which frames have been processed
    std::unordered_map<std::string, bool> window_was_decoding;
    std::unordered_map<std::string, bool> window_is_visible;  // actual ImGui visibility (prev frame)
    double inst_speed = 1.0;
    float set_playback_speed = 1.0f;
    PlaybackState ps;
    LiveTable table;

    int brightness = 0;
    float contrast = 1.0f;     // neutral contrastst
    bool pivot_midgray = true; // typical contrast feel

    // variables for project management
    ProjectManager pm = ProjectManager();
    pm.project_root_path = user_settings.default_project_root_path.empty()
                               ? red_data_dir
                               : user_settings.default_project_root_path;
    pm.media_folder = user_settings.default_media_root_path.empty()
                          ? media_root_dir
                          : user_settings.default_media_root_path;

    // Copy the shipped default_imgui_layout.ini into a project folder so new
    // projects (annotation or calibration) start with the standard dock layout.
    // If the project already has an imgui_layout.ini, this is a no-op.
    auto copy_default_layout_to_project = [&](const std::string &proj_path) {
        namespace fs = std::filesystem;
        fs::path dest = fs::path(proj_path) / "imgui_layout.ini";
        if (fs::exists(dest))
            return; // preserve existing layout
        for (const auto &candidate : {
                 window->exe_dir + "/../default_imgui_layout.ini",           // dev
                 window->exe_dir + "/../share/red/default_imgui_layout.ini", // brew
             }) {
            if (fs::exists(candidate)) {
                std::error_code ec;
                fs::copy_file(candidate, dest, ec);
                break;
            }
        }
    };

    // Switch ImGui layout ini to the project folder so each project keeps its own layout.
    // Before the main loop (CLI path): just redirect io.IniFilename and let NewFrame()
    // auto-load it. Mid-session (File menu): also explicitly load the project ini.
    bool main_loop_running = false;
    // Migrate renamed windows in project ini files so saved dock settings
    // carry over. Replaces old window name with new name. If the new name
    // already exists, removes the old section entirely to avoid duplicates.
    auto migrate_ini_window_names = [](const std::string &ini_path) {
        if (!std::filesystem::exists(ini_path)) return;
        std::ifstream in(ini_path);
        std::string content((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
        in.close();

        // "File Browser" → "Navigator" (renamed 2026-03-07)
        const std::string old_header = "[Window][File Browser]";
        const std::string new_header = "[Window][Navigator]";
        size_t old_pos = content.find(old_header);
        if (old_pos == std::string::npos) return; // nothing to migrate

        // Find the end of the old section (next "[" or end of file)
        size_t section_end = content.find("\n[", old_pos + 1);
        if (section_end == std::string::npos)
            section_end = content.size();
        else
            section_end += 1; // include the newline before next section

        if (content.find(new_header) != std::string::npos) {
            // New name already exists — remove old section entirely
            content.erase(old_pos, section_end - old_pos);
        } else {
            // No new name yet — rename the old section
            content.replace(old_pos, old_header.size(), new_header);
        }
        std::ofstream out(ini_path);
        out << content;
    };

    auto switch_ini_to_project = [&]() {
        project_ini_path = pm.project_path + "/imgui_layout.ini";
        // Seed new projects with the default dock layout
        copy_default_layout_to_project(pm.project_path);
        // Migrate old window names before loading
        migrate_ini_window_names(project_ini_path);
        io.IniFilename = project_ini_path.c_str();
        if (main_loop_running && std::filesystem::exists(project_ini_path)) {
            ImGui::LoadIniSettingsFromDisk(project_ini_path.c_str());
            // LoadIniSettingsFromDisk clears DockId on live windows during its
            // internal rebuild. Restore DockId/DockOrder from the loaded settings
            // so windows can re-dock into the newly created dock nodes.
            ImGuiContext* ctx = ImGui::GetCurrentContext();
            for (int i = 0; i < ctx->Windows.Size; i++) {
                ImGuiWindow* w = ctx->Windows[i];
                if (ImGuiWindowSettings* s = ImGui::FindWindowSettingsByWindow(w)) {
                    w->DockId = s->DockId;
                    w->DockOrder = s->DockOrder;
                }
            }
            // Mark all dock nodes alive so BeginDocked() doesn't undock windows
            // thinking the nodes have expired.
            for (int i = 0; i < ctx->DockContext.Nodes.Data.Size; i++) {
                ImGuiDockNode* node = (ImGuiDockNode*)ctx->DockContext.Nodes.Data[i].val_p;
                if (node)
                    node->LastFrameAlive = ctx->FrameCount;
            }
        }
    };

    // Consolidates the post-project-load sequence used by CLI load,
    // File > Open Project, and Annotate > Load Project.
    auto on_project_loaded = [&]() {
        switch_ini_to_project();
        std::map<std::string, std::string> empty_selected_files;
        load_videos(empty_selected_files, ps, pm,
                    window_was_decoding, demuxers, dc_context,
                    scene, label_buffer_size, decoder_threads,
                    is_view_focused);
        print_video_metadata(demuxers, pm.camera_names,
                             dc_context->seek_interval);
        std::string label_err;
        std::string most_recent_folder;
        if (!find_most_recent_labels(pm.keypoints_root_folder,
                                     most_recent_folder, label_err)) {
            if (load_keypoints(most_recent_folder, keypoints_map,
                               &skeleton, scene, pm.camera_names,
                               label_err, bbox_class_names)) {
                free_all_keypoints(keypoints_map, scene);
                popups.pushError(label_err);
            }
        }
        print_project_summary(pm, pm.skeleton_name, most_recent_folder);
    };

    ReprojectionTool rp_tool;

#ifdef __APPLE__
    // Per-camera last-uploaded frame number for Metal (skip redundant uploads)
    std::vector<int> mac_last_uploaded_frame(MAX_VIEWS, -1);
#endif

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::filesystem::path path = argv[i];
            // If a directory is passed, look for a sibling .redproj file
            if (std::filesystem::is_directory(path))
                path = path.parent_path() / (path.filename().string() + ".redproj");
            ProjectManager loaded;
            std::string err;
            if (!load_project_manager_json(&loaded, path, &err)) {
                popups.pushError(err);
            } else {
                pm = loaded;
                if (setup_project(pm, skeleton, skeleton_map, &err))
                    on_project_loaded();
                else
                    popups.pushError(err);
            }
        }
    }

    // Calibration Tool callbacks (bridge between extracted UI and main() locals)
    CalibrationToolCallbacks calib_cb;
    calib_cb.load_images = [&](std::map<std::string, std::string> &files) {
        load_images(files, ps, pm, imgs_names, scene, dc_context,
                    label_buffer_size, decoder_threads, is_view_focused,
                    window_was_decoding);
        input_is_imgs = true;
    };
    calib_cb.load_videos = [&]() {
        std::map<std::string, std::string> empty_selected_files;
        load_videos(empty_selected_files, ps, pm,
                    window_was_decoding, demuxers, dc_context,
                    scene, label_buffer_size, decoder_threads,
                    is_view_focused);
        input_is_imgs = false;
    };
    calib_cb.unload_media = [&]() {
        unload_media(ps, pm, demuxers, dc_context,
                     scene, decoder_threads,
                     is_view_focused, window_was_decoding);
    };
    calib_cb.copy_default_layout = [&](const std::string &proj_path) {
        copy_default_layout_to_project(proj_path);
    };
    calib_cb.switch_ini = [&](const std::string &proj_path) {
        project_ini_path = proj_path + "/imgui_layout.ini";
        io.IniFilename = project_ini_path.c_str();
        if (std::filesystem::exists(project_ini_path))
            ImGui::LoadIniSettingsFromDisk(project_ini_path.c_str());
    };
    calib_cb.print_metadata = [&]() {
        print_video_metadata(demuxers, pm.camera_names, dc_context->seek_interval);
    };
    calib_cb.deferred = &deferred;

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

        // Reset hovered bbox info at start of each frame
        hovered_bbox_cam = -1;
        hovered_bbox_idx = -1;
        hovered_bbox_confidence = 0.0f;
        hovered_bbox_class = -1;
        hovered_bbox_id = -1;

        // Reset hovered OBB info at start of each frame
        hovered_obb_cam = -1;
        hovered_obb_idx = -1;
        hovered_obb_confidence = 0.0f;
        hovered_obb_class = -1;
        hovered_obb_id = -1;

        // --- Update playback time ---
        auto now = std::chrono::steady_clock::now();

        if (ps.play_video) {
            ps.accumulated_play_time +=
                std::chrono::duration<double>(now - ps.last_play_time_start)
                    .count() *
                set_playback_speed;
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
                    if (ImGui::MenuItem("Skeleton Creator")) {
                        skeleton_creator_state.show = true;
                    }
                    if (ImGui::MenuItem("YOLO Export Tool")) {
                        yolo_export_state.show = true;
                    }
                    if (ImGui::MenuItem("JARVIS Export Tool")) {
                        jarvis_export_state.show = true;
                    }
                    if (ImGui::MenuItem("Spreadsheet")) {
                        table.is_open = true;
                    }
                    if (ImGui::MenuItem("Reprojection Error")) {
                        rp_tool.show_reprojection_error = true;
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Settings")) {
                    if (ImGui::MenuItem("Default Project Root")) {
                        IGFD::FileDialogConfig cfg;
                        cfg.countSelectionMax = 1;
                        cfg.path =
                            user_settings.default_project_root_path.empty()
                                ? red_data_dir
                                : user_settings.default_project_root_path;
                        cfg.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseDefaultProjectRoot",
                            "Set Default Project Root", nullptr, cfg);
                    }
                    if (ImGui::MenuItem("Default Media Root")) {
                        IGFD::FileDialogConfig cfg;
                        cfg.countSelectionMax = 1;
                        cfg.path =
                            user_settings.default_media_root_path.empty()
                                ? media_root_dir
                                : user_settings.default_media_root_path;
                        cfg.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseDefaultMediaRoot",
                            "Set Default Media Root", nullptr, cfg);
                    }
                    ImGui::EndMenu();
                }

                ImGui::EndMenuBar();
            }
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                        1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);

            if (!ps.video_loaded) {
                {
                    const char *items[] = {"CPU Buffer", "GPU Buffer"};
                    static int item_current = 0;
                    ImGui::Combo("Buffer Type", &item_current, items,
                                 IM_ARRAYSIZE(items));
                    if (item_current == 0) {
                        scene->use_cpu_buffer = true;
                    } else {
                        scene->use_cpu_buffer = false;
                    }
                }

                ImGui::InputInt("Buffer Size", &label_buffer_size);
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
                    inst_speed =
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
                    int denom = (int)roundf(1.0f / set_playback_speed);
                    if (denom <= 1)
                        snprintf(speed_label, sizeof(speed_label), "1x");
                    else
                        snprintf(speed_label, sizeof(speed_label), "1/%dx", denom);
                    ImGui::SliderFloat("##set_playback_speed", &set_playback_speed,
                                       1.0f / 16.0f, 1.0f, speed_label,
                                       ImGuiSliderFlags_Logarithmic);

                    // Row: Current speed readout
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Current Speed");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.2fx", inst_speed);

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
                ImGui::BeginDisabled(ps.play_video); // Disable if playing video

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
                    ImGui::SliderFloat("##contrast", &contrast, 0.0f, 3.0f,
                                       "%.2f");

                    // Brightness
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Brightness (beta)");
                    ImGui::SameLine();
                    HelpMarker("Shift pixel values. 0 = neutral.");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SliderInt("##brightness", &brightness, -150, 150);

                    // Reset row
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Display Preset");
                    ImGui::TableSetColumnIndex(1);
                    if (ImGui::Button("Reset##display")) {
                        contrast = 1.0f;
                        brightness = 0;
                        pivot_midgray = true;
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(restores neutral)");

                    ImGui::EndTable();
                }
                ImGui::EndDisabled();
            }
        }
        ImGui::End();

        DrawProjectWindow(pm, skeleton_map, skeleton, skeleton_dir, popups);

        // ===== Create Annotation Project dialog =====
        {
            std::string default_browse = user_settings.default_media_root_path.empty()
                                             ? media_root_dir
                                             : user_settings.default_media_root_path;
            DrawAnnotationDialog(annot_state, pm, skeleton_map, skeleton_dir,
                                 default_browse,
                                 [&](ProjectManager &pm_ref, std::string &err) -> bool {
                if (!ensure_dir_exists(pm_ref.project_path, &err))
                    return false;
                if (!setup_project(pm_ref, skeleton, skeleton_map, &err))
                    return false;
                std::filesystem::path redproj_path =
                    std::filesystem::path(pm_ref.project_path) / (pm_ref.project_name + ".redproj");
                if (!save_project_manager_json(pm_ref, redproj_path, &err))
                    return false;
                on_project_loaded();
                return true;
            });
        }


        if (ImGuiFileDialog::Instance()->Display("ChooseProjectDir", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                pm.project_root_path =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseCalibration", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                pm.calibration_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display(
                "ChooseDefaultProjectRoot", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string chosen =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                user_settings.default_project_root_path = chosen;
                // Live-update open dialogs that use this default
                calib_state.project.project_root_path = chosen;
                pm.project_root_path = chosen;
                save_user_settings(user_settings);
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display(
                "ChooseDefaultMediaRoot", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string chosen =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                std::string old_media_root =
                    user_settings.default_media_root_path;
                user_settings.default_media_root_path = chosen;
                pm.media_folder = chosen;
                annot_state.video_folder = chosen;
                if (calib_state.project.config_file.empty() ||
                    calib_state.project.config_file == old_media_root)
                    calib_state.project.config_file = chosen;
                save_user_settings(user_settings);
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseMedia", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto selected_files =
                    ImGuiFileDialog::Instance()->GetSelection();
                pm.media_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
                pm.project_name =
                    dir_difference(pm.media_folder, media_root_dir);
                pm.media_folder = pm.media_folder;
                load_videos(selected_files, ps, pm, window_was_decoding,
                            demuxers, dc_context, scene, label_buffer_size,
                            decoder_threads, is_view_focused);
                print_video_metadata(demuxers, pm.camera_names, dc_context->seek_interval);
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseImages", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto selected_files =
                    ImGuiFileDialog::Instance()->GetSelection();
                pm.media_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
                pm.project_name =
                    dir_difference(pm.media_folder, media_root_dir);
                pm.media_folder = pm.media_folder;
                load_images(selected_files, ps, pm, imgs_names, scene,
                            dc_context, label_buffer_size, decoder_threads,
                            is_view_focused, window_was_decoding);
                input_is_imgs = true;
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseProject", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                const auto sel = ImGuiFileDialog::Instance()->GetSelection();
                // Choose the picked file (single-select assumed)
                std::filesystem::path cfg_path;
                if (!sel.empty()) {
                    cfg_path =
                        std::filesystem::path(sel.begin()->second); // full path
                } else {
                    std::string full =
                        ImGuiFileDialog::Instance()->GetFilePathName(
                            IGFD_ResultMode_KeepInputFile);
                    if (!full.empty())
                        cfg_path = std::filesystem::path(full);
                    else
                        cfg_path = std::filesystem::path(
                            ImGuiFileDialog::Instance()->GetCurrentPath());
                }
                ProjectManager loaded;
                std::string err;
                if (!load_project_manager_json(&loaded, cfg_path, &err)) {
                    popups.pushError(err);
                } else {
                    pm = loaded;
                    if (setup_project(pm, skeleton, skeleton_map, &err))
                        on_project_loaded();
                    else
                        popups.pushError(err);
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseSkeleton", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                pm.skeleton_file =
                    ImGuiFileDialog::Instance()->GetFilePathName();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        // Annotation dialog handlers are now inside DrawAnnotationDialog()

        if (ImGuiFileDialog::Instance()->Display("LoadAnnotProject", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                const auto sel = ImGuiFileDialog::Instance()->GetSelection();
                std::filesystem::path cfg_path;
                if (!sel.empty()) {
                    cfg_path = std::filesystem::path(sel.begin()->second);
                } else {
                    std::string full =
                        ImGuiFileDialog::Instance()->GetFilePathName(
                            IGFD_ResultMode_KeepInputFile);
                    if (!full.empty())
                        cfg_path = std::filesystem::path(full);
                    else
                        cfg_path = std::filesystem::path(
                            ImGuiFileDialog::Instance()->GetCurrentPath());
                }
                ProjectManager loaded;
                std::string err;
                if (!load_project_manager_json(&loaded, cfg_path, &err)) {
                    popups.pushError(err);
                } else {
                    pm = loaded;
                    if (setup_project(pm, skeleton, skeleton_map, &err))
                        on_project_loaded();
                    else
                        popups.pushError(err);
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        // Skeleton creator dialog handlers are now inside DrawSkeletonCreatorWindow()

        // (deleted: ChooseBackgroundImage, LoadSkeletonForEdit handlers moved to skeleton_creator_window.h)

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

#ifndef __APPLE__
            // Automatic YOLO detection for current frame
            if (auto_yolo_labeling && !yolo_model_path.empty() &&
                skeleton.has_bbox) {
                if (!frameHasYoloDetections(current_frame_num, keypoints_map,
                                            &skeleton) &&
                    yolo_processed_frames.find(current_frame_num) ==
                        yolo_processed_frames.end()) {

                    // Mark frame as processed to avoid duplicate processing
                    yolo_processed_frames.insert(current_frame_num);

                    // Enable YOLO detection flag
                    yolo_detection = true;

                    std::cout << "Auto YOLO: Processing frame "
                              << current_frame_num << std::endl;

                    // Run YOLO inference on all cameras for this frame
                    for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                        if (ps.pause_seeked) {
                            unsigned char *frame_data =
                                scene->display_buffer[cam_id][select_corr_head]
                                    .frame;

                            if (frame_data) {
                                yolo_predictions[cam_id] = runYoloInference(
                                    yolo_model_path, frame_data,
                                    scene->image_width[cam_id],
                                    scene->image_height[cam_id]);

                                yolo_bboxes[cam_id].clear();
                                for (const auto &pred :
                                     yolo_predictions[cam_id]) {
                                    yolo_bboxes[cam_id].emplace_back(pred);
                                }
                            }
                        } else {
                            if (window_was_decoding[pm.camera_names[cam_id]]) {
                                unsigned char *frame_data =
                                    scene
                                        ->display_buffer[cam_id]
                                                        [select_corr_head]
                                        .frame;

                                if (frame_data) {
                                    yolo_predictions[cam_id] = runYoloInference(
                                        yolo_model_path, frame_data,
                                        scene->image_width[cam_id],
                                        scene->image_height[cam_id]);

                                    yolo_bboxes[cam_id].clear();
                                    for (const auto &pred :
                                         yolo_predictions[cam_id]) {
                                        yolo_bboxes[cam_id].emplace_back(pred);
                                    }
                                }
                            }
                        }
                    }

                    // Add YOLO detections to main bounding box system
                    if (!yolo_bboxes.empty() &&
                        std::any_of(yolo_bboxes.begin(), yolo_bboxes.end(),
                                    [](const auto &cam_bboxes) {
                                        return !cam_bboxes.empty();
                                    })) {
                        for (int cam_id = 0; cam_id < scene->num_cams;
                             cam_id++) {
                            if (!yolo_bboxes[cam_id].empty()) {
                                // Ensure keypoints structure exists
                                bool keypoints_find =
                                    keypoints_map.find(current_frame_num) !=
                                    keypoints_map.end();
                                if (!keypoints_find) {
                                    KeyPoints *keypoints =
                                        (KeyPoints *)malloc(sizeof(KeyPoints));
                                    allocate_keypoints(keypoints, scene,
                                                       &skeleton);
                                    keypoints_map[current_frame_num] =
                                        keypoints;
                                }

                                // Add YOLO detections to main bounding box
                                // system
                                int yolo_bbox_id = 0;
                                for (const auto &yolo_bbox :
                                     yolo_bboxes[cam_id]) {
                                    if (yolo_bbox.is_valid) {
                                        while (yolo_bbox.class_id >=
                                               (int)bbox_class_colors.size()) {
                                            create_new_bbox_class();
                                        }

                                        BoundingBox bbox;

                                        bbox.rect = new ImPlotRect(
                                            yolo_bbox.x_min, yolo_bbox.x_max,
                                            yolo_bbox.y_min, yolo_bbox.y_max);

                                        bbox.state = RectTwoPoints;
                                        bbox.class_id = yolo_bbox.class_id;
                                        bbox.id = yolo_bbox_id++;
                                        bbox.confidence = yolo_bbox.confidence;
                                        bbox.has_bbox_keypoints = false;
                                        bbox.bbox_keypoints2d = nullptr;
                                        bbox.active_kp_id = nullptr;

                                        if (skeleton.has_bbox &&
                                            skeleton.has_skeleton &&
                                            skeleton.num_nodes > 0) {
                                            allocate_bbox_keypoints(
                                                &bbox, scene, &skeleton);
                                        }

                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list[cam_id]
                                            .push_back(bbox);
                                    }
                                }

                                std::cout << "Auto YOLO: Added "
                                          << yolo_bboxes[cam_id].size()
                                          << " detections for camera " << cam_id
                                          << ", frame " << current_frame_num
                                          << std::endl;
                            }
                        }
                    }
                }
            }
#endif // !__APPLE__
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
                        }
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
                                contrast,          // e.g. 1.0f default
                                (float)brightness, // from ImGui slider
                                pivot_midgray, // pivot around mid-gray (128)
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

                                if (keypoints_find && skeleton.has_skeleton &&
                                    !skeleton.has_bbox) {
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
                                if (skeleton.has_bbox) {
                                    static bool shift_was_pressed = false;
                                    bool shift_pressed =
                                        ImGui::GetIO().KeyShift;

                                    bool keypoints_find =
                                        keypoints_map.find(current_frame_num) !=
                                        keypoints_map.end();
                                    if (!keypoints_find) {
                                        KeyPoints *keypoints =
                                            (KeyPoints *)malloc(
                                                sizeof(KeyPoints));
                                        allocate_keypoints(keypoints, scene,
                                                           &skeleton);
                                        keypoints_map[current_frame_num] =
                                            keypoints;
                                    }

                                    if (shift_pressed && !shift_was_pressed) {
                                        ImPlotPoint mouse =
                                            ImPlot::GetPlotMousePos();

                                        // Clamp mouse coordinates to frame
                                        // bounds
                                        double clamped_x = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_width[j],
                                                mouse.x));
                                        double clamped_y = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_height[j],
                                                mouse.y));

                                        // Delete existing bbox with the
                                        // same class_id and id
                                        if (keypoints_map.find(
                                                current_frame_num) !=
                                            keypoints_map.end()) {
                                            auto &bbox_list =
                                                keypoints_map[current_frame_num]
                                                    ->bbox2d_list[j];

                                            // Find and remove bbox with
                                            // same class_id and id
                                            bbox_list.erase(
                                                std::remove_if(
                                                    bbox_list.begin(),
                                                    bbox_list.end(),
                                                    [current_bbox_class,
                                                     current_bbox_id](
                                                        const BoundingBox
                                                            &bbox) {
                                                        return bbox.class_id ==
                                                                   current_bbox_class &&
                                                               bbox.id ==
                                                                   current_bbox_id;
                                                    }),
                                                bbox_list.end());
                                        }

                                        BoundingBox new_bbox;
                                        new_bbox.rect = new ImPlotRect(
                                            clamped_x, clamped_x, clamped_y,
                                            clamped_y);
                                        new_bbox.state = RectOnePoint;
                                        new_bbox.class_id =
                                            current_bbox_class; // Use
                                                                // currently
                                                                // selected
                                                                // class
                                        new_bbox.id =
                                            current_bbox_id; // Set bbox ID
                                        new_bbox.confidence = 1.0f;
                                        new_bbox.has_bbox_keypoints = false;
                                        new_bbox.bbox_keypoints2d = nullptr;
                                        new_bbox.active_kp_id = nullptr;

                                        // Allocate keypoints for this
                                        // bounding box only if skeleton has
                                        // both bbox and skeleton
                                        if (skeleton.has_bbox &&
                                            skeleton.has_skeleton &&
                                            skeleton.num_nodes > 0) {
                                            allocate_bbox_keypoints(
                                                &new_bbox, scene, &skeleton);
                                        }

                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list[j]
                                            .push_back(new_bbox);
                                    }

                                    // Only process bbox operations if
                                    // keypoints exist for this frame
                                    if (keypoints_map.find(current_frame_num) !=
                                        keypoints_map.end()) {
                                        for (auto &bbox :
                                             keypoints_map[current_frame_num]
                                                 ->bbox2d_list[j]) {
                                            if (bbox.state == RectOnePoint &&
                                                shift_pressed) {
                                                ImPlotPoint mouse =
                                                    ImPlot::GetPlotMousePos();
                                                // Clamp mouse coordinates
                                                // to frame bounds
                                                double clamped_x = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_width[j],
                                                        mouse.x));
                                                double clamped_y = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_height[j],
                                                        mouse.y));
                                                bbox.rect->X.Max = clamped_x;
                                                bbox.rect->Y.Max = clamped_y;
                                            }

                                            if (bbox.state == RectOnePoint &&
                                                !shift_pressed &&
                                                shift_was_pressed) {
                                                bbox.state = RectTwoPoints;

                                                double x_min =
                                                    std::min(bbox.rect->X.Min,
                                                             bbox.rect->X.Max);
                                                double x_max =
                                                    std::max(bbox.rect->X.Min,
                                                             bbox.rect->X.Max);
                                                double y_min =
                                                    std::min(bbox.rect->Y.Min,
                                                             bbox.rect->Y.Max);
                                                double y_max =
                                                    std::max(bbox.rect->Y.Min,
                                                             bbox.rect->Y.Max);

                                                bbox.rect->X.Min = x_min;
                                                bbox.rect->X.Max = x_max;
                                                bbox.rect->Y.Min = y_min;
                                                bbox.rect->Y.Max = y_max;

                                                // Auto-increment bbox ID
                                                // after finishing drawing
                                                current_bbox_id++;
                                            }
                                        }
                                    }

                                    shift_was_pressed = shift_pressed;
                                }
                            } else {
                                is_view_focused[j] = false;
                            }

                            // Plot bounding boxes (both with and without
                            // keypoints)
                            if (skeleton.has_bbox) {
                                bool keypoints_find =
                                    keypoints_map.find(current_frame_num) !=
                                    keypoints_map.end();
                                if (!keypoints_find) {
                                    KeyPoints *keypoints =
                                        (KeyPoints *)malloc(sizeof(KeyPoints));
                                    allocate_keypoints(keypoints, scene,
                                                       &skeleton);
                                    keypoints_map[current_frame_num] =
                                        keypoints;
                                }

                                // Determine which bbox is active (under
                                // mouse cursor)
                                ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                int active_bbox_idx = -1;

                                // Check which bbox the mouse is hovering
                                // over
                                for (int bbox_idx = 0;
                                     bbox_idx < keypoints_map[current_frame_num]
                                                    ->bbox2d_list[j]
                                                    .size();
                                     bbox_idx++) {
                                    BoundingBox &bbox =
                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list[j][bbox_idx];
                                    if (bbox.rect &&
                                        bbox.state == RectTwoPoints &&
                                        is_point_in_bbox(mouse.x, mouse.y,
                                                         bbox.rect)) {
                                        active_bbox_idx = bbox_idx;
                                        break;
                                    }
                                }

                                // Update global active bbox tracking for
                                // this camera
                                if (j < user_active_bbox_idx.size()) {
                                    user_active_bbox_idx[j] = active_bbox_idx;
                                }

                                // Plot multiple bounding boxes
                                for (int bbox_idx = 0;
                                     bbox_idx < keypoints_map[current_frame_num]
                                                    ->bbox2d_list[j]
                                                    .size();
                                     bbox_idx++) {
                                    BoundingBox &bbox =
                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list[j][bbox_idx];
                                    if (bbox.rect) {
                                        // Get color based on class_id
                                        ImVec4 bbox_color =
                                            ImVec4(0.5f, 1.0f, 1.0f,
                                                   1.0f); // Default fallback
                                        if (bbox.class_id >= 0 &&
                                            bbox.class_id <
                                                bbox_class_colors.size()) {
                                            bbox_color = bbox_class_colors
                                                [bbox.class_id];
                                        }

                                        // Reduce opacity for inactive
                                        // bboxes
                                        bool is_active_bbox =
                                            (bbox_idx == active_bbox_idx);
                                        if (!is_active_bbox) {
                                            bbox_color.w =
                                                0.6f; // Make inactive
                                                      // bboxes more
                                                      // transparent
                                        }

                                        // Draw completed bounding boxes
                                        if (bbox.state == RectTwoPoints) {
                                            bool bbox_clicked = false,
                                                 bbox_hovered = false,
                                                 bbox_held = false;

                                            // Store previous rect for this
                                            // specific bbox
                                            static std::map<std::pair<int, int>,
                                                            ImPlotRect>
                                                bbox_prev_rects; // frame ->
                                                                 // {camera,
                                                                 // bbox_idx}
                                                                 // ->
                                                                 // prev_rect
                                            auto bbox_key =
                                                std::make_pair(j, bbox_idx);

                                            // Initialize previous rect if
                                            // not exists
                                            if (bbox_prev_rects.find(
                                                    bbox_key) ==
                                                bbox_prev_rects.end()) {
                                                bbox_prev_rects[bbox_key] =
                                                    *bbox.rect;
                                            }

                                            ImPlotRect prev_bbox_rect =
                                                bbox_prev_rects[bbox_key];

                                            // Only allow interaction if
                                            // this is the active bbox or no
                                            // bbox is active
                                            ImPlotDragToolFlags drag_flags =
                                                ImPlotDragToolFlags_None;
                                            if (!is_active_bbox &&
                                                active_bbox_idx != -1) {
                                                drag_flags =
                                                    ImPlotDragToolFlags_NoInputs;
                                            }

                                            bool bbox_modified = MyDragRect(
                                                1000 + bbox_idx,
                                                &bbox.rect->X.Min,
                                                &bbox.rect->Y.Min,
                                                &bbox.rect->X.Max,
                                                &bbox.rect->Y.Max, bbox_color,
                                                drag_flags, &bbox_clicked,
                                                &bbox_hovered, &bbox_held);

                                            // Clamp bbox coordinates to
                                            // frame bounds after any
                                            // modification
                                            if (bbox_modified || bbox_held) {
                                                bbox.rect->X.Min = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_width[j],
                                                        bbox.rect->X.Min));
                                                bbox.rect->Y.Min = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_height[j],
                                                        bbox.rect->Y.Min));
                                                bbox.rect->X.Max = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_width[j],
                                                        bbox.rect->X.Max));
                                                bbox.rect->Y.Max = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_height[j],
                                                        bbox.rect->Y.Max));
                                            }

                                            // Display bbox ID on frame if
                                            // enabled
                                            if (show_bbox_ids) {
                                                // Position text above
                                                // top-right corner of bbox
                                                double text_x =
                                                    bbox.rect->X.Max -
                                                    10.0; // Offset to the
                                                          // left
                                                double text_y =
                                                    bbox.rect->Y.Max -
                                                    10.0; // Offset above
                                                          // the box
                                                ImPlot::PushStyleColor(
                                                    ImPlotCol_InlayText,
                                                    ImVec4(1.0f, 1.0f, 1.0f,
                                                           1.0f));
                                                ImPlot::PlotText(
                                                    std::to_string(bbox.id)
                                                        .c_str(),
                                                    text_x, text_y);
                                            }

                                            if (bbox_clicked || bbox_held ||
                                                bbox_modified) {
                                                active_bbox_idx = bbox_idx;
                                            }

                                            bool is_active_bbox =
                                                (bbox_idx == active_bbox_idx);

                                            // Scale bbox keypoints if bbox
                                            // was resized and this is the
                                            // active bbox
                                            if (bbox_modified &&
                                                bbox.has_bbox_keypoints &&
                                                bbox.bbox_keypoints2d &&
                                                is_active_bbox) {
                                                scale_bbox_keypoints(
                                                    &bbox, scene, &skeleton,
                                                    &prev_bbox_rect, bbox.rect);
                                                bbox_prev_rects[bbox_key] =
                                                    *bbox.rect; // Update
                                                                // stored
                                                                // previous
                                                                // rect
                                            }

                                            // Update previous rect when not
                                            // being dragged
                                            if (!bbox_held) {
                                                bbox_prev_rects[bbox_key] =
                                                    *bbox.rect;
                                            }

                                            // Handle keyboard shortcuts
                                            // when hovering over bounding
                                            // box
                                            if (bbox_hovered) {
                                                // Update hovered bbox info
                                                // for display
                                                hovered_bbox_cam = j;
                                                hovered_bbox_idx = bbox_idx;
                                                hovered_bbox_confidence =
                                                    bbox.confidence;
                                                hovered_bbox_class =
                                                    bbox.class_id;
                                                hovered_bbox_id = bbox.id;

                                                // Delete bounding box from
                                                // current camera when 'T'
                                                // key is pressed while
                                                // hovering
                                                if (ImGui::IsKeyPressed(
                                                        ImGuiKey_F, false) &&
                                                    !io.WantTextInput) {
                                                    // Clean up bbox
                                                    // keypoints before
                                                    // deletion
                                                    if (bbox.has_bbox_keypoints &&
                                                        bbox.bbox_keypoints2d) {
                                                        free(
                                                            bbox.bbox_keypoints2d);
                                                        bbox.bbox_keypoints2d =
                                                            nullptr;
                                                        free(bbox.active_kp_id);
                                                        bbox.active_kp_id =
                                                            nullptr;
                                                    }
                                                    // Mark for deletion by
                                                    // setting state to
                                                    // RectNull
                                                    delete bbox.rect;
                                                    bbox.rect = nullptr;
                                                    bbox.state = RectNull;
                                                    bbox.has_bbox_keypoints =
                                                        false;
                                                }

                                                // Delete bounding box from
                                                // all cameras when 'O' key
                                                // is pressed while hovering
                                                if (ImGui::IsKeyPressed(
                                                        ImGuiKey_O, false) &&
                                                    !io.WantTextInput) {
                                                    // Find this bbox's
                                                    // class_id and delete
                                                    // all bboxes with same
                                                    // class from all
                                                    // cameras
                                                    int target_class_id =
                                                        bbox.class_id;
                                                    for (int cam_idx = 0;
                                                         cam_idx <
                                                         scene->num_cams;
                                                         cam_idx++) {
                                                        auto &bbox_list =
                                                            keypoints_map
                                                                [current_frame_num]
                                                                    ->bbox2d_list
                                                                        [cam_idx];
                                                        for (auto &other_bbox :
                                                             bbox_list) {
                                                            if (other_bbox
                                                                        .class_id ==
                                                                    target_class_id &&
                                                                other_bbox
                                                                        .state !=
                                                                    RectNull &&
                                                                other_bbox
                                                                        .rect !=
                                                                    nullptr) {
                                                                // Clean up
                                                                // bbox
                                                                // keypoints
                                                                // before
                                                                // deletion
                                                                if (other_bbox
                                                                        .has_bbox_keypoints &&
                                                                    other_bbox
                                                                        .bbox_keypoints2d) {
                                                                    free(
                                                                        other_bbox
                                                                            .bbox_keypoints2d);
                                                                    other_bbox
                                                                        .bbox_keypoints2d =
                                                                        nullptr;
                                                                    free(
                                                                        other_bbox
                                                                            .active_kp_id);
                                                                    other_bbox
                                                                        .active_kp_id =
                                                                        nullptr;
                                                                }
                                                                delete other_bbox
                                                                    .rect;
                                                                other_bbox
                                                                    .rect =
                                                                    nullptr;
                                                                other_bbox
                                                                    .state =
                                                                    RectNull;
                                                                other_bbox
                                                                    .has_bbox_keypoints =
                                                                    false;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        // Draw bounding boxes being created
                                        // (one point set)
                                        else if (bbox.state == RectOnePoint) {
                                            // Draw a preview rectangle
                                            // while dragging
                                            double xs[5] = {bbox.rect->X.Min,
                                                            bbox.rect->X.Max,
                                                            bbox.rect->X.Max,
                                                            bbox.rect->X.Min,
                                                            bbox.rect->X.Min};
                                            double ys[5] = {bbox.rect->Y.Max,
                                                            bbox.rect->Y.Max,
                                                            bbox.rect->Y.Min,
                                                            bbox.rect->Y.Min,
                                                            bbox.rect->Y.Max};
                                            ImPlot::SetNextLineStyle(bbox_color,
                                                                     2.0f);
                                            ImPlot::PlotLine("##bbox_preview",
                                                             xs, ys, 5);
                                        }

                                        // Plot keypoints within this
                                        // bounding box (only if keypoints
                                        // are enabled and skeleton has
                                        // keypoints)
                                        if (bbox.state == RectTwoPoints &&
                                            keypoints_find &&
                                            skeleton.has_skeleton &&
                                            bbox.has_bbox_keypoints) {
                                            bool is_saved = true;
                                            // Only allow interaction with
                                            // keypoints if this is the
                                            // active bbox
                                            gui_plot_bbox_keypoints(
                                                &bbox, &skeleton, j,
                                                scene->num_cams, is_active_bbox,
                                                is_saved, bbox_idx);

                                            // Handle keypoint labeling with
                                            // W key for bounding box
                                            // keypoints (only on active
                                            // bbox)
                                            if (is_active_bbox &&
                                                ImGui::IsKeyPressed(ImGuiKey_W,
                                                                    false) &&
                                                !io.WantTextInput) {
                                                ImPlotPoint mouse =
                                                    ImPlot::GetPlotMousePos();
                                                if (is_point_in_bbox(
                                                        mouse.x, mouse.y,
                                                        bbox.rect)) {
                                                    u32 active_kp =
                                                        bbox.active_kp_id[j];
                                                    if (active_kp <
                                                        skeleton.num_nodes) {
                                                        bbox.bbox_keypoints2d
                                                            [j][active_kp]
                                                                .position = {
                                                            mouse.x, mouse.y};
                                                        bbox.bbox_keypoints2d
                                                            [j][active_kp]
                                                                .is_labeled =
                                                            true;
                                                        constrain_keypoint_to_bbox(
                                                            &bbox.bbox_keypoints2d
                                                                 [j][active_kp],
                                                            bbox.rect);
                                                        if (active_kp <
                                                            (skeleton
                                                                 .num_nodes -
                                                             1)) {
                                                            bbox.active_kp_id
                                                                [j]++;
                                                        }
                                                    }
                                                }
                                            }

                                            if (is_active_bbox) {
                                                u32 *active_kp =
                                                    &(bbox.active_kp_id[j]);

                                                if (ImGui::IsKeyPressed(
                                                        ImGuiKey_A, true) &&
                                                    !io.WantTextInput) {
                                                    if (*active_kp <= 0) {
                                                        *active_kp = 0;
                                                    } else {
                                                        (*active_kp)--;
                                                    }
                                                }

                                                if (ImGui::IsKeyPressed(
                                                        ImGuiKey_D, true) &&
                                                    !io.WantTextInput) {
                                                    if (*active_kp >=
                                                        skeleton.num_nodes -
                                                            1) {
                                                        *active_kp =
                                                            skeleton.num_nodes -
                                                            1;
                                                    } else {
                                                        (*active_kp)++;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Plot oriented bounding boxes
                            if (skeleton.has_obb) {
                                bool keypoints_find =
                                    keypoints_map.find(current_frame_num) !=
                                    keypoints_map.end();
                                if (!keypoints_find) {
                                    KeyPoints *keypoints =
                                        (KeyPoints *)malloc(sizeof(KeyPoints));
                                    allocate_keypoints(keypoints, scene,
                                                       &skeleton);
                                    keypoints_map[current_frame_num] =
                                        keypoints;
                                }

                                // ESC key cancels OBB creation
                                if (is_view_focused[j] &&
                                    ImGui::IsKeyPressed(ImGuiKey_Escape,
                                                        false) &&
                                    !io.WantTextInput) {
                                    // Find and cancel any incomplete OBB
                                    for (auto &obb :
                                         keypoints_map[current_frame_num]
                                             ->obb2d_list[j]) {
                                        if (obb.state == OBBFirstAxisPoint ||
                                            obb.state == OBBSecondAxisPoint) {
                                            obb.state = OBBNull;
                                            break;
                                        }
                                    }
                                }

                                // Handle OBB interaction with W key
                                if (is_view_focused[j] &&
                                    ImGui::IsKeyPressed(ImGuiKey_W, false) &&
                                    !io.WantTextInput) {
                                    ImPlotPoint mouse =
                                        ImPlot::GetPlotMousePos();

                                    // Find an OBB to continue or create a
                                    // new one
                                    bool found_incomplete_obb = false;
                                    for (auto &obb :
                                         keypoints_map[current_frame_num]
                                             ->obb2d_list[j]) {
                                        if (obb.state == OBBNull) {
                                            // Start new OBB - place first
                                            // axis point Clamp mouse
                                            // coordinates to frame bounds
                                            double clamped_x = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_width[j],
                                                         (double)mouse.x));
                                            double clamped_y = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_height[j],
                                                         (double)mouse.y));
                                            obb.axis_point1 =
                                                ImVec2(clamped_x, clamped_y);
                                            obb.state = OBBFirstAxisPoint;
                                            obb.class_id =
                                                current_bbox_class; // Use
                                                                    // currently
                                                                    // selected
                                                                    // class
                                            obb.confidence = 1.0f;
                                            found_incomplete_obb = true;
                                            break;
                                        } else if (obb.state ==
                                                   OBBFirstAxisPoint) {
                                            // Place second axis point
                                            // Clamp mouse coordinates to
                                            // frame bounds
                                            double clamped_x = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_width[j],
                                                         (double)mouse.x));
                                            double clamped_y = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_height[j],
                                                         (double)mouse.y));
                                            obb.axis_point2 =
                                                ImVec2(clamped_x, clamped_y);
                                            obb.state = OBBSecondAxisPoint;
                                            found_incomplete_obb = true;
                                            break;
                                        } else if (obb.state ==
                                                   OBBSecondAxisPoint) {
                                            // Place corner point and
                                            // complete the OBB Clamp mouse
                                            // coordinates to frame bounds
                                            double clamped_x = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_width[j],
                                                         (double)mouse.x));
                                            double clamped_y = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_height[j],
                                                         (double)mouse.y));
                                            obb.corner_point =
                                                ImVec2(clamped_x, clamped_y);
                                            obb.state = OBBThirdPoint;
                                            calculate_obb_properties(&obb);
                                            obb.state = OBBComplete;

                                            // Clear the construction points
                                            // after completion
                                            obb.axis_point1 = ImVec2(0, 0);
                                            obb.axis_point2 = ImVec2(0, 0);
                                            obb.corner_point = ImVec2(0, 0);

                                            // Auto-increment bbox ID after
                                            // finishing drawing OBB
                                            current_bbox_id++;

                                            found_incomplete_obb = true;
                                            break;
                                        }
                                    }

                                    // If no incomplete OBB found, create a
                                    // new one
                                    if (!found_incomplete_obb) {
                                        // Delete existing OBB with the same
                                        // class_id and id
                                        auto &obb_list =
                                            keypoints_map[current_frame_num]
                                                ->obb2d_list[j];
                                        obb_list.erase(
                                            std::remove_if(
                                                obb_list.begin(),
                                                obb_list.end(),
                                                [current_bbox_class,
                                                 current_bbox_id](
                                                    const OrientedBoundingBox
                                                        &obb) {
                                                    return obb.class_id ==
                                                               current_bbox_class &&
                                                           obb.id ==
                                                               current_bbox_id;
                                                }),
                                            obb_list.end());

                                        OrientedBoundingBox new_obb;
                                        // Clamp mouse coordinates to frame
                                        // bounds for new OBB
                                        double clamped_x = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_width[j],
                                                (double)mouse.x));
                                        double clamped_y = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_height[j],
                                                (double)mouse.y));
                                        new_obb.axis_point1 =
                                            ImVec2(clamped_x, clamped_y);
                                        new_obb.axis_point2 = ImVec2(0, 0);
                                        new_obb.corner_point = ImVec2(0, 0);
                                        new_obb.center = ImVec2(0, 0);
                                        new_obb.width = 0;
                                        new_obb.height = 0;
                                        new_obb.rotation = 0;
                                        new_obb.state = OBBFirstAxisPoint;
                                        new_obb.class_id =
                                            current_bbox_class; // Use
                                                                // currently
                                                                // selected
                                                                // class
                                        new_obb.id =
                                            current_bbox_id; // Set OBB ID
                                        new_obb.confidence = 1.0f;
                                        keypoints_map[current_frame_num]
                                            ->obb2d_list[j]
                                            .push_back(new_obb);
                                    }
                                }

                                // Handle OBB manipulation and interaction
                                static bool obb_dragging = false;
                                static size_t dragged_obb_idx = 0;

                                // Handle OBB construction point dragging
                                static bool obb_point_dragging = false;
                                static size_t dragged_point_obb_idx = 0;
                                static int dragged_point_type =
                                    0; // 0 = axis_point1, 1 = axis_point2,
                                       // 2 = corner_point

                                // Handle construction point dragging
                                if (is_view_focused[j] &&
                                    ImPlot::IsPlotHovered()) {
                                    ImPlotPoint mouse =
                                        ImPlot::GetPlotMousePos();
                                    ImVec2 mouse_vec = ImVec2(mouse.x, mouse.y);

                                    // Start dragging construction points
                                    if (ImGui::IsMouseClicked(
                                            ImGuiMouseButton_Left) &&
                                        !obb_point_dragging && !obb_dragging) {
                                        for (size_t obb_idx = 0;
                                             obb_idx <
                                             keypoints_map[current_frame_num]
                                                 ->obb2d_list[j]
                                                 .size();
                                             obb_idx++) {
                                            auto &obb =
                                                keypoints_map[current_frame_num]
                                                    ->obb2d_list[j][obb_idx];

                                            // Only allow dragging for
                                            // incomplete OBBs
                                            if (obb.state >=
                                                    OBBFirstAxisPoint &&
                                                obb.state < OBBComplete) {
                                                // Check if clicking near
                                                // axis_point1
                                                if (obb.state >=
                                                        OBBFirstAxisPoint &&
                                                    is_point_near(
                                                        mouse_vec,
                                                        obb.axis_point1)) {
                                                    obb_point_dragging = true;
                                                    dragged_point_obb_idx =
                                                        obb_idx;
                                                    dragged_point_type = 0;
                                                    break;
                                                }
                                                // Check if clicking near
                                                // axis_point2
                                                if (obb.state >=
                                                        OBBSecondAxisPoint &&
                                                    is_point_near(
                                                        mouse_vec,
                                                        obb.axis_point2)) {
                                                    obb_point_dragging = true;
                                                    dragged_point_obb_idx =
                                                        obb_idx;
                                                    dragged_point_type = 1;
                                                    break;
                                                }
                                                // Check if clicking near
                                                // corner_point
                                                if (obb.state >=
                                                        OBBThirdPoint &&
                                                    is_point_near(
                                                        mouse_vec,
                                                        obb.corner_point)) {
                                                    obb_point_dragging = true;
                                                    dragged_point_obb_idx =
                                                        obb_idx;
                                                    dragged_point_type = 2;
                                                    break;
                                                }
                                            }
                                        }
                                    }

                                    // Continue dragging
                                    if (obb_point_dragging &&
                                        ImGui::IsMouseDragging(
                                            ImGuiMouseButton_Left)) {
                                        auto &obb =
                                            keypoints_map[current_frame_num]
                                                ->obb2d_list
                                                    [j][dragged_point_obb_idx];

                                        // Clamp mouse coordinates to frame
                                        // bounds
                                        double clamped_x = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_width[j],
                                                (double)mouse_vec.x));
                                        double clamped_y = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_height[j],
                                                (double)mouse_vec.y));
                                        ImVec2 clamped_mouse =
                                            ImVec2(clamped_x, clamped_y);

                                        if (dragged_point_type == 0) {
                                            obb.axis_point1 = clamped_mouse;
                                        } else if (dragged_point_type == 1) {
                                            obb.axis_point2 = clamped_mouse;
                                        } else if (dragged_point_type == 2) {
                                            obb.corner_point = clamped_mouse;
                                            // Recalculate OBB properties if
                                            // dragging corner point
                                            if (obb.state == OBBThirdPoint) {
                                                calculate_obb_properties(&obb);
                                            }
                                        }
                                    }

                                    // Stop dragging
                                    if (obb_point_dragging &&
                                        ImGui::IsMouseReleased(
                                            ImGuiMouseButton_Left)) {
                                        obb_point_dragging = false;
                                    }
                                }

                                // Track which OBB is being hovered
                                int current_hovered_obb = -1;

                                // Check for hover when not dragging
                                if (!obb_dragging && !obb_point_dragging &&
                                    ImPlot::IsPlotHovered()) {
                                    ImPlotPoint mouse =
                                        ImPlot::GetPlotMousePos();

                                    // Check if hovering over any OBB
                                    for (size_t obb_idx = 0;
                                         obb_idx <
                                         keypoints_map[current_frame_num]
                                             ->obb2d_list[j]
                                             .size();
                                         obb_idx++) {
                                        auto &obb =
                                            keypoints_map[current_frame_num]
                                                ->obb2d_list[j][obb_idx];

                                        // Check if mouse is inside the OBB
                                        if (is_point_inside_obb(
                                                ImVec2(mouse.x, mouse.y),
                                                obb)) {
                                            current_hovered_obb = obb_idx;
                                            hovered_obb_cam = j;
                                            hovered_obb_idx = obb_idx;
                                            hovered_obb_class = obb.class_id;
                                            hovered_obb_confidence =
                                                obb.confidence;
                                            hovered_obb_id = obb.id;

                                            // Handle key presses for OBB
                                            // manipulation (similar to
                                            // bbox) Delete OBB from current
                                            // camera when 'T' key is
                                            // pressed while hovering
                                            if (ImGui::IsKeyPressed(ImGuiKey_T,
                                                                    false) &&
                                                !io.WantTextInput) {
                                                obb.state = OBBNull;
                                            }

                                            // Delete OBB from all cameras
                                            // when 'F' key is pressed while
                                            // hovering
                                            if (ImGui::IsKeyPressed(ImGuiKey_F,
                                                                    false) &&
                                                !io.WantTextInput) {
                                                int target_class_id =
                                                    obb.class_id;
                                                for (int cam = 0;
                                                     cam < scene->num_cams;
                                                     cam++) {
                                                    if (keypoints_map.count(
                                                            current_frame_num) >
                                                        0) {
                                                        auto &obb_list =
                                                            keypoints_map
                                                                [current_frame_num]
                                                                    ->obb2d_list
                                                                        [cam];
                                                        for (auto &other_obb :
                                                             obb_list) {
                                                            if (other_obb
                                                                    .class_id ==
                                                                target_class_id) {
                                                                other_obb
                                                                    .state =
                                                                    OBBNull;
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            // Switch OBB class when 'A' or
                                            // 'D' key is pressed while
                                            // hovering
                                            if (ImGui::IsKeyPressed(ImGuiKey_A,
                                                                    true) &&
                                                !io.WantTextInput) {
                                                obb.class_id =
                                                    (obb.class_id - 1 +
                                                     bbox_class_names.size()) %
                                                    bbox_class_names.size();
                                            }
                                            if (ImGui::IsKeyPressed(ImGuiKey_D,
                                                                    true) &&
                                                !io.WantTextInput) {
                                                obb.class_id =
                                                    (obb.class_id + 1) %
                                                    bbox_class_names.size();
                                            }

                                            break;
                                        }
                                    }
                                }

                                // Draw all OBBs for this camera
                                ImPlotPoint current_mouse =
                                    ImPlot::GetPlotMousePos();
                                // Clamp mouse coordinates for preview to
                                // frame bounds
                                double clamped_mouse_x = std::max(
                                    0.0, std::min((double)scene->image_width[j],
                                                  current_mouse.x));
                                double clamped_mouse_y = std::max(
                                    0.0,
                                    std::min((double)scene->image_height[j],
                                             current_mouse.y));
                                ImVec2 clamped_preview_mouse =
                                    ImVec2(clamped_mouse_x, clamped_mouse_y);
                                for (size_t obb_idx = 0;
                                     obb_idx < keypoints_map[current_frame_num]
                                                   ->obb2d_list[j]
                                                   .size();
                                     obb_idx++) {
                                    auto &obb = keypoints_map[current_frame_num]
                                                    ->obb2d_list[j][obb_idx];

                                    if (obb.state != OBBNull) {
                                        // Get color based on class_id (same
                                        // system as bboxes)
                                        ImVec4 obb_color =
                                            ImVec4(0.3f, 1.0f, 1.0f,
                                                   1.0f); // Default color
                                        if (obb.class_id >= 0 &&
                                            obb.class_id <
                                                bbox_class_colors.size()) {
                                            obb_color =
                                                bbox_class_colors[obb.class_id];
                                        }

                                        // Highlight when hovering,
                                        // dragging, or during construction
                                        bool is_active =
                                            (obb_dragging &&
                                             dragged_obb_idx == obb_idx) ||
                                            (current_hovered_obb ==
                                             (int)obb_idx) ||
                                            (!obb_dragging &&
                                             obb.state < OBBComplete);

                                        // Show preview when we have at
                                        // least one point and mouse is in
                                        // plot area
                                        bool show_preview =
                                            ((obb.state == OBBFirstAxisPoint ||
                                              obb.state ==
                                                  OBBSecondAxisPoint) &&
                                             ImPlot::IsPlotHovered() &&
                                             !obb_dragging &&
                                             !obb_point_dragging &&
                                             current_hovered_obb == -1);

                                        draw_obb(obb, is_active, obb_color,
                                                 clamped_preview_mouse,
                                                 show_preview);

                                        // Display OBB ID on frame if
                                        // enabled and OBB is complete
                                        if (show_bbox_ids &&
                                            obb.state == OBBComplete) {
                                            // Position text above top-right
                                            // corner of OBB
                                            double text_x =
                                                obb.center.x + obb.width / 2 +
                                                5.0; // Right side of OBB
                                            double text_y =
                                                obb.center.y + obb.height / 2 +
                                                5.0; // Above the OBB
                                            ImPlot::PlotText(
                                                std::to_string(obb.id).c_str(),
                                                text_x, text_y);
                                        }
                                    }
                                }
                            }

                            if (keypoints_find) {
                                // Only plot keypoints if skeleton has
                                // keypoints and we're not in bbox+keypoints
                                // mode
                                if (skeleton.has_skeleton &&
                                    !skeleton.has_bbox) {
                                    gui_plot_keypoints(
                                        keypoints_map.at(current_frame_num),
                                        &skeleton, j, scene->num_cams);
                                }

                                if (skeleton.name == "Rat4Box" ||
                                    skeleton.name == "Rat4Box3Ball") {
                                    gui_plot_bbox_from_keypoints(
                                        keypoints_map.at(current_frame_num),
                                        &skeleton, j, 4, 5);
                                }
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

            // Bounding box class switching keybinds
            if (ImGui::IsKeyPressed(ImGuiKey_Z, false) && !io.WantTextInput) {
                // Switch to previous class
                if (bbox_class_names.size() > 0) {
                    current_bbox_class =
                        (current_bbox_class - 1 + bbox_class_names.size()) %
                        bbox_class_names.size();
                    current_bbox_id = 0; // Reset bbox ID when switching classes
                }
            }

            if (ImGui::IsKeyPressed(ImGuiKey_X, false) && !io.WantTextInput) {
                if (bbox_class_names.size() > 0) {
                    // Switch to next class
                    current_bbox_class =
                        (current_bbox_class + 1) % bbox_class_names.size();
                    current_bbox_id = 0; // Reset bbox ID when switching classes
                }
            }

            // Bounding box ID switching keybinds within current class
            if (ImGui::IsKeyPressed(ImGuiKey_C, false) && !io.WantTextInput) {
                // Decrease bbox ID, stop at 0
                if (current_bbox_id > 0) {
                    current_bbox_id--;
                }
            }

            if (ImGui::IsKeyPressed(ImGuiKey_V, false) && !io.WantTextInput) {
                // Increment bbox ID (no wrap around)
                current_bbox_id++;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_N, false) && !io.WantTextInput) {
                create_new_bbox_class();
                // reset bbox id to 0
                current_bbox_id = 0;
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

        if (pm.plot_keypoints_flag) {
            DrawKeypointsWindow(
                pm, scene, skeleton, keypoints_map, current_frame_num,
                is_view_focused, bbox_class_names, current_bbox_class,
                bbox_class_colors, current_bbox_id, hovered_bbox_cam,
                hovered_bbox_idx, hovered_bbox_id, hovered_bbox_confidence,
                hovered_bbox_class, hovered_obb_cam, hovered_obb_idx,
                hovered_obb_id, hovered_obb_confidence, hovered_obb_class,
                show_bbox_ids, new_class_name);
        }

        if (keypoints_find) {
            DrawReprojectionWindow(keypoints_map[current_frame_num],
                                   pm.camera_names, scene, skeleton, rp_tool);
        }

        // Ctrl+S save: global so it works even when Labeling Tool tab
        // is hidden.  The flag is consumed by the save block inside
        // the Labeling Tool window (or after it).
        bool save_requested = false;
        if (pm.plot_keypoints_flag &&
            ImGui::GetIO().KeyCtrl &&
            ImGui::IsKeyPressed(ImGuiKey_S, false) &&
            !io.WantTextInput) {
            save_requested = true;
        }

        if (pm.plot_keypoints_flag) {

            if (ImGui::Begin("Labeling Tool")) {
                if (scene->num_cams > 1) {
                    bool keypoint_triangulated_all = true;
                    if (keypoints_find && scene->num_cams > 1) {
                        for (int j = 0; j < skeleton.num_nodes; j++) {
                            if (!keypoints_map.at(current_frame_num)
                                     ->kp3d[j]
                                     .is_triangulated) {
                                keypoint_triangulated_all = false;
                                break; // small optimization, exit early
                            }
                        }
                    } else {
                        keypoint_triangulated_all = false;
                    }
                    bool apply_color =
                        !keypoint_triangulated_all && keypoints_find;
                    if (apply_color) {
                        ImGui::PushStyleColor(
                            ImGuiCol_Button,
                            (ImVec4)ImColor::HSV(0.8, 1.0f, 1.0f));
                        ImGui::PushStyleColor(
                            ImGuiCol_ButtonHovered,
                            (ImVec4)ImColor::HSV(0.8, 0.9f, 0.8f));
                        ImGui::PushStyleColor(
                            ImGuiCol_ButtonActive,
                            (ImVec4)ImColor::HSV(0.8, 0.9f, 0.5f));
                    }

                    ImGui::BeginDisabled(!keypoints_find);
                    if (ImGui::Button("Triangulate")) {
                        reprojection(keypoints_map.at(current_frame_num),
                                     &skeleton, pm.camera_params, scene);
                    }
                    ImGui::EndDisabled();

                    if (apply_color) {
                        ImGui::PopStyleColor(3);
                    }

                    // T-key triangulation moved outside this window
                    // so it works even when the Labeling Tool tab is hidden.
                } else {
                    // Display message when skeleton is not properly loaded
                    ImGui::Text("Please load a skeleton to view keypoints.");
                }

                if (ImGui::Button("Update keypoints working directory")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = pm.project_path;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseKeypointsFolder",
                        "Choose keypoints working directory", nullptr, config);
                }
                ImGui::SameLine();
                ImGui::Text("%s", pm.keypoints_root_folder.c_str());

                if (ImGui::Button("Save Labeled Data")) {
                    save_requested = true;
                }
                if (last_saved != static_cast<std::time_t>(-1)) {
                    ImGui::SameLine();
                    ImGui::Text("Last saved: %s", ctime(&last_saved));
                }

                static bool load_old_format = false;
                if (ImGui::Button("Load Most Recent Labels")) {
                    free_all_keypoints(keypoints_map, scene);
                    std::string err;
                    if (load_old_format) {
                        if (load_keypoints_depreciated(keypoints_map, &skeleton,
                                                       pm.keypoints_root_folder,
                                                       scene, pm.camera_names,
                                                       err)) {
                            free_all_keypoints(keypoints_map, scene);
                            popups.pushError(err);
                        }

                    } else {
                        std::string most_recent_folder;
                        if (find_most_recent_labels(pm.keypoints_root_folder,
                                                    most_recent_folder, err)) {
                            popups.pushError(err);
                        } else {
                            if (load_keypoints(most_recent_folder,
                                               keypoints_map, &skeleton, scene,
                                               pm.camera_names, err,
                                               bbox_class_names)) {
                                free_all_keypoints(keypoints_map, scene);
                                popups.pushError(err);
                            }
                        }
                    }
                }
                ImGui::SameLine();
                ImGui::Checkbox("Old format", &load_old_format);

                if (ImGui::Button("Load From Selected")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = pm.keypoints_root_folder;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "LoadFromSelected", "Load from selected", nullptr,
                        config);
                }

                ImGui::Separator();

                auto next_labeled_frame_it = keypoints_map.end();

                // Find next labeled frame (forward, wrap)
                for (auto it = keypoints_map.upper_bound(current_frame_num);
                     it != keypoints_map.end(); ++it) {
                    if (has_any_labels(it->second, skeleton, scene)) {
                        next_labeled_frame_it = it;
                        break;
                    }
                }
                if (next_labeled_frame_it == keypoints_map.end()) {
                    for (auto it = keypoints_map.begin();
                         it != keypoints_map.upper_bound(current_frame_num);
                         ++it) {
                        if (has_any_labels(it->second, skeleton, scene)) {
                            next_labeled_frame_it = it;
                            break;
                        }
                    }
                }

                // Find previous labeled frame (no wrap)
                auto prev_labeled_frame_it = keypoints_map.end();
                auto lb = keypoints_map.lower_bound(current_frame_num);
                if (lb != keypoints_map.begin()) {
                    for (auto it = std::prev(lb);;) {
                        if (has_any_labels(it->second, skeleton, scene)) {
                            prev_labeled_frame_it = it;
                            break;
                        }
                        if (it == keypoints_map.begin())
                            break;
                        --it;
                    }
                }

                bool has_next_frame =
                    (next_labeled_frame_it != keypoints_map.end());
                int next_frame_num =
                    has_next_frame ? next_labeled_frame_it->first : -1;
                bool has_prev_frame =
                    (prev_labeled_frame_it != keypoints_map.end());
                int previous_frame_num =
                    has_prev_frame ? prev_labeled_frame_it->first : -1;

                if (ImGui::BeginTable("frame_nav_table", 2,
                                      ImGuiTableFlags_SizingFixedFit)) {
                    ImGui::TableNextRow();

                    // --- Next ---
                    ImGui::TableSetColumnIndex(0);

                    ImGui::BeginDisabled(!has_next_frame);
                    bool button_pressed =
                        ImGui::Button("Jump to next labeled frame");
                    ImGui::EndDisabled();

                    if (button_pressed && has_next_frame) {
                        ps.play_video = false;
                        seek_all_cameras(scene, next_frame_num,
                                         dc_context->video_fps, ps, true);
                    }

                    ImGui::TableSetColumnIndex(1);
                    if (has_next_frame)
                        ImGui::Text("%d", next_frame_num);
                    else
                        ImGui::Text("none");

                    // --- Previous ---
                    ImGui::TableNextRow();

                    ImGui::TableSetColumnIndex(0);
                    ImGui::BeginDisabled(!has_prev_frame);
                    if (ImGui::Button("Copy from previous labeled frame")) {
                        if (keypoints_find) {
                            // delete the keypoints if found
                            free_keypoints(keypoints_map[current_frame_num],
                                           scene);
                            keypoints_map.erase(current_frame_num);
                        }

                        KeyPoints *keypoints =
                            (KeyPoints *)malloc(sizeof(KeyPoints));
                        allocate_keypoints(keypoints, scene, &skeleton);
                        keypoints_map[current_frame_num] = keypoints;

                        KeyPoints *prev = keypoints_map[previous_frame_num];
                        KeyPoints *curr = keypoints_map[current_frame_num];
                        copy_keypoints(curr, prev, scene, &skeleton);
                    }
                    ImGui::EndDisabled();

                    ImGui::TableSetColumnIndex(1);
                    if (has_prev_frame)
                        ImGui::Text("%d", previous_frame_num);
                    else
                        ImGui::Text("none");

                    ImGui::EndTable();
                }

                size_t labeled_count = 0;
                for (const auto &[frame_num, keypoints] : keypoints_map) {
                    if (has_any_labels(keypoints, skeleton, scene,
                                       /*yolo_thresh=*/0.5f)) {
                        ++labeled_count;
                    }
                }
                ImGui::Text("Total labeled frames : %zu", labeled_count);

                // Only show YOLO button when current skeleton has bounding
                // boxes
                if (skeleton.has_bbox) {
                    if (ImGui::Button("Select YOLO")) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = yolo_model_dir;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseYoloModel", "Choose YOLO Model",
                            ".torchscript", config);
                    }
                }

                if (ImGuiFileDialog::Instance()->Display("ChooseYoloModel", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                    if (ImGuiFileDialog::Instance()->IsOk()) {
                        std::string model_path =
                            ImGuiFileDialog::Instance()->GetFilePathName();
                        if (!model_path.empty()) {
                            yolo_model_path = model_path;
                            std::cout
                                << "Selected YOLO model: " << yolo_model_path
                                << std::endl;
                        }
                    }
                    // close
                    ImGuiFileDialog::Instance()->Close();
                }

                // Show model path and run prediction button (only if model
                // is selected)
                if (!yolo_model_path.empty()) {
                    ImGui::Text("Selected model: %s", yolo_model_path.c_str());

                    // Automatic YOLO labeling checkbox
                    ImGui::Checkbox("Automatic YOLO Labeling",
                                    &auto_yolo_labeling);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Automatically run YOLO detection on current "
                            "and subsequent frames");
                    }

                    // YOLO parameter sliders
                    ImGui::SliderFloat("Confidence Threshold",
                                       &confidence_threshold, 0.01f, 0.99f,
                                       "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Minimum confidence score for detections");
                    }

                    ImGui::SliderFloat("NMS IoU Threshold", &iou_threshold,
                                       0.01f, 0.99f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Non-Maximum Suppression IoU threshold");
                    }

#ifndef __APPLE__
                    if (ImGui::Button("Run YOLO Prediction")) {
                        std::cout << "Running YOLO prediction on frame "
                                  << ps.to_display_frame_number << std::endl;

                        yolo_detection = true;

                        // Clear existing bounding boxes for current frame
                        // before running inference
                        if (skeleton.has_bbox) {
                            bool keypoints_find =
                                keypoints_map.find(current_frame_num) !=
                                keypoints_map.end();
                            if (keypoints_find) {
                                for (int cam_id = 0; cam_id < scene->num_cams;
                                     cam_id++) {
                                    auto &bbox_list =
                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list[cam_id];
                                    // Clear all bounding boxes for this
                                    // camera
                                    for (auto &bbox : bbox_list) {
                                        if (bbox.rect) {
                                            delete bbox.rect;
                                            bbox.rect = nullptr;
                                        }
                                        if (bbox.has_bbox_keypoints &&
                                            bbox.bbox_keypoints2d) {
                                            free(bbox.bbox_keypoints2d);
                                            bbox.bbox_keypoints2d = nullptr;
                                            free(bbox.active_kp_id);
                                            bbox.active_kp_id = nullptr;
                                        }
                                    }
                                    bbox_list.clear();
                                }
                                std::cout << "Cleared existing bounding "
                                             "boxes for frame "
                                          << current_frame_num << std::endl;
                            }
                        }
                        yolo_processed_frames.insert(current_frame_num);
                        for (int cam_id = 0; cam_id < scene->num_cams;
                             cam_id++) {
                            if (ps.pause_seeked) {
                                unsigned char *frame_data =
                                    scene
                                        ->display_buffer[cam_id]
                                                        [select_corr_head]
                                        .frame;

                                if (frame_data) {
                                    yolo_predictions[cam_id] = runYoloInference(
                                        yolo_model_path, frame_data,
                                        scene->image_width[cam_id],
                                        scene->image_height[cam_id]);

                                    yolo_bboxes[cam_id].clear();
                                    for (const auto &pred :
                                         yolo_predictions[cam_id]) {
                                        yolo_bboxes[cam_id].emplace_back(pred);
                                    }
                                }
                            } else {
                                if (window_was_decoding
                                        [pm.camera_names[cam_id]]) {
                                    unsigned char *frame_data =
                                        scene
                                            ->display_buffer[cam_id]
                                                            [select_corr_head]
                                            .frame;

                                    if (frame_data) {
                                        yolo_predictions[cam_id] =
                                            runYoloInference(
                                                yolo_model_path, frame_data,
                                                scene->image_width[cam_id],
                                                scene->image_height[cam_id]);

                                        yolo_bboxes[cam_id].clear();
                                        for (const auto &pred :
                                             yolo_predictions[cam_id]) {
                                            yolo_bboxes[cam_id].emplace_back(
                                                pred);
                                        }
                                    }
                                }
                            }
                        }

                        if (!yolo_bboxes.empty() &&
                            std::any_of(yolo_bboxes.begin(), yolo_bboxes.end(),
                                        [](const auto &cam_bboxes) {
                                            return !cam_bboxes.empty();
                                        })) {
                            // Convert YOLO detections to main bounding box
                            // system
                            for (int cam_id = 0; cam_id < scene->num_cams;
                                 cam_id++) {
                                if (!yolo_bboxes[cam_id].empty()) {
                                    // Ensure keypoints structure exists
                                    bool keypoints_find =
                                        keypoints_map.find(current_frame_num) !=
                                        keypoints_map.end();
                                    if (!keypoints_find) {
                                        KeyPoints *keypoints =
                                            (KeyPoints *)malloc(
                                                sizeof(KeyPoints));
                                        allocate_keypoints(keypoints, scene,
                                                           &skeleton);
                                        keypoints_map[current_frame_num] =
                                            keypoints;
                                    }

                                    // Add YOLO detections to main bounding
                                    // box system
                                    int yolo_bbox_id = 0;
                                    for (const auto &yolo_bbox :
                                         yolo_bboxes[cam_id]) {
                                        if (yolo_bbox.is_valid) {
                                            while (yolo_bbox.class_id >=
                                                   (int)bbox_class_colors.size()) {
                                                create_new_bbox_class();
                                            }

                                            BoundingBox bbox;

                                            // Create ImPlotRect from YOLO
                                            // coordinates (no Y-axis
                                            // flipping)
                                            bbox.rect = new ImPlotRect(
                                                yolo_bbox.x_min, // X.Min
                                                yolo_bbox.x_max, // X.Max
                                                yolo_bbox.y_min, // Y.Min
                                                yolo_bbox.y_max  // Y.Max
                                            );

                                            bbox.state = RectTwoPoints;
                                            bbox.class_id = yolo_bbox.class_id;
                                            bbox.id = yolo_bbox_id++;
                                            bbox.confidence =
                                                yolo_bbox.confidence;
                                            bbox.has_bbox_keypoints = false;
                                            bbox.bbox_keypoints2d = nullptr;
                                            bbox.active_kp_id = nullptr;

                                            // Allocate keypoints if
                                            // skeleton supports both bbox
                                            // and skeleton
                                            if (skeleton.has_bbox &&
                                                skeleton.has_skeleton &&
                                                skeleton.num_nodes > 0) {
                                                allocate_bbox_keypoints(
                                                    &bbox, scene, &skeleton);
                                            }

                                            // Add to main bounding box
                                            // system
                                            keypoints_map[current_frame_num]
                                                ->bbox2d_list[cam_id]
                                                .push_back(bbox);
                                        }
                                    }

                                    std::cout << "Added "
                                              << yolo_bboxes[cam_id].size()
                                              << " YOLO detections to main "
                                                 "bounding "
                                                 "box system for camera "
                                              << cam_id << std::endl;
                                }
                            }
                        }
                    }
#endif // !__APPLE__
                }
            }
            ImGui::End();

            // T-key triangulation: runs even when Labeling Tool tab is hidden
            if (keypoints_find &&
                ImGui::IsKeyPressed(ImGuiKey_T, false) &&
                !io.WantTextInput) {
                reprojection(keypoints_map.at(current_frame_num),
                             &skeleton, pm.camera_params, scene);
            }

            // Ctrl+S save: runs even when Labeling Tool tab is hidden
            if (save_requested) {
                if (skeleton.has_skeleton && !skeleton.has_bbox) {
                    save_keypoints(keypoints_map, &skeleton,
                                   pm.keypoints_root_folder,
                                   scene->num_cams, pm.camera_names,
                                   &input_is_imgs, imgs_names);
                    std::cout << "Saved skeleton keypoints data"
                              << std::endl;
                } else if (!skeleton.has_skeleton && skeleton.has_bbox) {
                    save_bboxes(keypoints_map, &skeleton,
                                pm.keypoints_root_folder, scene->num_cams,
                                pm.camera_names, &input_is_imgs,
                                imgs_names);
                    std::cout << "Saved bounding boxes data" << std::endl;
                } else if (skeleton.has_obb) {
                    save_obb(keypoints_map, &skeleton,
                             pm.keypoints_root_folder, pm.camera_names,
                             scene->num_cams, &imgs_names, &input_is_imgs,
                             bbox_class_names);
                    std::cout << "Saved oriented bounding boxes data"
                              << std::endl;
                } else if (skeleton.has_skeleton && skeleton.has_bbox) {
                    bool has_bbox_keypoints = false;
                    for (const auto &[frame_num, keypoints] :
                         keypoints_map) {
                        if (!keypoints)
                            continue;
                        for (int cam_id = 0;
                             cam_id < scene->num_cams &&
                             cam_id < MAX_VIEWS && !has_bbox_keypoints;
                             cam_id++) {
                            for (const auto &bbox :
                                 keypoints->bbox2d_list[cam_id]) {
                                if (bbox.state == RectTwoPoints &&
                                    bbox.has_bbox_keypoints) {
                                    has_bbox_keypoints = true;
                                    break;
                                }
                            }
                        }
                        if (has_bbox_keypoints)
                            break;
                    }

                    if (has_bbox_keypoints) {
                        save_keypoints(keypoints_map, &skeleton,
                                       pm.keypoints_root_folder,
                                       scene->num_cams, pm.camera_names,
                                       &input_is_imgs, imgs_names);
                        save_bbox_keypoints(
                            keypoints_map, &skeleton,
                            pm.keypoints_root_folder, scene->num_cams,
                            pm.camera_names, &input_is_imgs, imgs_names);
                        std::cout << "Saved skeleton keypoints and "
                                     "bounding box "
                                     "keypoints data"
                                  << std::endl;
                    } else {
                        save_keypoints(keypoints_map, &skeleton,
                                       pm.keypoints_root_folder,
                                       scene->num_cams, pm.camera_names,
                                       &input_is_imgs, imgs_names);
                        save_bboxes(keypoints_map, &skeleton,
                                    pm.keypoints_root_folder,
                                    scene->num_cams, pm.camera_names,
                                    &input_is_imgs, imgs_names);
                        std::cout << "Saved skeleton keypoints and "
                                     "bounding boxes data"
                                  << std::endl;
                    }
                } else {
                    save_keypoints(keypoints_map, &skeleton,
                                   pm.keypoints_root_folder,
                                   scene->num_cams, pm.camera_names,
                                   &input_is_imgs, imgs_names);
                    std::cout << "Saved skeleton keypoints data (fallback)"
                              << std::endl;
                }

                last_saved = time(NULL);
                toasts.pushSuccess("Labels saved");
            }
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseKeypointsFolder", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                pm.keypoints_root_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("LoadFromSelected", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto selected_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                free_all_keypoints(keypoints_map, scene);
                std::string err;
                if (load_keypoints(selected_folder, keypoints_map, &skeleton,
                                   scene, pm.camera_names, err,
                                   bbox_class_names)) {
                    free_all_keypoints(keypoints_map, scene);
                    popups.pushError(err);
                }
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        // YOLO/JARVIS/Calibration dialog handlers are now inside their respective Draw functions

        if (ImGui::IsKeyPressed(ImGuiKey_H, false) && !io.WantTextInput) {
            show_help_window = !show_help_window;
        }

        DrawHelpWindow(show_help_window);

        DrawSkeletonCreatorWindow(skeleton_creator_state, skeleton_dir);

        DrawLiveTable(table, "Spreadsheet", scene, dc_context->video_fps, ps,
                      pm.project_path);

        DrawYoloExportWindow(yolo_export_state, pm, skeleton_file_path, yolo_model_dir);

        DrawJarvisExportWindow(jarvis_export_state, pm, skeleton);

        DrawCalibrationToolWindow(calib_state, pm, ps, scene, dc_context,
                                  user_settings, red_data_dir, imgs_names, calib_cb
#ifdef __APPLE__
                                  , mac_last_uploaded_frame
#endif
        );

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
    cleanup_yolo_drag_boxes(); // Clean up YOLO drag boxes memory
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
