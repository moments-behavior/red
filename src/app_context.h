#pragma once
#include "imgui.h"
#include "imgui_internal.h"
#include "deferred_queue.h"
#include "gui/popup_stack.h"
#include "gui/toast.h"
#include "annotation.h"
#include "annotation_csv.h"
#include "media_loader.h"
#include "project.h"
#include "render.h"
#include "skeleton.h"
#include "user_settings.h"
#include "utils.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct DisplayState {
    int brightness = 0;
    float contrast = 1.0f;
    bool pivot_midgray = true;
    bool show_keypoints = true;
    bool show_masks = true;
    bool show_bboxes = true;
};

struct AppContext {
    // Core project/scene
    ProjectManager &pm;
    PlaybackState &ps;
    RenderScene *scene;
    DecoderContext *dc_context;

    // Skeleton + keypoints
    SkeletonContext &skeleton;
    std::map<std::string, SkeletonPrimitive> &skeleton_map;

    // Annotation model
    AnnotationMap &annotations;

    // UI infrastructure
    PopupStack &popups;
    ToastQueue &toasts;
    DeferredQueue &deferred;

    // Settings + paths
    UserSettings &user_settings;
    std::string &red_data_dir;
    std::string &skeleton_dir;

    // Media state
    std::vector<std::string> &imgs_names;
    std::vector<FFmpegDemuxer *> &demuxers;
    std::vector<std::thread> &decoder_threads;
    std::vector<bool> &is_view_focused;
    std::unordered_map<std::string, bool> &window_was_decoding;
    bool &input_is_imgs;
    int &label_buffer_size;
    int &current_frame_num;

    // Display
    DisplayState &display;

    // Layout
    gx_context *window;

    // Save trigger (set by toolbar Save button, consumed by labeling tool)
    bool &save_requested;

    // Per-project ini path (lives in main, referenced here)
    std::string &project_ini_path;
    bool &main_loop_running;


#ifdef __APPLE__
    std::vector<int> &mac_last_uploaded_frame;
#endif
};

// --- Free functions replacing lambdas that captured main() locals ---

// Copy the shipped default_imgui_layout.ini into a project folder.
// No-op if ini already exists.
inline void copy_default_layout_to_project(const AppContext &ctx, const std::string &proj_path) {
    namespace fs = std::filesystem;
    fs::path dest = fs::path(proj_path) / "imgui_layout.ini";
    if (fs::exists(dest))
        return;
    for (const auto &candidate : {
             ctx.window->exe_dir + "/../default_imgui_layout.ini",
             ctx.window->exe_dir + "/../share/red/default_imgui_layout.ini",
         }) {
        if (fs::exists(candidate)) {
            std::error_code ec;
            fs::copy_file(candidate, dest, ec);
            break;
        }
    }
}

// Migrate a single window header in an ini string. Returns true if content was modified.
inline bool migrate_ini_section(std::string &content,
                                const std::string &old_header,
                                const std::string &new_header) {
    size_t old_pos = content.find(old_header);
    if (old_pos == std::string::npos) return false;
    size_t section_end = content.find("\n[", old_pos + 1);
    if (section_end == std::string::npos)
        section_end = content.size();
    else
        section_end += 1;
    if (content.find(new_header) != std::string::npos)
        content.erase(old_pos, section_end - old_pos);
    else
        content.replace(old_pos, old_header.size(), new_header);
    return true;
}

// Migrate renamed windows in project ini files so saved dock settings carry over.
inline void migrate_ini_window_names(const std::string &ini_path) {
    if (!std::filesystem::exists(ini_path)) return;
    std::ifstream in(ini_path);
    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    in.close();

    bool changed = false;
    // v1 → v2: File Browser → Navigator (then Navigator → Controls below)
    changed |= migrate_ini_section(content,
        "[Window][File Browser]", "[Window][Navigator]");
    // v2 → v3: Navigator → Controls
    changed |= migrate_ini_section(content,
        "[Window][Navigator]", "[Window][Controls]");

    // v3 → v4: Remove [Window][Controls] entirely (no longer a window)
    {
        const std::string header = "[Window][Controls]";
        size_t pos = content.find(header);
        if (pos != std::string::npos) {
            size_t section_end = content.find("\n[", pos + 1);
            if (section_end == std::string::npos)
                section_end = content.size();
            else
                section_end += 1;
            content.erase(pos, section_end - pos);
            changed = true;
        }
    }

    // v5 → v6: Rename "Frames in the buffer" → "Frame Buffer"
    changed |= migrate_ini_section(content,
        "[Window][Frames in the buffer]", "[Window][Frame Buffer]");

    // v4 → v5: Sidebar dockspace 0x00000100 removed — remap tool windows
    // to main dockspace and strip the stale DockSpace node entry.
    {
        const std::string old_dock = "DockId=0x00000100";
        const std::string new_dock = "DockId=0x00000009";
        size_t pos = 0;
        while ((pos = content.find(old_dock, pos)) != std::string::npos) {
            content.replace(pos, old_dock.size(), new_dock);
            pos += new_dock.size();
            changed = true;
        }
        // Remove stale DockSpace node line for 0x00000100
        const std::string stale_node = "DockSpace ID=0x00000100";
        pos = content.find(stale_node);
        if (pos != std::string::npos) {
            size_t line_end = content.find('\n', pos);
            if (line_end != std::string::npos)
                line_end += 1;
            else
                line_end = content.size();
            content.erase(pos, line_end - pos);
            changed = true;
        }
    }

    if (changed) {
        std::ofstream out(ini_path);
        out << content;
    }
}

// Switch ImGui layout ini to the project folder.
inline void switch_ini_to_project(AppContext &ctx) {
    ctx.project_ini_path = ctx.pm.project_path + "/imgui_layout.ini";
    copy_default_layout_to_project(ctx, ctx.pm.project_path);
    migrate_ini_window_names(ctx.project_ini_path);
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = ctx.project_ini_path.c_str();
    if (ctx.main_loop_running && std::filesystem::exists(ctx.project_ini_path)) {
        ImGui::LoadIniSettingsFromDisk(ctx.project_ini_path.c_str());
        ImGuiContext* g = ImGui::GetCurrentContext();
        for (int i = 0; i < g->Windows.Size; i++) {
            ImGuiWindow* w = g->Windows[i];
            if (ImGuiWindowSettings* s = ImGui::FindWindowSettingsByWindow(w)) {
                w->DockId = s->DockId;
                w->DockOrder = s->DockOrder;
            }
        }
        for (int i = 0; i < g->DockContext.Nodes.Data.Size; i++) {
            ImGuiDockNode* node = (ImGuiDockNode*)g->DockContext.Nodes.Data[i].val_p;
            if (node)
                node->LastFrameAlive = g->FrameCount;
        }
    }
}

// Close project: auto-save, unload media, reset all project state.
inline void close_project(AppContext &ctx) {
    // 1. Auto-save annotations if project is loaded
    if (!ctx.pm.keypoints_root_folder.empty() && !ctx.annotations.empty()) {
        std::string save_err;
        AnnotationCSV::save_all(ctx.pm.keypoints_root_folder,
            ctx.skeleton.name, ctx.annotations,
            ctx.scene->num_cams, ctx.skeleton.num_nodes,
            ctx.pm.camera_names, &save_err);
    }

    // 2. Save ImGui ini
    ImGui::SaveIniSettingsToDisk(ImGui::GetIO().IniFilename);

    // 3. Unload media (stops threads, frees GPU)
    unload_media(ctx.ps, ctx.pm, ctx.demuxers, ctx.dc_context,
                 ctx.scene, ctx.decoder_threads, ctx.is_view_focused,
                 ctx.window_was_decoding);

    // 4. Clear annotations
    ctx.annotations.clear();

    // 5. Reset skeleton
    ctx.skeleton.num_nodes = 0;
    ctx.skeleton.num_edges = 0;
    ctx.skeleton.name.clear();
    ctx.skeleton.has_skeleton = false;
    ctx.skeleton.node_colors.clear();
    ctx.skeleton.edges.clear();
    ctx.skeleton.node_names.clear();

    // 6. Reset ProjectManager (preserve nothing)
    ctx.pm.project_path.clear();
    ctx.pm.project_name.clear();
    ctx.pm.project_root_path.clear();
    ctx.pm.calibration_folder.clear();
    ctx.pm.keypoints_root_folder.clear();
    ctx.pm.camera_params.clear();
    ctx.pm.camera_names.clear();
    ctx.pm.media_folder.clear();
    ctx.pm.skeleton_name.clear();
    ctx.pm.skeleton_file.clear();
    ctx.pm.load_skeleton_from_json = false;
    ctx.pm.plot_keypoints_flag = false;
    ctx.pm.show_project_window = false;
    ctx.pm.telecentric = false;
    ctx.pm.annotation_config = AnnotationConfig{};
    ctx.pm.jarvis_models.clear();
    ctx.pm.active_jarvis_model = -1;

    // 7. Reset display state (project-specific: different videos need different settings)
    ctx.display = DisplayState{};

    // 8. Reset frame state
    ctx.current_frame_num = 0;

    // 8. Clear media state
    ctx.imgs_names.clear();
    ctx.input_is_imgs = false;

    // 9. Reset mac state
#ifdef __APPLE__
    for (auto &v : ctx.mac_last_uploaded_frame) v = -1;
#endif

    // 10. Clear deferred queue
    ctx.deferred.flush();
}

// Post-project-load sequence: switch ini, load videos, load labels.
// print_metadata_fn: callback for print_video_metadata (static in red.cpp)
// print_summary_fn: callback for print_project_summary (static in red.cpp)
//   signature: void(const std::string &most_recent_folder)
inline void on_project_loaded(AppContext &ctx,
                              std::function<void()> print_metadata_fn = nullptr,
                              std::function<void(const std::string &)> print_summary_fn = nullptr) {
    // Note: close_project() should be called BEFORE this function for
    // project switching. It handles unload_media + annotations.clear.
    // For the startup path (no prior project), unload_media is a no-op.
    if (ctx.ps.video_loaded) {
        unload_media(ctx.ps, ctx.pm, ctx.demuxers, ctx.dc_context,
                     ctx.scene, ctx.decoder_threads,
                     ctx.is_view_focused, ctx.window_was_decoding);
    }
    ctx.annotations.clear();

    switch_ini_to_project(ctx);
    int expected_cameras = (int)ctx.pm.camera_names.size();
    std::map<std::string, std::string> empty_selected_files;
    load_videos(empty_selected_files, ctx.ps, ctx.pm,
                ctx.window_was_decoding, ctx.demuxers, ctx.dc_context,
                ctx.scene, ctx.label_buffer_size, ctx.decoder_threads,
                ctx.is_view_focused);
    if (print_metadata_fn) print_metadata_fn();
    int loaded_cameras = (int)ctx.pm.camera_names.size();
    if (loaded_cameras < expected_cameras) {
        int skipped = expected_cameras - loaded_cameras;
        ctx.toasts.push(
            std::to_string(skipped) + " camera(s) skipped (broken video headers). "
            "Create a new project without those videos.",
            Toast::Warning, 10.0f);
    }
    std::string label_err;
    std::string most_recent_folder;
    if (!AnnotationCSV::find_most_recent_labels(ctx.pm.keypoints_root_folder,
                                                 most_recent_folder, label_err)) {
        // Check for v1 format and convert if needed
        if (AnnotationCSV::is_v1_format(most_recent_folder, ctx.pm.camera_names)) {
            std::string v2_folder = most_recent_folder + "_v2";
            std::string conv_err;
            if (!AnnotationCSV::convert_v1_to_v2(most_recent_folder, v2_folder,
                                                   ctx.skeleton.name,
                                                   ctx.skeleton.num_nodes,
                                                   ctx.pm.camera_names, &conv_err)) {
                ctx.popups.pushError("V1 conversion failed: " + conv_err);
            } else {
                most_recent_folder = v2_folder;
            }
        }
        // Load v2 annotations
        ctx.annotations.clear();
        int num_cameras = ctx.scene ? (int)ctx.scene->num_cams : 0;
        if (AnnotationCSV::load_all(most_recent_folder, ctx.annotations,
                                      ctx.skeleton.name,
                                      ctx.skeleton.num_nodes, num_cameras,
                                      ctx.pm.camera_names, label_err)) {
            ctx.popups.pushError(label_err);
            ctx.annotations.clear();
        }
    }
    if (print_summary_fn) print_summary_fn(most_recent_folder);

    // Track in recent projects
    if (!ctx.pm.project_path.empty()) {
        std::string redproj = ctx.pm.project_path + "/" + ctx.pm.project_name + ".redproj";
        ctx.user_settings.push_recent_project(redproj);
        save_user_settings(ctx.user_settings);
    }
}
