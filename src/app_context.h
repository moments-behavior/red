#pragma once
#include "imgui.h"
#include "imgui_internal.h"
#include "deferred_queue.h"
#include "gui/popup_stack.h"
#include "gui/toast.h"
#include "gui/gui_save_load.h"
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
    std::map<u32, KeyPoints *> &keypoints_map;

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

// Post-project-load sequence: switch ini, load videos, load labels.
// print_metadata_fn: callback for print_video_metadata (static in red.cpp)
// print_summary_fn: callback for print_project_summary (static in red.cpp)
//   signature: void(const std::string &most_recent_folder)
inline void on_project_loaded(AppContext &ctx,
                              std::function<void()> print_metadata_fn = nullptr,
                              std::function<void(const std::string &)> print_summary_fn = nullptr) {
    switch_ini_to_project(ctx);
    std::map<std::string, std::string> empty_selected_files;
    load_videos(empty_selected_files, ctx.ps, ctx.pm,
                ctx.window_was_decoding, ctx.demuxers, ctx.dc_context,
                ctx.scene, ctx.label_buffer_size, ctx.decoder_threads,
                ctx.is_view_focused);
    if (print_metadata_fn) print_metadata_fn();
    std::string label_err;
    std::string most_recent_folder;
    if (!find_most_recent_labels(ctx.pm.keypoints_root_folder,
                                 most_recent_folder, label_err)) {
        if (load_keypoints(most_recent_folder, ctx.keypoints_map,
                           &ctx.skeleton, ctx.scene, ctx.pm.camera_names,
                           label_err)) {
            free_all_keypoints(ctx.keypoints_map, ctx.scene);
            ctx.popups.pushError(label_err);
        }
    }
    if (print_summary_fn) print_summary_fn(most_recent_folder);
}
