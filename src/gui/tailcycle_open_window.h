#ifndef RED_TAILCYCLE_OPEN_WINDOW
#define RED_TAILCYCLE_OPEN_WINDOW

// Open a tailcycle-dataset session for viewing.
//
// This is NOT the predictions importer. It opens a session someone else may
// have produced -- its own cameras, its own skeleton, its own frames -- rather
// than merging labels into the project you already have open.
//
// No project is written to disk. The session's group folder IS the media
// folder (media_loader reads <cam>/<frame>.jpg directly), the calibration comes
// from calibration.toml, and the skeleton from `names`, so opening one is
// read-only until you choose to save.

#include "app_context.h"
#include "imgui.h"
#include "media_loader.h"
#include "tailcycle_import.h"
#include "gui/panel.h"
#include "ImGuiFileDialog.h"
#include "misc/cpp/imgui_stdlib.h"

#include <filesystem>
#include <string>
#include <vector>

struct TailcycleOpenState {
    bool enabled = false;
    std::string session_dir;
    std::vector<std::string> groups;
    int group_idx = 0;
    std::string scanned_dir;      // which dir `groups` was listed from
    std::string status;
};

// Build a SkeletonContext from names+edges alone. A session carries its own
// keypoint axis and need not match any preset red ships -- §4 makes `names`
// the authority, and refusing to open a session whose skeleton red does not
// already know would defeat the point of reading other people's data.
inline void tailcycle_skeleton_from_session(const TailcycleImport::Session &s,
                                            SkeletonContext &out) {
    out = SkeletonContext{};
    out.has_skeleton = true;
    out.name = "tailcycle:" + s.session_id;
    out.node_names = s.node_names;
    out.num_nodes = (int)s.node_names.size();
    for (const auto &e : s.edges) out.edges.push_back({e.first, e.second});
    out.num_edges = (int)out.edges.size();
    out.node_colors.resize(out.num_nodes, ImVec4(1, 1, 1, 1));
    apply_keypoint_colormap(out, g_keypoint_colormap);
}

inline bool tailcycle_open_session(AppContext &ctx, const std::string &session_dir,
                                   const std::string &group_id, std::string *status) {
    namespace fs = std::filesystem;
    TailcycleImport::Session s;
    TailcycleImport::ImportStats st;
    if (!TailcycleImport::read_session(session_dir, group_id, &s, &st, status))
        return false;

    const fs::path gdir = fs::path(session_dir) / "groups" / s.group_id;
    if (!fs::is_directory(gdir)) {
        if (status) *status = "Group folder missing: " + gdir.string();
        return false;
    }
    // Rule 7: a group holds camera directories OR <cam>.mp4, never both. Which
    // one decides how the frames are read.
    bool has_dirs = false, has_videos = false;
    for (const auto &cam : s.camera_names) {
        if (fs::is_directory(gdir / cam)) has_dirs = true;
        if (fs::exists(gdir / (cam + ".mp4"))) has_videos = true;
    }
    if (has_dirs && has_videos) {
        if (status) *status = "Group holds both image directories and videos (rule 7).";
        return false;
    }
    if (!has_dirs && !has_videos) {
        if (status) *status = "Group holds no media for the declared cameras.";
        return false;
    }

    close_project(ctx);

    ctx.pm = ProjectManager{};
    ctx.pm.project_name = s.session_id;
    ctx.pm.project_path = session_dir;
    ctx.pm.media_folder = gdir.string();
    ctx.pm.camera_names = s.camera_names;
    ctx.pm.camera_params = s.calibration;
    ctx.pm.skeleton_name = "tailcycle:" + s.session_id;
    // Saving edits would write red's own CSVs beside the session rather than
    // back into the Parquet, so it is deliberately pointed inside the session.
    ctx.pm.keypoints_root_folder = (fs::path(session_dir) / "red_labels").string();

    tailcycle_skeleton_from_session(s, ctx.skeleton);
    ctx.annotations = s.annotations;
    ctx.input_is_imgs = has_dirs;

    if (has_dirs) {
        std::map<std::string, std::string> files;
        std::string err;
        if (!scan_per_camera_dirs(ctx.pm.media_folder, files, &err)) {
            if (status) *status = err;
            return false;
        }
        ctx.imgs_names.clear();
        load_images(files, ctx.ps, ctx.pm, ctx.imgs_names, ctx.scene, ctx.dc_context,
                    ctx.label_buffer_size, ctx.decoder_threads, ctx.is_view_focused,
                    ctx.window_was_decoding, ImageLayout::PerCameraDir);
    } else {
        std::map<std::string, std::string> none;
        load_videos(none, ctx.ps, ctx.pm, ctx.window_was_decoding, ctx.demuxers,
                    ctx.dc_context, ctx.scene, ctx.label_buffer_size,
                    ctx.decoder_threads, ctx.is_view_focused);
    }

    if (status)
        *status = "Opened " + s.session_id + "/" + s.group_id + " — " +
                  std::to_string(s.camera_names.size()) + " cameras, " +
                  std::to_string(s.n_frames) + " frames, " +
                  std::to_string(st.keypoint_rows) + " 2D and " +
                  std::to_string(st.points3d_rows) + " 3D labels" +
                  (has_dirs ? " (images)" : " (videos)");
    return true;
}

inline void DrawTailcycleOpenWindow(TailcycleOpenState &state, AppContext &ctx) {
    DrawPanel("Open tailcycle Dataset", state.enabled, [&]() {
        if (!TailcycleImport::available()) {
            ImGui::TextWrapped("This build has no Parquet support (Arrow was not found "
                               "at configure time).");
            return;
        }
        ImGui::TextDisabled("Point at a session folder: <dataset>/<split>/<session>/");
        ImGui::InputText("Session folder", &state.session_dir);
        ImGui::SameLine();
        if (ImGui::Button("Browse##tc_open")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = state.session_dir;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog("ChooseTailcycleSession",
                                                    "Choose Session Folder", nullptr, cfg);
        }

        // A session may hold several groups; a red project is one media folder,
        // so one has to be chosen.
        if (state.session_dir != state.scanned_dir) {
            state.scanned_dir = state.session_dir;
            state.groups.clear();
            state.group_idx = 0;
            std::string err;
            TailcycleImport::list_groups(state.session_dir, &state.groups, &err);
        }
        if (state.groups.size() > 1) {
            std::vector<const char *> labels;
            for (const auto &g : state.groups) labels.push_back(g.c_str());
            ImGui::Combo("Group", &state.group_idx, labels.data(), (int)labels.size());
        } else if (state.groups.size() == 1) {
            ImGui::Text("Group: %s", state.groups[0].c_str());
        }

        ImGui::Separator();
        ImGui::BeginDisabled(state.groups.empty());
        if (ImGui::Button("Open")) {
            const std::string gid =
                state.groups.empty() ? std::string()
                                     : state.groups[(size_t)state.group_idx];
            tailcycle_open_session(ctx, state.session_dir, gid, &state.status);
        }
        ImGui::EndDisabled();
        if (state.groups.empty() && !state.session_dir.empty())
            ImGui::TextDisabled("No groups/ found here — is this a session folder?");

        if (!state.status.empty()) {
            const bool bad = state.status.rfind("Opened", 0) != 0;
            ImGui::TextColored(bad ? ImVec4(1.0f, 0.45f, 0.35f, 1.0f)
                                   : ImVec4(0.4f, 0.9f, 0.5f, 1.0f),
                               "%s", state.status.c_str());
        }
        },
        [&]() {
        if (ImGuiFileDialog::Instance()->Display("ChooseTailcycleSession",
                                                 ImGuiWindowFlags_NoCollapse,
                                                 ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk())
                state.session_dir = ImGuiFileDialog::Instance()->GetCurrentPath();
            ImGuiFileDialog::Instance()->Close();
        }
        },
        ImVec2(560, 300));
}

#endif // RED_TAILCYCLE_OPEN_WINDOW
