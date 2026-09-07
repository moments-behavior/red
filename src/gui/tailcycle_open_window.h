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
#include "gui/gui_keypoints.h"
#include "tailcycle_export.h"
#include "tailcycle_import.h"
#include "gui/panel.h"
#include "ImGuiFileDialog.h"
#include "misc/cpp/imgui_stdlib.h"

#include <filesystem>
#include <string>
#include <vector>

struct TailcycleOpenState {
    bool show = false;
    std::string root;                 // the dataset root, or a session folder
    std::vector<TailcycleImport::SessionInfo> sessions;
    std::string scanned;              // which root `sessions` came from
    int group_idx = 0;                // for a session holding several groups
    int selected = -1;                // row currently open
    std::string status;

    // Enough of the open session to write corrections back over it. Held
    // rather than re-read because a save must reproduce the session's own
    // shape -- its group id, frame range and which label layers it carried --
    // not whatever an exporter would choose afresh.
    bool     open_valid = false;
    std::string open_dir, open_split, open_session_id, open_group_id;
    std::string open_source_video, open_units, open_labels;
    int      open_n_frames = 0, open_source_frame_start = 0;
    float    open_fps = 0.0f;
    bool     open_has_2d = false, open_has_3d = false;
    std::vector<std::string> open_camera_names;
    std::vector<CameraParams> open_calibration;
    std::vector<std::string> open_node_names;
    std::vector<std::pair<int,int>> open_edges;
    std::vector<std::string> open_animal_ids;
    bool     confirm_save = false;
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
                                   const std::string &group_id, std::string *status,
                                   TailcycleOpenState *remember = nullptr) {
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
    // Deliberately left empty: a session opens READ-ONLY.
    //
    // close_project() auto-saves whenever this is set and annotations exist,
    // which is always true after opening a session -- so pointing it anywhere
    // inside the dataset made red write CSVs into data it did not author,
    // every time a session was opened or switched.
    //
    // Round-tripping edits belongs in a tailcycle export, which writes the
    // format the session is already in, rather than red's CSVs beside it.
    ctx.pm.keypoints_root_folder.clear();
    // What load_project sets on a normal open. Without the first of these the
    // labelling overlay never draws, so the frames appear with no labels on
    // them and nothing says why.
    ctx.pm.plot_keypoints_flag = true;
    ctx.pm.show_project_window = false;

    tailcycle_skeleton_from_session(s, ctx.skeleton);
    ctx.annotations = s.annotations;

    // A session may carry 3D and no per-camera 2D at all -- johnson-mouse-tracked
    // is exactly that. §8 expects a consumer to derive what it needs ("2D from 3D
    // by projection"), and without it the camera views open with no labels on
    // them. red already does this for imported JARVIS predictions, and marks the
    // result Predicted, so follow that.
    //
    // These are derived, not observations. Re-exporting this session would write
    // them as real 2D rows, which §8 says not to store -- so treat an imported
    // 3D-only session as something to look at rather than a round trip.
    int reprojected = 0;
    if (!s.has_2d && s.has_3d) {
        for (auto &[frame, fis] : ctx.annotations)
          for (auto &fa : fis) {
            for (size_t n = 0; n < fa.kp3d.size(); n++) {
                const Keypoint3D &k3 = fa.kp3d[n];
                if (k3.source == Kp3DSource::None) continue;
                const Eigen::Vector3d p3d(k3.x, k3.y, k3.z);
                for (size_t c = 0; c < fa.cameras.size() && c < s.calibration.size(); c++) {
                    double px = 0, py = 0;
                    if (!reproject_3d_to_cam(p3d, s.calibration[c],
                                             s.calibration[c].image_width,
                                             s.calibration[c].image_height, px, py))
                        continue;   // behind the camera or outside the frame
                    Keypoint2D &kp = fa.cameras[c].keypoints[n];
                    kp.x = px;
                    kp.y = py;
                    kp.labeled = true;
                    kp.source = LabelSource::Predicted;
                    kp.confidence = k3.confidence;
                    reprojected++;
                }
            }
        }
    }
    ctx.input_is_imgs = has_dirs;

    if (has_dirs) {
        std::map<std::string, std::string> files;
        std::string err;
        if (!scan_per_camera_dirs(ctx.pm.media_folder, files, &err)) {
            if (status) *status = err;
            return false;
        }
        ctx.imgs_names.clear();
        // groups.pq declares the source recording's fps, so the clock-paced
        // playback speeds mean something here even though the frames are stills.
        load_images(files, ctx.ps, ctx.pm, ctx.imgs_names, ctx.scene, ctx.dc_context,
                    ctx.label_buffer_size, ctx.decoder_threads, ctx.is_view_focused,
                    ctx.window_was_decoding, ImageLayout::PerCameraDir, s.fps);
    } else {
        std::map<std::string, std::string> none;
        load_videos(none, ctx.ps, ctx.pm, ctx.window_was_decoding, ctx.demuxers,
                    ctx.dc_context, ctx.scene, ctx.label_buffer_size,
                    ctx.decoder_threads, ctx.is_view_focused);
    }

    if (remember) {
        remember->open_valid = true;
        remember->open_dir = session_dir;
        remember->open_split = s.split;
        remember->open_session_id = s.session_id;
        remember->open_group_id = s.group_id;
        remember->open_source_video = s.source_video;
        remember->open_units = s.units;
        remember->open_labels = s.labels;
        remember->open_n_frames = s.n_frames;
        remember->open_source_frame_start = s.source_frame_start;
        remember->open_fps = s.fps;
        remember->open_has_2d = s.has_2d;
        remember->open_has_3d = s.has_3d;
        remember->open_camera_names = s.camera_names;
        remember->open_calibration = s.calibration;
        remember->open_node_names = s.node_names;
        remember->open_edges = s.edges;
        remember->open_animal_ids = s.animal_ids;
    }

    if (status)
        *status = "Opened " + s.session_id + "/" + s.group_id + " — " +
                  std::to_string(s.camera_names.size()) + " cameras, " +
                  std::to_string(s.n_frames) + " frames, " +
                  std::to_string(st.keypoint_rows) + " 2D and " +
                  std::to_string(st.points3d_rows) + " 3D labels" +
                  (reprojected ? ", " + std::to_string(reprojected) +
                                     " 2D reprojected from 3D"
                               : std::string()) +
                  (has_dirs ? " (images)" : " (videos)");
    return true;
}

// Write corrections back over the session that was opened.
//
// Not the export window's job: that makes a NEW dataset and picks its own
// group id, frame range and layers. Saving in place has to reproduce this
// session's -- otherwise the tables would stop matching the frames sitting
// next to them in groups/.
//
// The media is untouched. export_session only ever creates groups/<id>/ and
// leaves it empty, so the frames already there survive.
inline bool tailcycle_save_session(AppContext &ctx, TailcycleOpenState &st,
                                   std::string *status) {
    namespace fs = std::filesystem;
    if (!st.open_valid) {
        if (status) *status = "No session is open.";
        return false;
    }
    TailcycleExport::ExportConfig cfg;
    // <dir> is <root>/<split>/<session>; export_session appends split/session,
    // so it has to start two levels up to land back on the same folder.
    cfg.output_folder = fs::path(st.open_dir).parent_path().parent_path().string();
    cfg.split = st.open_split;
    cfg.session_id = st.open_session_id;
    cfg.group_id = st.open_group_id;
    cfg.source_video = st.open_source_video;
    cfg.units = st.open_units.empty() ? "mm" : st.open_units;
    cfg.n_frames = st.open_n_frames;
    cfg.fps = st.open_fps;
    cfg.source_frame_start = st.open_source_frame_start;
    cfg.camera_names = st.open_camera_names;
    cfg.calibration = st.open_calibration;
    cfg.node_names = st.open_node_names;
    cfg.edges = st.open_edges;
    cfg.force_labels = st.open_labels.empty() ? "annotated" : st.open_labels;
    cfg.animal_ids = st.open_animal_ids;
    cfg.provenance_source = st.open_dir;
    // Keep the session's own shape: a 3D-only session stays 3D-only rather
    // than gaining a 2D layer red derived by reprojection (§8).
    cfg.layers = st.open_has_2d && st.open_has_3d
                     ? TailcycleExport::ExportConfig::Layers::TwoDAndThreeD
                 : st.open_has_3d ? TailcycleExport::ExportConfig::Layers::ThreeD
                                  : TailcycleExport::ExportConfig::Layers::TwoD;

    TailcycleExport::ExportStats est;
    return TailcycleExport::export_session(cfg, ctx.annotations, &est, status);
}

inline void DrawTailcycleOpenWindow(TailcycleOpenState &state, AppContext &ctx) {
    DrawPanel("Open tailcycle Dataset", state.show, [&]() {
        if (!TailcycleImport::available()) {
            ImGui::TextWrapped("This build has no Parquet support (Arrow was not found "
                               "at configure time).");
            return;
        }
        ImGui::TextDisabled("Point at a dataset root; a single session folder works too.");
        ImGui::InputText("Dataset", &state.root);
        ImGui::SameLine();
        if (ImGui::Button("Browse##tc_open")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = state.root;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog("ChooseTailcycleSession",
                                                    "Choose Dataset Folder", nullptr, cfg);
        }

        // Rescan when the path changes. The scan reads session.toml and one row
        // of groups.pq per session, never the label tables, so it stays quick
        // on a dataset whose tables run to millions of rows.
        if (state.root != state.scanned) {
            state.scanned = state.root;
            state.sessions.clear();
            state.selected = -1;
            state.group_idx = 0;
            if (!state.root.empty()) {
                std::string err;
                if (!TailcycleImport::scan_dataset(state.root, &state.sessions, &err))
                    state.status = err;
                // Open the first session straight away. Pointing at a dataset
                // is a request to look at it, and the list is for switching
                // between sessions rather than a gate in front of the first.
                if (!state.sessions.empty()) {
                    const auto &si = state.sessions.front();
                    if (tailcycle_open_session(
                            ctx, si.dir,
                            si.groups.empty() ? std::string() : si.groups[0],
                            &state.status, &state)) {
                        state.selected = 0;
                        ctx.user_settings.push_recent_project(state.root);
                        save_user_settings(ctx.user_settings);
                    }
                }
            }
        }

        if (state.sessions.empty()) {
            if (!state.root.empty())
                ImGui::TextDisabled("Nothing here — expected <split>/<session>/session.toml");
        } else {
            ImGui::Spacing();
            // ScrollX and fixed-fit sizing: docked into the left tab group the
            // panel is ~280px, far narrower than seven columns of session
            // metadata want. Scrolling sideways beats truncating names that
            // differ only in their tail.
            const ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                                       ImGuiTableFlags_SizingFixedFit |
                                       ImGuiTableFlags_ScrollX |
                                       ImGuiTableFlags_ScrollY;
            if (ImGui::BeginTable("##tc_sessions", 7, tf, ImVec2(0, 200))) {
                ImGui::TableSetupScrollFreeze(0, 1);
                for (const char *h : {"Split", "Session", "Labels", "Cams", "Frames",
                                      "Layers", "Groups"})
                    ImGui::TableSetupColumn(h);
                ImGui::TableHeadersRow();
                for (int i = 0; i < (int)state.sessions.size(); i++) {
                    const auto &si = state.sessions[i];
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::PushID(i);
                    if (ImGui::Selectable(si.split.c_str(), state.selected == i,
                                          ImGuiSelectableFlags_SpanAllColumns)) {
                        state.selected = i;
                        state.group_idx = 0;
                        const std::string gid =
                            si.groups.empty() ? std::string() : si.groups[0];
                        if (tailcycle_open_session(ctx, si.dir, gid, &state.status, &state)) {
                            // The ROOT goes in recents, not the session:
                            // reopening should bring back the list to choose
                            // from, which is how you actually use a dataset.
                            ctx.user_settings.push_recent_project(state.root);
                            save_user_settings(ctx.user_settings);
                        }
                    }
                    ImGui::PopID();
                    ImGui::TableNextColumn(); ImGui::TextUnformatted(si.session_id.c_str());
                    ImGui::TableNextColumn(); ImGui::TextUnformatted(si.labels.c_str());
                    ImGui::TableNextColumn(); ImGui::Text("%d", si.n_cameras);
                    ImGui::TableNextColumn(); ImGui::Text("%d", si.n_frames);
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(si.has_2d && si.has_3d ? "2D+3D"
                                           : si.has_2d           ? "2D"
                                           : si.has_3d           ? "3D"
                                                                 : "none");
                    ImGui::TableNextColumn(); ImGui::Text("%d", (int)si.groups.size());
                }
                ImGui::EndTable();
            }
            ImGui::TextDisabled("Click a row to open it.");

            // Only worth a control when there is a choice to make: a red project
            // is one media folder, so one group at a time.
            if (state.selected >= 0 &&
                state.sessions[state.selected].groups.size() > 1) {
                const auto &si = state.sessions[state.selected];
                std::vector<const char *> labels;
                for (const auto &g : si.groups) labels.push_back(g.c_str());
                if (ImGui::Combo("Group", &state.group_idx, labels.data(),
                                 (int)labels.size()))
                    tailcycle_open_session(ctx, si.dir, si.groups[(size_t)state.group_idx],
                                           &state.status, &state);
            }
        }

        // Saving corrections back over the session that is open. Separate
        // from the export window, which makes a NEW dataset -- this one
        // rewrites the tables in place and leaves the frames alone.
        if (state.open_valid) {
            ImGui::Separator();
            if (!state.confirm_save) {
                if (ImGui::Button("Save corrections to this session"))
                    state.confirm_save = true;
                ImGui::SameLine();
                ImGui::TextDisabled("%s/%s", state.open_split.c_str(),
                                    state.open_session_id.c_str());
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.35f, 1.0f),
                                   "Overwrite this session's label tables?");
                if (ImGui::Button("Overwrite")) {
                    state.confirm_save = false;
                    if (tailcycle_save_session(ctx, state, &state.status))
                        state.status = "Saved corrections to " + state.open_split +
                                       "/" + state.open_session_id;
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) state.confirm_save = false;
                ImGui::SameLine();
                ImGui::TextDisabled("frames are untouched");
            }
        }

        if (!state.status.empty()) {
            const bool bad = state.status.rfind("Opened", 0) != 0 &&
                             state.status.rfind("Saved", 0) != 0;
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
                state.root = ImGuiFileDialog::Instance()->GetCurrentPath();
            ImGuiFileDialog::Instance()->Close();
        }
        },
        ImVec2(720, 420));
}

#endif // RED_TAILCYCLE_OPEN_WINDOW
