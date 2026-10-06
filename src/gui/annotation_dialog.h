#pragma once
#include "imgui.h"
#include "app_context.h"
#include "gui/panel.h"
#include "gui/folder_dialog.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

// Discovery lives in media_loader.h now: a folder may hold videos, a
// directory per camera, or camera-prefixed image files, and which one it is
// has to be recorded on the project so reloading can do the same thing.

struct AnnotationDialogState {
    bool show = false;
    int skeleton_wait = -1;   // waiting on the Skeleton Creator (see "New...")
    bool was_shown = false;   // edge-detect the frame the dialog opens on
    MediaKind media_kind = MediaKind::Video;
    // Several cameras, not calibrated: 2D labels only. Chosen in the form
    // ("Cameras: Calibrated / Not calibrated"); one camera is always 2D.
    bool two_d_mode = false;
    std::string media_folder;
    std::vector<std::string> discovered_cameras;
    std::vector<bool> camera_selected;
    std::string status;

    // Opening the form is always a fresh start, and every entry point goes
    // through here. They used to each clear a different subset: the menu
    // cleared the camera lists but not the folder, both Welcome buttons
    // cleared nothing. Since the seed-from-open-media below only fires on an
    // empty folder, a leftover one made the field look filled with no cameras
    // under it, and re-picking that same folder was the only way back.
    void open() {
        show = true;
        two_d_mode = false;   // several cameras default to calibrated
        media_folder.clear();
        discovered_cameras.clear();
        camera_selected.clear();
        status.clear();
    }
};

// Callback signature: called after "Create Project" succeeds at setting up pm.
// The callback should do: switch_ini_to_project(), save .redproj, load_videos(), etc.
// Returns true on success, false on failure (sets error_message).
using AnnotationCreateCallback = std::function<bool(ProjectManager &pm, std::string &error_message)>;

inline void DrawAnnotationDialog(AnnotationDialogState &state,
                                 AppContext &ctx,
                                 const AnnotationCreateCallback &on_create) {
    auto &pm = ctx.pm;
    const auto &skeleton_map = ctx.skeleton_map;
    const auto &skeleton_dir = ctx.skeleton_dir;
    const std::string default_browse_path =
        ctx.user_settings.default_media_root_path.empty()
            ? ctx.default_dir
            : ctx.user_settings.default_media_root_path;
    // File dialog handlers (run every frame, even when window is hidden)
    if (display_folder_dialog("ChooseAnnotVideoDir")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::filesystem::path chosen(
                ImGuiFileDialog::Instance()->GetCurrentPath());
            state.media_folder = chosen.string();
            remember_media_dir(ctx, state.media_folder);
            state.discovered_cameras =
                discover_media_cameras(state.media_folder, &state.media_kind);
            state.camera_selected.assign(state.discovered_cameras.size(), true);
        }
        ImGuiFileDialog::Instance()->Close();
    }
    if (ImGuiFileDialog::Instance()->Display("ChooseAnnotRootDir", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk())
            pm.project_root_path = ImGuiFileDialog::Instance()->GetCurrentPath();
        ImGuiFileDialog::Instance()->Close();
    }
    if (ImGuiFileDialog::Instance()->Display("ChooseAnnotSkeleton", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            pm.skeleton_file = ImGuiFileDialog::Instance()->GetFilePathName();
            remember_skeleton_dir(ctx, pm.skeleton_file);
        }
        ImGuiFileDialog::Instance()->Close();
    }
    if (display_folder_dialog("ChooseAnnotCalib")) {
        if (ImGuiFileDialog::Instance()->IsOk())
            pm.calibration_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
        ImGuiFileDialog::Instance()->Close();
    }
    if (display_folder_dialog("ChooseAnnotTimestamps")) {
        if (ImGuiFileDialog::Instance()->IsOk())
            pm.timestamps_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
        ImGuiFileDialog::Instance()->Close();
    }

    // Seed the folder from whatever media is already open, the first frame
    // the dialog appears. Creating a project for footage you are looking at
    // was the whole point of File > Create Project (now File > New Project), which could only ever wrap
    // the open media and was greyed out otherwise; here it is a starting
    // value, and pointing the field somewhere else switches media instead of
    // being impossible. The empty check now only guards against re-seeding
    // while the form is already up -- open() clears the folder, so reopening
    // deliberately picks up whatever media is open NOW.
    if (state.show && !state.was_shown && state.media_folder.empty() &&
        ctx.ps.video_loaded && !pm.media_folder.empty()) {
        state.media_folder = pm.media_folder;
        state.discovered_cameras =
                discover_media_cameras(state.media_folder, &state.media_kind);
        state.camera_selected.assign(state.discovered_cameras.size(), true);
    }
    state.was_shown = state.show;

    if (!state.show) return;

    // Reflect the chosen mode onto the project every frame while shown. One
    // camera cannot triangulate, so it is 2D whatever was chosen; several are
    // 2D when marked not calibrated. 2D has no calibration / camera model.
    int n_cams_selected = 0;
    for (auto b : state.camera_selected) if (b) n_cams_selected++;
    const bool project_2d = state.two_d_mode || n_cams_selected <= 1;
    pm.annotation_2d = project_2d;
    pm.media_kind = media_kind_str(state.media_kind);
    if (project_2d) {
        pm.calibration_folder.clear();
        pm.telecentric = false;
    }

    ImGui::SetNextWindowSize(ImVec2(640, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("New Project", &state.show,
                     ImGuiWindowFlags_NoCollapse)) {
        // Leaving without creating: put back what the form wrote into pm.
        auto cancel = [&]() {
            state.show = false;
            pm.project_name.clear();
            pm.project_path.clear();
            pm.project_root_path.clear();
            pm.skeleton_file.clear();
            pm.skeleton_name.clear();
            pm.calibration_folder.clear();
            pm.timestamps_folder.clear();
            pm.camera_names.clear();
            pm.load_skeleton_from_json = false;
            pm.annotation_2d = false;
            state.status.clear();
            state.media_folder.clear();
            state.discovered_cameras.clear();
            state.camera_selected.clear();
        };

        // Build skeleton preset labels
        std::vector<const char *> annot_skel_labels;
        annot_skel_labels.reserve(skeleton_map.size());
        for (auto &kv : skeleton_map)
            annot_skel_labels.push_back(kv.first.c_str());
        static int annot_skeleton_idx = 0;
        if (annot_skeleton_idx >= (int)annot_skel_labels.size())
            annot_skeleton_idx = 0;

        // Three sections, in the order the choices are made: what to label,
        // how its cameras relate, which skeleton. Labels in a left column;
        // each field's Browse sits beside it.
        const ImGuiStyle &style = ImGui::GetStyle();
        const float label_w = 100.0f;
        const float gap = style.ItemInnerSpacing.x;
        auto btn_w = [&](const char *t) {
            return ImGui::CalcTextSize(t).x + style.FramePadding.x * 2.0f;
        };
        auto label = [&](const char *t) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(t);
            ImGui::SameLine(label_w);
        };
        // A path field filling the row up to a Browse button.
        auto path_row = [&](const char *t, const char *id, std::string *value,
                            const char *hint, const char *browse_id) {
            label(t);
            ImGui::SetNextItemWidth(ImMax(50.0f, ImGui::GetContentRegionAvail().x -
                                                     btn_w("Browse") - gap));
            const bool edited = hint ? ImGui::InputTextWithHint(id, hint, value)
                                     : ImGui::InputText(id, value);
            ImGui::SameLine(0.0f, gap);
            const bool browse = ImGui::Button(browse_id);
            return std::make_pair(edited, browse);
        };

        // ── Media ──
        ImGui::SeparatorText("Media");
        {
            auto [edited, browse] = path_row("Folder", "##annot_video_folder",
                                             &state.media_folder, nullptr,
                                             "Browse##annot_video");
            if (edited) {
                state.discovered_cameras =
                    discover_media_cameras(state.media_folder, &state.media_kind);
                state.camera_selected.assign(state.discovered_cameras.size(), true);
            }
            if (browse) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = state.media_folder.empty() ? media_browse_dir(ctx)
                                                      : state.media_folder;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                open_folder_dialog("ChooseAnnotVideoDir", "Choose Media Folder",
                                   media_folder_kind(), cfg);
            }
        }
        {
            const int n_cams = (int)state.discovered_cameras.size();
            label("Cameras");
            if (n_cams == 0) {
                ImGui::TextDisabled("(none found)");
                ImGui::SetCursorPosX(label_w);
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextDisabled(
                    "Videos: .mp4 or .avi files directly in the folder, one per "
                    "camera. Images: one directory per camera holding that "
                    "camera's frames, or <camera>_<frame>.jpg files in the folder "
                    "itself.");
                ImGui::PopTextWrapPos();
            } else {
                int n_sel = 0;
                for (auto b : state.camera_selected) if (b) n_sel++;
                const bool all = (n_sel == n_cams);
                ImGui::TextDisabled("%d found (%s)", n_cams,
                                    state.media_kind == MediaKind::Video ? "videos"
                                                                         : "images");
                ImGui::SameLine();
                if (ImGui::SmallButton(all ? "Select None" : "Select All"))
                    state.camera_selected.assign(n_cams, !all);
                // Two columns, filled down then across.
                const int n_rows = (n_cams + 1) / 2;
                ImGui::SetCursorPosX(label_w);
                if (ImGui::BeginTable("##annot_cam_grid", 2)) {
                    for (int row = 0; row < n_rows; row++) {
                        ImGui::TableNextRow();
                        for (int col = 0; col < 2; col++) {
                            const int idx = row + col * n_rows;
                            ImGui::TableSetColumnIndex(col);
                            if (idx >= n_cams) continue;
                            bool selected = state.camera_selected[idx];
                            if (ImGui::Checkbox((state.discovered_cameras[idx] +
                                                 "##cam_" + std::to_string(idx)).c_str(),
                                                &selected))
                                state.camera_selected[idx] = selected;
                        }
                    }
                    ImGui::EndTable();
                }
            }
        }

        // ── Cameras ──
        // Several: calibrated (3D) or not (2D); one camera is always 2D.
        int n_sel = 0;
        for (auto b : state.camera_selected) if (b) n_sel++;
        if (n_sel >= 1) {
            ImGui::SeparatorText("Cameras");
            if (n_sel == 1) {
                ImGui::TextDisabled("One camera: a 2D project (no triangulation).");
            } else {
                label("Type");
                if (ImGui::RadioButton("Calibrated: label in 2D, triangulate to 3D",
                                       !state.two_d_mode))
                    state.two_d_mode = false;
                ImGui::SetCursorPosX(label_w);
                if (ImGui::RadioButton("Not calibrated: 2D labels only",
                                       state.two_d_mode))
                    state.two_d_mode = true;
            }
            if (n_sel > 1 && !state.two_d_mode) {
                label("Model");
                ImGui::SetNextItemWidth(-FLT_MIN);
                int annot_cam_model = pm.telecentric ? 1 : 0;
                if (ImGui::Combo("##annot_cam_model", &annot_cam_model,
                                 "Projective (pinhole)\0Telecentric (affine DLT)\0"))
                    pm.telecentric = (annot_cam_model == 1);

                auto [edited, browse] = path_row("Calibration", "##annot_calibfolder",
                                                 &pm.calibration_folder, nullptr,
                                                 "Browse##annot_calib");
                (void)edited;
                if (pm.telecentric) {
                    ImGui::SetCursorPosX(label_w);
                    ImGui::TextDisabled("Expects Cam*_dlt.csv files");
                }
                if (browse) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.path = state.media_folder.empty() ? default_browse_path
                                                          : state.media_folder;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    open_folder_dialog("ChooseAnnotCalib", "Select Calibration Folder",
                                       calibration_folder_kind(), cfg);
                }
            }
            // Camera timestamps (optional, several video cameras): where the
            // per-camera timestamp files are, for the desync fix and Frame
            // Drops. red does not look for them anywhere else.
            if (n_sel > 1 && state.media_kind == MediaKind::Video) {
                auto [edited, browse] = path_row("Timestamps", "##annot_timestamps",
                                                 &pm.timestamps_folder, "optional",
                                                 "Browse##annot_timestamps");
                (void)edited;
                if (browse) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.path = state.media_folder.empty() ? default_browse_path
                                                          : state.media_folder;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    open_folder_dialog("ChooseAnnotTimestamps",
                                       "Select Camera Timestamps Folder",
                                       timestamps_folder_kind(), cfg);
                }
            } else {
                pm.timestamps_folder.clear();
            }
        }

        // ── Skeleton ──
        // A skeleton saved in the creator after "New..." becomes this one.
        if (std::string saved = skeleton_saved_since(ctx, state.skeleton_wait);
            !saved.empty()) {
            pm.load_skeleton_from_json = true;
            pm.skeleton_file = saved;
            state.skeleton_wait = -1;
        }
        ImGui::SeparatorText("Skeleton");
        {
            // The chosen one's control, New..., then File / Preset.
            const float mode_w = 90.0f;
            const float right = btn_w("New...") + mode_w + 2 * gap;
            if (pm.load_skeleton_from_json) {
                ImGui::SetNextItemWidth(ImMax(50.0f, ImGui::GetContentRegionAvail().x -
                                                         btn_w("Browse") - gap - right));
                ImGui::InputText("##annot_skelfile", &pm.skeleton_file);
                ImGui::SameLine(0.0f, gap);
                if (ImGui::Button("Browse##annot_skel")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = skeleton_dir;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseAnnotSkeleton", "Choose Skeleton", ".json", config);
                }
            } else {
                ImGui::BeginDisabled(annot_skel_labels.empty());
                ImGui::SetNextItemWidth(
                    ImMax(50.0f, ImGui::GetContentRegionAvail().x - right));
                ImGui::Combo("##annot_skeleton_preset", &annot_skeleton_idx,
                             annot_skel_labels.data(), (int)annot_skel_labels.size());
                ImGui::EndDisabled();
            }
            // Always there: draw a new skeleton. It is a file, so this goes
            // to File, where the saved skeleton is filled in.
            ImGui::SameLine(0.0f, gap);
            if (ImGui::Button("New...##annot_skel_new")) {
                state.skeleton_wait = open_skeleton_creator_for(ctx);
                pm.load_skeleton_from_json = true;
                pm.skeleton_name.clear();
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                ImGui::SetTooltip("Draw a new skeleton in the Skeleton Creator; "
                                  "saving it there picks it here.");
            ImGui::SameLine(0.0f, gap);
            int skel_mode = pm.load_skeleton_from_json ? 0 : 1;
            ImGui::SetNextItemWidth(mode_w);
            if (ImGui::Combo("##annot_skel_mode", &skel_mode, "File\0Preset\0")) {
                pm.load_skeleton_from_json = (skel_mode == 0);
                if (pm.load_skeleton_from_json) pm.skeleton_name.clear();
            }
        }
        pm.skeleton_name =
            pm.load_skeleton_from_json
                ? std::string()
                : (annot_skel_labels.empty() ? std::string()
                                             : std::string(annot_skel_labels[annot_skeleton_idx]));

        ImGui::Spacing();
        ImGui::Separator();

        // What stops Create, and what went wrong if it was tried -- both here,
        // beside the button.
        std::vector<const char *> missing;
        if (n_sel == 0)
            missing.push_back(state.discovered_cameras.empty()
                                  ? "a media folder with cameras in it"
                                  : "at least one camera ticked");
        if (pm.load_skeleton_from_json && pm.skeleton_file.empty())
            missing.push_back("a skeleton file");
        if (!state.two_d_mode && n_sel > 1 && pm.calibration_folder.empty())
            missing.push_back("a calibration folder");
        const bool annot_ok = missing.empty();
        if (!state.status.empty())
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s",
                               state.status.c_str());
        if (!annot_ok) {
            std::string need = "Still needed: ";
            for (size_t i = 0; i < missing.size(); i++) {
                if (i) need += (i + 1 == missing.size()) ? " and " : ", ";
                need += missing[i];
            }
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.35f, 1.0f), "%s", need.c_str());
        }

        // Cancel and Create, right-aligned.
        const char *create_label = "Create Project##annot_action";
        const float buttons_w = btn_w("Cancel") + style.ItemSpacing.x +
                                btn_w("Create Project");
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                             ImGui::GetContentRegionAvail().x - buttons_w);
        if (ImGui::Button("Cancel##annot_cancel")) {
            cancel();
            ImGui::End();
            return;
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!annot_ok);
        if (ImGui::Button(create_label)) {
            state.status.clear();
            pm.media_folder = state.media_folder;
            // Only include selected cameras
            pm.camera_names.clear();
            for (size_t i = 0; i < state.discovered_cameras.size(); i++)
                if (state.camera_selected[i])
                    pm.camera_names.push_back(state.discovered_cameras[i]);

            // No name or folder yet: it opens as Untitled, and the first
            // save asks for both. Replacing an Untitled project that has
            // unsaved labels asks first (run_or_confirm_unsaved).
            pm.project_name.clear();
            pm.project_path.clear();
            pm.untitled = true;
            AnnotationCreateCallback create = on_create;
            run_or_confirm_unsaved(ctx, [&ctx, &state, create]() {
                std::string error_message;
                if (!create(ctx.pm, error_message))
                    state.status = error_message;
                else
                    state.show = false;
            });
        }
        ImGui::EndDisabled();
    }
    ImGui::End();
}
