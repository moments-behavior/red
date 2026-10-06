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
    // was the whole point of File > Create Project, which could only ever wrap
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

    ImGui::SetNextWindowSize(ImVec2(720, 460), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Create Annotation Project", &state.show,
                     ImGuiWindowFlags_NoCollapse)) {

        // error banner
        if (!state.status.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.45f, 1.0f));
            ImGui::TextUnformatted(state.status.c_str());
            ImGui::PopStyleColor();
            ImGui::Separator();
        }

        // Back button — close dialog, return to welcome screen
        if (ImGui::SmallButton("< Back")) {
            state.show = false;
            // Reset pm fields that the annotation dialog writes to every frame
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
            // Reset dialog state
            state.status.clear();
            state.media_folder.clear();
            state.discovered_cameras.clear();
            state.camera_selected.clear();
            ImGui::End();
            return;
        }
        ImGui::Spacing();

        // Build skeleton preset labels
        std::vector<const char *> annot_skel_labels;
        annot_skel_labels.reserve(skeleton_map.size());
        for (auto &kv : skeleton_map)
            annot_skel_labels.push_back(kv.first.c_str());
        static int annot_skeleton_idx = 0;
        if (annot_skeleton_idx >= (int)annot_skel_labels.size())
            annot_skeleton_idx = 0;

        if (ImGui::BeginTable(
                "annotForm", 3,
                ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX |
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 160.0f);
            ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 110.0f);

            auto LabelCell = [](const char *t) {
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(t);
            };

            // ---- Video Folder ----
            ImGui::TableNextRow();
            LabelCell("Media Folder");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::InputText("##annot_video_folder", &state.media_folder)) {
                state.discovered_cameras =
                discover_media_cameras(state.media_folder, &state.media_kind);
                state.camera_selected.assign(state.discovered_cameras.size(), true);
            }
            ImGui::TableSetColumnIndex(2);
            if (ImGui::Button("Browse##annot_video")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = state.media_folder.empty() ? media_browse_dir(ctx)
                                                      : state.media_folder;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                open_folder_dialog("ChooseAnnotVideoDir", "Choose Media Folder",
                                   media_folder_kind(), cfg);
            }

            // ---- Cameras Found (checkboxes) ----
            ImGui::TableNextRow();
            {
                int n_selected = 0;
                for (size_t i = 0; i < state.camera_selected.size(); i++)
                    if (state.camera_selected[i]) n_selected++;
                std::string cam_label = "Cameras (" + std::to_string(n_selected) +
                                        "/" + std::to_string(state.discovered_cameras.size()) + ")";
                LabelCell(cam_label.c_str());
            }
            ImGui::TableSetColumnIndex(1);
            if (state.discovered_cameras.empty()) {
                ImGui::TextDisabled("(none found)");
                ImGui::TextWrapped(
                    "Videos: .mp4 or .avi files directly in the folder, one per "
                    "camera.\n"
                    "Images: one directory per camera holding that camera's "
                    "frames, or <camera>_<frame>.jpg files in the folder "
                    "itself.");
            } else {
                // Vertical 2-column layout for camera checkboxes
                int n_cams = (int)state.discovered_cameras.size();
                int n_rows = (n_cams + 1) / 2;
                if (ImGui::BeginTable("##annot_cam_grid", 2)) {
                    for (int row = 0; row < n_rows; row++) {
                        ImGui::TableNextRow();
                        for (int col = 0; col < 2; col++) {
                            int idx = row + col * n_rows;
                            ImGui::TableSetColumnIndex(col);
                            if (idx < n_cams) {
                                bool selected = state.camera_selected[idx];
                                if (ImGui::Checkbox(
                                        ("##cam_" + std::to_string(idx)).c_str(),
                                        &selected))
                                    state.camera_selected[idx] = selected;
                                ImGui::SameLine(0.0f, 2.0f);
                                ImGui::TextUnformatted(
                                    state.discovered_cameras[idx].c_str());
                            }
                        }
                    }
                    ImGui::EndTable();
                }
            }
            ImGui::TableSetColumnIndex(2);
            if (!state.discovered_cameras.empty()) {
                int n_sel = 0;
                for (auto b : state.camera_selected) if (b) n_sel++;
                bool all = (n_sel == (int)state.discovered_cameras.size());
                if (ImGui::Button(all ? "Select None" : "Select All", ImVec2(-FLT_MIN, 0))) {
                    state.camera_selected.assign(state.discovered_cameras.size(), !all);
                }
            } else {
                ImGui::Dummy(ImVec2(1, 1));
            }

            // ---- Skeleton ----
            // A skeleton saved in the creator after "New..." becomes this one.
            if (std::string saved = skeleton_saved_since(ctx, state.skeleton_wait);
                !saved.empty()) {
                pm.load_skeleton_from_json = true;
                pm.skeleton_file = saved;
                state.skeleton_wait = -1;
            }
            int skel_mode = pm.load_skeleton_from_json ? 0 : 1;

            ImGui::TableNextRow();
            LabelCell("Skeleton");
            ImGui::TableSetColumnIndex(1);
            {
                const char *ntxt = "New...##annot_skel_new";
                const float gap = ImGui::GetStyle().ItemInnerSpacing.x;
                const float new_w = ImGui::CalcTextSize("New...").x +
                                    ImGui::GetStyle().FramePadding.x * 2.0f;
                if (pm.load_skeleton_from_json) {
                    float avail = ImGui::GetContentRegionAvail().x;
                    const char *btxt = "Browse##annot_skel";
                    float browse_w = ImGui::CalcTextSize("Browse").x +
                                     ImGui::GetStyle().FramePadding.x * 2.0f;
                    ImGui::PushID("annot_skelfile");
                    ImGui::SetNextItemWidth(
                        ImMax(50.0f, avail - browse_w - new_w - 2 * gap));
                    ImGui::InputText("##path", &pm.skeleton_file);
                    ImGui::SameLine(0.0f, gap);
                    if (ImGui::Button(btxt)) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = skeleton_dir;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseAnnotSkeleton", "Choose Skeleton", ".json", config);
                    }
                    ImGui::PopID();
                } else {
                    ImGui::BeginDisabled(annot_skel_labels.empty());
                    ImGui::SetNextItemWidth(
                        ImMax(50.0f, ImGui::GetContentRegionAvail().x - new_w - gap));
                    ImGui::Combo("##annot_skeleton_preset", &annot_skeleton_idx,
                                 annot_skel_labels.data(), (int)annot_skel_labels.size());
                    ImGui::EndDisabled();
                }
                // Always there: draw a new skeleton. It is a file, so this
                // goes to File, where the saved skeleton is filled in.
                ImGui::SameLine(0.0f, gap);
                if (ImGui::Button(ntxt)) {
                    state.skeleton_wait = open_skeleton_creator_for(ctx);
                    pm.load_skeleton_from_json = true;
                    pm.skeleton_name.clear();
                }
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                    ImGui::SetTooltip("Draw a new skeleton in the Skeleton Creator; "
                                      "saving it there picks it here.");
            }
            ImGui::TableSetColumnIndex(2);
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::Combo("##annot_skel_mode", &skel_mode, "File\0Preset\0")) {
                pm.load_skeleton_from_json = (skel_mode == 0);
                if (pm.load_skeleton_from_json)
                    pm.skeleton_name.clear();
            }
            pm.skeleton_name =
                pm.load_skeleton_from_json
                    ? std::string()
                    : (annot_skel_labels.empty() ? std::string()
                                                 : std::string(annot_skel_labels[annot_skeleton_idx]));

            // ---- Calibrated or not, then Camera Model + Calibration ----
            // One form for both kinds: with several cameras the choice is
            // here; one camera is 2D (it cannot triangulate) and says so.
            {
                int n_sel = 0;
                for (auto b : state.camera_selected) if (b) n_sel++;
                if (n_sel > 1) {
                    ImGui::TableNextRow();
                    LabelCell("Cameras");
                    ImGui::TableSetColumnIndex(1);
                    if (ImGui::RadioButton("Calibrated: label in 2D, "
                                           "triangulate to 3D",
                                           !state.two_d_mode))
                        state.two_d_mode = false;
                    if (ImGui::RadioButton("Not calibrated: 2D labels only",
                                           state.two_d_mode))
                        state.two_d_mode = true;
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Dummy(ImVec2(1, 1));
                } else if (n_sel == 1) {
                    ImGui::TableNextRow();
                    LabelCell("Cameras");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextDisabled("One camera: a 2D project (no "
                                        "triangulation)");
                }
                if (n_sel > 1 && !state.two_d_mode) {
                // Camera Model selector
                ImGui::TableNextRow();
                LabelCell("Camera Model");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                int annot_cam_model = pm.telecentric ? 1 : 0;
                if (ImGui::Combo("##annot_cam_model", &annot_cam_model,
                                 "Projective (pinhole)\0Telecentric (affine DLT)\0")) {
                    pm.telecentric = (annot_cam_model == 1);
                }
                ImGui::TableSetColumnIndex(2);
                ImGui::Dummy(ImVec2(1, 1));

                // Calibration Folder
                ImGui::TableNextRow();
                LabelCell("Calibration Folder");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##annot_calibfolder", &pm.calibration_folder);
                if (pm.telecentric) {
                    ImGui::TextDisabled("Expects Cam*_dlt.csv files");
                }
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##annot_calib")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.path = state.media_folder.empty()
                                   ? default_browse_path
                                   : state.media_folder;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    open_folder_dialog("ChooseAnnotCalib", "Select Calibration Folder",
                                       calibration_folder_kind(), cfg);
                }
                }

                // Camera Timestamps (optional, several video cameras): where
                // the per-camera timestamp files are, for the desync fix and
                // Frame Drops. red does not look for them anywhere else.
                if (n_sel > 1 && state.media_kind == MediaKind::Video) {
                ImGui::TableNextRow();
                LabelCell("Camera Timestamps");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputTextWithHint("##annot_timestamps", "optional",
                                         &pm.timestamps_folder);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                    ImGui::SetTooltip(
                        "Folder with the cameras' frame timestamps --\n"
                        "sync_plan.json, Cam<name>_meta.csv or\n"
                        "cam<N>_timestamps_*.csv. Enables the desync fix\n"
                        "and Frame Drops for dropped frames. Leave empty\n"
                        "if you have none; set it later in\n"
                        "Tools > Camera Timestamps.");
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##annot_timestamps")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.path = state.media_folder.empty()
                                   ? default_browse_path
                                   : state.media_folder;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    open_folder_dialog("ChooseAnnotTimestamps", "Select Camera Timestamps Folder",
                                       timestamps_folder_kind(), cfg);
                }
                } else {
                    pm.timestamps_folder.clear();
                }
            }

            ImGui::EndTable();
        }

        ImGui::Separator();

        // Count selected cameras for validation
        int annot_n_selected = 0;
        for (size_t i = 0; i < state.camera_selected.size(); i++)
            if (state.camera_selected[i]) annot_n_selected++;

        // Validation. Five conditions used to be ANDed into one bool, and a
        // failing one greyed out Create with nothing else on screen -- so the
        // form said "no" without saying which field it meant, and the answer
        // was often a row that is only drawn under some conditions.
        std::vector<const char *> missing;
        if (annot_n_selected == 0)
            missing.push_back(state.discovered_cameras.empty()
                                  ? "a media folder with cameras in it"
                                  : "at least one camera ticked");
        if (pm.load_skeleton_from_json && pm.skeleton_file.empty())
            missing.push_back("a skeleton file");
        if (!state.two_d_mode && annot_n_selected > 1 &&
            pm.calibration_folder.empty())
            missing.push_back("a calibration folder");
        const bool annot_ok = missing.empty();

        if (!annot_ok) {
            std::string need = "Still needed: ";
            for (size_t i = 0; i < missing.size(); i++) {
                if (i) need += (i + 1 == missing.size()) ? " and " : ", ";
                need += missing[i];
            }
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.35f, 1.0f), "%s",
                               need.c_str());
        }

        // Right-align Create button
        float avail = ImGui::GetContentRegionAvail().x;
        const char *create_label = "Create Project##annot_action";
        float w = ImGui::CalcTextSize(create_label).x +
                  ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - w));

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
