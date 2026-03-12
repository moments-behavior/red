#pragma once
#include "imgui.h"
#include "app_context.h"
#include "gui/panel.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

inline std::vector<std::string> discover_mp4_cameras(const std::string &folder) {
    namespace fs = std::filesystem;
    std::vector<std::string> cameras;
    if (folder.empty() || !fs::is_directory(folder))
        return cameras;
    for (const auto &entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        // case-insensitive .mp4 check
        if (ext.size() == 4 &&
            (ext[1] == 'm' || ext[1] == 'M') &&
            (ext[2] == 'p' || ext[2] == 'P') &&
            (ext[3] == '4')) {
            cameras.push_back(entry.path().stem().string());
        }
    }
    std::sort(cameras.begin(), cameras.end());
    return cameras;
}

struct AnnotationDialogState {
    bool show = false;
    std::string video_folder;
    std::vector<std::string> discovered_cameras;
    std::vector<bool> camera_selected;
    std::string status;
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
            ? ctx.red_data_dir
            : ctx.user_settings.default_media_root_path;
    DrawPanel("Create Annotation Project", state.show,
        [&]() {
        // error banner
        if (!state.status.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.45f, 1.0f));
            ImGui::TextUnformatted(state.status.c_str());
            ImGui::PopStyleColor();
            ImGui::Separator();
        }

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
            LabelCell("Video Folder");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::InputText("##annot_video_folder", &state.video_folder)) {
                state.discovered_cameras = discover_mp4_cameras(state.video_folder);
                state.camera_selected.assign(state.discovered_cameras.size(), true);
            }
            ImGui::TableSetColumnIndex(2);
            if (ImGui::Button("Browse##annot_video")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = state.video_folder;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChooseAnnotVideoDir", "Choose Video Folder", nullptr, cfg);
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
                ImGui::TextDisabled("(none — select a folder with .mp4 files)");
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

            // ---- Project Name ----
            ImGui::TableNextRow();
            LabelCell("Project Name");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##annot_projname", &pm.project_name);
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ---- Project Root Path ----
            ImGui::TableNextRow();
            LabelCell("Project Root Path");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##annot_rootpath", &pm.project_root_path);
            ImGui::TableSetColumnIndex(2);
            if (ImGui::Button("Browse##annot_root")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = pm.project_root_path;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChooseAnnotRootDir", "Choose Project Root", nullptr, cfg);
            }

            // ---- Full Path (computed) ----
            {
                std::filesystem::path p =
                    std::filesystem::path(pm.project_root_path) / pm.project_name;
                pm.project_path = p.string();
            }
            ImGui::TableNextRow();
            LabelCell("Full Path");
            ImGui::TableSetColumnIndex(1);
            ImGui::BeginDisabled();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##annot_fullpath", &pm.project_path);
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ---- Skeleton ----
            int skel_mode = pm.load_skeleton_from_json ? 0 : 1;

            ImGui::TableNextRow();
            LabelCell("Skeleton");
            ImGui::TableSetColumnIndex(1);
            {
                if (pm.load_skeleton_from_json) {
                    float avail = ImGui::GetContentRegionAvail().x;
                    const char *btxt = "Browse##annot_skel";
                    float browse_w = ImGui::CalcTextSize(btxt).x +
                                     ImGui::GetStyle().FramePadding.x * 2.0f;
                    float gap = ImGui::GetStyle().ItemInnerSpacing.x;
                    ImGui::PushID("annot_skelfile");
                    ImGui::SetNextItemWidth(ImMax(50.0f, avail - browse_w - gap));
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
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::Combo("##annot_skeleton_preset", &annot_skeleton_idx,
                                 annot_skel_labels.data(), (int)annot_skel_labels.size());
                    ImGui::EndDisabled();
                }
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

            // ---- Camera Model + Calibration (only if multiple cameras) ----
            {
                int n_sel = 0;
                for (auto b : state.camera_selected) if (b) n_sel++;
                if (n_sel > 1) {
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
                    cfg.path = state.video_folder.empty()
                                   ? default_browse_path
                                   : state.video_folder;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseAnnotCalib", "Select Calibration Folder", nullptr, cfg);
                }
                }
            }

            ImGui::EndTable();
        }

        ImGui::Separator();

        // Count selected cameras for validation
        int annot_n_selected = 0;
        for (size_t i = 0; i < state.camera_selected.size(); i++)
            if (state.camera_selected[i]) annot_n_selected++;

        // Validation
        const bool annot_ok =
            !pm.project_name.empty() && !pm.project_root_path.empty() &&
            annot_n_selected > 0 &&
            (annot_n_selected <= 1 || !pm.calibration_folder.empty()) &&
            (!pm.load_skeleton_from_json || !pm.skeleton_file.empty());

        // Right-align Create button
        float avail = ImGui::GetContentRegionAvail().x;
        const char *create_label = "Create Project##annot_action";
        float w = ImGui::CalcTextSize(create_label).x +
                  ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - w));

        ImGui::BeginDisabled(!annot_ok);
        if (ImGui::Button(create_label)) {
            state.status.clear();
            pm.media_folder = state.video_folder;
            // Only include selected cameras
            pm.camera_names.clear();
            for (size_t i = 0; i < state.discovered_cameras.size(); i++)
                if (state.camera_selected[i])
                    pm.camera_names.push_back(state.discovered_cameras[i]);

            std::string error_message;
            if (!on_create(pm, error_message)) {
                state.status = error_message;
            } else {
                state.show = false;
            }
        }
        ImGui::EndDisabled();
        },
        [&]() {
        // File dialog handlers (run every frame)
        if (ImGuiFileDialog::Instance()->Display("ChooseAnnotVideoDir", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::filesystem::path chosen(
                    ImGuiFileDialog::Instance()->GetCurrentPath());
                state.video_folder = chosen.string();
                state.discovered_cameras = discover_mp4_cameras(state.video_folder);
                state.camera_selected.assign(state.discovered_cameras.size(), true);
                if (pm.project_name.empty())
                    pm.project_name = chosen.filename().string();
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseAnnotRootDir", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                pm.project_root_path =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseAnnotSkeleton", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                pm.skeleton_file = ImGuiFileDialog::Instance()->GetFilePathName();
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseAnnotCalib", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                pm.calibration_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            ImGuiFileDialog::Instance()->Close();
        }
        },
        ImVec2(720, 460), ImGuiWindowFlags_NoCollapse);
}
