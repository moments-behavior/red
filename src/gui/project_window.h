#pragma once
#include "app_context.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>

inline void DrawProjectWindow(AppContext &ctx) {
    auto &pm = ctx.pm;
    auto &skeleton_map = ctx.skeleton_map;
    auto &skeleton = ctx.skeleton;
    auto &skeleton_dir = ctx.skeleton_dir;
    auto &popups = ctx.popups;

    if (!pm.show_project_window)
        return;

    ImGuiWindowFlags win_flags = ImGuiWindowFlags_NoCollapse;
    ImGui::SetNextWindowSize(ImVec2(720, 460), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Create Project", &pm.show_project_window, win_flags)) {

        if (ImGui::BeginTable(
                "projectForm", 3,
                ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX |
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed,
                                    160.0f);
            ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthStretch,
                                    1.0f);
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed,
                                    110.0f);

            auto LabelCell = [](const char *t) {
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(t);
            };

            // ---- Project Name ----
            ImGui::TableNextRow();
            LabelCell("Project Name");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##projname", &pm.project_name);
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ---- Project Root Path ----
            ImGui::TableNextRow();
            LabelCell("Project Root Path");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##rootpath", &pm.project_root_path);
            ImGui::TableSetColumnIndex(2);
            if (ImGui::Button("Browse##project")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = pm.project_root_path;
                cfg.fileName = pm.project_name;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChooseProjectDir", "Choose Project Directory", nullptr,
                    cfg);
            }

            // ---- Full Path (computed) ----
            {
                std::filesystem::path p =
                    std::filesystem::path(pm.project_root_path) /
                    pm.project_name;
                pm.project_path = p.string();
            }

            ImGui::TableNextRow();
            LabelCell("Full Path");
            ImGui::TableSetColumnIndex(1);
            ImGui::BeginDisabled();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##fullpath", &pm.project_path);
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ---- Skeleton ----
            std::vector<const char *> labels_s;
            labels_s.reserve(skeleton_map.size());
            for (auto &kv : skeleton_map)
                labels_s.push_back(kv.first.c_str());
            static int skeleton_idx = 0;
            if (skeleton_idx >= (int)labels_s.size())
                skeleton_idx = 0;

            int mode = pm.load_skeleton_from_json ? 0 : 1;

            ImGui::TableNextRow();
            LabelCell("Skeleton");

            ImGui::TableSetColumnIndex(1);
            {
                if (pm.load_skeleton_from_json) {
                    float avail = ImGui::GetContentRegionAvail().x;
                    const char *btxt = "Browse##browse_skel";
                    float browse_w = ImGui::CalcTextSize(btxt).x +
                                     ImGui::GetStyle().FramePadding.x * 2.0f;
                    float gap = ImGui::GetStyle().ItemInnerSpacing.x;

                    ImGui::PushID("skelfile");
                    ImGui::SetNextItemWidth(
                        ImMax(50.0f, avail - browse_w - gap));
                    ImGui::InputText("##path", &pm.skeleton_file);
                    ImGui::SameLine(0.0f, gap);
                    if (ImGui::Button(btxt)) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = skeleton_dir;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseSkeleton", "Choose Skeleton", ".json",
                            config);
                    }
                    ImGui::PopID();
                } else {
                    ImGui::BeginDisabled(labels_s.empty());
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::Combo("##skeleton_preset", &skeleton_idx,
                                 labels_s.data(), (int)labels_s.size());
                    ImGui::EndDisabled();
                }
            }

            ImGui::TableSetColumnIndex(2);
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::Combo("##skeleton_mode_small", &mode,
                             "File\0Preset\0")) {
                pm.load_skeleton_from_json = (mode == 0);
                if (pm.load_skeleton_from_json)
                    pm.skeleton_name.clear();
            }

            pm.skeleton_name =
                pm.load_skeleton_from_json
                    ? std::string()
                    : (labels_s.empty() ? std::string()
                                        : std::string(labels_s[skeleton_idx]));

            // ---- Calibration Folder ----
            if (pm.camera_names.size() > 1) {
                ImGui::TableNextRow();
                LabelCell("Calibration Folder");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##calibfolder", &pm.calibration_folder);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##loadprojectcalibration")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.path = pm.project_root_path;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseCalibration", "Select Calibration Folder",
                        nullptr, cfg);
                }
            }

            ImGui::EndTable();
        }

        ImGui::Separator();

        const bool needed_ok =
            !pm.project_name.empty() && !pm.project_root_path.empty() &&
            (pm.camera_names.size() <= 1 || !pm.calibration_folder.empty()) &&
            (!pm.load_skeleton_from_json || !pm.skeleton_file.empty());

        float avail = ImGui::GetContentRegionAvail().x;
        const char *create_label = "Create Project##action";
        float w = ImGui::CalcTextSize(create_label).x +
                  ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - w));

        ImGui::BeginDisabled(!needed_ok);
        if (ImGui::Button(create_label)) {
            std::string err;
            if (!ensure_dir_exists(pm.project_path, &err))
                popups.pushError(err);

            IGFD::FileDialogConfig config;
            config.path = pm.project_path;
            config.fileName = "project.redproj";
            config.countSelectionMax = 1;
            config.flags = ImGuiFileDialogFlags_ConfirmOverwrite;
            ImGuiFileDialog::Instance()->OpenDialog(
                "SaveProjectFileDlg", "Save Project File",
                "Red Project{.redproj},All files{.*}", config);
        }
        ImGui::EndDisabled();
    }
    ImGui::End();

    if (ImGuiFileDialog::Instance()->Display("SaveProjectFileDlg", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string outPath =
                ImGuiFileDialog::Instance()->GetFilePathName();

            if (!ends_with_ci(outPath, ".redproj")) {
                outPath += ".redproj";
            }

            std::string err;
            if (!setup_project(pm, skeleton, skeleton_map, &err)) {
                popups.pushError(err);
            } else if (!save_project_manager_json(pm, outPath, &err)) {
                popups.pushError(err);
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }
}
