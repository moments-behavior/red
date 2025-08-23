#pragma once
#include "camera.h"
#include "render.h"
#include "skeleton.h"
#include "utils.h"
#include <ImGuiFileDialog.h>
#include <algorithm>
#include <fstream>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h> // for InputText(std::string&)
#include <string>
#include <vector>

struct ProjectManager {
    bool show_project_window = false;
    std::string project_root_path;
    std::string project_path;
    std::string suggest_project_name;
    bool load_skeleton_from_json = false;
    std::string project_skeleton_file;
    std::string project_calibration_folder;
    std::string keypoints_root_folder;
    bool plot_keypoints_flag = false;
    std::vector<CameraParams> camera_params;
    std::vector<std::string> camera_names;
};

inline void DrawProjectWindow(
    ProjectManager &pm, std::map<std::string, SkeletonPrimitive> &skeleton_map,
    SkeletonContext &skeleton, std::string &skeleton_dir, bool &show_error,
    std::string &error_message, render_scene *scene) {
    if (!pm.show_project_window)
        return;

    if (ImGui::Begin("Project", &pm.show_project_window)) {
        ImGui::InputText("Porject Name", &pm.suggest_project_name);
        ImGui::TextUnformatted("Project Root Path: ");
        ImGui::SameLine(0.0f, 0.0f);
        ImGui::Text("%s", pm.project_root_path.c_str());
        ImGui::SameLine();
        if (ImGui::Button("Browse##project")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = pm.project_root_path;
            cfg.fileName = pm.suggest_project_name;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseProjectDir", "Choose Project Directory", nullptr, cfg);
        }
        std::filesystem::path p = std::filesystem::path(pm.project_root_path) /
                                  pm.suggest_project_name;
        pm.project_path = p.string();
        ImGui::Text("%s", pm.project_path.c_str());
        ImGui::Checkbox("Load skeleton from file", &pm.load_skeleton_from_json);

        ImGui::BeginDisabled(!pm.load_skeleton_from_json);
        ImGui::InputText("Skeleton File##project", &pm.project_skeleton_file);
        ImGui::SameLine();
        if (ImGui::Button("Browse##loadprojectskeleton")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = skeleton_dir;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseSkeleton", "Choose Skeleton", ".json", config);
        }
        ImGui::EndDisabled();

        static std::vector<const char *> labels_s;
        static std::vector<SkeletonPrimitive *> vals_s;
        if (labels_s.empty()) {
            labels_s.reserve(skeleton_map.size());
            vals_s.reserve(skeleton_map.size());
            for (auto &[k, v] : skeleton_map) {
                labels_s.push_back(k.c_str()); // safe: map & keys don’t change
                vals_s.push_back(&v);
            }
        }
        static int skeleton_idx = 0;
        ImGui::BeginDisabled(pm.load_skeleton_from_json);
        if (ImGui::Combo("Select Skeleton", &skeleton_idx, labels_s.data(),
                         (int)labels_s.size())) {
            // selection changed
        }
        ImGui::EndDisabled();

        ImGui::InputText("Calibration Folder", &pm.project_calibration_folder);
        ImGui::SameLine();
        if (ImGui::Button("Browse##loadprojectcalibration")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = pm.project_root_path;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseCalibration", "Choose Calibration Folder", nullptr, cfg);
        }
        if (ImGui::Button("Create Project##action")) {
            bool load_calibration = true;
            if (scene->num_cams > 1) {
                for (u32 i = 0; i < scene->num_cams; i++) {
                    std::string cam_file = pm.project_calibration_folder + "/" +
                                           pm.camera_names[i] + ".yaml";
                    CameraParams cam;
                    if (camera_load_params_from_yaml(cam_file, cam,
                                                     error_message)) {
                        pm.camera_params.push_back(cam);
                    } else {
                        load_calibration = false;
                        pm.camera_params.clear();
                        show_error = true;
                        break;
                    }
                }
            }
            if (load_calibration) {
                skeleton.num_nodes = 0;
                skeleton.num_edges = 0;
                skeleton.name = "";
                skeleton.has_bbox = false;
                skeleton.has_skeleton = true;
                skeleton.node_colors.clear();
                skeleton.edges.clear();
                skeleton.node_names.clear();

                if (pm.load_skeleton_from_json) {
                    load_skeleton_json(pm.project_skeleton_file, &skeleton);
                } else {
                    skeleton_initialize(labels_s[skeleton_idx], &skeleton,
                                        *vals_s[skeleton_idx]);
                }

                pm.plot_keypoints_flag = true;
                pm.keypoints_root_folder = pm.project_path + "/labeled_data/";
            }

            if (!ensure_dir_exists(pm.project_path, &error_message)) {
                show_error = true;
            } else {
                std::filesystem::create_directory(pm.keypoints_root_folder);
                pm.show_project_window = false;
            }
        }
    }
    ImGui::End();
}
