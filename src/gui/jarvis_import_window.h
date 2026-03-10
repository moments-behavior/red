#pragma once
#include "imgui.h"
#include "app_context.h"
#include "gui/panel.h"
#include "jarvis_import.h"
#include "gui/gui_save_load.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <filesystem>
#include <string>

struct JarvisImportState {
    bool show = false;
    std::string data3d_path;       // path to data3D.csv
    float conf_threshold = 0.0f;
    bool done = false;
    JarvisImport::ImportResult result;
};

inline void DrawJarvisImportWindow(JarvisImportState &state, AppContext &ctx) {
    const auto &pm = ctx.pm;
    const auto &skeleton = ctx.skeleton;
    auto &scene = ctx.scene;
    auto &keypoints_map = ctx.keypoints_map;

    drawPanel("JARVIS Import Tool", state.show,
        [&]() {
        ImGui::SeparatorText("Import JARVIS Predictions");

        ImGui::Text("Calibration: %s",
                    pm.calibration_folder.empty()
                        ? "(none)" : pm.calibration_folder.c_str());
        ImGui::Text("Cameras:     %d", (int)pm.camera_names.size());
        ImGui::Text("Skeleton:    %s", skeleton.name.c_str());

        ImGui::Separator();

        ImGui::InputText("data3D.csv", &state.data3d_path);
        ImGui::SameLine();
        if (ImGui::Button("Browse##jarvis_import")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = state.data3d_path.empty()
                ? pm.project_path : state.data3d_path;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseJarvisData3D",
                "Choose data3D.csv", ".csv", cfg);
        }

        ImGui::SliderFloat("Min Confidence", &state.conf_threshold,
                           0.0f, 1.0f, "%.2f");

        ImGui::Separator();

        bool can_import = !state.data3d_path.empty() &&
                          !pm.calibration_folder.empty() &&
                          !pm.camera_names.empty() &&
                          pm.camera_params.size() == pm.camera_names.size();

        if (!can_import) ImGui::BeginDisabled();
        if (ImGui::Button("Import Predictions")) {
            // Determine output folder
            std::string predictions_root = pm.project_path + "/predictions";
            int img_h = 0;
            if (!pm.camera_params.empty()) {
                // Get image height from calibration
                // camera_params[0].k(1,2) is cy, approximate height as 2*cy
                img_h = (int)(pm.camera_params[0].k(1,2) * 2.0);
            }

            state.result = JarvisImport::import_jarvis_predictions(
                state.data3d_path,
                predictions_root,
                skeleton.name,
                pm.camera_params,
                pm.camera_names,
                img_h,
                state.conf_threshold);

            if (state.result.error.empty()) {
                state.done = true;
            }
        }
        if (!can_import) ImGui::EndDisabled();

        // Show result
        if (!state.result.error.empty()) {
            ImGui::TextColored(ImVec4(1,0.3f,0.3f,1), "Error: %s",
                               state.result.error.c_str());
        }

        if (state.done) {
            ImGui::TextColored(ImVec4(0,1,0,1),
                "Imported %d frames (%d filtered)",
                state.result.frames_imported,
                state.result.frames_filtered);
            ImGui::Text("Mean confidence: %.3f", state.result.mean_confidence);
            ImGui::Text("Output: %s", state.result.output_folder.c_str());

            ImGui::Separator();
            if (ImGui::Button("Load into RED")) {
                std::string err;
                std::vector<std::string> cam_names = pm.camera_names;
                int ret = load_keypoints(
                    state.result.output_folder,
                    keypoints_map, &ctx.skeleton, scene,
                    cam_names, err);
                if (ret == 0) {
                    // Load confidence data
                    JarvisImport::load_confidence(
                        state.result.output_folder,
                        keypoints_map,
                        scene->num_cams,
                        skeleton.num_nodes);
                    state.result.error.clear();
                    ImGui::TextColored(ImVec4(0,1,0,1),
                        "Predictions loaded!");
                } else {
                    state.result.error = "Load failed: " + err;
                }
            }
        }
        },
        [&]() {
        // File dialog handler
        if (ImGuiFileDialog::Instance()->Display(
                "ChooseJarvisData3D", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto sel = ImGuiFileDialog::Instance()->GetSelection();
                if (!sel.empty()) {
                    state.data3d_path = sel.begin()->second;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }
        },
        ImVec2(550, 380));
}
