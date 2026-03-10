#pragma once
#include "imgui.h"
#include "app_context.h"
#include "gui/panel.h"
#include "jarvis_export.h"
#include "annotation_csv.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <filesystem>
#include <string>
#include <thread>

struct JarvisExportState {
    bool show = false;
    std::string output_dir;
    float margin = 50.0f;
    float train_ratio = 0.9f;
    int seed = 42;
    int jpeg_quality = 95;
    bool in_progress = false;
    std::string status;

    // Cached label folder detection
    std::string label_folder;
    std::string label_display = "(none)";
    std::string label_cache_key;
};

inline void DrawJarvisExportWindow(JarvisExportState &state, AppContext &ctx) {
    const auto &pm = ctx.pm;
    const auto &skeleton = ctx.skeleton;
    drawPanel("JARVIS Export Tool", state.show,
        [&]() {
        ImGui::SeparatorText("Export Configuration");

        // Auto-detect label folder (cached to avoid per-frame work)
        if (state.label_cache_key != pm.keypoints_root_folder) {
            state.label_cache_key = pm.keypoints_root_folder;
            state.label_folder.clear();
            state.label_display = "(none)";
            if (!pm.keypoints_root_folder.empty()) {
                std::string most_recent;
                std::string tmp_err;
                if (AnnotationCSV::find_most_recent_labels(pm.keypoints_root_folder,
                                            most_recent, tmp_err) == 0) {
                    state.label_folder = most_recent;
                    state.label_display =
                        std::filesystem::path(most_recent)
                            .filename()
                            .string();
                    // Default output dir next to label folder
                    if (state.output_dir.empty()) {
                        state.output_dir =
                            std::filesystem::path(most_recent)
                                .parent_path()
                                .parent_path()
                                .string() +
                            "/jarvis_export";
                    }
                }
            }
        }

        ImGui::Text("Label Folder: %s", state.label_display.c_str());
        ImGui::Text("Calibration:  %s",
                    pm.calibration_folder.empty()
                        ? "(none)"
                        : pm.calibration_folder.c_str());
        ImGui::Text("Video Folder: %s",
                    pm.media_folder.empty()
                        ? "(none)"
                        : pm.media_folder.c_str());
        ImGui::Text("Cameras:      %d",
                    (int)pm.camera_names.size());

        ImGui::Separator();

        ImGui::InputText("Output Directory", &state.output_dir);
        ImGui::SameLine();
        if (ImGui::Button("Browse##jarvis_output")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = state.output_dir;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseJarvisExportOutputDir",
                "Choose Output Directory", nullptr, cfg);
        }

        ImGui::SliderFloat("Bbox Margin (px)", &state.margin,
                           0.0f, 200.0f);
        ImGui::SliderFloat("Train Ratio", &state.train_ratio,
                           0.5f, 0.99f);
        ImGui::InputInt("Random Seed", &state.seed);
        ImGui::SliderInt("JPEG Quality", &state.jpeg_quality,
                         10, 100);

        ImGui::Separator();

        if (!state.in_progress) {
            std::string validation_error;
            if (ImGui::Button("Start Export")) {
                if (state.label_folder.empty()) {
                    validation_error = "No labeled data found";
                } else if (pm.calibration_folder.empty()) {
                    validation_error = "No calibration folder set";
                } else if (pm.media_folder.empty()) {
                    validation_error = "No media folder set";
                } else if (state.output_dir.empty()) {
                    validation_error = "Output directory not set";
                } else if (pm.camera_names.empty()) {
                    validation_error = "No cameras loaded";
                } else {
                    state.in_progress = true;
                    state.status = "Starting export...";

                    JarvisExport::ExportConfig jcfg;
                    jcfg.label_folder = state.label_folder;
                    jcfg.calibration_folder = pm.calibration_folder;
                    jcfg.media_folder = pm.media_folder;
                    jcfg.output_folder = state.output_dir;
                    jcfg.camera_names = pm.camera_names;
                    jcfg.skeleton_name = skeleton.name;
                    jcfg.num_keypoints = skeleton.num_nodes;
                    jcfg.margin_pixel = state.margin;
                    jcfg.train_ratio = state.train_ratio;
                    jcfg.seed = state.seed;
                    jcfg.jpeg_quality = state.jpeg_quality;

                    // Copy node names and edges from skeleton
                    jcfg.node_names = skeleton.node_names;
                    for (const auto &e : skeleton.edges) {
                        jcfg.edges.push_back({e.x, e.y});
                    }

                    std::thread(
                        [jcfg, &state]() {
                            JarvisExport::export_jarvis_dataset(
                                jcfg, &state.status);
                            state.in_progress = false;
                        })
                        .detach();
                }
                if (!validation_error.empty()) {
                    state.status =
                        "Error: " + validation_error;
                }
            }
        } else {
            ImGui::BeginDisabled();
            ImGui::Button("Exporting...");
            ImGui::EndDisabled();
        }

        if (!state.status.empty()) {
            if (state.status.find("Error") !=
                std::string::npos) {
                ImGui::TextColored(
                    ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s",
                    state.status.c_str());
            } else if (state.status.find("completed") !=
                       std::string::npos) {
                ImGui::TextColored(
                    ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%s",
                    state.status.c_str());
            } else {
                ImGui::Text("%s", state.status.c_str());
            }
        }
        },
        [&]() {
        // File dialog handler (runs every frame)
        if (ImGuiFileDialog::Instance()->Display(
                "ChooseJarvisExportOutputDir", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                state.output_dir =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            ImGuiFileDialog::Instance()->Close();
        }
        },
        ImVec2(550, 400));
}
