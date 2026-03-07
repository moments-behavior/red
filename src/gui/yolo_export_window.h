#pragma once
#include "imgui.h"
#include "project.h"
#include "types.h"
#include "yolo_export.h"
#include <ImGuiFileDialog.h>
#include <filesystem>
#include <misc/cpp/imgui_stdlib.h>
#include <string>
#include <thread>
#include <vector>

struct YoloExportState {
    bool show = false;
    std::string label_dir;
    std::string video_dir;
    std::string output_dir;
    std::string skeleton_file;
    std::string class_names_file;
    std::vector<std::string> cam_names;
    int image_size = 640;
    float train_ratio = 0.7f;
    float val_ratio = 0.2f;
    float test_ratio = 0.1f;
    int seed = 42;
    YoloExportMode mode = YOLO_DETECTION;
    bool in_progress = false;
    std::string status;
};

inline void DrawYoloExportWindow(YoloExportState &state,
                                 const ProjectManager &pm,
                                 const std::string &skeleton_file_path,
                                 const std::string &yolo_model_dir) {
    // --- File dialog handlers (always process, even when window is hidden) ---
    if (ImGuiFileDialog::Instance()->Display(
            "ChooseYoloExportLabelDir", ImGuiWindowFlags_NoCollapse,
            ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string selected_path =
                ImGuiFileDialog::Instance()->GetCurrentPath();
            state.label_dir = selected_path;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display(
            "ChooseYoloExportVideoDir", ImGuiWindowFlags_NoCollapse,
            ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string selected_path =
                ImGuiFileDialog::Instance()->GetCurrentPath();
            state.video_dir = selected_path;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display(
            "ChooseYoloExportOutputDir", ImGuiWindowFlags_NoCollapse,
            ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string selected_path =
                ImGuiFileDialog::Instance()->GetCurrentPath();
            state.output_dir = selected_path;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display(
            "ChooseYoloExportSkeletonFile", ImGuiWindowFlags_NoCollapse,
            ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string selected_file =
                ImGuiFileDialog::Instance()->GetFilePathName();
            state.skeleton_file = selected_file;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display(
            "ChooseYoloExportClassFile", ImGuiWindowFlags_NoCollapse,
            ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string selected_file =
                ImGuiFileDialog::Instance()->GetFilePathName();
            state.class_names_file = selected_file;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // --- Early return if window is not shown ---
    if (!state.show)
        return;

    ImGui::SetNextWindowSize(ImVec2(600, 700), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("YOLO Export Tool", &state.show)) {
        ImGui::SeparatorText("Export Configuration");

        // Export mode selection
        const char *export_modes[] = {"Detection Dataset",
                                      "Pose Dataset", "OBB Dataset"};
        int current_mode = static_cast<int>(state.mode);

        auto apply_defaults = [&]() {
            if (pm.media_folder.empty())
                return; // guard if config not loaded yet
            state.mode = static_cast<YoloExportMode>(current_mode);
            if (state.mode == YOLO_DETECTION) {
                if (pm.project_path.empty()) {
                    state.output_dir =
                        pm.media_folder + "/yolo_detection_dataset";
                } else {
                    state.output_dir =
                        pm.project_path + "/yolo_detection_dataset";
                }
            } else if (state.mode == YOLO_POSE) {
                if (pm.project_path.empty()) {
                    state.output_dir =
                        pm.media_folder + "/yolo_pose_dataset";
                } else {
                    state.output_dir =
                        pm.project_path + "/yolo_pose_dataset";
                }
                if (!skeleton_file_path.empty())
                    state.skeleton_file = skeleton_file_path;
            } else {
                if (pm.project_path.empty()) {
                    state.output_dir =
                        pm.media_folder + "/yolo_obb_dataset";
                } else {
                    state.output_dir =
                        pm.project_path + "/yolo_obb_dataset";
                }
            }
        };

        static bool initialized = false;

        // 1) First-time init (once, when media_dir is known)
        if (!initialized && !pm.media_folder.empty()) {
            apply_defaults();
            initialized = true;
        }

        // 2) UI: apply when user changes mode
        if (ImGui::Combo("Export Mode", &current_mode, export_modes,
                         IM_ARRAYSIZE(export_modes))) {
            apply_defaults();
        }

        ImGui::Separator();
        // Directory and file inputs
        ImGui::Text("Input Directories:");
        ImGui::InputText("Label Directory", &state.label_dir);
        ImGui::SameLine();
        if (ImGui::Button("Browse##labels")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = state.label_dir;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseYoloExportLabelDir", "Choose Label Directory",
                nullptr, config);
        }

        ImGui::InputText("Video Directory", &state.video_dir);
        ImGui::SameLine();
        if (ImGui::Button("Browse##videos")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = state.video_dir;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseYoloExportVideoDir", "Choose Video Directory",
                nullptr, config);
        }

        ImGui::InputText("Output Directory", &state.output_dir);
        ImGui::SameLine();
        if (ImGui::Button("Browse##output")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = state.output_dir;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseYoloExportOutputDir", "Choose Output Directory",
                nullptr, config);
        }

        ImGui::Separator();

        // Configuration files
        if (state.mode == YOLO_POSE) {
            ImGui::Text("Skeleton Configuration:");
            ImGui::InputText("Skeleton File", &state.skeleton_file);
            ImGui::SameLine();
            if (ImGui::Button("Browse##skeleton")) {
                IGFD::FileDialogConfig config;
                config.countSelectionMax = 1;
                config.path = std::filesystem::current_path().string() +
                              "/skeleton";
                config.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChooseYoloExportSkeletonFile", "Choose Skeleton",
                    ".json", config);
            }
            ImGui::SameLine();
            if (ImGui::Button("Use Current")) {
                if (!skeleton_file_path.empty()) {
                    state.skeleton_file = skeleton_file_path;
                }
            }
        }

        ImGui::Text("Camera List:");
        for (size_t i = 0; i < state.cam_names.size(); i++) {
            ImGui::BulletText("%s", state.cam_names[i].c_str());
        }

        if (state.mode == YOLO_DETECTION ||
            state.mode == YOLO_OBB) {
            ImGui::Text("Class Names (Optional):");
            ImGui::InputText("Class Names File",
                             &state.class_names_file);
            ImGui::SameLine();
            if (ImGui::Button("Browse##classes")) {
                IGFD::FileDialogConfig config;
                config.countSelectionMax = 1;
                config.path = yolo_model_dir;
                config.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChooseYoloExportClassFile",
                    "Choose Class Names File", ".txt", config);
            }
        }

        ImGui::Separator();

        // Export parameters
        ImGui::Text("Export Parameters:");
        ImGui::SliderInt("Image Size", &state.image_size, 320,
                         1280, "%d");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Images will be resized to this "
                              "square size (e.g., 640x640)");
        }

        ImGui::SliderFloat("Train Ratio", &state.train_ratio,
                           0.1f, 0.9f, "%.2f");
        ImGui::SliderFloat("Val Ratio", &state.val_ratio, 0.05f,
                           0.5f, "%.2f");
        ImGui::SliderFloat("Test Ratio", &state.test_ratio, 0.05f,
                           0.5f, "%.2f");

        // Ensure ratios sum to 1.0
        float total_ratio = state.train_ratio +
                            state.val_ratio +
                            state.test_ratio;
        if (total_ratio > 0.001f) {
            ImGui::Text("Total: %.2f", total_ratio);
            if (total_ratio > 1.001f || total_ratio < 0.999f) {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f),
                                   "Warning: Ratios should sum to 1.0");
                if (ImGui::Button("Normalize Ratios")) {
                    state.train_ratio /= total_ratio;
                    state.val_ratio /= total_ratio;
                    state.test_ratio /= total_ratio;
                }
            }
        }

        ImGui::InputInt("Random Seed", &state.seed);
        if (ImGui::Button("Reset Defaults")) {
            state.image_size = 640;
            state.train_ratio = 0.7f;
            state.val_ratio = 0.2f;
            state.test_ratio = 0.1f;
            state.seed = 42;
            state.status = "";
        }

        ImGui::Separator();

        // Export buttons and status
        if (!state.in_progress) {
            if (ImGui::Button("Start Export", ImVec2(150, 30))) {
                // Validate inputs
                bool valid = true;
                std::string validation_error;

                if (state.label_dir.empty()) {
                    valid = false;
                    validation_error = "Label directory is required";
                } else if (state.video_dir.empty()) {
                    valid = false;
                    validation_error = "Video directory is required";
                } else if (state.output_dir.empty()) {
                    valid = false;
                    validation_error = "Output directory is required";
                } else if (state.mode == YOLO_POSE &&
                           state.skeleton_file.empty()) {
                    valid = false;
                    validation_error = "Skeleton file is required "
                                       "for pose datasets";
                }

                if (valid) {
                    state.in_progress = true;
                    state.status = "Starting export...";

                    // Setup export configuration
                    YoloExport::ExportConfig config;
                    config.label_dir = state.label_dir;
                    config.video_dir = state.video_dir;
                    config.output_dir = state.output_dir;
                    config.cam_names = state.cam_names;
                    config.skeleton_file = state.skeleton_file;
                    config.class_names_file =
                        std::string(state.class_names_file);
                    config.image_size = state.image_size;
                    config.split.train_ratio = state.train_ratio;
                    config.split.val_ratio = state.val_ratio;
                    config.split.test_ratio = state.test_ratio;
                    config.split.seed = state.seed;

                    // Run export in background thread
                    YoloExportMode export_mode = state.mode;
                    std::thread export_thread(
                        [config, export_mode, &state]() {
                            bool success = false;
                            if (export_mode == YOLO_DETECTION) {
                                success = YoloExport::
                                    export_yolo_detection_dataset(
                                        config, &state.status);
                            } else if (export_mode == YOLO_POSE) {
                                success = YoloExport::
                                    export_yolo_pose_dataset(
                                        config, &state.status);
                            } else {
                                success =
                                    YoloExport::export_yolo_obb_dataset(
                                        config, &state.status);
                            }

                            // Update status (note: this is not
                            // thread-safe, but for simple status
                            // updates it should be okay)
                            if (success) {
                                state.status = "Export completed "
                                               "successfully!";
                            } else {
                                state.status =
                                    "Export failed! Check console "
                                    "for details.";
                            }
                            state.in_progress = false;
                        });
                    export_thread.detach();
                } else {
                    state.status = "Error: " + validation_error;
                }
            }
        } else {
            ImGui::Text("Export in progress...");
            ImGui::SameLine();
        }

        if (!state.status.empty()) {
            if (state.status.find("Error") != std::string::npos ||
                state.status.find("failed") !=
                    std::string::npos) {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%s",
                                   state.status.c_str());
            } else if (state.status.find("completed") !=
                       std::string::npos) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%s",
                                   state.status.c_str());
            } else {
                ImGui::Text("%s", state.status.c_str());
            }
        }

        ImGui::Separator();

        // Quick setup buttons
        ImGui::Text("Quick Setup:");
        if (ImGui::Button("Use Current Data")) {
            state.video_dir = pm.media_folder;
            state.output_dir = pm.project_path + "/export";
            state.cam_names = pm.camera_names;
            // Set label dir to current keypoints folder
            if (!pm.keypoints_root_folder.empty()) {
                state.label_dir = pm.keypoints_root_folder;
            }

            // Set skeleton file to current one
            if (!skeleton_file_path.empty()) {
                state.skeleton_file = skeleton_file_path;
            }
        }
    }
    ImGui::End();
}
