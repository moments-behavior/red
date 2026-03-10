#pragma once
// export_window.h — Unified export window with format selector
//
// Replaces jarvis_export_window.h. Same layout pattern but with a format
// dropdown at top. Format-specific options appear/disappear based on selection.

#include "imgui.h"
#include "annotation.h"
#include "app_context.h"
#include "export_formats.h"
#include "gui/gui_save_load.h"
#include "gui/panel.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <filesystem>
#include <string>
#include <thread>

struct ExportWindowState {
    bool show = false;
    int format_idx = 0; // index into ExportFormats::Format enum
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

inline void DrawExportWindow(ExportWindowState &state, AppContext &ctx,
                              AnnotationMap &amap) {
    const auto &pm = ctx.pm;
    const auto &skeleton = ctx.skeleton;

    drawPanel("Export Tool", state.show,
        [&]() {
        // Live refresh: ensure AnnotationMap reflects latest keypoints
        refresh_keypoints_in_amap(amap, ctx.keypoints_map, ctx.skeleton, ctx.scene);

        // Format selector
        ImGui::SeparatorText("Format");
        static const char *format_labels[] = {
            "JARVIS", "JARVIS (with video index)", "COCO Keypoints",
            "DeepLabCut", "YOLO Pose", "YOLO Detection"
        };
        ImGui::Combo("Export Format", &state.format_idx, format_labels,
                     IM_ARRAYSIZE(format_labels));

        ImGui::SeparatorText("Project Info");

        // Auto-detect label folder (cached)
        if (state.label_cache_key != pm.keypoints_root_folder) {
            state.label_cache_key = pm.keypoints_root_folder;
            state.label_folder.clear();
            state.label_display = "(none)";
            if (!pm.keypoints_root_folder.empty()) {
                std::string most_recent, tmp_err;
                if (find_most_recent_labels(pm.keypoints_root_folder,
                                            most_recent, tmp_err) == 0) {
                    state.label_folder = most_recent;
                    state.label_display =
                        std::filesystem::path(most_recent).filename().string();
                    if (state.output_dir.empty()) {
                        state.output_dir =
                            std::filesystem::path(most_recent)
                                .parent_path().parent_path().string() + "/export";
                    }
                }
            }
        }

        ImGui::Text("Label Folder: %s", state.label_display.c_str());
        ImGui::Text("Calibration:  %s",
                    pm.calibration_folder.empty() ? "(none)" : pm.calibration_folder.c_str());

        auto fmt = static_cast<ExportFormats::Format>(state.format_idx);
        bool needs_video = (fmt == ExportFormats::JARVIS || fmt == ExportFormats::JARVIS_TR);
        if (needs_video) {
            ImGui::Text("Video Folder: %s",
                        pm.media_folder.empty() ? "(none)" : pm.media_folder.c_str());
        }
        ImGui::Text("Cameras:      %d", (int)pm.camera_names.size());

        int labeled_count = 0;
        for (const auto &[f, fa] : amap)
            if (frame_has_any_labels(fa)) ++labeled_count;
        ImGui::Text("Labeled:      %d frames", labeled_count);

        ImGui::SeparatorText("Output");

        ImGui::InputText("Output Directory", &state.output_dir);
        ImGui::SameLine();
        if (ImGui::Button("Browse##export_output")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = state.output_dir;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseExportOutputDir", "Choose Output Directory", nullptr, cfg);
        }

        ImGui::SeparatorText("Options");

        // Common options
        ImGui::SliderFloat("Train Ratio", &state.train_ratio, 0.5f, 0.99f);
        ImGui::InputInt("Random Seed", &state.seed);

        // Format-specific options
        if (fmt == ExportFormats::JARVIS || fmt == ExportFormats::JARVIS_TR ||
            fmt == ExportFormats::COCO || fmt == ExportFormats::YOLO_POSE ||
            fmt == ExportFormats::YOLO_DETECT) {
            ImGui::SliderFloat("Bbox Margin (px)", &state.margin, 0.0f, 200.0f);
        }
        if (needs_video) {
            ImGui::SliderInt("JPEG Quality", &state.jpeg_quality, 10, 100);
        }

        ImGui::Separator();

        // Export button
        if (!state.in_progress) {
            std::string validation_error;
            if (ImGui::Button("Start Export")) {
                if (state.label_folder.empty() && labeled_count == 0) {
                    validation_error = "No labeled data found";
                } else if (pm.calibration_folder.empty()) {
                    validation_error = "No calibration folder set";
                } else if (needs_video && pm.media_folder.empty()) {
                    validation_error = "No media folder set (required for " +
                                       std::string(format_labels[state.format_idx]) + ")";
                } else if (state.output_dir.empty()) {
                    validation_error = "Output directory not set";
                } else if (pm.camera_names.empty()) {
                    validation_error = "No cameras loaded";
                } else {
                    state.in_progress = true;
                    state.status = "Starting export...";

                    ExportFormats::ExportConfig ecfg;
                    ecfg.format             = fmt;
                    ecfg.label_folder       = state.label_folder;
                    ecfg.calibration_folder = pm.calibration_folder;
                    ecfg.media_folder       = pm.media_folder;
                    ecfg.output_folder      = state.output_dir;
                    ecfg.camera_names       = pm.camera_names;
                    ecfg.skeleton_name      = skeleton.name;
                    ecfg.num_keypoints      = skeleton.num_nodes;
                    ecfg.bbox_margin        = state.margin;
                    ecfg.train_ratio        = state.train_ratio;
                    ecfg.seed               = state.seed;
                    ecfg.jpeg_quality       = state.jpeg_quality;
                    ecfg.node_names         = skeleton.node_names;
                    for (const auto &e : skeleton.edges)
                        ecfg.edges.push_back({e.x, e.y});

                    // Copy the annotation map for thread safety
                    AnnotationMap amap_copy = amap;

                    std::thread(
                        [ecfg, amap_copy, &state]() {
                            ExportFormats::export_dataset(
                                ecfg.format, ecfg, amap_copy, &state.status);
                            state.in_progress = false;
                        })
                        .detach();
                }
                if (!validation_error.empty())
                    state.status = "Error: " + validation_error;
            }
        } else {
            ImGui::BeginDisabled();
            ImGui::Button("Exporting...");
            ImGui::EndDisabled();
        }

        // Status display
        if (!state.status.empty()) {
            if (state.status.find("Error") != std::string::npos)
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", state.status.c_str());
            else if (state.status.find("complete") != std::string::npos)
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%s", state.status.c_str());
            else
                ImGui::Text("%s", state.status.c_str());
        }
        },
        [&]() {
        // File dialog handler (runs every frame)
        if (ImGuiFileDialog::Instance()->Display(
                "ChooseExportOutputDir", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk())
                state.output_dir = ImGuiFileDialog::Instance()->GetCurrentPath();
            ImGuiFileDialog::Instance()->Close();
        }
        },
        ImVec2(550, 480));
}
