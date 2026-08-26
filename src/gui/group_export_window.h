#pragma once
// group_export_window.h — Tools -> Group JARVIS Export.
//
// Standalone tool (works with no project open) that merges many JARVIS
// datasets into one master dataset. Sources are added one at a time and are
// either a live RED project (.redproj) or an already-exported JARVIS dataset
// folder. All sources must share an identical keypoint set; train/val is
// re-split globally. The heavy lifting lives in src/jarvis_merge.h; this file
// is the ImGui front end.
//
// Threading mirrors export_window.h: the merge runs on a detached thread that
// writes only to a heap-allocated status string, handed back to the main
// thread via an atomic `finished` flag. `images_saved` is an atomic progress
// counter; `status` is only ever touched by the main thread.

#include "imgui.h"
#include "app_context.h"
#include "jarvis_merge.h"
#include "gui/panel.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

struct GroupExportState {
    bool show = false;

    std::vector<JarvisMerge::SourceInfo> sources;
    std::string output_dir;
    float train_ratio = 0.9f;
    float margin = 50.0f;
    int seed = 42;
    int jpeg_quality = 95;
    int scale_factor = 1; // project sources: write calibration so 3D reconstructs in (mm × scale_factor)

    // Thread-safe progress / handoff (see export_window.h for the protocol).
    std::atomic<bool> in_progress{false};
    std::atomic<int> images_saved{0};
    int images_total = 0;
    std::shared_ptr<std::string> finished_status;
    std::atomic<bool> finished{false};
    std::string status; // only written by the main thread
};

inline void DrawGroupExportWindow(GroupExportState &state, AppContext &ctx) {
    // Poll for worker completion (runs every frame, even when hidden).
    if (state.finished.load(std::memory_order_acquire)) {
        if (state.finished_status) state.status = *state.finished_status;
        state.finished_status.reset();
        state.in_progress.store(false, std::memory_order_relaxed);
        state.finished.store(false, std::memory_order_relaxed);
    }

    DrawPanel("Group JARVIS Export", state.show,
        [&]() {
        ImGui::TextWrapped(
            "Merge many JARVIS datasets into one master dataset. Add live "
            "projects (.redproj) and/or previously-exported dataset folders. "
            "All sources must share the same keypoints; train/val is re-split "
            "globally across everything.");

        ImGui::SeparatorText("Sources");

        if (ImGui::Button("Add Project (.redproj)")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 0; // multi-select
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "GroupAddProject", "Choose Project(s)", "Red Project{.redproj}", cfg);
        }
        ImGui::SameLine();
        if (ImGui::Button("Add Dataset Folder")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "GroupAddDataset", "Choose Exported Dataset Folder", nullptr, cfg);
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(state.sources.empty());
        if (ImGui::Button("Clear All")) state.sources.clear();
        ImGui::EndDisabled();

        // Reference keypoints = first valid source; used to flag mismatches.
        const std::vector<std::string> *ref_kp = nullptr;
        for (const auto &s : state.sources)
            if (s.valid) { ref_kp = &s.keypoint_names; break; }

        if (state.sources.empty()) {
            ImGui::TextDisabled("No sources added yet.");
        } else if (ImGui::BeginTable("group_sources", 4,
                       ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                       ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 70);
            ImGui::TableSetupColumn("Name");
            ImGui::TableSetupColumn("Status");
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 30);
            ImGui::TableHeadersRow();

            int remove_idx = -1;
            for (int i = 0; i < (int)state.sources.size(); ++i) {
                const auto &s = state.sources[i];
                bool kp_mismatch = s.valid && ref_kp && s.keypoint_names != *ref_kp;

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(s.kind == JarvisMerge::SourceInfo::Project
                                           ? "Project" : "Dataset");
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(s.display_name.c_str());
                ImGui::TableSetColumnIndex(2);
                if (!s.valid)
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "%s", s.message.c_str());
                else if (kp_mismatch)
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                                       "skeleton mismatch (%d kps)", (int)s.keypoint_names.size());
                else
                    ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%s", s.message.c_str());
                ImGui::TableSetColumnIndex(3);
                ImGui::PushID(i);
                if (ImGui::SmallButton("X")) remove_idx = i;
                ImGui::PopID();
            }
            ImGui::EndTable();
            if (remove_idx >= 0)
                state.sources.erase(state.sources.begin() + remove_idx);
        }

        // Aggregate counts across accepted sources.
        int accepted = 0, total_framesets = 0, total_images = 0;
        for (const auto &s : state.sources) {
            bool ok = s.valid && (!ref_kp || s.keypoint_names == *ref_kp);
            if (ok) { ++accepted; total_framesets += s.frame_count; total_images += s.image_count; }
        }
        if (accepted > 0)
            ImGui::Text("Accepted: %d sources, %d framesets, %d images",
                        accepted, total_framesets, total_images);

        ImGui::SeparatorText("Output");
        ImGui::InputText("Output Directory", &state.output_dir);
        ImGui::SameLine();
        if (ImGui::Button("Browse##group_output")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = state.output_dir;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "GroupChooseOutput", "Choose Output Directory", nullptr, cfg);
        }

        ImGui::SeparatorText("Options");
        ImGui::SliderFloat("Train Ratio", &state.train_ratio, 0.5f, 0.99f);
        ImGui::InputInt("Random Seed", &state.seed);
        ImGui::SliderFloat("Bbox Margin (px, projects only)", &state.margin, 0.0f, 200.0f);
        ImGui::SliderInt("JPEG Quality (projects only)", &state.jpeg_quality, 10, 100);
        ImGui::SliderInt("Scale factor (x1-x100; projects only)", &state.scale_factor, 1, 100);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Write each project source's calibration so JARVIS reconstructs\n"
                "3D in (mm x scale factor), so the integer-mm voxel grid resolves\n"
                "e.g. a ~3mm fly. Telecentric: projectionMatrix[0:2,0:3] is divided\n"
                "by the factor; perspective: T is multiplied by it. Dataset sources\n"
                "keep their baked-in scale. Predicted 3D comes out scaled — divide\n"
                "by the factor for mm. Check for the fly rig.");

        ImGui::Separator();

        if (!state.in_progress.load(std::memory_order_relaxed)) {
            if (ImGui::Button("Start Merge")) {
                std::string err;
                if (accepted == 0)
                    err = "No valid sources (matching skeletons) to merge";
                else if (state.output_dir.empty())
                    err = "Output directory not set";

                if (err.empty()) {
                    // Snapshot only the accepted sources for the worker thread.
                    std::vector<JarvisMerge::SourceInfo> picked;
                    for (const auto &s : state.sources) {
                        bool ok = s.valid && (!ref_kp || s.keypoint_names == *ref_kp);
                        if (ok) picked.push_back(s);
                    }

                    JarvisMerge::MergeConfig mcfg;
                    mcfg.output_folder = state.output_dir;
                    mcfg.train_ratio = state.train_ratio;
                    mcfg.seed = state.seed;
                    mcfg.margin_pixel = state.margin;
                    mcfg.jpeg_quality = state.jpeg_quality;
                    mcfg.scale_factor = state.scale_factor;

                    state.images_saved.store(0, std::memory_order_relaxed);
                    state.images_total = total_images;
                    state.in_progress.store(true, std::memory_order_relaxed);
                    state.status = "Merging...";

                    auto result_status = std::make_shared<std::string>();
                    state.finished_status = result_status;

                    std::thread(
                        [mcfg, picked, result_status, &state]() {
                            std::string thread_status;
                            JarvisMerge::merge_datasets(mcfg, picked, &thread_status,
                                                        &state.images_saved);
                            *result_status = std::move(thread_status);
                            state.finished.store(true, std::memory_order_release);
                        })
                        .detach();
                } else {
                    state.status = "Error: " + err;
                }
            }
        } else {
            ImGui::BeginDisabled();
            ImGui::Button("Merging...");
            ImGui::EndDisabled();
            if (state.images_total > 0) {
                int saved = state.images_saved.load(std::memory_order_relaxed);
                float progress = (float)saved / (float)state.images_total;
                char overlay[64];
                snprintf(overlay, sizeof(overlay), "%d / %d images", saved, state.images_total);
                ImGui::ProgressBar(progress, ImVec2(-1, 0), overlay);
            }
        }

        if (!state.status.empty()) {
            if (state.status.find("Error") != std::string::npos)
                ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "%s", state.status.c_str());
            else if (state.status.find("complete") != std::string::npos)
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "%s", state.status.c_str());
            else
                ImGui::Text("%s", state.status.c_str());
        }
        },
        [&]() {
        // File dialogs (run every frame).
        if (ImGuiFileDialog::Instance()->Display("GroupAddProject",
                ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto sel = ImGuiFileDialog::Instance()->GetSelection();
                for (const auto &[name, path] : sel)
                    state.sources.push_back(
                        JarvisMerge::scan_project(path, ctx.skeleton_map));
            }
            ImGuiFileDialog::Instance()->Close();
        }
        if (ImGuiFileDialog::Instance()->Display("GroupAddDataset",
                ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string dir = ImGuiFileDialog::Instance()->GetCurrentPath();
                state.sources.push_back(JarvisMerge::scan_dataset(dir));
            }
            ImGuiFileDialog::Instance()->Close();
        }
        if (ImGuiFileDialog::Instance()->Display("GroupChooseOutput",
                ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk())
                state.output_dir = ImGuiFileDialog::Instance()->GetCurrentPath();
            ImGuiFileDialog::Instance()->Close();
        }
        },
        ImVec2(640, 560));
}
