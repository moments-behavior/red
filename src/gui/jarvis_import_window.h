#pragma once
#include "imgui.h"
#include "app_context.h"
#include "annotation.h"
#include "annotation_csv.h"
#include "red3d_import.h"
#include "gui/panel.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <climits>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

// Import a red_csv v2 3D keypoints file (e.g. a cluster JARVIS prediction's
// keypoints3d.csv) into the memory-mapped .rpred prediction store — NOT the
// editable Labeling Tool. A whole-video prediction is too large to hold in the
// in-memory AnnotationMap, so it streams straight to the store and is auto-
// activated: it then shows as a read-only viewport overlay and feeds Pose Stats
// / Bouts, while the user promotes individual frames to edit on demand (Pose
// Stats "Fix this frame"). The file's #skeleton header is ignored; the store is
// written at the project skeleton's node count.
struct JarvisImportState {
    bool show = false;
    std::string keypoints3d_path;   // path to a red_csv v2 keypoints3d.csv
    bool done = false;
    Red3DImport::ImportStats result;
    std::string store_to_load;      // .rpred to activate (consumed by main loop)
    std::string error;
};

inline void DrawJarvisImportWindow(JarvisImportState &state, AppContext &ctx) {
    auto &pm = ctx.pm;
    const auto &skeleton = ctx.skeleton;

    DrawPanel("Import 3D Predictions", state.show,
        [&]() {
        ImGui::SeparatorText("Import predicted keypoints3d.csv");

        ImGui::Text("Skeleton:    %s", skeleton.name.c_str());
        ImGui::Text("Nodes:       %d", skeleton.num_nodes);
        ImGui::Text("Cameras:     %d", (int)pm.camera_names.size());
        ImGui::TextWrapped(
            "Streams the 3D into this project's prediction store (out-of-core), "
            "then activates it for the overlay, Pose Stats and Bouts. Promote "
            "individual frames to edit them (Pose Stats \"Fix this frame\"). The "
            "file's #skeleton header is ignored — the project skeleton is used.");

        ImGui::Separator();

        ImGui::InputText("keypoints3d.csv", &state.keypoints3d_path);
        ImGui::SameLine();
        if (ImGui::Button("Browse##red3d_import")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = state.keypoints3d_path.empty()
                ? pm.project_path : state.keypoints3d_path;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseRed3DCsv", "Choose keypoints3d.csv", ".csv", cfg);
        }

        ImGui::Separator();

        bool can_import = !state.keypoints3d_path.empty() &&
                          skeleton.num_nodes > 0 &&
                          !pm.project_path.empty();
        if (pm.camera_params.empty())
            ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1),
                "No calibration loaded — the overlay and per-frame promotion "
                "need calibration to reproject 3D to the camera views.");

        if (!can_import) ImGui::BeginDisabled();
        if (ImGui::Button("Import & Load")) {
            state.done = false;
            state.error.clear();

            std::string store_dir =
                (std::filesystem::path(pm.project_path) /
                 "predictions" / "red_store").string();
            uint32_t fps =
                (ctx.dc_context && ctx.dc_context->video_fps > 0)
                    ? (uint32_t)std::lround(ctx.dc_context->video_fps)
                    : 0u;
            uint32_t total_frames =
                (ctx.dc_context && ctx.dc_context->total_num_frame > 0 &&
                 ctx.dc_context->total_num_frame != INT_MAX)
                    ? (uint32_t)ctx.dc_context->total_num_frame
                    : 0u;  // 0 → importer derives max frame + 1

            state.result = Red3DImport::import_red3d_csv_to_store(
                state.keypoints3d_path, skeleton.num_nodes, store_dir, fps,
                total_frames);

            if (!state.result.error.empty()) {
                state.error = state.result.error;
            } else {
                state.store_to_load = state.result.store_path;  // main loop activates
                state.done = true;
                ctx.toasts.pushSuccess(
                    "Imported " + std::to_string(state.result.frames_stored) +
                    " frames into the prediction store");
            }
        }
        if (!can_import) ImGui::EndDisabled();

        if (!state.error.empty()) {
            ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "Error: %s",
                               state.error.c_str());
        }

        if (state.done) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0, 1, 0, 1),
                "Stored %d frames (%lld 3D keypoints)",
                state.result.frames_stored, state.result.kp3d_placed);
            ImGui::TextWrapped(
                "Active in the overlay, Pose Stats and Bouts. Use Pose Stats "
                "\"Fix this frame\" to edit a frame in the Labeling Tool.");
        }
        },
        [&]() {
        // File dialog handler
        if (ImGuiFileDialog::Instance()->Display(
                "ChooseRed3DCsv", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto sel = ImGuiFileDialog::Instance()->GetSelection();
                if (!sel.empty())
                    state.keypoints3d_path = sel.begin()->second;
            }
            ImGuiFileDialog::Instance()->Close();
        }
        },
        ImVec2(560, 380));
}
