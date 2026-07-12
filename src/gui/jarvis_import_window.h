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
// keypoints3d.csv) into the editable Labeling Tool: reprojects the 3D to 2D per
// camera, writes a proper labeled_data/<timestamp>/ folder (correct #skeleton
// header + per-camera 2D CSVs), and merges the result into the live annotations
// so it can be viewed/edited exactly like an auto-JARVIS prediction.
struct JarvisImportState {
    bool show = false;
    std::string keypoints3d_path;   // path to a red_csv v2 keypoints3d.csv
    bool done = false;
    Red3DImport::ImportStats result;
    std::string saved_folder;       // labeled_data/<ts>/ written on success
    std::string store_to_load;      // .rpred to activate (consumed by main loop)
    std::string store_status;       // prediction-store result line for the panel
    std::string error;
};

inline void DrawJarvisImportWindow(JarvisImportState &state, AppContext &ctx) {
    auto &pm = ctx.pm;
    const auto &skeleton = ctx.skeleton;
    auto &scene = ctx.scene;
    auto &annotations = ctx.annotations;

    DrawPanel("Import 3D Predictions", state.show,
        [&]() {
        ImGui::SeparatorText("Import predicted keypoints3d.csv");

        ImGui::Text("Skeleton:    %s", skeleton.name.c_str());
        ImGui::Text("Nodes:       %d", skeleton.num_nodes);
        ImGui::Text("Cameras:     %d", (int)pm.camera_names.size());
        ImGui::TextWrapped(
            "Reprojects each 3D keypoint into every camera as an editable "
            "prediction and saves to labeled_data/. The file's #skeleton header "
            "is ignored — the project skeleton above is used.");

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
                "No calibration loaded — 3D will import but 2D views stay empty.");

        ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1),
            "Large files (whole-video predictions) may freeze the UI for a while.");

        if (!can_import) ImGui::BeginDisabled();
        if (ImGui::Button("Import & Load")) {
            state.done = false;
            state.error.clear();
            state.saved_folder.clear();
            state.store_status.clear();

            // Per-camera pixel dims: prefer the loaded video, fall back to
            // calibration image size.
            int num_cams = (int)pm.camera_params.size();
            std::vector<int> img_w(num_cams), img_h(num_cams);
            for (int c = 0; c < num_cams; ++c) {
                int w = (scene && c < (int)scene->num_cams && scene->image_width)
                            ? (int)scene->image_width[c]
                            : pm.camera_params[c].image_width;
                int h = (scene && c < (int)scene->num_cams && scene->image_height)
                            ? (int)scene->image_height[c]
                            : pm.camera_params[c].image_height;
                img_w[c] = w;
                img_h[c] = h;
            }

            AnnotationMap loaded;
            state.result = Red3DImport::import_red3d_csv(
                state.keypoints3d_path, loaded, skeleton.num_nodes,
                pm.camera_params, img_w, img_h);

            if (!state.result.error.empty()) {
                state.error = state.result.error;
            } else if (state.result.frames_imported == 0) {
                state.error = "No frames with valid 3D keypoints found.";
            } else {
                // Persist to labeled_data/<ts>/ (correct #skeleton + 2D CSVs).
                std::string save_err;
                state.saved_folder = AnnotationCSV::save_all(
                    pm.keypoints_root_folder, skeleton.name, loaded,
                    num_cams, skeleton.num_nodes, pm.camera_names, &save_err);
                if (state.saved_folder.empty()) {
                    state.error = "Save failed: " + save_err;
                } else {
                    // Also write a .rpred prediction store from the imported 3D
                    // so the Bouts / Pose Stats tools (which read only from the
                    // store, not the editable AnnotationMap) can see it. Done
                    // BEFORE the merge below moves `loaded` frames out.
                    {
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
                                : 0u;  // 0 → writer derives max frame + 1
                        Red3DImport::StoreWriteResult sr =
                            Red3DImport::write_store_from_map(
                                loaded, skeleton.num_nodes, store_dir, fps,
                                total_frames);
                        if (sr.ok) {
                            state.store_to_load = sr.path;  // main loop activates
                            state.store_status =
                                "Prediction store: " +
                                std::to_string(sr.frames_stored) +
                                " frames (Bouts / Pose Stats)";
                        } else {
                            state.store_status =
                                "Prediction store not written: " + sr.error;
                        }
                    }

                    // Merge into the live Labeling Tool map (predictions win over
                    // any existing entry for the same frame).
                    for (auto &[frame, fa] : loaded)
                        annotations[frame] = std::move(fa);
                    pm.plot_keypoints_flag = true;   // reveal the Labeling Tool
                    state.done = true;
                    ctx.toasts.pushSuccess(
                        "Imported " + std::to_string(state.result.frames_imported) +
                        " frames into the Labeling Tool");
                }
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
                "Imported %d / %d frames", state.result.frames_imported,
                state.result.frames_total);
            ImGui::Text("3D keypoints: %lld   Reprojected 2D: %lld",
                        state.result.kp3d_placed, state.result.kp2d_placed);
            ImGui::TextWrapped("Saved to: %s", state.saved_folder.c_str());
            if (!state.store_status.empty())
                ImGui::TextWrapped("%s", state.store_status.c_str());
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
