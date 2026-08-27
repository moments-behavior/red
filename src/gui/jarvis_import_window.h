#pragma once
// jarvis_import_window.h — "Import JARVIS Predictions": read a JARVIS
// data3D.csv and send it to one of two destinations.
//
//   Prediction store (default) — streams the 3D into a sparse, memory-mapped
//     .rpred file. Predictions show as a read-only overlay and feed Pose Stats;
//     individual frames are promoted into the Labeling Tool on demand ("Fix
//     this frame"). Nothing but a tiny frame index is held in RAM, so a
//     whole-video import is cheap.
//
//   Editable labels — reprojects to 2D per camera straight into the
//     AnnotationMap, REPLACING whatever is loaded. Every frame lands in memory,
//     so this is for bout-sized files you intend to correct wholesale, not for
//     hours of footage.
//
// Reading/reprojection come from jarvis_import.h and prediction_store.h; this
// panel supplies the project's cameras/skeleton and runs them.
#include "imgui.h"
#include "app_context.h"
#include "annotation.h"
#include "gui/gui_keypoints.h"
#include "jarvis_import.h"
#include "prediction_store.h"
#include "gui/panel.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

struct JarvisImportState {
    bool show = false;
    std::string data3d_path;        // JARVIS data3D.csv chosen by the user
    int dest = 0;                   // 0 = prediction store, 1 = editable labels
    float conf_threshold = 0.0f;    // drop frames whose mean confidence is below
    bool done = false;
    std::string summary;            // success detail shown after a run
    std::string error;
    std::string store_to_load;      // .rpred to activate (consumed by main loop)
};

// Stream JARVIS predictions into a .rpred store. Returns "" on success,
// otherwise an error message. Frames are written in ascending order (std::map
// iterates sorted), which is what PredictionWriter's index requires.
inline std::string jarvis_import_to_store(
    const std::string &data3d_csv, const std::string &store_path,
    int num_nodes, float conf_threshold, uint32_t fps_hint,
    uint32_t total_frames_hint, int *out_frames, int *out_keypoints) {
    std::string err;
    auto preds = JarvisImport::read_jarvis_predictions(data3d_csv, 0.0f, &err);
    if (!err.empty()) return err;
    if (preds.empty()) return "No usable rows in " + data3d_csv;

    const int nkp = (int)preds.begin()->second.positions.size();
    if (out_keypoints) *out_keypoints = nkp;
    if (nkp != num_nodes)
        return "Keypoint count mismatch: file has " + std::to_string(nkp) +
               ", skeleton expects " + std::to_string(num_nodes) + ".";

    predstore::PredictionWriter w;
    if (!w.open(store_path, nkp))
        return "Could not create store: " + store_path;

    std::vector<float> buf((size_t)nkp * 4);
    int written = 0, last_frame = -1;
    for (const auto &[fid, p] : preds) {
        float mean = 0;
        for (float c : p.confidences) mean += c;
        mean /= (float)p.confidences.size();
        if (mean < conf_threshold) continue;
        for (int k = 0; k < nkp; ++k) {
            buf[(size_t)k * 4 + 0] = (float)p.positions[k].x();
            buf[(size_t)k * 4 + 1] = (float)p.positions[k].y();
            buf[(size_t)k * 4 + 2] = (float)p.positions[k].z();
            buf[(size_t)k * 4 + 3] = p.confidences[k];
        }
        if (!w.add_frame((uint32_t)fid, buf.data()))
            return "Write failed at frame " + std::to_string(fid);
        last_frame = fid;
        ++written;
    }
    if (written == 0) return "Every frame fell below the confidence threshold.";

    // total_video_frames drives Pose Stats' x-axis; fall back to the last
    // imported frame when the video length isn't known yet.
    uint32_t total = total_frames_hint > 0 ? total_frames_hint
                                           : (uint32_t)(last_frame + 1);
    if (!w.finalize(total, fps_hint))
        return "Failed to finalize the store (nothing was written).";
    if (out_frames) *out_frames = written;
    return "";
}

// Build editable annotations straight from a JARVIS data3D.csv — no CSV round
// trip. Reprojection uses reproject_3d_to_cam() so results match exactly what
// "Fix this frame" produces when promoting from a store.
//
// REPLACES the annotation map: whatever was loaded is cleared first. The map is
// only cleared once the file has parsed and the keypoint count checks out, so a
// bad file leaves the existing labels untouched.
struct JarvisLabelImportStats {
    int frames_imported = 0;
    int frames_filtered = 0;   // below the confidence threshold
    int num_keypoints   = 0;
};

inline std::string jarvis_import_to_labels(
    const std::string &data3d_csv, AnnotationMap &amap,
    const SkeletonContext &skel, const std::vector<CameraParams> &cams,
    const RenderScene *scene, float conf_threshold,
    JarvisLabelImportStats *out) {
    std::string err;
    auto preds = JarvisImport::read_jarvis_predictions(data3d_csv, 0.0f, &err);
    if (!err.empty()) return err;
    if (preds.empty()) return "No usable rows in " + data3d_csv;

    const int nkp = (int)preds.begin()->second.positions.size();
    out->num_keypoints = nkp;
    if (nkp != skel.num_nodes)
        return "Keypoint count mismatch: file has " + std::to_string(nkp) +
               ", skeleton expects " + std::to_string(skel.num_nodes) + ".";

    // Validation is done — safe to discard the current labels.
    amap.clear();

    const int ncam = scene ? (int)scene->num_cams : 0;
    for (const auto &[fid, p] : preds) {
        float mean = 0;
        for (float c : p.confidences) mean += c;
        mean /= (float)p.confidences.size();
        if (mean < conf_threshold) { out->frames_filtered++; continue; }

        FrameAnnotation &fa =
            get_or_create_frame(amap, (u32)fid, skel.num_nodes, ncam);
        for (int k = 0; k < nkp; ++k) {
            const Eigen::Vector3d &p3d = p.positions[k];
            const float c = p.confidences[k];
            if (std::isnan(p3d.x()) || std::isnan(p3d.y()) || std::isnan(p3d.z()))
                continue;
            fa.kp3d[k].x = p3d.x();
            fa.kp3d[k].y = p3d.y();
            fa.kp3d[k].z = p3d.z();
            fa.kp3d[k].set_imported(c);
            for (int cam = 0; cam < ncam && cam < (int)cams.size(); ++cam) {
                double px, py;
                if (reproject_3d_to_cam(p3d, cams[cam],
                                        (int)scene->image_width[cam],
                                        (int)scene->image_height[cam], px, py)) {
                    auto &kp2d = fa.cameras[cam].keypoints[k];
                    kp2d.x = px; kp2d.y = py; kp2d.labeled = true;
                    kp2d.confidence = c;
                    kp2d.source = LabelSource::Predicted;
                }
            }
        }
        out->frames_imported++;
    }
    if (out->frames_imported == 0)
        return "Nothing imported — all " + std::to_string(out->frames_filtered) +
               " frames fell below the confidence threshold.";
    return "";
}

inline void DrawJarvisImportWindow(JarvisImportState &state, AppContext &ctx) {
    auto &pm = ctx.pm;
    const auto &skeleton = ctx.skeleton;

    DrawPanel("Import JARVIS Predictions", state.show,
        [&]() {
        ImGui::SeparatorText("Import a JARVIS data3D.csv");

        ImGui::Text("Skeleton:  %s", skeleton.name.c_str());
        ImGui::Text("Nodes:     %d", skeleton.num_nodes);
        ImGui::Text("Cameras:   %d", (int)pm.camera_names.size());

        ImGui::Separator();

        ImGui::InputText("data3D.csv", &state.data3d_path);
        ImGui::SameLine();
        if (ImGui::Button("Browse##jarvis_import")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = pm.project_path.empty() ? "." : pm.project_path;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseJarvisData3D", "Choose data3D.csv", ".csv", cfg);
        }

        ImGui::SliderFloat("Min mean confidence", &state.conf_threshold,
                           0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Frames whose mean keypoint confidence is below "
                              "this are skipped entirely.");

        ImGui::SeparatorText("Send to");
        ImGui::RadioButton("Prediction store (read-only, out-of-core)",
                           &state.dest, 0);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Shows as an overlay and feeds Pose Stats. "
                              "Promote single frames to edit them. Safe for "
                              "whole-video files.");
        ImGui::RadioButton("Editable labels (in-memory, replaces current)",
                           &state.dest, 1);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Loads straight into the Labeling Tool as "
                              "editable keypoints, discarding whatever labels "
                              "are open. Every frame is held in memory — use "
                              "for bout-sized files.");
        if (state.dest == 1 && !ctx.annotations.empty())
            ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.2f, 1.0f),
                               "Replaces %d loaded frame(s) — save first if "
                               "they hold unsaved work.",
                               (int)ctx.annotations.size());

        // Preconditions. The store path needs no calibration (it holds 3D and
        // the overlay reprojects on the fly); the label path does, because it
        // must bake 2D positions into the CSVs.
        const bool no_project = pm.project_path.empty();
        const bool no_skel    = !skeleton.has_skeleton || skeleton.num_nodes <= 0;
        const bool no_file    = state.data3d_path.empty();
        const bool needs_cal  = (state.dest == 1) && pm.camera_params.empty();
        const char *blocked =
            no_project ? "Open a project first."
            : no_skel  ? "The project has no skeleton loaded."
            : no_file  ? "Choose a data3D.csv to import."
            : needs_cal ? "No calibration loaded — cannot bake 2D labels. "
                          "Import to a prediction store instead."
                        : nullptr;

        ImGui::Separator();
        ImGui::BeginDisabled(blocked != nullptr);
        if (ImGui::Button("Import")) {
            state.done = false;
            state.error.clear();
            state.summary.clear();

            if (state.dest == 0) {
                namespace fs = std::filesystem;
                std::string root = pm.project_path + "/predictions";
                std::error_code ec;
                fs::create_directories(root, ec);
                auto t = std::chrono::system_clock::to_time_t(
                    std::chrono::system_clock::now());
                char buf[32];
                std::strftime(buf, sizeof(buf), "%Y_%m_%d_%H_%M_%S",
                              std::localtime(&t));
                std::string store_path =
                    root + "/jarvis_" + std::string(buf) + ".rpred";

                uint32_t fps_hint = 0, total_hint = 0;
                if (ctx.dc_context) {
                    if (ctx.dc_context->video_fps > 0)
                        fps_hint = (uint32_t)std::lround(ctx.dc_context->video_fps);
                    if (ctx.dc_context->total_num_frame > 0)
                        total_hint = (uint32_t)ctx.dc_context->total_num_frame;
                }

                int frames = 0, nkp = 0;
                std::string err = jarvis_import_to_store(
                    state.data3d_path, store_path, skeleton.num_nodes,
                    state.conf_threshold, fps_hint, total_hint, &frames, &nkp);
                if (!err.empty()) {
                    state.error = err;
                } else {
                    state.store_to_load = store_path;  // main loop activates
                    state.done = true;
                    state.summary =
                        "Stored " + std::to_string(frames) + " frames to " +
                        store_path;
                    ctx.toasts.pushSuccess("Imported " +
                        std::to_string(frames) + " predicted frames");
                }
            } else {
                JarvisLabelImportStats st{};
                std::string err = jarvis_import_to_labels(
                    state.data3d_path, ctx.annotations, skeleton,
                    pm.camera_params, ctx.scene, state.conf_threshold, &st);
                if (!err.empty()) {
                    state.error = err;
                } else {
                    pm.plot_keypoints_flag = true;
                    state.done = true;
                    state.summary =
                        "Loaded " + std::to_string(st.frames_imported) +
                        " frames as editable labels, replacing what was open (" +
                        std::to_string(st.frames_filtered) +
                        " skipped below threshold)";
                    ctx.toasts.pushSuccess("Imported " +
                        std::to_string(st.frames_imported) + " frames");
                }
            }
        }
        ImGui::EndDisabled();
        if (blocked) {
            ImGui::SameLine();
            ImGui::TextDisabled("%s", blocked);
        }

        if (!state.error.empty()) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.90f, 0.28f, 0.28f, 1.0f), "%s",
                               state.error.c_str());
        } else if (state.done) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.35f, 0.80f, 0.45f, 1.0f), "Import OK");
            ImGui::TextWrapped("%s", state.summary.c_str());
            ImGui::TextDisabled(state.dest == 0
                ? "Open Pose Stats to review confidence; use \"Fix this frame\" "
                  "to promote a frame into the Labeling Tool."
                : "These are normal editable labels — correct them in the "
                  "Labeling Tool and Ctrl+S to save.");
        }
        },
        // always_fn: the file dialog must be pumped even when the panel is hidden.
        [&]() {
            if (ImGuiFileDialog::Instance()->Display(
                    "ChooseJarvisData3D", ImGuiWindowFlags_NoCollapse,
                    ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk())
                    state.data3d_path =
                        ImGuiFileDialog::Instance()->GetFilePathName();
                ImGuiFileDialog::Instance()->Close();
            }
        });
}
