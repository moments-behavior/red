#pragma once
// export_window.h -- Unified export window with format selector
//
// Replaces jarvis_export_window.h. Same layout pattern but with a format
// dropdown at top. Format-specific options appear/disappear based on selection.
//
// Thread safety: export runs on a detached thread. The thread writes to its own
// local std::string (thread_status), never to state.status directly. When the
// thread finishes, the final status is copied to state.status via atomic flag.
// state.in_progress is std::atomic<bool> for safe cross-thread signaling.

#include "imgui.h"
#include "annotation.h"
#include "app_context.h"
#include "export_formats.h"
#include "annotation_csv.h"
#include "gui/panel.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

struct ExportWindowState {
    bool show = false;
    int format_idx = 0; // 0=JARVIS, 1=COCO, 2=DLC, 3=YOLO Pose, 4=YOLO Detect, 5=Nerfstudio, 6=tailcycle
    // One row per session to write. The format makes split a directory level,
    // so a session belongs wholly to one split -- you build train/val/test by
    // exporting several ranges, not by ratio-splitting one. Frame-level random
    // splits are what rule 14 warns against: at 180 fps, frame N in train and
    // N+1 in val are near-identical, so the val score measures memorisation.
    struct TailcycleRange {
        int start = 0;
        int end = 0;            // inclusive; 0 with start 0 means "whole video"
        int split_idx = 0;      // train | val | test
    };
    std::vector<TailcycleRange> tailcycle_ranges{{}};
    char tailcycle_session_id[128] = "";
    bool tailcycle_include_triangulated_3d = false;
    std::string tailcycle_range_error;
    bool include_video_index = false; // JARVIS: include video_index.json
    int scale_factor = 1; // JARVIS: write calibration so 3D reconstructs in (mm × scale_factor)
    std::string output_dir;
    float margin = 50.0f;
    float train_ratio = 0.9f;
    int seed = 42;
    int jpeg_quality = 95;
    std::atomic<bool> in_progress{false};
    std::string status; // only written by main thread

    // Progress counters (written by export thread via atomic, read by main thread)
    std::atomic<int> images_saved{0};
    int images_total = 0; // set before thread launch, read-only during export

    // Thread → main thread communication for final status.
    // The thread writes to finished_status, then sets finished flag.
    // Main thread polls finished flag; when set, copies finished_status
    // into status and clears finished. This avoids any concurrent access
    // to std::string — the handoff is sequenced by the atomic.
    std::shared_ptr<std::string> finished_status; // heap-allocated, owned by thread+main
    std::atomic<bool> finished{false};

    // Cached label folder detection
    std::string label_folder;
    std::string label_display = "(none)";
    std::string label_cache_key;
};

inline void DrawExportWindow(ExportWindowState &state, AppContext &ctx,
                              AnnotationMap &amap) {
    const auto &pm = ctx.pm;
    const auto &skeleton = ctx.skeleton;

    // Poll for thread completion (runs every frame, even when window is hidden)
    if (state.finished.load(std::memory_order_acquire)) {
        // Thread is done and has released finished_status — safe to read
        if (state.finished_status)
            state.status = *state.finished_status;
        state.finished_status.reset();
        state.in_progress.store(false, std::memory_order_relaxed);
        state.finished.store(false, std::memory_order_relaxed);
    }

    DrawPanel("Export Tool", state.show,
        [&]() {

        // Format selector. Nerfstudio/3DGS needs camera calibration, so it is
        // omitted for 2D (uncalibrated) projects. The 2D formats (JARVIS, COCO,
        // DeepLabCut, YOLO) read per-camera 2D labels only.
        ImGui::SeparatorText("Format");
        std::vector<const char *> format_labels = {
            "JARVIS", "COCO Keypoints", "DeepLabCut", "YOLO Pose",
            "YOLO Detection"};
        std::vector<ExportFormats::Format> format_map = {
            ExportFormats::JARVIS, ExportFormats::COCO,
            ExportFormats::DEEPLABCUT, ExportFormats::YOLO_POSE,
            ExportFormats::YOLO_DETECT};
        if (!project_is_2d(pm)) {
            format_labels.push_back("Nerfstudio / 3DGS");
            format_map.push_back(ExportFormats::NERFSTUDIO);
        }
        // Offered only when this build has Arrow. Hiding it beats showing a
        // button that always fails -- same reason Nerfstudio is hidden for
        // uncalibrated projects.
        if (TailcycleExport::available()) {
            format_labels.push_back("tailcycle-dataset");
            format_map.push_back(ExportFormats::TAILCYCLE);
        }
        if (state.format_idx >= (int)format_labels.size())
            state.format_idx = 0;
        ImGui::Combo("Export Format", &state.format_idx, format_labels.data(),
                     (int)format_labels.size());

        auto fmt = format_map[state.format_idx];
        bool is_jarvis = (fmt == ExportFormats::JARVIS);

        // tailcycle-specific: one row per session, plus the 3D layer.
        if (fmt == ExportFormats::TAILCYCLE) {
            if (state.tailcycle_session_id[0] == '\0')
                snprintf(state.tailcycle_session_id,
                         sizeof(state.tailcycle_session_id), "%s",
                         pm.project_name.c_str());
            ImGui::InputText("Session ID", state.tailcycle_session_id,
                             sizeof(state.tailcycle_session_id));
            ImGui::SetItemTooltip(
                "Becomes the folder name, which IS the session id. Shared by "
                "every row below -- the split directory keeps them apart.");

            ImGui::SeparatorText("Splits");
            ImGui::TextDisabled(
                "One session per row. Leave end at 0 for the whole recording.");
            ImGui::TextDisabled(
                "Frames are extracted into each group, so the dataset is "
                "self-contained.");
            static const char *kSplits[] = {"train", "val", "test"};
            int remove_at = -1;
            for (size_t i = 0; i < state.tailcycle_ranges.size(); i++) {
                auto &r = state.tailcycle_ranges[i];
                ImGui::PushID((int)i);
                ImGui::SetNextItemWidth(90);
                ImGui::InputInt("##start", &r.start, 0, 0);
                if (r.start < 0) r.start = 0;
                ImGui::SameLine();
                ImGui::SetNextItemWidth(90);
                ImGui::InputInt("##end", &r.end, 0, 0);
                if (r.end < 0) r.end = 0;
                ImGui::SameLine();
                ImGui::SetNextItemWidth(90);
                ImGui::Combo("##split", &r.split_idx, kSplits, 3);
                ImGui::SameLine();
                ImGui::BeginDisabled(state.tailcycle_ranges.size() == 1);
                if (ImGui::Button("x")) remove_at = (int)i;
                ImGui::EndDisabled();
                if (i == 0) {
                    ImGui::SameLine();
                    ImGui::TextDisabled("start / end / split");
                }
                ImGui::PopID();
            }
            if (remove_at >= 0)
                state.tailcycle_ranges.erase(state.tailcycle_ranges.begin() + remove_at);
            if (ImGui::Button("+ Add split"))
                state.tailcycle_ranges.push_back({});

            // Overlapping ranges are the leak rule 14 exists to prevent, and a
            // validator only warns about it -- so catch it here, where it can
            // still be fixed.
            std::string range_err;
            for (size_t i = 0; i < state.tailcycle_ranges.size() && range_err.empty(); i++) {
                const auto &a = state.tailcycle_ranges[i];
                if (a.end != 0 && a.end < a.start) range_err = "A range ends before it starts.";
                for (size_t j = i + 1; j < state.tailcycle_ranges.size(); j++) {
                    const auto &b = state.tailcycle_ranges[j];
                    const int ae = a.end ? a.end : INT_MAX, be = b.end ? b.end : INT_MAX;
                    if (a.start <= be && b.start <= ae) {
                        range_err = "Ranges overlap -- the same frames would land in "
                                    "two splits.";
                        break;
                    }
                    if (a.split_idx == b.split_idx)
                        range_err = "Two rows share a split, so they would write the "
                                    "same session folder twice.";
                }
            }
            if (!range_err.empty())
                ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f), "%s", range_err.c_str());
            state.tailcycle_range_error = range_err;

            ImGui::Spacing();
            ImGui::Checkbox("Include triangulated 3D",
                            &state.tailcycle_include_triangulated_3d);
            ImGui::SetItemTooltip(
                "Off: the session ships 2D labels and calibration, and a consumer "
                "triangulates for itself.\n"
                "Triangulation is the only source of 3D in red, so OFF means no 3D "
                "layer at all -- not a reduced one.\n"
                "On: ships red's own solve, for a consumer that wants these exact "
                "numbers rather than its own.");

        }

        // JARVIS-specific: video index checkbox
        if (is_jarvis) {
            ImGui::Checkbox("Include video index (for semi-supervised training)",
                            &state.include_video_index);
            ImGui::SliderInt("Scale factor (x1-x100; for tightly-spaced keypoints)",
                             &state.scale_factor, 1, 100);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Write the calibration so JARVIS reconstructs 3D in\n"
                    "(mm x scale factor), so the integer-mm voxel grid resolves\n"
                    "e.g. a ~3mm fly. Telecentric: projectionMatrix[0:2,0:3] is\n"
                    "divided by the factor; perspective: T is multiplied by it.\n"
                    "Predicted 3D comes out scaled — divide by the factor for mm.\n"
                    "Check for the fly rig.");
        }

        ImGui::SeparatorText("Project Info");

        // Auto-detect label folder (cached)
        if (state.label_cache_key != pm.keypoints_root_folder) {
            state.label_cache_key = pm.keypoints_root_folder;
            state.label_folder.clear();
            state.label_display = "(none)";
            if (!pm.keypoints_root_folder.empty()) {
                std::string most_recent, tmp_err;
                if (AnnotationCSV::find_most_recent_labels(pm.keypoints_root_folder,
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

        ImGui::Text("Video Folder: %s",
                    pm.media_folder.empty() ? "(none — images will not be extracted)" : pm.media_folder.c_str());
        ImGui::Text("Cameras:      %d", (int)pm.camera_names.size());

        int kp_count = 0;
        for (const auto &[f, fa] : amap)
            if (frame_has_any_keypoints(fa)) ++kp_count;
        ImGui::Text("Annotated:    %d frames", kp_count);

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

        bool is_nerfstudio = (fmt == ExportFormats::NERFSTUDIO);

        // Common options (not applicable to Nerfstudio)
        if (!is_nerfstudio) {
            ImGui::SliderFloat("Train Ratio", &state.train_ratio, 0.5f, 0.99f);
            ImGui::InputInt("Random Seed", &state.seed);
        }

        // Format-specific options
        if (is_jarvis || fmt == ExportFormats::COCO ||
            fmt == ExportFormats::YOLO_POSE || fmt == ExportFormats::YOLO_DETECT) {
            ImGui::SliderFloat("Bbox Margin (px)", &state.margin, 0.0f, 200.0f);
        }
        if (is_nerfstudio) {
            ImGui::TextWrapped(
                "Exports camera calibration as transforms.json and extracts "
                "JPEG frames for 3D Gaussian Splatting / novel-view synthesis. "
                "Uses all annotated frames.");
            if (pm.camera_params.empty())
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                    "No calibration loaded — camera params required.");
        }
        if (!pm.media_folder.empty()) {
            ImGui::SliderInt("JPEG Quality", &state.jpeg_quality, 10, 100);
        }

        ImGui::Separator();

        // Export button
        if (!state.in_progress.load(std::memory_order_relaxed)) {
            std::string validation_error;
            // Resolve actual dispatch format (JARVIS vs JARVIS_TR based on checkbox)
            auto dispatch_fmt = fmt;
            if (is_jarvis && state.include_video_index)
                dispatch_fmt = ExportFormats::JARVIS_TR;

            if (ImGui::Button("Start Export")) {
                if (state.label_folder.empty() && kp_count == 0 && !is_nerfstudio) {
                    validation_error = "No annotations found";
                } else if (!project_is_2d(pm) && pm.calibration_folder.empty()) {
                    validation_error = "No calibration folder set";
                } else if (is_jarvis && pm.media_folder.empty()) {
                    validation_error = "No media folder set (required for JARVIS)";
                } else if (is_nerfstudio && pm.camera_params.empty()) {
                    validation_error = "No camera calibration loaded (required for Nerfstudio)";
                } else if (state.output_dir.empty()) {
                    validation_error = "Output directory not set";
                } else if (pm.camera_names.empty()) {
                    validation_error = "No cameras loaded";
                } else if (dispatch_fmt == ExportFormats::TAILCYCLE &&
                           !state.tailcycle_range_error.empty()) {
                    validation_error = state.tailcycle_range_error;
                } else if (dispatch_fmt == ExportFormats::TAILCYCLE &&
                           pm.media_folder.empty()) {
                    validation_error =
                        "No media folder set -- the group length must come from "
                        "the recording, not the labels";
                } else {
                    // Auto-save annotations before export so CSVs on disk are current.
                    // JARVIS reads from disk; other formats use AnnotationMap directly
                    // but saving is still good practice to avoid data loss.
                    if (!pm.keypoints_root_folder.empty() && !amap.empty()) {
                        std::string save_err;
                        std::string saved = AnnotationCSV::save_all(
                            pm.keypoints_root_folder, skeleton.name,
                            amap, ctx.scene ? (int)ctx.scene->num_cams : 0,
                            skeleton.num_nodes, pm.camera_names, &save_err);
                        if (!saved.empty()) {
                            // Update label folder to the freshly saved one
                            state.label_folder = saved;
                            state.label_display =
                                std::filesystem::path(saved).filename().string();
                            // Invalidate cache so next open re-detects
                            state.label_cache_key.clear();
                        }
                        // If save fails, proceed anyway — the old labels are still on disk
                    }

                    // Compute total expected images for progress bar
                    state.images_saved.store(0, std::memory_order_relaxed);
                    state.images_total = kp_count * (int)pm.camera_names.size();
                    state.in_progress.store(true, std::memory_order_relaxed);
                    state.status = "Exporting...";

                    ExportFormats::ExportConfig ecfg;
                    ecfg.format             = dispatch_fmt;
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
                    ecfg.camera_params      = pm.camera_params;
                    ecfg.telecentric        = pm.telecentric;
                    ecfg.scale_factor       = state.scale_factor;
                    // Per-camera image dims from the loaded video, so 2D /
                    // uncalibrated projects (no calibration YAML) can export.
                    if (ctx.scene) {
                        for (size_t i = 0; i < pm.camera_names.size(); i++) {
                            int w = (i < ctx.scene->num_cams)
                                        ? (int)ctx.scene->image_width[i] : 0;
                            int h = (i < ctx.scene->num_cams)
                                        ? (int)ctx.scene->image_height[i] : 0;
                            ecfg.image_width.push_back(w);
                            ecfg.image_height.push_back(h);
                        }
                    }
                    ecfg.node_names         = skeleton.node_names;
                    for (const auto &e : skeleton.edges)
                        ecfg.edges.push_back({e.x, e.y});

                    std::vector<ExportWindowState::TailcycleRange> tc_rows;
                    if (dispatch_fmt == ExportFormats::TAILCYCLE) {
                        ecfg.tailcycle_session_id = state.tailcycle_session_id;
                        ecfg.tailcycle_include_triangulated_3d =
                            state.tailcycle_include_triangulated_3d;
                        // n_frames must describe the media, not the labels: every
                        // frame index in the tables is validated against it, and
                        // the annotation range is usually a sparse subset.
                        if (ctx.input_is_imgs) {
                            ecfg.tailcycle_n_frames = (int)ctx.imgs_names.size();
                        } else if (!ctx.demuxers.empty() && ctx.demuxers[0]) {
                            ecfg.tailcycle_n_frames =
                                (int)ctx.demuxers[0]->GetNumFrames();
                            ecfg.tailcycle_fps =
                                (float)ctx.demuxers[0]->GetFramerate();
                        }
                        tc_rows = state.tailcycle_ranges;
                    }

                    // Copy the annotation map for thread safety
                    AnnotationMap amap_copy = amap;

                    // Allocate shared_ptr for thread → main status handoff.
                    // The thread writes the final status into this string,
                    // then sets the atomic finished flag. The main thread
                    // reads it on the next frame and copies to state.status.
                    auto result_status = std::make_shared<std::string>();
                    state.finished_status = result_status;

                    std::thread(
                        [ecfg, amap_copy, tc_rows, result_status, &state]() {
                            // Thread-local string for export_dataset to write into.
                            // No other thread touches this string.
                            std::string thread_status;
                            if (ecfg.format == ExportFormats::TAILCYCLE) {
                                // One session per row: split is a directory level,
                                // so each range is its own export.
                                static const char *kSplits[] = {"train", "val", "test"};
                                for (const auto &r : tc_rows) {
                                    ExportFormats::ExportConfig one = ecfg;
                                    one.tailcycle_frame_start = r.start;
                                    one.tailcycle_frame_end = r.end;
                                    one.tailcycle_split = kSplits[r.split_idx];
                                    std::string one_status;
                                    const bool ok = ExportFormats::export_dataset(
                                        one.format, one, amap_copy, &one_status,
                                        &state.images_saved);
                                    if (!thread_status.empty()) thread_status += "\n";
                                    thread_status += one_status;
                                    if (!ok) break;   // a refusal applies to them all
                                }
                            } else
                            ExportFormats::export_dataset(
                                ecfg.format, ecfg, amap_copy, &thread_status,
                                &state.images_saved);
                            // Copy final status into the shared handoff string,
                            // then signal completion. The release fence ensures
                            // the string write is visible before the flag.
                            *result_status = std::move(thread_status);
                            state.finished.store(true, std::memory_order_release);
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

            // Progress bar
            if (state.images_total > 0) {
                int saved = state.images_saved.load(std::memory_order_relaxed);
                float progress = (float)saved / (float)state.images_total;
                char overlay[64];
                snprintf(overlay, sizeof(overlay), "%d / %d images", saved, state.images_total);
                ImGui::ProgressBar(progress, ImVec2(-1, 0), overlay);
            }
        }

        // Status display
        if (!state.status.empty()) {
            if (state.status.find("Error") != std::string::npos)
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", state.status.c_str());
            else if (state.status.find("complete") != std::string::npos ||
                     state.status.find("Complete") != std::string::npos)
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
