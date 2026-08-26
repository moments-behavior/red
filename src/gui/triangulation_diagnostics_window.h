#pragma once
// triangulation_diagnostics_window.h — Annotation Project diagnostic panel.
// Computes triangulate-and-reproject residuals from the in-memory
// AnnotationMap + loaded CameraParams. Read-only: never mutates labels.

#include "annotation_diagnostics.h"
#include "app_context.h"
#include "gui/panel.h"
#include "skeleton.h"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

struct TriangulationDiagnosticsState {
    bool show = false;
    bool done = false;
    std::string status;
    AnnotationDiagnostics::Diagnostics result;
    bool show_per_keypoint = false;
    std::string last_export_path;
};

inline void DrawTriangulationDiagnosticsWindow(
    TriangulationDiagnosticsState &state, AppContext &ctx) {
    DrawPanel("Triangulation Diagnostics", state.show, [&]() {
        auto &pm = ctx.pm;
        auto *scene = ctx.scene;

        ImGui::TextWrapped(
            "Read-only diagnostic. For each labeled keypoint seen by >= 2 "
            "cameras: triangulate its 3D position from the multi-view "
            "observations, reproject into every camera that labeled it, "
            "and report ||reproj - observed||. Labels are not modified.");
        ImGui::Spacing();

        // ── Status snapshot of inputs ──
        int M = (int)pm.camera_params.size();
        bool videos_loaded = (scene && scene->num_cams > 0);
        int num_nodes = ctx.skeleton.num_nodes;
        int n_frames = (int)ctx.annotations.size();

        ImGui::Text("Cameras loaded:  %d  %s", M,
            M > 0 ? "" : "(no calibration)");
        ImGui::Text("Skeleton:        %s  (%d keypoints)",
            ctx.skeleton.has_skeleton ? ctx.skeleton.name.c_str() : "(none)",
            num_nodes);
        ImGui::Text("Labeled frames:  %d", n_frames);
        ImGui::Text("Videos loaded:   %s",
            videos_loaded ? "yes (for image dimensions)" : "NO - load videos first");
        ImGui::Spacing();

        bool can_run = (M > 0) && videos_loaded &&
                       (num_nodes > 0) && (n_frames > 0);

        ImGui::BeginDisabled(!can_run);
        if (ImGui::Button("Compute Diagnostics")) {
            std::vector<int> image_heights(M, 0);
            for (int m = 0; m < M && m < (int)scene->num_cams; m++) {
                image_heights[m] = scene->image_height[m];
            }
            state.result = AnnotationDiagnostics::compute(
                ctx.annotations, pm.camera_params, image_heights, num_nodes);
            state.done = state.result.success;
            if (state.result.success) {
                char buf[320];
                std::snprintf(buf, sizeof(buf),
                    "%d points triangulated, %d residuals, "
                    "overall RMSE %.4f px  (%d candidates skipped <2 views)",
                    state.result.base.n_points_triangulated,
                    (int)state.result.base.residuals.size(),
                    state.result.base.overall_rmse,
                    state.result.n_candidates_skipped_lt2_views);
                state.status = buf;
            } else {
                state.status = "Error: " + state.result.error;
                state.done = false;
            }
        }
        ImGui::EndDisabled();
        if (!can_run) {
            ImGui::SameLine();
            ImGui::TextDisabled("(need calibration + videos + labels)");
        }

        if (!state.status.empty()) {
            ImGui::Spacing();
            ImGui::TextWrapped("%s", state.status.c_str());
        }

        if (!(state.done && state.result.success)) {
            return;
        }

        const auto &d = state.result;
        const auto &b = d.base;

        ImGui::Spacing();
        ImGui::Separator();

        // ── Overall stats strip ──
        ImGui::Text("Overall (%d residuals across %d points, %d skipped):",
            (int)b.residuals.size(), b.n_points_triangulated,
            b.n_points_skipped);
        ImGui::Indent();
        ImGui::Text("Mean +- s.d.  %.4f +- %.4f px",
            b.overall_mean, b.overall_std);
        ImGui::Text("Median        %.4f px", b.overall_median);
        ImGui::SameLine(260);
        ImGui::Text("P95    %.4f px", b.overall_p95);
        ImGui::Text("Max           %.4f px", b.overall_max);
        ImGui::SameLine(260);
        ImGui::Text("RMSE   %.4f px", b.overall_rmse);
        ImGui::Unindent();

        // ── Histogram ──
        if (!b.residuals.empty()) {
            double hist_max = std::max(b.overall_p95 * 1.25, b.overall_max);
            hist_max = std::max(hist_max, 1.0);
            const int num_bins = 50;
            double bin_width = hist_max / num_bins;
            std::vector<double> bins(num_bins, 0);
            std::vector<double> centers(num_bins);
            for (int i = 0; i < num_bins; i++)
                centers[i] = (i + 0.5) * bin_width;
            for (const auto &r : b.residuals) {
                int bi = std::clamp(
                    (int)(r.error_px / bin_width), 0, num_bins - 1);
                bins[bi]++;
            }
            if (ImPlot::BeginPlot("Reprojection Error Distribution",
                                  ImVec2(-1, 200))) {
                ImPlot::SetupAxes("Error (px)", "Count");
                ImPlot::PlotBars("residuals",
                                 centers.data(), bins.data(),
                                 num_bins, bin_width);
                ImPlot::EndPlot();
            }
        }

        // Build camera display names: prefer pm.camera_names when it aligns
        // with pm.camera_params, else fall back to generic "Cam<idx>".
        int ncam = (int)b.per_camera.size();
        std::vector<std::string> cam_names(ncam);
        bool has_serials =
            (int)pm.camera_names.size() == ncam && !pm.camera_names.empty();
        for (int i = 0; i < ncam; i++) {
            cam_names[i] = has_serials
                ? pm.camera_names[i]
                : b.per_camera[i].name;
        }

        // ── Per-camera table ──
        if (ImGui::BeginTable("##td_per_cam", 8,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Camera");
            ImGui::TableSetupColumn("N");
            ImGui::TableSetupColumn("Mean");
            ImGui::TableSetupColumn("s.d.");
            ImGui::TableSetupColumn("Median");
            ImGui::TableSetupColumn("P95");
            ImGui::TableSetupColumn("Max");
            ImGui::TableSetupColumn("RMSE");
            ImGui::TableHeadersRow();
            for (int i = 0; i < ncam; i++) {
                const auto &pc = b.per_camera[i];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", cam_names[i].c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%d", pc.n_obs);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", pc.mean);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", pc.std);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", pc.median);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", pc.p95);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", pc.max);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", pc.rmse);
            }
            ImGui::EndTable();
        }

        // ── Per-keypoint table (collapsible — Fly50 has 50 kps) ──
        ImGui::Spacing();
        if (ImGui::TreeNode("Per-keypoint breakdown")) {
            if (ImGui::BeginTable("##td_per_kp", 8,
                    ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Keypoint");
                ImGui::TableSetupColumn("N pts");
                ImGui::TableSetupColumn("N obs");
                ImGui::TableSetupColumn("Mean");
                ImGui::TableSetupColumn("s.d.");
                ImGui::TableSetupColumn("Median");
                ImGui::TableSetupColumn("Max");
                ImGui::TableSetupColumn("RMSE");
                ImGui::TableHeadersRow();
                for (const auto &pk : d.per_keypoint) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    const char *kp_name =
                        ((int)pk.kp_id < (int)ctx.skeleton.node_names.size())
                            ? ctx.skeleton.node_names[pk.kp_id].c_str()
                            : "";
                    if (*kp_name)
                        ImGui::Text("%u: %s", pk.kp_id, kp_name);
                    else
                        ImGui::Text("%u", pk.kp_id);
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", pk.n_points);
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", pk.n_obs);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pk.mean);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pk.std);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pk.median);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pk.max);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", pk.rmse);
                }
                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // ── Export CSV ──
        ImGui::Spacing();
        if (ImGui::Button("Export CSV##td_export")) {
            std::string dir = pm.keypoints_root_folder.empty()
                ? pm.project_path : pm.keypoints_root_folder;
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            std::string out_path =
                dir + "/triangulation_diagnostics.csv";
            if (AnnotationDiagnostics::save_csv(d, cam_names, out_path)) {
                state.last_export_path = out_path;
                state.status = "Wrote " +
                    std::to_string(d.base.residuals.size()) +
                    " residuals to " + out_path;
            } else {
                state.status = "Failed to write " + out_path;
            }
        }
        if (!state.last_export_path.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("-> %s", state.last_export_path.c_str());
        }
    }, nullptr, ImVec2(700, 600));
}
