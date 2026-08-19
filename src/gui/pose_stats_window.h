#pragma once
// pose_stats_window.h — "Pose Stats" window: a confidence time-series for the
// active prediction store, synced to video playback.
//
// Plots mean prediction confidence (across all keypoints) over the whole video,
// plus any individual keypoints the user selects, as ImPlot line plots. A
// vertical line tracks the current playback frame, so as you play you can see
// where prediction quality is high vs. low and jump to sections worth
// inspecting (double-click the plot to seek there).
//
// For long videos the series is downsampled into buckets (≤ 4000) so the plot
// stays responsive regardless of length; the cache is rebuilt only when the
// active store changes.

#include "imgui.h"
#include "implot.h"

// ImPlot v1.0 obsoleted SetNextLineStyle()/SetNextMarkerStyle(); item styling is
// now passed per-call via ImPlotSpec. Built here in a C++17-friendly way
// (designated initialisers would need C++20).
static inline ImPlotSpec red_line_spec(const ImVec4 &col, float weight) {
    ImPlotSpec s;
    s.LineColor = col;
    s.LineWeight = weight;
    return s;
}

#include "skeleton.h"
#include "prediction_store.h"
#include "gui/panel.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct PoseStatsState {
    bool show = false;

    // Downsampled cache, rebuilt when the active store path changes.
    std::string cached_store_path;
    uint32_t total_frames = 0;
    uint32_t stored_frames = 0;
    int num_kp = 0;
    int num_buckets = 0;
    std::vector<float> bucket_x;      // representative frame per bucket [B]
    std::vector<float> overall_mean;  // mean conf across kps [B] (NaN = empty)
    std::vector<float> per_kp_mean;   // kp-major [num_kp * B] (NaN = empty)
    std::vector<char>  kp_selected;   // [num_kp] (char, not bool, for &ref)

    // Click-to-seek request, consumed by the main loop.
    bool seek_requested = false;
    int  seek_frame = 0;

    // A single click is held briefly before it seeks, so a double-click (which
    // resets the view) can cancel it — otherwise the first click of a
    // double-click would seek and its latency could swallow the second click.
    bool   pending_seek = false;
    int    pending_seek_frame = 0;
    double pending_seek_time = 0.0;

    bool refit = false;   // double-click → reset axes to full view next frame

    // "Fix this frame": promote the current frame's prediction into the
    // Labeling Tool for manual correction (consumed by the main loop).
    bool promote_requested = false;
    int  promote_frame = 0;
    std::string promote_status;
};

// One pass over stored frames → bucketed mean-confidence series (overall +
// per-keypoint). Buckets with no stored frames stay NaN so ImPlot leaves gaps.
inline void pose_stats_build(PoseStatsState &st,
                             const predstore::PredictionReader &store,
                             const SkeletonContext &skel) {
    st.total_frames = store.total_frames();
    st.stored_frames = store.stored_frames();
    st.num_kp = skel.num_nodes;
    const int nn = st.num_kp;
    // Never read past a stored frame's block: the store may hold fewer keypoints
    // than the current skeleton. Keypoints beyond nread stay NaN.
    const int nread = std::min(nn, (int)store.num_keypoints());
    const uint32_t total = std::max<uint32_t>(1, st.total_frames);
    const int B = (int)std::min<uint32_t>(total, 4000u);
    st.num_buckets = B;

    st.bucket_x.assign(B, 0.f);
    st.overall_mean.assign(B, std::nanf(""));
    st.per_kp_mean.assign((size_t)nn * B, std::nanf(""));
    if ((int)st.kp_selected.size() != nn) st.kp_selected.assign(nn, 0);

    std::vector<double> ov_sum(B, 0.0);
    std::vector<int> ov_cnt(B, 0);
    std::vector<double> kp_sum((size_t)nn * B, 0.0);
    std::vector<int> kp_cnt((size_t)nn * B, 0);

    for (uint32_t i = 0; i < st.stored_frames; ++i) {
        uint32_t fn = 0;
        const float *row = store.stored_at(i, &fn);
        if (!row) continue;
        int b = (int)(((uint64_t)fn * B) / total);
        if (b < 0) b = 0;
        if (b >= B) b = B - 1;
        double fsum = 0;
        int fcnt = 0;
        for (int k = 0; k < nread; ++k) {
            float c = row[k * 4 + 3];
            if (!std::isnan(c)) {
                fsum += c;
                fcnt++;
                kp_sum[(size_t)k * B + b] += c;
                kp_cnt[(size_t)k * B + b]++;
            }
        }
        if (fcnt > 0) {
            ov_sum[b] += fsum / fcnt;
            ov_cnt[b]++;
        }
    }

    for (int b = 0; b < B; ++b) {
        st.bucket_x[b] = (float)(((uint64_t)b * total) / (uint32_t)B);
        if (ov_cnt[b] > 0) st.overall_mean[b] = (float)(ov_sum[b] / ov_cnt[b]);
        for (int k = 0; k < nn; ++k) {
            int c = kp_cnt[(size_t)k * B + b];
            if (c > 0)
                st.per_kp_mean[(size_t)k * B + b] =
                    (float)(kp_sum[(size_t)k * B + b] / c);
        }
    }
}

inline void DrawPoseStatsWindow(PoseStatsState &st,
                                const predstore::PredictionReader &store,
                                const std::string &active_store_path,
                                SkeletonContext &skel,
                                int current_frame_num) {
    DrawPanel("Pose Stats", st.show, [&]() {
        if (!store.is_open() || active_store_path.empty()) {
            ImGui::TextDisabled("No prediction store loaded.");
            ImGui::TextWrapped(
                "Import predictions with Tools > Import JARVIS "
                "Predictions, sending them to a prediction store.");
            st.cached_store_path.clear();
            return;
        }

        // Rebuild the cache when the active store (or skeleton) changes.
        if (st.cached_store_path != active_store_path ||
            st.num_kp != skel.num_nodes) {
            pose_stats_build(st, store, skel);
            st.cached_store_path = active_store_path;
        }
        if (st.num_buckets <= 0) {
            ImGui::TextDisabled("Empty prediction store.");
            return;
        }

        // One compact control row so the confidence plot gets the vertical
        // space: selection buttons + a floating Keypoints picker + stats + a
        // help marker. The picker is a popup, so it costs no panel height.
        int n_sel = 0;
        for (char s : st.kp_selected) if (s) n_sel++;

        if (ImGui::SmallButton("Mean only"))
            std::fill(st.kp_selected.begin(), st.kp_selected.end(), 0);
        ImGui::SameLine();
        if (ImGui::SmallButton("All keypoints"))
            std::fill(st.kp_selected.begin(), st.kp_selected.end(), 1);
        ImGui::SameLine();
        char kp_btn[48];
        snprintf(kp_btn, sizeof(kp_btn), "Keypoints (%d) v###kp_btn", n_sel);
        if (ImGui::SmallButton(kp_btn)) ImGui::OpenPopup("##kp_popup");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Pick individual keypoints to overlay");
        if (ImGui::BeginPopup("##kp_popup")) {
            if (ImGui::BeginChild("##kp_pick", ImVec2(340, 220), true)) {
                int per_col = 0;
                for (int k = 0; k < st.num_kp; ++k) {
                    ImVec4 col = (k < (int)skel.node_colors.size())
                                     ? skel.node_colors[k]
                                     : ImVec4(1, 1, 1, 1);
                    ImGui::PushStyleColor(ImGuiCol_Text, col);
                    bool sel = st.kp_selected[k] != 0;
                    std::string name = (k < (int)skel.node_names.size())
                                           ? skel.node_names[k]
                                           : ("kp" + std::to_string(k));
                    if (ImGui::Checkbox((name + "##kp" + std::to_string(k)).c_str(),
                                        &sel))
                        st.kp_selected[k] = sel ? 1 : 0;
                    ImGui::PopStyleColor();
                    if (++per_col % 2 != 0) ImGui::SameLine(170);
                    else per_col = 0;
                }
            }
            ImGui::EndChild();
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%u/%u", st.stored_frames, st.total_frames);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Click plot to seek · drag to pan · "
                              "scroll to zoom · double-click to reset view");

        // Promote the current frame into the Labeling Tool for manual fixing.
        ImGui::SameLine();
        bool has_pred = store.frame((uint32_t)current_frame_num) != nullptr;
        if (!has_pred) ImGui::BeginDisabled();
        char fix_lbl[48];
        snprintf(fix_lbl, sizeof(fix_lbl), "Fix frame %d", current_frame_num);
        if (ImGui::SmallButton(fix_lbl)) {
            st.promote_requested = true;
            st.promote_frame = current_frame_num;
        }
        if (!has_pred) ImGui::EndDisabled();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(has_pred
                ? "Promote this predicted frame into the Labeling Tool's\n"
                  "\"Needs Improvement\" list so you can correct its keypoints."
                : "No prediction stored for this frame.");

        // ImPlot's native double-click "fit" snaps X to the DATA extent, which
        // for a sparse / sub-range store is just the predicted region — not the
        // whole video. Move native fit off the left mouse button so our own
        // handler owns left-double-click and resets X to the full [0, total]
        // range instead. Restored after EndPlot so other plots are unaffected.
        ImPlotInputMap &imap = ImPlot::GetInputMap();
        ImGuiMouseButton saved_fit = imap.Fit;
        imap.Fit = ImGuiMouseButton_Middle;

        // Double-click resets the X view to the full range next frame (Y is
        // pinned, so there is nothing to reset there).
        ImPlotCond xcond = st.refit ? ImPlotCond_Always : ImPlotCond_Once;
        st.refit = false;

        if (ImPlot::BeginPlot("##pose_stats", ImVec2(-1, -1))) {
            ImPlot::SetupAxis(ImAxis_X1, "Frame");
            // Confidence is bounded [0,1]; lock the Y axis to that range so it
            // can't be panned/zoomed off (and double-click never rescales it).
            ImPlot::SetupAxis(ImAxis_Y1, "Confidence", ImPlotAxisFlags_Lock);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, (double)st.total_frames, xcond);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Always);

            // Overall mean (bold white).
            ImPlot::PlotLine("Mean", st.bucket_x.data(),
                             st.overall_mean.data(), st.num_buckets,
                                 red_line_spec(ImVec4(1, 1, 1, 1), 2.0f));

            // Selected per-keypoint lines, in their skeleton colors.
            for (int k = 0; k < st.num_kp; ++k) {
                if (!st.kp_selected[k]) continue;
                ImVec4 col = (k < (int)skel.node_colors.size())
                                 ? skel.node_colors[k]
                                 : ImVec4(0.7f, 0.7f, 0.7f, 1);
                std::string name = (k < (int)skel.node_names.size())
                                       ? skel.node_names[k]
                                       : ("kp" + std::to_string(k));
                ImPlot::PlotLine(name.c_str(), st.bucket_x.data(),
                                 st.per_kp_mean.data() + (size_t)k * st.num_buckets,
                                 st.num_buckets,
                                     red_line_spec(col, 1.0f));
            }

            // Current-frame marker.
            double cf = (double)current_frame_num;
            ImPlot::PlotInfLines("##current", &cf, 1,
                                 red_line_spec(ImVec4(1.0f, 0.85f, 0.2f, 0.9f), 1.5f));

            // Single click (no drag) seeks; a second click while that seek is
            // still pending is a double-click → reset the view to full range;
            // left-drag pans. The seek is held for kSeekHoldSec (below) so the
            // double-click can cancel it. We use our own threshold for both the
            // hold and the double-click window so they stay consistent — that
            // lets it be shorter than ImGui's 0.30s default double-click time.
            if (ImPlot::IsPlotHovered() &&
                ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                ImVec2 d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                if (d.x * d.x + d.y * d.y < 9.0f) {  // < 3 px → click, not pan
                    if (st.pending_seek) {
                        st.refit = true;         // 2nd quick click → reset
                        st.pending_seek = false;
                    } else {
                        double mx = ImPlot::GetPlotMousePos().x;
                        if (mx >= 0 && mx < (double)st.total_frames) {
                            st.pending_seek = true;
                            st.pending_seek_frame = (int)(mx + 0.5);
                            st.pending_seek_time = ImGui::GetTime();
                        }
                    }
                }
            }
            ImPlot::EndPlot();
        }
        imap.Fit = saved_fit;  // restore native fit for other plots

        // Commit a held single-click seek once the double-click window passes
        // without a second click arriving.
        constexpr double kSeekHoldSec = 0.20;
        if (st.pending_seek &&
            ImGui::GetTime() - st.pending_seek_time > kSeekHoldSec) {
            st.seek_frame = st.pending_seek_frame;
            st.seek_requested = true;
            st.pending_seek = false;
        }
    }, nullptr, ImVec2(620, 420));
}
