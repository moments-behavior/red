#pragma once
// frame_drops_window.h — "Frame Drops" window: per-camera dropped-frame
// visualization for the desync-fix sync plan (g_sync_fix.plan).
//
// Two stacked plots with linked X axes (canonical trigger slots):
//   top    — one lane per camera; red marks at each gap (drawn at least 2 px
//            wide so a 10-frame gap is visible on a 500k-slot timeline), gray
//            shading where the camera hasn't started / has ended;
//   bottom — binned frames-lost curves (bold "All" aggregate + one line per
//            camera, legend toggles visibility).
// A vertical line tracks the current playback frame; clicking either plot
// seeks there (same debounced click/double-click scheme as Pose Stats).
//
// The window always draws in canonical-slot coordinates. When the Sync fix is
// OFF the playback timeline is mp4 indices, so the cursor and seek targets are
// translated through the reference camera (pm.camera_names[0]) — the same
// convention sync_fix_toggle uses.

#include "imgui.h"
#include "implot.h"
#include "app_context.h"
#include "global.h"
#include "gui/panel.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

struct FrameDropsState {
    bool show = false;

    // Bucketed frames-lost cache, rebuilt when the plan changes.
    std::string cached_key;
    int num_buckets = 0;
    std::vector<float> bucket_x;            // representative slot per bucket [B]
    std::vector<float> per_cam_lost;        // cam-major [n_cams * B]
    std::vector<float> agg_lost;            // [B]
    std::vector<std::string> cam_names;     // lane order = viewer camera order
    int64_t total_gaps = 0;
    int64_t total_lost = 0;

    // Click-to-seek request, consumed by the main loop (same scheme as
    // pending_seek_slot is canonical
    // and is mapped to a playback frame only when the seek commits.
    bool seek_requested = false;
    int  seek_frame = 0;
    bool    pending_seek = false;
    int64_t pending_seek_slot = 0;
    double  pending_seek_time = 0.0;
    bool refit = false;   // double-click → reset X to full range next frame

    float row_ratios[2] = {0.62f, 0.38f};   // subplot splitter (persists)
};

// Playback frame -> canonical slot (identity when the sync fix is on).
inline int64_t frame_drops_cursor_slot(const sync_plan::SyncPlan &plan,
                                       const sync_plan::SyncCam *cam0,
                                       bool fix_on, int64_t cur) {
    if (fix_on || !cam0)
        return std::clamp<int64_t>(cur, 0, plan.canonical_len - 1);
    return cam0->slot_of_pos(std::clamp<int64_t>(cur, 0, cam0->decoded_len - 1));
}

// Canonical slot -> playback seek frame (cam0's mp4 index when the fix is off).
inline int64_t frame_drops_seek_frame(const sync_plan::SyncCam *cam0,
                                      bool fix_on, int64_t slot) {
    if (fix_on || !cam0) return slot;
    return cam0->seek_pos(slot);
}

// Bucket every camera's gaps into a ≤4000-point frames-lost series. Iterates
// gaps (a few hundred), never slots, splitting each gap's lost count
// proportionally across the buckets it overlaps.
inline void frame_drops_build(FrameDropsState &st,
                              const sync_plan::SyncPlan &plan,
                              const std::vector<std::string> &camera_names) {
    st.cam_names.clear();
    for (const auto &name : camera_names)
        if (plan.cam(name)) st.cam_names.push_back(name);

    const int64_t total = std::max<int64_t>(1, plan.canonical_len);
    const int B = (int)std::min<int64_t>(total, 4000);
    const int n_cams = (int)st.cam_names.size();
    st.num_buckets = B;
    st.total_gaps = 0;
    st.total_lost = 0;

    st.bucket_x.assign(B, 0.f);
    for (int b = 0; b < B; ++b)
        st.bucket_x[b] = (float)(((int64_t)b * total) / B);
    st.per_cam_lost.assign((size_t)n_cams * B, 0.f);
    st.agg_lost.assign(B, 0.f);

    const double slots_per_bucket = (double)total / B;
    for (int i = 0; i < n_cams; ++i) {
        const sync_plan::SyncCam &cam = *plan.cam(st.cam_names[i]);
        st.total_gaps += (int64_t)cam.gaps.size();
        for (const auto &g : cam.gaps) {
            st.total_lost += g.lost;
            int b0 = (int)std::clamp<int64_t>(g.slot * B / total, 0, B - 1);
            int b1 = (int)std::clamp<int64_t>((g.slot + g.lost - 1) * B / total,
                                              0, B - 1);
            for (int b = b0; b <= b1; ++b) {
                double lo = std::max((double)g.slot, b * slots_per_bucket);
                double hi = std::min((double)(g.slot + g.lost),
                                     (b + 1) * slots_per_bucket);
                float part = (float)std::max(0.0, hi - lo);
                st.per_cam_lost[(size_t)i * B + b] += part;
                st.agg_lost[b] += part;
            }
        }
    }
}

// Shared click-to-seek handler, run inside each plot (Pose Stats scheme:
// release with <3 px drag = click; a second quick click cancels the pending
// seek and resets the view instead).
inline void frame_drops_plot_input(FrameDropsState &st, int64_t canonical_len) {
    if (!(ImPlot::IsPlotHovered() &&
          ImGui::IsMouseReleased(ImGuiMouseButton_Left)))
        return;
    ImVec2 d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
    if (d.x * d.x + d.y * d.y >= 9.0f) return;  // >= 3 px → pan, not click
    if (st.pending_seek) {
        st.refit = true;                        // 2nd quick click → reset view
        st.pending_seek = false;
    } else {
        double mx = ImPlot::GetPlotMousePos().x;
        if (mx >= 0 && mx < (double)canonical_len) {
            st.pending_seek = true;
            st.pending_seek_slot = (int64_t)(mx + 0.5);
            st.pending_seek_time = ImGui::GetTime();
        }
    }
}

inline void DrawFrameDropsWindow(FrameDropsState &st, AppContext &ctx) {
    DrawPanel("Frame Drops", st.show, [&]() {
        const sync_plan::SyncPlan &plan = g_sync_fix.plan;
        if (!plan.usable() || ctx.input_is_imgs || ctx.pm.camera_names.empty()) {
            ImGui::TextDisabled("No sync metadata found for this recording.");
            ImGui::TextWrapped(
                "Frame drops are read from Cam<serial>_meta.csv timestamp "
                "sidecars (or a cluster sync_plan.json) next to the videos.");
            st.cached_key.clear();
            return;
        }
        if (plan.status == sync_plan::Status::Clean) {
            ImGui::TextDisabled("No dropped frames — cameras aligned.");
            return;
        }

        std::string key = plan.source + ":" +
                          std::to_string(plan.canonical_len) + ":" +
                          std::to_string(plan.cams.size());
        if (st.cached_key != key) {
            frame_drops_build(st, plan, ctx.pm.camera_names);
            st.cached_key = key;
        }
        const int n_cams = (int)st.cam_names.size();
        if (st.num_buckets <= 0 || n_cams == 0) {
            ImGui::TextDisabled("Sync plan has no cameras.");
            return;
        }

        const sync_plan::SyncCam *cam0 = plan.cam(ctx.pm.camera_names[0]);
        const bool fix_on = ctx.dc_context->sync_fix_active.load();
        const int64_t cursor_slot = frame_drops_cursor_slot(
            plan, cam0, fix_on, ctx.current_frame_num);

        // Compact header row.
        ImGui::TextDisabled("%d cameras · %lld gaps · %lld frames lost",
                            n_cams, (long long)st.total_gaps,
                            (long long)st.total_lost);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Click a plot to seek · drag to pan · scroll to "
                              "zoom · double-click to reset view\nRed marks = "
                              "dropped frames; gray = camera not recording");
        if (!fix_on) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.4f, 1.0f),
                               "timeline shown in canonical trigger slots");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "The Sync fix is OFF, so playback runs on mp4 frame "
                    "indices.\nCursor and seeks are translated through the "
                    "reference camera (%s).",
                    ctx.pm.camera_names[0].c_str());
        }

        // Native double-click "fit" snaps to data extent; move it to Middle so
        // our left-double-click resets to the full [0, canonical_len] range
        // (same rationale as Pose Stats). Restored after the plots.
        ImPlotInputMap &imap = ImPlot::GetInputMap();
        ImGuiMouseButton saved_fit = imap.Fit;
        imap.Fit = ImGuiMouseButton_Middle;

        ImPlotCond xcond = st.refit ? ImPlotCond_Always : ImPlotCond_Once;
        st.refit = false;

        const double T = (double)plan.canonical_len;
        if (ImPlot::BeginSubplots("##fd_sub", 2, 1, ImVec2(-1, -1),
                                  ImPlotSubplotFlags_LinkCols |
                                      ImPlotSubplotFlags_NoTitle,
                                  st.row_ratios)) {
            // ---------------- Top: per-camera lanes ----------------
            if (ImPlot::BeginPlot("##fd_lanes", ImVec2(),
                                  ImPlotFlags_NoLegend |
                                      ImPlotFlags_NoMouseText)) {
                ImPlot::SetupAxis(ImAxis_X1, nullptr,
                                  ImPlotAxisFlags_NoTickLabels);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, T, xcond);
                ImPlot::SetupAxisZoomConstraints(ImAxis_X1, 50, T);
                ImPlot::SetupAxis(ImAxis_Y1, nullptr,
                                  ImPlotAxisFlags_Lock |
                                      ImPlotAxisFlags_Invert |
                                      ImPlotAxisFlags_NoGridLines);
                ImPlot::SetupAxisLimits(ImAxis_Y1, -0.5, n_cams - 0.5,
                                        ImPlotCond_Always);
                std::vector<double> tick_vals(n_cams);
                std::vector<const char *> tick_lbls(n_cams);
                for (int i = 0; i < n_cams; ++i) {
                    tick_vals[i] = (double)i;
                    tick_lbls[i] = st.cam_names[i].c_str();
                }
                ImPlot::SetupAxisTicks(ImAxis_Y1, tick_vals.data(), n_cams,
                                       tick_lbls.data());
                ImPlot::SetupFinish();

                ImPlotRect lim = ImPlot::GetPlotLimits();
                ImDrawList *dl = ImPlot::GetPlotDrawList();
                ImPlot::PushPlotClipRect();
                const ImU32 gap_col = IM_COL32(220, 60, 60, 220);
                const ImU32 gray_col = IM_COL32(128, 128, 128, 40);
                for (int i = 0; i < n_cams; ++i) {
                    const sync_plan::SyncCam &cam = *plan.cam(st.cam_names[i]);
                    const double y0 = i - 0.38, y1 = i + 0.38;
                    // Faint row background in the camera's series color, so
                    // lanes visually pair with the curves below.
                    ImVec4 cc = ImPlot::GetColormapColor(i);
                    dl->AddRectFilled(
                        ImPlot::PlotToPixels(0.0, y0),
                        ImPlot::PlotToPixels(T, y1),
                        IM_COL32((int)(cc.x * 255), (int)(cc.y * 255),
                                 (int)(cc.z * 255), 14));
                    // Gray where the camera isn't recording.
                    if (cam.start_slot > 0)
                        dl->AddRectFilled(
                            ImPlot::PlotToPixels(0.0, y0),
                            ImPlot::PlotToPixels((double)cam.start_slot, y1),
                            gray_col);
                    if (cam.end_slot < plan.canonical_len)
                        dl->AddRectFilled(
                            ImPlot::PlotToPixels((double)cam.end_slot, y0),
                            ImPlot::PlotToPixels(T, y1), gray_col);
                    // Red gap marks (≥2 px so they never vanish zoomed out).
                    auto first = std::lower_bound(
                        cam.gaps.begin(), cam.gaps.end(), lim.X.Min,
                        [](const sync_plan::Gap &g, double x) {
                            return (double)(g.slot + g.lost) < x;
                        });
                    for (auto it = first; it != cam.gaps.end() &&
                                          (double)it->slot <= lim.X.Max;
                         ++it) {
                        ImVec2 p0 = ImPlot::PlotToPixels((double)it->slot, y0);
                        ImVec2 p1 = ImPlot::PlotToPixels(
                            (double)(it->slot + it->lost), y1);
                        if (p1.x - p0.x < 2.0f) p1.x = p0.x + 2.0f;
                        dl->AddRectFilled(p0, p1, gap_col);
                    }
                }
                ImPlot::PopPlotClipRect();

                // Current-frame cursor.
                double cf = (double)cursor_slot;
                ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.85f, 0.2f, 0.9f), 1.5f);
                ImPlot::PlotInfLines("##current", &cf, 1);

                // Hover tooltip: which camera lane, which gap (with slop for
                // the 2 px minimum mark width).
                if (ImPlot::IsPlotHovered()) {
                    ImPlotPoint mp = ImPlot::GetPlotMousePos();
                    int lane = (int)std::lround(mp.y);
                    if (lane >= 0 && lane < n_cams) {
                        const sync_plan::SyncCam &cam =
                            *plan.cam(st.cam_names[lane]);
                        double tol = 3.0 * (lim.X.Max - lim.X.Min) /
                                     std::max(1.0f, ImPlot::GetPlotSize().x);
                        const sync_plan::Gap *hit = nullptr;
                        auto it = std::upper_bound(
                            cam.gaps.begin(), cam.gaps.end(), mp.x + tol,
                            [](double x, const sync_plan::Gap &g) {
                                return x < (double)g.slot;
                            });
                        if (it != cam.gaps.begin()) {
                            --it;
                            if (mp.x >= (double)it->slot - tol &&
                                mp.x < (double)(it->slot + it->lost) + tol)
                                hit = &*it;
                        }
                        if (hit) {
                            ImGui::BeginTooltip();
                            ImGui::Text("%s", st.cam_names[lane].c_str());
                            ImGui::Text("slots %lld..%lld — %lld frames lost",
                                        (long long)hit->slot,
                                        (long long)(hit->slot + hit->lost - 1),
                                        (long long)hit->lost);
                            ImGui::TextDisabled(
                                "mp4 resumes at index %lld",
                                (long long)hit->pos_before);
                            ImGui::EndTooltip();
                        } else if (mp.x < (double)cam.start_slot) {
                            ImGui::SetTooltip("%s: not started yet",
                                              st.cam_names[lane].c_str());
                        } else if (mp.x >= (double)cam.end_slot) {
                            ImGui::SetTooltip("%s: ended",
                                              st.cam_names[lane].c_str());
                        }
                    }
                }

                frame_drops_plot_input(st, plan.canonical_len);
                ImPlot::EndPlot();
            }

            // ---------------- Bottom: frames-lost curves ----------------
            if (ImPlot::BeginPlot("##fd_rate")) {
                ImPlot::SetupAxis(ImAxis_X1, "Trigger slot");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, T, xcond);
                ImPlot::SetupAxis(ImAxis_Y1, "Frames lost",
                                  ImPlotAxisFlags_AutoFit);
                ImPlot::SetupLegend(ImPlotLocation_NorthEast);

                ImPlot::SetNextLineStyle(ImVec4(1, 1, 1, 1), 2.0f);
                ImPlot::PlotLine("All", st.bucket_x.data(),
                                 st.agg_lost.data(), st.num_buckets);
                for (int i = 0; i < n_cams; ++i) {
                    ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(i), 1.0f);
                    ImPlot::PlotLine(
                        st.cam_names[i].c_str(), st.bucket_x.data(),
                        st.per_cam_lost.data() + (size_t)i * st.num_buckets,
                        st.num_buckets);
                }

                double cf = (double)cursor_slot;
                ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.85f, 0.2f, 0.9f), 1.5f);
                ImPlot::PlotInfLines("##current", &cf, 1);

                frame_drops_plot_input(st, plan.canonical_len);
                ImPlot::EndPlot();
            }
            ImPlot::EndSubplots();
        }
        imap.Fit = saved_fit;

        // Commit a held single-click seek once the double-click window passes,
        // mapping the canonical slot into playback coordinates.
        constexpr double kSeekHoldSec = 0.20;
        if (st.pending_seek &&
            ImGui::GetTime() - st.pending_seek_time > kSeekHoldSec) {
            st.seek_frame =
                (int)frame_drops_seek_frame(cam0, fix_on, st.pending_seek_slot);
            st.seek_requested = true;
            st.pending_seek = false;
        }
    }, nullptr, ImVec2(720, 460));
}
