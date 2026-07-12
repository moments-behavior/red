#pragma once
// bouts_window.h — segment the active prediction store into "bouts": contiguous
// frame ranges where the subject is well-predicted across the whole skeleton,
// for downstream kinematic analysis.
//
// Per-frame quality = mean keypoint confidence over ALL keypoints, counting
// missing keypoints as 0 (so partially-tracked frames score low). Segmentation
// is a Schmitt trigger (enter a bout when quality >= t_high, stay while
// >= t_low), bridging gaps up to max_gap frames (missing/low frames), then
// dropping bouts shorter than min_duration. Emits inclusive [start, end] ranges.

#include "imgui.h"
#include "skeleton.h"
#include "prediction_store.h"
#include "gui/panel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct Bout {
    int start = 0;        // inclusive first frame
    int end = 0;          // inclusive last frame
    int n_frames = 0;     // stored (present) good frames in the bout
    float mean_conf = 0;  // mean per-frame quality over the bout
};

struct BoutParams {
    float t_high = 0.50f;    // enter-bout quality threshold
    float t_low = 0.35f;     // stay-in-bout quality threshold (< t_high)
    int max_gap = 5;         // bridge gaps up to this many frames
    int min_duration = 10;   // drop bouts shorter than this many frames
};

struct BoutState {
    bool show = false;
    BoutParams params;
    std::vector<Bout> bouts;
    std::string cached_store_path;   // recompute when the active store changes
    bool dirty = true;
    long total_good_frames = 0;

    // Requests consumed by the main loop.
    bool seek_requested = false;
    int  seek_frame = 0;
    bool export_requested = false;
    std::string export_status;
};

// Mean confidence over all keypoints (missing = 0), i.e. per-frame quality.
inline float bout_frame_quality(const float *row, int nn) {
    double s = 0;
    for (int k = 0; k < nn; ++k) {
        float c = row[k * 4 + 3];
        if (!std::isnan(c)) s += c;
    }
    return nn > 0 ? (float)(s / nn) : 0.f;
}

inline std::vector<Bout> bout_segment(const predstore::PredictionReader &store,
                                      const SkeletonContext &skel,
                                      const BoutParams &p) {
    std::vector<Bout> bouts;
    // Read at most the store's own keypoint count so a store with fewer
    // keypoints than the current skeleton never over-reads a frame block.
    const int nn = std::min(skel.num_nodes, (int)store.num_keypoints());
    if (nn <= 0 || !store.is_open()) return bouts;

    bool in_bout = false;
    int bout_start = 0, last_good = 0;
    double conf_sum = 0;
    int conf_cnt = 0;

    auto finalize = [&]() {
        if (!in_bout) return;
        if (last_good - bout_start + 1 >= p.min_duration) {
            Bout b;
            b.start = bout_start;
            b.end = last_good;
            b.n_frames = conf_cnt;
            b.mean_conf = conf_cnt > 0 ? (float)(conf_sum / conf_cnt) : 0.f;
            bouts.push_back(b);
        }
        in_bout = false;
    };

    const uint32_t n = store.stored_frames();
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t fn = 0;
        const float *row = store.stored_at(i, &fn);
        if (!row) continue;
        float q = bout_frame_quality(row, nn);

        if (in_bout) {
            bool gap_break = (int)fn - last_good > p.max_gap + 1;
            if (gap_break || q < p.t_low) {
                finalize();  // close at last_good
            } else {
                last_good = (int)fn;
                conf_sum += q;
                conf_cnt++;
            }
        }
        if (!in_bout && q >= p.t_high) {
            in_bout = true;
            bout_start = (int)fn;
            last_good = (int)fn;
            conf_sum = q;
            conf_cnt = 1;
        }
    }
    finalize();
    return bouts;
}

inline void bouts_recompute(BoutState &st,
                            const predstore::PredictionReader &store,
                            const SkeletonContext &skel) {
    st.bouts = bout_segment(store, skel, st.params);
    st.total_good_frames = 0;
    for (const auto &b : st.bouts)
        st.total_good_frames += (b.end - b.start + 1);
    st.dirty = false;
}

inline void DrawBoutsWindow(BoutState &st,
                            const predstore::PredictionReader &store,
                            const std::string &active_store_path,
                            SkeletonContext &skel) {
    DrawPanel("Bouts", st.show, [&]() {
        if (!store.is_open() || active_store_path.empty()) {
            ImGui::TextDisabled("No prediction store loaded.");
            ImGui::TextWrapped(
                "Run Batch Predict (Send to: Predictions store) or load a saved "
                "store, then bouts of good tracking will be segmented here.");
            st.cached_store_path.clear();
            st.bouts.clear();
            return;
        }
        if (st.cached_store_path != active_store_path) {
            st.cached_store_path = active_store_path;
            st.dirty = true;
        }

        ImGui::TextWrapped(
            "Contiguous ranges where the whole skeleton is well predicted.");

        bool changed = false;
        ImGui::SetNextItemWidth(160);
        changed |= ImGui::SliderFloat("Enter threshold", &st.params.t_high,
                                      0.0f, 1.0f, "%.2f");
        ImGui::SetNextItemWidth(160);
        changed |= ImGui::SliderFloat("Stay threshold", &st.params.t_low,
                                      0.0f, 1.0f, "%.2f");
        if (st.params.t_low > st.params.t_high) st.params.t_low = st.params.t_high;
        ImGui::SetNextItemWidth(160);
        changed |= ImGui::SliderInt("Max gap (frames)", &st.params.max_gap, 0, 120);
        ImGui::SetNextItemWidth(160);
        changed |= ImGui::SliderInt("Min duration (frames)",
                                    &st.params.min_duration, 1, 600);
        if (changed) st.dirty = true;
        if (st.dirty) bouts_recompute(st, store, skel);

        const uint32_t fps = store.fps() > 0 ? store.fps() : 1;
        ImGui::Separator();
        ImGui::Text("%zu bouts · %ld good frames (%.1f s)", st.bouts.size(),
                    st.total_good_frames, (double)st.total_good_frames / fps);
        ImGui::SameLine();
        if (ImGui::SmallButton("Export CSV")) st.export_requested = true;
        if (!st.export_status.empty()) {
            bool err = st.export_status.find("Failed") != std::string::npos ||
                       st.export_status.find("Cannot") != std::string::npos;
            ImGui::TextColored(err ? ImVec4(1, 0.4f, 0.4f, 1)
                                   : ImVec4(0.5f, 1, 0.5f, 1),
                               "%s", st.export_status.c_str());
        }

        ImGuiTableFlags tf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                             ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
        if (ImGui::BeginTable("##bouts_tbl", 5, tf, ImVec2(0, 0))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("#");
            ImGui::TableSetupColumn("Start");
            ImGui::TableSetupColumn("End");
            ImGui::TableSetupColumn("Frames");
            ImGui::TableSetupColumn("Conf");
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < st.bouts.size(); ++i) {
                const Bout &b = st.bouts[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                char lbl[32];
                snprintf(lbl, sizeof(lbl), "%zu##bout%zu", i + 1, i);
                if (ImGui::Selectable(lbl, false,
                                      ImGuiSelectableFlags_SpanAllColumns)) {
                    st.seek_requested = true;
                    st.seek_frame = b.start;
                }
                ImGui::TableSetColumnIndex(1); ImGui::Text("%d", b.start);
                ImGui::TableSetColumnIndex(2); ImGui::Text("%d", b.end);
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%d", b.end - b.start + 1);
                ImGui::TableSetColumnIndex(4); ImGui::Text("%.2f", b.mean_conf);
            }
            ImGui::EndTable();
        }
    }, nullptr, ImVec2(440, 500));
}

// Write bouts to CSV next to the store. Columns match a form convenient for
// downstream kinematics. Returns a status string.
inline std::string bouts_export_csv(const BoutState &st,
                                    const std::string &store_path,
                                    uint32_t fps) {
    namespace fs = std::filesystem;
    if (store_path.empty()) return "Cannot export: no store path";
    fs::path out = store_path;
    out.replace_extension();
    out += "_bouts.csv";
    std::ofstream f(out);
    if (!f) return "Failed to open " + out.string();
    uint32_t safe_fps = fps > 0 ? fps : 1;
    f << "bout,start_frame,end_frame,n_frames,duration_s,mean_confidence\r\n";
    for (size_t i = 0; i < st.bouts.size(); ++i) {
        const Bout &b = st.bouts[i];
        int span = b.end - b.start + 1;
        f << (i + 1) << "," << b.start << "," << b.end << "," << span << ","
          << (double)span / safe_fps << "," << b.mean_conf << "\r\n";
    }
    return "Exported " + std::to_string(st.bouts.size()) + " bouts → " +
           out.filename().string();
}
