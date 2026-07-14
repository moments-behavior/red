#pragma once
// bout_filter_window.h — View -> Bout Filter panel.
//
// Runs a native (C++) walking-bout filter pipeline over the active .rpred
// prediction store: hard per-keypoint confidence floor, upright posture,
// floor-Z, Y/X arena-wall ("wall touching") filters, immobility splitting,
// contiguous bout detection, and walking-cycle validation. Every candidate
// bout is shown with an accept/reject status and, when rejected, the reason.
// Filter parameters are all editable here; the species/rig mapping (keypoint
// names, arena size, scale, fps) comes from a profile (prfs/<name>/profile.json,
// with a built-in fly fallback).
//
// The heavy numeric work lives in ../bout_filter_core.h (unit-tested against
// the original scipy pipeline). This file is UI + store-reading only.

#include "imgui.h"
#include "app_context.h"
#include "skeleton.h"
#include "prediction_store.h"
#include "gui/panel.h"
#include "../bout_filter_core.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

// One selectable profile: species/rig mapping + default filter values.
struct BoutFilterProfile {
    std::string name;
    std::string source;             // "built-in" or a file path
    float scale = 10.0f;
    int   fps = 800;
    float arena_x_mm = 23.5f;
    float arena_y_mm = 5.5f;
    std::string body_ref;
    std::vector<std::string> leg_tips;
    std::vector<std::string> x_wall_tips;
    boutfilter::Params defaults;
};

struct BoutFilterState {
    bool show = false;
    boutfilter::Params params;

    std::vector<BoutFilterProfile> profiles;
    int  profile_idx = -1;
    bool profiles_scanned = false;

    // Cached dense inputs (rebuilt only when the store or profile changes).
    boutfilter::Inputs inputs;
    bool inputs_valid = false;
    std::string build_error;                 // non-empty => cannot compute
    std::string cached_store_path;
    std::string cached_profile;              // profile name the inputs were built for

    boutfilter::Result result;
    bool dirty = true;                       // recompute compute() from inputs

    // Requests consumed by the main loop.
    bool seek_requested = false;
    int  seek_frame = 0;
    bool export_requested = false;
    std::string export_status;
};

// ── profile loading ──────────────────────────────────────────────────────────

inline BoutFilterProfile bout_filter_builtin_profile() {
    BoutFilterProfile p;
    p.name = "fly (built-in)";
    p.source = "built-in";
    p.body_ref = "Scutellum";
    p.leg_tips = {"T1L_TaTip", "T1R_TaTip", "T2L_TaTip",
                  "T2R_TaTip", "T3L_TaTip", "T3R_TaTip"};
    p.x_wall_tips = {"T1L_TaTip", "T1R_TaTip", "T3L_TaTip", "T3R_TaTip"};
    return p;  // Params + arena/scale/fps keep their struct defaults (fly)
}

inline bool bout_filter_parse_profile(const std::string &path, BoutFilterProfile &out) {
    try {
        std::ifstream f(path);
        if (!f) return false;
        nlohmann::json j; f >> j;
        out.source = path;
        out.name = j.value("name", std::filesystem::path(path).parent_path()
                                        .filename().string());
        out.scale = j.value("scale", 10.0f);
        out.fps = j.value("fps", 800);
        out.arena_x_mm = j.value("arena_x_mm", 23.5f);
        out.arena_y_mm = j.value("arena_y_mm", 5.5f);
        out.body_ref = j.value("body_ref_keypoint", std::string("Scutellum"));
        out.leg_tips = j.value("leg_tip_keypoints", std::vector<std::string>{});
        out.x_wall_tips = j.value("x_wall_keypoints", std::vector<std::string>{});
        boutfilter::Params &d = out.defaults;
        if (j.contains("defaults")) {
            const auto &dd = j["defaults"];
            d.confidence_enabled = dd.value("confidence_enabled", d.confidence_enabled);
            d.confidence_threshold = dd.value("confidence_threshold", d.confidence_threshold);
            d.confidence_gap_bridge = dd.value("confidence_gap_bridge", d.confidence_gap_bridge);
            d.upright_enabled = dd.value("upright_enabled", d.upright_enabled);
            d.floor_z_enabled = dd.value("floor_z_enabled", d.floor_z_enabled);
            d.floor_z_threshold = dd.value("floor_z_threshold", d.floor_z_threshold);
            d.y_wall_min = dd.value("y_wall_min", d.y_wall_min);
            d.y_wall_max = dd.value("y_wall_max", d.y_wall_max);
            d.x_wall_margin = dd.value("x_wall_margin", d.x_wall_margin);
            d.immobility_max_frames = dd.value("immobility_max_frames", d.immobility_max_frames);
            d.immobility_speed_threshold = dd.value("immobility_speed_threshold", d.immobility_speed_threshold);
            d.min_bout_frames = dd.value("min_bout_frames", d.min_bout_frames);
            d.max_gap_bridge = dd.value("max_gap_bridge", d.max_gap_bridge);
            d.min_walking_cycles = dd.value("min_walking_cycles", d.min_walking_cycles);
            d.min_distance_mm = dd.value("min_distance_mm", d.min_distance_mm);
            d.max_swing_duration = dd.value("max_swing_duration", d.max_swing_duration);
            d.swing_prominence = dd.value("swing_prominence", d.swing_prominence);
        }
        return true;
    } catch (...) {
        return false;
    }
}

inline void bout_filter_scan_profiles(BoutFilterState &st, const std::string &exe_dir) {
    namespace fs = std::filesystem;
    st.profiles.clear();
    // Search a few layouts so the profile is found in dev (single- and
    // multi-config generators, e.g. Windows build/Release/) and installed
    // (../share/red/prfs) trees on all platforms. A built-in fly profile is the
    // fallback if none are found, so the feature works with no files installed.
    const std::string roots[] = {
        exe_dir + "/prfs",
        exe_dir + "/../prfs",
        exe_dir + "/../../prfs",
        exe_dir + "/../share/red/prfs",
        exe_dir + "/../../share/red/prfs",
    };
    std::error_code ec;
    for (const std::string &root : roots) {
        if (!fs::is_directory(root, ec)) continue;
        for (const auto &e : fs::directory_iterator(root, ec)) {
            if (!e.is_directory()) continue;
            fs::path pj = e.path() / "profile.json";
            if (!fs::exists(pj)) continue;
            BoutFilterProfile p;
            if (!bout_filter_parse_profile(pj.string(), p)) continue;
            // Skip duplicates when several roots resolve to the same profile.
            bool dup = false;
            for (const auto &q : st.profiles) if (q.name == p.name) { dup = true; break; }
            if (!dup) st.profiles.push_back(p);
        }
    }
    if (st.profiles.empty()) st.profiles.push_back(bout_filter_builtin_profile());
    st.profile_idx = 0;
    st.params = st.profiles[0].defaults;
    st.profiles_scanned = true;
}

// ── build dense inputs from the store for a given profile ────────────────────

inline void bout_filter_build_inputs(BoutFilterState &st,
                                      const predstore::PredictionReader &store,
                                      const SkeletonContext &skel) {
    using namespace boutfilter;
    st.inputs = Inputs{};
    st.inputs_valid = false;
    st.build_error.clear();
    if (st.profile_idx < 0 || st.profile_idx >= (int)st.profiles.size()) {
        st.build_error = "No profile selected."; return;
    }
    const BoutFilterProfile &prof = st.profiles[st.profile_idx];
    // Mark which profile these (attempted) inputs are for, so an unresolved
    // error state doesn't retrigger a full store rescan every frame.
    st.cached_profile = prof.name;

    const int num_kp = (int)store.num_keypoints();
    if (num_kp != skel.num_nodes) {
        st.build_error = "Store has " + std::to_string(num_kp) +
            " keypoints but the loaded skeleton has " +
            std::to_string(skel.num_nodes) + ". Load a matching skeleton.";
        return;
    }
    auto find_kp = [&](const std::string &name) -> int {
        for (int i = 0; i < (int)skel.node_names.size(); ++i)
            if (skel.node_names[i] == name) return i;
        return -1;
    };
    int body_idx = find_kp(prof.body_ref);
    if (body_idx < 0) { st.build_error = "Body-ref keypoint '" + prof.body_ref +
        "' not in skeleton."; return; }
    std::vector<int> leg_idx;
    for (const auto &nm : prof.leg_tips) {
        int k = find_kp(nm);
        if (k < 0) { st.build_error = "Leg keypoint '" + nm + "' not in skeleton."; return; }
        leg_idx.push_back(k);
    }
    if (leg_idx.empty()) { st.build_error = "Profile has no leg keypoints."; return; }

    const int total = (int)store.total_frames();
    const float scale = prof.scale != 0 ? prof.scale : 1.0f;

    Inputs &in = st.inputs;
    in.total_frames = total;
    in.fps = store.fps() > 0 ? (int)store.fps() : prof.fps;
    in.arena_x_mm = prof.arena_x_mm;
    in.min_conf.assign(total, NAN);
    in.body_ref.x.assign(total, NAN);
    in.body_ref.y.assign(total, NAN);
    in.body_ref.z.assign(total, NAN);
    in.leg_tips.resize(leg_idx.size());
    in.leg_is_x_wall.assign(leg_idx.size(), 0);
    for (size_t l = 0; l < leg_idx.size(); ++l) {
        in.leg_tips[l].x.assign(total, NAN);
        in.leg_tips[l].y.assign(total, NAN);
        in.leg_tips[l].z.assign(total, NAN);
        // is this leg an x-wall tip?
        const std::string &nm = prof.leg_tips[l];
        in.leg_is_x_wall[l] = std::find(prof.x_wall_tips.begin(),
            prof.x_wall_tips.end(), nm) != prof.x_wall_tips.end() ? 1 : 0;
    }

    // One pass over stored frames: scatter needed keypoints + per-frame min conf.
    const uint32_t ns = store.stored_frames();
    for (uint32_t i = 0; i < ns; ++i) {
        uint32_t fn = 0;
        const float *row = store.stored_at(i, &fn);
        if (!row || (int)fn >= total) continue;
        // min confidence over ALL keypoints (any NaN => frame fails the floor)
        float m = std::numeric_limits<float>::infinity();
        bool anynan = false;
        for (int k = 0; k < num_kp; ++k) {
            float c = row[k * 4 + 3];
            if (std::isnan(c)) { anynan = true; break; }
            m = std::min(m, c);
        }
        in.min_conf[fn] = anynan ? NAN : m;
        auto put = [&](int kp, KpTrack &t) {
            t.x[fn] = row[kp * 4 + 0] / scale;
            t.y[fn] = row[kp * 4 + 1] / scale;
            t.z[fn] = row[kp * 4 + 2] / scale;
        };
        put(body_idx, in.body_ref);
        for (size_t l = 0; l < leg_idx.size(); ++l) put(leg_idx[l], in.leg_tips[l]);
    }

    st.inputs_valid = true;
    st.cached_store_path.clear();  // caller sets after this
    st.cached_profile = prof.name;
}

// ── CSV export ────────────────────────────────────────────────────────────────

inline std::string bout_filter_export_csv(const BoutFilterState &st,
                                           const std::string &store_path) {
    namespace fs = std::filesystem;
    if (store_path.empty()) return "Cannot export: no store path";
    fs::path out = store_path; out.replace_extension(); out += "_bout_filter.csv";
    std::ofstream f(out);
    if (!f) return "Failed to open " + out.string();
    f << "bout,start_frame,end_frame,n_frames,duration_s,min_cycles,status,reason,"
         "total_distance_mm,mean_speed_mm_s\r\n";
    int i = 0;
    for (const auto &b : st.result.bouts) {
        f << (++i) << "," << b.start << "," << b.end << "," << b.n_frames << ","
          << b.duration_s << "," << b.min_cycles << ","
          << (b.accepted ? "accepted" : "rejected") << ",\"" << b.reason << "\","
          << b.total_distance_mm << "," << b.mean_speed_mm_s << "\r\n";
    }
    return "Exported " + std::to_string(st.result.bouts.size()) + " bouts \xE2\x86\x92 " +
           out.filename().string();
}

// ── the panel ─────────────────────────────────────────────────────────────────

inline void DrawBoutFilterWindow(BoutFilterState &st,
                                 const predstore::PredictionReader &store,
                                 const std::string &active_store_path,
                                 SkeletonContext &skel,
                                 AppContext &ctx) {
    if (!st.profiles_scanned)
        bout_filter_scan_profiles(st, ctx.window ? ctx.window->exe_dir : "");

    DrawPanel("Bout Filter", st.show, [&]() {
        if (!store.is_open() || active_store_path.empty()) {
            ImGui::TextDisabled("No prediction store loaded.");
            ImGui::TextWrapped(
                "Run Batch Predict (Send to: Predictions store) or load a saved "
                "store, then walking bouts will be detected and filtered here.");
            st.cached_store_path.clear();
            st.inputs_valid = false;
            return;
        }

        // Rebuild dense inputs when the store or profile changes.
        if (st.cached_store_path != active_store_path ||
            (st.profile_idx >= 0 && st.profile_idx < (int)st.profiles.size() &&
             st.cached_profile != st.profiles[st.profile_idx].name)) {
            bout_filter_build_inputs(st, store, skel);
            st.cached_store_path = active_store_path;
            st.dirty = true;
        }

        // --- Profile selector ---
        ImGui::SetNextItemWidth(220);
        if (ImGui::BeginCombo("Profile",
                st.profile_idx >= 0 ? st.profiles[st.profile_idx].name.c_str() : "(none)")) {
            for (int i = 0; i < (int)st.profiles.size(); ++i) {
                bool sel = (i == st.profile_idx);
                if (ImGui::Selectable(st.profiles[i].name.c_str(), sel)) {
                    st.profile_idx = i;
                    st.params = st.profiles[i].defaults;
                    st.cached_profile.clear();  // force inputs rebuild
                    st.dirty = true;
                }
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset to profile defaults") && st.profile_idx >= 0) {
            st.params = st.profiles[st.profile_idx].defaults;
            st.dirty = true;
        }

        if (!st.build_error.empty()) {
            ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "%s", st.build_error.c_str());
            return;
        }
        if (!st.inputs_valid) { ImGui::TextDisabled("Preparing..."); return; }

        boutfilter::Params &p = st.params;
        bool ch = false;

        // --- Hard confidence floor (prominent) ---
        ImGui::SeparatorText("Confidence floor (hard)");
        ch |= ImGui::Checkbox("Enable##conf", &p.confidence_enabled);
        ImGui::TextDisabled("A frame is rejected unless EVERY keypoint's confidence "
                            "is at or above this floor.");
        ImGui::BeginDisabled(!p.confidence_enabled);
        ImGui::SetNextItemWidth(200);
        ch |= ImGui::SliderFloat("Min confidence", &p.confidence_threshold, 0.0f, 1.0f, "%.2f");
        ImGui::SetNextItemWidth(200);
        ch |= ImGui::SliderInt("Gap bridge (frames)##conf", &p.confidence_gap_bridge, 0, 120);
        ImGui::EndDisabled();

        if (ImGui::CollapsingHeader("Posture & floor")) {
            ch |= ImGui::Checkbox("Upright (body above all legs)", &p.upright_enabled);
            ch |= ImGui::Checkbox("Floor-Z reject", &p.floor_z_enabled);
            ImGui::BeginDisabled(!p.floor_z_enabled);
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderFloat("Floor Z (mm)", &p.floor_z_threshold, 0.0f, 5.0f, "%.2f");
            ImGui::EndDisabled();
        }
        if (ImGui::CollapsingHeader("Arena walls (wall touching)")) {
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderFloat("Y wall min (mm)", &p.y_wall_min, -1.0f, 10.0f, "%.2f");
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderFloat("Y wall max (mm)", &p.y_wall_max, -1.0f, 10.0f, "%.2f");
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderFloat("X wall margin (mm)", &p.x_wall_margin, 0.0f, 5.0f, "%.2f");
        }
        if (ImGui::CollapsingHeader("Immobility split")) {
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderInt("Max stationary (frames)", &p.immobility_max_frames, 0, 400);
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderFloat("Speed threshold (mm/s)", &p.immobility_speed_threshold, 0.0f, 50.0f, "%.1f");
        }
        if (ImGui::CollapsingHeader("Bout detection")) {
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderInt("Min duration (frames)", &p.min_bout_frames, 1, 2000);
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderInt("Max gap bridge (frames)", &p.max_gap_bridge, 0, 400);
        }
        if (ImGui::CollapsingHeader("Walking validation")) {
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderInt("Min walking cycles", &p.min_walking_cycles, 0, 20);
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderFloat("Min distance (mm)", &p.min_distance_mm, 0.0f, 50.0f, "%.1f");
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderInt("Max swing duration (frames)", &p.max_swing_duration, 1, 200);
            ImGui::SetNextItemWidth(200);
            ch |= ImGui::SliderFloat("Swing prominence (mm)", &p.swing_prominence, 0.0f, 1.0f, "%.3f");
        }
        if (ch) st.dirty = true;
        if (st.dirty) {
            st.result = boutfilter::compute(st.inputs, st.params);
            st.dirty = false;
        }

        // --- Summary + rejection breakdown ---
        const boutfilter::Result &R = st.result;
        const int fps = st.inputs.fps > 0 ? st.inputs.fps : 1;
        long acc_frames = 0;
        for (const auto &b : R.bouts) if (b.accepted) acc_frames += b.n_frames;
        ImGui::SeparatorText("Results");
        ImGui::Text("%d accepted / %d rejected of %d candidates \xC2\xB7 "
                    "%d/%d frames pass all filters (%.1f%%)",
                    R.n_accepted, R.n_rejected, R.n_candidates,
                    R.n_valid_frames, R.n_frames, R.pct_all);
        ImGui::Text("Accepted walking: %ld frames (%.1f s)", acc_frames,
                    (double)acc_frames / fps);
        ImGui::SameLine();
        if (ImGui::SmallButton("Export CSV")) st.export_requested = true;
        if (!st.export_status.empty()) {
            bool err = st.export_status.find("Failed") != std::string::npos ||
                       st.export_status.find("Cannot") != std::string::npos;
            ImGui::TextColored(err ? ImVec4(1, 0.4f, 0.4f, 1) : ImVec4(0.5f, 1, 0.5f, 1),
                               "%s", st.export_status.c_str());
        }

        if (R.n_rejected > 0) {
            std::map<std::string, int> reasons;
            for (const auto &b : R.bouts)
                if (!b.accepted) {
                    // group by the reason prefix (drop the numeric detail)
                    std::string r = b.reason;
                    size_t paren = r.find(" (");
                    if (paren != std::string::npos) r = r.substr(0, paren);
                    reasons[r]++;
                }
            std::string line = "Rejected: ";
            bool first = true;
            for (auto &kv : reasons) {
                if (!first) line += ", ";
                line += kv.first + " \xC3\x97" + std::to_string(kv.second);
                first = false;
            }
            ImGui::TextDisabled("%s", line.c_str());
        }

        // Per-filter frame pass rates.
        ImGui::TextDisabled("Frame pass: conf %.0f%% \xC2\xB7 upright %.0f%% \xC2\xB7 "
                            "floor %.0f%% \xC2\xB7 Ywall %.0f%% \xC2\xB7 Xwall %.0f%%",
                            R.pct_conf, R.pct_upright, R.pct_floor_ok,
                            R.pct_ywall_ok, R.pct_xwall_ok);

        // --- Bout table ---
        ImGuiTableFlags tf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                             ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
        if (ImGui::BeginTable("##bout_filter_tbl", 7, tf, ImVec2(0, 0))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("#");
            ImGui::TableSetupColumn("Start");
            ImGui::TableSetupColumn("End");
            ImGui::TableSetupColumn("Frames");
            ImGui::TableSetupColumn("Cycles");
            ImGui::TableSetupColumn("Status");
            ImGui::TableSetupColumn("Reason");
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < R.bouts.size(); ++i) {
                const boutfilter::ResultBout &b = R.bouts[i];
                ImGui::TableNextRow();
                ImU32 bg = b.accepted ? IM_COL32(40, 90, 45, 90)
                                      : IM_COL32(110, 40, 40, 90);
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, bg);
                ImGui::TableSetColumnIndex(0);
                char lbl[32];
                snprintf(lbl, sizeof(lbl), "%zu##bf%zu", i + 1, i);
                if (ImGui::Selectable(lbl, false, ImGuiSelectableFlags_SpanAllColumns)) {
                    st.seek_requested = true;
                    st.seek_frame = b.start;
                }
                if (ImGui::IsItemHovered() && (b.accepted ||
                        !std::isnan(b.total_distance_mm))) {
                    ImGui::SetTooltip(
                        "frames %d-%d (%.2f s)\ncycles %d\ndistance %.1f mm\n"
                        "mean speed %.1f mm/s",
                        b.start, b.end, b.duration_s, b.min_cycles,
                        b.total_distance_mm, b.mean_speed_mm_s);
                }
                ImGui::TableSetColumnIndex(1); ImGui::Text("%d", b.start);
                ImGui::TableSetColumnIndex(2); ImGui::Text("%d", b.end);
                ImGui::TableSetColumnIndex(3); ImGui::Text("%d", b.n_frames);
                ImGui::TableSetColumnIndex(4); ImGui::Text("%d", b.min_cycles);
                ImGui::TableSetColumnIndex(5);
                ImGui::TextColored(b.accepted ? ImVec4(0.5f, 1, 0.5f, 1)
                                              : ImVec4(1, 0.5f, 0.5f, 1),
                                   "%s", b.accepted ? "accept" : "reject");
                ImGui::TableSetColumnIndex(6);
                ImGui::TextUnformatted(b.reason.c_str());
            }
            ImGui::EndTable();
        }
    }, nullptr, ImVec2(480, 640));
}
