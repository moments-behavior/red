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
#include <set>
#include <string>
#include <vector>

// Group a rejection reason by its prefix, dropping the numeric detail
// (e.g. "too few walking cycles (1)" -> "too few walking cycles") so the
// results breakdown and the table filter checkboxes bucket consistently
// regardless of the exact count/distance/etc. in any one bout's reason.
inline std::string bout_reason_prefix(const std::string &reason) {
    size_t paren = reason.find(" (");
    return paren != std::string::npos ? reason.substr(0, paren) : reason;
}

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

    // Live preview: reproject threshold planes into the camera views
    // (bout_filter_preview.h). UI-only — never touches boutfilter::compute().
    bool  show_floor_preview = false;
    bool  show_ywall_preview = false;
    bool  show_xwall_preview = false;
    float preview_height_mm = 3.0f;   // top of the Y/X-wall rectangles

    std::vector<BoutFilterProfile> profiles;
    int  profile_idx = -1;
    bool profiles_scanned = false;

    // Cached dense inputs (rebuilt only when the store or profile changes).
    boutfilter::Inputs inputs;
    bool inputs_valid = false;
    std::string build_error;                 // non-empty => cannot compute
    std::string cached_store_path;
    std::string cached_profile;              // profile name the inputs were built for

    boutfilter::Result auto_result;          // raw compute() output
    boutfilter::Result result;               // curated = auto_result + edit overlay
    bool dirty = true;                       // recompute compute() from inputs

    // Manual-edit overlay (persisted to <store>_bout_filter_edits.json).
    boutfilter::BoutEdits edits;
    bool edits_dirty = true;                 // re-run apply_bout_edits over auto_result
    bool edits_save_requested = false;       // consumed by the main loop
    std::set<uint64_t> selected_ids;         // table multi-select (row keys)
    uint64_t result_version = 0;             // bump when result changes -> clears selection
    // Boundary-adjust popup scratch.
    uint64_t edit_target_id = 0;
    int edit_start_buf = 0, edit_end_buf = 0;

    // Table filter: which rows to show. UI-only — never touches compute().
    bool filter_show_accepted = true;
    bool filter_show_rejected = true;
    std::set<std::string> filter_hidden_reasons;  // rejection-reason prefixes hidden from the table

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

// ── manual-edit sidecar (JSON next to the store) ─────────────────────────────

inline std::string bout_filter_edits_path(const std::string &store_path) {
    std::filesystem::path out = store_path;
    out.replace_extension();
    out += "_bout_filter_edits.json";
    return out.string();
}

// Load the manual-edit overlay for a store. Missing/unparsable file clears the
// overlay (so switching stores never leaks edits). Robust to partial JSON.
inline void bout_filter_load_edits(BoutFilterState &st, const std::string &store_path) {
    st.edits = boutfilter::BoutEdits{};
    if (store_path.empty()) return;
    std::string path = bout_filter_edits_path(store_path);
    try {
        std::ifstream f(path);
        if (!f) return;
        nlohmann::json j; f >> j;
        boutfilter::BoutEdits &e = st.edits;
        e.schema_version = j.value("schema_version", 1);
        uint64_t max_id = 0;
        if (j.contains("manual_bouts")) {
            for (const auto &m : j["manual_bouts"]) {
                boutfilter::ManualBout b;
                b.id = m.value("id", (uint64_t)0);
                b.start = m.value("start", 0);
                b.end = m.value("end", 0);
                b.forced = (boutfilter::ForcedStatus)m.value("forced", 0);
                b.kind = (boutfilter::EditKind)m.value("kind", (int)boutfilter::EditKind::Manual);
                e.manual_bouts.push_back(b);
                max_id = std::max(max_id, b.id);
            }
        }
        if (j.contains("overrides")) {
            for (const auto &o : j["overrides"]) {
                boutfilter::StatusOverride s;
                s.id = o.value("id", (uint64_t)0);
                s.anchor.start = o.value("anchor_start", 0);
                s.anchor.end = o.value("anchor_end", 0);
                s.status = (boutfilter::ForcedStatus)o.value("status", 1);
                e.overrides.push_back(s);
                max_id = std::max(max_id, s.id);
            }
        }
        // Persisted next_id, but never below max(existing id)+1 (safety net).
        e.next_id = std::max<uint64_t>(j.value("next_id", (uint64_t)1), max_id + 1);
    } catch (...) {
        st.edits = boutfilter::BoutEdits{};
    }
}

inline std::string bout_filter_save_edits(const BoutFilterState &st,
                                          const std::string &store_path) {
    if (store_path.empty()) return "Cannot save edits: no store path";
    std::string path = bout_filter_edits_path(store_path);
    const boutfilter::BoutEdits &e = st.edits;
    // Nothing to persist: remove any stale sidecar so a cleared overlay doesn't
    // reappear on reload.
    if (e.empty()) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return "";
    }
    try {
        nlohmann::json j;
        j["schema_version"] = e.schema_version;
        j["next_id"] = e.next_id;
        j["manual_bouts"] = nlohmann::json::array();
        for (const auto &b : e.manual_bouts) {
            j["manual_bouts"].push_back({
                {"id", b.id}, {"start", b.start}, {"end", b.end},
                {"forced", (int)b.forced}, {"kind", (int)b.kind}});
        }
        j["overrides"] = nlohmann::json::array();
        for (const auto &s : e.overrides) {
            j["overrides"].push_back({
                {"id", s.id}, {"anchor_start", s.anchor.start},
                {"anchor_end", s.anchor.end}, {"status", (int)s.status}});
        }
        std::ofstream f(path);
        if (!f) return "Failed to write " + path;
        f << j.dump(2);
    } catch (...) {
        return "Failed to serialize edits";
    }
    return "";
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
         "total_distance_mm,mean_speed_mm_s,source,edit_kind,status_source\r\n";
    int i = 0;
    for (const auto &b : st.result.bouts) {
        const char *ekind = b.edit_kind == boutfilter::EditKind::Merge ? "merged"
                          : b.edit_kind == boutfilter::EditKind::Adjust ? "adjusted"
                          : b.edit_kind == boutfilter::EditKind::Manual ? "manual" : "";
        const char *source = b.edit_kind == boutfilter::EditKind::None ? "auto" : "manual";
        f << (++i) << "," << b.start << "," << b.end << "," << b.n_frames << ","
          << b.duration_s << "," << b.min_cycles << ","
          << (b.accepted ? "accepted" : "rejected") << ",\"" << b.reason << "\","
          << b.total_distance_mm << "," << b.mean_speed_mm_s << ","
          << source << "," << ekind << ","
          << (b.status_overridden ? "manual" : "auto") << "\r\n";
    }
    return "Exported " + std::to_string(st.result.bouts.size()) + " bouts \xE2\x86\x92 " +
           out.filename().string();
}

// ── manual-edit mutation helpers (operate on st.edits) ───────────────────────

// Stable per-row selection key: manual bouts keyed by their overlay id; auto
// rows keyed by start frame (unique within a result) with the high bit set so
// the two spaces never collide.
inline uint64_t bout_filter_row_key(const boutfilter::ResultBout &b) {
    if (b.edit_kind != boutfilter::EditKind::None) return b.edit_id;
    return 0x8000000000000000ull | (uint64_t)(uint32_t)b.start;
}

// Drop every overlay entry touching [lo,hi] so a new manual bout / override for
// that range replaces the old ones cleanly (keeps the sidecar tidy).
inline void bout_filter_clear_range(boutfilter::BoutEdits &e, int lo, int hi) {
    e.manual_bouts.erase(std::remove_if(e.manual_bouts.begin(), e.manual_bouts.end(),
        [&](const boutfilter::ManualBout &m) { return m.start <= hi && m.end >= lo; }),
        e.manual_bouts.end());
    e.overrides.erase(std::remove_if(e.overrides.begin(), e.overrides.end(),
        [&](const boutfilter::StatusOverride &o) {
            return o.anchor.start <= hi && o.anchor.end >= lo; }),
        e.overrides.end());
}

inline void bout_filter_mark_edited(BoutFilterState &st) {
    st.edits_dirty = true;
    st.edits_save_requested = true;
}

// Force a status on one displayed bout. Manual bouts store it inline; auto rows
// get a range-anchored StatusOverride (or clear it when status==None).
inline void bout_filter_set_status(BoutFilterState &st, const boutfilter::ResultBout &b,
                                    boutfilter::ForcedStatus status) {
    if (b.edit_kind != boutfilter::EditKind::None) {
        for (auto &m : st.edits.manual_bouts)
            if (m.id == b.edit_id) { m.forced = status; break; }
    } else {
        // remove any existing override on this range, then add if not clearing
        st.edits.overrides.erase(std::remove_if(st.edits.overrides.begin(),
            st.edits.overrides.end(), [&](const boutfilter::StatusOverride &o) {
                return o.anchor.start <= b.end && o.anchor.end >= b.start; }),
            st.edits.overrides.end());
        if (status != boutfilter::ForcedStatus::None)
            st.edits.overrides.push_back({st.edits.next_id++, {b.start, b.end}, status});
    }
    bout_filter_mark_edited(st);
}

// Reset a displayed bout to pure auto: remove the manual bout, or clear its
// status override.
inline void bout_filter_reset_bout(BoutFilterState &st, const boutfilter::ResultBout &b) {
    if (b.edit_kind != boutfilter::EditKind::None) {
        st.edits.manual_bouts.erase(std::remove_if(st.edits.manual_bouts.begin(),
            st.edits.manual_bouts.end(), [&](const boutfilter::ManualBout &m) {
                return m.id == b.edit_id; }), st.edits.manual_bouts.end());
    } else {
        bout_filter_set_status(st, b, boutfilter::ForcedStatus::None);
        return;  // set_status already marked edited
    }
    bout_filter_mark_edited(st);
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
        bool store_changed = st.cached_store_path != active_store_path;
        if (store_changed ||
            (st.profile_idx >= 0 && st.profile_idx < (int)st.profiles.size() &&
             st.cached_profile != st.profiles[st.profile_idx].name)) {
            bout_filter_build_inputs(st, store, skel);
            st.cached_store_path = active_store_path;
            st.dirty = true;
        }
        // Load the manual-edit overlay for this store (only on a store change,
        // not on a profile tweak) so curation follows the store across sessions.
        if (store_changed) {
            bout_filter_load_edits(st, active_store_path);
            st.edits_dirty = true;
            st.selected_ids.clear();
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

        // --- Live preview of the thresholds below, reprojected onto the
        // camera video views (bout_filter_preview.h / red.cpp). Independent
        // of the sliders' own recompute (`ch`/`st.dirty`) below. ---
        ImGui::SeparatorText("Camera preview");
        ImGui::TextColored(ImVec4(1.0f, 0.66f, 0.16f, 1), "\xE2\x96\xA0");
        ImGui::SameLine(0, 4);
        ImGui::Checkbox("Floor plane##prev", &st.show_floor_preview);
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.24f, 0.55f, 1.0f, 1), "\xE2\x96\xA0");
        ImGui::SameLine(0, 4);
        ImGui::Checkbox("Y walls##prev", &st.show_ywall_preview);
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.90f, 0.35f, 0.90f, 1), "\xE2\x96\xA0");
        ImGui::SameLine(0, 4);
        ImGui::Checkbox("X walls##prev", &st.show_xwall_preview);
        ImGui::BeginDisabled(!st.show_ywall_preview && !st.show_xwall_preview);
        ImGui::SetNextItemWidth(200);
        ImGui::SliderFloat("Preview height (mm)", &st.preview_height_mm, 0.5f, 20.0f, "%.1f");
        ImGui::EndDisabled();

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
            st.auto_result = boutfilter::compute(st.inputs, st.params);
            st.dirty = false;
            st.edits_dirty = true;
        }
        if (st.edits_dirty) {
            st.result = st.auto_result;
            boutfilter::apply_bout_edits(st.result, st.edits, st.inputs, st.params);
            st.edits_dirty = false;
            st.result_version++;
            st.selected_ids.clear();
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

        // --- Manual-edit action toolbar (operates on the row selection) ---
        std::vector<const boutfilter::ResultBout *> sel;
        for (const auto &b : R.bouts)
            if (st.selected_ids.count(bout_filter_row_key(b))) sel.push_back(&b);
        const int nsel = (int)sel.size();
        const int total_frames = st.inputs.total_frames;

        ImGui::SeparatorText("Manual edit");
        ImGui::BeginDisabled(nsel < 2);
        if (ImGui::SmallButton("Merge")) {
            int lo = INT32_MAX, hi = INT32_MIN;
            for (auto *b : sel) { lo = std::min(lo, b->start); hi = std::max(hi, b->end); }
            std::vector<const boutfilter::ResultBout *> ss = sel;
            std::sort(ss.begin(), ss.end(),
                      [](auto *a, auto *b) { return a->start < b->start; });
            bool gap = false;
            for (size_t i = 1; i < ss.size(); ++i)
                if (ss[i]->start > ss[i - 1]->end + 1) gap = true;
            if (gap) {
                st.edit_start_buf = lo; st.edit_end_buf = hi;
                ImGui::OpenPopup("Merge across gap?");
            } else {
                bout_filter_clear_range(st.edits, lo, hi);
                st.edits.manual_bouts.push_back({st.edits.next_id++, lo, hi,
                    boutfilter::ForcedStatus::None, boutfilter::EditKind::Merge});
                bout_filter_mark_edited(st);
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(nsel != 1);
        if (ImGui::SmallButton("Adjust\xE2\x80\xA6")) {
            st.edit_target_id = bout_filter_row_key(*sel[0]);
            st.edit_start_buf = sel[0]->start;
            st.edit_end_buf = sel[0]->end;
            ImGui::OpenPopup("Adjust boundaries");
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(nsel < 1);
        if (ImGui::SmallButton("Accept"))
            for (auto *b : sel) bout_filter_set_status(st, *b, boutfilter::ForcedStatus::Accept);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reject"))
            for (auto *b : sel) bout_filter_set_status(st, *b, boutfilter::ForcedStatus::Reject);
        ImGui::SameLine();
        if (ImGui::SmallButton("Clear override"))
            for (auto *b : sel) bout_filter_set_status(st, *b, boutfilter::ForcedStatus::None);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset to auto"))
            for (auto *b : sel) bout_filter_reset_bout(st, *b);
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(st.edits.empty());
        if (ImGui::SmallButton("Clear all edits")) {
            st.edits = boutfilter::BoutEdits{};
            bout_filter_mark_edited(st);
        }
        ImGui::EndDisabled();
        ImGui::TextDisabled("Click to select+seek \xC2\xB7 Ctrl+click to multi-select.");

        // Adjust-boundaries popup.
        if (ImGui::BeginPopupModal("Adjust boundaries", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SetNextItemWidth(120);
            ImGui::InputInt("Start##adj", &st.edit_start_buf);
            ImGui::SetNextItemWidth(120);
            ImGui::InputInt("End##adj", &st.edit_end_buf);
            bool ok_valid = st.edit_start_buf <= st.edit_end_buf &&
                            st.edit_end_buf >= 0 && st.edit_start_buf <= total_frames - 1;
            if (!ok_valid)
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                    "Start must be \xE2\x89\xA4 End and within [0, %d].", total_frames - 1);
            ImGui::BeginDisabled(!ok_valid);
            if (ImGui::Button("OK##adj")) {
                int ns = std::max(0, std::min(st.edit_start_buf, total_frames - 1));
                int ne = std::max(ns, std::min(st.edit_end_buf, total_frames - 1));
                int lo = ns, hi = ne;
                for (const auto &b : R.bouts)
                    if (bout_filter_row_key(b) == st.edit_target_id) {
                        lo = std::min(lo, b.start); hi = std::max(hi, b.end); break;
                    }
                bout_filter_clear_range(st.edits, lo, hi);
                st.edits.manual_bouts.push_back({st.edits.next_id++, ns, ne,
                    boutfilter::ForcedStatus::None, boutfilter::EditKind::Adjust});
                bout_filter_mark_edited(st);
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Cancel##adj")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
        // Merge-across-gap confirm popup.
        if (ImGui::BeginPopupModal("Merge across gap?", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Merging spans frames %d\xE2\x80\x93%d, absorbing the intervening\n"
                        "frames (including any rejected gap). Metrics are recomputed\n"
                        "over the full span. Continue?",
                        st.edit_start_buf, st.edit_end_buf);
            if (ImGui::Button("Merge##confirm")) {
                bout_filter_clear_range(st.edits, st.edit_start_buf, st.edit_end_buf);
                st.edits.manual_bouts.push_back({st.edits.next_id++,
                    st.edit_start_buf, st.edit_end_buf,
                    boutfilter::ForcedStatus::None, boutfilter::EditKind::Merge});
                bout_filter_mark_edited(st);
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel##confirm")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // Stale/orphaned edit banner.
        if (!R.edit_warnings.empty()) {
            ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1),
                "\xE2\x9A\xA0 %zu manual edit(s) no longer match the current detection.",
                R.edit_warnings.size());
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                for (const auto &w : R.edit_warnings) ImGui::TextUnformatted(w.c_str());
                ImGui::EndTooltip();
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Discard orphaned edits")) {
                const int tf2 = st.inputs.total_frames;
                st.edits.manual_bouts.erase(std::remove_if(st.edits.manual_bouts.begin(),
                    st.edits.manual_bouts.end(), [&](const boutfilter::ManualBout &m) {
                        return m.start > m.end || m.end < 0 ||
                               m.start > tf2 - 1 || tf2 <= 0; }),
                    st.edits.manual_bouts.end());
                st.edits.overrides.erase(std::remove_if(st.edits.overrides.begin(),
                    st.edits.overrides.end(), [&](const boutfilter::StatusOverride &o) {
                        int alen = o.anchor.end - o.anchor.start + 1; double best = 0;
                        for (const auto &ab : st.auto_result.bouts) {
                            int ov = std::min(ab.end, o.anchor.end) -
                                     std::max(ab.start, o.anchor.start) + 1;
                            if (ov <= 0) continue;
                            int den = std::max(alen, ab.n_frames);
                            best = std::max(best, den > 0 ? (double)ov / den : 0.0);
                        }
                        return best < 0.5; }),
                    st.edits.overrides.end());
                bout_filter_mark_edited(st);
            }
        }

        // --- Table filter: show/hide by accept/reject and specific reason ---
        ImGui::Checkbox(("Accepted (" + std::to_string(R.n_accepted) + ")").c_str(),
                        &st.filter_show_accepted);
        ImGui::SameLine();
        ImGui::Checkbox(("Rejected (" + std::to_string(R.n_rejected) + ")").c_str(),
                        &st.filter_show_rejected);
        if (R.n_rejected > 0) {
            std::map<std::string, int> reason_counts;
            for (const auto &b : R.bouts)
                if (!b.accepted) reason_counts[bout_reason_prefix(b.reason)]++;
            ImGui::BeginDisabled(!st.filter_show_rejected);
            ImGui::Indent();
            for (auto &kv : reason_counts) {
                bool shown = st.filter_hidden_reasons.find(kv.first) ==
                             st.filter_hidden_reasons.end();
                std::string label = kv.first + " (" + std::to_string(kv.second) + ")";
                if (ImGui::Checkbox(label.c_str(), &shown)) {
                    if (shown) st.filter_hidden_reasons.erase(kv.first);
                    else st.filter_hidden_reasons.insert(kv.first);
                }
            }
            ImGui::Unindent();
            ImGui::EndDisabled();
        }

        // Per-filter frame pass rates.
        ImGui::TextDisabled("Frame pass: conf %.0f%% \xC2\xB7 upright %.0f%% \xC2\xB7 "
                            "floor %.0f%% \xC2\xB7 Ywall %.0f%% \xC2\xB7 Xwall %.0f%%",
                            R.pct_conf, R.pct_upright, R.pct_floor_ok,
                            R.pct_ywall_ok, R.pct_xwall_ok);

        // --- Bout table ---
        ImGuiTableFlags tf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                             ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
        if (ImGui::BeginTable("##bout_filter_tbl", 8, tf, ImVec2(0, 0))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("#");
            ImGui::TableSetupColumn("Edit");
            ImGui::TableSetupColumn("Start");
            ImGui::TableSetupColumn("End");
            ImGui::TableSetupColumn("Frames");
            ImGui::TableSetupColumn("Cycles");
            ImGui::TableSetupColumn("Status");
            ImGui::TableSetupColumn("Reason");
            ImGui::TableHeadersRow();
            size_t shown = 0;
            for (size_t i = 0; i < R.bouts.size(); ++i) {
                const boutfilter::ResultBout &b = R.bouts[i];
                if (b.accepted) {
                    if (!st.filter_show_accepted) continue;
                } else {
                    if (!st.filter_show_rejected) continue;
                    if (st.filter_hidden_reasons.count(bout_reason_prefix(b.reason))) continue;
                }
                ImGui::TableNextRow();
                ImU32 bg = b.accepted ? IM_COL32(40, 90, 45, 90)
                                      : IM_COL32(110, 40, 40, 90);
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, bg);
                ImGui::TableSetColumnIndex(0);
                uint64_t key = bout_filter_row_key(b);
                bool is_sel = st.selected_ids.count(key) > 0;
                char lbl[32];
                snprintf(lbl, sizeof(lbl), "%zu##bf%zu", ++shown, i);
                if (ImGui::Selectable(lbl, is_sel, ImGuiSelectableFlags_SpanAllColumns)) {
                    if (ImGui::GetIO().KeyCtrl) {
                        if (is_sel) st.selected_ids.erase(key);
                        else st.selected_ids.insert(key);
                    } else {
                        st.selected_ids.clear();
                        st.selected_ids.insert(key);
                        st.seek_requested = true;
                        st.seek_frame = b.start;
                    }
                }
                if (ImGui::IsItemHovered() && (b.accepted ||
                        !std::isnan(b.total_distance_mm))) {
                    const char *origin =
                        b.edit_kind == boutfilter::EditKind::Merge ? "\n(merged, metrics recomputed)"
                      : b.edit_kind == boutfilter::EditKind::Adjust ? "\n(boundaries adjusted, metrics recomputed)"
                      : b.status_overridden ? "\n(status manually overridden)" : "";
                    ImGui::SetTooltip(
                        "frames %d-%d (%.2f s)\ncycles %d\ndistance %.1f mm\n"
                        "mean speed %.1f mm/s%s",
                        b.start, b.end, b.duration_s, b.min_cycles,
                        b.total_distance_mm, b.mean_speed_mm_s, origin);
                }
                ImGui::TableSetColumnIndex(1);
                const char *mark =
                    b.edit_kind == boutfilter::EditKind::Merge ? "M"
                  : b.edit_kind == boutfilter::EditKind::Adjust ? "A"
                  : b.status_overridden ? "*" : "";
                ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.35f, 1), "%s", mark);
                ImGui::TableSetColumnIndex(2); ImGui::Text("%d", b.start);
                ImGui::TableSetColumnIndex(3); ImGui::Text("%d", b.end);
                ImGui::TableSetColumnIndex(4); ImGui::Text("%d", b.n_frames);
                ImGui::TableSetColumnIndex(5); ImGui::Text("%d", b.min_cycles);
                ImGui::TableSetColumnIndex(6);
                ImGui::TextColored(b.accepted ? ImVec4(0.5f, 1, 0.5f, 1)
                                              : ImVec4(1, 0.5f, 0.5f, 1),
                                   "%s", b.accepted ? "accept" : "reject");
                ImGui::TableSetColumnIndex(7);
                ImGui::TextUnformatted(b.reason.c_str());
            }
            ImGui::EndTable();
        }
    }, nullptr, ImVec2(480, 640));
}
