#pragma once
// sync_plan.h — canonical trigger-timeline alignment for multi-camera playback.
//
// C++ port of cluster_pose/harness/check_sync.py (plan version 1). The rig
// hardware-triggers all cameras off a shared PTP clock; the viewer assumes the
// i-th decoded frame of every camera is the same trigger instant. A dropped
// frame in one camera breaks that for everything after it. This module reads
// per-camera timestamps (via camera_timestamps.h), reconstructs the canonical
// trigger timeline, and produces a per-camera remap:
//
//   has(t) — does the camera have a real frame at canonical slot t
//   pos(t) — which mp4 frame index delivers slot t
//   slot_of_pos(p) — inverse map: canonical slot of mp4 frame p
//
// Status: clean (identity mapping), trim (start/end skew only), reindex
// (interior drops). A plan can also be imported from a cluster_pose
// sync_plan.json found next to the videos.
//
// All spans are derived from frame COUNTS, never timestamp arithmetic — ppm
// clock drift over ~500k frames would accumulate a 1-2 slot error.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "camera_timestamps.h"
#include "json.hpp"

namespace sync_plan {

constexpr int PLAN_VERSION = 1;
constexpr double DROP_FACTOR = 1.5;  // gap = inter-frame delta >= factor * delta_ns

struct Gap {
    int64_t slot = 0;        // canonical slot of the first lost frame
    int64_t lost = 0;        // number of consecutive lost frames
    int64_t pos_before = 0;  // mp4 index of the first frame AFTER the gap
};

// Per-camera view of the canonical timeline (port of check_sync.py SyncCam).
struct SyncCam {
    int64_t start_slot = 0;
    int64_t decoded_len = 0;
    int64_t end_slot = 0;             // start_slot + decoded_len + sum(lost)
    std::vector<Gap> gaps;            // sorted by slot

    // Recompute end_slot and each gap's pos_before. Call after filling fields.
    void finalize() {
        std::sort(gaps.begin(), gaps.end(),
                  [](const Gap &a, const Gap &b) { return a.slot < b.slot; });
        int64_t total_lost = 0;
        for (auto &g : gaps) {
            g.pos_before = (g.slot - start_slot) - total_lost;
            total_lost += g.lost;
        }
        end_slot = start_slot + decoded_len + total_lost;
    }

    bool has(int64_t t) const {
        if (t < start_slot || t >= end_slot) return false;
        for (const auto &g : gaps) {
            if (g.slot <= t && t < g.slot + g.lost) return false;
            if (g.slot > t) break;
        }
        return true;
    }

    // mp4 frame index delivering canonical slot t (= count of present frames
    // before t). For t inside a gap this is the first frame after the gap.
    int64_t pos(int64_t t) const {
        int64_t lost_before = 0;
        for (const auto &g : gaps) {
            if (g.slot >= t) break;
            lost_before += g.lost;
        }
        return (t - start_slot) - lost_before;
    }

    // Canonical slot of mp4 frame p (inverse of pos over present slots).
    int64_t slot_of_pos(int64_t p) const {
        int64_t lost = 0;
        for (const auto &g : gaps) {
            if (g.pos_before > p) break;
            lost += g.lost;
        }
        return start_slot + p + lost;
    }

    // mp4 index of the first present frame at or after slot t, clamped into
    // [0, decoded_len). This is the seek target for slot t. Unlike pos(),
    // it is well-defined for a t in the middle of a gap: pos() subtracts a
    // gap's full lost count once t passes the gap start, which for mid-gap
    // t points several frames before the gap instead of just after it.
    int64_t seek_pos(int64_t t) const {
        if (t <= start_slot) return 0;
        if (t >= end_slot) return decoded_len - 1;
        int64_t absent = 0;
        for (const auto &g : gaps) {
            if (g.slot >= t) break;
            absent += std::min(g.lost, t - g.slot);
        }
        return (t - start_slot) - absent;
    }
};

enum class Status { Clean, Trim, Reindex };

inline const char *status_name(Status s) {
    switch (s) {
        case Status::Clean: return "clean";
        case Status::Trim: return "trim";
        case Status::Reindex: return "reindex";
    }
    return "?";
}

struct SyncPlan {
    bool valid = false;
    std::string error;                 // set when !valid (or warnings appended)
    std::string source;                // "meta.csv" or path of imported JSON

    int64_t delta_ns = 0;              // nominal inter-frame interval
    int64_t anchor_ts_ns = 0;          // earliest first_ts = canonical slot 0
    int64_t canonical_len = 0;         // max end_slot over cameras
    int64_t predict_start = 0;         // first slot where all cameras started
    int64_t predict_len = 0;           // slots where all cameras are present-ish
    int64_t first_drop_slot = -1;      // earliest gap slot, -1 if none
    Status status = Status::Clean;
    std::map<std::string, SyncCam> cams;  // keyed by FULL camera name

    const SyncCam *cam(const std::string &name) const {
        auto it = cams.find(name);
        return it == cams.end() ? nullptr : &it->second;
    }
    bool usable() const { return valid && !cams.empty(); }
};

namespace detail {

// Camera token used by camera_timestamps: the text after "Cam" in the stem.
inline std::string cam_token(const std::string &full_name) {
    if (full_name.rfind("Cam", 0) == 0 && full_name.size() > 3)
        return full_name.substr(3);
    return full_name;
}

}  // namespace detail

// Build a plan from loaded timestamps (port of check_sync.py analyze_recording).
// `full_cam_names` are the viewer camera names (video stems); timestamps in
// `ts` may be keyed by either the full name or its post-"Cam" token. Every
// loaded camera must have timestamps — a camera the plan cannot map would
// silently desync, so its absence invalidates the whole plan.
inline SyncPlan build(const camera_timestamps::CameraTimestamps &ts,
                      const std::vector<std::string> &full_cam_names,
                      int64_t delta_ns_override = 0) {
    SyncPlan plan;
    plan.source = "meta.csv";
    if (full_cam_names.empty()) {
        plan.error = "no cameras";
        return plan;
    }

    // Resolve each camera's timestamp series.
    std::map<std::string, const std::vector<int64_t> *> series;
    for (const auto &name : full_cam_names) {
        const std::string token = detail::cam_token(name);
        const std::vector<int64_t> *s = nullptr;
        if (ts.has(token)) s = &ts.frame_ns.at(token);
        else if (ts.has(name)) s = &ts.frame_ns.at(name);
        if (!s || s->size() < 2) {
            plan.error = "no timestamps for camera " + name;
            return plan;
        }
        series[name] = s;
    }

    // Nominal inter-frame interval: median over cameras of each camera's
    // median inter-frame delta (drops are rare, the median is unaffected).
    if (delta_ns_override > 0) {
        plan.delta_ns = delta_ns_override;
    } else {
        std::vector<int64_t> per_cam;
        for (const auto &kv : series) {
            int64_t p = camera_timestamps::frame_period_ns(*kv.second);
            if (p > 0) per_cam.push_back(p);
        }
        if (per_cam.empty()) {
            plan.error = "cannot infer frame interval";
            return plan;
        }
        plan.delta_ns = camera_timestamps::detail::median_i64(std::move(per_cam));
    }
    if (plan.delta_ns <= 0) {
        plan.error = "bad frame interval";
        return plan;
    }
    const double delta = (double)plan.delta_ns;

    // Anchor = earliest first timestamp across cameras = canonical slot 0.
    int64_t anchor = INT64_MAX;
    for (const auto &kv : series) anchor = std::min(anchor, kv.second->front());
    plan.anchor_ts_ns = anchor;

    const int64_t thr = (int64_t)std::llround(DROP_FACTOR * delta);
    bool any_drops = false, starts_zero = true, ends_equal = true;
    int64_t first_end = -1;

    for (const auto &name : full_cam_names) {
        const auto &ns = *series[name];
        SyncCam c;
        c.decoded_len = (int64_t)ns.size();
        c.start_slot = (int64_t)std::llround((double)(ns.front() - anchor) / delta);

        // Interior drops: gap positions first in mp4 coordinates, converted to
        // canonical slots via the cumulative lost count (check_sync.py:176-181).
        int64_t cum_lost = 0;
        for (size_t i = 1; i < ns.size(); ++i) {
            int64_t d = ns[i] - ns[i - 1];
            if (d < thr) continue;
            int64_t lost = (int64_t)std::llround((double)d / delta) - 1;
            if (lost <= 0) continue;
            Gap g;
            g.lost = lost;
            g.slot = c.start_slot + (int64_t)i + cum_lost;
            c.gaps.push_back(g);
            cum_lost += lost;
        }
        c.finalize();

        // Soft drift check: timestamp-implied span vs count-based span. Only
        // gross corruption trips this (tolerance >> ppm drift).
        int64_t true_span = c.decoded_len + cum_lost;
        int64_t ts_span =
            (int64_t)std::llround((double)(ns.back() - ns.front()) / delta) + 1;
        int64_t tol = std::max<int64_t>(3, (int64_t)(5e-4 * (double)true_span));
        if (std::llabs(ts_span - true_span) > tol) {
            plan.error = "timestamps inconsistent for camera " + name +
                         " (span " + std::to_string(ts_span) + " vs " +
                         std::to_string(true_span) + ")";
            return plan;
        }

        if (!c.gaps.empty()) any_drops = true;
        if (c.start_slot != 0) starts_zero = false;
        if (first_end < 0) first_end = c.end_slot;
        else if (c.end_slot != first_end) ends_equal = false;

        plan.canonical_len = std::max(plan.canonical_len, c.end_slot);
        plan.predict_start = std::max(plan.predict_start, c.start_slot);
        if (!c.gaps.empty())
            plan.first_drop_slot = (plan.first_drop_slot < 0)
                                       ? c.gaps.front().slot
                                       : std::min(plan.first_drop_slot,
                                                  c.gaps.front().slot);
        plan.cams[name] = std::move(c);
    }

    int64_t min_end = INT64_MAX;
    for (const auto &kv : plan.cams) min_end = std::min(min_end, kv.second.end_slot);
    plan.predict_len = std::max<int64_t>(0, min_end - plan.predict_start);

    plan.status = (!any_drops && starts_zero && ends_equal) ? Status::Clean
                  : !any_drops                              ? Status::Trim
                                                            : Status::Reindex;
    plan.valid = true;
    return plan;
}

// Import a cluster_pose sync_plan.json (plan version 1).
inline SyncPlan load_json(const std::string &path) {
    SyncPlan plan;
    plan.source = path;
    std::ifstream f(path);
    if (!f) {
        plan.error = "cannot open " + path;
        return plan;
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception &e) {
        plan.error = std::string("parse error: ") + e.what();
        return plan;
    }
    try {
        if (j.value("version", 0) != PLAN_VERSION) {
            plan.error = "unsupported plan version";
            return plan;
        }
        plan.delta_ns = j.at("delta_ns").get<int64_t>();
        plan.anchor_ts_ns = j.value("anchor_ts_ns", (int64_t)0);
        plan.canonical_len = j.at("canonical_len").get<int64_t>();
        plan.predict_start = j.value("predict_start", (int64_t)0);
        plan.predict_len = j.value("predict_len", (int64_t)0);
        std::string st = j.value("status", "reindex");
        plan.status = (st == "clean")  ? Status::Clean
                      : (st == "trim") ? Status::Trim
                                       : Status::Reindex;
        if (j.contains("first_drop_slot") && !j["first_drop_slot"].is_null())
            plan.first_drop_slot = j["first_drop_slot"].get<int64_t>();
        for (auto it = j.at("cameras").begin(); it != j.at("cameras").end(); ++it) {
            SyncCam c;
            c.start_slot = it.value().at("start_slot").get<int64_t>();
            c.decoded_len = it.value().at("decoded_len").get<int64_t>();
            for (const auto &gj : it.value().value("gaps", nlohmann::json::array())) {
                Gap g;
                g.slot = gj.at("slot").get<int64_t>();
                g.lost = gj.at("lost").get<int64_t>();
                c.gaps.push_back(g);
            }
            c.finalize();
            plan.cams[it.key()] = std::move(c);
        }
    } catch (const std::exception &e) {
        plan = SyncPlan{};
        plan.source = path;
        plan.error = std::string("bad plan json: ") + e.what();
        return plan;
    }
    plan.valid = !plan.cams.empty();
    if (!plan.valid) plan.error = "plan has no cameras";
    return plan;
}

// Cross-check each camera's plan decoded_len against the demuxed frame count
// (frame_counts keyed by full camera name; <=0 = unknown, skipped). A mismatch
// means a corrupt/truncated mp4, not a desync — refuse rather than misalign.
inline bool check_decoded_counts(SyncPlan &plan,
                                 const std::map<std::string, int> &frame_counts) {
    if (!plan.valid) return false;
    for (const auto &kv : plan.cams) {
        auto it = frame_counts.find(kv.first);
        if (it == frame_counts.end() || it->second <= 0) continue;
        int64_t tol = std::max<int64_t>(5, kv.second.decoded_len / 1000);
        if (std::llabs((int64_t)it->second - kv.second.decoded_len) > tol) {
            plan.valid = false;
            plan.error = "camera " + kv.first + ": video has " +
                         std::to_string(it->second) + " frames but plan expects " +
                         std::to_string(kv.second.decoded_len) +
                         " (corrupt video or stale plan)";
            return false;
        }
    }
    return true;
}

// Port of test_sync.py's mapping invariant: over [start_slot, end_slot) the
// present slots map exactly onto mp4 positions 0..decoded_len-1, monotonically
// and gaplessly, and slot_of_pos inverts pos.
inline bool self_check(const SyncPlan &plan, std::string *err = nullptr) {
    auto fail = [&](const std::string &m) {
        if (err) *err = m;
        return false;
    };
    for (const auto &kv : plan.cams) {
        const SyncCam &c = kv.second;
        int64_t expect_pos = 0;
        for (int64_t t = c.start_slot; t < c.end_slot; ++t) {
            if (!c.has(t)) continue;
            int64_t p = c.pos(t);
            if (p != expect_pos)
                return fail(kv.first + ": pos(" + std::to_string(t) + ")=" +
                            std::to_string(p) + " expected " +
                            std::to_string(expect_pos));
            if (c.slot_of_pos(p) != t)
                return fail(kv.first + ": slot_of_pos(" + std::to_string(p) +
                            ")=" + std::to_string(c.slot_of_pos(p)) +
                            " expected " + std::to_string(t));
            ++expect_pos;
        }
        if (expect_pos != c.decoded_len)
            return fail(kv.first + ": " + std::to_string(expect_pos) +
                        " present slots, decoded_len " +
                        std::to_string(c.decoded_len));
        if (c.has(c.start_slot - 1) || c.has(c.end_slot))
            return fail(kv.first + ": has() true outside span");
    }
    return true;
}

}  // namespace sync_plan
