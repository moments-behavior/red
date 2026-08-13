#pragma once
// pump_events_core.h — read pumpctl's dispense log and place each dispense on
// the video timeline (no ImGui deps, unit-tested by tests/test_pump_events.cpp).
//
// pumpctl (~/src/pumpctl) appends one JSON object per dispense to
// "pump_dispense_<YYYYmmdd_HHMMSS>.jsonl". With its "log into the newest
// recording folder" option on, that file lands in orange's recording folder
// next to the .mp4s and Cam*_meta.csv sidecars.
//
// The alignment is exact, not fitted: pumpctl stamps each record with the same
// NIC PTP hardware clock that orange writes into the `timestamp` column of
// Cam*_meta.csv, which camera_timestamps.h loads into frame_ns. So a dispense
// maps to a frame by binary-searching the reference camera's timestamps. We
// deliberately do NOT derive the frame from a nominal period and an anchor:
// sync_plan.h warns that accumulates ppm drift over long recordings, and an
// imported sync_plan.json may carry anchor_ts_ns == 0.
//
// Never correlate `realtime_ns` against a PTP recording with a fixed offset —
// pumpctl measured the PTP-to-CLOCK_REALTIME offset drifting +37.7 s -> +42.1 s
// across three days (~17 ppm), so a constant would be seconds wrong.
//
// The format carries no version field and no header; detect it by the filename
// glob plus the presence of `seq`/`monotonic_ns`.

#include "camera_timestamps.h"
#include "json.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace pump_events {

// Which clock a dispense is compared against. Set by the recording's metadata
// format, not by preference: the two formats put frame timestamps on different
// clocks, and mixing them is a multi-second error.
enum class ClockAxis {
    None,       // no usable frame timestamps -> events list, but cannot seek
    Ptp,        // orange Cam*_meta.csv `timestamp`  <- event ptp_ns
    Monotonic,  // orange Cam*_meta.csv `timestamp_sys` <- event monotonic_ns
    Realtime,   // lab ISO-8601 CSV (epoch ns)       <- event realtime_ns
};

inline const char *axis_name(ClockAxis a) {
    switch (a) {
        case ClockAxis::Ptp: return "PTP";
        case ClockAxis::Monotonic: return "monotonic";
        case ClockAxis::Realtime: return "realtime";
        default: return "none";
    }
}

// One dispense. Fields mirror pumpctl/src/dispense_log.cpp; the optional ones
// carry a has_* flag rather than a sentinel because their absence is meaningful
// (a microstep-mode dispense genuinely has no requested volume).
struct PumpEvent {
    // --- clocks (0 when the record omitted them) ---
    int64_t ptp_ns = 0;
    bool    has_ptp = false;      // "ptp_ns" is always present but may be null
    int64_t monotonic_ns = 0;
    int64_t realtime_ns = 0;
    std::string utc;              // ISO-8601, microsecond precision

    // --- what fired ---
    std::string pump;             // "x", "y", "z", ...
    std::string direction;        // "push" (dispense) | "pull" (draw)
    std::string mode;             // "uL" | "microsteps"
    std::string source;           // manual | repeat | experiment | jog
    std::string experiment;
    int step = 0;                 // 1-based, 0 when absent
    int loop = 0;

    double requested_uL = 0.0;    // microliters; only when mode == "uL"
    bool   has_volume = false;
    int    requested_ms = 0;
    int    steps = 0;
    int    delay_us = 0;
    // Predicted wall-clock length of the movement. The only duration signal in
    // the log — there is no completion record — and it already includes the 2x
    // firmware factor (steps * 2 * delay_us / 1000).
    double estimated_actual_ms = 0.0;
    std::string wire;             // literal serial bytes, or "(paced)"
    bool dry = false;             // pump had nothing (or not enough) to give

    // --- provenance. `seq` restarts at 1 in every file, so identity is the
    // pair, never seq alone. ---
    std::string src_file;
    int seq = 0;

    // --- resolved against the recording; -1 = could not be placed ---
    int frame = -1;
    int end_frame = -1;

    bool is_dispense() const { return direction != "pull"; }
};

struct LoadReport {
    int files = 0;
    int records = 0;
    int skipped_lines = 0;        // unparseable; a live log can be torn mid-write
    std::string error;            // set only when nothing could be read at all
};

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

// Append every record in one .jsonl to `out`. Returns false only if the file
// could not be opened; malformed lines are counted and skipped, matching
// pumpctl's own reader (tools/sync_frames.py). The log is appended live and
// flushed per record, so a torn final line is expected, not corruption.
inline bool parse_jsonl(const std::string &path, std::vector<PumpEvent> &out,
                        LoadReport &rep) {
    std::ifstream f(path);
    if (!f) return false;

    const std::string stem = std::filesystem::path(path).filename().string();
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line.empty()) continue;

        nlohmann::json j =
            nlohmann::json::parse(line, nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded() || !j.is_object()) { ++rep.skipped_lines; continue; }

        PumpEvent e;
        // "ptp_ns" is always emitted but is null when pumpctl could not read
        // /dev/ptpN (it needs elevated access); treat null as absent.
        if (j.contains("ptp_ns") && !j["ptp_ns"].is_null()) {
            e.ptp_ns = j["ptp_ns"].get<int64_t>();
            e.has_ptp = e.ptp_ns > 0;
        }
        e.monotonic_ns = j.value("monotonic_ns", (int64_t)0);
        e.realtime_ns  = j.value("realtime_ns", (int64_t)0);
        e.utc          = j.value("utc", std::string());
        e.pump         = j.value("pump", std::string());
        e.direction    = j.value("direction", std::string("push"));
        e.mode         = j.value("mode", std::string());
        e.source       = j.value("source", std::string());
        e.experiment   = j.value("experiment", std::string());
        e.step         = j.value("step", 0);
        e.loop         = j.value("loop", 0);
        if (j.contains("requested_uL") && j["requested_uL"].is_number()) {
            e.requested_uL = j["requested_uL"].get<double>();
            e.has_volume = true;
        }
        e.requested_ms        = j.value("requested_ms", 0);
        e.steps               = j.value("steps", 0);
        e.delay_us            = j.value("delay_us", 0);
        e.estimated_actual_ms = j.value("estimated_actual_ms", 0.0);
        e.wire                = j.value("wire", std::string());
        e.dry                 = j.value("dry", false);
        e.seq                 = j.value("seq", 0);
        e.src_file            = stem;

        out.push_back(std::move(e));
        ++rep.records;
    }
    ++rep.files;
    return true;
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

inline bool is_pump_log_name(const std::string &name) {
    return name.rfind("pump_dispense_", 0) == 0 &&
           name.size() > 6 &&
           name.compare(name.size() - 6, 6, ".jsonl") == 0;
}

// All pump logs in `folder`, sorted by name (which sorts by time, the filename
// being a local-time YYYYmmdd_HHMMSS stamp). pumpctl opens a fresh file every
// time it notices a new recording, so one session can span several.
inline std::vector<std::string> discover_logs(const std::string &folder) {
    namespace fs = std::filesystem;
    std::vector<std::string> out;
    if (folder.empty()) return out;
    std::error_code ec;
    if (!fs::is_directory(folder, ec)) return out;
    for (const auto &entry : fs::directory_iterator(folder, ec)) {
        if (!entry.is_regular_file()) continue;
        if (is_pump_log_name(entry.path().filename().string()))
            out.push_back(entry.path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Search the media folder, then its parent — the same order media_loader.h uses
// for timestamp sidecars, since videos are sometimes a level below the
// recording folder. Returns the first level that has any logs.
inline std::vector<std::string> discover_logs_with_parent(const std::string &folder) {
    std::vector<std::string> found = discover_logs(folder);
    if (!found.empty()) return found;
    std::error_code ec;
    std::filesystem::path parent = std::filesystem::path(folder).parent_path();
    if (!parent.empty()) return discover_logs(parent.string());
    return found;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

// Read every given log into one list, sorted by time. An empty log file is
// normal — pumpctl opens one at startup whether or not anything dispenses.
inline std::vector<PumpEvent> load_files(const std::vector<std::string> &paths,
                                         LoadReport &rep) {
    std::vector<PumpEvent> out;
    for (const auto &p : paths)
        if (!parse_jsonl(p, out, rep))
            rep.error = "could not open " + p;
    if (!out.empty()) rep.error.clear();
    // Sort on realtime_ns: it is the one clock present in every record
    // regardless of PTP availability, and is only used here for ordering.
    std::stable_sort(out.begin(), out.end(),
                     [](const PumpEvent &a, const PumpEvent &b) {
                         return a.realtime_ns < b.realtime_ns;
                     });
    return out;
}

// ---------------------------------------------------------------------------
// Placing events on the timeline
// ---------------------------------------------------------------------------

// Pick the axis the recording's frame timestamps live on. Monotonic is never
// chosen here — it is only ever a per-event fallback inside resolve(), for
// records that lack a PTP time on an otherwise-PTP recording.
inline ClockAxis pick_axis(camera_timestamps::Format fmt) {
    switch (fmt) {
        case camera_timestamps::Format::OrangePTP: return ClockAxis::Ptp;
        case camera_timestamps::Format::LabIsoCsv: return ClockAxis::Realtime;
        default:                                   return ClockAxis::None;
    }
}

// The event's timestamp on `axis`, or 0 when this record cannot supply it.
inline int64_t event_time_on(const PumpEvent &e, ClockAxis axis) {
    switch (axis) {
        case ClockAxis::Ptp:       return e.has_ptp ? e.ptp_ns : 0;
        case ClockAxis::Monotonic: return e.monotonic_ns;
        case ClockAxis::Realtime:  return e.realtime_ns;
        default:                   return 0;
    }
}

// Index of the frame whose timestamp is nearest `t`, or -1 when `t` falls more
// than `tol_ns` outside the recording. Clamping an out-of-range event to frame
// 0 would silently assert it happened during the video, so we refuse instead.
inline int nearest_frame(const std::vector<int64_t> &ref_ns, int64_t t,
                         int64_t tol_ns) {
    if (ref_ns.empty()) return -1;
    if (t < ref_ns.front() - tol_ns) return -1;
    if (t > ref_ns.back() + tol_ns) return -1;

    auto it = std::lower_bound(ref_ns.begin(), ref_ns.end(), t);
    if (it == ref_ns.end()) return (int)(ref_ns.size() - 1);
    if (it == ref_ns.begin()) return 0;
    auto prev = it - 1;
    return (int)((t - *prev <= *it - t) ? (prev - ref_ns.begin())
                                        : (it - ref_ns.begin()));
}

// Inputs for resolve(). ref_ns / ref_sys_ns are the reference camera's frame
// timestamps on the PTP and monotonic axes respectively (ref_sys_ns may be
// empty). offset_ns is the user's manual nudge.
struct ResolveInputs {
    const std::vector<int64_t> *ref_ns = nullptr;
    const std::vector<int64_t> *ref_sys_ns = nullptr;
    ClockAxis axis = ClockAxis::None;
    int64_t offset_ns = 0;
    int64_t tol_ns = 0;    // how far outside the recording still counts (0 = exact span)
};

// Fill frame/end_frame on every event. Events that cannot be placed keep -1.
// Returns how many were placed.
//
// On an orange recording a record whose ptp_ns was null still resolves, via
// monotonic_ns against the meta.csv timestamp_sys column — the two are the same
// CLOCK_MONOTONIC as long as the acquisition host has not rebooted between the
// recording and the dispense, which holds within a session.
inline int resolve(std::vector<PumpEvent> &events, const ResolveInputs &in) {
    int placed = 0;
    if (in.axis == ClockAxis::None || !in.ref_ns || in.ref_ns->empty()) {
        for (auto &e : events) { e.frame = -1; e.end_frame = -1; }
        return 0;
    }

    const bool have_sys = in.ref_sys_ns && in.ref_sys_ns->size() == in.ref_ns->size();

    for (auto &e : events) {
        ClockAxis axis = in.axis;
        const std::vector<int64_t> *series = in.ref_ns;

        // Per-event fallback: PTP recording, but this record has no PTP time.
        if (axis == ClockAxis::Ptp && !e.has_ptp) {
            if (have_sys && e.monotonic_ns > 0) {
                axis = ClockAxis::Monotonic;
                series = in.ref_sys_ns;
            } else {
                e.frame = -1; e.end_frame = -1;
                continue;
            }
        }

        const int64_t t = event_time_on(e, axis);
        if (t <= 0) { e.frame = -1; e.end_frame = -1; continue; }

        e.frame = nearest_frame(*series, t + in.offset_ns, in.tol_ns);
        if (e.frame < 0) { e.end_frame = -1; continue; }
        ++placed;

        if (e.estimated_actual_ms > 0.0) {
            const int64_t end_t =
                t + in.offset_ns + (int64_t)(e.estimated_actual_ms * 1e6);
            int ef = nearest_frame(*series, end_t, in.tol_ns);
            // A dispense running past the end of the recording ends at the last
            // frame rather than nowhere.
            e.end_frame = ef >= 0 ? ef : (int)series->size() - 1;
            if (e.end_frame < e.frame) e.end_frame = e.frame;
        } else {
            e.end_frame = e.frame;
        }
    }
    return placed;
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

// First event strictly after `frame` among those passing `keep`, or -1.
// Events are time-ordered but their frames need not be (a null-PTP record
// resolves on a different axis), so scan for the minimum rather than assuming.
template <typename Pred>
inline int next_after(const std::vector<PumpEvent> &events, int frame, Pred keep) {
    int best = -1;
    for (size_t i = 0; i < events.size(); ++i) {
        const PumpEvent &e = events[i];
        if (e.frame <= frame || !keep(e)) continue;
        if (best < 0 || e.frame < events[best].frame) best = (int)i;
    }
    return best;
}

// Last event strictly before `frame` among those passing `keep`, or -1.
template <typename Pred>
inline int prev_before(const std::vector<PumpEvent> &events, int frame, Pred keep) {
    int best = -1;
    for (size_t i = 0; i < events.size(); ++i) {
        const PumpEvent &e = events[i];
        if (e.frame < 0 || e.frame >= frame || !keep(e)) continue;
        if (best < 0 || e.frame > events[best].frame) best = (int)i;
    }
    return best;
}

// Human-readable volume. Microliters is the log's unit; only switch to mL where
// that would otherwise print an awkward number of digits.
inline std::string format_volume(const PumpEvent &e) {
    if (!e.has_volume) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d steps", e.steps);
        return buf;
    }
    char buf[64];
    if (e.requested_uL >= 1000.0)
        snprintf(buf, sizeof(buf), "%.3f mL", e.requested_uL / 1000.0);
    else
        snprintf(buf, sizeof(buf), "%.2f uL", e.requested_uL);
    return buf;
}

// "HH:MM:SS.mmm" from the record's UTC string, which is
// "YYYY-MM-DDTHH:MM:SS.ffffffZ". Falls back to the whole string if unexpected.
inline std::string format_clock(const PumpEvent &e) {
    const size_t tpos = e.utc.find('T');
    if (tpos == std::string::npos || e.utc.size() < tpos + 13) return e.utc;
    return e.utc.substr(tpos + 1, 12);
}

}  // namespace pump_events
