#pragma once
// camera_timestamps.h — Per-camera frame timestamp loading for multi-camera
// frame alignment in video calibration.
//
// Two on-disk metadata formats are supported, both normalized to int64
// nanoseconds on a clock that is shared across cameras, so that all downstream
// alignment logic is format-agnostic:
//
//   OrangePTP  — native acquisition ("orange"). Sidecar "Cam{cam}_meta.csv"
//                with header  frame_id,timestamp,timestamp_sys.  `timestamp` is
//                the PTP hardware clock (ns), shared across cameras (genlocked).
//   LabIsoCsv  — foreign lab software. User-supplied filename pattern, e.g.
//                "cam{cam}_timestamps_*.csv", no header, one ISO-8601 wall-clock
//                string per line per frame (e.g. 2026-06-01T17:11:46.7162880-04:00).
//
// `{cam}` is the camera token — the text after "Cam" in the video stem, i.e. the
// same key used everywhere else in the calibration pipeline (cam_ordered entry).
//
// No external dependencies beyond the C++17 standard library + <filesystem>.

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace camera_timestamps {

enum class Format { None, OrangePTP, LabIsoCsv };

// Normalized timestamps for a set of cameras. All times in int64 nanoseconds on
// a clock shared across cameras (PTP for orange, wall-clock for lab CSV).
struct CameraTimestamps {
    Format format = Format::None;
    // keyed by camera name; one entry per recorded metadata row.
    std::map<std::string, std::vector<int64_t>> frame_ns;
    // orange only (hardware frame counter); empty for lab CSV.
    std::map<std::string, std::vector<int64_t>> frame_id;
    // video frame count per camera, filled by the caller from a demux probe
    // (-1 = unknown). Used for the timestamp-rows vs. decoded-frames count check.
    std::map<std::string, int> video_frame_count;

    bool has(const std::string &cam) const {
        auto it = frame_ns.find(cam);
        return it != frame_ns.end() && !it->second.empty();
    }
};

// Per-camera integer frame offset relative to a reference camera, plus the
// diagnostics needed to decide whether the constant-offset model is trustworthy.
struct OffsetResult {
    std::map<std::string, int> offset;       // frames; reference camera is 0
    std::map<std::string, bool> constant;    // false => drift / dropped frame
    std::map<std::string, int> max_dev;      // worst |probe - offset|, in frames
    std::string reference;
};

// Result of validating loaded timestamps against the videos.
struct Validation {
    struct Cam {
        bool counts_ok = true;               // rows == video_frame_count
        int rows = 0;
        int video_frames = -1;
        std::vector<int> gap_indices;        // rows where dt > ~1.5x frame period
        std::vector<int> gap_missing;        // est. dropped frames per gap (0 = jitter)
        std::vector<int> frame_id_gaps;      // orange: rows where frame_id jumps >1
        int est_dropped = 0;                 // sum of gap_missing (likely real drops)
    };
    std::map<std::string, Cam> cams;
    bool counts_ok = true;                   // all cameras: rows == video frames
    bool clean = true;                       // counts_ok AND no gaps/drops anywhere
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace detail {

// Days since Unix epoch for a civil date (Howard Hinnant's algorithm).
inline int64_t days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153u * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (int)doe - 719468;
}

inline int median_int(std::vector<int> v) {
    if (v.empty()) return 0;
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
}

inline int64_t median_i64(std::vector<int64_t> v) {
    if (v.empty()) return 0;
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
}

// Minimal glob: '*' matches any run of characters, '?' a single character.
inline bool wildcard_match(const std::string &pat, const std::string &s) {
    size_t p = 0, n = 0, star = std::string::npos, mark = 0;
    while (n < s.size()) {
        if (p < pat.size() && (pat[p] == '?' || pat[p] == s[n])) {
            ++p; ++n;
        } else if (p < pat.size() && pat[p] == '*') {
            star = p++; mark = n;
        } else if (star != std::string::npos) {
            p = star + 1; n = ++mark;
        } else {
            return false;
        }
    }
    while (p < pat.size() && pat[p] == '*') ++p;
    return p == pat.size();
}

inline std::string substitute_cam(const std::string &pattern, const std::string &cam) {
    std::string out;
    for (size_t i = 0; i < pattern.size();) {
        if (pattern.compare(i, 5, "{cam}") == 0) { out += cam; i += 5; }
        else out += pattern[i++];
    }
    return out;
}

} // namespace detail

// Parse an ISO-8601 timestamp to nanoseconds since the Unix epoch (UTC).
// Accepts "YYYY-MM-DDTHH:MM:SS[.fffffff][(+|-)HH:MM]". Variable fractional digit
// count; trailing 'Z' or no zone is treated as UTC. Returns INT64_MIN on failure.
inline int64_t parse_iso8601_ns(const std::string &s) {
    int Y, Mo, D, H, Mi, S;
    // Expect at least "YYYY-MM-DDTHH:MM:SS"
    if (s.size() < 19) return INT64_MIN;
    auto two = [](const char *p, int &out) -> bool {
        return std::from_chars(p, p + 2, out).ec == std::errc();
    };
    {
        const char *p = s.c_str();
        if (std::from_chars(p, p + 4, Y).ec != std::errc()) return INT64_MIN;
        if (s[4] != '-' || s[7] != '-' || (s[10] != 'T' && s[10] != ' ')) return INT64_MIN;
        if (!two(p + 5, Mo) || !two(p + 8, D)) return INT64_MIN;
        if (s[13] != ':' || s[16] != ':') return INT64_MIN;
        if (!two(p + 11, H) || !two(p + 14, Mi) || !two(p + 17, S)) return INT64_MIN;
    }
    size_t i = 19;
    int64_t frac_ns = 0;
    if (i < s.size() && s[i] == '.') {
        ++i;
        int64_t scale = 100000000; // first digit = 1e8 ns
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            if (scale > 0) { frac_ns += (s[i] - '0') * scale; scale /= 10; }
            ++i; // extra digits beyond ns precision are ignored
        }
    }
    int64_t tz_off = 0; // seconds; local = UTC + tz_off
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
        int sign = (s[i] == '-') ? -1 : 1;
        int zh = 0, zm = 0;
        const char *p = s.c_str() + i + 1;
        if (i + 3 > s.size()) return INT64_MIN;
        std::from_chars(p, p + 2, zh);
        if (i + 6 <= s.size() && s[i + 3] == ':')
            std::from_chars(p + 3, p + 5, zm);
        else if (i + 5 <= s.size())
            std::from_chars(p + 2, p + 4, zm);
        tz_off = sign * (zh * 3600 + zm * 60);
    }
    int64_t days = detail::days_from_civil(Y, (unsigned)Mo, (unsigned)D);
    int64_t local_sec = days * 86400 + H * 3600 + Mi * 60 + S;
    int64_t utc_sec = local_sec - tz_off;
    return utc_sec * 1000000000LL + frac_ns;
}

// Parse an orange "Cam{cam}_meta.csv": header frame_id,timestamp,timestamp_sys.
// Fills frame_ns (from the PTP `timestamp` column) and frame_id. Returns false
// if the file cannot be opened or the header is unexpected.
inline bool parse_orange_meta(const std::string &path,
                              std::vector<int64_t> &out_ns,
                              std::vector<int64_t> &out_id) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    if (!std::getline(f, line)) return false;
    if (line.find("frame_id") == std::string::npos) return false;
    out_ns.clear(); out_id.clear();
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const char *b = line.c_str(), *e = b + line.size();
        int64_t fid = 0, ts = 0;
        auto r1 = std::from_chars(b, e, fid);
        if (r1.ec != std::errc() || r1.ptr >= e || *r1.ptr != ',') continue;
        auto r2 = std::from_chars(r1.ptr + 1, e, ts);
        if (r2.ec != std::errc()) continue;
        out_id.push_back(fid);
        out_ns.push_back(ts);
    }
    return !out_ns.empty();
}

// Parse a lab CSV: one ISO-8601 timestamp per line, no header.
inline bool parse_lab_csv(const std::string &path, std::vector<int64_t> &out_ns) {
    std::ifstream f(path);
    if (!f) return false;
    out_ns.clear();
    std::string line;
    while (std::getline(f, line)) {
        // trim trailing CR/space
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line.empty()) continue;
        int64_t ns = parse_iso8601_ns(line);
        if (ns == INT64_MIN) {
            if (out_ns.empty()) continue; // tolerate a stray header-ish first line
            return false;                 // mid-file corruption: fail loudly
        }
        out_ns.push_back(ns);
    }
    return !out_ns.empty();
}

// Among files in `folder` matching `glob`, pick the best for a camera: prefer an
// exact row-count match to `expected` (>0), else the file with the most rows.
inline std::string pick_lab_file(const std::string &folder, const std::string &glob,
                                 int expected) {
    namespace fs = std::filesystem;
    std::string best; long best_rows = -1; bool exact = false;
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(folder, ec)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (!detail::wildcard_match(glob, name)) continue;
        // count rows (cheap: count newlines)
        std::ifstream f(entry.path());
        long rows = 0; std::string l;
        while (std::getline(f, l)) if (!l.empty()) ++rows;
        bool is_exact = (expected > 0 && rows == expected);
        if (is_exact && !exact) { best = entry.path().string(); best_rows = rows; exact = true; }
        else if (is_exact && exact && rows > best_rows) { best = entry.path().string(); best_rows = rows; }
        else if (!exact && rows > best_rows) { best = entry.path().string(); best_rows = rows; }
    }
    return best;
}

// Auto-detect format and load timestamps for the given cameras.
//   folder         — directory containing sidecar metadata (usually the video folder)
//   cam_ordered    — camera tokens (post-"Cam" stems)
//   lab_pattern    — user-editable glob with {cam}, e.g. "cam{cam}_timestamps_*.csv"
//   expected_counts— optional video frame counts per camera (for lab disambiguation)
inline CameraTimestamps load(const std::string &folder,
                             const std::vector<std::string> &cam_ordered,
                             const std::string &lab_pattern,
                             const std::map<std::string, int> &expected_counts = {}) {
    namespace fs = std::filesystem;
    CameraTimestamps out;
    if (folder.empty() || !fs::is_directory(folder)) return out;

    // 1) Orange: Cam{cam}_meta.csv
    {
        std::map<std::string, std::vector<int64_t>> ns, ids;
        int found = 0;
        for (const auto &cam : cam_ordered) {
            std::string p = (fs::path(folder) / ("Cam" + cam + "_meta.csv")).string();
            std::vector<int64_t> a, b;
            if (fs::exists(p) && parse_orange_meta(p, a, b)) {
                ns[cam] = std::move(a); ids[cam] = std::move(b); ++found;
            }
        }
        if (found >= 2 || (found >= 1 && cam_ordered.size() == 1)) {
            out.format = Format::OrangePTP;
            out.frame_ns = std::move(ns);
            out.frame_id = std::move(ids);
            return out;
        }
    }

    // 2) Lab CSV: user pattern with {cam}
    if (!lab_pattern.empty()) {
        std::map<std::string, std::vector<int64_t>> ns;
        int found = 0;
        for (const auto &cam : cam_ordered) {
            std::string glob = detail::substitute_cam(lab_pattern, cam);
            int expected = -1;
            auto ec = expected_counts.find(cam);
            if (ec != expected_counts.end()) expected = ec->second;
            std::string p = pick_lab_file(folder, glob, expected);
            std::vector<int64_t> a;
            if (!p.empty() && parse_lab_csv(p, a)) { ns[cam] = std::move(a); ++found; }
        }
        if (found >= 2 || (found >= 1 && cam_ordered.size() == 1)) {
            out.format = Format::LabIsoCsv;
            out.frame_ns = std::move(ns);
            return out;
        }
    }

    return out; // Format::None
}

// Estimate the median inter-frame period (ns) for a camera.
inline int64_t frame_period_ns(const std::vector<int64_t> &ns) {
    if (ns.size() < 2) return 0;
    std::vector<int64_t> d;
    d.reserve(ns.size() - 1);
    for (size_t i = 1; i < ns.size(); ++i) d.push_back(ns[i] - ns[i - 1]);
    return detail::median_i64(std::move(d));
}

// Compute a constant integer frame offset per camera relative to `reference`
// (default: cam_ordered[0]). For each of several probe frames on the reference
// camera, the nearest-in-time frame on the other camera is found; the median
// index difference is the offset. If the difference is not constant across
// probes (clock drift or a dropped frame), `constant[cam]` is set false.
inline OffsetResult compute_offsets(const CameraTimestamps &ts,
                                    const std::vector<std::string> &cam_ordered,
                                    const std::string &reference = "") {
    OffsetResult res;
    if (cam_ordered.empty()) return res;
    std::string ref = reference.empty() ? cam_ordered[0] : reference;
    res.reference = ref;
    if (!ts.has(ref)) return res;
    const auto &R = ts.frame_ns.at(ref);

    // probe indices spread across the reference timeline
    std::vector<int> probes;
    int n = (int)R.size();
    int nprobe = std::min(64, n);
    for (int k = 0; k < nprobe; ++k)
        probes.push_back((int)((int64_t)k * (n - 1) / std::max(1, nprobe - 1)));

    for (const auto &cam : cam_ordered) {
        if (cam == ref) { res.offset[cam] = 0; res.constant[cam] = true; res.max_dev[cam] = 0; continue; }
        if (!ts.has(cam)) continue;
        const auto &C = ts.frame_ns.at(cam);
        // Only probes whose reference time falls strictly inside the partner's
        // time range count: a probe that clamps to the partner's first/last
        // frame reflects a non-overlapping span (e.g. a staggered start), not
        // drift, and would otherwise corrupt the constancy check.
        std::vector<int> diffs, all;
        for (int i : probes) {
            int64_t t = R[i];
            auto it = std::lower_bound(C.begin(), C.end(), t);
            if (it == C.begin() || it == C.end()) {
                int j = (it == C.begin()) ? 0 : (int)C.size() - 1;
                all.push_back(j - i);
                continue;
            }
            int j1 = (int)(it - C.begin());
            int j = (t - C[j1 - 1] <= C[j1] - t) ? j1 - 1 : j1;
            diffs.push_back(j - i);
            all.push_back(j - i);
        }
        const std::vector<int> &use = (diffs.size() >= 3) ? diffs : all;
        int off = detail::median_int(use);
        int maxdev = 0;
        for (int d : use) maxdev = std::max(maxdev, std::abs(d - off));
        res.offset[cam] = off;
        res.max_dev[cam] = maxdev;
        res.constant[cam] = (maxdev == 0);
    }
    return res;
}

// Validate loaded timestamps: row-count vs. decoded-frame count, intra-camera
// time gaps (grabber drops), and frame_id discontinuities (orange).
inline Validation validate(const CameraTimestamps &ts) {
    Validation v;
    for (const auto &kv : ts.frame_ns) {
        const std::string &cam = kv.first;
        const auto &ns = kv.second;
        Validation::Cam c;
        c.rows = (int)ns.size();
        auto vc = ts.video_frame_count.find(cam);
        c.video_frames = (vc != ts.video_frame_count.end()) ? vc->second : -1;
        if (c.video_frames >= 0 && c.video_frames != c.rows) {
            c.counts_ok = false;
            v.counts_ok = false;
        }
        int64_t period = frame_period_ns(ns);
        if (period > 0) {
            int64_t thresh = period * 3 / 2;
            for (size_t i = 1; i < ns.size(); ++i) {
                int64_t dt = ns[i] - ns[i - 1];
                if (dt <= thresh) continue;
                c.gap_indices.push_back((int)i);
                // est. dropped frames: a real grabber drop skips a whole sample
                // slot (dt ~= 2x period); a sub-2x spike is timing jitter (0).
                int missing = (int)((dt + period / 2) / period) - 1; // round(dt/period)-1
                if (dt < period * 19 / 10) missing = 0;               // <1.9x => jitter
                c.gap_missing.push_back(missing);
                c.est_dropped += missing;
            }
        }
        auto idit = ts.frame_id.find(cam);
        if (idit != ts.frame_id.end()) {
            const auto &ids = idit->second;
            for (size_t i = 1; i < ids.size(); ++i)
                if (ids[i] - ids[i - 1] != 1) c.frame_id_gaps.push_back((int)i);
        }
        if (!c.gap_indices.empty() || !c.frame_id_gaps.empty() || !c.counts_ok)
            v.clean = false;
        v.cams[cam] = std::move(c);
    }
    return v;
}

} // namespace camera_timestamps
